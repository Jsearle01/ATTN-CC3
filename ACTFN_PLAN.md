# ACTFN Implementation Plan

Exploration report for the ACTFN layer of the ATTN-CC3 port.
Based on reading ACTFN.MAC, LAYER.MAC, TRAIN.MAC, prototype.shf,
prototype-float.shf, gen-lut.shf, and the README.

## Section A — Inventory

### Routines in ACTFN.MAC

**1. RELU (macro)**
- Signature: `RELU Rn` — single register, inline
- Computes: `Rn = max(0, Rn)`
- PDP-11: TST + BPL + CLR = 3 instructions
- No LUT. No dependencies.
- Note: **Not used by the ATTN/11 transformer.** The architecture
  has no feed-forward network and no ReLU activation. Present in
  the NN11 library as a general utility.

**2. DRELU (macro)**
- Signature: `DRELU Rgrad, Ract` — zeroes gradient if activation <= 0
- Not used by ATTN/11 transformer (same reason as RELU).

**3. VRELU (subroutine)**
- Signature: `R0 = vector ptr, R1 = length` — in-place
- Not used by ATTN/11 transformer.

**4. SFTMX (subroutine)**
- Signature: `R0 = vector ptr (Q8), R1 = length` — in-place
- Computes: softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
- Dependencies: VMAX (from VECOP), FXDIV (from FXMATH), EXPTBL
- Algorithm:
  1. Find max via VMAX. (~20 instructions for VMAX call)
  2. For each element: subtract max, negate, divide by 8 (ASH #-3),
     clamp to [0,255], use as EXPTBL index, store exp value in-place,
     accumulate sum. (~15 instructions per element)
  3. For each element: divide by sum via FXDIV, store. (~10 instructions
     per element + FXDIV cost)
- Private storage: SF.VEC, SF.LEN, SF.MAX (6 bytes)
- This is the **only activation function used** by the transformer.

**5. EXPTBL (data)**
- 256 entries of 16-bit words, Q8 format
- Maps index i to round(exp(-i/32) * 256), clamped to non-negative
- Total size: 512 bytes (256 * 2)
- Key values: [0]=256 (exp(0)=1.0), [32]=94 (exp(-1)), [128]=5,
  [200+]=0
- Accessed via word-indexed lookup: `EXPTBL(R0)` where R0 = index*2

### Routines NOT in ACTFN.MAC but needed

**FXDIV — Q8 division (in FXMATH.MAC)**
- `R0 = R0 / R1` in Q8: shifts dividend left 8 (to Q16), then DIV
- On PDP-11: CLR R0 / MOV R2,R1 / ASHC #8 / DIV R3,R0
- **We do not have this in our fxmath.asm yet.** Must be added.
- On 6309: DIVD instruction (D / 8-bit operand) won't suffice — we
  need 32-bit / 16-bit division. Options:
  - Use DIVQ (Q / 16-bit memory = W quotient, D remainder) if available
  - Manual long division via repeated subtraction
  - This is the biggest open question for ACTFN.

**VMAX — Find maximum element and index (in VECOP.MAC)**
- `R0 = vector ptr, R1 = length → R0 = max value, R1 = max index`
- **We do not have this in our vecop.asm yet.** Must be added.
- Straightforward to port: loop comparing elements against running max.

### Routines in LAYER.MAC (not ACTFN, listed for context)

- EMBED: token + position embedding lookup
- PROJ: output projection via VTMUL
- ATTN: full self-attention (calls SFTMX internally)

These are the next layer (LAYER) and consume ACTFN's output.

## Section B — LUT Strategy

### EXPTBL

- **Format**: 256 entries, each a 16-bit word (Q8). Total: 512 bytes.
- **Content**: `EXPTBL[i] = round(exp(-i/32) * 256)`, clamped >= 0.
  Index 0 = 256 (1.0), index 255 = 0.
- **Generator**: Python script matching gen-lut.shf. The table data
  is already hardcoded in ACTFN.MAC — we can either generate it or
  embed it verbatim. Recommend: generate via Python for auditability,
  cross-check against the ACTFN.MAC values.
- **On 6309**: The table is 16-bit words, same as PDP-11. Accessed
  via `LDD table,X` where X = index * 2. The `ASL` for word offset
  on PDP-11 maps to `LSLD` or `LSLA` on 6309.

### LOGTBL

- **Format**: 257 entries (indices 0..256), 16-bit words, Q12.
- **Content**: `LOGTBL[x] = round(-ln(x/256) * 4096)`, unsigned.
  Index 0 clamped to a large value. Index 256 = 0.
- **Total size**: 514 bytes (257 * 2).
- **Usage**: Only in CLOSS (cross-entropy loss computation, every
  50 steps). Not in the training hot path.
- **Recommendation**: Defer to TRAIN implementation. Not needed for
  ACTFN validation.

### Load addresses

Current memory map from equates.inc:
- Code region: $0600-$0C72 (test harness + all code)
- Screen: $0400-$05FF (hardware)
- STR_BASE: $1800 (strings)
- BW_SCRATCH: $4420-$445F (64 bytes, now full)
- STACK_TOP: $4800
- FREE_BASE: $4800+ (test vectors loaded here by Lua)

**Proposed EXPTBL location**: Include inline in the code via FDB
directives (like ATTN/11 does). This avoids a separate load
mechanism. 512 bytes of table data is small enough to embed in
the binary. Place after the SFTMX routine code.

For the production binary, EXPTBL lives in the code region between
$0200 and wherever the code ends. For test harnesses, it's part
of the included source.

No separate Lua load needed for EXPTBL — it's assembled into the
binary. Only test vector data uses the Lua separate-load at $4800.

### LOGTBL location (deferred)

When implemented (TRAIN layer), embed similarly via FDB. 514 bytes.

## Section C — 6309 Translation Notes

### SFTMX calling convention

```
* Entry: X -> vector (Q8, 16-bit elements), B = length
* Exit:  vector replaced in-place with softmax probabilities
* Clobbers: X, Y, U, D, W, Q, CC
* Uses: SF_ private storage in BW_SCRATCH region
```

### Translation issues

**1. VMAX — missing primitive**

Must be added to vecop.asm:
```
* VMAX: X -> vector, B = length
* Exit: D = max value, X = index of max
* Simple loop: LDD ,Y++ / CMPD max / BLE skip / update max+index
```
No multiply, no LUT, no carry — straightforward.

**2. FXDIV — missing primitive**

Must be added to fxmath.asm. PDP-11 uses:
```
CLR R0 / MOV R2,R1 / ASHC #8,R0 / DIV R3,R0
```
This is: build 32-bit Q16 dividend (a << 8), divide by 16-bit divisor.

On 6309, DIVQ exists but needs verification:
- DIVQ: Q (32-bit D:W) / memory (16-bit) = W quotient, D remainder
- If DIVQ works: LDD a / LDW #0 / LSLD×8 (or shift Q left 8) /
  DIVQ divisor_addr / TFR W,D (quotient to D)
- Actually simpler: build Q = a * 256: `LDD a / LDW #0 / shift Q
  left 8`. But shifting Q left 8 requires 8 iterations of LSLD/ROLW
  or a byte-swap trick.
- **Cleanest 6309 approach**: `LDD a / CLRB` (D = a_hi:0, effectively
  a << 8 in high 16) then build Q with W = 0... no, Q = D:W so
  Q = (a<<8):0 = a * 256 * 65536. That's too much.
- **Correct approach**: We want (a * 256) / b. So:
  `a * 256` = a << 8. As 32-bit in Q:
  - High word = a >> 8 (signed)
  - Low word = (a << 8) & 0xFFFF
  - Then DIVQ by b gives quotient in W.
  Store a in memory, use byte-offset trick:
  `STD scratch+1 / sign-extend to scratch+0 / CLR scratch+3`
  Then LDQ from scratch, DIVQ by divisor.
- **Or**: multiply-based: `(a << 8) / b` is equivalent to
  `a * (256/b)` but that doesn't help on hardware.
- Need to verify DIVQ exists in LWASM. Test with a probe.

**3. ASH #-3 (index computation)**

PDP-11 `ASH #-3, R0` = arithmetic shift right by 3 = divide by 8.
On 6309: three LSRD (if unsigned) or three ASRD (if signed, but
the value is always non-negative after NEG). Use LSRD x3 since the
input is |x_i - max| >= 0.

Wait — LSRD is logical (fills with 0). ASRD doesn't exist per our
earlier testing. But the value is always non-negative so LSRD is
correct.

Actually, looking at ACTFN.MAC line 88: `ASH #-3., R0` — this is
a right shift by 3 of a value that was just NEGated (made positive).
So the input is >= 0 and LSRD works.

But LSRD shifts D (16-bit). We only need to shift by 3. Three LSRD
instructions = 6 cycles. Fine.

**4. ASL R0 (word offset)**

PDP-11 `ASL R0` = shift left 1 = multiply by 2 for word indexing.
On 6309: LSLD (shift D left 1). One instruction.

**5. Table lookup `EXPTBL(R0)`**

PDP-11 indexed addressing `EXPTBL(R0)` = load word at EXPTBL+R0.
On 6309: `LDD EXPTBL,X` where X = index*2. Or use `LEAX EXPTBL`
first, then `LDD X,D` — but 6309 doesn't support D as an index
register. Use: `LDX #EXPTBL / LDD D,X` — this uses D as a 16-bit
offset. Yes, `LDD D,X` is valid 6309 indexed addressing (D offset
from X).

**6. Division in normalization step**

Each element needs `exp_i / sum` via FXDIV. This is the expensive
operation — called once per element per softmax. For SEQ=8 (attention
scores) or V=10 (logits), that's 8-10 divisions per softmax call.

If DIVQ works on 6309, each FXDIV is:
- Build Q = exp_i << 8 (~6 instructions)
- DIVQ by sum (~1 instruction, but DIVQ may be slow: ~34 cycles)
- Extract quotient from W (~2 instructions)
- Total: ~40 cycles per element, vs ~50 on PDP-11 with ASHC+DIV

If DIVQ doesn't work, we need a software division loop (~100 cycles).

## Section D — Implementation Order

1. **FXDIV** — Add Q8 division to fxmath.asm. Test with the FXMATH
   harness (extend vectors.bin with division test cases).
   *Rationale*: SFTMX cannot be tested without division.

2. **VMAX** — Add max-find to vecop.asm. Test with VECOP harness.
   *Rationale*: SFTMX step 1 requires it.

3. **EXPTBL generator** — Python script emitting the 256-entry table
   as FDB directives in an includable .asm file, or as raw binary.
   Cross-check against ACTFN.MAC values.

4. **SFTMX** — The main routine. Depends on VMAX, FXDIV, EXPTBL.
   Test with dedicated vectors that verify:
   - Uniform input → uniform output (~1/N each)
   - One-hot dominant input → near-1.0 at max, near-0 elsewhere
   - All-equal input → all 1/N
   - Negative inputs → max-subtraction doesn't underflow
   - Sum of output ≈ 256 (1.0 in Q8)

5. **RELU/VRELU** — Trivial but not needed. Implement if time allows,
   skip if not. The transformer doesn't use ReLU.

## Section E — Open Questions

**Q1: Does DIVQ exist on HD6309 and assemble in LWASM?**
This is the critical unknown. If not, we need a software division
routine. Should be tested before any ACTFN code is written.

**Q2: SFTMX operates in-place — is this compatible with our harness?**
Yes, but the test harness must copy input vectors before calling
SFTMX so it can compare the in-place result against expected values
without losing the original.

**Q3: Does SFTMX's division-by-sum ever divide by zero?**
Only if all exp values round to zero (all inputs are far below max,
which can't happen since max is subtracted). The prototype handles
this: `if (= s 0) (zeros ...)`. We should add the same guard.

**Q4: Temperature/scale parameter?**
No. ATTN/11's softmax has no temperature parameter. The 1/sqrt(d)
scaling is applied to the attention scores BEFORE softmax (in ATTN
routine, via ASH), not inside SFTMX.

**Q5: Index computation: the `ASH #-3` in SFTMX divides by 8, but
gen-lut.shf says the table maps i to exp(-i/32). So the effective
mapping is: input delta → delta/8 → index → exp(-index/32) =
exp(-delta/256). Is this correct?**
Yes. The input delta (max - x_i) is in Q8, so its integer value is
delta/256 in real units. The table index = delta >> 3 = delta/8.
The table gives exp(-index/32) = exp(-(delta/8)/32) = exp(-delta/256).
This is exp(-(max-x_i)) in real units, which is correct for softmax
with numerical stability.

## Section F — Memory Budget

### Current usage (from t_matop.bin)

| Region | Range | Size |
|--------|-------|------|
| Code (harness+all) | $0600-$0C72 | 1650 B |
| BW_SCRATCH | $4420-$445F | 64 B (full) |

### ACTFN additions

| Component | Estimated size |
|-----------|---------------|
| FXDIV routine | ~30 bytes |
| VMAX routine | ~30 bytes |
| SFTMX routine | ~100 bytes |
| SF_ private storage | 6 bytes |
| EXPTBL (data) | 512 bytes |
| **Total ACTFN** | **~680 bytes** |

### Projected code end

$0C72 + 680 = ~$0F1A. Well below $1800 (STR_BASE), $4420
(BW_SCRATCH), and $4800 (VECS).

### Scratch needs

SFTMX needs 6 bytes of private storage (SF.VEC, SF.LEN, SF.MAX).
BW_SCRATCH is full (64 bytes used by MULSCR through MP_SCL).

Options:
- **Expand BW_SCRATCH** from 64 to 80 bytes. BW_END moves from
  $4460 to $4470. Still well below STACK_TOP at $4800. Requires
  updating equates.inc (BW_SCRATCH size, BW_END).
- **Use harness scratch** ($1700 region) for test-only storage.
  For production, the SF_ storage can share addresses with MP_*
  since SFTMX and MATOP are never called concurrently.

Recommend: expand BW_SCRATCH by 16 bytes (to 80) for SFTMX private
storage, giving headroom for future needs.

### LOGTBL (deferred)

514 bytes, needed only for TRAIN. Not in the ACTFN scope.
