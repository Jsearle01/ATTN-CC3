#!/usr/bin/env python3
"""
UPDAT reference generator — WUPDT, CVT16, INITW, RAND.

WUPDT_ONE: per-element SGD on split Q16 hi/lo weights.
  delta = grad >> (lr_shift-1); grad=0; w_hi:w_lo -= delta
  via NEG+ADD with carry propagation + sign extension. Simulates
  the 6309 16-bit sequence step-by-step to catch edge cases.

CVT16_ONE: Q16 -> Q8 middle-byte extraction.
  Q8 = ((hi & 0xFF) << 8) | ((lo >> 8) & 0xFF), sign-extended.
  Big-endian byte pickup from split arrays.

INITW_ONE: RAND -> [0,255] -> [-128,127] Q8 -> Q16 hi/lo.
  hi = sign_ext(Q8), lo = Q8 << 8.

RAND: 15-bit LCG seed = ((seed * 25173) + 13849) & 0x7FFF; init=887.
"""

from pathlib import Path


# ---- RAND / LCG ----

RN_MUL = 25173
RN_ADD = 13849
RN_MASK = 0x7FFF
RN_INIT = 887


class LCG:
    def __init__(self, seed=RN_INIT):
        self.seed = seed

    def rand(self):
        self.seed = ((self.seed * RN_MUL) + RN_ADD) & RN_MASK
        return self.seed


# ---- Helpers ----

def wrap16(x):
    x &= 0xFFFF
    if x >= 0x8000:
        x -= 0x10000
    return x


def asr(val, n):
    """Arithmetic right shift by n (sign-preserving). Python >> is arithmetic."""
    return val >> n


# ---- WUPDT_ONE reference ----
# Simulates the 6309 16-bit sequence step-by-step, mirroring:
#   NEGD / TFR D,W / ADDD lo / BCC nocy / LDD hi / ADDD #1 / STD hi
#   nocy: TSTW / BPL nosex / LDD hi / SUBD #1 / STD hi

def wupdt_element(w_hi, w_lo, grad, lr_shift):
    shift_count = lr_shift - 1

    # delta = grad >> shift_count (arithmetic)
    delta = asr(grad, shift_count)

    # neg_delta = -delta wrapped to 16 bits (two's complement via NEGD)
    neg_delta_unsigned = (-delta) & 0xFFFF
    neg_delta_signed = neg_delta_unsigned - 0x10000 if neg_delta_unsigned >= 0x8000 else neg_delta_unsigned

    # ADDD ,Y: w_lo += neg_delta (16-bit, unsigned add with carry)
    w_lo_u = w_lo & 0xFFFF
    sum_lo = w_lo_u + neg_delta_unsigned
    carry = 1 if sum_lo > 0xFFFF else 0
    new_w_lo = wrap16(sum_lo)

    # Carry propagation: w_hi += 1 if carry (ADDD #1)
    new_w_hi_u = (w_hi + carry) & 0xFFFF

    # Sign extension: if neg_delta was negative (TSTW / BPL), w_hi -= 1
    if neg_delta_signed < 0:
        new_w_hi_u = (new_w_hi_u - 1) & 0xFFFF

    new_w_hi = wrap16(new_w_hi_u)
    return new_w_hi, new_w_lo


def wupdt_ref(w_hi, w_lo, grad, lr_shift):
    n = len(w_hi)
    out_hi = []
    out_lo = []
    out_grad = [0] * n  # gradients zeroed after read
    for i in range(n):
        nh, nl = wupdt_element(w_hi[i], w_lo[i], grad[i], lr_shift)
        out_hi.append(nh)
        out_lo.append(nl)
    return out_hi, out_lo, out_grad


# ---- CVT16_ONE reference ----
# Matches LDA 1,X / LDB ,Y on big-endian storage:
#   hi stored as [hi_msb(bits31-24), hi_lsb(bits23-16)]
#   lo stored as [lo_msb(bits15-8),  lo_lsb(bits7-0)]
#   Q8 = bits[23:8] = (hi & 0xFF) << 8 | (lo >> 8) & 0xFF

def cvt16_element(w_hi, w_lo):
    hi_u = w_hi & 0xFFFF
    lo_u = w_lo & 0xFFFF
    q8 = ((hi_u & 0xFF) << 8) | ((lo_u >> 8) & 0xFF)
    return wrap16(q8)


def cvt16_ref(w_hi, w_lo):
    return [cvt16_element(w_hi[i], w_lo[i]) for i in range(len(w_hi))]


# ---- INITW_ONE reference ----

def initw_ref(count, rng):
    w_hi = []
    w_lo = []
    q8s = []
    for _ in range(count):
        r = rng.rand()
        q8 = (r & 0xFF) - 128       # [-128, 127]
        q8s.append(q8)
        hi = -1 if q8 < 0 else 0     # sign extension
        lo = wrap16(q8 << 8)         # q8 * 256, wrapped as signed 16
        w_hi.append(hi)
        w_lo.append(lo)
    return w_hi, w_lo, q8s


# ---- Self-test ----

def self_test():
    # LCG first 5 values from seed=887
    rng = LCG(RN_INIT)
    expected = [27292, 21477, 15138, 23651, 18680]
    for i, exp in enumerate(expected):
        v = rng.rand()
        assert v == exp, f"LCG step {i}: got {v}, want {exp}"

    # WUPDT: trivial (grad=256, lr_shift=1, initial w=0) -> w = -256
    h, l, g = wupdt_ref([0], [0], [256], 1)
    assert h == [-1] and l == [-256] and g == [0], f"wupdt trivial: h={h} l={l} g={g}"
    # The above: delta=256, neg_delta=-256 -> w_lo=0+(-256)=-256 (no carry out of 16-bit add
    # of 0x0000 + 0xFF00 = 0xFF00). No carry. neg_delta<0 -> w_hi -=1 = -1. ✓

    # WUPDT: positive grad with carry edge
    # w=(0, 0xFFFF), grad=-1, lr_shift=1 -> delta=-1, neg_delta=+1, w_lo=0xFFFF+1=0x10000
    # carry=1 -> w_hi=0+1=1. neg_delta>=0 -> no sign ext. Result (1, 0).
    h, l, _ = wupdt_ref([0], [-1], [-1], 1)
    assert h == [1] and l == [0], f"wupdt carry: h={h} l={l}"

    # CVT16: hi=0x0012, lo=0x3400 -> Q8 = 0x1234
    q = cvt16_ref([0x0012], [0x3400])
    assert q == [0x1234], f"cvt16: got {q}"

    # CVT16: negative
    # hi=-1 (0xFFFF), lo=-256 (0xFF00) -> Q8 bytes = (FF, FF) = 0xFFFF = -1
    q = cvt16_ref([-1], [-256])
    assert q == [-1], f"cvt16 neg: got {q}"

    # INITW first 3 from seed=887: Q8s should be [28, 101, -94] per our Python probe
    rng2 = LCG(RN_INIT)
    h, l, q8s = initw_ref(3, rng2)
    assert q8s == [28, 101, -94], f"initw q8: got {q8s}"
    assert h == [0, 0, -1], f"initw hi: got {h}"
    # lo values: 28*256=7168, 101*256=25856, -94*256=-24064
    assert l == [7168, 25856, -24064], f"initw lo: got {l}"

    print("self-test passed")


# ---- Test cases ----

def test_0_wupdt_basic():
    """Basic SGD: lr_shift=1 (no shift). Mix of positive/negative grads + weights."""
    w_hi = [0, 0, 0, 0]
    w_lo = [1000, 2000, -500, -1000]
    grad = [100, -200, 300, -50]
    lr_shift = 1
    new_hi, new_lo, new_grad = wupdt_ref(w_hi, w_lo, grad, lr_shift)
    return dict(name="UP_T0_WUPDT_BASIC", mode="wupdt",
                w_hi=w_hi, w_lo=w_lo, grad=grad, lr_shift=lr_shift,
                exp_hi=new_hi, exp_lo=new_lo, exp_grad=new_grad)


def test_1_wupdt_shifted():
    """Shifted SGD: lr_shift=4 (>>3). Exercises sign-ext path with non-zero w_hi."""
    w_hi = [0, 5, -1, 0]
    w_lo = [10000, 0, -32768, -1]
    grad = [8000, 800, 80, -8000]
    lr_shift = 4
    new_hi, new_lo, new_grad = wupdt_ref(w_hi, w_lo, grad, lr_shift)
    return dict(name="UP_T1_WUPDT_SHIFTED", mode="wupdt",
                w_hi=w_hi, w_lo=w_lo, grad=grad, lr_shift=lr_shift,
                exp_hi=new_hi, exp_lo=new_lo, exp_grad=new_grad)


def test_2_cvt16_basic():
    """Q16 -> Q8 middle-byte extraction. Positive, negative, zero, boundary."""
    w_hi = [0x0012, 0x0000, 0xFFFF & -1, 0x0000]
    w_lo = [0x3400, 0x8000, 0x0000, 0x0000]
    # Normalize w_hi values to signed 16-bit
    w_hi = [wrap16(v) for v in w_hi]
    q8 = cvt16_ref(w_hi, w_lo)
    return dict(name="UP_T2_CVT16_BASIC", mode="cvt16",
                w_hi=w_hi, w_lo=w_lo, exp_q8=q8)


def test_3_initw_rand():
    """Initialize 8 weights from seed=887. Validates RAND + INITW pipeline."""
    rng = LCG(RN_INIT)
    count = 8
    w_hi, w_lo, q8s = initw_ref(count, rng)
    return dict(name="UP_T3_INITW_RAND", mode="initw",
                count=count, seed=RN_INIT,
                exp_hi=w_hi, exp_lo=w_lo, exp_q8=q8s)


def test_4_wupdt_carry():
    """Carry + sign-extension edge cases."""
    # Element 0: w_lo near boundary, grad forces carry
    # Element 1: w_lo = 0xFFFF, grad=-1 -> neg_delta=+1 -> carry, w_hi+=1
    # Element 2: w_hi = 1, w_lo = 0, grad=1 -> neg_delta=-1 -> no carry but sign-ext
    # Element 3: w_hi = -1, w_lo = 0, grad=1 -> sign-ext with negative w_hi
    w_hi = [0, 0, 1, -1]
    w_lo = [0x7FFF, -1, 0, 0]
    grad = [-100, -1, 1, 1]
    lr_shift = 1
    # Normalize w_lo to signed 16-bit for Python
    w_lo = [wrap16(v) for v in w_lo]
    new_hi, new_lo, new_grad = wupdt_ref(w_hi, w_lo, grad, lr_shift)
    return dict(name="UP_T4_WUPDT_CARRY", mode="wupdt",
                w_hi=w_hi, w_lo=w_lo, grad=grad, lr_shift=lr_shift,
                exp_hi=new_hi, exp_lo=new_lo, exp_grad=new_grad)


def all_tests():
    return [test_0_wupdt_basic(), test_1_wupdt_shifted(),
            test_2_cvt16_basic(), test_3_initw_rand(),
            test_4_wupdt_carry()]


# ---- Emission ----

def emit_asm(path, results):
    lines = []
    lines.append("* ============================================================")
    lines.append("* updat_vectors.asm - UPDAT routines test vectors")
    lines.append("* Generated by gen_updat_vectors.py - do not hand-edit.")
    lines.append("* ============================================================")
    lines.append("")

    for idx, r in enumerate(results):
        name = r["name"]
        mode = r["mode"]
        lines.append(f"* Test {idx}: {name}  mode={mode}")

        if mode == "wupdt":
            cnt = len(r["w_hi"])
            lines.append(f"CNT_{idx}         FDB  {cnt}")
            lines.append(f"SHF_{idx}         FDB  {r['lr_shift']}")
            lines.append(f"WHI_{idx}         FDB  {','.join(str(v) for v in r['w_hi'])}")
            lines.append(f"WLO_{idx}         FDB  {','.join(str(v) for v in r['w_lo'])}")
            lines.append(f"GRAD_{idx}        FDB  {','.join(str(v) for v in r['grad'])}")
            lines.append(f"EXP_WHI_{idx}     FDB  {','.join(str(v) for v in r['exp_hi'])}")
            lines.append(f"EXP_WLO_{idx}     FDB  {','.join(str(v) for v in r['exp_lo'])}")
            lines.append(f"EXP_GRAD_{idx}    FDB  {','.join(str(v) for v in r['exp_grad'])}")
        elif mode == "cvt16":
            cnt = len(r["w_hi"])
            lines.append(f"CNT_{idx}         FDB  {cnt}")
            lines.append(f"WHI_{idx}         FDB  {','.join(str(v) for v in r['w_hi'])}")
            lines.append(f"WLO_{idx}         FDB  {','.join(str(v) for v in r['w_lo'])}")
            lines.append(f"EXP_Q8_{idx}      FDB  {','.join(str(v) for v in r['exp_q8'])}")
        elif mode == "initw":
            lines.append(f"CNT_{idx}         FDB  {r['count']}")
            lines.append(f"SEED_{idx}        FDB  {r['seed']}")
            lines.append(f"EXP_WHI_{idx}     FDB  {','.join(str(v) for v in r['exp_hi'])}")
            lines.append(f"EXP_WLO_{idx}     FDB  {','.join(str(v) for v in r['exp_lo'])}")
            lines.append(f"EXP_Q8_{idx}      FDB  {','.join(str(v) for v in r['exp_q8'])}")
        lines.append("")

    Path(path).write_text("\n".join(lines) + "\n")


def emit_log(path, results):
    lines = []
    lines.append("UPDAT reference generator log")
    lines.append("=" * 60)
    lines.append("")
    lines.append("RAND sequence from seed=887 (first 5 values):")
    rng = LCG(RN_INIT)
    for i in range(5):
        v = rng.rand()
        b = v & 0xFF
        q = b - 128
        lines.append(f"  call {i+1}: seed={v:5d}  byte={b:3d}  q8={q:+4d}")
    lines.append("")

    for idx, r in enumerate(results):
        lines.append(f"Test {idx}: {r['name']}  mode={r['mode']}")
        if r["mode"] == "wupdt":
            lines.append(f"  lr_shift = {r['lr_shift']} (shift_count = {r['lr_shift']-1})")
            lines.append(f"  w_hi     = {r['w_hi']}")
            lines.append(f"  w_lo     = {r['w_lo']}")
            lines.append(f"  grad     = {r['grad']}")
            lines.append(f"  exp_hi   = {r['exp_hi']}")
            lines.append(f"  exp_lo   = {r['exp_lo']}")
            lines.append(f"  exp_grad = {r['exp_grad']}  (zeroed)")
        elif r["mode"] == "cvt16":
            lines.append(f"  w_hi     = {r['w_hi']}")
            lines.append(f"  w_lo     = {r['w_lo']}")
            lines.append(f"  exp_q8   = {r['exp_q8']}")
        elif r["mode"] == "initw":
            lines.append(f"  count    = {r['count']}")
            lines.append(f"  seed     = {r['seed']}")
            lines.append(f"  exp_hi   = {r['exp_hi']}")
            lines.append(f"  exp_lo   = {r['exp_lo']}")
            lines.append(f"  exp_q8   = {r['exp_q8']}")
        lines.append("")

    Path(path).write_text("\n".join(lines) + "\n")


# ---- Main ----

def main():
    self_test()

    results = all_tests()

    out_dir = Path("tables")
    out_dir.mkdir(exist_ok=True)
    emit_asm(out_dir / "updat_vectors.asm", results)
    emit_log(out_dir / "updat_vectors.log", results)

    print(f"\nwrote tables/updat_vectors.asm ({len(results)} tests)")
    print("wrote tables/updat_vectors.log")
    print("\nSummary:")
    for r in results:
        if r["mode"] == "wupdt":
            print(f"  {r['name']}  n={len(r['w_hi'])} lr_shift={r['lr_shift']}")
        elif r["mode"] == "cvt16":
            print(f"  {r['name']}  n={len(r['w_hi'])}")
        elif r["mode"] == "initw":
            print(f"  {r['name']}  count={r['count']} seed={r['seed']}")


if __name__ == "__main__":
    main()
