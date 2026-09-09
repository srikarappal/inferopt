"""Port of the dataviz skill's validate_palette.js -- same maths, same thresholds.

The host has no node. The skill's rule is "the color part is computable, so
compute it", not "run this particular binary", so the checks are reproduced
exactly rather than reasoned about: OKLab, Machado CVD matrices, the same
bands and floors, verified against the reference palette's published numbers.
"""
import math, itertools, sys

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}   # OKLCH L
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

MACHADO = {
    "protan": ((0.152286, 1.052583, -0.204868),
               (0.114503, 0.786281, 0.099216),
               (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968),
               (0.280085, 0.672501, 0.047413),
               (-0.011820, 0.042940, 0.968881)),
    "tritan": ((1.255528, -0.076749, -0.178779),
               (-0.078411, 0.930809, 0.147602),
               (0.004733, 0.691367, 0.303900)),
}


def hex2srgb(h):
    h = h.strip().lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def s2lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin(h):
    return [s2lin(c) for c in hex2srgb(h)]


def rel_lum(h):
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted((rel_lum(a), rel_lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_from_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def oklch(h):
    L, a, b = oklab_from_lin(lin(h))
    return L, math.hypot(a, b)


def simulate(h, kind):
    r, g, b = lin(h)
    M = MACHADO[kind]
    return [min(1.0, max(0.0, M[i][0] * r + M[i][1] * g + M[i][2] * b)) for i in range(3)]


def delta_e(h1, h2, kind=None):
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.dist(a, b)


def validate(palette, mode="light", surface=None, pairs="adjacent", quiet=False):
    surface = surface or SURFACE[mode]
    lo, hi = BAND[mode]
    rep, ok = [], True

    out = [(c, round(oklch(c)[0], 3)) for c in palette if not (lo <= oklch(c)[0] <= hi)]
    ok &= not out
    rep.append(("Lightness band", not out,
                f"outside {lo}-{hi}: {out}" if out else f"all {len(palette)} inside {lo}-{hi}"))

    low = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    ok &= not low
    rep.append(("Chroma floor", not low,
                f"below floor (reads gray): {low}" if low else f"all {len(palette)} >= {CHROMA_FLOOR}"))

    plist = (list(itertools.combinations(range(len(palette)), 2)) if pairs == "all"
             else [(i, i + 1) for i in range(len(palette) - 1)])
    worst_cvd, wc_pair = (math.inf, None)
    for i, j in plist:
        d = min(delta_e(palette[i], palette[j], "protan"),
                delta_e(palette[i], palette[j], "deutan"))
        if d < worst_cvd:
            worst_cvd, wc_pair = d, (palette[i], palette[j])
    state = "pass" if worst_cvd >= CVD_TARGET else ("FLOOR" if worst_cvd >= CVD_FLOOR else "fail")
    ok &= worst_cvd >= CVD_FLOOR
    rep.append((f"CVD separation ({pairs})", worst_cvd >= CVD_FLOOR,
                f"worst {worst_cvd:.1f} [{state}] {wc_pair}"))

    worst_nor, wn_pair = (math.inf, None)
    for i, j in plist:
        d = delta_e(palette[i], palette[j])
        if d < worst_nor:
            worst_nor, wn_pair = d, (palette[i], palette[j])
    ok &= worst_nor >= NORMAL_FLOOR
    rep.append(("Normal-vision floor", worst_nor >= NORMAL_FLOOR,
                f"worst {worst_nor:.1f} (>= {NORMAL_FLOOR}) {wn_pair}"))

    weak = [(c, round(contrast(c, surface), 2)) for c in palette if contrast(c, surface) < 3.0]
    rep.append(("Contrast vs surface", True,
                f"WARN sub-3:1 (needs visible labels / table): {weak}" if weak
                else f"all >= 3:1 vs {surface}"))

    if not quiet:
        print(f"\n  mode={mode}  surface={surface}  pairs={pairs}  n={len(palette)}")
        for name, good, detail in rep:
            print(f"    {'PASS' if good else 'FAIL'}  {name:26s} {detail}")
        print(f"    => {'OK' if ok else 'PALETTE FAILS'}")
    return ok, worst_cvd, worst_nor


if __name__ == "__main__":
    pal = sys.argv[1].split(",")
    mode = sys.argv[2] if len(sys.argv) > 2 else "light"
    pairs = sys.argv[3] if len(sys.argv) > 3 else "all"
    validate([c.strip() for c in pal], mode=mode, pairs=pairs)
