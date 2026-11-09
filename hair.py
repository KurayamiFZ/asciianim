"""
ХАЙР  -  60 second ASCII piece, synced to the first minute of JMSN "Love Me".

Silent version: no music plays. Start your screen recorder, press SPACE, and
the 60 seconds run on the song's beat grid. Add the song afterwards in your
video editor / Instagram and line it up with the song's very first second
(the first heartbeat kick of the visuals is at 12.7 s, same as in the song).

Files in the SAME folder:
    hair.py                      (this file)
    hair_audio_analysis.json     (beat data measured from the song; optional)

Run:   pip install pygame numpy      (use pygame-ce if pygame won't install)
       python hayr.py

Keys:  SPACE start / pause   R restart   S screenshot   H show time   Esc quit
"""
import sys
import json
import math
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AUDIO_FILE = HERE / "hair_audio_60s.mp3"
ANALYSIS_FILE = HERE / "hair_audio_analysis.json"
WORD = "хайр"
DURATION = 60.0
SYNC_OFFSET = 0.0
USE_AUDIO = False          # set True (and keep hayr_audio_60s.mp3 here) to play the music too
START_ON_KEY = True        # window stays black until you press SPACE

# ---------------------------------------------------------------- layout
COLS, ROWS = 160, 48
CW, CH = 8, 14
W, H = COLS * CW, ROWS * CH            # 1280 x 672
SS = 2                                  # supersampling of the light field
GW, GH = COLS * SS, ROWS * SS
FONT_PX = 14
MONO = "consolas,menlo,dejavusansmono,couriernew,liberationmono,monospace"
TRAIL = 0.50

# ---------------------------------------------------------------- music grid (measured from the audio)
BAR0 = 12.72                            # first kick = bar downbeat
BPM = 80.55
BEAT = 60.0 / BPM
BAR = 4 * BEAT

RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
NR = len(RAMP) - 1
CIDX = {c: i for i, c in enumerate(RAMP)}

DEEP = np.array([0.50, 0.015, 0.05], np.float32)
RED = np.array([1.00, 0.08, 0.14], np.float32)
HOT = np.array([1.00, 0.42, 0.36], np.float32)
WHITE = np.array([1.00, 0.92, 0.88], np.float32)
COLD = np.array([0.55, 0.75, 1.00], np.float32)


# ---------------------------------------------------------------- small helpers
def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def vs(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def bump(t, a, b, c, d):
    return ease((t - a) / (b - a)) * (1 - ease((t - c) / (d - c)))


def kf(t, keys):
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t < t1:
            return v0 + (v1 - v0) * ease((t - t0) / (t1 - t0))
    return keys[-1][1]


def fbm(x, y, t):
    a = np.sin(x * 3.1 + t * 0.31 + np.sin(y * 2.3 - t * 0.23) * 1.7)
    b = np.sin(y * 4.7 - t * 0.27 + np.sin(x * 3.9 + t * 0.19) * 1.3)
    c = np.sin((x + y) * 7.9 + t * 0.41 + a * 2.0)
    return 0.5 + 0.5 * (0.5 * a + 0.3 * b + 0.2 * c)


def box(a, r, axis):
    if r <= 0:
        return a
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    p = np.pad(a, pad, mode="edge")
    c = np.cumsum(p, axis=axis, dtype=np.float64)
    zero = np.zeros_like(np.take(c, [0], axis=axis))
    c = np.concatenate([zero, c], axis=axis)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def blur2(a, rx, ry, it=2):
    for _ in range(it):
        a = box(box(a, rx, 1), ry, 0)
    return a


def _env(s):
    return np.exp(-s * 4.2) * (1 - np.exp(-s * 60.0))


_PEAK = float(np.max(_env(np.linspace(0, 1, 4000))))


def heartbeat(t):
    s = (t - BAR0) % BAR
    b1 = _env(s) / _PEAK
    b3 = 0.45 * _env((s - BAR / 2) % BAR) / _PEAK
    return float(max(b1, b3))


def beat_gain(t):
    g = 0.5 + 0.5 * ease((t - 6) / 8)
    g *= 1 - 0.5 * bump(t, 43.0, 44.5, 46.8, 48.0)
    g *= 1 + 0.45 * ease((t - 48.3) / 0.2)
    g *= 1 - 0.6 * ease((t - 57.6) / 1.2)
    return g


RINGS = [(BAR0 + k * BAR, s) for k, s in (
    (-3, .25), (-2, .30), (-1, .35), (0, .60), (1, .45), (2, .45), (3, .45), (4, 1.0),
    (5, .50), (6, .55), (7, .50), (8, 1.0), (9, .30), (10, .18), (11, .12),
    (12, 1.1), (13, .70), (14, 1.4), (15, .30))]
T_SHOCK = BAR0 + 4 * BAR      # 24.64
T_WORD = BAR0 + 8 * BAR       # 36.56  (word fully formed, whole-word flash)
T_RETURN = BAR0 + 12 * BAR    # 48.47  (bass enters, everything returns)
T_PEAK = BAR0 + 14 * BAR      # 54.43  (peak)


# ---------------------------------------------------------------- audio features
class Audio:
    def __init__(self):
        self.ok = False
        try:
            with open(ANALYSIS_FILE) as fh:
                d = json.load(fh)
            self.fps = float(d["fps"])
            self.a = {k: np.array(d[k], np.float32) for k in ("energy", "low", "mid", "high", "kick", "onset")}
            self.ok = True
        except Exception as e:  # works without the file, just less reactive
            print("note: analysis file not loaded (%s) - using the beat grid only" % e)

    def get(self, k, t):
        if not self.ok:
            return 0.0
        v = self.a[k]
        x = t * self.fps
        i = int(np.clip(x, 0, len(v) - 2))
        f = min(max(x - i, 0.0), 1.0)
        return float(v[i] * (1 - f) + v[i + 1] * f)

    def kick_env(self, t):
        if not self.ok:
            return 0.0
        v = self.a["kick"]
        i1 = int(np.clip(t * self.fps, 0, len(v) - 1))
        i0 = max(0, i1 - 16)
        seg = v[i0:i1 + 1]
        age = (i1 - np.arange(i0, i1 + 1)) / self.fps
        return float(np.max(np.clip(seg - 0.12, 0, None) * 1.2 * np.exp(-age * 7.0))) if len(seg) else 0.0


AUD = Audio()


# ---------------------------------------------------------------- one-time setup
def pygame_mask(wh, hh):
    import pygame
    pygame.font.init()
    names = "arial,segoeui,helveticaneue,dejavusans,liberationsans,freesans,notosans"
    size = hh
    f = pygame.font.SysFont(names, size, bold=False)
    while size > 10:
        f = pygame.font.SysFont(names, size, bold=False)
        w, h = f.size(WORD)
        if w <= 0.62 * wh and h <= 0.55 * hh:
            break
        size -= 2
    surf = f.render(WORD, True, (255, 255, 255))
    out = pygame.Surface((wh, hh))
    out.fill((0, 0, 0))
    out.blit(surf, ((wh - surf.get_width()) // 2, (hh - surf.get_height()) // 2))
    return pygame.surfarray.array3d(out)[:, :, 0].T.astype(np.float32) / 255


G = {}


def build(mask_fn=None):
    xs = (np.arange(GW, dtype=np.float32) + 0.5) * (CW / SS)
    ys = (np.arange(GH, dtype=np.float32) + 0.5) * (CH / SS)
    X = (((xs - W / 2) / H)[None, :] * np.ones((GH, 1), np.float32)).astype(np.float32)
    Y = (((ys - H / 2) / H)[:, None] * np.ones((1, GW), np.float32)).astype(np.float32)
    cx = (np.arange(COLS, dtype=np.float32) + 0.5) * CW
    cy = (np.arange(ROWS, dtype=np.float32) + 0.5) * CH
    XC = (((cx - W / 2) / H)[None, :] * np.ones((ROWS, 1), np.float32)).astype(np.float32)
    YC = (((cy - H / 2) / H)[:, None] * np.ones((1, COLS), np.float32)).astype(np.float32)
    G.update(X=X, Y=Y, RS=np.sqrt(X * X + Y * Y), XC=XC, YC=YC)
    rr, cc = np.mgrid[0:ROWS, 0:COLS]
    G["HSC"] = (((rr * 73856093) ^ (cc * 19349663) ^ 0x5bd1e995) % 1009 / 1009.0).astype(np.float32)
    G["RR"], G["CC"] = np.mgrid[0:GH, 0:GW]
    rg = np.random.default_rng(11)
    G["HASHF"] = rg.random((GH, GW)).astype(np.float32)

    # ---- word mask (field resolution)
    wh, hh = W // 2, H // 2
    hi = (mask_fn or pygame_mask)(wh, hh)
    ys_, xs_ = np.nonzero(hi > 0.5)
    hi = np.roll(hi, (hh // 2 - (ys_.min() + ys_.max()) // 2, wh // 2 - (xs_.min() + xs_.max()) // 2), (0, 1))
    vi = np.clip(((np.arange(GH) + 0.5) * (CH / SS) / H * hh).astype(int), 0, hh - 1)
    ui = np.clip(((np.arange(GW) + 0.5) * (CW / SS) / W * wh).astype(int), 0, wh - 1)
    mask = hi[np.ix_(vi, ui)]
    G["SHARP"] = blur2(mask, 1, 1, 1)
    G["GLOW"] = blur2(mask, 3, 2, 2)
    G["HALO"] = blur2(mask, 9, 5, 3)
    G["HALO"] /= max(G["HALO"].max(), 1e-6)
    G["GLOW"] /= max(G["GLOW"].max(), 1e-6)
    on_y, on_x = np.nonzero(mask > 0.5)
    G["WORD_BOX"] = (X[0, on_x.min()], X[0, on_x.max()], Y[on_y.min(), 0], Y[on_y.max(), 0])

    # ---- particles
    NP = 3200
    G["NP"] = NP
    # heart (implicit curve, same one the glowing heart uses), unit coordinates
    ang = rg.uniform(0, 2 * np.pi, 1300)
    lo, hi_r = np.zeros(1300), np.full(1300, 2.0)
    cy0 = 0.35
    for _ in range(36):
        mid = (lo + hi_r) / 2
        px, py = mid * np.cos(ang), cy0 + mid * np.sin(ang)
        f = (px * px + py * py - 1) ** 3 - px * px * py ** 3
        inside = f < 0
        lo = np.where(inside, mid, lo)
        hi_r = np.where(inside, hi_r, mid)
    rim_r = (lo + hi_r) / 2 * rg.uniform(0.985, 1.0, 1300)
    outline = np.stack([rim_r * np.cos(ang), cy0 + rim_r * np.sin(ang)], 1)
    cand = rg.uniform([-1.3, -1.1], [1.3, 1.4], (200000, 2))
    f = (cand[:, 0] ** 2 + cand[:, 1] ** 2 - 1) ** 3 - cand[:, 0] ** 2 * cand[:, 1] ** 3
    interior = cand[f < 0][:NP - 1300]
    hp = np.concatenate([outline, interior])[:NP]
    HU = np.stack([hp[:, 0], -(hp[:, 1] - 0.12)], 1).astype(np.float32)       # X = u*sz, Y = v*sz
    # word targets
    pick = rg.integers(0, len(on_x), NP)
    jx, jy = rg.uniform(-.5, .5, NP), rg.uniform(-.5, .5, NP)
    T = np.stack([X[0, on_x[pick]] + jx * CW / SS / H, Y[on_y[pick], 0] + jy * CH / SS / H], 1).astype(np.float32)
    # pair heart <-> word by angle so the flow is coherent (left side of the heart -> first letters)
    oh = np.argsort(np.arctan2(HU[:, 1], HU[:, 0]))
    ot = np.argsort(np.arctan2(T[:, 1], T[:, 0]))
    HUs, Ts = np.empty_like(HU), np.empty_like(T)
    HUs[ot] = HU[oh]
    Ts[ot] = T[ot]
    isint = (np.arange(NP) >= 1300)
    FI = np.empty(NP, bool)
    FI[ot] = isint[oh]
    G["HU"], G["T"], G["FI"] = HUs, Ts, FI
    ph = rg.random(NP).astype(np.float32)
    G["PR"] = np.stack([rg.random(NP), rg.random(NP), rg.random(NP), rg.random(NP)], 1).astype(np.float32)
    G["PH"] = ph
    x0, x1 = G["WORD_BOX"][0], G["WORD_BOX"][1]
    txn = np.clip((Ts[:, 0] - x0) / (x1 - x0 + 1e-6), 0, 1)
    G["DEP"] = (29.6 + 3.8 * txn + 0.4 * G["PR"][:, 0]).astype(np.float32)
    G["DET"] = (39.6 + 3.4 * G["PR"][:, 1]).astype(np.float32)
    G["RET"] = (49.0 + 2.6 * G["PR"][:, 0]).astype(np.float32)
    phi = rg.uniform(0, 2 * np.pi, NP)
    rad = 1.0 + 0.35 * rg.random(NP)
    G["E"] = np.stack([np.cos(phi) * 0.98 * rad, np.sin(phi) * 0.62 * rad], 1).astype(np.float32)
    # dust
    ND = 520
    G["ND"] = ND
    G["DUST"] = rg.random((ND, 5)).astype(np.float32)
    G["SPARKS"] = rg.random((60, 3)).astype(np.float32)


# ---------------------------------------------------------------- scene pieces
SZ0 = 0.235


def heart_size(t, beat):
    s = SZ0 * (0.62 + 0.38 * ease((t - 11) / 2.5))
    s *= 1 + 0.075 * beat
    age = t - T_SHOCK
    if age > 0:
        s *= 1 + 0.30 * (1 - math.exp(-age * 25)) * math.exp(-age * 2.6)
    s *= 1 - 0.05 * bump(t, 20.8, 21.4, 21.9, 22.6)
    s *= 1 + 0.10 * ease((t - T_PEAK) / 2.6) * (1 - ease((t - 57.0) / 1.0))
    return s


def heart_field(t, sz, hm, beat):
    X, Y = G["X"], G["Y"]
    hx, hy = X / sz, -Y / sz
    nC = fbm(X * 2.2, Y * 2.2, t)
    nD = fbm(X * 2.2 + 7, Y * 2.2 - 3, t * 0.9)
    wob = (1 - hm) * 0.42 + 0.012
    hxw, hyw = hx + wob * (nC - 0.5), hy + wob * (nD - 0.5)
    qc = np.sqrt(hxw * hxw + hyw * hyw) / 0.8 - 1.0
    hyh = hyw + 0.12
    fh = (hxw * hxw + hyh * hyh - 1.0) ** 3 - hxw * hxw * hyh ** 3
    qh = 0.5 * np.cbrt(fh)
    q = qc * (1 - hm) + qh * hm
    ins = np.clip(-q / 0.55, 0, 1)
    swirl = 0.5 + 0.5 * np.sin(3 * np.arctan2(hyw, hxw) + 7 * np.sqrt(hxw * hxw + hyw * hyw) - t * 1.4 + 3 * nC)
    inner = ins ** 0.7 * (0.10 + 0.30 * swirl) * (q < 0)
    diss = ease((t - 26.5) / 3.0)
    if diss > 0:
        keep = ((G["HASHF"] + t * 0.35) % 1.0) > diss * 0.85
        inner = inner * keep
    rim = np.exp(-(((q + 0.02) / 0.075) ** 2)) * (0.55 + 0.9 * fbm(X * 24, Y * 24, t * 1.5))
    halo = np.exp(-np.clip(q, 0, 9) * 3.6) * 0.12 * (0.5 + nC) * (q > 0)
    return rim, inner, halo


def particles(t, sz, beat):
    NP = G["NP"]
    PR, HU, T, E = G["PR"], G["HU"], G["T"], G["E"]
    DEP, DET, RET = G["DEP"], G["DET"], G["RET"]
    orb = ease((t - 26.4) / 1.6) * (1 - ease((t - 29.4) / 1.4))
    ang = 1.8 * orb * (t - 26.4)
    ca, sa = math.cos(ang), math.sin(ang)
    px, py = HU[:, 0] * sz, HU[:, 1] * sz
    scale = 1 + 0.38 * orb * (0.4 + PR[:, 3])
    hx, hy = (px * ca - py * sa) * scale, (px * sa + py * ca) * scale
    # stage 1 heart -> word
    e1 = vs((t - DEP) / 2.0)
    dx, dy = T[:, 0] - hx, T[:, 1] - hy
    ln = np.sqrt(dx * dx + dy * dy) + 1e-6
    off = (PR[:, 2] - 0.5) * 0.30 * np.sin(np.pi * e1)
    p1x = hx + dx * e1 + (-dy / ln) * off
    p1y = hy + dy * e1 + (dx / ln) * off
    # stage 2 word -> edge of the screen
    e2 = vs((t - DET) / 5.5)
    p2x, p2y = T[:, 0] + (E[:, 0] - T[:, 0]) * e2, T[:, 1] + (E[:, 1] - T[:, 1]) * e2
    # stage 3 edge -> heart
    e3 = vs((t - RET) / 3.0)
    p3x, p3y = E[:, 0] + (px - E[:, 0]) * e3, E[:, 1] + (py - E[:, 1]) * e3
    X = np.where(t < DET, p1x, np.where(t < RET, p2x, p3x))
    Y = np.where(t < DET, p1y, np.where(t < RET, p2y, p3y))
    al = vs((t - 16.0 - 2.4 * PR[:, 1]) / 1.2)
    al = np.where(t >= DET, 1 - 0.88 * e2, al)
    al = np.where(t >= RET, 0.12 + 0.88 * e3, al)
    al = al * (1 - vs((t - 57.0) / 1.3))
    flash = np.exp(-(((t - (DEP + 2.0)) / 0.35) ** 2))
    tw = 0.8 + 0.2 * np.sin(t * 5 + PR[:, 2] * 40)
    arrived = (e1 > 0.98) & (t < DET)
    spk = np.clip(np.sin(t * 3.0 + PR[:, 2] * 50), 0, 1) ** 3
    dimint = 1 - 0.62 * G["FI"] * (1 - e1)
    dimint = np.where(t >= RET, 1 - 0.70 * G["FI"], dimint)
    b = ((0.45 + 0.55 * PR[:, 3]) * (0.85 + 0.4 * beat) * tw + 0.7 * flash) * al * dimint
    b = np.where(arrived, b * (0.10 + 0.90 * spk), b)
    return X, Y, b, al


def dust(t, beat, high):
    D = G["DUST"]
    x = (D[:, 0] * 2 - 1) * 0.90 + 0.015 * np.sin(t * 0.5 + D[:, 4] * 6.28)
    y = ((D[:, 1] - t * (0.012 + 0.03 * D[:, 2])) % 1.0) - 0.5
    a = vs((t - 2.5 - 3.0 * D[:, 3]) / 3.0) * (1 - vs((t - 57.5) / 1.5))
    tw = np.clip(np.sin(t * (1.0 + 2.0 * D[:, 2]) + D[:, 4] * 40), 0, 1) ** 3
    b = a * (0.20 + 0.55 * tw + 0.35 * high * D[:, 2]) * (0.7 + 0.5 * beat)
    return x, y, b


def stamp_chars(idx, tint, li, X, Y, b, color, order=True):
    c = ((W / 2 + X * H) / CW).astype(int)
    r = ((H / 2 + Y * H) / CH).astype(int)
    ok = (c >= 0) & (c < COLS) & (r >= 0) & (r < ROWS) & (b > 0.06)
    if not ok.any():
        return
    c, r, bb = c[ok], r[ok], b[ok]
    col = color[ok] if color.ndim == 2 else np.broadcast_to(color, (len(bb), 3))
    keep = bb > li[r, c] * 0.55
    c, r, bb, col = c[keep], r[keep], bb[keep], col[keep]
    if len(bb) == 0:
        return
    o = np.argsort(bb)
    c, r, bb, col = c[o], r[o], bb[o], col[o]
    ch = np.where(bb < 0.18, CIDX["."], np.where(bb < 0.34, CIDX[":"], np.where(bb < 0.55, CIDX["+"],
                  np.where(bb < 0.80, CIDX["*"], CIDX["@"]))))
    idx[r, c] = ch
    tint[r, c] = col * np.clip(0.30 + 0.9 * bb, 0, 1.0)[:, None]
    li[r, c] = np.maximum(li[r, c], np.clip(bb, 0, 1))


def dir_char(dx, dy):
    a = np.degrees(np.arctan2(dy, dx)) % 180
    return np.where((a < 22.5) | (a >= 157.5), CIDX["-"],
                    np.where(a < 67.5, CIDX["\\"], np.where(a < 112.5, CIDX["|"], CIDX["/"])))


def stamp_path(idx, tint, li, pts, amt, color, alpha=1.0, head=True, pulse=None):
    n = len(pts)
    m = int(n * min(amt, 1.0))
    if m < 3 or alpha < 0.03:
        return
    p = pts[:m]
    d = np.gradient(p, axis=0)
    c = ((W / 2 + p[:, 0] * H) / CW).astype(int)
    r = ((H / 2 + p[:, 1] * H) / CH).astype(int)
    k = np.arange(m)
    b = alpha * (0.40 + 0.60 * np.exp(-(m - 1 - k) / (0.10 * n))) if head else np.full(m, alpha * 0.8)
    if pulse is not None:
        b = b + 0.7 * alpha * np.exp(-(((k / n) - pulse) / 0.05) ** 2)
    ok = (c >= 0) & (c < COLS) & (r >= 0) & (r < ROWS) & (b > li[np.clip(r, 0, ROWS - 1), np.clip(c, 0, COLS - 1)] * 0.9)
    c, r, b, d = c[ok], r[ok], b[ok], d[ok]
    if len(b) == 0:
        return
    idx[r, c] = dir_char(d[:, 0], d[:, 1])
    tint[r, c] = color[None, :] * np.clip(0.35 + 0.8 * b, 0, 1)[:, None]
    li[r, c] = np.maximum(li[r, c], np.clip(b, 0, 1))


def stamp_rays(idx, tint, li, cy, cx, reach, color, strength=1.0):
    for dist in range(1, reach + 1):
        fade = (1 - dist / (reach + 1)) * strength
        for dy, dx, ch in ((0, 1, "-"), (0, -1, "-"), (1, 0, "|"), (-1, 0, "|")):
            yy, xx = cy + dy * dist, cx + dx * dist * 2
            if 0 <= yy < ROWS and 0 <= xx < COLS and li[yy, xx] < fade:
                idx[yy, xx], li[yy, xx] = CIDX[ch], fade
                tint[yy, xx] = color * min(1.0, 0.3 + fade)
        if dist <= reach // 2:
            for dy, dx, ch in ((1, 1, "\\"), (-1, -1, "\\"), (1, -1, "/"), (-1, 1, "/")):
                yy, xx = cy + dy * dist, cx + dx * dist
                if 0 <= yy < ROWS and 0 <= xx < COLS and li[yy, xx] < fade * 0.7:
                    idx[yy, xx], li[yy, xx] = CIDX[ch], fade * 0.7  
                    tint[yy, xx] = color * min(1.0, 0.3 + fade * 0.7)
    if 0 <= cy < ROWS and 0 <= cx < COLS:
        idx[cy, cx], li[cy, cx] = CIDX["*"], 1.0
        tint[cy, cx] = np.maximum(color, 0.9 * WHITE)


# curves for veins / tendrils / roots / branches (built once)
def _curve_vein():
    s = np.linspace(0, 1, 90)
    return np.stack([0.012 * np.sin(s * 9), 0.06 - 0.36 * s], 1)


def _curve_tendril(side):
    # thin vine: rises from the top of the vein, arcs outward, droops and curls at the tip
    a = np.linspace(0, 1, 150)
    sg = -1 if side < 0 else 1
    x = sg * (0.03 + 0.30 * a ** 0.85)
    y = -0.29 - 0.11 * np.sin(a * np.pi * 0.85) + 0.035 * np.sin(a * 11) * a
    k = np.clip((a - 0.72) / 0.28, 0, 1)
    x = x - sg * 0.06 * k ** 2
    y = y + 0.13 * k ** 1.6
    return np.stack([x, y], 1)


def _curve_root(k):
    s = np.linspace(0, 1, 90)
    ends = [(-0.20, 0.40), (0.02, 0.45), (0.22, 0.38)]
    ex, ey = ends[k]
    return np.stack([ex * s + 0.05 * np.sin(s * 6 + k) * s, 0.27 + (ey - 0.27) * s], 1)


def _curve_branch(k, rg):
    a = rg.uniform(0, 2 * np.pi)
    s = np.linspace(0, 1, 70)
    r = 0.14 + 0.42 * s
    bend = rg.uniform(-0.9, 0.9)
    return np.stack([np.cos(a + bend * s) * r * 1.45, np.sin(a + bend * s) * r * 0.9], 1)


CURVES = {}


def curves():
    if not CURVES:
        rg = np.random.default_rng(5)
        CURVES["vein"] = _curve_vein()
        CURVES["tl"], CURVES["tr"] = _curve_tendril(-1), _curve_tendril(1)
        CURVES["roots"] = [_curve_root(k) for k in range(3)]
        CURVES["branches"] = [_curve_branch(k, rg) for k in range(9)]
    return CURVES


# ---------------------------------------------------------------- the frame
def bloom(img):
    b = np.clip(img - 0.55, 0, None)
    out = img
    for f, w_ in ((2, 0.26), (4, 0.22), (8, 0.18)):
        s_ = b.reshape(GH // f, f, GW // f, f, 3).mean((1, 3))
        for _ in range(2):
            s_ = (s_ * 2 + np.roll(s_, 1, 0) + np.roll(s_, -1, 0) + np.roll(s_, 1, 1) + np.roll(s_, -1, 1)) / 6
        out = out + w_ * np.repeat(np.repeat(s_, f, 0), f, 1)
    return out


def frame(t):
    X, Y, RS = G["X"], G["Y"], G["RS"]
    cv = curves()
    beat_grid = heartbeat(t) * beat_gain(t)
    kick = AUD.kick_env(t) if t > 12 else 0.0
    beat = max(beat_grid, 0.8 * kick)
    high, energy = AUD.get("high", t), AUD.get("energy", t)
    cold = 0.5 * bump(t, 42.2, 42.9, 47.2, 49.0)
    img = np.zeros((GH, GW, 3), np.float32)

    # --- red mist
    mist = 0.07 * ease((t - 8) / 5) * (1 - 0.5 * ease((t - 30) / 6)) + 0.04 * ease((t - 49) / 4)
    if mist > 0.002:
        a = 0.06 * t
        xr, yr = X * math.cos(a) - Y * math.sin(a), X * math.sin(a) + Y * math.cos(a)
        m = np.clip((fbm(xr * 1.6, yr * 1.9, t * 0.6) - 0.35) * 1.8, 0, 1) ** 2.0
        img += (m * mist * (0.8 + 0.5 * energy))[..., None] * DEEP

    # --- the first pixel / core
    grow = ease((t - 0.7) / 2.8)
    if grow > 0:
        r2 = X * X + Y * Y
        sig = 0.006 + 0.007 * grow + 0.022 * beat * (0.4 + 0.6 * grow)
        core = np.exp(-r2 / (sig * sig)) * grow * (1.0 - 0.55 * ease((t - 13) / 3))
        img += core[..., None] * HOT * 1.1
        img += (np.exp(-r2 / ((sig * 3) ** 2)) * 0.16 * grow)[..., None] * RED

    # --- shockwave rings on the bar downbeats
    for rt, rs in RINGS:
        age = t - rt
        if 0 < age < 1.8:
            R = age * 0.55
            amp = rs * math.exp(-age * 1.7) * (1 - math.exp(-age * 40))
            img += (amp * np.exp(-(((RS - R) / 0.022) ** 2)))[..., None] * RED * 1.1

    # --- the heart
    hf = ease((t - 10.5) / 3.0) * (1 - ease((t - 29.0) / 2.5)) + 0.38 * bump(t, 40.6, 41.2, 42.0, 43.2) \
        + ease((t - 50.4) / 3.6) * (1 - ease((t - 56.8) / 1.2))
    hf = min(hf, 1.15)
    sz = heart_size(t, beat)
    if hf > 0.01:
        hm = 1.0 if t > 30 else ease((t - 13) / 4.5)
        rim, inner, halo = heart_field(t, sz, hm, beat)
        img += (hf * rim * (0.9 + 0.55 * beat))[..., None] * (RED * 0.9 + HOT * 0.35 * beat)
        img += (hf * inner * (1 + 0.7 * beat))[..., None] * RED
        img += (hf * halo * (0.6 + 0.7 * beat))[..., None] * DEEP * 1.6

    # --- the word
    wa = kf(t, ((32.5, 0.0), (36.2, 0.85), (36.6, 1.0), (39.0, 0.95), (42.0, 0.45), (45.0, 0.28), (47.6, 0.10),
                (48.45, 0.25), (49.2, 1.0), (54.4, 1.25), (57.0, 1.0), (58.2, 0.45), (59.2, 0.2), (60.0, 0.0)))
    if wa > 0.005:
        sharp, glow, halo_w = G["SHARP"], G["GLOW"], G["HALO"]
        amp = 3.4 * bump(t, 42.6, 43.4, 44.8, 46.2)
        if amp > 0.05:
            sh = (amp * np.sin(Y * 38 + t * 2.2)).astype(int)
            cc2 = np.clip(G["CC"] + sh, 0, GW - 1)
            sharp, glow = sharp[G["RR"], cc2], glow[G["RR"], cc2]
        flash = 1.0 * math.exp(-max(t - T_WORD, 0) * 4.0) * (t >= T_WORD) + 0.9 * math.exp(-max(t - T_RETURN, 0) * 3.0) * (t >= T_RETURN) \
            + 0.8 * math.exp(-max(t - T_PEAK, 0) * 2.5) * (t >= T_PEAK)
        band = 0.0
        if 37.0 < t < 38.4:
            bx = -0.65 + 1.3 * ease((t - 37.0) / 1.2)
            band = 1.3 * np.exp(-(((X - bx) / 0.08) ** 2))
        spark = 1.0 + 0.55 * beat + flash
        wcol = RED * 0.7 + HOT * (0.45 + 0.4 * min(1.0, beat + flash))
        img += (wa * np.clip(sharp * 1.25 - 0.1, 0, 1) * spark)[..., None] * wcol
        img += (wa * np.clip(glow - sharp, 0, None) * 0.12)[..., None] * RED
        img += (wa * halo_w * 0.07)[..., None] * DEEP * 1.5
        if not np.isscalar(band):
            img += (wa * sharp * band)[..., None] * WHITE
        dis = ease((t - 58.8) / 1.2)
        if dis > 0:
            img *= (G["HASHF"] > dis)[..., None] * 0.9 + 0.1 * (1 - dis)

    # --- cold glint, brightness wave
    if 42.2 < t < 43.8:
        pos = -1.0 + 2.0 * ease((t - 42.3) / 1.2)
        d = X * 0.866 - Y * 0.5
        img += (0.65 * np.exp(-(((d - pos) / 0.035) ** 2)) * math.sin(math.pi * (t - 42.2) / 1.6))[..., None] * COLD
    if 52.8 < t < 54.6:
        pos = -0.9 + 1.8 * ease((t - 52.8) / 1.4)
        img += (0.42 * np.exp(-(((X - pos) / 0.10) ** 2)) * (1 - ease((t - 54.0) / 0.6)))[..., None] * (RED + 0.4 * HOT)

    # --- grade
    img = bloom(img)
    img = 1 - np.exp(-1.9 * np.clip(img, 0, None))
    img *= (1 - 0.35 * vs((RS - 0.30) / 0.55))[..., None]
    fade = ease(t / 0.8) * (1 - ease((t - 59.0) / 1.0))
    img *= fade
    cells = img.reshape(ROWS, SS, COLS, SS, 3).mean((1, 3))
    v = cells.max(-1)
    li = np.clip(v, 0, 1) ** 0.8
    ri = np.clip((li + (G["HSC"] - 0.5) * 0.06) * NR, 0, NR).astype(np.int32)
    ri = np.where(v < 0.012, 0, np.maximum(ri, 1))
    idx = ri.astype(np.int32)
    hue = cells / np.maximum(v, 1e-4)[..., None]
    tint = (hue * (0.35 + 0.65 * np.clip(v * 1.3, 0, 1))[..., None]).astype(np.float32)
    tint = np.where((v < 0.012)[..., None], 0, tint)
    li = li.astype(np.float32)

    # --- stamped characters: veins, tendrils, roots, branches
    gF = fade
    a_vein = ease((t - 4.0) / 3.0)
    if 4.0 < t < 17:
        stamp_path(idx, tint, li, cv["vein"], a_vein, RED, alpha=gF * (1 - ease((t - 12) / 4)),
                   pulse=((t * 0.22) % 1.0) if t > 5 else None)
        ta = ease((t - 7.0) / 2.2)
        for key in ("tl", "tr"):
            stamp_path(idx, tint, li, cv[key], ta, RED, alpha=gF * 0.9 * (1 - ease((t - 12) / 4)))
    if 19.0 < t < 29.5:
        for k, rt in enumerate(cv["roots"]):
            pts = rt.copy()
            pts[:, 1] += (sz - SZ0) * 0.0
            stamp_path(idx, tint, li, pts, ease((t - 19.0 - 0.2 * k) / 1.6), RED * 0.9,
                       alpha=gF * (1 - ease((t - 28) / 1.5)), pulse=((t - 20.0) * 0.8) % 1.0 if t > 20 else None)
    if 52.0 < t < 58.5:
        for k, br in enumerate(cv["branches"]):
            stamp_path(idx, tint, li, br, ease((t - 52.0 - 0.12 * k) / 1.8), RED * 0.95,
                       alpha=gF * (1 - ease((t - 56.5) / 1.5)))

    # --- particles
    px, py, pb, pal = particles(t, sz, beat)
    pcol = np.clip(RED[None, :] + (HOT - RED)[None, :] * np.clip(pb - 0.35, 0, 1)[:, None] * 1.6, 0, 1)
    pcol = pcol * (1 - cold) + COLD * cold * 0.9
    stamp_chars(idx, tint, li, px, py, pb * gF, pcol.astype(np.float32))
    dx_, dy_, db = dust(t, beat, high)
    dcol = np.broadcast_to(RED * (1 - cold) * 0.9 + HOT * 0.3 + COLD * cold * 0.9, (len(db), 3)).astype(np.float32)
    stamp_chars(idx, tint, li, dx_, dy_, db * gF, dcol)

    # --- sparkle rays on the brightest glints
    PR = G["PR"]
    gl = np.clip(np.sin(t * 2.3 + PR[:, 2] * 40), 0, 1) ** 10 * pal * (PR[:, 0] > 0.55)
    if t > 17:
        k = 4 + int(6 * ease((t - 48.0) / 4.0))
        top = np.argsort(gl)[::-1][:k]
        rcol = (RED * (1 - cold) + COLD * cold) * 0.4 + HOT * 0.6
        for i in top:
            if gl[i] < 0.45:
                break
            cx = int((W / 2 + px[i] * H) / CW)
            cy = int((H / 2 + py[i] * H) / CH)
            if 0 <= cx < COLS and 0 <= cy < ROWS:
                stamp_rays(idx, tint, li, cy, cx, int(3 + 6 * gl[i] * (0.6 + beat)), rcol.astype(np.float32), gF)
    # spark inside the second letter just before the return
    if 47.9 < t < T_RETURN + 0.1:
        x0, x1, y0, y1 = G["WORD_BOX"]
        sx, sy = x0 + 0.375 * (x1 - x0), 0.5 * (y0 + y1)
        cx, cy = int((W / 2 + sx * H) / CW), int((H / 2 + sy * H) / CH)
        stamp_rays(idx, tint, li, cy, cx, int(2 + 8 * ease((t - 47.9) / 0.57)), HOT, 1.0)
    # final sweeping ray across the word
    if 56.0 < t < 57.4:
        x0, x1, y0, y1 = G["WORD_BOX"]
        row = int((H / 2 + 0.5 * (y0 + y1) * H) / CH)
        head = int(ease((t - 56.0) / 1.1) * (COLS + 10)) - 5
        for c in range(max(0, head - 40), min(COLS, head + 1)):
            a = (1 - (head - c) / 40.0) * (1 - ease((t - 57.0) / 0.4))
            if a > 0.05:
                idx[row, c], tint[row, c] = CIDX["-" if c < head else "*"], WHITE * min(1.0, 0.3 + a)
                li[row, c] = 1.0
        if 0 <= head < COLS:
            stamp_rays(idx, tint, li, row, head, 5, WHITE, 1.0)

    # the word at 60 fades out: tint already multiplied by the field fade, apply to stamps too
    tint = (tint * (1 - ease((t - 59.0) / 1.0))).astype(np.float32)
    return idx, np.clip(tint, 0, 1), {"beat": beat}


# ---------------------------------------------------------------- window
def make_atlas():
    import pygame
    font = pygame.font.SysFont(MONO, FONT_PX, bold=True)
    atlas = []
    for ch in RAMP:
        s = pygame.Surface((CW, CH))
        g = font.render(ch, True, (255, 255, 255))
        s.blit(g, ((CW - g.get_width()) // 2, (CH - g.get_height()) // 2))
        atlas.append(pygame.surfarray.array3d(s)[:, :, 0].T.astype(np.float32) / 255)
    return (np.stack(atlas) * 255).astype(np.uint8)


def main():
    import pygame
    pygame.init()
    info = pygame.display.Info()
    k = 1.0
    if info.current_w > 0 and info.current_h > 0:
        k = min(1.0, 0.92 * info.current_w / W, 0.88 * info.current_h / H)
    size = (int(W * k), int(H * k))
    window = pygame.display.set_mode(size)
    pygame.display.set_caption("ХАЙР")
    print("building scene ...")
    build()
    atlas = make_atlas()
    shape = pygame.Surface((W, H))
    trail = pygame.Surface((W, H))
    screen = pygame.Surface((W, H))
    fadeval = (int(TRAIL * 255),) * 3
    font_small = pygame.font.SysFont(MONO, 16)

    audio_ok = False
    if USE_AUDIO:
        try:
            pygame.mixer.init(44100, -16, 2, 512)
            pygame.mixer.music.load(str(AUDIO_FILE))
            audio_ok = True
        except Exception as e:
            print("could not load audio (%s) - running silent" % e)

    def blur(surf, kk, rad):
        small = pygame.transform.smoothscale(surf, (W // kk, H // kk))
        if hasattr(pygame.transform, "gaussian_blur"):
            small = pygame.transform.gaussian_blur(small, max(1, rad // kk))
        return pygame.transform.smoothscale(small, (W, H))

    def glow_layer(surf, amount):
        layer = surf.copy()
        layer.fill((int(255 * min(max(amount, 0.0), 1.0)),) * 3, special_flags=pygame.BLEND_MULT)
        return layer

    state = {"t0": 0.0, "off": 0.0, "paused": False, "frozen": 0.0, "show": False, "fresh": False}

    def start(wait=False):
        trail.fill((0, 0, 0))
        if wait:
            pygame.display.set_caption("ХАЙР  -  press SPACE to start")
            print("Window is black on purpose. Start your recorder, then press SPACE.")
            state.update(t0=time.perf_counter(), off=0.0, paused=True, frozen=0.0, fresh=True)
        else:
            if audio_ok:
                pygame.mixer.music.play()
            pygame.display.set_caption("ХАЙР")
            state.update(t0=time.perf_counter(), off=0.0, paused=False, fresh=False)

    def now():
        return time.perf_counter() - state["t0"] + state["off"]

    start(START_ON_KEY)
    clock = pygame.time.Clock()
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    if state["paused"]:
                        state.update(t0=time.perf_counter() - state["frozen"], off=0.0, paused=False)
                        if state["fresh"]:
                            state["fresh"] = False
                            pygame.display.set_caption("ХАЙР")
                            if audio_ok:
                                pygame.mixer.music.play()
                        elif audio_ok:
                            pygame.mixer.music.unpause()
                    else:
                        state.update(frozen=now(), paused=True)
                        if audio_ok:
                            pygame.mixer.music.pause()
                elif e.key == pygame.K_r:
                    start(START_ON_KEY)
                elif e.key == pygame.K_h:
                    state["show"] = not state["show"]
                elif e.key == pygame.K_s:
                    pygame.image.save(screen, str(HERE / ("hayr_%d.png" % int(time.time()))))
        if not state["paused"]:
            if audio_ok and pygame.mixer.music.get_busy():
                pos = pygame.mixer.music.get_pos() / 1000.0
                if pos >= 0:
                    clk = time.perf_counter() - state["t0"]
                    state["off"] += 0.08 * ((pos - clk) - state["off"])
            t = now() + SYNC_OFFSET
        else:
            t = state["frozen"] + SYNC_OFFSET
        t = max(t, 0.0)

        if t <= DURATION + 0.5:
            idx, tint, inf = frame(min(t, DURATION))
            mask = atlas[idx.T].transpose(0, 3, 1, 2).reshape(COLS * CW, ROWS * CH)
            pygame.surfarray.blit_array(shape, np.repeat(mask[..., None], 3, 2))
            col = (tint * 255).astype(np.uint8)
            tintsurf = pygame.transform.scale(pygame.surfarray.make_surface(col.transpose(1, 0, 2)), (W, H))
            shape.blit(tintsurf, (0, 0), special_flags=pygame.BLEND_MULT)
            trail.fill(fadeval, special_flags=pygame.BLEND_MULT)
            trail.blit(shape, (0, 0), special_flags=pygame.BLEND_MAX)
            bt = inf["beat"]
        else:
            trail.fill((0, 0, 0))
            bt = 0.0
        screen.fill((0, 0, 0))
        screen.blit(trail, (0, 0))
        screen.blit(glow_layer(blur(trail, 4, 5), 0.60 + 0.35 * bt), (0, 0), special_flags=pygame.BLEND_ADD)
        screen.blit(glow_layer(blur(trail, 8, 22), 0.38 + 0.25 * bt), (0, 0), special_flags=pygame.BLEND_ADD)
        if state["show"]:
            screen.blit(font_small.render("t %05.2f" % t, True, (150, 140, 150)), (24, 20))
        window.blit(screen if size == (W, H) else pygame.transform.smoothscale(screen, size), (0, 0))
        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    sys.exit(main())