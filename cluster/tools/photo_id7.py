#!/usr/bin/env python3
"""photo_id7.py - ID7 Corsa as a literal 1:1 copy of BMW M GmbH's photo (option A): the photo IS the static layer.

    python3 cluster/tools/photo_id7.py                         # design/out/id7_Corsa_photo_static.png (+ live at the photo's values)
    python3 cluster/tools/photo_id7.py --sim fuel=0.4,temp=80  # design/out/id7_Corsa_photo_live.png

The static layer is refs/id7/Corsa.png (the photo resampled to 1920x720, not in git) with a true black and the moving
parts taken out: the fuel and temperature fills and their bars. The live parts are rebuilt from the same photo's pixels:
the fill under the level line (the photo's own, as photographed: fuel full, temperature 106 C), the bar moved and
turned onto the level line. Everything derived from the photo is written to design/out (ignored by git): the photo
never goes into the repo. Geometry used only to cut the photo: the user's point editor lines (B, C) and the levels.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import gen_id7 as G  # noqa: E402

W, H = 1920, 720
BLACK = 7                       # the photo's black level: at or below it -> 0 (the panel's own glow stays above it)
LABELS = {'fuel': [(257.5, 428.5, 270.5, 449.5), (268.5, 490.5, 312.5, 511.5), (322.5, 535.5, 343.5, 557.5)],
          'temp': [(1611.5, 438, 1659, 458.5), (1596.5, 489.5, 1645, 510), (1565, 533.5, 1601.5, 554)]}
BAR_BOX = {'fuel': (236, 427, 314, 457), 'temp': (1600, 494, 1672, 524)}
BAR_END = {'fuel': None, 'temp': 1626.0}       # temp: on the photo the bar ends between the two 0s of '100'


def load_photo():
    a = np.asarray(Image.open(os.path.join(ROOT, 'design', 'refs', 'id7', 'Corsa.png')).convert('RGB')).astype(np.float32)
    return np.clip((a - BLACK) * 255.0 / (255 - BLACK), 0, 255)


def poly_mask(poly, grow=0.0):
    m = Image.new('L', (W * 4, H * 4), 0)
    ImageDraw.Draw(m).polygon([(x * 4, y * 4) for x, y in poly], fill=255)
    m = m.resize((W, H), Image.BOX)
    if grow:
        m = m.filter(ImageFilter.MaxFilter(int(2 * grow) * 2 + 1))
    return np.asarray(m).astype(np.float32) / 255


def editor_line(cfg, key, name):
    e = next(i for i in cfg['elements'][key]['editor']['items'] if i['name'] == name)
    return np.array(G.fillet_path(e['pts'], e['radii'], 0.25)), e['width']


def level(cfg, key, v):
    """Level line at v (0..1) on the display: point on the photo's C and direction (fuel: as is; temp: mirrored)."""
    p, d = G.g10_level_point(cfg, key, v)
    if key == 'temp':
        ax = cfg['elements']['temp']['mirror_axis']
        return np.array([2 * ax - p[0], p[1]]), np.array([d[0], -d[1]])
    return np.array(p), np.array(d)


def below(p, d, X, Y):
    """1 under the level line (y down), 0 above, antialiased over a pixel."""
    s = d[0] * (Y - p[1]) - d[1] * (X - p[0])
    return np.clip(s + 0.5, 0, 1)


def photo_C(key):
    """C (the fill's lit edge) followed on the photo itself at 10x (ridge across the line), display px."""
    import trace_id7 as T
    seed = np.array(T.SEEDS_FUEL['C'], float)
    if key == 'fuel':
        img, org = T.load_x10('M8_fuel_x10'), (190, 400)
    else:
        img, org = T.load_x10('M8_temp_x10'), (1550, 400)
        seed[:, 0] = 2 * 952.5 - seed[:, 0]                    # a starting track only: the ridge finds the photo's line
    Pp, Wd, Pk, ok = T.ridge(img.min(2), org, T._ext(seed, 10.0), 2.4, min_peak=50)
    g = np.flatnonzero(Pk >= 0.45 * np.median(Pk[len(Pk) // 4:3 * len(Pk) // 4]))
    C, w = Pp[g[0]:g[-1] + 1], float(np.median(Wd[ok]))
    if key == 'temp':                                           # above 106 C the photo shows no C: the fuel's, mirrored,
        F, _ = photo_C('fuel')                                  # joined where the photo's own begins
        F = F.copy(); F[:, 0] = 2 * 952.5 - F[:, 0]
        top = C[np.argmin(C[:, 1])]
        j = int(np.argmin(np.linalg.norm(F - top, axis=1)))
        up = F[:j] + (top - F[j])
        C = np.vstack([up, C]) if np.linalg.norm(C[0] - top) < 1 else np.vstack([C, up[::-1]])
        if C[0, 1] > C[-1, 1]:
            C = C[::-1]
    return C, w


def extended(C, top=22.0, bottom=8.0):
    """C carried on straight past its ends (the fill reaches beyond the drawn edge)."""
    if C[0, 1] > C[-1, 1]:
        C = C[::-1]
    t0 = (C[0] - C[4]) / np.linalg.norm(C[0] - C[4]); t1 = (C[-1] - C[-5]) / np.linalg.norm(C[-1] - C[-5])
    return np.vstack([C[0] + t0 * np.linspace(top, 0.5, 12)[:, None], C, C[-1] + t1 * np.linspace(0.5, bottom, 6)[:, None]])


def fill_region(cfg, key):
    """Where the photo moves with the level: C and the lit fill inside it (from 1.2 px outside C's edge on the track
    side, to 60 px past C away from the track). The track, B, the ticks and the labels stay the photo's own."""
    C, wC = photo_C(key)
    C = extended(C)
    B, wB = editor_line(cfg, key, 'B')                          # only to tell the track side from the fill side
    # C's normal (from its tangent smoothed over ~8 px), turned away from B by a majority vote: one consistent side
    tg = np.gradient(C, axis=0)
    k = np.ones(33) / 33
    tg = np.stack([np.convolve(np.pad(tg[:, j], 16, mode='edge'), k, 'valid') for j in (0, 1)], 1)
    tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9
    nC = np.stack([-tg[:, 1], tg[:, 0]], 1)
    away = np.array([np.dot(nC[i], C[i] - B[int(np.argmin(np.linalg.norm(B - C[i], axis=1)))]) for i in range(len(C))])
    nC *= 1 if np.median(away) > 0 else -1
    near = C - nC * (wC / 2 + 2.2)
    far = C + nC * 60
    return np.vstack([near, far[::-1]])


def label_mask(key, photo):
    """The labels' glyph pixels (white) in their boxes, grown 1 px: they stay static, over everything."""
    m = np.zeros((H, W), np.float32)
    v = photo.min(-1)
    for x0, y0, x1, y1 in LABELS[key]:
        sl = (slice(int(y0), int(np.ceil(y1))), slice(int(x0), int(np.ceil(x1))))
        m[sl] = np.clip((v[sl] - 150) / 90, 0, 1)                # the glyph's own white only (its fringe follows the fill)
    return m


def bar_mask(key, photo):
    """The photographed bar: red pixels in its box; where a label hides it, its measured top and bottom edges are
    carried straight across (the bar's own colour there). Returns the mask and the bar's source image."""
    x0, y0, x1, y1 = BAR_BOX[key]
    r = photo[..., 0] - np.maximum(photo[..., 1], photo[..., 2])
    m = np.zeros((H, W), np.float32)
    m[y0:y1, x0:x1] = np.clip((r[y0:y1, x0:x1] - 20) / 60, 0, 1)
    src = photo.copy()
    hidden = np.zeros(W, bool)
    for a0, b0, a1, b1 in LABELS[key]:
        if a1 > x0 and a0 < x1 and b1 > y0 and b0 < y1:
            hidden[int(a0) - 1:int(np.ceil(a1)) + 1] = True
    cols, tops, bots = [], [], []
    for x in range(x0, x1):
        if hidden[x]:
            continue
        rr = np.flatnonzero(m[y0:y1, x] > 0.5)
        if len(rr) > 2:
            cols.append(x); tops.append(y0 + rr[0]); bots.append(y0 + rr[-1] + 1)
    if len(cols) > 5:
        kt, kb = np.polyfit(cols, tops, 1), np.polyfit(cols, bots, 1)
        col = np.median(photo[m > 0.8], 0)
        lo = max(x0, int(BAR_END[key])) if BAR_END[key] else x0
        for x in range(lo, x1):
            if not hidden[x]:
                continue
            t, b = np.polyval(kt, x + 0.5), np.polyval(kb, x + 0.5)
            ys = np.arange(H) + 0.5
            cov = np.clip(np.minimum(ys - t, b - ys) + 0.5, 0, 1)
            m[:, x] = np.maximum(m[:, x], cov)
            src[:, x] = src[:, x] * (1 - cov[:, None]) + col * cov[:, None]
    return m, src


def mid_tick(cfg, key, photo, reg):
    """The 1/2 (fuel) / 100 C (temperature) line across the dark track, from B to C: live (shown under the level only).
    Its band: the level line at the mid value, from C back across the track to B, 4 px wide; static there shows the
    track's own colour (the photo's track next to it, white marks left out)."""
    v = 0.5
    p, d = level(cfg, key, v)
    B, wB = editor_line(cfg, key, 'B')
    b = B[int(np.argmin(np.linalg.norm(B - p, axis=1)))]
    L = float(np.linalg.norm(b - p)) - (wB / 2 + 2.0)            # stops short of B: its tick there stays
    back = -d if np.dot(b - p, d) < 0 else d
    n = np.array([-d[1], d[0]])
    a0, a1 = p - back * 1.0, p + back * L
    band = poly_mask([tuple(a0 + n * 2.2), tuple(a1 + n * 2.2), tuple(a1 - n * 2.2), tuple(a0 - n * 2.2)])
    band = band * (1 - np.clip(reg, 0, 1))
    ring = poly_mask([tuple(a0 + n * 6), tuple(a1 + n * 6), tuple(a1 - n * 6), tuple(a0 - n * 6)]) - band
    sel = (ring > 0.9) & (photo.max(-1) < 90) & (reg < 0.1)
    track = np.median(photo[sel], 0) if sel.any() else np.zeros(3)
    return band, track


def build(cfg):
    photo = load_photo()
    X, Y = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    static = photo.copy()
    parts = {}
    for key in ('fuel', 'temp'):
        reg = poly_mask(fill_region(cfg, key))
        Cp, wCp = photo_C(key)                                  # C itself, whole length: live
        cm = Image.new('L', (W * 4, H * 4), 0)
        ImageDraw.Draw(cm).line([(x * 4, y * 4) for x, y in extended(Cp)], fill=255, width=int((wCp + 6.0) * 4), joint='curve')
        cmask = np.asarray(cm.resize((W, H), Image.BOX)).astype(np.float32) / 255
        reg = np.maximum(reg, cmask)
        # the coloured marks (hot red, cold blue, the reserve) stay the photo's: out of the region, outside the bar's box
        r_ = photo[..., 0] - np.maximum(photo[..., 1], photo[..., 2]); b_ = photo[..., 2] - photo[..., 0]
        col = ((r_ > 30) | (b_ > 30)).astype(np.float32)
        bx0, by0, bx1, by1 = BAR_BOX[key]
        col[by0 - 14:by1 + 14, bx0 - 14:bx1 + 14] = 0             # the bar and its reddish glow are live
        col = np.asarray(Image.fromarray((col * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))).astype(np.float32) / 255
        reg = np.maximum(reg * (1 - col), cmask)             # C's bluish fringe is not a mark
        lab = label_mask(key, photo)
        bar, bsrc = bar_mask(key, photo)
        # 'empty': what the photo shows in the region above its level (temperature, above 106 C), as a flat colour
        pt, dt = level(cfg, 'temp', (cfg['elements']['temp']['photo_value'] - 50) / 100.0)
        rt = poly_mask(fill_region(cfg, 'temp'))
        above = (rt > 0.9) & (below(pt, dt, X, Y) < 0.1) & (photo.max(-1) < 60)
        empty = np.median(photo[above], 0) if above.any() else np.zeros(3)
        barc = np.asarray(Image.fromarray((np.clip(bar, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))).astype(np.float32) / 255
        cut = np.clip(np.maximum(reg, barc) - lab, 0, 1)[..., None]
        static = static * (1 - cut) + empty * cut
        band, track = mid_tick(cfg, key, photo, reg)
        bc = np.clip(band - lab, 0, 1)[..., None]
        static = static * (1 - bc) + track * bc
        parts[key] = {'reg': np.maximum(reg, band), 'lab': lab, 'bar': bar, 'bsrc': bsrc, 'empty': empty}
    return photo, static, parts


def warp_bar(photo, mask, p0, d0, p1, d1):
    """The bar's pixels moved from level line (p0, d0) to (p1, d1): a rotation about p0 then a shift to p1."""
    a0, a1 = np.arctan2(d0[1], d0[0]), np.arctan2(d1[1], d1[0])
    rot = a1 - a0
    c, s = np.cos(rot), np.sin(rot)
    X, Y = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    # inverse map: target -> source
    dx, dy = X - p1[0], Y - p1[1]
    sx = p0[0] + c * dx + s * dy
    sy = p0[1] - s * dx + c * dy
    def samp(img):
        return G.g._bilinear(img, sx - 0.5, sy - 0.5) if hasattr(G.g, '_bilinear') else _bil(img, sx - 0.5, sy - 0.5)
    rgb = np.stack([_bil(photo[..., k] * mask, sx - 0.5, sy - 0.5) for k in range(3)], -1)
    al = _bil(mask, sx - 0.5, sy - 0.5)
    return rgb, al


def _bil(img, xs, ys):
    h, w = img.shape
    ok = (xs >= 0) & (ys >= 0) & (xs < w - 1) & (ys < h - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 2); y0 = np.clip(np.floor(ys).astype(int), 0, h - 2)
    fx, fy = np.clip(xs - x0, 0, 1), np.clip(ys - y0, 0, 1)
    v = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
         + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
    return np.where(ok, v, 0)


def live(cfg, photo, static, parts, tel):
    out = static.copy()
    X, Y = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    t = cfg['elements']['temp']
    v_ph = {'fuel': 1.0, 'temp': (t['photo_value'] - t['scale'][0]) / (t['scale'][1] - t['scale'][0])}
    val = {'fuel': tel.get('fuel', 1.0), 'temp': (tel.get('temp', t['photo_value']) - t['scale'][0]) / (t['scale'][1] - t['scale'][0])}
    for key in ('fuel', 'temp'):
        P = parts[key]
        v = float(np.clip(val[key], 0, 1))
        p, d = level(cfg, key, v)
        under = below(p, d, X, Y)
        src = photo
        if key == 'temp' and v > v_ph['temp']:
            # above 106 C the photo has no fill: there the fuel's photo, mirrored onto the temperature
            ax = t['mirror_axis']; sh = int(round(2 * ax))
            mir = np.zeros_like(photo)
            xs = np.arange(W); xm = sh - 1 - xs; ok = (xm >= 0) & (xm < W)
            mir[:, ok] = photo[:, xm[ok]]
            pt, dt = level(cfg, 'temp', v_ph['temp'])
            src = np.where((below(pt, dt, X, Y) > 0.5)[..., None], photo, mir)
        a = (np.clip(P['reg'] - P['lab'], 0, 1) * under)[..., None]
        out = out * (1 - a) + src * a
        p0, d0 = level(cfg, key, v_ph[key])
        rgb, al = warp_bar(P['bsrc'], P['bar'], p0, d0, p, d)
        al = (al * (1 - P['lab']))[..., None]
        out = out * (1 - al) + np.where(al > 0, rgb / np.maximum(al, 1e-3), 0) * al
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sim')
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(ROOT, 'layouts', 'id7', 'Corsa', 'config.json'), encoding='utf-8'))
    photo, static, parts = build(cfg)
    out_dir = os.path.join(ROOT, 'design', 'out'); os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(np.clip(static + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(out_dir, 'id7_Corsa_photo_static.png'))
    tel = {}
    if a.sim:
        for kv in a.sim.split(','):
            k, v = kv.split('='); tel[k] = float(v)
    img = live(cfg, photo, static, parts, tel)
    name = 'id7_Corsa_photo_live.png'
    Image.fromarray(np.clip(img + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(out_dir, name))
    print('design/out/id7_Corsa_photo_static.png, design/out/' + name)


if __name__ == '__main__':
    main()
