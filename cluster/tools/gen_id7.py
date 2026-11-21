#!/usr/bin/env python3
"""gen_id7.py - the iDrive 7 layouts (Strada, then Corsa), drawn from their config.json, one component at a time.

    python3 cluster/tools/gen_id7.py id7/Strada                       # layouts/id7/Strada/static.png
    python3 cluster/tools/gen_id7.py id7/Strada --compare             # + design/out/id7_Strada_compare.png
    python3 cluster/tools/gen_id7.py id7/Strada --sim fuel=0.4        # + design/out/id7_Strada_live.png

A design tool (Python, PC only). Shares the glyphs, the icons and the supersampling with gen_static.py (ID8). The
reference photo (cluster/design/refs/id7/<layout>.png) stays out of git.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_static as g  # noqa: E402

ROOT = g.ROOT
SS = g.SS


# ---------------------------------------------------------------- geometry

def px(cfg, pts):
    W, H = cfg['grid']
    return [(x * W, y * H) for x, y in pts]


def densify(pts, step=1.0):
    """A smooth curve through the measured points (Catmull-Rom), sampled about every 'step' px."""
    P = [pts[0]] + list(pts) + [pts[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = (np.array(P[i + k], float) for k in (-1, 0, 1, 2))
        n = max(2, int(np.linalg.norm(p2 - p1) / step))
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(tuple(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)))
    out.append(tuple(pts[-1]))
    return out


def offset(poly, d):
    """The polyline moved d px to its right-hand side seen from its start (inwards for a gauge drawn top -> bottom
    on the left)."""
    a = np.array(poly, float)
    t = np.gradient(a, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-t[:, 1], t[:, 0]], 1) * -1                  # (dy, -dx)
    return [tuple(p) for p in a + nrm * d]


def nearest(poly, p):
    a = np.array(poly, float)
    i = int(np.argmin(((a - np.array(p)) ** 2).sum(1)))
    return i, tuple(a[i])


def interp(stops, v):
    return float(np.interp(v, [s[0] for s in stops], [s[1] for s in stops]))


# ---------------------------------------------------------------- the fuel gauge

def fuel_paths(cfg):
    el = cfg['elements']['fuel']
    W, H = cfg['grid']
    a = px(cfg, el['A'])
    A = [a[0]] + densify(a[1:])                                 # its top: a short stroke parallel to the spike's right edge
    Boff = offset(densify(px(cfg, el['A'])[1:]), el['B_offset'])
    Boff = [p for p in Boff if p[0] <= el['B_end_x'] * W and p[1] >= 428]
    top = px(cfg, el['B_top']) + [Boff[0]]
    B = []
    for a, b in zip(top[:-1], top[1:]):                          # straight segments: the spike stays sharp
        k = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1])))
        B += [(a[0] + (b[0] - a[0]) * i / k, a[1] + (b[1] - a[1]) * i / k) for i in range(k)]
    B += Boff
    C = densify(px(cfg, el['C']))
    return A, B, C


def level_point(cfg, C, v):
    """Where level v (0..1) meets C, and its line's direction across the fill."""
    el = cfg['elements']['fuel']
    y = interp(el['levels']['y'], v)
    W = cfg['grid'][0]
    cands = [p for p in C if p[0] < 240]                       # the arc part (the bottom line is level 0)
    if v <= 0:
        p = min(C, key=lambda q: abs(q[0] - 238))
    else:
        p = min(cands, key=lambda q: abs(q[1] - y))
    s = interp(el['levels']['slope'], v)
    d = np.array([1.0, s]); d /= np.linalg.norm(d)
    return p, d


def _line(d, pts, w, fill):
    d.line([(x * SS, y * SS) for x, y in pts], fill=fill, width=int(round(w * SS)), joint='curve')
    r = w * SS / 2
    for x, y in (pts[0], pts[-1]):
        d.ellipse([x * SS - r, y * SS - r, x * SS + r, y * SS + r], fill=fill)


def draw_fuel_static(img, cfg):
    """Rail (A, B, the cap), track, ticks, the lit edge C (fading at its right end), reserve mark (supersampled canvas)."""
    el = cfg['elements']['fuel']
    W, H = cfg['grid']
    col = el['colors']
    A, B, C = fuel_paths(cfg)
    d = ImageDraw.Draw(img)
    # the track: between B (from where it turns down) and C
    i0, _ = nearest(B, C[0])
    track = B[i0:] + list(reversed(C[:nearest(C, B[-1])[0] + 1]))
    d.polygon([(x * SS, y * SS) for x, y in track], fill=tuple(col['track']))
    if 'track_ext' in el:                                       # past B's end, under C, over the reserve's left part
        d.polygon([(x * SS, y * SS) for x, y in px(cfg, el['track_ext'])], fill=tuple(col['track']))
    # the reserve mark, a horizontal gradient
    rp = px(cfg, el['reserve']['poly'])
    m = Image.new('L', img.size, 0)
    ImageDraw.Draw(m).polygon([(x * SS, y * SS) for x, y in rp], fill=255)
    xs = [p[0] for p in rp]
    xx = (np.arange(img.size[0]) + 0.5) / SS
    f = np.clip((xx - min(xs)) / (max(xs) - min(xs)), 0, 1)[None, :, None]
    c0, c1 = np.array(col['reserve'], float), np.array(col['reserve_end'], float)
    y0, y1 = int(min(p[1] for p in rp) * SS) - 2, int(max(p[1] for p in rp) * SS) + 2
    band = np.repeat(c0 + (c1 - c0) * f, y1 - y0, axis=0)
    fill = Image.fromarray(np.clip(band + 0.5, 0, 255).astype(np.uint8), 'RGB').convert('RGBA')
    img.paste(fill, (0, y0), m.crop((0, y0, img.size[0], y1)))
    lc = tuple(col['line'])
    _line(d, A, el['A_width'], lc)
    tail = el.get('B_tail', 0)
    _line(d, B[:len(B) - tail] if tail else B, el['B_width'], lc)
    if tail:                                                    # B's end: a thin point over the reserve
        pts = np.array(B[len(B) - tail - 1:], float)
        tg = np.gradient(pts, axis=0); tg /= np.linalg.norm(tg, axis=1, keepdims=True)
        nn = np.stack([-tg[:, 1], tg[:, 0]], 1)
        hw = el['B_width'] / 2 * np.linspace(1, 0.12, len(pts))[:, None]
        poly = list(pts + nn * hw) + list((pts - nn * hw)[::-1])
        d.polygon([(x * SS, y * SS) for x, y in poly], fill=lc)
    tr = el.get('trail')
    for t in el.get('triangles', []):                           # on B's inner side, apex towards C, with a trail
        p, dv = level_point(cfg, C, t['at'])
        i, q = nearest(B, p)
        tg = np.array(B[min(i + 1, len(B) - 1)]) - np.array(B[max(i - 1, 0)]); tg /= np.linalg.norm(tg)
        inw = np.array(p) - np.array(q); inw /= np.linalg.norm(inw)
        q = np.array(q) + inw * el['B_width'] / 2 * 0.8
        apex = q + inw * t['h']
        if tr:
            _line(d, [tuple(apex), tuple(p)], tr['width'], tuple(tr['color']))
        tri = [q + tg * t['half'], apex, q - tg * t['half']]
        d.polygon([(x * SS, y * SS) for x, y in tri], fill=lc)
    # C, its right end fading out
    fx0, fx1 = el['C_fade'][0] * W, el['C_fade'][1] * W
    m = Image.new('L', img.size, 0)
    _line(ImageDraw.Draw(m), C, el['C_width'], 255)
    a = np.asarray(m).astype(np.float32)
    a *= np.clip((fx1 - xx) / (fx1 - fx0), 0, 1)[None, :]
    layer = Image.new('RGBA', img.size, lc + (255,))
    img.paste(layer, (0, 0), Image.fromarray(a.astype(np.uint8)))


def draw_fuel_labels(img, cfg):
    el = cfg['elements']['fuel']
    W, H = cfg['grid']
    lb = el['labels']
    for it in lb['items']:
        tmp = Image.new('RGBA', (400, 100), (0, 0, 0, 0))
        g.draw_number(tmp, it['text'], 10, 50, it['h'], 32.5, lb['color'], align='left', sx=it.get('sx', 1.0))
        bb = tmp.getbbox()
        cx, cy = it['at'][0] * W, it['at'][1] * H
        part = tmp.crop(bb)
        img.alpha_composite(part, (int(round(cx - part.width / 2)), int(round(cy - part.height / 2))))
    ic = lb['icon']
    g.draw_pump(img, ic['at'][0] * W, ic['at'][1] * H, ic['size'][0], ic['size'][1], lb['color'])


def draw_fuel_level(img, cfg, v):
    """The live part: the fill lit from C up to level v, and the level bar there."""
    el = cfg['elements']['fuel']
    W, H = cfg['grid']
    col = el['colors']
    A, B, C = fuel_paths(cfg)
    v = float(np.clip(v, 0, 1))
    p, dvec = level_point(cfg, C, v)
    i = nearest(C, p)[0]
    far = 120
    top = tuple(np.array(p) + dvec * far)
    poly = C[i:] + [(C[-1][0] + 30, C[-1][1] - 4), (top[0], top[1])]
    x0, y0 = int(min(q[0] for q in poly)) - 2, int(min(q[1] for q in poly)) - 2
    x1, y1 = int(max(q[0] for q in poly)) + 3, int(max(q[1] for q in poly)) + 3
    m = Image.new('L', ((x1 - x0) * SS, (y1 - y0) * SS), 0)
    ImageDraw.Draw(m).polygon([((x - x0) * SS, (y - y0) * SS) for x, y in poly], fill=255)
    m = np.asarray(m.resize((x1 - x0, y1 - y0), Image.BOX)).astype(np.float32) / 255
    X, Y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    dist = g.poly_distance(C, X, Y)
    st = el['fill']['dist']
    k = np.interp(dist, [s[0] for s in st], [s[1] for s in st])
    fx0, fx1 = el['fill']['fade_x'][0] * W, el['fill']['fade_x'][1] * W
    k = k * np.clip((fx1 - X) / (fx1 - fx0), 0, 1) * m
    arr = np.asarray(img).astype(np.float32)
    sub = arr[y0:y1, x0:x1, :3]
    arr[y0:y1, x0:x1, :3] = sub + (np.array(col['fill'], np.float32) - sub) * k[..., None]
    img = Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8), 'RGBA')
    tr = el.get('trail')
    if tr:                                                      # the trails go on faintly over the lit fill
        lm = Image.new('L', (W * SS, H * SS), 0)
        ld = ImageDraw.Draw(lm)
        for t in el.get('triangles', []):
            if t['at'] >= v:
                continue
            q, dq = level_point(cfg, C, t['at'])
            _line(ld, [q, tuple(np.array(q) + dq * tr['fill_len'])], tr['width'], 255)
        la = np.asarray(lm.resize((W, H), Image.BOX)).astype(np.float32) / 255
        arr = np.asarray(img).astype(np.float32)
        arr[..., :3] += (255 - arr[..., :3]) * (la * tr['fill_gain'])[..., None] * (arr[..., :3].mean(-1, keepdims=True) > 30)
        img = Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8), 'RGBA')
    # the bar: along the level line from C, thick, its far end tapering and darker
    bl, bt = el['bar']['len'], el['bar']['thick']
    n = np.array([-dvec[1], dvec[0]])
    big = Image.new('RGBA', (img.size[0] * SS, img.size[1] * SS), (0, 0, 0, 0))
    bd = ImageDraw.Draw(big)
    steps = 40
    c0, c1 = np.array(col['bar'], float), np.array(col['bar_end'], float)
    for s in range(steps):
        t0, t1 = s / steps, (s + 1) / steps
        half = bt / 2 * (1 - el['bar'].get('taper', 0.35) * t1 ** 2)
        a0, a1 = np.array(p) + dvec * bl * t0, np.array(p) + dvec * bl * t1
        quad = [a0 + n * half, a1 + n * half, a1 - n * half, a0 - n * half]
        c = tuple(int(round(v_)) for v_ in c0 + (c1 - c0) * t1 ** 1.5) + (255,)
        bd.polygon([(q[0] * SS, q[1] * SS) for q in quad], fill=c)
    img.alpha_composite(big.resize(img.size, Image.LANCZOS))
    return img


# ---------------------------------------------------------------- build

def build_static(cfg):
    W, H = cfg['grid']
    big = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 255))
    draw_fuel_static(big, cfg)
    img = big.resize((W, H), Image.LANCZOS)
    draw_fuel_labels(img, cfg)
    return img


def draw_dynamic(img, cfg, tel):
    if 'fuel' in tel:
        img = draw_fuel_level(img, cfg, tel['fuel'])
        draw_fuel_labels(img, cfg)                              # the labels stay over the bar (as on the photo)
    return img


def compare(img, ref_path, out_path, box):
    ref = Image.open(ref_path).convert('RGB')
    a, b = ref.crop(box), img.convert('RGB').crop(box)
    k = 6
    w, h = (box[2] - box[0]) * k, (box[3] - box[1]) * k
    o = Image.new('RGB', (w * 3 + 20, h), (25, 25, 25))
    for j, im in enumerate((a, b, Image.blend(a, b, 0.5))):
        o.paste(im.resize((w, h), Image.LANCZOS), (j * (w + 10), 0))
    o.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('layout')
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--sim')
    a = ap.parse_args()
    cat, name = a.layout.split('/')
    folder = os.path.join(ROOT, 'layouts', cat, name)
    cfg = json.load(open(os.path.join(folder, 'config.json'), encoding='utf-8'))
    static = build_static(cfg)
    static.save(os.path.join(folder, 'static.png'))
    print('static: ' + os.path.relpath(os.path.join(folder, 'static.png'), ROOT))
    out_dir = os.path.join(ROOT, 'design', 'out')
    os.makedirs(out_dir, exist_ok=True)
    tel = {}
    if a.sim:
        for kv in a.sim.split(','):
            k, v = kv.split('=')
            tel[k] = float(v)
        live = draw_dynamic(static.copy(), cfg, tel)
        live.convert('RGB').save(os.path.join(out_dir, '%s_%s_live.png' % (cat, name)))
        print('live: design/out/%s_%s_live.png' % (cat, name))
    if a.compare:
        live = draw_dynamic(static.copy(), cfg, tel or {'fuel': 1.0})
        ref = os.path.join(ROOT, 'design', 'refs', cat, name + '.png')
        print('compare: ' + os.path.relpath(compare(live, ref, os.path.join(out_dir, '%s_%s_compare.png' % (cat, name)),
                                                    (115, 405, 315, 590)), ROOT))


if __name__ == '__main__':
    main()
