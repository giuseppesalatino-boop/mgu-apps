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

def fuel_paths(cfg, key='fuel'):
    el = cfg['elements'][key]
    W, H = cfg['grid']
    A = densify(px(cfg, el['A']), step=0.75)
    B_corner_px = px(cfg, [el['B_corner']])[0]
    if 'B' in el:                                               # measured on its own (M8 reference)
        B = [B_corner_px] + densify(px(cfg, el['B']), step=0.75)
    else:                                                       # A's parallel
        B_raw = offset(A, el['B_offset'])
        B_body = [p for p in B_raw if p[0] <= el['B_end_x'] * W and p[1] >= B_corner_px[1] + 2.0]
        B = [B_corner_px] + B_body
    C = densify(px(cfg, el['C']), step=0.75)
    return A, B, C


def level_point(cfg, C, v, key='fuel'):
    """Where level v (0..1) meets C, and its line's direction across the fill."""
    el = cfg['elements'][key]
    y = interp(el['levels']['y'], v)
    W = cfg['grid'][0]
    lv = el['levels']
    cands = [p for p in C if p[0] < lv.get('arc_max_x', 240)]  # the arc part (the bottom line is level 0)
    if v <= 0:
        p = min(C, key=lambda q: abs(q[0] - lv.get('x0', 238)))
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


def _strip(d, pts, w0, w1, c0, c1, steps=None):
    """A line from pts[0] to pts[-1] whose width and colour (RGBA) go from w0/c0 to w1/c1 (supersampled draw)."""
    a, b = np.array(pts[0], float), np.array(pts[-1], float)
    v = b - a; L = np.linalg.norm(v); v /= L; nn = np.array([-v[1], v[0]])
    steps = steps or max(8, int(L * 2))
    for s in range(steps):
        t0, t1 = s / steps, (s + 1) / steps
        hw = (w0 + (w1 - w0) * (t0 + t1) / 2) / 2
        p0, p1 = a + v * L * t0, a + v * L * t1
        quad = [p0 + nn * hw, p1 + nn * hw, p1 - nn * hw, p0 - nn * hw]
        c = tuple(int(round(c0[k] + (c1[k] - c0[k]) * (t0 + t1) / 2)) for k in range(len(c0)))
        d.polygon([(q[0] * SS, q[1] * SS) for q in quad], fill=c)


def draw_fuel_static(img, cfg, trails=True, level=None, key='fuel'):
    """The fixed part (supersampled canvas): the reserve, the thin line A, the thicker line B with its hook and its thin
    end, the triangles on B and the grey trails fading towards the labels."""
    el = cfg['elements'][key]
    W, H = cfg['grid']
    col = el['colors']
    A, B, C = fuel_paths(cfg, key)
    d = ImageDraw.Draw(img)
    # the reserve mark, a horizontal gradient (unless traced in layers: then drawn below, over the lines)
    rp = px(cfg, el['reserve']['poly'])
    layered = bool(el['reserve'].get('layers'))
    hidden = el['reserve'].get('hidden')                        # temp: its own traced overlay takes its place
    m = Image.new('L', img.size, 0)
    ImageDraw.Draw(m).polygon([(x * SS, y * SS) for x, y in rp], fill=255)
    xs = [p[0] for p in rp]
    xx = (np.arange(img.size[0]) + 0.5) / SS
    f = np.clip((xx - min(xs)) / (max(xs) - min(xs)), 0, 1)[None, :, None]
    c0, c1 = np.array(col['reserve'], float), np.array(col['reserve_end'], float)
    y0, y1 = int(min(p[1] for p in rp) * SS) - 2, int(max(p[1] for p in rp) * SS) + 2
    band = np.repeat(c0 + (c1 - c0) * f, y1 - y0, axis=0)
    fill = Image.fromarray(np.clip(band + 0.5, 0, 255).astype(np.uint8), 'RGB').convert('RGBA')
    if not layered and not hidden:
        img.paste(fill, (0, y0), m.crop((0, y0, img.size[0], y1)))
    lc = tuple(col['line'])
    tr = el['trail']
    for t in (el.get('trails', []) if trails else []):         # grey, fading out towards the label
        pts = px(cfg, t['pts'])
        _strip(d, pts, t['width'], t['width'] * 0.7, tuple(tr['c0']), tuple(tr['c1']))
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
    bz = el.get('B_zone')
    if bz:                                                      # B's top part in the zone's colour (temp: hot)
        seg = [q for q in B if q[1] <= bz['y_max'] * H]
        if len(seg) > 1:
            _line(d, seg, el['B_width'], tuple(bz['color']))
    tp = el.get('top')
    if tp and not tp.get('hidden'):                             # the '1': the wedge towards the label and the barb
        _strip(d, px(cfg, tp['wedge']), tp['w0'], tp['w1'], tuple(tp['c0']), tuple(tp['c1']))
        _line(d, px(cfg, tp['barb']), tp['barb_width'], tuple(tp.get('barb_color', lc)))
    sh = el['reserve'].get('shade')
    if layered:                                                 # the 10x vector trace: flat layers, lowest first
        for lay in el['reserve']['layers']:
            for poly in lay['polys']:
                d.polygon([(x * SS, y * SS) for x, y in px(cfg, poly)], fill=tuple(lay['color']))
    elif sh and not hidden:
        d.polygon([(x * SS, y * SS) for x, y in px(cfg, sh['pts'])], fill=tuple(sh['color']))
    for t in el.get('triangles', []):                           # on B's inner side, apex towards C
        p, dvec = level_point(cfg, C, t['at'], key)
        p_arr = np.array(p, dtype=float)
        _, q = nearest(B, p)
        q_arr = np.array(q, dtype=float)

        # Intersezione della linea di livello (passante per p con direzione -dvec) con il bordo interno di B
        Bi = offset(B, el['B_width'] / 2.0)
        _, qb = nearest(Bi, p - dvec * np.linalg.norm(p_arr - q_arr))
        qb_arr = np.array(qb, dtype=float)

        # 1) Tacca bianca che attraversa l'intera fascia grigia da B fino a C
        # (solo sotto il livello, insieme alla fascia grigia: nel layer fisso no)
        if level is not None and t['at'] < level:
            tick_w = t.get('line_w', 2.2)
            _line(d, [tuple(qb_arr), tuple(p_arr)], tick_w, lc)

        # 2) Invito triangolare su B (solo se h > 0, es. su 1/2)
        if t.get('h', 0) > 0:
            idx = nearest(Bi, qb_arr)[0]
            tg = np.array(Bi[min(idx + 1, len(Bi) - 1)]) - np.array(Bi[max(idx - 1, 0)])
            tg /= np.linalg.norm(tg) + 1e-9
            inw = p_arr - qb_arr
            inw /= np.linalg.norm(inw) + 1e-9
            apex = qb_arr + inw * t['h']
            tri = [tuple(qb_arr - tg * t['half']), tuple(apex), tuple(qb_arr + tg * t['half'])]
            d.polygon([(x * SS, y * SS) for x, y in tri], fill=lc)


def _rp_mask(cfg, key='fuel'):
    """The reserve mark's shape at grid size (the track and C keep off it)."""
    W, H = cfg['grid']
    m = Image.new('L', (W * SS, H * SS), 0)
    ImageDraw.Draw(m).polygon([(x * SS, y * SS) for x, y in px(cfg, cfg['elements'][key]['reserve']['poly'])], fill=255)
    return m.resize((W, H), Image.BOX)


def _below(cfg, p, dvec, X, Y):
    """1 under the level line through p (direction dvec), 0 above, anti-aliased over a pixel."""
    s = dvec[0] * (Y - p[1]) - dvec[1] * (X - p[0])
    return np.clip(s + 0.5, 0, 1)


def draw_fuel_labels(img, cfg, key='fuel'):
    el = cfg['elements'][key]
    if el['labels'].get('layers'):                              # traced at 10x: the photo's own glyphs
        draw_layers(img, cfg, el['labels']['layers'])
        return
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


def draw_fuel_level(img, cfg, v, key='fuel'):
    """The live part: the fill lit from C up to level v, and the level bar there."""
    el = cfg['elements'][key]
    W, H = cfg['grid']
    col = el['colors']
    A, B, C = fuel_paths(cfg, key)
    v = float(np.clip(v, 0, 1))
    p, dvec = level_point(cfg, C, v, key)
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
    sy = el['fill'].get('dist_scale_y')
    if sy:                                                      # the lit band widens towards the bottom
        dist = dist / np.interp(Y, [s[0] for s in sy], [s[1] for s in sy])
    k = np.interp(dist, [s[0] for s in st], [s[1] for s in st])
    gy = el['fill'].get('gain_y')
    if gy:                                                      # dimmer rows (temp: towards the bottom)
        k = k * np.interp(Y, [s[0] for s in gy], [s[1] for s in gy])
    fx0, fx1 = el['fill']['fade_x'][0] * W, el['fill']['fade_x'][1] * W
    k = k * np.clip((fx1 - X) / (fx1 - fx0), 0, 1) * m * _below(cfg, p, dvec, X, Y)
    # the grey track between B and C and C itself, both only under the level
    big = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 0))
    bd = ImageDraw.Draw(big)
    Bi = offset(B, el['B_width'] / 2)
    i0 = nearest(Bi, C[0])[0]
    trk = Bi[i0:] + list(reversed(C[:nearest(C, B[-1])[0] + 1]))
    bd.polygon([(x * SS, y * SS) for x, y in trk], fill=tuple(col['track']) + (255,))
    if 'track_ext' in el:
        bd.polygon([(x * SS, y * SS) for x, y in px(cfg, el['track_ext'])], fill=tuple(col['track']) + (255,))
    _line(bd, C, el['C_width'], tuple(col['line']) + (255,))
    lay = np.asarray(big.resize((W, H), Image.LANCZOS)).astype(np.float32)
    XX, YY = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    fx0c, fx1c = el['C_fade'][0] * W, el['C_fade'][1] * W
    lay[..., 3] *= _below(cfg, p, dvec, XX, YY) * np.clip((fx1c - XX) / (fx1c - fx0c), 0, 1)
    lay[..., 3] *= 1 - np.asarray(_rp_mask(cfg, key)).astype(np.float32) / 255
    img.alpha_composite(Image.fromarray(np.clip(lay + 0.5, 0, 255).astype(np.uint8), 'RGBA'))
    arr = np.asarray(img).astype(np.float32)
    sub = arr[y0:y1, x0:x1, :3]
    tex = el['fill'].get('layers')
    if tex:                                                     # the photographed fill (10x trace), cut at the level
        T = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        draw_layers(T, cfg, tex)
        T = np.asarray(T).astype(np.float32)[y0:y1, x0:x1]
        a = T[..., 3:] / 255 * (m * _below(cfg, p, dvec, X, Y))[..., None]
        arr[y0:y1, x0:x1, :3] = sub * (1 - a) + T[..., :3] * a
    else:
        arr[y0:y1, x0:x1, :3] = sub + (np.array(col['fill'], np.float32) - sub) * k[..., None]
    img = Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8), 'RGBA')
    tr = el.get('trail')
    if tr and not tex:                                                      # the trails go on faintly over the lit fill
        lm = Image.new('L', (W * SS, H * SS), 0)
        ld = ImageDraw.Draw(lm)
        for t in el.get('triangles', []):
            if t['at'] >= v:
                continue
            q, dq = level_point(cfg, C, t['at'], key)
            _line(ld, [q, tuple(np.array(q) + dq * tr['fill_len'])], tr['width'], 255)
        la = np.asarray(lm.resize((W, H), Image.BOX)).astype(np.float32) / 255
        arr = np.asarray(img).astype(np.float32)
        arr[..., :3] += (255 - arr[..., :3]) * (la * tr['fill_gain'])[..., None] * (arr[..., :3].mean(-1, keepdims=True) > 30)
        img = Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8), 'RGBA')
    # the bar: along the level line from the rail (B's inner edge), thick, its far end softening out (close-up photo)
    if el['bar'].get('layers'):                                 # the photographed bar (10x trace), at this level
        n = np.array([-dvec[1], dvec[0]])
        big = Image.new('RGBA', (img.size[0] * SS, img.size[1] * SS), (0, 0, 0, 0))
        bd = ImageDraw.Draw(big)
        P0 = np.array(p, float)
        for lay in el['bar']['layers']:
            for poly in lay['polys']:
                q = [P0 + dvec * u + n * w_ for u, w_ in poly]
                bd.polygon([(x * SS, y * SS) for x, y in q], fill=tuple(lay['color']) + (255,))
        img.alpha_composite(big.resize(img.size, Image.LANCZOS))
        return img
    bl, bt = el['bar']['len'], el['bar']['thick']
    n = np.array([-dvec[1], dvec[0]])
    if el['bar'].get('from') == 'C':                            # from the lit edge (M8 reference)
        start = np.array(p, dtype=float)
    else:                                                       # from the rail
        q = np.array(nearest(offset(B, el['B_width'] / 2), p)[1])
        start = np.array(p) + dvec * float(np.dot(q - np.array(p), dvec))
        bl += float(np.linalg.norm(np.array(p) - start))
    big = Image.new('RGBA', (img.size[0] * SS, img.size[1] * SS), (0, 0, 0, 0))
    bd = ImageDraw.Draw(big)
    steps = 60
    c0, c1 = np.array(col['bar'], float), np.array(col['bar_end'], float)
    fo = el['bar'].get('fade', 0.4)
    for s in range(steps):
        t0, t1 = s / steps, (s + 1) / steps
        half = bt / 2 * (1 - el['bar'].get('taper', 0.35) * t1 ** 2)
        a0, a1 = start + dvec * bl * t0, start + dvec * bl * t1
        quad = [a0 + n * half, a1 + n * half, a1 - n * half, a0 - n * half]
        al = 1 - float(g.smooth(1 - fo, 1.0, (t0 + t1) / 2))
        c = tuple(int(round(v_)) for v_ in c0 + (c1 - c0) * t1 ** 1.5) + (int(round(255 * al)),)
        bd.polygon([(q_[0] * SS, q_[1] * SS) for q_ in quad], fill=c)
    img.alpha_composite(big.resize(img.size, Image.LANCZOS))
    return img


# ---------------------------------------------------------------- the user's point editor (lines and fillets, shapes)

def fillet_path(pts, radii, step=0.25):
    """The polyline through pts with a circular fillet of radius r at each interior vertex (clamped to fit), sampled."""
    P = np.array(pts, float); n = len(P); out = [P[0]]
    for i in range(1, n - 1):
        u = P[i] - P[i - 1]; w = P[i + 1] - P[i]; lu = np.linalg.norm(u); lw = np.linalg.norm(w); u /= lu; w /= lw
        th = np.arccos(np.clip(u @ w, -1, 1)); r = float(radii[i - 1] or 0)
        if r <= 0 or th < 1e-4:
            out.append(P[i]); continue
        tl = r * np.tan(th / 2); tmax = min(lu if i == 1 else lu / 2, lw if i == n - 2 else lw / 2)
        if tl > tmax:
            tl = tmax; r = tl / np.tan(th / 2)
        T1, T2 = P[i] - u * tl, P[i] + w * tl
        cr = np.sign(u[0] * w[1] - u[1] * w[0]); C = T1 + np.array([-u[1], u[0]]) * r * cr
        a0 = np.arctan2(*(T1 - C)[::-1]); a1 = np.arctan2(*(T2 - C)[::-1]); da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
        m = max(2, int(abs(da) * r / step))
        out += [C + r * np.array([np.cos(a), np.sin(a)]) for a in a0 + da * np.linspace(0, 1, m)]
    out.append(P[-1])
    return [tuple(q) for q in out]


def _hex(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def editor_items(cfg, key, which):
    ed = cfg['elements'][key].get('editor')
    if not ed:
        return []
    return [e for e in ed['items'] if (e['name'] == 'C') == (which == 'live')]


def draw_editor(img, cfg, key, which='static', clip=None):
    """The editor's items (static: all but C; live: C), supersampled; clip = (p, d): only under that level line."""
    W, H = cfg['grid']
    big = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    for e in editor_items(cfg, key, which):
        col = _hex(e.get('color') or '#ffffff') + (255,)
        if e['kind'] == 'forma':
            d.polygon([(x * SS, y * SS) for x, y in e['pts']], fill=col)
            continue
        pts = fillet_path(e['pts'], e['radii'])
        if e['kind'] == 'tacca':
            d.line([(x * SS, y * SS) for x, y in pts], fill=col, width=max(1, int(round(e['width'] * SS))))
        else:
            _line(d, pts, e['width'], col)
    lay = np.asarray(big.resize((W, H), Image.LANCZOS)).astype(np.float32)
    if clip is not None:
        XX, YY = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        lay[..., 3] *= _below(cfg, clip[0], clip[1], XX, YY)
    img.alpha_composite(Image.fromarray(np.clip(lay + 0.5, 0, 255).astype(np.uint8), 'RGBA'))
    return img


# ---------------------------------------------------------------- a fuel-type gauge rebuilt at 10x (trace_id7 fuel10)

def rebuilt(cfg, key):
    return 'extras' in cfg['elements'].get(key, {})


def g10_level_point(cfg, key, v):
    """The level line at v (0..1): its point on C (by arc position) and its direction."""
    el = cfg['elements'][key]
    C = np.array(densify(px(cfg, el['C']), step=0.25))
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    lv = el['levels']                                           # each scale line's point on C -> its arc on C
    ar = [float(arc[int(np.argmin(np.linalg.norm(C - np.array(q[1:]), axis=1)))]) for q in lv['pts']]
    a = np.interp(v, [q[0] for q in lv['pts']], ar)
    p = C[int(np.argmin(np.abs(arc - a)))]
    k = np.interp(v, [q[0] for q in lv['slope']], [q[1] for q in lv['slope']])
    d = np.array([1.0, k])
    return p, d / np.linalg.norm(d)


def g10_static(img, cfg, key):
    """A and B (measured widths and colours), the static white marks, the reserve (supersampled, then onto img)."""
    W, H = cfg['grid']
    el = cfg['elements'][key]
    big = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    _line(d, densify(px(cfg, el['A']), 0.5), el['A_width'], tuple(el['colors']['A']) + (255,))
    _line(d, densify(px(cfg, el['B']), 0.5), el['B_width'], tuple(el['colors']['line']) + (255,))
    img.alpha_composite(big.resize((W, H), Image.LANCZOS))
    draw_layers(img, cfg, el['extras'])
    if el['reserve'].get('layers') and not el['reserve'].get('hidden'):
        draw_layers(img, cfg, el['reserve']['layers'])


def g10_level(img, cfg, key, v):
    """The live part at v: track, C and fill as photographed, cut at the level line; the bar at the level."""
    W, H = cfg['grid']
    el = cfg['elements'][key]
    v = float(np.clip(v, 0, 1))
    p, dvec = g10_level_point(cfg, key, v)
    T = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    draw_layers(T, cfg, el['fill']['layers'])
    T = np.asarray(T).astype(np.float32)
    XX, YY = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    a = T[..., 3:] / 255 * _below(cfg, p, dvec, XX, YY)[..., None]
    arr = np.asarray(img).astype(np.float32)
    arr[..., :3] = arr[..., :3] * (1 - a) + T[..., :3] * a
    img = Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8), 'RGBA')
    n = np.array([-dvec[1], dvec[0]])
    big = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 0))
    bd = ImageDraw.Draw(big)
    for lay in el['bar']['layers']:
        for poly in lay['polys']:
            q = [p + dvec * u + n * w_ for u, w_ in poly]
            bd.polygon([(x * SS, y * SS) for x, y in q], fill=tuple(lay['color']) + (255,))
    img.alpha_composite(big.resize((W, H), Image.LANCZOS))
    return img


# ---------------------------------------------------------------- the temperature gauge (Corsa): the fuel gauge's mirror

def mirror(img, axis, back=False):
    """img mirrored about x = axis (an integer or half-integer: whole-pixel shift, lossless): x -> 2 axis - x."""
    sc = img.size[0] / 1920.0                                   # grid or supersampled canvas
    sh = int(round((2 * axis - 1920) * sc))
    a = np.asarray(img)
    a = np.roll(a, -sh, axis=1)[:, ::-1] if back else np.roll(a[:, ::-1], sh, axis=1)
    return Image.fromarray(np.ascontiguousarray(a), img.mode)


def draw_temp_static(img, cfg):
    """The fuel gauge's drawing in the mirrored frame (temp's lines stored there), mirrored onto the right."""
    W, H = cfg['grid']
    el = cfg['elements']['temp']
    if 'editor' in el:                                          # the user's drawing, on the display as it is
        draw_editor(img, cfg, 'temp')
        return
    if rebuilt(cfg, 'temp'):                                    # 10x rebuild: drawn on a clear layer, mirrored
        lay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        g10_static(lay, cfg, 'temp')
        img.alpha_composite(mirror(lay, el['mirror_axis']))
        draw_temp_overlay(img, cfg)
        return
    big = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 0))
    draw_fuel_static(big, cfg, key='temp')
    big = mirror(big, el['mirror_axis'])
    img.alpha_composite(big.resize((W, H), Image.LANCZOS))
    draw_temp_overlay(img, cfg)


def draw_layers(img, cfg, layers):
    """Traced colour layers (display coordinates): flat polygons, lowest first, supersampled."""
    W, H = cfg['grid']
    big = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    for lay in layers:
        for poly in lay['polys']:
            d.polygon([(x * W * SS, y * H * SS) for x, y in poly], fill=tuple(lay['color']) + (255,))
    img.alpha_composite(big.resize((W, H), Image.LANCZOS))


def draw_temp_overlay(img, cfg):
    ov = cfg['elements']['temp'].get('overlay', {})
    for k in ('cold', 'hot'):
        if k in ov:
            draw_layers(img, cfg, ov[k])


def draw_temp_labels(img, cfg):
    """50 / 100 / 150 and the icon (°C and the oil can), on the display as measured (not mirrored)."""
    W, H = cfg['grid']
    lb = cfg['elements']['temp']['labels']
    if lb.get('layers'):                                        # traced at 10x: the photo's own glyphs
        draw_layers(img, cfg, lb['layers'])
        return
    for it in lb['items']:
        x0, y0, x1, y1 = it['box'][0] * W, it['box'][1] * H, it['box'][2] * W, it['box'][3] * H
        tmp = Image.new('RGBA', (900, 300), (0, 0, 0, 0))
        g.draw_number(tmp, it['text'], 20, 150, 200, 31.5, tuple(lb['color']), align='left')
        part = tmp.crop(tmp.getbbox())
        big = Image.new('RGBA', (int(x1 - int(x0) + 2) * SS, int(y1 - int(y0) + 2) * SS), (0, 0, 0, 0))
        big.alpha_composite(part.resize((int(round((x1 - x0) * SS)), int(round((y1 - y0) * SS))), Image.LANCZOS),
                            (int(round((x0 - int(x0)) * SS)), int(round((y0 - int(y0)) * SS))))
        img.alpha_composite(big.resize((big.width // SS, big.height // SS), Image.LANCZOS), (int(x0), int(y0)))
    u = lb.get('unit')
    if u:
        _text_in_box(img, cfg, u['text'], u['box'], u['font'], u['color'])
    ic = lb.get('icon')
    if ic:                                                      # the oil can fitted to its traced box (the drawing
        x0, y0, x1, y1 = ic['box'][0] * W, ic['box'][1] * H, ic['box'][2] * W, ic['box'][3] * H   # spans 1..45.5 x -3..25.5
        kx, ky = (x1 - x0) / 44.5, (y1 - y0) / 28.5                                                   # of its 48 x 32 cell)
        g.draw_oil(img, x0 - 1 * kx, y0 + 3 * ky, 48 * kx, 32 * ky, tuple(ic['color']))


def _text_in_box(img, cfg, text, box, font, color):
    """Text (a project font) stretched into its traced box."""
    from PIL import ImageFont
    W, H = cfg['grid']
    x0, y0, x1, y1 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    f = ImageFont.truetype(os.path.join(HERE, 'fonts-src', font), 400)
    tmp = Image.new('L', (2400, 700), 0)
    ImageDraw.Draw(tmp).text((50, 50), text, font=f, fill=255)
    part = tmp.crop(tmp.getbbox())
    m = Image.new('L', (int(x1 - int(x0) + 2) * SS, int(y1 - int(y0) + 2) * SS), 0)
    m.paste(part.resize((int(round((x1 - x0) * SS)), int(round((y1 - y0) * SS))), Image.LANCZOS),
            (int(round((x0 - int(x0)) * SS)), int(round((y0 - int(y0)) * SS))))
    m = m.resize((m.width // SS, m.height // SS), Image.LANCZOS)
    lay = Image.new('RGBA', m.size, tuple(color) + (0,))
    lay.putalpha(m)
    img.alpha_composite(lay, (int(x0), int(y0)))


def draw_temp_level(img, cfg, t_c):
    """The live fill at t_c (°C): the fuel gauge's level drawing in the mirrored frame, then the rail over it."""
    W, H = cfg['grid']
    el = cfg['elements']['temp']
    lo, hi = el['scale']
    v = float(np.clip((t_c - lo) / (hi - lo), 0, 1))
    m = mirror(img, el['mirror_axis'])
    if rebuilt(cfg, 'temp'):
        m = g10_level(m, cfg, 'temp', v)
        img = mirror(m, el['mirror_axis'], back=True)
        if 'editor' in el:
            p, dv = g10_level_point(cfg, 'temp', v)            # the level line, back on the display
            ax = el['mirror_axis']
            pd, dd = np.array([2 * ax - p[0], p[1]]), np.array([dv[0], -dv[1]])
            draw_editor(img, cfg, 'temp', 'live', clip=(pd, dd))
            draw_editor(img, cfg, 'temp')
        else:
            draw_temp_overlay(img, cfg)
        draw_temp_labels(img, cfg)
        return img
    m = draw_fuel_level(m, cfg, v, key='temp')
    rail = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 0))
    draw_fuel_static(rail, cfg, trails=False, level=v, key='temp')
    m.alpha_composite(rail.resize((W, H), Image.LANCZOS))
    img = mirror(m, el['mirror_axis'], back=True)
    draw_temp_overlay(img, cfg)
    draw_temp_labels(img, cfg)
    return img


# ---------------------------------------------------------------- the rev counter (Corsa): two traced blades

def _by_y(poly):
    a = np.array(poly, float)
    o = np.argsort(a[:, 1])
    return a[o, 1], a[o, 0]


def _x_on(poly, y):
    yy, xx = _by_y(poly)
    return np.interp(y, yy, xx)


def blades(cfg):
    r = cfg['elements'].get('revs', {})
    return [r[k] for k in ('left', 'right') if k in r]


def blade_lines(cfg, b):
    """One blade in display px: side (+1 left blade, inwards = +x; -1 right), O / E / I densified."""
    sd = 1 if b['side'] == 'left' else -1
    return sd, densify(px(cfg, b['O']), 0.5), densify(px(cfg, b['E']), 0.5), densify(px(cfg, b['I']), 0.5)


def o_inner(cfg, b, O, y):
    """x of O's inner edge at y."""
    sd = 1 if b['side'] == 'left' else -1
    return _x_on(O, y) + sd * interp(b['O_width'], y) / 2


def _profile(stops, d):
    st = np.array(stops, float)
    return np.interp(d, st[:, 0], st[:, 1] / st[0, 1])


def draw_blade_body(img, cfg, b):
    """The body's light (1x, numpy): base colour by y between E and I, E's and I's light falling off by distance,
    each line's outer side dropping within ~1.6 px (the dark gap to O / the black outside)."""
    W, H = cfg['grid']
    sd, O, E, I = blade_lines(cfg, b)
    top = [(p[0] * W, p[1] * H) for p in b['O_top']]
    outer = top + O                                             # O and, over the '8', its thin end
    y0 = int(min(p[1] for p in outer)) - 1
    y1 = int(max(p[1] for p in I)) + 1
    xs_all = [p[0] for p in outer + I]
    x0, x1 = int(min(xs_all)) - 3, int(max(xs_all)) + 4
    X, Y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    xo, xi = _x_on(outer, Y), _x_on(I, Y)
    e_top = min(p[1] for p in E)
    xe = np.where(Y < e_top, xo, _x_on(E, Y))                   # no lit edge over the '8': the body starts at O
    dE, dI = g.poly_distance(E, X, Y), g.poly_distance(I, X, Y)
    inE = sd * (X - xe) >= 0                                    # inwards of E
    inI = sd * (xi - X) >= 0                                    # inwards of I
    band = inE & inI
    gap = (sd * (X - xo) > 0) & ~inE
    out_ = ~inI & (sd * (X - xi) < 3)
    base = np.array(b['base'], float)
    rgb = np.stack([np.interp(Y, base[:, 0], base[:, k]) for k in (1, 2, 3)], -1)
    # E's light: none along the zones (they take its place), then up to its measured peak
    z_end = max([z['y'][1] * H for z in b['zones']] or [0]) + 1.0
    eP = np.array(b['E_peak'], float)
    gE = np.interp(Y, eP[:, 0], eP[:, 1]) * np.clip((Y - z_end) / max(1.0, 232 - z_end), 0, 1)
    gI = np.interp(Y, *np.array(b['I_peak'], float).T) * np.clip((Y - b['I_top'] + 3) / 3, 0, 1)
    out_side = [[0, 1.0], [0.8, 0.55], [1.6, 0.0]]
    LE = np.where(inE, _profile(b['glow_E'], dE), _profile(out_side, dE)) * gE
    LI = np.where(inI, _profile(b['glow_I'], dI), _profile(out_side, dI)) * gI
    lit = np.where(band, 1.0, 0.0)[..., None] * rgb + (np.where(band | gap, LE, 0) + np.where(band | out_, LI, 0))[..., None]
    lit = np.clip(lit, 0, 255)
    keep = band | gap | out_
    arr = np.asarray(img).astype(np.float32)
    sub = arr[y0:y1, x0:x1, :3]
    arr[y0:y1, x0:x1, :3] = np.where(keep[..., None], np.maximum(sub, lit), sub)
    return Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8), 'RGBA')


def _strip_var(d, pts, widths, fill):
    """A polyline with a width per point (supersampled draw)."""
    a = np.array(pts, float)
    for (p, q), w0, w1 in zip(zip(a[:-1], a[1:]), widths[:-1], widths[1:]):   # one quad per segment + round
        v = q - p                                               # joins: no self-crossing outline, no holes
        L = np.linalg.norm(v)
        if L < 1e-6:
            continue
        n = np.array([-v[1], v[0]]) / L
        quad = [p + n * w0 / 2, q + n * w1 / 2, q - n * w1 / 2, p - n * w0 / 2]
        d.polygon([(x * SS, y * SS) for x, y in quad], fill=fill)
        r = w1 / 2
        d.ellipse([(q[0] - r) * SS, (q[1] - r) * SS, (q[0] + r) * SS, (q[1] + r) * SS], fill=fill)


def draw_blade_lines(big, cfg, b, col=(250, 250, 252)):
    """O (width by y) with its thin end over the '8', the tick wedges and the slits, the zones (supersampled canvas)."""
    W, H = cfg['grid']
    d = ImageDraw.Draw(big)
    sd, O, E, I = blade_lines(cfg, b)
    halo = b.get('tick_halo', 1.2)                              # the dark ring round each wedge (over the body)
    for t in b['ticks']:
        w = px(cfg, t['wedge'])
        d.polygon([(x * SS, y * SS) for x, y in w], fill=(0, 0, 0, 255), outline=(0, 0, 0, 255),
                  width=max(1, int(round(2 * halo * SS))))
        r = abs(w[2][1] - w[1][1]) / 2 + halo
        cx, cy = w[1][0] - sd * (r - halo), (w[1][1] + w[2][1]) / 2
        d.ellipse([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS], fill=(0, 0, 0, 255))
    tg = np.gradient(np.array(O, float), axis=0)               # O_width is measured across a row: the width
    tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9      # across the line is that times |dy| of its tangent
    _strip_var(d, O, [interp(b['O_width'], p[1]) * abs(t[1]) for p, t in zip(O, tg)], col + (255,))
    top = b['O_top']
    for p, q in zip(top[:-1], top[1:]):                        # the thin end: brightness as measured
        a = (p[2] + q[2]) / 2 / 255
        c = tuple(int(round(v * min(1, a / 0.78))) for v in col) + (255,)
        _line(d, [(p[0] * W, p[1] * H), (q[0] * W, q[1] * H)], 2.4, c)
    for t in b['ticks']:
        w = px(cfg, t['wedge'])
        d.polygon([(x * SS, y * SS) for x, y in w], fill=col + (255,))
        r = abs(w[2][1] - w[1][1]) / 2
        cx, cy = w[1][0] - sd * r, (w[1][1] + w[2][1]) / 2
        d.ellipse([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS], fill=col + (255,))
    for t in b['ticks']:
        if t['slit']:
            d.polygon([(x * SS, y * SS) for x, y in px(cfg, t['slit'])], fill=(0, 0, 0, 255))
    for z in b['zones']:
        ya, yb = z['y'][0] * H, z['y'][1] * H
        yy = np.linspace(ya, yb, 12)
        xin = [o_inner(cfg, b, O, y) for y in yy]
        L = [(x + sd * z['off'][0], y) for x, y in zip(xin, yy)]
        R = [(x + sd * z['off'][1], y) for x, y in zip(xin, yy)]
        d.polygon([(x * SS, y * SS) for x, y in L + R[::-1]], fill=tuple(z['color']) + (255,))


def draw_blade_floor(big, cfg, b, col=(250, 250, 252)):
    """Under 0: the white zero line across the band and the dark red floor down to O's foot (supersampled)."""
    d = ImageDraw.Draw(big)
    sd, O, E, I = blade_lines(cfg, b)
    rows, last = [], None
    for r in b['floor']:
        if sd * (r[2] - r[1]) > 4 and (last is None or abs(r[1] - last) < 4):
            rows.append(r); last = r[1]
    if len(rows) > 1:
        ys = [r[0] for r in rows]
        for k in range(len(rows) - 1):                          # row by row: its colour
            a, c = rows[k], rows[k + 1]
            q = [(a[1], a[0] - 0.5), (a[2], a[0] - 0.5), (c[2], c[0] - 0.5), (c[1], c[0] - 0.5)]
            d.polygon([(x * SS, y * SS) for x, y in q], fill=tuple(a[3:6]) + (255,))
    z = b.get('zero_line')
    if z:
        ya, yb = z['y']
        xa = _x_on(E, (ya + yb) / 2) - sd * 2
        d.polygon([(xa * SS, ya * SS), (z['x_end'] * SS, ya * SS), (z['x_end'] * SS, yb * SS), (xa * SS, yb * SS)],
                  fill=col + (255,))


def draw_blade_digits(big, cfg, b, col=(254, 254, 254)):
    """The even numbers, each fitted into its traced box (BMW digits, supersampled)."""
    W, H = cfg['grid']
    for dg in b['digits']:
        x0, y0, x1, y1 = dg['box'][0] * W, dg['box'][1] * H, dg['box'][2] * W, dg['box'][3] * H
        tmp = Image.new('RGBA', (600, 300), (0, 0, 0, 0))
        g.draw_number(tmp, dg['text'], 20, 150, 200, 260, col, align='left')
        part = tmp.crop(tmp.getbbox()).resize((max(1, int(round((x1 - x0) * SS))), max(1, int(round((y1 - y0) * SS)))), Image.LANCZOS)
        big.alpha_composite(part, (int(round(x0 * SS)), int(round(y0 * SS))))


def rev_y(b, v, H):
    vy = [y * H for y in b['values_y']]
    return float(np.interp(v, np.arange(len(vy)), vy))


def draw_rev_level(img, cfg, b, v):
    """The live bar at v (x1000/min): the grey block from 0 up to it, the red bar, the red glow above (traced table)."""
    W, H = cfg['grid']
    f = b['fill']
    sd, O, E, I = blade_lines(cfg, b)
    yv = rev_y(b, v, H)
    hb = (f['bar'][1] - f['bar'][0]) / 2
    y0z = rev_y(b, 0, H)
    # the glow: rows above the bar x 6 stops across E -> I; the photo's own base (its top row) taken off
    gl = np.array([r for r in f['glow'] if r[0] <= 5 or max(r[1:]) < 0.95], float)
    b0 = f.get('glow_base', 0.08)                              # the body's own red under the glow
    amp = np.clip((gl[:, 1:] - b0) / (1 - b0), 0, 1)
    gl = np.vstack([gl, [gl[-1, 0] + f.get('glow_tail', 12)] + [0] * 6])   # its far end fades out
    amp = np.vstack([amp, np.zeros(6)])
    ya, yb = int(yv - hb - gl[-1, 0]) - 1, int(yv - hb) + 1
    X, Y = np.meshgrid(np.arange(W) + 0.5, np.arange(max(0, ya), yb) + 0.5)
    xe, xi = _x_on(E, Y), _x_on(I, Y)
    u = (X - xe) / (xi - xe)
    dy = (yv - hb) - Y
    us = np.linspace(0.08, 0.92, 6)
    A = np.zeros(X.shape)
    for k in range(6):                                          # bilinear: dy along the table, u across
        A += np.interp(dy, gl[:, 0], amp[:, k]) * np.clip(1 - np.abs(u - us[k]) / (us[1] - us[0]), 0, 1)
    A *= (u >= 0) & (u <= 1) & (dy >= 0)
    dt = f.get('dots')
    if dt:                                                      # the dot grid: lines with brighter crossings
        def near(c, o, per):
            q = (c - o) / per
            return np.exp(-(((q - np.round(q)) * per) / dt['sigma']) ** 2)
        gx, gy = near(X, dt['origin'][0], dt['period'][0]), near(Y, dt['origin'][1], dt['period'][1])
        A = np.clip(A * (dt['c'] + dt['a'] * (gx + gy) + dt['b'] * gx * gy), 0, 1)
    arr = np.asarray(img).astype(np.float32)
    sub = arr[max(0, ya):yb, :, :3]
    sub[..., 0] = np.maximum(sub[..., 0], A * 255)
    sub[..., 1] *= 1 - A
    sub[..., 2] *= 1 - A
    img = Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8), 'RGBA')
    # the bar and the block (supersampled)
    big = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    def band(ya_, yb_, grow=0.0):
        yy = np.linspace(ya_, yb_, 10)
        L = [(_x_on(E, y) - sd * grow, y) for y in yy]
        R = [(_x_on(I, y) + sd * grow, y) for y in yy]
        return [(x * SS, y * SS) for x, y in L + R[::-1]]
    fw = b.get('frame_w', [3.0, 3.0])
    def band2(ya_, yb_, gl_, gr_):
        yy = np.linspace(ya_, yb_, 10)
        L = [(_x_on(E, y) - sd * gl_, y) for y in yy]
        R = [(_x_on(I, y) + sd * gr_, y) for y in yy]
        return [(x * SS, y * SS) for x, y in L + R[::-1]]
    if y0z - (yv + hb) > 1.5:
        d.polygon(band2(yv + hb + 1.0, y0z, fw[0] / 2, fw[1] / 2), fill=(250, 250, 252, 255))     # the frame
        d.polygon(band2(yv + hb + 2.3, y0z, -fw[0] / 2, -fw[1] / 2), fill=tuple(f['block']) + (255,))
    pb = band(yv - hb, yv + hb, 0.3)
    m = Image.new('L', big.size, 0)
    ImageDraw.Draw(m).polygon(pb, fill=255)
    big.paste(Image.new('RGBA', big.size, tuple(f['bar_color']) + (255,)), (0, 0), m)
    img.alpha_composite(big.resize((W, H), Image.LANCZOS))
    return img


def photo_rpm(cfg):
    """The value the photo shows (its bar's centre on the left blade's scale)."""
    b = cfg['elements']['revs']['left']
    H = cfg['grid'][1]
    yc = sum(b['fill']['bar']) / 2
    vy = [y * H for y in b['values_y']]
    return float(np.interp(-yc, [-y for y in vy], np.arange(len(vy))))


# ---------------------------------------------------------------- the centre (Corsa): gear, speed, line

def _glyph_box(cfg, gl):
    W, H = cfg['grid']
    xs = [p[0] * W for p in gl['outline']]; ys = [p[1] * H for p in gl['outline']]
    return min(xs), min(ys), max(xs), max(ys)


def draw_traced(img, cfg, gl, color):
    """A traced glyph (outline minus its counters), supersampled into img."""
    W, H = cfg['grid']
    x0, y0, x1, y1 = _glyph_box(cfg, gl)
    ox, oy = int(x0) - 2, int(y0) - 2
    w_, h_ = int(x1 - ox) + 3, int(y1 - oy) + 3
    m = Image.new('L', (w_ * SS, h_ * SS), 0)
    d = ImageDraw.Draw(m)
    P = lambda poly: [((x * W - ox) * SS, (y * H - oy) * SS) for x, y in poly]
    d.polygon(P(gl['outline']), fill=255)
    for h in gl['holes']:
        d.polygon(P(h), fill=0)
    m = m.resize((w_, h_), Image.LANCZOS)
    lay = Image.new('RGBA', (w_, h_), tuple(color) + (0,))
    lay.putalpha(m)
    img.alpha_composite(lay, (ox, oy))


def draw_center_value(img, cfg, part, text):
    """Gear or speed: the traced glyph when it is the photographed one, else BMW's big glyphs with the measured
    strokes, centred on the traced glyph's centre, same top."""
    el = cfg['elements']['center'][part]
    if text == el['text']:
        draw_traced(img, cfg, el['glyph'], el['color'])
        return
    x0, y0, x1, y1 = _glyph_box(cfg, el['glyph'])
    st = dict(el['style'])
    widths = [st.get('letter_w', st['w']) if ch.isalpha() else (st['w'] * 0.62 if ch == '1' else st['w']) for ch in text]
    total = sum(widths) + st['gap'] * (len(text) - 1)
    g.draw_big(img, text, (x0 + x1) / 2 + total / 2, y0, st, el['color'])


def draw_center_static(big, img_unused, cfg):
    """The line and its end pieces (supersampled canvas)."""
    W, H = cfg['grid']
    d = ImageDraw.Draw(big)
    for l_ in cfg['elements']['center']['line']:
        d.polygon([(x * W * SS, y * H * SS) for x, y in l_['poly']], fill=tuple(l_['color']) + (255,))


def draw_center_unit(img, cfg):
    """km/h fitted into its traced box (the project's Barlow, the nearest to the cluster's own sans)."""
    from PIL import ImageFont
    W, H = cfg['grid']
    u = cfg['elements']['center']['unit']
    x0, y0, x1, y1 = u['box'][0] * W, u['box'][1] * H, u['box'][2] * W, u['box'][3] * H
    f = ImageFont.truetype(os.path.join(HERE, 'fonts-src', u['font']), 400)
    tmp = Image.new('L', (2400, 700), 0)
    ImageDraw.Draw(tmp).text((50, 50), u['text'], font=f, fill=255)
    part = tmp.crop(tmp.getbbox())
    ww, hh = int(round((x1 - x0) * SS)), int(round((y1 - y0) * SS))
    m = Image.new('L', (int(x1 - int(x0) + 2) * SS, int(y1 - int(y0) + 2) * SS), 0)
    m.paste(part.resize((ww, hh), Image.LANCZOS), (int(round((x0 - int(x0)) * SS)), int(round((y0 - int(y0)) * SS))))
    m = m.resize((m.width // SS, m.height // SS), Image.LANCZOS)
    lay = Image.new('RGBA', m.size, tuple(u['color']) + (0,))
    lay.putalpha(m)
    img.alpha_composite(lay, (int(x0), int(y0)))


# ---------------------------------------------------------------- build

def build_static(cfg):
    W, H = cfg['grid']
    el = cfg['elements']
    big = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 255))
    if 'fuel' in el and not rebuilt(cfg, 'fuel'):
        draw_fuel_static(big, cfg)
    img = big.resize((W, H), Image.LANCZOS)
    if 'editor' in el.get('fuel', {}):
        draw_editor(img, cfg, 'fuel')
    elif rebuilt(cfg, 'fuel'):
        g10_static(img, cfg, 'fuel')
    for b in blades(cfg):
        img = draw_blade_body(img, cfg, b)
    if 'revs' in el:
        top = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 0))
        for b in blades(cfg):
            draw_blade_floor(top, cfg, b)
            draw_blade_lines(top, cfg, b)
            draw_blade_digits(top, cfg, b)
        img.alpha_composite(top.resize((W, H), Image.LANCZOS))
    if 'center' in el:
        top = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 0))
        draw_center_static(top, img, cfg)
        img.alpha_composite(top.resize((W, H), Image.LANCZOS))
        draw_center_unit(img, cfg)
    if 'fuel' in el:
        draw_fuel_labels(img, cfg)
    if 'temp' in el:
        draw_temp_static(img, cfg)
        draw_temp_labels(img, cfg)
    return img


def draw_dynamic(img, cfg, tel):
    if 'temp' in tel and 'temp' in cfg['elements']:
        img = draw_temp_level(img, cfg, tel['temp'])
    if 'center' in cfg['elements']:
        if 'gear' in tel:
            draw_center_value(img, cfg, 'gear', str(tel['gear']))
        if 'speed' in tel:
            draw_center_value(img, cfg, 'speed', str(int(round(float(tel['speed'])))))
    if 'rpm' in tel and 'revs' in cfg['elements']:
        W, H = cfg['grid']
        for b in blades(cfg):
            img = draw_rev_level(img, cfg, b, tel['rpm'])
        top = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 0))   # the lines and the numbers stay over the bar
        for b in blades(cfg):
            draw_blade_lines(top, cfg, b)
        img.alpha_composite(top.resize((W, H), Image.LANCZOS))
    if 'fuel' in tel and 'editor' in cfg['elements']['fuel']:
        v = float(np.clip(tel['fuel'], 0, 1))
        img = g10_level(img, cfg, 'fuel', v)                   # the fill (as photographed) and the bar at the level
        p, dv = g10_level_point(cfg, 'fuel', v)
        draw_editor(img, cfg, 'fuel', 'live', clip=(p, dv))    # C under the level
        draw_editor(img, cfg, 'fuel')                          # the drawing over it
        draw_fuel_labels(img, cfg)
    elif 'fuel' in tel and rebuilt(cfg, 'fuel'):
        img = g10_level(img, cfg, 'fuel', tel['fuel'])
        draw_fuel_labels(img, cfg)
    elif 'fuel' in tel:
        img = draw_fuel_level(img, cfg, tel['fuel'])
        W, H = cfg['grid']                                      # the rail (lines, triangles, trails) stays over the fill
        rail = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 0))
        draw_fuel_static(rail, cfg, trails=False, level=float(np.clip(tel['fuel'], 0, 1)))            # over the fill the trails show as lighter lines
        img.alpha_composite(rail.resize((W, H), Image.LANCZOS))
        draw_fuel_labels(img, cfg)                              # the labels stay over the bar (as on the photo)
    return img


def photo_tel(cfg):
    """The values the reference shows: fuel full, the rev bar at idle, P and 0 km/h."""
    el = cfg['elements']
    tel = {'fuel': 1.0}
    if 'revs' in el:
        tel['rpm'] = photo_rpm(cfg)
    if 'temp' in el:
        tel['temp'] = el['temp'].get('photo_value', 105)
    if 'center' in el:
        tel['gear'] = el['center']['gear']['text']; tel['speed'] = el['center']['speed']['text']
    return tel


def compare(img, ref_path, out_path, box):
    ref = Image.open(ref_path).convert('RGB')
    a, b = ref.crop(box), img.convert('RGB').crop(box)
    k = max(1, min(6, 900 // (box[2] - box[0])))             # small boxes enlarged, wide ones at 1:1
    w, h = (box[2] - box[0]) * k, (box[3] - box[1]) * k
    tall = w > 1.5 * h                                          # wide boxes: photo / generated / 50% stacked
    o = Image.new('RGB', (w, h * 3 + 20) if tall else (w * 3 + 20, h), (25, 25, 25))
    for j, im in enumerate((a, b, Image.blend(a, b, 0.5))):
        o.paste(im.resize((w, h), Image.LANCZOS), (0, j * (h + 10)) if tall else (j * (w + 10), 0))
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
            tel[k] = v if k == 'gear' else float(v)
        live = draw_dynamic(static.copy(), cfg, tel)
        live.convert('RGB').save(os.path.join(out_dir, '%s_%s_live.png' % (cat, name)))
        print('live: design/out/%s_%s_live.png' % (cat, name))
    if a.compare:
        live = draw_dynamic(static.copy(), cfg, tel or photo_tel(cfg))
        ref = os.path.join(ROOT, 'design', 'refs', cat, cfg.get('reference', name) + '.png')
        print('compare: ' + os.path.relpath(compare(live, ref, os.path.join(out_dir, '%s_%s_compare.png' % (cat, name)),
                                                    tuple(cfg.get('compare_box', (115, 405, 315, 590)))), ROOT))


if __name__ == '__main__':
    main()
