#!/usr/bin/env python3
"""gen_static.py - draws a layout's static layer from its config.json (positions normalized to the 1920x720
calibration grid), and checks it against the reference photo.

    python3 cluster/tools/gen_static.py id8/A                 # cluster/layouts/id8/A/static.png
    python3 cluster/tools/gen_static.py id8/A --compare       # + cluster/design/out/id8_A_compare.png

--compare stacks the reference photo (cluster/design/refs/<category>/<layout>.png, not in git), the generated layer
and the two blended 50 % with each element's outline in magenta, the calibration grid (lines every 50 px) on top.
Runs on a PC (Pillow, numpy); the car only ever loads the finished picture.
"""
import argparse
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SS = 4                      # supersampling: drawn 4x larger, scaled down


def fillet_polyline(pts, samples=12):
    """[(x, y, r), ...] in px -> a dense polyline with every inner vertex rounded by a circular arc of radius r
    (r 0 = sharp). The arc is tangent to both segments; r is clipped so the arc fits the shorter segment."""
    out = [pts[0][:2]]
    for i in range(1, len(pts) - 1):
        (x0, y0, _), (x1, y1, r), (x2, y2, _) = pts[i - 1], pts[i], pts[i + 1]
        if r <= 0:
            out.append((x1, y1))
            continue
        ax, ay = x0 - x1, y0 - y1
        bx, by = x2 - x1, y2 - y1
        la, lb = math.hypot(ax, ay), math.hypot(bx, by)
        ax, ay, bx, by = ax / la, ay / la, bx / lb, by / lb
        ang = math.acos(max(-1.0, min(1.0, ax * bx + ay * by)))           # the inner angle at the vertex
        t = r / math.tan(ang / 2)                                         # vertex -> tangent point
        t = min(t, 0.5 * la, 0.5 * lb)
        r = t * math.tan(ang / 2)
        p0 = (x1 + ax * t, y1 + ay * t)
        p1 = (x1 + bx * t, y1 + by * t)
        hx, hy = ax + bx, ay + by
        hl = math.hypot(hx, hy)
        c = (x1 + hx / hl * r / math.sin(ang / 2), y1 + hy / hl * r / math.sin(ang / 2))   # the arc's centre
        a0 = math.atan2(p0[1] - c[1], p0[0] - c[0])
        a1 = math.atan2(p1[1] - c[1], p1[0] - c[0])
        da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi                 # the short way round
        for k in range(samples + 1):
            a = a0 + da * k / samples
            out.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a)))
    out.append(pts[-1][:2])
    return out


def to_px(cfg, pts):
    W, H = cfg['grid']
    return [(p[0] * W, p[1] * H, p[2] if len(p) > 2 else 0) for p in pts]


def expand(cfg):
    """Elements with "mirror_of" get the source's geometry mirrored about x 0.5 (and its bevel); marks and numbers
    then go to the other side of their lines ("_side": "right"). Idempotent."""
    for el in cfg['elements'].values():
        src = cfg['elements'].get(el.get('mirror_of', ''))
        if not src:
            continue
        for key in ('outline', 'parallel', 'band', 'inner', 'rail', 'track'):
            if key in src and key not in el:
                el[key] = [[round(1 - p[0], 6)] + list(p[1:]) for p in src[key]]
        for key in ('bevel', 'gaps', 'gap_px', 'colors', 'labels'):
            if key in src and key not in el:
                el[key] = src[key]
        el['_side'] = 'right'
    return cfg


def panel_shapes(cfg):
    """The central frame's closed shapes (px polygons): the main panel (its left half and the mirror) and both wings."""
    W, H = cfg['grid']
    out = {}
    for name, el in cfg['elements'].items():
        if 'half' not in el:
            continue
        half = to_px(cfg, el['half'])
        right = [(W - x, y, r) for x, y, r in reversed(half[1:-1])]
        if 'gear_cut' in el:                                  # the gear cut-out is not the speed one's mirror
            gc = to_px(cfg, el['gear_cut'])
            right[2], right[3] = gc[1], gc[0]                 # its inner side: bottom vertex, then top vertex
        if 'top_right' in el:                                 # the top right is not the top left's mirror
            right = right[:4] + to_px(cfg, el['top_right'])
        full = half + right
        out[name] = fillet_closed(full)
        if 'wing_half' in el:
            wing = to_px(cfg, el['wing_half'])
            out[name + '.wing_l'] = fillet_closed(wing)
            wr = to_px(cfg, el['wing_right']) if 'wing_right' in el else [(W - x, y, r) for x, y, r in reversed(wing)]
            out[name + '.wing_r'] = fillet_closed(wr)
    return out


def fillet_closed(pts):
    """A closed polygon [(x, y, r)] with its corners rounded (fillet_polyline on the ring)."""
    ring = pts[-1:] + pts + pts[:1]
    return fillet_polyline(ring)[1:-1]


def outlines(cfg):
    """Every element's outline as a dense px polyline (the reference for --compare)."""
    out = {}
    expand(cfg)
    for name, poly in panel_shapes(cfg).items():
        out[name] = poly + [poly[0]]
    for name, el in cfg['elements'].items():
        for key in ('outline', 'parallel', 'band', 'inner'):
            if key in el:
                out[name + '.' + key] = fillet_polyline(to_px(cfg, el[key]))
    return out


def x_on(poly, y):
    """x where the polyline first crosses the height y (None if it doesn't)."""
    for (xa, ya), (xb, yb) in zip(poly[:-1], poly[1:]):
        if (ya - y) * (yb - y) <= 0 and ya != yb:
            return xa + (xb - xa) * (y - ya) / (yb - ya)
    return None


def tick_rects(cfg, only=None):
    """Every element's (or only that one's) graduated marks as [x0, y0, x1, y1] px: level dashes from its parallel."""
    W, H = cfg['grid']
    out = []
    expand(cfg)
    for el in ([only] if only is not None else cfg['elements'].values()):
        t = el.get('ticks')
        if not t or 'parallel' not in el:
            continue
        par = fillet_polyline(to_px(cfg, el['parallel']))
        sc = t['scales'].get((el.get('numbers') or {}).get('scale', ''))
        for yn in (sc['y'] if isinstance(sc, dict) else t['y']):
            y = yn * H
            x = x_on(par, y)
            if x is not None:
                s = -1 if el.get('_side') == 'right' else 1             # inwards, away from the outer profile
                xa, xb = x + s * t['start'], x + s * (t['start'] + t['length'])
                out.append([min(xa, xb), y - t['thick'] / 2, max(xa, xb), y + t['thick'] / 2])
    return out


def bracket_zones(cfg, el):
    """A bracket's zones as polygons (px): the whole bracket, outer profile to the fill track's edge, and outer profile
    to the parallel. Each runs down the outer profile, back up its inner line through the foot's end cut, and closes
    along the top line."""
    P = lambda k: fillet_polyline(to_px(cfg, el[k]))
    out = P('outline')
    end = [(el['inner'][-2][0] * cfg['grid'][0], el['inner'][-2][1] * cfg['grid'][1])]     # the foot's upper end
    zone = lambda k: out + end + P(k)[::-1]                # closes back along the top line
    return zone('inner'), zone('band'), zone('parallel')


def poly_distance(poly, X, Y):
    """Distance (px) from every point of the grids X, Y to the polyline."""
    best = np.full(X.shape, 1e9)
    for (ax, ay), (bx, by) in zip(poly[:-1], poly[1:]):
        ex, ey = bx - ax, by - ay
        L2 = ex * ex + ey * ey or 1e-9
        h = np.clip(((X - ax) * ex + (Y - ay) * ey) / L2, 0, 1)
        best = np.minimum(best, np.hypot(X - ax - ex * h, Y - ay - ey * h))
    return best


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def bevel(img, cfg, el):
    """The 3D look on a bracket already filled flat (img: RGBA at grid size): a brightness profile across the stripe
    (a tube with a glossy highlight) and across the band (a hollow), from each pixel's distances to the zone's two
    edges, as config 'bevel' says."""
    bv = el.get('bevel')
    if not bv:
        return img
    P = lambda k: fillet_polyline(to_px(cfg, el[k]))
    out, par, band, inner = P('outline'), P('parallel'), P('band'), P('inner')
    xs = [p[0] for p in out + inner]
    ys = [p[1] for p in out + inner]
    x0, x1, y0, y1 = int(min(xs)) - 2, int(max(xs)) + 3, int(min(ys)) - 2, int(max(ys)) + 3
    X, Y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    whole, to_band, to_par = bracket_zones(cfg, el)

    def mask(poly):
        m = Image.new('L', (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(m).polygon([(x - x0, y - y0) for x, y in poly], fill=255)
        return np.asarray(m.filter(ImageFilter.GaussianBlur(bv.get('soft', 1.0)))) / 255.0

    m_par, m_band = mask(to_par), mask(to_band)
    m_fill = mask(whole) - m_band
    rgb = np.asarray(img, np.float32)[y0:y1, x0:x1, :3].copy()
    # the stripe: a tube, darker at both rims, with a glossy highlight
    st = bv['stripe']
    d_o, d_p = poly_distance(out, X, Y), poly_distance(par, X, Y)
    t = d_o / (d_o + d_p + 1e-6)
    f = st['edge'] + (1 - st['edge']) * np.sin(np.pi * np.clip(t, 0, 1)) ** 0.6
    g = st['gloss'] * np.exp(-((t - st['gloss_at']) / st['gloss_w']) ** 2)
    shaded = np.clip(rgb * f[..., None] + (255 - rgb * f[..., None]) * g[..., None], 0, 255)
    rgb = rgb + (shaded - rgb) * np.clip(m_par, 0, 1)[..., None]
    # the band: a hollow, rising from the gap, a touch darker before the inner line
    bd = bv['band']
    d_b, d_i = poly_distance(band, X, Y), poly_distance(inner, X, Y)
    t = d_b / (d_b + d_i + 1e-6)
    f = bd['edge'] + (1 - bd['edge']) * smooth(0, bd['rise'], t)
    f = f - (1 - bd['inner']) * smooth(0.72, 0.95, t)
    col = np.array(el['colors']['band'], np.float32)
    keep_line = smooth(1.2, 2.4, d_i)                                   # the inner line stays as drawn
    rgb = rgb + (col * f[..., None] - rgb) * (np.clip(m_fill, 0, 1) * keep_line)[..., None]
    arr = np.asarray(img).copy()
    arr[y0:y1, x0:x1, :3] = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, 'RGBA')


# BMW's squared digits, measured on the reference (px of a 29 x 22 cell; the 1 is 18 wide): every digit is a union of
# rounded bars [x0, y0, x1, y1], the 7's stroke a slanted quad. Uprights 7 thick, bars 5, middle bar y 8.5..13.5.
_T, _M, _B = (0, 0, 29, 5), (0, 8.5, 29, 13.5), (0, 17, 29, 22)
_L, _R = 7, 22
DIGITS = {
    '0': [_T, _B, (0, 0, _L, 22), (_R, 0, 29, 22)],
    '1': [(0, 0, 18, 5), (11, 0, 18, 22)],
    '2': [_T, _M, _B, (_R, 0, 29, 13.5), (0, 8.5, _L, 22)],
    '3': [_T, (7, 8.5, 29, 13.5), _B, (_R, 0, 29, 22)],
    '4': [(12, 0, 29, 4), ('quad', (12, 0), (19.5, 0), (7, 14.5), (0, 14.5)), (0, 13, 31, 18), (_R, 0, 29, 22)],   # BMW's: a slanted left stroke, the bar low and past the upright
    '5': [_T, (0, 0, _L, 13.5), _M, (_R, 8.5, 29, 22), _B],
    '6': [_T, (0, 0, _L, 22), _M, (_R, 8.5, 29, 22), _B],
    '7': [_T, ('quad', (20.5, 5), (30, 5), (18, 22), (7, 22))],
    '8': [_T, _M, _B, (0, 0, _L, 22), (_R, 0, 29, 22)],
    '9': [_T, (0, 0, _L, 13.5), _M, (_R, 0, 29, 22), _B],
}
DIGITS['/'] = [('quad', (12, 0), (17, 0), (5, 22), (0, 22))]
DIGIT_W = {'1': 18, '/': 17}


def big_glyph(ch, w, h, tb, tu):
    """BMW's big squared glyphs (speed and gear), built from the cell w x h, bar thickness tb and upright thickness tu
    (measured on the photo's 57 and D): parts as in DIGITS ([x0, y0, x1, y1] or ('quad', ...) or ('round', ...))."""
    m0, m1 = h * 0.41, h * 0.41 + tb                       # the middle bar
    T, M, B = (0, 0, w, tb), (0, m0, w, m1), (0, h - tb, w, h)
    L, R = (0, 0, tu, h), (w - tu, 0, w, h)
    up = lambda y0, y1, right=False: ((w - tu, y0, w, y1) if right else (0, y0, tu, y1))
    g = {
        '0': [T, B, L, R], '8': [T, M, B, L, R],
        '1': [(w * 0.35, 0, w * 0.35 + tu, h), (w * 0.12, 0, w * 0.35 + tu, tb)],
        '2': [T, M, B, up(0, m1, True), up(m0, h)], '3': [T, (w * 0.25, m0, w, m1), B, R],
        '4': [('quad', (w * 0.3, 0), (w * 0.3 + tu, 0), (tu, m1), (0, m1)), (0, m0 + tb * 0.6, w, m1 + tb * 0.6),
              (w - tu * 0.85, 0, w, h)],
        '5': [T, up(0, m1), M, up(m0, h, True), B], '6': [T, L, M, up(m0, h, True), B],
        '7': [T, ('quad', (w - tu * 1.1, tb), (w, tb), (w * 0.27 + tu * 1.1, h), (w * 0.27, h))],
        '9': [T, up(0, m1), M, R, B],
        'P': [L, T, M, up(0, m1, True)], 'R': [L, T, M, up(0, m1, True), ('quad', (w * 0.45, m1), (w * 0.45 + tu, m1), (w, h), (w - tu, h))],
        'N': [L, R, ('quad', (0, 0), (tu, 0), (w, h), (w - tu, h))], 'S': [('round', 0, 0, w, m1, h * 0.22), ('hole', tu, tb, w + 1, m0, 0),               # rounded, unlike the 5
              ('round', 0, m0, w, h, h * 0.22), ('hole', -1, m1, w - tu, h - tb, 0)],
        'M': [L, R, T, ((w - tu * 0.8) / 2, 0, (w + tu * 0.8) / 2, h * 0.7)],
        'D': [('round', 0, 0, w, h, h * 0.32), (0, 0, w * 0.5, h),                         # square left, rounded right
              ('hole', tu * 0.75, tb, w - tu * 0.95, h - tb, h * 0.16), ('hole', tu * 0.75, tb, w * 0.5, h - tb, 0)],
    }
    return g[ch]


def draw_big(img, text, x_right, y_top, d, color, ss=4):
    """Big glyphs (config 'digits': w, h, bar, upright, gap, letter_w) right-aligned at x_right, top at y_top."""
    w, h, tb, tu, gap = d['w'], d['h'], d['bar'], d['upright'], d['gap']
    widths = [d.get('letter_w', w) if ch.isalpha() else (w * 0.62 if ch == '1' else w) for ch in text]
    total = sum(widths) + gap * (len(text) - 1)
    k = 1.0
    if d.get('_min_left') is not None and x_right - total < d['_min_left']:
        k = (x_right - d['_min_left']) / total               # three digits: scaled down to fit the cut-out
    x0 = x_right - total * k
    Wd, Hd = int(total * k) + 6, int(h * k) + 6
    m = Image.new('L', (Wd * ss, Hd * ss), 0)
    dr = ImageDraw.Draw(m)
    x = 3.0
    for ch, cw in zip(text, widths):
        for part in big_glyph(ch, cw, h, tb, tu):
            if part[0] == 'quad':
                dr.polygon([((x + px_ * k) * ss, (3 + py * k) * ss) for px_, py in part[1:]], fill=255)
            elif part[0] in ('round', 'hole'):
                _, a, b, c2, e, r = part
                dr.rounded_rectangle([(x + a * k) * ss, (3 + b * k) * ss, (x + c2 * k) * ss, (3 + e * k) * ss],
                                     radius=r * k * ss, fill=255 if part[0] == 'round' else 0)
            else:
                a, b, c2, e = part
                dr.rounded_rectangle([(x + a * k) * ss, (3 + b * k) * ss, (x + c2 * k) * ss - 1, (3 + e * k) * ss - 1],
                                     radius=2.5 * k * ss, fill=255)
        x += (cw + gap) * k
    m = m.resize((Wd, Hd), Image.LANCZOS)
    layer = Image.new('RGBA', m.size, tuple(color) + (0,))
    layer.putalpha(m)
    img.alpha_composite(layer, (int(round(x0)) - 3, int(round(y_top)) - 3))


def draw_number(img, text, x_edge, y_mid, h, advance, color, ss=8, align='right', sx=1.0, gap=None):
    """Text of BMW's digits into img (RGBA, grid px), its right (or left) edge at x_edge, centred on y_mid, h px tall;
    sx narrows the glyphs, gap (px) overrides the spacing that advance gives."""
    k = h / 22.0
    widths = [DIGIT_W.get(ch, 29) * k * sx for ch in text]
    space = gap if gap is not None else (advance - 29) * k
    total = sum(widths) + space * (len(text) - 1)
    x0, y0 = (x_edge - total if align == 'right' else x_edge), y_mid - h / 2
    W, H = int(total) + 4, int(h) + 4
    m = Image.new('L', (W * ss, H * ss), 0)
    d = ImageDraw.Draw(m)
    x = 2.0
    for ch, w in zip(text, widths):
        for part in DIGITS[ch]:
            if part[0] == 'quad':
                d.polygon([((x + px_ * k * sx) * ss, (2 + py * k) * ss) for px_, py in part[1:]], fill=255)
            else:
                a, b, c, e = part
                if c - a <= 8 and sx != 1.0:          # an upright keeps its thickness when the digit is narrowed
                    a2 = a * sx if a < 14.5 else c * sx - (c - a)
                    c2 = a2 + (c - a)
                else:
                    a2, c2 = a * sx, c * sx
                d.rounded_rectangle([(x + a2 * k) * ss, (2 + b * k) * ss, (x + c2 * k) * ss - 1, (2 + e * k) * ss - 1],
                                    radius=1.6 * k * ss, fill=255)
        x += w + space
    m = m.resize((W, H), Image.LANCZOS)
    layer = Image.new('RGBA', (W, H), tuple(color) + (0,))
    layer.putalpha(m)
    img.alpha_composite(layer, (int(round(x0)) - 2, int(round(y0)) - 2))


def draw_numbers(img, cfg):
    W, H = cfg['grid']
    for el in cfg['elements'].values():
        n = el.get('numbers')
        if not n:
            continue
        out = fillet_polyline(to_px(cfg, el['outline']))
        sc = el['ticks']['scales'][n['scale']]
        vals, ys = (sc['values'], sc['y']) if isinstance(sc, dict) else (sc, el['ticks']['y'])
        for v, yn in zip(vals, ys):
            y = yn * H
            left = n.get('align') == 'left'
            draw_number(img, str(v), x_on(out, y) + (n['gap'] if left else -n['gap']), y, n['h'], n['advance'],
                        n['color'], align='left' if left else 'right')


FONTS = os.path.join(ROOT, 'tools', 'fonts-src')


def draw_labels(img, cfg, ss=8):
    """Text labels: the text's ink is scaled to the box's height (box: normalized x0, y0, x1, y1 measured on the photo),
    left-aligned; for the photo's own text its width matches the box too."""
    W, H = cfg['grid']
    for el in cfg['elements'].values():
        lb = el.get('label')
        if not lb:
            continue
        text = lb['text']
        src = cfg['elements'].get(lb.get('scale_of', ''), {}).get('numbers', {})
        if str(src.get('scale', '')).endswith('mph') and lb.get('text_mph'):
            text = lb['text_mph']
        x0, y0, x1, y1 = lb['box'][0] * W, lb['box'][1] * H, lb['box'][2] * W, lb['box'][3] * H
        font = ImageFont.truetype(os.path.join(FONTS, lb['font']), 200)
        sw = int(round(200 * lb.get('bold', 0)))                # extra weight, as a fraction of the font size
        l, t, r, b = font.getbbox(lb.get('fit_text', lb['text']), stroke_width=sw)  # the photo's text sets the scale
        k = (y1 - y0) / (b - t)
        l2, t2, r2, b2 = font.getbbox(text, stroke_width=sw)
        m = Image.new('L', (r2 - l2 + 8, max(b, b2) - t + 8), 0)      # room for descenders (the p of mph)
        ImageDraw.Draw(m).text((4 - l2, 4 - t), text, font=font, fill=255, stroke_width=sw, stroke_fill=255)
        if text != lb.get('fit_text', lb['text']) and lb.get('lift_mph'):
            y0 -= lb.get('lift_mph', 0)                              # raised so the descender clears the line below
        m = m.resize((max(1, round(m.width * k)), max(1, round(m.height * k))), Image.LANCZOS)
        kx = 1.0
        layer = Image.new('RGBA', m.size, tuple(lb['color']) + (0,))
        layer.putalpha(m)
        img.alpha_composite(layer, (int(round(x0 - 4 * k * kx)), int(round(y0 - 4 * k))))


def draw_panel(img, cfg):
    """The central frame (supersampled canvas): its shape filled with the row gradient, the wings flat."""
    W, H = cfg['grid']
    for name, poly in panel_shapes(cfg).items():
        base = name.split('.')[0]
        col = cfg['elements'][base].get('colors')
        if not col:
            continue
        m = Image.new('L', img.size, 0)
        ImageDraw.Draw(m).polygon([(x * SS, y * SS) for x, y in poly], fill=255)
        stops = col['wing'] if name.endswith(('.wing_l', '.wing_r')) else col['rows']
        if stops and not isinstance(stops[0][0], (list, tuple)) and len(stops) == 3 and isinstance(stops[0], int):
            fill = Image.new('RGBA', img.size, tuple(stops) + (255,))
        else:
            ys, cs = [r[0] for r in stops], np.array([r[1] for r in stops], np.float32)
            yy = (np.arange(img.size[1]) + 0.5) / SS
            grad = np.stack([np.interp(yy, ys, cs[:, k]) for k in range(3)], -1)
            arr = np.repeat(grad[:, None, :], img.size[0], axis=1).astype(np.uint8)
            fill = Image.fromarray(np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)]), 'RGBA')
        img.paste(fill, (0, 0), m)


def draw_bar(img, cfg):
    """The grey bar under the frame: slanted ends, lighter along its middle (supersampled canvas)."""
    W, H = cfg['grid']
    for el in cfg['elements'].values():
        b = el.get('bar')
        if not b:
            continue
        x0, x1 = b['x'][0] * W, b['x'][1] * W
        y0, y1 = b['y'][0] * H, b['y'][1] * H
        sl = b['slant']
        poly = [(x0 + sl, y0), (x1 - sl, y0), (x1, y1), (x0, y1)]
        m = Image.new('L', img.size, 0)
        ImageDraw.Draw(m).polygon([(x * SS, y * SS) for x, y in poly], fill=255)
        yy = (np.arange(img.size[1]) + 0.5) / SS
        t = np.clip((yy - y0) / (y1 - y0), 0, 1)
        f = b['edge'] + (1 - b['edge']) * np.sin(np.pi * t) ** 0.7
        col = np.array(b['color'], np.float32)[None, :] * f[:, None]
        arr = np.repeat(np.clip(col, 0, 255)[:, None, :], img.size[0], axis=1).astype(np.uint8)
        img.paste(Image.fromarray(np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)]), 'RGBA'), (0, 0), m)


def draw_gauges(d, cfg, sc):
    """The small slanted gauges (fuel, oil): rail and empty track, the track cut by its gaps (polygons in grid px)."""
    W, H = cfg['grid']
    for el in cfg['elements'].values():
        if 'track' not in el:
            continue
        col = el['colors']
        P = lambda k: [(p[0] * W, p[1] * H) for p in el[k]]
        lerp = lambda a, b, t: (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        rtl, rtr, rbr, rbl = P('rail')
        for z in el.get('rail_zones') or [{'from': 0.0, 'to': 1.0, 'color': col['rail']}]:
            a, b = z['from'], z['to']                     # fractions from the bottom (cold / empty) end
            d.polygon(sc([lerp(rbl, rtl, b), lerp(rbr, rtr, b), lerp(rbr, rtr, a), lerp(rbl, rtl, a)]), fill=tuple(z['color']))
        tl, tr_, br, bl = P('track')
        span = bl[1] - tl[1]
        cuts = [0.0] + sorted(el.get('gaps', [])) + [1.0]
        g = el.get('gap_px', 0) / 2 / span
        for i in range(len(cuts) - 1):
            a, b = cuts[i] + (g if i else 0), cuts[i + 1] - (g if i < len(cuts) - 2 else 0)
            # t runs from the bottom (empty, 0) to the top (full, 1)
            d.polygon(sc([lerp(bl, tl, b), lerp(br, tr_, b), lerp(br, tr_, a), lerp(bl, tl, a)]), fill=tuple(col['track']))


def gauge_bevel(img, cfg, el):
    """Light 3D on a slanted gauge (img: RGBA at grid size): a brightness profile across the rail and the track,
    t from 0 at each strip's outer edge (its polygon's first edge) to 1 at the other."""
    bv = el.get('bevel')
    if not bv or 'track' not in el:
        return img
    W, H = cfg['grid']
    arr = np.asarray(img).copy()
    rgb = arr[..., :3].astype(np.float32)
    for key in ('rail', 'track'):
        tl, tr_, br, bl = [(p[0] * W, p[1] * H) for p in el[key]]
        y0, y1 = int(min(tl[1], bl[1])), int(max(tl[1], bl[1])) + 1
        x0, x1 = int(min(tl[0], tr_[0], br[0], bl[0])) - 1, int(max(tl[0], tr_[0], br[0], bl[0])) + 2
        X, Y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        f = (Y - tl[1]) / (bl[1] - tl[1])
        xo = tl[0] + (bl[0] - tl[0]) * f                     # the outer edge at that height
        xi = tr_[0] + (br[0] - tr_[0]) * f                    # the inner edge
        t = np.clip((X - xo) / (xi - xo), 0, 1)
        if key == 'rail':
            b = bv['rail']
            m = b['edge'] + (1 - b['edge']) * np.sin(np.pi * t) ** 0.6
            g = b['gloss'] * np.exp(-((t - b['gloss_at']) / b['gloss_w']) ** 2)
        else:
            b = bv['track']
            m = 1 + b['lit'] * np.exp(-t / b['lit_w']) - b['shade'] * smooth(0.5, 1.0, t)
            g = np.zeros_like(t)
        sub = rgb[y0:y1, x0:x1]
        mk = Image.new('L', (x1 - x0, y1 - y0), 0)           # only the strip itself (not the bracket around it)
        ImageDraw.Draw(mk).polygon([(px_ - x0, py - y0) for px_, py in (tl, tr_, br, bl)], fill=255)
        lit = np.asarray(mk) > 127
        out = np.clip(sub * m[..., None] + (255 - sub * m[..., None]) * g[..., None], 0, 255)
        rgb[y0:y1, x0:x1] = np.where(lit[..., None], out, sub)
    arr[..., :3] = (rgb + 0.5).astype(np.uint8)
    return Image.fromarray(arr, 'RGBA')


def draw_pump(img, x, y, w, h, color, ss=8):
    """BMW's fuel pump icon with its arrow (the filler side), drawn in a w x h px box at x, y; traced on the photo's
    35 x 28 px icon."""
    k = (w / 35.0, h / 28.0)
    m = Image.new('L', (int(w + 4) * ss, int(h + 4) * ss), 0)
    d = ImageDraw.Draw(m)
    P = lambda px_, py: ((2 + px_ * k[0]) * ss, (2 + py * k[1]) * ss)
    lw = int(round(2.6 * ss))
    d.rounded_rectangle([P(1, 1), P(16, 12)], radius=2 * ss, outline=255, width=int(round(3 * ss)))   # the window
    d.polygon([P(-1.5, 27), P(0.5, 22), P(17, 22), P(18, 27)], fill=255)                     # the foot, flared
    d.rounded_rectangle([P(0, 10), P(16.5, 27)], radius=1.5 * ss, fill=255)                # the body
    d.line([P(15, 3.5), P(19, 6.5), P(21, 9.5), P(21, 20), P(19.5, 23.5), P(16, 24.5)], fill=255, width=lw,
           joint='curve')                                                                   # the hose, apart from the body
    d.polygon([P(26, 7), P(34.5, 11.5), P(26, 17)], fill=255)                               # the arrow
    m = m.resize((int(w + 4), int(h + 4)), Image.LANCZOS)
    layer = Image.new('RGBA', m.size, tuple(color) + (0,))
    layer.putalpha(m)
    img.alpha_composite(layer, (int(round(x)) - 2, int(round(y)) - 2))


def _icon_canvas(w, h, ss):
    m = Image.new('L', (int(w + 4) * ss, int(h + 4) * ss), 0)
    k = (w / 48.0, h / 32.0)
    P = lambda px_, py: ((2 + px_ * k[0]) * ss, (2 + py * k[1]) * ss)
    return m, ImageDraw.Draw(m), P


def _icon_paste(img, m, x, y, w, h, color):
    m = m.resize((int(w + 4), int(h + 4)), Image.LANCZOS)
    layer = Image.new('RGBA', m.size, tuple(color) + (0,))
    layer.putalpha(m)
    img.alpha_composite(layer, (int(round(x)) - 2, int(round(y)) - 2))


def _thermometer(d, P, ss, cx, top, bulb_y, lw):
    """ISO 2575's thermometer: a stem, a round bulb, three marks on its right (long, short, long)."""
    d.line([P(cx, top), P(cx, bulb_y)], fill=255, width=int(round(2.8 * ss)))
    d.ellipse([P(cx - 3.2, bulb_y - 3.2), P(cx + 3.2, bulb_y + 3.2)], fill=255)
    for ty, ln in ((top + 1, 4.5), (top + 4.5, 3), (top + 8, 4.5)):
        d.line([P(cx + 2.5, ty), P(cx + 2.5 + ln, ty)], fill=255, width=lw)


def draw_oil(img, x, y, w, h, color, ss=8):
    """ISO 7000-0247, engine oil temperature: an oil can (handle, body, spout, a drop) with a thermometer standing in
    it, in the theme's line weight; w x h px box (48 x 32 on the photo)."""
    m, d, P = _icon_canvas(w, h, ss)
    lw = int(round(2.6 * ss))
    _thermometer(d, P, ss, 22, -3, 18, int(round(2 * ss)))
    d.line([P(9, 14), P(19, 14)], fill=255, width=lw)                                      # the can's top, left of it
    d.line([P(25, 14), P(34, 14)], fill=255, width=lw)                                     # ... and right of it
    d.line([P(9, 14), P(13, 25.5), P(31.5, 25.5), P(34, 14)], fill=255, width=lw, joint='curve')     # the body
    d.line([P(9, 14), P(4, 10.5), P(1, 13), P(4.5, 17.5), P(10.5, 18)], fill=255, width=lw, joint='curve')   # the handle loop
    d.line([P(33.5, 16.5), P(43.5, 10)], fill=255, width=lw)                                # the spout
    d.ellipse([P(41.5, 14.5), P(45.5, 19.5)], fill=255)                                     # the drop
    _icon_paste(img, m, x, y, w, h, color)


def draw_coolant(img, x, y, w, h, color, ss=8):
    """ISO 7000-0246, engine coolant temperature: a thermometer whose bulb stands on two waves, in the theme's line
    weight; same box as the oil icon."""
    m, d, P = _icon_canvas(w, h, ss)
    lw = int(round(2.6 * ss))
    _thermometer(d, P, ss, 24, -3, 15, int(round(2 * ss)))
    for wy in (22, 28):                                                                     # two waves, 2 periods
        pts = [P(6 + i * 1.5, wy + 1.8 * math.sin(i * 1.5 / 36 * 4 * math.pi)) for i in range(25)]
        d.line(pts, fill=255, width=lw, joint='curve')
    _icon_paste(img, m, x, y, w, h, color)


def draw_gauge_labels(img, cfg):
    W, H = cfg['grid']
    for el in cfg['elements'].values():
        lb = el.get('labels') if 'track' in el else None
        if not lb:
            continue
        if 'variants' in lb:                                      # picked by the engine (petrol / diesel)
            lb = dict(lb, **lb['variants'][cfg.get('engine', 'petrol')])
        imperial = cfg.get('units') == 'imperial'
        for it in (lb.get('items_imperial') if imperial and lb.get('items_imperial') else lb['items']):
            draw_number(img, it['text'], it['at'][0] * W, it['at'][1] * H, it['h'], 32.5, lb['color'],
                        align=it.get('align', 'left'), sx=lb.get('sx', 1.0), gap=lb.get('gap'))
        for tx in lb.get('text', []):
            if imperial and tx.get('text_imperial'):
                tx = dict(tx, text=tx['text_imperial'], fit_text=tx['text'])
            draw_labels(img, {'grid': cfg['grid'], 'elements': {'t': {'label': tx}}})
        ic = lb.get('icon')
        if ic:
            draw = {'pump': draw_pump, 'oil': draw_oil, 'coolant': draw_coolant}[ic['kind']]
            draw(img, ic['at'][0] * W, ic['at'][1] * H, ic['size'][0], ic['size'][1], lb['color'])


def build_static(cfg):
    """The static layer, RGBA at grid size: each bracket filled zone by zone (band, gap, stripe), then its lines and marks."""
    expand(cfg)
    for el in cfg['elements'].values():                        # the engine picks the rev scale
        rz = el.get('rev_zones')
        if rz and cfg.get('engine') in rz and el.get('numbers'):
            el['numbers']['scale'] = rz[cfg['engine']]['scale']
    W, H = cfg['grid']
    img = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 255))
    d = ImageDraw.Draw(img)
    sc = lambda poly: [(x * SS, y * SS) for x, y in poly]
    for el in cfg['elements'].values():
        col = el.get('colors')
        if not col or 'band' not in el:
            continue
        whole, to_band, to_par = bracket_zones(cfg, el)
        d.polygon(sc(whole), fill=tuple(col['band']))
        d.polygon(sc(to_band), fill=tuple(col['gap']))
        d.polygon(sc(to_par), fill=tuple(col['stripe']))
        out = fillet_polyline(to_px(cfg, el['outline']))
        top = [p for p in out if p[1] <= out[0][1] + 0.5]                       # the top line (level part)
        d.line(sc(top), fill=tuple(col['top_line']), width=3 * SS)
        d.line(sc(fillet_polyline(to_px(cfg, el['inner']))[1:]), fill=tuple(col['inner_line']), width=2 * SS, joint='curve')
        for x0, y0, x1, y1 in tick_rects(cfg, el):
            d.rectangle([x0 * SS, y0 * SS, x1 * SS, y1 * SS], fill=tuple(col['tick']))
    draw_panel(img, cfg)
    draw_bar(img, cfg)
    d = ImageDraw.Draw(img)
    draw_gauges(d, cfg, sc)
    img = img.resize((W, H), Image.LANCZOS)
    for el in cfg['elements'].values():
        if el.get('colors') and 'band' in el:
            img = bevel(img, cfg, el)
    for el in cfg['elements'].values():
        img = gauge_bevel(img, cfg, el)
    draw_numbers(img, cfg)
    draw_labels(img, cfg)
    draw_gauge_labels(img, cfg)
    W, H = cfg['grid']
    for el in cfg['elements'].values():                       # the live readouts, sampled with the photo's values
        dg = el.get('digits')
        if dg and el.get('sample'):
            dg = dict(dg, _min_left=dg['min_left'] * W if 'min_left' in dg else None)
            draw_big(img, el['sample'], dg['right'] * W, dg['top'] * H, dg, dg['color'])
    return img


def grid_overlay(img, step=50):
    d = ImageDraw.Draw(img, 'RGBA')
    W, H = img.size
    for x in range(0, W + 1, step):
        d.line([(x, 0), (x, H)], fill=(0, 255, 120, 120 if x % 100 == 0 else 50))
    for y in range(0, H + 1, step):
        d.line([(0, y), (W, y)], fill=(0, 255, 120, 120 if y % 100 == 0 else 50))
    for x in range(0, W, 100):
        for y in range(0, H, 100):
            d.text((x + 3, y + 2), '%d/%d' % (x, y), fill=(255, 255, 0, 200))


def compare(cfg, static, ref_path, out_path):
    W, H = cfg['grid']
    ref = Image.open(ref_path).convert('RGB').resize((W, H))
    gen = static.convert('RGB')
    blend = Image.blend(ref, gen, 0.5)
    d = ImageDraw.Draw(blend)
    for poly in outlines(cfg).values():
        d.line(poly, fill=(255, 0, 255), width=1)
    for r in tick_rects(cfg):
        d.rectangle(r, outline=(255, 0, 255))
    for im in (ref, gen, blend):
        grid_overlay(im)
    out = Image.new('RGB', (W, H * 3 + 20), (30, 30, 30))
    for i, im in enumerate((ref, gen, blend)):
        out.paste(im, (0, i * (H + 10)))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description='Draw a layout\'s static layer from its config.json')
    ap.add_argument('layout', help='category/layout, e.g. id8/A')
    ap.add_argument('--compare', action='store_true', help='stack it against the reference photo')
    a = ap.parse_args()
    cat, name = a.layout.split('/')
    folder = os.path.join(ROOT, 'layouts', cat, name)
    cfg = json.load(open(os.path.join(folder, 'config.json'), encoding='utf-8'))
    static = build_static(cfg)
    static.save(os.path.join(folder, 'static.png'))
    print('static: ' + os.path.relpath(os.path.join(folder, 'static.png'), ROOT))
    if a.compare:
        ref = os.path.join(ROOT, 'design', 'refs', cat, name + '.png')
        print('compare: ' + os.path.relpath(compare(cfg, static, ref, os.path.join(ROOT, 'design', 'out', '%s_%s_compare.png' % (cat, name))), ROOT))


if __name__ == '__main__':
    main()
