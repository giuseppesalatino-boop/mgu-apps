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
        for key in ('bevel', 'gaps', 'gap_px', 'colors'):
            if key in src and key not in el:
                el[key] = src[key]
        el['_side'] = 'right'
    return cfg


def outlines(cfg):
    """Every element's outline as a dense px polyline (the reference for --compare)."""
    out = {}
    expand(cfg)
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
DIGIT_W = {'1': 18}


def draw_number(img, text, x_edge, y_mid, h, advance, color, ss=8, align='right'):
    """Text of BMW's digits into img (RGBA, grid px), its right (or left) edge at x_edge, centred on y_mid, h px tall."""
    k = h / 22.0
    widths = [DIGIT_W.get(ch, 29) * k for ch in text]
    total = sum(widths) + (advance - 29) * k * (len(text) - 1)
    x0, y0 = (x_edge - total if align == 'right' else x_edge), y_mid - h / 2
    W, H = int(total) + 4, int(h) + 4
    m = Image.new('L', (W * ss, H * ss), 0)
    d = ImageDraw.Draw(m)
    x = 2.0
    for ch, w in zip(text, widths):
        for part in DIGITS[ch]:
            if part[0] == 'quad':
                d.polygon([((x + px_ * k) * ss, (2 + py * k) * ss) for px_, py in part[1:]], fill=255)
            else:
                a, b, c, e = part
                d.rounded_rectangle([(x + a * k) * ss, (2 + b * k) * ss, (x + c * k) * ss - 1, (2 + e * k) * ss - 1],
                                    radius=1.6 * k * ss, fill=255)
        x += w + (advance - 29) * k
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
        l, t, r, b = font.getbbox(lb['text'], stroke_width=sw)  # the photo's text sets the scale
        k = (y1 - y0) / (b - t)
        l2, t2, r2, b2 = font.getbbox(text, stroke_width=sw)
        m = Image.new('L', (r2 - l2 + 8, max(b, b2) - t + 8), 0)      # room for descenders (the p of mph)
        ImageDraw.Draw(m).text((4 - l2, 4 - t), text, font=font, fill=255, stroke_width=sw, stroke_fill=255)
        if text != lb['text']:
            y0 -= lb.get('lift_mph', 0)                              # raised so the descender clears the line below
        m = m.resize((max(1, round(m.width * k)), max(1, round(m.height * k))), Image.LANCZOS)
        kx = 1.0
        layer = Image.new('RGBA', m.size, tuple(lb['color']) + (0,))
        layer.putalpha(m)
        img.alpha_composite(layer, (int(round(x0 - 4 * k * kx)), int(round(y0 - 4 * k))))


def draw_gauges(d, cfg, sc):
    """The small slanted gauges (fuel, oil): rail and empty track, the track cut by its gaps (polygons in grid px)."""
    W, H = cfg['grid']
    for el in cfg['elements'].values():
        if 'track' not in el:
            continue
        col = el['colors']
        P = lambda k: [(p[0] * W, p[1] * H) for p in el[k]]
        d.polygon(sc(P('rail')), fill=tuple(col['rail']))
        tl, tr_, br, bl = P('track')
        lerp = lambda a, b, t: (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        span = bl[1] - tl[1]
        cuts = [0.0] + sorted(el.get('gaps', [])) + [1.0]
        g = el.get('gap_px', 0) / 2 / span
        for i in range(len(cuts) - 1):
            a, b = cuts[i] + (g if i else 0), cuts[i + 1] - (g if i < len(cuts) - 2 else 0)
            # t runs from the bottom (empty, 0) to the top (full, 1)
            d.polygon(sc([lerp(bl, tl, b), lerp(br, tr_, b), lerp(br, tr_, a), lerp(bl, tl, a)]), fill=tuple(col['track']))


def build_static(cfg):
    """The static layer, RGBA at grid size: each bracket filled zone by zone (band, gap, stripe), then its lines and marks."""
    expand(cfg)
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
    draw_gauges(d, cfg, sc)
    img = img.resize((W, H), Image.LANCZOS)
    for el in cfg['elements'].values():
        if el.get('colors') and 'band' in el:
            img = bevel(img, cfg, el)
    draw_numbers(img, cfg)
    draw_labels(img, cfg)
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
