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
from PIL import Image, ImageDraw, ImageFilter

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


def outlines(cfg):
    """Every element's outline as a dense px polyline (the reference for --compare)."""
    out = {}
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


def tick_rects(cfg):
    """Every element's graduated marks as [x0, y0, x1, y1] px: level dashes measured from its parallel."""
    W, H = cfg['grid']
    out = []
    for el in cfg['elements'].values():
        t = el.get('ticks')
        if not t or 'parallel' not in el:
            continue
        par = fillet_polyline(to_px(cfg, el['parallel']))
        for yn in t['y']:
            y = yn * H
            x = x_on(par, y)
            if x is not None:
                out.append([x + t['start'], y - t['thick'] / 2, x + t['start'] + t['length'], y + t['thick'] / 2])
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


def build_static(cfg):
    """The static layer, RGBA at grid size: each bracket filled zone by zone (band, gap, stripe), then its lines and marks."""
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
        for x0, y0, x1, y1 in tick_rects(cfg):
            d.rectangle([x0 * SS, y0 * SS, x1 * SS, y1 * SS], fill=tuple(col['tick']))
    img = img.resize((W, H), Image.LANCZOS)
    for el in cfg['elements'].values():
        if el.get('colors') and 'band' in el:
            img = bevel(img, cfg, el)
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
