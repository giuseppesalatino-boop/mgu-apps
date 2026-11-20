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

from PIL import Image, ImageDraw

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


def build_static(cfg):
    """The static layer, RGBA at grid size. So far only the outlines are drawn (the first step: geometry)."""
    W, H = cfg['grid']
    img = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 255))
    d = ImageDraw.Draw(img)
    for poly in outlines(cfg).values():
        d.line([(x * SS, y * SS) for x, y in poly], fill=(90, 190, 255, 255), width=3 * SS, joint='curve')
    for x0, y0, x1, y1 in tick_rects(cfg):
        d.rectangle([x0 * SS, y0 * SS, x1 * SS, y1 * SS], fill=(215, 225, 240, 255))
    return img.resize((W, H), Image.LANCZOS)


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
