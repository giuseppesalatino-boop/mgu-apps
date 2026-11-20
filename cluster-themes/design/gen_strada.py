#!/usr/bin/env python3
"""gen_strada.py - writes the Strada layout (apps/dkombi4_layouts/hud/layouts/strada.json).

    python3 design/gen_strada.py

Strada is BMW's iDrive 7 Live Cockpit layout with the speed on the left, the revs on the right and the map in the
middle. Every number here was measured on a photo of a G-series cluster (design/photos/ref1.png, not in the repo),
straightened to 1920 x 720 with an affine fit (x' = (x - 145) * 1.686 + 100, y' = (y - 88) * 1.713 + 28) and a shear
that puts the two scales' axis of symmetry on x 960 (x'' = x' + 10 - 0.048 * (y' - 160)): design/photos/ref1_fix.png.
Edit the numbers here and run this again; don't edit the JSON by hand.
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'apps', 'dkombi4_layouts', 'hud', 'layouts', 'strada.json')

# The left scale's outer line, from its end by the middle (the bottom line, which steps up to y 610 towards the map)
# round the left side to its top end. Traced off the photo (the leftmost bright pixel of each row, averaged with the
# mirror of the right side), at most 32 points. The right scale is its mirror.
PATH = [[870, 610], [752, 610], [747, 611.5], [724, 629], [718, 632], [300, 632], [280, 630.5], [266, 627], [256, 621],
        [246, 612], [207, 570], [183, 536], [169, 512], [127, 428], [111, 388], [103, 360], [100.5, 344], [100.5, 328],
        [102, 312], [107, 292], [114.5, 276], [122, 264], [146.5, 232], [180.5, 196], [238, 148], [294.5, 108],
        [323, 91], [352, 76], [380, 64], [500, 43], [580, 35], [740, 26]]

# The point the long marks and the needle point at: BMW's run like spokes (slanted at the bottom, flat at the left)
CENTER = (700, 330)

# The numbers' centres on the photo: km/h 0..330 every 30, and the revs 0..8 (right side, screen x)
KMH = [(446, 568), (343, 568), (280, 497), (230, 407), (205, 320), (258, 238), (332, 180), (397, 130), (480, 107),
       (568, 92), (657, 88), (742, 85)]
RPM = [(1485, 575), (1598, 552), (1668, 452), (1715, 338), (1668, 245), (1583, 172), (1478, 125), (1363, 105),
       (1240, 97)]
LABEL_H = 27            # k_scale's box height; a centred label's box is (cx - 55, cy - 12, 110, 27)

TYRES = (1402, 357)     # the car glyph in the middle of the tyre block
TYRE_ICON_X = 1364


def seg_hit(p, d, a, b):
    """Where the ray p + t d (t > 0) crosses the segment a-b, or None."""
    ex, ey = b[0] - a[0], b[1] - a[1]
    den = d[0] * ey - d[1] * ex
    if abs(den) < 1e-9:
        return None
    t = ((a[0] - p[0]) * ey - (a[1] - p[1]) * ex) / den
    u = ((a[0] - p[0]) * d[1] - (a[1] - p[1]) * d[0]) / den
    return t if t > 0 and -1e-6 <= u <= 1 + 1e-6 else None


def nearest(p, a, b):
    ex, ey = b[0] - a[0], b[1] - a[1]
    h = max(0.0, min(1.0, ((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / (ex * ex + ey * ey or 1)))
    return (a[0] + ex * h, a[1] + ey * h)


def mark(label, digits):
    """A numbered mark: from the scale line, along the spoke from CENTER through the label, in to the label's edge."""
    dx, dy = label[0] - CENTER[0], label[1] - CENTER[1]
    L = math.hypot(dx, dy)
    d = (dx / L, dy / L)
    ts = [t for t in (seg_hit(label, d, PATH[i], PATH[i + 1]) for i in range(len(PATH) - 1)) if t is not None]
    if ts:
        t = min(ts)
        start = (label[0] + d[0] * t, label[1] + d[1] * t)
    else:                       # past the line's end (330 km/h): the line's nearest point
        start = min((nearest(label, PATH[i], PATH[i + 1]) for i in range(len(PATH) - 1)),
                    key=lambda q: math.hypot(q[0] - label[0], q[1] - label[1]))
    half = 0.5 * ((18 * digits + 6) * abs(d[0]) + LABEL_H * abs(d[1])) + 10   # the label's half extent along the spoke
    end = (label[0] + d[0] * half, label[1] + d[1] * half)
    return [round(start[0], 1), round(start[1], 1), round(end[0], 1), round(end[1], 1)]


def lerp_pt(pts, k):
    i = min(len(pts) - 2, int(k))
    f = k - i
    return (pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f)


def text(id_, x, y, w, h, txt, font, color='fg', align='center', **kw):
    t = {'id': id_, 'type': 'text', 'z': 5, 'x': round(x), 'y': round(y), 'w': w, 'h': h, 'text': txt, 'font': font,
         'color': color, 'align': align}
    t.update(kw)
    return t


def label(id_, c, txt, **kw):
    return text(id_, c[0] - 55, c[1] - 12, 110, LABEL_H, txt, 'k_scale', **kw)


def main():
    rpm_left = [(1920 - x, y) for x, y in RPM]
    # mph: BMW's mph face isn't measured; its numbers (0..200 every 20) sit where the km/h scale has that speed
    mph = [lerp_pt(KMH, m * 1.609344 / 30) for m in range(0, 201, 20)]
    marks = {
        'metric': [mark(c, len(str(i * 30))) for i, c in enumerate(KMH)],
        'us': [mark(c, len(str(i * 20))) for i, c in enumerate(mph)],
        'rpm': [mark(c, 1) for c in rpm_left],
    }
    W = []
    W.append({
        'id': 'mview', 'type': 'mview', 'z': 1, 'x': 0, 'y': 0, 'w': 1920, 'h': 660, 'class': 'content',
        'path': PATH, 'marks': marks, 'center': list(CENTER), 'needle': 70,
        # the fuel dial (the oil dial is its mirror): a circle through the photo's bracket, full at the top, empty below
        'dial': [240, 455, 195], 'dialAng': [188, 130],
        # the line under the speed: the double dash at x 300, the line on to x 720 at y 393; no mirror on the right
        'line': [300, 720, 393, 880], 'lineRight': False,
        # the dark panel inside each scale: its edge facing the middle, top / middle / bottom, and how far inside the line
        'panel': [[785, 95], [555, 345], [600, 600], 34],
        'tyres': [TYRES[0], TYRES[1], TYRE_ICON_X],
        # the Waze map window, an oval that fades into the black ("Show Waze map inside the theme")
        'hole': [660, 110, 600, 500], 'holeWhen': 'mapUp', 'holeInset': 0, 'holeOval': 70,
    })
    for i, c in enumerate(mph):
        W.append(label('spd_us%d' % i, c, str(i * 20), toggle='units=us'))
    for i, c in enumerate(KMH):
        W.append(label('spd_km%d' % i, c, str(i * 30), toggle='!units=us'))
    for i, c in enumerate(RPM):
        W.append(label('rpm%d' % i, c, str(i)))
    # the speed: big figures right-aligned to x 637 with the unit after them, over the line
    W.append(text('speed', 307, 316, 330, 68, '{{speed}}', 'k_speed', align='right'))
    W.append(text('speed_unit', 652, 358, 100, 29, '{{u:speedLc}}', 's_unit', 'dim', 'left'))
    W.append(text('scale_unit', 392, 598, 140, 29, '{{u:speedLc}}', 's_unit', 'dim'))
    W.append(text('rpm_unit', 1300, 600, 220, 29, '1/min x1000', 's_unit', 'dim'))
    # the tyre block: pressure over temperature at each corner, the units on top
    W.append(text('tyre_pu', 1392, 213, 90, 29, '{{u:pressure}}', 's_unit', align='left'))
    W.append(text('tyre_tu', 1365, 248, 80, 29, '{{u:temp}}', 's_unit'))
    for corner, cx, cy in (('FL', 1350, 300), ('FR', 1457, 300), ('RL', 1348, 392), ('RR', 1455, 392)):
        W.append(text('tp_' + corner, cx - 40, cy - 13, 80, 26, '{{tyreP:%s}}' % corner, 'k_val'))
        W.append(text('tt_' + corner, cx - 40, cy + 20, 80, 26, '{{tyreT:%s}}' % corner, 'k_val'))
    # the corner dials' numbers: fuel 1 / 1/2 / 0, oil 150 / 100 / 50
    for i, (txt, c) in enumerate((('1', (75, 447)), ('1/2', (103, 530)), ('0', (155, 588)))):
        W.append(text('fuel%d' % i, c[0] - 30, c[1] - 11, 60, 22, txt, 'k_gl'))
    for i, (t, c) in enumerate((('{{temp:150}}', (1822, 475)), ('{{temp:100}}', (1798, 540)), ('{{temp:50}}', (1745, 597)))):
        W.append(text('oil%d' % i, c[0] - 45, c[1] - 11, 90, 22, t, 'k_gl'))
    W.append({'id': 'fuel_icon', 'type': 'icon', 'icon': 'fuel', 'z': 5, 'x': 4, 'y': 575, 'w': 24, 'h': 28, 'color': 'dim'})
    W.append({'id': 'oil_icon', 'type': 'icon', 'icon': 'oilcan', 'z': 5, 'x': 1842, 'y': 600, 'w': 36, 'h': 24, 'color': 'dim'})
    W.append(text('oil_unit', 1862, 568, 30, 29, '{{u:temp}}', 's_unit', 'dim', 'left'))
    # the bottom row, as BMW's: range with the fuel pump, the clock, (the street: no signal), outside temperature;
    # BMW's own gear letter shows in the gap between the two covers
    W.append({'id': 'strip_cover_l', 'type': 'fill', 'class': 'content', 'z': 0, 'x': 0, 'y': 660, 'w': 1380, 'h': 60,
              'color': 'bg', 'stripRow': 'l'})
    W.append({'id': 'strip_cover_r', 'type': 'fill', 'class': 'content', 'z': 0, 'x': 1460, 'y': 660, 'w': 460, 'h': 60,
              'color': 'bg', 'stripRow': 'r'})
    W.append({'id': 'range_icon', 'type': 'icon', 'icon': 'fuel', 'z': 3, 'x': 136, 'y': 664, 'w': 26, 'h': 28,
              'color': 'fg', 'stripRow': 'l'})
    W.append(text('strip_range', 166, 678, 84, 26, '{{range}}', 'k_val', align='right', z=3, stripRow='l'))
    W.append(text('strip_range_u', 256, 676, 40, 29, '{{u:distLc}}', 's_unit', 'fg', 'left', z=3, stripRow='l'))
    W.append(text('strip_time', 337, 678, 124, 26, '{{time}}', 'k_val', align='right', z=3, stripRow='l'))
    W.append(text('strip_ampm', 466, 676, 35, 29, '{{ampmLc}}', 's_unit', 'fg', 'left', z=3, stripRow='l'))
    W.append(text('strip_out', 1560, 678, 80, 26, '{{ambient}}', 'k_val', align='right', z=3, stripRow='r'))
    W.append(text('strip_out_u', 1656, 676, 27, 29, '{{u:temp}}', 's_unit', 'fg', 'left', z=3, stripRow='r'))
    layout = {
        'schemaVersion': 2, 'name': 'Strada', 'theme': 'mdash', 'mapLive': True,
        '_about': "BMW iDrive 7 Live Cockpit with the speed on the left, the revs on the right and the map between them, "
                  "measured on a photo of a G-series cluster (generated by design/gen_strada.py: edit that and run it). "
                  "One mview effect layer draws the black backdrop, both scales (outer line traced off the photo, the "
                  "right one mirrored; BMW's numbered marks are not even along the line, so each number's mark is a "
                  "spoke from the line in towards it and a value goes between its two marks), the needles, the red "
                  "glow, the red zone, the dark panels, the fuel and oil dials and the tyre block's cross and car; the "
                  "numbers and readouts are text widgets. Not drawn, for want of a signal: the speed limit sign and the "
                  "street name. Widget reference: shared/hud.js header.",
        'widgets': W,
    }
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(layout, f, indent=2)
        f.write('\n')
    print('strada: %d widgets -> %s' % (len(W), os.path.relpath(OUT, ROOT)))


if __name__ == '__main__':
    main()
