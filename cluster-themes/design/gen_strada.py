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

# BMW's fuel dial, the outer line of its double frame: from the end of the slanted top cap, round the left side, to
# the end of the foot along the bottom (the photo's points, straightened). Corsa has the same dials at the same place.
BRACKET = [[61.5, 432.3], [35.0, 422.0], [20.6, 442.5], [9.2, 468.2], [6.3, 493.9], [13.1, 528.2], [28.5, 559.0],
           [47.5, 583.0], [71.8, 605.3], [104.5, 625.8], [154.7, 634.4]]
# The oil dial's frame (screen px, from its top cap round the right side to its foot): BMW's is not the fuel one's
# mirror, it is narrower and sits lower
BRACKET_R = [[1821.1, 446.0], [1868.7, 437.4], [1877.4, 456.8], [1887.1, 485.4], [1886.9, 514.0], [1883.0, 542.4],
             [1871.9, 571.0], [1853.9, 593.8], [1824.7, 616.8], [1795.3, 631.0], [1767.6, 636.6], [1739.1, 642.5]]

# The point the long marks and the needle point at: BMW's run like spokes (slanted at the bottom, flat at the left)
CENTER = (700, 330)

# The numbers' centres on the photo: km/h 0..330 every 30, and the revs 0..8 (right side, screen x)
KMH = [(446, 568), (343, 568), (280, 497), (230, 407), (205, 320), (258, 238), (332, 180), (397, 130), (480, 107),
       (568, 92), (657, 88), (742, 85)]
RPM = [(1485, 575), (1598, 552), (1668, 452), (1715, 338), (1668, 245), (1583, 172), (1478, 125), (1363, 105),
       (1240, 97)]
LABEL_H = 32            # a centred label's box is (cx - 60, cy - 16, 120, 32): k_scale's cell

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
    return text(id_, c[0] - 60, c[1] - 16, 120, LABEL_H, txt, kw.pop('font', 'k_scale'), **kw)


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
        # the fuel dial's frame (the oil dial is its mirror): its outer line from the top cap's end round to the foot
        'bracket': BRACKET, 'bracketR': BRACKET_R, 'bracketY': [453, 552, 602], 'bracketYR': [470, 548, 600],
        # the line under the speed: the double dash at x 300, the line on to x 720 at y 393; no mirror on the right
        'line': [300, 720, 393, 880], 'lineRight': False,
        # the dark panel inside each scale: its edge facing the middle, top / middle / bottom, and how far inside the line
        # (left: its top, middle and bottom corner, how far inside the line it starts, its bottom edge's y)
        'panel': [[780, 95], [540, 330], [600, 565], 62, 565],
        # the right one is not the left one's mirror: top, middle, bottom corner (screen px) and its bottom edge's y
        'panelR': [[1195, 100], [1255, 330], [1330, 570], 570],
        'tyres': [TYRES[0], TYRES[1], TYRE_ICON_X],
        # the Waze map window, an oval that fades into the black ("Show Waze map inside the theme")
        'hole': [660, 110, 600, 500], 'holeWhen': 'mapUp', 'holeInset': 0, 'holeOval': 70,
    })
    for i, c in enumerate(mph):
        W.append(label('spd_us%d' % i, c, str(i * 20), toggle='units=us'))
    for i, c in enumerate(KMH):
        W.append(label('spd_km%d' % i, c, str(i * 30), toggle='!units=us'))
    for i, c in enumerate(RPM):
        W.append(text('rpm%d' % i, c[0] - 40, c[1] - 20, 80, 40, str(i), 'k_rpm'))
    # the speed: big figures right-aligned to x 637 with the unit after them, over the line
    W.append(text('speed', 297, 309, 340, 82, '{{speed}}', 'k_speed', align='right'))
    W.append(text('speed_unit', 652, 352, 100, 36, '{{u:speedLc}}', 'k_unit', 'fg', 'left'))
    W.append(text('scale_unit', 392, 594, 140, 36, '{{u:speedLc}}', 'k_unit', 'dim'))
    W.append(text('rpm_unit', 1300, 596, 220, 36, '1/min x1000', 'k_unit', 'dim'))
    # the tyre block: pressure over temperature at each corner, the units on top
    W.append(text('tyre_pu', 1384, 210, 90, 36, '{{u:pressure}}', 'k_unit', align='left'))
    W.append(text('tyre_tu', 1362, 245, 80, 36, '{{u:temp}}', 'k_unit'))
    for corner, cx, cy in (('FL', 1350, 300), ('FR', 1457, 300), ('RL', 1348, 392), ('RR', 1455, 392)):
        W.append(text('tp_' + corner, cx - 45, cy - 14, 90, 28, '{{tyreP:%s}}' % corner, 'k_val'))
        W.append(text('tt_' + corner, cx - 45, cy + 19, 90, 28, '{{tyreT:%s}}' % corner, 'k_val'))
    # the corner dials' numbers: fuel 1 / 1/2 / 0, oil 150 / 100 / 50
    for i, (txt, c) in enumerate((('1', (75, 447)), ('1/2', (103, 530)), ('0', (155, 588)))):
        W.append(text('fuel%d' % i, c[0] - 35, c[1] - 13, 70, 26, txt, 'k_gl'))
    for i, (t, c) in enumerate((('{{temp:150}}', (1822, 475)), ('{{temp:100}}', (1798, 540)), ('{{temp:50}}', (1745, 597)))):
        W.append(text('oil%d' % i, c[0] - 50, c[1] - 13, 100, 26, t, 'k_gl'))
    W.append({'id': 'fuel_icon', 'type': 'icon', 'icon': 'fuel', 'z': 5, 'x': 2, 'y': 564, 'w': 30, 'h': 34, 'color': 'fg'})
    W.append({'id': 'oil_icon', 'type': 'icon', 'icon': 'oilcan', 'z': 5, 'x': 1848, 'y': 600, 'w': 36, 'h': 24, 'color': 'fg'})
    W.append(text('oil_unit', 1866, 562, 50, 36, '{{u:temp}}', 'k_unit', 'fg', 'left'))
    # the bottom row, as BMW's: range with the fuel pump, the clock, (the street: no signal), outside temperature;
    # BMW's own gear letter shows in the gap between the two covers
    W.append({'id': 'strip_cover_l', 'type': 'fill', 'class': 'content', 'z': 0, 'x': 0, 'y': 660, 'w': 1380, 'h': 60,
              'color': 'bg', 'stripRow': 'l'})
    W.append({'id': 'strip_cover_r', 'type': 'fill', 'class': 'content', 'z': 0, 'x': 1460, 'y': 660, 'w': 460, 'h': 60,
              'color': 'bg', 'stripRow': 'r'})
    W.append({'id': 'range_icon', 'type': 'icon', 'icon': 'fuel', 'z': 3, 'x': 124, 'y': 664, 'w': 32, 'h': 36,
              'color': 'fg', 'stripRow': 'l'})
    W.append(text('strip_range', 160, 668, 100, 38, '{{range}}', 'k_strip', align='right', z=3, stripRow='l'))
    W.append(text('strip_range_u', 262, 670, 46, 36, '{{u:distLc}}', 'k_unit', 'fg', 'left', z=3, stripRow='l'))
    W.append(text('strip_time', 322, 668, 136, 38, '{{time}}', 'k_strip', align='right', z=3, stripRow='l'))
    W.append(text('strip_ampm', 462, 670, 44, 36, '{{ampmLc}}', 'k_unit', 'fg', 'left', z=3, stripRow='l'))
    W.append(text('strip_out', 1530, 668, 120, 38, '{{ambient}}', 'k_strip', align='right', z=3, stripRow='r'))
    W.append(text('strip_out_u', 1654, 670, 40, 36, '{{u:temp}}', 'k_unit', 'fg', 'left', z=3, stripRow='r'))
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
