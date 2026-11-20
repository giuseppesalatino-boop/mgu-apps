#!/usr/bin/env python3
"""oem_corsa.py - a 30 fps reference renderer of BMW's iDrive 7 "Corsa" cluster layout (gear between two rev blades).

The calibration bench for the Corsa theme: it draws the layout from the measurements in CONFIG so they can be checked
against the straightened photo (design/photos/ref2_fix.png, not in the repo) before they go into the theme's
generator and shaders. Not part of the car app.

    python3 design/oem_corsa.py                        # 10 s of demo telemetry, frame times, last frame -> corsa.png
    python3 design/oem_corsa.py --grid --ref design/photos/ref2_fix.png --frames 1 --save calib.png
    python3 design/oem_corsa.py --window               # in a window, if pygame is installed

Needs numpy and Pillow (pygame only for --window). Fonts: tools/fonts-src (Michroma, Barlow; SIL OFL).
Everything that doesn't move is drawn once by build_static_layers() at SUPERSAMPLE x and scaled down; a frame is a
copy of that plus the moving parts (blade fills, bars, dial levels, needles, numbers), blended into small regions only.
"""
import argparse
import math
import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

WIDTH, HEIGHT = 1920, 720
DEBUG_GRID = False          # True: the 0.1 grid and the geometric references over the static layer
SUPERSAMPLE = 3             # the static layer is drawn this many times larger, then scaled down (anti-aliasing)
TARGET_FPS = 30

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, '..', 'tools', 'fonts-src')

# ---------------------------------------------------------------------------------------------------------------------
# CONFIG: every position is normalized (x: 0..1 of WIDTH, y: 0..1 of HEIGHT), measured on the straightened photo
# (1920 x 720, the two halves symmetric about x = 0.5). Lengths across are fractions of WIDTH ("w"), along y of HEIGHT
# ("h"). Colours are RGB 0..255. The right half mirrors the left unless it has its own entry.
# ---------------------------------------------------------------------------------------------------------------------
CONFIG = {
    'colors': {
        'bg': (0, 0, 0),
        'white': (238, 240, 244),
        'text': (242, 244, 247),
        'dim': (150, 155, 162),
        'line_grey': (120, 125, 132),
        'card': (34, 36, 40),             # the side cards' fill
        'card_rim': (92, 96, 104),
        'card_dot': (58, 61, 66),         # the cards' perforation
        'glass_hi': (205, 208, 214),      # the blades' glass by the outer edge
        'glass_lo': (38, 40, 45),         # ... by the inner edge
        'red': (226, 28, 36),
        'red_hot': (255, 70, 55),
        'orange': (255, 150, 60),
        'blue': (60, 130, 255),
        'green': (64, 222, 80),
        'track': (78, 82, 88),            # the bars' empty track
    },
    # The rev blades. The left blade's outer edge: x = bend.x + k * (sqrt(dy^2 + round^2) - round), dy = y - bend.y
    # (k_top above the bend, k_bot below): a bow, traced row by row off the photo (both blades averaged).
    'blade': {
        'bend': (0.2568, 0.4583),         # (493, 330) px
        'k_top': 0.85, 'k_bot': 0.455,    # px across per px down
        'round_h': 0.1208,                # 87 px: the bend's roundness
        'y_top': 0.1083, 'y_bot': 0.8306, # 78 / 598 px: the blade's ends
        # width across (w) at a height (h), measured: narrow at the top, widest at the bend
        'width': [(0.1083, 0.0250), (0.1528, 0.0297), (0.2917, 0.0401), (0.4583, 0.0443), (0.5694, 0.0406),
                  (0.7083, 0.0359), (0.8306, 0.0313)],
        'foot_h': 0.0347,                 # 25 px: the outlined block at the bottom
        'border_w': 0.0016,               # 3 px: the bright outer edge
        # the y of each thousand rpm, 0..8 (the photo's labels and marks: not even, the bow foreshortens them)
        'rpm_y': [0.8139, 0.7292, 0.6444, 0.5472, 0.4500, 0.3611, 0.2722, 0.2000, 0.1278],
        'labels_x': {0: 0.3469, 2: 0.3302, 4: 0.3208, 6: 0.3396, 8: 0.3672},   # 666, 634, 616, 652, 705 px
        'tick_long': 0.0125, 'tick_short': 0.0063,   # 24 / 12 px in from the outer edge
        'rpm_max': 8000, 'redline': 7500,
        'glow_up_h': 0.075,               # the red glow above the rpm fades over 54 px
        'lit_fade_h': 0.18,               # ... and the lit part below it over 130 px
    },
    'center': {
        'gear': (0.5104, 0.4444), 'gear_h': 0.139,          # 980, 320 px; 100 px tall
        'mode_icon': (0.5677, 0.4792), 'mode_icon_size': (0.0234, 0.0556),
        'line_y': 0.5486, 'line_x': (0.3828, 0.6380),       # 395 px, from 735 to 1225 px
        'line_cap_w': 0.0339,                               # 65 px bright caps at both ends
        'speed': (0.5130, 0.6431), 'speed_h': 0.0972,       # 985, 463 px; 70 px tall
        'speed_unit': (0.5964, 0.6208),                     # "km/h", 1145, 447 px
        'rpm_unit': (0.7526, 0.8236),                       # "1/min x1000", 1445, 593 px
    },
    # The side cards: the left one's outer edge (from its lower left round its top), its inner edge keeps 'gap' off
    # the blade, its bottom is straight at 'bottom_y'. The right card is its mirror.
    'card': {
        'outer': [(0.0521, 0.6667), (0.0495, 0.4583), (0.0615, 0.3472), (0.0885, 0.2500), (0.1302, 0.1778),
                  (0.1823, 0.1389), (0.2448, 0.1222), (0.3063, 0.1194)],
        'gap_w': 0.0156,                  # 30 px off the blade
        'bottom_y': 0.7847,               # 565 px
        'corner_w': 0.0313,               # 60 px: the rounding where the bottom meets the outer edge
        'dot_pitch_w': 0.0026,            # 5 px perforation
    },
    # The two bars on the left card: coolant (70..130 C, red from 120) and boost (0..20 psi)
    'bars': {
        'coolant': {'x': (0.0990, 0.2135), 'y': 0.3639, 'h': 0.0111, 'min': 70, 'max': 130, 'red_from': 120,
                    'labels': [(70, 0.0990, 0.4028), (100, 0.1615, 0.4028), (130, 0.2198, 0.4056)],
                    'icon': (0.1120, 0.2986), 'unit': (0.2057, 0.3083), 'unit_text': '°C'},
        'boost': {'x': (0.0990, 0.2135), 'y': 0.5444, 'h': 0.0111, 'min': 0, 'max': 20, 'red_from': None,
                  'labels': [(0, 0.0911, 0.5861), (10, 0.1589, 0.5861), (20, 0.2214, 0.5903)],
                  'icon': (0.1094, 0.4792), 'unit': (0.2005, 0.4931), 'unit_text': 'psi'},
    },
    # The tyre block on the right card: the car in the middle, pressure over temperature at each corner
    'tyres': {
        'car': (0.8344, 0.4931), 'icon': (0.8073, 0.3167), 'unit_p': (0.8307, 0.3167), 'unit_t': (0.8292, 0.3653),
        'cells': {'FL': (0.8047, 0.4139, 0.4653), 'FR': (0.8620, 0.4167, 0.4667),
                  'RL': (0.8047, 0.5389, 0.5903), 'RR': (0.8620, 0.5417, 0.5903)},   # x, y of pressure, y of temp
        'cross_h': [(0.7943, 0.8240), (0.8448, 0.8750)],    # the horizontal strokes' x ranges, at the car's y
        'cross_v': [(0.3958, 0.4583), (0.5347, 0.6042)],    # the vertical strokes' y ranges, at the car's x
    },
    # BMW's fuel and oil dials: the outer line of each double frame, from the top cap's end round to the foot's end
    # (the oil one is its own, not the fuel one's mirror), the y of full / half / empty (not evenly spaced), labels.
    'dials': {
        'fuel': {'frame': [(0.0320, 0.6004), (0.0182, 0.5861), (0.0107, 0.6146), (0.0048, 0.6503), (0.0033, 0.6860),
                           (0.0068, 0.7336), (0.0148, 0.7764), (0.0247, 0.8097), (0.0374, 0.8407), (0.0544, 0.8692),
                           (0.0806, 0.8811)],
                 'levels_y': (0.6292, 0.7667, 0.8361),
                 'labels': [('1', 0.0385, 0.6250), ('1/2', 0.0534, 0.7326), ('0', 0.0807, 0.8160)],
                 'icon': (0.0078, 0.8056), 'zone': 'reserve'},
        'oil': {'frame': [(0.9485, 0.6194), (0.9733, 0.6075), (0.9778, 0.6344), (0.9829, 0.6742), (0.9828, 0.7139),
                          (0.9807, 0.7533), (0.9749, 0.7931), (0.9656, 0.8247), (0.9504, 0.8567), (0.9351, 0.8764),
                          (0.9206, 0.8842), (0.9058, 0.8924)],
                'levels_y': (0.6528, 0.7611, 0.8333),
                'labels': [('150', 0.9479, 0.6597), ('100', 0.9362, 0.7500), ('50', 0.9089, 0.8264)],
                'icon': (0.9714, 0.8500), 'unit': (0.9792, 0.8056), 'zone': 'temp', 'min': 50, 'max': 150},
        'channel': (7.0, 19.0),           # px in from the frame line: the dull tube, then its bright inner line
        'surface': 56.0,                  # px of lit curved surface past the tube
        'slope': 0.14,                    # the needle and the dividers rise this much per px inwards
        'tube_lit': (0.62, 0.625, 0.64), 'tube_dark': (0.11, 0.112, 0.12),          # the channel, 0..1 RGB
        'surface_lit': (0.95, 0.955, 0.965), 'surface_dark': (0.2, 0.205, 0.215),   # the surface past it
        'surface_fall': 28.0,             # px over which the surface's light fades inwards
    },
    'strip': {
        'range_icon': (0.0781, 0.9444), 'range': (0.1302, 0.9444), 'range_unit': (0.1328, 0.9444),
        'time': (0.2448, 0.9444), 'ampm': (0.2474, 0.9444), 'ambient': (0.8594, 0.9444), 'ambient_unit': (0.8620, 0.9444),
        'h': 0.0500,                      # 36 px tall figures
    },
    'limit_sign': {'c': (0.1677, 0.7542), 'r_h': 0.0514},  # 322, 543 px, 37 px
    # the light falling below the blades: thin rays down from the foot and a line on towards the middle (left side)
    'floor': {'rays': [((0.3255, 0.8333), (0.3125, 0.9861)), ((0.3411, 0.8333), (0.3542, 0.9861))],
              'line': ((0.3464, 0.8306), (0.3958, 0.7500))},
    'fonts': {
        # name: (file, px at 1920 x 720, stroke px, squeeze across)
        'gear': ('Michroma-Regular.ttf', 118, 6, 0.80),
        'speed': ('Michroma-Regular.ttf', 84, 4, 0.80),
        'rpm': ('Michroma-Regular.ttf', 40, 2, 0.82),
        'scale': ('Michroma-Regular.ttf', 25, 1, 0.82),
        'value': ('Michroma-Regular.ttf', 28, 1, 0.82),
        'strip': ('Michroma-Regular.ttf', 36, 2, 0.80),
        'unit': ('BarlowSemiCondensed-SemiBold.ttf', 30, 0, 1.0),
        'sign': ('BarlowCondensed-SemiBold.ttf', 34, 0, 1.0),
        'grid': ('Barlow-Medium.ttf', 14, 0, 1.0),
    },
    'alarms': {'coolant_c': 120, 'oil_c': 140, 'fuel': 0.12, 'blink_hz': 2.0},
}


# ---------------------------------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------------------------------
def px(p, s=1.0):
    """A normalized point -> pixels at scale s."""
    return (p[0] * WIDTH * s, p[1] * HEIGHT * s)


def mirror_x(nx):
    return 1.0 - nx


def rgb01(c):
    return np.array(c, np.float32) / 255.0


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Sprite:
    """A small RGBA patch (float 0..1, straight alpha) placed at x, y on the frame."""
    __slots__ = ('rgb', 'a', 'x', 'y')

    def __init__(self, rgb, a, x, y):
        self.rgb, self.a, self.x, self.y = rgb, a, int(x), int(y)


def blend(frame, rgb, a, x, y):
    """Blend an RGB (h, w, 3 float 0..1) patch with alpha (h, w) into the uint8 frame at x, y, clipped."""
    h, w = a.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(WIDTH, x + w), min(HEIGHT, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    sa = a[y0 - y:y1 - y, x0 - x:x1 - x, None]
    sc = rgb[y0 - y:y1 - y, x0 - x:x1 - x] if rgb.ndim == 3 else rgb
    dst = frame[y0:y1, x0:x1].astype(np.float32) * (1.0 / 255.0)
    frame[y0:y1, x0:x1] = ((dst + (sc - dst) * sa) * 255.0 + 0.5).astype(np.uint8)


def rgba_image_to_arrays(img):
    a = np.asarray(img, np.float32) * (1.0 / 255.0)
    return a[..., :3].copy(), a[..., 3].copy()


# ---------------------------------------------------------------------------------------------------------------------
# text: rendered once per (font, string) and kept; the numbers that change come back from the cache in a few us
# ---------------------------------------------------------------------------------------------------------------------
class TextCache:
    def __init__(self):
        self.fonts, self.cache = {}, {}

    def font(self, name, scale=1.0):
        key = (name, scale)
        if key not in self.fonts:
            file, size, stroke, sx = CONFIG['fonts'][name]
            path = os.path.join(FONTS, file)
            f = ImageFont.truetype(path, max(6, int(round(size * scale)))) if os.path.exists(path) else ImageFont.load_default()
            self.fonts[key] = (f, int(round(stroke * scale)), sx)
        return self.fonts[key]

    def image(self, text, name, color, scale=1.0):
        """RGBA PIL image of the text, squeezed as CONFIG says."""
        key = (text, name, color, scale)
        img = self.cache.get(key)
        if img is None:
            f, stroke, sx = self.font(name, scale)
            l, t, r, b = f.getbbox(text, stroke_width=stroke)
            w, h = max(1, r - l + 2), max(1, b - t + 2)
            img = Image.new('RGBA', (w, h), color + (0,))
            ImageDraw.Draw(img).text((1 - l, 1 - t), text, font=f, fill=color + (255,), stroke_width=stroke,
                                     stroke_fill=color + (255,))
            if sx != 1.0:
                img = img.resize((max(1, round(w * sx)), h), Image.LANCZOS)
            self.cache[key] = img
        return img

    def sprite(self, text, name, color, anchor, align='center'):
        """A Sprite of the text centred vertically on anchor (normalized), aligned left / center / right on its x."""
        key = ('spr', text, name, color, anchor, align)
        spr = self.cache.get(key)
        if spr is None:
            img = self.image(text, name, color)
            x, y = px(anchor)
            x -= {'left': 0, 'center': img.width / 2, 'right': img.width}[align]
            rgb, a = rgba_image_to_arrays(img)
            spr = self.cache[key] = Sprite(rgb, a, round(x), round(y - img.height / 2))
        return spr


TEXT = TextCache()


def draw_text_static(canvas, text, name, color, anchor, align='center', s=1.0):
    """Text into a (supersampled) static PIL canvas."""
    img = TEXT.image(text, name, color, s)
    x, y = px(anchor, s)
    x -= {'left': 0, 'center': img.width / 2, 'right': img.width}[align]
    canvas.alpha_composite(img, (int(round(x)), int(round(y - img.height / 2))))


# ---------------------------------------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------------------------------------
def blade_outer_x(y, s=1.0):
    """The left blade's outer edge x at pixel row(s) y (scale s)."""
    B = CONFIG['blade']
    bx, by = px(B['bend'], s)
    r = B['round_h'] * HEIGHT * s
    dy = np.asarray(y, np.float64) - by
    k = np.where(dy < 0, B['k_top'], B['k_bot'])
    return bx + k * (np.sqrt(dy * dy + r * r) - r)


def blade_width(y, s=1.0):
    B = CONFIG['blade']
    ys = [p[0] * HEIGHT * s for p in B['width']]
    ws = [p[1] * WIDTH * s for p in B['width']]
    return np.interp(y, ys, ws)


def rpm_to_y(rpm, s=1.0):
    B = CONFIG['blade']
    t = np.clip(rpm / 1000.0, 0, len(B['rpm_y']) - 1)
    return np.interp(t, np.arange(len(B['rpm_y'])), B['rpm_y']) * HEIGHT * s


def polyline_field(pts, X, Y):
    """For each pixel: arc length s of the nearest point of the polyline, signed distance sd (+ to the right of the
    direction of travel, i.e. inside the bracket), and how far past the polyline's ends the nearest point lies."""
    best = np.full(X.shape, 1e9)
    S = np.zeros(X.shape)
    SD = np.zeros(X.shape)
    OFF = np.zeros(X.shape)
    acc = 0.0
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        ex, ey = bx - ax, by - ay
        L = math.hypot(ex, ey) or 1e-6
        hr = ((X - ax) * ex + (Y - ay) * ey) / (L * L)
        h = np.clip(hr, 0, 1)
        dx, dy = X - (ax + ex * h), Y - (ay + ey * h)
        d = np.hypot(dx, dy)
        m = d < best
        best = np.where(m, d, best)
        S = np.where(m, acc + h * L, S)
        SD = np.where(m, (ey * dx - ex * dy) / L, SD)
        OFF = np.where(m, np.abs(hr - h) * L, OFF)
        acc += L
    return S, SD, OFF, acc


# ---------------------------------------------------------------------------------------------------------------------
# static pieces (drawn at SUPERSAMPLE x)
# ---------------------------------------------------------------------------------------------------------------------
def card_polygon(s, right=False):
    """The left card's outline (closed polygon, px at scale s), its corners rounded by Chaikin cuts."""
    C = CONFIG['card']
    outer = [px(p, s) for p in C['outer']]
    yb = C['bottom_y'] * HEIGHT * s
    top_y = outer[-1][1]
    gap = C['gap_w'] * WIDTH * s
    inner = [(float(blade_outer_x(y, s)) - gap, y) for y in np.linspace(top_y, yb, 24)]
    corner = C['corner_w'] * WIDTH * s
    bottom = [(inner[-1][0], yb), (outer[0][0] + corner, yb), (outer[0][0], yb - corner)]
    pts = outer + inner + bottom
    for _ in range(3):                                   # Chaikin: rounds every corner a little
        out = []
        for i in range(len(pts)):
            p, q = pts[i], pts[(i + 1) % len(pts)]
            out += [(0.75 * p[0] + 0.25 * q[0], 0.75 * p[1] + 0.25 * q[1]),
                    (0.25 * p[0] + 0.75 * q[0], 0.25 * p[1] + 0.75 * q[1])]
        pts = out
    if right:
        pts = [(WIDTH * s - x, y) for x, y in pts]
    return pts


def draw_cards(canvas, s):
    col = CONFIG['colors']
    pitch = max(2, int(round(CONFIG['card']['dot_pitch_w'] * WIDTH * s)))
    for right in (False, True):
        poly = card_polygon(s, right)
        mask = Image.new('L', canvas.size, 0)
        ImageDraw.Draw(mask).polygon(poly, fill=255)
        layer = Image.new('RGBA', canvas.size, col['card'] + (0,))
        layer.putalpha(mask)
        canvas.alpha_composite(layer)
        # the perforation: a dot every pitch px, inside the card
        dots = Image.new('L', canvas.size, 0)
        dd = ImageDraw.Draw(dots)
        x0, y0, x1, y1 = (min(p[0] for p in poly), min(p[1] for p in poly), max(p[0] for p in poly), max(p[1] for p in poly))
        rr = pitch * 0.22
        for yy in np.arange(y0, y1, pitch):
            for xx in np.arange(x0 + (pitch / 2 if int(yy / pitch) % 2 else 0), x1, pitch):
                dd.ellipse((xx - rr, yy - rr, xx + rr, yy + rr), fill=150)
        dots = Image.fromarray(np.minimum(np.asarray(dots), np.asarray(mask)))
        dl = Image.new('RGBA', canvas.size, col['card_dot'] + (0,))
        dl.putalpha(dots)
        canvas.alpha_composite(dl)
        ImageDraw.Draw(canvas).line(poly + [poly[0]], fill=col['card_rim'] + (200,), width=max(1, int(s * 1.2)))


def blade_arrays(s):
    """The left blade at scale s over its bounding box: the static RGBA (glass, edges, foot, marks, red zone), the
    mask of its glass for the moving fill, and the box (x0, y0)."""
    B, col = CONFIG['blade'], CONFIG['colors']
    yt, yb = B['y_top'] * HEIGHT * s, B['y_bot'] * HEIGHT * s
    ys = np.arange(int(yt) - 2, int(yb) + 3)
    xo = blade_outer_x(ys + 0.5, s)
    w = blade_width(ys + 0.5, s)
    x0, x1 = int(xo.min()) - 4, int((xo + w).max()) + 4
    xs = np.arange(x0, x1)
    X, Y = np.meshgrid(xs + 0.5, ys + 0.5)
    XO = blade_outer_x(Y, s)
    Wd = blade_width(Y, s)
    d = X - XO                                   # px into the blade from its outer edge
    t = d / Wd
    aa = 1.0                                     # one px of edge softening at this scale
    inside = np.clip(d / aa + 0.5, 0, 1) * np.clip((Wd - d) / aa + 0.5, 0, 1) \
        * np.clip((Y - yt) / aa + 0.5, 0, 1) * np.clip((yb - Y) / aa + 0.5, 0, 1)
    rgb = np.zeros(X.shape + (3,), np.float32)
    a = np.zeros(X.shape, np.float32)

    def put(color, alpha):
        nonlocal rgb, a
        alpha = np.clip(alpha, 0, 1).astype(np.float32)
        rgb = rgb * (1 - alpha[..., None]) + rgb01(color) * alpha[..., None]
        a = a + (1 - a) * alpha

    # the glass: bright by the outer edge, dark inside, a little lighter towards the bottom
    shade = (1 - np.clip(t, 0, 1)) ** 1.6
    glass = rgb01(col['glass_lo']) + (rgb01(col['glass_hi']) - rgb01(col['glass_lo'])) * shade[..., None]
    rgb, a = glass.astype(np.float32), (inside * 0.92).astype(np.float32)
    # the foot: an outlined lighter block
    foot = B['foot_h'] * HEIGHT * s
    infoot = (Y > yb - foot)
    put((150, 154, 162), inside * infoot * 0.55)
    put(col['white'], inside * np.clip(1.6 * s - np.abs(Y - (yb - foot)), 0, 1))
    # the edges: a bright outer line with a fainter second line, a thin grey inner line
    bw = B['border_w'] * WIDTH * s
    ends = np.clip((Y - yt) + 0.5, 0, 1) * np.clip((yb - Y) + 0.5, 0, 1)
    put(col['white'], np.clip(bw - np.abs(d - bw * 0.5), 0, 1) * ends)
    put(col['white'], 0.45 * np.clip(0.7 * s - np.abs(d - 2.4 * bw), 0, 1) * ends)
    put(col['line_grey'], np.clip(0.9 * s - np.abs(d - Wd), 0, 1) * ends)
    # the red zone: from the redline to the top, by the outer edge
    yr = rpm_to_y(B['redline'], s)
    put(col['red'], np.clip(Y < yr, 0, 1) * np.clip(1 - np.abs(d - 2.2 * bw) / (2.2 * bw), 0, 1) * ends)
    # the marks: every 1000 rpm long, every 500 short, from the outer edge in
    for k in range(0, B['rpm_max'] // 500 + 1):
        ym = rpm_to_y(k * 500, s)
        ln = (B['tick_long'] if k % 2 == 0 else B['tick_short']) * WIDTH * s
        put(col['white'], np.clip(1.4 * s - np.abs(Y - ym), 0, 1) * (d > 0) * np.clip(ln - d, 0, 1))
    # the moving fill's mask: the glass, inside the edges
    fill_mask = inside * np.clip((d - 1.5 * bw) / (bw * 0.6), 0, 1) * np.clip((Wd - d - 1.0 * s) / s, 0, 1)
    return rgb, a, fill_mask.astype(np.float32), (x0, int(ys[0]))


def dial_arrays(key, s):
    """One fuel / oil dial over its box at scale s: RGBA lit (as if the level were at the top) and unlit, the pixel
    y and the signed distance sd (for the needle and the level), the box."""
    D, col = CONFIG['dials'], CONFIG['colors']
    cfg = D[key]
    right = key == 'oil'
    pts = [px(p, s) for p in cfg['frame']]
    if right:                                         # walk it in the left frame's direction (the inside on the right)
        pts = [(WIDTH * s - x, y) for x, y in pts]
    xs_all = [p[0] for p in pts]
    ys_all = [p[1] for p in pts]
    m = 70 * s
    x0, x1 = int(min(xs_all) - 12 * s), int(max(xs_all) + m)
    y0, y1 = int(min(ys_all) - 12 * s), int(max(ys_all) + 12 * s)
    X, Y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    S, SD, OFF, total = polyline_field(pts, X, Y)     # SD + inside the bracket (the frame runs anticlockwise)
    c0, c1 = D['channel'][0] * s, D['channel'][1] * s
    surf = D['surface'] * s
    in_span = 1 - smoothstep(0, 1.0 * s, OFF)
    in_soft = 1 - smoothstep(0, 14 * s, OFF)
    end_zone = smoothstep(total - 62 * s, total - 58 * s, S)
    hot = (1 - smoothstep(44 * s, 48 * s, S)) if right else np.zeros_like(S)
    cold = smoothstep(total - 42 * s, total - 38 * s, S) if right else np.zeros_like(S)
    reserve = np.zeros_like(S) if right else end_zone
    yt = [v * HEIGHT * s for v in cfg['levels_y']]

    def build(lit):
        rgb = np.zeros(X.shape + (3,), np.float32)
        a = np.zeros(X.shape, np.float32)

        def put(color, alpha):
            nonlocal rgb, a
            alpha = np.clip(alpha, 0, 1).astype(np.float32)
            c = color if isinstance(color, np.ndarray) and color.ndim == 3 else rgb01(color)
            rgb = rgb * (1 - alpha[..., None]) + c * alpha[..., None]
            a = a + (1 - a) * alpha

        # the channel: a dull tube
        u = (SD - c0) / (c1 - c0)
        shade = 0.4 + 0.6 * np.sin(np.pi * np.clip(u, 0, 1))
        base = D['tube_lit'] if lit else D['tube_dark']
        ch = np.array(base, np.float32)[None, None, :] * (0.7 + 0.5 * shade)[..., None]
        zone_red = np.array((0.62, 0.06, 0.08), np.float32)[None, None, :] * (0.7 + 0.5 * shade)[..., None]
        zone_blue = np.array((0.10, 0.32, 0.80), np.float32)[None, None, :] * (0.7 + 0.5 * shade)[..., None]
        ch = ch + (zone_red - ch) * (np.maximum(reserve, hot) * 0.85)[..., None]
        ch = ch + (zone_blue - ch) * (cold * 0.85)[..., None]
        put(ch, (SD > c0 - 1 * s) * (SD < c1 + 1 * s) * np.clip((SD - c0 + 0.5 * s) / s, 0, 1) * in_soft)
        # the surface past the tube: catches the light lower down
        k = SD - c1
        fall = np.exp(-k / (D['surface_fall'] * s)) * (1 - smoothstep(0.86 * surf, 1.28 * surf, k))
        low = smoothstep(yt[0] - 20 * s, yt[2] + 10 * s, Y)
        sf = (np.array(D['surface_lit'], np.float32)[None, None, :] * (0.55 + 0.45 * low)[..., None]) if lit \
            else np.broadcast_to(np.array(D['surface_dark'], np.float32), X.shape + (3,)).copy()
        sf = sf + (np.array((0.85, 0.08, 0.1), np.float32) - sf) * (reserve * 0.6 * np.exp(-np.maximum(k, 0) / (10 * s)))[..., None]
        put(sf, (k > 0) * (k < 1.35 * surf) * fall * (0.95 if lit else 0.7) * in_soft)
        # the tube's thin bright inner line
        put((230, 232, 237), np.clip(0.7 * s - np.abs(SD - c1), 0, 1) * in_span)
        # the dividers at the half and at 0
        for yd in (yt[1], yt[2]):
            put((216, 219, 224), 0.8 * np.clip(0.6 * s - np.abs(Y - yd + (SD - 4 * s) * D['slope']), 0, 1)
                * (SD > 4 * s) * (1 - smoothstep(30 * s, 70 * s, SD)) * in_span)
        # the frame: a white line and a fainter one inside it; oil's hot top red, its cold end blue
        fc = np.broadcast_to(rgb01(col['white']), X.shape + (3,)).copy()
        fc = fc + (rgb01(col['red']) - fc) * hot[..., None]
        fc = fc + (rgb01(col['blue']) - fc) * cold[..., None]
        put((220, 222, 228), np.clip(0.8 * s - np.abs(SD - 4.2 * s), 0, 1) * in_span)
        put(fc, np.clip(1.3 * s - np.abs(SD), 0, 1) * in_span)
        # notches: small wedges in from the frame at the eighths of the level's travel
        for j in range(4):
            v = j * 0.25 + 0.125
            yn = np.interp(v, [0, 0.5, 1], [yt[2], yt[1], yt[0]])
            put(col['white'], np.clip(2.6 * s - 0.45 * (SD - 4 * s) - np.abs(Y - yn), 0, 1)
                * (SD > 4 * s) * (SD < 8.5 * s) * in_span)
        return rgb, a

    lit = build(True)
    unlit = build(False)
    return lit, unlit, Y, SD, in_soft, (x0, y0), right


def draw_icons(canvas, s):
    """The small line icons: coolant, boost, tyre, fuel pump, oil can, range pump."""
    col = CONFIG['colors']
    d = ImageDraw.Draw(canvas)
    lw = max(1, int(round(2 * s)))

    def thermo(c):
        x, y = px(c, s)
        d.line([(x, y - 14 * s), (x, y + 6 * s)], fill=col['white'], width=lw)
        for i in range(3):
            d.line([(x, y - 10 * s + i * 5 * s), (x + 6 * s, y - 10 * s + i * 5 * s)], fill=col['white'], width=lw)
        for k in (-1, 1):
            yy = y + (10 + 4 * (k + 1)) * s
            d.arc([x - 14 * s, yy - 4 * s, x + 14 * s, yy + 4 * s], 0, 180, fill=col['white'], width=lw)

    def turbo(c):
        x, y = px(c, s)
        r = 13 * s
        d.ellipse([x - r, y - r, x + r, y + r], outline=col['white'], width=lw)
        for k in range(6):
            a0 = k * math.pi / 3
            d.arc([x - r * 0.75, y - r * 0.75, x + r * 0.75, y + r * 0.75], math.degrees(a0), math.degrees(a0) + 40,
                  fill=col['white'], width=lw)
        d.line([(x + r, y), (x + r + 8 * s, y)], fill=col['white'], width=lw)

    def tyre(c):
        x, y = px(c, s)
        r = 12 * s
        d.ellipse([x - r, y - r, x + r, y + r], outline=col['white'], width=lw)
        for k in range(5):
            a0 = k * 2 * math.pi / 5
            d.ellipse([x + 6 * s * math.cos(a0) - 2 * s, y + 6 * s * math.sin(a0) - 2 * s,
                       x + 6 * s * math.cos(a0) + 2 * s, y + 6 * s * math.sin(a0) + 2 * s], fill=col['white'])

    def pump(c, scale=1.0):
        x, y = px(c, s)
        k = s * scale
        d.rounded_rectangle([x - 8 * k, y - 12 * k, x + 4 * k, y + 12 * k], radius=2 * k, outline=col['white'], width=lw)
        d.rectangle([x - 5 * k, y - 9 * k, x + 1 * k, y - 3 * k], fill=col['white'])
        d.line([(x + 4 * k, y - 4 * k), (x + 9 * k, y), (x + 9 * k, y + 8 * k)], fill=col['white'], width=lw)
        d.polygon([(x + 14 * k, y - 3 * k), (x + 20 * k, y), (x + 14 * k, y + 3 * k)], fill=col['white'])

    def oilcan(c):
        x, y = px(c, s)
        d.polygon([(x - 16 * s, y - 2 * s), (x + 6 * s, y - 2 * s), (x + 16 * s, y - 8 * s), (x + 8 * s, y + 6 * s),
                   (x - 16 * s, y + 6 * s)], outline=col['white'])
        d.line([(x - 4 * s, y - 2 * s), (x - 4 * s, y - 8 * s)], fill=col['white'], width=lw)
        d.line([(x + 18 * s, y + 2 * s), (x + 18 * s, y + 6 * s)], fill=col['white'], width=lw)

    thermo(CONFIG['bars']['coolant']['icon'])
    turbo(CONFIG['bars']['boost']['icon'])
    tyre(CONFIG['tyres']['icon'])
    pump(CONFIG['dials']['fuel']['icon'], 1.1)
    pump(CONFIG['strip']['range_icon'], 1.25)
    oilcan(CONFIG['dials']['oil']['icon'])


def draw_center_static(canvas, s):
    C, col = CONFIG['center'], CONFIG['colors']
    d = ImageDraw.Draw(canvas)
    y = C['line_y'] * HEIGHT * s
    x0, x1 = C['line_x'][0] * WIDTH * s, C['line_x'][1] * WIDTH * s
    cap = C['line_cap_w'] * WIDTH * s
    d.line([(x0 + cap, y), (x1 - cap, y)], fill=(170, 175, 184), width=max(1, int(2 * s)))
    for a, b in ((x0, x0 + cap), (x1 - cap, x1)):          # the bright end caps, tapering towards the middle
        steps = 12
        for i in range(steps):
            u0, u1 = i / steps, (i + 1) / steps
            xa, xb = a + (b - a) * u0, a + (b - a) * u1
            outer = (u0 if a > x0 else 1 - u0)
            wdt = max(1, int(round((2 + 3 * outer) * s)))
            d.line([(xa, y), (xb, y)], fill=col['white'], width=wdt)
    # the floor: rays down from each blade's foot and a line on towards the middle, faint
    F = CONFIG['floor']
    for right in (False, True):
        mx = (lambda p: (mirror_x(p[0]), p[1])) if right else (lambda p: p)
        for a, b in F['rays']:
            d.line([px(mx(a), s), px(mx(b), s)], fill=(70, 74, 80), width=max(1, int(s)))
        a, b = F['line']
        d.line([px(mx(a), s), px(mx(b), s)], fill=(150, 154, 160), width=max(1, int(1.5 * s)))


def draw_static_text(canvas, s):
    col = CONFIG['colors']
    B = CONFIG['blade']
    for r, x in B['labels_x'].items():
        y = rpm_to_y(r * 1000) / HEIGHT
        draw_text_static(canvas, str(r), 'rpm', col['text'], (x, y), s=s)
        draw_text_static(canvas, str(r), 'rpm', col['text'], (mirror_x(x) - 0.0004, y), s=s)
    for b in CONFIG['bars'].values():
        for v, x, y in b['labels']:
            draw_text_static(canvas, str(v), 'scale', col['text'], (x, y), s=s)
        draw_text_static(canvas, b['unit_text'], 'unit', col['text'], b['unit'], s=s)
    T = CONFIG['tyres']
    draw_text_static(canvas, 'psi', 'unit', col['text'], T['unit_p'], 'left', s=s)
    draw_text_static(canvas, '°C', 'unit', col['text'], T['unit_t'], s=s)
    for key in ('fuel', 'oil'):
        for txt, x, y in CONFIG['dials'][key]['labels']:
            draw_text_static(canvas, txt, 'scale', col['text'], (x, y), s=s)
    draw_text_static(canvas, '°C', 'unit', col['text'], CONFIG['dials']['oil']['unit'], 'left', s=s)
    draw_text_static(canvas, 'km/h', 'unit', col['text'], CONFIG['center']['speed_unit'], 'left', s=s)
    draw_text_static(canvas, '1/min x1000', 'unit', col['dim'], CONFIG['center']['rpm_unit'], s=s)
    St = CONFIG['strip']
    draw_text_static(canvas, 'km', 'unit', col['text'], St['range_unit'], 'left', s=s)
    draw_text_static(canvas, '°C', 'unit', col['text'], St['ambient_unit'], 'left', s=s)


def draw_tyre_glyph(canvas, s):
    T, col = CONFIG['tyres'], CONFIG['colors']
    d = ImageDraw.Draw(canvas)
    cx, cy = px(T['car'], s)
    lw = max(1, int(round(1.6 * s)))
    for xa, xb in T['cross_h']:
        d.line([(xa * WIDTH * s, cy), (xb * WIDTH * s, cy)], fill=col['white'], width=lw)
    for ya, yb in T['cross_v']:
        d.line([(cx, ya * HEIGHT * s), (cx, yb * HEIGHT * s)], fill=col['white'], width=lw)
    d.rounded_rectangle([cx - 7 * s, cy - 15 * s, cx + 7 * s, cy + 15 * s], radius=5 * s, outline=col['green'], width=lw)
    for sx in (-1, 1):
        for sy in (-1, 1):
            d.rectangle([cx + sx * 11 * s - 1.6 * s, cy + sy * 9 * s - 4 * s, cx + sx * 11 * s + 1.6 * s, cy + sy * 9 * s + 4 * s],
                        fill=col['green'])


def draw_bars_static(canvas, s):
    col = CONFIG['colors']
    d = ImageDraw.Draw(canvas)
    for b in CONFIG['bars'].values():
        x0, x1 = b['x'][0] * WIDTH * s, b['x'][1] * WIDTH * s
        y, h = b['y'] * HEIGHT * s, b['h'] * HEIGHT * s
        d.rectangle([x0, y - h / 2, x1, y + h / 2], fill=col['track'])
        if b['red_from'] is not None:
            xr = x0 + (x1 - x0) * (b['red_from'] - b['min']) / (b['max'] - b['min'])
            d.rectangle([xr, y - h / 2, x1, y + h / 2], fill=col['red'])
        for v, _, _ in b['labels']:                      # small marks above the bar at the numbered values
            xm = x0 + (x1 - x0) * (v - b['min']) / (b['max'] - b['min'])
            d.line([(xm, y - h / 2 - 5 * s), (xm, y - h / 2)], fill=col['white'], width=max(1, int(1.5 * s)))


def draw_debug_grid(canvas, s):
    """The 0.1 grid with its labels, the axis of symmetry and the main geometric references."""
    d = ImageDraw.Draw(canvas)
    grid = (0, 255, 120)
    for i in range(11):
        x, y = i / 10 * WIDTH * s, i / 10 * HEIGHT * s
        d.line([(x, 0), (x, HEIGHT * s)], fill=grid + (110,), width=max(1, int(s)))
        d.line([(0, y), (WIDTH * s, y)], fill=grid + (110,), width=max(1, int(s)))
        draw_text_static(canvas, '%.1f' % (i / 10), 'grid', (255, 255, 0), (min(i / 10 + 0.004, 0.985), 0.012), 'left', s)
        draw_text_static(canvas, '%.1f' % (i / 10), 'grid', (255, 255, 0), (0.004, min(i / 10 + 0.012, 0.985)), 'left', s)
    d.line([(WIDTH * s / 2, 0), (WIDTH * s / 2, HEIGHT * s)], fill=(255, 0, 255, 200), width=max(1, int(s)))
    ref = (255, 220, 0, 230)

    def cross(p, r=6):
        x, y = px(p, s)
        d.line([(x - r * s, y), (x + r * s, y)], fill=ref, width=max(1, int(s)))
        d.line([(x, y - r * s), (x, y + r * s)], fill=ref, width=max(1, int(s)))

    B = CONFIG['blade']
    for right in (False, True):
        bx, by = B['bend']
        cross((mirror_x(bx) if right else bx, by), 10)
        ys = np.linspace(B['y_top'] * HEIGHT * s, B['y_bot'] * HEIGHT * s, 60)
        xs = blade_outer_x(ys, s)
        if right:
            xs = WIDTH * s - xs
        d.line(list(zip(xs, ys)), fill=(0, 200, 255, 220), width=max(1, int(s)))
        d.line(card_polygon(s, right), fill=(255, 140, 0, 200), width=max(1, int(s)))
    for r in range(9):
        y = rpm_to_y(r * 1000, s)
        d.line([(0.29 * WIDTH * s, y), (0.37 * WIDTH * s, y)], fill=(255, 80, 80, 160), width=max(1, int(s)))
    for key in ('fuel', 'oil'):
        d.line([px(p, s) for p in CONFIG['dials'][key]['frame']], fill=(0, 200, 255, 220), width=max(1, int(s)))
        for y in CONFIG['dials'][key]['levels_y']:
            xs = [p[0] for p in CONFIG['dials'][key]['frame']]
            d.line([(min(xs) * WIDTH * s, y * HEIGHT * s), ((max(xs) + 0.04) * WIDTH * s, y * HEIGHT * s)],
                   fill=(255, 80, 80, 160), width=max(1, int(s)))
    C = CONFIG['center']
    for p in (C['gear'], C['speed'], C['mode_icon'], CONFIG['tyres']['car'], CONFIG['limit_sign']['c']):
        cross(p)
    for b in CONFIG['bars'].values():
        d.rectangle([b['x'][0] * WIDTH * s, (b['y'] - b['h']) * HEIGHT * s, b['x'][1] * WIDTH * s,
                     (b['y'] + b['h']) * HEIGHT * s], outline=(255, 140, 0, 220), width=max(1, int(s)))


def build_static_layers(width=WIDTH, height=HEIGHT, debug_grid=DEBUG_GRID, reference=None):
    """Everything that doesn't move, drawn once at SUPERSAMPLE x and scaled down. Returns the cache the frames use."""
    assert (width, height) == (WIDTH, HEIGHT), 'CONFIG is normalized; set WIDTH / HEIGHT at the top to change size'
    s = SUPERSAMPLE
    col = CONFIG['colors']
    t0 = time.perf_counter()
    canvas = Image.new('RGBA', (width * s, height * s), col['bg'] + (255,))
    draw_cards(canvas, s)
    # the glow under the blades' feet, blurred once
    glow = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for right in (False, True):
        cx = (mirror_x(0.335) if right else 0.335) * width * s
        gd.ellipse([cx - 70 * s, 0.82 * height * s, cx + 70 * s, 0.93 * height * s], fill=(120, 124, 132, 110))
    canvas.alpha_composite(glow.filter(ImageFilter.GaussianBlur(18 * s)))
    draw_center_static(canvas, s)
    # the blades (left as computed, right its mirror)
    brgb, ba, bmask, (bx0, by0) = blade_arrays(s)
    for right in (False, True):
        r, a = (brgb[:, ::-1], ba[:, ::-1]) if right else (brgb, ba)
        x = (width * s - bx0 - a.shape[1]) if right else bx0
        img = Image.fromarray(np.dstack([np.clip(r * 255, 0, 255), np.clip(a * 255, 0, 255)]).astype(np.uint8), 'RGBA')
        canvas.alpha_composite(img, (int(x), int(by0)))
    draw_bars_static(canvas, s)
    draw_tyre_glyph(canvas, s)
    draw_icons(canvas, s)
    dials = {}
    for key in ('fuel', 'oil'):
        lit, unlit, Y, SD, soft, (dx0, dy0), right = dial_arrays(key, s)
        dials[key] = (lit, unlit, Y, SD, soft, (dx0, dy0), right)
    draw_static_text(canvas, s)
    if debug_grid:
        draw_debug_grid(canvas, s)
    static = canvas.resize((width, height), Image.LANCZOS).convert('RGB')
    if reference:
        ref = Image.open(reference).convert('RGB').resize((width, height))
        static = Image.blend(static, ref, 0.5)
    frame0 = np.asarray(static, np.uint8).copy()

    # the moving parts' precomputed patches at 1x: the blades' fill mask, the dials' lit / unlit halves
    def down(arr):
        """Average s x s blocks (the supersampled arrays down to 1x)."""
        h, w = arr.shape[:2]
        h2, w2 = h // s, w // s
        arr = arr[:h2 * s, :w2 * s]
        return arr.reshape(h2, s, w2, s, *arr.shape[2:]).mean(axis=(1, 3)).astype(np.float32)

    cache = {'frame': frame0, 'build_ms': 0.0}
    mask1 = down(bmask)
    yrows = (by0 / s + np.arange(mask1.shape[0]) + 0.5).astype(np.float32)
    cache['blade'] = {'mask': mask1, 'mask_r': mask1[:, ::-1].copy(), 'y': yrows,
                      'x': int(round(bx0 / s)), 'x_r': int(round((width * s - bx0) / s)) - mask1.shape[1],
                      'y0': int(round(by0 / s))}
    cache['dials'] = {}
    for key, (lit, unlit, Y, SD, soft, (dx0, dy0), right) in dials.items():
        L = (down(lit[0]), down(lit[1]))
        U = (down(unlit[0]), down(unlit[1]))
        Yd, SDd, So = down(Y) / s, down(SD) / s, down(soft)
        if right:                                   # back from the left frame to the screen
            L = (L[0][:, ::-1].copy(), L[1][:, ::-1].copy())
            U = (U[0][:, ::-1].copy(), U[1][:, ::-1].copy())
            Yd, SDd, So = Yd[:, ::-1].copy(), SDd[:, ::-1].copy(), So[:, ::-1].copy()
            x = width - int(round(dx0 / s)) - L[1].shape[1]
        else:
            x = int(round(dx0 / s))
        cache['dials'][key] = {'lit': L, 'unlit': U, 'Y': Yd, 'SD': SDd, 'soft': So, 'x': x, 'y': int(round(dy0 / s)),
                               'levels': [v * height for v in CONFIG['dials'][key]['levels_y']]}
    cache['build_ms'] = (time.perf_counter() - t0) * 1000
    return cache


# ---------------------------------------------------------------------------------------------------------------------
# the moving parts, every frame
# ---------------------------------------------------------------------------------------------------------------------
RED = rgb01(CONFIG['colors']['red'])
RED_HOT = rgb01(CONFIG['colors']['red_hot'])
WHITE = rgb01(CONFIG['colors']['white'])


def _blade_fill(frame, bl, rpm):
    B = CONFIG['blade']
    yr = float(rpm_to_y(rpm))
    y = bl['y']
    below = y >= yr
    gain = np.where(below, 0.45 + 0.55 * np.exp(-(y - yr) / (B['lit_fade_h'] * HEIGHT)),
                    0.6 * np.exp(-(yr - y) / (B['glow_up_h'] * HEIGHT))).astype(np.float32)
    marker = np.clip(3.0 - np.abs(y - yr), 0, 1).astype(np.float32)
    rows = slice(max(0, int(yr - B['glow_up_h'] * HEIGHT * 4) - bl['y0']), None)
    for m, x in ((bl['mask'], bl['x']), (bl['mask_r'], bl['x_r'])):
        mm = m[rows]
        a = mm * gain[rows, None]
        color = RED[None, None, :] + (RED_HOT - RED)[None, None, :] * marker[rows, None, None]
        blend(frame, np.broadcast_to(color, mm.shape + (3,)), a, x, bl['y0'] + rows.start)
        blend(frame, np.broadcast_to(np.array((1.0, 0.78, 0.6), np.float32), mm.shape + (3,)), mm * marker[rows, None] * 0.9,
              x, bl['y0'] + rows.start)


def _dial(frame, d, v):
    D = CONFIG['dials']
    t = d['levels']
    v = float(np.clip(v, 0, 1))
    yl = np.interp(v, [0, 0.5, 1], [t[2], t[1], t[0]])
    lit = d['Y'] > yl
    rgb = np.where(lit[..., None], d['lit'][0], d['unlit'][0])
    a = np.where(lit, d['lit'][1], d['unlit'][1])
    blend(frame, rgb, a, d['x'], d['y'])
    # the needle: a red bar across at the level, rising inwards, fading out towards the numbers
    SD = d['SD']
    nd = np.clip(2.6 - np.abs(d['Y'] - yl + (SD - 4) * D['slope']), 0, 1) * (SD > 3.5) \
        * (1 - smoothstep(22, 66, SD)) * d['soft']
    blend(frame, np.broadcast_to(np.array((0.97, 0.12, 0.15), np.float32), nd.shape + (3,)), nd.astype(np.float32), d['x'], d['y'])


def _bar(frame, b, value, alarm):
    x0, x1 = int(b['x'][0] * WIDTH), int(b['x'][1] * WIDTH)
    y, h = b['y'] * HEIGHT, b['h'] * HEIGHT
    f = float(np.clip((value - b['min']) / (b['max'] - b['min']), 0, 1))
    xe = int(round(x0 + (x1 - x0) * f))
    ya, yb = int(round(y - h / 2)), int(round(y + h / 2))
    if xe > x0:
        frame[ya:yb, x0:xe] = (np.array(CONFIG['colors']['red'] if alarm else CONFIG['colors']['white'])).astype(np.uint8)
    frame[ya - 3:yb + 3, max(x0, xe - 2):xe + 1] = 255           # the value's marker, a little taller than the bar


def _limit_sign(frame, value):
    L = CONFIG['limit_sign']
    key = ('sign', value)
    spr = TEXT.cache.get(key)
    if spr is None:
        r = L['r_h'] * HEIGHT
        k = 4
        img = Image.new('RGBA', (int(2 * r * k) + 4, int(2 * r * k) + 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        c = img.width / 2
        d.ellipse([c - r * k, c - r * k, c + r * k, c + r * k], fill=(232, 34, 38, 255))
        d.ellipse([c - r * k * 0.78, c - r * k * 0.78, c + r * k * 0.78, c + r * k * 0.78], fill=(250, 250, 250, 255))
        img = img.resize((img.width // k, img.height // k), Image.LANCZOS)
        num = TEXT.image(str(value), 'sign', (40, 40, 44))
        img.alpha_composite(num, (int(img.width / 2 - num.width / 2), int(img.height / 2 - num.height / 2)))
        rgb, a = rgba_image_to_arrays(img)
        x, y = px(L['c'])
        spr = TEXT.cache[key] = Sprite(rgb, a, x - img.width / 2, y - img.height / 2)
    blend(frame, spr.rgb, spr.a, spr.x, spr.y)


def _text(frame, txt, font, color, anchor, align='center'):
    spr = TEXT.sprite(txt, font, color, anchor, align)
    blend(frame, spr.rgb, spr.a, spr.x, spr.y)


def _mode_icon(frame, mode):
    """BMW's drive-mode stack beside the gear: 4 slanted bars (Sport), 3 (Comfort), 2 (Eco Pro)."""
    key = ('mode', mode)
    spr = TEXT.cache.get(key)
    if spr is None:
        n = {'sport': 4, 'comfort': 3, 'eco': 2}.get(mode, 3)
        w, h = (int(CONFIG['center']['mode_icon_size'][0] * WIDTH), int(CONFIG['center']['mode_icon_size'][1] * HEIGHT))
        k = 4
        img = Image.new('RGBA', (w * k, h * k), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        bh, gap = h * k / 5.5, h * k / 14
        for i in range(4):
            y = h * k - (i + 1) * (bh + gap)
            on = i < n
            d.polygon([(0, y + bh), (w * k * 0.82, y + bh), (w * k, y), (w * k * 0.18, y)],
                      fill=(240, 242, 246, 255) if on else (90, 94, 100, 255))
        img = img.resize((w, h), Image.LANCZOS)
        rgb, a = rgba_image_to_arrays(img)
        x, y = px(CONFIG['center']['mode_icon'])
        spr = TEXT.cache[key] = Sprite(rgb, a, x - w / 2, y - h / 2)
    blend(frame, spr.rgb, spr.a, spr.x, spr.y)


def update_dynamic_layer(static_cache, telemetry_dict, t_now=0.0):
    """One frame: the static layer plus everything that moves. telemetry_dict holds real values (see demo_telemetry);
    a missing key shows as '--' or leaves its part unlit. Returns an HxWx3 uint8 array."""
    T = telemetry_dict
    col = CONFIG['colors']
    frame = static_cache['frame'].copy()
    A = CONFIG['alarms']
    blink = (int(t_now * A['blink_hz'] * 2) % 2) == 0

    rpm = T.get('rpm')
    if rpm is not None:
        _blade_fill(frame, static_cache['blade'], min(float(rpm), CONFIG['blade']['rpm_max']))
    fuel = T.get('fuel')
    if fuel is not None:
        _dial(frame, static_cache['dials']['fuel'], fuel)
    oil = T.get('oil_c')
    if oil is not None:
        o = CONFIG['dials']['oil']
        _dial(frame, static_cache['dials']['oil'], (oil - o['min']) / (o['max'] - o['min']))
    cool = T.get('coolant_c')
    if cool is not None:
        _bar(frame, CONFIG['bars']['coolant'], cool, cool >= A['coolant_c'] and blink)
    boost = T.get('boost_psi')
    if boost is not None:
        _bar(frame, CONFIG['bars']['boost'], boost, False)

    C = CONFIG['center']
    _text(frame, str(T.get('gear', '-')), 'gear', col['text'], C['gear'])
    _mode_icon(frame, T.get('drive_mode', 'comfort'))
    spd = T.get('speed_kmh')
    _text(frame, '--' if spd is None else str(int(round(spd))), 'speed', col['text'], C['speed'])
    tyres = T.get('tyres') or {}
    for corner, (x, yp, yt) in CONFIG['tyres']['cells'].items():
        p, tc = tyres.get(corner, (None, None))
        _text(frame, '--' if p is None else '%d' % round(p), 'value', col['text'], (x, yp))
        _text(frame, '--' if tc is None else '%d' % round(tc), 'value', col['text'], (x, yt))
    St = CONFIG['strip']
    rng = T.get('range_km')
    low_fuel = fuel is not None and fuel < A['fuel']
    _text(frame, '--' if rng is None else '%d' % rng, 'strip', (255, 176, 40) if low_fuel else col['text'], St['range'], 'right')
    _text(frame, T.get('clock', '--:--'), 'strip', col['text'], St['time'], 'right')
    _text(frame, T.get('ampm', ''), 'unit', col['text'], St['ampm'], 'left')
    amb = T.get('ambient_c')
    _text(frame, '--' if amb is None else '%+.1f' % amb, 'strip', col['text'], St['ambient'], 'right')
    if T.get('speed_limit'):
        _limit_sign(frame, int(T['speed_limit']))
    if oil is not None and oil >= A['oil_c'] and blink:                # oil too hot: its numbers flash red
        _text(frame, '150', 'scale', col['red'], CONFIG['dials']['oil']['labels'][0][1:])
    return frame


# ---------------------------------------------------------------------------------------------------------------------
# demo telemetry and the 30 fps loop
# ---------------------------------------------------------------------------------------------------------------------
def demo_telemetry(t):
    """A drive: revs climbing through the gears, speed, slow temperatures, a little boost."""
    cyc = (t % 8.0) / 8.0
    gear = 1 + int(cyc * 4)
    rpm = 900 + (cyc * 4 % 1.0) ** 0.8 * 5600
    speed = 12 + cyc * 120
    return {'rpm': rpm, 'speed_kmh': speed, 'gear': gear, 'drive_mode': 'sport',
            'fuel': 0.78 - 0.02 * math.sin(t * 0.1), 'oil_c': 92 + 6 * math.sin(t * 0.05),
            'coolant_c': 88 + 3 * math.sin(t * 0.07), 'boost_psi': max(0.0, (rpm - 1500) / 4000 * 14),
            'tyres': {'FL': (36, 28), 'FR': (36, 28), 'RL': (36, 25), 'RR': (36, 25)},
            'range_km': 448, 'clock': '3:45', 'ampm': 'pm', 'ambient_c': 21.0, 'speed_limit': 10}


def run_loop_30fps(seconds=10.0, frames=None, window=False, save=None, telemetry=demo_telemetry,
                   debug_grid=DEBUG_GRID, reference=None):
    cache = build_static_layers(WIDTH, HEIGHT, debug_grid, reference)
    print('static layers: %.0f ms (once)' % cache['build_ms'])
    screen = None
    if window:
        try:
            import pygame
            pygame.init()
            screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption('Corsa - 30 fps')
        except ImportError:
            print('pygame is not installed: running without a window')
    period = 1.0 / TARGET_FPS
    t_start = last = time.perf_counter()
    times, n, frame = [], 0, None
    next_report = t_start + 1.0
    while True:
        now = time.perf_counter()
        dt = now - last                     # the frame's delta time, for anything that animates by time
        last = now
        t = now - t_start
        f0 = time.perf_counter()
        frame = update_dynamic_layer(cache, telemetry(t), t)
        if screen is not None:
            import pygame
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    frames = n
            pygame.surfarray.blit_array(screen, frame.swapaxes(0, 1))
            pygame.display.flip()
        work = time.perf_counter() - f0
        times.append(work * 1000)
        n += 1
        if now >= next_report:
            print('t %5.1f s  dt %5.1f ms  frame %5.2f ms avg, %5.2f ms max (budget %.1f ms)'
                  % (t, dt * 1000, sum(times[-30:]) / len(times[-30:]), max(times[-30:]), period * 1000))
            next_report += 1.0
        if (frames is not None and n >= frames) or (frames is None and t >= seconds):
            break
        sleep = period - (time.perf_counter() - now)
        if sleep > 0:
            time.sleep(sleep)
    avg = sum(times) / len(times)
    print('%d frames: %.2f ms per frame on average, %.2f ms worst, %s the 33.3 ms budget'
          % (n, avg, max(times), 'within' if max(times) < period * 1000 else 'OVER'))
    if save and frame is not None:
        Image.fromarray(frame).save(save)
        print('last frame -> ' + save)
    return avg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seconds', type=float, default=10.0)
    ap.add_argument('--frames', type=int)
    ap.add_argument('--window', action='store_true', help='show it (pygame)')
    ap.add_argument('--save', default='corsa.png', help='the last frame as PNG ("" = none)')
    ap.add_argument('--grid', action='store_true', help='DEBUG_GRID on')
    ap.add_argument('--ref', help='blend this straightened photo at 50 %% into the static layer')
    a = ap.parse_args()
    run_loop_30fps(a.seconds, a.frames, a.window, a.save or None, debug_grid=a.grid or DEBUG_GRID, reference=a.ref)


if __name__ == '__main__':
    main()
