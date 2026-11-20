#!/usr/bin/env python3
"""sim_anim.py - an animated preview of a layout's live layer: a pull through the gears, rendered at 30 fps.

    python3 cluster/tools/sim_anim.py id8/A [--engine diesel] [--seconds 12]   # cluster/design/out/<cat>_<name>_anim.mp4

A design tool (Python, PC only): the static layer is drawn once, each frame draws the live layer over a copy of it
(gen_static.draw_dynamic) and the result is fitted to the car's visible area. Needs ffmpeg.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_static as g  # noqa: E402

ROOT = os.path.dirname(HERE)
FPS = 30
# km/h per 1000 rpm in each gear (an M4-like box), and the rpm to shift at in the scenario per engine
RATIO = [9.0, 15.0, 22.0, 29.5, 37.0, 44.0]
SHIFT = {'diesel': 4720, 'petrol': 7100}
IDLE = {'diesel': 800, 'petrol': 900}


def scenario(t, eng):
    """The telemetry at time t: idle, then a pull through the gears in M (each gear held a moment past the last red
    so the flash shows), the rpm dropping on each upshift."""
    idle, top = IDLE[eng], SHIFT[eng]
    if t < 1.0:
        return {'speed': 0.0, 'rpm': idle, 'gear': 'M1'}
    t -= 1.0
    rpm_climb = [1.2, 1.6, 2.1, 2.7, 3.4, 4.2]             # seconds to climb from the drop to the shift rpm, per gear
    start = 1800 if eng == 'diesel' else 2600
    for i, dur in enumerate(rpm_climb):
        lo = start if i else idle + 400
        if t < dur:
            f = t / dur
            rpm = lo + (top - lo) * (1 - (1 - f) ** 1.6)
            return {'speed': rpm / 1000 * RATIO[i], 'rpm': rpm, 'gear': 'M%d' % (i + 1)}
        t -= dur
        if t < 0.15:                                         # the shift itself: the rpm falls to the next gear's
            nxt = top * RATIO[i] / RATIO[min(i + 1, len(RATIO) - 1)]
            rpm = top + (nxt - top) * (t / 0.15)
            return {'speed': top / 1000 * RATIO[i], 'rpm': rpm, 'gear': 'M%d' % (i + 2 if i + 1 < len(RATIO) else i + 1)}
        t -= 0.15
        start = top * RATIO[i] / RATIO[min(i + 1, len(RATIO) - 1)]
    return {'speed': top / 1000 * RATIO[-1], 'rpm': top, 'gear': 'M%d' % len(RATIO)}


_STATE = {}


def _init(cfg_path, eng):
    cfg = json.load(open(cfg_path, encoding='utf-8'))
    cfg['engine'] = eng
    cfg['elements']['speed']['sample'] = ''
    cfg['elements']['gear']['sample'] = ''
    static = g.build_static(cfg)
    g.fit_view(static, cfg)                                   # finds the scale once
    _STATE.update(cfg=cfg, static=static, fit=cfg['view']['_fit'])


def _frame(args):
    i, out_dir, eng = args
    cfg, static, fit = _STATE['cfg'], _STATE['static'], _STATE['fit']
    tel = scenario(i / FPS, eng)
    tel.update(fuel=0.5, temp_c=90, oil_c=95, frame=i)
    img = g.draw_dynamic(static.copy(), cfg, tel)
    W, H = cfg['grid']
    img = g._scale(img, float(fit['scale']), float(fit['cx']), float(fit['cy']), W, H)
    img.convert('RGB').save(os.path.join(out_dir, '%05d.png' % i))
    return i


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('layout')
    ap.add_argument('--engine', default='diesel')
    ap.add_argument('--seconds', type=float, default=16.5)
    a = ap.parse_args()
    cat, name = a.layout.split('/')
    cfg_path = os.path.join(ROOT, 'layouts', cat, name, 'config.json')
    tmp = tempfile.mkdtemp(prefix='anim_')
    n = int(a.seconds * FPS)
    with Pool(os.cpu_count(), initializer=_init, initargs=(cfg_path, a.engine)) as pool:
        for k, _ in enumerate(pool.imap_unordered(_frame, [(i, tmp, a.engine) for i in range(n)], chunksize=4)):
            if k % 60 == 0:
                print('%d / %d frames' % (k, n), flush=True)
    out = os.path.join(ROOT, 'design', 'out', '%s_%s_anim.mp4' % (cat, name))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', os.path.join(tmp, '%05d.png'),
                           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', out])
    shutil.rmtree(tmp)
    print('anim: ' + os.path.relpath(out, ROOT))


if __name__ == '__main__':
    main()
