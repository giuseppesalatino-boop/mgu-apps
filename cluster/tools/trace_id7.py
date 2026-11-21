#!/usr/bin/env python3
"""trace_id7.py - automatic contour tracing on the 4x enlargement of the BMW M reference (ID7 Corsa / M View).

    python3 cluster/tools/trace_id7.py revs id7/Corsa     # the rev counter blades (4x)
    python3 cluster/tools/trace_id7.py center id7/Corsa   # gear, speed, km/h, the line (4x)
    python3 cluster/tools/trace_id7.py fuel10 id7/Strada id7/Corsa   # the fuel gauge rebuilt at 10x
    python3 cluster/tools/trace_id7.py temp10 id7/Corsa   # the temperature gauge at 10x (run tempmirror10 952.5 first)
    python3 cluster/tools/trace_id7.py tempzones10 / reserve10 / labels10 ...   # hot-cold marks, reserve, labels

Reads cluster/design/refs/id7/M8_mview_x4.png (photo * 0.3104 * 4, not in git). Every coordinate is measured on the
4x grid with sub-pixel half-maximum crossings / weighted centroids, returned in display px and normalized on 1920x720.
A design tool (Python, PC only).
"""
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
K = 4
W, H = 1920, 720


def load(mirror_axis=None):
    a = np.asarray(Image.open(os.path.join(ROOT, 'design', 'refs', 'id7', 'M8_mview_x4.png')).convert('RGB')).astype(np.float32)
    if mirror_axis is not None:                               # the right blade seen as a left one: x' = 2*axis - x
        a = a[:, ::-1]
        sh = int(round((2 * mirror_axis - W) * K))            # flipped image's x' = W*K - 1 - x  ->  shift to 2*axis - x
        a = np.roll(a, sh, axis=1)
    return a


def crossing(row, i0, i1, lvl):
    """Sub-pixel position between samples i0 and i1 where row crosses lvl."""
    a, b = row[i0], row[i1]
    if a == b:
        return float(i0)
    return i0 + (lvl - a) / (b - a) * (i1 - i0)


def centroid(row, lo, hi, base):
    lo, hi = max(0, lo), min(len(row), hi)
    seg = np.clip(row[lo:hi] - base, 0, None)
    if seg.sum() <= 0:
        return None
    return float((np.arange(lo, hi) * seg).sum() / seg.sum())


def trace_blade(img, y0, y1, x_start, y_seed):
    """Per display row: O (outer white line: left/right half-max edges), E (the lit edge inside the gap), I (the thin
    inner line), and a flag for the tick rows (the white run goes on inward)."""
    M = img.min(2)                                            # white lines only (red fill / zones drop out)
    Lm = img.mean(2)
    rows = {}
    order = list(range(y_seed, y1)) + list(range(y_seed - 1, y0 - 1, -1))
    for yd in order:
        r = M[yd * K:(yd + 1) * K].mean(0)
        rl = Lm[yd * K:(yd + 1) * K].mean(0)
        xs = int(x_start * K)
        hit = np.where(r[xs:] > 128)[0]
        if not len(hit):
            continue
        i = xs + hit[0]
        ol = float(crossing(r, i - 1, i, 128)) / K
        j = i
        while j < len(r) - 1 and r[j + 1] > 128:
            j += 1
        orr = float(crossing(r, j, j + 1, 128)) / K
        rec = {'O_l': ol, 'O_r': orr, 'tick': bool((orr - ol) > 6.5)}
        # E: the lit edge, first peak after O within 10 px
        if not rec['tick']:
            lo2, hi2 = int((orr + 1) * K), int((orr + 12) * K)
            s2 = rl[lo2:hi2]
            if len(s2) and s2.max() > 50:
                pk = lo2 + int(np.argmax(s2))
                c = centroid(rl, pk - 4, pk + 5, s2.max() * 0.5)
                rec['E'] = c / K if c is not None else None
                rec['E_peak'] = float(s2.max())
        # I: the thin inner line = the last peak before the black outside the blade (< 8 for 3 px)
        st = int(((rec.get('E') or orr + 8) + 3) * K)
        blk = rl[st:st + 110 * K] < 8
        run = np.convolve(blk.astype(int), np.ones(3 * K, int), 'valid')
        hitb = np.where(run == 3 * K)[0]
        rec['I'] = None
        if len(hitb):
            e = st + hitb[0]
            lo = e - 7 * K
            pk = lo + int(np.argmax(rl[lo:e]))
            c = centroid(rl, pk - 6, pk + 7, rl[pk] * 0.5)
            if c is not None:
                rec['I'] = c / K
                rec['I_peak'] = float(rl[pk])
        rows[yd] = rec
    return dict(sorted(rows.items()))


def body_profile(img, rows, y_list, n=12):
    """The body's brightness across the band, from just past E to just before I, at n stops (mean RGB over y_list)."""
    acc = []
    for yd in y_list:
        rc = rows.get(yd)
        if not rc or rc['tick'] or not rc.get('E') or not rc['I']:
            continue
        r = img[yd * K:(yd + 1) * K].mean(0)
        xs = np.linspace(rc['E'], rc['I'], n) * K
        acc.append([r[int(round(x))].tolist() for x in xs])
    return np.array(acc).mean(0).round(0).astype(int).tolist() if acc else None


def simplify(pts, tol=0.15):
    """Douglas-Peucker on a polyline (display px)."""
    pts = np.array(pts, float)
    if len(pts) < 3:
        return pts.tolist()
    a, b = pts[0], pts[-1]
    d = b - a
    nd = np.linalg.norm(d) + 1e-9
    dist = np.abs(d[0] * (pts[:, 1] - a[1]) - d[1] * (pts[:, 0] - a[0])) / nd
    k = int(np.argmax(dist))
    if dist[k] > tol:
        return simplify(pts[:k + 1], tol)[:-1] + simplify(pts[k:], tol)
    return [a.tolist(), b.tolist()]


def smooth1(v, n=9):
    k = np.ones(n) / n
    vp = np.pad(v, n // 2, mode='edge')
    return np.convolve(vp, k, 'valid')


def measure_blade(img, x_start, y_top=149.0, y_bot=561.5):
    """All the blade's parts in display px (left-blade orientation): O, ticks, E, I, the body's light, the zones."""
    M = img.min(2)
    Lm = img.mean(2)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    sub = np.arange(int(y_top * K), int(y_bot * K))
    ol, orr = [], []
    for s in sub:
        r = M[s]
        xs = int(x_start * K)
        hit = np.where(r[xs:] > 128)[0]
        if not len(hit):
            ol.append(np.nan); orr.append(np.nan); continue
        i = xs + hit[0]
        j = i
        while j < len(r) - 1 and r[j + 1] > 128:
            j += 1
        ol.append(crossing(r, i - 1, i, 128) / K)
        orr.append(crossing(r, j, j + 1, 128) / K)
    ys = (sub + 0.5) / K
    ol, orr = np.array(ol), np.array(orr)
    w = orr - ol
    wmed = float(np.nanmedian(w))
    # the clean rows: no tick (run too wide) and no slit (left edge pushed in); iterate a smooth expectation
    ok = np.isfinite(w) & (w < wmed + 1.0)
    for _ in range(3):
        exp_l = np.interp(ys, ys[ok], smooth1(ol[ok], 13))
        ok = np.isfinite(w) & (w < wmed + 1.0) & (np.abs(ol - exp_l) < 0.8)
    exp_l = np.interp(ys, ys[ok], smooth1(ol[ok], 13))
    exp_r = np.interp(ys, ys[ok], smooth1(orr[ok], 13))
    bad = ~ok
    # ticks: clusters of bad sub-rows
    ticks = []
    idx = np.where(bad)[0]
    groups = np.split(idx, np.where(np.diff(idx) > 6)[0] + 1) if len(idx) else []
    for gidx in groups:
        if len(gidx) < 6:
            continue
        # the expected edges across the tick: a line through the clean sub-rows just above and below it
        okr = np.where(ok)[0]
        nb = np.concatenate([okr[okr < gidx[0]][-8:], okr[okr > gidx[-1]][:8]])
        fl, fr = np.polyfit(ys[nb], ol[nb], 1), np.polyfit(ys[nb], orr[nb], 1)
        exp_l[gidx], exp_r[gidx] = np.polyval(fl, ys[gidx]), np.polyval(fr, ys[gidx])
        wide = gidx[(orr[gidx] > exp_r[gidx] + 1.0)]
        if not len(wide):
            continue
        tip = float(np.nanmax(orr[wide]))
        ext = tip - float(np.interp(ys[wide].mean(), ys, exp_r))
        if not 4.0 < ext < 45 or ys[wide[0]] < y_top + 3 or ys[wide[-1]] > y_bot - 4:
            continue
        near = wide[orr[wide] > tip - 1.0]
        slit = gidx[ol[gidx] > exp_l[gidx] + 1.0]
        t = {'y0': float(ys[wide[0]]), 'y1': float(ys[wide[-1]]), 'tip': [tip, float(ys[near].mean())],
             'tip_h': float(ys[near[-1]] - ys[near[0]] + 0.25)}
        if len(slit):
            k = slit[int(np.argmax(ol[slit]))]
            t['slit'] = {'y0': float(ys[slit[0]] - 0.25), 'y1': float(ys[slit[-1]] + 0.25), 'apex': [float(ol[k]), float(ys[k])]}
        ticks.append(t)
    # O: the clean centre line, simplified
    cO = [((ol[i] + orr[i]) / 2, ys[i]) for i in np.where(ok)[0][::2]]
    wbins = [[float(y), float(np.median(w[ok & (np.abs(ys - y) < 15)]))] for y in range(int(y_top) + 10, int(y_bot), 30)]
    return {'O': simplify(cO, 0.12), 'O_width': wbins, 'ticks': ticks, 'O_exp': (ys, exp_l, exp_r)}


def poly_distance(poly, X, Y):
    best = np.full(X.shape, 1e9)
    for (ax, ay), (bx, by) in zip(poly[:-1], poly[1:]):
        ex, ey = bx - ax, by - ay
        L2 = ex * ex + ey * ey or 1e-9
        h = np.clip(((X - ax) * ex + (Y - ay) * ey) / L2, 0, 1)
        best = np.minimum(best, np.hypot(X - ax - ex * h, Y - ay - ey * h))
    return best


def x_at(poly, y):
    p = np.array(poly)
    o = np.argsort(p[:, 1])
    return np.interp(y, p[o, 1], p[o, 0])


def blobs(mask, y0, y1, x0, x1, min_rows=2):
    """Shapes of a colour mask as polygons (sub-row left/right edges), display px."""
    out, cur = [], None
    for s in range(int(y0 * K), int(y1 * K)):
        row = mask[s, int(x0 * K):int(x1 * K)]
        idx = np.where(row)[0]
        if len(idx):
            seg = (idx[0] / K + x0, (idx[-1] + 1) / K + x0)
            if cur is not None and cur['last'] == s - 1 and abs(seg[0] - cur['L'][-1][0]) < 3:
                cur['L'].append((seg[0], s / K)); cur['R'].append((seg[1], s / K)); cur['last'] = s
                continue
            if cur is not None:
                out.append(cur)
            cur = {'L': [(seg[0], s / K)], 'R': [(seg[1], s / K)], 'last': s}
    if cur is not None:
        out.append(cur)
    polys = []
    for b in out:
        if len(b['L']) < min_rows * K:
            continue
        L = simplify(b['L'], 0.2); R = simplify(b['R'], 0.2)
        polys.append([[round(x, 2), round(y, 2)] for x, y in L + R[::-1]])
    return polys


def measure_body(img, rows, O):
    """E and I lines, and the light across the body: base by y, the glow by distance from E and from I."""
    Lm = img.mean(2)
    ticks_y = [y for y, r in rows.items() if r['tick']]
    near_tick = lambda y: any(abs(y - t) < 4 for t in ticks_y)
    E = [(r['E'], y + 0.5) for y, r in rows.items() if r.get('E') and 168 <= y <= 440 and not near_tick(y)
         and 5.5 < r['E'] - r['O_r'] < 10]
    Ipts = [(r['I'], y + 0.5) for y, r in rows.items() if r['I'] and 166 <= y <= 548]
    # drop the I points off the line (digits, the block's corner): keep those within 1.5 px of a running median
    Ia = np.array(Ipts)
    med = np.array([np.median(Ia[max(0, i - 6):i + 7, 0]) for i in range(len(Ia))])
    Ipts = [tuple(p) for p, m in zip(Ia, med) if abs(p[0] - m) < 1.5]
    keep = [Ipts[-1]]                                         # from the bottom up: no jumps
    for p in Ipts[-2::-1]:
        if abs(p[0] - keep[-1][0]) <= 1.2 * abs(p[1] - keep[-1][1]) + 1.5:
            keep.append(p)
    Ipts = keep[::-1]
    gE = [[y, float(np.median([rows[k]['E_peak'] for k in rows if abs(k - y) < 12 and rows[k].get('E_peak')
                               and not near_tick(k)] or [0]))] for y in range(170, 450, 20)]
    gI = [[y, float(np.median([rows[k]['I_peak'] for k in rows if abs(k - y) < 12 and rows[k].get('I_peak')]
                              or [0]))] for y in range(170, 550, 20)]
    # profiles on the mid rows
    ys = np.arange(250, 425)
    dEv, dIv, Lv, Yv = [], [], [], []
    for y in ys:
        if near_tick(y) or y not in rows:
            continue
        xe, xi = x_at(E, y + 0.5), x_at(Ipts, y + 0.5)
        xs = np.arange(xe, xi, 0.25)
        dEv += list(xs - xe); dIv += list(xi - xs)
        Lv += list(Lm[y * K + 1, (xs * K).astype(int)]); Yv += [y] * len(xs)
    dEv, dIv, Lv = map(np.array, (dEv, dIv, Lv))
    mid = (dEv > 14) & (dIv > 14)
    base = float(np.median(Lv[mid]))
    prof = lambda d: [[float(b), round(float(np.median(Lv[(d >= b - 0.25) & (d < b + 0.25) & (np.minimum(dEv, dIv) == d)]) - base), 1)]
                      for b in np.arange(0, 16.5, 1.0)]
    rgb_mid = []
    for y in range(180, 450, 20):
        xe, xi = x_at(E, y + 0.5), x_at(Ipts, y + 0.5)
        if xi - xe < 30:
            continue
        seg = img[y * K:(y + 1) * K, int((xe + 14) * K):int((xi - 14) * K)].reshape(-1, 3)
        rgb_mid.append([y] + np.median(seg, 0).round(1).tolist())
    return {'E': simplify(E, 0.12), 'I': simplify(Ipts, 0.12), 'E_peak': gE, 'I_peak': gI,
            'glow_E': prof(dEv), 'glow_I': prof(dIv), 'base': rgb_mid}


def norm(pts, nd=5):
    return [[round(x / W, nd), round(y / H, nd)] for x, y in pts]


def measure_zones(img, ex, y0, y1):
    """The coloured marks along the inside of O (rev zones): per sub-row the mean colour of the band right of O,
    grouped into red / yellow runs; their y range and their horizontal offsets from O's inner edge."""
    ys_, xl, xr, rr = ex
    segs, cur = [], None
    for s in range(int(y0 * K), int(y1 * K)):
        y = (s + 0.5) / K
        x0 = float(np.interp(y, ys_, xr))
        row = img[s, int(x0 * K):int((x0 + 12) * K)]
        R, G, B = row[:, 0], row[:, 1], row[:, 2]
        red = (R > 150) & (G < 110) & (R - B > 100)
        yel = (R > 150) & (G >= 110) & (R - B > 90)
        kind = 'red' if red.sum() > 2 * K and red.sum() >= yel.sum() else ('yellow' if yel.sum() > 2 * K else None)
        if kind:
            m = red | yel
            idx = np.where(m)[0]
            off = (idx[0] / K, (idx[-1] + 1) / K)
            col = row[m].mean(0)
        if cur and (not kind or kind != cur['kind']):
            segs.append(cur); cur = None
        if kind:
            if not cur:
                cur = {'kind': kind, 'y0': y, 'offs': [], 'cols': []}
            cur['y1'] = y; cur['offs'].append(off); cur['cols'].append(col)
    if cur:
        segs.append(cur)
    out = []
    for z in segs:
        if z['y1'] - z['y0'] < 1.0:
            continue
        o = np.median(np.array(z['offs']), 0)
        out.append({'kind': z['kind'], 'y': [round(z['y0'] - 0.125, 2), round(z['y1'] + 0.125, 2)],
                    'off': [round(float(o[0]), 2), round(float(o[1]), 2)],
                    'color': np.median(np.array(z['cols']), 0).round().astype(int).tolist()})
    return out


def measure_top(img, x0, x1, y0, y1):
    """O's thin end above the '8' (column by column): the ridge's y and its brightness."""
    Lm = img.mean(2)
    pts = []
    for x in np.arange(x0, x1, 1.0):
        col = Lm[int(y0 * K):int(y1 * K), int(x * K) + 1]
        pk = int(np.argmax(col))
        if col[pk] < 25:
            break
        c = centroid(col, max(0, pk - 6), pk + 7, col[pk] * 0.5)
        pts.append([round(float(x + 0.125), 2), round(y0 + c / K + 0.125, 2), int(col[pk])])
    return pts


def measure_digits(img, ticks_y, I):
    """The even scale numbers: the white blob right of I at each even tick (flood fill on the 1x grid), its box."""
    M = img.min(2)[K // 2::K, K // 2::K] > 128                # 1x sampling of the 4x image
    out = []
    for v, yc in ticks_y:
        if v % 2:
            continue
        xi = float(x_at(I, min(yc, I[-1][1] if I else yc)))
        found, x = None, int(xi + 6)
        while x < int(xi + 90) and not found:
            seed = next(((int(yc) + dy, x) for dy in (0, -2, 2, -4, 4, -6, 6) if M[int(yc) + dy, x]), None)
            if not seed:
                x += 1; continue
            seen, st = {seed}, [seed]
            while st and len(seen) < 3000:
                y, xx = st.pop()
                for q in ((y + 1, xx), (y - 1, xx), (y, xx + 1), (y, xx - 1)):
                    if q not in seen and M[q]:
                        seen.add(q); st.append(q)
            ys_, xs_ = zip(*seen)
            w_, h_ = max(xs_) - min(xs_) + 1, max(ys_) - min(ys_) + 1
            if 12 <= w_ <= 40 and 17 <= h_ <= 30:
                found = seen
            x = max(xs_) + 1
        if not found:
            continue
        seen = found
        ys_, xs_ = zip(*seen)
        # refine the box on the 4x grid
        A = img.min(2)[min(ys_) * K - K:(max(ys_) + 2) * K, min(xs_) * K - K:(max(xs_) + 2) * K] > 128
        r_, c_ = np.where(A.any(1))[0], np.where(A.any(0))[0]
        y0, x0 = min(ys_) - 1, min(xs_) - 1
        out.append({'text': str(v), 'box': [round(float(x0 + c_[0] / K), 2), round(float(y0 + r_[0] / K), 2),
                                             round(float(x0 + (c_[-1] + 1) / K), 2), round(float(y0 + (r_[-1] + 1) / K), 2)]})
    return out


def measure_fill(img, E, I):
    """The live rev bar as photographed (idle): the red bar's rows, the grey block under it (frame and body) and the
    red glow above it as a table: rows above the bar x 6 stops across the band (E -> I), red channel 0..1."""
    Rm = img[..., 0]
    yy = np.arange(480 * K, 530 * K)
    xc = [(x_at(E, y / K) + x_at(I, y / K)) / 2 for y in yy]
    px_ = np.array([img[y, int((x - 4) * K):int((x + 4) * K)].mean(0) for y, x in zip(yy, xc)])
    red = (px_[:, 0] > 200) & (px_[:, 1] > 15)
    runs = np.split(np.where(red)[0], np.where(np.diff(np.where(red)[0]) > 1)[0] + 1)
    rr = yy[max(runs, key=len)]                               # the longest run of red rows: the bar
    bar = [float(rr[0] / K), float((rr[-1] + 1) / K)]
    glow = []
    for dy in list(range(1, 12, 2)) + list(range(14, 104, 6)):
        y = bar[0] - dy
        xe, xi = x_at(E, y), x_at(I, y)
        st = []
        for u in np.linspace(0.08, 0.92, 6):
            x = xe + (xi - xe) * u
            hy = 1 if dy < 12 else 3.5                       # a whole dot period away from the bar (the dots averaged out)
            st.append(round(float(np.mean(Rm[int((y - hy) * K):int((y + hy) * K), int((x - 3.5) * K):int((x + 3.5) * K)]) / 255), 3))
        glow.append([dy] + st)
    blk = img[int((bar[1] + 6) * K):int((bar[1] + 24) * K), int((x_at(E, bar[1] + 15) + 6) * K):int((x_at(I, bar[1] + 15) - 6) * K)]
    # the dot grid over the glow: its period and phase (peaks of the red channel's column / row means)
    def peaks(v, o):
        v = np.convolve(v, np.ones(3) / 3, 'same')
        return [o + i / K for i in range(2, len(v) - 2) if v[i] == v[i - 2:i + 3].max() and v[i] > np.percentile(v, 60)]
    yb = bar[0]
    xm = (x_at(E, yb - 30) + x_at(I, yb - 30)) / 2
    sub = Rm[int((yb - 50) * K):int((yb - 6) * K), int((xm - 18) * K):int((xm + 18) * K)]
    cx, cy = peaks(sub.mean(0), xm - 18), peaks(sub.mean(1), yb - 50)
    cx = [c for c in cx if not any(0 < c - d < 4 for d in cx)]
    cy = [c for c in cy if not any(0 < c - d < 4 for d in cy)]
    per = float(np.median(np.diff(cx + [])) if len(cx) > 1 else 7.1)
    py = float(np.median(np.diff(cy))) if len(cy) > 1 else per
    py = py / 2 if py > 1.5 * per else py
    dots = {'origin': [round(cx[0], 2), round(cy[0], 2)], 'period': [round(per, 2), round(py, 2)],
            'c': 0.86, 'a': 0.24, 'b': 0.2, 'sigma': 0.9}
    return {'dots': dots, 'bar': [round(v, 2) for v in bar], 'bar_color': img[int((bar[0] + bar[1]) / 2 * K), int(xc[len(xc) // 2] * K)].round().astype(int).tolist(),
            'block': np.median(blk.reshape(-1, 3), 0).round().astype(int).tolist(), 'glow': glow}


def zero_line(img, y0, I):
    """The white line across the band at 0 (the block's floor): its rows and its right end (left: the tick)."""
    M = img.min(2)
    xi = x_at(I, y0)
    rows = [s for s in range(int((y0 - 3) * K), int((y0 + 4) * K)) if (M[s, int((xi - 8) * K):int((xi - 2) * K)] > 128).all()]
    if not rows:
        return None
    s = rows[len(rows) // 2]
    r = M[s, int((xi - 2) * K):int((xi + 12) * K)]
    end = xi - 2 + np.where(r > 128)[0][-1] / K if (r > 128).any() else xi
    k = np.where(r < 128)[0]
    end = xi - 2 + (k[0] / K if len(k) else 0)
    return {'y': [round(rows[0] / K, 2), round((rows[-1] + 1) / K, 2)], 'x_end': round(float(end), 2)}


def measure_floor(img, ys_, xr, y0, I):
    """The dark red floor under the zero line, down to the foot: per row its right edge and its colour."""
    Lm = img.mean(2)
    out = []
    for y in np.arange(y0 + 2.5, 562.0, 1.0):
        xl = float(np.interp(y, ys_, xr)) + 3.0                 # past O and its dark gap
        row = Lm[int(y * K) + 1, int(xl * K):int((xl + 60) * K)]
        hi = np.where(row > 12)[0]
        if not len(hi):
            continue
        cut = np.where(np.diff(hi) > 2 * K)[0]                    # stop before the digit
        e = hi[cut[0]] if len(cut) else hi[-1]
        xe = xl + (e + 1) / K
        x_in = xl + hi[0] / K
        if xe - x_in < 4:
            continue
        rgb = np.median(img[int(y * K):int(y * K) + K, int((x_in + 1) * K):int((xe - 1) * K)].reshape(-1, 3), 0)
        out.append([round(float(y), 2), round(float(x_in), 2), round(float(xe), 2)] + rgb.round().astype(int).tolist())
    return out


def build_blade(axis=None, x_start=565):
    img = load(axis)
    rows = trace_blade(img, 135, 568, x_start, 345)
    m = measure_blade(img, x_start)
    b = measure_body(img, rows, m['O'])
    tk = sorted(m['ticks'], key=lambda t: -t['tip'][1])            # 0 at the bottom
    ticks_y = [(i, t['tip'][1]) for i, t in enumerate(tk)]
    # E beyond its measured ends: parallel to O at the end offsets
    ys_, xl, xr = m['O_exp']
    E = b['E']
    # the grey block's white sides (as photographed) are E's and I's lower ends: measured row by row
    M = img.min(2)
    fl, fr = [], []
    for yb in np.arange(524, 543, 2.0):
        r = M[int(yb * K) + 1]
        x0 = int((float(np.interp(yb, ys_, xr)) + 1) * K)
        b_ = r[x0:x0 + 60 * K] > 128
        ed = np.flatnonzero(np.diff(b_.astype(int)))
        runs = [((x0 + ed[i] + 1) / K, (x0 + ed[i + 1] + 1) / K) for i in range(0 if b_[0] == 0 else 1, len(ed) - 1, 2)]
        runs = [q for q in runs if q[1] - q[0] > 1.0]
        if len(runs) >= 2:
            fl.append([(runs[0][0] + runs[0][1]) / 2, float(yb), runs[0][1] - runs[0][0]])
            fr.append([(runs[1][0] + runs[1][1]) / 2, float(yb), runs[1][1] - runs[1][0]])
    I = [p for p in b['I'] if p[1] <= 505] + [q[:2] for q in fr]
    E = E + [q[:2] for q in fl]
    dE_lo = E[-1][0] - float(np.interp(E[-1][1], ys_, xr))
    E = E + [[float(np.interp(y, ys_, xr)) + dE_lo, float(y)] for y in np.arange(E[-1][1] + 4, 552, 4)]
    dI_lo = I[-1][0] - float(np.interp(I[-1][1], ys_, xr))
    I = I + [[float(np.interp(y, ys_, xr)) + dI_lo, float(y)] for y in np.arange(I[-1][1] + 6, 553, 6)]
    O = [p for p in m['O'] if p[1] >= 153]
    O_top = measure_top(img, int(O[0][0]), 800, 132, 156)
    # above I's top the body goes on, faint, up to O's thin end: its right edge on the rows over the '8'
    Lm = img.mean(2)
    ext = []
    for y in (140, 143, 146):
        row = Lm[y * K + 1, int(740 * K):int(800 * K)]
        idx = np.where(row > 7)[0]
        if len(idx):
            ext.append([740 + (idx[-1] + 1) / K, y + 0.5])
    I = sorted(ext + I, key=lambda p: p[1])
    digits = measure_digits(img, ticks_y, I)
    zl = zero_line(img, ticks_y[0][1], I)
    floor = measure_floor(img, ys_, xr, ticks_y[0][1], I)
    d0 = next((d['box'] for d in digits if d['text'] == '0'), None)
    if d0 and zl:                                             # rows beside the '0': the right edge runs from the
        free = [r for r in floor if r[0] > d0[3] + 0.5]       # zero line's end to the first row under the digit
        if free:
            for r in floor:
                if r[0] <= d0[3] + 0.5:
                    r[2] = round(float(np.interp(r[0], [zl['y'][1], free[0][0]], [zl['x_end'], free[0][2]])), 2)
    el = {
        'frame_w': [round(float(np.median([q[2] for q in fl])), 2) if fl else 3.0,
                    round(float(np.median([q[2] for q in fr])), 2) if fr else 3.0],
        'O': O, 'O_width': m['O_width'], 'O_top': O_top, 'I_top': I[len(ext)][1],
        'ticks': [{'wedge': [[float(np.interp(t['y0'], ys_, xr)) - 0.5, t['y0']], [t['tip'][0], t['tip'][1] - t['tip_h'] / 2],
                             [t['tip'][0], t['tip'][1] + t['tip_h'] / 2], [float(np.interp(t['y1'], ys_, xr)) - 0.5, t['y1']]],
                   'slit': ([[float(np.interp(t['slit']['y0'], ys_, xl)) - 1.0, t['slit']['y0']], t['slit']['apex'],
                             [float(np.interp(t['slit']['y1'], ys_, xl)) - 1.0, t['slit']['y1']]] if 'slit' in t else None)}
                  for t in tk],
        'values_y': [round(y, 2) for _, y in ticks_y],
        'E': E, 'I': I, 'E_peak': b['E_peak'], 'I_peak': b['I_peak'], 'glow_E': b['glow_E'], 'glow_I': b['glow_I'],
        'base': b['base'],
        'zones': measure_zones(img, (ys_, xl, xr, None), 150, 230),
        'digits': digits,
        'fill': measure_fill(img, E, I),
        'zero_line': zl,
        'floor': floor,
    }
    return el


def to_display(el, axis):
    """Mirror a blade traced on the flipped image back to the display (x -> 2*axis - x), and normalize."""
    def fx(x):
        return 2 * axis - x if axis is not None else x
    def P(pts):
        return [[round(fx(p[0]) / W, 5), round(p[1] / H, 5)] + list(p[2:]) for p in pts]
    out = dict(el)
    for k in ('O', 'E', 'I', 'O_top'):
        out[k] = P(el[k])
    out['floor'] = [[r[0], round(fx(r[1]), 2), round(fx(r[2]), 2)] + r[3:] for r in el['floor']]
    if el['zero_line']:
        out['zero_line'] = dict(el['zero_line'], x_end=round(fx(el['zero_line']['x_end']), 2))
    f = el['fill']['dots']
    out['fill'] = dict(el['fill'], dots=dict(f, origin=[round(fx(f['origin'][0]), 2), f['origin'][1]]))
    out['ticks'] = [{'wedge': P(t['wedge']), 'slit': P(t['slit']) if t['slit'] else None} for t in el['ticks']]
    out['values_y'] = [round(y / H, 5) for y in el['values_y']]
    for d in out['digits']:
        x0, y0, x1, y1 = d['box']
        xa, xb = sorted((fx(x0), fx(x1)))
        d['box'] = [round(xa / W, 5), round(y0 / H, 5), round(xb / W, 5), round(y1 / H, 5)]
    out['zones'] = [dict(z, y=[round(v / H, 5) for v in z['y']]) for z in el['zones']]
    out['side'] = 'right' if axis is not None else 'left'
    return out


def trace_shapes(img, box, thr=128, chan='min'):
    """Shapes inside box (display px) traced sub-row by sub-row: each white run is linked to the run above it that
    overlaps it; a shape's outline is its runs' left edges down and right edges up (shapes that are x-convex per
    row; a glyph's counter comes back as its own dark shape via trace_holes). Returns polygons, display px."""
    x0, y0, x1, y1 = box
    A = img.min(2) if chan == 'min' else img.mean(2)
    shapes, open_ = [], []
    for s_ in range(int(y0 * K), int(y1 * K)):
        row = A[s_, int(x0 * K):int(x1 * K)] > thr
        e = np.flatnonzero(np.diff(np.r_[0, row.astype(int), 0]))
        runs = [(x0 + e[i] / K, x0 + e[i + 1] / K) for i in range(0, len(e), 2)]
        nxt = []
        for r in runs:
            hit = next((sh for sh in open_ if sh['R'][-1][0] > r[0] and sh['L'][-1][0] < r[1] and sh['last'] == s_ - 1
                        and not sh.get('taken')), None)
            if hit is None:
                hit = {'L': [], 'R': []}
                shapes.append(hit)
            hit['L'].append((r[0], s_ / K)); hit['R'].append((r[1], s_ / K)); hit['last'] = s_; hit['taken'] = True
            nxt.append(hit)
        for sh in nxt:
            sh['taken'] = False
        open_ = nxt
    out = []
    for sh in shapes:
        if len(sh['L']) < 2:
            continue
        L = sh['L'] + [(sh['L'][-1][0], sh['L'][-1][1] + 1 / K)]
        R = sh['R'] + [(sh['R'][-1][0], sh['R'][-1][1] + 1 / K)]
        L = simplify(L, 0.12); R = simplify(R, 0.12)
        out.append([[round(x, 2), round(y, 2)] for x, y in L + R[::-1]])
    return out


def trace_glyph(img, box, thr=128):
    """A glyph with counters: its outline (the white extent per sub-row) and its holes (dark runs between white runs)."""
    x0, y0, x1, y1 = box
    A = img.min(2)
    L, R, holes, cur = [], [], [], None
    for s_ in range(int(y0 * K), int(y1 * K)):
        row = A[s_, int(x0 * K):int(x1 * K)] > thr
        idx = np.flatnonzero(row)
        if not len(idx):
            continue
        y = s_ / K
        L.append((x0 + idx[0] / K, y)); R.append((x0 + (idx[-1] + 1) / K, y))
        gaps = np.flatnonzero(np.diff(idx) > 1)
        if len(gaps):
            g0 = x0 + (idx[gaps[0]] + 1) / K; g1 = x0 + idx[gaps[0] + 1] / K
            if cur is None:
                cur = {'L': [], 'R': []}; holes.append(cur)
            cur['L'].append((g0, y)); cur['R'].append((g1, y))
        else:
            cur = None
    def poly(Lp, Rp):
        Lp = Lp + [(Lp[-1][0], Lp[-1][1] + 1 / K)]; Rp = Rp + [(Rp[-1][0], Rp[-1][1] + 1 / K)]
        return [[round(x, 2), round(y, 2)] for x, y in simplify(Lp, 0.12) + simplify(Rp, 0.12)[::-1]]
    return {'outline': poly(L, R), 'holes': [poly(h['L'], h['R']) for h in holes if len(h['L']) > 2]}


def measure_center(img):
    """Component 2: the gear (P), the speed (0) with km/h, the horizontal line with its end pieces."""
    Lm = img.mean(2)
    P = trace_glyph(img, (900, 290, 1040, 390))
    Z = trace_glyph(img, (920, 415, 1020, 485))
    line = trace_shapes(img, (760, 390, 1175, 402), thr=90, chan='mean')
    line = [p for p in line if (max(q[0] for q in p) - min(q[0] for q in p)) > 2
            and (max(q[1] for q in p) - min(q[1] for q in p)) > 1.5]          # no fringe pieces
    # colours: the median of each part's lit pixels
    def col(poly):
        xs = [q[0] for q in poly]; ys = [q[1] for q in poly]
        sub = img[int(min(ys) * K):int(max(ys) * K), int(min(xs) * K):int(max(xs) * K)].reshape(-1, 3)
        sub = sub[sub.mean(1) > 90]
        return np.median(sub, 0).round().astype(int).tolist()
    km = img[int(425 * K):int(447 * K), int(1072 * K):int(1124 * K)]
    m = km.min(2) > 60
    r_, c_ = np.where(m.any(1))[0], np.where(m.any(0))[0]
    kmbox = [round(1072 + c_[0] / K, 2), round(425 + r_[0] / K, 2), round(1072 + (c_[-1] + 1) / K, 2), round(425 + (r_[-1] + 1) / K, 2)]
    kmcol = np.percentile(km.reshape(-1, 3)[m.reshape(-1)], 90, axis=0).round().astype(int).tolist()
    return {'gear': {'text': 'P', 'glyph': P, 'color': col(P['outline'])},
            'speed': {'text': '0', 'glyph': Z, 'color': col(Z['outline'])},
            'unit': {'text': 'km/h', 'box': kmbox, 'color': kmcol},
            'line': [{'poly': p, 'color': col(p)} for p in sorted(line, key=lambda p: p[0][0])]}


PHOTO = os.path.join(ROOT, 'design', 'refs', 'bmw_press', 'P90346920_highRes_m-view-05-2019.jpg')
PLACE = (0.3104, 39.1, -312.7)                                 # display = photo * s + (ox, oy)


def make_x10(box, name):
    """The photo resampled straight onto the display grid x 10 over box (display px): refs/id7/<name>.png."""
    s_, ox, oy = PLACE
    X0, Y0, X1, Y1 = box
    src = Image.open(PHOTO).convert('RGB')
    px0, py0 = (X0 - ox) / s_, (Y0 - oy) / s_
    px1, py1 = (X1 - ox) / s_, (Y1 - oy) / s_
    c0x, c0y = int(px0) - 2, int(py0) - 2
    crop = src.crop((c0x, c0y, int(px1) + 3, int(py1) + 3))
    a_ = 1 / (s_ * 10)
    big = crop.transform(((X1 - X0) * 10, (Y1 - Y0) * 10), Image.AFFINE, (a_, 0, px0 - c0x, 0, a_, py0 - c0y),
                         resample=Image.BICUBIC)
    out = os.path.join(ROOT, 'design', 'refs', 'id7', name + '.png')
    big.save(out)
    return out


def load_x10(name):
    return np.asarray(Image.open(os.path.join(ROOT, 'design', 'refs', 'id7', name + '.png')).convert('RGB')).astype(np.float32)


def vectorize_x10(img10, origin, box, layers, key, blur=3):
    """Colour layers of a region of a 10x enlargement (display grid x 10) as polygons in display px: each layer is a
    mask key(img) >= level, traced row by row at 10x (each white run linked to the overlapping run above it, every
    chain an outline: left edges down, right edges up), simplified to 0.05 px; painted in order (lowest level first)
    they rebuild the photo's shading. Returns [{level, color, polys}]."""
    from PIL import ImageFilter
    ox, oy = origin
    x0, y0, x1, y1 = [int(round(v * 10)) for v in ((box[0] - ox), (box[1] - oy), (box[2] - ox), (box[3] - oy))]
    sub = Image.fromarray(img10[y0:y1, x0:x1].astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur))
    a = np.asarray(sub).astype(np.float32)
    v = key(a)
    out = []
    for i, lv in enumerate(layers):
        m = v >= lv
        nxt = layers[i + 1] if i + 1 < len(layers) else 1e9
        band = m & (v < nxt)
        if not band.any():
            continue
        col = np.median(a[band], 0).round().astype(int).tolist()
        shapes, open_ = [], []
        for r in range(m.shape[0]):
            e = np.flatnonzero(np.diff(np.r_[0, m[r].astype(int), 0]))
            runs = [(e[j], e[j + 1]) for j in range(0, len(e), 2) if e[j + 1] - e[j] >= 2]
            nxt_open = []
            for r0, r1 in runs:
                hit = next((sh for sh in open_ if sh['R'][-1][0] > r0 and sh['L'][-1][0] < r1 and not sh.get('t')), None)
                if hit is None:
                    hit = {'L': [], 'R': []}; shapes.append(hit)
                hit['L'].append((r0, r)); hit['R'].append((r1, r)); hit['t'] = True
                nxt_open.append(hit)
            for sh in nxt_open:
                sh['t'] = False
            open_ = nxt_open
        polys = []
        for sh in shapes:
            if len(sh['L']) < 3:
                continue
            L = sh['L'] + [(sh['L'][-1][0], sh['L'][-1][1] + 1)]
            R = sh['R'] + [(sh['R'][-1][0], sh['R'][-1][1] + 1)]
            pts = simplify(L, 0.5) + simplify(R, 0.5)[::-1]
            polys.append([[round(ox + (x0 + x) / 10, 2), round(oy + (y0 + y) / 10, 2)] for x, y in pts])
        out.append({'level': lv, 'color': col, 'polys': polys})
    return out


def _bilinear(img, xs, ys):
    """img (H x W) sampled at float pixel coordinates."""
    h, w = img.shape
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 2); y0 = np.clip(np.floor(ys).astype(int), 0, h - 2)
    fx, fy = np.clip(xs - x0, 0, 1), np.clip(ys - y0, 0, 1)
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def _catmull(pts, step):
    P = [pts[0]] + list(pts) + [pts[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = (np.array(P[i + k], float) for k in (-1, 0, 1, 2))
        n = max(2, int(np.linalg.norm(p2 - p1) / step))
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(np.array(pts[-1], float))
    return np.array(out)


def densify_np(pts, step):
    return _catmull([tuple(q) for q in pts], step)


def ridge(key10, origin, pts, hw, min_peak=90, step=1.0, iters=3):
    """A bright line followed on the 10x image: at every point (display px) the profile across the line (+-hw px,
    0.1 px steps); its centre = the centroid of the part above half the local peak, its width = that part's length.
    Points without a clear peak (gaps, ticks) are bridged. Returns (centre points, widths, peaks), display px."""
    ox, oy = origin
    P = _catmull(pts, step)
    for _ in range(iters):
        tg = np.gradient(P, axis=0); tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9
        nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
        ss = np.arange(-hw, hw + 1e-6, 0.1)
        X = (P[:, None, 0] + nr[:, None, 0] * ss[None] - ox) * 10 - 0.5
        Y = (P[:, None, 1] + nr[:, None, 1] * ss[None] - oy) * 10 - 0.5
        prof = _bilinear(key10, X, Y)
        C, Wd, Pk, ok = [], [], [], []
        for i, pr in enumerate(prof):
            # the local maximum nearest the track (a brighter neighbouring line must not pull it over)
            pkx = [j for j in range(1, len(pr) - 1) if pr[j] >= pr[j - 1] and pr[j] >= pr[j + 1] and pr[j] >= 0.5 * pr.max()]
            k = min(pkx, key=lambda j: abs(ss[j])) if pkx else int(np.argmax(pr))
            pk = pr[k]
            if pk < min_peak:
                C.append(0.0); Wd.append(0.0); Pk.append(pk); ok.append(False); continue
            half = pk / 2
            lo = k
            while lo > 0 and pr[lo - 1] >= half:
                lo -= 1
            hi = k
            while hi < len(pr) - 1 and pr[hi + 1] >= half:
                hi += 1
            seg = pr[lo:hi + 1] - half
            C.append(float((ss[lo:hi + 1] * seg).sum() / seg.sum())); Wd.append((hi - lo + 1) * 0.1); Pk.append(pk); ok.append(True)
        ok = np.array(ok); C = np.array(C); Wd = np.array(Wd)
        idx = np.arange(len(P))
        if ok.sum() < 2:
            break
        C = np.interp(idx, idx[ok], C[ok]); Wd = np.interp(idx, idx[ok], Wd[ok])
        k_ = np.ones(5) / 5
        C = np.convolve(np.pad(C, 2, mode='edge'), k_, 'valid')
        P = P + nr * C[:, None]
    return P, Wd, np.array(Pk), ok


# coarse starting tracks (whole px) for the fuel gauge's A, B, C: they only start the ridge search, so a rebuild
# never depends on the previous one; the temperature gauge starts from their mirror image (its own frame)
SEEDS_FUEL = {"A": [[235, 419], [224, 436], [216, 452], [215, 468], [219, 484], [225, 500], [232, 516], [241, 532], [251, 548], [263, 563], [278, 577], [294, 584], [314, 586], [332, 586]], "B": [[232, 432], [223, 448], [220, 464], [223, 480], [229, 496], [236, 512], [245, 528], [255, 544], [266, 559], [282, 573], [298, 579], [303, 580]], "C": [[239, 446], [234, 464], [238, 480], [244, 496], [252, 512], [261, 528], [274, 545], [294, 563], [310, 566], [342, 563], [350, 560]]}


def _ext(pts, L=8.0):
    pts = np.array(pts, float)
    a0 = pts[0] + (pts[0] - pts[1]) / np.linalg.norm(pts[0] - pts[1]) * L
    a1 = pts[-1] + (pts[-1] - pts[-2]) / np.linalg.norm(pts[-1] - pts[-2]) * L
    return np.vstack([a0, pts, a1])


def gauge10(img10, origin, seeds, label_boxes, bar_box):
    """The fuel-type gauge rebuilt from a 10x image only (display px, the fuel's orientation: fill to the right of C).
    seeds: rough A / B / C polylines that only start the ridge search; every returned value is measured.
      - A, B, C: ridge centres (trimmed where the line ends), widths and colours;
      - levels: arc position on C and slope of each scale line (dark lines in the lit fill, the bar for 'full');
      - extras: the white marks not explained by A / B (the ticks, the top corner), as colour layers (static);
      - fill: everything between B's inner edge and the fill's far side (track, C, fill, the 1/2 tick), labels
        and red filled in from their sides, as layers cut live at the level;
      - bar: the red level bar, layers local to its level point."""
    from PIL import ImageDraw as IDr
    ox, oy = origin
    key = img10.min(2)
    Lm = img10.mean(2)
    H10, W10 = key.shape
    out, lines = {}, {}
    for name, hw in (('A', 2.6), ('B', 2.6), ('C', 2.4)):
        P, Wd, Pk, ok = ridge(key, origin, _ext(seeds[name], 14.0), hw, min_peak=50)
        mid = Pk[len(Pk) // 4:3 * len(Pk) // 4]
        g_ = np.flatnonzero(Pk >= 0.45 * float(np.median(mid)))    # where the line ends (its own brightness)
        P, Wd = P[g_[0]:g_[-1] + 1], Wd[g_[0]:g_[-1] + 1]
        w_ = float(np.median(Wd[Wd > 0]))
        rgb = np.median(img10[np.clip(((P[:, 1] - oy) * 10).astype(int), 0, H10 - 1),
                              np.clip(((P[:, 0] - ox) * 10).astype(int), 0, W10 - 1)], 0)
        lines[name] = (P, w_)
        out[name] = [[round(float(x), 2), round(float(y), 2)] for x, y in simplify([tuple(q) for q in P], 0.08)]
        out[name + '_width'] = round(w_, 2)
        out[name + '_color'] = rgb.round().astype(int).tolist()
    # the scale lines: darker lines leaving C into the lit fill (contrast to their parallels, 4..26 px from C)
    PC = lines['C'][0]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(PC, axis=0), axis=1))]
    cand = []
    for i in range(len(PC)):
        for k in np.arange(-0.45, 0.01, 0.025):
            d = np.array([1, k]); d /= np.linalg.norm(d); nrm = np.array([-d[1], d[0]])
            q = PC[i] + d * np.arange(4, 26, 0.5)[:, None]
            v = lambda qq: _bilinear(Lm, (qq[:, 0] - ox) * 10 - .5, (qq[:, 1] - oy) * 10 - .5).mean()
            c = min(v(q - nrm * 2.5), v(q + nrm * 2.5)) - v(q)
            cand.append((c, i, k))
    cand.sort(reverse=True)
    picked = []
    for c, i, k in cand:
        if c < 8:
            break
        if all(abs(i - j) > 8 for _, j, _ in picked):
            picked.append((c, i, k))
    out['_scale_lines'] = [[round(float(arc[i]), 2), round(float(k), 3), round(float(c), 1)] for c, i, k in sorted(picked, key=lambda x: x[1])]
    # the bar: its slope (red pixels left of the label) and its point on C
    R_, G_, B_c = img10[..., 0], img10[..., 1], img10[..., 2]
    x0b, y0b, x1b, y1b = bar_box
    sub = (R_ > 200) & (G_ < 120)
    ys, xs = np.nonzero(sub[int((y0b - oy) * 10):int((y1b - oy) * 10), int((x0b - ox) * 10):int((x1b - ox) * 10)])
    k_bar = float(np.polyfit(x0b + xs / 10, y0b + ys / 10, 1)[0])
    out['_bar_slope'] = round(k_bar, 3)
    out['_arc'] = [round(float(v), 2) for v in arc[[0, -1]]]
    return out, lines


def gauge10_build(img10, origin, seeds, level_pick, label_boxes, bar_box, top_box, region_far, far_len=85, bar_img=None,
                  anchor=None):
    """The whole fuel-type gauge from the 10x image (see gauge10): returns the config fields, display px."""
    from PIL import ImageDraw as IDr
    ox, oy = origin
    g, lines = gauge10(img10, origin, seeds, label_boxes, bar_box)
    key = img10.min(2)
    H10, W10 = key.shape
    T = lambda x, y: ((x - ox) * 10, (y - oy) * 10)
    PC = lines['C'][0]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(PC, axis=0), axis=1))]
    # levels: each scale line's point on C (absolute, so the scale does not depend on where C's trace starts) and
    # slope; picks come as points, or as arc lengths from an anchor point on C
    a0 = float(arc[int(np.argmin(np.linalg.norm(PC - np.array(anchor), axis=1)))]) if anchor is not None else 0.0
    pts_ = []
    for v, q, k in level_pick:
        if np.ndim(q) == 0:
            q = PC[int(np.argmin(np.abs(arc - (a0 + q))))]
        pts_.append([v, round(float(q[0]), 2), round(float(q[1]), 2)])
    lv = {'pts': pts_, 'slope': [[v, k] for v, _, k in level_pick]}
    def lp(v):
        ar = [float(arc[int(np.argmin(np.linalg.norm(PC - np.array(q[1:]), axis=1)))]) for q in lv['pts']]
        a_ = np.interp(v, [q[0] for q in lv['pts']], ar)
        i = int(np.argmin(np.abs(arc - a_)))
        k = np.interp(v, [q[0] for q in lv['slope']], [q[1] for q in lv['slope']])
        d = np.array([1.0, k]); return PC[i], d / np.linalg.norm(d)
    # the region that follows the level: between B's inner edge and the fill's far side, up to the 'full' line
    PB, wB = lines['B']
    tg = np.gradient(PB, axis=0); tg /= np.linalg.norm(tg, axis=1, keepdims=True)
    nB = np.stack([-tg[:, 1], tg[:, 0]], 1)
    side = 1 if _bilinear(img10.mean(2), *[((PB + nB * 5)[:, j] - (ox, oy)[j]) * 10 - .5 for j in (0, 1)]).mean() > \
        _bilinear(img10.mean(2), *[((PB - nB * 5)[:, j] - (ox, oy)[j]) * 10 - .5 for j in (0, 1)]).mean() else -1
    Bi = PB + nB * side * (wB / 2 + 0.6)
    p1, d1 = lp(1.0)
    above = lambda q: (d1[0] * (q[1] - p1[1]) - d1[1] * (q[0] - p1[0])) < -2.5
    keepC = [q for q in PC if not above(q)]                     # C only up to 'full' (above it it never shows)
    g['C'] = [[round(float(x), 2), round(float(y), 2)] for x, y in simplify([tuple(q) for q in keepC], 0.08)]
    Bi = [q for q in Bi if not above(q)]
    region = Bi + region_far + [tuple(p1 + d1 * far_len)]
    m = Image.new('L', (W10, H10), 0)
    IDr.Draw(m).polygon([T(x, y) for x, y in region], fill=255)
    m = np.asarray(m) > 0
    tex = img10.copy()
    for (a0, b0, a1, b1) in label_boxes:                       # labels out: rows filled in from their sides
        y0, y1, x0, x1 = int((b0 - oy) * 10), int((b1 - oy) * 10), int((a0 - ox) * 10), int((a1 - ox) * 10)
        for r in range(y0, y1):
            l_, r_ = tex[r, x0 - 1].copy(), tex[r, x1].copy()
            tex[r, x0:x1] = l_ + (r_ - l_) * np.linspace(0, 1, x1 - x0)[:, None]
    red = (tex[..., 0] - np.maximum(tex[..., 1], tex[..., 2])) > 40
    tex[red] = 0
    tex[~m] = 0
    xs_ = [q[0] for q in region]; ys_ = [q[1] for q in region]
    fill = vectorize_x10(tex, origin, (min(xs_) - 1, min(ys_) - 1, max(xs_) + 1, max(ys_) + 1),
                         list(range(6, 250, 8)), key=lambda a: a.mean(-1), blur=4)
    # static extras: white not explained by A / B, outside the level region, and the top corner (minus A)
    lm = Image.new('L', (W10, H10), 0); d = IDr.Draw(lm)
    for nm in ('A', 'B'):
        P_, w_ = lines[nm]
        d.line([T(x, y) for x, y in P_], fill=255, width=int((w_ + 1.6) * 10), joint='curve')
    lm = np.asarray(lm) > 0
    tb = np.zeros_like(m)
    x0t, y0t, x1t, y1t = top_box
    tb[int((y0t - oy) * 10):int((y1t - oy) * 10), int((x0t - ox) * 10):int((x1t - ox) * 10)] = True
    la = Image.new('L', (W10, H10), 0)
    IDr.Draw(la).line([T(x, y) for x, y in lines['A'][0]], fill=255, width=int((lines['A'][1] + 1.6) * 10), joint='curve')
    la = np.asarray(la) > 0
    ex = img10.copy()
    for (a0, b0, a1, b1) in label_boxes:
        ex[int((b0 - oy) * 10):int((b1 - oy) * 10), int((a0 - ox) * 10):int((a1 - ox) * 10)] = 0
    near = Image.new('L', (W10, H10), 0)                       # only next to A and B (no neighbouring readings)
    for nm in ('A', 'B'):
        P_, w_ = lines[nm]
        IDr.Draw(near).line([T(x, y) for x, y in P_], fill=255, width=int((w_ + 14) * 10), joint='curve')
    nb_ = Image.new('L', (W10, H10), 0)                        # the marks on B's inner side (ticks) are static
    IDr.Draw(nb_).line([T(x, y) for x, y in lines['B'][0]], fill=255, width=int((lines['B'][1] + 12) * 10), joint='curve')
    nb_ = np.asarray(nb_) > 0
    keep = ((~lm & (np.asarray(near) > 0) & (~m | nb_)) | (tb & ~la)) & ~red
    ex[~keep] = 0
    def white(a):
        v = np.where(a.max(-1) - a.min(-1) < 70, a.min(-1), 0)   # white / grey only (not the blue or red marks' edges)
        core, ok_ = v >= 200, v >= 90
        for _ in range(300):
            g_ = core.copy()
            g_[1:] |= core[:-1]; g_[:-1] |= core[1:]; g_[:, 1:] |= core[:, :-1]; g_[:, :-1] |= core[:, 1:]
            g_ &= ok_
            if (g_ == core).all():
                break
            core = g_
        return np.where(core, v, 0)
    allx = [q[0] for nm in ('A', 'B') for q in lines[nm][0]]; ally = [q[1] for nm in ('A', 'B') for q in lines[nm][0]]
    exm = white(ex) > 0                                         # ... and out of the live texture
    if exm.any():
        fill = vectorize_x10(np.where(exm[..., None], 0, tex), origin, (min(xs_) - 1, min(ys_) - 1, max(xs_) + 1, max(ys_) + 1),
                             list(range(6, 250, 8)), key=lambda a: a.mean(-1), blur=4)
    extras = vectorize_x10(ex, origin, (min(allx) - 6, min(ally) - 6, max(allx) + 14, max(ally) + 6),
                           list(range(90, 256, 14)), key=white, blur=1.5)
    # the bar, local to the 'full' point
    nb = np.array([-d1[1], d1[0]])
    bimg = (img10 if bar_img is None else bar_img).copy()
    bx0, by0, bx1, by1 = [int(round(v)) for v in (((bar_box[0] - ox) * 10), ((bar_box[1] - oy) * 10), ((bar_box[2] - ox) * 10), ((bar_box[3] - oy) * 10))]
    rm = (bimg[..., 0] - np.maximum(bimg[..., 1], bimg[..., 2])) > 80
    for (a0, b0, a1, b1) in label_boxes:                       # a label over the bar: the bar goes on under it as the
        y0, y1, x0, x1 = int((b0 - oy) * 10), int((b1 - oy) * 10), int((a0 - ox) * 10), int((a1 - ox) * 10)   # same slanted band
        if x1 < bx0 or x0 > bx1 or y1 < by0 or y0 > by1:
            continue
        cols = [c for c in range(max(bx0, x0), min(bx1, x1 + 400)) if not (x0 <= c < x1) or True]
        best = None
        for rng in (range(x1 + 5, min(bx1, x1 + 220)), range(max(bx0, x0 - 220), x0 - 5)):   # the side where the bar shows
            tops, bots, cs = [], [], []
            for c in rng:
                r_ = np.flatnonzero(rm[by0:by1, c])
                if len(r_) > 20:
                    tops.append(by0 + r_[0]); bots.append(by0 + r_[-1]); cs.append(c)
            if best is None or len(cs) > len(best[2]):
                best = (tops, bots, cs)
        tops, bots, cs = best
        if len(cs) < 20:
            continue
        kt, kb = np.polyfit(cs, tops, 1), np.polyfit(cs, bots, 1)
        ne = np.argsort(np.abs(np.array(cs) - (x0 + x1) / 2))[:10]      # its colour next to the label
        col = np.median(np.concatenate([bimg[int(tops[j] + 15):int(bots[j] - 15), cs[j]] for j in ne]), 0)
        for c in range(max(0, x0 - 2), x1 + 3):
            t_, b_ = np.polyval(kt, c), np.polyval(kb, c)
            col_ = bimg[:, c].copy()
            col_[y0:y1] = (10, 10, 12)
            rr = np.arange(len(col_))
            band = (rr >= t_) & (rr <= b_) & (rr >= y0 - 20) & (rr < y1 + 20)
            col_[band] = col
            bimg[:, c] = col_
    bar = vectorize_x10(bimg, origin, bar_box, [8, 16, 24] + list(range(32, 256, 12)),
                        key=lambda a: np.where(a[..., 0] - a[..., 1:].max(-1) > np.minimum(25, a[..., 0] * 0.5), a[..., 0], 0))
    NZ = lambda pp: [[round(x / W, 6), round(y / H, 6)] for x, y in pp]
    lay = lambda L_: [{'level': l_['level'], 'color': l_['color'], 'polys': [NZ(q) for q in l_['polys']]} for l_ in L_]
    g['levels'] = lv
    g['fill_layers'] = lay(fill)
    g['extras'] = lay(extras)
    g['bar_layers'] = [{'level': l_['level'], 'color': l_['color'],
                        'polys': [[[round(float(np.dot(np.array(q) - p1, d1)), 2), round(float(np.dot(np.array(q) - p1, nb)), 2)]
                                   for q in poly] for poly in l_['polys']]} for l_ in bar]
    for nm in ('A', 'B', 'C'):
        g[nm] = NZ(g[nm])
    return g


def apply_gauge(el, g, about):
    """Replace a gauge element's measured fields by the 10x rebuild (labels, reserve and overlays kept)."""
    for k in [k for k in el if k not in ('labels', 'reserve', 'overlay', 'mirror_axis', 'scale', 'photo_value')]:
        el.pop(k)
    el['_about'] = about
    el['A'], el['A_width'] = g['A'], g['A_width']
    el['B'], el['B_width'], el['B_corner'], el['B_tail'] = g['B'], g['B_width'], g['B'][0], 0
    el['C'], el['C_width'] = g['C'], g['C_width']
    el['colors'] = {'line': g['B_color'], 'A': g['A_color'], 'C': g['C_color']}
    el['levels'] = g['levels']
    el['fill'] = {'layers': g['fill_layers'], '_about': "track, C, the 1/2 tick and the lit fill as photographed (full), "
                  "between B's inner edge and the fill's far side: colour layers cut live at the level"}
    el['extras'] = g['extras']
    el['bar'] = {'layers': g['bar_layers'], '_about': "the bar at 'full', in px along / across its level line from its point on C"}
    el['_scale_lines'] = g['_scale_lines']


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else 'blade'
    if what == 'revs':                                        # trace_id7.py revs id7/Corsa: into its config.json
        revs = {'left': to_display(build_blade(None, 565), None), 'right': to_display(build_blade(960.0, 583), 960.0)}
        path = os.path.join(ROOT, 'layouts', sys.argv[2], 'config.json')
        cfg = json.load(open(path, encoding='utf-8'))
        el = cfg['elements'].setdefault('revs', {})
        for side in ('left', 'right'):
            keep = {k: v for k, v in el.get(side, {}).items() if k.startswith('_')}
            el[side] = dict(revs[side], **keep)
        json.dump(cfg, open(path, 'w', encoding='utf-8'), indent=1)
        print('revs -> ' + os.path.relpath(path, ROOT))
    if what == 'center':                                      # trace_id7.py center id7/Corsa
        c = measure_center(load(None))
        def nm(poly):
            return [[round(x / W, 5), round(y / H, 5)] for x, y in poly]
        for k in ('gear', 'speed'):
            g_ = c[k]['glyph']
            c[k]['glyph'] = {'outline': nm(g_['outline']), 'holes': [nm(h) for h in g_['holes']]}
        c['unit']['box'] = [round(c['unit']['box'][0] / W, 5), round(c['unit']['box'][1] / H, 5),
                            round(c['unit']['box'][2] / W, 5), round(c['unit']['box'][3] / H, 5)]
        for l_ in c['line']:
            l_['poly'] = nm(l_['poly'])
        path = os.path.join(ROOT, 'layouts', sys.argv[2], 'config.json')
        cfg = json.load(open(path, encoding='utf-8'))
        old = cfg['elements'].get('center', {})
        cfg['elements']['center'] = dict(c, **{k: v for k, v in old.items() if k.startswith('_') or k not in c})
        json.dump(cfg, open(path, 'w', encoding='utf-8'), indent=1)
        print('center -> ' + os.path.relpath(path, ROOT))
    if what == 'fuel10':                                      # trace_id7.py fuel10 id7/Strada id7/Corsa
        cfg = json.load(open(os.path.join(ROOT, 'layouts', sys.argv[2], 'config.json'), encoding='utf-8'))
        seeds = {k: np.array(v, float) for k, v in SEEDS_FUEL.items()}
        g = gauge10_build(load_x10('M8_fuel_x10'), (190, 400), seeds,
                          [(0.0, 150.63, -0.375), (0.25, 129.89, -0.325), (0.5, 79.17, -0.325), (0.75, 26.97, -0.2), (1.0, 0.0, -0.142)],
                          [(257.5, 428.5, 270.5, 449.5), (268.5, 490.5, 312.5, 511.5), (322.5, 535.5, 343.5, 557.5)],
                          (232, 428, 312, 456), (224, 412, 252, 434), [(352, 566), (366, 552), (352, 500), (330, 462), (305, 440)],
                          anchor=(240.4, 445.5))                # the scale lines' arcs were read from there (dark lines, bar)
        about = ("The fuel gauge, rebuilt on BMW M GmbH's photo only (M8 Competition, M View), at 10x on the display grid "
                 "(M8_fuel_x10.png, not in git): A, B, C followed as ridges across the lines (centres, widths, colours); "
                 "the scale (levels: arc position on C and slope of each line) read from the dark scale lines in the lit fill, "
                 "'full' from the red bar; the ticks and the top corner as static white layers (extras); track, C and fill "
                 "as colour layers cut live at the level; the bar local to its level point; labels and reserve as layers.")
        for lo_ in sys.argv[2:]:
            path = os.path.join(ROOT, 'layouts', lo_, 'config.json')
            c2 = json.load(open(path, encoding='utf-8'))
            apply_gauge(c2['elements']['fuel'], g, about)
            json.dump(c2, open(path, 'w', encoding='utf-8'), indent=1)
        print('fuel ->', sys.argv[2:], {k: (len(v) if isinstance(v, list) else v) for k, v in g.items() if not k.startswith('_')})
    if what == 'tempmirror10':                                # the temp photo mirrored onto the fuel's 10x grid
        ax = float(sys.argv[2])
        fx_, tx_ = load_x10('M8_fuel_x10'), load_x10('M8_temp_x10')
        off = int(round((2 * ax - 190 - 1550) * 10))
        jj = off - 1 - np.arange(fx_.shape[1]); ok_ = (jj >= 0) & (jj < tx_.shape[1])
        tm = np.zeros_like(fx_); tm[:, ok_] = tx_[:, jj[ok_]]
        Image.fromarray(tm.astype(np.uint8)).save(os.path.join(ROOT, 'design', 'refs', 'id7', 'M8_temp_mirror_x10.png'))
        print('ok')
    if what == 'temp10':                                      # trace_id7.py temp10 id7/Corsa
        from PIL import ImageDraw as IDr
        path = os.path.join(ROOT, 'layouts', sys.argv[2], 'config.json')
        cfg = json.load(open(path, encoding='utf-8'))
        f, tp = cfg['elements']['fuel'], cfg['elements']['temp']
        ax = tp['mirror_axis']
        mb = lambda b: (2 * ax - b[2], b[1], 2 * ax - b[0], b[3])
        t_labels = [mb(b) for b in [(1611.5, 438, 1659, 458.5), (1596.5, 489.5, 1645, 510), (1565, 533.5, 1601.5, 554)]]
        f_labels = [(257.5, 428.5, 270.5, 449.5), (268.5, 490.5, 312.5, 511.5), (322.5, 535.5, 343.5, 557.5)]
        tm = load_x10('M8_temp_mirror_x10')
        fx_ = load_x10('M8_fuel_x10').copy()
        for (a0, b0, a1, b1) in f_labels:                     # the fuel's labels out before it lends its fill
            y0, y1, x0, x1 = int((b0 - 400) * 10), int((b1 - 400) * 10), int((a0 - 190) * 10), int((a1 - 190) * 10)
            for r in range(y0, y1):
                l_, r_ = fx_[r, x0 - 1].copy(), fx_[r, x1].copy()
                fx_[r, x0:x1] = l_ + (r_ - l_) * np.linspace(0, 1, x1 - x0)[:, None]
        # A and B on both photos -> x_t = a x_f + b y + c (least squares on points at equal y)
        seeds_t = {k: np.array(v, float) for k, v in SEEDS_FUEL.items()}
        _, lt = gauge10(tm, (190, 400), seeds_t, [], (240, 495, 300, 520))
        rows = []
        for k in ('A', 'B'):
            Ft = np.array(f[k]) * [W, H]; Tt = lt[k][0]
            Fs, Ts = Ft[np.argsort(Ft[:, 1])], Tt[np.argsort(Tt[:, 1])]
            for y in np.arange(max(Fs[0, 1], Ts[0, 1]) + 3, min(Fs[-1, 1], Ts[-1, 1]) - 3, 2.0):
                xf, xt = np.interp(y, Fs[:, 1], Fs[:, 0]), np.interp(y, Ts[:, 1], Ts[:, 0])
                if abs(np.gradient(Fs[:, 0], Fs[:, 1])[np.searchsorted(Fs[:, 1], y) - 1]) < 2.5:
                    rows.append((xf, y, xt))
        R_ = np.array(rows)
        sol = np.linalg.lstsq(np.c_[R_[:, 0], R_[:, 1], np.ones(len(R_))], R_[:, 2], rcond=None)[0]
        res = R_[:, 2] - np.c_[R_[:, 0], R_[:, 1], np.ones(len(R_))] @ sol
        print('fuel -> temp: x_t = %.4f x + %.4f y + %.2f, residual rms %.2f px' % (*sol, np.sqrt((res ** 2).mean())))
        # the fuel's 10x image warped onto the temp grid
        H10, W10 = tm.shape[:2]
        Xg, Yg = np.meshgrid(190 + (np.arange(W10) + 0.5) / 10, 400 + (np.arange(H10) + 0.5) / 10)
        Xf = (Xg - sol[1] * Yg - sol[2]) / sol[0]
        warped = np.stack([_bilinear(fx_[..., c], (Xf - 190) * 10 - .5, (Yg - 400) * 10 - .5) for c in range(3)], -1)
        # the temp photo's level: the line along its bar (where its own C starts)
        PCt = lt['C'][0]
        pt = PCt[0]; kb = -0.325
        above = (Yg - pt[1]) - kb * (Xg - pt[0]) < -16.0      # clear of the photo's own bar
        # the level region (B inner edge .. far side): above the photo level -> the warped fuel, else the temp photo
        PB, wB = lt['B']
        tg = np.gradient(PB, axis=0); tg /= np.linalg.norm(tg, axis=1, keepdims=True); nB = np.stack([-tg[:, 1], tg[:, 0]], 1)
        Bi = PB + nB * (wB / 2 + 0.6)
        if _bilinear(tm.mean(2), (Bi[:, 0] - 190) * 10, (Bi[:, 1] - 400) * 10).mean() < _bilinear(tm.mean(2), ((PB - nB * 3)[:, 0] - 190) * 10, ((PB - nB * 3)[:, 1] - 400) * 10).mean():
            Bi = PB - nB * (wB / 2 + 0.6)
        m = Image.new('L', (W10, H10), 0)
        IDr.Draw(m).polygon([((x - 190) * 10, (y - 400) * 10) for x, y in list(map(tuple, Bi)) + [(352, 566), (372, 548), (372, 410), (Bi[0][0], 410)]], fill=255)
        reg = np.asarray(m) > 0
        nbm = Image.new('L', (W10, H10), 0)
        IDr.Draw(nbm).line([((x - 190) * 10, (y - 400) * 10) for x, y in PB], fill=255, width=int((wB + 11) * 10), joint='curve')
        comp = np.where((reg & above & ~(np.asarray(nbm) > 0))[..., None], warped, tm).astype(np.float32)
        # the C seed for the upper part: the fuel's C through the map
        Cf = np.array(f['C']) * [W, H]
        seeds_t['C'] = np.c_[sol[0] * Cf[:, 0] + sol[1] * Cf[:, 1] + sol[2], Cf[:, 1]]
        # the scale: the fuel's levels carried over (arc positions on the fuel's C -> points -> the temp C)
        picks = [(q[0], np.array([sol[0] * q[1] + sol[1] * q[2] + sol[2], q[2]]), round(float(k_[1] / sol[0]), 3))
                 for q, k_ in zip(f['levels']['pts'], f['levels']['slope'])]
        print('temp levels', picks)
        v_ph = (tp['photo_value'] - tp['scale'][0]) / (tp['scale'][1] - tp['scale'][0])
        # the bar box around the photo's bar, label boxes: the temp's (mirrored)
        g = gauge10_build(comp, (190, 400), seeds_t, picks, t_labels, (238, 488, 300, 522), (224, 412, 252, 434), [(352, 566), (366, 552), (352, 500), (330, 462), (305, 440)], bar_img=tm)
        # the bar was traced at the photo's level (v_ph), not at 'full': re-express it local to that level point
        lv = g['levels']
        Cn = np.array(densify_np(np.array(g['C']) * [W, H], 0.25))
        carc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Cn, axis=0), axis=1))]
        def lpt(v):
            ar = [float(carc[int(np.argmin(np.linalg.norm(Cn - np.array(q[1:]), axis=1)))]) for q in lv['pts']]
            a_ = np.interp(v, [q[0] for q in lv['pts']], ar)
            k_ = np.interp(v, [q[0] for q in lv['slope']], [q[1] for q in lv['slope']])
            d_ = np.array([1.0, k_]); return Cn[int(np.argmin(np.abs(carc - a_)))], d_ / np.linalg.norm(d_)
        p1, d1 = lpt(1.0); pv, dv = lpt(v_ph)
        n1, nv = np.array([-d1[1], d1[0]]), np.array([-dv[1], dv[0]])
        for l_ in g['bar_layers']:
            l_['polys'] = [[[round(float(np.dot(p1 + d1 * u + n1 * w_ - pv, dv)), 2), round(float(np.dot(p1 + d1 * u + n1 * w_ - pv, nv)), 2)]
                            for u, w_ in poly] for poly in l_['polys']]
        # its far end: on the photo the bar shows in the second 0's counter, not in the first's -> it ends between
        # them (display x 1626, the second 0's outer left edge, 1628, less its 2 px outline)
        umax = float((2 * ax - 1626.0 - pv[0]) / dv[0])
        def clip_u(poly, um):                                   # Sutherland-Hodgman against u <= um
            out_ = []
            for i_ in range(len(poly)):
                a_, b_ = poly[i_ - 1], poly[i_]
                ina, inb = a_[0] <= um, b_[0] <= um
                if inb:
                    if not ina:
                        t_ = (um - a_[0]) / (b_[0] - a_[0]); out_.append([um, round(a_[1] + t_ * (b_[1] - a_[1]), 2)])
                    out_.append(b_)
                elif ina:
                    t_ = (um - a_[0]) / (b_[0] - a_[0]); out_.append([um, round(a_[1] + t_ * (b_[1] - a_[1]), 2)])
            return out_
        for l_ in g['bar_layers']:
            l_['polys'] = [q for q in (clip_u(poly, umax) for poly in l_['polys']) if len(q) >= 3]
        about = ("The temperature gauge, rebuilt on BMW M GmbH's photo only, in the frame mirrored about x %.1f (drawn by the "
                 "fuel gauge's code, then mirrored): A, B, the extras, the bar and the fill under 106 °C from its own photo at "
                 "10x; above 106 °C (not shown on the photo) C, the fill and the scale come from the fuel gauge's photo warped "
                 "onto it by the fit of their A and B lines (x_t = %.4f x + %.4f y + %.2f)." % (ax, *sol))
        apply_gauge(tp, g, about)
        tp['fill']['_about'] += "; under 106 °C the temperature's own photo, above it the fuel's (warped)"
        tp['bar']['_about'] = "the bar as photographed at 106 °C, local to that level point"
        tp['reserve']['hidden'] = True
        json.dump(cfg, open(path, 'w', encoding='utf-8'), indent=1)
        print('temp ->', sys.argv[2])
    if what == 'labels10':                                    # trace_id7.py labels10 id7/Strada id7/Corsa
        def white(a):                                         # the glyphs only: pixels joined to a glyph's core
            v = a.min(-1)
            keep, ok_ = v >= 225, v >= 110
            for _ in range(200):
                g_ = keep.copy()
                g_[1:] |= keep[:-1]; g_[:-1] |= keep[1:]; g_[:, 1:] |= keep[:, :-1]; g_[:, :-1] |= keep[:, 1:]
                g_ &= ok_
                if (g_ == keep).all():
                    break
                keep = g_
            return np.where(keep, v, 0)
        lv = list(range(110, 256, 12))
        f10, t10 = load_x10('M8_fuel_x10'), load_x10('M8_temp_x10')
        def boxes(img10, origin, bxs):
            out = []
            for b_ in bxs:
                for l_ in vectorize_x10(img10, origin, b_, lv, white, blur=1.5):
                    out.append({'level': l_['level'], 'color': l_['color'],
                                'polys': [[[round(x / W, 6), round(y / H, 6)] for x, y in p_] for p_ in l_['polys']]})
            return sorted(out, key=lambda l_: l_['level'])
        # the photo's white glyphs (display px boxes, 1.5 px margin): 1, 1/2, 0, the pump and its arrow
        fuel_b = [(257.5, 428.5, 270.5, 449.5), (268.5, 490.5, 312.5, 511.5), (322.5, 535.5, 343.5, 557.5),
                  (208.5, 535.5, 229.5, 559.5), (230.5, 542.5, 241.5, 554.5)]
        temp_b = [(1611.5, 438, 1659, 458.5), (1596.5, 489.5, 1645, 510), (1565, 533.5, 1601.5, 554),
                  (1672, 522.5, 1696.5, 542.5), (1657.5, 540, 1694, 560.5)]
        FL, TL = boxes(f10, (190, 400), fuel_b), boxes(t10, (1550, 400), temp_b)
        for lo_ in sys.argv[2:]:
            path = os.path.join(ROOT, 'layouts', lo_, 'config.json')
            cfg = json.load(open(path, encoding='utf-8'))
            cfg['elements']['fuel']['labels']['layers'] = FL
            cfg['elements']['fuel']['labels']['_layers'] = ("1, 1/2, 0, the pump and its arrow vectorized on the 10x "
                                                            "enlargement (white = the min channel, levels 110..254)")
            if 'temp' in cfg['elements']:
                cfg['elements']['temp']['labels']['layers'] = TL
                cfg['elements']['temp']['labels']['_layers'] = "150, 100, 50, °C and the oil can, the same way"
            json.dump(cfg, open(path, 'w', encoding='utf-8'), indent=1)
            print('labels -> ' + lo_, len(FL), len(TL))
    if what == 'tempzones10':                                 # trace_id7.py tempzones10 id7/Corsa: hot (red) / cold (blue)
        img10 = load_x10('M8_temp_x10')
        lv = [8, 16, 24] + list(range(32, 256, 12))
        red = lambda a: np.where(a[..., 0] - a[..., 1:].max(-1) > np.minimum(25, a[..., 0] * 0.5), a[..., 0], 0)
        blue = lambda a: np.where(a[..., 2] - a[..., 0] > np.minimum(25, a[..., 2] * 0.5), a[..., 2], 0)
        parts = {'hot': vectorize_x10(img10, (1550, 400), (1628, 412, 1692, 452), lv, red),
                 'cold': vectorize_x10(img10, (1550, 400), (1550, 550, 1615, 584), lv, blue)}
        path = os.path.join(ROOT, 'layouts', sys.argv[2], 'config.json')
        cfg = json.load(open(path, encoding='utf-8'))
        t_ = cfg['elements']['temp']
        t_['overlay'] = {'_about': "the hot (red, top) and cold (blue, bottom) marks vectorized on the 10x enlargement "
                         "of the temperature gauge itself (M8_temp_x10.png, from x 1550, y 400): colour layers painted "
                         "flat, lowest first, over the mirrored drawing, in display coordinates (not mirrored)"}
        for k_, lay in parts.items():
            t_['overlay'][k_] = [{'level': l_['level'], 'color': l_['color'],
                                  'polys': [[[round(x / W, 6), round(y / H, 6)] for x, y in p_] for p_ in l_['polys']]} for l_ in lay]
        json.dump(cfg, open(path, 'w', encoding='utf-8'), indent=1)
        print('temp overlay -> ' + sys.argv[2], {k_: sum(len(l_['polys']) for l_ in v_) for k_, v_ in parts.items()})
    if what == 'reserve10':                                   # trace_id7.py reserve10 id7/Strada id7/Corsa
        img10 = np.asarray(Image.open(os.path.join(ROOT, 'design', 'refs', 'id7', 'M8_fuel_x10.png')).convert('RGB')).astype(np.float32)
        lay = vectorize_x10(img10, (190, 400), (298, 558, 366, 592), [8, 16, 24] + list(range(32, 250, 12)),
                            key=lambda a: np.where(a[..., 0] - a[..., 1:].max(-1) > np.minimum(25, a[..., 0] * 0.5), a[..., 0], 0))
        for lo_ in sys.argv[2:]:
            path = os.path.join(ROOT, 'layouts', lo_, 'config.json')
            cfg = json.load(open(path, encoding='utf-8'))
            r = cfg['elements']['fuel']['reserve']
            r['layers'] = [{'level': l_['level'], 'color': l_['color'],
                            'polys': [[[round(x / W, 6), round(y / H, 6)] for x, y in p] for p in l_['polys']]} for l_ in lay]
            r['_layers'] = ("the reserve mark vectorized on the 10x enlargement of the photo (M8_fuel_x10.png, display grid "
                            "x 10 from x 190, y 400): red-channel layers (red over green/blue by 25+) at levels 40..235, "
                            "each painted flat in its median colour, lowest first: the bright front edges, the darker top "
                            "face and its shading come out as on the photo")
            json.dump(cfg, open(path, 'w', encoding='utf-8'), indent=1)
            print('reserve layers -> ' + lo_, [len(l_['polys']) for l_ in lay])
    if what == 'blade':
        axis = float(sys.argv[2]) if len(sys.argv) > 2 else None
        img = load(axis)
        rows = trace_blade(img, 135, 568, 565, 345)
        json.dump({str(k): v for k, v in rows.items()}, sys.stdout)


if __name__ == '__main__':
    main()
