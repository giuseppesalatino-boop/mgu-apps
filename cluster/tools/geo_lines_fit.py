#!/usr/bin/env python3
"""geo_lines_fit.py - the fuel gauge's lines as straight lines and circular fillets only (ID7 Strada / Corsa).

    python3 cluster/tools/geo_lines_fit.py

A (the thin outer line) and C (the fill's lit edge) fitted on BMW's M8 photo at 10x (refs/id7/M8_fuel_x10.png, not in
git): vertices by Douglas-Peucker, then vertices and fillet radii by least squares (Nelder-Mead); B (the thick line) is
A's offset inwards by the measured distance, sharp where a fillet is tighter than the offset. Writes
design/out/{carburante,temperatura}_linee.svg, design/out/linee_geometriche.json and A / B into the layouts' fuel and
temp (the temperature gauge: the same drawing, mirrored about x 952.5). A design tool (Python, PC only).
"""

import sys, json, numpy as np
sys.path.insert(0, '/home/user/mgu-apps/cluster/tools'); import trace_id7 as t

def dp(P, tol):
    a, b = P[0], P[-1]; d = b - a; nd = np.linalg.norm(d) + 1e-9
    dist = np.abs(d[0] * (P[:, 1] - a[1]) - d[1] * (P[:, 0] - a[0])) / nd
    k = int(np.argmax(dist))
    if dist[k] > tol and 0 < k < len(P) - 1:
        return dp(P[:k + 1], tol)[:-1] + dp(P[k:], tol)
    return [a, b]

def build(V, R):
    """V vertices (n x 2), R radii for interior vertices -> list of primitives and dense samples."""
    n = len(V); prims = []; pts = []
    T = [(V[0], None)]
    fil = []
    for i in range(1, n - 1):
        u = V[i] - V[i - 1]; Lu = np.linalg.norm(u); u /= Lu
        w = V[i + 1] - V[i]; Lw = np.linalg.norm(w); w /= Lw
        th = np.arccos(np.clip(u @ w, -1, 1))
        if th < 1e-4:
            fil.append(None); continue
        tmax = min(Lu if i == 1 else Lu / 2, Lw if i == n - 2 else Lw / 2)
        r = max(R[i - 1], 3.0)
        tl = r * np.tan(th / 2)
        if tl > tmax: tl = tmax; r = tl / np.tan(th / 2)
        T1, T2 = V[i] - u * tl, V[i] + w * tl
        cr = u[0] * w[1] - u[1] * w[0]; nrm = np.array([-u[1], u[0]]) * np.sign(cr)
        C = T1 + nrm * r
        fil.append((T1, T2, C, r, np.sign(cr)))
    cur = V[0]
    for i in range(1, n - 1):
        f = fil[i - 1]
        if f is None: continue
        T1, T2, C, r, s = f
        prims.append(('L', cur, T1)); prims.append(('A', T1, T2, C, r, s)); cur = T2
    prims.append(('L', cur, V[-1]))
    for p in prims:
        if p[0] == 'L':
            a, b = p[1], p[2]; m = max(2, int(np.linalg.norm(b - a) / 0.2))
            pts.append(a + (b - a) * np.linspace(0, 1, m)[:, None])
        else:
            _, a, b, C, r, s = p
            a0 = np.arctan2(*(a - C)[::-1]); a1 = np.arctan2(*(b - C)[::-1])
            da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
            m = max(2, int(abs(da) * r / 0.2))
            ang = a0 + da * np.linspace(0, 1, m)
            pts.append(C + r * np.c_[np.cos(ang), np.sin(ang)])
    return prims, np.vstack(pts)

def err(P, S):
    d = np.sqrt(((P[:, None, :] - S[None, :, :]) ** 2).sum(-1))
    return d.min(1), d.min(0)

def cost(x, n, P):
    V = x[:2 * n].reshape(n, 2); R = x[2 * n:]
    _, S = build(V, R)
    a, b = err(P, S)
    return np.mean(a ** 2) + np.mean(b ** 2) + 0.2 * (a.max() ** 2)

def nelder(f, x0, step, it=4000):
    n = len(x0); X = [x0] + [x0 + np.eye(n)[i] * step[i] for i in range(n)]; F = [f(x) for x in X]
    for _ in range(it):
        o = np.argsort(F); X = [X[i] for i in o]; F = [F[i] for i in o]
        c = np.mean(X[:-1], 0); xr = c + (c - X[-1]); fr = f(xr)
        if fr < F[0]:
            xe = c + 2 * (c - X[-1]); fe = f(xe)
            X[-1], F[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < F[-2]: X[-1], F[-1] = xr, fr
        else:
            xc = c + 0.5 * (X[-1] - c); fc = f(xc)
            if fc < F[-1]: X[-1], F[-1] = xc, fc
            else:
                X = [X[0]] + [X[0] + 0.5 * (x - X[0]) for x in X[1:]]; F = [F[0]] + [f(x) for x in X[1:]]
        if abs(F[-1] - F[0]) < 1e-7: break
    return X[int(np.argmin(F))]

def fit(P, tol):
    P = P[::2]
    V = np.array(dp(P, tol), float); n = len(V)
    R = np.full(n - 2, 8.0)
    x = np.r_[V.ravel(), R]
    step = np.r_[np.full(2 * n, 1.0), np.full(n - 2, 6.0)]
    for _ in range(3):
        x = nelder(lambda q: cost(q, n, P), x, step, 3000); step = step * 0.4
    V = x[:2 * n].reshape(n, 2); R = np.maximum(x[2 * n:], 0)
    prims, S = build(V, R)
    a, b = err(P, S)
    return V, R, prims, max(a.max(), b.max()), a.mean()

def ridges(img, seeds):
    _, ln = t.gauge10(img, (190, 400), seeds, [], (232, 428, 312, 456))
    return {k: ln[k][0] for k in ('A', 'B', 'C')}, {k: ln[k][1] for k in ('A', 'B', 'C')}

seeds = {k: np.array(v, float) for k, v in t.SEEDS_FUEL.items()}
rid, wid = ridges(t.load_x10('M8_fuel_x10'), seeds)
PA, PB = rid['A'], rid['B']
dBA = np.array([np.min(np.linalg.norm(PA - q, axis=1)) for q in PB])
PB = PB[dBA > 3.5]                                    # B's trace without the part that slid onto A
d = float(np.median(dBA[(dBA > 3.5)]))
def fitn(P, tol):
    V, R, prims, emax, emean = fit(P, tol)
    return V, np.maximum(R, 3.0), emax, emean
VA, RA, e1, e2 = fitn(PA, 3.0)
VC, RC, e3, e4 = fitn(rid['C'], 3.0)
print('A vertici', len(VA), 'raggi', np.round(RA, 1).tolist(), 'scarto max %.2f medio %.2f' % (e1, e2))
print('C vertici', len(VC), 'raggi', np.round(RC, 1).tolist(), 'scarto max %.2f medio %.2f' % (e3, e4))
primsA, SA = build(VA, RA)
# B = offset of A towards the inside, the end segments extended, cut at B's photo ends
Vx = VA.copy()
Vx[0] = VA[0] + (VA[0] - VA[1]) / np.linalg.norm(VA[0] - VA[1]) * 15
pr, _ = build(Vx, RA)
def offset(prims, dd):
    out = []
    for p in prims:
        if p[0] == 'L':
            a, b = p[1], p[2]; u = (b - a) / np.linalg.norm(b - a); n = np.array([-u[1], u[0]])
            out.append(['L', a + n * dd, b + n * dd])
        else:
            _, a, b, C, r, s = p; ra = r - s * dd
            out.append(['A', C + (a - C) * ra / r, C + (b - C) * ra / r, C, ra, s])
    fixed = []                                            # an inner offset of a fillet tighter than the offset:
    for i, p in enumerate(out):                           # a sharp corner where the two offset lines meet
        if p[0] == 'A' and p[4] <= 0.05:
            l0, l1 = fixed[-1], out[i + 1]
            a0, u0 = l0[1], l0[2] - l0[1]; a1, u1 = l1[1], l1[2] - l1[1]
            tt = np.linalg.solve(np.c_[u0, -u1], a1 - a0)[0]
            X = a0 + u0 * tt
            fixed[-1] = ['L', l0[1], X]; out[i + 1] = ['L', X, l1[2]]
            continue
        fixed.append(p)
    return fixed
def samples(o):
    S = []
    for p in o:
        if p[0] == 'L': S.append(p[1] + (p[2] - p[1]) * np.linspace(0, 1, 100)[:, None])
        else:
            _, a, b, C, r, s = p; a0 = np.arctan2(*(a - C)[::-1]); a1 = np.arctan2(*(b - C)[::-1])
            da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi; ang = a0 + da * np.linspace(0, 1, 100)
            S.append(C + r * np.c_[np.cos(ang), np.sin(ang)])
    return S
best = min((err(PB[::3], np.vstack(samples(offset(pr, sg * d))))[0].mean(), sg) for sg in (1, -1))
B = offset(pr, best[1] * d)
print('B = offset di A a %.2f px, scarto medio dalla foto %.2f px' % (d, best[0]))
# cut B at its photo ends: drop primitives before / after, cut inside the one that holds the end
def cut(o, P0, P1):
    S = samples(o)
    def where(P):
        k = min(range(len(o)), key=lambda i: np.min(np.linalg.norm(S[i] - P, axis=1)))
        j = int(np.argmin(np.linalg.norm(S[k] - P, axis=1))); return k, S[k][j]
    k0, q0 = where(P0); k1, q1 = where(P1)
    o = [list(p) for p in o[k0:k1 + 1]]
    o[0][1] = q0; o[-1][2] = q1
    return o
B = cut(B, PB[0], PB[-1])
primsC, _ = build(VC, RC)
geo = {'A': primsA, 'B': B, 'C': primsC}
W = {k: float(wid[k]) for k in ('A', 'B', 'C')}
def svgpath(o, mx=None):
    f = (lambda q: (2 * mx - q[0], q[1])) if mx else (lambda q: (q[0], q[1]))
    s = 'M %.3f %.3f' % f(o[0][1])
    for p in o:
        if p[0] == 'L': s += ' L %.3f %.3f' % f(p[2])
        else:
            sw = (1 if p[5] > 0 else 0) if not mx else (0 if p[5] > 0 else 1)
            s += ' A %.3f %.3f 0 0 %d %.3f %.3f' % (p[4], p[4], sw, *f(p[2]))
    return s
for name, x0, mx in (('carburante', 190, None), ('temperatura', 1550, 952.5)):
    L = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="%d 400 180 200" width="720" height="800">' % x0,
         '<rect x="%d" y="400" width="180" height="200" fill="#000"/>' % x0]
    for k in ('A', 'B', 'C'):
        L.append('<path fill="none" stroke="#fff" stroke-width="%.2f" stroke-linecap="round" d="%s"/>' % (W[k], svgpath(geo[k], mx)))
    L.append('</svg>')
    open('/home/user/mgu-apps/cluster/design/out/%s_linee.svg' % name, 'w').write('\n'.join(L))
ser = lambda o: [[p[0]] + [np.round(np.asarray(q, float), 3).tolist() if isinstance(q, np.ndarray) else round(float(q), 3) for q in p[1:]] for p in o]
json.dump({'mirror_axis': 952.5, 'widths': W, 'B_offset_from_A': d, 'A': ser(primsA), 'B': ser(B), 'C': ser(primsC)},
          open('/home/user/mgu-apps/cluster/design/out/linee_geometriche.json', 'w'), indent=1)

g = json.load(open('/home/user/mgu-apps/cluster/design/out/linee_geometriche.json'))
def sample(o, step=0.25):
    S = []
    for p in o:
        a = np.array(p[1], float); b = np.array(p[2], float)
        if p[0] == 'L':
            m = max(2, int(np.linalg.norm(b - a) / step)); S.append(a + (b - a) * np.linspace(0, 1, m)[:, None])
        else:
            C = np.array(p[3], float); r = p[4]
            a0 = np.arctan2(*(a - C)[::-1]); a1 = np.arctan2(*(b - C)[::-1])
            da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi; m = max(2, int(abs(da) * r / step))
            ang = a0 + da * np.linspace(0, 1, m); S.append(C + r * np.c_[np.cos(ang), np.sin(ang)])
    P = np.vstack(S); keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-3]
    return [[round(x / 1920, 6), round(y / 720, 6)] for x, y in P[keep]]
for lo, keys in (('Strada', ['fuel']), ('Corsa', ['fuel', 'temp'])):
    path = '/home/user/mgu-apps/cluster/layouts/id7/%s/config.json' % lo
    c = json.load(open(path))
    for k in keys:
        el = c['elements'][k]
        for n in ('A', 'B'):
            el[n] = sample(g[n]); el[n + '_width'] = round(g['widths'][n], 2)
        el['B_corner'] = el['B'][0]
        el['geometry'] = {'_about': "A and B as straight lines and circular fillets only (fitted to BMW's photo at 10x): "
                          "A measured, B its offset %.2f px inwards; fuel and temperature the same drawing (the temperature "
                          "mirrored about x 952.5). A/B above are these primitives sampled every 0.25 px." % g['B_offset_from_A'],
                          'A': g['A'], 'B': g['B'], 'C': g['C']}
    json.dump(c, open(path, 'w'), indent=1)
    print(lo, 'ok')
