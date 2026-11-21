"""sync_editor.py EXPORT_DIR - copy the point editor's saved items (docs exported from its database) into the layouts' config (fuel: Strada + Corsa, temp: Corsa)."""
import json, glob, re, sys
src = sys.argv[1]
ed = {'fuel': [], 'temp': []}
for f in glob.glob(src + '/*.json'):
    d = json.load(open(f)); d = d.get('data', d)
    ed[d['zone']].append({k: d.get(k) for k in ('name', 'kind', 'pts', 'radii', 'width', 'color', 'note')})
key = lambda e: (0 if e['kind'] == 'linea' else 1, e['name'] if e['kind'] == 'linea' else int(re.sub(r'\D', '', e['name']) or 0))
for z in ed: ed[z].sort(key=key)
for lo, zs in (('Strada', ['fuel']), ('Corsa', ['fuel', 'temp'])):
    p = '/home/user/mgu-apps/cluster/layouts/id7/%s/config.json' % lo; c = json.load(open(p))
    for z in zs: c['elements'][z]['editor']['items'] = ed[z]
    json.dump(c, open(p, 'w'), indent=1)
print({z: len(v) for z, v in ed.items()})
