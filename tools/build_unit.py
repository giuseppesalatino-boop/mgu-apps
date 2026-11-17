#!/usr/bin/env python3
"""build_unit.py - the MGU Store's build of one app of this collection on the head unit (mgu-repo.json).

    python3 tools/build_unit.py <id> <app folder>

The Store runs it in a copy of this repository, with the oap-sdk it generates from a BMW app on the
unit in vendor/idrive7-apps/oap-sdk, Python 3.12 and Node 8, and no network. It puts the app folder
(true-time/app, ecu-live-data/app, dpf-monitoring/app, ...) into the shared kit as apps/<id> (a copy:
the links the apps' tools/setup.js make on a PC aren't in the repository's tarball), with <id> in its
app.json (the Store installs a second app of the same id as <id>2), and runs the kit's build:
vendor/idrive7-apps/dist/<id>@<version>.tgz.
Works on a PC too (with an SDK or the emulator placeholder in vendor/idrive7-apps/oap-sdk).
"""
import json
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.join(ROOT, 'vendor', 'idrive7-apps')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('id')
    parser.add_argument('app_dir')
    parser.add_argument('--language', default='en')
    parser.add_argument('--without-sdk', action='store_true',
                        help='build a package that MGU Deploy supplies with its on-unit SDK')
    args = parser.parse_args()
    app_id = args.id
    src = os.path.abspath(args.app_dir)
    if not os.path.isfile(os.path.join(src, 'app.json')):
        sys.exit('build_unit: no app.json in ' + src)
    dst = os.path.join(KIT, 'apps', app_id)
    if os.path.realpath(dst) != os.path.realpath(src):
        if os.path.islink(dst):
            os.remove(dst)
        elif os.path.isdir(dst):
            shutil.rmtree(dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(src, dst)
    manifest = os.path.join(dst, 'app.json')
    with open(manifest, encoding='utf-8') as f:
        spec = json.load(f)
    if spec.get('id') != app_id:
        spec['id'] = app_id
        with open(manifest, 'w', encoding='utf-8') as f:
            json.dump(spec, f, indent=2)
    build = [sys.executable, os.path.join(KIT, 'kit', 'tools', 'build.py'), '--language', args.language]
    if args.without_sdk:
        licenses = [os.path.join(src, '..', 'LICENSE'), os.path.join(ROOT, 'LICENSE'), os.path.join(KIT, 'LICENSE')]
        if spec.get('vendor') == 'boostline':
            licenses.append(os.path.join(ROOT, 'vendor', 'boostline', 'LICENSE'))
        with open(os.path.join(dst, 'NOTICE'), 'w', encoding='utf-8') as notice:
            for license_path in licenses:
                with open(license_path, encoding='utf-8') as license_file:
                    notice.write(license_file.read().rstrip() + '\n\n')
        build.append('--without-sdk')
    subprocess.check_call(build + [app_id], cwd=KIT)
    sys.path.insert(0, os.path.join(KIT, 'kit', 'tools'))
    from localize import package_version
    version = spec['version'] if args.without_sdk else package_version(spec['version'], args.language)
    out = os.path.join(KIT, 'dist', '%s@%s.tgz' % (app_id, version))
    if not os.path.exists(out):
        sys.exit('build_unit: no ' + out)
    print('build_unit: ' + os.path.relpath(out, ROOT))


if __name__ == '__main__':
    main()
