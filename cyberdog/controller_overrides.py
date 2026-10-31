"""Write a copy of controllers.yaml with Cyberdog's local overrides applied.

Usage: controller_overrides BASE OUTPUT OVERRIDE...

The original file is never modified. Override keys may be dotted
(reference.position_rate_limit) and must already exist in the base file with
the same shape, so a typo cannot silently add a parameter nobody reads.
"""
import sys

import yaml


def merge(base, overrides):
    for node, section in overrides.items():
        if node not in base:
            raise KeyError(f'{node} is not in the base controllers file')
        target = base[node]['ros__parameters']
        for key, value in section['ros__parameters'].items():
            *path, leaf = key.split('.')
            parent = target
            for part in path:
                parent = parent[part]
            if leaf not in parent:
                raise KeyError(f'{node}: {key} is not in the base controllers file')
            old = parent[leaf]
            if isinstance(old, list) != isinstance(value, list) or (
                    isinstance(old, list) and len(old) != len(value)):
                raise ValueError(f'{node}: {key} does not match the base shape')
            parent[leaf] = value
    return base


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 3:
        raise SystemExit(__doc__)
    with open(argv[0]) as f:
        base = yaml.safe_load(f)
    for path in argv[2:]:
        with open(path) as f:
            merge(base, yaml.safe_load(f))
    with open(argv[1], 'w') as f:
        yaml.safe_dump(base, f, sort_keys=False)
