#!/usr/bin/env python3
"""Where the routed experts are read from, and reads like the engine's for the drive probes.

The engine reads every role (gate, up, down) of an expert with O_DIRECT, straight from the model's GGUF ranges
(widened to 4 KiB boundaries; GGML_MOE_EXPERT_GGUF_ALIAS_TSV indexes them) or from an expert pack when one is
given.  On 176 both read alike wherever the drive's region is healthy (2026-10-19); a file that reads far slower
than the others sits on degraded blocks and comes back once rewritten (copied and moved over the original).
The GGUF ranges come from the header alone (gguf_header.py, standard library only).
usage: expert_source.py index MODEL [PACK] DIR   -> DIR/experts.json (+ DIR/experts.tsv, the GGUF alias index),
                                                   unless DIR holds them for MODEL (and PACK) already, newer than
                                                   every file they index
       expert_source.py iolog DIR N SEED OUT [FILE]   -> fio iolog: N random experts (of FILE only, if given),
                                                         three role reads each, as O_DIRECT ranges
       expert_source.py facts MODEL [ALIAS]    -> what run-ssd-llama.sh derives settings from, as shell assignments:
                                                  expert_types (the experts' ggml types), alias_unaligned (entries
                                                  of ALIAS, a GGUF alias index, that do not start on a PAGE boundary)
       expert_source.py lane-args DIR          -> the arguments of test-ssd-vram-group --time for the experts DIR
                                                  indexes: --latent L --hidden H --up TYPE --down TYPE (the routed
                                                  experts' shape, and each role's most common type over the layers)
       expert_source.py dense MODEL HOMES OUT  -> OUT, the alias index of the dense matrices HOMES (tensor<TAB>ram|drive
                                                  lines, tools/adaptive DENSE_HOMES) puts on the drive, block by
                                                  block (blocks()), and as shell assignments what HOMES holds:
                                                  dense_drive (tensors), dense_drive_bytes, dense_ram, dense_ram_bytes,
                                                  and dense_stage_bytes, the engine's pinned stage of the drive's
                                                  reads (two halves of the most blocks a matrix has, each slot the
                                                  largest block widened by a PAGE); a drive row of a tensor that is
                                                  not a matrix with a block split stops it
experts.json: where the experts are (source, model, pack, alias, files), how many (roles = role slices, experts,
layers), what one costs to read: bytes_per_role and role_types per role (largest over the layers; the types every
layer uses), expert_bytes (the largest expert's three roles), expert_read_bytes (the largest expert's reads widened
to PAGE boundaries, what the engine reads and stages), total_bytes (every role slice), aligned_roles (slices that
start on a PAGE boundary); dense: every other tensor of the model, the dense weights (name, layer, type, bytes, rows,
row_bytes, n_blocks), in the order of the header.  n_blocks is how many equal blocks of whole rows a matrix is read
in from the drive when it is homed there (blocks()); none for a tensor that is not a matrix or has no such split.
"""
import json
import os
import random
import re
import struct
import sys

import gguf_header

PAGE = 4096
ROLE = re.compile(r'^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$')


def is_role(t):
    return bool(ROLE.match(t.name)) and len(t.shape) == 3


def gguf_tensors(model):
    """the routed expert tensors (one per layer and role) of every shard of MODEL, from the headers"""
    return [t for h in gguf_header.read_model(model) for t in h.tensors if is_role(t)]


def role_slice(tensors):
    """the largest role slice (one expert's part of a role tensor) of the routed expert tensors among `tensors`"""
    return max(t.nbytes // t.shape[-1] for t in tensors if is_role(t))


def blocks(rows, nbytes, largest):
    """The blocks a matrix of `rows` rows and `nbytes` bytes is read in from the drive: the fewest equal blocks of
    whole rows, each at most `largest` bytes (the largest role slice, the size of the transport's expert reads), n in
    [n0, 2 n0] with n0 = ceil(nbytes / largest) and n dividing the rows; None when there is none.  The transport
    places block i at i x the block's bytes in a target of exactly the matrix's size, so a short or padded last block
    cannot be."""
    n0 = max(1, -(-nbytes // largest))
    return next((n for n in range(n0, 2 * n0 + 1) if rows % n == 0), None)


def dense_facts(tensors, largest):
    out = []
    for t in tensors:
        m = re.match(r'^blk\.(\d+)\.', t.name)
        rows = 1
        for d in t.shape[1:]:
            rows *= d
        out.append({'name': t.name, 'layer': int(m.group(1)) if m else None, 'type': t.type_name, 'bytes': t.nbytes,
                    'rows': rows, 'row_bytes': t.nbytes // rows,
                    'n_blocks': blocks(rows, t.nbytes, largest) if len(t.shape) == 2 else None})
    return out


def gguf_entries(tensors):
    rows = []
    for t in tensors:
        n = t.shape[-1]
        nb = t.nbytes // n
        rows += [(t.name, e, nb, os.path.abspath(t.path), t.offset + e * nb) for e in range(n)]
    return rows


def widened(o, nb):
    return (o + nb + PAGE - 1) // PAGE * PAGE - o // PAGE * PAGE


def pack_entries(pack):
    with open(pack, 'rb') as f:
        magic, version, _, n, _ = struct.unpack('<16sIIQQ', f.read(40))
        assert magic.startswith(b'GGMLMOEPACKv1') and version == 1, f'{pack}: not an expert pack'
        raw = f.read(n * 152)
    path = os.path.abspath(pack)
    rows = []
    for i in range(n):
        name, e, _, off, nb = struct.unpack_from('<128siIQQ', raw, i * 152)
        rows.append((name.split(b'\0')[0].decode(), e, nb, path, off))
    return rows


def current(d, model, pack):
    """the index in D if it is MODEL's (and PACK's) and newer than every file it indexes, else None"""
    try:
        info = json.load(open(f'{d}/experts.json'))
        made = os.path.getmtime(f'{d}/experts.json')
        if info['model'] != os.path.abspath(model) or info['pack'] != (pack and os.path.abspath(pack)):
            return None
        if info['alias'] and not os.path.isfile(info['alias']):
            return None
        if any(os.path.getmtime(p) > made for p in {os.path.abspath(model), *info['files']}):
            return None
        if 'dense' not in info:     # an index from before the dense weights were indexed
            return None
        return info
    except (OSError, ValueError, KeyError, TypeError):
        return None


def index(model, pack, d):
    info = current(d, model, pack)
    if info is None:
        print(f'indexing the experts of {pack or model} into {d}', file=sys.stderr)
        info = build(model, pack, d)
    print(json.dumps({k: v for k, v in info.items() if k not in ('files', 'dense')}))


def build(model, pack, d):
    os.makedirs(d, exist_ok=True)
    every = [t for h in gguf_header.read_model(model) for t in h.tensors]
    tensors = [t for t in every if is_role(t)]
    rows = pack_entries(pack) if pack else gguf_entries(tensors)
    assert rows, f'no routed expert tensors in {model}'
    if not pack:
        with open(f'{d}/experts.tsv.tmp', 'w') as f:
            f.write('tensor\texpert_idx\tnbytes\tsource_path\toffset\n')
            f.writelines(f'{t}\t{e}\t{nb}\t{p}\t{o}\n' for t, e, nb, p, o in rows)
        os.replace(f'{d}/experts.tsv.tmp', f'{d}/experts.tsv')
    per_role, types, expert, read = {}, {}, {}, {}
    for t in tensors:
        types.setdefault(ROLE.match(t.name).group(2), set()).add(t.type_name)
    for t, e, nb, _, o in rows:
        m = ROLE.match(t)
        per_role[m.group(2)] = max(per_role.get(m.group(2), 0), nb)
        key = (int(m.group(1)), e)
        expert[key] = expert.get(key, 0) + nb
        read[key] = read.get(key, 0) + widened(o, nb)
    info = {'source': 'pack' if pack else 'gguf', 'model': os.path.abspath(model), 'pack': pack and os.path.abspath(pack),
            'alias': None if pack else os.path.abspath(f'{d}/experts.tsv'), 'roles': len(rows),
            'experts': 1 + max(e for _, e, _, _, _ in rows), 'layers': len({layer for layer, _ in expert}),
            'bytes_per_role': dict(sorted(per_role.items())), 'role_types': {r: sorted(v) for r, v in sorted(types.items())},
            'expert_bytes': max(expert.values()), 'expert_read_bytes': max(read.values()),
            'total_bytes': sum(nb for _, _, nb, _, _ in rows),
            'files': sorted({p for *_, p, _ in rows}), 'aligned_roles': sum(1 for *_, o in rows if o % PAGE == 0),
            'dense': dense_facts([t for t in every if not is_role(t)], role_slice(every))}
    with open(f'{d}/experts.json.tmp', 'w') as f:
        json.dump(info, f, indent=1)
    os.replace(f'{d}/experts.json.tmp', f'{d}/experts.json')
    return info


def load(d):
    info = json.load(open(f'{d}/experts.json'))
    if info['pack']:
        return info, pack_entries(info['pack'])
    rows = []
    with open(info['alias']) as f:
        next(f)
        for line in f:
            t, e, nb, p, o = line.rstrip('\n').split('\t')
            rows.append((t, int(e), int(nb), p, int(o)))
    return info, rows


def iolog(d, n, seed, out, only=None):
    info, rows = load(d)
    where = {(t, e): (p, o, nb) for t, e, nb, p, o in rows}
    experts = sorted({(int(ROLE.match(t).group(1)), e) for t, e, _, p, _ in rows if not only or p == os.path.abspath(only)})
    assert experts, f'no experts in {only}'
    rng = random.Random(seed)
    lines, files = [], set()
    for layer, e in (rng.choice(experts) for _ in range(n)):
        for role in ('gate', 'up', 'down'):
            p, o, nb = where[(f'blk.{layer}.ffn_{role}_exps.weight', e)]
            start, end = o // PAGE * PAGE, (o + nb + PAGE - 1) // PAGE * PAGE
            lines.append(f'{p} read {start} {end - start}\n')
            files.add(p)
    with open(out, 'w') as f:
        f.write('fio version 2 iolog\n' + ''.join(f'{p} add\n{p} open\n' for p in sorted(files)))
        f.writelines(lines)
        f.write(''.join(f'{p} close\n' for p in sorted(files)))


def facts(model, alias=None):
    types = sorted({t.type_name for t in gguf_tensors(model)})
    unaligned = 0
    if alias:
        with open(alias) as f:
            next(f)
            unaligned = sum(1 for line in f if int(line.rsplit('\t', 1)[1]) % PAGE)
    print(f"expert_types='{' '.join(types)}'\nalias_unaligned={unaligned}")


def lane_args(d):
    """The probe's per-expert lane timing at this model's expert shape and types (B4): up is [latent, hidden, E]."""
    info = json.load(open(f'{d}/experts.json'))
    tensors = gguf_tensors(info['model'])
    ups = [t for t in tensors if ROLE.match(t.name).group(2) == 'up']
    downs = [t for t in tensors if ROLE.match(t.name).group(2) == 'down']
    assert ups and downs, f'no routed up and down tensors in {info["model"]}'

    def common(ts):
        names = [t.type_name for t in ts]
        return max(sorted(set(names)), key=names.count)
    print(f'--latent {ups[0].shape[0]} --hidden {ups[0].shape[1]} --up {common(ups)} --down {common(downs)}')


def dense(model, homes, out):
    every = [t for h in gguf_header.read_model(model) for t in h.tensors]
    tensors = {t.name: t for t in every if not is_role(t)}
    largest = role_slice(every)
    facts = {x['name']: x for x in dense_facts(tensors.values(), largest)}
    rows, seen, totals, stage = [], set(), {'drive': [0, 0], 'ram': [0, 0]}, [0, 0]
    for n, line in enumerate(open(homes), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        name, home = (line.split('\t') + [''])[:2]
        if home not in totals or name in seen:
            sys.exit(f'{homes}:{n}: not a tensor<TAB>ram|drive line of a tensor not named before: {line}')
        if name not in facts:
            sys.exit(f'{homes}:{n}: {name} is not a dense weight of {model}')
        seen.add(name)
        x, t = facts[name], tensors[name]
        totals[home][0] += 1
        totals[home][1] += x['bytes']
        if home != 'drive':
            continue
        if not x['n_blocks']:
            sys.exit(f'{homes}:{n}: {name} ({x["rows"]} rows of {x["row_bytes"]} B) has no split into equal blocks of '
                     f'whole rows of at most {largest} B: it cannot be homed on the drive')
        bb = x['bytes'] // x['n_blocks']
        rows += [(name, i, bb, os.path.abspath(t.path), t.offset + i * bb) for i in range(x['n_blocks'])]
        stage = [max(stage[0], x['n_blocks']), max(stage[1], (bb + 2 * PAGE - 1) // PAGE * PAGE)]
    with open(f'{out}.tmp', 'w') as f:
        f.write('tensor\texpert_idx\tnbytes\tsource_path\toffset\n')
        f.writelines(f'{t}\t{e}\t{nb}\t{p}\t{o}\n' for t, e, nb, p, o in rows)
    os.replace(f'{out}.tmp', out)
    print(f"dense_drive={totals['drive'][0]}\ndense_drive_bytes={totals['drive'][1]}\n"
          f"dense_ram={totals['ram'][0]}\ndense_ram_bytes={totals['ram'][1]}\ndense_stage_bytes={2 * stage[0] * stage[1]}")


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[:1] == ['index'] and len(a) in (3, 4):
        index(a[1], a[2] if len(a) == 4 else None, a[-1])
    elif a[:1] == ['facts'] and len(a) in (2, 3):
        facts(a[1], a[2] if len(a) == 3 else None)
    elif a[:1] == ['lane-args'] and len(a) == 2:
        lane_args(a[1])
    elif a[:1] == ['dense'] and len(a) == 4:
        dense(a[1], a[2], a[3])
    elif a[:1] == ['iolog'] and len(a) in (5, 6):
        iolog(a[1], int(a[2]), int(a[3]), a[4], a[5] if len(a) == 6 else None)
    else:
        sys.exit(__doc__)
