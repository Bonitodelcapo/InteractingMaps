"""Writes experiments_walkthrough.ipynb from walkthrough/cells/cells_<NN>_<ID>.py.

The notebook is generated: edit the cell files, then rebuild. From the repo root:

    python walkthrough/build_nb.py                     # every file -> experiments_walkthrough.ipynb
    python walkthrough/build_nb.py OUT.ipynb A2 A3 A4  # core + those files -> OUT (a test build)

Each cells file holds CELLS = [(kind, tags, src)]; NN orders the sections.
"""
import glob
import importlib.util
import os
import re
import sys

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(os.path.dirname(HERE), 'experiments_walkthrough.ipynb')


def overview_md(ids, path=os.path.join(os.path.dirname(HERE), 'experiments.md')):
    """experiments.md as the notebook's overview; the IDs with a section link to it."""
    with open(path, encoding='utf-8') as f:
        text = f.read()
    text = re.sub(r'^\| ([A-E]\d+) \|', lambda m: (f'| [{m[1]}](#{m[1]}) |' if m[1] in ids
                                                    else m[0]), text, flags=re.M)
    return '<a id="overview"></a>\n\n' + text


def anchored(src):
    """'## A3 · title' -> anchor + title + back link to the overview."""
    m = re.match(r'## ([A-E]\d+) ·', src)
    if not m:
        return src
    first, _, rest = src.partition('\n')
    return f'<a id="{m[1]}"></a>\n\n{first} · [↑ overview](#overview)\n{rest}'


def build(out=REPO, only=None):
    cells = []
    for path in sorted(glob.glob(os.path.join(HERE, 'cells', 'cells_*.py'))):
        fid = os.path.basename(path)[:-3].split('_', 2)[2]
        if only and fid not in ('core', *only):
            continue
        spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for kind, tags, src in mod.CELLS:
            src = src.strip('\n')
            c = (nbformat.v4.new_markdown_cell(anchored(src)) if kind == 'md'
                 else nbformat.v4.new_code_cell(src))
            c.metadata['tags'] = list(tags)
            cells.append(c)
    ids = {m[1] for c in cells if c.cell_type == 'markdown'
           for m in [re.search(r'<a id="([A-E]\d+)"></a>', c.source)] if m}
    for c in cells:
        if 'overview' in c.metadata['tags']:
            c.source = overview_md(ids)
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nbformat.write(nb, out)
    print(f'{len(cells)} cells -> {out}')


if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else build()
