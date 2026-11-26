"""Writes experiments_walkthrough.ipynb from walkthrough/cells/cells_<NN>_<ID>.py.

The notebook is generated: edit the cell files, then rebuild. From the repo root:

    python walkthrough/build_nb.py                     # every file -> experiments_walkthrough.ipynb
    python walkthrough/build_nb.py OUT.ipynb A2 A3 A4  # core + those files -> OUT (a test build)

Each cells file holds CELLS = [(kind, tags, src)]; NN orders the sections.
"""
import glob
import importlib.util
import os
import sys

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(os.path.dirname(HERE), 'experiments_walkthrough.ipynb')


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
            c = (nbformat.v4.new_markdown_cell(src) if kind == 'md'
                 else nbformat.v4.new_code_cell(src))
            c.metadata['tags'] = list(tags)
            cells.append(c)
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nbformat.write(nb, out)
    print(f'{len(cells)} cells -> {out}')


if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else build()
