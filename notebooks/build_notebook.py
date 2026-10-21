"""Construit le .ipynb à partir du script à cellules (« # %% » et « # %% [markdown] »)."""
import sys

import nbformat


def cells_from_script(path):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if kind == "md":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            cells.append(nbformat.v4.new_code_cell(text))

    for line in open(path, encoding="utf-8").read().splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("md" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    nb = nbformat.v4.new_notebook()
    nb.cells = cells_from_script(src)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbformat.write(nb, dst)
    print(f"{dst} : {len(nb.cells)} cellules")
