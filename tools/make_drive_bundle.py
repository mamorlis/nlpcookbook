#!/usr/bin/env python3
"""Google Drive に置くためのまとめを dist/ に作る。

    python tools/make_drive_bundle.py

- dist/nlpcookbook_notebooks.zip : notebooks/ 以下の全ノートブック
- dist/00_はじめに.ipynb         : Drive 上で zip を展開するためのノートブック

この 2 つを Drive の「Colab Notebooks/nlpcookbook/」に置き、00_はじめに.ipynb を Colab で開いて実行する。
"""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
FOLDER = "/content/drive/MyDrive/Colab Notebooks/nlpcookbook"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}


def main():
    DIST.mkdir(exist_ok=True)
    notebooks = sorted((ROOT / "notebooks").glob("*/*.ipynb"))
    with zipfile.ZipFile(DIST / "nlpcookbook_notebooks.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for nb in notebooks:
            z.write(nb, nb.relative_to(ROOT / "notebooks"))

    chapters = {}
    for nb in notebooks:
        meta = json.loads(nb.read_text(encoding="utf-8"))
        title = "".join(meta["cells"][0]["source"]).split("\n")[0].lstrip("# ")
        chapters.setdefault(nb.parent.name, []).append((meta["metadata"]["nlpcookbook"]["id"], title, nb.name))
    index = []
    for chapter, items in chapters.items():
        index.append(f"\n### {chapter}\n")
        index += [f"- {i} {t} (`{chapter}/{name}`)" for i, t, name in items]

    cells = [
        md(f"# NLP クックブック: はじめに\n\n自然言語処理の小ネタ集 (全 {len(notebooks)} レシピ) を Google Drive に展開します。\n\n"
           "1. 下のセルを上から順に実行する (Drive へのアクセス許可を求められます)\n"
           "2. Drive の `Colab Notebooks/nlpcookbook/` に章ごとのフォルダができる\n"
           "3. 好きなノートブックを開いて実行する\n\n"
           "最新版は GitHub にあります: https://github.com/mamorlis/nlpcookbook"),
        code("from google.colab import drive\n\ndrive.mount(\"/content/drive\")"),
        code(f"import zipfile\nfrom pathlib import Path\n\nfolder = Path(\"{FOLDER}\")\n"
             "with zipfile.ZipFile(folder / \"nlpcookbook_notebooks.zip\") as z:\n"
             "    z.extractall(folder)   # 同じ名前のノートブックは上書きされる\n"
             "for d in sorted(p for p in folder.iterdir() if p.is_dir()):\n"
             "    print(d.name, len(list(d.glob(\"*.ipynb\"))), \"レシピ\")"),
        md("## レシピ一覧\n" + "\n".join(index)),
    ]
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"colab": {"provenance": [], "toc_visible": True},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}}
    (DIST / "00_はじめに.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    size = (DIST / "nlpcookbook_notebooks.zip").stat().st_size
    print(f"{len(notebooks)} notebooks → dist/nlpcookbook_notebooks.zip ({size / 1e6:.1f} MB), dist/00_はじめに.ipynb")


if __name__ == "__main__":
    main()
