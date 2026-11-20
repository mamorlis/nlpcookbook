#!/usr/bin/env python3
"""recipes/*.md から Colab 用ノートブック (notebooks/**/*.ipynb) と README の目次を生成する。

    python tools/build.py              # ノートブックと目次を生成 (実行はしない)
    python tools/build.py --run        # 未実行・変更ありのレシピを実行して出力を埋め込む
    python tools/build.py --run -k bm25  # スラッグに bm25 を含むレシピだけ実行
    python tools/build.py --run --force  # すべて再実行

レシピの書式は CONTRIBUTING.md を参照。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "recipes"
OUT = ROOT / "notebooks"
REPO = "mamorlis/nlpcookbook"
BRANCH = "main"

HEAD_RE = re.compile(r"^## (.+?)\s*\{([a-z0-9_]+)((?:\s+[a-z]+)*)\}\s*$")
FENCE_RE = re.compile(r"^```python\s*$")


@dataclass
class Recipe:
    chapter_no: int | str  # 本文の章は整数、付録は "A" など
    chapter_dir: str
    chapter_title: str
    no: int
    title: str
    slug: str
    tags: set[str]
    cells: list[tuple[str, str]] = field(default_factory=list)  # (kind, source)

    @property
    def ident(self) -> str:
        if isinstance(self.chapter_no, str):  # 付録は A-01 の形
            return f"{self.chapter_no}-{self.no:02d}"
        return f"{self.chapter_no:02d}-{self.no:02d}"

    @property
    def relpath(self) -> str:
        return f"notebooks/{self.chapter_dir}/{self.slug}.ipynb"

    @property
    def github_url(self) -> str:
        # README の表は絶対 URL にする (相対リンクは、開いている URL によっては github.com/<owner>/notebooks/... に解決されて切れる)
        return f"https://github.com/{REPO}/blob/{BRANCH}/{self.relpath}"

    @property
    def colab_url(self) -> str:
        return f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{self.relpath}"

    @property
    def digest(self) -> str:
        h = hashlib.sha256()
        for kind, src in self.cells:
            if kind == "code":
                h.update(src.encode())
                h.update(b"\0")
        return h.hexdigest()[:16]


def split_cells(lines: list[str]) -> list[tuple[str, str]]:
    cells, buf, in_code = [], [], False

    def flush(kind: str) -> None:
        text = "\n".join(buf).strip("\n")
        if text.strip():
            cells.append((kind, text))
        buf.clear()

    for line in lines:
        if not in_code and FENCE_RE.match(line):
            flush("markdown")
            in_code = True
        elif in_code and line.rstrip() == "```":
            flush("code")
            in_code = False
        else:
            buf.append(line)
    if in_code:
        raise ValueError("閉じていないコードブロックがあります")
    flush("markdown")
    return cells


def parse_chapter(path: Path) -> tuple[str, str, list[Recipe]]:
    m = re.match(r"(\d+|[A-Z])_", path.stem)
    if not m:
        raise ValueError(f"{path.name}: ファイル名は NN_name.md （付録は A_name.md）の形式にしてください")
    chapter_no = int(m.group(1)) if m.group(1).isdigit() else m.group(1)
    lines = path.read_text(encoding="utf-8").split("\n")
    if not lines[0].startswith("# "):
        raise ValueError(f"{path.name}: 1 行目は '# 章タイトル' にしてください")
    chapter_title = lines[0][2:].strip()

    intro: list[str] = []
    recipes: list[Recipe] = []
    body: list[str] | None = None
    in_code = False
    for line in lines[1:]:
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and line.startswith("## "):
            m = HEAD_RE.match(line)
            if not m:
                raise ValueError(f"{path.name}: 見出しの書式が不正です: {line}")
            if body is not None:
                recipes[-1].cells = split_cells(body)
            body = []
            recipes.append(
                Recipe(chapter_no, path.stem, chapter_title, len(recipes) + 1,
                       m.group(1), m.group(2), set(m.group(3).split()))
            )
        elif body is None:
            intro.append(line)
        else:
            body.append(line)
    if body is not None:
        recipes[-1].cells = split_cells(body)
    return chapter_title, "\n".join(intro).strip(), recipes


def md_cell(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src}


def to_notebook(r: Recipe) -> dict:
    badge = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({r.colab_url})"
    notes = []
    if "gpu" in r.tags:
        notes.append("GPU ランタイム推奨（ランタイム → ランタイムのタイプを変更 → T4 GPU）")
    if "key" in r.tags:
        notes.append("API キーが必要（Colab 左の 🔑 シークレットに登録）")
    head = f"# {r.title}\n\n{badge}\n\nNLP クックブック {r.ident} ／ {r.chapter_title}"
    if notes:
        head += "\n\n> " + " ／ ".join(notes)
    cells = [md_cell(head)]
    for kind, src in r.cells:
        cells.append(md_cell(src) if kind == "markdown" else code_cell(src))
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index}"
    meta = {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "nlpcookbook": {"id": r.ident, "slug": r.slug, "digest": r.digest, "executed": False},
    }
    if "gpu" in r.tags:
        meta["accelerator"] = "GPU"
        meta["colab"]["gpuType"] = "T4"
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}


MAX_LINES = 60
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
WIDGET_RE = re.compile(r"\d+%\|| \[00:00<|^(Text|IntSlider|Output|HBox|VBox)\(")


def clean_outputs(nb: dict) -> None:
    """警告・進捗バーなどを落として、読んで意味のある出力だけ残す。"""
    count = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        cell["execution_count"] = count
        cell["metadata"] = {}
        outs = []
        if "".join(cell["source"]).lstrip().startswith(("!pip", "%pip", "!apt", "!python -m spacy download")):
            cell["outputs"] = []  # インストールのログは残さない
            continue
        for o in cell.get("outputs", []):
            if o["output_type"] == "stream":
                if o.get("name") == "stderr":
                    continue
                o["text"] = "".join(o["text"])
                if outs and outs[-1]["output_type"] == "stream":
                    outs[-1]["text"] += o["text"]
                    continue
            elif o["output_type"] in ("display_data", "execute_result"):
                data = o.get("data", {})
                had_widget = any("widget" in k for k in data)
                for k in [k for k in data if "widget" in k or k == "application/javascript"]:
                    del data[k]
                plain = "".join(data.get("text/plain", ""))
                if set(data) <= {"text/plain"} and (had_widget or WIDGET_RE.search(plain)):
                    continue  # ウィジェットや進捗バーの残骸
                if not data or set(data) == {"text/plain"} and o["output_type"] == "display_data" \
                        and "".join(data["text/plain"]).startswith(("HBox", "VBox", "<IPython")):
                    continue
                o["metadata"] = {}
                if o["output_type"] == "execute_result":
                    o["execution_count"] = count
            outs.append(o)
        for o in outs:
            if o["output_type"] == "stream":
                text = ANSI_RE.sub("", o["text"])
                # \r で上書きされる進捗表示は、各行の最後の状態だけ残す
                lines = [line.rstrip("\r").split("\r")[-1] for line in text.split("\n")]
                if len(lines) > MAX_LINES:
                    lines = lines[:MAX_LINES] + ["...（以下省略）"]
                o["text"] = "\n".join(lines)
        cell["outputs"] = outs
    nb["metadata"].pop("widgets", None)


def execute(r: Recipe, timeout: int) -> tuple[Recipe, dict | None, str]:
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.from_dict(to_notebook(r))
    with tempfile.TemporaryDirectory(prefix=f"nlpcb_{r.slug}_") as tmp:
        client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                                resources={"metadata": {"path": tmp}})
        try:
            client.execute()
        except Exception as e:  # noqa: BLE001
            lines = ANSI_RE.sub("", str(e)).strip().splitlines()
            return r, None, "\n".join(line[:300] for line in lines[-12:])
    out = json.loads(nbformat.writes(nb))
    clean_outputs(out)
    out["metadata"] = to_notebook(r)["metadata"]
    out["metadata"]["nlpcookbook"]["executed"] = True
    return r, out, ""


def preinstall(recipes: list[Recipe]) -> None:
    """並列実行中に pip が競合しないよう、必要なパッケージを先にまとめて入れておく。"""
    import shlex
    import subprocess

    for line in sorted({ln.strip() for r in recipes for kind, src in r.cells if kind == "code"
                        for ln in src.splitlines() if ln.startswith("!pip install")}):
        res = subprocess.run([sys.executable, "-m", "pip", *shlex.split(line)[1:]],
                             capture_output=True, text=True)
        if res.returncode:
            print(f"pip failed: {line}\n{res.stderr[-800:]}", flush=True)


def dump(nb: dict, path: Path) -> None:
    for cell in nb["cells"]:  # 差分が読みやすいよう、複数行文字列は行のリストで保存する
        if isinstance(cell["source"], str):
            cell["source"] = cell["source"].splitlines(keepends=True)
        for o in cell.get("outputs", []):
            if isinstance(o.get("text"), str):
                o["text"] = o["text"].splitlines(keepends=True)
    ids = [cell["id"] for cell in nb["cells"]]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", i) for i in ids):
        raise ValueError(f"{path}: セル ID が不正または重複しています")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def write_index(chapters: list[tuple[str, str, list[Recipe]]]) -> None:
    total = sum(len(rs) for _, _, rs in chapters)
    main_total = sum(len(rs) for _, _, rs in chapters if rs and isinstance(rs[0].chapter_no, int))
    count = f"全 {main_total} レシピ（付録を含めると {total} レシピ）" if main_total != total else f"全 {total} レシピ"
    lines = [f"{count}。タイトルをクリックすると GitHub 上で、"
             "バッジをクリックすると Google Colab で開きます。", ""]
    for title, _, rs in chapters:
        if rs:
            head = chapter_heading(rs[0].chapter_no, title)
            lines.append(f"- [{head}](#{slugify(head)}) ({len(rs)})")
    for title, intro, rs in chapters:
        if not rs:
            continue
        lines += ["", f"### {chapter_heading(rs[0].chapter_no, title)}", ""]
        if intro:
            lines += [intro, ""]
        lines += ["| # | やりたいこと | Colab |", "|---|---|---|"]
        for r in rs:
            mark = "".join(m for t, m in (("gpu", " ⚡"), ("key", " 🔑")) if t in r.tags)
            badge = f"[![Colab](https://colab.research.google.com/assets/colab-badge.svg)]({r.colab_url})"
            lines.append(f"| {r.ident} | [{r.title}]({r.github_url}){mark} | {badge} |")
    lines += ["", "⚡ = GPU ランタイム推奨、🔑 = API キーが必要"]
    block = "\n".join(lines)
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- INDEX:START -->", "<!-- INDEX:END -->"
    if start not in text:
        raise SystemExit("README.md に INDEX マーカーがありません")
    pre, rest = text.split(start, 1)
    post = rest.split(end, 1)[1]
    readme.write_text(f"{pre}{start}\n{block}\n{end}{post}", encoding="utf-8")


def chapter_heading(no: int | str, title: str) -> str:
    """README の章の見出し。本文の章は「3. 章名」、付録は「付録 A. 章名」。"""
    return f"付録 {no}. {title}" if isinstance(no, str) else f"{no}. {title}"


def chapter_files() -> list[Path]:
    """本文の章 (NN_*.md) を番号順に並べ、付録 (A_*.md など) をその後ろに置く。"""
    return sorted(SRC.glob("[0-9]*.md")) + sorted(SRC.glob("[A-Z]_*.md"))


def slugify(title: str) -> str:
    """GitHub の見出しアンカーに合わせる (記号を除去し、空白をハイフンに)。"""
    s = re.sub(r"[^\w\- ]", "", title.lower())
    return s.replace(" ", "-")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="レシピを実行して出力を埋め込む")
    ap.add_argument("--force", action="store_true", help="実行済みでも再実行する")
    ap.add_argument("-k", default="", help="スラッグまたは章ディレクトリ名の部分一致で絞り込む")
    ap.add_argument("-j", type=int, default=3, help="並列数")
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()

    chapters = [parse_chapter(p) for p in chapter_files()]
    recipes = [r for _, _, rs in chapters for r in rs]
    slugs = [r.slug for r in recipes]
    dup = {s for s in slugs if slugs.count(s) > 1}
    if dup:
        raise SystemExit(f"スラッグが重複しています: {sorted(dup)}")

    todo: list[Recipe] = []
    wanted = set()
    for r in recipes:
        path = ROOT / r.relpath
        wanted.add(path)
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        old_meta = (old or {}).get("metadata", {}).get("nlpcookbook", {})
        reusable = old_meta.get("executed") and old_meta.get("digest") == r.digest
        selected = args.k in r.slug or args.k in r.chapter_dir
        runnable = "norun" not in r.tags and "key" not in r.tags
        if args.run and runnable and selected and (args.force or not reusable):
            todo.append(r)
            if not old:
                dump(to_notebook(r), path)
            continue
        nb = to_notebook(r)
        if reusable:  # コードが同じなら実行済みの出力を引き継ぐ
            code_old = [c for c in old["cells"] if c["cell_type"] == "code"]
            code_new = [c for c in nb["cells"] if c["cell_type"] == "code"]
            for co, cn in zip(code_old, code_new):
                cn["outputs"], cn["execution_count"] = co["outputs"], co["execution_count"]
            meta = nb["metadata"]
            clean_outputs(nb)
            nb["metadata"] = meta
            nb["metadata"]["nlpcookbook"]["executed"] = True
        dump(nb, path)

    for stale in set(OUT.glob("**/*.ipynb")) - wanted:
        stale.unlink()
        print(f"removed {stale.relative_to(ROOT)}")

    write_index(chapters)
    print(f"{len(recipes)} recipes / {len(chapters)} chapters")

    failed = []
    if todo:
        print(f"running {len(todo)} recipes ...", flush=True)
        preinstall(todo)
        with cf.ThreadPoolExecutor(args.j) as ex:
            for r, nb, err in ex.map(lambda r: execute(r, args.timeout), todo):
                if nb is None:
                    failed.append(r)
                    print(f"FAIL {r.ident} {r.slug}\n{err}\n", flush=True)
                else:
                    dump(nb, ROOT / r.relpath)
                    print(f"ok   {r.ident} {r.slug}", flush=True)
    if failed:
        print(f"{len(failed)} failed: {' '.join(r.slug for r in failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
