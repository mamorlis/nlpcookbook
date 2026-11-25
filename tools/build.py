#!/usr/bin/env python3
"""recipes/*.md から Colab 用ノートブック (notebooks/**/*.ipynb) と README の目次を生成する。

    python tools/build.py              # ノートブックと目次を生成 (実行はしない)
    python tools/build.py --lint       # 書式・相互参照・表記ルールを検査し、警告があれば終了コード 1
    python tools/build.py --run        # 未実行・変更ありのレシピを実行して出力を埋め込む
    python tools/build.py --run -k bm25  # スラッグに bm25 を含むレシピだけ実行
    python tools/build.py --run --force  # すべて再実行

レシピの書式は CONTRIBUTING.md を参照。
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures as cf
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "recipes"
OUT = ROOT / "notebooks"
REPO = "mamorlis/nlpcookbook"
BRANCH = "main"
COLAB_BADGE = "https://colab.research.google.com/assets/colab-badge.svg"
STARTER_SLUGS = ("nfkc", "janome", "wikipedia_api")
README_FOOTER = (
    "\n\n## 使い方\n\n"
    "- **Colab で開く**: 目次のバッジをクリックします。自分用に書き換えるときは「ドライブにコピーを保存」してください。\n"
    "- **Google Drive に置く**: `notebooks/` 以下を Drive の `Colab Notebooks/nlpcookbook/` にコピーすれば、Drive から直接開けます。\n\n"
    "## レシピの追加・修正\n\n"
    "ノートブックは `recipes/*.md` から自動生成しています。`.ipynb` を直接編集せず、Markdown を編集してから\n"
    "`python tools/build.py` を実行してください。詳しくは [CONTRIBUTING.md](https://github.com/mamorlis/nlpcookbook/blob/main/CONTRIBUTING.md) を参照。\n\n"
    "## ライセンス\n\n"
    "[Apache License 2.0](https://github.com/mamorlis/nlpcookbook/blob/main/LICENSE)\n"
)

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
    badge = f"[![Open In Colab]({COLAB_BADGE})]({r.colab_url})"
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
ANSI_RE = re.compile(r"\x1b\[\??[0-9;]*[A-Za-z]")
WIDGET_RE = re.compile(r"\d+%\|| \[00:00<|^(Text|IntSlider|Output|HBox|VBox)\(")
INSTALL_PREFIXES = (
    "!pip install",
    "%pip install",
    "!apt install",
    "!apt-get install",
    "!python -m spacy download",
)


def is_install_cell(source: str | list[str]) -> bool:
    """空行とコメント行を除いたすべての行がインストール専用コマンドか判定する。"""
    lines = [ln.strip() for ln in "".join(source).splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return bool(lines) and all(ln.startswith(INSTALL_PREFIXES) for ln in lines)


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
        if is_install_cell(cell["source"]):
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
                if "text/html" in data:
                    html = "".join(data["text/html"])
                    if "<progress" in html:
                        html = re.sub(r"\s*<div>\s*<progress\b[^>]*>.*?</progress>.*?</div>\s*", "", html, flags=re.DOTALL)
                        if html.strip():
                            data["text/html"] = html.splitlines(keepends=True) if isinstance(data["text/html"], list) else html
                        else:
                            del data["text/html"]
                    elif not html.strip():
                        del data["text/html"]
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


def cuda_available() -> bool:
    """実行環境の Python で CUDA が使えるかを 1 回だけ確認する。"""
    res = subprocess.run(
        [sys.executable, "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"],
        capture_output=True,
    )
    return res.returncode == 0


def build_readme(chapters: list[tuple[str, str, list[Recipe]]]) -> str:
    """README.md 全体（冒頭・初めての方へ・目次・末尾）を組み立てて返す。"""
    by_slug = {r.slug: r for _, _, rs in chapters for r in rs}
    starter_lines = []
    for slug in STARTER_SLUGS:
        r = by_slug.get(slug)
        if r is None:
            raise ValueError(f"スターターレシピが見つかりません: {slug}")
        badge = f"[![Open In Colab]({COLAB_BADGE})]({r.colab_url})"
        starter_lines.append(f"- {r.ident} [{r.title}]({r.github_url}) {badge}")

    header_lines = [
        "# 自然言語処理クックブック",
        "",
        "自然言語処理の「やりたいこと」に対して「こうやればいい」をまとめた小ネタ集です。",
        "1 レシピ = 1 ノートブックで、すべて Google Colab でそのまま実行できます。",
        "",
        "- 日本語のテキスト処理を中心に、英語・多言語も扱います。",
        "- 各ノートブックは上から順に実行するだけで動きます（必要なライブラリのインストールを含む）。",
        "- 実行結果を埋め込んであるので、GitHub 上で眺めるだけでも使えます。",
        "- 理論や背景を体系的に学びたい方は、姉妹編の [自然言語処理の教科書](https://github.com/mamorlis/nlpbook) もあわせてご覧ください。",
        "",
        "## 初めての方へ",
        "",
        "次の3件は CPU で試せます。リンク先で実行例を読み、自分の文章で試すときは「Open In Colab」を開いてください。",
        "",
        *starter_lines,
        "",
        "目次の ⚡ は GPU ランタイム推奨、🔑 は API キーが必要なレシピです。",
        "",
        "## 目次",
    ]

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
            badge = f"[![Colab]({COLAB_BADGE})]({r.colab_url})"
            lines.append(f"| {r.ident} | [{r.title}]({r.github_url}){mark} | {badge} |")
    lines += ["", "⚡ = GPU ランタイム推奨、🔑 = API キーが必要"]
    block = "\n".join(lines)

    readme = ROOT / "README.md"
    end = "<!-- INDEX:END -->"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        post = text.split(end, 1)[1] if end in text else README_FOOTER
    else:
        post = README_FOOTER
    header = "\n".join(header_lines)
    return f"{header}\n\n<!-- INDEX:START -->\n{block}\n{end}{post}"


def write_index(chapters: list[tuple[str, str, list[Recipe]]]) -> None:
    (ROOT / "README.md").write_text(build_readme(chapters), encoding="utf-8")


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


def _sanitize_prose(text: str) -> str:
    """インラインコードと Markdown リンクの URL を除いて、散文の検査用テキストを作る。"""
    # ASCII のみのコードは "CODE" に（(`oracle=True`) を検出するため）、日本語を含むコードは "日本語" に置換する
    text = re.sub(r"`([^`]+)`", lambda m: "CODE" if m.group(1).isascii() else "日本語", text)
    return re.sub(r"\]\([^)]+\)", "]", text)


def _scan_quoted(text: str, start: int) -> tuple[str | None, int]:
    """start 位置の '「' に対応する '」' までをネストを数えて取り出す。"""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "「":
            depth += 1
        elif text[i] == "」":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
    return None, len(text)


NON_COLAB_MODULES = {
    "fugashi", "unidic", "unidic_lite", "ipadic", "mecab", "MeCab",
    "sudachipy", "sudachidict_core", "janome", "pyknp", "kytea", "Mykytea",
    "pyopenjtalk", "neologdn", "jaconv", "mojimoji", "zenhan", "ftfy",
    "trafilatura", "fitz", "docx", "datasketch", "evaluate",
    "sacrebleu", "rouge_score", "bert_score", "comet", "sklearn_crfsuite",
    "seqeval", "rank_bm25", "faiss", "qdrant_client", "chromadb", "bm25s",
    "umap", "hdbscan", "bertopic", "optuna", "cleanlab", "krippendorff",
    "matplotlib_fontja", "errant", "spacy_udpipe", "stanza", "ginza", "kwja",
    "ja_timex", "symspellpy", "spellchecker", "Levenshtein", "rapidfuzz",
    "vllm", "bitsandbytes", "outlines", "trl", "gradio", "whisper",
    "faster_whisper", "gtts", "edge_tts", "anthropic", "genai", "tiktoken",
    "llama_cpp", "tuned_lens", "nnsight", "pycocoevalcap", "open_clip",
    "datatrove", "kenlm", "CaboCha", "loguru", "rhoknp", "pendulum",
    "marisa_trie",
}

PIP_PROVIDERS: dict[str, set[str]] = {
    "fitz": {"pymupdf"},
    "docx": {"python-docx"},
    "MeCab": {"mecab-python3", "fugashi"},
    "mecab": {"mecab-python3", "fugashi"},
    "unidic_lite": {"unidic-lite", "transformers"},
    "comet": {"unbabel-comet"},
    "spellchecker": {"pyspellchecker"},
    "Mykytea": {"kytea", "mykytea"},
    "Levenshtein": {"python-levenshtein", "levenshtein"},
    "genai": {"google-genai"},
    "whisper": {"openai-whisper"},
    "open_clip": {"open-clip-torch"},
    "faiss": {"faiss-cpu", "faiss-gpu"},
    "umap": {"umap-learn"},
    "llama_cpp": {"llama-cpp-python"},
    # 推移的依存
    "rhoknp": {"kwja"},
    "pendulum": {"ja-timex"},
    "loguru": {"datatrove"},
    "sacrebleu": {"unbabel-comet"},
}


def _join_continuations(lines: list[str]) -> list[str]:
    """バックスラッシュで続く複数行を 1 行につなぐ。"""
    joined: list[str] = []
    buf = ""
    for line in lines:
        if buf:
            buf += " " + line.lstrip()
        else:
            buf = line
        if buf.endswith("\\"):
            buf = buf[:-1]
        else:
            joined.append(buf)
            buf = ""
    if buf:
        joined.append(buf)
    return joined


def _strip_magic_lines(src: str) -> str:
    """! / % で始まるシェル・マジック行（複数行継続を含む）を除いた Python コードを返す。"""
    lines = src.splitlines()
    if lines and lines[0].lstrip().startswith("%%"):
        return ""
    kept: list[str] = []
    in_magic_cont = False
    for line in lines:
        stripped = line.lstrip()
        if in_magic_cont:
            in_magic_cont = line.rstrip().endswith("\\")
            continue
        if stripped.startswith(("!", "%")):
            in_magic_cont = line.rstrip().endswith("\\")
            continue
        kept.append(line)
    return "\n".join(kept)


def _extract_imports(src: str) -> tuple[set[str], set[str]]:
    """コードセルからトップレベル import 名と sys.path に追加されたローカルモジュール候補を返す。"""
    clean = _strip_magic_lines(src)
    mods: set[str] = set()
    local_dirs: set[str] = set()
    if not clean.strip():
        return mods, local_dirs
    try:
        tree = ast.parse(clean)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mods.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods.add(node.module.split(".")[0])
    except SyntaxError:
        for line in clean.splitlines():
            m = re.match(r"^\s*import\s+(.+)", line)
            if m:
                for part in m.group(1).split(","):
                    name = part.strip().split()[0].split(".")[0]
                    if name:
                        mods.add(name)
            m = re.match(r"^\s*from\s+([A-Za-z0-9_]+)\s+import\b", line)
            if m:
                mods.add(m.group(1))
    for m in re.finditer(r"""sys\.path\.(?:insert|append)\([^)]*?["']([^"']+)["']""", clean):
        tail = Path(m.group(1)).name
        if tail:
            local_dirs.add(tail.removesuffix("-py"))
    return mods, local_dirs


def _extract_installed_packages(code_cells: list[str]) -> tuple[set[str], set[str]]:
    """コードセル内の pip install と %%writefile から、インストール済みパッケージ名とローカル .py 名を集める。"""
    installed: set[str] = set()
    local_mods: set[str] = set()
    skip_next_flags = {"-i", "--index-url", "--extra-index-url", "-f", "--find-links", "-r", "--requirement",
                       "-c", "--constraint", "-e", "--editable", "-t", "--target", "--prefix", "--root"}
    for src in code_cells:
        lines = src.splitlines()
        if lines and lines[0].lstrip().startswith("%%writefile"):
            parts = lines[0].split()
            if len(parts) >= 2 and parts[1].endswith(".py"):
                local_mods.add(Path(parts[1]).stem)
        for line in _join_continuations(lines):
            s = line.strip()
            if not s.startswith(("!", "%")):
                continue
            cmd = s.lstrip("!%").strip()
            # 複合コマンドを && / ; / || で分け、直前の cd <dir> も覚えておく
            cur_dir = ""
            for seg in re.split(r"\s*(?:&&|\|\||;)\s*", cmd):
                seg = seg.strip()
                if seg.startswith("cd "):
                    cur_dir = Path(seg[3:].strip().split()[0]).name
                m = re.search(r"\b(?:pip3?|python3?\s+-m\s+pip)\s+install\b(.*)", seg)
                if not m:
                    continue
                try:
                    tokens = shlex.split(m.group(1), comments=True)
                except ValueError:
                    tokens = m.group(1).split()
                skip = False
                for tok in tokens:
                    if skip:
                        skip = False
                        continue
                    if tok in skip_next_flags:
                        skip = True
                        continue
                    if tok.startswith("-") or tok in {">", ">>", "2>&1", "|"} or tok.startswith((">", "<")):
                        continue
                    if tok in {".", "./"} and cur_dir:
                        installed.add(cur_dir.lower())
                    elif "/" in tok or tok.endswith(".whl"):
                        stem = Path(tok.rstrip("/")).name
                        if stem in {"python", "python3"}:
                            stem = Path(tok.rstrip("/")).parent.name
                        if stem:
                            installed.add(stem.lower())
                    else:
                        pkg = re.split(r"[=<>!~\[;]", tok)[0].strip().lower()
                        if pkg:
                            installed.add(pkg)
                            installed.add(pkg.replace("_", "-"))
    return installed, local_mods


def lint_chapters(chapters: list[tuple[str, str, list[Recipe]]]) -> list[str]:
    """章タイトル・レシピタイトルの相互参照、全角括弧、非推奨パッケージ、pip install 漏れを検査する。"""
    warnings: list[str] = []
    chap_titles = {ct for ct, _, _ in chapters}
    recipe_titles = {r.title for _, _, rs in chapters for r in rs}
    chap_ref_re = re.compile(r"「([^」]+)」の章")
    recipe_ref_re = re.compile(r"(?:(?:この章|「[^」]+」の章|[^\s、。（「]+の章)の|(?:前|次)のレシピ)(?=「)")
    fw_ascii_re = re.compile(r"（([\x20-\x7E]+)）")
    fw_space_re = re.compile(r"（ | ）")

    for ct, intro, rs in chapters:
        chap_dir = rs[0].chapter_dir if rs else ct
        prose_targets: list[tuple[str, str]] = [(f"{chap_dir} (intro)", intro)] if intro else []
        for r in rs:
            prose_targets.append((f"{r.ident} {r.slug} (title)", r.title))
            code_cells: list[str] = []
            for kind, src in r.cells:
                if kind == "markdown":
                    prose_targets.append((f"{r.ident} {r.slug}", src))
                elif kind == "code":
                    code_cells.append(src)
                    if "japanize-matplotlib" in src or "japanize_matplotlib" in src:
                        warnings.append(f"{r.ident} {r.slug}: 非推奨の japanize-matplotlib が使われています")
            imported: set[str] = set()
            local_mods: set[str] = set()
            for src in code_cells:
                mods, dirs = _extract_imports(src)
                imported |= mods
                local_mods |= dirs
            installed, write_mods = _extract_installed_packages(code_cells)
            local_mods |= write_mods
            for mod in sorted((imported - local_mods) & NON_COLAB_MODULES):
                accepted = {mod.lower(), mod.lower().replace("_", "-")} | PIP_PROVIDERS.get(mod, set())
                if not (accepted & installed):
                    warnings.append(f"{r.ident} {r.slug}: {mod} に対応する !pip install が見つかりません")

        for loc, raw in prose_targets:
            prose = _sanitize_prose(raw)
            for m in chap_ref_re.finditer(prose):
                if m.group(1) not in chap_titles:
                    warnings.append(f"{loc}: 章タイトルの参照が見つかりません: 「{m.group(1)}」")
            for m in recipe_ref_re.finditer(prose):
                pos = m.end()
                while pos < len(prose):
                    title, end = _scan_quoted(prose, pos)
                    if title is None:
                        warnings.append(f"{loc}: レシピ参照の括弧が閉じていません: {prose[pos:pos + 30]!r}")
                        break
                    if title not in recipe_titles:
                        warnings.append(f"{loc}: レシピの参照が見つかりません: 「{title}」")
                    nxt = re.match(r"^(?:[とや・、]?\s*)?(?=「)", prose[end:])
                    if not nxt:
                        break
                    pos = end + nxt.end()
            for m in fw_ascii_re.finditer(prose):
                warnings.append(f"{loc}: ASCII のみに全角括弧が使われています: {m.group(0)}")
            for m in fw_space_re.finditer(prose):
                ctx = prose[max(0, m.start() - 10) : min(len(prose), m.end() + 10)].replace("\n", " ")
                warnings.append(f"{loc}: 全角括弧の内側に空白があります: ...{ctx}...")

    return warnings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="レシピを実行して出力を埋め込む")
    ap.add_argument("--force", action="store_true", help="実行済みでも再実行する")
    ap.add_argument("--allow-cpu-for-gpu", action="store_true", help="CPU 環境でも gpu タグのレシピを実行する")
    ap.add_argument("--lint", action="store_true", help="書式・相互参照・表記ルールを検査し、警告があれば終了コード 1 で終了する")
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

    has_cuda = cuda_available() if args.run and not args.allow_cpu_for_gpu else False

    existing_by_slug: dict[str, dict] = {}
    for p in sorted(OUT.glob("**/*.ipynb")):
        nb_old = json.loads(p.read_text(encoding="utf-8"))
        slug_old = nb_old.get("metadata", {}).get("nlpcookbook", {}).get("slug")
        if slug_old:
            existing_by_slug[slug_old] = nb_old

    todo: list[Recipe] = []
    wanted = set()
    for r in recipes:
        path = ROOT / r.relpath
        wanted.add(path)
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else existing_by_slug.get(r.slug)
        old_meta = (old or {}).get("metadata", {}).get("nlpcookbook", {})
        reusable = old_meta.get("executed") and old_meta.get("digest") == r.digest
        selected = args.k in r.slug or args.k in r.chapter_dir
        runnable = "norun" not in r.tags and "key" not in r.tags
        if args.run and runnable and selected and (args.force or not reusable):
            if "gpu" in r.tags and not args.allow_cpu_for_gpu and not has_cuda:
                print(f"skip {r.ident} {r.slug} (gpu recipe on CPU host; pass --allow-cpu-for-gpu to override)")
            else:
                todo.append(r)
                if not path.exists():
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

    for d in sorted((p for p in OUT.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()

    write_index(chapters)
    print(f"{len(recipes)} recipes / {len(chapters)} chapters")

    warnings = lint_chapters(chapters)
    for w in warnings:
        print(f"lint: {w}", file=sys.stderr)
    if args.lint:
        if warnings:
            print(f"lint: {len(warnings)} warnings", file=sys.stderr)
            sys.exit(1)
        print("lint: 0 warnings")

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
