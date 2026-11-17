# データの収集と入出力

テキストを集める、ファイルや音声から取り出す、きれいにする、品質で選ぶ、分割する。実験の前にやること。JSON Lines や CSV の読み書きのようなファイル操作の小技は、付録の「Colab と Python の小技」を参照。

## 青空文庫から作品をダウンロードして本文を取り出す {aozora_download}

著作権の切れた日本語の文学作品を、手軽に使える日本語コーパスとして取得したい。ルビや注記を取り除きたい。

```python
import io, re, urllib.request, zipfile

def aozora(url):
    """青空文庫の「テキストファイル (ルビあり)」の zip の URL から、本文だけを取り出す"""
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    raw = z.read(z.namelist()[0]).decode("cp932")           # 青空文庫のテキストは Shift_JIS
    header, _, body = re.split(r"-{50,}", raw)[:3]           # 冒頭の「テキスト中に現れる記号について」を除く
    body = body.split("底本：")[0]                            # 末尾の底本情報を除く
    body = re.sub(r"《.+?》", "", body)                       # ルビ: 吾輩《わがはい》
    body = re.sub(r"［＃.+?］", "", body)                     # 入力者注: ［＃「言＋墟のつくり」、第4水準2-88-74］
    body = body.replace("｜", "")                             # ルビの開始位置を示す記号
    return header.strip().split("\r\n")[:2], body.replace("\r\n", "\n").strip()

(title, author), text = aozora("https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip")
print(title, "/", author, "/", len(text), "文字")
print(text[:120])
```

```python
# 生のテキストはこうなっている
with urllib.request.urlopen("https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip") as r:
    raw = zipfile.ZipFile(io.BytesIO(r.read())).read("wagahaiwa_nekodearu.txt").decode("cp932")
start = raw.index("吾輩《わがはい》は猫である。")
print(raw[start:start + 150])
```

- zip の URL は、各作品の図書カード (`https://www.aozora.gr.jp/cards/人物ID/card作品ID.html`) の「ファイルのダウンロード」欄にある。
- 全作品の一覧は「公開中 作家別作品一覧拡充版」の CSV として配布されている。
- 新字新仮名の作品を選ぶと現代の解析器で扱いやすい。旧字旧仮名の作品は UniDic の近代文語用辞書などを検討する。

## Wikipedia の記事本文を API で取得する {wikipedia_api}

特定の Wikipedia 記事の本文を、HTML を解析せずにプレーンテキストで取得したい。

```python
import requests

API = "https://ja.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "nlp-cookbook-example/0.1 (educational notebook)"}   # 利用者を識別できる User-Agent を付けるのがルール

def fetch(title):
    params = {"action": "query", "prop": "extracts", "explaintext": 1, "titles": title, "format": "json", "redirects": 1}
    pages = requests.get(API, params=params, headers=HEADERS, timeout=30).json()["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("extract", "")

text = fetch("自然言語処理")
print(len(text), "文字")
print(text[:300])
```

```python
# キーワードで記事を検索する
def search(query, limit=5):
    params = {"action": "query", "list": "search", "srsearch": query, "srlimit": limit, "format": "json"}
    return [hit["title"] for hit in requests.get(API, params=params, headers=HEADERS, timeout=30).json()["query"]["search"]]

print(search("機械翻訳 ニューラル"))
```

- 数件〜数百件ならこの方法で。連続して呼ぶときは間隔を空ける (`time.sleep(1)`)。短時間に大量に呼ぶと 429 (Too Many Requests) で拒否される。
- 全記事が必要なら API を叩かずに、ダンプか Hugging Face の `wikimedia/wikipedia` データセットを使う（Transformers の基本操作の章のストリーミングのレシピを参照）。
- 本文のライセンスは CC BY-SA。再配布するときは出典の表示と同一ライセンスでの公開が必要。

## Web ページから本文だけを抽出する (trafilatura) {web_extract}

ニュース記事やブログのページから、メニューや広告を除いた本文テキストだけを取り出したい。

```python
!pip install -q trafilatura
```

```python
import trafilatura

url = "https://www.aozora.gr.jp/guide/aozora_bunko_hayawakari.html"
html = trafilatura.fetch_url(url)
text = trafilatura.extract(html, include_comments=False, include_tables=False)
print(len(html), "文字の HTML →", len(text), "文字の本文")
print(text[:300])
```

```python
# タイトルや日付などのメタデータも一緒に、JSON で取り出す
import json

result = json.loads(trafilatura.extract(html, output_format="json", with_metadata=True))
print({k: result.get(k) for k in ["title", "author", "date", "hostname"]})
```

- 本文らしさをテキスト密度やタグの構造から推定している。BeautifulSoup でサイトごとにセレクタを書くより、多様なサイトに強い。
- JavaScript で描画されるページは `fetch_url` では取得できない。Playwright などで HTML を取得してから `extract` に渡す。
- スクレイピングの前に、サイトの利用規約と robots.txt を確認する。アクセス間隔を空け、著作権（特に再配布）に注意する。

## PDF からテキストを抽出する (PyMuPDF) {pdf_extract}

論文や報告書の PDF から本文テキストを取り出したい。ページごと、段組みを考慮して取りたい。

```python
!pip install -q pymupdf
```

```python
import urllib.request
import fitz  # PyMuPDF

urllib.request.urlretrieve("https://arxiv.org/pdf/1706.03762", "paper.pdf")   # Attention Is All You Need
doc = fitz.open("paper.pdf")
print(len(doc), "ページ", doc.metadata.get("title"))

page = doc[0]
print(page.get_text()[:400])
```

```python
import re

# ブロック (段落) 単位で取り出すと、位置情報が得られ、読み順の制御やヘッダ・フッタの除去ができる
blocks = doc[1].get_text("blocks")     # (x0, y0, x1, y1, テキスト, ブロック番号, 種類)
for x0, y0, x1, y1, text, *_ in blocks[:4]:
    print(f"({x0:5.0f}, {y0:5.0f})  {text[:70]!r}")

def clean(text):
    text = re.sub(r"-\n(\w)", r"\1", text)      # 行末のハイフネーションをつなぐ
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)   # 段落内の改行を空白に (日本語の場合は空文字に)
    return re.sub(r"[ \t]+", " ", text).strip()

full = "\n\n".join(clean(b[4]) for p in doc for b in p.get_text("blocks", sort=True) if b[6] == 0)
print(len(full), "文字")
print(full[1500:1900])
```

- PDF は「文字と座標の集まり」で、段落や読み順の情報を持たない。2 段組み、表、数式、脚注は崩れやすい。
- スキャンされた PDF（画像）からは抽出できない。OCR が必要（`page.get_textpage_ocr()` や、専用の OCR / 文書解析モデル）。
- 論文の構造（章、参考文献）まで欲しいなら GROBID、レイアウトを保った Markdown が欲しいなら `pymupdf4llm` や Docling などがある。

## Word ファイルからテキストと表を取り出す {docx_extract}

`.docx` 形式の文書から、段落のテキストや見出し、表の中身を取り出したい。

```python
!pip install -q python-docx
```

```python
from docx import Document

# 動作確認用の Word ファイルを作る
doc = Document()
doc.add_heading("実験報告書", level=1)
doc.add_paragraph("本報告書では、日本語の文書分類の実験結果を述べる。")
doc.add_heading("結果", level=2)
doc.add_paragraph("提案手法はベースラインを上回った。")
table = doc.add_table(rows=3, cols=2)
for row, (a, b) in zip(table.rows, [("手法", "正解率"), ("ベースライン", "0.85"), ("提案手法", "0.91")]):
    row.cells[0].text, row.cells[1].text = a, b
doc.save("report.docx")
```

```python
import pandas as pd

doc = Document("report.docx")
for p in doc.paragraphs:
    if p.text.strip():
        print(f"[{p.style.name}] {p.text}")

for t in doc.tables:
    rows = [[cell.text for cell in row.cells] for row in t.rows]
    print(pd.DataFrame(rows[1:], columns=rows[0]))
```

- 見出しはスタイル名（`Heading 1`、日本語版 Word で作ると `見出し 1`）で判別できる。
- 古い `.doc` 形式は読めない。LibreOffice で変換する (`!soffice --headless --convert-to docx file.doc`)。
- PowerPoint は `python-pptx`、Excel は `pandas.read_excel` (openpyxl) で同様に読める。

## Whisper で日本語の音声を書き起こす（モデルの比較、長い音声、字幕） {whisper_asr gpu}

録音した音声（講演、会議、インタビュー）を日本語のテキストに書き起こして、NLP の材料にしたい。どのモデルを使えばどれくらい正確で速いか、長い音声や字幕 (SRT) はどう扱うか、どんな誤りが出るかも確かめたい。

```python
!pip install -q jiwer silero-vad
```

```python
import io
import numpy as np
import pandas as pd
import soundfile as sf
from huggingface_hub import hf_hub_download
from IPython.display import Audio

# FLEURS（Google、CC BY 4.0）の日本語の開発用データ: 機械翻訳の評価データ FLoRes の文（英語の記事の翻訳）を読み上げた音声と、その文
# Common Voice の日本語なども使えるが、配布元の利用条件を確かめる
path = hf_hub_download("google/fleurs", "parquet-data/ja_jp/validation-00000-of-00001.parquet", repo_type="dataset")   # 約 220MB
df = pd.read_parquet(path).drop_duplicates("id").sort_values("id").reset_index(drop=True)   # 同じ文を複数の話者が読んでいるので 1 文 1 録音にする
SR = 16000                                                   # Whisper は 16kHz のモノラルを前提にする
df["audio"] = [sf.read(io.BytesIO(a["bytes"]), dtype="float32") for a in df["audio"]]
assert all(sr == SR for _, sr in df["audio"])                # ほかの音声は ffmpeg -ar 16000 -ac 1 で 16kHz のモノラルにしてから渡す
df["audio"] = [x for x, _ in df["audio"]]
print(len(df), "文", round(sum(map(len, df["audio"])) / SR / 60, 1), "分")
print(df["raw_transcription"][3])
Audio(df["audio"][3], rate=SR)
```

```python
import time
import unicodedata
import jiwer
import torch
from transformers import pipeline

def normalize(s):
    """CER を測る前の正規化: 全角半角をそろえ、句読点・記号と空白を除く"""
    s = unicodedata.normalize("NFKC", s)
    return "".join(c for c in s if not unicodedata.category(c).startswith(("P", "Z")))

BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
DTYPE = torch.bfloat16 if BF16 else torch.float16
GEN = {"language": "ja", "task": "transcribe"}          # 言語を指定しないと、最初の 30 秒から言語を推定する
refs = [normalize(s) for s in df["raw_transcription"]]
audio_sec = sum(map(len, df["audio"])) / SR

MODELS = {"small": "openai/whisper-small",                       # Apache 2.0
          "large-v3-turbo": "openai/whisper-large-v3-turbo",     # MIT
          "kotoba-v2.0": "kotoba-tech/kotoba-whisper-v2.0"}      # Apache 2.0（日本語のテレビ音声 ReazonSpeech で蒸留）
hyps, rows = {}, []
for short, name in MODELS.items():                     # 3 モデルで T4 なら 5〜10 分程度（ダウンロードを含む）
    asr = pipeline("automatic-speech-recognition", model=name, dtype=DTYPE, device_map="auto")
    asr(df["audio"][0], generate_kwargs=GEN)           # 1 回目は初期化で遅いので、測る前に 1 回動かす
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t = time.time()
    out = asr(list(df["audio"]), batch_size=16, generate_kwargs=GEN)
    sec = time.time() - t
    hyps[short] = [o["text"] for o in out]
    rows.append({"モデル": short, "パラメータ（億）": asr.model.num_parameters() / 1e8,
                 "CER": jiwer.cer(refs, [normalize(h) for h in hyps[short]]), "時間（秒）": sec,
                 "実時間の何倍速": audio_sec / sec, "最大メモリ (GB)": torch.cuda.max_memory_allocated() / 1e9})
    del asr; torch.cuda.empty_cache()
pd.DataFrame(rows).round(3)
```

```python
# 文単位で再標本化するブートストラップで、CER の 95% 信頼区間とモデル間の差の信頼区間を求める
def errors(ref, hyp):
    m = jiwer.process_characters(ref, normalize(hyp))
    return m.substitutions + m.deletions + m.insertions

n_chars = np.array([len(r) for r in refs])
err = {k: np.array([errors(r, h) for r, h in zip(refs, v)]) for k, v in hyps.items()}
rng = np.random.default_rng(0)
idx = rng.integers(0, len(refs), (2000, len(refs)))
boot = {k: e[idx].sum(1) / n_chars[idx].sum(1) for k, e in err.items()}
for k, b in boot.items():
    print(f"{k:15s} CER {err[k].sum() / n_chars.sum():.3f}  95% CI [{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]")
for a, b in [("small", "large-v3-turbo"), ("kotoba-v2.0", "large-v3-turbo")]:
    d = boot[a] - boot[b]
    print(f"{a} - {b}: {d.mean():+.3f}  95% CI [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")
```

```python
import re

# 典型的な誤り: 数字、カタカナ語（外国の固有名詞を含む）、漢字の語が、語の単位でどれくらい正しく出るか
def hit_rate(pattern, model):
    found = total = 0
    for r, h in zip(refs, hyps[model]):
        words = re.findall(pattern, r)
        found += sum(w in normalize(h) for w in words); total += len(words)
    return found / total, total

for model in hyps:
    rates = {k: hit_rate(p, model) for k, p in [("数字", r"\d+"), ("カタカナ 3 字以上", r"[ァ-ヴー]{3,}"), ("漢字 3 字以上", r"[一-龥]{3,}")]}
    print(f"{model:15s}", "  ".join(f"{k} {r:.2f}（{n} 個）" for k, (r, n) in rates.items()), f"  1 - CER {1 - err[model].sum() / n_chars.sum():.2f}")

# large-v3-turbo の誤りが多い文
for i in np.argsort(-err["large-v3-turbo"] / n_chars)[:6]:
    print(f"\n正解  : {df['raw_transcription'][i]}")
    for model in ["large-v3-turbo", "kotoba-v2.0"]:
        print(f"{model:6s}: {hyps[model][i]}")
```

```python
# 長い音声: 15 文をつなげた約 4 分の音声（文の間に 1 秒の無音）。録音ごとに音量が違うのは、会議で話者とマイクの距離が違うのに近い
long_audio = np.concatenate([x for a in df["audio"][:15] for x in (a, np.zeros(SR, np.float32))])
long_ref = normalize("".join(df["raw_transcription"][:15]))
sf.write("long.wav", long_audio, SR)
# ffmpeg の dynaudnorm で、短い区間ごとに音量をそろえる（f: 区間の長さ（ミリ秒）、g: 平滑化する区間数、m: 最大の増幅率）
!ffmpeg -y -loglevel error -i long.wav -af dynaudnorm=f=200:g=11:m=100 long_norm.wav
long_norm = sf.read("long_norm.wav", dtype="float32")[0]

starts = np.cumsum([0] + [len(a) + SR for a in df["audio"][:15]])
db = lambda x: 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)
print(round(len(long_audio) / SR), "秒")
print("文ごとの音量 (dBFS)       :", [round(db(long_audio[s:e - SR])) for s, e in zip(starts, starts[1:])])
print("文ごとの音量（そろえたあと）:", [round(db(long_norm[s:e - SR])) for s, e in zip(starts, starts[1:])])
```

```python
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", dtype=DTYPE, device_map="auto")

def report(name, text, sec):
    m = jiwer.process_characters(long_ref, normalize(text))
    print(f"{name:30s} CER {m.cer:.3f}（置換 {m.substitutions:3d} / 削除 {m.deletions:3d} / 挿入 {m.insertions:3d}）、{sec:.1f} 秒")

# 30 秒を超える音声は、chunk_length_s で区切るか、return_timestamps=True で逐次処理する（どちらもないと ValueError になる）
results = {}
for name, audio, kw in [("逐次処理（音量そのまま）", long_audio, dict(return_timestamps=True)),
                        ("逐次処理", long_norm, dict(return_timestamps=True)),
                        ("chunk_length_s=30", long_norm, dict(chunk_length_s=30, batch_size=8)),
                        ("chunk_length_s=30 + タイムスタンプ", long_norm, dict(chunk_length_s=30, batch_size=8, return_timestamps=True))]:
    t = time.time()
    results[name] = asr(audio, generate_kwargs=GEN, **kw)
    report(name, results[name]["text"], time.time() - t)

# 文ごとに、正解の文字のうち認識結果に正しく出た割合を見る
bounds = np.cumsum([0] + [len(normalize(t)) for t in df["raw_transcription"][:15]])
def sentence_recall(text):
    ok = np.zeros(len(long_ref), bool)
    for c in jiwer.process_characters(long_ref, normalize(text)).alignments[0]:
        if c.type == "equal":
            ok[c.ref_start_idx:c.ref_end_idx] = True
    return [round(float(ok[a:b].mean()), 2) for a, b in zip(bounds, bounds[1:])]

for name in ["逐次処理（音量そのまま）", "逐次処理", "chunk_length_s=30"]:
    print(f"{name:24s}", sentence_recall(results[name]["text"]))
print(results["chunk_length_s=30"]["text"][:150])

# タイムスタンプ付きの分割で、正解より多く出てくる 12 文字の並び（二重に出たか、挿入されたところ）
h = normalize(results["chunk_length_s=30 + タイムスタンプ"]["text"])
dups = sorted({h[i:i + 12] for i in range(len(h) - 11) if h.count(h[i:i + 12]) > long_ref.count(h[i:i + 12])})
print(len(dups), "種類:", dups[:4])
```

```python
from silero_vad import load_silero_vad, get_speech_timestamps

# 無音のところで区切り、30 秒未満の区間をまとめてバッチで認識する
vad = load_silero_vad()                                # Silero VAD (MIT): 発話区間を検出するモデル
speech = get_speech_timestamps(torch.from_numpy(long_norm), vad, sampling_rate=SR, max_speech_duration_s=25)
segments = []                                          # 発話区間を、25 秒を超えない範囲でまとめる
for s in speech:
    if segments and s["end"] - segments[-1][0] <= 25 * SR:
        segments[-1][1] = s["end"]
    else:
        segments.append([s["start"], s["end"]])
print(len(speech), "個の発話区間 →", len(segments), "個の区間:", [round((e - s) / SR, 1) for s, e in segments])

t = time.time()
outs = asr([long_norm[s:e] for s, e in segments], batch_size=8, generate_kwargs=GEN, return_timestamps=True)
report("無音で区切ってバッチで認識", "".join(o["text"] for o in outs), time.time() - t)
```

```python
# タイムスタンプ付きの出力を字幕 (SRT) にする
def srt_time(t):
    ms = round(t * 1000)
    h, ms = divmod(ms, 3_600_000); m, ms = divmod(ms, 60_000); s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

subs = []
for (s, e), o in zip(segments, outs):
    for c in o["chunks"]:
        start, end = c["timestamp"]
        end = end if end is not None else (e - s) / SR          # 区間の最後で終わりの時刻が出ないことがある
        subs.append((s / SR + start, s / SR + end, c["text"].strip()))
srt = "\n".join(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n" for i, (a, b, text) in enumerate(subs, 1))
with open("long.srt", "w", encoding="utf-8") as f:
    f.write(srt)
print(len(subs), "行の字幕")
print(srt[:500])
```

```python
# 幻覚と繰り返し: 音声のないところに文字を出す
rng = np.random.default_rng(0)
tests = {"無音 10 秒": np.zeros(10 * SR, np.float32),
         "雑音 10 秒": (rng.standard_normal(10 * SR) * 0.01).astype(np.float32),
         "発話のあとに無音 15 秒": np.concatenate([df["audio"][0], np.zeros(15 * SR, np.float32)])}
for name, x in tests.items():
    text = asr(x, generate_kwargs=GEN)["text"]
    n_speech = len(get_speech_timestamps(torch.from_numpy(x), vad, sampling_rate=SR))
    print(f"{name}: VAD の発話区間 {n_speech} 個 / 認識結果 {len(text)} 文字: {text if len(text) <= 100 else text[:50] + '…'}")
print("正解:", df["raw_transcription"][0])
```

- FLEURS の 135 文（約 29 分）で、CER は large-v3-turbo 0.038、kotoba-whisper-v2.0 0.054、small 0.114 で、差はブートストラップの信頼区間で 0 を含まない。FLEURS は英語の記事を訳した文の読み上げで外国の固有名詞が多く、話し言葉では順位が変わりうる（kotoba のモデルカードでは ReazonSpeech の評価データで large-v3 より CER が低い）。
- kotoba は turbo より速く、small は小さいのに速くなかった。1 トークンずつ生成するデコーダの層数（small 12、turbo 4、kotoba 2）が効くためと考えられる。GPU メモリの最大は 5.7GB（small、バッチ 16）。
- 語の単位で正しく出た割合は、3 字以上のカタカナ語（turbo 0.85、kotoba 0.82、small 0.72）も漢字の語（0.90、0.84、0.67）も 1 − CER より低い。1 字違えば外れになるためで、カタカナ語（外国の固有名詞を含む）だけに誤りが集中するとは言えない。「カタルーニャ語 → 語るうね語」「両替 → 業外」「千年 → 1000年」のような誤りや表記の違いが出る（正規化は評価指標と統計的検定の章の「音声認識や OCR の誤り率 (WER / CER) を計算する」を参照）。
- 約 4 分の音声 1 本で、逐次処理（`return_timestamps=True`）は CER 0.019 だったが、`chunk_length_s=30` で分けると繰り返しや文の脱落で 0.684、タイムスタンプを付けても二重の出力や挿入で 0.187 になった。モデルカードも精度を優先するなら逐次処理、速さなら分割としているが、この例では分割による精度の落ち方が大きい。
- 音量が 40dB ほど小さい 3 文のうち 2 文は丸ごと抜けた。ffmpeg の dynaudnorm で音量をそろえると、どの文も 9 割以上の文字が出た。
- Silero VAD で無音のところで区切ってバッチで認識すると、CER 0.022 と逐次処理に近く、時間は短かった。区間の開始時刻を足せば字幕 (SRT) になる。句読点が要るなら日本語の形態素解析と文分割の章の「句読点のない文章に読点と句点を付ける（句読点の復元）」を参照。
- 無音や雑音にも文字を出す（幻覚）が、VAD はどちらも発話区間 0 個と判定したので、VAD で区切れば防げる。発話のあとの無音では、この例では幻覚は出なかった。
- 書き起こしの先は、固有表現抽出の章の「音声認識の結果から固有表現を抽出する」と、要約と平易化の章の「会議の書き起こしから議事録を作る（決定事項・ToDo を JSON で抜き出す）」を参照。議事録には pyannote などの話者分離を組み合わせる（モデルの利用条件への同意と HF のトークンが要る。kotoba-whisper-v2.2 は話者分離を組み込んだパイプライン）。字幕を作って翻訳する例は、機械翻訳の章の「動画の字幕を作って翻訳する（区切り直し、文脈、文字数の制約）」を参照。

## テキストデータの重複と不良データを取り除く {dedup_clean}

集めたテキストに、完全な重複、空白や大文字小文字だけが違う重複、短すぎる・長すぎる行、文字化けが混ざっている。学習前に掃除したい。

```python
import re
import unicodedata
import pandas as pd

raw = ["今日は良い天気です。", "今日は良い天気です。", "今日は　良い天気です。 ", "ＮＬＰは楽しい", "NLPは楽しい", "", "あ", "!!!!!!!!!!!!",
       "縺薙ｓ縺ｫ縺｡縺ｯ荳也阜", "https://example.com/only-url", "機械翻訳の品質は近年大きく向上した。", "This line is English only.",
       "長い文" * 300]
df = pd.DataFrame({"text": raw})

def normalize_key(s):
    """重複判定用のキー: 正規化して空白と記号を除く (原文は書き換えない)"""
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[\s\W_]+", "", s)

def ja_ratio(s):
    return len(re.findall(r"[ぁ-んァ-ヶ一-龥]", s)) / max(1, len(s))

def looks_mojibake(s):
    return len(re.findall(r"[縺繧繝荳螟譁蜿]", s)) / max(1, len(s)) > 0.2   # UTF-8 を Shift_JIS と誤解したときに頻出する文字

df["key"] = df["text"].map(normalize_key)
checks = {
    "空・短すぎる": df["key"].str.len() < 3,
    "長すぎる": df["text"].str.len() > 500,
    "日本語が少ない": df["text"].map(ja_ratio) < 0.3,
    "文字化け": df["text"].map(looks_mojibake),
    "重複": df.duplicated("key", keep="first"),
}
df["reason"] = ""
for name, mask in checks.items():
    df.loc[mask & (df["reason"] == ""), "reason"] = name
print(df["reason"].replace("", "(残す)").value_counts())
df[df["reason"] == ""]["text"].tolist()
```

- 何を何件捨てたかを理由ごとに記録しておく。フィルタのバグでデータの大半を捨てていた、という事故に気づける。
- 学習データと評価データの間の重複（リーク）も必ず確認する: `set(train_keys) & set(test_keys)`。
- ほぼ同じ内容の文書（ニアデュプリケート）は、この章の「MinHash と LSH で大量文書からほぼ重複を見つける」で検出する。

## MinHash と LSH で大量文書からほぼ重複を見つける {minhash_lsh}

数十万件の文書から、コピペや微修正による「ほぼ同じ文書」を、全ペア比較せずに高速に見つけたい。

```python
!pip install -q datasketch
```

```python
from datasketch import MinHash, MinHashLSH

def shingles(text, n=3):
    text = "".join(text.split())
    return {text[i:i + n] for i in range(len(text) - n + 1)}

def minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for s in shingles(text):
        m.update(s.encode("utf-8"))
    return m

docs = {
    "a": "自然言語処理は、人間が日常的に使っている言語をコンピュータに処理させる一連の技術である。",
    "b": "自然言語処理とは、人間が日常的に使っている言語をコンピューターに処理させる技術のことである。",
    "c": "機械学習は、データから規則性を学習してタスクを解くアルゴリズムの研究分野である。",
    "d": "自然言語処理は、人間が日常的に使っている言語をコンピュータに処理させる一連の技術である！",
}
mh = {k: minhash(v) for k, v in docs.items()}

def jaccard(a, b):
    A, B = shingles(docs[a]), shingles(docs[b])
    return len(A & B) / len(A | B)

for x, y in [("a", "b"), ("a", "c"), ("a", "d")]:
    print(f"{x}-{y}: 真の Jaccard = {jaccard(x, y):.2f}  MinHash による推定 = {mh[x].jaccard(mh[y]):.2f}")
```

```python
lsh = MinHashLSH(threshold=0.5, num_perm=128)
for k, m in mh.items():
    lsh.insert(k, m)
for k in docs:
    print(k, "と似ている:", sorted(set(lsh.query(mh[k])) - {k}))
```

- MinHash は集合の Jaccard 係数を固定長の署名で近似する。LSH は署名が似たものだけを同じバケットに入れ、候補を絞る。
- 大規模な事前学習コーパスの重複除去で標準的に使われている。意味的な重複（言い換え）は拾えないので、それには埋め込みを使う。

## 事前学習用の Web テキストを品質で選ぶ（規則によるフィルタと、LLM の採点で学習した品質分類器） {pretrain_data_filter gpu}

Web から集めた日本語テキストを LLM の事前学習に使う前に、品質の低い文書を除きたい。(1) FineWeb-2 が日本語用に調整した規則（繰り返しの多さ、句読点で終わる行の割合、記号の多さなど）と、(2) FineWeb-Edu の方式の品質分類器（LLM に一部の文書の教育的な価値を採点させ、その点数を文の埋め込みからの回帰で学習して全体に当てる）で選び、それぞれで捨てられた文書、残る割合、2 つの方法の一致を確かめる。

（T4 の GPU ランタイムで、モデルのダウンロードを含めて全体で 20〜25 分程度かかる。）

```python
# datatrove: Hugging Face のデータ処理ライブラリ（FineWeb の作成に使われた）。日本語の語の分割に spaCy + SudachiPy を使う。fasteners は語のリストのダウンロードに要る
!pip install -q datatrove==0.10.1 sudachipy==0.7.0 sudachidict_core==20261015 fasteners==0.20
```

```python
import numpy as np
import pandas as pd
from datasets import load_dataset

# mC4 の日本語（allenai/c4 の multilingual、ODC-BY。Common Crawl の利用規約にも従う）の検証用の 1 ファイル（約 40MB）から 3,000 文書
web = load_dataset("allenai/c4", data_files="multilingual/c4-ja-validation.tfrecord-00000-of-00008.json.gz", split="train")
docs = web.shuffle(seed=0).select(range(3000))
df = pd.DataFrame({"text": docs["text"], "url": docs["url"]})
print(len(web), "文書から", len(df), "文書。文字数:", df["text"].str.len().describe().round(0).to_dict())

def short(s, n=70):
    """表示用: 改行を / にして先頭 n 文字だけにする"""
    return s.replace("\n", " / ")[:n]
```

```python
# (1) 規則によるフィルタ。datatrove のフィルタに、FineWeb-2 が日本語 (jpn_Jpan) 用に調整した閾値を渡す
# （閾値は huggingface/fineweb-2 の configs/jpn_Jpan.yml と fineweb-2-pipeline.py、Apache 2.0）
# 最初に引っかかった規則で止めずに、すべての規則を全文書に当てて理由を記録する（Colab で 5〜8 分）
from datatrove.data import Document
from loguru import logger
from datatrove.pipeline.filters import C4BadWordsFilter, FineWebQualityFilter, GopherQualityFilter, GopherRepetitionFilter

logger.remove()      # datatrove のログ（語のリストのダウンロードの表示など）を出さない
filters = {
    # C4 の不適切な語のリスト（LDNOOBW の日本語、CC BY 4.0）の語を含む文書を除く。FineWeb-2 は使っていない
    "不適切な語": C4BadWordsFilter(default_language="ja"),
    # Gopher の繰り返しの規則: 重複行の割合、最頻の n-gram (n=2〜4) と重複する n-gram (n=5〜10) が占める文字の割合
    "繰り返し": GopherRepetitionFilter(
        language="ja", dup_para_frac=0, dup_line_char_frac=0, dup_para_char_frac=0, dup_line_frac=0.328,
        top_n_grams=((2, 0.239), (3, 0.196), (4, 0.172)),
        dup_n_grams=((5, 0.243), (6, 0.225), (7, 0.207), (8, 0.19), (9, 0.175), (10, 0.159))),
    # FineWeb の規則: 句読点で終わる行の割合、重複行の文字の割合、改行の多さ（FineWeb-2 は短い行の規則を外している）
    "行の形": FineWebQualityFilter(language="ja", short_line_thr=999, char_duplicates_ratio=0.1, line_punct_thr=0.096, new_line_ratio=0.13),
    # Gopher の品質の規則: 語数、平均の語の長さ、記号（# と …）の多さ、文字を含む語の割合、よく使われる語（の、に、を…）を 2 種類以上含むか
    "語の統計": GopherQualityFilter(
        language="ja", min_avg_word_length=1, max_avg_word_length=6, max_non_alpha_words_ratio=0.759, min_stop_words=2,
        stop_words=["の", "に", "を", "は", "た", "て", "が", "と", "で", "年", "し", "・", "月", "れ", "さ"]),
}

def apply(f, i, text):
    r = f.filter(Document(text=text, id=str(i)))
    return "" if r is True else r[1]          # 通れば "", 落ちれば理由の名前

R = pd.DataFrame({k: [apply(f, i, t) for i, t in enumerate(df["text"])] for k, f in filters.items()})
df["rule_keep"] = (R == "").all(axis=1)
chars = df["text"].str.len()
print(f"規則をすべて通った文書: {df['rule_keep'].sum()} / {len(df)} ({df['rule_keep'].mean():.1%})、文字数では {chars[df['rule_keep']].sum() / chars.sum():.1%}")
print(pd.DataFrame({"落ちた文書": (R != "").sum(), "その規則だけで落ちた文書": ((R != "").sum(axis=1).eq(1).values[:, None] & (R != "")).sum()}))
for k in R:
    print(k, R[k].value_counts().drop("", errors="ignore").to_dict())
```

```python
from collections import Counter

# 不適切な語の規則で、どの語に一致したか（日本語は語の境界を見ずに部分文字列で探す）
bad_re = filters["不適切な語"]._get_badwords("ja")
hits = Counter(w for t in df.loc[R["不適切な語"] != "", "text"] for w in set(bad_re.findall(t.lower())))
print(f"一致した語は {len(hits)} 種類。上位の多くは成人向けの語なので、ここでは表示しない")
print("普通の語で一致した文書の数:", {w: hits[w] for w in ["嫌い", "支配", "ラバー"]})

# 理由ごとに、落ちた文書の例を見る。表示する例は、不適切な語に引っかからず、成人向けの語も含まない文書から選ぶ（表示のためだけ）
clean_view = (R["不適切な語"] == "") & ~df["text"].str.contains("エロ|アダルト|無修正|風俗")
for k in ["繰り返し", "行の形", "語の統計"]:
    for reason, n in R[k].value_counts().items():
        if reason and n >= 10:
            print(f"--- {k}: {reason}（{n} 文書）")
            for t in df.loc[(R[k] == reason) & clean_view, "text"].head(2):
                print("   ", short(t))
print("--- 規則をすべて通った文書（無作為に 6 件）")
for t in df.loc[df["rule_keep"]].sample(6, random_state=0)["text"]:
    print("   ", short(t))
```

```python
# (2) 品質分類器の学習データ: 1,000 文書を LLM に 0〜5 点で採点させる（FineWeb-Edu の論文の付録 F.1 にある加点方式のプロンプトを、簡略化して日本語にしたもの）
# LLM: Qwen3-4B-Instruct-2507（Alibaba の Qwen チーム、Apache 2.0）。T4 で 8〜12 分
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # T4 は bf16 が使えないので float16
name = "Qwen/Qwen3-4B-Instruct-2507"
tok = AutoTokenizer.from_pretrained(name)
tok.padding_side = "left"
lm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()

PROMPT = """以下は Web ページから抜き出した文章です。小学校から中学校までの学習に役立つ、教育的な価値があるかを、次の加点方式で 0〜5 点で採点してください。

- 教育的な話題について基本的な情報を含んでいれば 1 点を加える（広告や宣伝など無関係な内容が混じっていてもよい）。
- 教育に関係する要素を扱っているが、学校の基準にはそろっておらず、教育的でない内容が混じっていたり、まとまりが悪かったりする場合は、さらに 1 点を加える。
- 教育に使うのに適していて、学校の教育課程に関係する重要な概念を紹介していれば、3 点目を加える。説明が網羅的でなくても、余計な情報が含まれていてもよい。
- 小学校から中学校の学習に非常に関係が深く役立ち、筋の通った明快な書き方をしていれば、4 点目を加える。教科書の一節やチュートリアルのような内容で、余計な内容がほとんどない。
- 小学校から中学校の教育に非常に優れていて、詳しい説明があり、余計な内容がまったくなければ、5 点目を加える。

文章:
{text}

点数を 0〜5 の整数 1 つだけで答えてください。"""
digit_ids = [tok.convert_tokens_to_ids(str(k)) for k in range(6)]     # 「0」〜「5」はそれぞれ 1 トークン

@torch.no_grad()
def llm_score(texts, batch_size=8):
    """次のトークンが「0」〜「5」になる確率から、点数の期待値を返す（文書は先頭 1,000 文字だけを見せる）"""
    prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(text=t[:1000])}], add_generation_prompt=True, tokenize=False)
               for t in texts]
    order = np.argsort([len(p) for p in prompts])      # 長さの近いものをまとめてパディングを減らす
    scores, mass = np.zeros(len(texts)), np.zeros(len(texts))
    for i in range(0, len(texts), batch_size):
        idx = order[i:i + batch_size]
        enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True).to(lm.device)
        p = lm(**enc, logits_to_keep=1).logits[:, -1].float().softmax(-1)[:, digit_ids].cpu().numpy()
        mass[idx] = p.sum(1)                            # 数字以外のトークンに確率が流れていないかの確認用
        scores[idx] = (p / p.sum(1, keepdims=True)) @ np.arange(6)
    return scores, mass

ann = df.sample(1000, random_state=0).index           # 採点する文書（全体から無作為に選ぶ）
df.loc[ann, "llm"], mass = llm_score(list(df.loc[ann, "text"]))
print(f"数字のトークンの確率の合計: 中央値 {np.median(mass):.3f}、最小 {mass.min():.3f}")
print("点数（四捨五入）の分布:", df.loc[ann, "llm"].round().astype(int).value_counts().sort_index().to_dict())
print(f"GPU メモリの最大: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

```python
a = df.loc[ann].sort_values("llm")
print("--- LLM の点数が高い文書")
for _, r in a[clean_view[a.index]].tail(8)[::-1].iterrows():
    print(f"{r.llm:4.2f}  {short(r.text)}")
print("--- 規則をすべて通ったが、LLM の点数が 0.5 未満の文書（無作為に 6 件）")
for _, r in a[clean_view[a.index] & a["rule_keep"] & (a["llm"] < 0.5)].sample(6, random_state=0).iterrows():
    print(f"{r.llm:4.2f}  {short(r.text)}")
```

```python
# LLM の点数を、文の埋め込み + リッジ回帰で学習する。5 分割の交差検証で、採点していない文書への当てはまりを確かめる
import gc
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict

lm = None; gc.collect(); torch.cuda.empty_cache()      # LLM の GPU メモリを空ける
st = SentenceTransformer("intfloat/multilingual-e5-small", device="cuda")   # MIT ライセンス。最大 512 トークン
X = st.encode(["passage: " + t[:1000] for t in df["text"]], normalize_embeddings=True, batch_size=128)   # LLM に見せた範囲（先頭 1,000 文字）。e5 はさらに 512 トークンで切る

# 「残す」文書: LLM の点数を四捨五入して 2 点以上（FineWeb-Edu は 3 点以上。ここで 2 にした理由は注意書き）
y = df.loc[ann, "llm"].to_numpy()
good = y >= 1.5
reg = RidgeCV(alphas=np.logspace(-2, 2, 9))
oof = cross_val_predict(reg, X[ann], y, cv=KFold(5, shuffle=True, random_state=0))
print(f"交差検証の予測: 最大 {oof.max():.2f}、スピアマンの順位相関 {spearmanr(y, oof).statistic:.3f}、"
      f"2 点以上（{good.sum()} 文書）の判定の ROC AUC {roc_auc_score(good, oof):.3f}")
# 回帰の予測は平均（ほとんどが 0 点）の側に縮むので、1.5 では切れない。交差検証の予測で F1 が最大になる閾値を選ぶ
prec, rec, ths = precision_recall_curve(good, oof)
f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
best = f1[:-1].argmax()
THRESHOLD = ths[best]
print(f"分類器の閾値 {THRESHOLD:.2f}: 適合率 {prec[best]:.3f}、再現率 {rec[best]:.3f}、F1 {f1[best]:.3f}")

reg.fit(X[ann], y)                                     # 採点した 1,000 文書すべてで学習し直し、3,000 文書に当てる
df["clf"] = reg.predict(X)
df["clf_keep"] = df["clf"] >= THRESHOLD
print(f"分類器で残る文書: {df['clf_keep'].sum()} / {len(df)} ({df['clf_keep'].mean():.1%})、"
      f"文字数では {chars[df['clf_keep']].sum() / chars.sum():.1%}（採点した 1,000 文書で LLM が 2 点以上: {good.mean():.1%}）")
```

```python
# 規則と分類器の一致
from sklearn.metrics import cohen_kappa_score

print(pd.crosstab(df["rule_keep"].map({True: "規則: 残す", False: "規則: 捨てる"}),
                  df["clf_keep"].map({True: "分類器: 残す", False: "分類器: 捨てる"}), margins=True, margins_name="計"))
print(f"一致率 {(df['rule_keep'] == df['clf_keep']).mean():.3f}、カッパ係数 {cohen_kappa_score(df['rule_keep'], df['clf_keep']):.3f}")
rk = df.loc[ann, "rule_keep"]
print(f"採点した 1,000 文書で、LLM が 2 点以上の割合: 規則で残す文書 {good[rk].mean():.1%}、規則で捨てる文書 {good[~rk].mean():.1%}、"
      f"LLM の判定と規則のカッパ係数 {cohen_kappa_score(rk, good):.3f}")
both = df["rule_keep"] & df["clf_keep"]
print(f"両方で残す: {both.sum()} 文書 ({both.mean():.1%})、文字数では {chars[both].sum() / chars.sum():.1%}")

# 規則で落ちた理由ごとの、分類器の点数の平均（採点した文書では LLM の点数も）
# 複数の規則で落ちた文書は、最初に引っかかった規則（filters の順）に数える。10 文書以上の理由だけを表示する
reason = R.apply(lambda row: next((f"{k}: {v}" for k, v in row.items() if v), "（規則をすべて通る）"), axis=1)
g = df.assign(reason=reason).groupby("reason")
print(pd.DataFrame({"文書数": g.size(), "分類器の点数": g["clf"].mean(), "LLM の点数（採点した文書）": g["llm"].mean()}).query("文書数 >= 10").round(2).sort_values("分類器の点数").to_string())

print("--- 規則では捨てるが、分類器の点数が高い文書")
for i in df[~df["rule_keep"] & clean_view].nlargest(6, "clf").index:
    print(f"{df.clf[i]:4.2f}  [{reason[i]}]  {short(df.text[i], 60)}")
```

- この章の「テキストデータの重複と不良データを取り除く」の次の段階にあたる。そこで完全な重複や明らかな不良を除き（大量の文書のほぼ重複は、この章の「MinHash と LSH で大量文書からほぼ重複を見つける」）、ここで文書の品質で選ぶ。規則は FineWeb-2（Penedo ら 2025）が日本語用に調整した閾値を datatrove に渡したもので、繰り返しの規則と語の統計の規則は Gopher（Rae ら 2021）、行の形の規則は FineWeb（Penedo ら 2024）にならっている。FineWeb-2 は C4（Raffel ら 2020）の規則を使っていないが、ここでは C4 の不適切な語のリストを加えた。
- 規則をすべて通ったのは 3,000 文書のうち 904 文書（30.1%、文字数では 24.0%）だった。落とした数が多いのは、数字や記号だけの語が多すぎる（gopher_below_alpha_threshold、1,047 文書）、句読点で終わる行が 9.6% 未満（line_punct_ratio、879 文書）、5 語の連なりの重複が多い（duplicated_5_n_grams、374 文書）で、商品名や型番の羅列、メニューや見出しだけの行が並ぶページ、中国語のページが落ちた。mC4 はページの行をそのまま残している（Xue ら 2021 は C4 の句読点の規則の代わりに、200 文字以上の行が 3 行以上あることを求めた）。trafilatura で本文を取り出した FineWeb-2 のテキストに合わせた閾値なので、このデータでは多めに落ちている可能性がある（FineWeb-2 のデータでは確かめていない）。閾値は、落ちた例と残った例を見て決め直す。
- 不適切な語の規則は 762 文書 (25.4%)を落とし、うち 368 文書はこの規則だけで落ちた。日本語は語の境界を見ずに部分文字列で探すので、「嫌い」（121 文書）、「支配」（39 文書）、「ラバー」（33 文書）のような普通の語でも落ちる。datatrove も「sm」「グロ」「女の子」は落としすぎるとして除外している。使うなら、一致した語を数えてリストを見直す。
- 規則をすべて通った文書にも、「中郡大磯町で切手買取りの優良業者とは？」や「ホワイトニング 効果なしでできるダイエット」のような、地名や語を差し替えて量産された検索エンジン向けのページが残った。文章の形が整っていれば規則は通るので、内容は分類器で見る。
- LLM の採点は FineWeb-Edu（Penedo ら 2024）の方式にならった。FineWeb-Edu は Llama-3-70B-Instruct に 46 万ページを 0〜5 点で採点させ（点数の理由も書かせる）、Snowflake-arctic-embed-m に回帰の出力層を付けて学習し、四捨五入して 3 点以上を残した（3 点の判定で F1 82%）。ここでは T4 に載る Qwen3-4B-Instruct-2507 に 1,000 文書を採点させ、理由を書かせずに次のトークンの確率から期待値をとった（数字のトークンの確率の合計は最小でも 0.912）。点数は 0 点に偏り（889 文書）、3 点以上は 23 文書 (2.3%)しかなかったので、2 点以上（55 文書、5.5%）を残す側にした。中学理科の「消化と吸収」や地震計の解説が 4 点前後になった一方、「総ビリルビン」の用語解説は 0 点だった。4B の LLM の採点は、人や大きな LLM の判断と比べて確かめていない。
- 埋め込み + リッジ回帰の分類器は、5 分割の交差検証でスピアマンの順位相関 0.601、2 点以上の判定の ROC AUC 0.916 だった。ほとんどが 0 点のデータで回帰すると予測が 0 の側に縮み（交差検証の予測の最大は 1.67）、LLM と同じ 1.5 では切れないので、F1 が最大になる 0.70 を閾値にした（適合率 0.418、再現率 0.600、F1 0.493）。分類器では 3,000 文書のうち 240 文書（8.0%、文字数では 9.3%）が残った。同じ予測で閾値を選んだので F1 は甘めで、正例も 55 文書しかない。採点する文書を増やす（FineWeb-Edu の学習データは 41 万件）と改善の余地がある。
- 規則と分類器の一致率は 0.704 だが、偶然の一致を除いたカッパ係数は 0.111 と低く、両方で残すのは 128 文書（4.3%、文字数では 3.7%）だった。方向はそろっていて、規則で落ちた文書は理由によらず分類器の点数の平均が 0.04〜0.18 と、規則を通った文書 (0.32)より低く、採点した文書で LLM が 2 点以上の割合も規則で残す文書で 9.6%、捨てる文書で 3.6% だった。規則で捨てるが分類器の点数が高い文書には、読書感想文の課題図書の紹介や Weblio の「円周率の日」があり、先頭を見た範囲では、パンくずやメニューの行が多く、句読点や繰り返しの規則で落ちていた。安い規則で明らかな不良を除いてから分類器を当てるのが普通の順序だが、規則が良い文書も捨てていることは、こうして確かめる。
- 教育的な価値は、事前学習に役立つ品質の見方の 1 つにすぎない。言語モデルのパープレキシティで選ぶ方法は、コーパスの統計と可視化の章の「KenLM で n-gram 言語モデルを作り、パープレキシティでコーパスのノイズを見つける」を参照（同じ mC4 のファイルを使っている）。ライセンスは、mC4 が ODC-BY（Common Crawl の利用規約にも従う）、Qwen3-4B-Instruct-2507 が Apache 2.0、multilingual-e5-small が MIT、datatrove と FineWeb-2 の設定が Apache 2.0、LDNOOBW の語のリストが CC BY 4.0。GPU メモリの最大は 10.5 GB（T4 と同じ float16 で実行した値）。

## データを学習・開発・評価に正しく分割する {data_split}

データをランダムに分けるだけでは、クラスの偏りや、同じ文書・同じ書き手由来のデータが両側に入るリークが起きる。適切な分割をしたい。

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split

rng = np.random.default_rng(0)
n = 1000
df = pd.DataFrame({
    "text": [f"文 {i}" for i in range(n)],
    "label": rng.choice(["pos", "neg", "neu"], n, p=[0.6, 0.3, 0.1]),
    "author": rng.integers(0, 50, n),          # 同じ書き手の文が複数ある (学習者コーパス、レビュー、対話など)
})

# 1. 層化分割: クラスの比率を各分割で保つ (8:1:1)
train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=0)
dev, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=0)
print(pd.DataFrame({name: d["label"].value_counts(normalize=True).round(3) for name, d in [("train", train), ("dev", dev), ("test", test)]}))
print("書き手の重なり (train と test):", len(set(train["author"]) & set(test["author"])), "人")
```

```python
# 2. グループ分割: 同じ書き手のデータが学習と評価にまたがらないようにする
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(gss.split(df, groups=df["author"]))
print("書き手の重なり:", len(set(df["author"].iloc[train_idx]) & set(df["author"].iloc[test_idx])), "人")

# 3. 層化 + グループの交差検証
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
for fold, (tr, te) in enumerate(cv.split(df, df["label"], groups=df["author"])):
    print(fold, len(tr), len(te), df["label"].iloc[te].value_counts(normalize=True).round(2).to_dict())
```

```python
# 分割結果は ID で保存して固定する (毎回ランダムに分け直さない)
df["split"] = "train"
df.loc[df.index[test_idx], "split"] = "test"
df[["text", "split"]].head(3)
```

- 同じ文書から切り出した文、同じ人が書いた文、同じ原文の翻訳や言い換えは、必ず同じ側に入れる。両側にまたがると評価が不当に高くなる。
- 時系列のあるデータ（ニュース、SNS）は、ランダムではなく時間で分割する（過去で学習、未来で評価）。
- test は最後に 1 回だけ使う。ハイパーパラメータの調整やモデル選択は dev で行う。
