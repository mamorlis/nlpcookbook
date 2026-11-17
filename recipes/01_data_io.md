# データの収集と入出力

テキストを集める、ファイルや音声から取り出す、きれいにして分割する。実験の前にやること。JSON Lines や CSV の読み書きのようなファイル操作の小技は、付録の「Colab と Python の小技」を参照。

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

## Wikipedia の記事本文を API で取得する {wikipedia_api norun}

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
- 書き起こしの先は、固有表現抽出の章の「音声認識の結果から固有表現を抽出する」と、要約と平易化の章の「会議の書き起こしから議事録を作る（決定事項・ToDo を JSON で抜き出す）」を参照。議事録には pyannote などの話者分離を組み合わせる（モデルの利用条件への同意と HF のトークンが要る。kotoba-whisper-v2.2 は話者分離を組み込んだパイプライン）。

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
- ほぼ同じ内容の文書（ニアデュプリケート）は、文字列処理と正規化の章の MinHash のレシピで検出する。

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
