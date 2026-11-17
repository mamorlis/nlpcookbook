# データの収集と入出力

テキストを集める、ファイルから取り出す、きれいにして分割する。実験の前にやること。JSON Lines や CSV の読み書きのようなファイル操作の小技は、付録の「Colab と Python の小技」を参照。

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
- 全記事が必要なら API を叩かずに、ダンプか Hugging Face の `wikimedia/wikipedia` データセットを使う (Transformers の基本操作の章のストリーミングのレシピを参照)。
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
- スクレイピングの前に、サイトの利用規約と robots.txt を確認する。アクセス間隔を空け、著作権 (特に再配布) に注意する。

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
- スキャンされた PDF (画像) からは抽出できない。OCR が必要 (`page.get_textpage_ocr()` や、専用の OCR / 文書解析モデル)。
- 論文の構造 (章、参考文献) まで欲しいなら GROBID、レイアウトを保った Markdown が欲しいなら `pymupdf4llm` や Docling などがある。

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

- 見出しはスタイル名 (`Heading 1`、日本語版 Word で作ると `見出し 1`) で判別できる。
- 古い `.doc` 形式は読めない。LibreOffice で変換する (`!soffice --headless --convert-to docx file.doc`)。
- PowerPoint は `python-pptx`、Excel は `pandas.read_excel` (openpyxl) で同様に読める。

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
- 学習データと評価データの間の重複 (リーク) も必ず確認する: `set(train_keys) & set(test_keys)`。
- ほぼ同じ内容の文書 (ニアデュプリケート) は、文字列処理と正規化の章の MinHash のレシピで検出する。

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
- 時系列のあるデータ (ニュース、SNS) は、ランダムではなく時間で分割する (過去で学習、未来で評価)。
- test は最後に 1 回だけ使う。ハイパーパラメータの調整やモデル選択は dev で行う。
