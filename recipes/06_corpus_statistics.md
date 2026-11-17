# コーパスの統計と可視化

単語を数える、共起や特徴語を調べる、分布を図にする、文体から書き手を推定する、n-gram 言語モデルでノイズの混じった文を見つける。コーパスを手に入れたらまずやること。

## 文の長さや文字数の基本統計をとる {corpus_stats}

コーパスの規模感（文数、平均文長、長さの分布）をまず把握したい。

```python
!pip install -q fugashi unidic-lite
```

```python
import io, re, urllib.request, zipfile
import pandas as pd
from fugashi import Tagger

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

tagger = Tagger()
text = aozora("https://www.aozora.gr.jp/cards/000148/files/752_ruby_2438.zip")
sents = [s.strip() for s in re.split(r"(?<=[。？！])|\n", text) if s and s.strip()]
df = pd.DataFrame({"sentence": sents})
df["chars"] = df["sentence"].str.len()
df["words"] = df["sentence"].map(lambda s: len(tagger(s)))
df[["chars", "words"]].describe(percentiles=[0.5, 0.9, 0.99]).round(1)
```

```python
ax = df["words"].clip(upper=100).hist(bins=50)
ax.set_xlabel("words per sentence"); ax.set_ylabel("sentences")

print("最長の文:", df.loc[df["words"].idxmax(), "sentence"][:80], "...")
print("最短の文:", df.nsmallest(5, "chars")["sentence"].tolist())
```

- 極端に長い文・短い文は、文分割の失敗やノイズであることが多い。最大・最小の実例は必ず目で見る。

## 単語の頻度を数える {word_freq}

テキスト中でどの単語が何回出てくるかを数えて、頻度順に並べたい。

```python
!pip install -q fugashi unidic-lite
```

```python
import io, re, urllib.request, zipfile

def aozora(url):
    """青空文庫の zip (ルビあり) を取得して本文だけを返す"""
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

text = aozora("https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip")  # 吾輩は猫である
print(len(text), "文字")
```

```python
from collections import Counter
from fugashi import Tagger

tagger = Tagger()
# 注意: fugashi のノードは次の解析で上書きされるので、必要な属性をその場で取り出して保存する
words = [(w.surface, w.feature.pos1, w.feature.pos2, w.feature.orthBase or w.surface)
         for line in text.split("\n") for w in tagger(line)]
print("延べ語数:", len(words))

freq_all = Counter(surface for surface, *_ in words)
print(freq_all.most_common(15))

# 名詞だけ (原形で集計)
freq_noun = Counter(base for surface, pos1, pos2, base in words if pos1 == "名詞" and pos2 in ("普通名詞", "固有名詞"))
print(freq_noun.most_common(15))
```

```python
import pandas as pd

df = pd.DataFrame(freq_noun.most_common(), columns=["word", "freq"])
df["ratio"] = df["freq"] / df["freq"].sum()
df["cumulative"] = df["ratio"].cumsum()
df.head(10)
```

- 何も絞らないと上位は助詞と句読点で埋まる。見たいものに応じて品詞で絞る。
- `[w for line in lines for w in tagger(line)]` のようにノードそのものを溜め込むと、中身が壊れて品詞や原形がでたらめになる（fugashi の落とし穴）。

## matplotlib で日本語を文字化けさせずに表示する {matplotlib_ja}

グラフの軸ラベルや凡例の日本語が □□□（豆腐）になるのを直したい。

```python
!pip install -q matplotlib-fontja
```

```python
import matplotlib.pyplot as plt
import matplotlib_fontja  # import するだけで IPAex ゴシックが既定のフォントになる

words, counts = ["猫", "主人", "人間", "先生", "迷亭"], [270, 933, 280, 250, 340]
plt.bar(words, counts)
plt.title("単語の出現頻度"); plt.xlabel("単語"); plt.ylabel("頻度")
plt.show()
```

```python
# seaborn を使う場合は、set_theme がフォントを上書きするので後からもう一度設定する
import seaborn as sns

sns.set_theme(style="whitegrid")
matplotlib_fontja.japanize()
sns.barplot(x=words, y=counts)
plt.title("seaborn でも日本語")
plt.show()
```

- 以前の定番 `japanize-matplotlib` は Python 3.12 以降で動かない（distutils 廃止のため）。`matplotlib-fontja` はその後継。
- 別のフォントを使いたいときは `!apt-get -y install fonts-noto-cjk` して `plt.rcParams["font.family"] = "Noto Sans CJK JP"`。

## Zipf の法則を確かめる（順位-頻度プロット） {zipf}

単語の頻度が順位にほぼ反比例すること（Zipf の法則）を、両対数グラフで確かめたい。

```python
!pip install -q fugashi unidic-lite
```

```python
import io, re, urllib.request, zipfile
from collections import Counter
from fugashi import Tagger

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

text = aozora("https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip")
tagger = Tagger()
freq = Counter(w.surface for line in text.split("\n") for w in tagger(line))
print("異なり語数:", len(freq), "延べ語数:", sum(freq.values()))
```

```python
import matplotlib.pyplot as plt
import numpy as np

counts = np.array(sorted(freq.values(), reverse=True))
ranks = np.arange(1, len(counts) + 1)
slope, intercept = np.polyfit(np.log(ranks), np.log(counts), 1)  # 両対数で直線回帰

plt.loglog(ranks, counts, ".", markersize=3, label="observed")
plt.loglog(ranks, np.exp(intercept) * ranks ** slope, label=f"slope = {slope:.2f}")
plt.xlabel("rank"); plt.ylabel("frequency"); plt.legend(); plt.show()

hapax = sum(1 for c in counts if c == 1)
print(f"1 回しか出ない語 (hapax): {hapax} ({hapax / len(counts):.0%})")
```

- 傾きが -1 付近なら Zipf の法則どおり。語彙の半分前後は 1 回しか出現しない、というのも普遍的な性質。

## 語彙の豊富さを測る（TTR とその仲間） {lexical_diversity}

テキストがどれくらい多様な語彙を使っているかを数値化したい（作家の比較、学習者の習熟度の推定など）。

```python
!pip install -q fugashi unidic-lite
```

```python
import io, math, re, urllib.request, zipfile
from fugashi import Tagger

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

tagger = Tagger()

def tokens(text):
    return [w.feature.lemma or w.surface for line in text.split("\n") for w in tagger(line) if w.feature.pos1 != "補助記号"]

def mattr(toks, window=500):
    """移動平均 TTR: 一定幅の窓ごとの TTR を平均する (テキスト長の影響を受けにくい)"""
    if len(toks) < window:
        return len(set(toks)) / len(toks)
    return sum(len(set(toks[i:i + window])) for i in range(len(toks) - window + 1)) / (len(toks) - window + 1) / window

base = "https://www.aozora.gr.jp/cards/"
works = {"吾輩は猫である": "000148/files/789_ruby_5639.zip", "坊っちゃん": "000148/files/752_ruby_2438.zip",
         "走れメロス": "000035/files/1567_ruby_4948.zip", "銀河鉄道の夜": "000081/files/456_ruby_145.zip"}
print(f"{'':14}{'延べ':>8}{'異なり':>7}{'TTR':>7}{'Guiraud':>9}{'MATTR':>7}")
for title, path in works.items():
    t = tokens(aozora(base + path))
    N, V = len(t), len(set(t))
    print(f"{title:　<8}{N:>9}{V:>9}{V / N:>8.3f}{V / math.sqrt(N):>8.1f}{mattr(t):>8.3f}")
```

- 素の TTR（異なり語数 ÷ 延べ語数）はテキストが長いほど小さくなるため、長さの違うテキストの比較には使えない。MATTR や MTLD を使う。

## 文字や単語のエントロピーを計算する {entropy}

テキストの情報量（1 文字あたり何ビットか）を計算して、言語やテキストの冗長性を比べたい。

```python
import math
from collections import Counter

def entropy(seq):
    counts = Counter(seq)
    total = sum(counts.values())
    return -sum(c / total * math.log2(c / total) for c in counts.values())

def conditional_entropy(seq, n=2):
    """直前の n-1 個を知っているときの次の 1 個のエントロピー H(X_n | X_1..X_{n-1})"""
    grams = list(zip(*[seq[i:] for i in range(n)]))
    return entropy(grams) - entropy([g[:-1] for g in grams])

ja = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。" * 3
en = "I am a cat. As yet I have no name. I have no idea where I was born. All I remember is that I was miaowing in a dampish dark place. " * 3

for name, text in [("日本語", ja), ("英語", en)]:
    print(f"{name}: 文字種 {len(set(text)):3}  H(1文字) = {entropy(text):.2f} bit  H(次の文字|直前の文字) = {conditional_entropy(list(text)):.2f} bit")
```

- 文脈を長くするほど条件付きエントロピーは下がる。その極限がその言語のエントロピーレートで、言語モデルの交差エントロピー（= log2 パープレキシティ）はその上界になる。
- 短いテキストでは過小評価になる。実際の推定には大きなコーパスが必要。

## 品詞の分布を調べる・比べる {pos_distribution}

テキストの品詞構成（名詞が多い、動詞が多い…）を調べて、文体やジャンルの違いを見たい。

```python
!pip install -q fugashi unidic-lite
```

```python
import io, re, urllib.request, zipfile
from collections import Counter
import pandas as pd
from fugashi import Tagger

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

tagger = Tagger()
base = "https://www.aozora.gr.jp/cards/"
works = {"吾輩は猫である": "000148/files/789_ruby_5639.zip", "坊っちゃん": "000148/files/752_ruby_2438.zip",
         "走れメロス": "000035/files/1567_ruby_4948.zip", "銀河鉄道の夜": "000081/files/456_ruby_145.zip"}

dist = {}
for title, path in works.items():
    c = Counter(w.feature.pos1 for line in aozora(base + path).split("\n") for w in tagger(line))
    total = sum(c.values())
    dist[title] = {pos: n / total for pos, n in c.items()}

df = pd.DataFrame(dist).fillna(0).sort_values("吾輩は猫である", ascending=False)
(df * 100).round(1)
```

```python
# MVR (相の類 ÷ 用の類 × 100): 形容詞・副詞などが動詞に対して多いほど「ありさま描写的」(樺島・寿岳の文体指標)
mvr = (df.loc[["形容詞", "形状詞", "副詞", "連体詞"]].sum() / df.loc["動詞"] * 100).round(1)
print(mvr)
```

## キーワードの前後の文脈を一覧する (KWIC) {kwic}

ある単語がどんな文脈で使われているかを、キーワードを中央に揃えて一覧したい。

```python
import io, re, urllib.request, zipfile

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

text = aozora("https://www.aozora.gr.jp/cards/000148/files/752_ruby_2438.zip")  # 坊っちゃん

def kwic(text, keyword, width=15, limit=15):
    flat = text.replace("\n", "")
    for m in list(re.finditer(keyword, flat))[:limit]:
        left = flat[max(0, m.start() - width):m.start()]
        right = flat[m.end():m.end() + width]
        print(f"{left:　>{width}} 【{m.group()}】 {right}")

kwic(text, "赤シャツ")
```

```python
# 正規表現も使える: 「〜ぞなもし」の直前を見る
kwic(text, "ぞな、?もし", width=12, limit=8)
```

```python
# 英語は NLTK の concordance が便利
import nltk
nltk.download("gutenberg", quiet=True)
from nltk.corpus import gutenberg
from nltk.text import Text

Text(gutenberg.words("melville-moby_dick.txt")).concordance("monstrous", width=70, lines=8)
```

- 日本語は全角スペース (`　`) で埋めると等幅フォントできれいに揃う。

## 単語の共起を数えて PMI で結びつきの強さを測る {pmi}

同じ文に一緒に出てきやすい単語の組を見つけたい。単なる頻度ではなく、偶然より多く共起しているかで測りたい。

```python
!pip install -q fugashi unidic-lite
```

```python
import io, re, urllib.request, zipfile
from collections import Counter
from itertools import combinations
from fugashi import Tagger

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

text = aozora("https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip")
tagger = Tagger()
sentences = [s for s in re.split(r"[。\n]", text) if s.strip()]
docs = [{w.feature.orthBase or w.surface for w in tagger(s)
         if w.feature.pos1 in ("名詞", "動詞", "形容詞") and w.feature.pos2 not in ("非自立可能", "数詞")} for s in sentences]
print(len(docs), "文")
```

```python
import math

N = len(docs)
word_count = Counter(w for d in docs for w in d)
pair_count = Counter(pair for d in docs for pair in combinations(sorted(d), 2))

def pmi(a, b):
    a, b = sorted((a, b))
    return math.log2(pair_count[a, b] * N / (word_count[a] * word_count[b]))

print("共起頻度の上位:", pair_count.most_common(8))
strong = sorted(((pmi(a, b), a, b, c) for (a, b), c in pair_count.items() if c >= 10), reverse=True)
for score, a, b, c in strong[:15]:
    print(f"{score:5.2f}  {a} - {b}  (共起 {c} 回)")
```

- PMI は低頻度語の組を過大評価するので、共起頻度の下限（ここでは 10）を設けるのが必須。
- 負の値を 0 に切り上げた PPMI を成分とする共起行列は、古典的な単語ベクトルになる。

## よく使われる言い回し（コロケーション）を抽出する {collocations}

`strong tea` `of course` のような、偶然以上によく隣り合って現れる単語の組を見つけたい。

```python
import nltk
nltk.download("gutenberg", quiet=True)
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder, TrigramAssocMeasures, TrigramCollocationFinder
from nltk.corpus import gutenberg

words = [w.lower() for w in gutenberg.words("austen-emma.txt") if w.isalpha()]
finder = BigramCollocationFinder.from_words(words)
finder.apply_freq_filter(10)  # 10 回未満の bigram は捨てる

measures = BigramAssocMeasures()
print("頻度  :", [" ".join(p) for p, _ in finder.ngram_fd.most_common(8)])
print("PMI   :", [" ".join(p) for p in finder.nbest(measures.pmi, 8)])
print("尤度比:", [" ".join(p) for p in finder.nbest(measures.likelihood_ratio, 8)])
print("t 検定:", [" ".join(p) for p in finder.nbest(measures.student_t, 8)])
```

```python
tri = TrigramCollocationFinder.from_words(words)
tri.apply_freq_filter(8)
print([" ".join(t) for t in tri.nbest(TrigramAssocMeasures().likelihood_ratio, 8)])
```

- PMI は固有名詞などの稀な組、尤度比や t 検定は高頻度の機能語の組を上位に出しやすい。目的に合う指標を選ぶ。
- 日本語なら、分かち書きした単語リストをそのまま `from_words` に渡せばよい。

## 2 つのコーパスを比べて特徴語を見つける（対数尤度比） {keyness}

コーパス A にはコーパス B と比べてどんな語が有意に多いか、を統計的に取り出したい。

```python
!pip install -q fugashi unidic-lite
```

```python
import io, math, re, urllib.request, zipfile
from collections import Counter
from fugashi import Tagger

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

tagger = Tagger()

def count(url):
    return Counter((w.feature.lemma or w.surface).split("-")[0] for line in aozora(url).split("\n") for w in tagger(line) if w.feature.pos1 != "補助記号")

A = count("https://www.aozora.gr.jp/cards/000148/files/752_ruby_2438.zip")  # 坊っちゃん (夏目漱石)
B = count("https://www.aozora.gr.jp/cards/000081/files/456_ruby_145.zip")   # 銀河鉄道の夜 (宮沢賢治)
```

```python
def log_likelihood(a, b, na, nb):
    """Dunning の対数尤度比 G2。3.84 以上で 5% 水準、15.13 以上で 0.01% 水準で有意"""
    e1, e2 = na * (a + b) / (na + nb), nb * (a + b) / (na + nb)
    return 2 * ((a * math.log(a / e1) if a else 0) + (b * math.log(b / e2) if b else 0))

na, nb = sum(A.values()), sum(B.values())
rows = []
for w in set(A) | set(B):
    g2 = log_likelihood(A[w], B[w], na, nb)
    rows.append((g2 if A[w] / na > B[w] / nb else -g2, w, A[w], B[w]))
rows.sort()

print("坊っちゃんに特徴的:", [(w, a, b) for _, w, a, b in rows[::-1][:15]])
print("銀河鉄道の夜に特徴的:", [(w, a, b) for _, w, a, b in rows[:15]])
```

- 内容語だけでなく機能語（一人称「おれ」、文末「た/ました」）も上位に来る。文体の違いの分析に使える。
- カイ二乗検定より低頻度語に頑健なので、コーパス言語学ではこの指標 (keyness) が標準的に使われる。

## 文体の特徴から文章の書き手を推定する（計量文体学） {authorship_attribution}

作者を伏せた文章の断片を、読点の打ち方や助詞の使い方などの文体の特徴から、どの作家が書いたか当てたい。話題（内容語）ではなく文体で当てていることも確かめたい。

```python
!pip install -q datasets fugashi unidic-lite matplotlib-fontja
```

```python
import random, re
from collections import Counter
import numpy as np
import pandas as pd
from datasets import load_dataset
from fugashi import Tagger

# 青空文庫を整形したデータセット (GLOBIS 大学、CC BY 4.0。収録作品はパブリックドメイン)。約 240MB、初回はダウンロードに 1〜2 分
ds = load_dataset("globis-university/aozorabunko-clean", split="train")

AUTHORS = ["夏目漱石", "森鴎外", "芥川竜之介", "太宰治", "宮沢賢治", "坂口安吾"]
MAX_WORKS, MAX_FRAGS, FRAG_CHARS = 24, 4, 2000
random.seed(0)

def series(title):
    """版違い (「（初稿）」) や続き物 (「彼　第二」) を同じ作品として扱うための名前"""
    return re.sub(r"[（(].*?[）)]|[\s　]*第[一二三四五六七八九十]+$", "", title).strip()

works = {a: {} for a in AUTHORS}
for ex in ds:
    m = ex["meta"]
    name = m["姓"] + m["名"]
    # 表記をそろえるため新字新仮名、ジャンルをそろえるため小説・物語 (NDC 913。児童書の K913 も含む) に限る。
    # 随筆や序文を兼ねるもの (「NDC 913 914」など) と短すぎる作品は除く
    if (name in works and m["役割フラグ"] == "著者" and m["文字遣い種別"] == "新字新仮名"
            and m["分類番号"] in ("NDC 913", "NDC K913") and len(ex["text"]) >= 2 * FRAG_CHARS):
        # 連作 (「明治開化　安吾捕物」の各回など) は作品名が同じで作品 ID が別。登場人物が共通なので 1 本だけ残す
        works[name].setdefault(series(m["作品名"]), ex["text"])

def fragments(text):
    """段落の切れ目で約 FRAG_CHARS 字ずつに区切り、作品全体から等間隔に MAX_FRAGS 個を選ぶ"""
    chunks, buf = [], ""
    for para in text.split("\n"):
        buf += para.strip() + "\n"
        if len(buf) >= FRAG_CHARS:
            chunks.append(buf); buf = ""
    idx = np.linspace(0, len(chunks) - 1, min(MAX_FRAGS, len(chunks))).round().astype(int)
    return [chunks[i] for i in sorted(set(idx))]

rows = []
for author, ws in works.items():
    for title, text in random.sample(list(ws.items()), min(MAX_WORKS, len(ws))):
        # 作品名は作家をまたいで重なる (「手紙」「母」など) ので、作家名と組にしてグループの ID にする
        rows += [{"author": author, "work_id": f"{author}/{title}", "text": f} for f in fragments(text)]
df = pd.DataFrame(rows)
print(df.groupby("author").agg(作品数=("work_id", "nunique"), 断片数=("text", "size")))
```

```python
# 形態素解析して、文体の特徴量を作るための列を用意する (Colab の CPU で 10〜20 秒)
tagger = Tagger()
FUNC = {"助詞", "助動詞"}          # 機能語として扱う品詞
KEEP = FUNC | {"補助記号"}          # 内容語を伏せるときに残す品詞

def analyze(text):
    toks = [(w.surface, w.feature.pos1, w.feature.pos2) for line in text.split("\n") for w in tagger(line)]
    sents = [s for s in re.split(r"(?<=[。！？])|\n", text) if s.strip()]
    n_chars = len(re.sub(r"\s", "", text))
    talk = sum(len(t) for t in re.findall(r"「[^」]*」", text))
    return {
        # 1. 文体の基本統計 (文の長さ、読点の多さ、字種の割合、会話文の割合)
        "文の長さ (平均)": np.mean([len(s) for s in sents]),
        "文の長さ (標準偏差)": np.std([len(s) for s in sents]),
        "1 文あたりの読点": text.count("、") / len(sents),
        "漢字の割合": len(re.findall(r"[一-龥々]", text)) / n_chars,
        "ひらがなの割合": len(re.findall(r"[ぁ-ん]", text)) / n_chars,
        "カタカナの割合": len(re.findall(r"[ァ-ヶー]", text)) / n_chars,
        "会話文の割合": talk / n_chars,
        # 2. 読点の直前の品詞 (どこで読点を打つか)
        "comma_pos": " ".join(f"{p1}-{p2}" for (s, p1, p2), nxt in zip(toks, toks[1:]) if nxt[0] == "、"),
        "comma_char": " ".join(text[i - 1] for i in range(1, len(text)) if text[i] == "、"),   # 読点の直前の文字
        # 3. 助詞・助動詞の並び
        "func": " ".join(s for s, p1, _ in toks if p1 in FUNC),
        # 4. 品詞の並び
        "pos": " ".join(p1 for _, p1, _ in toks),
        # 5. 内容語を品詞名に置き換えた文 (助詞・助動詞・記号だけが残る)
        "masked": " ".join(s if p1 in KEEP else f"[{p1}]" for s, p1, _ in toks),
        "masked_np": " ".join(s if p1 in FUNC else f"[{p1}]" for s, p1, _ in toks if p1 != "補助記号"),  # 記号も除く
        # 6. 内容語だけ (話題の手がかり。比較用)
        "content": " ".join(s for s, p1, p2 in toks if p1 in ("名詞", "動詞", "形容詞") and p2 != "非自立可能"),
    }

feats = pd.DataFrame([analyze(t) for t in df["text"]])
STATS = [c for c in feats.columns if feats[c].dtype != object]
print(feats[STATS].groupby(df["author"]).mean().round(3).to_string())
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def words(ngram=(1, 1)):   # 空白区切りのトークン列を n-gram の TF-IDF にする
    return TfidfVectorizer(token_pattern=r"\S+", lowercase=False, ngram_range=ngram, min_df=2, sublinear_tf=True)

SETS = {   # 名前: (使う列, 特徴量の作り方)
    "文体の基本統計":            (STATS, StandardScaler()),
    "読点の直前の文字":          ("comma_char", words()),
    "読点の直前の品詞":          ("comma_pos", words()),
    "助詞・助動詞 (1〜2-gram)":  ("func", words((1, 2))),
    "品詞 (1〜3-gram)":          ("pos", words((1, 3))),
    "内容語を伏せた文 (1〜3-gram)": ("masked", words((1, 3))),
    "　同 (記号も除く)":          ("masked_np", words((1, 3))),
    "内容語 (比較用)":           ("content", words()),
    "文字 (1〜3-gram、全部入り)": ("text", TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2, sublinear_tf=True)),
}
data = pd.concat([df[["author", "work_id", "text"]], feats], axis=1)
y, groups = data["author"].values, data["work_id"].values
cv = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(data, y, groups))  # 作品単位で 5 分割

def work_bootstrap(correct, n_boot=1000, seed=0):
    """作品を単位に復元抽出して、正解率の 95% 信頼区間を求める"""
    rng = np.random.default_rng(seed)
    by_work = pd.Series(correct).groupby(groups)
    s, n = by_work.sum().values, by_work.size().values
    idx = rng.integers(0, len(s), (n_boot, len(s)))
    return np.percentile(s[idx].sum(1) / n[idx].sum(1), [2.5, 97.5])

preds, results = {}, []   # 9 種類 × 5 分割の学習。Colab の CPU で 2〜3 分
for name, (col, vec) in SETS.items():
    pred = np.empty(len(y), dtype=object)
    for tr, te in cv:
        clf = make_pipeline(vec, LogisticRegression(C=10, max_iter=5000))
        clf.fit(data.iloc[tr][col], y[tr])
        pred[te] = clf.predict(data.iloc[te][col])
    preds[name] = pred
    lo, hi = work_bootstrap(pred == y)
    results.append((name, accuracy_score(y, pred), lo, hi, f1_score(y, pred, average="macro")))
pd.DataFrame(results, columns=["特徴量", "正解率", "95%CI 下", "95%CI 上", "マクロ F1"]).round(3)
```

```python
def paired_diff(a, b, n_boot=1000, seed=0):
    """同じ作品の組を復元抽出して、2 つの特徴量の正解率の差の 95% 信頼区間を求める"""
    rng = np.random.default_rng(seed)
    by = pd.DataFrame({"a": preds[a] == y, "b": preds[b] == y}).groupby(groups)
    sa, sb, n = by["a"].sum().values, by["b"].sum().values, by.size().values
    idx = rng.integers(0, len(n), (n_boot, len(n)))
    d = (sa[idx].sum(1) - sb[idx].sum(1)) / n[idx].sum(1)
    return (sa.sum() - sb.sum()) / n.sum(), *np.percentile(d, [2.5, 97.5])

M = "内容語を伏せた文 (1〜3-gram)"
for a, b in [("内容語 (比較用)", M), ("文字 (1〜3-gram、全部入り)", M), (M, "　同 (記号も除く)")]:
    d, lo, hi = paired_diff(a, b)
    print(f"{a} − {b.strip()}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
```

```python
# Burrows の Delta: 頻出語の相対頻度を z 得点にし、各作家の平均との差の絶対値の平均が最も小さい作家を答える
from sklearn.feature_extraction.text import CountVectorizer

n_tok = data["masked"].str.split().str.len().values[:, None]   # 断片の全トークン数 (相対頻度の分母)

def delta_predict(col, n_mfw):
    pred = np.empty(len(y), dtype=object)
    for tr, te in cv:
        cnt = CountVectorizer(token_pattern=r"\S+", lowercase=False, max_features=n_mfw).fit(data.iloc[tr][col])  # 学習側の頻出語
        X = cnt.transform(data[col]).toarray().astype(float)
        X /= n_tok                                                         # 全トークン数に対する相対頻度
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Z = (X - mu) / sd
        centroids = np.stack([Z[tr][y[tr] == a].mean(0) for a in AUTHORS])
        dist = np.abs(Z[te][:, None, :] - centroids[None]).mean(2)         # Delta
        pred[te] = np.array(AUTHORS)[dist.argmin(1)]
    return pred

for col, label in [("func", "助詞・助動詞"), ("masked", "内容語を伏せた文の語")]:
    for n_mfw in (50, 150):
        pred = delta_predict(col, n_mfw)
        lo, hi = work_bootstrap(pred == y)
        print(f"Delta ({label}、上位 {n_mfw} 語): 正解率 {accuracy_score(y, pred):.3f} [{lo:.3f}, {hi:.3f}]")
```

```python
# 間違え方を見る (内容語を伏せた文の分類器)
print(pd.crosstab(pd.Series(y, name="正解"), pd.Series(preds["内容語を伏せた文 (1〜3-gram)"], name="予測")))

# 作家ごとに特に多い・少ない助詞・助動詞 (相対頻度の z 得点の平均)
cnt = CountVectorizer(token_pattern=r"\S+", lowercase=False, max_features=40).fit(data["func"])
X = cnt.transform(data["func"]).toarray() / n_tok
Z = pd.DataFrame((X - X.mean(0)) / X.std(0), columns=cnt.get_feature_names_out()).groupby(y).mean()
for a in AUTHORS:
    z = Z.loc[a].sort_values()
    print(f"{a}: 多い {list(z.index[::-1][:5])}  少ない {list(z.index[:3])}")

# 比較用: 内容語の分類器は何を手がかりにしているか (手がかりを見るために全データで学習し、係数の大きい語を出す)
clf = make_pipeline(words(), LogisticRegression(C=10, max_iter=5000)).fit(data["content"], y)
vocab = clf[0].get_feature_names_out()
for a, coef in zip(clf[1].classes_, clf[1].coef_):
    print(f"{a}: {list(vocab[coef.argsort()[::-1][:10]])}")
```

```python
import matplotlib.pyplot as plt
import matplotlib_fontja
from sklearn.decomposition import PCA

# 作品ごとに助詞・助動詞の相対頻度をまとめ、主成分分析で 2 次元にする
W = pd.DataFrame(X, columns=cnt.get_feature_names_out()).groupby(data["work_id"].values).mean()
work_author = data.groupby("work_id")["author"].first().loc[W.index]
pca = PCA(n_components=2, random_state=0)
P = pca.fit_transform(StandardScaler().fit_transform(W))

fig, ax = plt.subplots(figsize=(7, 6))
for a in AUTHORS:
    m = (work_author == a).values
    ax.scatter(P[m, 0], P[m, 1], label=a, alpha=0.7)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})"); ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
ax.set_title("助詞・助動詞の頻度による作品の散らばり"); ax.legend(); plt.show()

for i in range(2):
    load = pd.Series(pca.components_[i], index=W.columns).sort_values()
    print(f"PC{i + 1}: + 側 {list(load.index[::-1][:6])}  - 側 {list(load.index[:6])}")
```

- 学習と評価は作品単位で分ける（`groups` に作家名と作品名）。同じ作品の断片が両側に入ると、登場人物の名前や話題で当たり、評価が甘くなる。青空文庫では連作（「明治開化　安吾捕物」の各回など）が別々の作品 ID で入っているので、作品名で 1 本にまとめた。題名の違う連作（芥川の「保吉」もの）はこれでは除けない。データの収集と入出力の章の「データを学習・開発・評価に正しく分割する」も参照。
- 6 人 × 24 作品の 486 断片（約 2000 字ずつ）で、内容語を品詞名に伏せた文の 1〜3-gram が正解率 0.944（作品単位のブートストラップで 95% 信頼区間 0.916〜0.969。ブートストラップについては評価指標と統計的検定の章の「ブートストラップ法で評価値の信頼区間を求める」を参照）。品詞の並びだけで 0.881、助詞・助動詞だけで 0.860 だった。読点は、直前の文字 (0.788) のほうが直前の品詞 (0.650) より効いた。読点の打ち方には書き手の特徴が現れ、特にどの文字・単語の後に打つかに最もよく現れる、と報告されている（金・樺島・村上 1993; Jin & Murakami 1993; 金 1994）。助詞の分布だけで日記の書き手を判別した研究もある（金 1997）。
- 内容語 (0.975) と文字 n-gram (0.977) は、伏せた文より高い（作品単位の対応のあるブートストラップで、差は +0.031 [+0.004, +0.061] と +0.033 [+0.010, +0.060]）。全データで学習した内容語の分類器の係数を見ると、「事・所・訳」（漱石）、「ひと・いま」（太宰）、「処・為事」（鴎外）、「心もち」（芥川）のような表記の癖と、「保吉」（芥川）、「津田」（漱石）のように複数の作品に出る名前が混ざっている。係数の読み方はテキスト分類の章の「分類に効いている単語を調べる（線形モデルの係数）」を参照。
- 伏せた文でも、少なくとも固有名詞や話題の語に頼らずに当てられる。ただし、ジャンル（童話の敬体、推理ものの会話文）、底本の表記（記号の使い方）、時代の差は残る。記号も除くと 0.922 に下がった（差 +0.023 [+0.000, +0.049]）。宮沢賢治は 65 断片中 62 が正解で、多い機能語は「まし・でし・ます」。作品の多くが童話 (NDC K913) の敬体なので、作家の文体というよりジャンルの差で当たっている面がある。内容語を伏せる考え方は Stamatatos (2017) の text distortion と同じ。原論文は品詞名ではなく、頻出語の上位 k 語だけを残し、ほかの語を文字ごと `*` に置き換える。
- Burrows の Delta (Burrows 2002) は、頻出語の相対頻度（ここでは全トークン数で割る）を z 得点にして、その差の絶対値の平均で作家との距離を測る。ここでは正解率 0.648〜0.763 で、ロジスティック回帰より大きく低い。Burrows は約 1500 語を超える文章で精度がよいと報告しており、2000 字ほどの断片では短すぎる可能性がある。
- 主成分分析の図では、PC1 の + 側（「に・を・ず・ぬ・より」）に森鴎外が、- 側（「だ・から・な・ん・よ・まし」）に宮沢賢治が寄る。PC2 の - 側には、芥川に多い「ばかり・ながら」と、です・ます体の「ませ・です・まし」が混ざっていて、芥川と宮沢賢治が下に来る。漱石・太宰・坂口安吾は重なる（2 成分で分散の 29%）。品詞の構成で作品を比べる方法は、この章の「品詞の分布を調べる・比べる」を参照。
- 新字新仮名の作品は、底本の段階で表記が現代の字体と仮名遣いに直されている。漢字の割合や「云う / 言う」のような表記の特徴には、作家だけでなく底本や編集方針の違いも混じる。

## KenLM で n-gram 言語モデルを作り、パープレキシティでコーパスのノイズを見つける {ngram_lm}

Web から集めたコーパスに混じった、文章になっていない段落（キーワードの羅列、外国語、記号の繰り返しなど）を、きれいな文章で学習した言語モデルのパープレキシティで見つけて除きたい。

```python
# KenLM をビルドする。版をそろえるためにコミットを固定する (Colab で 3〜5 分)
!apt-get install -y -q libboost-program-options-dev libboost-system-dev libboost-thread-dev libboost-test-dev > /dev/null 2>&1
!git clone -q https://github.com/kpu/kenlm && cd kenlm && git checkout -q 4cb443e
# ビルドの出力は build.log に書き、失敗したときだけ末尾を表示する
!mkdir -p kenlm/build && cd kenlm/build && (cmake .. > build.log 2>&1 && make -j4 lmplz build_binary >> build.log 2>&1 || tail -n 20 build.log)
!ls kenlm/build/bin
# Python から使うモジュール。同梱の kenlm.cpp は古い Cython で生成されていて Python 3.13 ではビルドできないので、作り直してから入れる
!pip install -q cython==3.1.4 && cd kenlm && python -m cython -3 --cplus python/kenlm.pyx -o python/kenlm.cpp && pip install -q .
```

```python
!pip install -q datasets fugashi unidic-lite matplotlib-fontja
```

```python
import itertools, re, unicodedata
from datasets import load_dataset
from fugashi import Tagger

tagger = Tagger("-Owakati")

def tokenize(s):
    """学習と採点で同じ前処理をする: NFKC で正規化して小文字にし、数字を 0 にそろえてから分かち書きする"""
    s = re.sub(r"\d", "0", unicodedata.normalize("NFKC", s).lower())
    return tagger.parse(s).strip()

def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[。！？!?])|\n", text) if s.strip()]

# 「きれいな文章」の見本として日本語 Wikipedia (wikimedia/wikipedia の 20231101.ja、CC BY-SA 3.0 / GFDL) の先頭 3,000 記事を使う
# 100 記事に 1 つは学習に使わず、パープレキシティの基準をとるために残す (Colab で 1〜2 分)
wiki = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
with open("wiki.txt", "w") as f_train, open("wiki_dev.txt", "w") as f_dev:
    for i, article in enumerate(itertools.islice(wiki, 3000)):
        f = f_dev if i % 100 == 0 else f_train
        for s in sentences(article["text"]):
            if len(s) >= 10:                   # 見出しなどの短い行は除く
                f.write(tokenize(s) + "\n")
!wc -l wiki.txt wiki_dev.txt
!head -n 3 wiki.txt
```

```python
# 5-gram の言語モデルを学習する (修正 Kneser-Ney 平滑化)。-S は使うメモリの上限、--prune 0 1 1 1 は 2-gram 以上で 1 回しか出ないものを捨てる
!kenlm/build/bin/lmplz -o 5 -S 2G --prune 0 1 1 1 < wiki.txt > wiki.arpa 2> lmplz.log
!kenlm/build/bin/build_binary wiki.arpa wiki.bin > /dev/null 2>&1
!ls -lh wiki.arpa wiki.bin
!sed -n 1,8p wiki.arpa
```

```python
import kenlm

lm = kenlm.Model("wiki.bin")

def ppl_tok(toks):
    """分かち書き済みの文の列のパープレキシティ。文ごとの log10 確率を足し、文末記号 </s> を含むトークン数で割る"""
    logp = sum(lm.score(t) for t in toks)          # lm.score は文頭・文末の記号 <s> </s> を自動で付ける
    n = sum(len(t.split()) + 1 for t in toks)
    return 10 ** (-logp / n)

def ppl(sents):
    return ppl_tok([tokenize(s) for s in sents])

dev = [line.strip() for line in open("wiki_dev.txt")]   # 前処理済み
print(f"{lm.order}-gram、Wikipedia の残しておいた {len(dev)} 文の PPL = {ppl_tok(dev):.1f}")
for s in ["吾輩は猫である。", "猫は吾輩である。", "である猫吾輩は。", "東京都は日本の首都である。", "SampleGallery SampleGallery SampleGallery"]:
    print(f"{ppl([s]):10.1f}  {s}")

# トークンごとの log10 確率、使われた n-gram の長さ、未知語かどうか
for s in ["推しのアクスタを買ったけど草", "그렇게 많은 오늘이"]:
    t = tokenize(s)
    print(" ".join(f"{w}({logp:.1f},{n}{',未知語' if oov else ''})" for w, (logp, n, oov) in zip(t.split() + ["</s>"], lm.full_scores(t))))
```

```python
import pandas as pd

# 採点する Web コーパス: mC4 の日本語 (allenai/c4 の multilingual、ODC-BY。Common Crawl の利用規約にも従う) の検証用の 1 ファイル (約 40MB)
web = load_dataset("allenai/c4", data_files="multilingual/c4-ja-validation.tfrecord-00000-of-00008.json.gz", split="train")

# CCNet と同じく段落 (改行で区切られた行) ごとにパープレキシティを計算する (Colab で 1 分ほど)
rows = []
for doc_id, doc in enumerate(web.select(range(3000))):
    for para in doc["text"].split("\n"):
        if toks := [tokenize(s) for s in sentences(para)]:
            rows.append((doc_id, para, sum(len(t.split()) + 1 for t in toks), ppl_tok(toks)))
df = pd.DataFrame(rows, columns=["doc", "text", "tokens", "ppl"])
d = df[df["tokens"] >= 10]           # 短い段落は値がばらつくので、以下では 10 トークン以上の段落だけを見る
print(len(web), "文書のうち 3000 文書、", len(df), "段落 (10 トークン以上は", len(d), "段落)")
d["ppl"].describe(percentiles=[0.1, 0.33, 0.5, 0.66, 0.9, 0.99]).round(1)
```

```python
import matplotlib.pyplot as plt
import matplotlib_fontja
import numpy as np

pd.set_option("display.max_colwidth", 70)
ax = np.log10(d["ppl"]).hist(bins=60)
ax.set_xlabel("log10 パープレキシティ"); ax.set_ylabel("段落数 (10 トークン以上)"); plt.show()

print("PPL が最も高い段落"); display(d.nlargest(12, "ppl")[["ppl", "text"]].round(0))
print("PPL が最も低い段落"); display(d.nsmallest(8, "ppl")[["ppl", "text"]].round(1))
```

```python
# CCNet のように、PPL の三分位で head (Wikipedia に近い)・middle・tail に分け、無作為に 6 段落ずつ見る
q1, q2 = d["ppl"].quantile([1 / 3, 2 / 3])
d = d.assign(bucket=pd.cut(d["ppl"], [0, q1, q2, np.inf], labels=["head", "middle", "tail"]))
print(f"三分位の境界: {q1:.0f}, {q2:.0f}")
for b, g in d.groupby("bucket", observed=True):
    print(f"--- {b}")
    for _, r in g.sample(6, random_state=0).iterrows():
        print(f"{r.ppl:9.0f}  {r.text[:60]}")
```

```python
# 除く: 上位 10% (PPL が高い側) を捨てる。境界のすぐ内側と外側を目で確かめてから閾値を決める
th = d["ppl"].quantile(0.9)
removed = d["ppl"] > th
print(f"閾値 PPL = {th:.0f}、除く段落 {removed.sum()} / {len(d)}")
print(f"段落の平均の文字数: 除く段落 {d.loc[removed, 'text'].str.len().mean():.0f}、残す段落 {d.loc[~removed, 'text'].str.len().mean():.0f}")
print("--- 境界のすぐ外側 (除かれる)")
for _, r in d[removed].nsmallest(8, "ppl").iterrows():
    print(f"{r.ppl:9.0f}  {r.text[:60]}")
print("--- 境界のすぐ内側 (残る)")
for _, r in d[~removed].nlargest(8, "ppl").iterrows():
    print(f"{r.ppl:9.0f}  {r.text[:60]}")
clean = df[~df.index.isin(d.index[removed])]
print(f"文字数: {df['text'].str.len().sum():,} → {clean['text'].str.len().sum():,}")
```

- CCNet（Wenzek ら 2020）の言語モデルによる絞り込みにならった。CCNet は言語ごとに Wikipedia で SentencePiece と 5-gram の Kneser-Ney モデル (KenLM) を学習し、Web のテキストをパープレキシティの三分位で head・middle・tail に分ける。論文は段落ごとに計算すると書いているが、公開コード (facebookresearch/cc_net) は段落のスコアを文書ごとに合算し、文書を分けている。ここでは段落ごとに見る。分かち書きは fugashi にし、小文字化と数字の 0 への置き換えは公開コードの正規化をまねた。論文は、tail にも用途によっては役立つ文章があるとして、パープレキシティで捨てることまではしていない。
- 最も PPL が高い段落は、タイ文字の結合記号の繰り返し、「SampleGallery」の繰り返し、ベトナム語や韓国語の文、英単語の羅列だった。mC4 はページ単位で言語を判定しているので、日本語とされたページにこうした段落が混じる。この例では韓国語の語はすべて未知語になった。一方で、閾値のすぐ内側にはロシア語のスパムや英語の文が残った。外国語を除くのは言語判定のほうが確実なので、英語・多言語の前処理の章の「高精度な言語判定と、混在テキストの言語区間検出 (lingua)」と組み合わせる。
- 低い側にもノイズがある。PPL が最も低い段落は、年号の連続、ISBN、数字の列、日付 (PPL 1.4〜3.4) だった。数字を 0 にそろえたので、同じトークンの繰り返しになり予測しやすい。高い側だけでなく低い側も目で見て、文字種の割合などの規則や、データの収集と入出力の章の「テキストデータの重複と不良データを取り除く」と組み合わせる。
- パープレキシティは「Wikipedia にどれだけ近いか」であって、文章の質そのものではない。head には日付だけの行 (PPL 5) や、それらしく見えて意味の通らない宣伝文 (PPL 281) が入り、tail には顔文字の入ったブログの文 (PPL 4233) が入った。「推しのアクスタを買ったけど草」も、使われた n-gram はほとんど 1〜2-gram で、確率が低い。くだけた文や専門分野の文を集めたいときに Wikipedia の言語モデルで切ると、欲しいものまで捨てる。CCNet の論文も、ブログのコメントや専門的なフォーラムが tail に入ると書いている。
- 10 トークン以上の段落の上位 10%（PPL 10278 より上、10,161 / 101,616 段落）を除くと、文字数は 1,245 万から 1,023 万に 18% 減った。除いた段落は平均 218 文字で、残した段落（109 文字）の 2 倍の長さがあるため、段落数の割合より文字数の減り方が大きい。境界の両側には似た商品名の羅列が並んでいて、閾値に決まった正解はない。境界の近くを見て決め、文字数の減り方も確かめる。10 トークン未満の段落は値がばらつくので、ここでは判定せずに残した。
- lmplz は修正 Kneser-Ney 平滑化で学習し、未知語のための `<unk>` も自動で入れるので、未知語の確率は 0 にならない。学習と採点で前処理 (`tokenize`) をそろえないと値が意味をなさない。`-S` はメモリの上限で、Colab に合わせて 2G にした。`--prune` で ARPA は 137MB、バイナリは 87MB になった。機械翻訳の章の「Moses で統計的機械翻訳システムを作る」も同じ lmplz と build_binary を使っている。ニューラル言語モデルでは、言語モデルの確率と内部を調べるの章の「言語モデルで文の自然さ（パープレキシティ）を測る」を参照。
- KenLM (Heafield 2011) は LGPL 2.1 以降（一部のファイルは別のライセンス）で、2025-03-30 のコミット 4cb443e に固定した。PyPI の kenlm 0.3.0 も同梱の `kenlm.cpp` も Python 3.13 ではビルドできなかったので、Cython で作り直している。データは日本語 Wikipedia (CC BY-SA 3.0 / GFDL) と mC4 の日本語（ODC-BY。Common Crawl の利用規約にも従う）。12 万段落の採点は 2 コアで 30 秒ほどだった（mC4 はダウンロード済みの状態で測った）。
- 仕組みを自分で実装して確かめる版は、付録「仕組みを理解するための実装」の「n-gram 言語モデルを自分で実装してパープレキシティを測り、文を生成する」にある。
