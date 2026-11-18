# コーパスの統計と可視化

単語を数える、共起や特徴語を調べる、分布を図にする、n-gram 言語モデルでノイズの混じった文を見つける。コーパスを手に入れたらまずやること。

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

- 内容語だけでなく機能語も上位に来る。坊っちゃんでは一人称「おれ」が見出し語の「俺」（477 対 1）として 1 位に、銀河鉄道の夜では文末の「ます」「です」が上位に出る。文体の違いの分析に使える。
- UniDic の見出し語で数えるので、人名の「山嵐」は「ヤマアラシ」（155 対 0）として出る。「赤シャツ」は「赤」と「シャツ」に、「野だ」は「野」に分かれる。
- カイ二乗検定より低頻度語に頑健なので、コーパス言語学ではこの指標 (keyness) が標準的に使われる。

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
