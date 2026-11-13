# コーパスの統計と可視化

単語を数える、特徴語を見つける、分布を図にする。コーパスを手に入れたらまずやること。

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
- `[w for line in lines for w in tagger(line)]` のようにノードそのものを溜め込むと、中身が壊れて品詞や原形がでたらめになる (fugashi の落とし穴)。

## Zipf の法則を確かめる (順位-頻度プロット) {zipf}

単語の頻度が順位にほぼ反比例すること (Zipf の法則) を、両対数グラフで確かめたい。

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

## matplotlib で日本語を文字化けさせずに表示する {matplotlib_ja}

グラフの軸ラベルや凡例の日本語が □□□ (豆腐) になるのを直したい。

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

- 以前の定番 `japanize-matplotlib` は Python 3.12 以降で動かない (distutils 廃止のため)。`matplotlib-fontja` はその後継。
- 別のフォントを使いたいときは `!apt-get -y install fonts-noto-cjk` して `plt.rcParams["font.family"] = "Noto Sans CJK JP"`。

## 日本語のワードクラウドを作る {wordcloud_ja}

テキストの頻出語をワードクラウドで可視化したい。

```python
!pip install -q wordcloud fugashi unidic-lite matplotlib-fontja
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

text = aozora("https://www.aozora.gr.jp/cards/000081/files/456_ruby_145.zip")  # 銀河鉄道の夜
tagger = Tagger()
stop = {"よう", "こと", "もの", "それ", "これ", "そこ", "どこ", "なに", "ところ", "とき", "うち", "ほう"}
freq = Counter(w.surface for line in text.split("\n") for w in tagger(line)
               if w.feature.pos1 == "名詞" and w.feature.pos2 != "数詞" and w.surface not in stop and len(w.surface) > 1)
print(freq.most_common(10))
```

```python
import matplotlib.pyplot as plt
import matplotlib_fontja
from wordcloud import WordCloud

wc = WordCloud(font_path=matplotlib_fontja.get_font_ttf_path(),  # 日本語フォントの指定が必須
               width=800, height=450, background_color="white", max_words=150, colormap="viridis")
wc.generate_from_frequencies(freq)
plt.figure(figsize=(10, 6)); plt.imshow(wc); plt.axis("off"); plt.show()
wc.to_file("wordcloud.png");
```

- 分かち書き済みの文字列を `wc.generate(text)` に渡す方法もあるが、頻度辞書を渡すほうが品詞やストップワードを制御しやすい。

## TF-IDF で文書ごとの特徴語を取り出す {tfidf_keywords}

複数の文書があるとき、それぞれの文書を特徴づける単語 (他の文書にはあまり出ない語) を見つけたい。

```python
!pip install -q fugashi unidic-lite
```

```python
import io, re, urllib.request, zipfile

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

base = "https://www.aozora.gr.jp/cards/"
docs = {
    "吾輩は猫である": aozora(base + "000148/files/789_ruby_5639.zip"),
    "坊っちゃん": aozora(base + "000148/files/752_ruby_2438.zip"),
    "走れメロス": aozora(base + "000035/files/1567_ruby_4948.zip"),
    "銀河鉄道の夜": aozora(base + "000081/files/456_ruby_145.zip"),
}
```

```python
import pandas as pd
from fugashi import Tagger
from sklearn.feature_extraction.text import TfidfVectorizer

tagger = Tagger()

def tokenize(text):
    return [w.surface for line in text.split("\n") for w in tagger(line)
            if w.feature.pos1 == "名詞" and w.feature.pos2 in ("普通名詞", "固有名詞") and len(w.surface) > 1]

vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False, sublinear_tf=True)
X = vec.fit_transform(docs.values())
vocab = vec.get_feature_names_out()

top = {title: [vocab[i] for i in row.toarray().ravel().argsort()[::-1][:10]] for title, row in zip(docs, X)}
pd.DataFrame(top)
```

- 自前のトークナイザを渡すときは `token_pattern=None` を付ける (警告を消す)。
- `sublinear_tf=True` は頻度を 1+log(tf) に圧縮する。長い文書で頻出語が支配的になるのを防ぐ。
- 文書数が少ないと IDF があまり効かない。段落や章を 1 文書とみなすと粒度を変えられる。

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

- PMI は低頻度語の組を過大評価するので、共起頻度の下限 (ここでは 10) を設けるのが必須。
- 負の値を 0 に切り上げた PPMI を成分とする共起行列は、古典的な単語ベクトルになる。

## よく使われる言い回し (コロケーション) を抽出する {collocations}

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

## 語彙の豊富さを測る (TTR とその仲間) {lexical_diversity}

テキストがどれくらい多様な語彙を使っているかを数値化したい (作家の比較、学習者の習熟度の推定など)。

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

- 素の TTR (異なり語数 ÷ 延べ語数) はテキストが長いほど小さくなるため、長さの違うテキストの比較には使えない。MATTR や MTLD を使う。

## n-gram 言語モデルを作ってパープレキシティを測り、文を生成する {ngram_lm}

古典的な n-gram 言語モデルを学習して、文の「それらしさ」を数値化したい。文も生成してみたい。

```python
!pip install -q fugashi unidic-lite
```

```python
import io, re, urllib.request, zipfile
from fugashi import Tagger

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

tagger = Tagger()
text = aozora("https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip")
sents = [[w.surface for w in tagger(s)] for s in re.split(r"(?<=。)|\n", text) if s and s.strip()]
train, test = sents[:-300], sents[-300:]
print(len(train), len(test))
```

```python
from nltk.lm import KneserNeyInterpolated, Laplace, Vocabulary
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.util import ngrams

n = 3
models = {}
for name, cls in [("Laplace", Laplace), ("Kneser-Ney", KneserNeyInterpolated)]:
    data, words = padded_everygram_pipeline(n, train)
    lm = cls(n, vocabulary=Vocabulary(words, unk_cutoff=2))  # 1 回しか出ない語は <UNK> として学習
    lm.fit(data)
    models[name] = lm

def perplexity(lm, sentences):
    grams = [g for s in sentences for g in ngrams(pad_both_ends(lm.vocab.lookup(s), n=n), n)]
    return lm.perplexity(grams)

for name, lm in models.items():
    print(f"{name:11} テスト PPL = {perplexity(lm, test[:100]):.1f}")
```

```python
lm = models["Kneser-Ney"]
for s in ["吾輩は猫である。", "猫は吾輩である。", "である猫吾輩は。"]:
    print(f"{perplexity(lm, [[w.surface for w in tagger(s)]]):10.1f}  {s}")

for seed in range(3):
    out = lm.generate(30, text_seed=["<s>", "<s>"], random_seed=seed)
    print("".join(w for w in out if w not in ("<s>", "</s>")))
```

- 未知語があると確率 0 (PPL = ∞) になるので、低頻度語を `<UNK>` に置き換えて学習する (`unk_cutoff`)。
- スムージングで未知の n-gram の確率を 0 にしないのが肝。加算 (Laplace) より Kneser-Ney のほうがずっと良い (PPL が低い)。
- 本格的に使うなら KenLM (高速・省メモリ)。ニューラル言語モデルのパープレキシティは Transformers の章を参照。

## 共起ネットワークを描く {cooccurrence_network}

よく一緒に出てくる単語同士を線で結んだネットワーク図で、テキストの話題の構造を俯瞰したい。

```python
!pip install -q fugashi unidic-lite matplotlib-fontja
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

text = aozora("https://www.aozora.gr.jp/cards/000035/files/1567_ruby_4948.zip")  # 走れメロス
tagger = Tagger()
docs = [{w.surface for w in tagger(s) if w.feature.pos1 == "名詞" and w.feature.pos2 in ("普通名詞", "固有名詞") and len(w.surface) > 1}
        for s in re.split(r"[。\n]", text) if s.strip()]
word_count = Counter(w for d in docs for w in d)
pair_count = Counter(p for d in docs for p in combinations(sorted(d), 2))
```

```python
import matplotlib.pyplot as plt
import matplotlib_fontja
import networkx as nx

G = nx.Graph()
for (a, b), c in pair_count.items():
    jaccard = c / (word_count[a] + word_count[b] - c)
    if c >= 3 and jaccard >= 0.1:
        G.add_edge(a, b, weight=jaccard)

communities = nx.community.greedy_modularity_communities(G)  # 色分け用にコミュニティ検出
color = {w: i for i, com in enumerate(communities) for w in com}

plt.figure(figsize=(11, 8))
pos = nx.spring_layout(G, k=0.6, seed=0)
nx.draw_networkx_nodes(G, pos, node_size=[word_count[w] * 40 for w in G], node_color=[color[w] for w in G], cmap="Set3")
nx.draw_networkx_edges(G, pos, width=[G[a][b]["weight"] * 6 for a, b in G.edges], alpha=0.4)
nx.draw_networkx_labels(G, pos, font_family=plt.rcParams["font.family"], font_size=11)
plt.axis("off"); plt.show()
```

- 線を引く基準は、共起回数の下限と Jaccard 係数の下限の 2 つで調整する。線が多すぎると毛玉になる。
- `draw_networkx_labels` は rcParams のフォントを見ないので `font_family` を明示する。

## 2 つのコーパスを比べて特徴語を見つける (対数尤度比) {keyness}

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

- 内容語だけでなく機能語 (一人称「おれ」、文末「た/ました」) も上位に来る。文体の違いの分析に使える。
- カイ二乗検定より低頻度語に頑健なので、コーパス言語学ではこの指標 (keyness) が標準的に使われる。

## 文の長さや文字数の基本統計をとる {corpus_stats}

コーパスの規模感 (文数、平均文長、長さの分布) をまず把握したい。

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

## 品詞の分布を調べる・比べる {pos_distribution}

テキストの品詞構成 (名詞が多い、動詞が多い…) を調べて、文体やジャンルの違いを見たい。

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

## 文字や単語のエントロピーを計算する {entropy}

テキストの情報量 (1 文字あたり何ビットか) を計算して、言語やテキストの冗長性を比べたい。

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

- 文脈を長くするほど条件付きエントロピーは下がる。その極限がその言語のエントロピーレートで、言語モデルの交差エントロピー (= log2 パープレキシティ) はその上界になる。
- 短いテキストでは過小評価になる。実際の推定には大きなコーパスが必要。
