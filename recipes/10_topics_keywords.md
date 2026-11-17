# トピックとキーワードの抽出

ワードクラウドで眺める、文書を代表する語 (特徴語やキーワード) を選ぶ、共起ネットワークを描く、文書集合に潜む話題 (トピック) を見つける、急に増えた語を見つける (バースト検出)。

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

## 文書からキーワードを抽出する (埋め込みによる KeyBERT 方式) {keyword_extraction}

1 つの文書から、その内容を代表するキーワードを数個選びたい。コーパス全体の統計なしで、単独の文書に対して行いたい。

```python
!pip install -q fugashi unidic-lite
```

```python
import numpy as np
from fugashi import Tagger
from sentence_transformers import SentenceTransformer

tagger = Tagger()
model = SentenceTransformer("intfloat/multilingual-e5-small")

def candidates(text):
    """連続する名詞をつなげた複合名詞を候補にする"""
    out, buf = [], []
    for w in list(tagger(text)) + [None]:
        if w is not None and w.feature.pos1 in ("名詞", "接頭辞", "接尾辞") and w.feature.pos2 != "数詞":
            buf.append(w.surface)
        else:
            if buf and len("".join(buf)) > 1:
                out.append("".join(buf))
            buf = []
    return list(dict.fromkeys(out))

def keywords(text, top_k=6, diversity=0.5):
    cands = candidates(text)
    doc = model.encode("query: " + text, normalize_embeddings=True)
    emb = model.encode(["query: " + c for c in cands], normalize_embeddings=True)
    sim_doc, sim_cand = emb @ doc, emb @ emb.T
    selected = [int(sim_doc.argmax())]
    while len(selected) < min(top_k, len(cands)):      # MMR: 文書に近く、かつ選択済みの語とは似ていないものを選ぶ
        rest = [i for i in range(len(cands)) if i not in selected]
        mmr = [(1 - diversity) * sim_doc[i] - diversity * sim_cand[i, selected].max() for i in rest]
        selected.append(rest[int(np.argmax(mmr))])
    return [(cands[i], round(float(sim_doc[i]), 3)) for i in selected]

text = ("文法誤り訂正は、学習者が書いた文に含まれる誤りを自動的に訂正する自然言語処理のタスクである。"
        "近年は系列変換モデルや大規模言語モデルを用いた手法が主流となっており、擬似誤りデータによる事前学習が性能向上に寄与している。"
        "評価には編集単位の適合率と再現率に基づく F0.5 が広く用いられる。")
print("候補:", candidates(text))
print("キーワード:", keywords(text))
```

- 「文書全体の埋め込みに近い語句ほど、その文書を代表している」という考え方。学習もコーパスも要らない。
- MMR で多様性を持たせないと、「言語モデル」「大規模言語モデル」のように似た語ばかりが選ばれる。
- 統計ベースなら TF-IDF (この章の「TF-IDF で文書ごとの特徴語を取り出す」を参照)、英語なら YAKE や RAKE、グラフベースなら TextRank などがある。ライブラリは `keybert` や `pke`。

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

## LDA で文書集合のトピックを見つける {lda_topics}

大量の文書がどんな話題で構成されているのかを、ラベルなしで把握したい。各文書がどの話題をどれくらい含むかも知りたい。

```python
!pip install -q fugashi unidic-lite
```

```python
import io, tarfile, urllib.request
import pandas as pd
from fugashi import Tagger

def load_livedoor():
    url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
    with urllib.request.urlopen(url) as r:
        tar = tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")
    rows = []
    for m in tar.getmembers():
        parts = m.name.split("/")
        if m.isfile() and len(parts) == 3 and parts[2] != "LICENSE.txt":
            lines = tar.extractfile(m).read().decode("utf-8").split("\n")
            rows.append({"label": parts[1], "title": lines[2], "text": "\n".join(lines[3:])})
    return pd.DataFrame(rows)

df = load_livedoor().sample(3000, random_state=0).reset_index(drop=True)
tagger = Tagger()

def nouns(text):
    return " ".join(w.surface for w in tagger(text[:1500]) if w.feature.pos1 == "名詞" and w.feature.pos2 == "普通名詞" and len(w.surface) > 1)

df["nouns"] = df["text"].map(nouns)
```

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(token_pattern=r"\S+", min_df=5, max_df=0.3)   # LDA には TF-IDF ではなく単語の出現回数を渡す
X = vec.fit_transform(df["nouns"])
lda = LatentDirichletAllocation(n_components=9, learning_method="batch", max_iter=30, random_state=0)
doc_topic = lda.fit_transform(X)        # 文書 × トピックの分布 (各行の和は 1)

vocab = vec.get_feature_names_out()
for k, weights in enumerate(lda.components_):
    print(k, " ".join(vocab[i] for i in weights.argsort()[::-1][:10]))
```

```python
# 各トピックと元のカテゴリの対応を見る (答え合わせ)
df["topic"] = doc_topic.argmax(axis=1)
pd.crosstab(df["label"], df["topic"])
```

- トピックは「単語の確率分布」、文書は「トピックの確率分布」として表される。1 つの文書が複数のトピックを持てるのが、クラスタリングとの違い。
- トピック数は自分で決める。パープレキシティやコヒーレンスも参考になるが、最終的には「人間が見て解釈できるか」で選ぶ。
- 前処理が結果を大きく左右する。名詞に絞る、高頻度語・低頻度語を除く (`max_df`、`min_df`) のが基本。
- 大規模なコーパスや、コヒーレンスの計算・可視化 (pyLDAvis) まで行うなら gensim の `LdaModel` が定番。

## NMF でトピックを抽出する {nmf_topics}

LDA より速く、短い文書でも安定しやすい方法でトピックを抽出したい。TF-IDF 行列を非負値行列因子分解する。

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
docs = [d for d in data.data if len(d) > 200]
vec = TfidfVectorizer(stop_words="english", min_df=5, max_df=0.5, max_features=20000, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
X = vec.fit_transform(docs)

nmf = NMF(n_components=12, init="nndsvd", random_state=0, max_iter=400)
W = nmf.fit_transform(X)      # 文書 × トピックの重み
H = nmf.components_           # トピック × 単語の重み
vocab = vec.get_feature_names_out()
for k, row in enumerate(H):
    print(f"{k:2} {' '.join(vocab[i] for i in row.argsort()[::-1][:9])}")
```

```python
# 各トピックを最もよく代表する文書
for k in [0, 1, 2]:
    i = W[:, k].argmax()
    print(k, "|", " ".join(docs[i].split())[:110])
```

- TF-IDF 行列 X を、非負の 2 つの行列の積 W × H で近似する。非負なので「足し算だけ」で文書が表現され、解釈しやすい。
- 確率モデルではないので、LDA のような確率的な解釈はできないが、決定的 (`init="nndsvd"`) で再現性が高い。
- ツイートやタイトルのような短文には、LDA より NMF のほうが良い結果になることが多い。

## 文埋め込みとクラスタリングでトピックを作る (BERTopic 方式) {embedding_topics}

単語の共起ではなく、文の意味に基づいてトピックを作りたい。クラスタごとに特徴的な語を取り出して名前を付けたい。

```python
!pip install -q fugashi unidic-lite umap-learn
```

```python
import io, tarfile, urllib.request
import pandas as pd
from sentence_transformers import SentenceTransformer

def load_livedoor():
    url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
    with urllib.request.urlopen(url) as r:
        tar = tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")
    rows = []
    for m in tar.getmembers():
        parts = m.name.split("/")
        if m.isfile() and len(parts) == 3 and parts[2] != "LICENSE.txt":
            lines = tar.extractfile(m).read().decode("utf-8").split("\n")
            rows.append({"label": parts[1], "title": lines[2], "text": "\n".join(lines[3:])})
    return pd.DataFrame(rows)

df = load_livedoor().sample(1500, random_state=0).reset_index(drop=True)
model = SentenceTransformer("intfloat/multilingual-e5-small")
emb = model.encode(["query: " + t for t in df["title"]], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
```

```python
import numpy as np
import umap
from fugashi import Tagger
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

# 1. 次元削減してからクラスタリング (高次元のままだと距離が効きにくい)
reduced = umap.UMAP(n_neighbors=15, n_components=5, metric="cosine", random_state=0).fit_transform(emb)
df["cluster"] = KMeans(n_clusters=12, n_init=10, random_state=0).fit_predict(reduced)

# 2. クラスタごとに文書を連結して 1 つの大きな文書とみなし、クラスタ単位の TF-IDF (c-TF-IDF) で特徴語を選ぶ
tagger = Tagger()
tok = lambda s: [w.surface for w in tagger(s) if w.feature.pos1 == "名詞" and w.feature.pos2 in ("普通名詞", "固有名詞") and len(w.surface) > 1]
joined = df.groupby("cluster")["title"].apply(" ".join)
vec = CountVectorizer(tokenizer=tok, token_pattern=None, lowercase=False)
counts = vec.fit_transform(joined).toarray()
tf = counts / counts.sum(axis=1, keepdims=True)
idf = np.log(1 + counts.sum(axis=1).mean() / counts.sum(axis=0))
ctfidf = tf * idf
vocab = vec.get_feature_names_out()

for c in joined.index:
    top = [vocab[i] for i in ctfidf[c].argsort()[::-1][:7]]
    main = df[df.cluster == c]["label"].value_counts()
    print(f"{c:2} ({(df.cluster == c).sum():3} 件, 最多カテゴリ {main.index[0]} {main.iloc[0] / main.sum():.0%}) {' '.join(top)}")
```

- この「埋め込み → UMAP → クラスタリング → c-TF-IDF」の流れをまとめたライブラリが BERTopic (`pip install bertopic`)。密度ベースの HDBSCAN でクラスタ数を自動で決め、どこにも属さない外れ値も扱える。
- 短い文 (タイトル、ツイート、アンケートの回答) に特に向く。長い文書は段落に分けてから埋め込む。
- トピックの名前付けは、特徴語と代表文書を LLM に渡して要約させると読みやすくなる。

## 日付のあるテキストから急に増えた語を見つける (バースト検出) {burst_detection}

日付の付いたニュース記事の流れから、ある時期に急に増えた語 (話題) と、その期間を見つけたい。

```python
!pip install -q fugashi unidic-lite matplotlib-fontja
```

```python
import io, tarfile, urllib.request
import pandas as pd

# livedoor ニュースコーパス (株式会社ロンウイット配布)。記事は CC BY-ND 2.1 JP (表示・改変禁止)
def load_livedoor():
    url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
    with urllib.request.urlopen(url) as r:
        tar = tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")
    rows = []
    for m in tar.getmembers():
        parts = m.name.split("/")
        if m.isfile() and len(parts) == 3 and parts[2] != "LICENSE.txt":
            lines = tar.extractfile(m).read().decode("utf-8").split("\n")
            rows.append({"label": parts[1], "date": lines[1], "title": lines[2], "text": "\n".join(lines[3:])})
    return pd.DataFrame(rows)

df = load_livedoor()
df["date"] = pd.to_datetime(df["date"].str[:10])
# カテゴリごとに収録の時期が違うので、毎月の記事がある「Sports Watch」だけを使う。
# 収録の最初と最後の月 (2010-02 は 12 件、2012-09 は 3 件) は記事が少ないので除く
df = df[(df.label == "sports-watch") & (df.date >= "2010-03-01") & (df.date < "2012-09-01")].reset_index(drop=True)
df["month"] = df["date"].dt.to_period("M")
print(len(df), "件")
print(df.groupby("month").size().to_string())
```

```python
import re
from collections import Counter
from fugashi import Tagger

tagger = Tagger()

def clean(title, text):
    """時期によって書式が変わる定型部分 (見出しの【Sports Watch】、関連リンクの一覧) を除く"""
    title = re.sub(r"^【[^】]*】\s*", "", title)
    text = text.split("関連リンク")[0]
    return title + "\n" + "\n".join(l for l in text.split("\n") if not l.startswith(("・", "■", "【")))

def nouns(text):
    return {w.surface for w in tagger(text) if w.feature.pos1 == "名詞" and w.feature.pos2 in ("普通名詞", "固有名詞")
            and len(w.surface) > 1}

df["words"] = [nouns(clean(a, b)) for a, b in zip(df["title"], df["text"])]   # 1 記事に何回出ても 1 と数える (文書頻度)

months = pd.period_range(df["month"].min(), df["month"].max(), freq="M")
d = df.groupby("month").size().reindex(months, fill_value=0).to_numpy()          # 各月の記事数
total = Counter(w for ws in df["words"] for w in ws)
vocab = sorted(w for w, c in total.items() if c >= 10)                                # 全体で 10 記事以上に出る語
R = pd.DataFrame(0, index=vocab, columns=months)                                  # 語 × 月 の記事数
vs = set(vocab)
for m, ws in zip(df["month"], df["words"]):
    for w in ws & vs:
        R.loc[w, m] += 1
print(R.shape)
```

```python
# 方法 1: 単純な頻度の比。その月の出現率を、直前 3 か月の出現率と比べる
def ratio_table(R, d, prev=3, min_count=5):
    rows = []
    for t in range(prev, len(d)):
        cur = (R.iloc[:, t] + 0.5) / (d[t] + 1)
        base = (R.iloc[:, t - prev:t].sum(axis=1) + 0.5) / (d[t - prev:t].sum() + 1)
        for w in R.index[R.iloc[:, t] >= min_count]:
            rows.append({"word": w, "month": str(R.columns[t]), "count": R.loc[w, R.columns[t]], "ratio": cur[w] / base[w]})
    return pd.DataFrame(rows).sort_values("ratio", ascending=False, kind="stable").reset_index(drop=True)

ratio = ratio_table(R, d)
ratio.head(15)
```

```python
import numpy as np
from scipy.stats import binom

# 方法 2: Kleinberg (2002) のバースト検出 (バッチのデータ用の 2 状態モデル)。
# 各月の記事数 d_t のうち語を含む記事数 r_t が、通常状態 (出現率 p0) と
# バースト状態 (出現率 p1 = s * p0) のどちらで説明しやすいかを、状態を上げるコスト gamma * log(n) と合わせて
# ビタビアルゴリズムで決める
def kleinberg_bursts(r, d, s=2.0, gamma=1.0):
    r, d = np.asarray(r), np.asarray(d)
    n = len(r)
    p0 = r.sum() / d.sum()
    p1 = min(s * p0, 0.9999)
    cost = -np.stack([binom.logpmf(r, d, p0), binom.logpmf(r, d, p1)])   # 2 × n
    up = gamma * np.log(n)
    C = np.array([cost[0, 0], cost[1, 0] + up])
    back = np.zeros((n, 2), dtype=int)
    for t in range(1, n):
        to0 = [C[0], C[1]]                # 状態を下げるコストは 0
        to1 = [C[0] + up, C[1]]
        back[t] = [np.argmin(to0), np.argmin(to1)]
        C = np.array([min(to0) + cost[0, t], min(to1) + cost[1, t]])
    q = np.zeros(n, dtype=int)
    q[-1] = C.argmin()
    for t in range(n - 1, 0, -1):
        q[t - 1] = back[t, q[t]]
    # 連続するバースト状態を 1 つのバーストとし、強さ = 区間内のコストの差の合計
    out, gain = [], cost[0] - cost[1]
    for t in range(n):
        if q[t] == 1 and (t == 0 or q[t - 1] == 0):
            start = t
        if q[t] == 1 and (t == n - 1 or q[t + 1] == 0):
            out.append((start, t, gain[start:t + 1].sum()))
    return out

rows = [{"word": w, "begin": str(months[b]), "end": str(months[e]), "b": b, "e": e, "weight": wt}
        for w in R.index for b, e, wt in kleinberg_bursts(R.loc[w].to_numpy(), d)]
bursts = pd.DataFrame(rows).sort_values("weight", ascending=False, kind="stable").reset_index(drop=True)
print(len(bursts), "個のバースト")
bursts.head(15)[["word", "begin", "end", "weight"]]
```

```python
# 2 つの方法の比較: 頻度の比の上位 15 件が、Kleinberg では何位のバーストの期間に入っているか
def kleinberg_rank(w, m):
    t = months.get_loc(pd.Period(m, "M"))
    hit = bursts[(bursts.word == w) & (bursts.b <= t) & (bursts.e >= t)]
    return hit.index[0] + 1 if len(hit) else None

cmp = ratio.head(15).copy()
cmp["kleinberg_rank"] = pd.array([kleinberg_rank(w, m) for w, m in zip(cmp.word, cmp.month)], dtype="Int64")
print("比が 3 倍以上の (語, 月):", (ratio.ratio >= 3).sum(), "件 / Kleinberg のバースト:", len(bursts), "個")
cmp
```

```python
import matplotlib.pyplot as plt
import matplotlib_fontja

# 強いバースト上位 20 個の期間を横棒で示す
top = bursts.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(top)), top["e"] - top["b"] + 1, left=top["b"] - 0.5, color="tab:orange")   # 月 i を i-0.5〜i+0.5 に描く
ax.set_yticks(range(len(top)), top["word"])
ticks = list(range(0, len(months), 3))
ax.set_xticks(ticks, [str(months[i]) for i in ticks], rotation=45)
ax.set_xlim(-0.5, len(months) - 0.5); ax.grid(axis="x", alpha=0.3)
ax.set_title("Kleinberg のバースト (強さの上位 20)")
plt.tight_layout(); plt.show()
```

```python
# 語ごとの出現率の推移。橙の帯が Kleinberg のバースト期間、赤い点が頻度の比で 3 倍以上になった月
words = ["ロンドン", "メダル", "なでしこ", "震災"]
fig, axes = plt.subplots(len(words), 1, figsize=(10, 2.2 * len(words)), sharex=True)
for ax, w in zip(axes, words):
    ax.plot(range(len(months)), R.loc[w] / d, marker=".", color="tab:blue")
    for _, b in bursts[bursts.word == w].iterrows():
        ax.axvspan(b["b"] - 0.5, b["e"] + 0.5, color="tab:orange", alpha=0.3)
    hit = ratio[(ratio.word == w) & (ratio.ratio >= 3)]
    idx = [months.get_loc(pd.Period(m, "M")) for m in hit["month"]]
    ax.scatter(idx, (R.loc[w] / d).iloc[idx], color="red", zorder=3)
    ax.set_ylabel(w); ax.set_ylim(bottom=0)
axes[-1].set_xticks(ticks, [str(months[i]) for i in ticks], rotation=45)
axes[-1].set_xlim(-0.5, len(months) - 0.5)
fig.suptitle("語を含む記事の割合 (月ごと)")
plt.tight_layout(); plt.show()
```

- 頻度の比は計算が簡単だが、直前にほとんど出ていない語が数記事出ただけで比が大きくなる。この出力でも比の上位 15 件は 5〜9 記事の語で、そのうち 6 件は Kleinberg のバーストの期間に入っていない。逆に直前の数か月も多い月は比が大きくならないので、「ロンドン」のように 2012 年に数か月かけて増えた語は、2012 年には 3 倍を超えない。
- Kleinberg のバースト検出は KDD 2002 で発表された (雑誌版は Data Mining and Knowledge Discovery, 2003)。ここでは論文のバッチ用のモデルで、出現率 p0 = R/D の状態と p1 = s·p0 の状態のどちらにいるかを、二項分布の負の対数尤度と状態を上げるコスト γ ln n の合計が最小になるように決める。頻度の比と違って始まりと終わりの期間が得られ、記事数の少ない月のばらつきを二項分布で考慮した強さ (p1 で説明したときのコストの減りの合計) で順位を付けられる。
- `s` を大きくすると大きく増えた語だけがバーストになる。`gamma` を大きくするとバーストに入りにくくなり、弱い増加や短い増加はバーストになりにくい。月の記事数が 19〜64 件と少ないので月単位にまとめたが、記事の多いデータなら週や日の単位にできる。
- PyPI の `burst_detection` (0.1.3) は NumPy から削除された `np.float` を使っていて、検証に使った Colab のイメージ (NumPy 2.4) ではエラーになり、分類も "Free for non-commercial use" なので、ここでは 2 状態のビタビを自分で書いた。メールのように 1 件ずつ届く時刻の列には、論文の前半の連続時間のモデル (R の `bursts` パッケージなど) を使う。
- livedoor ニュースコーパスの記事は CC BY-ND 2.1 JP (表示・改変禁止)。カテゴリごとに収録の時期が違う (例: topic-news は 2011 年 10 月から、smax は 2012 年 4 月から) ので、収録の始まりがバーストに見えないよう Sports Watch だけを使い、時期によって変わる定型文 (見出しの【Sports Watch】、関連リンクの一覧) も除いた。それでも上位の「ネット」「アサヒ」「週刊」「同誌」などは話題より書き方の変化を拾った可能性があり (中身は確かめていない)、検出結果は元の記事を読んで確かめる。
- 名詞を 1 語ずつ数えているので、「川澄」「穂希」「奈穂美」のように人名が分かれて別々に出る。名詞の連続をつないで数える (この章の「文書からキーワードを抽出する (埋め込みによる KeyBERT 方式)」の `candidates`) と読みやすくなる。
- 語の代わりに、この章の「LDA で文書集合のトピックを見つける」や「文埋め込みとクラスタリングでトピックを作る (BERTopic 方式)」で各文書に割り当てたトピックを数えて r_t にすれば、同じ関数で話題のバーストを検出できる。2 つの期間だけを比べるなら、コーパスの統計と可視化の章の「2 つのコーパスを比べて特徴語を見つける (対数尤度比)」も使える。
