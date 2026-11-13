# トピック・キーワード・要約

文書集合に潜む話題を見つける、文書を代表する語や文を選ぶ。教師なしで全体像をつかむための手法。

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
- 統計ベースなら TF-IDF (コーパスの統計と特徴語の章を参照)、英語なら YAKE や RAKE、グラフベースなら TextRank などがある。ライブラリは `keybert` や `pke`。

## 重要な文を選んで抽出型要約を作る (LexRank) {extractive_summary}

長い文章から、内容を代表する文をいくつか抜き出して要約にしたい。生成モデルと違い、原文にない内容が混ざらない。

```python
!pip install -q fugashi unidic-lite
```

```python
import re
import networkx as nx
import numpy as np
from fugashi import Tagger
from sklearn.feature_extraction.text import TfidfVectorizer

tagger = Tagger()
tok = lambda s: [w.feature.orthBase or w.surface for w in tagger(s) if w.feature.pos1 in ("名詞", "動詞", "形容詞") and w.feature.pos2 != "非自立可能"]

def lexrank(text, n=3, threshold=0.1):
    sents = [s.strip() for s in re.split(r"(?<=[。！？])", text) if s.strip()]
    X = TfidfVectorizer(tokenizer=tok, token_pattern=None, lowercase=False).fit_transform(sents)
    sim = (X @ X.T).toarray()                      # 文同士のコサイン類似度
    np.fill_diagonal(sim, 0)
    sim[sim < threshold] = 0
    scores = nx.pagerank(nx.from_numpy_array(sim), weight="weight")   # 多くの文と似ている文 = 中心的な文
    top = sorted(sorted(scores, key=scores.get, reverse=True)[:n])    # スコア上位を、元の順番で並べる
    return [sents[i] for i in top], [round(scores[i], 3) for i in range(len(sents))]

text = ("自然言語処理は、人間が日常的に使っている言語をコンピュータに処理させる技術である。"
        "その歴史は 1950 年代の機械翻訳の研究にさかのぼる。"
        "初期の機械翻訳は人手で書かれた規則に基づいていたが、十分な精度は得られなかった。"
        "1990 年代になると、大量の対訳データから統計的に翻訳を学習する統計的機械翻訳が主流となった。"
        "2010 年代半ばには、ニューラルネットワークを用いたニューラル機械翻訳が登場し、翻訳の精度は飛躍的に向上した。"
        "現在では、機械翻訳に限らず、自然言語処理のほとんどのタスクでニューラルネットワークが使われている。"
        "特に、大量のテキストで事前学習された大規模言語モデルは、少ない追加学習で多様なタスクに適用できる。"
        "ところで、昨日の夕食はカレーだった。")
summary, scores = lexrank(text)
print(scores)
for s in summary:
    print("・", s)
```

- 文をノード、類似度を辺の重みとするグラフを作り、PageRank で中心的な文を選ぶ。無関係な文 (最後の文) はどの文とも似ていないのでスコアが低い。
- 類似度の計算を TF-IDF から文埋め込みに替えると、言い換えにも強くなる。
- ニュース記事では「先頭の 3 文を取る」(Lead-3) が非常に強いベースラインになる。要約手法を評価するときは必ず比較する。
- 英語を含む各種の抽出型要約アルゴリズムは `sumy` にまとまっている。
