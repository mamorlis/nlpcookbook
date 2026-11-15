# 文埋め込みとクラスタリング

文をベクトルにして意味の近さを測る。言い換えの検出、クラスタリング、可視化。検索への応用は「検索」の章を参照。

## 文をベクトルにして意味の近さを測る (Sentence Transformers) {sentence_embeddings}

言い回しが違っても意味が近い文を見つけたい。文を埋め込みベクトルにしてコサイン類似度を計算する。

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-small")

sentences = ["今日は天気が良いので散歩に行きます。", "晴れているから外を歩いてくるよ。", "明日の会議は午後 3 時からです。",
             "The weather is nice, so I'll go for a walk.", "機械学習モデルの学習には GPU が必要だ。"]
# E5 系のモデルは入力の先頭に "query: " か "passage: " を付ける約束になっている
emb = model.encode(["query: " + s for s in sentences], normalize_embeddings=True)
print(emb.shape)
```

```python
import pandas as pd

sim = model.similarity(emb, emb)  # コサイン類似度の行列
pd.DataFrame(sim.numpy(), index=[s[:12] for s in sentences], columns=range(len(sentences))).round(2)
```

- 日本語と英語の同じ意味の文も近くなる (多言語モデル)。
- モデルごとにプレフィックスの約束が違う (E5: `query:` / `passage:`、Ruri: `検索クエリ:` / `検索文書:` など)。モデルカードを必ず確認する。
- 日本語特化なら `cl-nagoya/ruri-v3-310m` や `sbintuitions/sarashina-embedding-v2-1b` など。[JMTEB](https://github.com/sbintuitions/JMTEB) のリーダーボードが参考になる。

## 文をクラスタリングして話題ごとにまとめる {sentence_clustering}

大量の文 (アンケートの自由記述、問い合わせなど) を、意味の近いもの同士で自動的にグループ分けしたい。

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans

texts = [
    "アプリがすぐ落ちる", "起動するとクラッシュします", "ログインできません", "パスワードを忘れてログインできない",
    "料金が高すぎる", "月額プランをもっと安くしてほしい", "解約方法がわからない", "退会の手続きを教えてください",
    "動作が重くて使いにくい", "読み込みが遅い", "値段の割に機能が少ない", "サインインの画面でエラーが出る",
]
model = SentenceTransformer("intfloat/multilingual-e5-small")
emb = model.encode(["query: " + t for t in texts], normalize_embeddings=True)
```

```python
# クラスタ数を決め打ちする場合: k-means
labels = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(emb)
for c in range(4):
    print(c, [t for t, l in zip(texts, labels) if l == c])
```

```python
# クラスタ数がわからない場合: 距離のしきい値で止める階層的クラスタリング
agg = AgglomerativeClustering(n_clusters=None, distance_threshold=0.13, metric="cosine", linkage="average")
labels = agg.fit_predict(emb)
for c in np.unique(labels):
    print(c, [t for t, l in zip(texts, labels) if l == c])
```

- しきい値はモデルによって適切な値が違う (E5 は全体に類似度が高めに出る)。少量のデータで目視して決める。
- 各クラスタの名前付けは、クラスタ内の代表文 (中心に最も近い文) を見るか、LLM に要約させるとよい。

## 重複・言い換えの文ペアを見つける {paraphrase_mining}

コーパスの中から、ほぼ同じ内容を述べている文のペアを洗い出したい (FAQ の整理、データの重複除去)。

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("intfloat/multilingual-e5-small")
sentences = [
    "返品はいつまで可能ですか？", "商品を返したいのですが期限はありますか。", "送料はいくらかかりますか？",
    "配送料を教えてください。", "支払い方法には何がありますか。", "クレジットカードは使えますか？",
    "返品の締め切りを知りたい。", "営業時間は何時から何時までですか？", "お店は何時に開きますか。",
]
pairs = util.paraphrase_mining(model, ["query: " + s for s in sentences], top_k=3)
for score, i, j in pairs[:8]:
    print(f"{score:.3f}  {sentences[i]}  ⇔  {sentences[j]}")
```

```python
# しきい値以上のペアをつないで、重複グループにまとめる
clusters = util.community_detection(model.encode(["query: " + s for s in sentences], convert_to_tensor=True),
                                    threshold=0.89, min_community_size=2)
for c in clusters:
    print([sentences[i] for i in c])
```

- `paraphrase_mining` は全ペアの総当たりを避けてチャンク単位で計算するので、数万文でも実行できる。

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

## 文埋め込みを 2 次元に可視化する (UMAP) {embedding_umap}

文の埋め込みベクトルを散布図にして、データセットの全体像 (クラスの分かれ方、外れ値) を眺めたい。

```python
!pip install -q umap-learn
```

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

cats = ["rec.sport.baseball", "sci.space", "comp.graphics", "talk.politics.mideast", "rec.autos"]
data = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))
keep = [i for i, d in enumerate(data.data) if len(d) > 100][:1500]
texts, y = [data.data[i][:1000] for i in keep], np.array([data.target[i] for i in keep])

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
print(emb.shape)
```

```python
import matplotlib.pyplot as plt
import umap

Z = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(emb)
plt.figure(figsize=(9, 7))
for c, name in enumerate(data.target_names):
    plt.scatter(Z[y == c, 0], Z[y == c, 1], s=6, label=name)
plt.legend(markerscale=3); plt.xticks([]); plt.yticks([]); plt.show()
```

- UMAP は t-SNE より速く、大域的な配置も比較的保たれる。`n_neighbors` を大きくすると大域構造、小さくすると局所構造を重視する。
- 点にマウスを載せて原文を見たいなら Plotly: `px.scatter(x=Z[:,0], y=Z[:,1], color=..., hover_name=texts)`。

## 文の意味的類似度 (STS) でモデルを評価する {sts_eval}

文埋め込みモデルが出す類似度が、人間の判断とどれくらい合っているかを測りたい。

```python
!pip install -q datasets
```

```python
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer

ds = load_dataset("sentence-transformers/stsb", split="test")
print(ds[0])

for name in ["sentence-transformers/all-MiniLM-L6-v2", "intfloat/multilingual-e5-small"]:
    model = SentenceTransformer(name)
    prefix = "query: " if "e5" in name else ""
    e1 = model.encode([prefix + s for s in ds["sentence1"]], normalize_embeddings=True, batch_size=64)
    e2 = model.encode([prefix + s for s in ds["sentence2"]], normalize_embeddings=True, batch_size=64)
    cos = (e1 * e2).sum(axis=1)
    print(f"{name:42} Spearman = {spearmanr(cos, ds['score']).statistic:.4f}  Pearson = {pearsonr(cos, ds['score']).statistic:.4f}")
```

- STS の評価は Spearman の順位相関で報告するのが慣例。
- 日本語なら JGLUE の JSTS や JSICK を使う。複数タスクでまとめて評価するなら MTEB / JMTEB。

## 埋め込みを量子化・次元削減して省メモリ化する {embedding_quantization}

数百万件の埋め込みを保存するとメモリを圧迫する。精度をあまり落とさずにサイズを小さくしたい。

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
from sklearn.datasets import fetch_20newsgroups

docs = [d[:500] for d in fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes")).data if len(d) > 200][:2000]
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(docs, normalize_embeddings=True, batch_size=64)
queries, corpus = emb[:200], emb[200:]
truth = (queries @ corpus.T).argsort(axis=1)[:, -10:]       # float32 での上位 10 件を正解とする

def recall(q, c, score):
    top = score(q, c).argsort(axis=1)[:, -10:]
    return np.mean([len(set(a) & set(b)) / 10 for a, b in zip(truth, top)])

int8 = quantize_embeddings(emb, precision="int8")
binary = quantize_embeddings(emb, precision="ubinary")      # 1 次元 = 1 ビット
hamming = lambda q, c: -np.unpackbits(q[:, None] ^ c[None], axis=-1).sum(-1)
dot = lambda q, c: q.astype(np.float32) @ c.astype(np.float32).T

print(f"float32: {emb.nbytes / len(emb):6.0f} バイト/件  recall@10 = 1.00")
print(f"int8   : {int8.nbytes / len(int8):6.0f} バイト/件  recall@10 = {recall(int8[:200], int8[200:], dot):.2f}")
print(f"binary : {binary.nbytes / len(binary):6.0f} バイト/件  recall@10 = {recall(binary[:200], binary[200:], hamming):.2f}")
half = emb[:, :192] / np.linalg.norm(emb[:, :192], axis=1, keepdims=True)
print(f"前半 192 次元: {half.nbytes / len(half):4.0f} バイト/件  recall@10 = {recall(half[:200], half[200:], dot):.2f}")
```

- バイナリ量子化で 1/32 のサイズになる。バイナリで上位 100 件を粗く取り、float32 で再スコアリングすると精度をほぼ回復できる。
- 「先頭の次元だけ使う」方法は、マトリョーシカ表現学習 (MRL) で訓練されたモデルなら精度がほとんど落ちない (このモデルは MRL ではない)。
