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

## BERT の出力から文ベクトルを自分で作る (mean pooling) {mean_pooling}

Sentence Transformers を使わずに、モデルの隠れ状態から文の埋め込みを計算したい。パディングを正しく無視するのがポイント。

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

name = "intfloat/multilingual-e5-small"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name).eval()

sentences = ["query: 今日は天気が良いので散歩に行きます。", "query: 晴れた。歩こう。",
             "query: 明日の会議は午後 3 時から 5 階の大会議室で行いますので、資料を 10 部印刷して持ってきてください。"]
batch = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    hidden = model(**batch).last_hidden_state            # (文数, トークン数, 次元)
print(hidden.shape)

mask = batch["attention_mask"].unsqueeze(-1).float()      # パディング部分を 0 にするマスク
mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1)       # 本物のトークンだけで平均
emb = F.normalize(mean, dim=-1)
print((emb @ emb.T).round(decimals=3))
```

```python
# よくある間違い: パディングも含めて平均してしまう / [CLS] を使う
naive = F.normalize(hidden.mean(dim=1), dim=-1)
cls = F.normalize(hidden[:, 0], dim=-1)
print("パディング込みの平均:", [round(x, 3) for x in (naive @ naive.T)[0].tolist()])
print("[CLS] ベクトル:      ", [round(x, 3) for x in (cls @ cls.T)[0].tolist()])

from sentence_transformers import SentenceTransformer
st = torch.tensor(SentenceTransformer(name).encode(sentences, normalize_embeddings=True))
print("Sentence Transformers と一致:", torch.allclose(st, emb, atol=1e-4))
```

- どのプーリング (平均 / CLS / 最後のトークン) を使うべきかはモデルの学習方法で決まる。モデルカードの指定に従う。
- 素の BERT (文埋め込み用に学習されていないモデル) の平均ベクトルは、類似度計算にはあまり向かない。

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

## 文埋め込みを PCA・t-SNE・UMAP で 2 次元に可視化して比べる {embedding_visualization}

文の埋め込みベクトルを散布図にして、データセットの全体像 (クラスの分かれ方、外れ値) を眺めたい。次元削減の手法による見え方の違いも知りたい。

```python
!pip install -q umap-learn matplotlib-fontja
```

```python
import io, tarfile, urllib.request
import numpy as np
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

df = load_livedoor()
df = df[df["label"].isin(["sports-watch", "movie-enter", "smax", "dokujo-tsushin", "kaden-channel"])]
df = df.groupby("label").sample(200, random_state=0).reset_index(drop=True)
model = SentenceTransformer("intfloat/multilingual-e5-small")
emb = model.encode(["query: " + t for t in df["title"]], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
print(emb.shape)
```

```python
import time
import matplotlib.pyplot as plt
import matplotlib_fontja
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

methods = {
    "PCA": lambda X: PCA(n_components=2, random_state=0).fit_transform(X),
    "t-SNE": lambda X: TSNE(n_components=2, perplexity=30, metric="cosine", init="pca", random_state=0).fit_transform(X),
    "UMAP": lambda X: umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(X),
}
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
coords = {}
for ax, (name, f) in zip(axes, methods.items()):
    t = time.perf_counter()
    Z = coords[name] = f(emb)
    for label in df["label"].unique():
        m = (df["label"] == label).to_numpy()
        ax.scatter(Z[m, 0], Z[m, 1], s=8, label=label)
    ax.set_title(f"{name} ({time.perf_counter() - t:.1f} 秒)"); ax.set_xticks([]); ax.set_yticks([])
axes[0].legend(markerscale=2, fontsize=9)
plt.tight_layout(); plt.show()
```

```python
# 2 次元に落としたあとも、元の空間での近傍が保たれているかを数値で比べる (trustworthiness: 1 に近いほど良い)
from sklearn.manifold import trustworthiness
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

pca = PCA(n_components=50, random_state=0).fit(emb)
print(f"PCA: 上位 2 成分の寄与率 {pca.explained_variance_ratio_[:2].sum():.1%}、50 成分で {pca.explained_variance_ratio_.sum():.1%}")
for name, Z in coords.items():
    knn = cross_val_score(KNeighborsClassifier(10), Z, df["label"], cv=5).mean()
    print(f"{name:6} trustworthiness = {trustworthiness(emb, Z, n_neighbors=10, metric='cosine'):.3f}  2 次元での kNN 正解率 = {knn:.3f}")
```

- PCA は線形の射影で、軸に意味がある (分散が最大の方向) が、2 次元では情報の大半が落ちる。全体の広がりを見るのに向く。高速で、結果が毎回同じ。
- t-SNE は近い点同士を近くに置くことを重視する。クラスタははっきり見えるが、クラスタ間の距離や大きさには意味がない。`perplexity` (近傍の数の目安) で見え方が変わる。
- UMAP は t-SNE に近い見え方で、より速く、大域的な配置も比較的保たれる。新しい点を後から同じ空間に射影することもできる (`transform`)。
- どの手法でも、2 次元の図は「眺める」ためのもの。クラスタリングや類似度の計算は、元の高次元のベクトルで行う。
- 点にマウスを載せて原文を見たいなら Plotly: `px.scatter(x=Z[:,0], y=Z[:,1], color=df["label"], hover_name=df["title"])`。

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
