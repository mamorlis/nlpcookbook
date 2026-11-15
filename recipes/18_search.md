# 検索

キーワード検索と意味検索、その組み合わせと並べ替え。質問応答や RAG の土台になる。

## TF-IDF とコサイン類似度で似た文書を探す {tfidf_similarity}

ニューラルモデルを使わずに、単語の重なりに基づいて類似文書を高速に検索したい。

```python
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
docs = [d for d in data.data if len(d) > 200][:5000]
vec = TfidfVectorizer(stop_words="english", max_df=0.5, min_df=2, sublinear_tf=True)
X = vec.fit_transform(docs)   # 疎行列 (文書数 × 語彙数)。行は L2 正規化済み
print(X.shape, f"非ゼロ要素の割合 {X.nnz / (X.shape[0] * X.shape[1]):.4f}")
```

```python
def search(query, k=3):
    sims = cosine_similarity(vec.transform([query]), X).ravel()
    for i in np.argsort(-sims)[:k]:
        print(f"  {sims[i]:.3f}  {' '.join(docs[i].split())[:90]}")

for q in ["space shuttle launch NASA", "best graphics card for windows"]:
    print(q)
    search(q)
```

```python
# 文書同士の類似度: 0 番の文書に似た文書
sims = cosine_similarity(X[0], X).ravel()
print(" ".join(docs[0].split())[:100])
for i in np.argsort(-sims)[1:3]:
    print(f"  {sims[i]:.3f}  {' '.join(docs[i].split())[:90]}")
```

- 日本語の場合は `TfidfVectorizer(tokenizer=分かち書き関数, token_pattern=None)` とする。
- 文字 n-gram (`analyzer="char", ngram_range=(2, 3)`) にすると、分かち書きなしで表記ゆれにも強くなる。

## BM25 で日本語のキーワード検索をする {bm25_ja}

単語の一致に基づく古典的な検索 (BM25) を日本語文書に対して行いたい。固有名詞や型番の検索には今でも強い。

```python
!pip install -q rank_bm25 fugashi unidic-lite
```

```python
from fugashi import Tagger
from rank_bm25 import BM25Okapi

tagger = Tagger()

def tokenize(text):
    return [w.feature.orthBase or w.surface for w in tagger(text.lower())
            if w.feature.pos1 not in ("助詞", "助動詞", "補助記号", "空白")]

docs = [
    "富士山は標高 3776 メートルで、日本で最も高い山である。",
    "東京スカイツリーは高さ 634 メートルの電波塔である。",
    "琵琶湖は滋賀県にある日本最大の湖である。",
    "信濃川は全長 367 キロメートルで、日本で最も長い川である。",
    "形態素解析とは、文を単語に分割して品詞を付与する処理である。",
    "BERT は Transformer のエンコーダを用いた事前学習モデルである。",
]
bm25 = BM25Okapi([tokenize(d) for d in docs])

for q in ["日本で一番高い山", "BERT とは何ですか", "日本一大きい湖はどこ？"]:
    scores = bm25.get_scores(tokenize(q))
    print(q, tokenize(q))
    for i in scores.argsort()[::-1][:2]:
        print(f"  {scores[i]:.2f}  {docs[i]}")
```

- 3 つ目のクエリのように語が一致しない言い換え (大きい ↔ 最大) には弱い。埋め込み検索と組み合わせると補い合える (ハイブリッド検索のレシピを参照)。
- 大規模なら `bm25s` (高速な実装) や Elasticsearch / OpenSearch、Pyserini を使う。

## 埋め込みで意味検索 (セマンティックサーチ) をする {semantic_search}

キーワードが一致しなくても、質問と意味が近い文書を検索したい。

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("intfloat/multilingual-e5-small")

docs = [
    "富士山は標高 3776 メートルで、日本で最も高い山である。",
    "東京スカイツリーは高さ 634 メートルの電波塔である。",
    "琵琶湖は滋賀県にある日本最大の湖である。",
    "信濃川は全長 367 キロメートルで、日本で最も長い川である。",
    "形態素解析とは、文を単語に分割して品詞を付与する処理である。",
    "Transformer は自己注意機構に基づくニューラルネットワークである。",
    "カレーライスは明治時代に日本に伝わった料理である。",
]
doc_emb = model.encode(["passage: " + d for d in docs], normalize_embeddings=True, convert_to_tensor=True)

def search(query, k=3):
    q = model.encode("query: " + query, normalize_embeddings=True, convert_to_tensor=True)
    for hit in util.semantic_search(q, doc_emb, top_k=k)[0]:
        print(f"  {hit['score']:.3f}  {docs[hit['corpus_id']]}")

for q in ["日本一大きい湖はどこ？", "単語に区切る技術", "What is the tallest mountain in Japan?"]:
    print(q)
    search(q)
```

- 文書側の埋め込みは 1 回計算して保存しておき、検索時はクエリだけ埋め込む。
- 数十万件までなら総当たり (この方法) で十分速い。それ以上は近似最近傍探索 (次項の FAISS) を使う。

## FAISS で大量のベクトルから高速に近傍検索する {faiss_index}

数十万〜数億件の埋め込みベクトルから、クエリに近いものを高速に取り出したい。

```python
!pip install -q faiss-cpu
```

```python
import time
import faiss
import numpy as np

rng = np.random.default_rng(0)
d, n = 384, 200_000
# 実際の埋め込みに似せて、1000 個のクラスタ中心 + ノイズで人工データを作る
centers = rng.standard_normal((1000, d), dtype=np.float32)
xb = centers[rng.integers(0, 1000, n)] + 0.7 * rng.standard_normal((n, d), dtype=np.float32)
xq = centers[rng.integers(0, 1000, 100)] + 0.7 * rng.standard_normal((100, d), dtype=np.float32)
faiss.normalize_L2(xb); faiss.normalize_L2(xq)   # 正規化しておけば内積 = コサイン類似度

flat = faiss.IndexFlatIP(d)                      # 厳密な総当たり
flat.add(xb)
t = time.perf_counter(); D, I = flat.search(xq, 10); t_flat = time.perf_counter() - t
print(f"Flat: {t_flat * 1000:.0f} ms / 100 クエリ")
```

```python
# 近似探索: HNSW (グラフ) と IVF (クラスタリング)
hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
hnsw.add(xb)
hnsw.hnsw.efSearch = 64                           # 探索時に保持する候補数 (大きいほど正確で遅い)

ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, 256, faiss.METRIC_INNER_PRODUCT)
ivf.train(xb); ivf.add(xb)
ivf.nprobe = 16                                   # 探索するクラスタ数 (大きいほど正確で遅い)

for name, index in [("HNSW", hnsw), ("IVF", ivf)]:
    t = time.perf_counter(); _, I2 = index.search(xq, 10); dt = time.perf_counter() - t
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(I, I2)])
    print(f"{name}: {dt * 1000:.0f} ms / 100 クエリ, recall@10 = {recall:.2f}")
```

```python
faiss.write_index(hnsw, "index.faiss")           # 保存と読み込み
index = faiss.read_index("index.faiss")
print(index.ntotal)
```

- 近似探索は速度と recall のトレードオフ。`efSearch` や `nprobe` を上げると recall が上がり、遅くなる。自分のデータで必ず recall を測る。
- 100 万件以下で精度重視なら Flat か HNSW、メモリを節約したいなら IVF + PQ (直積量子化)。
- GPU 版は `faiss-gpu`。メタデータによる絞り込みや永続化まで欲しければ Qdrant や Chroma などのベクトル DB を検討する。

## キーワード検索と埋め込み検索を組み合わせる (ハイブリッド検索、RRF) {hybrid_search}

BM25 と埋め込み検索の結果を統合して、両方の長所を活かしたい。

```python
!pip install -q rank_bm25 fugashi unidic-lite
```

```python
import numpy as np
from fugashi import Tagger
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

docs = [
    "RTX 4090 は 24GB の VRAM を搭載した GPU である。",
    "グラフィックボードはゲームや機械学習の計算を高速化する部品である。",
    "RX 7900 XTX は AMD のハイエンド GPU である。",
    "CPU はコンピュータの中央演算処理装置である。",
    "メモリが不足すると、大きなモデルは読み込めない。",
    "画像処理用の半導体は深層学習の学習に広く使われている。",
]
tagger = Tagger()
tok = lambda s: [w.surface.lower() for w in tagger(s) if w.feature.pos1 not in ("助詞", "助動詞", "補助記号", "空白")]
bm25 = BM25Okapi([tok(d) for d in docs])
model = SentenceTransformer("intfloat/multilingual-e5-small")
doc_emb = model.encode(["passage: " + d for d in docs], normalize_embeddings=True)
```

```python
def rrf(rankings, k=60):
    """Reciprocal Rank Fusion: 各ランキングでの順位の逆数を足し合わせる"""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

def search(query, topn=3):
    sparse = list(np.argsort(-bm25.get_scores(tok(query))))
    dense = list(np.argsort(-(doc_emb @ model.encode("query: " + query, normalize_embeddings=True))))
    print(query)
    for name, ranking in [("BM25  ", sparse), ("Dense ", dense), ("Hybrid", rrf([sparse, dense]))]:
        print(f"  {name}", [int(i) for i in ranking[:topn]])
    print("  →", docs[rrf([sparse, dense])[0]])

search("RTX 4090 の VRAM 容量")
search("深層学習に使うグラボ")
```

- RRF はスコアの尺度が違うランキング同士を、順位だけを使って統合する。調整するパラメータがほぼなく (k=60 が定番)、頑健。
- スコアを正規化して重み付き和をとる方法もあるが、重みの調整が必要になる。

## クロスエンコーダで検索結果を並べ替える (リランキング) {rerank}

検索の上位候補を、より精度の高いモデルで並べ替えて、本当に関連する文書を一番上に持ってきたい。

```python
!pip install -q fugashi unidic-lite
```

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("hotchpotch/japanese-reranker-cross-encoder-xsmall-v1", max_length=512)

query = "日本で一番高い山は何ですか？"
candidates = [
    "日本の山の標高ランキングは登山者の間でよく話題になる。",
    "エベレストは世界で最も高い山で、標高は 8849 メートルである。",
    "富士山は標高 3776 メートルで、日本の最高峰である。",
    "北岳は標高 3193 メートルで、富士山に次いで日本で 2 番目に高い。",
    "日本で一番長い川は信濃川である。",
]
scores = reranker.predict([(query, c) for c in candidates])
for s, c in sorted(zip(scores, candidates), reverse=True):
    print(f"{s:.3f}  {c}")
```

```python
# rank() を使うと並べ替えまでやってくれる
for r in reranker.rank(query, candidates, top_k=2, return_documents=True):
    print(r["corpus_id"], round(float(r["score"]), 3), r["text"])
```

- バイエンコーダ (文埋め込み) はクエリと文書を別々にベクトル化するので速いが粗い。クロスエンコーダは両方を連結して入力するので正確だが遅い。
- 定番の構成: BM25 や埋め込み検索で上位 100 件を取得 → クロスエンコーダで上位 10 件に絞る。
- 多言語なら `BAAI/bge-reranker-v2-m3`、日本語で高精度なら `hotchpotch/japanese-reranker-cross-encoder-large-v1` など。
