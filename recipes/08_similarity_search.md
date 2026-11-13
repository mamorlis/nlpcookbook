# 文埋め込み・類似度・検索

文をベクトルにして類似度を測る、似た文書を探す、検索システムを組み立てる。

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

- 3 つ目のクエリのように語が一致しない言い換え (大きい ↔ 最大) には弱い。埋め込み検索と組み合わせると補い合える (次項)。
- 大規模なら `bm25s` (高速な実装) や Elasticsearch / OpenSearch、Pyserini を使う。

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

## 編集距離とあいまい文字列マッチング (RapidFuzz) {edit_distance}

表記ゆれやタイプミスを含む文字列同士の近さを測りたい。候補リストから一番近いものを選びたい。

```python
!pip install -q rapidfuzz
```

```python
from rapidfuzz import fuzz, process
from rapidfuzz.distance import DamerauLevenshtein, JaroWinkler, Levenshtein

for a, b in [("コンピューター", "コンピュータ"), ("自然言語処理", "自然原語処理"), ("recieve", "receive"), ("東京都立大学", "首都大学東京")]:
    print(f"{a} / {b}: 編集距離 {Levenshtein.distance(a, b)}  正規化類似度 {Levenshtein.normalized_similarity(a, b):.2f}"
          f"  転置込み {DamerauLevenshtein.distance(a, b)}  Jaro-Winkler {JaroWinkler.similarity(a, b):.2f}")
print(Levenshtein.editops("recieve", "receive"))
```

```python
# 候補の中から近いものを探す
universities = ["東京大学", "京都大学", "東京工業大学", "東京都立大学", "一橋大学", "大阪大学", "東京科学大学", "東北大学"]
for q in ["東京都率大学", "ひとつばし大学", "東工大"]:
    print(q, process.extract(q, universities, scorer=fuzz.WRatio, limit=2))
```

```python
# 単語単位の編集距離 (リストを渡す)
ref, hyp = "the cat sat on the mat".split(), "the cat sit on mat".split()
print(Levenshtein.distance(ref, hyp), Levenshtein.editops(ref, hyp))
```

- 「東工大」のような略称や「ひとつばし」のような読みの違いは、文字の編集距離では拾えない。読みに変換してから比べるか、別名辞書を使う。
- RapidFuzz は C++ 実装で、数百万件の突き合わせにも使える (`process.cdist` で行列計算)。

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

## MinHash と LSH で大量文書からほぼ重複を見つける {minhash_lsh}

数十万件の文書から、コピペや微修正による「ほぼ同じ文書」を、全ペア比較せずに高速に見つけたい。

```python
!pip install -q datasketch
```

```python
from datasketch import MinHash, MinHashLSH

def shingles(text, n=3):
    text = "".join(text.split())
    return {text[i:i + n] for i in range(len(text) - n + 1)}

def minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for s in shingles(text):
        m.update(s.encode("utf-8"))
    return m

docs = {
    "a": "自然言語処理は、人間が日常的に使っている言語をコンピュータに処理させる一連の技術である。",
    "b": "自然言語処理とは、人間が日常的に使っている言語をコンピューターに処理させる技術のことである。",
    "c": "機械学習は、データから規則性を学習してタスクを解くアルゴリズムの研究分野である。",
    "d": "自然言語処理は、人間が日常的に使っている言語をコンピュータに処理させる一連の技術である！",
}
mh = {k: minhash(v) for k, v in docs.items()}

def jaccard(a, b):
    A, B = shingles(docs[a]), shingles(docs[b])
    return len(A & B) / len(A | B)

for x, y in [("a", "b"), ("a", "c"), ("a", "d")]:
    print(f"{x}-{y}: 真の Jaccard = {jaccard(x, y):.2f}  MinHash による推定 = {mh[x].jaccard(mh[y]):.2f}")
```

```python
lsh = MinHashLSH(threshold=0.5, num_perm=128)
for k, m in mh.items():
    lsh.insert(k, m)
for k in docs:
    print(k, "と似ている:", sorted(set(lsh.query(mh[k])) - {k}))
```

- MinHash は集合の Jaccard 係数を固定長の署名で近似する。LSH は署名が似たものだけを同じバケットに入れ、候補を絞る。
- 大規模な事前学習コーパスの重複除去で標準的に使われている。意味的な重複 (言い換え) は拾えないので、それには埋め込みを使う。

## 言語をまたいで意味の近い文を対応付ける (対訳文の抽出) {cross_lingual_similarity}

日本語と英語の文集合から、互いに翻訳になっている文のペアを見つけたい (対訳コーパスの構築)。

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/LaBSE")

ja = ["猫がソファの上で寝ている。", "明日は雨が降るでしょう。", "この論文は新しい翻訳手法を提案する。", "駅までの道を教えてください。", "私はコーヒーより紅茶が好きだ。"]
en = ["It will probably rain tomorrow.", "I prefer tea to coffee.", "Could you tell me the way to the station?",
      "This paper proposes a novel translation method.", "The stock market crashed.", "A cat is sleeping on the sofa."]

sim = model.similarity(model.encode(ja, normalize_embeddings=True), model.encode(en, normalize_embeddings=True)).numpy()
for i, row in enumerate(sim):
    j = row.argmax()
    print(f"{row[j]:.2f}  {ja[i]}  ⇔  {en[j]}")
```

```python
# 相互最近傍 (双方向で互いに 1 位) かつしきい値以上のペアだけを採用すると、誤対応が減る
best_en, best_ja = sim.argmax(axis=1), sim.argmax(axis=0)
pairs = [(ja[i], en[j], sim[i, j]) for i, j in enumerate(best_en) if best_ja[j] == i and sim[i, j] > 0.6]
print(len(pairs), "ペア")
```

- LaBSE は 109 言語の対訳文で学習された、対訳文検索向けの文埋め込みモデル。
- 大規模な対訳抽出では、近傍との類似度の比を使う margin-based スコアリングが標準的。

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

## 巨大な辞書の語をテキストから一括で探す (Aho-Corasick) {aho_corasick}

数万〜数百万語の辞書 (商品名、病名、NG ワードなど) に載っている語を、長いテキストから高速にすべて見つけたい。

```python
!pip install -q pyahocorasick
```

```python
import ahocorasick

terms = {"東京": "LOC", "東京都": "LOC", "京都": "LOC", "東京都立大学": "ORG", "自然言語処理": "TERM", "言語": "TERM", "機械翻訳": "TERM"}
A = ahocorasick.Automaton()
for term, label in terms.items():
    A.add_word(term, (term, label))
A.make_automaton()

text = "東京都立大学では自然言語処理と機械翻訳を研究している。京都にも行った。"
for end, (term, label) in A.iter(text):          # 重なりも含めてすべての一致
    print(end - len(term) + 1, end + 1, term, label)
```

```python
# 最長一致だけ残す (重なったら長い語を優先)
for end, (term, label) in A.iter_long(text):
    print(end - len(term) + 1, end + 1, term, label)
```

```python
# 正規表現の | 連結と速度を比べる
import random, re, time

random.seed(0)
vocab = ["".join(random.choices("あいうえおかきくけこさしすせそ", k=random.randint(3, 6))) for _ in range(50000)]
big = "".join(random.choices("あいうえおかきくけこさしすせそ", k=200000))
B = ahocorasick.Automaton()
for w in vocab:
    B.add_word(w, w)
B.make_automaton()

t = time.perf_counter(); n1 = sum(1 for _ in B.iter(big)); t1 = time.perf_counter() - t
t = time.perf_counter(); n2 = sum(1 for w in vocab[:500] if w in big); t2 = time.perf_counter() - t
print(f"Aho-Corasick (5 万語): {t1:.3f} 秒 / 単純な in 判定 (500 語だけ): {t2:.3f} 秒")
```

- テキストを 1 回走査するだけで、辞書の語数に関係なくすべての一致を列挙できる。
- 「東京都」の中の「京都」のような誤一致は最長一致でも完全には防げない。形態素境界と照合すると精度が上がる。
