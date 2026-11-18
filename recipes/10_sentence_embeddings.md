# 文埋め込みとクラスタリング

文をベクトルにして意味の近さを測る。言い換えの検出、クラスタリング、可視化、対照学習による文埋め込みモデルの学習。検索への応用は「検索」の章を参照。

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

- E5 は類似度が全体に高く出て、幅が狭い。ここでは無関係な文どうしでも 0.72〜0.84 ある。値の大きさではなく順位で比べる。いちばん近いのは言い換えの文 0 と文 1（0.91）。
- 多言語モデルなので英語の文も比べられるが、無関係な文との差は小さい。英訳（文 3）にいちばん近いのは文 0（0.85）だが、文 0 と無関係な会議の文（文 2）も 0.84 ある。文 1 と英訳は 0.81 で、文 1 と文 2 の 0.82 より低い。
- モデルごとにプレフィックスの約束が違う（E5: `query:` / `passage:`、Ruri: `検索クエリ:` / `検索文書:` など）。モデルカードを必ず確認する。
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

- どのプーリング（平均 / CLS / 最後のトークン）を使うべきかはモデルの学習方法で決まる。モデルカードの指定に従う。
- 素の BERT（文埋め込み用に学習されていないモデル）の平均ベクトルは、類似度計算にはあまり向かない。対照学習で文埋め込みモデルにすると何が変わるかは、この章の「文埋め込み用でない BERT を対照学習で文埋め込みモデルにする（SimCSE、Sentence Transformers）」を参照。

## 重複・言い換えの文ペアを見つける {paraphrase_mining}

コーパスの中から、ほぼ同じ内容を述べている文のペアを洗い出したい（FAQ の整理、データの重複除去）。

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

大量の文（アンケートの自由記述、問い合わせなど）を、意味の近いもの同士で自動的にグループ分けしたい。

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

- しきい値はモデルによって適切な値が違う（E5 は全体に類似度が高めに出る）。少量のデータで目視して決める。
- 各クラスタの名前付けは、クラスタ内の代表文（中心に最も近い文）を見るか、LLM に要約させるとよい。

## 文埋め込みを PCA・t-SNE・UMAP で 2 次元に可視化して比べる {embedding_visualization}

文の埋め込みベクトルを散布図にして、データセットの全体像（クラスの分かれ方、外れ値）を眺めたい。次元削減の手法による見え方の違いも知りたい。

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

- PCA は線形の射影で、軸に意味がある（分散が最大の方向）が、2 次元では情報の大半が落ちる。全体の広がりを見るのに向く。高速で、結果が毎回同じ。
- t-SNE は近い点同士を近くに置くことを重視する。クラスタははっきり見えるが、クラスタ間の距離や大きさには意味がない。`perplexity`（近傍の数の目安）で見え方が変わる。
- UMAP は t-SNE に近い見え方で、大域的な配置も比較的保たれる。件数が多いと t-SNE より速い（図の 23.2 秒は、初回の実行に numba のコンパイルの時間が入るため）。新しい点を後から同じ空間に射影することもできる (`transform`)。
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

## 文埋め込み用でない BERT を対照学習で文埋め込みモデルにする（SimCSE、Sentence Transformers） {simcse gpu}

分野に合った文埋め込みモデルがなく、手元にあるのは文埋め込み用に学習されていない BERT とラベルなしの文（あれば NLI のような文の組）だけである。Sentence Transformers の `SentenceTransformerTrainer` と `MultipleNegativesRankingLoss` で対照学習 (SimCSE) をして、文埋め込みモデルを作る。

```python
!pip install -q fugashi unidic-lite datasets accelerate
```

```python
import json, random, re, urllib.request
import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers.utils import logging

logging.set_verbosity_error()   # モデルを読むたびに出る「使わない重み」の一覧を出さない
random.seed(0); np.random.seed(0); torch.manual_seed(0)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う

# 学習データ: 日本語 Wikipedia (wikimedia/wikipedia の 20231101.ja、CC BY-SA 3.0 / GFDL) の先頭の記事から文を切り出す
# streaming=True なので全体 (約 4GB) はダウンロードしない
wiki = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
sents = []
for article in wiki:
    for line in article["text"].split("\n"):
        sents += [s for s in re.findall(r"[^。]+。", line) if 20 <= len(s) <= 100]
    if len(sents) >= 40000:
        break
random.shuffle(sents)
print(len(sents), "文", sents[:3])

# 評価データ: JGLUE の JSTS v1.3 (0〜5 の類似度、CC BY-SA 4.0)。検証セット (1457 組) でチェックポイントを選び、テストセット (1589 組) で最後に測る
def load_jsts(split):
    url = f"https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jsts-v1.3/{split}-v1.3.json"
    return [json.loads(l) for l in urllib.request.urlopen(url).read().decode("utf-8").splitlines()]
jsts_valid, jsts_test = load_jsts("valid"), load_jsts("test")
print(len(jsts_valid), "組 /", len(jsts_test), "組", jsts_test[1])
```

```python
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator   # 5.7 より前の版では sentence_transformers.evaluation

name = "tohoku-nlp/bert-base-japanese-v3"   # 東北大の日本語 BERT (Apache 2.0)。MLM で事前学習しただけで、文埋め込み用の学習はしていない

def new_model():
    # Sentence Transformers 用の設定がないモデル名を渡すと、BERT + 平均プーリングの文埋め込みモデルが組み立てられる
    model = SentenceTransformer(name, device="cuda")
    model.max_seq_length = 64
    return model

def sts_evaluator(data, split):
    return EmbeddingSimilarityEvaluator([d["sentence1"] for d in data], [d["sentence2"] for d in data],
                                        [d["label"] / 5 for d in data], batch_size=256, name=f"jsts-{split}")
dev_eval, test_eval = sts_evaluator(jsts_valid, "dev"), sts_evaluator(jsts_test, "test")
jsts_spearman = lambda model: test_eval(model)["jsts-test_spearman_cosine"]   # この章の「文の意味的類似度 (STS) でモデルを評価する」と同じ指標

bert = new_model()
print(bert)
print(f"学習前の BERT (平均プーリング): JSTS 検証の Spearman = {dev_eval(bert)['jsts-dev_spearman_cosine']:.4f}"
      f"  テストの Spearman = {jsts_spearman(bert):.4f}")
```

```python
import shutil
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.training_args import BatchSamplers   # 5.7 より前の版では sentence_transformers.training_args
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss, CachedMultipleNegativesRankingLoss   # 5.7 より前の版では sentence_transformers.losses

def train(train_ds, batch_size=64, scale=20.0, lr=3e-5, seed=0, eval_steps=50):
    """train_ds の 1 列目と 2 列目の同じ行が正例、バッチ内の他の行の 2 列目 (と 3 列目) が負例。scale は 1 / 温度"""
    model = new_model()
    args = SentenceTransformerTrainingArguments(
        output_dir=f"ckpt-{len(train_ds)}-{batch_size}-{scale}-{seed}", save_only_model=True,
        num_train_epochs=1, per_device_train_batch_size=batch_size, learning_rate=lr, fp16=not BF16, bf16=BF16, seed=seed, report_to="none", disable_tqdm=True, logging_steps=eval_steps,
        # eval_steps ごとに JSTS の検証セットで測り、最後に最良のチェックポイントに戻す (テストセットでは選ばない)
        eval_strategy="steps", eval_steps=eval_steps, save_strategy="steps", save_steps=eval_steps, save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_jsts-dev_spearman_cosine",
        # 同じ文が同じバッチに入ると、その文が自分の正例と同じなのに負例として扱われる (偽の負例)。それを避けるバッチの作り方
        batch_sampler=BatchSamplers.NO_DUPLICATES)
    if batch_size <= 64:
        loss = MultipleNegativesRankingLoss(model, scale=scale)
    else:
        # 大きなバッチは GradCache 版にする。64 文ずつ順伝播するのでメモリはバッチ 64 程度で済み、損失は同じになる (ドロップアウトの乱数も再現する)。
        # 普通の MultipleNegativesRankingLoss でバッチ 256 にすると、このノートブックの最大メモリが 15.9GB になり、T4 (16GB) には収まらない
        loss = CachedMultipleNegativesRankingLoss(model, scale=scale, mini_batch_size=64)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds, loss=loss, evaluator=dev_eval)
    trainer.train()
    best = trainer.state.best_model_checkpoint.split("-")[-1]
    print(f"選んだチェックポイント: {best} ステップ目 (検証セットの Spearman {trainer.state.best_metric:.4f})")
    # チェックポイントは 1 つ約 0.44GB で、ckpt-* に最良と最新の 2 つまで残る。最良のモデルはもう読み込んであるので消す
    shutil.rmtree(args.output_dir)
    return model

# 教師なし SimCSE: anchor と positive に同じ文を入れる。学習中はドロップアウト (p=0.1) が有効なので、同じ文でも 2 つの埋め込みは少しずれる
unsup_ds = Dataset.from_dict({"anchor": sents[:12800], "positive": sents[:12800]})

# 12800 文、バッチ 64 で 200 ステップ。乱数の種を変えて 3 回学習する (Colab の T4 で 5〜8 分の見込み)
scores = []
for seed in range(3):
    simcse = train(unsup_ds, seed=seed)
    scores.append(jsts_spearman(simcse))
    print(f"教師なし SimCSE (seed={seed}): JSTS テストの Spearman = {scores[-1]:.4f}")
print(f"平均 {np.mean(scores):.4f} (最小 {min(scores):.4f}、最大 {max(scores):.4f})")
```

```python
# 学習前後で類似度の値がどう変わるか (simcse は上で最後に学習した seed=2 のモデル)
def jsts_cos(model, data=jsts_test):
    e1 = model.encode([d["sentence1"] for d in data], batch_size=256, normalize_embeddings=True, convert_to_tensor=True)
    e2 = model.encode([d["sentence2"] for d in data], batch_size=256, normalize_embeddings=True, convert_to_tensor=True)
    return (e1 * e2).sum(-1).float().cpu().numpy()

gold = np.array([d["label"] for d in jsts_test])
pairs = [("猫がソファの上で寝ている。", "ソファで猫が眠っています。"),
         ("猫がソファの上で寝ている。", "犬が公園を走り回っている。"),
         ("猫がソファの上で寝ている。", "株価が大きく値下がりした。")]
for a, b in pairs:
    r = [m.similarity(m.encode([a]), m.encode([b])).item() for m in (bert, simcse)]
    print(f"学習前 {r[0]:.3f}  学習後 {r[1]:.3f}  {a} / {b}")

for label, m in [("学習前", bert), ("学習後", simcse)]:
    # ランダムな文の組の類似度の平均: 素の BERT はどの文同士も似ている (異方性)
    e = m.encode(sents[-512:], batch_size=256, normalize_embeddings=True, convert_to_tensor=True)
    rand = (e @ e.T)[~torch.eye(512, dtype=bool, device=e.device)].mean().item()
    # 人手の類似度が低い組 (0〜1) と高い組 (4〜5) のコサイン類似度の分布
    c = jsts_cos(m)
    lo, hi = c[gold <= 1], c[gold >= 4]
    print(f"{label}: ランダムな組の平均 {rand:.3f}  JSTS の 0〜1 の組 {np.mean(lo):.3f} (90 パーセンタイル {np.percentile(lo, 90):.3f})"
          f"  4〜5 の組 {np.mean(hi):.3f} (10 パーセンタイル {np.percentile(hi, 10):.3f})")
```

```python
# 温度 (scale = 1 / 温度) とバッチサイズを変えてみる (それぞれ seed=0。既定の scale=20 は温度 0.05。Colab の T4 で 6〜10 分の見込み)
for scale, b in [(100.0, 64), (1.0, 64), (20.0, 256)]:
    print(f"温度 {1 / scale:<4} バッチ {b:<3}: JSTS テストの Spearman = {jsts_spearman(train(unsup_ds, batch_size=b, scale=scale, eval_steps=200 // (b // 64) // 4)):.4f}")
```

```python
# 教師あり SimCSE: JSNLI (SNLI の日本語訳、京都大学、CC BY-SA 4.0) の「含意」を正例、「矛盾」をハードネガティブにする
import io, zipfile
url = "https://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi?down=https://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JSNLI/jsnli_1.1.zip&name=JSNLI.zip"
with urllib.request.urlopen(url) as r:
    z = zipfile.ZipFile(io.BytesIO(r.read()))
rows = [l.split("\t") for l in z.read("jsnli_1.1/train_w_filtering.tsv").decode("utf-8").splitlines()]
unspace = lambda s: re.sub(r"(?<![A-Za-z0-9]) | (?![A-Za-z0-9])", "", s.strip())   # 単語区切りの空白を取る
hyp = {}
for label, premise, hypothesis in rows:
    hyp.setdefault(unspace(premise), {}).setdefault(label, unspace(hypothesis))
triples = [(p, h["entailment"], h["contradiction"]) for p, h in hyp.items() if "entailment" in h and "contradiction" in h]
random.shuffle(triples)
print(len(triples), "組", triples[0])

# 3 列 (anchor, positive, negative) にすると、MultipleNegativesRankingLoss は negative の列もバッチ全体で負例に加える
sup_ds = Dataset.from_dict(dict(zip(["anchor", "positive", "negative"], map(list, zip(*triples[:16000])))))
# 16000 組、バッチ 64 で 250 ステップ (ダウンロードを含めて Colab の T4 で 2〜4 分の見込み)
sup = train(sup_ds, lr=5e-5)
print(f"教師あり SimCSE: JSTS テストの Spearman = {jsts_spearman(sup):.4f}")
```

```python
# 差がばらつきの範囲かどうか: 評価の 1589 組をブートストラップで再標本化して、Spearman の差の 95% 信頼区間を求める
cos = {"学習前": jsts_cos(bert), "教師なし": jsts_cos(simcse), "教師あり": jsts_cos(sup)}
idx = np.random.default_rng(0).integers(0, len(gold), (1000, len(gold)))
for a, b in [("学習前", "教師なし"), ("学習前", "教師あり")]:
    d = [spearmanr(cos[b][i], gold[i]).statistic - spearmanr(cos[a][i], gold[i]).statistic for i in idx]
    print(f"{b} − {a}: {np.mean(d):+.4f}  95% 信頼区間 [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")

# 保存して読み直せば、この章のほかのレシピと同じように SentenceTransformer として使える
sup.save("simcse-sup-ja")
reloaded = SentenceTransformer("simcse-sup-ja")
print(reloaded.similarity(reloaded.encode([pairs[0][0]]), reloaded.encode([pairs[0][1]])))
print(f"最大メモリ {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
```

- 以下の数値は、A5000 で float16 の経路（`BF16 = False`、T4 と同じ dtype）で実行したもの（最大メモリ 6.9GB）。普通の `MultipleNegativesRankingLoss` でバッチ 256 にしたときは 15.9GB になったので、バッチ 64 より大きいときは `CachedMultipleNegativesRankingLoss`（GradCache、Gao ら 2021）を使う。64 文ずつ順伝播して勾配をためるので、メモリはバッチの大きさにほとんどよらない。そのかわり学習は遅くなる。
- SimCSE は Gao, Yao, Chen (EMNLP 2021) の手法。教師なしでは、同じ文を 2 回エンコードしたときのドロップアウト (p=0.1) のマスクの違いだけがデータ拡張になり、バッチ内の他の文が負例になる。論文の教師なしは英語 Wikipedia の 100 万文、バッチ 64、学習率 3e-5、温度 0.05 で、[CLS] に MLP を載せて学習し、評価時は MLP を外す。教師ありは SNLI と MNLI（論文の記載で 31.4 万組）、バッチ 512、学習率 5e-5、3 エポックである。公開コードの教師なしの例 (run_unsup_example.sh) は最大 32 トークンで学習している。このレシピは平均プーリングで MLP を使わず、最大の長さを 64 トークンにし（学習に使う 1 万 2800 文のうち、32 トークンを超える文は 50%、64 トークンを超える文は 2.5%。評価も同じ長さで行う）、教師なしは 1 万 2800 文、教師ありはバッチ 64 で 1 万 6000 組（1 エポック未満）に縮めた。論文と同じく開発用のデータ（ここでは JSTS の検証セット）で最良のチェックポイントを選んでいる。
- JSTS のテストセット（1589 組）の Spearman は、学習前の平均プーリングで 0.7536、教師なし SimCSE で 0.7543〜0.7624（乱数の種 3 つ、平均 0.7585）になった。seed=2 (0.7589) と学習前との差 (+0.0052) のブートストラップの 95% 信頼区間は −0.0078〜+0.0185 で 0 をまたぎ、この規模ではばらつきと区別できない。検証セットでも 0.7409 から 0.750〜0.756 にしか上がっていない。損失は 100 ステップで 0.0002 まで下がり、同じ文の組を当てる課題はすぐに簡単になってしまう。日本語 SimCSE の報告（Tsukagoshi ら 2023）は、約 100 万文で学習し、ハイパーパラメータとチェックポイントを開発セット (JSICK) で選んで、同じ BERT で JSTS の検証セットが 74.09（学習前、平均プーリング）から 78.70（5 回の平均）に上がったと報告している。教師なしで効果を出すには、文の数を大きく増やす必要がある。
- 順位 (Spearman) はほとんど変わらなくても、類似度の値は大きく変わる。ランダムな文の組のコサイン類似度の平均は 0.834 から 0.108 に、「猫が…」と「株価が…」は 0.755 から 0.060 に下がった。学習前は、人手の類似度が 0〜1 の組の平均が 0.888、4〜5 の組の平均が 0.967 で、どの組も高い値に詰まっていて（異方性）、0〜1 の組の 90 パーセンタイル (0.924) と 4〜5 の組の 10 パーセンタイル (0.942) の差は 0.02 しかない。学習後は 0.523 と 0.658 で、差が 0.13 に広がった。この章の「BERT の出力から文ベクトルを自分で作る (mean pooling)」で、素の BERT の平均ベクトルは類似度の計算にあまり向かないとしたのはこの点である。値をしきい値に使うなら、モデルを替えるたびにしきい値を決め直す。
- 温度を 1.0 にすると、損失は 3.22 付近で下げ止まり（ln 64 ≈ 4.16 から少ししか下がらない）、Spearman はテストで 0.7311、検証でも最良が 0.7227 で、どちらも学習前より悪い。チェックポイントの選択は、学習前のモデルより悪くなるのは防げない。温度 0.01 (0.7729) は、温度 0.05・バッチ 64 の seed=0 (0.7624) を 0.01 上回ったが、種による幅 (0.008) と同じくらいで、種 1 つだけなので優劣は言えない。バッチ 256 は 0.7602 だった。バッチ 256 は全部で 50 ステップしかなく、12 ステップ目のチェックポイントが選ばれたので、ほとんど学習していない。論文の付録は、学習率を合わせればバッチサイズにはあまり敏感でない、としている。ここではテストセットで比べているが、実務でハイパーパラメータを選ぶときは検証セットの値で選ぶ。
- 教師あり SimCSE（JSNLI の含意を正例、矛盾をハードネガティブ）は 0.8261 で、学習前との差は +0.0721（95% 信頼区間 +0.0544〜+0.0906。評価指標と統計的検定の章の「ブートストラップ法で評価値の信頼区間を求める」を参照）だった。選ばれたのは最初の 50 ステップ目のチェックポイントで、少ない組でも効く。JSNLI は SNLI を機械翻訳して自動で絞り込んだものである。また SNLI の元は Flickr30k の画像の説明文、JSTS は YJ Captions（MS COCO の画像の日本語説明文）で、分野が近いことも効いている可能性がある。JSNLI の仮説には短い同じ文がよく出るので、`BatchSamplers.NO_DUPLICATES` で同じ文が同じバッチに入らないようにしている（Sentence Transformers の説明が `MultipleNegativesRankingLoss` に勧めている設定）。
- 検索の章の「検索用の埋め込みモデルを自分のデータで微調整する」も同じ `MultipleNegativesRankingLoss` を使っている。Sentence Transformers の説明でも、既定の設定（コサイン類似度、scale=20 = 温度 0.05）は InfoNCE や SimCSE の損失とも呼ばれるとしている。違うのは正例の組の作り方だけで、検索の章は（クエリ, 文書）、教師なし SimCSE は（文, 同じ文）、教師ありは（前提, 含意）に矛盾を足した三つ組である。
- データとモデルの出典とライセンス: 日本語 Wikipedia（wikimedia/wikipedia、CC BY-SA 3.0 / GFDL）、JGLUE の JSTS（ヤフー、CC BY-SA 4.0）、JSNLI（京都大学、CC BY-SA 4.0）、tohoku-nlp/bert-base-japanese-v3 (Apache 2.0)。Sentence Transformers は Apache 2.0。SimCSE の損失と学習ループを自分で書く版は、付録「仕組みを理解するための実装」の「SimCSE の損失と学習ループを自分で書いて、素の BERT から文埋め込みモデルを作る」にある。

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
- 「先頭の次元だけ使う」方法は、マトリョーシカ表現学習 (MRL) で訓練されたモデルなら精度がほとんど落ちない（このモデルは MRL ではない）。
