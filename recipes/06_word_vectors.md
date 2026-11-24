# 単語ベクトル

単語を数百次元のベクトルで表し、類似度やアナロジーを計算する。2 言語の単語ベクトルを線形写像で重ね合わせて、対訳語を引くレシピもある。文脈によって変わる単語ベクトル (BERT) は「Transformers の基本操作」の章を、共起行列から単語ベクトルを自分で作るレシピ（PPMI と SVD）は付録の「仕組みを理解するための実装」を参照。

## word2vec を自分のコーパスで学習する {word2vec_train}

手元のコーパスから単語ベクトルを学習して、似た単語を調べたい。

```python
!pip install -q gensim fugashi unidic-lite
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
base = "https://www.aozora.gr.jp/cards/000148/files/"
text = "\n".join(aozora(base + f) for f in ["789_ruby_5639.zip", "752_ruby_2438.zip", "773_ruby_5968.zip"])  # 猫・坊っちゃん・こころ
sentences = [[w.surface for w in tagger(s)] for s in re.split(r"[。\n]", text) if s.strip()]
print(len(sentences), "文", sum(map(len, sentences)), "語")
```

```python
from gensim.models import Word2Vec

model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, sg=1, negative=5, epochs=20, workers=2, seed=0)
print("語彙:", len(model.wv))
for w in ["先生", "猫", "東京", "学校"]:
    print(w, [(x, round(s, 2)) for x, s in model.wv.most_similar(w, topn=6)])
```

```python
model.save("w2v.model")                       # 追加学習できる形式
model.wv.save_word2vec_format("w2v.txt")      # 他のツールでも読めるテキスト形式
print(Word2Vec.load("w2v.model").wv["猫"][:5])
```

- `sg=1` が skip-gram、`sg=0` が CBOW。小さいコーパスでは skip-gram + エポック多めがよい。
- 数十万語程度のコーパスでは品質は限定的。実用には数億語規模か、学習済みベクトルを使う。

## fastText で未知語にもベクトルを与える {fasttext_train}

word2vec は語彙にない単語を扱えない。文字 n-gram を使う fastText なら、未知語や表記ゆれにもベクトルを与えられる。

```python
!pip install -q gensim
```

```python
import nltk
nltk.download("brown", quiet=True)
from nltk.corpus import brown
from gensim.models import FastText, Word2Vec

sentences = [[w.lower() for w in s] for s in brown.sents()]
ft = FastText(sentences, vector_size=100, window=5, min_count=5, min_n=3, max_n=6, epochs=10, workers=2, seed=0)
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=5, epochs=10, workers=2, seed=0)
```

```python
for w in ["governmental", "governmentally", "goverment", "unhappily"]:
    in_vocab = w in ft.wv.key_to_index
    print(f"{w:16} 語彙内: {in_vocab!s:6} fastText: {[x for x, _ in ft.wv.most_similar(w, topn=4)]}")
    if w not in w2v.wv:
        print(f"{'':16} word2vec: KeyError (語彙にない)")
```

- 単語ベクトル = 単語自身 + 文字 n-gram（`min_n`〜`max_n` 文字）のベクトルの和。綴りが似ている語は近くなる。
- 形態が豊かな言語（ドイツ語、トルコ語、フィンランド語など）やタイポの多いテキストで特に有効。
- Meta 公開の学習済みモデル（157 言語、`cc.ja.300.bin` など）は `fasttext` パッケージで読める（各 4GB 以上）。

## 学習済みの英語単語ベクトルを読み込んで類似語を調べる {pretrained_vectors_en}

自分で学習せずに、公開されている学習済み単語ベクトル（GloVe など）をすぐ使いたい。

```python
!pip install -q gensim
```

```python
import gensim.downloader as api

print([name for name in api.info()["models"]][:13])
wv = api.load("glove-wiki-gigaword-100")  # 約 130MB。40 万語 × 100 次元
print(wv.vector_size, len(wv))
```

```python
print(wv.most_similar("tokyo", topn=5))
print("類似度:", wv.similarity("cat", "dog"), wv.similarity("cat", "car"))
print("仲間外れ:", wv.doesnt_match(["breakfast", "lunch", "dinner", "computer"]))
print("語彙にある？", "tokyo" in wv, "Tokyo" in wv)  # GloVe は小文字のみ
```

- `word2vec-google-news-300` (1.6GB) や `fasttext-wiki-news-subwords-300` (1GB) も同じ API で読める。
- 戻り値は `KeyedVectors`。`wv["word"]` で numpy 配列が得られる。

## 学習済みの日本語単語ベクトルを使う (chiVe) {pretrained_vectors_ja}

日本語の学習済み単語ベクトルで、類似語検索やアナロジーをしたい。

```python
!pip install -q gensim
!wget -q https://sudachi.s3-ap-northeast-1.amazonaws.com/chive/chive-1.3-mc90_gensim.tar.gz
!tar xzf chive-1.3-mc90_gensim.tar.gz
```

```python
from gensim.models import KeyedVectors

wv = KeyedVectors.load("chive-1.3-mc90_gensim/chive-1.3-mc90.kv")
print(wv.vector_size, "次元", len(wv), "語")
for w in ["自然言語処理", "寿司", "嬉しい"]:
    print(w, [x for x, _ in wv.most_similar(w, topn=7)])
```

```python
print(wv.most_similar(positive=["東京", "フランス"], negative=["日本"], topn=3))
print(wv.most_similar(positive=["兄", "女"], negative=["男"], topn=3))
print(wv.most_similar(positive=["大きい", "低い"], negative=["小さい"], topn=3))
```

- chiVe は国語研日本語ウェブコーパス（約 1 億ページ）で学習した 300 次元の word2vec。語は Sudachi の正規化形。
- `mc90` は頻度 90 回以上の約 41 万語 (0.5GB)。語彙を増やすなら `mc5`（約 320 万語、3GB 以上）。
- 入力を Sudachi で分割・正規化 (`normalized_form()`) してから引くと語彙に当たりやすい。

## 単語ベクトルでアナロジー（王 − 男 + 女 = 女王）を解く {analogy}

単語ベクトルの足し算・引き算で「A にとっての B は、C にとっての何か」を求めたい。

```python
!pip install -q gensim
```

```python
import gensim.downloader as api

wv = api.load("glove-wiki-gigaword-100")

def analogy(a, b, c, topn=3):
    """a : b = c : ?"""
    return [w for w, _ in wv.most_similar(positive=[b, c], negative=[a], topn=topn)]

print("man : king = woman :", analogy("man", "king", "woman"))
print("japan : tokyo = france :", analogy("japan", "tokyo", "france"))
print("good : better = bad :", analogy("good", "better", "bad"))
print("walk : walked = go :", analogy("walk", "walked", "go"))
print("japan : sushi = italy :", analogy("japan", "sushi", "italy"))
```

```python
# 標準的なアナロジー評価セット (Google analogy test set) で正解率を測る
from gensim.test.utils import datapath

score, sections = wv.evaluate_word_analogies(datapath("questions-words.txt"))
print(f"全体の正解率: {score:.3f}")
for sec in sections[:5]:
    n = len(sec["correct"]) + len(sec["incorrect"])
    print(f"  {sec['section']:28} {len(sec['correct']) / max(1, n):.3f} ({n} 問)")
```

- `most_similar` は入力に使った語を答えから除外する。除外しないと、多くの場合 b か c 自身が最近傍になる。

## 単語ベクトルを 2 次元に落として可視化する {word_vectors_viz}

単語ベクトルの空間で、似た単語が近くに集まっている様子を散布図で見たい。

```python
!pip install -q gensim
```

```python
import gensim.downloader as api
import numpy as np

wv = api.load("glove-wiki-gigaword-100")
groups = {
    "country": ["japan", "china", "france", "germany", "italy", "brazil", "india", "canada"],
    "city": ["tokyo", "beijing", "paris", "berlin", "rome", "london", "delhi", "ottawa"],
    "animal": ["cat", "dog", "horse", "lion", "tiger", "rabbit", "monkey", "elephant"],
    "food": ["sushi", "pizza", "bread", "rice", "noodles", "cheese", "apple", "banana"],
    "verb": ["run", "walk", "eat", "drink", "sleep", "write", "read", "speak"],
}
words = [w for ws in groups.values() for w in ws]
X = np.array([wv[w] for w in words])
```

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, (name, Z) in zip(axes, [("PCA", PCA(n_components=2).fit_transform(X)),
                                ("t-SNE", TSNE(n_components=2, perplexity=8, random_state=0).fit_transform(X))]):
    i = 0
    for label, ws in groups.items():
        ax.scatter(Z[i:i + len(ws), 0], Z[i:i + len(ws), 1], label=label)
        for j, w in enumerate(ws):
            ax.annotate(w, Z[i + j], fontsize=9, xytext=(3, 3), textcoords="offset points")
        i += len(ws)
    ax.set_title(name); ax.legend()
plt.show()
```

- PCA は大域的な構造（動物は上、食べ物は右下、国と都市は左に集まるなど）を保ち、t-SNE は近傍のまとまりを強調する。t-SNE の軸や距離に意味はない。どちらの図でも、国と都市は分かれずに混ざる。
- 点が近い単語はラベルが重なって読みにくい（PCA の lion と tiger、t-SNE の drink と eat など）。
- 語数が多いときは t-SNE の `perplexity` を 30 前後に。数万語なら UMAP が速い。

## 単語類似度データセットで単語ベクトルを評価する {wordsim_eval}

単語ベクトルの良し悪しを、人間が付けた単語類似度スコアとの相関で定量的に評価したい。

```python
!pip install -q gensim
```

```python
import gensim.downloader as api
from gensim.test.utils import datapath

for name in ["glove-wiki-gigaword-50", "glove-wiki-gigaword-100"]:
    wv = api.load(name)
    pearson, spearman, oov = wv.evaluate_word_pairs(datapath("wordsim353.tsv"))
    print(f"{name:26} Spearman = {spearman.statistic:.3f}  Pearson = {pearson.statistic:.3f}  未知語率 = {oov:.1f}%")
```

```python
# 中身を見る: 人間の評価 (0〜10) とモデルのコサイン類似度
import pandas as pd

df = pd.read_csv(datapath("wordsim353.tsv"), sep="\t", comment="#", names=["w1", "w2", "human"])
df["model"] = [wv.similarity(a.lower(), b.lower()) if a.lower() in wv and b.lower() in wv else None for a, b in zip(df.w1, df.w2)]
df.sample(8, random_state=0)
```

- 順位相関 (Spearman) を使うのが慣例。値のスケールが違っても、順序が合っていればよいため。
- WordSim-353 は「類似」と「関連」(coffee-cup) が混ざっている。純粋な類似度を測るなら SimLex-999。日本語には JWSAN や日本語単語類似度データセット (JWSD) がある。

## 単語ベクトルに含まれるジェンダーバイアスを測る {embedding_bias}

単語ベクトルが、職業名などに社会的な偏り（ステレオタイプ）を取り込んでいるかを調べたい。

```python
!pip install -q gensim
```

```python
import gensim.downloader as api
import numpy as np

wv = api.load("glove-wiki-gigaword-100")

# 性別の方向 = (he - she), (man - woman) などの差の平均
pairs = [("he", "she"), ("man", "woman"), ("his", "her"), ("father", "mother"), ("boy", "girl"), ("male", "female")]
g = np.mean([wv[a] - wv[b] for a, b in pairs], axis=0)
g /= np.linalg.norm(g)

jobs = ["nurse", "engineer", "teacher", "programmer", "receptionist", "doctor", "homemaker", "scientist",
        "librarian", "mechanic", "secretary", "professor", "dancer", "soldier", "carpenter", "hairdresser"]
scores = sorted(((float(wv[w] @ g / np.linalg.norm(wv[w])), w) for w in jobs))
for s, w in scores:
    print(f"{w:14}{s:+.3f}  {'#' * int(abs(s) * 60)}{'  ← she' if s < 0 else '  → he'}")
```

- 値が負なら she 側、正なら he 側に寄っている。コーパス中の使われ方の偏りが、そのままベクトルに現れる。
- 体系的に測るなら WEAT (Word Embedding Association Test)。バイアス除去 (debiasing) の研究も多いが、完全には除けないことが知られている。

## 単語ベクトルの平均で文ベクトルを作る {avg_word_vectors}

文や文書をベクトルにする最も手軽な方法として、含まれる単語のベクトルを平均したい。

```python
!pip install -q gensim
```

```python
import re
import gensim.downloader as api
import numpy as np

wv = api.load("glove-wiki-gigaword-100")

def sent_vec(text):
    words = [w for w in re.findall(r"[a-z]+", text.lower()) if w in wv]
    return np.mean([wv[w] for w in words], axis=0) if words else np.zeros(wv.vector_size)

def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

query = "How do I cook pasta?"
docs = ["Boil the spaghetti in salted water for ten minutes.", "The stock market fell sharply today.",
        "A recipe for making noodles at home.", "My cat sleeps all day on the sofa."]
for d in sorted(docs, key=lambda d: -cos(sent_vec(query), sent_vec(d))):
    print(f"{cos(sent_vec(query), sent_vec(d)):.3f}  {d}")
```

```python
# SIF 風の重み付け: 高頻度語の重みを下げる (順位を頻度の代わりに使った簡易版)
def weighted_vec(text, a=1e-3):
    words = [w for w in re.findall(r"[a-z]+", text.lower()) if w in wv]
    weights = [a / (a + 1 / (wv.key_to_index[w] + 2) / 10) for w in words]  # Zipf 則で順位から頻度を近似
    return np.average([wv[w] for w in words], axis=0, weights=weights)

for d in sorted(docs, key=lambda d: -cos(weighted_vec(query), weighted_vec(d))):
    print(f"{cos(weighted_vec(query), weighted_vec(d)):.3f}  {d}")
```

- 語順は無視されるが、驚くほど強いベースライン。まずこれを試し、足りなければ文埋め込みモデル（文埋め込みとクラスタリングの章）に進む。
- 単純平均は `the` `of` などの機能語に引っ張られて、どの文も似てしまう。ストップワード除去か頻度による重み付けが効く。

## Doc2Vec で文書ベクトルを学習する {doc2vec}

文書そのものにベクトルを割り当てて、似た文書を探したい（word2vec の文書版）。

```python
!pip install -q gensim
```

```python
import nltk
nltk.download("reuters", quiet=True)
from nltk.corpus import reuters
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

ids = reuters.fileids()[:3000]
docs = [TaggedDocument([w.lower() for w in reuters.words(i) if w.isalpha()], [n]) for n, i in enumerate(ids)]
# workers=1 にすると、seed を固定したときに毎回同じ結果になる（スレッドが 2 つ以上だと処理の順番で結果が変わる）
model = Doc2Vec(docs, vector_size=100, window=5, min_count=3, epochs=20, dm=0, workers=1, seed=0)
```

```python
q = 0
print("クエリ:", reuters.categories(ids[q]), " ".join(docs[q].words[:25]))
for n, sim in model.dv.most_similar(q, topn=3):
    print(f"{sim:.2f}", reuters.categories(ids[n]), " ".join(docs[n].words[:25]))
```

```python
# 学習に使っていない新しい文書のベクトルを推論する
vec = model.infer_vector("oil prices rose after opec agreed to cut production".split(), epochs=50)
for n, sim in model.dv.most_similar([vec], topn=3):
    print(f"{sim:.2f}", reuters.categories(ids[n]), " ".join(docs[n].words[:15]))
```

- `dm=1` が PV-DM（文脈語 + 文書ベクトルで予測）、`dm=0` が PV-DBOW（文書ベクトルだけで単語を予測）。ここでは `dm=0` を使った。このデータでは `dm=1` にすると、上の推論の例で石油 (crude) の記事が上位に来なかった。
- 現在は事前学習済みの文埋め込みモデルのほうが手軽で高精度なことが多い。ドメイン特化の大量文書がある場合の選択肢。

## 2 言語の単語ベクトルを線形写像で対応させて対訳語を引く {cross_lingual_word_vectors}

日本語と英語で別々に学習した単語ベクトルを、少数の対訳ペアで学習した線形写像で重ね合わせ、辞書にない単語の訳語を引きたい。

```python
# MUSE の日英対訳辞書: 学習用 (よく使う語 0〜5000 番目) とテスト用 (5000〜6500 番目)
!wget -q https://dl.fbaipublicfiles.com/arrival/dictionaries/ja-en.0-5000.txt
!wget -q https://dl.fbaipublicfiles.com/arrival/dictionaries/ja-en.5000-6500.txt
```

```python
import gzip, urllib.request
import numpy as np

def load_vec(lang, n=50000):
    """fastText の Common Crawl ベクトル (.vec.gz, 約 1.3GB) を、ストリーミングで頻度順の先頭 n 語だけ読む"""
    url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{lang}.300.vec.gz"
    words, vecs = [], np.zeros((n, 300), dtype=np.float32)
    with urllib.request.urlopen(url) as r, gzip.open(r, "rt", encoding="utf-8") as f:
        f.readline()  # 1 行目は「語数 次元数」
        for i, line in enumerate(f):
            if i == n:
                break
            w, *v = line.rstrip().split(" ")
            words.append(w)
            vecs[i] = np.array(v, dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)  # 長さ 1 にそろえる (内積 = コサイン類似度)
    return words, {w: i for i, w in enumerate(words)}, vecs

ja_words, ja_id, JA = load_vec("ja")  # 1 言語あたり約 35MB のダウンロード
en_words, en_id, EN = load_vec("en")

def load_dict(path):
    d = {}  # 日本語 → 英訳の集合 (訳語が複数ある語も多い)
    for line in open(path, encoding="utf-8"):
        s, t = line.split()
        if s in ja_id and t in en_id:
            d.setdefault(s, set()).add(t)
    return d

train, test = load_dict("ja-en.0-5000.txt"), load_dict("ja-en.5000-6500.txt")
pairs = [(s, t) for s, ts in train.items() for t in sorted(ts)]
print(f"学習: {len(train)} 語 ({len(pairs)} ペア)  テスト: {len(test)} 語")
print("テストの例:", [(s, sorted(ts)) for s, ts in list(test.items())[:5]])
```

```python
X = JA[[ja_id[s] for s, _ in pairs]]  # 日本語側のベクトル
Y = EN[[en_id[t] for _, t in pairs]]  # 対応する英語側のベクトル

# 1. Mikolov らの翻訳行列: ‖XW − Y‖² を最小にする W (最小二乗法)
W_ls = np.linalg.lstsq(X, Y, rcond=None)[0]
# 2. 直交プロクラステス: W を回転 (直交行列) に限る。XᵀY の特異値分解 UΣVᵀ から W = UVᵀ
U, _, Vt = np.linalg.svd(X.T @ Y)
W_pro = U @ Vt
print("WᵀW と単位行列の差:", round(float(np.abs(W_ls.T @ W_ls - np.eye(300)).max()), 3), "(最小二乗)",
      round(float(np.abs(W_pro.T @ W_pro - np.eye(300)).max()), 6), "(プロクラステス)")

def nn(Q, k=5):
    """写像したベクトルに、コサイン類似度で最も近い英単語を k 個返す"""
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    return np.argsort(-(Q @ EN.T), axis=1)[:, :k]

def precision(top, queries, k):
    return np.mean([any(en_words[j] in test[s] for j in row[:k]) for row, s in zip(top, queries)])

queries = list(test)
Q = JA[[ja_id[s] for s in queries]]
for name, W in [("最小二乗", W_ls), ("プロクラステス", W_pro)]:
    top = nn(Q @ W)
    print(f"{name:8}  P@1 = {precision(top, queries, 1):.3f}  P@5 = {precision(top, queries, 5):.3f}")
```

```python
# CSLS: どの語の近傍にも現れる「ハブ」の英単語を割り引いて、最近傍探索の偏りを抑える
#   CSLS(x, y) = 2 cos(Wx, y) − r_en(Wx) − r_ja(y)   (r は相手側の言語の近傍 10 語との平均類似度)
def csls(Q, W, k=5, knn=10):
    S = JA @ W
    S /= np.linalg.norm(S, axis=1, keepdims=True)  # 日本語の全語彙 (5 万語) を写像したもの
    r_ja = np.zeros(len(EN), dtype=np.float32)     # 英単語ごとに、写像した日本語の近傍との平均類似度
    for i in range(0, len(EN), 2000):
        sims = EN[i:i + 2000] @ S.T
        r_ja[i:i + 2000] = np.partition(sims, -knn, axis=1)[:, -knn:].mean(1)
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    sims = Q @ EN.T
    r_en = np.partition(sims, -knn, axis=1)[:, -knn:].mean(1)
    return np.argsort(-(2 * sims - r_en[:, None] - r_ja[None]), axis=1)[:, :k]

from collections import Counter
for name, W in [("最小二乗", W_ls), ("プロクラステス", W_pro)]:
    for method, top in [("最近傍", nn(Q @ W)), ("CSLS", csls(Q @ W, W))]:
        hubs = Counter(en_words[row[0]] for row in top).most_common(3)
        print(f"{name:8} {method:4}  P@1 = {precision(top, queries, 1):.3f}  P@5 = {precision(top, queries, 5):.3f}"
              f"  1 位に多く出た語: {hubs}")
```

```python
# うまくいく例と失敗する例 (どれも学習用の辞書にない語。猫と犬はテスト用の辞書にもない)
examples = ["猫", "犬", "寿司", "ひまわり", "休暇", "レバー", "押す", "微分", "等しい", "ジェームス", "ディック", "土井", "三次"]
top = csls(JA[[ja_id[w] for w in examples]] @ W_pro, W_pro)
for w, row in zip(examples, top):
    gold = "/".join(sorted(test.get(w, []))) or "-"
    print(f"{w:6} 正解: {gold:20} 予測: {[en_words[j] for j in row]}")
```

```python
# 対訳ペアの数を減らすと: 300 ペアは次元数 (300) より少なく、最小二乗では W が決まらない
rng = np.random.default_rng(0)
perm = rng.permutation(len(pairs))
for n in [300, 1000, 3000, len(pairs)]:
    Xn, Yn = X[perm[:n]], Y[perm[:n]]
    U, _, Vt = np.linalg.svd(Xn.T @ Yn)
    p_ls = precision(nn(Q @ np.linalg.lstsq(Xn, Yn, rcond=None)[0], 1), queries, 1)
    p_pro = precision(nn(Q @ (U @ Vt), 1), queries, 1)
    print(f"{n:5} ペア  P@1  最小二乗 {p_ls:.3f}  プロクラステス {p_pro:.3f}")
```

- 単語ベクトル空間の形は言語が違っても似ているので、線形写像（ほぼ回転）で重ねられる（Mikolov ら 2013）。写像を直交行列に限ると、日本語の中での語どうしの距離が変わらず、少ないペアでも過学習しにくい（Xing ら 2015）。CSLS は MUSE（Conneau ら 2018）で提案された。
- 評価は、テスト用の日本語 1 語に対して英語 5 万語から探し、上位に辞書の訳語のどれかが入れば正解とする。全ペア（5300 ペア）で学習すると、最近傍の P@1 は最小二乗 0.532、プロクラステス 0.547 で差は小さい。CSLS にすると 0.589 と 0.570 に上がり、このデータでは最小二乗のほうが伸びた。ペアが少ないとプロクラステスのほうがはるかに頑健で、300 ペアでは最小二乗の P@1 が 0.001 とほぼ 0 で、プロクラステスは 0.238 だった。
- 失敗の典型は、固有名詞（ジェームス → george、ディック → mr、名字の土井は意味のない語）、意味は近いが訳ではない語（微分 → algebra などの数学の語、等しい → negligible、tantamount）、トークン化や用法のずれ（三次 → cubic が引けない）。多義語は、片方の語義（レバー → lever、liver は出ない）だけが出やすい。
- 英語側は大文字・小文字を区別した語彙なので、`Sunflower` のような大文字の変種が候補の枠を使う。MUSE の辞書は小文字だけ。
- fastText の Common Crawl ベクトルは CC BY-SA 3.0。MUSE の対訳辞書は CC BY-NC 4.0（非商用に限る）。MUSE の日本語の辞書には、カタカナの人名や自動抽出による誤りが多く含まれる。
- 対訳ペアなしで写像を学習する方法（敵対的学習、VecMap の自己学習）もある。同じ回転による位置合わせを、時代の違うコーパスどうしに使うのが語の意味と意味の変化の章の「時代別の単語ベクトルを位置合わせして意味が変わった単語を探す」。単語ではなく文を言語をまたいで探すなら、検索の章の「言語をまたいで検索する（クロスリンガル検索）」。
