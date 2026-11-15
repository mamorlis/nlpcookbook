# 語彙知識の獲得

コーパスから、上位語・下位語、同じカテゴリの語、言い換えなどの語彙知識を自動的に集める。パターンによる抽出、ブートストラッピングと意味ドリフト、グラフに基づく半教師あり学習まで。

## 「A などの B」のパターンで上位語と下位語の組を集める {hypernym_patterns}

「犬は動物の一種」のような上位・下位の関係 (is-a 関係) を、テキストから自動的に集めたい。

```python
!pip install -q datasets fugashi unidic-lite
```

```python
import itertools, re
from collections import Counter, defaultdict
from datasets import load_dataset
from fugashi import Tagger

# 日本語 Wikipedia の先頭から 1.2 万記事の冒頭部分を使う (ストリーミングで必要な分だけ読む)
ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
articles = [ex["text"][:1500] for ex in itertools.islice(ds, 12000)]
sentences = [s for a in articles for s in re.split(r"(?<=。)|\n", a) if 10 <= len(s) <= 150]
print(len(articles), "記事", len(sentences), "文")
```

```python
tagger = Tagger()

def chunks(sentence):
    """連続する名詞を 1 つの名詞句にまとめ、(文字列, 名詞句かどうか) の列にする"""
    out = []
    for w in tagger(sentence):
        is_noun = w.feature.pos1 in ("名詞", "接尾辞") and w.feature.pos2 != "数詞" or w.feature.pos1 == "接頭辞"
        if is_noun and out and out[-1][1]:
            out[-1] = (out[-1][0] + w.surface, True)
        else:
            out.append((w.surface, is_noun))
    return out

# パターン: 「A などの B」「A や C などの B」「A といった B」「A をはじめとする B」→ A (と C) は B の下位語
MARKERS = [("など", "の"), ("と", "いっ", "た"), ("を", "はじめ", "と", "する"), ("と", "いう")]
pairs = Counter()
for s in sentences:
    c = chunks(s)
    for i, (text, is_noun) in enumerate(c):
        if not is_noun:
            continue
        for marker in MARKERS:
            j = i + 1 + len(marker)
            if tuple(x for x, _ in c[i + 1:j]) == marker and j < len(c) and c[j][1]:
                hyponyms = [text]
                k = i
                while k >= 2 and c[k - 1][0] in ("や", "、", "・") and c[k - 2][1]:       # 「A や C などの B」の並列をさかのぼる
                    hyponyms.append(c[k - 2][0]); k -= 2
                for h in hyponyms:
                    pairs[h, c[j][0], "".join(marker)] += 1
print(len(pairs), "組")
print(Counter(m for _, _, m in pairs).most_common())
```

```python
# 上位語ごとに下位語をまとめる
by_hypernym = defaultdict(Counter)
for (hypo, hyper, marker), n in pairs.items():
    if marker != "という" and len(hypo) >= 2 and len(hyper) >= 2:
        by_hypernym[hyper][hypo] += n
for hyper in ["言語", "スポーツ", "動物", "施設", "楽器", "金属", "作品", "問題"]:
    print(hyper, "←", " ".join(w for w, _ in by_hypernym[hyper].most_common(15)))

print("\n下位語の種類が多い上位語:", [(h, len(c)) for h, c in sorted(by_hypernym.items(), key=lambda x: -len(x[1]))[:15]])
print("「という」の例:", [(a, b) for (a, b, m), n in pairs.most_common(3000) if m == "という"][:12])
```

- 「A などの B」のような言い回しは、A が B の一種であることを高い確率で示す。英語の "B such as A" にあたり、Hearst パターンと呼ばれる。
- パターンごとに精度が違う。「などの」「といった」は信頼でき、「という」は「〜という意味」「〜という名前」のような上位語でない組を多く拾う。パターン別に抜き取って精度を確かめる。
- 「問題」「もの」「作品」のような抽象的な語は、何の上位語にもなれてしまう。下位語の種類が極端に多い上位語は、辞書として使う前に除くか、別扱いにする。
- 並列 (「A、B などの C」) をさかのぼるとき、文の区切りを越えて無関係な語 (「一方」「傍ら」) を拾うことがある。係り受け解析を使えば減らせる。
- 名詞句の範囲の取り方 (どこまでを 1 語とするか) で結果が大きく変わる。ここでは連続する名詞をまとめただけなので、「の」でつながる長い名詞句は切れている。
- 再現率は低い (パターンに当てはまる文にしか効かない)。大量のテキストに適用して数で補うか、次項のブートストラッピングでパターンを増やす。

## 少数の例から同じカテゴリの語を増やす (ブートストラッピング) {bootstrapping}

「日本、フランス、中国」のような数個の例 (シード) を与えるだけで、同じカテゴリの語をコーパスから集めたい。

```python
!pip install -q datasets fugashi unidic-lite
```

```python
import itertools, math, re
from collections import Counter
import numpy as np
import scipy.sparse as sp
from datasets import load_dataset
from fugashi import Tagger

ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
articles = [ex["text"][:1500] for ex in itertools.islice(ds, 12000)]
tagger = Tagger()

# (名詞句, その前後の文脈パターン) の共起を数える。パターンは「左の語 X 右の語」と「X 右の語 その次の語」の 2 種類
cooc = Counter()
for a in articles:
    for s in re.split(r"(?<=。)|\n", a):
        if not 10 <= len(s) <= 150:
            continue
        c = []
        for w in tagger(s):
            if w.feature.pos1 == "名詞" and c and c[-1][1]:
                c[-1] = (c[-1][0] + w.surface, True)
            else:
                c.append((w.surface, w.feature.pos1 == "名詞"))
        c = [("<s>", False)] + c + [("</s>", False), ("</s>", False)]
        for i in range(1, len(c) - 2):
            if c[i][1] and len(c[i][0]) >= 2 and not c[i][0].isdigit():
                cooc[c[i][0], f"{c[i - 1][0]} X {c[i + 1][0]}"] += 1
                cooc[c[i][0], f"X {c[i + 1][0]} {c[i + 2][0]}"] += 1
print(len(cooc), "種類の共起")
```

```python
# 行がパターン、列が語 (インスタンス) の行列を作る。要素は PMI (最大値で割って 0〜1 にする)
n_inst, n_pat = Counter(), Counter()
for (i, p), n in cooc.items():
    n_inst[i] += n; n_pat[p] += n
instances = [i for i, n in n_inst.items() if n >= 5]
patterns = [p for p, n in n_pat.items() if n >= 5]
inst_id, pat_id = {w: k for k, w in enumerate(instances)}, {w: k for k, w in enumerate(patterns)}
total, rows, cols, vals = sum(cooc.values()), [], [], []
for (i, p), n in cooc.items():
    if i in inst_id and p in pat_id:
        pmi = math.log2(n * total / (n_inst[i] * n_pat[p]))
        if pmi > 0:
            rows.append(pat_id[p]); cols.append(inst_id[i]); vals.append(pmi)
M = sp.csr_matrix((vals, (rows, cols)), shape=(len(patterns), len(instances)))
M = M / M.max()
print(M.shape, "非ゼロ要素", M.nnz)
```

```python
def bootstrap(seeds, iterations=1):
    """シード → シードとよく共起するパターン → そのパターンとよく共起する語、を繰り返す"""
    inst_score = np.zeros(len(instances))
    inst_score[[inst_id[s] for s in seeds]] = 1.0
    for _ in range(iterations):
        pat_score = M @ inst_score; pat_score /= pat_score.max()        # パターンの信頼度: 信頼できる語と強く共起するほど高い
        inst_score = M.T @ pat_score; inst_score /= inst_score.max()    # 語の信頼度: 信頼できるパターンと強く共起するほど高い
    return inst_score, pat_score

for seeds in [["日本", "フランス", "中国"], ["ピアノ", "ギター", "ヴァイオリン"], ["夏目漱石", "森鴎外", "芥川龍之介"], ["サッカー", "野球", "テニス"]]:
    inst_score, pat_score = bootstrap(seeds)
    print("シード:", seeds)
    print("   パターン:", [patterns[k] for k in np.argsort(-pat_score)[:8]])
    print("   獲得した語:", [instances[k] for k in np.argsort(-inst_score) if instances[k] not in seeds][:18])
```

- ブートストラッピングは「シードとよく一緒に出る文脈 (パターン) を見つけ、そのパターンに当てはまる別の語を集める」ことを繰り返す。正解のラベルを数個しか必要としない。
- ここでの計算は Espresso というアルゴリズムを簡略化したもの。パターンの信頼度と語の信頼度を、PMI を重みにして交互に計算する。
- 「X を弾く」(楽器)、「X の小説」(作家) のように、カテゴリに特有のパターンが自動的に選ばれる。パターンを見ると、何を手がかりにしたかが分かる。
- 1 回の反復でも、かなり良い候補が得られる。ただし「音楽」「多く」のような、カテゴリの外の語も混ざる。
- 反復を増やせば候補は増えるが、次項のように別の問題 (意味ドリフト) が起きる。

## ブートストラッピングの意味ドリフトを観察して抑える {semantic_drift}

ブートストラッピングを繰り返すと、シードと関係のない語ばかりが集まるようになってしまう (意味ドリフト)。なぜ起きるかを確かめ、抑える方法を試したい。

```python
!pip install -q datasets fugashi unidic-lite
```

```python
import itertools, math, re
from collections import Counter
import numpy as np
import scipy.sparse as sp
from datasets import load_dataset
from fugashi import Tagger

ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
articles = [ex["text"][:1500] for ex in itertools.islice(ds, 12000)]
tagger = Tagger()

cooc = Counter()
for a in articles:
    for s in re.split(r"(?<=。)|\n", a):
        if not 10 <= len(s) <= 150:
            continue
        c = []
        for w in tagger(s):
            if w.feature.pos1 == "名詞" and c and c[-1][1]:
                c[-1] = (c[-1][0] + w.surface, True)
            else:
                c.append((w.surface, w.feature.pos1 == "名詞"))
        c = [("<s>", False)] + c + [("</s>", False), ("</s>", False)]
        for i in range(1, len(c) - 2):
            if c[i][1] and len(c[i][0]) >= 2 and not c[i][0].isdigit():
                cooc[c[i][0], f"{c[i - 1][0]} X {c[i + 1][0]}"] += 1
                cooc[c[i][0], f"X {c[i + 1][0]} {c[i + 2][0]}"] += 1

n_inst, n_pat = Counter(), Counter()
for (i, p), n in cooc.items():
    n_inst[i] += n; n_pat[p] += n
instances = [i for i, n in n_inst.items() if n >= 5]
patterns = [p for p, n in n_pat.items() if n >= 5]
inst_id, pat_id = {w: k for k, w in enumerate(instances)}, {w: k for k, w in enumerate(patterns)}
total, rows, cols, vals = sum(cooc.values()), [], [], []
for (i, p), n in cooc.items():
    if i in inst_id and p in pat_id and (pmi := math.log2(n * total / (n_inst[i] * n_pat[p]))) > 0:
        rows.append(pat_id[p]); cols.append(inst_id[i]); vals.append(pmi)
M = sp.csr_matrix((vals, (rows, cols)), shape=(len(patterns), len(instances)))
M = sp.csr_matrix(M / M.max())

def seed_vector(seeds):
    v = np.zeros(len(instances)); v[[inst_id[s] for s in seeds]] = 1.0
    return v

def top(scores, k=12):
    return " ".join(instances[i] for i in np.argsort(-scores)[:k])
```

```python
# 反復を増やすと、どのシードから始めても同じ語に収束する
SEEDS = {"国": ["日本", "フランス", "中国"], "楽器": ["ピアノ", "ギター", "ヴァイオリン"], "作家": ["夏目漱石", "森鴎外", "芥川龍之介"]}
final = {}
for name, seeds in SEEDS.items():
    v = seed_vector(seeds)
    for it in range(1, 31):
        p = M @ v; p /= p.max(); v = M.T @ p; v /= v.max()
        if it in (1, 2, 5, 30):
            print(f"{name} 反復 {it:2}: {top(v)}")
    final[name] = v
print("30 回後の順位の一致 (上位 100 語の重なり):", len(set(np.argsort(-final["国"])[:100]) & set(np.argsort(-final["楽器"])[:100])))
```

```python
# 理由: この反復は行列 A = MᵀM を繰り返し掛けること (べき乗法) と同じで、シードに関係なく A の第 1 固有ベクトルに収束する
from scipy.sparse.linalg import svds

_, s, vt = svds(M, k=1)
principal = np.abs(vt[0])
print("第 1 固有ベクトルの上位:", top(principal))
print("30 回後のスコアとの相関:", round(float(np.corrcoef(principal, final["国"])[0, 1]), 4))
print("収束先に多い語は、多くのパターンと共起する語:", [(instances[i], int((M[:, i] > 0).sum())) for i in np.argsort(-principal)[:6]])
```

```python
# 対策 1: フォン・ノイマン カーネル。反復 1 回目、2 回目、... の結果を、回数が増えるほど小さい重み (α の n 乗) で足し合わせる
lam = float(s[0]) ** 2                              # A の最大固有値。α / lam を掛けることで和が収束する

def von_neumann(seeds, alpha, terms=50):
    term, score = seed_vector(seeds), np.zeros(len(instances))
    for _ in range(terms):
        term = (alpha / lam) * (M.T @ (M @ term)); score += term
    return score

# 対策 2: 正則化ラプラシアン カーネル。多くの語とつながっている語 (ハブ) のスコアを、次数で割り引く
from scipy.sparse.linalg import LinearOperator, cg

degree = np.asarray(M.T @ (M @ np.ones(len(instances)))).ravel()
d = 1 / np.sqrt(degree + 1e-12)

def regularized_laplacian(seeds, beta):
    op = LinearOperator((len(instances),) * 2, matvec=lambda x: x + beta * (x - d * (M.T @ (M @ (d * x)))))    # (I + βL) x = シード を解く
    return cg(op, seed_vector(seeds), maxiter=200)[0]

for name, seeds in SEEDS.items():
    print(name)
    for alpha in [0.01, 0.9, 0.99]:
        print(f"   フォン・ノイマン α={alpha}: {top(von_neumann(seeds, alpha))}")
    for beta in [1, 100]:
        print(f"   正則化ラプラシアン β={beta}: {top(regularized_laplacian(seeds, beta))}")
```

- 意味ドリフトは、多くのパターンと共起する一般的な語 (「もの」「作品」「こと」) が一度入ると、そこから無関係なパターンと語が芋づる式に増えていく現象。
- 簡略化した Espresso の反復は、行列のべき乗法と同じ計算になっている。繰り返すほどシードの情報が消え、どのシードでも同じ順位 (グラフの中心性) に収束する。これは HITS アルゴリズムが権威ノードを求める計算と同じ。
- フォン・ノイマン カーネルは、α が小さいとシードの近くを重視し、大きいとグラフ全体の中心性に近づく。正則化ラプラシアンは、ハブを通る経路を割り引くので、一般的な語に流れにくい (その代わり、頻度の低い語が上位に来やすい)。
- 実際の Espresso は、各反復で信頼度の高いパターンと語だけを少数ずつ残すことで、ドリフトを遅らせている。反復回数を増やしすぎない、一般的すぎるパターンをストップリストで除く、といった対策も使われる。
- どの方法でも、パラメータ (α、β、反復回数) で「シードへの近さ」と「広がり」の釣り合いを調整することになる。少数の正解データで確かめながら決める。

## 少数のラベルをグラフで広げて分類する (ラベル伝播と相互 k 近傍グラフ) {label_propagation_knn}

ラベル付きの文書がごく少ししかないとき、ラベルのない大量の文書も使って分類の精度を上げたい。

```python
!pip install -q scikit-learn fugashi unidic-lite
```

```python
import io, re, tarfile, urllib.request
import numpy as np
import pandas as pd
from fugashi import Tagger
from sklearn.feature_extraction.text import TfidfVectorizer

def load_livedoor():
    url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
    with urllib.request.urlopen(url) as r:
        tar = tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")
    rows = []
    for m in tar.getmembers():
        parts = m.name.split("/")
        if m.isfile() and len(parts) == 3 and parts[2] not in ("LICENSE.txt", "README.txt"):
            lines = tar.extractfile(m).read().decode("utf-8").split("\n")
            rows.append({"label": parts[1], "text": lines[2] + "\n" + "\n".join(lines[3:])[:600]})
    return pd.DataFrame(rows)

df = load_livedoor().sample(3000, random_state=0).reset_index(drop=True)
tagger = Tagger()
tokenize = lambda t: [w.surface for w in tagger(t) if w.feature.pos1 in ("名詞", "動詞", "形容詞")]
X = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, min_df=3, max_df=0.5, sublinear_tf=True).fit_transform(df["text"])
y = pd.factorize(df["label"])[0]
print(X.shape, df["label"].nunique(), "クラス")
```

```python
import scipy.sparse as sp

S = (X @ X.T).toarray()                    # TF-IDF は正規化済みなので、内積がコサイン類似度
np.fill_diagonal(S, 0)

def knn_graph(S, k, mutual):
    """各文書から類似度の高い k 個に辺を張る。mutual=True なら、互いに k 近傍に入っている組だけを残す"""
    n = len(S)
    idx = np.argsort(-S, axis=1)[:, :k]
    A = sp.csr_matrix((np.ones(n * k), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    A = A.multiply(A.T) if mutual else A.maximum(A.T)
    return sp.csr_matrix(A.multiply(S))    # 辺の重みは類似度

for k in [10, 30]:
    for mutual in [False, True]:
        A = knn_graph(S, k, mutual)
        degree = np.asarray((A > 0).sum(axis=1)).ravel()
        same = (y[A.nonzero()[0]] == y[A.nonzero()[1]]).mean()
        print(f"k={k} {'相互 k 近傍' if mutual else 'k 近傍'}: 辺 {A.nnz // 2} / 次数の最大 {degree.max()} / 孤立した文書 {(degree == 0).sum()} / 同じクラスどうしを結ぶ辺の割合 {same:.3f}")
```

```python
def label_propagation(A, y, labeled, alpha=0.9, iterations=50):
    """ラベル拡散法: 隣の文書のスコアを受け取ることを繰り返す (alpha は隣からもらう割合)"""
    d = np.asarray(A.sum(axis=1)).ravel()
    d_inv = np.where(d > 0, 1 / np.sqrt(np.maximum(d, 1e-12)), 0)
    W = sp.diags(d_inv) @ A @ sp.diags(d_inv)
    Y = np.zeros((len(y), y.max() + 1)); Y[labeled, y[labeled]] = 1
    F = Y.copy()
    for _ in range(iterations):
        F = alpha * (W @ F) + (1 - alpha) * Y
    return F.argmax(axis=1)

from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)
results = []
for per_class in [2, 5, 20]:
    for trial in range(5):
        labeled = np.concatenate([rng.choice(np.where(y == c)[0], per_class, replace=False) for c in range(y.max() + 1)])
        rest = np.setdiff1d(np.arange(len(y)), labeled)
        row = {"1 クラスあたりのラベル数": per_class,
               "ロジスティック回帰 (ラベル付きのみ)": (LogisticRegression(C=10, max_iter=1000).fit(X[labeled], y[labeled]).predict(X[rest]) == y[rest]).mean()}
        for name, mutual in [("ラベル伝播 (k 近傍)", False), ("ラベル伝播 (相互 k 近傍)", True)]:
            row[name] = (label_propagation(knn_graph(S, 30, mutual), y, labeled)[rest] == y[rest]).mean()
        results.append(row)
pd.DataFrame(results).groupby("1 クラスあたりのラベル数").mean().round(3)
```

```python
# k 近傍グラフでは、多くの文書の近傍に入る「ハブ」ができる。ハブは別のクラスの文書どうしをつないでしまう
A = knn_graph(S, 30, mutual=False)
degree = np.asarray((A > 0).sum(axis=1)).ravel()
hubs = np.argsort(-degree)[:5]
for h in hubs:
    neighbors = A[h].nonzero()[1]
    print(f"次数 {degree[h]} / クラス {df['label'][h]} / 近傍のうち同じクラス {np.mean(y[neighbors] == y[h]):.2f} | {df['text'][h].splitlines()[0][:40]}")
```

- グラフに基づく半教師あり学習は、「似ている文書は同じラベルを持つ」という仮定で、少数のラベルをグラフの辺に沿って広げる。ラベルが極端に少ないときに効果が大きい。
- 結果はグラフの作り方に強く左右される。普通の k 近傍グラフは、多くの点の近傍になる「ハブ」を生み、ハブを通って違うクラスにラベルが流れる。
- 相互 k 近傍グラフ (互いに近傍である組だけを結ぶ) はハブができず、同じクラスどうしを結ぶ辺の割合が高い。その代わり、どこにもつながらない孤立点ができる。この実験では、分類の正解率の差は小さく、ラベル数によって優劣が入れ替わった。
- ラベルが増えると、普通の教師あり学習との差は縮まり、1 クラス 20 件では逆転している。数十件以上のラベルがあるなら、まずは事前学習済みモデルのファインチューニングや埋め込み + 分類器を試す。
- 類似度に文埋め込みを使えば、同じ方法をそのまま適用できる。高次元の埋め込みでもハブは生じる (ハブネス問題)。

## 対訳コーパスを介して言い換えを集める (ピボット) {paraphrase_pivot}

「同じ外国語に訳される表現どうしは、言い換えである可能性が高い」という考え方で、対訳コーパスから言い換えの候補を集めたい。

```python
!pip install -q datasets fugashi unidic-lite
```

```python
import math, re
from collections import Counter, defaultdict
from datasets import load_dataset
from fugashi import Tagger

ds = load_dataset("Helsinki-NLP/opus-100", "en-ja", split="train")
tagger = Tagger()
pairs = []
for ex in ds.select(range(200000)):
    ja, en = ex["translation"]["ja"].strip(), ex["translation"]["en"].strip().lower()
    if 0 < len(ja) <= 40 and 0 < len(en.split()) <= 15 and re.search(r"[ぁ-んァ-ヶ一-龥]", ja):
        ja_words = [(w.feature.lemma or w.surface).split("-")[0] for w in tagger(ja) if w.feature.pos1 in ("名詞", "動詞", "形容詞", "形状詞", "副詞")]
        en_words = re.findall(r"[a-z']+", en)
        if ja_words and en_words:
            pairs.append((ja_words, en_words))
print(len(pairs), "文対", pairs[0])
```

```python
# IBM モデル 1 で単語の翻訳確率 t(英語 | 日本語) を推定する (EM アルゴリズム)
def ibm_model1(pairs, iterations=5):
    t = {}
    for it in range(iterations):
        default = 1.0 if it == 0 else 1e-6          # 最初はすべて同じ確率。2 回目以降、表から落とした組にはごく小さい確率を与える
        count, total = defaultdict(float), defaultdict(float)
        for src, tgt in pairs:
            for e in tgt:
                probs = [t.get((e, f), default) for f in src]
                z = sum(probs)                      # e がどの f から訳されたかの確率を、現在の t で按分する
                for f, p in zip(src, probs):
                    count[e, f] += p / z; total[f] += p / z
        t = {(e, f): c / total[f] for (e, f), c in count.items() if c / total[f] > 1e-4}      # 確率の小さい組は捨てて、表を小さく保つ
    return t

t_en_ja = ibm_model1(pairs)                                  # P(英語 | 日本語)
t_ja_en = ibm_model1([(en, ja) for ja, en in pairs])         # P(日本語 | 英語)

def translations(table, word, k=6):
    return sorted(((round(p, 2), a) for (a, b), p in table.items() if b == word), reverse=True)[:k]

for w in ["嬉しい", "素晴らしい", "始める", "病院", "多分"]:
    print(w, translations(t_en_ja, w))
```

```python
# ピボット: P(言い換え | 元の語) = Σ_英語 P(言い換え | 英語) × P(英語 | 元の語)
by_ja, by_en = defaultdict(dict), defaultdict(dict)
for (e, f), p in t_en_ja.items():
    by_ja[f][e] = p
for (f, e), p in t_ja_en.items():
    by_en[e][f] = p
freq = Counter(w for ja, _ in pairs for w in ja)
# 機能語を経由すると、どんな語とでも結び付いてしまう。英語の機能語と、日本語の軽い動詞・形式名詞は使わない
EN_STOP = set("a an the to of in on at for with by from and or but is am are was were be been being i you he she it we they me him her us them my your his its our their this that these those what which who do does did not no so if then just very can will would should could have has had there here".split())
JA_STOP = {"為る", "居る", "有る", "無い", "事", "成る", "良い", "そう", "行く", "言う", "呉れる", "物", "此の", "其の", "何", "様", "所", "今", "遣る", "来る", "見る", "思う", "知る"}

def paraphrases(word, k=8, min_freq=20):
    score = Counter()
    for e, p1 in by_ja[word].items():
        if e in EN_STOP:
            continue
        for f, p2 in by_en[e].items():
            if f != word and f not in JA_STOP and freq[f] >= min_freq:
                score[f] += p1 * p2
    return [(w, round(s, 3)) for w, s in score.most_common(k)]

for w in ["嬉しい", "素晴らしい", "始める", "病院", "多分", "怖い", "話す", "綺麗", "仕事", "直ぐ"]:
    print(w, "→", paraphrases(w))
```

```python
# 誤りの原因を見る: どの英語を経由したか
def explain(word, para):
    return sorted(((round(p1 * by_en[e].get(para, 0), 4), e) for e, p1 in by_ja[word].items() if para in by_en[e]), reverse=True)[:4]

for word, para in [("嬉しい", "幸せ"), ("話す", "言う"), ("怖い", "恐れる")]:
    print(word, "→", para, "経由した英語:", explain(word, para))
```

- 2 つの表現が同じ外国語の表現に訳されるなら、それらは言い換えの候補になる (バイリンガル・ピボッティング)。英語の PPDB など、大規模な言い換えデータベースがこの方法で作られた。
- ここでは単語アライメントの最も単純なモデル (IBM モデル 1) で単語の翻訳確率を求めた。句の言い換えを集めるなら、機械翻訳の章の Moses が作るフレーズテーブルを使う。
- 機能語 (英語の to、the、日本語の「為る」「居る」) は、どの語の翻訳候補にも顔を出すので、そのまま計算すると言い換えの上位を占めてしまう。両側の機能語を除くのが必須。
- 多義語を経由すると誤る (英語の 1 語が複数の意味を持つと、無関係な語どうしが結び付く)。反義語や、上位語・下位語も混ざりやすい。経由した語を確認すると原因が分かる。
- 得られるのは「候補」であり、文脈によって言い換えられるかどうかは変わる。単語ベクトルの類似度や言語モデルのスコアと組み合わせて絞り込む。
- 言い換えは、テキスト平易化 (難しい語をやさしい語に)、検索のクエリ拡張、データ拡張、評価指標 (言い換えを正解と認める) に使われる。
