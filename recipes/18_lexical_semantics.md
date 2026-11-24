# 語の意味と意味の変化

文脈から語の意味を判定し、コーパスから語彙知識を集め、意味の変化を見つける。WordNet で同義語・上位語を引く方法、上位語・下位語や同じカテゴリの語、言い換えの自動獲得（パターンによる抽出、ブートストラッピングと意味ドリフト、対訳を介した言い換え）、語義曖昧性解消と語義推定、慣用句・機能表現の検出と字義どおりの用法との区別、時代の違うコーパスを比べて頻度や使われ方が変わった単語を見つける方法（単語ベクトルの位置合わせ、近傍語や共起の比較、文脈化ベクトル、評価データでの検証）。少数のシードから始める半教師あり学習（自己学習・Yarowsky アルゴリズムと共訓練 Co-training、意味ドリフトの演習を含む）については、姉妹編『[自然言語処理の教科書](https://github.com/mamorlis/nlpbook)』の第 1 章も参照。

## WordNet で同義語・上位語・類似度を調べる（日本語も） {wordnet}

単語の同義語や上位概念を辞書から引きたい。日本語 WordNet も使いたい。

```python
import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)  # 多言語 WordNet (NLTK のバージョンによって 1.4 か 2.0 を使う)
nltk.download("omw-2.0", quiet=True)
from nltk.corpus import wordnet as wn

for syn in wn.synsets("bank")[:4]:
    print(syn.name(), syn.lemma_names(), "-", syn.definition())
```

```python
dog, cat, car = wn.synset("dog.n.01"), wn.synset("cat.n.01"), wn.synset("car.n.01")
print("上位語:", [s.name() for s in dog.hypernyms()])
print("下位語:", [s.name() for s in dog.hyponyms()][:5])
print("上位語をたどる:", " > ".join(s.lemma_names()[0] for s in dog.hypernym_paths()[0]))
print("類似度 dog-cat:", round(dog.wup_similarity(cat), 3), " dog-car:", round(dog.wup_similarity(car), 3))
```

```python
# 日本語 WordNet (Open Multilingual Wordnet 経由)
for syn in wn.synsets("犬", lang="jpn"):
    print(syn.name(), syn.lemma_names("jpn"), "|", syn.lemma_names()[:3])
print(sorted({l for s in wn.synsets("美しい", lang="jpn") for l in s.lemma_names("jpn")})[:15])
```

- 同義語によるデータ拡張 (EDA) や、語義曖昧性解消のベースライン (Lesk) に使える。

## 「A などの B」のパターンで上位語と下位語の組を集める {hypernym_patterns}

「犬は動物の一種」のような上位・下位の関係（is-a 関係）を、テキストから自動的に集めたい。

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
- 並列（「A、B などの C」）をさかのぼるとき、文の区切りを越えて無関係な語（「一方」「傍ら」）を拾うことがある。係り受け解析を使えば減らせる。
- 名詞句の範囲の取り方（どこまでを 1 語とするか）で結果が大きく変わる。ここでは連続する名詞をまとめただけなので、「の」でつながる長い名詞句は切れている。
- 再現率は低い（パターンに当てはまる文にしか効かない）。大量のテキストに適用して数で補うか、次項のブートストラッピングでパターンを増やす。

## 少数の例から同じカテゴリの語を増やす（ブートストラッピング） {bootstrapping}

「日本、フランス、中国」のような数個の例（シード）を与えるだけで、同じカテゴリの語をコーパスから集めたい。

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

- ブートストラッピングは「シードとよく一緒に出る文脈（パターン）を見つけ、そのパターンに当てはまる別の語を集める」ことを繰り返す。正解のラベルを数個しか必要としない。
- ここでの計算は Espresso というアルゴリズムを簡略化したもの。パターンの信頼度と語の信頼度を、PMI を重みにして交互に計算する。
- 「X を弾く」（楽器）、「X の小説」（作家）のように、カテゴリに特有のパターンが自動的に選ばれる。パターンを見ると、何を手がかりにしたかが分かる。
- 1 回の反復でも、かなり良い候補が得られる。ただし「音楽」「多く」のような、カテゴリの外の語も混ざる。
- 反復を増やせば候補は増えるが、次項のように別の問題（意味ドリフト）が起きる。

## ブートストラッピングの意味ドリフトを観察して抑える {semantic_drift}

ブートストラッピングを繰り返すと、シードと関係のない語ばかりが集まるようになってしまう（意味ドリフト）。なぜ起きるかを確かめ、抑える方法を試したい。

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

- 意味ドリフトは、多くのパターンと共起する一般的な語（「もの」「作品」「こと」）が一度入ると、そこから無関係なパターンと語が芋づる式に増えていく現象。
- 簡略化した Espresso の反復は、行列のべき乗法と同じ計算になっている。繰り返すほどシードの情報が消え、どのシードでも同じ順位（グラフの中心性）に収束する。これは HITS アルゴリズムが権威ノードを求める計算と同じ。
- フォン・ノイマン カーネルは、α が小さいとシードの近くを重視し、大きいとグラフ全体の中心性に近づく。正則化ラプラシアンは、ハブを通る経路を割り引くので、一般的な語に流れにくい（その代わり、頻度の低い語が上位に来やすい）。
- 実際の Espresso は、各反復で信頼度の高いパターンと語だけを少数ずつ残すことで、ドリフトを遅らせている。反復回数を増やしすぎない、一般的すぎるパターンをストップリストで除く、といった対策も使われる。
- どの方法でも、パラメータ（α、β、反復回数）で「シードへの近さ」と「広がり」の釣り合いを調整することになる。少数の正解データで確かめながら決める。
- 少数の正解をグラフの辺に沿って広げる考え方は、文書の分類にも使える（テキスト分類の章の「少数のラベルをグラフで広げて分類する（scikit-learn の LabelSpreading）」）。ハブを通って違うものに広がってしまう問題も共通している。

## 対訳コーパスを介して言い換えを集める（ピボット） {paraphrase_pivot}

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
# IBM モデル 1 で単語の翻訳確率 t(英語 | 日本語) を推定する（EM アルゴリズム）
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

# 語は UniDic の原形の表記で指定する（「きれい」「綺麗」は「奇麗」、「すぐ」は「直ぐ」になる）
for w in ["嬉しい", "素晴らしい", "始める", "病院", "多分", "怖い", "話す", "奇麗", "仕事", "直ぐ"]:
    print(w, "→", paraphrases(w))
```

```python
# 誤りの原因を見る: どの英語を経由したか
def explain(word, para):
    return sorted(((round(p1 * by_en[e].get(para, 0), 4), e) for e, p1 in by_ja[word].items() if para in by_en[e]), reverse=True)[:4]

for word, para in [("嬉しい", "幸せ"), ("話す", "言う"), ("怖い", "恐れる")]:
    print(word, "→", para, "経由した英語:", explain(word, para))
```

- 2 つの表現が同じ外国語の表現に訳されるなら、それらは言い換えの候補になる（バイリンガル・ピボッティング）。英語の PPDB など、大規模な言い換えデータベースがこの方法で作られた。
- ここでは単語アライメントの最も単純なモデル（IBM モデル 1）で単語の翻訳確率を求めた。句の言い換えを集めるなら、機械翻訳の章の Moses が作るフレーズテーブルを使う。
- 機能語（英語の to、the、日本語の「為る」「居る」）は、どの語の翻訳候補にも顔を出すので、そのまま計算すると言い換えの上位を占めてしまう。両側の機能語を除くのが必須。
- 多義語を経由すると誤る（英語の 1 語が複数の意味を持つと、無関係な語どうしが結び付く）。反義語や、上位語・下位語も混ざりやすい。経由した語を確認すると原因が分かる。
- 得られるのは「候補」であり、文脈によって言い換えられるかどうかは変わる。単語ベクトルの類似度や言語モデルのスコアと組み合わせて絞り込む。
- 言い換えは、テキスト平易化（難しい語をやさしい語に）、検索のクエリ拡張、データ拡張、評価指標（言い換えを正解と認める）に使われる。

## 多義語の語義を文脈から判定する（語義曖昧性解消） {wsd_en}

`bank` が「銀行」なのか「土手」なのかを、文脈から自動で判定したい。WordNet の語義のどれに当たるかを選ぶ。

```python
import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk

sentences = ["I deposited my salary at the bank this morning.", "We had a picnic on the bank of the river.",
             "The pilot had to bank the plane sharply to the left.", "He caught a huge bass in the lake.",
             "She plays bass in a jazz band."]
targets = ["bank", "bank", "bank", "bass", "bass"]

# 1. Lesk アルゴリズム: 文脈の単語と、語義の定義文の単語の重なりが最も大きい語義を選ぶ
for sent, word in zip(sentences, targets):
    syn = lesk(nltk.word_tokenize(sent), word)
    print(f"{syn.name():22} {syn.definition()[:60]}\n    ← {sent}")
```

```python
# 2. 定義文の埋め込みと文脈の埋め込みの類似度で選ぶ (単語の重なりがなくても意味が近ければ選べる)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def wsd(sentence, word, pos=None):
    synsets = wn.synsets(word, pos=pos)
    glosses = [f"{word}: {s.definition()}. {' '.join(s.examples())}" for s in synsets]
    sims = model.encode(glosses, normalize_embeddings=True) @ model.encode(sentence, normalize_embeddings=True)
    best = sims.argsort()[::-1]
    return [(synsets[i].name(), round(float(sims[i]), 3), synsets[i].definition()) for i in best[:2]]

for sent, word in zip(sentences, targets):
    (name, score, definition), second = wsd(sent, word)
    print(f"{name:22} {score}  {definition[:60]}\n    ← {sent}   (次点: {second[0]} {second[1]})")
```

- 単純な Lesk は、定義文が短いので単語がほとんど重ならず、精度が低い。埋め込みを使うと大きく改善する。
- 「常に最頻の語義（WordNet の 1 番目の語義）を選ぶ」というベースラインが非常に強いことで有名。評価するときは必ず比較する。
- WordNet の語義は細かすぎるという批判もある（`bank` の名詞だけで 10 個）。応用では、粗い語義にまとめて使うことが多い。
- 日本語 WordNet (`wn.synsets("首", lang="jpn")`) と多言語の埋め込みで同じことができるが、定義文が英語であることもあり、試した範囲では精度は低かった。日本語では次項の語義推定のほうが扱いやすい。
- 語ではなく句の単位で、慣用句の意味か字義どおりかを文脈から判定する例は、この章の「慣用句・機能表現を見つけて字義どおりの用法と区別する」を参照。

## 文脈化ベクトルをクラスタリングして語義を自動で分ける（語義推定） {word_sense_induction}

辞書の語義に頼らずに、コーパス中の用例を「同じ意味で使われているもの同士」にグループ分けしたい。新語や専門用語の意味の分析に使える。

```python
!pip install -q fugashi unidic-lite
```

```python
import torch
from transformers import AutoModel, AutoTokenizer

name = "tohoku-nlp/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name).eval()

def target_vector(sentence, target):
    """文中の target に当たるトークンの文脈化ベクトル (複数トークンなら平均) を返す"""
    enc = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    pieces = tokenizer.tokenize(target)
    start = next(i for i in range(len(tokens)) if tokens[i:i + len(pieces)] == pieces)
    with torch.no_grad():
        hidden = model(**enc, output_hidden_states=True).hidden_states
    layers = torch.stack(hidden[-4:]).mean(0)[0]              # 最後の 4 層の平均
    return layers[start:start + len(pieces)].mean(0)

sentences = [
    "タクシーのドライバーに行き先を伝えた。", "長距離トラックのドライバーは睡眠不足になりやすい。", "彼は F1 のドライバーとして活躍した。", "バスのドライバーが不足している。",
    "プリンタのドライバーをインストールし直した。", "グラフィックボードのドライバーを最新版に更新する。", "古い OS 用のドライバーが見つからない。", "ドライバーの不具合で画面が映らない。",
    "ネジを締めるためにプラスのドライバーを使う。", "工具箱からドライバーとペンチを取り出した。", "精密ドライバーで眼鏡のネジを直した。", "電動ドライバーがあると家具の組み立てが楽だ。",
]
X = torch.stack([target_vector(s, "ドライバー") for s in sentences])
X = torch.nn.functional.normalize(X, dim=-1).numpy()
print(X.shape)
```

```python
from sklearn.cluster import KMeans

labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)
for c in sorted(set(labels)):
    print(c, [s for s, l in zip(sentences, labels) if l == c])
```

```python
# 用例同士の類似度を見る: 同じ語義の用例は近く、違う語義の用例は遠い
import pandas as pd

sim = pd.DataFrame(X @ X.T, index=[s[:10] for s in sentences]).round(2)
sim.iloc[[0, 1, 4, 5, 8, 9], [0, 1, 4, 5, 8, 9]]
```

- 同じ「ドライバー」でも、運転手・ソフトウェア・工具で文脈化ベクトルが分かれる。これを利用して、語義の一覧を与えずに用例を分類するのが語義推定 (word sense induction)。
- クラスタ数は自分で決める必要がある。語義の数は用途によって変わるので、唯一の正解はない。
- 話題がはっきり違う語義は分かれやすい。「高い」（値段・背・声）のように同じ構文で使われる語義は、ベクトルが近くなって分かれにくい。
- 時期の違うコーパスで同じ語のベクトルの分布を比べると、意味変化の検出（「やばい」の意味の広がりなど）に応用できる。

## 時代別のコーパスを作って頻度が変わった単語を調べる {diachronic_corpus}

書かれた時代の違うテキストを集めて、時代によって使われる頻度が大きく変わった単語を調べたい。

```python
!pip install -q datasets fugashi unidic-lite
```

```python
import math, random, re
from collections import Counter
from datasets import load_dataset
from fugashi import Tagger

ds = load_dataset("globis-university/aozorabunko-clean", split="train")     # 青空文庫の約 1.7 万作品 (約 240MB)
print(ds[0]["meta"]["作品名"], ds[0]["meta"]["姓"], ds[0]["meta"]["生年月日"], ds[0]["meta"]["文字遣い種別"])

def era(meta):
    """作者の生年で時代を分ける。表記の違いが混ざらないよう、新字新仮名の作品だけを使う"""
    year = (meta.get("生年月日") or "")[:4]
    if meta["文字遣い種別"] != "新字新仮名" or not year.isdigit():
        return None
    return "旧" if 1850 <= int(year) < 1880 else "新" if 1895 <= int(year) < 1930 else None

texts = {"旧": [], "新": []}
for ex in ds:
    if era(ex["meta"]):
        texts[era(ex["meta"])].append(ex["text"])
print({e: len(t) for e, t in texts.items()})
```

```python
tagger = Tagger()
random.seed(0)

def count_words(works, max_chars=10_000_000):
    counts, n = Counter(), 0
    for text in random.sample(works, len(works)):
        text = text[:30000]                                   # 長編 1 作品に偏らないよう、各作品の冒頭だけを使う
        in_work = Counter(w.surface for w in tagger(text) if w.feature.pos1 == "名詞" and w.feature.pos2 == "普通名詞")
        counts.update({w: min(c, 5) for w, c in in_work.items()})     # 1 作品の中で何度出ても 5 回までしか数えない (登場人物名などの偏りを抑える)
        n += len(text)
        if n > max_chars:
            break
    return counts

freq = {e: count_words(t) for e, t in texts.items()}
total = {e: sum(c.values()) for e, c in freq.items()}
print(total)
```

```python
def log_ratio(word, smooth=5):
    """100 万語あたりの頻度の比の対数 (正なら新しい時代に増えた)。頻度 0 で発散しないよう、少し足してから比べる"""
    old = (freq["旧"][word] + smooth) / total["旧"]
    new = (freq["新"][word] + smooth) / total["新"]
    return math.log2(new / old)

vocab = [w for w in set(freq["旧"]) | set(freq["新"]) if freq["旧"][w] + freq["新"][w] >= 50 and len(w) >= 2]
ranked = sorted(vocab, key=log_ratio)
fmt = lambda ws: " ".join(f"{w}({freq['旧'][w]}→{freq['新'][w]})" for w in ws)
print("増えた語:", fmt(ranked[::-1][:30]))
print("減った語:", fmt(ranked[:30]))
for w in ["ラジオ", "映画", "放送", "電話", "汽車", "馬車", "書生", "細君"]:
    print(f"{w}: 100 万語あたり {freq['旧'][w] / total['旧'] * 1e6:.0f} → {freq['新'][w] / total['新'] * 1e6:.0f} (対数比 {log_ratio(w):+.2f})")
```

- 意味の変化を調べる前に、まず頻度の変化を見る。新しい事物の登場（ラジオ、映画）や、すたれた事物（馬車、書生）の減少は頻度に表れる。ただし、汽車や細君のように、ほとんど変わらない語もある。
- 時代の情報として、ここでは作者の生年を使った。作品の発表年が分かるコーパス（新聞、雑誌、国会会議録）なら、より正確に分けられる。
- 時代による違いと、ジャンルや作者の違いは混ざる。特定の作品にしか出ない語（登場人物の名前など）が上位に来やすいので、各作品から使う量をそろえ、1 作品あたりの回数に上限を設け、固有名詞を除いている。
- 形態素解析の単位にも注意する。UniDic では「自動車」が「自動 / 車」に分かれるので、そのままでは数えられない。
- 日本語の本格的な通時コーパスには、国立国語研究所の日本語歴史コーパス (CHJ) や昭和・平成書き言葉コーパスがある（利用には申請が必要）。

## 時代別の単語ベクトルを位置合わせして意味が変わった単語を探す {semantic_change_w2v}

時代によって意味や使われ方が変わった単語を、単語ベクトルを使って自動的に見つけたい。

```python
!pip install -q datasets fugashi unidic-lite gensim
```

```python
import random, re
import numpy as np
from datasets import load_dataset
from fugashi import Tagger
from gensim.models import Word2Vec

ds = load_dataset("globis-university/aozorabunko-clean", split="train")
tagger = Tagger()
random.seed(0)

def era(meta):
    year = (meta.get("生年月日") or "")[:4]
    if meta["文字遣い種別"] != "新字新仮名" or not year.isdigit():
        return None
    return "旧" if 1850 <= int(year) < 1880 else "新" if 1895 <= int(year) < 1930 else None

texts, common_nouns = {"旧": [], "新": []}, set()
for ex in ds:
    if era(ex["meta"]):
        texts[era(ex["meta"])].append(ex["text"])

def tokenize(works, max_chars=20_000_000):
    sentences, n = [], 0
    for text in random.sample(works, len(works)):
        for s in re.split(r"(?<=[。！？])|\n", text[:30000]):
            if len(s.strip()) >= 5:
                words = tagger(s.strip())
                common_nouns.update(w.surface for w in words if w.feature.pos2 == "普通名詞")
                sentences.append([w.surface for w in words]); n += len(s)
        if n > max_chars:
            break
    return sentences

corpus = {e: tokenize(t) for e, t in texts.items()}
print({e: len(s) for e, s in corpus.items()})
```

```python
# 時代ごとに別々の word2vec を学習する (それぞれ 1 分ほど)
models = {e: Word2Vec(s, vector_size=100, window=5, min_count=30, sg=1, epochs=3, workers=4, seed=0).wv for e, s in corpus.items()}

# 2 つの空間は向きがばらばらなので、そのままでは比べられない。両方に十分出てくる語を使って、回転で重ね合わせる (直交プロクラステス)
vocab = [w for w in models["旧"].index_to_key if w in models["新"].key_to_index
         and models["旧"].get_vecattr(w, "count") >= 100 and models["新"].get_vecattr(w, "count") >= 100]
A = np.stack([models["旧"][w] for w in vocab]); A /= np.linalg.norm(A, axis=1, keepdims=True)
B = np.stack([models["新"][w] for w in vocab]); B /= np.linalg.norm(B, axis=1, keepdims=True)
U, _, Vt = np.linalg.svd(A.T @ B)
R = U @ Vt                                         # A @ R が B に最も近くなる回転行列
similarity = dict(zip(vocab, ((A @ R) * B).sum(axis=1)))
print(len(vocab), "語 / 位置合わせ前の平均コサイン", round(float((A * B).sum(axis=1).mean()), 3), "→ 後", round(float(np.mean(list(similarity.values()))), 3))
```

```python
# 重ね合わせた後もベクトルが離れている語 = 使われ方が変わった語
targets = [w for w in vocab if w in common_nouns and re.fullmatch(r"[一-龥ァ-ヶー]{2,}", w)]
ranked = sorted(targets, key=similarity.get)
print("変化が大きい:", [(w, round(float(similarity[w]), 2)) for w in ranked[:30]])
print("変化が小さい:", [(w, round(float(similarity[w]), 2)) for w in ranked[-15:]])

for w in ["原子", "同志", "活動", "戦争", "自由", "映画", "言葉"]:
    old = [x for x, _ in models["旧"].most_similar(w, topn=6)]
    new = [x for x, _ in models["新"].most_similar(w, topn=6)]
    print(f"{w} ({similarity[w]:.2f})  旧: {' '.join(old)}  →  新: {' '.join(new)}")
```

- 時代ごとに単語ベクトルを学習し、回転で重ね合わせてから同じ単語のベクトルを比べる。コサイン類似度が低い語ほど、周りに出てくる語（文脈）が変わっている。
- 「原子」（物理の用語 → 爆弾）、「同志」（仲間一般 → 政治運動の仲間）のように、近い語の顔ぶれを並べると、何が変わったかが読み取れる。数値だけでなく、必ず近傍語を確認する。
- 上位には、人名と同じ形の語や、特定の作品に偏って出てくる語も混ざる。本当の意味変化と、コーパスの偏りを区別するのは人の仕事になる。
- 頻度の低い語はベクトルが不安定で、それだけで「変化が大きい」と判定されやすい。頻度の下限を設け、できれば乱数シードを変えて結果が安定しているか確かめる。
- この方法は Hamilton らの手法（SGNS + 直交プロクラステス）で、意味変化の検出の標準的なベースライン。
- 同じ直交プロクラステスで 2 言語の単語ベクトルを重ね、対訳語を引く例は、単語ベクトルの章の「2 言語の単語ベクトルを線形写像で対応させて対訳語を引く」を参照。

## 近傍語の入れ替わりで意味の変化を測る {semantic_change_neighbors}

単語ベクトルの位置合わせをせずに、近い語の顔ぶれがどれだけ入れ替わったかで意味の変化を測りたい。

```python
!pip install -q datasets fugashi unidic-lite gensim
```

```python
import random, re
import numpy as np
from datasets import load_dataset
from fugashi import Tagger
from gensim.models import Word2Vec

ds = load_dataset("globis-university/aozorabunko-clean", split="train")
tagger = Tagger()
random.seed(0)

def era(meta):
    year = (meta.get("生年月日") or "")[:4]
    if meta["文字遣い種別"] != "新字新仮名" or not year.isdigit():
        return None
    return "旧" if 1850 <= int(year) < 1880 else "新" if 1895 <= int(year) < 1930 else None

texts, common_nouns = {"旧": [], "新": []}, set()
for ex in ds:
    if era(ex["meta"]):
        texts[era(ex["meta"])].append(ex["text"])

def tokenize(works, max_chars=20_000_000):
    sentences, n = [], 0
    for text in random.sample(works, len(works)):
        for s in re.split(r"(?<=[。！？])|\n", text[:30000]):
            if len(s.strip()) >= 5:
                words = tagger(s.strip())
                common_nouns.update(w.surface for w in words if w.feature.pos2 == "普通名詞")
                sentences.append([w.surface for w in words]); n += len(s)
        if n > max_chars:
            break
    return sentences

models = {e: Word2Vec(tokenize(t), vector_size=100, window=5, min_count=30, sg=1, epochs=3, workers=4, seed=0).wv for e, t in texts.items()}
vocab = [w for w in models["旧"].index_to_key if w in models["新"].key_to_index
         and models["旧"].get_vecattr(w, "count") >= 100 and models["新"].get_vecattr(w, "count") >= 100]
```

```python
K = 25
shared = set(vocab)

def neighbors(wv, word, k=K):
    return [w for w, _ in wv.most_similar(word, topn=k * 4) if w in shared][:k]      # 両方の時代にある語だけを近傍とする

def jaccard_change(word):
    """近傍語の集合がどれだけ入れ替わったか (1 − Jaccard 係数)"""
    a, b = set(neighbors(models["旧"], word)), set(neighbors(models["新"], word))
    return 1 - len(a & b) / len(a | b)

def second_order_change(word):
    """両時代の近傍語を合わせた集合に対する類似度の並びを比べる。同じ語の集合を物差しにするので、位置合わせが要らない"""
    ref = sorted(set(neighbors(models["旧"], word)) | set(neighbors(models["新"], word)))
    a = np.array([models["旧"].similarity(word, r) for r in ref])
    b = np.array([models["新"].similarity(word, r) for r in ref])
    return 1 - float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))

for w in ["原子", "同志", "活動", "戦争", "言葉", "必要"]:
    print(f"{w}: 近傍の入れ替わり {jaccard_change(w):.2f} / 二次の類似度による変化 {second_order_change(w):.3f}")
```

```python
# 位置合わせによる方法 (直交プロクラステス) と、どれくらい同じ結果になるか
from scipy.stats import spearmanr

A = np.stack([models["旧"][w] for w in vocab]); A /= np.linalg.norm(A, axis=1, keepdims=True)
B = np.stack([models["新"][w] for w in vocab]); B /= np.linalg.norm(B, axis=1, keepdims=True)
U, _, Vt = np.linalg.svd(A.T @ B)
procrustes = dict(zip(vocab, 1 - ((A @ U @ Vt) * B).sum(axis=1)))

targets = [w for w in vocab if w in common_nouns and re.fullmatch(r"[一-龥ァ-ヶー]{2,}", w)]
sample = random.sample(targets, 500)
local = {w: second_order_change(w) for w in sample}
print("順位相関:", round(spearmanr([procrustes[w] for w in sample], [local[w] for w in sample]).statistic, 3))
print("近傍で見て変化が大きい:", sorted(sample, key=local.get, reverse=True)[:20])
print("位置合わせで見て変化が大きい:", sorted(sample, key=procrustes.get, reverse=True)[:20])
```

- 近傍語の比較は、ベクトル空間どうしを重ねる必要がない。別々のモデル、別々の次元数、別々の手法で作ったベクトルでも比べられる。
- 位置合わせによる方法は、文脈全体の変化（連想の変化など）を広く拾う。近傍語による方法は、入れ替え可能な語の変化、つまり語義そのものの変化に敏感だとされている。
- 近傍の数 K によって結果が変わる。小さいと偶然の入れ替わりに左右され、大きいと変化が薄まる。
- 2 つの方法の順位はある程度一致するが、同じではない。複数の指標で上位に来る語から調べるのが堅実。

## 共起する語の変化から意味の変化を読み取る (PPMI) {semantic_change_ppmi}

ある単語と一緒に使われる語が、時代によってどう変わったかを直接見たい。ニューラルネットを使わず、共起の統計だけで調べる。

```python
!pip install -q datasets fugashi unidic-lite
```

```python
import math, random, re
from collections import Counter, defaultdict
from datasets import load_dataset
from fugashi import Tagger

ds = load_dataset("globis-university/aozorabunko-clean", split="train")
tagger = Tagger()
random.seed(0)

def era(meta):
    year = (meta.get("生年月日") or "")[:4]
    if meta["文字遣い種別"] != "新字新仮名" or not year.isdigit():
        return None
    return "旧" if 1850 <= int(year) < 1880 else "新" if 1895 <= int(year) < 1930 else None

texts = {"旧": [], "新": []}
for ex in ds:
    if era(ex["meta"]):
        texts[era(ex["meta"])].append(ex["text"])

TARGETS = ["原子", "同志", "活動", "戦争", "自由", "言葉", "必要", "機械", "放送", "電話"]
CONTENT = {"名詞", "動詞", "形容詞", "形状詞"}

def cooccurrence(works, window=5, max_chars=20_000_000):
    """対象語の前後 window 語に出てくる内容語（対象語そのものは除く）を数える"""
    pair, word, n = defaultdict(Counter), Counter(), 0
    for text in random.sample(works, len(works)):
        for s in re.split(r"(?<=[。！？])|\n", text[:30000]):
            words = [w.surface if w.feature.pos2 == "固有名詞" else (w.feature.lemma or w.surface).split("-")[0]     # 活用語は原形にまとめる
                     for w in tagger(s) if w.feature.pos1 in CONTENT]
            word.update(words)
            for i, w in enumerate(words):
                if w in TARGETS:
                    pair[w].update(c for c in words[max(0, i - window):i] + words[i + 1:i + 1 + window] if c != w)
        n += min(len(text), 30000)
        if n > max_chars:
            break
    return pair, word

stats = {e: cooccurrence(t) for e, t in texts.items()}
print({e: (sum(word.values()), {t: sum(pair[t].values()) for t in TARGETS[:4]}) for e, (pair, word) in stats.items()})
```

```python
def ppmi(e, target, min_count=5):
    """PPMI(対象語, 文脈語) = max(0, log2 [P(対象語, 文脈語) / (P(対象語) P(文脈語))])"""
    pair, word = stats[e]
    total, n_target = sum(word.values()), sum(pair[target].values())
    return {c: max(0.0, math.log2((n / n_target) / (word[c] / total))) for c, n in pair[target].items() if n >= min_count}

def cosine(a, b):
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))

for t in TARGETS:
    old, new = ppmi("旧", t), ppmi("新", t)
    print(f"{t}: 類似度 {cosine(old, new):.2f}")
```

```python
# どの共起語が増えて、どの共起語が減ったか
def diff(target, k=10):
    old, new = ppmi("旧", target), ppmi("新", target)
    delta = {c: new.get(c, 0.0) - old.get(c, 0.0) for c in set(old) | set(new)}
    ranked = sorted(delta, key=delta.get)
    return ranked[:k], ranked[::-1][:k]

for t in ["原子", "同志", "活動", "放送"]:
    lost, gained = diff(t)
    print(f"{t}\n   旧い時代に特徴的: {' '.join(lost)}\n   新しい時代に特徴的: {' '.join(gained)}")
```

- PPMI のベクトルは、各次元が「どの語と共起するか」をそのまま表す。時代が違っても次元の意味が同じなので、位置合わせなしで比べられ、どの共起語が変化に効いたかも読める。
- word2vec のような密なベクトルより、変化の中身を説明しやすい。「原子」と「爆弾」の共起が増えた、のように具体的な根拠を示せる。
- 頻度の低い語どうしの PMI は不安定に大きくなる。共起回数の下限を設ける、平滑化する、といった対処が要る。
- 対象語の頻度が時代で大きく違うと、共起語の種類の数が変わり、それだけで類似度が下がる。頻度の影響を差し引いて解釈する。

## 複数の時代にまたがる意味の変化を 1 つのモデルで追う (Temporal Referencing) {temporal_referencing}

3 つ以上の時代にわたって、ある単語の使われ方がいつ変わったのかを見たい。時代ごとのモデルの位置合わせをせずに済ませたい。

```python
!pip install -q datasets fugashi unidic-lite gensim japanize-matplotlib
```

```python
import random, re
import numpy as np
from datasets import load_dataset
from fugashi import Tagger
from gensim.models import Word2Vec

ds = load_dataset("globis-university/aozorabunko-clean", split="train")
tagger = Tagger("-Owakati")
random.seed(0)
PERIODS = [(1850, 1870), (1870, 1885), (1885, 1895), (1895, 1905), (1905, 1930)]     # 作者の生年で 5 つに分ける
TARGETS = ["原子", "同志", "活動", "戦争", "自由", "放送", "言葉", "必要"]

def period(meta):
    year = (meta.get("生年月日") or "")[:4]
    if meta["文字遣い種別"] != "新字新仮名" or not year.isdigit():
        return None
    return next((i for i, (a, b) in enumerate(PERIODS) if a <= int(year) < b), None)

texts = [[] for _ in PERIODS]
for ex in ds:
    if period(ex["meta"]) is not None:
        texts[period(ex["meta"])].append(ex["text"])

# 対象語だけに時代の印を付ける (原子 → 原子_0, 原子_1, ...)。文脈語はそのままなので、全時代で共有される
sentences, counts = [], {}
for p, works in enumerate(texts):
    n = 0
    for text in random.sample(works, len(works)):
        for s in re.split(r"(?<=[。！？])|\n", text[:30000]):
            if len(s.strip()) >= 5:
                sentences.append([f"{w}_{p}" if w in TARGETS else w for w in tagger.parse(s.strip()).split()]); n += len(s)
        if n > 12_000_000:
            break
    counts[p] = n
print(len(sentences), "文 / 時代ごとの文字数", counts)
```

```python
wv = Word2Vec(sentences, vector_size=100, window=5, min_count=10, sg=1, epochs=3, workers=4, seed=0).wv

def similarity_matrix(word):
    """時代 × 時代の類似度行列。値が急に下がる境目が、使われ方の変わった時期"""
    keys = [f"{word}_{p}" for p in range(len(PERIODS))]
    return np.array([[wv.similarity(a, b) if a in wv and b in wv else np.nan for b in keys] for a in keys])

for w in ["原子", "同志", "言葉"]:
    print(w, "各時代での出現回数:", [int(wv.get_vecattr(f"{w}_{p}", "count")) if f"{w}_{p}" in wv else 0 for p in range(len(PERIODS))])
    print(similarity_matrix(w).round(2))
```

```python
import japanize_matplotlib
import matplotlib.pyplot as plt

labels = [f"{a}-{b - 1}" for a, b in PERIODS]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, w in zip(axes.ravel(), TARGETS):
    im = ax.imshow(similarity_matrix(w), vmin=0.3, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(labels)), labels, rotation=45, fontsize=8); ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_title(w)
fig.colorbar(im, ax=axes, shrink=0.6, label="コサイン類似度")
plt.show()

# 各時代での近傍語
for w in ["原子", "同志"]:
    for p in range(len(PERIODS)):
        if f"{w}_{p}" in wv:
            near = [x for x, _ in wv.most_similar(f"{w}_{p}", topn=12) if "_" not in x][:7]
            print(f"{w} ({labels[p]} 年生まれの作者): {' '.join(near)}")
```

- 対象語だけを時代別の別の単語として扱い、1 つのモデルで学習する。全時代が同じベクトル空間に入るので、位置合わせによる誤差が生じない。
- 時代 × 時代の類似度行列を見ると、変化が徐々に進んだのか、ある時期を境に急に起きたのかが分かる。隣り合う時代だけを比べるより、情報が多い。
- 対象語をあらかじめ決めておく必要がある。全単語を対象にすると、時代ごとのデータが細切れになって学習できない。
- 用例の少ない時代のベクトルは不安定なので、各時代での出現回数を必ず確認する。回数が少ない時代との類似度は、意味の変化ではなく雑音かもしれない。

## BERT の文脈化ベクトルで単語の使われ方の変化を調べる {semantic_change_bert}

1 つ 1 つの用例を文脈化ベクトルにして、時代ごとの用法の広がりや割合の変化を調べたい。

```python
!pip install -q datasets fugashi unidic-lite scikit-learn
```

```python
import random, re
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

ds = load_dataset("globis-university/aozorabunko-clean", split="train")
random.seed(0)
TARGETS = ["原子", "同志", "活動", "言葉", "必要"]

def era(meta):
    year = (meta.get("生年月日") or "")[:4]
    if meta["文字遣い種別"] != "新字新仮名" or not year.isdigit():
        return None
    return "旧" if 1850 <= int(year) < 1880 else "新" if 1895 <= int(year) < 1930 else None

# 対象語を含む文 (用例) を、時代ごとに最大 150 件ずつ集める
usages = {(t, e): [] for t in TARGETS for e in ["旧", "新"]}
order = list(range(len(ds))); random.shuffle(order)
for i in order:
    e = era(ds[i]["meta"])
    if e is None:
        continue
    per_work = {t: 0 for t in TARGETS}
    for s in re.split(r"(?<=[。！？])|\n", ds[i]["text"][:30000]):
        for t in TARGETS:
            if t in s and 10 <= len(s) <= 120 and len(usages[t, e]) < 150 and per_work[t] < 3:     # 1 作品からは 3 件まで
                usages[t, e].append(s.strip()); per_work[t] += 1
    if all(len(u) >= 150 for u in usages.values()):
        break
print({k: len(v) for k, v in usages.items()})
```

```python
name = "tohoku-nlp/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name).eval()

@torch.no_grad()
def usage_vectors(word, sentences, batch_size=32):
    """文の中の対象語の位置にあたるトークンのベクトル (最終層) を取り出す"""
    word_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
    vectors = []
    for i in range(0, len(sentences), batch_size):
        enc = tokenizer(sentences[i:i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=128)
        hidden = model(**enc).last_hidden_state
        for ids, h in zip(enc["input_ids"].tolist(), hidden):
            for j in range(len(ids) - len(word_ids) + 1):
                if ids[j:j + len(word_ids)] == word_ids:
                    vectors.append(h[j:j + len(word_ids)].mean(0).numpy()); break
    return np.stack(vectors)

vectors = {k: usage_vectors(k[0], v) for k, v in usages.items()}
print({k: v.shape[0] for k, v in vectors.items()})
```

```python
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances

def prt(a, b):
    """時代ごとの平均ベクトル (プロトタイプ) どうしの距離"""
    return float(cosine_distances(a.mean(0, keepdims=True), b.mean(0, keepdims=True))[0, 0])

def apd(a, b):
    """時代をまたぐ用例の組の、平均距離 (average pairwise distance)"""
    return float(cosine_distances(a, b).mean())

def jsd(a, b, k=4):
    """全用例をクラスタリングして、各クラスタ (≒ 用法) の割合の分布を時代間で比べる (Jensen-Shannon 距離)"""
    labels = KMeans(k, n_init=10, random_state=0).fit_predict(np.vstack([a, b]))
    p = np.bincount(labels[:len(a)], minlength=k) / len(a); q = np.bincount(labels[len(a):], minlength=k) / len(b)
    m = (p + q) / 2
    kl = lambda x, y: float(np.sum(x[x > 0] * np.log2(x[x > 0] / y[x > 0])))
    return (kl(p, m) + kl(q, m)) / 2, labels, p, q

for t in TARGETS:
    a, b = vectors[t, "旧"], vectors[t, "新"]
    print(f"{t}: PRT {prt(a, b):.3f} / APD {apd(a, b):.3f} / 時代内の平均距離 旧 {apd(a, a):.3f} 新 {apd(b, b):.3f} / JSD {jsd(a, b)[0]:.3f}")
```

```python
# クラスタごとに用例を見て、どの用法が増減したかを確かめる
def show_clusters(word, k=4, n=2):
    a, b = vectors[word, "旧"], vectors[word, "新"]
    _, labels, p, q = jsd(a, b, k)
    sents = usages[word, "旧"][:len(a)] + usages[word, "新"][:len(b)]
    for c in np.argsort(q - p):
        print(f"[クラスタ {c}] 旧 {p[c]:.0%} → 新 {q[c]:.0%}")
        for s in [s for s, l in zip(sents, labels) if l == c][:n]:
            print("    ", s[:70])

show_clusters("同志")
show_clusters("原子")
```

- 単語ベクトル (word2vec) は 1 語に 1 つのベクトルしかなく、複数の語義が混ざる。文脈化ベクトルは用例ごとに別のベクトルになるので、「どの用法が増えたか」まで調べられる。
- PRT は平均どうしの距離、APD は用例の組の平均距離、JSD は用法（クラスタ）の割合の変化。同じ語でも指標によって順位が変わるので、複数を並べて見る。
- 「同志」は、単語ベクトルでは大きく変化して見えたが、ここではほとんど差が出ない。用例を読むと、どちらの時代も「若いもの同志」のような「同士」の意味が大半で、政治的な用法は一部にすぎない。少数の新しい用法は、平均や距離の指標には表れにくい。
- 学習済みの BERT は現代語のテキストで学習されている。古い時代の文章や表記には弱く、文体の違いが「意味の変化」として表れてしまうことがある。
- 用例の数が少ないと、どの指標も不安定になる。時代ごとに 100 件以上を目安にし、1 つの作品に偏らないように集める。
- クラスタは語義と一致するとは限らない。文体や構文の違いで分かれることも多いので、必ず用例を読んで確かめる。

## 意味変化の検出手法を評価データで比べる (SemEval-2020 Task 1) {semantic_change_eval}

意味変化の度合いを測る手法が、人手で付けた「変化の大きさ」とどれくらい合うかを確かめ、手法どうしを比べたい。

```python
!pip install -q gensim
!wget -q https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_eng.zip && unzip -q -o semeval2020_ulscd_eng.zip
```

```python
import gzip
import numpy as np
from gensim.models import Word2Vec
from scipy.stats import spearmanr

base = "semeval2020_ulscd_eng/"
gold = dict(line.split("\t") for line in open(base + "truth/graded.txt").read().strip().split("\n"))
gold = {w: float(v) for w, v in gold.items()}                 # 対象語 37 語と、人手の注釈から求めた変化の度合い (0〜1)
corpora = {}
for name, path in [("1810-1860", "corpus1/lemma/ccoha1.txt.gz"), ("1960-2010", "corpus2/lemma/ccoha2.txt.gz")]:
    with gzip.open(base + path, "rt", encoding="utf-8") as f:
        corpora[name] = [line.split() for line in f]          # 見出し語化済み。対象語には attack_nn のように品詞が付いている
    print(name, len(corpora[name]), "文", sum(map(len, corpora[name])), "語")
print(sorted(gold.items(), key=lambda x: -x[1])[:6], "...")
```

```python
c1, c2 = corpora.values()
models = [Word2Vec(c, vector_size=100, window=10, min_count=10, sg=1, negative=5, epochs=5, workers=4, seed=0).wv for c in (c1, c2)]

# 手法 1: 直交プロクラステスで位置合わせしてコサイン距離
vocab = [w for w in models[0].index_to_key if w in models[1].key_to_index]
A = np.stack([models[0][w] for w in vocab]); A = (A - A.mean(0)); A /= np.linalg.norm(A, axis=1, keepdims=True)
B = np.stack([models[1][w] for w in vocab]); B = (B - B.mean(0)); B /= np.linalg.norm(B, axis=1, keepdims=True)
U, _, Vt = np.linalg.svd(A.T @ B)
index = {w: i for i, w in enumerate(vocab)}
procrustes = {w: 1 - float((A[index[w]] @ U @ Vt) @ B[index[w]]) for w in gold}

# 手法 2: 近傍語の入れ替わり (位置合わせなし)
def neighbor_change(word, k=50):
    shared = set(vocab)
    near = [[w for w, _ in m.most_similar(word, topn=k * 3) if w in shared][:k] for m in models]
    ref = sorted(set(near[0]) | set(near[1]))
    a, b = (np.array([m.similarity(word, r) for r in ref]) for m in models)
    return 1 - float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
neighbor = {w: neighbor_change(w) for w in gold}

# ベースライン: 相対頻度の変化
count = [{w: m.get_vecattr(w, "count") / sum(map(len, c)) for w in gold} for m, c in zip(models, (c1, c2))]
frequency = {w: abs(np.log(count[1][w] / count[0][w])) for w in gold}
```

```python
words = sorted(gold)
for name, scores in [("頻度の変化 (ベースライン)", frequency), ("近傍語の入れ替わり", neighbor), ("位置合わせ + コサイン距離", procrustes)]:
    r = spearmanr([gold[w] for w in words], [scores[w] for w in words])
    print(f"{name}: Spearman = {r.statistic:.3f} (p = {r.pvalue:.3f})")

print("正解で変化が大きい語:", sorted(gold, key=gold.get, reverse=True)[:8])
print("手法が変化が大きいとした語:", sorted(procrustes, key=procrustes.get, reverse=True)[:8])
for w in ["plane_nn", "tip_vb", "chairman_nn"]:
    print(w, round(gold[w], 2), "|", [x for x, _ in models[0].most_similar(w, topn=5)], "→", [x for x, _ in models[1].most_similar(w, topn=5)])
```

- SemEval-2020 Task 1 は、2 つの時代のコーパスと、対象語ごとの変化の度合い（人手の注釈から計算）を提供する。手法の出したスコアとの順位相関 (Spearman) で評価する。英語のほか、ドイツ語、ラテン語、スウェーデン語がある。
- 対象語が 37 語しかないので、相関係数の信頼区間は広い。0.1 程度の差は、乱数シードを変えるだけで入れ替わる。複数のシードと複数の言語で確かめる。
- 「頻度が変わった語は意味も変わった」という単純なベースラインと必ず比べる。これを大きく超えられない手法は、意味ではなく頻度を見ているだけかもしれない。
- このタスクでは、単語ベクトルを位置合わせする古典的な方法が、文脈化ベクトルを使う方法と同等以上の成績を出したことが知られている。新しい手法が常に良いとは限らない。
- 日本語には、日本語歴史コーパスなどを使った評価データ (JaSemChange) がある。コーパスの利用申請が必要。

## 慣用句・機能表現を見つけて字義どおりの用法と区別する {idiom_detection gpu}

「足を洗う」が「悪い仕事をやめる」（慣用）なのか「足を水で洗う」（字義）なのかを、文脈から判定したい。形態素解析の原形の並びで候補を見つける段階と、文脈から慣用か字義かを決める段階に分ける。

```python
!pip install -q fugashi unidic-lite
```

```python
!git clone -q --depth 1 https://github.com/nlp-waseda/OpenMWE
```

```python
from fugashi import Tagger

tagger = Tagger()

# 複単語表現の辞書: 原形 (または表層) の並び。助詞は「は」「も」に変わることがあるので候補を並べる
PATTERNS = {
    "足を洗う": ([{"足"}, {"を", "は", "も"}, {"洗う"}], 2),     # 2 = 要素の間に挟まってよい語の数
    "骨が折れる": ([{"骨"}, {"が", "は", "も"}, {"折れる"}], 2),
    "顔を出す": ([{"顔"}, {"を", "は", "も"}, {"出す"}], 2),
    "手を焼く": ([{"手"}, {"を", "は", "も"}, {"焼く"}], 2),
    "腹が立つ": ([{"腹"}, {"が", "は", "も"}, {"立つ"}], 1),
    "について": ([{"に"}, {"つく", "付く"}, {"て"}], 0),          # 機能表現は語を挟まない
    "に対して": ([{"に"}, {"対する"}, {"て"}], 0),
    "にもかかわらず": ([{"に"}, {"も"}, {"関わる", "拘る"}, {"ず"}], 0),
}

def find_mwes(sentence):
    """文中の候補を (表現, 開始文字位置, 終了文字位置) のリストで返す"""
    words, pos = [], 0
    for w in tagger(sentence):
        start = sentence.index(w.surface, pos); pos = start + len(w.surface)
        words.append((w.surface, w.feature.lemma, start, pos))
    hits = []
    for name, (pattern, gap) in PATTERNS.items():
        for i, w in enumerate(words):
            if not ({w[0], w[1]} & pattern[0]):
                continue
            j, ok = i, True
            for p in pattern[1:]:                           # 次の要素を gap 語以内で探す
                nxt = next((k for k in range(j + 1, min(j + 2 + gap, len(words))) if {words[k][0], words[k][1]} & p), None)
                if nxt is None:
                    ok = False; break
                j = nxt
            if ok:
                hits.append((name, words[i][2], words[j][3]))
    return hits

for s in ["彼は三年前に裏の仕事から足を洗った。", "砂浜を歩いたあと、水道で足をきれいに洗った。", "彼の態度には本当に腹が立つ。",
          "雨にもかかわらず大勢が集まった。", "質問に対して丁寧に答えた。", "泥が靴について取れない。",
          "足が痛くて、手を洗った。", "骨は折れていなかった。"]:
    print(s, "→", [(n, s[a:b]) for n, a, b in find_mwes(s)])
```

```python
# 手作りのデータ: 表現ごとに、少数の例 (慣用 4 + 字義 4) と、評価用 (慣用 4 + 字義 4)。ラベル 1 = 慣用・機能表現、0 = 字義どおり
DATA = {
    "足を洗う": {
        "few": [(1, "彼は二十歳で暴走族から足を洗った。"), (1, "ギャンブルからはもう足を洗ったつもりだ。"), (1, "裏の商売から足を洗って、田舎で農業を始めた。"), (1, "詐欺グループから足を洗うのは簡単ではない。"),
                (0, "泥だらけになったので、玄関の前で足を洗った。"), (0, "海から上がって、シャワーで足を洗う。"), (0, "寝る前に石けんで足を洗う習慣がある。"), (0, "子どもの足を洗ってからタオルで拭いた。")],
        "test": [(1, "組を抜けて足を洗ってから、もう十年になる。"), (1, "夜の仕事から足を洗いたいと相談された。"), (1, "盗みから足を洗った男が、更生施設で働いている。"), (1, "彼女は芸能界から足を洗って会社員になった。"),
                 (0, "畑仕事のあとは、井戸の水で足を洗う。"), (0, "温泉の入口に足を洗う場所がある。"), (0, "砂浜を歩いたあと、水道で足をきれいに洗った。"), (0, "介護の仕事では、お年寄りの足を洗うこともある。")]},
    "骨が折れる": {
        "few": [(1, "大量の書類を整理するのは骨が折れる。"), (1, "子ども三人の世話はなかなか骨が折れる。"), (1, "頑固な父を説得するのに骨が折れた。"), (1, "古い資料を探すのは骨が折れる作業だ。"),
                (0, "階段から落ちて、腕の骨が折れた。"), (0, "スキーで転んで足の骨が折れた。"), (0, "骨が折れていないか、医者に診てもらった。"), (0, "年を取ると、少し転んだだけで骨が折れることがある。")],
        "test": [(1, "英語の論文を一人で翻訳するのは骨が折れた。"), (1, "この山道を自転車で登るのは骨が折れる。"), (1, "新しい制度に慣れるまでが骨が折れる。"), (1, "全員の予定を合わせるのは本当に骨が折れる。"),
                 (0, "交通事故で首の骨が折れる重傷を負った。"), (0, "転倒した選手は、手首の骨が折れていた。"), (0, "傘の骨が折れて使えなくなった。"), (0, "犬が骨をかじっていたら、骨が折れて二つになった。")]},
    "顔を出す": {
        "few": [(1, "久しぶりに同窓会に顔を出した。"), (1, "忙しくても、週に一度は実家に顔を出すようにしている。"), (1, "打ち上げに少しだけ顔を出してから帰った。"), (1, "社長が珍しく現場に顔を出した。"),
                (0, "窓から顔を出して外の様子をうかがった。"), (0, "亀が甲羅から顔を出した。"), (0, "布団から顔を出すと、部屋は寒かった。"), (0, "カーテンの隙間から子どもが顔を出した。")],
        "test": [(1, "時間があれば、明日の勉強会にも顔を出してください。"), (1, "彼は最近、サークルに全く顔を出さない。"), (1, "取引先の新年会に顔を出す予定だ。"), (1, "退院した祖父が久しぶりに町内会に顔を出した。"),
                 (0, "車の窓から顔を出すのは危ない。"), (0, "モグラが土の中から顔を出した。"), (0, "ドアの陰から猫が顔を出した。"), (0, "水面から顔を出して大きく息を吸った。")]},
    "手を焼く": {
        "few": [(1, "息子のいたずらに手を焼いている。"), (1, "新しいシステムの不具合に手を焼いた。"), (1, "頑固な客の対応に手を焼く。"), (1, "雑草の多さには毎年手を焼く。"),
                (0, "熱いフライパンに触って手を焼いてしまった。"), (0, "たき火で手を焼かないように気をつけて。"), (0, "アイロンで手を焼いて、水ぶくれができた。"), (0, "花火の火で手を焼いた。")],
        "test": [(1, "反抗期の娘に手を焼いている。"), (1, "野良猫のいたずらに近所中が手を焼いている。"), (1, "このチームは相手の守備に手を焼いた。"), (1, "方言が強くて、聞き取りに手を焼いた。"),
                 (0, "オーブンの天板で手を焼いて、病院に行った。"), (0, "炭火で手を焼かないよう、長いトングを使う。"), (0, "やけどの手当てをしたが、手を焼いた跡が残った。"), (0, "鍋の取っ手が熱くて、うっかり手を焼いた。")]},
    "について": {
        "few": [(1, "環境問題について議論した。"), (1, "この点について質問があります。"), (1, "将来について考える時間がほしい。"), (1, "事故の原因について調べている。"),
                (0, "ガイドの後について歩いた。"), (0, "泥が靴について取れない。"), (0, "米粒が頬について、みんなに笑われた。"), (0, "名前を呼ばれるまで席について待った。")],
        "test": [(1, "新製品の価格について発表があった。"), (1, "日本の歴史について本を書いた。"), (1, "彼の発言について、どう思いますか。"), (1, "安全対策について説明を受けた。"),
                 (0, "列の最後について順番を待った。"), (0, "猫の毛が服について困る。"), (0, "有名な先生について、三年間ピアノを習った。"), (0, "全員が席について、会議が始まった。")]},
}

# 第 1 段階: 原形の並びで候補を見つける。対象の表現が見つかった位置を、次の段階で使う
examples = []                                               # (表現, 分割, ラベル, 文, 開始, 終了)
for name, d in DATA.items():
    for split in ["few", "test"]:
        for label, s in d[split]:
            span = [(a, b) for n, a, b in find_mwes(s) if n == name]
            assert span, f"見つからない: {s}"
            examples.append((name, split, label, s, *span[0]))
print(len(examples), "文で候補を検出できた")
```

```python
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

name = "tohoku-nlp/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(name)
bert = AutoModel.from_pretrained(name).eval().to("cuda")

@torch.no_grad()
def span_vectors(items, batch_size=64):
    """(文, 開始, 終了) のリストから、表現の部分のトークンの文脈化ベクトル (最後の 4 層の平均) を返す"""
    out = []
    for i in range(0, len(items), batch_size):
        ids, spans = [], []
        for s, a, b in items[i:i + batch_size]:
            left, mid, right = (tokenizer.encode(x, add_special_tokens=False) for x in (s[max(0, a - 60):a], s[a:b], s[b:b + 60]))
            ids.append([tokenizer.cls_token_id] + left + mid + right + [tokenizer.sep_token_id])
            spans.append((1 + len(left), 1 + len(left) + len(mid)))
        n = max(map(len, ids))
        input_ids = torch.tensor([x + [tokenizer.pad_token_id] * (n - len(x)) for x in ids], device="cuda")
        mask = (input_ids != tokenizer.pad_token_id).long()
        hidden = torch.stack(bert(input_ids=input_ids, attention_mask=mask, output_hidden_states=True).hidden_states[-4:]).mean(0)
        out += [hidden[k, a:b].mean(0).cpu().numpy() for k, (a, b) in enumerate(spans)]
    X = np.stack(out)
    return X / np.linalg.norm(X, axis=1, keepdims=True)

X = span_vectors([(s, a, b) for _, _, _, s, a, b in examples])
print(X.shape)
```

```python
from sklearn.linear_model import LogisticRegression

names = list(DATA)
def evaluate(predict):
    """評価用の 40 文で、表現ごとと全体の正解率を出す"""
    acc = {}
    for n in names:
        idx = [i for i, e in enumerate(examples) if e[0] == n and e[1] == "test"]
        acc[n] = np.mean([predict(i) == examples[i][2] for i in idx])
    acc["全体"] = np.mean(list(acc.values()))
    return acc

# 第 2 段階 (1): 表現ごとに、少数の例 (慣用 4 + 字義 4) でロジスティック回帰を学習する
few_shot = {}
for n in names:
    idx = [i for i, e in enumerate(examples) if e[0] == n and e[1] == "few"]
    few_shot[n] = LogisticRegression(C=10, max_iter=1000).fit(X[idx], [examples[i][2] for i in idx])
results = {"常に慣用 (ベースライン)": evaluate(lambda i: 1),
           "BERT + 少数の例 (表現ごと)": evaluate(lambda i: few_shot[examples[i][0]].predict(X[i:i + 1])[0])}
for k, v in results.items():
    print(f"{k:24}", " ".join(f"{n} {a:.0%}" for n, a in v.items()))
```

```python
# 公開の慣用句コーパス OpenMWE: 慣用句 146 個の用例に、慣用 (I) か字義 (L) かのラベルが付いている
import glob, random, re

def load_openmwe(path, n_per_label=None, seed=0):
    rows = {"I": [], "L": []}
    for line in open(path, encoding="euc-jp", errors="replace"):
        label, _, lemma, text = line.rstrip("\n").split(" ", 3)
        if "<" not in text or ">" not in text:
            continue
        a, b = text.index("<"), text.rindex(">")
        clean = text[:a] + re.sub("[<>]", "", text[a:b + 1]) + text[b + 1:].replace("<", "").replace(">", "")
        rows[label].append((lemma, int(label == "I"), clean, a, a + len(re.sub("[<>]", "", text[a:b + 1]))))
    random.Random(seed).shuffle(rows["I"]); random.Random(seed).shuffle(rows["L"])
    return rows["I"][:n_per_label] + rows["L"][:n_per_label]

corpus = {}
for path in sorted(glob.glob("OpenMWE/D/*")):
    rows = load_openmwe(path)
    if sum(r[1] for r in rows) >= 100 and sum(1 - r[1] for r in rows) >= 100:   # 慣用・字義とも 100 例以上ある慣用句
        corpus[rows[0][0]] = rows
print(len(corpus), "個の慣用句。例:", list(corpus)[:8])
print([n for n in names if n in corpus], "は OpenMWE にもある")
```

```python
# 第 2 段階 (2): OpenMWE の多数の例で学習する
# (a) 同じ慣用句の例を増やすと、どこまで上がるか (OpenMWE から取り分けた 200 例と、手作りの評価用で測る)
for n in [n for n in names if n in corpus]:
    rows = corpus[n]
    I, L = [r for r in rows if r[1] == 1], [r for r in rows if r[1] == 0]
    held = I[:100] + L[:100]
    Xh = span_vectors([r[2:] for r in held]); yh = [r[1] for r in held]
    line = [f"手作りの 8 例: OpenMWE {few_shot[n].score(Xh, yh):.0%}"]
    for k in [4, 16, 64]:
        train = I[100:100 + k] + L[100:100 + k]
        clf = LogisticRegression(C=10, max_iter=1000).fit(span_vectors([r[2:] for r in train]), [r[1] for r in train])
        idx = [i for i, e in enumerate(examples) if e[0] == n and e[1] == "test"]
        line.append(f"k={k:2}: OpenMWE {clf.score(Xh, yh):.0%} 手作り {clf.score(X[idx], [examples[i][2] for i in idx]):.0%}")
    print(f"{n}:", " / ".join(line))
```

```python
# (b) 評価する 5 表現以外の慣用句だけで、表現によらない「慣用か字義か」の分類器を作る
train = [r for n, rows in corpus.items() if n not in names for r in (
    [x for x in rows if x[1] == 1][:60] + [x for x in rows if x[1] == 0][:60])]
clf_all = LogisticRegression(C=1, max_iter=2000).fit(span_vectors([r[2:] for r in train]), [r[1] for r in train])
print(len(train), "例 (", len(corpus) - sum(n in corpus for n in names), "個の慣用句) で学習")
for n in [n for n in names if n in corpus]:
    held = [r for r in corpus[n] if r[1] == 1][:100] + [r for r in corpus[n] if r[1] == 0][:100]
    print(f"{n}: OpenMWE の 200 例で {clf_all.score(span_vectors([r[2:] for r in held]), [r[1] for r in held]):.0%}")
results["BERT + 他の慣用句で学習"] = evaluate(lambda i: clf_all.predict(X[i:i + 1])[0])
print("手作りの評価用:", " ".join(f"{n} {a:.0%}" for n, a in results["BERT + 他の慣用句で学習"].items()))
print("誤り:", [(examples[i][3], examples[i][2]) for i in range(len(examples))
             if examples[i][1] == "test" and clf_all.predict(X[i:i + 1])[0] != examples[i][2]])
```

```python
# 第 2 段階 (3): 小さな LLM に、2 つの意味のどちらかを選ばせる (A と B のトークンの確率を比べる)
from transformers import AutoModelForCausalLM

del bert; torch.cuda.empty_cache()
llm_name = "Qwen/Qwen3-4B-Instruct-2507"     # Alibaba の Qwen チーム、Apache 2.0
llm_tok = AutoTokenizer.from_pretrained(llm_name)
llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16, device_map="cuda").eval()

MEANING = {"足を洗う": ("悪い仕事や生活をやめる", "足を水などで洗う"), "骨が折れる": ("苦労する、手間がかかる", "体などの骨が折れる"),
           "顔を出す": ("集まりや場所を訪れる、参加する", "物の陰や中から顔をのぞかせる"), "手を焼く": ("扱いに困る", "手をやけどする"),
           "について": ("「〜に関して」という意味の機能表現", "「付く」の本来の意味 (くっつく、従う、席に着くなど)")}
A, B = (llm_tok.encode(x, add_special_tokens=False)[0] for x in ("A", "B"))

@torch.no_grad()
def llm_predict(i, shots=()):
    n, _, _, s, _, _ = examples[i]
    idiom, literal = MEANING[n]
    ask = lambda s: f"次の文の「{n}」はどちらの意味ですか。A か B の 1 文字で答えてください。\n文: {s}\nA: {idiom}\nB: {literal}"
    msgs = []
    for label, t in shots:
        msgs += [{"role": "user", "content": ask(t)}, {"role": "assistant", "content": "A" if label else "B"}]
    msgs.append({"role": "user", "content": ask(s)})
    enc = llm_tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
    logits = llm(**enc).logits[0, -1]
    return int(logits[A] > logits[B])

results["LLM (例なし)"] = evaluate(lambda i: llm_predict(i))
results["LLM (少数の例 8 個)"] = evaluate(lambda i: llm_predict(i, DATA[examples[i][0]]["few"]))
print("例なしの LLM の誤り:", [(examples[i][3], examples[i][2]) for i in range(len(examples))
                           if examples[i][1] == "test" and llm_predict(i) != examples[i][2]])
import pandas as pd
(pd.DataFrame(results).T * 100).round().astype(int)        # 手作りの評価用 40 文の正解率 (%)
```

- 第 1 段階は辞書と照合するだけなので、字義どおりの用法も候補になる（「骨は折れていなかった」も「骨が折れる」として見つかる）。助詞の入れ替わり（は・も）と、間に挟まる語（「足をきれいに洗う」）を許すかどうかで、見落としと拾いすぎの釣り合いが変わる。機能表現は語を挟まないので間を空けない。ここで作った 80 文は、すべて候補として見つかった。
- 第 2 段階は、語義曖昧性解消と同じ考え方で、表現の部分の文脈化ベクトルを分類する。手作りの評価用 40 文（表現ごとに慣用 4・字義 4）では、表現ごとに 8 例で学習しただけで全問正解した。ただし手作りの文は典型的すぎて易しい。同じ 8 例で学習した分類器を OpenMWE の Web の用例 200 例で測ると、「足を洗う」96%、「骨が折れる」95%、「顔を出す」82% に下がる。評価は実際の文章に近いデータでも行う。
- 「顔を出す」は、OpenMWE の例を 1 クラス 64 例に増やしても 83% にとどまった。OpenMWE では「集まりに顔を出す」（訪れる）だけでなく「悪癖が顔を出す」「土から顔を出す（作物）」のような比喩も慣用とされ、一方で「キノコが顔を出していた」は字義とされている。慣用と字義の境界がはっきりしない表現ほど難しい。
- 評価する表現を含まない 63 個の慣用句（7,560 例）で学習した「慣用か字義か」の分類器も、手作りの評価用で 95%、学習していない慣用句の OpenMWE の用例で 82〜96% だった。慣用的な用法に共通する文脈の特徴があるので、例のない表現にもある程度使える。誤った 2 文は「介護の仕事では、お年寄りの足を洗う」（文中の「仕事」に引かれたらしい）と「日本の歴史について」で、機能表現は学習データに入っていないことも効いている。
- 小さな LLM (Qwen3-4B-Instruct-2507) に 2 つの意味から選ばせると、例なしで 92%、8 例を見せて 95% だった。例なしでは「夜の仕事から足を洗いたい」を字義と、「有名な先生について習った」（付き従う）を機能表現と、「手を焼いた跡」を慣用と誤った。学習データを用意しなくてよい代わりに、表現ごとに 2 つの意味の説明を書く必要がある。
- 検証の環境では、モデルを読み込み済みの状態で全体が約 1 分。Colab では BERT（約 0.4GB）と LLM（約 8GB） のダウンロードに数分かかる。LLM の比較のセルだけ GPU を多く使う（float16 で T4 に収まる）。
- 語の意味を文脈から決める点で、前の「多義語の語義を文脈から判定する（語義曖昧性解消）」「文脈化ベクトルをクラスタリングして語義を自動で分ける（語義推定）」と同じ問題である。慣用句の辞書そのものを作るには、コーパス統計の章の「よく使われる言い回し（コロケーション）を抽出する」のように、よく一緒に現れる語の組を集めて候補にする方法がある（ただしコロケーションには「雨が降る」のように字義どおりのものも多い）。
- 日本語の慣用句コーパス OpenMWE（Hashimoto and Kawahara, 2008）は、慣用句 146 個の約 10 万例に慣用・字義のラベルを付けたもの。文字コードは EUC-JP。利用条件は BSD 3 条項に似た再配布可のライセンスで、Web の文を集めたものなので削除の依頼があれば利用者にも削除を求めるとされている。tohoku-nlp/bert-base-japanese-v3 と Qwen3-4B-Instruct-2507 は Apache 2.0、fugashi は MIT、unidic-lite はパッケージが MIT で、同梱の UniDic は GPL、LGPL、BSD から選べる。
