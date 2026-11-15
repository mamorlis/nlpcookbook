# 意味の変化 (通時的な分析)

時代の違うコーパスを比べて、頻度や使われ方が変わった単語を見つける。単語ベクトルの位置合わせ、近傍語や共起の比較、文脈化ベクトル、評価データでの検証まで。

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

- 意味の変化を調べる前に、まず頻度の変化を見る。新しい事物の登場 (ラジオ、映画) や、言い方の交代 (細君 → 妻) は頻度に表れる。
- 時代の情報として、ここでは作者の生年を使った。作品の発表年が分かるコーパス (新聞、雑誌、国会会議録) なら、より正確に分けられる。
- 時代による違いと、ジャンルや作者の違いは混ざる。特定の作品にしか出ない語 (登場人物の名前など) が上位に来やすいので、各作品から使う量をそろえ、1 作品あたりの回数に上限を設け、固有名詞を除いている。
- 形態素解析の単位にも注意する。UniDic では「自動車」が「自動 / 車」に分かれるので、そのままでは数えられない。
- 日本語の本格的な通時コーパスには、国立国語研究所の日本語歴史コーパス (CHJ) や昭和・平成書き言葉コーパスがある (利用には申請が必要)。

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

- 時代ごとに単語ベクトルを学習し、回転で重ね合わせてから同じ単語のベクトルを比べる。コサイン類似度が低い語ほど、周りに出てくる語 (文脈) が変わっている。
- 「原子」(物理の用語 → 爆弾)、「同志」(仲間一般 → 政治運動の仲間) のように、近い語の顔ぶれを並べると、何が変わったかが読み取れる。数値だけでなく、必ず近傍語を確認する。
- 上位には、人名と同じ形の語や、特定の作品に偏って出てくる語も混ざる。本当の意味変化と、コーパスの偏りを区別するのは人の仕事になる。
- 頻度の低い語はベクトルが不安定で、それだけで「変化が大きい」と判定されやすい。頻度の下限を設け、できれば乱数シードを変えて結果が安定しているか確かめる。
- この方法は Hamilton らの手法 (SGNS + 直交プロクラステス) で、意味変化の検出の標準的なベースライン。

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
- 位置合わせによる方法は、文脈全体の変化 (連想の変化など) を広く拾う。近傍語による方法は、入れ替え可能な語の変化、つまり語義そのものの変化に敏感だとされている。
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
    """対象語の前後 window 語に出てくる内容語を数える"""
    pair, word, n = defaultdict(Counter), Counter(), 0
    for text in random.sample(works, len(works)):
        for s in re.split(r"(?<=[。！？])|\n", text[:30000]):
            words = [w.surface if w.feature.pos2 == "固有名詞" else (w.feature.lemma or w.surface).split("-")[0]     # 活用語は原形にまとめる
                     for w in tagger(s) if w.feature.pos1 in CONTENT]
            word.update(words)
            for i, w in enumerate(words):
                if w in TARGETS:
                    pair[w].update(words[max(0, i - window):i] + words[i + 1:i + 1 + window])
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
- PRT は平均どうしの距離、APD は用例の組の平均距離、JSD は用法 (クラスタ) の割合の変化。同じ語でも指標によって順位が変わるので、複数を並べて見る。
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

- SemEval-2020 Task 1 は、2 つの時代のコーパスと、対象語ごとの変化の度合い (人手の注釈から計算) を提供する。手法の出したスコアとの順位相関 (Spearman) で評価する。英語のほか、ドイツ語、ラテン語、スウェーデン語がある。
- 対象語が 37 語しかないので、相関係数の信頼区間は広い。0.1 程度の差は、乱数シードを変えるだけで入れ替わる。複数のシードと複数の言語で確かめる。
- 「頻度が変わった語は意味も変わった」という単純なベースラインと必ず比べる。これを大きく超えられない手法は、意味ではなく頻度を見ているだけかもしれない。
- このタスクでは、単語ベクトルを位置合わせする古典的な方法が、文脈化ベクトルを使う方法と同等以上の成績を出したことが知られている。新しい手法が常に良いとは限らない。
- 日本語には、日本語歴史コーパスなどを使った評価データ (JaSemChange) がある。コーパスの利用申請が必要。
