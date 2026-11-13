# 系列ラベリング

トークンごとにラベルを付ける。BIO タグの扱い、CRF、BERT による固有表現抽出の学習と評価。

## 文字位置のアノテーションを BIO タグ列に変換する {spans_to_bio}

「何文字目から何文字目が人名」という形式 (doccano や brat の出力) のアノテーションを、トークンごとの BIO タグに変換したい。

```python
!pip install -q fugashi unidic-lite
```

```python
from fugashi import Tagger

tagger = Tagger()

def tokenize_with_offsets(text):
    pos, out = 0, []
    for w in tagger(text):
        start = text.index(w.surface, pos)
        pos = start + len(w.surface)
        out.append((w.surface, start, pos))
    return out

def spans_to_bio(text, spans):
    """spans: [(開始文字位置, 終了文字位置, ラベル), ...]"""
    tokens = tokenize_with_offsets(text)
    tags = ["O"] * len(tokens)
    for s, e, label in spans:
        inside = [i for i, (_, ts, te) in enumerate(tokens) if ts < e and te > s]   # スパンと重なるトークン
        for n, i in enumerate(inside):
            tags[i] = ("B-" if n == 0 else "I-") + label
        if inside and (tokens[inside[0]][1] != s or tokens[inside[-1]][2] != e):
            print(f"  警告: 境界がトークンと合わない: {text[s:e]!r} → {''.join(tokens[i][0] for i in inside)!r}")
    return [t for t, _, _ in tokens], tags

# doccano の JSONL 形式: {"text": ..., "label": [[start, end, label], ...]}
record = {"text": "山田太郎は2020年に京都大学を卒業し、東京都で働いている。",
          "label": [[0, 4, "PER"], [5, 10, "DATE"], [11, 15, "ORG"], [20, 22, "LOC"]]}
tokens, tags = spans_to_bio(record["text"], record["label"])
for tok, tag in zip(tokens, tags):
    print(f"{tok}\t{tag}")
```

```python
# アノテーションの境界がトークンの境界と合わない例
tokens, tags = spans_to_bio("東京都立大学の教授に会った。", [(0, 3, "LOC")])
print(list(zip(tokens, tags)))
```

- 「東京」だけがアノテーションされているのに、トークナイザが「東京都」を 1 語にする、といった境界のずれが必ず起きる。件数を数えて、多ければ文字単位のタグ付けやサブワードの利用を検討する。
- I- で始まる系列を許す IOB1、末尾と単独を区別する BIOES (BILOU) などの変種がある。データセットがどれかを確認する。

## BIO タグ列から固有表現のスパンを取り出す {bio_to_spans}

モデルが出力した BIO タグ列を、「どこからどこまでが何のエンティティか」という形に戻したい。不正なタグ列にも対処したい。

```python
def bio_to_spans(tags):
    """[(開始トークン位置, 終了トークン位置 (含まない), ラベル), ...] を返す。B- なしで始まる I- は新しいスパンの開始とみなす"""
    spans, start, label = [], None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, _, lab = tag.partition("-")
        if start is not None and (prefix != "I" or lab != label):   # 今のスパンが終わる
            spans.append((start, i, label))
            start, label = None, None
        if prefix == "B" or (prefix == "I" and start is None):      # 新しいスパンが始まる
            start, label = i, lab
    return spans

tokens = ["山田", "太郎", "は", "京都", "大学", "を", "卒業", "し", "、", "東京", "都", "で", "働く"]
tags = ["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "O", "O", "O", "O", "B-LOC", "I-LOC", "O", "O"]
for s, e, label in bio_to_spans(tags):
    print(label, "".join(tokens[s:e]))
```

```python
# モデルの出力にありがちな不正な系列
bad = ["I-PER", "I-PER", "O", "B-ORG", "I-LOC", "O", "B-PER", "B-PER"]
print(bio_to_spans(bad))

def repair(tags):
    """不正な I- を B- に直して、正しい BIO 系列にする"""
    out, prev = [], "O"
    for tag in tags:
        if tag.startswith("I-") and prev[2:] != tag[2:]:
            tag = "B-" + tag[2:]
        out.append(tag)
        prev = tag
    return out

print(repair(bad))
```

- 不正な系列の扱い方 (捨てる / 直す) で評価値が変わる。seqeval のデフォルトは、ここと同じく「I- で始まっても新しいスパンとみなす」寛容な方式。
- 根本的に防ぐには、CRF 層や、遷移制約付きのビタビ復号を使う。

## 固有表現抽出をエンティティ単位で評価する (seqeval) {seqeval}

系列ラベリングの結果を、トークン単位の正解率ではなく、エンティティ単位の適合率・再現率・F1 で評価したい。

```python
!pip install -q seqeval
```

```python
from seqeval.metrics import accuracy_score, classification_report, f1_score
from seqeval.scheme import IOB2

y_true = [["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "O", "O"], ["B-LOC", "O", "O", "B-PER", "O"], ["O", "O", "B-ORG", "I-ORG", "I-ORG"]]
y_pred = [["B-PER", "I-PER", "O", "B-ORG", "O", "O", "O"], ["B-LOC", "O", "O", "B-PER", "O"], ["O", "O", "B-LOC", "I-LOC", "I-LOC"]]

print("トークン単位の正解率:", round(accuracy_score(y_true, y_pred), 3))
print("エンティティ単位の F1:", round(f1_score(y_true, y_pred), 3))
print(classification_report(y_true, y_pred, digits=3))
```

```python
# 厳密モード: 不正な系列 (B- なしの I-) をエンティティとして数えない
bad_pred = [["I-PER", "I-PER", "O", "B-ORG", "I-ORG", "O", "O"], ["B-LOC", "O", "O", "B-PER", "O"], ["O", "O", "B-ORG", "I-ORG", "I-ORG"]]
print("default:", f1_score(y_true, bad_pred), " strict:", f1_score(y_true, bad_pred, mode="strict", scheme=IOB2))
```

- エンティティ単位では、範囲とラベルの両方が完全に一致したときだけ正解。境界が 1 トークンずれても、偽陽性 1 + 偽陰性 1 になる。
- O タグが大半を占めるので、トークン単位の正解率は高く出すぎて指標にならない。
- 論文で報告されるのは通常 micro 平均の F1。

## CRF で固有表現抽出器を学習する {crf_tagger}

ニューラルネットを使わずに、手作りの特徴量と条件付き確率場 (CRF) で系列ラベリングをしたい。CPU で数十秒で学習できる。

```python
!pip install -q sklearn-crfsuite seqeval
```

```python
import nltk
nltk.download("conll2002", quiet=True)
from nltk.corpus import conll2002

train = list(conll2002.iob_sents("esp.train"))   # スペイン語の固有表現データ: (単語, 品詞, BIO タグ)
test = list(conll2002.iob_sents("esp.testb"))
print(len(train), len(test), train[0][:6])

def word_features(sent, i):
    w, pos = sent[i][0], sent[i][1]
    f = {"bias": 1.0, "lower": w.lower(), "suffix3": w[-3:], "suffix2": w[-2:], "is_upper": w.isupper(),
         "is_title": w.istitle(), "is_digit": w.isdigit(), "pos": pos}
    for d in (-1, 1):   # 前後 1 語の特徴
        if 0 <= i + d < len(sent):
            w2, pos2 = sent[i + d][0], sent[i + d][1]
            f.update({f"{d}:lower": w2.lower(), f"{d}:is_title": w2.istitle(), f"{d}:pos": pos2})
        else:
            f["BOS" if d == -1 else "EOS"] = True
    return f

X_train = [[word_features(s, i) for i in range(len(s))] for s in train]
y_train = [[t for _, _, t in s] for s in train]
X_test = [[word_features(s, i) for i in range(len(s))] for s in test]
y_test = [[t for _, _, t in s] for s in test]
```

```python
import sklearn_crfsuite
from seqeval.metrics import classification_report

crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
crf.fit(X_train, y_train)
print(classification_report(y_test, crf.predict(X_test), digits=3))
```

```python
# 学習された遷移の重み: O → I-xxx のような不正な遷移には大きな負の重みが付く
from collections import Counter

trans = Counter(crf.transition_features_)
print("起きやすい遷移:", [(a, b, round(w, 1)) for (a, b), w in trans.most_common(5)])
print("起きにくい遷移:", [(a, b, round(w, 1)) for (a, b), w in trans.most_common()[-5:]])
```

- CRF はラベル間の遷移も学習するので、B- なしの I- のような不正な系列を出しにくい。
- 特徴量を自分で設計できる・学習が速い・挙動を解釈できるのが利点。少量データや計算資源が限られる場面では今でも有力。

## ビタビアルゴリズムを実装して最適なラベル列を求める {viterbi}

HMM や CRF の復号に使われるビタビアルゴリズムを、自分で実装して動きを確かめたい。

```python
import numpy as np

def viterbi(emission, transition, start):
    """emission: (系列長, ラベル数), transition: (ラベル数, ラベル数), start: (ラベル数,) の対数スコア"""
    T, K = emission.shape
    score = start + emission[0]
    back = np.zeros((T, K), dtype=int)
    for t in range(1, T):
        cand = score[:, None] + transition      # cand[i, j] = 直前が i で今が j のときのスコア
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + emission[t]
    path = [int(score.argmax())]
    for t in range(T - 1, 0, -1):               # 後ろからたどる
        path.append(int(back[t, path[-1]]))
    return path[::-1], float(score.max())

labels = ["名詞", "動詞", "助詞"]
words = ["猫", "が", "魚", "を", "食べる"]
log = lambda p: np.log(np.array(p) + 1e-12)
start = log([0.7, 0.2, 0.1])
transition = log([[0.2, 0.2, 0.6],     # 名詞 → 名詞 / 動詞 / 助詞
                  [0.3, 0.3, 0.4],     # 動詞 →
                  [0.6, 0.39, 0.01]])  # 助詞 →
emission = log([[0.9, 0.05, 0.05],     # 猫
                [0.3, 0.1, 0.6],       # が (「蛾」かもしれない)
                [0.9, 0.05, 0.05],     # 魚
                [0.1, 0.1, 0.8],       # を
                [0.1, 0.85, 0.05]])    # 食べる

path, score = viterbi(emission, transition, start)
print([f"{w}/{labels[i]}" for w, i in zip(words, path)], round(score, 3))
```

```python
# 貪欲法 (各位置で独立に最大のラベルを選ぶ) との比較、および総当たりとの一致の確認
from itertools import product

greedy = emission.argmax(axis=1).tolist()
def total(p):
    return start[p[0]] + emission[0, p[0]] + sum(transition[a, b] + emission[t, b] for t, (a, b) in enumerate(zip(p, p[1:]), 1))
brute = max(product(range(3), repeat=5), key=total)
print("貪欲法:", [labels[i] for i in greedy], " 総当たり:", [labels[i] for i in brute], " 一致:", list(brute) == path)
```

- 計算量は O(系列長 × ラベル数²)。総当たりの O(ラベル数^系列長) を動的計画法で避けている。
- 遷移スコアに「O → I- は −∞」のような制約を入れれば、ニューラルモデルの出力に対しても正しい BIO 系列だけを出力させられる。

## HMM で品詞タガーを作る {hmm_tagger}

隠れマルコフモデル (HMM) による古典的な品詞タグ付けを学習・評価して、ベースラインと比べたい。

```python
import nltk
nltk.download("treebank", quiet=True)
nltk.download("universal_tagset", quiet=True)
from nltk.corpus import treebank
from nltk.tag import DefaultTagger, UnigramTagger, hmm, tnt

sents = list(treebank.tagged_sents(tagset="universal"))
train, test = sents[:3500], sents[3500:]
print(len(train), len(test), train[0][:6])

baseline = UnigramTagger(train, backoff=DefaultTagger("NOUN"))   # 各単語に最頻の品詞を付けるだけ
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(train, estimator=lambda fd, bins: nltk.LidstoneProbDist(fd, 0.01, bins))

tnt_tagger = tnt.TnT(unk=DefaultTagger("NOUN"), Trained=True)   # trigram HMM + 未知語は名詞とみなす
tnt_tagger.train(train)

print("ユニグラム (最頻タグ):   ", round(baseline.accuracy(test), 4))
print("HMM (bigram):            ", round(tagger.accuracy(test), 4))
print("TnT (trigram + 未知語処理):", round(tnt_tagger.accuracy(test), 4))
```

```python
for sent in ["They can fish .", "I saw her duck .", "Time flies like an arrow ."]:
    print(tagger.tag(sent.split()))
```

```python
# 未知語と既知語に分けて正解率を見る
vocab = {w for s in train for w, _ in s}
known = unknown = known_ok = unknown_ok = 0
for s in test:
    for (w, gold), (_, pred) in zip(s, tagger.tag([w for w, _ in s])):
        if w in vocab:
            known += 1; known_ok += gold == pred
        else:
            unknown += 1; unknown_ok += gold == pred
print(f"既知語 {known_ok / known:.3f} ({known} 語) / 未知語 {unknown_ok / unknown:.3f} ({unknown} 語)")
```

- 「各単語の最頻タグ」だけで 9 割を超える。品詞タグ付けの難しさは、残りの曖昧な語と未知語にある。
- 素朴な HMM は未知語の出力確率をうまく見積もれず、未知語でほぼ当たらない。そのせいで全体ではユニグラムのベースラインに負けることすらある。
- HMM は単語の綴りを見ないのが弱点。接尾辞や大文字などの特徴を使える CRF やニューラルモデルが有利な理由。

## 単語分割を文字単位の系列ラベリングとして解く {char_segmentation}

日本語の単語分割を「各文字が単語の先頭 (B) か途中 (I) か」を当てる問題として、CRF で学習したい。

```python
!pip install -q sklearn-crfsuite fugashi unidic-lite regex
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

# MeCab の分割結果を正解とみなす (本来は人手で分割したコーパスを使う)
tagger = Tagger()
text = aozora("https://www.aozora.gr.jp/cards/000148/files/752_ruby_2438.zip")
sents = [[w.surface for w in tagger(s)] for s in re.split(r"(?<=。)|\n", text) if s and s.strip()]
train, test = sents[:2200], sents[2200:]

def to_chars_tags(words):
    chars = [c for w in words for c in w]
    tags = [t for w in words for t in ["B"] + ["I"] * (len(w) - 1)]
    return chars, tags
```

```python
import regex
import sklearn_crfsuite

def char_type(c):
    for name in ["Han", "Hiragana", "Katakana", "Latin", "Nd"]:
        if regex.match(rf"\p{{{name}}}", c):
            return name
    return "Other"

def features(chars, i):
    f = {"bias": 1.0}
    for d in range(-2, 3):          # 前後 2 文字の文字と文字種
        if 0 <= i + d < len(chars):
            f[f"c{d}"] = chars[i + d]
            f[f"t{d}"] = char_type(chars[i + d])
    for d in (-1, 0):               # 文字 bigram と文字種の変わり目
        if 0 <= i + d and i + d + 1 < len(chars):
            f[f"bi{d}"] = chars[i + d] + chars[i + d + 1]
            f[f"tt{d}"] = char_type(chars[i + d]) + ">" + char_type(chars[i + d + 1])
    return f

def dataset(sents):
    X, y = [], []
    for words in sents:
        chars, tags = to_chars_tags(words)
        X.append([features(chars, i) for i in range(len(chars))]); y.append(tags)
    return X, y

X_train, y_train = dataset(train)
X_test, y_test = dataset(test)
crf = sklearn_crfsuite.CRF(c1=0.1, c2=0.1, max_iterations=60).fit(X_train, y_train)
```

```python
def segment(sentence):
    chars = list(sentence)
    tags = crf.predict_single([features(chars, i) for i in range(len(chars))])
    return "".join((" " if t == "B" and i else "") + c for i, (c, t) in enumerate(zip(chars, tags))).split()

def word_f1(gold_sents, pred_sents):
    """単語単位の F1: 開始位置と終了位置が両方一致した単語だけを正解とする"""
    tp = n_gold = n_pred = 0
    for gold, pred in zip(gold_sents, pred_sents):
        spans = lambda ws: {(sum(map(len, ws[:i])), sum(map(len, ws[:i + 1]))) for i in range(len(ws))}
        g, p = spans(gold), spans(pred)
        tp += len(g & p); n_gold += len(g); n_pred += len(p)
    prec, rec = tp / n_pred, tp / n_gold
    return 2 * prec * rec / (prec + rec)

pred = [segment("".join(ws)) for ws in test]
print("単語 F1:", round(word_f1(test, pred), 4))
print(segment("親譲りの無鉄砲で小供の時から損ばかりしている。"))
print(segment("最新のスマートフォンで自然言語処理を勉強する。"))
```

- 辞書を使わない「点予測」に近い方式 (KyTea など) の簡易版。学習データにない分野の語 (2 文目) では誤りが増える。
- 中国語の単語分割では、B/M/E/S (先頭・中間・末尾・単独) の 4 タグが標準的。

## BERT をファインチューニングして日本語の固有表現抽出器を作る {bert_ner_finetune gpu norun}

日本語の固有表現抽出モデルを、文字位置のアノテーション付きデータから学習したい。

```python
!pip install -q datasets seqeval accelerate
```

```python
import json, urllib.request
from datasets import Dataset

# ストックマーク社の Wikipedia 固有表現データセット (CC BY-SA 3.0)
url = "https://raw.githubusercontent.com/stockmarkteam/ner-wikipedia-dataset/main/ner.json"
data = json.load(urllib.request.urlopen(url))
print(len(data), data[0])

types = sorted({e["type"] for d in data for e in d["entities"]})
labels = ["O"] + [f"{p}-{t}" for t in types for p in ("B", "I")]
label2id = {l: i for i, l in enumerate(labels)}
ds = Dataset.from_list([{"text": d["text"], "spans": [[*e["span"], e["type"]] for e in d["entities"]]} for d in data])
ds = ds.train_test_split(test_size=0.2, seed=0)
```

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

model_name = "xlm-roberta-base"   # 文字位置 (offset mapping) を返せる高速トークナイザを持つモデルを使う
tokenizer = AutoTokenizer.from_pretrained(model_name)

def encode(example):
    enc = tokenizer(example["text"], truncation=True, max_length=256, return_offsets_mapping=True)
    tags = []
    for s, e in enc["offset_mapping"]:
        if s == e:                                  # 特殊トークンは損失計算から除外
            tags.append(-100)
            continue
        tag = "O"
        for start, end, typ in example["spans"]:
            start, end = int(start), int(end)
            if s < end and e > start:               # トークンがスパンと重なる
                tag = ("B-" if s <= start else "I-") + typ
                break
        tags.append(label2id[tag])
    enc["labels"] = tags
    enc.pop("offset_mapping")
    return enc

tokenized = ds.map(encode, remove_columns=["text", "spans"])
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label2id)
```

```python
import numpy as np
from seqeval.metrics import classification_report, f1_score
from transformers import Trainer, TrainingArguments

def to_tags(logits, gold):
    pred = np.argmax(logits, axis=-1)
    y_true = [[labels[g] for g in row if g != -100] for row in gold]
    y_pred = [[labels[p] for p, g in zip(prow, grow) if g != -100] for prow, grow in zip(pred, gold)]
    return y_true, y_pred

def compute_metrics(eval_pred):
    y_true, y_pred = to_tags(*eval_pred)
    return {"f1": f1_score(y_true, y_pred)}

args = TrainingArguments(
    output_dir="ner-out", num_train_epochs=5, learning_rate=5e-5, warmup_ratio=0.1, weight_decay=0.01,
    per_device_train_batch_size=32, per_device_eval_batch_size=64, eval_strategy="epoch", save_strategy="no",
    fp16=True, logging_steps=50, report_to="none", seed=0,
)
trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["test"],
                  data_collator=DataCollatorForTokenClassification(tokenizer), compute_metrics=compute_metrics)
trainer.train()
out = trainer.predict(tokenized["test"])
print(classification_report(*to_tags(out.predictions, out.label_ids), digits=3))
```

```python
from transformers import pipeline

ner = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=0)
text = "ソニーグループの吉田憲一郎会長は、東京都港区の本社で新型プレイステーションを発表した。"
for e in ner(text):
    print(text[e["start"]:e["end"]], e["entity_group"], round(float(e["score"]), 2))
```

- T4 GPU で 5 エポック約 5 分、エンティティ単位の F1 は 0.85 前後。
- ここではサブワードすべてにラベルを付けている (先頭だけに付けて残りを -100 にする流儀もある)。評価は本来、単語や文字の単位に戻して行うのが厳密。
- 日本語特化の BERT (`tohoku-nlp/bert-base-japanese-v3`) のほうが精度は高いが、トークナイザが文字位置を返さないので、位置合わせを自分で実装する必要がある。
