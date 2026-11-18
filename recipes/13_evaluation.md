# 評価指標と統計的検定

分類・抽出・生成の出力を数値で評価し、その差が偶然でないかを確かめる。翻訳・要約・質問応答・検索に固有の指標は、それぞれの章を参照。LLM に生成文を採点させる方法は「LLM の評価と分析」の章にある。

## scikit-learn で適合率・再現率・F1 を集計する（micro / macro 平均と抽出の評価） {prf_manual}

問い合わせの自動振り分けの結果を、クラスごとの適合率・再現率・F1 と micro / macro 平均で正しく集計したい。固有表現や関係の抽出のように、予測と正解が「集合」になる出力も同じ道具で評価したい。

```python
# scikit-learn は Colab に最初から入っている (BSD-3-Clause)。検証に使った版は下の出力の 1 行目にある。
# 版をそろえるなら、このセルの前に別のセルを作って pip install でその版を指定して入れ、ランタイムを再起動する
import sklearn
from collections import Counter
from sklearn.metrics import (accuracy_score, classification_report, multilabel_confusion_matrix,
                             precision_recall_fscore_support)

print("scikit-learn", sklearn.__version__)

# （正解, 予測）の組と件数。「個人情報」は 3 件しかなく、一度も正しく予測されていない
pairs = {("料金", "料金"): 40, ("料金", "その他"): 5, ("解約", "解約"): 12, ("解約", "料金"): 3,
         ("故障", "故障"): 6, ("故障", "その他"): 4, ("個人情報", "その他"): 2, ("個人情報", "解約"): 1,
         ("その他", "その他"): 25, ("その他", "料金"): 3}
gold = [g for (g, p), n in pairs.items() for _ in range(n)]
pred = [p for (g, p), n in pairs.items() for _ in range(n)]
print("正解のクラスの件数:", Counter(gold))

# zero_division=0: 一度も予測されなかったクラスの適合率 (0 / 0) を 0 とし、警告を出さない
print(classification_report(gold, pred, digits=3, zero_division=0))
print("正解率:", round(accuracy_score(gold, pred), 3))
```

```python
import numpy as np
import pandas as pd

# 「その他」を除いた、振り分けたいクラスだけでの評価。labels= で集計に使うクラスを選ぶ
targets = ["料金", "解約", "故障", "個人情報"]
for avg in ["micro", "macro", "weighted"]:
    p, r, f, _ = precision_recall_fscore_support(gold, pred, labels=targets, average=avg, zero_division=0)
    print(f"{avg:8} (その他を除く): P = {p:.3f}  R = {r:.3f}  F1 = {f:.3f}")

# 表として扱う (output_dict=True)。F1 の低い順に並べると弱いクラスが分かる
rep = pd.DataFrame(classification_report(gold, pred, labels=targets, output_dict=True, zero_division=0)).T
print(rep.loc[targets].astype({"support": int}).sort_values("f1-score").round(3))

# 誤りの分析には件数も見る。multilabel_confusion_matrix はクラスごとに [[TN, FP], [FN, TP]] を返す
mcm = multilabel_confusion_matrix(gold, pred, labels=targets)
print(pd.DataFrame({"TP": mcm[:, 1, 1], "FP": mcm[:, 0, 1], "FN": mcm[:, 1, 0]}, index=targets))

# zero_division=np.nan にすると、0 / 0 になったクラスの適合率を macro 平均から除く
for zd in [0, np.nan]:
    p, r, f, _ = precision_recall_fscore_support(gold, pred, labels=targets, average="macro", zero_division=zd)
    print(f"zero_division={zd}: macro P = {p:.3f}  macro F1 = {f:.3f}")

# 「macro-F1」の 2 つの流儀: クラスごとの F1 の平均（scikit-learn の macro）と、macro P と macro R の調和平均
p, r, f, _ = precision_recall_fscore_support(gold, pred, labels=targets, average="macro", zero_division=0)
print(f"F1 の平均 = {f:.3f}   macro P と macro R の調和平均 = {2 * p * r / (p + r):.3f}")
```

```python
from sklearn.preprocessing import MultiLabelBinarizer

# 抽出の評価: 文書ごとに（文字列, 種類）の集合を比べる。LLM で抜き出した固有表現や、（主語, 関係, 目的語）の 3 つ組でも同じ
gold_docs = [{("山田", "PER"), ("京都", "LOC"), ("ソニー", "ORG")},
             {("田中", "PER"), ("大阪", "LOC")},
             {("トヨタ", "ORG")},
             set(),                                     # 何も含まない文書
             set()]
pred_docs = [{("山田", "PER"), ("京都", "ORG"), ("東京", "LOC")},
             {("田中", "PER"), ("大阪", "LOC")},
             set(),
             {("日本", "LOC")},
             set()]                                     # 正解も予測も空

def set_prf(gold_docs, pred_docs, average="micro", zero_division=0):
    """各文書の集合を 0/1 の行列にし、「要素ごとの 2 値分類」として集計する。micro は全文書の TP・FP・FN を合算する"""
    mlb = MultiLabelBinarizer().fit(gold_docs + pred_docs)
    p, r, f, _ = precision_recall_fscore_support(mlb.transform(gold_docs), mlb.transform(pred_docs),
                                                 average=average, zero_division=zero_division)
    return round(p, 3), round(r, 3), round(f, 3)

print("完全一致 (micro)         :", set_prf(gold_docs, pred_docs))
# 文書ごとの F1 の平均。正解も予測も空の文書を 0 点 (zero_division=0) と 1 点 (=1) のどちらにするかで値が変わる
# （片方だけ空の文書は F1 が 0 のままで、zero_division で変わるのは P・R だけ）
print("文書ごとの平均 (samples, 空は 0):", set_prf(gold_docs, pred_docs, average="samples"))
print("文書ごとの平均 (samples, 空は 1):", set_prf(gold_docs, pred_docs, average="samples", zero_division=1))
# 種類を無視して文字列だけで比べる（緩い一致）
print("種類を無視 (micro)       :", set_prf([{s for s, _ in d} for d in gold_docs],
                                          [{s for s, _ in d} for d in pred_docs]))
# 種類ごとの内訳: その種類の要素だけを残して集計する
for t in ["PER", "LOC", "ORG"]:
    print(f"  {t}:", set_prf([{e for e in d if e[1] == t} for d in gold_docs],
                             [{e for e in d if e[1] == t} for d in pred_docs]))
```

- 単一ラベルの多クラス分類で全クラスを集計に入れると、micro 平均の P・R・F1 はどれも正解率と一致する（classification_report が micro の行の代わりに accuracy の行を出すのはこのため）。「その他」のような背景クラスを `labels=` で除くと一致しなくなり、この例では micro-F1 が 0.841 で、正解率 0.822 とは違う値になった。どのクラスを集計に入れたかを必ず書く。
- macro 平均は件数によらず各クラスを同じ重さで平均するので、件数の少ないクラスの失敗が見える。この例では「個人情報」（3 件）は TP が 0、FN が 3 で F1 が 0 になり、その他を除いた macro-F1 は 0.622、micro-F1 は 0.841 だった。weighted 平均 (0.821) は件数で重み付けするので件数の多いクラスに引っ張られ、少数クラスの失敗が隠れやすい。
- 一度も予測されなかったクラスは適合率が 0 / 0 になる。`zero_division` を指定しないと UndefinedMetricWarning を出して 0 として扱い、`zero_division=np.nan` にするとそのクラスを平均から除く。この例では macro 適合率が 0.698 から 0.931 に上がった（再現率は 0 と定まるので F1 は変わらない）。件数 (support) と、どちらにしたかを一緒に報告する。
- 「macro-F1」には、クラスごとの F1 の平均（scikit-learn の `average="macro"`）と、macro 適合率と macro 再現率の調和平均の 2 つの流儀があり、この例でも 0.622 と 0.629 で値が違う。どちらを使ったかを書き、比べる相手の論文や評価スクリプトがどちらかを確かめる。
- 抽出の評価は、要素ごとの 2 値分類とみなすと `MultiLabelBinarizer` と `precision_recall_fscore_support` で集計できる。`average="micro"` は全文書の TP・FP・FN を合算する集計（この例で 0.5）で、`average="samples"` は文書ごとの F1 の平均である。後者は、正解も予測も空の文書の F1 を 0 点にするか 1 点にするか (`zero_division`) で変わり、この例では 0.267 と 0.467 になった（片方だけ空の文書は F1 が 0 のままで、変わるのは P・R だけ）。空の文書が多いデータでは micro を主に使う。集合にすると同じ表現が 2 回出てきても 1 つに数えるので、出現ごとに評価するなら文字位置を要素に含める。BIO タグ列の評価は、固有表現抽出の章の「固有表現抽出をエンティティ単位で評価する (seqeval)」を使う。
- 実データでの分類の評価と混同行列は、テキスト分類の章の「分類結果を評価する（適合率・再現率・F1・混同行列）」を参照。評価値の不確かさは、この章の「ブートストラップ法で評価値の信頼区間を求める」で信頼区間を付けて報告する。
- 仕組みを自分で実装して確かめる版は、付録「仕組みを理解するための実装」の「適合率・再現率・F1 を自分で計算する（micro と macro）」にある。

## モデルの確信度が当てになるか調べる（較正、ECE） {calibration_ece}

モデルが「確率 90%」と言った予測は、本当に 90% 当たっているのか。確信度の信頼性を評価し、補正したい。

```python
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB

cats = ["rec.sport.baseball", "rec.sport.hockey", "sci.space", "sci.med", "comp.graphics"]
data = fetch_20newsgroups(subset="all", categories=cats, remove=("headers", "footers", "quotes"))
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.4, stratify=data.target, random_state=0)
vec = TfidfVectorizer(stop_words="english", min_df=2)
Xtr, Xte = vec.fit_transform(X_train), vec.transform(X_test)

def ece(probs, labels, n_bins=10):
    """Expected Calibration Error: 確信度で区間に分け、|正解率 − 平均確信度| を事例数で重み付けして平均する"""
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == labels[m]).mean() - conf[m].mean())
    return total

models = {"ナイーブベイズ": MultinomialNB(alpha=0.01), "SVM + 較正 (Platt)": CalibratedClassifierCV(LinearSVC(), method="sigmoid", cv=5)}
probs = {}
for name, model in models.items():
    probs[name] = model.fit(Xtr, y_train).predict_proba(Xte)
    print(f"{name:20} 正解率 {(probs[name].argmax(1) == y_test).mean():.3f}  平均確信度 {probs[name].max(1).mean():.3f}  ECE {ece(probs[name], y_test):.3f}")
```

```python
import matplotlib.pyplot as plt

plt.plot([0, 1], [0, 1], "k--", label="perfect")
for (name, p), label in zip(probs.items(), ["Naive Bayes", "SVM + Platt scaling"]):
    conf, correct = p.max(axis=1), p.argmax(axis=1) == y_test
    bins = np.linspace(0.2, 1, 9)
    idx = np.digitize(conf, bins)
    xs = [conf[idx == i].mean() for i in range(1, len(bins)) if (idx == i).any()]
    ys = [correct[idx == i].mean() for i in range(1, len(bins)) if (idx == i).any()]
    plt.plot(xs, ys, "o-", label=label)
plt.xlabel("confidence"); plt.ylabel("accuracy"); plt.legend(); plt.show()
```

- 対角線より下は自信過剰（確信度のほうが正解率より高い）、上は自信不足。この例では、較正した SVM のほうが自信不足気味で ECE が大きい。較正の良し悪しはモデルとデータの組み合わせ次第なので、思い込みで決めずに測る。
- ニューラルモデルの較正には、検証データでソフトマックスの温度を 1 つだけ調整する temperature scaling が簡単で効果的。
- 正解率が同じでも、較正の良し悪しは違いうる。確信度でしきい値を設けて人手に回すような運用では、較正が重要になる。

## 音声認識や OCR の誤り率 (WER / CER) を計算する {wer_cer}

音声認識や OCR の出力が、正解とどれくらい違うかを単語誤り率・文字誤り率で測りたい。

```python
!pip install -q jiwer
```

```python
import jiwer

ref = ["the quick brown fox jumps over the lazy dog", "natural language processing is fun"]
hyp = ["the quick brown fox jumped over lazy dog", "natural language processing fun is"]
out = jiwer.process_words(ref, hyp)
print(f"WER = {out.wer:.3f}  (置換 {out.substitutions} / 削除 {out.deletions} / 挿入 {out.insertions} / 正解 {out.hits})")
print(jiwer.visualize_alignment(out))
```

```python
# 日本語は単語境界がないので、文字誤り率 (CER) を使うのが一般的
import unicodedata

def normalize(s):
    s = unicodedata.normalize("NFKC", s)
    return "".join(c for c in s if not unicodedata.category(c).startswith(("P", "Z")))   # 句読点と空白を除く

ref_ja = ["今日は良い天気ですね。", "自然言語処理の研究をしています。"]
hyp_ja = ["今日はいい天気ですね", "自然原語処理の研究をしてます。"]
print("CER (そのまま):", round(jiwer.cer(ref_ja, hyp_ja), 3))
print("CER (正規化後):", round(jiwer.cer([normalize(s) for s in ref_ja], [normalize(s) for s in hyp_ja]), 3))
```

- WER =（置換 + 削除 + 挿入）÷ 参照の単語数。挿入が多いと 100% を超えることもある。
- 「良い / いい」「3 / 三」のような表記の違いも誤りに数えられる。評価前の正規化（句読点、全角半角、数字の表記）の方針を決めて明記する。

## BERTScore で意味的な近さを評価する {bertscore}

表現が違っても意味が同じなら高く評価したい。n-gram の一致ではなく、文脈化埋め込みの類似度で測る。

```python
!pip install -q bert-score
```

```python
from bert_score import score

refs = ["猫はマットの上に座っています。", "彼は昨日、東京に行きました。", "この本は非常に面白い。", "この本は非常に面白い。"]
hyps = ["猫がマットの上に座っている。", "昨日、彼は東京へ出かけた。", "この書籍はとても興味深い。", "この本は非常に面白くない。"]

P, R, F = score(hyps, refs, lang="ja", verbose=False)   # lang="ja" では bert-base-multilingual-cased が使われる
for h, r, f in zip(hyps, refs, F.tolist()):
    print(f"{f:.3f}  {h}  /  {r}")
print("平均 F1:", F.mean().item())
```

```python
# 無関係な文同士でも 0.6 前後になる (値の範囲が狭い) ことに注意。設定を表すハッシュも取得できる
(P, R, F), setting = score(["今日は天気が良い。"], ["会議は午後 3 時に始まります。"], lang="ja", verbose=False, return_hash=True)
print(round(F.item(), 3), setting)
```

- 言い換え（「本 / 書籍」「面白い / 興味深い」）にも高いスコアが付く。一方で、否定のような小さいが決定的な違い（4 つ目の例）には鈍感。
- 値はモデルと層に依存する。報告するときはハッシュを添える。英語などでは `rescale_with_baseline=True` で値の範囲を広げて読みやすくできる。
- 翻訳評価では、人手評価で学習した COMET などのほうが人間の判断との相関が高い。

## 生成テキストの多様性と繰り返しを測る (distinct-n) {distinct_n}

対話システムや生成モデルの出力が、単調で似た文ばかりになっていないか、同じ語句を繰り返していないかを数値化したい。

```python
!pip install -q fugashi unidic-lite
```

```python
from collections import Counter
from fugashi import Tagger

tagger = Tagger("-Owakati")
tokenize = lambda s: tagger.parse(s).split()

def ngrams(tokens, n):
    return list(zip(*[tokens[i:] for i in range(n)]))

def distinct_n(texts, n):
    """出力全体での、異なり n-gram 数 ÷ 延べ n-gram 数"""
    all_ngrams = [g for t in texts for g in ngrams(tokenize(t), n)]
    return len(set(all_ngrams)) / max(1, len(all_ngrams))

def repetition_rate(text, n=3):
    """1 つの出力の中で、繰り返し現れた n-gram の割合"""
    counts = Counter(ngrams(tokenize(text), n))
    total = sum(counts.values())
    return sum(c for c in counts.values() if c > 1) / max(1, total)

bland = ["そうですね。", "それはいいですね。", "そうですね、いいと思います。", "わかりません。", "そうですね。"]
diverse = ["京都なら秋の東福寺がおすすめです。", "私は辛い料理が苦手なんです。", "最近は古い映画ばかり観ています。", "猫を二匹飼っています。", "明日は雨らしいので傘を忘れずに。"]
for name, texts in [("無難な応答", bland), ("多様な応答", diverse)]:
    print(f"{name}: distinct-1 = {distinct_n(texts, 1):.3f}  distinct-2 = {distinct_n(texts, 2):.3f}")

loop = "私は猫が好きです。私は猫が好きです。私は猫が好きです。なぜなら猫はかわいいからです。"
print("繰り返し率:", round(repetition_rate(loop), 3), "/", round(repetition_rate(diverse[0]), 3))
```

- distinct-n は出力の総量が増えると下がるので、比較するシステム間で出力数と長さを揃える。
- 多様性だけ高くても意味がない（ランダムな文字列が最も多様）。品質の指標と必ず組み合わせて見る。
- 複数の出力同士の類似度を測る Self-BLEU も、多様性の指標としてよく使われる（低いほど多様）。

## ブートストラップ法で評価値の信頼区間を求める {bootstrap_ci}

テストセットで正解率 85% だったが、その値がどれくらい不確かなのか、信頼区間を付けて報告したい。

```python
import numpy as np

rng = np.random.default_rng(0)
n = 500
gold = rng.integers(0, 2, n)
pred = np.where(rng.random(n) < 0.85, gold, 1 - gold)      # 正解率が約 85% の予測を人工的に作る

def bootstrap_ci(metric, gold, pred, n_boot=2000, alpha=0.05):
    """テスト事例を復元抽出して指標を計算し直すことを繰り返し、分布の両端を信頼区間とする"""
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(gold), len(gold))
        stats.append(metric(gold[idx], pred[idx]))
    return metric(gold, pred), np.percentile(stats, 100 * alpha / 2), np.percentile(stats, 100 * (1 - alpha / 2))

accuracy = lambda g, p: (g == p).mean()
point, lo, hi = bootstrap_ci(accuracy, gold, pred)
print(f"正解率 = {point:.3f}  95% 信頼区間 [{lo:.3f}, {hi:.3f}]")
```

```python
# F1 のように単純な平均でない指標でも、同じ手順で求められる
from sklearn.metrics import f1_score

point, lo, hi = bootstrap_ci(lambda g, p: f1_score(g, p), gold, pred)
print(f"F1 = {point:.3f}  95% 信頼区間 [{lo:.3f}, {hi:.3f}]")

# テストセットの大きさと信頼区間の幅
for size in [100, 500, 2000, 10000]:
    g = rng.integers(0, 2, size); p = np.where(rng.random(size) < 0.85, g, 1 - g)
    _, lo, hi = bootstrap_ci(accuracy, g, p, n_boot=500)
    print(f"n = {size:5}: 幅 ±{(hi - lo) / 2 * 100:.1f} ポイント")
```

- 500 件のテストセットでは、正解率に ±3 ポイント程度の不確かさがある。1 ポイントの差を議論するには数千件が必要。
- BLEU のようなコーパス単位の指標も、文をリサンプリングして計算し直せば同じように扱える（`sacrebleu` にも実装がある）。
- これは「テストセットの選び方」による不確かさ。学習の乱数シードによるばらつきは別に評価する必要がある。

## 2 つのシステムの差が有意かどうかを検定する {significance_test}

システム A が B より 1 ポイント高かった。この差は偶然なのか、本当に A のほうが良いと言えるのかを確かめたい。

```python
import numpy as np

rng = np.random.default_rng(0)
n = 1000
gold = rng.integers(0, 2, n)
difficulty = rng.random(n)                                  # 事例ごとの難しさ (両システムに共通)
pred_a = np.where(difficulty + rng.normal(0, 0.15, n) < 0.86, gold, 1 - gold)
pred_b = np.where(difficulty + rng.normal(0, 0.15, n) < 0.83, gold, 1 - gold)
correct_a, correct_b = pred_a == gold, pred_b == gold
print(f"A = {correct_a.mean():.3f}  B = {correct_b.mean():.3f}  差 = {correct_a.mean() - correct_b.mean():+.3f}")
```

```python
def paired_bootstrap(a, b, n_boot=10000):
    """対応のあるブートストラップ検定: 同じ事例を両システムについてリサンプリングし、A が B を上回らない割合を p 値とする"""
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return (diffs <= 0).mean()

def approximate_randomization(a, b, n_perm=10000):
    """近似並べ替え検定: 事例ごとに A と B の結果をランダムに入れ替え、観測された差以上の差が出る割合を p 値とする (両側)"""
    observed = abs(a.mean() - b.mean())
    swap = rng.random((n_perm, len(a))) < 0.5
    diffs = np.abs(np.where(swap, b, a).mean(axis=1) - np.where(swap, a, b).mean(axis=1))
    return (diffs >= observed).mean()

print("対応のあるブートストラップ (片側) p =", paired_bootstrap(correct_a, correct_b))
print("近似並べ替え検定 (両側) p =", approximate_randomization(correct_a, correct_b))
```

```python
# 分類の正誤なら、McNemar 検定が使える (片方だけが正解した事例の数だけを見る)
from statsmodels.stats.contingency_tables import mcnemar

table = [[(correct_a & correct_b).sum(), (correct_a & ~correct_b).sum()],
         [(~correct_a & correct_b).sum(), (~correct_a & ~correct_b).sum()]]
print("A だけ正解:", table[0][1], " B だけ正解:", table[1][0], " McNemar p =", round(mcnemar(table, exact=True).pvalue, 4))
```

- 同じテストセットで 2 つのシステムを比べるときは、事例ごとの対応を使う「対応のある」検定にする。対応を無視すると検出力が大きく落ちる。
- p < 0.05 は「差がない場合にこの結果が出る確率が 5% 未満」という意味で、差の大きさや重要さは示さない。効果量と信頼区間も併せて報告する。
- たくさんのシステムや設定を比べると、偶然「有意」になるものが出てくる（多重比較）。Bonferroni 補正などを検討する。

## 乱数シードによるばらつきを考慮して結果を報告する {seed_variance}

ファインチューニングは乱数シードで結果が変わる。1 回の実行の数字ではなく、複数回の平均とばらつきで手法を比較したい。

```python
import numpy as np
from scipy import stats

# 5 つのシードで学習したときのテスト F1 (例)
baseline = np.array([84.2, 85.1, 83.6, 84.9, 84.4])
proposed = np.array([85.0, 85.9, 84.1, 86.2, 85.3])

for name, x in [("baseline", baseline), ("proposed", proposed)]:
    ci = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
    print(f"{name}: {x.mean():.2f} ± {x.std(ddof=1):.2f} (標準偏差)  95% 信頼区間 [{ci[0]:.2f}, {ci[1]:.2f}]  最小 {x.min()} 最大 {x.max()}")
```

```python
t, p = stats.ttest_ind(proposed, baseline, equal_var=False)       # Welch の t 検定
u, p_u = stats.mannwhitneyu(proposed, baseline, alternative="two-sided")   # 正規性を仮定しないノンパラメトリック検定
d = (proposed.mean() - baseline.mean()) / np.sqrt((proposed.var(ddof=1) + baseline.var(ddof=1)) / 2)
print(f"差 = {proposed.mean() - baseline.mean():+.2f}  Welch の t 検定 p = {p:.3f}  Mann-Whitney p = {p_u:.3f}  効果量 d = {d:.2f}")

# やってはいけない比較: 提案手法の最良のシードとベースラインの 1 回の結果を比べる
print(f"最良同士の差: {proposed.max() - baseline.max():+.1f} / 提案の最良 vs ベースラインの最悪: {proposed.max() - baseline.min():+.1f}")
```

- 小さなデータセットでの BERT のファインチューニングでは、シードだけで数ポイント変わることがある。最低でも 3〜5 シードの平均と標準偏差を報告する。
- シード数が少ないと検定の検出力は低い。有意にならなくても「差がない」とは言えない。
- 両手法で同じシードを使ったからといって、対応のあるデータになるわけではない（初期値やデータ順が同じでも、手法が違えば別の乱数列になる）。

## アノテーションの一致度を測る（κ 係数、Krippendorff の α） {agreement}

複数人が付けたラベルがどれくらい一致しているかを、偶然の一致を差し引いて評価したい。

```python
!pip install -q krippendorff statsmodels
```

```python
import numpy as np
from sklearn.metrics import cohen_kappa_score

a = ["pos", "pos", "neg", "neu", "pos", "neg", "neg", "pos", "neu", "pos", "neg", "pos"]
b = ["pos", "neu", "neg", "neu", "pos", "neg", "pos", "pos", "neg", "pos", "neg", "pos"]
print("単純一致率:", round(float(np.mean([x == y for x, y in zip(a, b)])), 3))
print("Cohen の κ:", round(cohen_kappa_score(a, b), 3))

# 順序のあるラベル (1〜5 の評点など) は、近い食い違いを軽く扱う重み付き κ を使う
r1, r2 = [1, 2, 3, 4, 5, 3, 2, 4, 5, 1], [1, 3, 3, 5, 5, 2, 2, 4, 4, 2]
print("重みなし κ:", round(cohen_kappa_score(r1, r2), 3), " 2 次重み付き κ:", round(cohen_kappa_score(r1, r2, weights="quadratic"), 3))
```

```python
# 3 人以上: Fleiss の κ
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

ratings = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1], [2, 2, 1], [0, 0, 0], [1, 1, 2], [2, 2, 2], [0, 0, 1]])   # 行 = 事例、列 = 評定者
table, _ = aggregate_raters(ratings)
print("Fleiss の κ:", round(fleiss_kappa(table), 3))
```

```python
# 欠損があってもよく、名義・順序・間隔尺度に対応する Krippendorff の α
import krippendorff

data = np.array([[1, 2, 3, 3, 2, 1, 4, 1, 2, np.nan],      # 行 = 評定者、列 = 事例 (nan = 未評定)
                 [1, 2, 3, 3, 2, 2, 4, 1, 2, 5],
                 [np.nan, 3, 3, 3, 2, 3, 4, 2, 2, 5]])
for level in ["nominal", "ordinal", "interval"]:
    print(f"α ({level}):", round(krippendorff.alpha(reliability_data=data, level_of_measurement=level), 3))
```

- κ の目安 (Landis & Koch): 0.41〜0.60 中程度、0.61〜0.80 かなりの一致、0.81 以上ほぼ完全。ただし絶対的な基準ではない。
- κ はラベルの分布が偏っていると、一致率が高くても低く出る。混同行列も併せて見る。
- 一致度が低いときは、アノテータではなくガイドラインに問題があることが多い。食い違った事例を見て、基準を明確にする。

## 自動評価指標と人手評価の相関を調べる {metric_correlation}

提案した自動評価指標や LLM による評価が、人間の判断とどれくらい一致しているかを、相関係数で示したい。

```python
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

rng = np.random.default_rng(0)
human = rng.integers(1, 6, 200).astype(float)                    # 人手評価 (1〜5)
metric_a = human * 15 + rng.normal(0, 12, 200)                   # 人手評価とよく合う指標
metric_b = np.exp(human) + rng.normal(0, 15, 200)                # 単調だが非線形な関係の指標
metric_c = rng.normal(50, 15, 200)                               # 無関係な指標

print(f"{'':10}{'Pearson':>9}{'Spearman':>10}{'Kendall':>9}")
for name, m in [("metric A", metric_a), ("metric B", metric_b), ("metric C", metric_c)]:
    print(f"{name:10}{pearsonr(human, m).statistic:9.3f}{spearmanr(human, m).statistic:10.3f}{kendalltau(human, m).statistic:9.3f}")
```

```python
# システム単位の相関: 各システムの平均スコア同士を比べる (文単位より高く出やすい)
n_sys, n_sent = 8, 100
quality = np.linspace(2.5, 4.0, n_sys)                            # システムの真の品質
human_s = quality[:, None] + rng.normal(0, 1.0, (n_sys, n_sent))
metric_s = quality[:, None] * 10 + rng.normal(0, 12, (n_sys, n_sent))
print("文単位:", round(pearsonr(human_s.ravel(), metric_s.ravel()).statistic, 3),
      " システム単位:", round(pearsonr(human_s.mean(1), metric_s.mean(1)).statistic, 3))

# 相関係数の信頼区間 (ブートストラップ)
idx = rng.integers(0, 200, (2000, 200))
boots = [spearmanr(human[i], metric_a[i]).statistic for i in idx]
print("Spearman の 95% 信頼区間:", np.percentile(boots, [2.5, 97.5]).round(3))
```

- Pearson は線形な関係の強さ、Spearman と Kendall は順位の一致を測る。非線形だが単調な関係 (metric B) は Pearson だと過小評価される。
- 同点の多い離散的な評価（1〜5 の評点）には Kendall の τ がよく使われる。
- システム単位の相関は、システム数が少ないと不安定。数個のシステムでの高い相関は当てにならない。
- 相関係数の信頼区間は、評価の組を復元抽出して求めている。考え方はこの章の「ブートストラップ法で評価値の信頼区間を求める」と同じ。
- LLM による評価（LLM-as-a-judge）は、LLM の評価と分析の章の「手元の LLM に生成文を採点させる (LLM-as-a-judge)」で扱う。そこでも、この相関で採点の信頼性を確かめている。
