# 評価指標と統計的検定

翻訳・要約・検索・分類の出力を数値で評価し、その差が偶然でないかを確かめる。

## 機械翻訳を BLEU で評価する (sacreBLEU) {bleu}

翻訳システムの出力を、参照訳との n-gram の一致度 (BLEU) で評価したい。日本語の出力も正しく評価したい。

```python
!pip install -q "sacrebleu[ja]"
```

```python
from sacrebleu.metrics import BLEU

hyps = ["猫がマットの上に座っている。", "彼は昨日東京へ行った。", "この本はとても面白いです。"]
refs = ["猫はマットの上に座っています。", "彼は昨日、東京に行きました。", "この本は非常に面白い。"]

bleu = BLEU(tokenize="ja-mecab")           # 日本語は MeCab で分割してから n-gram を数える
result = bleu.corpus_score(hyps, [refs])   # 参照は「参照セットのリスト」で渡す (複数参照に対応するため)
print(result)
print(bleu.get_signature())                # 論文にはこの署名を載せる
```

```python
# トークナイズの違いでスコアは大きく変わる
for tok in ["ja-mecab", "char", "13a"]:
    print(f"{tok:10} {BLEU(tokenize=tok).corpus_score(hyps, [refs]).score:.2f}")

# 英語 (デフォルトの 13a トークナイザ)。参照訳が 2 セットある場合
hyps_en = ["The cat sat on the mat.", "He went to Tokyo yesterday."]
refs_en = [["The cat is sitting on the mat.", "He went to Tokyo yesterday."], ["A cat sat on the mat.", "Yesterday he traveled to Tokyo."]]
print(BLEU().corpus_score(hyps_en, refs_en))
```

```python
# 文単位の BLEU (平滑化が必要。コーパス単位の値とは性質が違う)
sent_bleu = BLEU(tokenize="ja-mecab", effective_order=True)
for h, r in zip(hyps, refs):
    print(f"{sent_bleu.sentence_score(h, [r]).score:5.1f}  {h}")
```

- BLEU は、モデルの出力をデトークナイズした「普通の文」に対して計算する。自前で分かち書きした文字列を渡してはいけない。
- 日本語を既定の `13a` で評価すると、文全体が 1〜数トークンになり、ほぼ意味のない値になる。必ず `ja-mecab` を指定する。
- BLEU はコーパス全体の n-gram を集計する指標。文ごとの BLEU の平均とは一致しない。
- 異なるトークナイズ・異なるテストセットの BLEU 同士は比較できない。署名を必ず報告する。

## chrF と TER で翻訳を評価する {chrf_ter}

BLEU 以外の自動評価指標も併せて報告したい。単語分割に依存しない指標が欲しい。

```python
!pip install -q "sacrebleu[ja]"
```

```python
from sacrebleu.metrics import BLEU, CHRF, TER

hyps = ["猫がマットの上に座っている。", "彼は昨日東京へ行った。", "この本はとても面白いです。"]
refs = ["猫はマットの上に座っています。", "彼は昨日、東京に行きました。", "この本は非常に面白い。"]

chrf = CHRF()                 # 文字 n-gram の F スコア (既定は 6-gram まで、β=2)
chrfpp = CHRF(word_order=2)   # chrF++: 単語 1-gram, 2-gram も加える
ter = TER(asian_support=True, normalized=True)   # 翻訳編集率: 参照に直すための編集回数 (低いほど良い)

print(chrf.corpus_score(hyps, [refs]), "|", chrf.get_signature())
print(chrfpp.corpus_score(hyps, [refs]))
print(ter.corpus_score(hyps, [refs]))
```

```python
# 形態が豊かな言語や表記ゆれでは、chrF のほうが BLEU より人手評価との相関が高い傾向がある
hyp_en, ref_en = ["He goes to schools every days."], ["He goes to school every day."]
print(BLEU().corpus_score(hyp_en, [ref_en]).score, CHRF().corpus_score(hyp_en, [ref_en]).score)
```

- chrF は文字単位なので、日本語でも単語分割の方法に左右されない。語尾だけ違う語にも部分点が入る。
- いずれも表層の一致を見る指標で、言い換えは評価できない。意味を見るには BERTScore や COMET などの学習ベースの指標を併用する。

## 要約を ROUGE で評価する (日本語対応) {rouge_ja}

要約の出力を、参照要約との単語の重なり (ROUGE-1/2/L) で評価したい。日本語でも正しく計算したい。

```python
!pip install -q rouge-score fugashi unidic-lite
```

```python
from fugashi import Tagger
from rouge_score import rouge_scorer

class MecabTokenizer:
    """rouge-score の既定のトークナイザは英数字以外を捨ててしまうので、日本語用のものを渡す"""
    def __init__(self):
        self.tagger = Tagger("-Owakati")
    def tokenize(self, text):
        return self.tagger.parse(text).split()

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=MecabTokenizer())
ref = "政府は来年度予算案を閣議決定し、一般会計の総額は過去最大となった。"
for hyp in ["政府は来年度予算案を決定し、総額は過去最大となった。", "来年度の予算は過去最大の規模である。", "今日は天気が良い。"]:
    s = scorer.score(ref, hyp)    # 引数の順番は (参照, 出力)
    print(f"R1 {s['rouge1'].fmeasure:.3f}  R2 {s['rouge2'].fmeasure:.3f}  RL {s['rougeL'].fmeasure:.3f}  {hyp}")
```

```python
# 既定のトークナイザに日本語を渡すと、何も残らず 0 になる
default = rouge_scorer.RougeScorer(["rouge1"])
print(default.score(ref, "政府は来年度予算案を決定し、総額は過去最大となった。"))

# コーパス全体では、文書ごとの F 値を平均して報告するのが一般的
import numpy as np
refs = [ref, "新しい言語モデルが発表され、性能が大幅に向上した。"]
hyps = ["政府は予算案を決定した。", "新しいモデルが発表された。"]
print(round(float(np.mean([scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)])), 3))
```

- ROUGE-1/2 は単語 1-gram / 2-gram の重なり、ROUGE-L は最長共通部分列。再現率・適合率・F 値のどれを報告しているかを明記する。
- 日本語で「ROUGE が 0 になる」「異常に高い/低い」ときは、ほぼトークナイザの問題。

## BERTScore で意味的な近さを評価する {bertscore}

表現が違っても意味が同じなら高く評価したい。n-gram の一致ではなく、文脈化埋め込みの類似度で測る。

```python
!pip install -q bert-score
```

```python
from bert_score import score

refs = ["猫はマットの上に座っています。", "彼は昨日、東京に行きました。", "この本は非常に面白い。", "会議は午後 3 時に始まります。"]
hyps = ["猫がマットの上に座っている。", "昨日、彼は東京へ出かけた。", "この書籍はとても興味深い。", "この本は非常に面白くない。"]

P, R, F = score(hyps, refs, lang="ja", verbose=False)   # lang="ja" では bert-base-multilingual-cased が使われる
for h, r, f in zip(hyps, refs, F.tolist()):
    print(f"{f:.3f}  {h}  /  {r}")
print("平均 F1:", F.mean().item())
```

```python
# スコアが 0.6〜1.0 の狭い範囲に固まって読みにくいので、ベースラインで再スケールする
P, R, F = score(hyps, refs, lang="ja", rescale_with_baseline=True, verbose=False)
print([round(f, 3) for f in F.tolist()])
```

- 言い換え (「本 / 書籍」「面白い / 興味深い」) にも高いスコアが付く。一方で、否定のような小さいが決定的な違い (4 つ目の例) には鈍感。
- 値はモデルと層に依存する。報告するときは `bert_score` のハッシュ (`return_hash=True`) を添える。
- 翻訳評価では、人手評価で学習した COMET などのほうが人間の判断との相関が高い。

## COMET で翻訳の品質を評価する {comet gpu norun}

機械翻訳の評価で現在標準的に使われている、人手評価に基づいて学習されたニューラル評価指標を使いたい。

```python
!pip install -q unbabel-comet
```

```python
from comet import download_model, load_from_checkpoint

model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))   # 原文・訳文・参照訳を使うモデル (約 2.3GB)

data = [
    {"src": "吾輩は猫である。名前はまだ無い。", "mt": "I am a cat. I don't have a name yet.", "ref": "I am a cat. As yet I have no name."},
    {"src": "吾輩は猫である。名前はまだ無い。", "mt": "I am a dog. My name is not here.", "ref": "I am a cat. As yet I have no name."},
    {"src": "彼は昨日東京に行った。", "mt": "He went to Tokyo yesterday.", "ref": "Yesterday he went to Tokyo."},
]
out = model.predict(data, batch_size=8, gpus=1)
for d, s in zip(data, out.scores):
    print(f"{s:.3f}  {d['mt']}")
print("システムスコア:", round(out.system_score, 4))
```

```python
# 参照訳なしで評価する品質推定 (QE) モデル。利用規約への同意と Hugging Face へのログインが必要
# from huggingface_hub import login; login()
# qe = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))
# print(qe.predict([{"src": "彼は昨日東京に行った。", "mt": "He went to Tokyo yesterday."}], batch_size=8, gpus=1).scores)
```

- BLEU と違い、原文も入力に使う。意味が合っていれば表現が違っても高く評価され、人手評価との相関が高い。
- スコアはモデルごとに尺度が違う。異なる COMET モデルの値を比べてはいけない。モデル名を必ず報告する。
- Colab では、インストール時に依存ライブラリのバージョンが入れ替わることがある。エラーが出たらランタイムを再起動する。

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

- WER = (置換 + 削除 + 挿入) ÷ 参照の単語数。挿入が多いと 100% を超えることもある。
- 「良い / いい」「3 / 三」のような表記の違いも誤りに数えられる。評価前の正規化 (句読点、全角半角、数字の表記) の方針を決めて明記する。

## 適合率・再現率・F1 を自分で計算する (micro と macro) {prf_manual}

ライブラリに頼らず、適合率・再現率・F1 を定義から計算して、micro 平均と macro 平均の違いを理解したい。

```python
from collections import Counter

gold = ["A", "A", "A", "A", "A", "A", "B", "B", "B", "C"]
pred = ["A", "A", "A", "A", "A", "B", "B", "B", "A", "A"]

labels = sorted(set(gold) | set(pred))
tp, fp, fn = Counter(), Counter(), Counter()
for g, p in zip(gold, pred):
    if g == p:
        tp[g] += 1
    else:
        fp[p] += 1     # p と予測したが違った
        fn[g] += 1     # g を見逃した

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f

per_class = {}
for c in labels:
    per_class[c] = prf(tp[c], fp[c], fn[c])
    print(f"{c}: P = {per_class[c][0]:.3f}  R = {per_class[c][1]:.3f}  F1 = {per_class[c][2]:.3f}  (TP {tp[c]}, FP {fp[c]}, FN {fn[c]})")

macro_f1 = sum(f for _, _, f in per_class.values()) / len(labels)        # クラスごとの F1 の単純平均
micro = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))         # 全クラスの TP/FP/FN を合算してから計算
print(f"macro-F1 = {macro_f1:.3f}   micro-F1 = {micro[2]:.3f}   正解率 = {sum(tp.values()) / len(gold):.3f}")
```

```python
from sklearn.metrics import precision_recall_fscore_support

p, r, f, _ = precision_recall_fscore_support(gold, pred, average="macro", zero_division=0)
print("scikit-learn の macro:", round(p, 3), round(r, 3), round(f, 3))

# 抽出タスク (固有表現抽出など、正解と予測が「集合」の場合) の F1
gold_set, pred_set = {("山田", "PER"), ("京都", "LOC"), ("ソニー", "ORG")}, {("山田", "PER"), ("京都", "ORG"), ("東京", "LOC")}
n = len(gold_set & pred_set)
print("抽出の P/R/F1:", [round(x, 3) for x in prf(n, len(pred_set) - n, len(gold_set) - n)])
```

- 単一ラベルの多クラス分類では、micro-F1 は正解率と一致する。
- macro-F1 は少数クラス (ここでは一度も当たっていない C) の影響を強く受ける。クラスの偏りがあるときに、モデルの弱点が見える。
- 「macro-F1」には「クラスごとの F1 の平均」と「macro 適合率と macro 再現率の調和平均」の 2 つの流儀があり、値が違う。前者が一般的。

## アノテーションの一致度を測る (κ 係数、Krippendorff の α) {agreement}

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
- BLEU のようなコーパス単位の指標も、文をリサンプリングして計算し直せば同じように扱える (`sacrebleu` にも実装がある)。
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
- たくさんのシステムや設定を比べると、偶然「有意」になるものが出てくる (多重比較)。Bonferroni 補正などを検討する。

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
- 両手法で同じシードを使ったからといって、対応のあるデータになるわけではない (初期値やデータ順が同じでも、手法が違えば別の乱数列になる)。

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
- 同点の多い離散的な評価 (1〜5 の評点) には Kendall の τ がよく使われる。
- システム単位の相関は、システム数が少ないと不安定。数個のシステムでの高い相関は当てにならない。

## 質問応答を完全一致と F1 で評価する {qa_em_f1}

質問応答の出力を、正解と完全に一致したか (EM)、部分的にどれくらい重なっているか (F1) で評価したい。

```python
!pip install -q fugashi unidic-lite
```

```python
import re
import unicodedata
from collections import Counter
from fugashi import Tagger

tagger = Tagger("-Owakati")

def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s。、．，,.!?！？「」『』()（）]", "", text)

def exact_match(pred, golds):
    return float(any(normalize(pred) == normalize(g) for g in golds))

def token_f1(pred, golds, unit="char"):
    """正解が複数あるときは、最も高い F1 を採用する"""
    tok = (lambda s: list(normalize(s))) if unit == "char" else (lambda s: tagger.parse(s).split())
    best = 0.0
    for g in golds:
        p, t = tok(pred), tok(g)
        common = sum((Counter(p) & Counter(t)).values())
        if common:
            prec, rec = common / len(p), common / len(t)
            best = max(best, 2 * prec * rec / (prec + rec))
    return best

examples = [("富士山", ["富士山"]), ("富士山です。", ["富士山"]), ("3776 メートル", ["3776m", "3776メートル"]),
            ("１９４５年", ["1945年", "1945"]), ("夏目漱石と森鴎外", ["夏目漱石"]), ("わかりません", ["東京"])]
for pred, golds in examples:
    print(f"EM {exact_match(pred, golds):.0f}  F1(文字) {token_f1(pred, golds):.2f}  F1(単語) {token_f1(pred, golds, 'word'):.2f}  {pred!r} / {golds}")
```

- 英語の SQuAD では、冠詞と句読点を除いて小文字化し、空白区切りの単語で F1 を計算する。日本語 (JSQuAD など) では文字単位の F1 が使われる。
- LLM の出力は「富士山です。」のように文になりがちで、EM が不当に低くなる。プロンプトで形式を指定するか、「正解を含むか」での判定や LLM による判定を併用する。

## 検索・ランキングを評価する (Recall@k、MRR、nDCG) {ranking_metrics}

検索システムやリランカーが、正解の文書をどれくらい上位に出せているかを測りたい。

```python
import numpy as np

def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / len(relevant)

def reciprocal_rank(ranked, relevant):
    for i, doc in enumerate(ranked, 1):
        if doc in relevant:
            return 1 / i
    return 0.0

def average_precision(ranked, relevant):
    hits, total = 0, 0.0
    for i, doc in enumerate(ranked, 1):
        if doc in relevant:
            hits += 1
            total += hits / i            # 正解が出た位置での適合率
    return total / len(relevant)

def ndcg_at_k(ranked, gains, k):
    """gains: {文書 ID: 関連度 (0, 1, 2, ...)}。上位ほど大きい重みで関連度を足し、理想的な順位での値で割る"""
    dcg = sum(gains.get(doc, 0) / np.log2(i + 1) for i, doc in enumerate(ranked[:k], 1))
    ideal = sum(g / np.log2(i + 1) for i, g in enumerate(sorted(gains.values(), reverse=True)[:k], 1))
    return float(dcg / ideal) if ideal else 0.0

# クエリごとの (システムの出力順位, 関連度付きの正解)
runs = {
    "q1": (["d3", "d1", "d7", "d2", "d9"], {"d1": 2, "d2": 1}),
    "q2": (["d5", "d4", "d8", "d6", "d1"], {"d5": 2}),
    "q3": (["d2", "d9", "d4", "d3", "d6"], {"d7": 2, "d6": 1}),
}
rows = []
for q, (ranked, gains) in runs.items():
    rel = set(gains)
    rows.append([recall_at_k(ranked, rel, 1), recall_at_k(ranked, rel, 5), reciprocal_rank(ranked, rel), average_precision(ranked, rel), ndcg_at_k(ranked, gains, 5)])
    print(q, [round(x, 3) for x in rows[-1]])
print("平均: Recall@1 {:.3f}  Recall@5 {:.3f}  MRR {:.3f}  MAP {:.3f}  nDCG@5 {:.3f}".format(*np.mean(rows, axis=0)))
```

- RAG の検索段階なら Recall@k (正解が上位 k 件に入っているか)、最終的な順位付けなら MRR や nDCG を見る。
- MRR は最初の正解の順位だけを見る。正解が複数あって段階的な関連度が付いているなら nDCG。
- 標準的な実装は `pytrec_eval` や `ranx`。論文で報告するなら、それらを使うのが安全。

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
- 多様性だけ高くても意味がない (ランダムな文字列が最も多様)。品質の指標と必ず組み合わせて見る。
- 複数の出力同士の類似度を測る Self-BLEU も、多様性の指標としてよく使われる (低いほど多様)。

## モデルの確信度が当てになるか調べる (較正、ECE) {calibration_ece}

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

- 対角線より下は自信過剰 (確信度のほうが正解率より高い)、上は自信不足。この例では、較正した SVM のほうが自信不足気味で ECE が大きい。較正の良し悪しはモデルとデータの組み合わせ次第なので、思い込みで決めずに測る。
- ニューラルモデルの較正には、検証データでソフトマックスの温度を 1 つだけ調整する temperature scaling が簡単で効果的。
- 正解率が同じでも、較正の良し悪しは違いうる。確信度でしきい値を設けて人手に回すような運用では、較正が重要になる。
