# 感情分析と意見・有害表現の分析

レビューや SNS の投稿からポジティブ・ネガティブを判定する（学習済みモデルと評価極性辞書）、8 種類の感情の強さを推定する (WRIME)、観点ごとの極性を判定する（アスペクトベース感情分析）、有害な表現を検出する。一般的なテキスト分類の手法は「テキスト分類」の章を参照。評価極性分析の基礎と実践については、姉妹編『[自然言語処理の教科書](https://github.com/mamorlis/nlpbook)』の第 2 章（2-1 評価極性分析）も参照。

## 日本語の感情分析（ポジネガ判定）をする {sentiment_ja}

レビューや SNS の投稿がポジティブかネガティブかを、学習済みモデルですぐ判定したい。

```python
from transformers import pipeline

clf = pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
texts = ["この映画は本当に素晴らしかった。もう一度観たい。", "最悪の接客で、二度と行かないと決めた。", "明日は月曜日です。",
         "期待していたほどではなかったが、悪くはない。", "面白くないわけがない。", "すごいね、また遅刻したんだ。"]
for text, r in zip(texts, clf(texts)):
    print(f"{r['label']:9}{r['score']:.2f}  {text}")
```

```python
# 全クラスの確率を得る
for r in clf("値段は高いけれど、品質には満足しています。", top_k=None):
    print(r["label"], round(r["score"], 3))
```

- 二重否定、皮肉、対比（「〜だが…」）は誤りやすい。ドメインが違うと精度が落ちるので、自分のデータで少量でも評価する。評価極性辞書と規則で判定し、効いた語を示す方法は、この章の「評価極性辞書で感情を判定し、効いた語を示す（否定と逆接の規則）」を参照。
- ここで使ったのは多言語モデル（英語・日本語・中国語など 12 言語）。日本語専用なら `koheiduck/bert-japanese-finetuned-sentiment`、英語なら `cardiffnlp/twitter-roberta-base-sentiment-latest` など。
- 喜び・悲しみ・怒りなどの感情の種類まで知りたい場合は、この章の「喜び・悲しみ・怒りなど 8 種類の感情の強さを推定する (WRIME)」を、観点（味、価格など）ごとに判定したい場合は「観点ごとにポジティブ・ネガティブを判定する（アスペクトベース感情分析）」を参照。

## 評価極性辞書で感情を判定し、効いた語を示す（否定と逆接の規則） {sentiment_lexicon}

学習済みモデルの判定は根拠が見えにくい。公開の評価極性辞書で文中の語のポジ・ネガを数え、否定と逆接の規則を加えて判定し、どの語が効いたかを示したい。

```python
!pip install -q fugashi==1.5.2 unidic-lite==1.0.8
```

```python
import re, urllib.request

# 日本語評価極性辞書 ver.1.0（東北大学 乾・岡崎研究室）。クレジットを明記すれば商用利用も可能。
# 用言編は 小林ら (2005)「意見抽出のための評価表現の収集」、名詞編は 東山ら (2008)「述語の選択選好性に着目した名詞評価極性の獲得」を引用する
BASE = "https://www.cl.ecei.tohoku.ac.jp/resources/sent_lex/"
fetch = lambda name: urllib.request.urlopen(BASE + name).read().decode("utf-8").splitlines()

lexicon = {}
# 名詞編: 語 \t 極性のタグ \t 判定の観点。タグが p（ポジ）と n（ネガ）のものだけを使う（e の中立と、?p?n などの曖昧なものは除く）
for line in fetch("pn.csv.m3.120408.trim"):
    f = line.split("\t")
    if len(f) >= 2 and f[1] in ("p", "n"):
        lexicon[f[0]] = 1 if f[1] == "p" else -1
n_noun = len(lexicon)
for line in fetch("wago.121808.pn"):             # 用言編: 「ポジ（評価）」などのタグ \t 形態素を空白で区切った表現
    tag, _, expr = line.partition("\t")
    expr = re.sub(r" (だ|です|の|な|と|に)$", "", expr).replace(" ", "")   # 「至便 です」は「至便」、「使い 難い」は「使い難い」で引く
    if expr:
        lexicon.setdefault(expr, 1 if tag.startswith("ポジ") else -1)   # 名詞編と重なる表現は名詞編の極性を優先する
print(f"名詞 {n_noun} 語 + 用言 {len(lexicon) - n_noun} 語 = {len(lexicon)} 語")
print({w: lexicon.get(w) for w in ["満足", "最悪", "美味しい", "悪い", "使い難い", "本当", "値段", "高い"]})
```

```python
from fugashi import Tagger

tagger = Tagger()
ADVERSATIVE_PARTICLES = {"が", "けど", "けれど", "けれども", "のに", "ものの"}
ADVERSATIVE_CONJ = {"しかし", "でも", "だが", "だけど", "けれど", "けれども", "ところが", "ただ"}

def lookup(toks, i):
    """位置 i から始まる最長の辞書の語（4 形態素まで）。最後の形態素は活用した形のほか基本形でも引く"""
    if toks[i].feature.pos1 in ("助詞", "助動詞", "補助記号"):
        return None
    for n in (4, 3, 2, 1):
        if i + n <= len(toks):
            head, last = "".join(t.surface for t in toks[i:i + n - 1]), toks[i + n - 1]
            for form in (last.surface, last.feature.orthBase, last.feature.lemma):
                if form and head + form in lexicon:
                    return head + form, n
    return None

def is_negation(t):   # 「ない」「ぬ」「ず」と、「良くない」「問題がない」の「ない」（形容詞「無い」）
    return (t.feature.pos1 == "助動詞" and t.feature.lemma in ("ない", "ぬ", "ず")) or (t.feature.pos1 == "形容詞" and t.feature.lemma == "無い")

def is_adversative(toks, i):
    t = toks[i]
    if t.feature.pos1 == "接続詞":
        return t.surface in ADVERSATIVE_CONJ
    if t.feature.pos2 == "接続助詞":
        return t.surface in ADVERSATIVE_PARTICLES
    # unidic-lite は「美味しいが」の「が」を格助詞と解析することがあるので、用言の直後の「が」も逆接とみなす
    return t.surface == "が" and i > 0 and toks[i - 1].feature.pos1 in ("動詞", "形容詞", "助動詞")

def analyze(text, negation=True, adversative=True):
    toks, hits, i = list(tagger(text)), [], 0
    while i < len(toks):
        found = lookup(toks, i)
        if found:
            word, n = found
            hits.append({"語": word, "極性": lexicon[word], "否定": 0, "重み": 1.0, "end": i + n - 1})
            i += n
            continue
        t = toks[i]
        if negation and hits and is_negation(t):
            # 直前の極性語との間が付属語（「は」「て」「わけが」など）だけなら反転する。「良くはない」「面白くないわけがない」
            between = toks[hits[-1]["end"] + 1:i]
            if all(b.feature.pos1 in ("助詞", "助動詞", "接尾辞") or b.feature.pos2 == "非自立可能" or b.feature.lemma in ("訳", "筈")
                   for b in between):
                hits[-1]["極性"] *= -1
                hits[-1]["否定"] += 1
        if adversative and is_adversative(toks, i):
            for h in hits:
                h["重み"] *= 0.5   # 逆接より前を半分にする（= 後ろを重くする）
        i += 1
    return sum(h["極性"] * h["重み"] for h in hits), hits

def explain(text):
    score, hits = analyze(text)
    label = "ポジ" if score > 0 else "ネガ" if score < 0 else "判定なし"
    why = "  ".join(f"{h['語']}{'（否定×%d）' % h['否定'] if h['否定'] else ''} {h['極性'] * h['重み']:+g}" for h in hits)
    print(f"{label:4} {score:+5.1f}  {text}\n{'':12}根拠: {why or '辞書の語なし'}")

for text in ["この映画は本当に素晴らしかった。もう一度観たい。", "最悪の接客で、二度と行かないと決めた。", "明日は月曜日です。",
             "期待していたほどではなかったが、悪くはない。", "面白くないわけがない。", "値段は高いけれど、品質には満足しています。",
             "料理は美味しいが、接客が悪い。", "めっちゃ嬉しい。"]:
    explain(text)
```

```python
import numpy as np
import pandas as pd

# WRIME Ver.2（宮内ら 2022 / Suzuki et al., LREC 2022, https://github.com/ids-cv/wrime）: SNS の投稿 35,000 件に、書き手本人と
# 読み手 3 人が感情極性 (-2〜+2) を付けたもの。CC BY-NC-ND 4.0（非商用・改変禁止）
df = pd.read_csv("https://raw.githubusercontent.com/ids-cv/wrime/master/wrime-ver2.tsv", sep="\t")
df["readers"] = df[[f"Reader{i}_Sentiment" for i in (1, 2, 3)]].mean(axis=1)   # 読み手 3 人の平均（実数）
test = df[(df["Train/Dev/Test"] == "test") & (df["readers"] != 0)].reset_index(drop=True)   # 平均が 0（中立）の投稿は除く
y = np.where(test["readers"] > 0, "positive", "negative")
print(f"評価: {len(test)} 件（書き手 {test['UserID'].nunique()} 人）ポジの割合 {(y == 'positive').mean():.3f}")

scores, rows = {}, {}
for name, kw in [("辞書だけ", dict(negation=False, adversative=False)), ("+ 否定で反転", dict(adversative=False)), ("+ 否定 + 逆接", {})]:
    s = np.array([analyze(t, **kw)[0] for t in test["Sentence"]])
    c = s != 0   # 辞書の語がない（または打ち消し合った）文は判定しない
    scores[name] = s
    rows[name] = {"判定できた割合": c.mean(), "判定できた文の正解率": (np.where(s > 0, "positive", "negative") == y)[c].mean()}
print(pd.DataFrame(rows).T.round(3))

# 規則を足して判定（ポジ / ネガ / 判定なし）が変わった文だけで比べる。判定なしは不正解と数える
correct = lambda s: np.where(s > 0, "positive", np.where(s < 0, "negative", "")) == y
for before, after in [("辞書だけ", "+ 否定で反転"), ("+ 否定で反転", "+ 否定 + 逆接")]:
    ch = np.sign(scores[before]) != np.sign(scores[after])
    print(f"{after}: 判定が変わった {ch.sum()} 件で、正解 {correct(scores[before])[ch].sum()} 件 → {correct(scores[after])[ch].sum()} 件")
```

```python
from transformers import pipeline

# この章の「日本語の感情分析（ポジネガ判定）をする」と同じ多言語モデル (Apache 2.0)。Colab の CPU で 5〜8 分程度
clf = pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
order = np.argsort(test["Sentence"].str.len().to_numpy())   # 長さの近い文をまとめてパディングを減らす
out = clf(list(test["Sentence"].iloc[order]), top_k=None, truncation=True, batch_size=32)
model_pred = np.empty(len(test), dtype=object)
posneg = lambda rs: max((r for r in rs if r["label"] != "neutral"), key=lambda r: r["score"])["label"]   # neutral を除き、ポジとネガの高いほう
model_pred[order] = [posneg(rs) for rs in out]

lex = scores["+ 否定 + 逆接"]
covered = lex != 0
lex_pred = np.where(lex > 0, "positive", "negative")
acc = lambda p, m=slice(None): (p == y)[m].mean()
print(f"モデル        : 全体 {acc(model_pred):.3f}   辞書が判定できた {covered.sum()} 件で {acc(model_pred, covered):.3f}")
print(f"辞書 + 規則   :                辞書が判定できた {covered.sum()} 件で {acc(lex_pred, covered):.3f}")
print(f"辞書 → モデル : 全体 {acc(np.where(covered, lex_pred, model_pred)):.3f}（辞書で判定できない文だけモデルを使う）")

# 辞書とモデルの正解率の差（判定できた文）の 95% 信頼区間。同じ人の投稿は似ているので、書き手を単位にリサンプリングする
rng = np.random.default_rng(0)
users = test["UserID"].to_numpy()
rows_of = {u: np.flatnonzero((users == u) & covered) for u in np.unique(users)}
diff = [acc(lex_pred, ix) - acc(model_pred, ix)
        for ix in (np.concatenate([rows_of[u] for u in rng.choice(list(rows_of), len(rows_of))]) for _ in range(1000))]
print(f"差（辞書 - モデル）= {acc(lex_pred, covered) - acc(model_pred, covered):+.3f}  95% CI {np.percentile(diff, 2.5):+.3f}〜{np.percentile(diff, 97.5):+.3f}")
```

```python
from collections import Counter

# 効いた語を評価データ全体で集計する: 語の向き（否定の反転後）が、投稿の正解と逆だった割合の高い語
agree, disagree = Counter(), Counter()
for text, gold, s in zip(test["Sentence"], y, lex):
    for h in analyze(text)[1] if s != 0 else []:
        (agree if (h["極性"] > 0) == (gold == "positive") else disagree)[h["語"]] += 1
stat = pd.DataFrame({"正解と同じ向き": agree, "逆向き": disagree}).fillna(0).astype(int)
stat["逆向きの割合"] = (stat["逆向き"] / stat.sum(axis=1)).round(2)
stat["辞書の極性"] = [lexicon[w] for w in stat.index]
print(stat[stat[["正解と同じ向き", "逆向き"]].sum(axis=1) >= 15].sort_values("逆向きの割合", ascending=False).head(8))

# 判定できなかった投稿によく出る内容語（辞書にない語の候補）
miss = Counter(t.feature.orthBase for text in test["Sentence"][~covered] for t in tagger(text)
               if t.feature.pos1 in ("名詞", "形容詞", "動詞") and t.feature.pos2 not in ("数詞", "非自立可能") and t.feature.orthBase and len(t.surface) > 1)
print("判定できなかった投稿に多い語:", [w for w, _ in miss.most_common(15)])
```

```python
# 弱点の例（自作の文）。モデルの判定（上と同じく neutral を除いたポジとネガの高いほう）と並べる
for text in ["素晴らしい対応ですね、2 時間も待たされるなんて。", "このラーメン、マジで神ってる。", "エモい曲で泣いた。",
             "バッテリーの持ちが短い。", "値段が高い。", "値段が安い。", "評価が高い。"]:
    explain(text)
    print(f"{'':12}モデル: {posneg(clf(text, top_k=None))}")

# 辞書は書き換えるだけで直せる（分野の語を足す、誤りの多い語を外す）
lexicon.update({"神ってる": 1, "エモい": 1})
explain("このラーメン、マジで神ってる。")
```

- 辞書の判定は、どの語がどちら向きに効いたかをそのまま示せる。評価データ全体で集計すると、正解と逆向きに働きやすい語が分かる。WRIME では「目茶」（「めっちゃ」は語彙素で引くとこれになり、辞書ではネガなので、「めっちゃ嬉しい。」は「嬉しい」と打ち消し合って判定なしになる）、「大丈夫」「結構」「明日」（いずれもポジ）などで、強調の副詞や、文脈で向きの決まらない語が極性語として数えられている。こうした語を辞書から外したり、分野の語を足したり（最後のセル）するのは、モデルの再学習より手軽である。学習済みモデルの根拠を予測のあとから近似して見る方法は、テキスト分類の章の「個々の予測の根拠を可視化する (LIME)」を参照。辞書では、根拠が判定の仕組みそのものである点が違う。
- WRIME Ver.2 のテスト（読み手 3 人の平均が中立でない 2,223 件）で、スコアが 0 でなく判定できた文は 69.9%、その 1,553 件の正解率は辞書 + 規則が 0.759、モデルが 0.740 だった。差 +0.019 の書き手単位の 95% 信頼区間は -0.007〜+0.057 で 0 をまたぎ、テストの書き手は 10 人しかないので、どちらが上とは言えない。判定できない文だけモデルに任せると、全体の正解率はモデルだけの 0.728 から 0.741 になった。
- 否定の反転で判定が変わった 146 件では正解が 48 件から 65 件に、逆接の規則で判定が変わった 56 件では 4 件から 31 件に増えた。逆接の規則では、判定できた割合も 68.4% から 69.9% に増えた（ポジとネガの語が打ち消し合っていた文に判定が付く）。ただし規則は直前の極性語しか見ない簡単なもので、「期待していたほどではなかった」の「期待」の反転のように偶然合うものも含む。従属節の否定や否定の作用域は扱えない。否定の扱いは意味解析の章の「否定・モダリティ・時制を取り出す」を参照。
- 皮肉（「素晴らしい対応ですね、2 時間も待たされるなんて。」）はポジの語だけを数えて誤る。「神ってる」「エモい」のような新しい語は辞書になく（辞書は 2008 年版）、「高い」「安い」「短い」のように対象によって極性が変わる語は入っていない。「値段」は名詞としてネガなので、「値段が安い。」もネガになる。モデルも皮肉は誤り、「バッテリーの持ちが短い。」「値段が高い。」もポジとした。新しい語の文にもモデルは判定を返したが、「エモい曲で泣いた。」はネガだった。
- 形態素解析は unidic-lite (UniDic) だが、辞書の用言編は IPAdic の分割で書かれているので、複数の形態素からなる表現は表層を連結して引いている。分割の違いで引けない表現が残る。
- 日本語評価極性辞書（用言編 約 5 千、名詞編 約 8 千 5 百の表現）は東北大学 乾・岡崎研究室の作成で、クレジットを明記すれば商用利用も可能（用言編は小林ら 2005、名詞編は東山ら 2008 を引用する）。配布元は研究室の旧サイトで、移転などで読めなくなったら、この辞書を JSON にして同梱したライブラリ oseti（MIT ライセンス。否定の反転はするが逆接の重み付けはなく、形態素解析に MeCab を使う）から取り出せる。WRIME は CC BY-NC-ND 4.0（非商用・改変禁止）で、ここでは評価にだけ使った。モデルは Apache 2.0。
- 評価極性辞書と否定・逆接ヒューリスティクスによる文単位・文書単位の集約（第 1〜3 節）および語彙ネットワークと平均場イジングスピンモデルによる極性辞書の自動獲得（第 4 節）の演習は、姉妹編『[自然言語処理の教科書](https://github.com/mamorlis/nlpbook)』第 2 章の [評価極性辞書と文単位・文書単位の極性分析](https://github.com/mamorlis/nlpbook/blob/main/notebooks/ch2_sentiment_lexicon.ipynb) も参照。

## 喜び・悲しみ・怒りなど 8 種類の感情の強さを推定する (WRIME) {emotion_ja}

ポジティブかネガティブかだけでなく、投稿に表れた喜び・悲しみ・怒りなどの感情を、種類ごとに強さとして推定したい。書き手本人と読み手とで感じ方がどう違うかも見る。

```python
!pip install -q fugashi unidic-lite   # 後半で使う BERT の日本語トークナイザーに必要
```

```python
import numpy as np
import pandas as pd

# WRIME Ver.1 (梶原ら, NAACL 2021, https://github.com/ids-cv/wrime): SNS の投稿 43,200 件に、Plutchik の 8 感情の強度
# (0:無 1:弱 2:中 3:強) を書き手本人 1 人と読み手 3 人が付けたもの。CC BY-NC-ND 4.0 (非商用・改変禁止)
url = "https://raw.githubusercontent.com/ids-cv/wrime/master/wrime-ver1.tsv"
df = pd.read_csv(url, sep="\t")
EMO = ["Joy", "Sadness", "Anticipation", "Surprise", "Anger", "Fear", "Disgust", "Trust"]
JA = ["喜び", "悲しみ", "期待", "驚き", "怒り", "恐れ", "嫌悪", "信頼"]
W = [f"Writer_{e}" for e in EMO]   # 書き手 (主観)
# 読み手 (客観) は 3 人の強度の平均 (実数)。TSV の「Avg. Readers_*」列は平均を四捨五入した整数なので使わない
for e in EMO:
    df[f"Readers_{e}"] = df[[f"Reader{i}_{e}" for i in (1, 2, 3)]].mean(axis=1)
R = [f"Readers_{e}" for e in EMO]
train, test = df[df["Train/Dev/Test"] == "train"], df[df["Train/Dev/Test"] == "test"]
print(len(df), "件", df["Train/Dev/Test"].value_counts().to_dict())
print("書き手の人数:", df.groupby("Train/Dev/Test")["UserID"].nunique().to_dict(),
      " 学習とテストで書き手が重ならない:", set(train["UserID"]).isdisjoint(test["UserID"]))

def mean_over_readers(f):   # 読み手 3 人 (または 3 組) について平均する
    return lambda e: np.mean([f(e, i) for i in (1, 2, 3)])

pd.DataFrame({
    "書き手 ≥1 の割合": [(df[f"Writer_{e}"] >= 1).mean() for e in EMO],
    "読み手 1 人 ≥1 の割合": [mean_over_readers(lambda e, i: (df[f"Reader{i}_{e}"] >= 1).mean())(e) for e in EMO],
    "書き手と読み手 1 人の相関": [mean_over_readers(lambda e, i: df[f"Writer_{e}"].corr(df[f"Reader{i}_{e}"]))(e) for e in EMO],
    "読み手どうしの相関": [mean_over_readers(lambda e, i: df[f"Reader{i}_{e}"].corr(df[f"Reader{i % 3 + 1}_{e}"]))(e) for e in EMO],
}, index=JA).round(3)
```

```python
# 書き手は強い感情 (3) を付けたのに、読み手 3 人が全員 0 を付けた投稿
# 本文の出典: WRIME Ver.1 (梶原ら, NAACL 2021, https://github.com/ids-cv/wrime)、
# CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/)。先頭 60 字の抜粋
for e, ja in [("Anger", "怒り"), ("Trust", "信頼")]:
    strong = df[df[f"Writer_{e}"] == 3]
    hit = strong[(strong[[f"Reader{i}_{e}" for i in (1, 2, 3)]] == 0).all(axis=1)]
    print(f"{ja}: 書き手が 3 を付けた {len(strong)} 件のうち、読み手 3 人とも 0 は {len(hit)} 件")
    for s in hit["Sentence"].sample(2, random_state=0):
        print("   ", s[:60])
```

```python
from sentence_transformers import SentenceTransformer

# 学習データ 40,000 件から 10,000 件を使う。Colab の CPU で埋め込みの計算に約 3〜5 分
sub = train.sample(10000, random_state=0)
embedder = SentenceTransformer("intfloat/multilingual-e5-small")
encode = lambda texts: embedder.encode(["query: " + t for t in texts], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
X_train, X_test = encode(sub["Sentence"].tolist()), encode(test["Sentence"].tolist())
print(X_train.shape, X_test.shape)
```

```python
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score

# 8 感情の強度をまとめて回帰する (出力 8 次元のリッジ回帰。正則化の強さは学習データ内の一個抜き交差検証で選ぶ)
alphas = [0.1, 0.3, 1, 3, 10]
ridge_w = RidgeCV(alphas=alphas).fit(X_train, sub[W])   # 書き手の強度を予測
ridge_r = RidgeCV(alphas=alphas).fit(X_train, sub[R])   # 読み手 3 人の平均強度を予測
pred_emb = {"書き手": ridge_w.predict(X_test), "読み手": ridge_r.predict(X_test)}

def evaluate(pred):
    """感情ごとに、強度との相関 r と、「強度 ≥1 (その感情あり)」を当てる ROC-AUC を求める"""
    rows = []
    for who, cols in [("書き手", W), ("読み手", R)]:
        for i, ja in enumerate(JA):
            y = test[cols[i]].to_numpy()
            rows.append({"感情": ja, "立場": who, "正例数": int((y >= 1).sum()),
                         "相関 r": np.corrcoef(y, pred[who][:, i])[0, 1], "AUC": roc_auc_score(y >= 1, pred[who][:, i])})
    return pd.DataFrame(rows).pivot(index="感情", columns="立場").loc[JA]

res_emb = evaluate(pred_emb)
res_emb.round(3)
```

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# WRIME Ver.1 (NC-ND) の train で、書き手 8 + 読み手 3 人の平均 8 の計 16 個の強度を回帰するよう学習した BERT。
# モデルカードの表示は CC BY-SA 3.0
name = "patrickramos/bert-base-japanese-v2-wrime-fine-tune"
tokenizer = AutoTokenizer.from_pretrained(name)
bert = AutoModelForSequenceClassification.from_pretrained(name).eval()

def bert_predict(texts, bs=64):
    order = np.argsort([len(t) for t in texts])   # 長さの近い文をまとめてパディングを減らす
    out = np.zeros((len(texts), bert.config.num_labels), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]
            batch = tokenizer([texts[j] for j in idx], padding=True, truncation=True, max_length=128, return_tensors="pt")
            out[idx] = bert(**batch).logits.numpy()
    return out

# Colab の CPU で、テスト 2,000 件の予測に約 3〜5 分 (モデルのダウンロード約 450MB は別)
logits = bert_predict(test["Sentence"].tolist())
pred_bert = {"書き手": logits[:, :8], "読み手": logits[:, 8:]}
res_bert = evaluate(pred_bert)
pd.concat({"埋め込み + リッジ": res_emb["相関 r"], "BERT (WRIME で学習)": res_bert["相関 r"]}, axis=1).round(3)
```

```python
# 8 感情の平均の相関と、その 95% 信頼区間。テストは 10 人の投稿なので、投稿ではなく書き手を単位に
# リサンプリングする (同じ人の投稿どうしは似ているので、投稿単位だと区間が狭く出すぎる)
rng = np.random.default_rng(0)
users = test["UserID"].to_numpy()
rows_of = {u: np.flatnonzero(users == u) for u in np.unique(users)}
boots = [np.concatenate([rows_of[u] for u in rng.choice(list(rows_of), len(rows_of))]) for _ in range(300)]

def mean_r(cols, pred, ix):
    y, p = test[cols].to_numpy()[ix], pred[ix]
    return np.mean([np.corrcoef(y[:, i], p[:, i])[0, 1] for i in range(8)])

for model_name, pred in [("埋め込み + リッジ", pred_emb), ("BERT (WRIME で学習)", pred_bert)]:
    for who, cols in [("書き手", W), ("読み手", R)]:
        b = [mean_r(cols, pred[who], ix) for ix in boots]
        print(f"{model_name:16} {who}: 平均 r = {mean_r(cols, pred[who], slice(None)):.3f}"
              f"  (書き手単位のブートストラップ 95% CI {np.percentile(b, 2.5):.3f}〜{np.percentile(b, 97.5):.3f})")
```

```python
texts = ["やっと試験に合格した!今夜はお祝いだ", "財布を落とした…最悪", "明日からの旅行が楽しみすぎて眠れない",
         "また電車が遅れてる。いい加減にしてほしい", "夜中に変な物音がして怖い"]
logits = bert_predict(texts)
for who, sl in [("書き手", slice(0, 8)), ("読み手", slice(8, 16))]:
    print(f"\nBERT の予測 ({who})")
    print(pd.DataFrame(logits[:, sl], columns=JA, index=[t[:14] for t in texts]).round(1).to_string())
```

- 多くの感情で、書き手は読み手より多くの感情を報告する（恐れだけは逆）。特に怒りと信頼は、強度 1 以上を付けた割合が書き手で 12.9% と 21.7%、読み手 1 人あたりでは 2.8% と 5.1% しかない。書き手が信頼を 3（強）と付けた 1,947 件のうち 1,220 件は、読み手 3 人とも 0 だった。本人が感じていても文面に出ていない感情は、テキストからは推定しにくい。
- 書き手と読み手 1 人の相関は、どの感情でも読み手どうしの相関より低い。差が大きいのは怒り（0.401 と 0.627）と期待（0.410 と 0.581）。信頼は読み手どうしでも 0.216 と一致しにくい。
- 8 感情の平均の相関は、どちらのモデルでも読み手のほうが書き手より高かった（埋め込み + リッジで 0.473 と 0.361、BERT で 0.665 と 0.464。書き手単位のブートストラップ信頼区間は重ならない）。文面から伝わる感情のほうが推定しやすいことに加え、読み手は 3 人の平均なので 1 人分のラベルよりノイズが少ないことも効いている。用途に応じて、どちらのラベルで学習・評価するかを決める（「読んだ人にどう伝わるか」なら読み手、「本人の状態」なら書き手）。
- ほとんど 0 ばかりの目標では、相関 r が低く出やすい。読み手の怒りはテストで強度 1 以上が 37 件しかなく、埋め込み + リッジの相関は 0.241 と書き手 (0.408) より低いが、「その感情あり」を当てる AUC は 0.840 と書き手 (0.792) より高い。出現の少ない感情は、正例数を確かめ、相関と AUC の両方を見る。
- 強度を回帰しておけば、しきい値を決めて「この感情あり / なし」のマルチラベル分類にも使える。しきい値の決め方や評価は、テキスト分類の章の「1 つの文書に複数のラベルを付ける（マルチラベル分類）」「クラスの偏ったデータを扱う（クラス重み・しきい値調整）」と同じ。回帰なので、予測が 0 未満になることもある。
- WRIME で学習済みの BERT は、16 項目すべてで埋め込み + リッジより相関が高かった。ただし BERT は学習データ 40,000 件全部でモデル全体を微調整しており、条件は対等ではない。また、このモデルの学習コードは学習中の評価にテスト分割を使っている。最良のエポックを選ぶ設定ではなく最終エポックのモデルだが、テストを見ながら設定を決めた可能性は否定できないので、BERT の数値はやや楽観的に見る。
- WRIME の学習・検証・テストは書き手で分かれている（テストの 10 人は学習に出てこない）。同じ人の投稿が学習とテストの両方に入ると、書き手の癖を覚えて評価が甘くなる。信頼区間も、投稿ではなく書き手を単位にリサンプリングして求めた。ブートストラップによる信頼区間の求め方は評価指標と統計的検定の章の「ブートストラップ法で評価値の信頼区間を求める」を参照。
- WRIME は CC BY-NC-ND 4.0（非商用・改変禁止）。BERT のモデルは CC BY-SA 3.0 と表示されているが、WRIME で微調整したものなので、商用で使う場合は WRIME の条件も含めて確認が必要。ポジネガだけでよければ、この章の「日本語の感情分析（ポジネガ判定）をする」を使う。

## 観点ごとにポジティブ・ネガティブを判定する（アスペクトベース感情分析） {absa_ja gpu}

「売上は伸びたが利益は減った」のような文で、文全体ではなく観点（アスペクト）ごとに、売上=ポジ、利益=ネガと判定したい。有価証券報告書の ABSA データセット chABSA を使う。

```python
!pip install -q fugashi unidic-lite
```

```python
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import GroupShuffleSplit

# chABSA（TIS 株式会社 / chakki-works、CC BY 4.0、https://github.com/chakki-works/chABSA-dataset）を第三者が Hugging Face に複製したもの。
# README の配布 URL はリンク切れのため、複製の版を固定して使う（複製が削除されると読めなくなる）
ds = load_dataset("zenless-lab/chABSA", revision="254de3bf9771bcafb40812cecd33e6cdb234bb49")
sents = pd.concat([ds["train"].to_pandas(), ds["test"].to_pandas()], ignore_index=True)
# 1 行 =（文, 観点の語, 極性）。from / to は文中の文字位置
df = pd.DataFrame([{"company": s.document_id, "sid": i, "sentence": s.sentence, "target": o["text"],
                    "start": o["from"], "end": o["to"], "label": o["polarity"]}
                   for i, s in enumerate(sents.itertuples()) for o in s.opinions])
print(len(sents), "文 /", len(df), "観点", df["label"].value_counts().to_dict())
df = df[df["label"] != "neutral"].reset_index(drop=True)   # 中立はごく少ないので除き、ポジ・ネガの 2 値にする

# 同じ会社の文が学習と評価の両方に入らないよう、会社単位で分ける
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(df, groups=df["company"]))
train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
test["mixed"] = test.groupby("sid")["label"].transform("nunique") > 1   # ポジとネガの観点が同居する文か
print("学習:", len(train), " 評価:", len(test), " うち対比の文の観点:", int(test["mixed"].sum()))
ex = test[test["mixed"]].iloc[0]
print(ex["sentence"]); print(test[test["sid"] == ex["sid"]][["target", "label"]].to_string(index=False))
```

```python
from sklearn.metrics import accuracy_score, f1_score
from transformers import pipeline

def report(name, pred):
    y, p = test["label"].to_numpy(), np.asarray(pred)
    m = test["mixed"].to_numpy()
    print(f"{name:24} 正解率 {accuracy_score(y, p):.3f}  macro-F1 {f1_score(y, p, average='macro'):.3f}"
          f"  | 対比の文での正解率 {accuracy_score(y[m], p[m]):.3f}")

# (1) 文全体の感情分析（この章の「日本語の感情分析（ポジネガ判定）をする」と同じモデル）の結果を、文中のすべての観点に配る
sent_clf = pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
uniq = test.drop_duplicates("sid")
out = sent_clf(list(uniq["sentence"]), top_k=None, truncation=True, batch_size=32)
sent_pred = {sid: max((r for r in rs if r["label"] != "neutral"), key=lambda r: r["score"])["label"]   # ポジとネガの高いほう
             for sid, rs in zip(uniq["sid"], out)}
report("文全体の感情分析", test["sid"].map(sent_pred))

# 文ごとに 1 つのラベルしか出せない方法の上限: 各文で多数派の極性を（正解を見て）全観点に付けた場合
majority = test.groupby("sid")["label"].agg(lambda s: s.mode()[0])
report("文単位の上限 (正解の多数派)", test["sid"].map(majority))

# 観点の語と文だけを見る方法の上限: 同じ文の同じ位置の語に、カテゴリ違い（例: business#sales と business#profit）で逆の極性が付いた行がある
span = test.groupby(["sid", "start", "end"])["label"]
print("同じ語に逆の極性が付いた行:", int((span.transform("nunique") > 1).sum()))
report("語と文だけで判定する上限", span.transform(lambda s: s.mode()[0]))
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# (2) 観点の語から次の読点・句点までを取り出す。日本語では「売上高は増加し、」のように、観点の後ろの述語に評価が出やすい
def window(r):
    rest = r.sentence[r.start:]
    cut = min([i for i in (rest.find("、", len(r.target)), rest.find("。")) if i >= 0], default=len(rest))
    return rest[:cut]

def tfidf_lr():
    return make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True), LogisticRegression(C=10, max_iter=1000))

whole = tfidf_lr().fit(train["sentence"], train["label"])                 # 比較用: 同じデータで学習した文全体の分類器
local = tfidf_lr().fit(train.apply(window, axis=1), train["label"])
report("TF-IDF (文全体)", whole.predict(test["sentence"]))
report("TF-IDF (観点の後ろの区間)", local.predict(test.apply(window, axis=1)))
print(test.head(3).apply(window, axis=1).tolist())
```

```python
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

# (3) 「観点の語」と「観点に印を付けた文」を文ペアにして BERT を微調整する（T4 で 5 分程度）
name = "tohoku-nlp/bert-base-japanese-v3"   # この章の「BERT をファインチューニングして文書分類する (Trainer)」と同じモデル (Apache 2.0)
tok = AutoTokenizer.from_pretrained(name)

def to_dataset(d):
    marked = [r.sentence[:r.start] + "【" + r.target + "】" + r.sentence[r.end:] for r in d.itertuples()]   # 同じ語が 2 回出ても区別できる
    labels = (d["label"] == "positive").astype(int).tolist() if "label" in d else [0] * len(d)
    ds = Dataset.from_dict({"aspect": list(d["target"]), "text": marked, "label": labels})
    return ds.map(lambda b: tok(b["aspect"], b["text"], truncation="only_second", max_length=192), batched=True, remove_columns=["aspect", "text"])

set_seed(0)   # 分類ヘッドの初期化も固定する
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2, id2label={0: "negative", 1: "positive"},
                                                           label2id={"negative": 0, "positive": 1})
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
args = TrainingArguments(output_dir="absa-bert", num_train_epochs=3, learning_rate=3e-5, warmup_steps=50, weight_decay=0.01,
                         per_device_train_batch_size=32, per_device_eval_batch_size=64, bf16=BF16, fp16=not BF16,
                         save_strategy="no", logging_steps=50, report_to="none", seed=0)
trainer = Trainer(model=model, args=args, train_dataset=to_dataset(train), data_collator=DataCollatorWithPadding(tok))
trainer.train()

def predict(d):
    return np.where(trainer.predict(to_dataset(d)).predictions.argmax(-1) == 1, "positive", "negative").tolist()

report("文ペアの微調整 (BERT)", predict(test))
print(f"GPU メモリの最大使用量: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

```python
# 新しい文で試す。観点の語を与えると、文中の位置を探して判定する
def absa(sentence, targets):
    d = pd.DataFrame([{"sentence": sentence, "target": t, "start": sentence.index(t), "end": sentence.index(t) + len(t)} for t in targets])
    return dict(zip(targets, predict(d)))

print(absa("売上高は前期を上回ったものの、原材料価格の高騰により営業利益は減少した。", ["売上高", "原材料価格", "営業利益"]))
print(absa("国内の販売は低調だったが、海外の販売は好調に推移した。", ["国内の販売", "海外の販売"]))
print("文全体の感情分析:", sent_clf("国内の販売は低調だったが、海外の販売は好調に推移した。")[0])
```

- 文全体の感情分析は 1 つの文に 1 つのラベルしか出せない。評価の 1,449 観点のうち 522 は対比の文（ポジとネガが同居）にあり、正解を見て文ごとに多数派を選んでも、対比の文の正解率は 0.625 が上限になる。
- 同じデータで学習した TF-IDF でも、文全体が入力だと対比の文は 0.569 だが、観点の後ろの区間だけにすると 0.822 に上がる。入力に「どの観点か」を含めることが重要である。
- 文ペア（観点, 観点に【】で印を付けた文）で BERT を微調整すると、全体 0.968、対比の文 0.935 になった。1 回の実行の値で、種を固定しても GPU の計算は完全には再現しない（set_seed を入れる前を含む検証の 3 回の実行では、全体で 0.967〜0.971 だった）。評価件数から見た 95% 区間はおよそ全体 ±0.009、対比の文 ±0.02 で、手法どうしの差はこれよりずっと大きい。
- 同じ文の同じ語に、カテゴリ違い（例: 「運輸業」の business#sales はポジ、business#profit はネガ）で逆の極性が付いた行が評価に 34 ある。語と文だけを入力にする方法では、正解率は全体 0.988、対比の文 0.967 が上限になる。カテゴリも区別したいなら、カテゴリを入力に加える。
- 新しい決算の文 2 つ（観点 5 つ）では期待どおりに判定した（正解ラベルのない例）。学習データは有価証券報告書だけなので、分野外の文は当てにならない。検証では「料理は美味しいが接客が悪い」の「料理」の判定が、同じコードの実行ごとにポジとネガで変わった。
- 数値は検証の GPU で float16 の経路（`BF16 = False`、T4 と同じ dtype）で実行したもので、GPU メモリの最大使用量は 4.7 GB だった。(1) と (2) は CPU だけで動く。テキスト分類の章の「ラベル付きデータなしで分類する（ゼロショット分類、NLI）」で「売上高は好調だ」という仮説の含意を判定させたり、LLM に観点ごとの極性を JSON で答えさせたりする方法もある。
- chABSA は TIS の久保・中山によるデータセットで、2016 年度の有価証券報告書 230 社分の「経営成績の概要」のうち、観点の付いた 3,215 文（全 6,119 文のうち）に観点の語、カテゴリ（company#sales など）、極性を付けたもの（CC BY 4.0、https://github.com/chakki-works/chABSA-dataset）。ここでは Hugging Face 上の第三者による複製を使った（文数・観点数は論文と一致する。複製にはライセンスの記載がないので、元データの CC BY 4.0 に従う）。
- 観点の語は与えられている前提で、極性だけを判定した。文から観点の語を見つけるのは系列ラベリングの問題になる（固有表現抽出の章の「文字位置のアノテーションを BIO タグ列に変換する」「BERT をファインチューニングして日本語の固有表現抽出器を作る」）。

## 有害な表現を検出する {toxicity_detection}

投稿やコメントの中から、暴言や差別的な表現を含むものを自動的に見つけたい。

```python
!pip install -q sentence-transformers scikit-learn
```

```python
import numpy as np
import pandas as pd

url = "https://raw.githubusercontent.com/inspection-ai/japanese-toxic-dataset/main/data/subset.csv"
df = pd.read_csv(url)
# 複数の作業者の判定を集計する: 「Toxic」「Very Toxic」と判定した人の割合を有害度とする
df["toxicity"] = (df["Toxic"] + df["Very Toxic"]) / df["annotation_num"]
df["label"] = (df["toxicity"] >= 0.5).astype(int)
print(len(df), "件 / 有害:", int(df["label"].sum()))
print("判定が割れた (全員一致ではない) 割合:", round(((df["toxicity"] > 0) & (df["toxicity"] < 1)).mean(), 3))
print(df[["text", "toxicity"]].sample(6, random_state=1).to_string(index=False))
```

```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# 方法 1: 禁止語のリスト (最も単純なベースライン)
NG_WORDS = ["死ね", "殺", "バカ", "馬鹿", "クソ", "糞", "カス", "ゴミ", "キモ", "うざ", "消えろ", "アホ", "ブス", "デブ", "ガイジ"]
keyword = df["text"].apply(lambda t: float(any(w in t for w in NG_WORDS)))

# 方法 2: 文埋め込み + ロジスティック回帰 (5 分割交差検証)
embedder = SentenceTransformer("intfloat/multilingual-e5-small")
X = embedder.encode(["query: " + t for t in df["text"]], normalize_embeddings=True, batch_size=64)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
proba = cross_val_predict(LogisticRegression(C=10, max_iter=1000, class_weight="balanced"), X, df["label"], cv=cv, method="predict_proba")[:, 1]

for name, score in [("禁止語リスト", keyword), ("埋め込み + 分類器", proba)]:
    print(f"{name}: ROC-AUC {roc_auc_score(df['label'], score):.3f} / 平均適合率 {average_precision_score(df['label'], score):.3f}")
print("禁止語リストに当たった件数:", int(keyword.sum()), "/ そのうち有害:", int(df["label"][keyword == 1].sum()))
```

```python
# しきい値で適合率と再現率が入れ替わる。見逃しと誤検出のどちらを避けたいかで決める
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(df["label"], proba)
for t in [0.3, 0.5, 0.7, 0.9]:
    i = np.searchsorted(thresholds, t)
    print(f"しきい値 {t}: 適合率 {precision[i]:.2f} / 再現率 {recall[i]:.2f}")

df["proba"] = proba
print("\n見逃し (有害だがスコアが低い):"); print(df[df["label"] == 1].sort_values("proba").head(5)[["text", "toxicity", "proba"]].to_string(index=False))
print("\n誤検出 (有害でないがスコアが高い):"); print(df[df["label"] == 0].sort_values("proba").tail(5)[["text", "toxicity", "proba"]].to_string(index=False))
```

- 有害かどうかの判断は人によって割れる。多数決で 0 / 1 にする前に、判定の分布を見る。割れた例を学習や評価から除くのか、有害度として連続値で扱うのかを決める。
- 禁止語リストは説明しやすいが、遠回しな表現や文脈に依存する攻撃を見逃し、無害な用法（「バカ売れ」）を拾う。学習した分類器と組み合わせるのが普通。
- 有害な投稿は全体のごく一部なので、正解率ではなく適合率・再現率・平均適合率で評価する。しきい値は、運用（自動で消すのか、人が確認するのか）に合わせて決める（テキスト分類の章の「クラスの偏ったデータを扱う（クラス重み・しきい値調整）」も参照）。
- 「しね」のようなひらがな書きや伏せ字は、表記が変わるだけで見逃されやすい。文字の正規化（ひらがな・カタカナの統一など）をしてから判定する、表記ゆれを含むデータで学習する、といった対策がある。
- 分類器は、特定の集団を指す語が出てくるだけで有害と判定しがちである（差別に反対する投稿まで有害とする）。誤検出の例を読んで、そうした偏りがないか確かめる。
- データは数百件と小さい。実際には、対象とするサービスの投稿で数千件以上の注釈データを作る。何を有害とするかの基準（ガイドライン）を先に決めることが、モデルの選択より重要。
