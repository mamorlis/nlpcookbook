# テキスト分類

文書にラベルを付ける。古典的な TF-IDF と分類結果の分析から、文埋め込みによる分類とゼロショット分類、少ないラベルで学習する方法 (データ拡張、能動学習、弱教師あり学習、ラベル誤りの検出、グラフによる半教師あり学習)、BERT のファインチューニング、言語間転移、感情分析、有害な表現の検出まで。

## TF-IDF とロジスティック回帰で日本語のニュース記事を分類する {tfidf_logreg_ja}

日本語の文書分類を、まずは軽くて強い古典的な手法でやってみたい。livedoor ニュースコーパス (9 カテゴリ) を使う。

```python
!pip install -q fugashi unidic-lite
```

```python
import io, tarfile, urllib.request
import pandas as pd

def load_livedoor():
    """livedoor ニュースコーパス (CC BY-ND) を DataFrame にする"""
    url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
    with urllib.request.urlopen(url) as r:
        tar = tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")
    rows = []
    for m in tar.getmembers():
        parts = m.name.split("/")
        if m.isfile() and len(parts) == 3 and parts[2] != "LICENSE.txt":
            lines = tar.extractfile(m).read().decode("utf-8").split("\n")  # 1 行目 URL, 2 行目 日付, 3 行目 タイトル
            rows.append({"label": parts[1], "title": lines[2], "text": "\n".join(lines[3:])})
    return pd.DataFrame(rows)

df = load_livedoor()
print(len(df))
df["label"].value_counts()
```

```python
from fugashi import Tagger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

tagger = Tagger("-Owakati")
df["wakati"] = [tagger.parse(t[:2000]).strip() for t in df["text"]]   # 先頭 2000 文字を分かち書き

train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=0)
clf = make_pipeline(
    TfidfVectorizer(token_pattern=r"\S+", min_df=3, max_df=0.5, sublinear_tf=True),   # 分かち書き済みなので空白で区切る
    LogisticRegression(C=10, max_iter=1000),
)
clf.fit(train["wakati"], train["label"])
print("正解率:", accuracy_score(test["label"], clf.predict(test["wakati"])))
```

```python
new = "日本代表は後半に 2 点を奪って逆転し、ワールドカップ予選で勝利を収めた。"
print(clf.predict([tagger.parse(new)]), clf.predict_proba([tagger.parse(new)]).max().round(2))
```

- 先に分かち書きして空白区切りの文字列にしておくと、scikit-learn の標準的な部品がそのまま使える。
- このベースラインで 9 割を超える。BERT を持ち出す前に、まず TF-IDF + 線形モデルを試すのが定石。

## 分類結果を評価する (適合率・再現率・F1・混同行列) {classification_report}

正解率だけでなく、クラスごとの適合率・再現率・F1 と、どのクラスをどのクラスに間違えたかを見たい。

```python
!pip install -q matplotlib-fontja
```

```python
import io, tarfile, urllib.request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

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
train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=0)
# タイトルだけを使う難しめの設定。文字 n-gram なら分かち書きも不要
clf = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True), LogisticRegression(C=10, max_iter=1000))
clf.fit(train["title"], train["label"])
pred = clf.predict(test["title"])
```

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

print(classification_report(test["label"], pred, digits=3))
print("macro-F1:", round(f1_score(test["label"], pred, average="macro"), 3),
      " micro-F1 (= 正解率):", round(f1_score(test["label"], pred, average="micro"), 3))

dummy = DummyClassifier(strategy="most_frequent").fit(train["title"], train["label"])
print("最頻クラスを答えるだけのベースライン:", round(accuracy_score(test["label"], dummy.predict(test["title"])), 3))
```

```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay.from_predictions(test["label"], pred, normalize="true", values_format=".2f",
                                        xticks_rotation=60, cmap="Blues", ax=ax, colorbar=False)
plt.show()
```

- macro-F1 はクラスごとの F1 の単純平均で、少数クラスも同じ重みで評価する。不均衡データでは正解率より macro-F1 を見る。
- 混同行列は行が正解、列が予測。`normalize="true"` で行ごとの割合 (= 再現率) になる。
- 必ず「何も考えないベースライン」と比べる。

## 分類に効いている単語を調べる (線形モデルの係数) {feature_importance}

モデルがどの単語を手がかりに各クラスと判断しているのかを確認したい。データのリーク (カンニング) の発見にも役立つ。

```python
!pip install -q fugashi unidic-lite
```

```python
import io, tarfile, urllib.request
import numpy as np
import pandas as pd
from fugashi import Tagger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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
tagger = Tagger("-Owakati")
vec = TfidfVectorizer(token_pattern=r"\S+", min_df=3, max_df=0.5, sublinear_tf=True)
X = vec.fit_transform(tagger.parse(t[:2000]).strip() for t in df["text"])
clf = LogisticRegression(C=10, max_iter=1000).fit(X, df["label"])
```

```python
vocab = vec.get_feature_names_out()
top = {label: vocab[np.argsort(-coef)[:12]] for label, coef in zip(clf.classes_, clf.coef_)}
pd.DataFrame(top)
```

```python
# 1 つの文書について、どの語がどれだけ寄与したか (tf-idf 値 × 係数)
i = 0
row = X[i].toarray().ravel()
c = list(clf.classes_).index(df["label"][i])
contrib = row * clf.coef_[c]
print(df["label"][i], "|", df["title"][i])
print([(vocab[j], round(float(contrib[j]), 2)) for j in np.argsort(-contrib)[:10]])
```

- カテゴリ名そのもの (「独女」「peachy」「smax」) や、本文に含まれる関連記事の URL の断片 (`livedoor`、`detail`) が上位に来る。現実のタスクでは、こうした「答えが漏れている」特徴に頼っていないかを確認する。

## 個々の予測の根拠を可視化する (LIME) {lime_explain}

ある文書がなぜそのクラスに分類されたのか、どの単語が効いたのかを、モデルの中身に関係なく説明したい。

```python
!pip install -q lime
```

```python
from lime.lime_text import LimeTextExplainer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

cats = ["alt.atheism", "soc.religion.christian"]
train = fetch_20newsgroups(subset="train", categories=cats)   # あえてヘッダを残す
test = fetch_20newsgroups(subset="test", categories=cats)
clf = make_pipeline(TfidfVectorizer(sublinear_tf=True), LogisticRegression(C=10, max_iter=1000)).fit(train.data, train.target)
print("正解率:", round(clf.score(test.data, test.target), 3))
```

```python
explainer = LimeTextExplainer(class_names=["atheism", "christian"], random_state=0)
i = 83
exp = explainer.explain_instance(test.data[i], clf.predict_proba, num_features=10, num_samples=2000)
print("正解:", cats[test.target[i]], " 予測確率:", clf.predict_proba([test.data[i]]).round(2))
for word, weight in exp.as_list():
    print(f"{weight:+.3f}  {word}")
```

```python
from IPython.display import HTML

HTML(exp.as_html(text=False))   # 棒グラフで表示 (text=True にすると原文の色付き表示も付く)
```

- LIME は、単語をランダムに消した文を多数作って予測の変化を調べ、局所的な線形モデルで近似する。分類器が何であっても (BERT でも) 使える。
- この例では `Posting`、`Host`、`edu` などメールのヘッダ由来の語が効いている。高い正解率が本質的でない手がかりによるものだと分かる、有名な例。
- 日本語は `LimeTextExplainer(split_expression=lambda s: 分かち書き(s).split(), bow=False)` のように分割方法を渡す。

## モデルが自信満々に間違えた事例を調べる (エラー分析) {error_analysis}

精度の数字だけでなく、どんな事例を間違えているのかを見たい。アノテーション誤りの発見にもつながる。

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline

cats = ["rec.sport.baseball", "rec.sport.hockey", "sci.space", "sci.med", "comp.graphics"]
data = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))
pipe = make_pipeline(TfidfVectorizer(stop_words="english", sublinear_tf=True), LogisticRegression(C=10, max_iter=1000))

# 交差検証で、すべての事例について「その事例を学習に使っていないモデル」の予測確率を得る
proba = cross_val_predict(pipe, data.data, data.target, cv=5, method="predict_proba", n_jobs=-1)
df = pd.DataFrame({"text": [" ".join(t.split())[:110] for t in data.data], "gold": data.target, "pred": proba.argmax(axis=1),
                   "p_gold": proba[np.arange(len(proba)), data.target], "p_pred": proba.max(axis=1)})
df["gold"], df["pred"] = df["gold"].map(lambda i: cats[i]), df["pred"].map(lambda i: cats[i])
print("正解率:", round((df.gold == df.pred).mean(), 3))
```

```python
pd.set_option("display.max_colwidth", 120)
errors = df[df.gold != df.pred].sort_values("p_gold")   # 正解ラベルの確率が低い順 = 自信満々に間違えた順
errors.head(8)
```

```python
print("空に近い文書の割合 (誤り / 全体):", round((errors.text.str.len() < 20).mean(), 2), "/", round((df.text.str.len() < 20).mean(), 2))
pd.crosstab(errors.gold, errors.pred)
```

- 上位には「本文が空」「引用だけ」「ラベルのほうが怪しい」事例が並ぶ。モデルの改善より先にデータの掃除が必要だと分かることが多い。
- この「交差検証の予測確率でラベル誤りを探す」考え方を体系化したのが Confident Learning (`cleanlab`)。この章の「学習データの中の付け間違えたラベルを見つける (Confident Learning)」を参照。

## 複数の分類器を交差検証で比較する {compare_classifiers}

ナイーブベイズ、ロジスティック回帰、SVM などのうち、どれが自分のデータに合うかを公平に比べたい。

```python
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

cats = ["rec.sport.baseball", "sci.space", "comp.graphics", "talk.politics.mideast", "rec.autos", "sci.med"]
data = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))

models = {
    "MultinomialNB": MultinomialNB(alpha=0.1),
    "ComplementNB": ComplementNB(alpha=0.1),
    "LogisticRegression": LogisticRegression(C=10, max_iter=1000),
    "LinearSVC": LinearSVC(C=1),
    "SGD (hinge)": SGDClassifier(random_state=0),
    "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
rows = []
for name, model in models.items():
    pipe = make_pipeline(TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2), model)
    res = cross_validate(pipe, data.data, data.target, cv=cv, scoring=["accuracy", "f1_macro"])
    rows.append({"model": name, "accuracy": res["test_accuracy"].mean(), "±": res["test_accuracy"].std(),
                 "macro-F1": res["test_f1_macro"].mean(), "学習時間 (秒)": res["fit_time"].mean()})
pd.DataFrame(rows).round(3).sort_values("macro-F1", ascending=False)
```

- ベクトル化を `Pipeline` に入れておくのが重要。そうしないと、IDF の計算に検証データが混ざって評価が甘くなる (リーク)。
- 高次元で疎なテキスト特徴には線形モデル (LR、線形 SVM、NB) が強く、決定木系は相性が良くない。

## ハイパーパラメータをグリッドサーチで調整する {grid_search}

TF-IDF の設定や正則化の強さなど、複数のハイパーパラメータの最良の組み合わせを交差検証で探したい。

```python
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

cats = ["rec.sport.baseball", "sci.space", "comp.graphics", "talk.politics.mideast"]
train = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))
test = fetch_20newsgroups(subset="test", categories=cats, remove=("headers", "footers", "quotes"))

pipe = Pipeline([("tfidf", TfidfVectorizer(stop_words="english")), ("clf", LogisticRegression(max_iter=1000))])
grid = GridSearchCV(pipe, {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__sublinear_tf": [False, True],
    "tfidf__min_df": [1, 3],
    "clf__C": [0.1, 1, 10, 100],
}, cv=3, scoring="f1_macro", n_jobs=-1)
grid.fit(train.data, train.target)

print("最良:", grid.best_params_, round(grid.best_score_, 4))
print("テスト macro-F1:", round(grid.score(test.data, test.target), 4))
```

```python
res = pd.DataFrame(grid.cv_results_)[["param_tfidf__ngram_range", "param_tfidf__sublinear_tf", "param_tfidf__min_df", "param_clf__C", "mean_test_score", "std_test_score"]]
res.sort_values("mean_test_score", ascending=False).head(5).round(4)
```

- パラメータ名は `ステップ名__パラメータ名`。パイプラインごと探索することで、前処理の設定も一緒に最適化できる。
- 探索空間が広いときは `RandomizedSearchCV` や Optuna を使う。
- 上位の組み合わせのスコア差が標準偏差より小さければ、実質的に同等。

## 学習データを増やすと精度がどう変わるか調べる (学習曲線) {learning_curve}

あと何件アノテーションすれば精度が上がるのか、もう頭打ちなのかを判断したい。

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, learning_curve
from sklearn.pipeline import make_pipeline

cats = ["rec.sport.baseball", "sci.space", "comp.graphics", "talk.politics.mideast", "rec.autos", "sci.med"]
data = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))
pipe = make_pipeline(TfidfVectorizer(stop_words="english", sublinear_tf=True), LogisticRegression(C=10, max_iter=1000))

sizes, train_scores, val_scores = learning_curve(
    pipe, data.data, data.target, train_sizes=[50, 100, 200, 400, 800, 1600, 2800],
    cv=StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0), scoring="f1_macro", n_jobs=-1)

for n, v in zip(sizes, val_scores.mean(axis=1)):
    print(f"{n:5} 件: macro-F1 = {v:.3f}")
plt.plot(sizes, train_scores.mean(axis=1), "o-", label="train")
plt.plot(sizes, val_scores.mean(axis=1), "o-", label="validation")
plt.fill_between(sizes, val_scores.mean(axis=1) - val_scores.std(axis=1), val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.2)
plt.xscale("log"); plt.xlabel("training examples"); plt.ylabel("macro-F1"); plt.legend(); plt.show()
```

- 横軸を対数にして直線的に伸びているなら、データを倍にすれば同じ幅だけ上がると期待できる。寝てきたら頭打ち。
- 学習データでのスコアと検証スコアの差が大きいのは過学習の印。データを増やすか、正則化を強める。

## クラスの偏ったデータを扱う (クラス重み・しきい値調整) {imbalanced}

陽性が数 % しかないデータ (スパム、有害投稿、異常報告など) で、少数クラスをきちんと検出したい。

```python
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split

data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
y = (data.target == data.target_names.index("sci.med")).astype(int)   # 「医療」だけを陽性 (約 5%) とする 2 値分類
X_train, X_test, y_train, y_test = train_test_split(data.data, y, test_size=0.3, stratify=y, random_state=0)
vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2)
Xtr, Xte = vec.fit_transform(X_train), vec.transform(X_test)
print(f"陽性の割合: {y.mean():.1%}")

for cw in [None, "balanced"]:
    clf = LogisticRegression(C=1, max_iter=1000, class_weight=cw).fit(Xtr, y_train)
    print(f"\nclass_weight={cw}")
    print(classification_report(y_test, clf.predict(Xte), digits=3, target_names=["other", "med"]))
```

```python
# しきい値を動かして適合率と再現率のバランスを選ぶ
import matplotlib.pyplot as plt

clf = LogisticRegression(C=1, max_iter=1000).fit(Xtr, y_train)
prob = clf.predict_proba(Xte)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, prob)
f1 = 2 * precision * recall / (precision + recall + 1e-9)
best = f1[:-1].argmax()
print(f"平均適合率 (AP) = {average_precision_score(y_test, prob):.3f}")
print(f"しきい値 0.5 → F1 = {f1[np.searchsorted(thresholds, 0.5)]:.3f} / 最良しきい値 {thresholds[best]:.2f} → F1 = {f1[best]:.3f}")

plt.plot(recall, precision); plt.scatter(recall[best], precision[best], color="red", label="best F1")
plt.xlabel("recall"); plt.ylabel("precision"); plt.legend(); plt.show()
```

- 不均衡データでは正解率は役に立たない (全部「陰性」と答えても 95%)。少数クラスの適合率・再現率・F1、あるいは PR 曲線下面積 (AP) を見る。
- `class_weight="balanced"` は少数クラスの誤りを重く扱う。再現率が上がり、適合率は下がる。
- しきい値は検証データで決める (テストデータで決めてはいけない)。「見逃しを減らしたい」のか「誤検出を減らしたい」のか、用途に応じて選ぶ。

## 1 つの文書に複数のラベルを付ける (マルチラベル分類) {multilabel}

1 つの記事が「経済」かつ「国際」のように、複数のカテゴリに同時に属する場合の分類をしたい。

```python
import nltk
nltk.download("reuters", quiet=True)
from nltk.corpus import reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

train_ids = [i for i in reuters.fileids() if i.startswith("training/")]
test_ids = [i for i in reuters.fileids() if i.startswith("test/")]
print(len(train_ids), len(test_ids), "カテゴリ数:", len(reuters.categories()))
print(reuters.categories(train_ids[1]), reuters.raw(train_ids[1])[:80].replace("\n", " "))

mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform([reuters.categories(i) for i in train_ids])   # 0/1 の行列 (文書数 × ラベル数)
Y_test = mlb.transform([reuters.categories(i) for i in test_ids])
vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2)
X_train, X_test = vec.fit_transform(reuters.raw(i) for i in train_ids), vec.transform(reuters.raw(i) for i in test_ids)
```

```python
clf = OneVsRestClassifier(LinearSVC(C=1)).fit(X_train, Y_train)   # ラベルごとに 2 値分類器を学習
pred = clf.predict(X_test)
print("micro-F1:", round(f1_score(Y_test, pred, average="micro"), 3),
      " macro-F1:", round(f1_score(Y_test, pred, average="macro", zero_division=0), 3),
      " 完全一致率:", round((pred == Y_test).all(axis=1).mean(), 3))
for i in range(3):
    print("正解:", mlb.inverse_transform(Y_test[i:i + 1])[0], " 予測:", mlb.inverse_transform(pred[i:i + 1])[0])
```

- micro-F1 は頻出ラベルに、macro-F1 は稀なラベルに引っ張られる。両者の差が大きいのは、稀なラベルが当たっていない証拠。
- BERT でやる場合は `AutoModelForSequenceClassification(..., problem_type="multi_label_classification")` とし、ラベルを float の 0/1 ベクトルで与える (損失が BCE になる)。

## 学習した分類器を保存・読み込みする {save_sklearn_model}

学習済みの scikit-learn の分類器 (前処理込み) をファイルに保存して、別のノートブックやアプリで使いたい。

```python
import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

cats = ["rec.sport.baseball", "sci.space", "comp.graphics"]
train = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))
pipe = make_pipeline(TfidfVectorizer(stop_words="english", sublinear_tf=True), LogisticRegression(max_iter=1000))
pipe.fit(train.data, train.target)

joblib.dump({"model": pipe, "labels": train.target_names}, "classifier.joblib", compress=3)
```

```python
import os
import sklearn

bundle = joblib.load("classifier.joblib")
model, labels = bundle["model"], bundle["labels"]
for text in ["The rocket reached orbit after launch.", "He hit a home run in the ninth inning."]:
    print(labels[model.predict([text])[0]], "|", text)
print(f"{os.path.getsize('classifier.joblib') / 1e6:.1f} MB, scikit-learn {sklearn.__version__}")
```

- ベクトライザと分類器を `Pipeline` にまとめて保存するのが鉄則。別々に保存すると、語彙と重みの対応がずれる事故が起きる。
- pickle 形式なので、読み込む側の scikit-learn のバージョンを揃える。また、信頼できないファイルを `load` してはいけない (任意コードが実行されうる)。
- Colab ではランタイムが切れるとファイルが消える。Google Drive に保存するか、ダウンロードしておく。

## 文埋め込みを特徴量にして少ない学習データで分類する {embedding_classifier}

ラベル付きデータが各クラス数十件しかない。事前学習済みの文埋め込みを特徴量にして分類器を学習したい。

```python
import io, tarfile, urllib.request
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

df = load_livedoor().sample(1800, random_state=0).reset_index(drop=True)
model = SentenceTransformer("intfloat/multilingual-e5-small")
emb = model.encode(["query: " + t for t in df["title"]], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
print(emb.shape)
```

```python
pool_idx, test_idx = train_test_split(df.index, test_size=600, stratify=df["label"], random_state=0)
tfidf = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True).fit(df["title"])
X_tfidf = tfidf.transform(df["title"])

for n_per_class in [4, 16, 64]:
    train_idx = df.loc[pool_idx].groupby("label").sample(n_per_class, random_state=0).index
    acc_emb = LogisticRegression(C=10, max_iter=1000).fit(emb[train_idx], df["label"][train_idx]).score(emb[test_idx], df["label"][test_idx])
    acc_tfidf = LogisticRegression(C=10, max_iter=1000).fit(X_tfidf[train_idx], df["label"][train_idx]).score(X_tfidf[test_idx], df["label"][test_idx])
    print(f"各クラス {n_per_class:3} 件: 埋め込み + LR = {acc_emb:.3f}   TF-IDF + LR = {acc_tfidf:.3f}")
```

- 学習データが少ないほど、事前学習済みの埋め込みの効果が大きい。データが増えると TF-IDF との差は縮まる。
- 埋め込みは 1 回計算して保存しておけば、分類器の学習は一瞬で終わる。モデル本体を更新しないので GPU も不要。
- さらに精度を上げたいなら、対照学習で埋め込みモデル自体を少数例で微調整する SetFit がある。

## ラベル付きデータなしで分類する (ゼロショット分類、NLI) {zero_shot_nli}

学習データがまったくない状態で、自分で決めたカテゴリにテキストを分類したい。

```python
from transformers import PreTrainedTokenizerFast, pipeline

name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
# transformers 5 系の DebertaV2Tokenizer は全角の記号や英数字を [UNK] にしてしまう (issue #49622) ので、tokenizer.json から直接読む
tokenizer = PreTrainedTokenizerFast.from_pretrained(name)
clf = pipeline("zero-shot-classification", model=name, tokenizer=tokenizer)

labels = ["スポーツ", "経済", "テクノロジー", "料理", "政治"]
texts = ["大谷翔平が今季 50 本目のホームランを放った。",
         "トヨタ自動車の純利益は前年比 20% 増加した。",
         "新しい言語モデルは従来より少ない計算量で高い性能を達成した。",
         "玉ねぎを飴色になるまで炒めてから、カレールーを加えます。"]
for text in texts:
    r = clf(text, labels, hypothesis_template="この文は{}に関するものです。")
    print(f"{r['labels'][0]}({r['scores'][0]:.2f}) / {r['labels'][1]}({r['scores'][1]:.2f})  {text}")
```

```python
# 複数ラベルが同時に成り立つ場合 (各ラベルを独立に判定)
r = clf("新型スマートフォンの販売が好調で、アップルの株価が上昇した。", labels,
        hypothesis_template="この文は{}に関するものです。", multi_label=True)
print([(l, round(s, 2)) for l, s in zip(r["labels"], r["scores"])])
```

- 仕組み: 「この文は◯◯に関するものです」という仮説が入力文から含意されるかを NLI モデルに判定させ、含意の確率が最も高いラベルを選ぶ。
- `hypothesis_template` は対象の言語で書く。言い回しによって精度が変わるので、いくつか試す。
- ラベル数だけモデルを実行するので遅い。大量に処理するなら、次項の埋め込みによる方法か、ゼロショットの結果を教師にして小さなモデルを学習する。

## ラベルの説明文との類似度でゼロショット分類する {zero_shot_embedding}

ゼロショット分類を高速に行いたい。テキストと「ラベルの説明文」を同じ空間に埋め込み、最も近いラベルを選ぶ。

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-small")

label_desc = {
    "スポーツ": "野球やサッカーなどスポーツの試合や選手に関する話題",
    "経済": "金融、株価、企業業績、景気など経済に関する話題",
    "テクノロジー": "AI、ソフトウェア、スマートフォンなど技術に関する話題",
    "料理": "レシピ、食材、調理方法など料理に関する話題",
    "政治": "選挙、国会、政府、外交など政治に関する話題",
}
label_emb = model.encode(["query: " + d for d in label_desc.values()], normalize_embeddings=True)

texts = ["大谷翔平が今季 50 本目のホームランを放った。", "日銀は政策金利を据え置くことを決定した。",
         "新しい言語モデルは従来より少ない計算量で高い性能を達成した。", "玉ねぎを飴色になるまで炒めてから、カレールーを加えます。",
         "首相は来月の衆議院解散を表明した。"]
sims = model.encode(["query: " + t for t in texts], normalize_embeddings=True) @ label_emb.T
for text, row in zip(texts, sims):
    order = np.argsort(-row)
    names = list(label_desc)
    print(f"{names[order[0]]}({row[order[0]]:.2f}) / {names[order[1]]}({row[order[1]]:.2f})  {text}")
```

- ラベル名だけでなく、少し長い説明文や典型例を埋め込むと精度が上がる。各ラベルに数件の例文があるなら、その平均ベクトル (セントロイド) を使う。
- 1 位と 2 位のスコア差が小さいものは「分類不能」として人手に回す、という運用がしやすい。

## 言い換えたデータを足して少ない学習データを補う (データ拡張) {data_augmentation gpu}

ラベル付きデータが各クラス 10 件しかない。学習データを機械的に言い換えて水増しすれば分類の精度が上がるのか、同義語の置換 (EDA)、逆翻訳、LLM による言い換えを比べて確かめたい。

(T4 の GPU ランタイムで、全体で 10 分程度かかる。)

```python
!pip install -q fugashi unidic-lite
```

```python
import io, random, tarfile, urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_livedoor():
    """livedoor ニュースコーパス (CC BY-ND 2.1 JP。改変禁止なので、言い換えたデータは手元の実験だけに使い、配布しない)"""
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
pool, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=0)
# タイトルだけで 9 クラスに分類する。学習データの選び方 (乱数の種) を 5 通り変えて、結果のばらつきも見る
SEEDS, N = range(5), 10
real3x = {s: pool.groupby("label").sample(3 * N, random_state=s) for s in SEEDS}   # 比較用: 本物のデータを 3 倍 (各クラス 30 件) 集めた場合
small = {s: d.groupby("label").head(N) for s, d in real3x.items()}                  # 各クラス 10 件 (計 90 件) の学習データ
used = sorted(set().union(*[set(d.index) for d in small.values()]))                 # 5 通りのどれかで使う文 = 言い換える対象
print("学習データ:", len(small[0]), "件 × 5 通り / 言い換える文:", len(used), "件 / テスト:", len(test), "件")
augs = {}   # 手法名 → {文の番号: [言い換え 1, 言い換え 2]}
```

```python
# (1) EDA: 同義語の置換・同義語の挿入・語の入れ替え・語の削除のどれか 1 つを、文の約 1 割の語に施す
import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)   # 日本語 WordNet (NICT) を含む多言語 WordNet
from fugashi import Tagger
from nltk.corpus import wordnet as wn

tagger = Tagger()

def synonyms(word):
    return sorted({l.replace("_", "") for s in wn.synsets(word, lang="jpn") for l in s.lemma_names("jpn")} - {word})

def eda_once(title, rng, alpha):
    tokens = list(tagger(title))
    words = [w.surface for w in tokens]
    nouns = [i for i, w in enumerate(tokens) if w.feature.pos1 == "名詞" and w.feature.pos2 != "固有名詞" and synonyms(w.surface)]
    n = max(1, round(alpha * len(words)))
    op = rng.choice(["置換", "挿入", "入替", "削除"])
    w = words[:]
    if op == "置換":       # 名詞を WordNet の同義語に置き換える (活用する語は形が崩れるので名詞だけ)
        for i in rng.sample(nouns, min(n, len(nouns))):
            w[i] = rng.choice(synonyms(w[i]))
    elif op == "挿入":     # どれかの名詞の同義語を、ランダムな位置に入れる
        for _ in range(n if nouns else 0):
            w.insert(rng.randrange(len(w) + 1), rng.choice(synonyms(words[rng.choice(nouns)])))
    elif op == "入替":     # ランダムな 2 語を入れ替える
        for _ in range(n):
            i, j = rng.sample(range(len(w)), 2)
            w[i], w[j] = w[j], w[i]
    else:                  # 各語を確率 alpha で消す
        w = [x for x in w if rng.random() > alpha] or [rng.choice(w)]
    return "".join(w), op

def eda(title, rng, alpha=0.1, tries=10):
    for _ in range(tries):        # 何も消えなかった・置き換える名詞がなかったなどで元の文と同じになったら、やり直す
        new, op = eda_once(title, rng, alpha)
        if new != title:
            break
    return new, op

rng = random.Random(0)
eda_out = {i: [eda(pool.title[i], rng) for _ in range(2)] for i in used}
augs["EDA"] = {i: [t for t, _ in v] for i, v in eda_out.items()}
for i in used[:6]:
    print(pool.title[i], "\n   →", " / ".join(f"[{op}] {t}" for t, op in eda_out[i]))
```

```python
# (2) 逆翻訳: 日本語 → 英語 → 日本語と往復させる (翻訳の章の「逆翻訳で擬似対訳データを作る」と同じ考え方)
# Colab の T4 で 3 分程度
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BF16 = torch.cuda.get_device_capability()[0] >= 8      # T4 は bf16 が使えないので float16
DTYPE = torch.bfloat16 if BF16 else torch.float16
name = "facebook/nllb-200-distilled-600M"              # Meta の NLLB (CC BY-NC 4.0、非商用)
mt_tok = AutoTokenizer.from_pretrained(name)
mt = AutoModelForSeq2SeqLM.from_pretrained(name, dtype=DTYPE).to("cuda").eval()

@torch.no_grad()
def translate(texts, src, tgt, n=1, num_beams=4, batch_size=64):
    mt_tok.src_lang = src
    out = []
    for i in range(0, len(texts), batch_size):
        enc = mt_tok(texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
        gen = mt.generate(**enc, forced_bos_token_id=mt_tok.convert_tokens_to_ids(tgt), num_beams=num_beams,
                          num_return_sequences=n, max_length=128)
        out += mt_tok.batch_decode(gen, skip_special_tokens=True)
    return out

titles = [pool.title[i] for i in used]
en = translate(titles, "jpn_Jpan", "eng_Latn", num_beams=1)   # 行きは貪欲法で軽くする
ja = translate(en, "eng_Latn", "jpn_Jpan", n=2)          # ビームの上位 2 つを言い換えにする
augs["逆翻訳"] = {i: ja[2 * k:2 * k + 2] for k, i in enumerate(used)}
for k in range(6):
    print(titles[k], "\n   →", en[k], "\n   →", " / ".join(augs["逆翻訳"][used[k]]))
del mt; torch.cuda.empty_cache()
```

```python
# (3) LLM による言い換え: 指示チューニング済みの 3B モデルに見出しを書き換えさせる
# Colab の T4 で 2 分程度
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM

name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"    # SB Intuitions (MIT ライセンス)
tok = AutoTokenizer.from_pretrained(name)
print("差し替える前:", tok.tokenize("みのもんた引退、新卒初任給"))
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tok.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
print("差し替えた後:", tok.tokenize("みのもんた引退、新卒初任給"))
tok.padding_side = "left"
lm = AutoModelForCausalLM.from_pretrained(name, dtype=DTYPE).to("cuda").eval()
print(f"GPU メモリ: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

def prompt(title):
    msg = f"次のニュース記事の見出しを、意味を変えずに別の言い方で書き換えてください。書き換えた見出しだけを 1 行で出力してください。\n\n見出し: {title}"
    return tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=False)

def clean(s):
    s = s.strip().split("\n")[0].strip()
    return s[1:-1] if s.startswith("「") and s.endswith("」") else s

@torch.no_grad()
def paraphrase(texts, n=2, batch_size=64):
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok([prompt(t) for t in texts[i:i + batch_size]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = lm.generate(**enc, max_new_tokens=40, do_sample=True, temperature=0.8, top_p=0.9, num_return_sequences=n,
                          pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        out += [clean(s) for s in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return out

torch.manual_seed(0)
para = paraphrase(titles)
augs["LLM"] = {i: para[2 * k:2 * k + 2] for k, i in enumerate(used)}
for k in range(6):
    print(titles[k], "\n   →", " / ".join(augs["LLM"][used[k]]))
del lm; torch.cuda.empty_cache()
```

```python
# 拡張なし・各拡張・本物 3 倍で学習し、同じテストデータで正解率を比べる (分類器は TF-IDF + LR と、文埋め込み + LR)
# 5 通り × 5 条件 × 2 分類器 = 50 回の学習を CPU で行うので、Colab で 3 分程度
from scipy import stats
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

st = SentenceTransformer("intfloat/multilingual-e5-small", device="cuda")
all_texts = sorted(set(test["title"]) | set(pool["title"]) | {t for a in augs.values() for v in a.values() for t in v})
vec = dict(zip(all_texts, st.encode(["query: " + t for t in all_texts], normalize_embeddings=True, batch_size=256)))
embed = lambda texts: np.stack([vec[t] for t in texts])
X_test = embed(test["title"])

def evaluate(texts, labels):
    tfidf = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True), LogisticRegression(C=10, max_iter=1000))
    acc1 = tfidf.fit(texts, labels).score(test["title"], test["label"])
    acc2 = LogisticRegression(C=10, max_iter=1000).fit(embed(texts), labels).score(X_test, test["label"])
    return acc1, acc2

res = {}
for s in SEEDS:
    d = small[s]
    res["拡張なし", s] = evaluate(list(d["title"]), list(d["label"]))
    for method, a in augs.items():   # 元の 90 件 + 言い換え 180 件
        res[method, s] = evaluate(list(d["title"]) + [t for i in d.index for t in a[i]],
                                  list(d["label"]) + [l for l in d["label"] for _ in range(2)])
    res["本物 3 倍", s] = evaluate(list(real3x[s]["title"]), list(real3x[s]["label"]))

rows = []
for method in ["拡張なし", *augs, "本物 3 倍"]:
    acc, base = np.array([res[method, s] for s in SEEDS]), np.array([res["拡張なし", s] for s in SEEDS])
    row = {"手法": method}
    for j, clf_name in enumerate(["TF-IDF", "埋め込み"]):
        diff = acc[:, j] - base[:, j]                                   # 同じ種どうしの差 (対応のある比較)
        half = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
        row[f"{clf_name} 正解率"] = f"{acc[:, j].mean():.3f} ± {acc[:, j].std(ddof=1):.3f}"
        row[f"{clf_name} 差 [95%CI]"] = f"{diff.mean():+.3f} [{diff.mean() - half:+.3f}, {diff.mean() + half:+.3f}]"
    rows.append(row)
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).to_string(index=False))
```

```python
# 拡張でラベルが変わっていないか: 学習用プールのうち言い換えの対象でない文 (約 5500 件) で学習した分類器を「審判」にして、
# 元の文と言い換えた文の予測ラベルを調べる。元の文を審判の学習に入れると、答えを覚えているので比較にならない
# (この審判は分析のためだけのもの。実際に少量のデータしかない場面では使えない)
rest = pool.drop(used)
judge = LogisticRegression(C=10, max_iter=1000).fit(embed(rest["title"]), rest["label"])
gold = np.array([pool.label[i] for i in used])
pred_orig = judge.predict(embed(titles))
print(f"元の文     : 審判の予測が元のラベルと一致 {np.mean(pred_orig == gold):.1%}")
for method, a in augs.items():
    aug_texts = [t for i in used for t in a[i]]
    pred = judge.predict(embed(aug_texts))
    cos = (embed(aug_texts) * np.repeat(embed(titles), 2, axis=0)).sum(axis=1)
    print(f"{method:6}: 審判の予測が元のラベルと一致 {np.mean(pred == np.repeat(gold, 2)):.1%} / 元の文との埋め込みの類似度 (平均) {cos.mean():.3f}")

# 元の文は正しく分類されるのに、言い換えると別のクラスになる例
for method, a in augs.items():
    print(f"\n--- {method}")
    shown = 0
    for k, i in enumerate(used):
        p = judge.predict(embed(a[i][:1]))[0]
        if pred_orig[k] == gold[k] and p != gold[k]:
            print(f"[{gold[k]} → {p}] {titles[k]}\n    → {a[i][0]}")
            shown += 1
        if shown == 4:
            break
```

- EDA は Wei and Zou (EMNLP-IJCNLP 2019) が提案した 4 つの操作 (同義語の置換・挿入、語の入れ替え・削除)。英語では同義語を WordNet から取る。日本語では分かち書きしてから、日本語 WordNet の同義語を使う。ここでは活用で形が崩れないように、同義語を使う操作は名詞だけにした。それでも「CM」が長さの単位 (センチメートル) を表す「糎」に置き換わるなど、語義の取り違えは避けられない。
- 逆翻訳はもともと機械翻訳の学習データを増やす方法 (Sennrich ら, ACL 2016。機械翻訳の章の「逆翻訳で擬似対訳データを作る」)。分類では、往復させた文を元の文の言い換えとして使う (Xie らの UDA, NeurIPS 2020 など)。翻訳では人が書いた文が訳文側に残るが、ここでは往復した機械訳そのものを学習に使うので、誤訳がそのまま混ざる。
- この実験 (タイトルで 9 クラス、各クラス 10 件、5 通りの学習データ) では、どの拡張も正解率を上げなかった。EDA と LLM の言い換えは、差の 95% 信頼区間が 0 をまたぐか 0 以下 (平均で最大 1.1 ポイント減) で、逆翻訳は 3〜5 ポイント下がった。いっぽう、本物のデータを 3 倍にすると 6.5〜9 ポイント上がる。1 件を言い換えた 2 件は、本物の 2 件の代わりにならない (データを増やす効果はこの章の「学習データを増やすと精度がどう変わるか調べる (学習曲線)」も参照)。
- 差が小さいときは 1 回の結果で判断せず、同じ学習データ (同じ種) で拡張の有無を比べ、その差の平均と信頼区間を見る (対応のある比較)。この実験でも、種による正解率の標準偏差 (TF-IDF で 0.01〜0.02) は、EDA や LLM の言い換えによる差と同じくらいか、それより大きい。なお、この信頼区間は学習データの選び方のばらつきだけを表す。テストデータと言い換え (EDA の乱数、LLM のサンプリング) は 1 回しか作っていない。
- 審判の分類器 (言い換えの対象でない文で学習) で見ると、元の文の 83% が元のラベルに分類されるのに対し、EDA は 79%、LLM は 71%、逆翻訳は 53% だった。往復翻訳では人名が落ちたり別人 (「ツーサ・エリカ」) になったりして、見出しの意味が壊れている。LLM の言い換えは意味がおおむね保たれていても、話題の重心が移って (「映画大コケか」→「主演の韓国映画、予想外の低成績」) 別のカテゴリらしくなることがある。
- Longpre ら (Findings of EMNLP 2020) は、EDA と逆翻訳が、ファインチューニングした事前学習済み Transformer の精度を安定して上げないと報告している。ここでは埋め込みモデルを固定してロジスティック回帰を学習しており、設定は違う。この実験に限っていえば、埋め込みを使った分類器のほうが、拡張による低下が大きかった。
- transformers 5 系では、sarashina2.2 のトークナイザーが本来の SentencePiece (Unigram) と違う分割になる (上の出力では「みのもんた」が「み / のも / ん / た」、「初任給」が 1 字ずつに分かれる) ので、上のようにモデル部分を Unigram に戻して使う。直したあとも、言い換えに英語が混ざる (「quite the financier」「Incident」) ことがある。言い換えが元の文から離れすぎていないか (埋め込みの類似度) や、ラベルが変わっていないかを確かめてから使う。機械翻訳の章の「擬似対訳データを往復翻訳の一致度でふるいにかける」と同じ発想で、類似度の低いものを捨てる方法もある (ここでは試していない)。
- データは CC BY-ND (改変禁止) なので、言い換えたデータを公開・配布してはいけない。NLLB は CC BY-NC 4.0 (非商用)、sarashina2.2 は MIT、multilingual-e5-small は MIT、日本語 WordNet は NICT の独自ライセンス (著作権表示を残せば利用・改変・再配布可)。

## ラベルを付ける文書を賢く選んで、少ないラベルで精度を上げる (能動学習) {active_learning}

アノテーションの予算が限られているので、ランダムに選ぶより「モデルが迷っている文書」から優先してラベルを付け、少ない件数で精度を上げたい。livedoor ニュースコーパスの記事タイトルで、ラベルを少しずつ増やすシミュレーションをして確かめる。

```python
import io, tarfile, urllib.request
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def load_livedoor():
    """livedoor ニュースコーパス (株式会社ロンウイット配布、記事は CC BY-ND 2.1 JP) を DataFrame にする"""
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

df = load_livedoor().sample(4000, random_state=0).reset_index(drop=True)   # CPU で回せるよう 4000 件に減らす
y = df["label"].to_numpy()
pool_idx, test_idx = train_test_split(np.arange(len(df)), test_size=1000, stratify=y, random_state=0)
print("プール (ラベルを付ける候補):", len(pool_idx), " テスト:", len(test_idx))

# 特徴量は 2 種類。ラベルを使わないので、最初に一度だけ作っておけばよい
tfidf = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True, min_df=2)   # 1 回しか出ない n-gram は捨てて軽くする
X_tfidf = tfidf.fit(df["title"].iloc[pool_idx]).transform(df["title"])   # 語彙と IDF はプールだけから作る
model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")   # MIT ライセンス
# ダウンロードとモデルの読み込みを含めて、2 コアで 2〜4 分
X_emb = model.encode(["query: " + t for t in df["title"]], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
print(X_tfidf.shape, X_emb.shape)
```

```python
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

def make_clf():
    # 何百回も学習し直すので、速い liblinear (クラスごとの 2 値分類 = one-vs-rest) を使う
    return OneVsRestClassifier(LogisticRegression(C=10, solver="liblinear"))

def uncertainty(proba, strategy):
    """値が大きいほど「モデルが迷っている」"""
    p = np.sort(proba, axis=1)[:, ::-1]
    if strategy == "least_confident":
        return 1 - p[:, 0]                                  # 最大の確率が低い
    if strategy == "margin":
        return -(p[:, 0] - p[:, 1])                         # 1 位と 2 位の確率の差が小さい
    if strategy == "entropy":
        return -(proba * np.log(proba + 1e-12)).sum(axis=1) # 確率の分布が平ら

def simulate(X, strategy, seed, init=None, n_init=20, batch=40, budget=420):
    """ラベル付きを n_init 件から batch 件ずつ増やし、各段階のテスト正解率を返す。
    ラベルは本当は人が付けるが、ここでは正解を隠しておいて、選ばれたものだけ見せる"""
    rng = np.random.default_rng(seed)
    labeled = list(rng.choice(pool_idx, n_init, replace=False) if init is None else init)
    accs = []
    while True:
        clf = make_clf().fit(X[labeled], y[labeled])
        accs.append(clf.score(X[test_idx], y[test_idx]))
        if len(labeled) >= budget:
            return np.array(accs), labeled
        rest = np.setdiff1d(pool_idx, labeled)
        if strategy == "random":
            new = rng.choice(rest, batch, replace=False)
        else:
            new = rest[np.argsort(-uncertainty(clf.predict_proba(X[rest]), strategy))[:batch]]
        labeled += list(new)

strategies = ["random", "least_confident", "margin", "entropy"]
seeds = range(10)
sizes = np.arange(20, 421, 40)   # 20, 60, ..., 420 件
at = {n: i for i, n in enumerate(sizes)}
# 2 特徴量 × 4 方法 × 10 種で、分類器を 880 回学習し直す。2 コアで 1〜4 分 (Colab でも数分以内)
res = {(f, s): np.array([simulate(X, s, seed)[0] for seed in seeds])
       for f, X in [("emb", X_emb), ("tfidf", X_tfidf)] for s in strategies}

for f, X in [("emb", X_emb), ("tfidf", X_tfidf)]:
    full = make_clf().fit(X[pool_idx], y[pool_idx]).score(X[test_idx], y[test_idx])
    print(f"[{f}] プール全件 ({len(pool_idx)} 件) で学習したときの正解率: {full:.3f}")
    print(pd.DataFrame({s: [f"{res[f, s][:, i].mean():.3f} ± {res[f, s][:, i].std():.3f}" for i in [at[20], at[100], at[220], at[420]]]
                        for s in strategies}, index=[f"{n} 件" for n in [20, 100, 220, 420]]).to_string(), "\n")
```

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, f in zip(axes, ["emb", "tfidf"]):
    for s in strategies:
        m, sd = res[f, s].mean(axis=0), res[f, s].std(axis=0)
        ax.plot(sizes, m, "o-", ms=3, label=s)
        ax.fill_between(sizes, m - sd, m + sd, alpha=0.15)
    ax.set_title(f); ax.set_xlabel("labeled examples"); ax.grid(alpha=0.3)
axes[0].set_ylabel("test accuracy"); axes[0].legend(); plt.show()

# 同じ種 (= 同じ最初の 20 件) どうしで、ランダムとの差を取る
for f in ["emb", "tfidf"]:
    for s in strategies[1:]:
        d = res[f, s] - res[f, "random"]
        print(f"[{f}] {s:15}", "  ".join(f"{n}件: {d[:, i].mean():+.3f}±{d[:, i].std():.3f} ({(d[:, i] > 0).sum()}/10 勝)"
                                          for i, n in [(at[100], 100), (at[220], 220), (at[420], 420)]))
```

```python
from sklearn.neighbors import NearestNeighbors

# どんな文書が選ばれたか: 種 0 で 420 件まで選んだ集合を比べる
nn = NearestNeighbors(n_neighbors=11, metric="cosine").fit(X_emb[pool_idx])
def typicality(idx):
    """近い 10 件との平均コサイン類似度。低いほど周りに似た文書がない (外れ値寄り)"""
    dist, _ = nn.kneighbors(X_emb[idx])
    return 1 - dist[:, 1:].mean(axis=1)

rows = {}
for s in strategies:
    _, chosen = simulate(X_emb, s, seed=0)
    chosen = np.array(chosen[20:])   # 最初のランダムな 20 件を除いた、方法が選んだ 400 件
    cnt = pd.Series(y[chosen]).value_counts()
    rows[s] = {"典型度 (平均)": typicality(chosen).mean(), "タイトルの長さ": df["title"].iloc[chosen].str.len().mean(),
               "最多クラスの件数": f"{cnt.iloc[0]} ({cnt.index[0]})", "最少クラスの件数": f"{cnt.iloc[-1]} ({cnt.index[-1]})"}
rows["(プール全体)"] = {"典型度 (平均)": typicality(pool_idx).mean(), "タイトルの長さ": df["title"].iloc[pool_idx].str.len().mean()}
display(pd.DataFrame(rows).T.round(3))

_, chosen = simulate(X_emb, "entropy", seed=0, budget=60)
print("エントロピーで最初に選ばれた 40 件 (の一部):")
for i in chosen[20:30]:
    print(f"  [{y[i]}] {df['title'][i]}")
```

```python
from sklearn.cluster import KMeans

# コールドスタート: 最初の 20 件をランダムに選ぶか、クラスタリングで代表的な文書を選ぶか
def kmeans_init(seed, k=20):
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(X_emb[pool_idx])
    return [pool_idx[np.argmin(((X_emb[pool_idx] - c) ** 2).sum(axis=1))] for c in km.cluster_centers_]   # 重心に最も近い文書

for name, make in [("random", lambda seed: np.random.default_rng(seed).choice(pool_idx, 20, replace=False)), ("k-means", kmeans_init)]:
    accs, ncls = [], []
    for seed in seeds:
        init = make(seed)
        ncls.append(len(set(y[init])))
        accs.append(simulate(X_emb, "entropy", seed, init=init, budget=100)[0])
    accs = np.array(accs)
    print(f"最初の 20 件 = {name:8}: 含まれるクラス数 {np.mean(ncls):.1f}/9   正解率 20 件 {accs[:, 0].mean():.3f}±{accs[:, 0].std():.3f}"
          f"   (その後エントロピーで) 100 件 {accs[:, -1].mean():.3f}±{accs[:, -1].std():.3f}")
```

- 能動学習 (active learning) は、モデルに「次にラベルを付けると役立ちそうな文書」を選ばせ、人がそれにラベルを付けてモデルを学習し直す、を繰り返す。ここでは正解ラベルを隠しておき、選ばれた文書のラベルだけを使うことで、人の作業をシミュレーションしている。データと特徴量は、この章の「文埋め込みを特徴量にして少ない学習データで分類する」と同じ構成にした。livedoor ニュースコーパスは CC BY-ND 2.1 JP (改変禁止)、`intfloat/multilingual-e5-small` は MIT ライセンス。
- 不確実性サンプリングは Lewis と Gale (1994, SIGIR) がテキスト分類で提案した。確信度・マージン・エントロピーなどの選び方の整理は Settles のサーベイ (2009) にある。
- 種 10 個の平均で、マージン (1 位と 2 位の確率の差) で選ぶ方法は 420 件の時点でランダムより正解率が高く、同じ種どうしの差は埋め込みで +0.022 ± 0.017 (10 種中 9 勝)、TF-IDF で +0.026 ± 0.015 (10 勝) だった。100 件の時点では勝ち負けが半々で、差は種によるばらつきに埋もれている。ラベルが少ないうちはモデルの確率が当てにならないためと考えられる。
- 確信度の低さ (least confident) やエントロピーで選ぶと、序盤はランダムより悪かった (100 件で、埋め込みのエントロピーは -0.180、TF-IDF でも -0.129、どちらも 10 種すべてで負け)。9 クラスでは、どのクラスにも当てはまりそうな曖昧な文書ほど確率が平らになり、優先して選ばれやすいためと考えられる。マージンは上位 2 クラスの差だけを見るので、この影響を受けにくいと考えられる。
- 選ばれたラベル付きデータは母集団の縮図ではない。種 0 の 1 回だけの比較だが、エントロピーで選んだ 400 件では smax が 15 件しかなかった (ランダムでは最少のクラスでも 30 件)。確信度やエントロピーで選んだ集合はタイトルもやや短い (プール平均 37.6 文字に対し 34〜35 文字)。この集合で精度を測ると偏るので、評価用のデータは必ず別にランダムに取っておく。また、能動学習で集めたデータが別のモデルの学習でもランダムより良いとは限らないことを Lowell ら (2019, EMNLP) が報告している。
- 最初のモデルがないとき (コールドスタート) は、不確実性を計算できないので最初の数十件は別の方法で選ぶ。k-means の各クラスタの代表を選ぶと、最初の 20 件に含まれるクラス数は平均 7.9 から 8.8 に増えた。ただし正解率の差 (20 件で 0.319 と 0.306) は種によるばらつきの範囲内だった。
- ランダムに選んだときの曲線は、この章の「学習データを増やすと精度がどう変わるか調べる (学習曲線)」と同じ考え方の曲線である。能動学習はこの曲線を左に (少ない件数側に) ずらす工夫だが、ここではプール全件 (3000 件) で学習した正解率 (埋め込み 0.813、TF-IDF 0.838) には 420 件ではまだ届かない。ラベルを付ける代わりにルールでまとめてラベルを付ける方法は、この章の「ルールでラベルを付けて分類器を学習する (弱教師あり学習、Snorkel)」を参照。

## ルールでラベルを付けて分類器を学習する (弱教師あり学習、Snorkel) {weak_supervision}

人手でラベルを付ける代わりに、キーワードや既存のモデルを使った「ラベル付け関数」を何本か書いてラベルを作り、そのラベルで分類器を学習したい。livedoor ニュースコーパスのうち 4 カテゴリ (スポーツ・映画・スマートフォン・独身女性の生活) を使う。

```python
!pip install -q snorkel
```

```python
import io, re, tarfile, urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# livedoor ニュースコーパス (株式会社ロンウイット配布、CC BY-ND 2.1 JP: 改変禁止)。記事は読み込んで使うだけで、改変したものを配布しない
CATS = ["sports-watch", "movie-enter", "smax", "dokujo-tsushin"]
ABSTAIN, SPORTS, MOVIE, SMAX, DOKUJO = -1, 0, 1, 2, 3   # -1 は「判断しない (棄権)」

url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
with urllib.request.urlopen(url) as r:
    tar = tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")
rows = []
for m in tar.getmembers():
    parts = m.name.split("/")
    if m.isfile() and len(parts) == 3 and parts[1] in CATS and parts[2] != "LICENSE.txt":
        lines = tar.extractfile(m).read().decode("utf-8").split("\n")   # 1 行目 URL, 2 行目 日付, 3 行目 タイトル
        rows.append({"y": CATS.index(parts[1]), "title": lines[2], "text": lines[2] + "\n" + "\n".join(lines[3:])[:300]})
df = pd.DataFrame(rows)
# train の正解ラベルは「使わない」前提。正解はラベル付け関数の評価と比較のためだけに見る
train, test = train_test_split(df, test_size=0.3, stratify=df["y"], random_state=0)
train, test = train.reset_index(drop=True), test.reset_index(drop=True)
print(len(train), len(test))
```

```python
from sentence_transformers import SentenceTransformer
from snorkel.labeling import LFAnalysis, PandasLFApplier, labeling_function

def keyword_lf(name, pattern, label):
    """正規表現に当たればそのラベル、当たらなければ棄権するラベル付け関数を作る"""
    rx = re.compile(pattern)
    @labeling_function(name=name)
    def lf(x):
        return label if rx.search(x.text) else ABSTAIN
    return lf

lfs = [
    keyword_lf("sports_kw", r"試合|選手|五輪|サッカー|野球|ゴール|優勝", SPORTS),
    keyword_lf("coach", r"監督", SPORTS),                    # わざと入れた質の悪い関数 (映画の「監督」にも当たる)
    keyword_lf("movie_kw", r"映画|上映|公開|主演|劇場", MOVIE),
    keyword_lf("smax_kw", r"Android|スマホ|スマートフォン|ドコモ|NTT|au|ソフトバンク|アプリ", SMAX),   # au は "beautiful" などの英単語の一部にも当たる
    keyword_lf("dokujo_kw", r"独女|結婚|彼氏|婚活|アラサー|女性", DOKUJO),   # 「独女」は配信元のサイト名 (独女通信) なので、ほぼ答えの漏れ
    keyword_lf("title_bracket", r"^[^\n]*『", MOVIE),        # タイトルに『』があれば作品名とみなす
]

# 既存のモデルもラベル付け関数にできる: タイトルとラベルの説明文の埋め込みの類似度 (ゼロショット)。1 位と 2 位の差が小さいときは棄権
embedder = SentenceTransformer("intfloat/multilingual-e5-small")   # MIT ライセンス
desc = ["野球やサッカーなどスポーツの試合や選手に関する話題", "映画の公開、俳優、監督など映画に関する話題",
        "スマートフォンや携帯電話、モバイル通信に関する話題", "独身女性の恋愛、結婚、仕事、生活に関する話題"]
desc_emb = embedder.encode(["query: " + d for d in desc], normalize_embeddings=True)
for d in (train, test):   # 3,500 件のタイトルの埋め込み。Colab の CPU で 5〜8 分
    sims = embedder.encode(["query: " + t for t in d["title"]], normalize_embeddings=True, batch_size=64) @ desc_emb.T
    top2 = np.sort(sims, axis=1)[:, -2:]
    d["zero_shot"] = np.where(top2[:, 1] - top2[:, 0] > 0.02, sims.argmax(axis=1), ABSTAIN)

@labeling_function()
def zero_shot(x):
    return x.zero_shot
lfs.append(zero_shot)

applier = PandasLFApplier(lfs)
L_train, L_test = applier.apply(train, progress_bar=False), applier.apply(test, progress_bar=False)   # 文書数 × 関数の数の行列
# 説明のために学習データ全件の正解で評価している。本来は train の正解は使えないので、少量の人手のラベル付きの開発データで評価する
# Coverage: 棄権しなかった割合、Overlaps: ほかの関数と重なった割合、Conflicts: ほかの関数と違うラベルを付けた割合
summary = LFAnalysis(L_train, lfs).lf_summary(Y=train["y"].values)
print(summary[["Coverage", "Overlaps", "Conflicts", "Emp. Acc."]].round(3))
print("どれか 1 つでも当たった割合:", round((L_train != ABSTAIN).any(axis=1).mean(), 3))
```

```python
from snorkel.labeling.model import LabelModel, MajorityLabelVoter

# 方法 1: 多数決。同票なら棄権
mv = MajorityLabelVoter(cardinality=4)
# 方法 2: ラベルモデル。関数どうしの一致・不一致のパターンだけから各関数の正解率を推定し、重み付きでまとめる (正解ラベルは使わない)
label_model = LabelModel(cardinality=4, verbose=False)
label_model.fit(L_train, n_epochs=500, lr=0.01, seed=0, progress_bar=False)
print(pd.DataFrame({"推定した正解率": label_model.get_weights(), "実際の正解率": summary["Emp. Acc."]}).round(3))

y_train = train["y"].values
pred_mv = mv.predict(L_train, tie_break_policy="abstain")
pred_lm = label_model.predict(L_train, tie_break_policy="abstain")
both = (pred_mv != ABSTAIN) & (pred_lm != ABSTAIN)
for name, pred in [("多数決", pred_mv), ("ラベルモデル", pred_lm)]:
    c = pred != ABSTAIN
    print(f"{name:7}: 付与率 {c.mean():.3f}  付与した中の正解率 {(pred[c] == y_train[c]).mean():.3f}"
          f"  (両方が付与した {both.sum()} 件での正解率 {(pred[both] == y_train[both]).mean():.3f})")
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True, min_df=2).fit(train["text"])
X_train, X_test = vec.transform(train["text"]), vec.transform(test["text"])

y_test = test["y"].values

def test_pred(mask, labels):   # このセルは Colab の CPU で 1〜2 分
    # 簡単のため、確率的なラベルではなく最も確からしいラベル (棄権しなかった文書だけ) で学習する
    return LogisticRegression(C=10, max_iter=1000).fit(X_train[mask], labels[mask]).predict(X_test)

pred_test_lm = label_model.predict(L_test, tie_break_policy="random")   # 参考: ラベル付け関数だけでテストを分類 (全関数が棄権した文書は当て推量)
hit = (L_test != ABSTAIN).any(axis=1)
print(f"ラベル付け関数 + ラベルモデルでそのまま分類: {(pred_test_lm == y_test).mean():.3f}"
      f"  (どれかの関数が当たった {hit.sum()} 件: {(pred_test_lm[hit] == y_test[hit]).mean():.3f} / 全関数が棄権した {(~hit).sum()} 件: {(pred_test_lm[~hit] == y_test[~hit]).mean():.3f})")
for name, pred in [("多数決", pred_mv), ("ラベルモデル", pred_lm)]:
    p = test_pred(pred != ABSTAIN, pred)
    print(f"{name}のラベルで学習 ({(pred != ABSTAIN).sum()} 件): {(p == y_test).mean():.3f}"
          f"  (全関数が棄権した {(~hit).sum()} 件: {(p[~hit] == y_test[~hit]).mean():.3f})")
print(f"人手のラベルで学習 (全 {len(train)} 件): {(test_pred(np.ones(len(train), bool), y_train) == y_test).mean():.3f}")
for n in [10, 25, 50]:   # 人手のラベルが各クラス n 件しかない場合。選ぶ文書を 5 通り変える
    accs = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        idx = np.concatenate([rng.choice(np.where(y_train == k)[0], n, replace=False) for k in range(4)])
        mask = np.zeros(len(train), bool); mask[idx] = True
        accs.append((test_pred(mask, y_train) == y_test).mean())
    print(f"人手のラベルで学習 (各クラス {n} 件): 平均 {np.mean(accs):.3f}  範囲 {min(accs):.3f}〜{max(accs):.3f}")
print("テスト件数:", len(test))
```

- ラベル付け関数の出力を、関数どうしの一致・不一致から推定した正解率で重み付けしてまとめ、そのラベルで分類器を学習する枠組みは、Ratner ら (2016, NeurIPS (当時 NIPS)) の「データプログラミング」。snorkel の `LabelModel` は、Ratner ら (2019, AAAI) の行列補完による推定を実装しており、正解ラベルを見ずに各関数の正解率を推定する。原論文は確率的なラベルのまま分類器を学習する (ノイズを考慮した損失) が、ここでは簡単のため最も確からしいラベルで学習した。
- 推定した正解率は、わざと入れた質の悪い関数 `coach` (実際 0.384) を 0.429 と低く見積もり、重みを下げた。一方で `title_bracket` (実際 0.718) は 0.890 と高く見積もった。`zero_shot` の 1.000 は `get_weights()` が上限 1.0 で切った値で、実際は 0.966。推定はあくまで目安である。一般に、ラベルモデルは関数どうしが (正解のラベルを固定すれば) 独立だと仮定しており、同じ手がかりを見る関数が何本もあると推定が狂いやすい。
- 多数決とラベルモデルは、両方がラベルを付けた 2,148 件では正解率がほぼ同じ (0.947 と 0.948) だった。ラベルモデルは多数決が同票で棄権した文書にもラベルを付けるので付与率が高い (0.874 → 0.944) が、その分だけ正解率は下がる (0.920)。ラベルモデルが常に多数決より良いわけではないので、両方を試す。
- ラベル付け関数で直接テストデータを分類すると正解率 0.890 で、全関数が棄権した 62 件 (5.9%) は当て推量になり 0.258 だった。そのラベルで学習した TF-IDF の分類器は 0.973 (多数決) / 0.978 (ラベルモデル) で、全関数が棄権した 62 件でも 0.919 / 0.935 だった。分類器はキーワード以外の手がかりも学ぶので、関数が当たらない文書にも汎化する。これが最後に分類器を学習する理由。なお 0.973 と 0.978 の差は、テスト 1,053 件のうち 5 件ほどで、ばらつきの範囲である。
- 人手のラベル全 2,457 件で学習すると 0.988 で、弱教師ありより 0.010 (10 件ほど) 高い。人手のラベルが各クラス 10 件 / 25 件 / 50 件のときは、選ぶ文書を 5 通り変えた平均で 0.952 / 0.969 / 0.978 だった。このデータでは、弱教師ありのラベルは人手の各クラス 25〜50 件 (計 100〜200 件) と同程度である。
- ここでは説明のために、学習データ全件の正解ラベルで Coverage や Emp. Acc. を計算したが、本来は使えない。実際には数百件の人手のラベル付きの開発データを作り、それを見ながら関数を足したり直したりする。この例では同じ程度の件数 (各クラス 50 件) で直接学習しても 0.978 が出るので、少量の人手のラベルで十分に分類できる課題では、ルールを書く手間に見合わない。ラベル付けが高くつく課題や、データが大量に流れてくる課題で効く。
- ラベル付け関数で作ったラベルには誤りが残る。この章の「学習データの中の付け間違えたラベルを見つける (Confident Learning)」の方法で、そうした誤りを洗い出すこともできる。
- snorkel (Apache 2.0) は、README に開発チームが商用の Snorkel Flow に注力すると書かれており、最後のリリースは 0.10.0 (2024 年 2 月)。今の Colab (Python 3.13) では `pip install` で入り、動いた。データの livedoor ニュースコーパスは CC BY-ND 2.1 JP (改変禁止)、埋め込みモデル `intfloat/multilingual-e5-small` は MIT ライセンス。

## 学習データの中の付け間違えたラベルを見つける (Confident Learning) {label_noise}

学習データに紛れ込んだラベルの誤りを、交差検証の予測確率から自動で洗い出したい。livedoor ニュースコーパスにわざと誤りを入れて、どれだけ見つかるか、除くと精度がどう変わるかを確かめる。

```python
!pip install -q fugashi unidic-lite cleanlab
```

```python
import io, tarfile, urllib.request
import numpy as np
import pandas as pd
from fugashi import Tagger
from sklearn.model_selection import train_test_split

def load_livedoor():
    """livedoor ニュースコーパス (CC BY-ND 2.1 JP) を DataFrame にする"""
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
tagger = Tagger("-Owakati")
df["wakati"] = [tagger.parse(t[:2000]).strip() for t in df["text"]]   # Colab の CPU で 1〜2 分
names = sorted(df["label"].unique())
train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=0)
train = train.reset_index(drop=True)
y_true = train["label"].map(names.index).to_numpy()   # 元のラベル (整数 ID)
y_test = test["label"].map(names.index).to_numpy()
print(f"学習 {len(train)} 件 / テスト {len(test)} 件")
```

```python
# まず、同じタイトルの記事が別のラベルで二重に入っていないかを確かめる (矛盾するラベルの典型)
# 記事: livedoor ニュースの各配信元。収集・配布: 株式会社ロンウイット (livedoor ニュースコーパス、CC BY-ND 2.1 JP)
df[df.groupby("title")["label"].transform("nunique") > 1][["label", "title"]]
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

def make_clf():
    return make_pipeline(TfidfVectorizer(token_pattern=r"\S+", min_df=3, max_df=0.5, sublinear_tf=True),
                         LogisticRegression(C=10, max_iter=1000))

cv = StratifiedKFold(5, shuffle=True, random_state=0)
# 付け替え先を決めるため、誤りを入れる前のラベルで交差検証し、各クラスで最も取り違えやすいクラスを求める (Colab の CPU で 2 分程度)
proba_orig = cross_val_predict(make_clf(), train["wakati"], y_true, cv=cv, method="predict_proba", n_jobs=-1)   # 最後のセルでも使う
cm = confusion_matrix(y_true, proba_orig.argmax(axis=1))
np.fill_diagonal(cm, 0)
swap = cm.argmax(axis=1)
print({names[i]: names[j] for i, j in enumerate(swap)})

NOISE_RATE = 0.2   # 誤りの割合。0.1 や 0.4 に変えて試すとよい
rng = np.random.default_rng(0)
flip = rng.random(len(train)) < NOISE_RATE   # 誤りを入れる事例 (テストデータには入れない)
noisy = {
    "系統的": np.where(flip, swap[y_true], y_true),   # 取り違えやすいクラスに付け替える
    "ランダム": np.where(flip, (y_true + rng.integers(1, len(names), len(train))) % len(names), y_true),   # ばらばらなクラスに付け替える
}
print(f"わざと入れた誤り: {flip.sum()} 件 ({NOISE_RATE:.0%})")
```

```python
from cleanlab.filter import find_label_issues

rows, proba, issues = [], {}, {}
for kind, y in noisy.items():
    # 交差検証で、すべての事例について「その事例を学習に使っていないモデル」の予測確率を得る (1 回 2 分程度)
    p = proba[kind] = cross_val_predict(make_clf(), train["wakati"], y, cv=cv, method="predict_proba", n_jobs=-1)
    # (1) 予測と付いたラベルが食い違うものをすべて候補にする (論文の C_confusion。この章の「エラー分析」のやり方)
    c_confusion = np.flatnonzero(p.argmax(axis=1) != y)
    # (2) confident joint の非対角を自分で実装: クラス j のしきい値 = ラベル j が付いた事例での p(j) の平均。
    #     しきい値を超えたクラスのうち確率最大のものが付いたラベルと違えば候補 (filter_by="confident_learning" と同じ考え方)
    thr = np.array([p[y == j, j].mean() for j in range(len(names))])
    over = p >= thr
    cj_offdiag = np.flatnonzero(over.any(axis=1) & (np.where(over, p, -1).argmax(axis=1) != y))
    # (3) cleanlab の既定 (prune by noise rate): confident joint から (付いたラベル, 本当のラベル) の組ごとの誤りの件数を推定し、
    #     組ごとに margin (p(本当のラベル) - p(付いたラベル)) の大きい順にその件数を選ぶ。結果は自信度の低い順に並べる
    issues[kind] = find_label_issues(y, p, filter_by="prune_by_noise_rate", return_indices_ranked_by="self_confidence")
    for method, idx in [("予測 ≠ 付いたラベル (C_confusion)", c_confusion), ("confident joint の非対角 (自作)", cj_offdiag),
                        ("prune by noise rate (cleanlab)", issues[kind])]:
        rows.append({"誤り": kind, "方法": method, "候補数": len(idx),
                     "適合率": flip[idx].mean(), "再現率": flip[idx].sum() / flip.sum()})
pd.DataFrame(rows).round(3)
```

```python
boot_rng = np.random.default_rng(0)
def boot_ci(a, b, n=1000):
    """正解の 0/1 の差のブートストラップ 95% 区間。テスト事例の選び方によるばらつきだけを表す (学習の乱数などは含まない)"""
    d = a.astype(float) - b
    s = [d[boot_rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return f"{d.mean():+.3f} (95% 区間 {np.percentile(s, 2.5):+.3f} 〜 {np.percentile(s, 97.5):+.3f})"

def fit_eval(idx, y):
    return make_clf().fit(train["wakati"].iloc[idx], y[idx]).predict(test["wakati"]) == y_test

# 5 通り学習してテストする (Colab の CPU で 3〜4 分)
all_idx = np.arange(len(train))
ideal = fit_eval(all_idx, y_true)
print(f"誤りを入れる前: テスト正解率 {ideal.mean():.3f}")
res = {}
for kind, y in noisy.items():
    keep = np.setdiff1d(all_idx, issues[kind])
    res[kind] = fit_eval(all_idx, y), fit_eval(keep, y)
    print(f"[{kind}] 誤りを含んだまま {res[kind][0].mean():.3f} → 候補 {len(issues[kind])} 件を除いて {res[kind][1].mean():.3f}"
          f" (残った誤り {(flip[keep]).sum()} 件 / 除いた正しい事例 {(~flip[issues[kind]]).sum()} 件)")
    print("    除いた効果:", boot_ci(res[kind][1], res[kind][0]))
print("誤りを含んだままの差 (ランダム − 系統的):", boot_ci(res["ランダム"][0], res["系統的"][0]))
```

```python
# わざと入れた誤りとは別に、元のデータにもともとある「怪しい」事例を探す (誤りを入れる前のラベルとその交差検証の確率を使う)
# 記事: livedoor ニュースの各配信元。収集・配布: 株式会社ロンウイット (livedoor ニュースコーパス、CC BY-ND 2.1 JP)
pd.set_option("display.max_colwidth", 70)
idx = find_label_issues(y_true, proba_orig, filter_by="prune_by_noise_rate", return_indices_ranked_by="self_confidence")
cand = pd.DataFrame({"title": train["title"], "付いたラベル": [names[i] for i in y_true], "予測": [names[i] for i in proba_orig.argmax(axis=1)],
                     "p(付いたラベル)": proba_orig[np.arange(len(train)), y_true].round(3)}).loc[idx]
print("元のデータの候補:", len(idx), "件")
cand.head(10)
```

- 交差検証の予測確率を使う点は、この章の「モデルが自信満々に間違えた事例を調べる (エラー分析)」と同じ。そこでのやり方にあたる (1) 「予測 ≠ 付いたラベル」は、再現率は高い (系統的 0.916、ランダム 0.989) が、適合率は 0.655 / 0.777 で、系統的では 3 割強、ランダムでは 2 割強の候補がわざと入れた誤りではない。
- Confident Learning は Northcutt, Jiang, Chuang が提案した (arXiv 2019 年、JAIR 70 巻、2021 年)。誤り方が (付いたラベル, 本当のラベル) の組ごとに決まる class-conditional noise を前提に、その件数表 (confident joint) を推定する。論文では (1) を基準となる最も単純な方法 C_confusion、(2) を confident joint の非対角を使う方法、(3) を prune by noise rate と呼ぶ。この実験では (2) は再現率が低め (0.724 / 0.770) で、(3) は適合率と再現率の釣り合いがよかった。`cleanlab` のライセンスは Apache-2.0。
- 実際の付け間違いは似たクラスの取り違えに偏り、手法の前提にも合うので、系統的な誤りを主にした。付け替え先は、誤りを入れる前のデータの交差検証の混同行列で、各クラスが最も間違えられやすいクラスにした (独女通信 ↔ Peachy、Sports Watch ↔ トピックニュースなど)。系統的な誤りはランダムな誤りより見つけにくく (cleanlab の適合率 0.725 対 0.909)、害も大きかった (誤りを含んだままの正解率 0.864 対 0.925)。
- 候補を除いて学習し直すと、系統的な誤りでは 0.864 → 0.920 (+0.056、95% 区間 +0.041〜+0.070) と大きく戻ったが、ランダムな誤りでは 0.925 → 0.937 (+0.012、区間 +0.000〜+0.022) と効果が小さかった。ランダムな誤りは、もともとの害が小さいぶん除いて得られるものも少ない。誤りを入れる前の 0.949 に届かないのは、残った誤り (196 件) と、一緒に除いた正しい事例 (388 件) のためと考えられる。区間はテスト事例のブートストラップで求めたもので (評価指標と統計的検定の章の「ブートストラップ法で評価値の信頼区間を求める」を参照)、テスト事例の選び方によるばらつきだけを表し、1 回の試行の結果である。`NOISE_RATE` を変えて試すとよい。
- 元のデータの候補 (21 件) の最上位は、重複を調べたセルで見つかった記事の kaden-channel 側 (予測 smax) だった。家電チャンネルに入っている新庄剛志の野球の記事 (予測 sports-watch) や、独女通信のプレゼントやキャンペーンの記事 (予測 peachy) も上位にある。このコーパスのラベルは配信元のサイト名なので、多くは付け間違いというより内容が別のサイトに近い境界例である。候補は自動で消さず、人が読んで「直す・除く・そのまま」を決める。重複は完全一致で先に洗い出す。
- 学習に使った事例の確率は過信しているので、必ず交差検証の確率を使う。論文では除いた後にクラスごとに損失を重み付けし直すが、ここでは単に除いて学習し直している。
- この章の「ルールでラベルを付けて分類器を学習する (弱教師あり学習、Snorkel)」で作ったラベルにも誤りが含まれる。このレシピの方法で候補を洗い出して確かめられる。
- livedoor ニュースコーパスは CC BY-ND 2.1 JP (改変禁止)。記事そのものは改変していない (付け替えたのはカテゴリのラベルだけ) が、それでも、ラベルを付け替えたデータを livedoor ニュースコーパスとして配布しない。

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
- ラベルが増えると、普通の教師あり学習との差は縮まり、1 クラス 20 件では逆転している。数十件以上のラベルがあるなら、まずは事前学習済みモデルのファインチューニングや、埋め込み + 分類器 (この章の「文埋め込みを特徴量にして少ない学習データで分類する」) を試す。
- 類似度に文埋め込みを使えば、同じ方法をそのまま適用できる。高次元の埋め込みでもハブは生じる (ハブネス問題)。
- 少数の例から語の集合を広げるブートストラッピングでも、ハブ (一般的な語) を通って無関係な語に広がる「意味ドリフト」が起きる (語彙知識と意味の変化の章の「ブートストラッピングの意味ドリフトを観察して抑える」)。

## BERT をファインチューニングして文書分類する (Trainer) {bert_finetune gpu norun}

事前学習済みの日本語 BERT を自分の分類データでファインチューニングして、高精度な分類器を作りたい。

```python
!pip install -q fugashi unidic-lite datasets evaluate accelerate
```

```python
import io, tarfile, urllib.request
import pandas as pd
from datasets import ClassLabel, Dataset

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
names = sorted(df["label"].unique())
ds = Dataset.from_pandas(df).cast_column("label", ClassLabel(names=names))   # 文字列ラベル → 整数 ID
ds = ds.train_test_split(test_size=0.2, stratify_by_column="label", seed=0)
print(ds)
```

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

model_name = "tohoku-nlp/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(batch):
    return tokenizer(batch["title"], batch["text"], truncation="only_second", max_length=256)

tokenized = ds.map(preprocess, batched=True, remove_columns=["title", "text"])
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(names), id2label=dict(enumerate(names)), label2id={n: i for i, n in enumerate(names)})
```

```python
import evaluate
import numpy as np
from transformers import Trainer, TrainingArguments

accuracy, f1 = evaluate.load("accuracy"), evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    return {**accuracy.compute(predictions=pred, references=labels),
            **f1.compute(predictions=pred, references=labels, average="macro")}

args = TrainingArguments(
    output_dir="out", num_train_epochs=3, learning_rate=3e-5, warmup_steps=50, weight_decay=0.01,
    per_device_train_batch_size=32, per_device_eval_batch_size=64,
    eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="f1",
    fp16=True, logging_steps=50, report_to="none", seed=0,
)
trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["test"],
                  data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
trainer.train()
print(trainer.evaluate())
```

```python
# 保存して pipeline で推論
from transformers import pipeline

trainer.save_model("livedoor-bert"); tokenizer.save_pretrained("livedoor-bert")
clf = pipeline("text-classification", model="livedoor-bert", device=0)
print(clf("新型スマートフォンは有機 EL ディスプレイと高性能カメラを搭載し、来月発売される。"))
```

- T4 GPU で 1 エポック約 2 分。正解率は 95〜97% 程度になる。
- 学習率 (1e-5〜5e-5) とエポック数 (2〜5) が最も効くハイパーパラメータ。小さいデータではシードによるばらつきも大きいので、複数シードで平均をとる。
- 本来は train / dev / test の 3 分割にし、dev でモデル選択、test は最後に 1 回だけ使う。

## 英語のデータだけで学習して日本語を分類する (ゼロショットの言語間転移) {crosslingual_transfer gpu}

日本語のラベル付きデータがないが、同じタスクの英語のデータならたくさんある。多言語モデルを英語のデータだけで微調整して、日本語の分類にそのまま使いたい。自然言語推論 (NLI、含意・中立・矛盾の 3 クラス) を例に、英語のデータを機械翻訳して学習した場合 (translate-train)、日本語のデータで学習した場合、日本語のデータを少しだけ足した場合 (few-shot) と比べる。

```python
import csv, io, urllib.request, zipfile
import pandas as pd

LABELS = ["entailment", "neutral", "contradiction"]   # 3 つのデータで共通のラベル (含意・中立・矛盾)
N_TRAIN = 12000                                       # 各設定の学習件数 (1 エポック)

# 英語: SNLI (Bowman et al. 2015、CC BY-SA 4.0)。画像キャプションを前提文にして、仮説文をクラウドソーシングで書いたもの
hf = "https://huggingface.co/datasets/stanfordnlp/snli/resolve/main/plain_text/"
snli, snli_dev = [d[d["label"] >= 0].assign(label=lambda d: d["label"].map(dict(enumerate(LABELS))))   # -1 はラベルが割れたもの
                  for d in (pd.read_parquet(hf + f"{s}-00000-of-00001.parquet") for s in ["train", "validation"])]

# 英語を機械翻訳した日本語: JSNLI v1.1 (京都大学 黒橋研、CC BY-SA 4.0)。SNLI を機械翻訳したもの。dev はクラウドソーシングで翻訳の質とラベルを確かめてある
url = "https://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi?down=https://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JSNLI/jsnli_1.1.zip&name=JSNLI.zip"
with urllib.request.urlopen(url) as r:
    z = zipfile.ZipFile(io.BytesIO(r.read()))
def read_jsnli(name):
    d = pd.read_csv(z.open(f"jsnli_1.1/{name}"), sep="\t", names=["label", "premise", "hypothesis"], quoting=csv.QUOTE_NONE)
    return d.assign(premise=d["premise"].str.replace(" ", ""), hypothesis=d["hypothesis"].str.replace(" ", ""))   # 分かち書きの空白を除く
jsnli, jsnli_dev = read_jsnli("train_w_filtering.tsv"), read_jsnli("dev.tsv")

# 日本語: JGLUE の JNLI v1.3 (ヤフーと早稲田大学、CC BY-SA 4.0)。日本語の画像キャプション (YJ Captions) から、SNLI とは独立に作ったもの
gh = "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jnli-v1.3/"
jnli, jnli_test = [pd.read_json(gh + f"{s}-v1.3.json", lines=True).rename(columns={"sentence1": "premise", "sentence2": "hypothesis"})
                  for s in ["train", "test"]]

for name, d in [("SNLI", snli), ("JSNLI", jsnli), ("JNLI", jnli), ("JNLI test", jnli_test)]:
    print(f"{name:8} {len(d):7} 件  ラベルの割合:", d["label"].value_counts(normalize=True).round(2).to_dict())
print(jsnli[["premise", "hypothesis", "label"]].head(2).to_string(index=False))
print(jnli[["premise", "hypothesis", "label"]].head(2).to_string(index=False))
```

```python
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)
from tokenizers import normalizers
from transformers.utils import logging

logging.set_verbosity_error()   # 分類用の層を新しく初期化したという報告を出さない

MODEL = "microsoft/mdeberta-v3-base"   # DeBERTaV3 の多言語版。XLM-R と同じ CC100 で事前学習 (MIT ライセンス)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
# transformers 5 系の DebertaV2Tokenizer は、sentencepiece の NFKC 正規化を落とし、全角の記号や英数字 (（、ＡＢＣ、１２３ など) を
# [UNK] にしてしまう (issue #49622)。このモデルには tokenizer.json がないので、正規化の先頭に NFKC を足して元の spm.model の分割にそろえる
tokenizer.backend_tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), tokenizer.backend_tokenizer.normalizer])
print(tokenizer.tokenize("Ｔシャツを着た２人の子ども（左）"))   # [UNK] が出ないことを確かめる
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う

def encode(df):
    ds = Dataset.from_dict({"premise": list(df["premise"]), "hypothesis": list(df["hypothesis"]),
                            "label": [LABELS.index(l) for l in df["label"]]})
    return ds.map(lambda b: tokenizer(b["premise"], b["hypothesis"], truncation=True, max_length=128),
                  batched=True, remove_columns=["premise", "hypothesis"])

def train(df, init=MODEL, epochs=1, lr=3e-5):
    set_seed(0)
    model = AutoModelForSequenceClassification.from_pretrained(
        # 重みが float16 で保存されていて (212 個中 210 個)、既定では float16 のまま読まれるので、float32 で読む
        # (fp16 の混合精度の学習には float32 の重みが必要)
        init, dtype=torch.float32, num_labels=3,
        id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
    args = TrainingArguments(output_dir="out", num_train_epochs=epochs, learning_rate=lr, weight_decay=0.01,
                             warmup_steps=0.1,   # 1 未満は全ステップに対する割合
                             per_device_train_batch_size=32, per_device_eval_batch_size=128, save_strategy="no",
                             logging_strategy="no", report_to="none", bf16=BF16, fp16=not BF16, seed=0)
    trainer = Trainer(model=model, args=args, train_dataset=encode(df), data_collator=DataCollatorWithPadding(tokenizer))
    trainer.train()
    return trainer

EVAL = {"SNLI dev (英語)": snli_dev.sample(2000, random_state=0), "JSNLI dev (SNLI の翻訳)": jsnli_dev, "JNLI test (日本語)": jnli_test}
EVAL_DS = {k: encode(d) for k, d in EVAL.items()}

JNLI_PRED = {}   # 設定ごとの JNLI test の予測 (後でブートストラップと混同行列に使う)

def evaluate(trainer, setting=None):
    row, preds = {}, {}
    for name, ds in EVAL_DS.items():
        preds[name] = trainer.predict(ds).predictions.argmax(-1)
        row[name] = round(accuracy_score(ds["label"], preds[name]), 3)
    gold, pred = EVAL_DS["JNLI test (日本語)"]["label"], preds["JNLI test (日本語)"]
    row["JNLI macro-F1"] = round(f1_score(gold, pred, average="macro"), 3)
    row["JNLI の予測の割合 (含意/中立/矛盾)"] = "/".join(f"{x:.2f}" for x in np.bincount(pred, minlength=3) / len(pred))
    if setting:
        JNLI_PRED[setting] = pred
    return row

print("JNLI test の多数派 (中立) を答え続けたときの正解率:", round((jnli_test["label"] == "neutral").mean(), 3))
```

```python
# 3 つの設定で学習して比べる (Colab の T4 で合計 6〜8 分の見込み)
results = {}
trainer_en = train(snli.sample(N_TRAIN, random_state=0))
results["英語 (SNLI) で学習 = ゼロショット転移"] = evaluate(trainer_en, "zero-shot")
trainer_en.save_model("mdeberta-snli")                     # 後で few-shot の出発点にする
results["機械翻訳 (JSNLI) で学習 = translate-train"] = evaluate(train(jsnli.sample(N_TRAIN, random_state=0)), "translate-train")
results["日本語 (JNLI) で学習"] = evaluate(train(jnli.sample(N_TRAIN, random_state=0)))
pd.set_option("display.width", 200)
pd.DataFrame(results).T
```

```python
# JNLI test の 2508 件をブートストラップして 95% 信頼区間を出す。評価データの抽出によるばらつきだけで、乱数の種によるばらつきは含まない
gold = np.array(EVAL_DS["JNLI test (日本語)"]["label"])
c_zs, c_tt = [(JNLI_PRED[k] == gold).astype(float) for k in ["zero-shot", "translate-train"]]   # 正解なら 1
rng = np.random.default_rng(0)
idx = rng.integers(0, len(gold), (1000, len(gold)))   # 2 つの設定で同じ組を引き直す (対応のあるブートストラップ)
lo, hi = np.percentile(c_zs[idx].mean(1), [2.5, 97.5])
print(f"ゼロショット転移の JNLI 正解率 {c_zs.mean():.3f} (95% 信頼区間 {lo:.3f}〜{hi:.3f})")
lo, hi = np.percentile((c_tt[idx] - c_zs[idx]).mean(1), [2.5, 97.5])
print(f"translate-train − ゼロショット転移 = {c_tt.mean() - c_zs.mean():+.3f} (95% 信頼区間 {lo:+.3f}〜{hi:+.3f})")

# どのラベルを取り違えているか (行が正解、列が予測)
from sklearn.metrics import confusion_matrix
print(pd.DataFrame(confusion_matrix(gold, JNLI_PRED["zero-shot"]), index=LABELS, columns=LABELS))
```

```python
# few-shot: 英語で学習したモデルに、日本語の学習データを少しだけ足して続けて学習する (Colab の T4 で 2〜3 分の見込み)
for k in [100, 1000]:
    few = jnli.sample(k, random_state=1)
    epochs = 10 if k == 100 else 3
    r_cont = evaluate(train(few, init="mdeberta-snli", epochs=epochs))     # 英語 → 日本語 k 件
    r_only = evaluate(train(few, epochs=epochs))                         # 日本語 k 件だけ (英語なし)
    print(f"JNLI {k:4} 件: 英語で学習してから追加 → 正解率 {r_cont['JNLI test (日本語)']:.3f} / macro-F1 {r_cont['JNLI macro-F1']:.3f}"
          f"   日本語だけで学習 → 正解率 {r_only['JNLI test (日本語)']:.3f} / macro-F1 {r_only['JNLI macro-F1']:.3f}")
```

```python
# 英語だけで学習したモデルに、日本語の文の組を入力してみる
from transformers import pipeline

clf = pipeline("text-classification", model="mdeberta-snli", tokenizer=tokenizer, device=0)
pairs = [("公園で子どもたちがサッカーをしている。", "子どもたちが外で遊んでいる。"),
         ("公園で子どもたちがサッカーをしている。", "子どもたちは教室で授業を受けている。"),
         ("公園で子どもたちがサッカーをしている。", "子どもたちは試合に勝った。")]
for p, h in pairs:
    r = clf({"text": p, "text_pair": h})
    print(f"{r['label']:13} {r['score']:.2f}  {p} → {h}")
```

- 多言語モデルは、英語のデータだけで微調整しても日本語の入力をある程度扱える。手順はこの章の「BERT をファインチューニングして文書分類する (Trainer)」と同じで、モデルを多言語のものに替えるだけでよい。
- 上の表では、3 つの比較で言語の効果とデータセットの効果を切り分けている。(1) 言語だけが違う: 英語で学習したモデルは SNLI dev で 0.863、JSNLI dev で 0.849 で、差は 1.4 ポイントだった。(2) 言語をそろえても作り方が違う: 機械翻訳した JSNLI で学習しても JNLI では 0.673 で、JNLI で学習した場合 (0.893) に遠い。(3) 逆向き: JNLI で学習したモデルは SNLI dev で 0.535、JSNLI dev で 0.645 になる。translate-train のモデルも、英語の SNLI dev では 0.838 と下がる。SNLI (Flickr30k) も JNLI (YJ Captions) も画像キャプションが元で分野は近いので、JNLI との差の大部分は、ラベルの付け方の違いによるものと考えられる。なお、translate-train の学習データは機械翻訳した翻訳調の日本語である。
- JSNLI dev は、SNLI の dev を機械翻訳し、日本語として意味が通るか、ラベルが正しいかをクラウドソーシングで確かめて絞り込んだ 3,916 組である (JSNLI の README)。SNLI dev より易しい組に偏っている可能性があり、1.4 ポイントは言語の効果を小さく見せているかもしれない。また機械翻訳の文なので、JSNLI dev での translate-train (0.858) とゼロショット転移 (0.849) の比較は、translate-train に有利である。
- JNLI では、ゼロショット転移の正解率は 0.634 (95% 信頼区間 0.615〜0.654) で、多数派のクラスを答え続けた場合 (0.544) は上回る。translate-train との差は +0.039 (対応のあるブートストラップで 95% 信頼区間 +0.024〜+0.055) だった。この区間は評価データの抽出によるばらつきだけで、乱数の種によるばらつきは含まない。種 1 つ、1 回の学習の結果なので、複数の種で確かめる。ブートストラップによる信頼区間の求め方は評価指標と統計的検定の章の「ブートストラップ法で評価値の信頼区間を求める」を参照。
- 混同行列を見ると、英語で学習したモデルは JNLI の中立 1365 件のうち 430 件を含意と判定している。ラベルの割合も、SNLI はほぼ均等なのに対し、JNLI は中立が半分を超える。言語間転移を試す前に、元のデータと使いたいデータで、ラベルの定義や割合がそろっているかを確かめる。
- few-shot は、この設定 (学習率・エポック数は調整していない、種 1 つ) では、英語で学習したモデルに JNLI の 100 件を足すと正解率 0.813、1000 件で 0.860 になり、JNLI の 12000 件で学習した場合 (0.893) に近づいた。同じ件数の日本語だけで学習すると 0.690 と 0.756 だった。学習データが小さいほど (特に 100 件では) 種によって結果が大きく変わるので、複数の種で平均をとる。
- mDeBERTa-v3-base のモデルカードでは、英語のデータだけで学習して XNLI の 15 言語 (英語を含む) の dev で評価した正解率の平均が、XLM-R base より高い (79.8 と 76.2)。この章の「ラベル付きデータなしで分類する (ゼロショット分類、NLI)」で使った `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` も、このモデルを MNLI と XNLI (日本語を含まない 15 言語) で微調整したもので、日本語への適用は言語間転移にあたる。
- 検索の章の「言語をまたいで検索する (クロスリンガル検索)」は多言語の埋め込みで言語をまたいだ。ここでは多言語の事前学習モデルを微調整した分類器で言語をまたいでいる。データは SNLI、JSNLI、JNLI のいずれも CC BY-SA 4.0、モデルは MIT ライセンス。CC BY-SA の継承の条件が学習したモデルにかかるかは確定していないが、かかる可能性があるので、配布するときは注意する。

## 日本語の感情分析 (ポジネガ判定) をする {sentiment_ja}

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

- 二重否定、皮肉、対比 (「〜だが…」) は誤りやすい。ドメインが違うと精度が落ちるので、自分のデータで少量でも評価する。
- ここで使ったのは多言語モデル (英語・日本語・中国語など 12 言語)。日本語専用なら `koheiduck/bert-japanese-finetuned-sentiment`、英語なら `cardiffnlp/twitter-roberta-base-sentiment-latest` など。
- 喜び・悲しみ・怒りなどの感情の種類まで知りたい場合は、この章の「喜び・悲しみ・怒りなど 8 種類の感情の強さを推定する (WRIME)」を、観点 (味、価格など) ごとに判定したい場合は「観点ごとにポジティブ・ネガティブを判定する (アスペクトベース感情分析)」を参照。

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

- 多くの感情で、書き手は読み手より多くの感情を報告する (恐れだけは逆)。特に怒りと信頼は、強度 1 以上を付けた割合が書き手で 12.9% と 21.7%、読み手 1 人あたりでは 2.8% と 5.1% しかない。書き手が信頼を 3 (強) と付けた 1,947 件のうち 1,220 件は、読み手 3 人とも 0 だった。本人が感じていても文面に出ていない感情は、テキストからは推定しにくい。
- 書き手と読み手 1 人の相関は、どの感情でも読み手どうしの相関より低い。差が大きいのは怒り (0.401 と 0.627) と期待 (0.410 と 0.581)。信頼は読み手どうしでも 0.216 と一致しにくい。
- 8 感情の平均の相関は、どちらのモデルでも読み手のほうが書き手より高かった (埋め込み + リッジで 0.473 と 0.361、BERT で 0.665 と 0.464。書き手単位のブートストラップ信頼区間は重ならない)。文面から伝わる感情のほうが推定しやすいことに加え、読み手は 3 人の平均なので 1 人分のラベルよりノイズが少ないことも効いている。用途に応じて、どちらのラベルで学習・評価するかを決める (「読んだ人にどう伝わるか」なら読み手、「本人の状態」なら書き手)。
- ほとんど 0 ばかりの目標では、相関 r が低く出やすい。読み手の怒りはテストで強度 1 以上が 37 件しかなく、埋め込み + リッジの相関は 0.241 と書き手 (0.408) より低いが、「その感情あり」を当てる AUC は 0.840 と書き手 (0.792) より高い。出現の少ない感情は、正例数を確かめ、相関と AUC の両方を見る。
- 強度を回帰しておけば、しきい値を決めて「この感情あり / なし」のマルチラベル分類にも使える。しきい値の決め方や評価は、この章の「1 つの文書に複数のラベルを付ける (マルチラベル分類)」「クラスの偏ったデータを扱う (クラス重み・しきい値調整)」と同じ。回帰なので、予測が 0 未満になることもある。
- WRIME で学習済みの BERT は、16 項目すべてで埋め込み + リッジより相関が高かった。ただし BERT は学習データ 40,000 件全部でモデル全体を微調整しており、条件は対等ではない。また、このモデルの学習コードは学習中の評価にテスト分割を使っている。最良のエポックを選ぶ設定ではなく最終エポックのモデルだが、テストを見ながら設定を決めた可能性は否定できないので、BERT の数値はやや楽観的に見る。
- WRIME の学習・検証・テストは書き手で分かれている (テストの 10 人は学習に出てこない)。同じ人の投稿が学習とテストの両方に入ると、書き手の癖を覚えて評価が甘くなる。信頼区間も、投稿ではなく書き手を単位にリサンプリングして求めた。ブートストラップによる信頼区間の求め方は評価指標と統計的検定の章の「ブートストラップ法で評価値の信頼区間を求める」を参照。
- WRIME は CC BY-NC-ND 4.0 (非商用・改変禁止)。BERT のモデルは CC BY-SA 3.0 と表示されているが、WRIME で微調整したものなので、商用で使う場合は WRIME の条件も含めて確認が必要。ポジネガだけでよければ、この章の「日本語の感情分析 (ポジネガ判定) をする」を使う。

## 観点ごとにポジティブ・ネガティブを判定する (アスペクトベース感情分析) {absa_ja gpu}

「売上は伸びたが利益は減った」のような文で、文全体ではなく観点 (アスペクト) ごとに、売上=ポジ、利益=ネガと判定したい。有価証券報告書の ABSA データセット chABSA を使う。

```python
!pip install -q fugashi unidic-lite
```

```python
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import GroupShuffleSplit

# chABSA (TIS 株式会社 / chakki-works、CC BY 4.0、https://github.com/chakki-works/chABSA-dataset) を第三者が Hugging Face に複製したもの。
# README の配布 URL はリンク切れのため、複製の版を固定して使う (複製が削除されると読めなくなる)
ds = load_dataset("zenless-lab/chABSA", revision="254de3bf9771bcafb40812cecd33e6cdb234bb49")
sents = pd.concat([ds["train"].to_pandas(), ds["test"].to_pandas()], ignore_index=True)
# 1 行 = (文, 観点の語, 極性)。from / to は文中の文字位置
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

# (1) 文全体の感情分析 (この章の「日本語の感情分析 (ポジネガ判定) をする」と同じモデル) の結果を、文中のすべての観点に配る
sent_clf = pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
uniq = test.drop_duplicates("sid")
out = sent_clf(list(uniq["sentence"]), top_k=None, truncation=True, batch_size=32)
sent_pred = {sid: max((r for r in rs if r["label"] != "neutral"), key=lambda r: r["score"])["label"]   # ポジとネガの高いほう
             for sid, rs in zip(uniq["sid"], out)}
report("文全体の感情分析", test["sid"].map(sent_pred))

# 文ごとに 1 つのラベルしか出せない方法の上限: 各文で多数派の極性を (正解を見て) 全観点に付けた場合
majority = test.groupby("sid")["label"].agg(lambda s: s.mode()[0])
report("文単位の上限 (正解の多数派)", test["sid"].map(majority))

# 観点の語と文だけを見る方法の上限: 同じ文の同じ位置の語に、カテゴリ違い (例: business#sales と business#profit) で逆の極性が付いた行がある
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

# (3) 「観点の語」と「観点に印を付けた文」を文ペアにして BERT を微調整する (T4 で 5 分程度)
name = "tohoku-nlp/bert-base-japanese-v3"   # この章の「BERT をファインチューニングして文書分類する」と同じモデル (Apache 2.0)
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

- 文全体の感情分析は 1 つの文に 1 つのラベルしか出せない。評価の 1,449 観点のうち 522 は対比の文 (ポジとネガが同居) にあり、正解を見て文ごとに多数派を選んでも、対比の文の正解率は 0.625 が上限になる。
- 同じデータで学習した TF-IDF でも、文全体が入力だと対比の文は 0.569 だが、観点の後ろの区間だけにすると 0.822 に上がる。入力に「どの観点か」を含めることが重要である。
- 文ペア (観点, 観点に【】で印を付けた文) で BERT を微調整すると、全体 0.967、対比の文 0.937 になった。1 回の実行の値で、種を固定しても GPU の計算は完全には再現しない (set_seed を入れる前を含む検証の 3 回の実行では、全体で 0.967〜0.971 だった)。評価件数から見た 95% 区間はおよそ全体 ±0.009、対比の文 ±0.02 で、手法どうしの差はこれよりずっと大きい。
- 同じ文の同じ語に、カテゴリ違い (例: 「運輸業」の business#sales はポジ、business#profit はネガ) で逆の極性が付いた行が評価に 34 ある。語と文だけを入力にする方法では、正解率は全体 0.988、対比の文 0.967 が上限になる。カテゴリも区別したいなら、カテゴリを入力に加える。
- 新しい決算の文 2 つ (観点 5 つ) では期待どおりに判定した (正解ラベルのない例)。学習データは有価証券報告書だけなので、分野外の文は当てにならない。検証では「料理は美味しいが接客が悪い」の「料理」の判定が、同じコードの実行ごとにポジとネガで変わった。
- 数値は A5000 で float16 の経路 (`BF16 = False`、T4 と同じ dtype) で実行したもので、GPU メモリの最大使用量は 4.7 GB だった。(1) と (2) は CPU だけで動く。この章の「ラベル付きデータなしで分類する (ゼロショット分類、NLI)」で「売上高は好調だ」という仮説の含意を判定させたり、LLM に観点ごとの極性を JSON で答えさせたりする方法もある。
- chABSA は TIS の久保・中山によるデータセットで、2016 年度の有価証券報告書 230 社分の「経営成績の概要」のうち、観点の付いた 3,215 文 (全 6,119 文のうち) に観点の語、カテゴリ (company#sales など)、極性を付けたもの (CC BY 4.0、https://github.com/chakki-works/chABSA-dataset)。ここでは Hugging Face 上の第三者による複製を使った (文数・観点数は論文と一致する。複製にはライセンスの記載がないので、元データの CC BY 4.0 に従う)。
- 観点の語は与えられている前提で、極性だけを判定した。文から観点の語を見つけるのは系列ラベリングの問題になる (固有表現抽出の章の「文字位置のアノテーションを BIO タグ列に変換する」「BERT をファインチューニングして日本語の固有表現抽出器を作る」)。

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
- 禁止語リストは説明しやすいが、遠回しな表現や文脈に依存する攻撃を見逃し、無害な用法 (「バカ売れ」) を拾う。学習した分類器と組み合わせるのが普通。
- 有害な投稿は全体のごく一部なので、正解率ではなく適合率・再現率・平均適合率で評価する。しきい値は、運用 (自動で消すのか、人が確認するのか) に合わせて決める。
- 「しね」のようなひらがな書きや伏せ字は、表記が変わるだけで見逃されやすい。文字の正規化 (ひらがな・カタカナの統一など) をしてから判定する、表記ゆれを含むデータで学習する、といった対策がある。
- 分類器は、特定の集団を指す語が出てくるだけで有害と判定しがちである (差別に反対する投稿まで有害とする)。誤検出の例を読んで、そうした偏りがないか確かめる。
- データは数百件と小さい。実際には、対象とするサービスの投稿で数千件以上の注釈データを作る。何を有害とするかの基準 (ガイドライン) を先に決めることが、モデルの選択より重要。
