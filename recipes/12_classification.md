# テキスト分類

文書にラベルを付ける。古典的な TF-IDF から BERT のファインチューニング、ゼロショットまで。

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
- この「交差検証の予測確率でラベル誤りを探す」考え方を体系化したのが Confident Learning (`cleanlab`)。

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
from transformers import pipeline

clf = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

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
- 喜び・悲しみ・怒りなどの感情の種類まで知りたい場合は、WRIME データセットで学習されたモデルを探す。

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
