# トピックとキーワードの抽出

ワードクラウドで眺める、文書を代表する語（特徴語やキーワード）を選ぶ、共起ネットワークを描く、文書集合に潜む話題（トピック）を見つける、急に増えた語を見つける（バースト検出）、アンケートの自由記述を話題ごとに集計する。

## 日本語のワードクラウドを作る {wordcloud_ja}

テキストの頻出語をワードクラウドで可視化したい。

```python
!pip install -q wordcloud fugashi unidic-lite matplotlib-fontja
```

```python
import io, re, urllib.request, zipfile
from collections import Counter
from fugashi import Tagger

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

text = aozora("https://www.aozora.gr.jp/cards/000081/files/456_ruby_145.zip")  # 銀河鉄道の夜
tagger = Tagger()
stop = {"よう", "こと", "もの", "それ", "これ", "そこ", "どこ", "なに", "ところ", "とき", "うち", "ほう"}
freq = Counter(w.surface for line in text.split("\n") for w in tagger(line)
               if w.feature.pos1 == "名詞" and w.feature.pos2 != "数詞" and w.surface not in stop and len(w.surface) > 1)
print(freq.most_common(10))
```

```python
import matplotlib.pyplot as plt
import matplotlib_fontja
from wordcloud import WordCloud

wc = WordCloud(font_path=matplotlib_fontja.get_font_ttf_path(),  # 日本語フォントの指定が必須
               width=800, height=450, background_color="white", max_words=150, colormap="viridis")
wc.generate_from_frequencies(freq)
plt.figure(figsize=(10, 6)); plt.imshow(wc); plt.axis("off"); plt.show()
wc.to_file("wordcloud.png");
```

- 分かち書き済みの文字列を `wc.generate(text)` に渡す方法もあるが、頻度辞書を渡すほうが品詞やストップワードを制御しやすい。

## TF-IDF で文書ごとの特徴語を取り出す {tfidf_keywords}

複数の文書があるとき、それぞれの文書を特徴づける単語（他の文書にはあまり出ない語）を見つけたい。

```python
!pip install -q fugashi unidic-lite
```

```python
import io, re, urllib.request, zipfile

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

base = "https://www.aozora.gr.jp/cards/"
docs = {
    "吾輩は猫である": aozora(base + "000148/files/789_ruby_5639.zip"),
    "坊っちゃん": aozora(base + "000148/files/752_ruby_2438.zip"),
    "走れメロス": aozora(base + "000035/files/1567_ruby_4948.zip"),
    "銀河鉄道の夜": aozora(base + "000081/files/456_ruby_145.zip"),
}
```

```python
import pandas as pd
from fugashi import Tagger
from sklearn.feature_extraction.text import TfidfVectorizer

tagger = Tagger()

def tokenize(text):
    return [w.surface for line in text.split("\n") for w in tagger(line)
            if w.feature.pos1 == "名詞" and w.feature.pos2 in ("普通名詞", "固有名詞") and len(w.surface) > 1]

vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False, sublinear_tf=True)
X = vec.fit_transform(docs.values())
vocab = vec.get_feature_names_out()

top = {title: [vocab[i] for i in row.toarray().ravel().argsort()[::-1][:10]] for title, row in zip(docs, X)}
pd.DataFrame(top)
```

- 自前のトークナイザを渡すときは `token_pattern=None` を付ける（警告を消す）。
- `sublinear_tf=True` は頻度を 1+log(tf) に圧縮する。長い文書で頻出語が支配的になるのを防ぐ。
- 文書数が少ないと IDF があまり効かない。段落や章を 1 文書とみなすと粒度を変えられる。
- 1 語ではなく複合語の専門用語を取り出したいときは、日本語の形態素解析と文分割の章の「専門用語の候補を FLR と C-value でスコア付けして抽出する」を参照。

## 文書からキーワードを抽出する（埋め込みによる KeyBERT 方式） {keyword_extraction}

1 つの文書から、その内容を代表するキーワードを数個選びたい。コーパス全体の統計なしで、単独の文書に対して行いたい。

```python
!pip install -q fugashi unidic-lite
```

```python
import numpy as np
from fugashi import Tagger
from sentence_transformers import SentenceTransformer

tagger = Tagger()
model = SentenceTransformer("intfloat/multilingual-e5-small")

def candidates(text):
    """連続する名詞をつなげた複合名詞を候補にする"""
    out, buf = [], []
    for w in list(tagger(text)) + [None]:
        if w is not None and w.feature.pos1 in ("名詞", "接頭辞", "接尾辞") and w.feature.pos2 != "数詞":
            buf.append(w.surface)
        else:
            if buf and len("".join(buf)) > 1:
                out.append("".join(buf))
            buf = []
    return list(dict.fromkeys(out))

def keywords(text, top_k=6, diversity=0.5):
    cands = candidates(text)
    doc = model.encode("query: " + text, normalize_embeddings=True)
    emb = model.encode(["query: " + c for c in cands], normalize_embeddings=True)
    sim_doc, sim_cand = emb @ doc, emb @ emb.T
    selected = [int(sim_doc.argmax())]
    while len(selected) < min(top_k, len(cands)):      # MMR: 文書に近く、かつ選択済みの語とは似ていないものを選ぶ
        rest = [i for i in range(len(cands)) if i not in selected]
        mmr = [(1 - diversity) * sim_doc[i] - diversity * sim_cand[i, selected].max() for i in rest]
        selected.append(rest[int(np.argmax(mmr))])
    return [(cands[i], round(float(sim_doc[i]), 3)) for i in selected]

text = ("文法誤り訂正は、学習者が書いた文に含まれる誤りを自動的に訂正する自然言語処理のタスクである。"
        "近年は系列変換モデルや大規模言語モデルを用いた手法が主流となっており、擬似誤りデータによる事前学習が性能向上に寄与している。"
        "評価には編集単位の適合率と再現率に基づく F0.5 が広く用いられる。")
print("候補:", candidates(text))
print("キーワード:", keywords(text))
```

- 「文書全体の埋め込みに近い語句ほど、その文書を代表している」という考え方。学習もコーパスも要らない。
- MMR で多様性を持たせないと、「言語モデル」「大規模言語モデル」のように似た語ばかりが選ばれる。
- 統計ベースなら TF-IDF（この章の「TF-IDF で文書ごとの特徴語を取り出す」を参照）、英語なら YAKE や RAKE、グラフベースなら TextRank などがある。ライブラリは `keybert` や `pke`。
- 名詞の連続から専門用語の候補を作り、連接の多様さ (FLR) や入れ子の頻度 (C-value) でスコアを付ける方法は、日本語の形態素解析と文分割の章の「専門用語の候補を FLR と C-value でスコア付けして抽出する」を参照。

## 共起ネットワークを描く {cooccurrence_network}

よく一緒に出てくる単語同士を線で結んだネットワーク図で、テキストの話題の構造を俯瞰したい。

```python
!pip install -q fugashi unidic-lite matplotlib-fontja
```

```python
import io, re, urllib.request, zipfile
from collections import Counter
from itertools import combinations
from fugashi import Tagger

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

text = aozora("https://www.aozora.gr.jp/cards/000035/files/1567_ruby_4948.zip")  # 走れメロス
tagger = Tagger()
docs = [{w.surface for w in tagger(s) if w.feature.pos1 == "名詞" and w.feature.pos2 in ("普通名詞", "固有名詞") and len(w.surface) > 1}
        for s in re.split(r"[。\n]", text) if s.strip()]
word_count = Counter(w for d in docs for w in d)
pair_count = Counter(p for d in docs for p in combinations(sorted(d), 2))
```

```python
import matplotlib.pyplot as plt
import matplotlib_fontja
import networkx as nx

G = nx.Graph()
for (a, b), c in pair_count.items():
    jaccard = c / (word_count[a] + word_count[b] - c)
    if c >= 3 and jaccard >= 0.1:
        G.add_edge(a, b, weight=jaccard)

communities = nx.community.greedy_modularity_communities(G)  # 色分け用にコミュニティ検出
color = {w: i for i, com in enumerate(communities) for w in com}

plt.figure(figsize=(11, 8))
pos = nx.spring_layout(G, k=0.6, seed=0)
nx.draw_networkx_nodes(G, pos, node_size=[word_count[w] * 40 for w in G], node_color=[color[w] for w in G], cmap="Set3")
nx.draw_networkx_edges(G, pos, width=[G[a][b]["weight"] * 6 for a, b in G.edges], alpha=0.4)
nx.draw_networkx_labels(G, pos, font_family=plt.rcParams["font.family"], font_size=11)
plt.axis("off"); plt.show()
```

- 線を引く基準は、共起回数の下限と Jaccard 係数の下限の 2 つで調整する。線が多すぎると毛玉になる。
- `draw_networkx_labels` は rcParams のフォントを見ないので `font_family` を明示する。

## LDA で文書集合のトピックを見つける {lda_topics}

大量の文書がどんな話題で構成されているのかを、ラベルなしで把握したい。各文書がどの話題をどれくらい含むかも知りたい。

```python
!pip install -q fugashi unidic-lite
```

```python
import io, tarfile, urllib.request
import pandas as pd
from fugashi import Tagger

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

df = load_livedoor().sample(3000, random_state=0).reset_index(drop=True)
tagger = Tagger()

def nouns(text):
    return " ".join(w.surface for w in tagger(text[:1500]) if w.feature.pos1 == "名詞" and w.feature.pos2 == "普通名詞" and len(w.surface) > 1)

df["nouns"] = df["text"].map(nouns)
```

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(token_pattern=r"\S+", min_df=5, max_df=0.3)   # LDA には TF-IDF ではなく単語の出現回数を渡す
X = vec.fit_transform(df["nouns"])
lda = LatentDirichletAllocation(n_components=9, learning_method="batch", max_iter=30, random_state=0)
doc_topic = lda.fit_transform(X)        # 文書 × トピックの分布 (各行の和は 1)

vocab = vec.get_feature_names_out()
for k, weights in enumerate(lda.components_):
    print(k, " ".join(vocab[i] for i in weights.argsort()[::-1][:10]))
```

```python
# 各トピックと元のカテゴリの対応を見る (答え合わせ)
df["topic"] = doc_topic.argmax(axis=1)
pd.crosstab(df["label"], df["topic"])
```

- トピックは「単語の確率分布」、文書は「トピックの確率分布」として表される。1 つの文書が複数のトピックを持てるのが、クラスタリングとの違い。
- トピック数は自分で決める。パープレキシティやコヒーレンスも参考になるが、最終的には「人間が見て解釈できるか」で選ぶ。
- 前処理が結果を大きく左右する。名詞に絞る、高頻度語・低頻度語を除く（`max_df`、`min_df`）のが基本。
- 大規模なコーパスや、コヒーレンスの計算・可視化 (pyLDAvis) まで行うなら gensim の `LdaModel` が定番。

## NMF でトピックを抽出する {nmf_topics}

LDA より速く、短い文書でも安定しやすい方法でトピックを抽出したい。TF-IDF 行列を非負値行列因子分解する。

```python
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

names = fetch_20newsgroups(subset="train").target_names
cats = [c for c in names if not c.startswith("talk.politics")]   # 政治の話題のグループは除く
data = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))

# 署名のように、5 件以上の投稿に同じ形で現れる行を除く
def lines(d):
    return {l.strip() for l in d.split("\n") if len(l.strip()) > 10}
count = Counter(l for d in data.data for l in lines(d))
repeated = {l for l, n in count.items() if n >= 5}
def clean(d):
    return "\n".join(l for l in d.split("\n") if l.strip() not in repeated)

docs = [c for c in map(clean, data.data) if len(c) > 200]
vec = TfidfVectorizer(stop_words="english", min_df=5, max_df=0.5, max_features=20000, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
X = vec.fit_transform(docs)

nmf = NMF(n_components=12, init="nndsvd", random_state=0, max_iter=400)
W = nmf.fit_transform(X)      # 文書 × トピックの重み
H = nmf.components_           # トピック × 単語の重み
vocab = vec.get_feature_names_out()
for k, row in enumerate(H):
    print(f"{k:2} {' '.join(vocab[i] for i in row.argsort()[::-1][:9])}")
```

```python
# 各トピックを最もよく代表する文書
for k in [3, 5, 8]:
    i = W[:, k].argmax()
    print(k, "|", " ".join(docs[i].split())[:110])
```

- TF-IDF 行列 X を、非負の 2 つの行列の積 W × H で近似する。非負なので「足し算だけ」で文書が表現され、解釈しやすい。
- 確率モデルではないので、LDA のような確率的な解釈はできないが、決定的 (`init="nndsvd"`) で再現性が高い。
- ツイートやタイトルのような短文には、LDA より NMF のほうが良い結果になることが多い。
- 署名や定型文のように同じ文字列を繰り返す投稿者がいると、その語だけのトピックができてしまう。`remove=("headers", "footers", "quotes")` でも署名は取り切れないので、多くの投稿に出てくる行を除いた。

## 文埋め込みとクラスタリングでトピックを作る（BERTopic 方式） {embedding_topics}

単語の共起ではなく、文の意味に基づいてトピックを作りたい。クラスタごとに特徴的な語を取り出して名前を付けたい。

```python
!pip install -q fugashi unidic-lite umap-learn
```

```python
import io, tarfile, urllib.request
import pandas as pd
from sentence_transformers import SentenceTransformer

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

df = load_livedoor().sample(1500, random_state=0).reset_index(drop=True)
model = SentenceTransformer("intfloat/multilingual-e5-small")
emb = model.encode(["query: " + t for t in df["title"]], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
```

```python
import numpy as np
import umap
from fugashi import Tagger
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

# 1. 次元削減してからクラスタリング (高次元のままだと距離が効きにくい)
reduced = umap.UMAP(n_neighbors=15, n_components=5, metric="cosine", random_state=0).fit_transform(emb)
df["cluster"] = KMeans(n_clusters=12, n_init=10, random_state=0).fit_predict(reduced)

# 2. クラスタごとに文書を連結して 1 つの大きな文書とみなし、クラスタ単位の TF-IDF (c-TF-IDF) で特徴語を選ぶ
tagger = Tagger()
tok = lambda s: [w.surface for w in tagger(s) if w.feature.pos1 == "名詞" and w.feature.pos2 in ("普通名詞", "固有名詞") and len(w.surface) > 1]
joined = df.groupby("cluster")["title"].apply(" ".join)
vec = CountVectorizer(tokenizer=tok, token_pattern=None, lowercase=False)
counts = vec.fit_transform(joined).toarray()
tf = counts / counts.sum(axis=1, keepdims=True)
idf = np.log(1 + counts.sum(axis=1).mean() / counts.sum(axis=0))
ctfidf = tf * idf
vocab = vec.get_feature_names_out()

for c in joined.index:
    top = [vocab[i] for i in ctfidf[c].argsort()[::-1][:7]]
    main = df[df.cluster == c]["label"].value_counts()
    print(f"{c:2} ({(df.cluster == c).sum():3} 件, 最多カテゴリ {main.index[0]} {main.iloc[0] / main.sum():.0%}) {' '.join(top)}")
```

- この「埋め込み → UMAP → クラスタリング → c-TF-IDF」の流れをまとめたライブラリが BERTopic (`pip install bertopic`)。密度ベースの HDBSCAN でクラスタ数を自動で決め、どこにも属さない外れ値も扱える。
- 短い文（タイトル、ツイート、アンケートの回答）に特に向く。長い文書は段落に分けてから埋め込む。
- トピックの名前付けは、特徴語と代表文書を LLM に渡して要約させると読みやすくなる。この章の「アンケートの自由記述を話題ごとに集計して報告する（LLM によるラベル付け）」で、その方法を使っている。

## アンケートの自由記述を話題ごとに集計して報告する（LLM によるラベル付け） {survey_free_text gpu}

アンケートの自由記述が数百件ある。文埋め込みでクラスタリングして話題を見つけ、各話題に LLM でラベルと説明を付け、件数と代表的な回答を報告用の表と図にしたい。1 件が複数の話題にまたがる回答の扱い、ラベルの粒度の調整、少数意見の取りこぼしの確かめ方も知りたい。

（T4 の GPU ランタイムで、モデルのダウンロードを含めて全体で 10 分程度かかる。）

```python
!pip install -q matplotlib-fontja==1.1.0
```

```python
import random
import re
import pandas as pd

# このレシピのために作った架空のアンケート（市立図書館の利用者アンケート「図書館へのご意見・ご要望をご自由にお書きください」）。
# 話題ごとに手で書いた文を 1 つ（単一の話題）か 2 つ（複数の話題）つないで回答にした。実在の回答ではない。
# 正解の話題 (true) は、最後の答え合わせにだけ使う
FRAG = {
    "開館時間": [
        "平日も夜9時まで開いていると、仕事帰りに寄れて助かります。", "閉館が18時なので、会社員はほとんど利用できません。",
        "月曜休館をやめて、別の曜日を休みにしてほしい。", "土日だけでも朝9時より早く開けてもらえないでしょうか。",
        "年末年始の休館期間が長すぎると思います。", "夜間開館の日を増やしてほしいです。", "開館時間を延長してください。",
        "祝日に閉まっていることがあり、何度か無駄足になりました。", "試験期間中だけでも閉館時間を遅くしてほしい。",
        "館内整理日がいつなのか分かりにくいです。", "日曜日は17時に閉まるのが早すぎる。", "朝早くから開いているのはありがたい。",
        "分館は平日の昼間しか開いていないので、もう少し時間を広げてほしい。", "仕事が終わってから行くと、もう閉まっています。"],
    "座席": [
        "自習席が少なく、土日は朝から満席です。", "学生が席を長時間占領していて、本を読む場所がない。", "座席の予約制を導入してほしい。",
        "窓際の閲覧席は明るくて気に入っています。", "机が狭くてノートパソコンと本を同時に広げられない。",
        "椅子が硬く、長く座っていると腰が痛くなります。", "グループで話し合いができる部屋がほしい。",
        "荷物だけ置いて席を取る人がいて困っている。", "新聞コーナーのソファーがいつも同じ人で埋まっている。", "学習室を増やしてほしいです。",
        "席数が足りないので、使っていない会議室を開放してはどうか。", "閲覧席の照明が暗くて字が読みにくい。",
        "一人用の仕切りのある席があると集中できる。", "夏休みは中高生で席が全く空いていない。"],
    "蔵書": [
        "新刊の入荷が遅く、話題の本はいつも予約待ちです。", "専門書が古いものばかりで、最新の版がない。", "雑誌の種類を増やしてほしい。",
        "リクエストした本をすぐに購入してもらえて嬉しかった。", "人気の小説を何冊か複本で置いてほしい。",
        "郷土資料が充実しているのはこの図書館の良いところだと思う。", "プログラミングやITの本が少ない。",
        "漫画も少し置いてもらえると若い人が来やすいと思う。", "電子書籍の貸出を始めてほしい。",
        "資格試験の問題集が古い年度のものしかない。", "大活字本をもっと増やしてほしい。", "同じシリーズの途中の巻が抜けていることが多い。",
        "古い本が汚れていて、手に取る気になれないものがある。", "ビジネス書の棚をもう少し充実させてほしい。"],
    "予約・Web": [
        "ホームページの蔵書検索が使いにくい。", "予約した本の取り置き期限が短すぎる。", "スマホから予約や延長ができるようにしてほしい。",
        "予約の順番があと何人か分かるようにしてほしい。", "貸出の延長手続きがネットでできるのは便利です。",
        "Webサイトのデザインが古く、どこに何があるか分からない。", "予約の本が届いたメールがよく迷惑メールに入ってしまう。",
        "パスワードを忘れたときの再発行が窓口でしかできないのは不便。", "自動貸出機の操作がわかりにくく、毎回職員さんを呼んでいる。",
        "予約できる冊数の上限を増やしてほしい。", "検索結果に貸出中かどうかがすぐ表示されないのが不便です。",
        "カードを忘れてもスマホの画面で借りられるようにしてほしい。"],
    "職員": [
        "カウンターの職員さんがいつも親切で、気持ちよく利用できます。", "レファレンスで調べ物を手伝っていただき、大変助かりました。",
        "一部の職員の言葉づかいが冷たく感じることがある。", "本の場所を尋ねたら、棚まで案内してくれてありがたかった。",
        "窓口が混んでいるのに一人しか対応していないことがある。", "子どもにも優しく声をかけてくださり感謝しています。",
        "質問しても「分かりません」で終わってしまい残念だった。", "司書の方のおすすめ本の展示がいつも楽しみです。",
        "返却が少し遅れただけで強い口調で注意された。", "職員の皆さんの丁寧な対応に感謝しています。", "電話の問い合わせにも丁寧に答えてもらえた。"],
    "駐車場・交通": [
        "駐車場が狭く、土日は停められないことが多い。", "駐輪場に屋根がなく、雨の日は自転車が濡れる。",
        "駅から遠いので、バスの本数を増やしてほしい。", "駐車場の無料時間を2時間に延ばしてほしい。",
        "駐車場の出入口が狭くて車をぶつけそうになる。", "最寄りのバス停から図書館までの道が分かりにくい。",
        "駐輪場がいっぱいで、歩道に自転車があふれている。", "近くの市営駐車場と提携してほしい。", "送迎の車が入口前に停まっていて危ない。"],
    "静かさ・マナー": [
        "館内で携帯電話で話している人がいて気になる。", "中高生のおしゃべりがうるさいときがある。", "静かな環境で読書ができて満足しています。",
        "イヤホンの音漏れを注意してほしい。", "飲食禁止のはずなのに、お菓子を食べている人がいる。", "職員同士の私語が意外と響いている。",
        "キーボードの音が気になるので、パソコンを使える席を分けてほしい。", "寝ている人のいびきが気になることがある。",
        "マナーの悪い利用者には、もっと注意してほしい。"],
    "Wi-Fi・電源": [
        "館内のWi-Fiがすぐ切れる。", "電源の使える席を増やしてほしい。", "無料Wi-Fiがあるのは便利です。",
        "Wi-Fiの接続に毎回登録が必要で面倒。", "パソコン席のコンセントが壊れていることがある。",
        "オンライン会議に使えるくらい通信速度を上げてほしい。", "スマホの充電ができる場所があると助かる。", "調べ物用のパソコンが古くて動作が遅い。"],
    "子ども・イベント": [
        "子ども向けのおはなし会を毎回楽しみにしています。", "児童書コーナーをもう少し広くしてほしい。",
        "小さい子どもが少し声を出しても気にならないスペースがあると助かる。", "夏休みの工作教室に参加して、子どもが本に興味を持つようになった。",
        "大人向けの講座や読書会も開いてほしい。", "授乳室とおむつ替えの場所を分かりやすくしてほしい。",
        "絵本の数が多くて、子どもと一緒に選ぶのが楽しい。", "中高生向けの本のコーナーがあるとよい。"],
    # 少数意見: 件数が少なく、ほかの話題と同じ語（カウンター、棚、本）を含む
    "バリアフリー": [
        "車いすだとカウンターが高くて、職員さんと話しにくい。", "二階へのエレベーターがなく、足の悪い母は一階しか利用できない。",
        "書架の間が狭くて、車いすでは奥の棚まで行けません。", "入口のスロープが急で、一人では上れない。",
        "多目的トイレが一階に一か所しかないので増やしてほしい。"],
    "外国語": [
        "英語や中国語の本がほとんどないので、外国出身の家族は使いにくい。", "館内の案内表示に英語も入れてほしい。",
        "やさしい日本語の本のコーナーがあると、日本語を勉強している人に役立つと思う。", "利用案内のパンフレットを多言語で作ってほしい。"],
}
rng = random.Random(0)
rows = [{"text": f, "true": [t]} for t, fs in FRAG.items() for f in fs]
main = [t for t in FRAG if t not in ("バリアフリー", "外国語")]
for _ in range(80):                                     # 2 つの話題にまたがる回答
    t1, t2 = rng.sample(main, 2)
    rows.append({"text": rng.choice(FRAG[t1]) + rng.choice(FRAG[t2]), "true": [t1, t2]})
for t in ["特になし", "特にありません。", "なし", "とくにないです", "特にない", "なし。", "今のままで十分です。", "いつもありがとうございます。"]:
    rows.append({"text": t, "true": []})
for r in rows:                                           # 前置きや結びの言葉を付けて、表現を少しばらつかせる
    if r["true"]:
        r["text"] = rng.choice(["", "", "", "いつも利用しています。", "週に一度は来ています。"]) + r["text"]
        r["text"] += rng.choice(["", "", "", "よろしくお願いします。", "ご検討ください。"])
df = pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)
print(len(df), "件")
df.head(8)
```

```python
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 「特になし」のような中身のない回答は、件数だけ数えて分析から除く
# 「今のままで十分です。」やお礼だけの回答は正規表現では除かず、LLM に話題なしと判断させる（手順 4 でラベルなしに出る）
empty = df["text"].str.fullmatch(r"(特に|とくに)?(なし|ない|ないです|ありません)。?")
print("中身のない回答:", empty.sum(), "件")
ans = df[~empty].reset_index(drop=True)

torch.manual_seed(0)
name = "Qwen/Qwen3-4B-Instruct-2507"     # Apache 2.0。思考 (thinking) をしない版
tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.padding_side = "left"           # バッチ生成では左パディング
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()

@torch.no_grad()
def generate(prompts, max_new_tokens=100, batch_size=16):
    """プロンプトのリストをバッチで貪欲生成する（大規模言語モデルの章の「複数のプロンプトをまとめてバッチ生成する」と同じ）"""
    outs = []
    for i in range(0, len(prompts), batch_size):
        texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
                 for p in prompts[i:i + batch_size]]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(llm.device)
        out = llm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        outs += [s.strip() for s in tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return outs

def extract_json(text):
    """最初の JSON オブジェクトを取り出す（大規模言語モデルの章の「LLM の出力から JSON を取り出して検証する」と同じ考え方）"""
    start = text.find("{")
    try:
        return json.JSONDecoder().raw_decode(text[start:])[0] if start >= 0 else {}
    except json.JSONDecodeError:
        return {}
```

```python
# 1. 回答ごとに、意見の対象になっている話題を短い句で LLM に書き出させる（1 件に話題が複数あれば複数。Colab の T4 で 1〜2 分）
PHRASE_PROMPT = """次の図書館の利用者アンケートの回答から、意見や要望の対象になっている話題を、短い名詞句（15 字以内）で書き出してください。
話題が複数あればすべて書き出し、1 行に 1 つずつ「- 」で始めてください。挨拶やお礼だけの言葉は話題にしません。話題がなければ「- なし」と書いてください。

回答: {text}"""
outs = generate([PHRASE_PROMPT.format(text=t) for t in ans["text"]], max_new_tokens=60)
ans["phrases"] = [[p for p in (re.sub(r"^[-・\s]+", "", line).strip() for line in o.split("\n")) if p and p != "なし"] for o in outs]
print("句の数の分布:", ans["phrases"].map(len).value_counts().sort_index().to_dict())
ans[["text", "phrases"]].head(6)
```

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# 2. 句を文埋め込みにしてクラスタリングする。回答の文全体より、言い回しや挨拶に引きずられにくい
ph = ans[["phrases"]].explode("phrases").dropna().reset_index().rename(columns={"index": "answer", "phrases": "phrase"})
model = SentenceTransformer("cl-nagoya/ruri-v3-310m")   # Apache 2.0。日本語の文埋め込みモデル。クラスタリングには「トピック: 」を前に付ける
emb = model.encode(["トピック: " + t for t in ph["phrase"]], normalize_embeddings=True, batch_size=64, show_progress_bar=False)

K = 15          # 少し細かめに分けておき、あとでまとめる（粒度の調整を参照）
km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(emb)
ph["cluster"] = km.labels_
cent = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
ph["sim"] = (emb * cent[ph["cluster"]]).sum(axis=1)        # 自分のクラスタの中心とのコサイン類似度
print(len(ph), "個の句")
for c in ph["cluster"].value_counts().index:
    g = ph[ph.cluster == c].sort_values("sim", ascending=False)
    print(f"[{c:2}] {len(g):3} 個 | " + "、".join(g["phrase"].drop_duplicates().head(6)))
```

```python
# 3. 各クラスタに LLM でラベルと説明を付ける。中心に近い句 10 個とそれ以外から無作為に 10 個、元の回答 3 件を見せる
LABEL_PROMPT = """次は、図書館の利用者アンケートの自由記述から取り出した話題のうち、似たものを集めたものです。
これら全体に共通する話題を表すラベル（12 字以内）と、どのような意見があるかの説明（40 字以内）を考えてください。
一部の話題だけに当てはまる狭いラベルにはしないでください。
出力は {{"label": "...", "description": "..."}} の形の JSON だけにしてください。

話題: {phrases}

回答の例:
{answers}"""
prompts = []
for c in range(K):
    g = ph[ph.cluster == c].sort_values("sim", ascending=False).drop_duplicates("phrase")
    phrases = list(g["phrase"].head(10)) + list(g["phrase"].iloc[10:].sample(min(10, max(len(g) - 10, 0)), random_state=0))
    answers = "\n".join("- " + ans["text"][a] for a in g["answer"].drop_duplicates().head(3))
    prompts.append(LABEL_PROMPT.format(phrases="、".join(phrases), answers=answers))
info = [extract_json(o) for o in generate(prompts, max_new_tokens=100)]
codebook = pd.DataFrame({"cluster": range(K), "ラベル": [d.get("label", f"クラスタ{c}") for c, d in enumerate(info)],
                         "説明": [d.get("description", "") for d in info], "句の数": ph["cluster"].value_counts().sort_index().values})
codebook
```

```python
# 4. ラベルの一覧（コードブック）を選択肢にして、全回答に当てはまるラベルをすべて LLM に選ばせる（マルチラベル。Colab の T4 で 2〜3 分）
choices = "\n".join(f"{i}. {r.ラベル}: {r.説明}" for i, r in enumerate(codebook.itertuples(), 1))
MULTI_PROMPT = """次の図書館の利用者アンケートの回答が触れている話題を、下の一覧からすべて選んでください。
番号だけをカンマ区切りで出力してください（例: 2,5）。どれにも当てはまらない場合は 0 と出力してください。

話題:
{choices}

回答: {text}"""
outs = generate([MULTI_PROMPT.format(choices=choices, text=t) for t in ans["text"]], max_new_tokens=12)
ans["labels"] = [sorted({int(x) - 1 for x in re.findall(r"\d+", o) if 0 < int(x) <= K}) for o in outs]
ans["from_cluster"] = [sorted(set(ph[ph.answer == i]["cluster"])) for i in range(len(ans))]   # 句のクラスタから決めたラベル
print("ラベルの数の分布:", ans["labels"].map(len).value_counts().sort_index().to_dict())
print("句のクラスタと LLM の選択が一致した回答:", f"{np.mean([a == b for a, b in zip(ans.labels, ans.from_cluster)]):.0%}")
print(f"最大メモリ: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

```python
import matplotlib.pyplot as plt
import matplotlib_fontja

# 5. 報告用の表と図: ラベルごとの回答数（1 件が複数のラベルに数えられる）、割合、説明、代表的な回答
def representatives(c, n=2):
    """そのラベルが付いた回答のうち、ほかのラベルが少なく、句がクラスタの中心に近い順に n 件"""
    g = ph[(ph.cluster == c) & ph.answer.map(lambda a: c in ans["labels"][a])].copy()
    g["n_labels"] = g["answer"].map(lambda a: len(ans["labels"][a]))
    g = g.sort_values(["n_labels", "sim"], ascending=[True, False])
    return " / ".join(ans["text"][a] for a in g["answer"].drop_duplicates().head(n))

report = codebook[["ラベル", "説明"]].copy()
report["回答数"] = [sum(c in l for l in ans["labels"]) for c in range(K)]
report["割合"] = report["回答数"] / len(ans)
report["代表的な回答"] = [representatives(c) for c in range(K)]
report = report.sort_values("回答数", ascending=False)
report.to_csv("survey_report.csv", index=False, encoding="utf-8-sig")   # Excel で開けるように BOM 付きで保存
display(report.style.format({"割合": "{:.1%}"}).hide(axis="index"))

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh(report["ラベル"][::-1], report["回答数"][::-1], color="tab:blue")
for i, n in enumerate(report["回答数"][::-1]):
    ax.text(n + 0.3, i, str(n), va="center")
ax.set_xlabel("回答数（複数のラベルに数える）")
ax.set_title(f"図書館へのご意見・ご要望（{len(ans)} 件、中身のない回答 {empty.sum()} 件を除く）")
plt.tight_layout(); plt.savefig("survey_topics.png", dpi=150); plt.show()
```

```python
# 6. 粒度の調整: 細かいラベルを、LLM に大きな分類にまとめさせる（どの分類にも 1 回ずつ入っているかを確かめる）
GROUP_PROMPT = """次は、図書館の利用者アンケートの自由記述から作った話題の一覧です。これらを 6 個程度の大きな分類にまとめてください。
すべての番号をどれか 1 つの分類に入れ、{{"分類名": [番号, ...], ...}} の形の JSON だけを出力してください。

{items}"""
items = "\n".join(f"{i}. {r.ラベル}: {r.説明}" for i, r in enumerate(codebook.itertuples(), 1))
groups = {g: [int(c) - 1 for c in cs] for g, cs in extract_json(generate([GROUP_PROMPT.format(items=items)], max_new_tokens=200)[0]).items()}
assigned = sorted(c for cs in groups.values() for c in cs)
print("すべてのラベルが 1 回ずつ入っている:", assigned == list(range(K)))
for g, cs in groups.items():
    n = sum(any(c in l for c in cs) for l in ans["labels"])
    print(f"{g}（{n} 件）: " + "、".join(codebook["ラベル"][c] for c in cs if 0 <= c < K))
```

```python
# 7. 少数意見の取りこぼしを確かめる
# (1) LLM がどのラベルにも当てはまらないとした回答: コードブックにない話題の候補
print("■ どのラベルにも当てはまらない回答:", (ans["labels"].map(len) == 0).sum(), "件")
for t in ans[ans["labels"].map(len) == 0]["text"]:
    print("  ", t)
# (2) 自分のクラスタの中心から遠い句: クラスタになじまない意見の候補
print("■ クラスタの中心から遠い句")
for _, r in ph.nsmallest(10, "sim").iterrows():
    print(f"  {r.sim:.3f} {r.phrase} [{codebook['ラベル'][r.cluster]}] ← {ans['text'][r.answer]}")
# (3) クラスタ数を増やしたときに現れる小さなクラスタ
lab = KMeans(n_clusters=30, n_init=10, random_state=0).fit_predict(emb)
size = pd.Series(lab).value_counts()
print("■ K=30 で句が 4 個以下のクラスタ")
for c in size[size <= 4].index:
    print(f"  {size[c]} 個: " + "、".join(ph["phrase"][lab == c]))
```

```python
# (4) 取りこぼしの候補をまとめて LLM に見せ、新しいラベルが要るかを考えさせる
cand = list(dict.fromkeys(list(ans[ans["labels"].map(len) == 0]["text"]) + [ans["text"][a] for a in ph.nsmallest(10, "sim")["answer"]]))
NEW_PROMPT = """次は、図書館の利用者アンケートの回答のうち、既存の話題の一覧に当てはまりにくかったものです。
既存の話題: {labels}

この中に、既存の話題とは別の話題で、2 件以上の回答に共通するものがあれば、
{{"topics": [{{"label": "...", "answers": [回答の番号, ...]}}]}} の形の JSON だけを出力してください。

{answers}"""
out = generate([NEW_PROMPT.format(labels="、".join(codebook["ラベル"]), answers="\n".join(f"{i}. {t}" for i, t in enumerate(cand, 1)))],
               max_new_tokens=300)[0]
for t in extract_json(out).get("topics", []):
    print(t.get("label"), "←", [cand[i - 1] for i in t.get("answers", []) if 0 < i <= len(cand)])
```

```python
# (5) 件数が少なくても見落としたくない観点は、クラスタに頼らず文埋め込みの検索で確かめる（ruri-v3 の検索用の接頭辞を使う）
checks = ["車いすや段差、エレベーターなどのバリアフリー", "外国語や多言語での案内", "けがや事故などの安全"]
doc = model.encode(["検索文書: " + t for t in ans["text"]], normalize_embeddings=True, show_progress_bar=False)
q = model.encode(["検索クエリ: " + c for c in checks], normalize_embeddings=True)
for c, sims in zip(checks, q @ doc.T):
    print("■", c)
    for i in sims.argsort()[::-1][:5]:
        print(f"  {sims[i]:.3f} [{'、'.join(codebook['ラベル'][j] for j in ans['labels'][i]) or 'ラベルなし'}] {ans['text'][i]}")
```

```python
from sklearn.metrics import adjusted_rand_score

# 8. 答え合わせ（この架空のデータでは正解の話題が分かっている）
# 単一の話題の回答で、クラスタと正解の話題の一致度 (ARI) を比べる: 回答の文全体をクラスタリングした場合と、句をクラスタリングした場合
single = ans["true"].map(len) == 1
ans_emb = model.encode(["トピック: " + t for t in ans["text"]], normalize_embeddings=True, show_progress_bar=False)
direct = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(ans_emb)
first = [l[0] if l else -1 for l in ans["from_cluster"]]
truth = ans["true"][single].map(lambda t: t[0])
print(f"ARI 回答の文全体: {adjusted_rand_score(truth, direct[single]):.3f} / 句: {adjusted_rand_score(truth, np.array(first)[single]):.3f}")

# 各ラベルが付いた回答のうち、最も多い正解の話題とその割合
check_rows = []
for c in report.index:
    top = pd.Series([t for l, ts in zip(ans["labels"], ans["true"]) if c in l for t in ts]).value_counts()
    if len(top):
        check_rows.append({"ラベル": codebook["ラベル"][c], "回答数": report["回答数"][c], "最多の正解の話題": top.index[0],
                           "その回答数": top.iloc[0]})
display(pd.DataFrame(check_rows).style.hide(axis="index"))

# どのラベルでも最多にならなかった正解の話題（コードブックにラベルができなかった話題の候補）と、その回答数
covered = {r["最多の正解の話題"] for r in check_rows}
n_true = ans["true"].explode().value_counts()
print("どのラベルでも最多にならなかった正解の話題:", {t: int(n_true[t]) for t in FRAG if t not in covered})

# 少数意見（バリアフリー、外国語）の回答に付いたラベル
for _, r in ans[ans["true"].map(lambda ts: ts[0] in ("バリアフリー", "外国語") if ts else False)].iterrows():
    print(f"{r.true[0]}: {[codebook['ラベル'][c] for c in r.labels] or 'ラベルなし'} | {r.text}")
```

- データはこのレシピのために作った架空のアンケート（196 件、うち中身のない回答 6 件）で、話題ごとに手で書いた文を 1 つか 2 つつないで回答にした。同じ文が別の回答の一部として何度も出てくるので、実際の自由記述より話題が分かれやすい。LLM は Qwen3-4B-Instruct-2507、文埋め込みは cl-nagoya/ruri-v3-310m で、どちらも Apache 2.0。float16 での GPU メモリの最大は 11.3GB だった。
- この出力では、単一の話題の回答でクラスタと正解の話題の一致度 (ARI) は、回答の文全体をクラスタリングすると 0.114、LLM で取り出した句をクラスタリングすると 0.389 だった。文全体では「いつも利用しています。」のような前置きや言い回しに引きずられやすいことや、2 つの話題にまたがる回答も一緒にクラスタリングしていることが差の原因と考えられるが、確かめてはいない（句の側は、回答の句のクラスタのうち番号が最小のもので比べている）。句に分けると、2 つの話題にまたがる回答（この出力では 87 件が句を 2 つ持つ）も話題ごとに数えられる。この章の「文埋め込みとクラスタリングでトピックを作る（BERTopic 方式）」は LLM を使わずに c-TF-IDF の特徴語で名前を付ける方法、文埋め込みとクラスタリングの章の「文をクラスタリングして話題ごとにまとめる」は文をそのままクラスタリングする基本形である。
- クラスタに付けたラベルを固定の一覧（コードブック）にして、全回答を LLM で付け直すと、件数をラベルの定義に基づいて数えられ、次回の調査でも同じ一覧で比べられる。この出力では、句のクラスタから決めたラベルと LLM が選んだラベルが完全に一致した回答は 45% で、「送迎車の駐車場」（中身は駐車場全般）や「利用環境の整備」のように中身と合わないラベルや曖昧なラベルもある。コードブックにラベルができない話題もある。この出力では、正解の話題では「静かさ・マナー」より多い 22 件の「子ども・イベント」にラベルができず（答え合わせのどのラベルでも最多にならなかった）、児童書コーナーの句は「読書環境の整備」のクラスタに入り、おはなし会、工作教室、授乳室の 3 件はラベルなしになった。報告の前に人がコードブックを見直してラベルと説明を直し（足りない話題は加える）、手順 4 からやり直す。表の回答数は 1 件を複数のラベルに数えるので、合計は回答数を超える（報告ではそのことを書き添える）。
- 粒度は、クラスタ数 `K` を変えるか、細かいラベルを LLM にまとめさせて調整する。この出力では 15 個のラベルが 6 個の分類にまとまった。LLM のまとめ方は抜けや重複が起きうるので、すべてのラベルが 1 回ずつ入っているかを確かめている。
- 件数の少ない意見は、大きなラベルに吸収されやすい。この出力では、バリアフリーの回答 5 件のうち 4 件が「読書環境の整備」に入り、外国語の 4 件は 2 件が「本の充実と予約上限」、2 件がラベルなしになった。取りこぼしの候補（ラベルなしの回答と、中心から遠い句を持つ回答）を LLM に見直させると「案内表示の多言語化」が見つかった（道案内の回答 2 件も混ざった）が、バリアフリーと子ども・イベントは見つからなかった。見落としたくない観点（安全、バリアフリー、差別など）は、件数にかかわらず検索でも確かめると取りこぼしを減らせる。この出力では、外国語の 4 件は上位 4 件に入ったが、バリアフリーは 5 件のうち 3 件が上位 3 件に入り、2 件は上位 5 件に入らなかった。
- 実際のアンケートの自由記述には、名前や連絡先などの個人情報が書かれていることがある。LLM の API（LLM の API の章の「Gemini に決まった形式（JSON スキーマ）で出力させる」など）を使えばより大きなモデルでラベルを付けられるが、外部に送る前に、調査の同意の範囲や組織の規則を確かめ、必要なら個人情報を伏せる。

## 日付のあるテキストから急に増えた語を見つける（バースト検出） {burst_detection}

日付の付いたニュース記事の流れから、ある時期に急に増えた語（話題）と、その期間を見つけたい。

```python
!pip install -q fugashi unidic-lite matplotlib-fontja
```

```python
import io, tarfile, urllib.request
import pandas as pd

# livedoor ニュースコーパス (株式会社ロンウイット配布)。記事は CC BY-ND 2.1 JP (表示・改変禁止)
def load_livedoor():
    url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
    with urllib.request.urlopen(url) as r:
        tar = tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")
    rows = []
    for m in tar.getmembers():
        parts = m.name.split("/")
        if m.isfile() and len(parts) == 3 and parts[2] != "LICENSE.txt":
            lines = tar.extractfile(m).read().decode("utf-8").split("\n")
            rows.append({"label": parts[1], "date": lines[1], "title": lines[2], "text": "\n".join(lines[3:])})
    return pd.DataFrame(rows)

df = load_livedoor()
df["date"] = pd.to_datetime(df["date"].str[:10])
# カテゴリごとに収録の時期が違うので、毎月の記事がある「Sports Watch」だけを使う。
# 収録の最初と最後の月 (2010-02 は 12 件、2012-09 は 3 件) は記事が少ないので除く
df = df[(df.label == "sports-watch") & (df.date >= "2010-03-01") & (df.date < "2012-09-01")].reset_index(drop=True)
df["month"] = df["date"].dt.to_period("M")
print(len(df), "件")
print(df.groupby("month").size().to_string())
```

```python
import re
from collections import Counter
from fugashi import Tagger

tagger = Tagger()

def clean(title, text):
    """時期によって書式が変わる定型部分 (見出しの【Sports Watch】、関連リンクの一覧) を除く"""
    title = re.sub(r"^【[^】]*】\s*", "", title)
    text = text.split("関連リンク")[0]
    return title + "\n" + "\n".join(l for l in text.split("\n") if not l.startswith(("・", "■", "【")))

def nouns(text):
    return {w.surface for w in tagger(text) if w.feature.pos1 == "名詞" and w.feature.pos2 in ("普通名詞", "固有名詞")
            and len(w.surface) > 1}

df["words"] = [nouns(clean(a, b)) for a, b in zip(df["title"], df["text"])]   # 1 記事に何回出ても 1 と数える (文書頻度)

months = pd.period_range(df["month"].min(), df["month"].max(), freq="M")
d = df.groupby("month").size().reindex(months, fill_value=0).to_numpy()          # 各月の記事数
total = Counter(w for ws in df["words"] for w in ws)
vocab = sorted(w for w, c in total.items() if c >= 10)                                # 全体で 10 記事以上に出る語
R = pd.DataFrame(0, index=vocab, columns=months)                                  # 語 × 月 の記事数
vs = set(vocab)
for m, ws in zip(df["month"], df["words"]):
    for w in ws & vs:
        R.loc[w, m] += 1
print(R.shape)
```

```python
# 方法 1: 単純な頻度の比。その月の出現率を、直前 3 か月の出現率と比べる
def ratio_table(R, d, prev=3, min_count=5):
    rows = []
    for t in range(prev, len(d)):
        cur = (R.iloc[:, t] + 0.5) / (d[t] + 1)
        base = (R.iloc[:, t - prev:t].sum(axis=1) + 0.5) / (d[t - prev:t].sum() + 1)
        for w in R.index[R.iloc[:, t] >= min_count]:
            rows.append({"word": w, "month": str(R.columns[t]), "count": R.loc[w, R.columns[t]], "ratio": cur[w] / base[w]})
    return pd.DataFrame(rows).sort_values("ratio", ascending=False, kind="stable").reset_index(drop=True)

ratio = ratio_table(R, d)
ratio.head(15)
```

```python
import numpy as np
from scipy.stats import binom

# 方法 2: Kleinberg (2002) のバースト検出 (バッチのデータ用の 2 状態モデル)。
# 各月の記事数 d_t のうち語を含む記事数 r_t が、通常状態 (出現率 p0) と
# バースト状態 (出現率 p1 = s * p0) のどちらで説明しやすいかを、状態を上げるコスト gamma * log(n) と合わせて
# ビタビアルゴリズムで決める
def kleinberg_bursts(r, d, s=2.0, gamma=1.0):
    r, d = np.asarray(r), np.asarray(d)
    n = len(r)
    p0 = r.sum() / d.sum()
    p1 = min(s * p0, 0.9999)
    cost = -np.stack([binom.logpmf(r, d, p0), binom.logpmf(r, d, p1)])   # 2 × n
    up = gamma * np.log(n)
    C = np.array([cost[0, 0], cost[1, 0] + up])
    back = np.zeros((n, 2), dtype=int)
    for t in range(1, n):
        to0 = [C[0], C[1]]                # 状態を下げるコストは 0
        to1 = [C[0] + up, C[1]]
        back[t] = [np.argmin(to0), np.argmin(to1)]
        C = np.array([min(to0) + cost[0, t], min(to1) + cost[1, t]])
    q = np.zeros(n, dtype=int)
    q[-1] = C.argmin()
    for t in range(n - 1, 0, -1):
        q[t - 1] = back[t, q[t]]
    # 連続するバースト状態を 1 つのバーストとし、強さ = 区間内のコストの差の合計
    out, gain = [], cost[0] - cost[1]
    for t in range(n):
        if q[t] == 1 and (t == 0 or q[t - 1] == 0):
            start = t
        if q[t] == 1 and (t == n - 1 or q[t + 1] == 0):
            out.append((start, t, gain[start:t + 1].sum()))
    return out

rows = [{"word": w, "begin": str(months[b]), "end": str(months[e]), "b": b, "e": e, "weight": wt}
        for w in R.index for b, e, wt in kleinberg_bursts(R.loc[w].to_numpy(), d)]
bursts = pd.DataFrame(rows).sort_values("weight", ascending=False, kind="stable").reset_index(drop=True)
print(len(bursts), "個のバースト")
bursts.head(15)[["word", "begin", "end", "weight"]]
```

```python
# 2 つの方法の比較: 頻度の比の上位 15 件が、Kleinberg では何位のバーストの期間に入っているか
def kleinberg_rank(w, m):
    t = months.get_loc(pd.Period(m, "M"))
    hit = bursts[(bursts.word == w) & (bursts.b <= t) & (bursts.e >= t)]
    return hit.index[0] + 1 if len(hit) else None

cmp = ratio.head(15).copy()
cmp["kleinberg_rank"] = pd.array([kleinberg_rank(w, m) for w, m in zip(cmp.word, cmp.month)], dtype="Int64")
print("比が 3 倍以上の (語, 月):", (ratio.ratio >= 3).sum(), "件 / Kleinberg のバースト:", len(bursts), "個")
cmp
```

```python
import matplotlib.pyplot as plt
import matplotlib_fontja

# 強いバースト上位 20 個の期間を横棒で示す
top = bursts.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(top)), top["e"] - top["b"] + 1, left=top["b"] - 0.5, color="tab:orange")   # 月 i を i-0.5〜i+0.5 に描く
ax.set_yticks(range(len(top)), top["word"])
ticks = list(range(0, len(months), 3))
ax.set_xticks(ticks, [str(months[i]) for i in ticks], rotation=45)
ax.set_xlim(-0.5, len(months) - 0.5); ax.grid(axis="x", alpha=0.3)
ax.set_title("Kleinberg のバースト (強さの上位 20)")
plt.tight_layout(); plt.show()
```

```python
# 語ごとの出現率の推移。橙の帯が Kleinberg のバースト期間、赤い点が頻度の比で 3 倍以上になった月
words = ["ロンドン", "メダル", "なでしこ", "震災"]
fig, axes = plt.subplots(len(words), 1, figsize=(10, 2.2 * len(words)), sharex=True)
for ax, w in zip(axes, words):
    ax.plot(range(len(months)), R.loc[w] / d, marker=".", color="tab:blue")
    for _, b in bursts[bursts.word == w].iterrows():
        ax.axvspan(b["b"] - 0.5, b["e"] + 0.5, color="tab:orange", alpha=0.3)
    hit = ratio[(ratio.word == w) & (ratio.ratio >= 3)]
    idx = [months.get_loc(pd.Period(m, "M")) for m in hit["month"]]
    ax.scatter(idx, (R.loc[w] / d).iloc[idx], color="red", zorder=3)
    ax.set_ylabel(w); ax.set_ylim(bottom=0)
axes[-1].set_xticks(ticks, [str(months[i]) for i in ticks], rotation=45)
axes[-1].set_xlim(-0.5, len(months) - 0.5)
fig.suptitle("語を含む記事の割合 (月ごと)")
plt.tight_layout(); plt.show()
```

- 頻度の比は計算が簡単だが、直前にほとんど出ていない語が数記事出ただけで比が大きくなる。この出力でも比の上位 15 件は 5〜9 記事の語で、そのうち 6 件は Kleinberg のバーストの期間に入っていない。逆に直前の数か月も多い月は比が大きくならないので、「ロンドン」のように 2012 年に数か月かけて増えた語は、2012 年には 3 倍を超えない。
- Kleinberg のバースト検出は KDD 2002 で発表された（雑誌版は Data Mining and Knowledge Discovery, 2003）。ここでは論文のバッチ用のモデルで、出現率 p0 = R/D の状態と p1 = s·p0 の状態のどちらにいるかを、二項分布の負の対数尤度と状態を上げるコスト γ ln n の合計が最小になるように決める。頻度の比と違って始まりと終わりの期間が得られ、記事数の少ない月のばらつきを二項分布で考慮した強さ（p1 で説明したときのコストの減りの合計）で順位を付けられる。
- `s` を大きくすると大きく増えた語だけがバーストになる。`gamma` を大きくするとバーストに入りにくくなり、弱い増加や短い増加はバーストになりにくい。月の記事数が 19〜64 件と少ないので月単位にまとめたが、記事の多いデータなら週や日の単位にできる。
- PyPI の `burst_detection` (0.1.3) は NumPy から削除された `np.float` を使っていて、検証に使った Colab のイメージ (NumPy 2.4) ではエラーになり、分類も "Free for non-commercial use" なので、ここでは 2 状態のビタビを自分で書いた。メールのように 1 件ずつ届く時刻の列には、論文の前半の連続時間のモデル（R の `bursts` パッケージなど）を使う。
- livedoor ニュースコーパスの記事は CC BY-ND 2.1 JP（表示・改変禁止）。カテゴリごとに収録の時期が違う（例: topic-news は 2011 年 10 月から、smax は 2012 年 4 月から）ので、収録の始まりがバーストに見えないよう Sports Watch だけを使い、時期によって変わる定型文（見出しの【Sports Watch】、関連リンクの一覧）も除いた。それでも上位の「ネット」「アサヒ」「週刊」「同誌」などは話題より書き方の変化を拾った可能性があり（中身は確かめていない）、検出結果は元の記事を読んで確かめる。
- 名詞を 1 語ずつ数えているので、「川澄」「穂希」「奈穂美」のように人名が分かれて別々に出る。名詞の連続をつないで数える（この章の「文書からキーワードを抽出する（埋め込みによる KeyBERT 方式）」の `candidates`）と読みやすくなる。
- 語の代わりに、この章の「LDA で文書集合のトピックを見つける」や「文埋め込みとクラスタリングでトピックを作る（BERTopic 方式）」で各文書に割り当てたトピックを数えて r_t にすれば、同じ関数で話題のバーストを検出できる。2 つの期間だけを比べるなら、コーパスの統計と可視化の章の「2 つのコーパスを比べて特徴語を見つける（対数尤度比）」も使える。
