# 検索

転置索引と BM25 によるキーワード検索、埋め込みによる意味検索、その組み合わせと並べ替え、評価、モデルの微調整。質問応答や RAG の土台になる。

## 転置索引を自作してブール検索・フレーズ検索をする {inverted_index}

検索エンジンの基本である転置索引を自分で作り、AND / OR / NOT や語順どおりのフレーズで検索したい。

```python
!pip install -q fugashi unidic-lite
```

```python
from collections import defaultdict
from fugashi import Tagger

tagger = Tagger()
docs = [
    "東京都立大学で自然言語処理の研究をしている。",
    "自然言語処理では機械翻訳の研究が盛んだ。",
    "東京で機械学習の勉強会が開かれた。",
    "京都大学は言語処理の研究で知られる。",
    "機械翻訳は自然な言語を処理する技術の一つである。",
    "東京と京都を結ぶ新幹線に乗った。",
]

def tokenize(text):
    return [w.surface for w in tagger(text) if w.feature.pos1 not in ("助詞", "助動詞", "補助記号")]

# 転置索引: 単語 → {文書番号: [出現位置, ...]}
index = defaultdict(dict)
for doc_id, doc in enumerate(docs):
    for pos, term in enumerate(tokenize(doc)):
        index[term].setdefault(doc_id, []).append(pos)

print(len(index), "語")
for term in ["東京", "研究", "処理"]:
    print(term, dict(index[term]))
```

```python
ALL = set(range(len(docs)))
posting = lambda term: set(index.get(term, {}))

def AND(*terms): return set.intersection(*[posting(t) for t in terms])
def OR(*terms): return set.union(*[posting(t) for t in terms])
def NOT(term): return ALL - posting(term)

def phrase(*terms):
    """語がこの順番で隣り合って出現する文書 (位置情報を使う)"""
    hits = set()
    for doc_id in AND(*terms):
        if any(all(p + i in index[t][doc_id] for i, t in enumerate(terms)) for p in index[terms[0]][doc_id]):
            hits.add(doc_id)
    return hits

queries = {
    "東京 AND 研究": AND("東京", "研究"),
    "機械 AND (翻訳 OR 学習)": posting("機械") & OR("翻訳", "学習"),
    "研究 AND NOT 東京": posting("研究") & NOT("東京"),
    'フレーズ "自然 言語 処理"': phrase("自然", "言語", "処理"),
    "自然 AND 言語 AND 処理": AND("自然", "言語", "処理"),
}
for q, result in queries.items():
    print(f"{q}\n   → {sorted(result)} {[docs[i][:18] for i in sorted(result)]}")
```

- 転置索引は「この単語を含む文書はどれか」を即座に引ける構造。全文書を走査せずに済むので、数億件の文書でも高速に検索できる。
- 出現位置まで持っておくと、フレーズ検索や「近くに出現する」という条件の検索ができる。最後の 2 つの結果の違い (語順どおりに並んでいるか) がその効果。
- 日本語では単語の区切り方が検索結果を左右する。「東京都立大学」が 1 語になる辞書だと「東京」で検索しても当たらない。対策は、この章の文字 n-gram による検索。
- 実用の検索エンジン (Lucene、Elasticsearch、Tantivy) も、この構造を圧縮し、スコア計算と組み合わせたもの。

## TF-IDF とコサイン類似度で似た文書を探す {tfidf_similarity}

ニューラルモデルを使わずに、単語の重なりに基づいて類似文書を高速に検索したい。

```python
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
docs = [d for d in data.data if len(d) > 200][:5000]
vec = TfidfVectorizer(stop_words="english", max_df=0.5, min_df=2, sublinear_tf=True)
X = vec.fit_transform(docs)   # 疎行列 (文書数 × 語彙数)。行は L2 正規化済み
print(X.shape, f"非ゼロ要素の割合 {X.nnz / (X.shape[0] * X.shape[1]):.4f}")
```

```python
def search(query, k=3):
    sims = cosine_similarity(vec.transform([query]), X).ravel()
    for i in np.argsort(-sims)[:k]:
        print(f"  {sims[i]:.3f}  {' '.join(docs[i].split())[:90]}")

for q in ["space shuttle launch NASA", "best graphics card for windows"]:
    print(q)
    search(q)
```

```python
# 文書同士の類似度: 0 番の文書に似た文書
sims = cosine_similarity(X[0], X).ravel()
print(" ".join(docs[0].split())[:100])
for i in np.argsort(-sims)[1:3]:
    print(f"  {sims[i]:.3f}  {' '.join(docs[i].split())[:90]}")
```

- 日本語の場合は `TfidfVectorizer(tokenizer=分かち書き関数, token_pattern=None)` とする。
- 文字 n-gram (`analyzer="char", ngram_range=(2, 3)`) にすると、分かち書きなしで表記ゆれにも強くなる。

## BM25 を自分で実装してパラメータの働きを確かめる {bm25_scratch}

キーワード検索の標準的なスコアである BM25 が、何を計算しているのかを理解したい。パラメータ k1 と b を変えると何が起きるかを見たい。

```python
!pip install -q rank_bm25 fugashi unidic-lite
```

```python
import math
from collections import Counter
import numpy as np
from fugashi import Tagger

tagger = Tagger()
tok = lambda s: [w.surface for w in tagger(s) if w.feature.pos1 not in ("助詞", "助動詞", "補助記号")]

docs = [
    "猫は可愛い。",
    "猫と犬はどちらも人気のあるペットで、猫は室内で、犬は散歩が必要なので屋外でも飼われることが多い。",
    "猫、猫、猫。猫が大好きだ。",
    "犬の散歩は毎日の習慣である。",
    "ペットを飼うと生活が豊かになると言われている。",
]
tokenized = [tok(d) for d in docs]
N, avgdl = len(docs), np.mean([len(d) for d in tokenized])
df = Counter(t for d in tokenized for t in set(d))

def bm25(query, k1=1.5, b=0.75):
    scores = []
    for d in tokenized:
        tf, total = Counter(d), 0.0
        for t in tok(query):
            if t not in tf:
                continue
            idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)           # 珍しい語ほど重い
            norm = tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))   # 出現回数は頭打ち、長い文書は割り引く
            total += idf * norm
        scores.append(total)
    return np.array(scores)

for q in ["猫", "猫 ペット"]:
    print(q, bm25(q).round(2).tolist())
```

```python
# k1: 出現回数をどこまで重視するか / b: 文書の長さでどこまで割り引くか
print("文書の長さ:", [len(d) for d in tokenized])
for k1, b in [(1.5, 0.75), (0.0, 0.75), (10.0, 0.75), (1.5, 0.0), (1.5, 1.0)]:
    s = bm25("猫", k1, b)
    print(f"k1={k1:4} b={b:4}  スコア {s[:3].round(2).tolist()}  順位 {np.argsort(-s)[:3].tolist()}")
```

```python
# ライブラリの実装と順位が一致することを確かめる (IDF の式に細かい違いがあるので値は少し違う)
from rank_bm25 import BM25Okapi

lib = BM25Okapi(tokenized, k1=1.5, b=0.75).get_scores(tok("猫 ペット"))
mine = bm25("猫 ペット")
print(lib.round(2).tolist(), np.argsort(-lib).tolist())
print(mine.round(2).tolist(), np.argsort(-mine).tolist())
```

- BM25 = 「語の珍しさ (IDF)」×「その文書での出現回数 (頭打ちあり)」を、クエリの語について足したもの。
- k1 を 0 にすると出現回数を無視する (含むか含まないかだけ)。大きくすると、同じ語を繰り返す文書ほど高くなる。
- b を 0 にすると文書の長さを無視し、1 にすると長さに比例して割り引く。長い文書は語を多く含みやすいので、その有利さを打ち消している。
- 既定値 (k1=1.2〜2.0、b=0.75) は多くの場合そのままでよい。タイトルのような短い文書ばかりなら b を小さくする。

## BM25 で日本語のキーワード検索をする {bm25_ja}

単語の一致に基づく古典的な検索 (BM25) を日本語文書に対して行いたい。固有名詞や型番の検索には今でも強い。

```python
!pip install -q rank_bm25 fugashi unidic-lite
```

```python
from fugashi import Tagger
from rank_bm25 import BM25Okapi

tagger = Tagger()

def tokenize(text):
    return [w.feature.orthBase or w.surface for w in tagger(text.lower())
            if w.feature.pos1 not in ("助詞", "助動詞", "補助記号", "空白")]

docs = [
    "富士山は標高 3776 メートルで、日本で最も高い山である。",
    "東京スカイツリーは高さ 634 メートルの電波塔である。",
    "琵琶湖は滋賀県にある日本最大の湖である。",
    "信濃川は全長 367 キロメートルで、日本で最も長い川である。",
    "形態素解析とは、文を単語に分割して品詞を付与する処理である。",
    "BERT は Transformer のエンコーダを用いた事前学習モデルである。",
]
bm25 = BM25Okapi([tokenize(d) for d in docs])

for q in ["日本で一番高い山", "BERT とは何ですか", "日本一大きい湖はどこ？"]:
    scores = bm25.get_scores(tokenize(q))
    print(q, tokenize(q))
    for i in scores.argsort()[::-1][:2]:
        print(f"  {scores[i]:.2f}  {docs[i]}")
```

- 3 つ目のクエリのように語が一致しない言い換え (大きい ↔ 最大) には弱い。埋め込み検索と組み合わせると補い合える (ハイブリッド検索のレシピを参照)。
- 大規模なら `bm25s` (高速な実装) や Elasticsearch / OpenSearch、Pyserini を使う。

## Elasticsearch で日本語の全文検索をする (kuromoji) {elasticsearch_ja}

自前の BM25 ではなく、検索エンジンの Elasticsearch で日本語の文書を全文検索したい。形態素解析器 kuromoji を組み込んだアナライザを作り、スコアやハイライト付きで検索結果を見たい。

```python
# 所要時間の目安: Colab で 2〜4 分 (本体の配布物が約 840MB)
ES = "elasticsearch-9.5.6"
!wget -q https://artifacts.elastic.co/downloads/elasticsearch/{ES}-linux-x86_64.tar.gz
!tar -xzf {ES}-linux-x86_64.tar.gz && rm {ES}-linux-x86_64.tar.gz
!{ES}/bin/elasticsearch-plugin install --batch analysis-kuromoji > /dev/null
# 学習用の設定: 1 台構成、手元 (127.0.0.1) だけで待ち受け、セキュリティ機能 (認証・TLS) は無効。
# 機械学習の機能は使わないので切る (メモリの節約)
!printf 'discovery.type: single-node\nnetwork.host: 127.0.0.1\nhttp.port: 9200\nxpack.security.enabled: false\nxpack.ml.enabled: false\n' >> {ES}/config/elasticsearch.yml
# ヒープ (Java のメモリ) は 1GB に固定する (指定しないと搭載メモリと役割から自動で決まる)
!printf -- '-Xms1g\n-Xmx1g\n' > {ES}/config/jvm.options.d/heap.options
# Elasticsearch は root では起動できないので、Colab に最初からいる daemon ユーザーの持ち物にしておく
!chown -R daemon:daemon {ES}
!pip install -q "elasticsearch>=9,<10" rank_bm25 fugashi unidic-lite
```

```python
import subprocess, time, urllib.request
import elasticsearch

# バックグラウンドで起動する (Colab で 1 分ほど)。root では起動を拒否されるので daemon ユーザーで動かす
server = subprocess.Popen([f"{ES}/bin/elasticsearch"], user="daemon", group="daemon",
                          stdout=open("es.log", "w"), stderr=subprocess.STDOUT)
for _ in range(180):   # HTTP で応答するまで待つ
    if server.poll() is not None:   # 起動に失敗して終了した
        print(open("es.log").read()[-3000:]); break
    try:
        urllib.request.urlopen("http://localhost:9200", timeout=1); break
    except Exception:
        time.sleep(1)
es = elasticsearch.Elasticsearch("http://localhost:9200")
info = es.info()
print("サーバー", info["version"]["number"], "/ Lucene", info["version"]["lucene_version"], "/ クライアント", elasticsearch.__versionstr__)
print([p["component"] for p in es.cat.plugins(format="json")])
```

```python
# トークンの分かれ方を _analyze API で見る: 標準のアナライザ / kuromoji の通常モード / 検索モード
text = "関西国際空港で東京スカイツリーのお土産を買いました"
show = lambda **kw: print([t["token"] for t in es.indices.analyze(text=text, **kw)["tokens"]])
show(analyzer="standard")
show(tokenizer={"type": "kuromoji_tokenizer", "mode": "normal"})
show(tokenizer={"type": "kuromoji_tokenizer", "mode": "search"})
# 品詞での除外 (助詞・助動詞など) と原形化 (買い → 買う) まで入れたもの
show(tokenizer={"type": "kuromoji_tokenizer", "mode": "search"},
     filter=["kuromoji_baseform", "kuromoji_part_of_speech", "cjk_width", "ja_stop", "kuromoji_stemmer", "lowercase"])
```

```python
# 索引 (index) を作る。アナライザとマッピング (フィールドの型) はここで決める
settings = {
    "number_of_shards": 1, "number_of_replicas": 0,
    "analysis": {
        "tokenizer": {"ja_search": {"type": "kuromoji_tokenizer", "mode": "search"}},
        "analyzer": {"ja": {"type": "custom", "tokenizer": "ja_search",
                            "char_filter": ["kuromoji_iteration_mark"],
                            "filter": ["kuromoji_baseform", "kuromoji_part_of_speech", "cjk_width",
                                       "ja_stop", "kuromoji_stemmer", "lowercase"]}}},
    # BM25 のパラメータを変えた類似度も定義できる (b=0: 文書の長さで割り引かない)
    "similarity": {"bm25_nolen": {"type": "BM25", "k1": 1.2, "b": 0.0}},
}
mappings = {"properties": {
    "text": {"type": "text", "analyzer": "ja",
             "fields": {"nolen": {"type": "text", "analyzer": "ja", "similarity": "bm25_nolen"}}},
    "category": {"type": "keyword"},   # 絞り込み用 (解析しない)
}}
es.indices.delete(index="docs", ignore_unavailable=True)
es.indices.create(index="docs", settings=settings, mappings=mappings)
```

```python
from elasticsearch import helpers

# この章の「BM25 で日本語のキーワード検索をする」と同じ文書
docs = [
    ("地理", "富士山は標高 3776 メートルで、日本で最も高い山である。"),
    ("建築", "東京スカイツリーは高さ 634 メートルの電波塔である。"),
    ("地理", "琵琶湖は滋賀県にある日本最大の湖である。"),
    ("地理", "信濃川は全長 367 キロメートルで、日本で最も長い川である。"),
    ("技術", "形態素解析とは、文を単語に分割して品詞を付与する処理である。"),
    ("技術", "BERT は Transformer のエンコーダを用いた事前学習モデルである。"),
]
actions = ({"_index": "docs", "_id": str(i), "_source": {"category": c, "text": t}} for i, (c, t) in enumerate(docs))
print(helpers.bulk(es, actions, refresh=True))   # refresh=True: 登録した文書をすぐ検索できるようにする
print(es.count(index="docs")["count"], "件")
```

```python
ja = lambda s: [t["token"] for t in es.indices.analyze(index="docs", analyzer="ja", text=s)["tokens"]]

def search(query, field="text", category=None, size=2):
    q = {"match": {field: query}}   # クエリも同じアナライザで解析され、どれかの語を含む文書が当たる (OR)
    if category:   # bool の filter はスコアに影響しない絞り込み
        q = {"bool": {"must": [q], "filter": [{"term": {"category": category}}]}}
    res = es.search(index="docs", query=q, size=size, highlight={"fields": {field: {}}})
    print(query, ja(query), f"→ {res['hits']['total']['value']} 件ヒット")
    for h in res["hits"]["hits"]:
        print(f"  {h['_score']:.3f}  [{h['_source']['category']}] {h['highlight'][field][0]}")

for q in ["日本で一番高い山", "BERT とは何ですか", "日本一大きい湖はどこ？"]:
    search(q)
search("日本", size=4)
search("日本", category="技術")
print(ja("高さ"), ja("お土産を買いました"))
```

```python
# match は既定で OR。minimum_should_match で「クエリの語のうち何割を含むか」を指定できる
for msm in ["1", "75%", "100%"]:
    res = es.search(index="docs", query={"match": {"text": {"query": "日本で一番高い山", "minimum_should_match": msm}}})
    print(f"{msm:>4}", [(h["_id"], round(h["_score"], 3)) for h in res["hits"]["hits"]])
```

```python
# 自前の BM25 (この章の「BM25 で日本語のキーワード検索をする」の rank_bm25 + fugashi) とスコアを並べる
import numpy as np
from fugashi import Tagger
from rank_bm25 import BM25Okapi

tagger = Tagger()
tok = lambda s: [w.feature.orthBase or w.surface for w in tagger(s.lower())
                 if w.feature.pos1 not in ("助詞", "助動詞", "補助記号", "空白")]
bm25 = BM25Okapi([tok(t) for _, t in docs])
for q in ["日本で一番高い山", "日本 最も 長い", "メートル"]:
    es_score = np.zeros(len(docs))
    for h in es.search(index="docs", query={"match": {"text": q}})["hits"]["hits"]:
        es_score[int(h["_id"])] = h["_score"]
    print(q, "\n   Elasticsearch:", es_score.round(2).tolist(), "\n   rank_bm25    :", bm25.get_scores(tok(q)).round(2).tolist())
```

```python
# スコアの内訳 (explain): Lucene の BM25 の式の各項が見える
def print_tree(e, depth=0):
    print("  " * depth + f"{e['value']:.3f}  {e['description'][:60]}")
    for d in e["details"]:
        print_tree(d, depth + 1)

res = es.search(index="docs", query={"match": {"text": "日本"}}, explain=True, size=1)
print_tree(res["hits"]["hits"][0]["_explanation"])
```

```python
# BM25 の b を 0 にしたフィールド (text.nolen) では、文書の長さで割り引かなくなる
for field in ["text", "text.nolen"]:
    hits = es.search(index="docs", query={"match": {field: "日本"}})["hits"]["hits"]
    print(f"{field:10}", [(h["_id"], len(ja(h["_source"]["text"])), round(h["_score"], 3)) for h in hits])
```

```python
# 後片付け: サーバーを止める (終了処理に 30 秒ほどかかる)
es.close()
server.terminate()
server.wait(timeout=120)
print("終了コード", server.returncode)   # 143 = SIGTERM で終了した
```

- Elasticsearch は、検索ライブラリ Lucene の転置索引を HTTP の API で使えるようにした検索サーバー。この版 (9.5.6) は Lucene 10.5.2 を使っている。スコアの既定は BM25 で、explain の内訳のとおり k1=1.2、b=0.75、IDF は log(1 + (N − n + 0.5) / (n + 0.5))。boost の 2.2 は k1 + 1 にあたり、この章の「BM25 を自分で実装してパラメータの働きを確かめる」の式と同じ形になる。
- 自前の BM25 (rank_bm25 の BM25Okapi) と比べると、スコアが 0 より大きい文書の順位は同じだが、スコアの値とヒットする文書の数が違う。BM25Okapi の IDF は log((N − n + 0.5) / (n + 0.5)) なので、6 件中 3 件に出る「日本」の IDF が 0 になり、「日本」だけが一致する文書のスコアが 0 になる。Lucene の IDF は 1 を足してから対数をとるので、必ず正になる。k1 の既定値 (rank_bm25 は 1.5、Lucene は 1.2) や、単語の区切り方 (unidic-lite と kuromoji に内蔵の辞書) も違う。
- kuromoji の検索モードは、長い複合語を、全体 (関西国際空港) と部分 (関西 / 国際 / 空港) の両方のトークンにするので、「空港」のような部分の語でも当たるようになる。通常モードでは複合語が 1 語のまま。標準のアナライザ (standard) は、漢字やひらがなを 1 文字ずつに分けてしまう (カタカナの続きは 1 語)。
- 品詞による除外 (`kuromoji_part_of_speech`) とストップワード (`ja_stop`) で、「で」「を」「とは」「です」のような機能語が消える (「BERT とは何ですか」→ bert, 何)。`kuromoji_baseform` は活用語を原形に戻す (買い → 買う)。その結果、「高さ」の「高」も「高い」になり、「日本で一番高い山」で東京スカイツリーの文書がヒットしている。原形化は検索漏れを減らすが、このような意図しない一致も増やす。
- 「日本一」は 1 語として解析されるので、「日本」とは一致しない。「大きい」と「最大」のような言い換えに弱いのも自前の BM25 と同じで、埋め込み検索との組み合わせ (この章の「キーワード検索と埋め込み検索を組み合わせる」) が有効。
- `match` は既定で、クエリの語のどれかを含めばヒットする (OR)。`minimum_should_match` を 75% にすると、「日本で一番高い山」(4 語) では 3 語以上を含む富士山の文書だけになる。bool の `filter` はスコアを変えずに絞り込む。BM25 のパラメータは索引の `similarity` で指定でき、b=0 のフィールド (text.nolen) では、文書の長さ (dl) が 6、8、10 と違っても「日本」のスコアが同じになる。アナライザやマッピングは索引を作るときに決め、変えるときは文書を入れ直す。
- ここではセキュリティ機能 (認証と HTTPS) を無効にしている。127.0.0.1 だけで待ち受ける学習用の設定であり、外から接続できるサーバーでは無効にしない。設定しないと、初回の起動時に証明書が作られて HTTPS と認証が必須になり、バックグラウンドで起動した場合は `bin/elasticsearch-reset-password` で elastic ユーザーのパスワードを設定する必要がある。
- 配布物 (tar.gz) のライセンスは Elastic License 2.0。ソースコードの大部分は AGPLv3、SSPL、Elastic License 2.0 から選べる (リポジトリの LICENSE.txt)。Apache License 2.0 だったころの Elasticsearch (7.10.2) のコードから派生した OpenSearch もあり、こちらは Apache License 2.0 で、analysis-kuromoji プラグインも用意されている。

## 単語に区切らず文字 n-gram で検索する (表記ゆれに強い検索) {char_ngram_search}

形態素解析の区切り方や表記ゆれのせいで検索に当たらない。単語に区切らずに、文字 2-gram で索引を作りたい。

```python
!pip install -q fugashi unidic-lite
```

```python
import unicodedata
import numpy as np
from fugashi import Tagger
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "東京都立大学のサーバーにコンピューターを接続する。",
    "大阪府立図書館でインターフェイスの本を借りた。",
    "引っ越しの見積もりをウェブで申し込んだ。",
    "京都の寺を巡るツアーに参加した。",
    "新しいプリンタのドライバをインストールした。",
    "国際会議で機械翻訳の研究を発表した。",
]
queries = ["コンピュータ", "サーバ", "インターフェース", "引越し 見積り", "プリンター ドライバー", "都立大", "京都"]

tagger = Tagger()
norm = lambda s: unicodedata.normalize("NFKC", s).lower()
word_vec = TfidfVectorizer(tokenizer=lambda s: [w.surface for w in tagger(norm(s))], token_pattern=None, lowercase=False)
char_vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 2), preprocessor=lambda s: norm(s).replace(" ", ""), lowercase=False)
Xw, Xc = word_vec.fit_transform(docs), char_vec.fit_transform(docs)

def top(vec, X, q):
    sims = (X @ vec.transform([q]).T).toarray().ravel()
    i = int(sims.argmax())
    return f"{sims[i]:.2f} {docs[i][:16]}" if sims[i] > 0 else "(ヒットなし)"

for q in queries:
    print(f"{q}\n   単語: {top(word_vec, Xw, q)}\n   文字 2-gram: {top(char_vec, Xc, q)}")
```

```python
# 文字 n-gram の弱点: 単語の一部に一致してしまう
print([g for g in char_vec.build_analyzer()("東京都")])
for q in ["京都", "都立"]:
    sims = (Xc @ char_vec.transform([q]).T).toarray().ravel()
    print(q, "→", [(round(float(s), 2), docs[i][:10]) for i, s in enumerate(sims) if s > 0])
```

- 文字 n-gram は、長音の有無 (サーバ / サーバー)、送り仮名 (引っ越し / 引越し)、複合語の一部 (都立大) のような表記の違いに強い。検索漏れが減る。
- かわりに「東京都」の中の「京都」のような、意味のない一致が増える。単語の検索と併用し、単語で一致したものを上位にする構成が実用的。
- 日本語の全文検索エンジンでは、形態素解析による索引と 2-gram の索引を両方持つのが定石。
- 表記ゆれの吸収には、検索前の正規化 (NFKC、長音の統一、Sudachi の正規化形) も有効。

## 検索結果の上位からクエリを拡張する (擬似適合性フィードバック) {query_expansion}

短いクエリでは語が足りず、関連文書に当たらない。最初の検索結果の上位に出てくる語をクエリに足して、もう一度検索したい。

```python
!pip install -q rank_bm25 fugashi unidic-lite
```

```python
import io, tarfile, urllib.request
from collections import Counter
import numpy as np
import pandas as pd
from fugashi import Tagger
from rank_bm25 import BM25Okapi

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
tagger = Tagger()
tok = lambda s: [w.surface.lower() for w in tagger(s) if w.feature.pos1 in ("名詞", "動詞", "形容詞") and w.feature.pos2 not in ("非自立可能", "数詞") and len(w.surface) > 1]
corpus = [tok(t[:600]) for t in df["text"]]
bm25 = BM25Okapi(corpus)
doc_freq = Counter(t for d in corpus for t in set(d))
```

```python
def expand(query_terms, top_docs=5, n_terms=5):
    """上位文書に多く、コーパス全体では珍しい語 (TF × IDF) をクエリに足す"""
    first = np.argsort(-bm25.get_scores(query_terms))[:top_docs]
    tf = Counter(t for i in first for t in corpus[i])
    weight = {t: c * np.log(len(corpus) / doc_freq[t]) for t, c in tf.items() if t not in query_terms}
    return [t for t, _ in sorted(weight.items(), key=lambda x: -x[1])[:n_terms]]

for q in ["スマホ 電池", "サッカー 日本代表", "結婚 悩み"]:
    terms = tok(q)
    added = expand(terms)
    before = np.argsort(-bm25.get_scores(terms))[:3]
    after = np.argsort(-bm25.get_scores(terms + added))[:3]
    print(f"{q} + {added}")
    print("   拡張前:", [df['title'][i][:22] for i in before])
    print("   拡張後:", [df['title'][i][:22] for i in after])
```

```python
# 効果を測る: カテゴリが同じ記事を「関連あり」とみなして、上位 10 件の適合率を比べる
queries = {"スマホ アプリ": "smax", "映画 公開": "movie-enter", "試合 選手": "sports-watch", "恋愛 結婚": "dokujo-tsushin", "家電 売れ筋": "kaden-channel", "パソコン ソフト": "it-life-hack"}
rows = []
for q, label in queries.items():
    terms = tok(q)
    p = lambda ts: (df["label"].to_numpy()[np.argsort(-bm25.get_scores(ts))[:10]] == label).mean()
    rows.append({"クエリ": q, "拡張前 P@10": p(terms), "拡張後 P@10": p(terms + expand(terms)), "足した語": " ".join(expand(terms))})
print(pd.DataFrame(rows).to_string(index=False))
```

- 「最初の検索の上位はだいたい関連文書だろう」と仮定して、そこから特徴的な語を拾う。人の判断を使わないので「擬似」適合性フィードバックと呼ぶ。
- 上位が外れていると、無関係な語が足されて検索がさらにずれる (クエリドリフト)。足す語数と使う文書数は控えめにする。
- この例でも、拡張で適合率が上がったのは 6 クエリ中 1 つだけで、多くは人名や番組名のような偶然の語が足されて下がっている。クエリ拡張は「入れれば良くなる」ものではないので、必ず評価データで効果を確かめる。
- 元のクエリの語を、足した語より重くするのが普通 (ここでは同じ重み)。RM3 と呼ばれる方式が古典的な標準。
- 同義語辞書 (WordNet) による拡張や、LLM にクエリを言い換えさせる方法もある。埋め込み検索は、同じ効果を内部で実現していると言える。

## 埋め込みで意味検索 (セマンティックサーチ) をする {semantic_search}

キーワードが一致しなくても、質問と意味が近い文書を検索したい。

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("intfloat/multilingual-e5-small")

docs = [
    "富士山は標高 3776 メートルで、日本で最も高い山である。",
    "東京スカイツリーは高さ 634 メートルの電波塔である。",
    "琵琶湖は滋賀県にある日本最大の湖である。",
    "信濃川は全長 367 キロメートルで、日本で最も長い川である。",
    "形態素解析とは、文を単語に分割して品詞を付与する処理である。",
    "Transformer は自己注意機構に基づくニューラルネットワークである。",
    "カレーライスは明治時代に日本に伝わった料理である。",
]
doc_emb = model.encode(["passage: " + d for d in docs], normalize_embeddings=True, convert_to_tensor=True)

def search(query, k=3):
    q = model.encode("query: " + query, normalize_embeddings=True, convert_to_tensor=True)
    for hit in util.semantic_search(q, doc_emb, top_k=k)[0]:
        print(f"  {hit['score']:.3f}  {docs[hit['corpus_id']]}")

for q in ["日本一大きい湖はどこ？", "単語に区切る技術", "What is the tallest mountain in Japan?"]:
    print(q)
    search(q)
```

- 文書側の埋め込みは 1 回計算して保存しておき、検索時はクエリだけ埋め込む。
- 数十万件までなら総当たり (この方法) で十分速い。それ以上は近似最近傍探索 (次項の FAISS) を使う。

## FAISS で大量のベクトルから高速に近傍検索する {faiss_index}

数十万〜数億件の埋め込みベクトルから、クエリに近いものを高速に取り出したい。

```python
!pip install -q faiss-cpu
```

```python
import time
import faiss
import numpy as np

rng = np.random.default_rng(0)
d, n = 384, 200_000
# 実際の埋め込みに似せて、1000 個のクラスタ中心 + ノイズで人工データを作る
centers = rng.standard_normal((1000, d), dtype=np.float32)
xb = centers[rng.integers(0, 1000, n)] + 0.7 * rng.standard_normal((n, d), dtype=np.float32)
xq = centers[rng.integers(0, 1000, 100)] + 0.7 * rng.standard_normal((100, d), dtype=np.float32)
faiss.normalize_L2(xb); faiss.normalize_L2(xq)   # 正規化しておけば内積 = コサイン類似度

flat = faiss.IndexFlatIP(d)                      # 厳密な総当たり
flat.add(xb)
t = time.perf_counter(); D, I = flat.search(xq, 10); t_flat = time.perf_counter() - t
print(f"Flat: {t_flat * 1000:.0f} ms / 100 クエリ")
```

```python
# 近似探索: HNSW (グラフ) と IVF (クラスタリング)
hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
hnsw.add(xb)
hnsw.hnsw.efSearch = 64                           # 探索時に保持する候補数 (大きいほど正確で遅い)

ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, 256, faiss.METRIC_INNER_PRODUCT)
ivf.train(xb); ivf.add(xb)
ivf.nprobe = 16                                   # 探索するクラスタ数 (大きいほど正確で遅い)

for name, index in [("HNSW", hnsw), ("IVF", ivf)]:
    t = time.perf_counter(); _, I2 = index.search(xq, 10); dt = time.perf_counter() - t
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(I, I2)])
    print(f"{name}: {dt * 1000:.0f} ms / 100 クエリ, recall@10 = {recall:.2f}")
```

```python
faiss.write_index(hnsw, "index.faiss")           # 保存と読み込み
index = faiss.read_index("index.faiss")
print(index.ntotal)
```

- 近似探索は速度と recall のトレードオフ。`efSearch` や `nprobe` を上げると recall が上がり、遅くなる。自分のデータで必ず recall を測る。
- 100 万件以下で精度重視なら Flat か HNSW、メモリを節約したいなら IVF + PQ (直積量子化)。
- GPU 版は `faiss-gpu`。メタデータによる絞り込みや永続化まで欲しければ Qdrant や Chroma などのベクトル DB を検討する。

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

## キーワード検索と埋め込み検索を組み合わせる (ハイブリッド検索、RRF) {hybrid_search}

BM25 と埋め込み検索の結果を統合して、両方の長所を活かしたい。

```python
!pip install -q rank_bm25 fugashi unidic-lite
```

```python
import numpy as np
from fugashi import Tagger
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

docs = [
    "RTX 4090 は 24GB の VRAM を搭載した GPU である。",
    "グラフィックボードはゲームや機械学習の計算を高速化する部品である。",
    "RX 7900 XTX は AMD のハイエンド GPU である。",
    "CPU はコンピュータの中央演算処理装置である。",
    "メモリが不足すると、大きなモデルは読み込めない。",
    "画像処理用の半導体は深層学習の学習に広く使われている。",
]
tagger = Tagger()
tok = lambda s: [w.surface.lower() for w in tagger(s) if w.feature.pos1 not in ("助詞", "助動詞", "補助記号", "空白")]
bm25 = BM25Okapi([tok(d) for d in docs])
model = SentenceTransformer("intfloat/multilingual-e5-small")
doc_emb = model.encode(["passage: " + d for d in docs], normalize_embeddings=True)
```

```python
def rrf(rankings, k=60):
    """Reciprocal Rank Fusion: 各ランキングでの順位の逆数を足し合わせる"""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

def search(query, topn=3):
    sparse = list(np.argsort(-bm25.get_scores(tok(query))))
    dense = list(np.argsort(-(doc_emb @ model.encode("query: " + query, normalize_embeddings=True))))
    print(query)
    for name, ranking in [("BM25  ", sparse), ("Dense ", dense), ("Hybrid", rrf([sparse, dense]))]:
        print(f"  {name}", [int(i) for i in ranking[:topn]])
    print("  →", docs[rrf([sparse, dense])[0]])

search("RTX 4090 の VRAM 容量")
search("深層学習に使うグラボ")
```

- RRF はスコアの尺度が違うランキング同士を、順位だけを使って統合する。調整するパラメータがほぼなく (k=60 が定番)、頑健。
- スコアを正規化して重み付き和をとる方法もあるが、重みの調整が必要になる。

## クロスエンコーダで検索結果を並べ替える (リランキング) {rerank}

検索の上位候補を、より精度の高いモデルで並べ替えて、本当に関連する文書を一番上に持ってきたい。

```python
!pip install -q fugashi unidic-lite
```

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("hotchpotch/japanese-reranker-cross-encoder-xsmall-v1", max_length=512)

query = "日本で一番高い山は何ですか？"
candidates = [
    "日本の山の標高ランキングは登山者の間でよく話題になる。",
    "エベレストは世界で最も高い山で、標高は 8849 メートルである。",
    "富士山は標高 3776 メートルで、日本の最高峰である。",
    "北岳は標高 3193 メートルで、富士山に次いで日本で 2 番目に高い。",
    "日本で一番長い川は信濃川である。",
]
scores = reranker.predict([(query, c) for c in candidates])
for s, c in sorted(zip(scores, candidates), reverse=True):
    print(f"{s:.3f}  {c}")
```

```python
# rank() を使うと並べ替えまでやってくれる
for r in reranker.rank(query, candidates, top_k=2, return_documents=True):
    print(r["corpus_id"], round(float(r["score"]), 3), r["text"])
```

- バイエンコーダ (文埋め込み) はクエリと文書を別々にベクトル化するので速いが粗い。クロスエンコーダは両方を連結して入力するので正確だが遅い。
- 定番の構成: BM25 や埋め込み検索で上位 100 件を取得 → クロスエンコーダで上位 10 件に絞る。
- 多言語なら `BAAI/bge-reranker-v2-m3`、日本語で高精度なら `hotchpotch/japanese-reranker-cross-encoder-large-v1` など。

## 複数の特徴量からランキングを学習する (Learning to Rank、LambdaMART) {learning_to_rank}

BM25 のスコア、埋め込みの類似度、文書の長さなど、手元にある複数の手がかりをどう組み合わせて並べるかを、関連度のラベル付きデータから学習したい。クイズの質問と Wikipedia の段落に関連の有無のラベルを付けた JQaRA を使い、LightGBM の LambdaMART で学習して、BM25、埋め込み、RRF と比べる。

```python
!pip install -q rank_bm25 fugashi unidic-lite
```

```python
import numpy as np
import pandas as pd

# JQaRA: クイズの質問ごとに Wikipedia の段落の候補 (dev は 50 件、test は 100 件) が付いている。
# ラベルは、答えの文字列を含み、答えを導くのに有用と確かめた段落が 1、それ以外が 0
base = "hf://datasets/hotchpotch/JQaRA/data/"
dev = pd.read_parquet(base + "dev-00000-of-00001.parquet")
test = pd.read_parquet(base + "test-00000-of-00001.parquet")
for name, d in [("dev", dev), ("test", test)]:
    by_pos = d.groupby(d.groupby("q_id").cumcount())["label"].mean()   # 配布時の並び順の i 番目が正解である割合
    print(f"{name}: {d['q_id'].nunique()} 質問, 正解の割合 (並び順の 1〜8 番目) {by_pos.head(8).round(2).tolist()}")

# 配布時の並び順は、dev でも test でも上位ほど正解が多い。並び順が答えを漏らさないよう、候補の順番は混ぜる
# 時間の都合で、dev (学習用) と test (評価用) から 100 質問ずつ選ぶ
rng = np.random.default_rng(0)
def sample(d, n):
    qids = rng.choice(d["q_id"].unique(), n, replace=False)
    return d[d["q_id"].isin(qids)].sample(frac=1, random_state=0).sort_values("q_id", kind="stable")

df = pd.concat([sample(dev, 100).assign(split="train"), sample(test, 100).assign(split="test")], ignore_index=True)
assert df.groupby("q_id")["split"].nunique().max() == 1      # 同じ質問が学習と評価の両方に入っていない
print(len(df), "組 |", df["question"].iloc[0], df["answers"].iloc[0])
```

```python
from fugashi import Tagger
from rank_bm25 import BM25Okapi

tagger = Tagger()
def tokenize(text):
    return [w.feature.orthBase or w.surface for w in tagger(text.lower())
            if w.feature.pos1 not in ("助詞", "助動詞", "補助記号", "空白")]

titles = [tokenize(t) for t in df["title"]]
bodies = [tokenize(t) for t in df["text"]]
# IDF は手元の候補全体 (200 質問分) から計算する簡略版。本来は検索対象の全文書から計算する
bm25 = BM25Okapi([t + b for t, b in zip(titles, bodies)])   # タイトル + 本文
bm25_title = BM25Okapi(titles)                               # タイトルだけ

rows = {}
for qid, g in df.groupby("q_id", sort=False):
    q = tokenize(g["question"].iloc[0]); qs = set(q); idx = g.index.to_list()
    for i, s, st in zip(idx, bm25.get_batch_scores(q, idx), bm25_title.get_batch_scores(q, idx)):
        rows[i] = {"bm25": s, "bm25_title": st,
                   "coverage": len(qs & (set(titles[i]) | set(bodies[i]))) / len(qs),   # クエリの語のうち文書に出てくる割合
                   "title_coverage": len(qs & set(titles[i])) / len(qs)}
df = df.join(pd.DataFrame.from_dict(rows, orient="index"))
df["doc_len"] = df["text"].str.len()

# 第 1 段: BM25 で各質問の上位 20 件に絞る。以降はこの 20 件の並べ替えを比べる
df["bm25_rank"] = df.groupby("q_id")["bm25"].rank(ascending=False, method="first")
cand = df[df["bm25_rank"] <= 20].reset_index(drop=True)
recall = cand.groupby("q_id")["label"].sum() / df.groupby("q_id")["label"].sum()
print(len(cand), "候補 | 正解のうち上位 20 件に残った割合:", recall.groupby(cand.groupby("q_id")["split"].first()).mean().round(3).to_dict())
```

```python
from sentence_transformers import SentenceTransformer

# 4000 段落の埋め込み。Colab の CPU で 8〜12 分かかる
model = SentenceTransformer("intfloat/multilingual-e5-small")
P = model.encode(["passage: " + t + " " + x for t, x in zip(cand["title"], cand["text"])], batch_size=32, normalize_embeddings=True)
questions = cand["question"].unique().tolist()
Q = dict(zip(questions, model.encode(["query: " + q for q in questions], normalize_embeddings=True)))
cand["dense"] = (P * np.stack([Q[q] for q in cand["question"]])).sum(axis=1)   # コサイン類似度
cand["dense_rank"] = cand.groupby("q_id")["dense"].rank(ascending=False, method="first")
print(cand[["bm25", "bm25_title", "dense", "coverage", "title_coverage", "doc_len", "label"]].describe().round(2).loc[["mean", "min", "max"]])
```

```python
import lightgbm as lgb

features = ["bm25", "bm25_title", "dense", "coverage", "title_coverage", "doc_len"]
params = dict(objective="lambdarank", n_estimators=100, learning_rate=0.05, num_leaves=7, min_child_samples=20, random_state=0, verbose=-1)
train, ev = cand[cand["split"] == "train"], cand[cand["split"] == "test"].copy()

# group: クエリごとの候補数を、行の並び順どおりに並べたもの。同じクエリの行は連続していなければならない
group = train.groupby("q_id", sort=False).size().to_numpy()
assert (train["q_id"] != train["q_id"].shift()).sum() == len(group)
print("group:", group[:8], "... 合計", group.sum(), "=", len(train), "行")

ranker = lgb.LGBMRanker(**params)
ranker.fit(train[features], train["label"], group=group)
ev["ltr"] = ranker.predict(ev[features])
ev["rrf"] = 1 / (60 + ev["bm25_rank"]) + 1 / (60 + ev["dense_rank"])    # この章の「キーワード検索と埋め込み検索を組み合わせる (ハイブリッド検索、RRF)」と同じ RRF (k=60)
```

```python
def reciprocal_rank(ranked, relevant):
    for i, doc in enumerate(ranked, 1):
        if doc in relevant:
            return 1 / i
    return 0.0

def ndcg_at_k(ranked, gains, k):
    dcg = sum(gains.get(doc, 0) / np.log2(i + 1) for i, doc in enumerate(ranked[:k], 1))
    ideal = sum(g / np.log2(i + 1) for i, g in enumerate(sorted(gains.values(), reverse=True)[:k], 1))
    return float(dcg / ideal) if ideal else 0.0

# 正解は第 1 段で落ちたものも含めて数える (理想の nDCG は元の候補全体の正解から計算)
gold = {qid: {d: 1 for d in s} for qid, s in df[df["label"] == 1].groupby("q_id")["id"]}

def evaluate(data, col):
    nd, rr = [], []
    for qid, g in data.groupby("q_id"):
        ranked = g.sort_values(col, ascending=False)["id"].tolist()
        nd.append(ndcg_at_k(ranked, gold[qid], 10)); rr.append(reciprocal_rank(ranked[:10], set(gold[qid])))
    return np.array(nd), np.array(rr)

methods = [("BM25", "bm25"), ("埋め込み", "dense"), ("RRF", "rrf"), ("LTR (LambdaMART)", "ltr"), ("上限 (20 件中の正解を全部上に)", "label")]
results = {name: evaluate(ev, col) for name, col in methods}
print(pd.DataFrame({k: {"nDCG@10": v[0].mean(), "MRR@10": v[1].mean()} for k, v in results.items()}).T.round(4))

def paired_ci(diff, n_boot=2000, seed=0):
    """質問ごとの差の平均の 95% 信頼区間 (質問を復元抽出するブートストラップ)"""
    r = np.random.default_rng(seed)
    means = [diff[r.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    return diff.mean(), *np.percentile(means, [2.5, 97.5])

for a, b in [("LTR (LambdaMART)", "RRF"), ("LTR (LambdaMART)", "BM25"), ("RRF", "埋め込み")]:
    diff = results[a][0] - results[b][0]
    m_, lo, hi = paired_ci(diff)
    print(f"nDCG@10 の差 {a} − {b}: {m_:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"  (良い {(diff > 0).sum()} / 悪い {(diff < 0).sum()} / 同じ {(diff == 0).sum()} 質問)")

imp = pd.Series(ranker.booster_.feature_importance("gain"), index=features)
print((imp / imp.sum()).sort_values(ascending=False).round(3))
```

```python
# 比較: 評価と同じ分布のデータ (test の別の質問) で学習したらどうなるか。test の 100 質問を、質問単位で 5 分割して交差検証する
from sklearn.model_selection import GroupKFold

ev["ltr_cv"] = 0.0
imps = []
for tr_idx, te_idx in GroupKFold(n_splits=5).split(ev, groups=ev["q_id"]):
    a = ev.iloc[tr_idx]
    m = lgb.LGBMRanker(**params).fit(a[features], a["label"], group=a.groupby("q_id", sort=False).size().to_numpy())
    ev.iloc[te_idx, ev.columns.get_loc("ltr_cv")] = m.predict(ev.iloc[te_idx][features])
    imps.append(m.booster_.feature_importance("gain"))
nd, rr = evaluate(ev, "ltr_cv")
m_, lo, hi = paired_ci(nd - results["RRF"][0])
print(f"LTR (test の別の質問で学習): nDCG@10 {nd.mean():.4f}  MRR@10 {rr.mean():.4f}  | RRF との差 {m_:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
imp = pd.Series(np.mean(imps, axis=0), index=features)
print((imp / imp.sum()).sort_values(ascending=False).round(3))
```

- Learning to Rank (LTR) は、クエリと文書の組ごとの特徴量から並べる順番を学習する。文書ごとの関連度を当てる pointwise、同じクエリの 2 文書のどちらが上かを当てる pairwise、並び全体の良さを目的にする listwise に分けられる。LambdaMART は、ペアの順序の誤りに「入れ替えたら nDCG がどれだけ変わるか」で重みを付けた勾配を勾配ブースティング木で学習する方式で、LightGBM の objective="lambdarank" がこれにあたる。クロスエンコーダのスコア (この章の「クロスエンコーダで検索結果を並べ替える (リランキング)」) も特徴量の 1 つにできる。
- `group` は「最初の 20 行が 1 つ目のクエリ、次の 20 行が 2 つ目…」という件数の並びで、合計が行数と一致しなければならない。同じクエリの行は連続させる。ここでは全クエリが 20 件だが、件数はクエリごとに違ってよい。予測は行ごとのスコアなので `group` は要らず、クエリの中でスコア順に並べればよい。
- 学習と評価はクエリ単位で分ける。同じクエリの候補が両方に入ると、そのクエリに特有の傾向を覚えたまま評価することになり、数値が甘くなる。ここでは dev の質問で学習して test の質問で評価し (assert で確認)、最後のセルは `GroupKFold` で質問単位に分割している。また配布データの候補は上位ほど正解が多い (最初のセルの出力で、dev は 1〜5 番目が 1.0〜0.29 で 6 番目以降は 0、test も 1〜8 番目で 0.61 から 0.22 に下がる) ので、順番が答えを漏らさないよう混ぜてから使う。データの質問と答えは JAQKET 由来で CC BY-SA 4.0、段落は Wikipedia 由来 (CC BY-SA 4.0 または GFDL)。
- 100 質問の評価では、どの方法の差も誤差の範囲にある。dev で学習した LTR は nDCG@10 0.478 で RRF の 0.484 より低いが、質問ごとの差のブートストラップ 95% 信頼区間は [−0.023, +0.011] で 0 をまたぎ、勝ち負けも 37 対 43 である。BM25 との差 (+0.013、[−0.016, +0.043]) も同様。RRF と埋め込みの差 (+0.023、[+0.003, +0.043]) だけは 0 をまたがなかったが、いくつも比べたうちの 1 つなので強くは言えない。信頼区間の考え方は評価指標と統計的検定の章の「ブートストラップ法で評価値の信頼区間を求める」を参照。LTR を入れるなら、もっと多くの質問で調整なしの RRF より良いことを確かめてからにする。
- dev と test では性質が違う。dev で学習したモデルの重要度は dense に偏り (0.889)、test の別の質問で学習すると bm25 や coverage にも分散する (dense 0.402)。BM25 の上位 20 件に残る正解の割合も dev 93.4%、test 51.5% と違う。README によれば、候補数が dev 50 件・test 100 件で、dev は正解を LLM で確かめた最大 5 件に絞っており、test のほうが正解が多い。また候補自体が e5-large など 5 つの埋め込みモデルの検索結果を RRF で絞ったもので、ここでの BM25 の第 1 段はその中での絞り込みにすぎない。dense の重要度が高いのは、候補を作った方法と同系統だからという可能性もある。ただし、これらが性能差の原因だとは、この実験からは言えない。なお最後のセルの LTR (nDCG@10 0.486、RRF との差 +0.002、[−0.020, +0.023]) は、評価と同じ分布のラベルで学習しているのに対し、BM25・埋め込み・RRF はラベルを使っていない。しかも特徴量や設定は、試行の段階で test の結果を見ながら決めたので、楽観的な参考値である。
- 第 1 段で落ちた正解は並べ替えでは取り戻せない。20 件の候補の中の正解を全部上に並べたときの上限は nDCG@10 0.700・MRR@10 0.920 で、どの方法 (0.46〜0.49) も方法同士の差よりはるかに大きく上限を下回る。test の正解のうち上位 20 件に残ったのは 51.5% なので、第 1 段をハイブリッド検索にする、候補を増やすなどで上限を上げる余地が大きい。
- 木のモデルは値の大小で分岐するので、特徴量の標準化は要らない。重要度 (gain) は分岐で損失を減らした量の合計で、似た特徴量の間で分け合われ、学習データで大きく変わる (上の 2 つのモデルのとおり)。ラベルは 0 以上の整数で、0〜3 のような段階もそのまま渡せる。LightGBM の lambdarank は既定で関連度 l の重みを 2^l − 1 とし、31 以上のラベルはエラーになる (`label_gain` で変えられる)。この章の「検索・ランキングを評価する (Recall@k、MRR、nDCG)」の `ndcg_at_k` は関連度をそのまま足すので、段階のあるラベルでは LightGBM の nDCG と値が一致しない (0/1 なら同じ)。
- クリックログを正解の代わりにすると、上位に表示された文書ほどクリックされる位置バイアスまで学習してしまう。クリックされなかった文書は、見られていないだけかもしれない。表示位置ごとのクリックされやすさを推定して補正する方法 (unbiased learning to rank) があり、LightGBM の `Dataset` には表示位置を渡す `position` 引数がある (4.6.0 で確認)。

## 検索結果の重複を減らして多様にする (MMR) {mmr_diversify}

検索結果の上位が似た文書ばかりになるのを避けたい。関連度を保ちつつ、内容の違う文書を選びたい。

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-small")
docs = [
    "京都の紅葉の名所といえば東福寺。通天橋からの眺めが有名だ。",
    "東福寺の通天橋は紅葉の名所として知られ、秋には多くの人が訪れる。",
    "紅葉の季節、東福寺の通天橋は大変混雑する。",
    "嵐山の渡月橋周辺も、秋には山が色づいて美しい。",
    "京都の秋の味覚といえば松茸や栗を使った料理である。",
    "紅葉のライトアップは永観堂が人気で、夜間拝観が行われる。",
    "京都駅から東福寺までは JR 奈良線で一駅である。",
    "北海道の紅葉は 9 月下旬から始まり、大雪山が有名だ。",
]
doc_emb = model.encode(["passage: " + d for d in docs], normalize_embeddings=True)
query = "京都の紅葉のおすすめ"
q = model.encode("query: " + query, normalize_embeddings=True)
rel = doc_emb @ q

def mmr(rel, doc_emb, k=4, lam=0.7):
    """lam × クエリとの関連度 − (1 − lam) × 選択済みの文書との最大類似度 が最大のものを 1 つずつ選ぶ"""
    selected = [int(rel.argmax())]
    while len(selected) < k:
        rest = [i for i in range(len(rel)) if i not in selected]
        scores = [lam * rel[i] - (1 - lam) * (doc_emb[i] @ doc_emb[selected].T).max() for i in rest]
        selected.append(rest[int(np.argmax(scores))])
    return selected

print("関連度順:")
for i in np.argsort(-rel)[:4]:
    print(f"   {rel[i]:.3f} {docs[i]}")
for lam in [0.7, 0.5]:
    print(f"MMR (lambda={lam}):")
    for i in mmr(rel, doc_emb, lam=lam):
        print(f"   {rel[i]:.3f} {docs[i]}")
```

```python
# 選ばれた文書同士の類似度の平均 (低いほど多様)
def redundancy(ids):
    sims = doc_emb[ids] @ doc_emb[ids].T
    return float((sims.sum() - len(ids)) / (len(ids) * (len(ids) - 1)))

top = list(np.argsort(-rel)[:4])
print("関連度順:", round(redundancy(top), 3), " MMR:", round(redundancy(mmr(rel, doc_emb, lam=0.5)), 3))
```

- MMR (Maximal Marginal Relevance) は、「クエリに近い」と「すでに選んだものと違う」の両方を満たす文書を順に選ぶ。λ が 1 なら関連度順、小さいほど多様性を重視する。
- RAG では、同じ内容のチャンクを何個も渡しても情報が増えない。MMR で多様にすると、限られた件数で広い範囲の情報を渡せる。
- 多様にしすぎると関連の薄い文書が入る。λ は 0.5〜0.8 あたりから試す。
- 完全に同じ内容の文書は、検索より前に重複除去しておく (文字列処理の章の MinHash を参照)。

## 長い文書を分割して検索し、文書単位にまとめる {long_document_retrieval}

埋め込みモデルには入力長の上限があり、長い文書は先頭しか見てもらえない。文書の後半に書かれた内容でも検索に当たるようにしたい。

```python
import io, re, tarfile, urllib.request
import numpy as np
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

df = load_livedoor()
df = df[df["text"].str.len().between(1500, 4000)].sample(200, random_state=0).reset_index(drop=True)
split = lambda t: [s.strip() for s in re.split(r"(?<=[。！？])|\n+", t) if s and len(s.strip()) > 15]
sents = [split(t) for t in df["text"]]

# クエリ: 各文書の「後ろ 3 分の 1」にある 1 文 (正解はその文書)。文書の後半の内容を探す状況を模している
rng = np.random.default_rng(0)
queries = [s[int(rng.integers(len(s) * 2 // 3, len(s)))] for s in sents]
print(len(df), "文書 / 平均", int(df["text"].str.len().mean()), "文字 |", queries[0][:50])
```

```python
model = SentenceTransformer("intfloat/multilingual-e5-small")
model.max_seq_length = 256
enc = lambda texts, prefix: model.encode([prefix + t for t in texts], normalize_embeddings=True, batch_size=32, show_progress_bar=False)
q_emb = enc(queries, "query: ")

# 方法 1: 文書をそのまま埋め込む (上限を超えた分は切り捨てられ、先頭しか反映されない)
whole = enc(df["text"].tolist(), "passage: ")
rank_whole = np.argsort(-(q_emb @ whole.T), axis=1)

# 方法 2: 約 300 文字のチャンクに分けて埋め込み、文書のスコア = その文書のチャンクの最大スコア
chunks, owner = [], []
for doc_id, ss in enumerate(sents):
    cur = ""
    for s in ss + [None]:
        if s is None or len(cur) + len(s) > 300:
            if cur:
                chunks.append(cur); owner.append(doc_id)
            cur = ""
        if s:
            cur += s
owner = np.array(owner)
chunk_emb = enc(chunks, "passage: ")
sims = q_emb @ chunk_emb.T
doc_scores = np.stack([sims[:, owner == d].max(axis=1) for d in range(len(df))], axis=1)
rank_chunk = np.argsort(-doc_scores, axis=1)

def report(name, ranking):
    rank = np.array([np.where(ranking[i] == i)[0][0] + 1 for i in range(len(ranking))])
    print(f"{name:28} Recall@1 = {(rank <= 1).mean():.3f}  Recall@5 = {(rank <= 5).mean():.3f}  MRR = {(1 / rank).mean():.3f}")

print(len(chunks), "チャンク")
report("文書全体を 1 つの埋め込みに", rank_whole)
report("チャンクに分けて最大値", rank_chunk)
```

- 入力長の上限を超えた部分は黙って切り捨てられる。エラーにならないので、気づかないまま「後半が検索に当たらない」状態になりやすい。
- チャンクに分けて、文書のスコアをチャンクの最大値 (または上位数個の和) とするのが基本の対処。どのチャンクが当たったかも分かるので、該当箇所を示せる。
- この実験のクエリは文書中の文そのものなので、数値は実際より高く出る。差の向きを見るための簡易な確認である。
- 上限の長い埋め込みモデル (8192 トークンなど) もあるが、長い文書を 1 つのベクトルに押し込むと細部が薄まる。長さの上限が大きくても、分割は有効なことが多い。

## 言語をまたいで検索する (クロスリンガル検索) {cross_lingual_search}

日本語のクエリで英語の文書を、あるいは英語のクエリで日本語の文書を検索したい。翻訳せずに、多言語の埋め込みで直接探したい。

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-small")
docs = [
    ("en", "The Transformer architecture relies entirely on self-attention and dispenses with recurrence."),
    ("en", "Mount Fuji is the highest mountain in Japan, standing 3,776 meters tall."),
    ("en", "Sourdough bread is made by fermenting dough with wild yeast and lactobacilli."),
    ("ja", "形態素解析とは、文を単語に分割して品詞を付与する処理である。"),
    ("ja", "琵琶湖は滋賀県にある日本最大の湖である。"),
    ("ja", "天ぷらは、魚介や野菜に衣を付けて油で揚げた料理である。"),
    ("zh", "长城是中国古代的军事防御工程。"),
    ("fr", "La tour Eiffel est le monument le plus visité de Paris."),
]
doc_emb = model.encode(["passage: " + d for _, d in docs], normalize_embeddings=True)

def search(query, k=2):
    sims = doc_emb @ model.encode("query: " + query, normalize_embeddings=True)
    return [(round(float(sims[i]), 3), docs[i][0], docs[i][1][:44]) for i in np.argsort(-sims)[:k]]

for q in ["日本で一番高い山", "自己注意機構に基づくニューラルネットワーク", "パンの作り方", "How do you split Japanese text into words?",
          "largest lake in Japan", "パリの観光名所", "万里の長城"]:
    print(q)
    for hit in search(q):
        print("   ", hit)
```

```python
# 同じ言語の文書が上に来やすい偏りを確かめる: 同じ内容の日本語と英語の文書がある場合
pairs = ["富士山は日本で最も高い山で、標高は 3776 メートルである。", "Mount Fuji is the highest mountain in Japan, standing 3,776 meters tall."]
emb = model.encode(["passage: " + p for p in pairs], normalize_embeddings=True)
for q in ["日本で一番高い山は？", "What is the highest mountain in Japan?"]:
    sims = emb @ model.encode("query: " + q, normalize_embeddings=True)
    print(q, "→ 日本語の文書", round(float(sims[0]), 3), "/ 英語の文書", round(float(sims[1]), 3))
```

- 多言語の埋め込みモデルは、言語が違っても意味が近い文を近くに置くように学習されている。クエリも文書も翻訳せずに、そのまま検索できる。
- 同じ内容なら、クエリと同じ言語の文書のほうが少し高いスコアになりやすい (言語バイアス)。言語を混ぜた検索結果では、他言語の文書が下位に沈むことがある。
- 小さなモデルでは取り違えも起きる。上の例では、「日本で一番高い山」に琵琶湖 (日本最大の湖) が、「largest lake in Japan」に富士山が上位に来ている。「日本」「一番」のような共通部分に引かれるためで、より大きなモデル (multilingual-e5-large、BGE-M3 など) やリランキングで改善する。
- 固有名詞や専門用語が言語によって大きく違う場合は弱い。クエリを機械翻訳してから検索する方法と比べて、良いほうを使う。
- BM25 のようなキーワード検索は、言語が違うと語が一致しないので使えない。ハイブリッド検索にするなら、クエリの翻訳と組み合わせる。

## 検索用の埋め込みモデルを自分のデータで微調整する {embedding_finetune}

汎用の埋め込みモデルでは、自分の分野の検索で精度が足りない。クエリと正解文書の組から、埋め込みモデルを追加学習したい。

```python
!pip install -q datasets accelerate
```

```python
import io, tarfile, urllib.request
import numpy as np
import pandas as pd

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

# (タイトル, 本文) を (クエリ, 正解文書) の組とみなす。本文の冒頭はタイトルと重なりやすいので、途中の部分を使って難しくする
df = load_livedoor().sample(2100, random_state=0).reset_index(drop=True)
df["passage"] = df["text"].map(lambda t: " ".join(t.split())[200:500])
df = df[df["passage"].str.len() > 100].reset_index(drop=True)
train, test = df.iloc[:1200], df.iloc[1200:1800].reset_index(drop=True)
print(len(train), "組で学習 /", len(test), "組で評価")
```

```python
from sentence_transformers import SentenceTransformer

def evaluate(model):
    q = model.encode(["query: " + t for t in test["title"]], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    d = model.encode(["passage: " + t for t in test["passage"]], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    ranking = np.argsort(-(q @ d.T), axis=1)
    rank = np.array([np.where(ranking[i] == i)[0][0] + 1 for i in range(len(test))])
    return {"Recall@1": round(float((rank <= 1).mean()), 3), "Recall@10": round(float((rank <= 10).mean()), 3), "MRR": round(float((1 / rank).mean()), 3)}

model = SentenceTransformer("intfloat/multilingual-e5-small")
model.max_seq_length = 128
print("微調整前:", evaluate(model))
```

```python
from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss

train_ds = Dataset.from_dict({"anchor": ["query: " + t for t in train["title"]], "positive": ["passage: " + t for t in train["passage"]]})
loss = MultipleNegativesRankingLoss(model)   # バッチ内の他の文書を「不正解」として使う対照学習
args = SentenceTransformerTrainingArguments(output_dir="e5-finetuned", num_train_epochs=1, per_device_train_batch_size=32, learning_rate=2e-5,
                                            warmup_steps=5, logging_steps=10, save_strategy="no", report_to="none", seed=0)
SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds, loss=loss).train()
print("微調整後:", evaluate(model))
model.save("e5-finetuned")
```

- 学習データは (クエリ, 正解文書) の組だけでよい。バッチ内の他の組の文書が自動的に不正解の例になるので、バッチサイズが大きいほど学習が効く。
- 1200 組・1 エポック (CPU で数分) では、改善はごくわずか (Recall@1 で 0.715 → 0.723)。はっきりした効果を得るには、数万組以上のデータと GPU での学習が要る。学習データには、実際のクエリのログや、LLM に文書から作らせた質問 (質問応答の章の質問生成) が学習データになる。
- 「正解ではないが紛らわしい文書」(ハードネガティブ) を BM25 や元のモデルの検索結果から集めて加えると、さらに伸びる。
- 評価データは必ず学習に使っていない組にする。微調整後は、分野外のクエリでの性能が下がっていないかも確かめる。

## 入力の途中で候補を出す・打ち間違いを直す (サジェストと「もしかして」) {search_suggest}

検索窓に数文字入れた時点で候補を出したい。打ち間違いや、ひらがなで入力された語にも対応したい。

```python
!pip install -q rapidfuzz fugashi unidic-lite jaconv
```

```python
import bisect
import jaconv
from fugashi import Tagger
from rapidfuzz import fuzz, process

# 候補の語と人気度 (検索回数など)
terms = {"自然言語処理": 950, "自然言語理解": 120, "自然言語生成": 200, "機械翻訳": 800, "機械学習": 990, "形態素解析": 400, "係り受け解析": 150,
         "固有表現抽出": 300, "感情分析": 350, "質問応答": 280, "文法誤り訂正": 180, "大規模言語モデル": 700, "検索拡張生成": 260, "言語モデル": 500}

tagger = Tagger()
def reading(word):
    return jaconv.kata2hira("".join(w.feature.kana or w.surface for w in tagger(word)))

# 表記と読みの両方を、前方一致で引けるように並べておく
entries = sorted([(t, t) for t in terms] + [(reading(t), t) for t in terms])
keys = [k for k, _ in entries]

def suggest(prefix, k=5):
    """前方一致する候補を、人気度の高い順に返す (二分探索なので、候補が数百万あっても速い)"""
    prefix = jaconv.kata2hira(prefix)
    lo = bisect.bisect_left(keys, prefix)
    hi = bisect.bisect_right(keys, prefix + "￿")
    found = {t for _, t in entries[lo:hi]}
    return sorted(found, key=lambda t: -terms[t])[:k]

for prefix in ["自然", "しぜんげんごし", "機械", "きかい", "けい", "ぶんぽう"]:
    print(prefix, "→", suggest(prefix))
```

```python
# 前方一致で見つからないときは、編集距離の近い語を「もしかして」として出す
def did_you_mean(query, limit=2, threshold=60):
    by_surface = process.extract(query, list(terms), scorer=fuzz.ratio, limit=limit)
    by_reading = process.extract(reading(query), {t: reading(t) for t in terms}, scorer=fuzz.ratio, limit=limit)
    best = {}
    for name, score in [(m[0], m[1]) for m in by_surface] + [(m[2], m[1]) for m in by_reading]:
        best[name] = max(best.get(name, 0), score)
    return [(t, round(s)) for t, s in sorted(best.items(), key=lambda x: -x[1]) if s >= threshold][:limit]

for q in ["自然原語処理", "機会翻訳", "けいたいそかいせき", "固有表現注出", "大規模言語モデル", "らーめん"]:
    hits = suggest(q, 1)
    print(q, "→", hits if hits else f"もしかして: {did_you_mean(q)}")
```

- サジェストは「前方一致 + 人気順」が基本。候補をソートしておけば二分探索で引ける。大規模にはトライ木や、検索エンジンの補完機能を使う。
- 日本語では、変換前のひらがなやローマ字で入力されることが多い。読みでも引けるようにしておくと、「しぜんげんごし」の時点で候補を出せる。
- 「機会翻訳」のような同音の誤変換は、表記の編集距離では遠くても、読みに直すと完全に一致する。表記と読みの両方で照合するのが有効。
- しきい値未満なら何も出さない。無関係な語を「もしかして」と出すと、かえって邪魔になる。
