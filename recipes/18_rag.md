# RAG (検索拡張生成)

手元の文書を検索して LLM に答えさせる。分割・検索・生成・出典・評価まで、CPU で動く小型モデルで一通り組み立てる。検索の部品 (埋め込み、BM25、ハイブリッド、リランキング) は「文埋め込み・類似度・検索」の章を参照。

## 検索した文書を LLM に渡して答えさせる (最小の RAG) {rag_minimal}

LLM が知らない情報 (社内文書、最新情報) について、関連文書を検索してから答えさせたい。GPU なしで動かしたい。

```python
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

docs = [
    "クックブック食堂の営業時間は平日 11 時から 15 時、土日は 11 時から 20 時です。月曜日は定休日です。",
    "クックブック食堂の看板メニューは、3 日間煮込んだ特製ビーフカレー (950 円) です。大盛りは 100 円増しです。",
    "クックブック食堂は 2019 年に京都市左京区で開業しました。店主の田中さんはインドで 2 年間修行しました。",
    "駐車場は店舗の裏に 3 台分あります。満車の場合は近隣のコインパーキングをご利用ください。",
    "予約は電話 (075-000-0000) のみ受け付けています。5 名以上の場合は前日までにご連絡ください。",
    "支払いは現金と QR コード決済に対応しています。クレジットカードは使えません。",
]
embedder = SentenceTransformer("intfloat/multilingual-e5-small")
doc_emb = embedder.encode(["passage: " + d for d in docs], normalize_embeddings=True)

name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"   # 3B の日本語モデル。CPU でも 1 回答あたり数秒〜十数秒 (メモリ約 7GB)
tokenizer = AutoTokenizer.from_pretrained(name)
llm = AutoModelForCausalLM.from_pretrained(name).eval()

def chat(messages, max_new_tokens=100):
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
```

```python
def retrieve(question, k=2):
    sims = doc_emb @ embedder.encode("query: " + question, normalize_embeddings=True)
    return [(int(i), docs[i]) for i in sims.argsort()[::-1][:k]]

def rag(question):
    hits = retrieve(question)
    context = "\n".join(f"[{i}] {d}" for i, d in hits)
    prompt = (f"以下の資料だけに基づいて、質問に一文で答えてください。資料に答えがなければ「わかりません」と答えてください。\n\n"
              f"# 資料\n{context}\n\n# 質問\n{question}")
    return chat([{"role": "user", "content": prompt}]), [i for i, _ in hits]

for q in ["カレーはいくらですか？", "日曜日は何時まで開いていますか？", "クレジットカードで払えますか？"]:
    answer, sources = rag(q)
    print(f"Q: {q}\n  検索なし: {chat([{'role': 'user', 'content': 'クックブック食堂について質問です。' + q}])[:70]}\n  RAG: {answer} (参照: {sources})")
```

- RAG = 検索 (Retrieval) + 生成 (Generation)。検索なしだと、モデルは知らない店についてもっともらしい嘘 (ハルシネーション) を答えるか、答えられない。
- 精度の大半は検索で決まる。まず「正解の文書が上位 k 件に入っているか」を評価し、ハイブリッド検索やリランキング (類似度と検索の章) で改善する。
- もっと軽くしたいなら `sarashina2.2-1b-instruct-v0.1` や `0.5b` に替えられるが、資料を読み違えることが増える。GPU があるなら 7B 以上のモデルにすると安定する。

## 長い文書をチャンクに分割する {chunking}

RAG のために長い文書を検索しやすい大きさの断片に区切りたい。文の途中で切らず、前後を少し重ねたい。

```python
import io, re, urllib.request, zipfile

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

text = aozora("https://www.aozora.gr.jp/cards/000035/files/1567_ruby_4948.zip")   # 走れメロス
print(len(text), "文字")

def chunk_by_sentences(text, max_chars=400, overlap_sents=1):
    """文の境界を保ったまま、max_chars を超えないように詰める。隣のチャンクと overlap_sents 文だけ重ねる"""
    sents = [s.strip() for s in re.split(r"(?<=[。！？」])(?![」。！？])|\n+", text) if s and s.strip()]
    chunks, cur = [], []
    for s in sents:
        if cur and sum(map(len, cur)) + len(s) > max_chars:
            chunks.append("".join(cur))
            cur = cur[-overlap_sents:] if overlap_sents else []
        cur.append(s)
    if cur:
        chunks.append("".join(cur))
    return chunks

chunks = chunk_by_sentences(text)
print(len(chunks), "チャンク / 文字数の最小・平均・最大:", min(map(len, chunks)), sum(map(len, chunks)) // len(chunks), max(map(len, chunks)))
for c in chunks[:2]:
    print("---", c[:150], "...")
```

```python
# トークン数を基準にする場合 (埋め込みモデルの最大長に合わせる)
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")

def chunk_by_tokens(text, max_tokens=256, overlap=32):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return [tok.decode(ids[i:i + max_tokens]) for i in range(0, len(ids), max_tokens - overlap)]

tchunks = chunk_by_tokens(text)
print(len(tchunks), "チャンク", [len(tok(c)["input_ids"]) for c in tchunks[:5]])
```

- チャンクが小さすぎると文脈が失われ、大きすぎると検索の精度が落ちて LLM に渡すトークンも増える。200〜500 トークン程度から試す。
- 構造のある文書 (Markdown、論文、マニュアル) は、見出しや段落を単位にするのが最も良い。各チャンクに文書タイトルや見出しを付け足すと検索精度が上がる。
- トークン数だけで機械的に切ると文の途中で切れる。両方を組み合わせる (文で区切り、トークン数で上限を設ける) のが実用的。

## 小さなチャンクで検索して、周りの文脈ごと LLM に渡す {rag_parent_child}

チャンクを小さくすると検索は当たりやすいが、LLM に渡す文脈が足りなくなる。検索の単位と、読ませる単位を分けたい。

```python
import io, re, urllib.request, zipfile
from sentence_transformers import SentenceTransformer

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

text = aozora("https://www.aozora.gr.jp/cards/000035/files/1567_ruby_4948.zip")   # 走れメロス
sents = [s.strip() for s in re.split(r"(?<=[。！？」])(?![」。！？])|\n+", text) if s and s.strip()]

# 子: 1 文ずつ (検索の単位) / 親: 前後 3 文を含む窓 (LLM に渡す単位)
def parent(i, window=3):
    return "".join(sents[max(0, i - window):i + window + 1])

model = SentenceTransformer("intfloat/multilingual-e5-small")
emb = model.encode(["passage: " + s for s in sents], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
print(len(sents), "文")
```

```python
def search(question, k=3):
    sims = emb @ model.encode("query: " + question, normalize_embeddings=True)
    hits, used = [], set()
    for i in sims.argsort()[::-1]:
        if any(abs(int(i) - j) <= 3 for j in used):     # 既に採用した窓と重なる文は飛ばす
            continue
        used.add(int(i)); hits.append((int(i), float(sims[i])))
        if len(hits) == k:
            break
    return hits

for q in ["メロスの妹は誰と結婚するのか", "王はなぜ人を殺すのか"]:
    print(q)
    for i, score in search(q, k=2):
        print(f"  {score:.3f} 当たった文: {sents[i][:50]}")
        print(f"        渡す文脈: {parent(i)[:120]}...")
```

- 「small-to-big」「parent document retrieval」と呼ばれる方法。1 文〜数文の小さな単位で索引を作り、ヒットした位置の周辺 (段落や節の全体) を LLM に渡す。
- 親の単位は、前後の窓でも、元の段落・節・ページでもよい。構造のある文書なら、見出し単位の節を親にするのが自然。
- 近い位置の文が複数ヒットすると同じ文脈を重複して渡すことになるので、重なりを除く。

## ベクトル DB に保存して、メタデータで絞り込んで検索する (Qdrant) {rag_vector_db}

埋め込みを毎回計算し直さずに保存しておきたい。「この部署の文書だけ」「今年の文書だけ」のように条件で絞り込んで検索したい。

```python
!pip install -q qdrant-client
```

```python
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-small")
docs = [
    {"text": "有給休暇は入社 6 か月後に 10 日付与されます。", "dept": "人事", "year": 2025},
    {"text": "有給休暇は入社時に 5 日、6 か月後にさらに 10 日付与されます。", "dept": "人事", "year": 2026},
    {"text": "経費精算は月末までに申請システムから行ってください。", "dept": "経理", "year": 2026},
    {"text": "出張旅費の上限は 1 泊あたり 12,000 円です。", "dept": "経理", "year": 2025},
    {"text": "出張旅費の上限は 1 泊あたり 15,000 円に改定されました。", "dept": "経理", "year": 2026},
    {"text": "社内 Wi-Fi のパスワードは四半期ごとに変更されます。", "dept": "情シス", "year": 2026},
    {"text": "ノート PC の貸与期間は原則 4 年です。", "dept": "情シス", "year": 2025},
]

client = QdrantClient(path="qdrant_db")      # ローカルのフォルダに保存 (サーバー不要)。":memory:" ならメモリ上のみ
if not client.collection_exists("manual"):
    client.create_collection("manual", vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
vectors = model.encode(["passage: " + d["text"] for d in docs], normalize_embeddings=True)
client.upsert("manual", points=[models.PointStruct(id=i, vector=v.tolist(), payload=d) for i, (v, d) in enumerate(zip(vectors, docs))])
print(client.count("manual"))
```

```python
def search(query, limit=3, **conditions):
    must = [models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k, v in conditions.items()]
    result = client.query_points("manual", query=model.encode("query: " + query, normalize_embeddings=True).tolist(),
                                 limit=limit, query_filter=models.Filter(must=must) if must else None)
    return [(round(p.score, 3), p.payload["dept"], p.payload["year"], p.payload["text"]) for p in result.points]

print("絞り込みなし:")
for r in search("出張のホテル代はいくらまで？"):
    print("  ", r)
print("2026 年の文書だけ:")
for r in search("出張のホテル代はいくらまで？", year=2026):
    print("  ", r)
print("人事の文書だけ:")
for r in search("休みは何日もらえる？", dept="人事", limit=2):
    print("  ", r)
```

```python
# 開き直しても残っている (永続化)。文書の更新は同じ ID で upsert、削除は delete
client.close()
client = QdrantClient(path="qdrant_db")
print(client.count("manual").count, "件")
client.delete("manual", points_selector=models.PointIdsList(points=[3]))   # 古い旅費規程を削除
print([r[3] for r in search("出張のホテル代はいくらまで？", limit=2)])
```

- 古い版と新しい版の文書が両方ヒットすると、LLM は古い情報で答えてしまうことがある。日付や版でのフィルタ、古い文書の削除が実用上とても重要。
- 数万件までなら NumPy や FAISS で十分。ベクトル DB の利点は、永続化、メタデータでの絞り込み、文書の追加・更新・削除が簡単なこと。
- Qdrant のほかに Chroma、LanceDB、pgvector (PostgreSQL) などがある。ローカルのファイルで動くものから始めると手軽。

## 回答に出典を付けさせ、その出典を検証する {rag_citation}

RAG の回答がどの資料に基づいているかを示したい。LLM が付けた出典番号が本当に根拠になっているかも確かめたい。

```python
import re
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

docs = [
    "クックブック食堂の営業時間は平日 11 時から 15 時、土日は 11 時から 20 時です。月曜日は定休日です。",
    "クックブック食堂の看板メニューは、3 日間煮込んだ特製ビーフカレー (950 円) です。大盛りは 100 円増しです。",
    "クックブック食堂は 2019 年に京都市左京区で開業しました。店主の田中さんはインドで 2 年間修行しました。",
    "駐車場は店舗の裏に 3 台分あります。満車の場合は近隣のコインパーキングをご利用ください。",
    "予約は電話 (075-000-0000) のみ受け付けています。5 名以上の場合は前日までにご連絡ください。",
    "支払いは現金と QR コード決済に対応しています。クレジットカードは使えません。",
]
embedder = SentenceTransformer("intfloat/multilingual-e5-small")
doc_emb = embedder.encode(["passage: " + d for d in docs], normalize_embeddings=True)

name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
llm = AutoModelForCausalLM.from_pretrained(name).eval()

def chat(messages, max_new_tokens=120):
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def answer_with_citations(question, k=3):
    sims = doc_emb @ embedder.encode("query: " + question, normalize_embeddings=True)
    ids = [int(i) for i in sims.argsort()[::-1][:k]]
    context = "\n".join(f"[{n}] {docs[i]}" for n, i in enumerate(ids, 1))
    prompt = ("以下の資料だけに基づいて質問に答えてください。各文の末尾に、根拠にした資料の番号を [1] のように必ず付けてください。\n\n"
              f"# 資料\n{context}\n\n# 質問\n{question}")
    return chat([{"role": "user", "content": prompt}]), ids

question = "何時まで営業していて、支払いには何が使えますか？"
answer, ids = answer_with_citations(question)
print(answer)
```

```python
# 検証: 回答の各文について、(1) 引用番号が実在するか (2) 引用先の資料がその文を含意するか (NLI)
nli_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
nli_tok = AutoTokenizer.from_pretrained(nli_name)
nli = AutoModelForSequenceClassification.from_pretrained(nli_name).eval()

def entailment(premise, hypothesis):
    with torch.no_grad():
        probs = nli(**nli_tok(premise, hypothesis, return_tensors="pt", truncation=True)).logits.softmax(-1)[0]
    return probs[0].item()      # このモデルのラベル 0 が entailment

def verify(answer, ids):
    for sent in [s for s in re.split(r"(?<=[。\]])\s*", answer) if s.strip()]:
        cited = [int(n) for n in re.findall(r"\[(\d+)\]", sent)]
        claim = re.sub(r"\[\d+\]", "", sent).strip()
        if not claim:
            continue
        if not cited:
            print(f"  出典なし      | {claim}")
        for n in cited:
            if not 1 <= n <= len(ids):
                print(f"  存在しない番号 [{n}] | {claim}")
            else:
                p = entailment(docs[ids[n - 1]], claim)
                print(f"  [{n}] 含意 {p:.2f} {'○' if p > 0.5 else '×'} | {claim}")

verify(answer, ids)
print("--- わざと出典を付け替えた回答 ---")
verify("営業時間は土日は 20 時までです [2]。クレジットカードが使えます [3]。", ids)
```

- 出典番号は「LLM がそう書いた」だけで、正しい保証はない。存在しない番号や、内容と合わない番号が付くことがある。
- NLI による検証は、LLM を使わずに「この資料からこの文が言えるか」を機械的に確かめる方法。回答を文に分けて 1 文ずつ調べるのがコツ。
- 複数の資料を組み合わせないと言えない文は、資料を連結して前提にする。NLI モデルは長い前提や数値の比較が苦手な点に注意。

## 答えが資料にないときに「わかりません」と答えさせる {rag_refusal}

検索結果に答えがないのに、LLM が無理に答えを作ってしまうのを防ぎたい。検索の段階と生成の段階の両方で歯止めをかける。

```python
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

docs = [
    "クックブック食堂の営業時間は平日 11 時から 15 時、土日は 11 時から 20 時です。月曜日は定休日です。",
    "クックブック食堂の看板メニューは、3 日間煮込んだ特製ビーフカレー (950 円) です。大盛りは 100 円増しです。",
    "クックブック食堂は 2019 年に京都市左京区で開業しました。店主の田中さんはインドで 2 年間修行しました。",
    "駐車場は店舗の裏に 3 台分あります。満車の場合は近隣のコインパーキングをご利用ください。",
    "予約は電話 (075-000-0000) のみ受け付けています。5 名以上の場合は前日までにご連絡ください。",
    "支払いは現金と QR コード決済に対応しています。クレジットカードは使えません。",
]
embedder = SentenceTransformer("intfloat/multilingual-e5-small")
doc_emb = embedder.encode(["passage: " + d for d in docs], normalize_embeddings=True)

answerable = ["カレーはいくらですか？", "駐車場はありますか？", "月曜日は開いていますか？", "店主はどこで修行しましたか？"]
unanswerable = ["テイクアウトはできますか？", "アレルギー対応のメニューはありますか？", "明日の東京の天気は？", "日本で一番高い山は？"]
for label, qs in [("答えがある", answerable), ("答えがない", unanswerable)]:
    for q in qs:
        sims = doc_emb @ embedder.encode("query: " + q, normalize_embeddings=True)
        print(f"{label}  最大類似度 {sims.max():.3f}  {q}")
```

```python
name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
llm = AutoModelForCausalLM.from_pretrained(name).eval()

def chat(messages, max_new_tokens=80):
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def rag(question, threshold=0.80, k=2, strict=True):
    sims = doc_emb @ embedder.encode("query: " + question, normalize_embeddings=True)
    if sims.max() < threshold:                       # 1 段目: 関連する資料がそもそも見つからない
        return "(検索で棄却) 申し訳ありませんが、その質問にはお答えできません。"
    context = "\n".join(f"- {docs[i]}" for i in sims.argsort()[::-1][:k])
    rule = "資料に答えが書かれていない場合は、推測せずに「わかりません」とだけ答えてください。" if strict else ""
    prompt = f"以下の資料だけに基づいて、質問に一文で答えてください。{rule}\n\n# 資料\n{context}\n\n# 質問\n{question}"
    return chat([{"role": "user", "content": prompt}])   # 2 段目: 資料はあるが答えは載っていない場合は、LLM に断らせる

for q in ["カレーはいくらですか？", "明日の東京の天気は？", "テイクアウトはできますか？", "アレルギー対応のメニューはありますか？"]:
    print(f"Q: {q}\n   指示あり: {rag(q)}\n   指示なし: {rag(q, strict=False)}")
```

- 類似度のしきい値だけでは「話題は近いが答えは載っていない」質問 (テイクアウト、アレルギー) を弾けない。プロンプトでの指示と二段構えにする。
- しきい値はモデルごとに違う (E5 は無関係な文でも 0.7 台後半が出る)。答えがある質問とない質問の例を用意して、分布を見て決める。
- 「わかりません」と言わせすぎると、答えられる質問まで断るようになる。断るべき質問と答えるべき質問の両方で評価する。

## 検索しやすい形に質問を書き換える (文脈の補完と HyDE) {rag_query_rewrite}

対話の途中の「それはいくら？」のような質問は、そのままでは検索に当たらない。質問を検索向けに書き換えたい。

```python
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

docs = [
    "クックブック食堂の営業時間は平日 11 時から 15 時、土日は 11 時から 20 時です。月曜日は定休日です。",
    "クックブック食堂の看板メニューは、3 日間煮込んだ特製ビーフカレー (950 円) です。大盛りは 100 円増しです。",
    "クックブック食堂は 2019 年に京都市左京区で開業しました。店主の田中さんはインドで 2 年間修行しました。",
    "駐車場は店舗の裏に 3 台分あります。満車の場合は近隣のコインパーキングをご利用ください。",
    "予約は電話 (075-000-0000) のみ受け付けています。5 名以上の場合は前日までにご連絡ください。",
    "支払いは現金と QR コード決済に対応しています。クレジットカードは使えません。",
]
embedder = SentenceTransformer("intfloat/multilingual-e5-small")
doc_emb = embedder.encode(["passage: " + d for d in docs], normalize_embeddings=True)

name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
llm = AutoModelForCausalLM.from_pretrained(name).eval()

def chat(messages, max_new_tokens=80):
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def top(query, k=2, prefix="query: "):
    sims = doc_emb @ embedder.encode(prefix + query, normalize_embeddings=True)
    return [(round(float(sims[i]), 3), docs[i][:32]) for i in sims.argsort()[::-1][:k]]
```

```python
# 1. 対話の文脈を補って、単独で意味が通じる質問にする
def rewrite(history, question):
    dialogue = "\n".join(f"{'ユーザー' if role == 'user' else 'アシスタント'}: {text}" for role, text in history)
    prompt = ("次の対話の最後の質問を、対話を読んでいない人にも意味が通じる、検索用の 1 つの質問に書き換えてください。"
              f"書き換えた質問だけを出力してください。\n\n{dialogue}\nユーザー: {question}")
    return chat([{"role": "user", "content": prompt}]).strip("「」")

history = [("user", "クックブック食堂の看板メニューは何ですか？"), ("assistant", "3 日間煮込んだ特製ビーフカレーです。")]
for q in ["それはいくらですか？", "大盛りにできますか？", "車で行っても大丈夫？"]:
    r = rewrite(history, q)
    print(f"{q} → {r}\n   そのまま検索: {top(q, 1)}\n   書き換えて検索: {top(r, 1)}")
```

```python
# 2. HyDE: 質問に対する「仮の答え」を LLM に書かせ、その文章で検索する (質問文より、答えの文のほうが文書に似ている)
def hyde(question):
    prompt = f"次の質問に対する答えとして、ありそうな内容を飲食店の案内文の形で 1 文だけ書いてください。事実でなくて構いません。\n\n質問: {question}"
    return chat([{"role": "user", "content": prompt}])

for q in ["カードで払いたいんだけど", "大人数で行きたい"]:
    h = hyde(q)
    print(f"{q}\n   仮の答え: {h}\n   そのまま検索: {top(q, 1)}\n   HyDE で検索: {top(h, 1, prefix='passage: ')}")
```

- 複数ターンの RAG では、質問の書き換えはほぼ必須。「それ」「そこ」や省略された主語を補わないと、検索が的外れになる。
- HyDE (Hypothetical Document Embeddings) の仮の答えは内容が間違っていてよい。語彙や文体が文書に近づくことで検索が当たりやすくなる。ただし LLM の呼び出しが 1 回増える。
- 書き換えの結果は必ずログに残す。回答がおかしいとき、書き換えで意味が変わっていた、ということがよくある。

## 検索の評価セットを作って手法を比べる {rag_eval_retrieval}

RAG の検索部分を改善したい。BM25、埋め込み、ハイブリッドのどれが自分のデータに合うのかを、数値で比べたい。

```python
!pip install -q rank_bm25 fugashi unidic-lite
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

# 評価セット: 「記事のタイトル」を質問、「その記事の本文」を正解の文書とみなす (人手のラベルなしで作れる)
df = load_livedoor().sample(600, random_state=0).reset_index(drop=True)
corpus = [t[:400] for t in df["text"]]
queries = df["title"].tolist()[:200]            # 正解は同じ番号の文書
print(len(corpus), "文書", len(queries), "クエリ |", queries[0], "→", corpus[0][:40].replace("\n", " "))
```

```python
from fugashi import Tagger
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

tagger = Tagger()
tok = lambda s: [w.surface.lower() for w in tagger(s) if w.feature.pos1 not in ("助詞", "助動詞", "補助記号", "空白")]
bm25 = BM25Okapi([tok(d) for d in corpus])
model = SentenceTransformer("intfloat/multilingual-e5-small")
doc_emb = model.encode(["passage: " + d for d in corpus], normalize_embeddings=True, batch_size=32, show_progress_bar=False)
q_emb = model.encode(["query: " + q for q in queries], normalize_embeddings=True, batch_size=32, show_progress_bar=False)

sparse = np.array([np.argsort(-bm25.get_scores(tok(q))) for q in queries])      # 各クエリについて、文書番号を良い順に並べたもの
dense = np.argsort(-(q_emb @ doc_emb.T), axis=1)

def rrf(rankings, k=60):
    scores = np.zeros(rankings[0].shape)
    for r in rankings:
        scores[np.arange(len(r))[:, None], r] += 1 / (k + np.arange(1, r.shape[1] + 1))
    return np.argsort(-scores, axis=1)

def evaluate(ranking):
    rank = np.array([np.where(ranking[i] == i)[0][0] + 1 for i in range(len(ranking))])   # 正解文書の順位
    return {"Recall@1": (rank <= 1).mean(), "Recall@5": (rank <= 5).mean(), "Recall@20": (rank <= 20).mean(), "MRR": (1 / rank).mean()}

pd.DataFrame({"BM25": evaluate(sparse), "埋め込み": evaluate(dense), "ハイブリッド (RRF)": evaluate(rrf([sparse, dense]))}).round(3).T
```

```python
# どちらか一方しか当てられなかったクエリを見ると、手法の得意・不得意が分かる
rank_s = np.array([np.where(sparse[i] == i)[0][0] for i in range(len(queries))])
rank_d = np.array([np.where(dense[i] == i)[0][0] for i in range(len(queries))])
print("BM25 だけ 1 位:", [queries[i] for i in np.where((rank_s == 0) & (rank_d > 4))[0][:3]])
print("埋め込みだけ 1 位:", [queries[i] for i in np.where((rank_d == 0) & (rank_s > 4))[0][:3]])
```

- RAG の改善は、まず検索の評価セットを作るところから。50〜200 件の「質問と正解文書の組」があれば、手法やチャンク設定の比較ができる。
- タイトルを質問に使う方法は手軽だが、実際のユーザーの質問とは性質が違う (本文と同じ語を含みやすく、BM25 に有利)。実運用のログや、LLM に文書から質問を作らせた合成データも併用する。
- Recall@k の k は、LLM に渡す件数に合わせる。5 件渡すなら Recall@5 が、そのまま「正解を読ませられた割合」になる。

## RAG の回答を LLM を使わずに評価する {rag_eval_answer}

RAG の回答が正しいか、資料に基づいているかを自動で確かめたい。評価のたびに LLM を呼ばずに、軽い方法で回したい。

```python
import re
import unicodedata
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 評価データ: 質問、検索された資料、システムの回答、正解 (answerable=False は「答えられない」が正解)
data = [
    {"q": "カレーはいくらですか？", "context": "看板メニューは特製ビーフカレー (950 円) です。大盛りは 100 円増しです。", "answer": "特製ビーフカレーは 950 円です。", "gold": ["950 円", "950円"], "answerable": True},
    {"q": "日曜日は何時までですか？", "context": "営業時間は平日 11 時から 15 時、土日は 11 時から 20 時です。", "answer": "日曜日は 15 時までです。", "gold": ["20 時", "20時"], "answerable": True},
    {"q": "クレジットカードは使えますか？", "context": "支払いは現金と QR コード決済に対応しています。クレジットカードは使えません。", "answer": "はい、主要なクレジットカードがご利用いただけます。", "gold": ["使えません", "使えない"], "answerable": True},
    {"q": "テイクアウトはできますか？", "context": "駐車場は店舗の裏に 3 台分あります。", "answer": "わかりません。", "gold": [], "answerable": False},
    {"q": "アレルギー対応はありますか？", "context": "看板メニューは特製ビーフカレー (950 円) です。", "answer": "はい、アレルギー対応メニューをご用意しています。", "gold": [], "answerable": False},
    {"q": "駐車場は何台分ですか？", "context": "駐車場は店舗の裏に 3 台分あります。満車の場合は近隣のコインパーキングをご利用ください。", "answer": "店舗の裏に 3 台分あり、料金は無料です。", "gold": ["3 台", "3台"], "answerable": True},
]

norm = lambda s: re.sub(r"\s", "", unicodedata.normalize("NFKC", s))
REFUSAL = re.compile(r"わかりません|分かりません|お答えできません|記載がありません|書かれていません")

nli_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
tok = AutoTokenizer.from_pretrained(nli_name)
nli = AutoModelForSequenceClassification.from_pretrained(nli_name).eval()

def faithfulness(context, answer):
    """回答の各文が資料から含意される確率の最小値 (1 文でも資料にないことを言っていれば低くなる)"""
    sents = [s for s in re.split(r"(?<=[。！？])", answer) if s.strip()]
    with torch.no_grad():
        probs = nli(**tok([context] * len(sents), sents, return_tensors="pt", padding=True, truncation=True)).logits.softmax(-1)
    return probs[:, 0].min().item()

rows = []
for d in data:
    refused = bool(REFUSAL.search(d["answer"]))
    rows.append({
        "質問": d["q"], "回答": d["answer"][:18],
        "正解を含む": any(norm(g) in norm(d["answer"]) for g in d["gold"]) if d["answerable"] else None,
        "正しく断った": refused if not d["answerable"] else None,
        "忠実性": None if refused else round(faithfulness(d["context"], d["answer"]), 2),
    })
pd.DataFrame(rows)
```

```python
df = pd.DataFrame(rows)
print(f"正解を含む割合 (答えられる質問): {df['正解を含む'].dropna().mean():.2f}")
print(f"正しく断った割合 (答えられない質問): {df['正しく断った'].dropna().mean():.2f}")
print(f"忠実性が 0.5 未満の回答 (資料にないことを述べている疑い): {(df['忠実性'].dropna() < 0.5).sum()} 件")
```

- 3 つの観点を分けて見る: 正しさ (正解を含むか)、忠実性 (資料に基づいているか)、棄却 (答えがないときに断れるか)。正しいが資料にない、資料どおりだが質問に答えていない、ということが起こる。
- 最後の例は「3 台分」は正しいが「料金は無料」は資料にない。正解を含むかだけを見ていると、この種の付け足し (ハルシネーション) を見逃す。
- NLI による忠実性は LLM による判定より粗いが、速くて安く、結果が安定している。変更のたびに回す回帰テストに向く。
- 最終的な品質の確認には、人手の評価か、LLM による評価 (LLM の章の LLM-as-a-judge) を併用する。
