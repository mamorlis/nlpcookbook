# 大規模言語モデル (LLM)

小さなオープンモデルを Colab 上で動かす方法と、Claude API を使う方法。プロンプト、構造化出力、RAG、LoRA。

## オープンな LLM とチャットする (チャットテンプレート) {llm_chat_local}

Hugging Face Hub の指示チューニング済みモデルを手元で動かして、システムプロンプト付きの対話をしたい。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"   # 0.5B の小型日本語モデル。CPU でも動く
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                             device_map="auto" if torch.cuda.is_available() else None).eval()

def chat(messages, max_new_tokens=200, **kwargs):
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id, **kwargs)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

messages = [
    {"role": "system", "content": "あなたは親切な日本語教師です。簡潔に答えてください。"},
    {"role": "user", "content": "「は」と「が」の違いを、例文を 1 つずつ挙げて説明してください。"},
]
reply = chat(messages)
print(reply)
```

```python
# 対話を続ける: これまでのやりとりをすべて messages に積んで渡す (モデル自体は状態を持たない)
messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": "今の説明を、小学生にもわかるように一文で言い直してください。"}]
print(chat(messages))
```

```python
# チャットテンプレートが実際に作る文字列を確認する
print(tokenizer.apply_chat_template(messages[:2], add_generation_prompt=True, tokenize=False))
```

- 0.5B のモデルは日本語としては流暢だが、内容は不正確なことが多い (上の説明も文法的には誤りを含む)。動作確認用と割り切り、実用には大きなモデルを使う。
- モデルごとに特殊トークンの形式が違う。自分で文字列を組み立てず、必ず `apply_chat_template` を使う。
- `add_generation_prompt=True` で「ここからアシスタントの発話」という印が末尾に付く。忘れるとモデルがユーザーの発話の続きを書いてしまう。
- 同じコードで、GPU があれば `Qwen/Qwen2.5-7B-Instruct`、`google/gemma-3-4b-it`、`sbintuitions/sarashina2.2-3b-instruct-v0.1` などのより大きなモデルに差し替えられる。

## 生成結果を 1 文字ずつ表示する (ストリーミング) {llm_streaming_local}

生成が終わるまで待たずに、出力されたそばから表示したい。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()

messages = [{"role": "user", "content": "自然言語処理とは何か、3 文で説明してください。"}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)   # 標準出力に逐次表示
_ = model.generate(**inputs, max_new_tokens=120, do_sample=False, streamer=streamer, pad_token_id=tokenizer.eos_token_id)
```

```python
# 自分のコードで受け取りたい場合 (Web アプリに流すなど): 別スレッドで生成し、イテレータから読む
from threading import Thread
from transformers import TextIteratorStreamer

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
thread = Thread(target=model.generate, kwargs=dict(**inputs, max_new_tokens=60, do_sample=False, streamer=streamer, pad_token_id=tokenizer.eos_token_id))
thread.start()
chunks = [chunk for chunk in streamer]
thread.join()
print(len(chunks), "個の断片:", chunks[:8])
```

## Few-shot プロンプトでテキストを分類する {llm_few_shot}

学習なしで、いくつかの例をプロンプトに入れるだけで LLM に分類させたい。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()

def chat(messages, max_new_tokens=10):
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

INSTRUCTION = "次のレビューの感情を「ポジティブ」「ネガティブ」のどちらかで答えてください。ラベルだけを出力してください。"
examples = [("料理がとても美味しくて、店員さんも親切でした。", "ポジティブ"), ("1 時間待たされた上に、注文を間違えられた。", "ネガティブ"),
            ("値段のわりに量が少なくてがっかり。", "ネガティブ"), ("雰囲気が良く、また来たいと思った。", "ポジティブ")]

def build(text, shots):
    messages = [{"role": "system", "content": INSTRUCTION}]
    for x, y in shots:               # 例は「ユーザーの入力 → アシスタントの出力」の対話として与える
        messages += [{"role": "user", "content": x}, {"role": "assistant", "content": y}]
    return messages + [{"role": "user", "content": text}]

tests = [("最高の体験でした。友人にも勧めたい。", "ポジティブ"), ("二度と行きません。", "ネガティブ"), ("部屋が汚く、対応も最悪だった。", "ネガティブ"),
         ("期待以上のサービスに感動した。", "ポジティブ"), ("可もなく不可もなく、といったところ。でも駅から近いのは助かる。", "ポジティブ"),
         ("写真と全然違う料理が出てきた。", "ネガティブ")]
for n_shots in [0, 4]:
    preds = [chat(build(t, examples[:n_shots])) for t, _ in tests]
    acc = sum(p.startswith(g) for p, (_, g) in zip(preds, tests)) / len(tests)
    print(f"{n_shots}-shot: 正解率 {acc:.2f}  出力: {preds}")
```

- 例があると、出力の「形式」が揃う効果が大きい (ゼロショットでは余計な説明が付きがち)。
- 例の選び方と順序で結果が変わる。ラベルの偏りがあると、その多数派ラベルに引っ張られる。
- 出力は必ず後処理する (前後の空白や句点を除く、想定外の出力を検出する)。次項のように尤度で選ばせる方法もある。

## 選択肢の尤度を比べて LLM に分類させる {llm_choice_scoring}

LLM に自由に生成させると、選択肢以外の答えや余計な文が返ってくる。各選択肢の生成確率を直接比べて、必ず選択肢の中から選ばせたい。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()

@torch.no_grad()
def loglik(messages, choices):
    """各選択肢を応答として続けたときの対数尤度"""
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    scores = []
    for c in choices:
        choice_ids = tokenizer(c, add_special_tokens=False, return_tensors="pt")["input_ids"]
        ids = torch.cat([prompt_ids, choice_ids], dim=1)
        logp = model(ids).logits[0, prompt_ids.shape[1] - 1:-1].log_softmax(-1)      # 選択肢の各トークンの予測分布
        scores.append(logp[torch.arange(choice_ids.shape[1]), choice_ids[0]].sum().item())
    return torch.tensor(scores)

labels = ["ポジティブ", "ネガティブ"]
ask = lambda text: [{"role": "user", "content": f"次のレビューの感情を「ポジティブ」「ネガティブ」のどちらかで答えてください。\n\nレビュー: {text}"}]

texts = ["最高の体験でした。友人にも勧めたい。", "二度と行きません。", "部屋が汚く、対応も最悪だった。", "期待以上のサービスに感動した。",
         "写真と全然違う料理が出てきた。", "値段は高いが、それだけの価値はある。"]
for text in texts:
    probs = loglik(ask(text), labels).softmax(0)
    print(labels[probs.argmax()], [round(p, 3) for p in probs.tolist()], text)
```

```python
# 較正: 中身のない入力に対するスコア (= モデルがもともと持っているラベルの偏り) を差し引く
prior = loglik(ask("（なし）"), labels)
print("偏り:", dict(zip(labels, [round(p, 3) for p in prior.softmax(0).tolist()])))
for text in texts:
    probs = (loglik(ask(text), labels) - prior).softmax(0)
    print(labels[probs.argmax()], [round(p, 3) for p in probs.tolist()], text)
```

- 出力が必ず選択肢のどれかになり、確信度 (確率) も得られる。小さなモデルでも安定する。
- モデルは特定のラベルを好む偏りを持つ。内容のない入力でのスコアを差し引く「文脈較正」(Zhao et al., 2021) で補正できる。
- トークン数の多い選択肢は不利になる。長さで割る、あるいは選択肢を「A」「B」「C」のような 1 トークンの記号にして回避する。
- API 経由のモデルでは対数尤度が取れないことが多い。その場合は構造化出力 (enum) で選択肢を制約する。

## 複数のプロンプトをまとめてバッチ生成する {llm_batch_generation}

大量の入力を LLM で処理したい。1 件ずつ `generate` すると遅いので、バッチでまとめて生成したい。

```python
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()
tokenizer.padding_side = "left"               # 生成モデルのバッチ処理では必ず左パディング
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

words = ["形態素解析", "機械翻訳", "固有表現抽出", "要約", "質問応答", "感情分析", "音声認識", "情報検索"]
prompts = [tokenizer.apply_chat_template([{"role": "user", "content": f"「{w}」を一文で説明してください。"}],
                                         add_generation_prompt=True, tokenize=False) for w in words]

@torch.no_grad()
def generate(batch):
    enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
    out = model.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

t = time.perf_counter(); single = [generate([p])[0] for p in prompts]; t1 = time.perf_counter() - t
t = time.perf_counter(); batched = generate(prompts); t2 = time.perf_counter() - t
print(f"1 件ずつ: {t1:.1f} 秒 / バッチ: {t2:.1f} 秒")
for w, s in zip(words, batched):
    print(f"{w}: {s.strip()}")
```

- 右パディングのままだと、短いプロンプトの後ろにパディングが入り、その続きを生成することになって出力が壊れる。
- バッチ内の長さを揃える (長さでソートする) とパディングが減って効率が良い。
- 数万件以上を本格的に処理するなら、vLLM などの推論サーバー (連続バッチ処理) が桁違いに速い。

## LLM の出力から JSON を取り出して検証する {llm_json_output}

LLM に JSON で答えさせて、プログラムで扱いたい。余計な文や壊れた JSON が返ってきても落ちないようにしたい。

```python
!pip install -q pydantic
```

```python
import json
import re
from typing import Literal
from pydantic import BaseModel, Field, ValidationError

class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    score: int = Field(ge=1, le=5)
    aspects: list[str]

def extract_json(text):
    """コードブロックや前後の説明文があっても、最初の JSON オブジェクトを取り出す"""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find("{")
    if start < 0:
        raise ValueError("JSON が見つかりません")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])   # 最初のオブジェクトだけ読む (後ろに文が続いていてもよい)
    return obj

outputs = [
    '{"sentiment": "positive", "score": 5, "aspects": ["料理", "接客"]}',
    'はい、分析結果は以下のとおりです。\n```json\n{"sentiment": "negative", "score": 2, "aspects": ["待ち時間"]}\n```\nご参考になれば幸いです。',
    '{"sentiment": "やや肯定的", "score": 4, "aspects": ["価格"]}',
    '{"sentiment": "neutral", "score": 7, "aspects": "立地"}',
    '{"sentiment": "positive", "score": 4, "aspects": ["雰囲気"',
]
for out in outputs:
    try:
        print("OK  ", Review.model_validate(extract_json(out)))
    except (ValueError, ValidationError) as e:
        print("NG  ", str(e).split("\n")[0], "|", out[:40].replace("\n", " "))
```

```python
# 失敗したらエラー内容を伝えて再生成させるループ (llm は「メッセージ列 → 文字列」の関数なら何でもよい)
def generate_validated(llm, messages, schema, max_retries=2):
    for _ in range(max_retries + 1):
        out = llm(messages)
        try:
            return schema.model_validate(extract_json(out))
        except (ValueError, ValidationError) as e:
            messages = messages + [{"role": "assistant", "content": out},
                                   {"role": "user", "content": f"出力が不正です: {e}\nスキーマに従う JSON だけを出力し直してください。"}]
    raise RuntimeError("有効な JSON が得られませんでした")

fake_outputs = iter([outputs[2], outputs[0]])   # 1 回目は不正、2 回目は正しい出力を返すダミーの LLM で動作確認
print(generate_validated(lambda m: next(fake_outputs), [{"role": "user", "content": "..."}], Review))
print(json.dumps(Review.model_json_schema(), ensure_ascii=False)[:150], "...")   # スキーマはプロンプトに埋め込める
```

- プロンプトには、スキーマ (`model_json_schema()`) と出力例を入れる。
- 可能なら、API やライブラリの「構造化出力」機能を使うのが確実 (文法的に正しい JSON しか生成できないように制約される)。Claude API の例は後述。ローカルモデルなら `outlines` などの制約付きデコードのライブラリがある。

## プロンプトをテンプレートで管理し、似た例を動的に選ぶ {prompt_template}

プロンプトをコードに直書きせずテンプレートとして管理したい。Few-shot の例を、入力に似たものから自動で選びたい。

```python
!pip install -q jinja2
```

```python
from jinja2 import Template

TEMPLATE = Template("""あなたは日本語学習者の作文を添削する教師です。
誤りを訂正した文だけを出力してください。誤りがなければ、そのまま出力してください。
{% if examples %}
{% for ex in examples -%}
入力: {{ ex.src }}
出力: {{ ex.tgt }}

{% endfor -%}
{% endif %}入力: {{ text }}
出力:""")

examples = [
    {"src": "私は学校を行きます。", "tgt": "私は学校に行きます。"},
    {"src": "昨日は雨が降ります。", "tgt": "昨日は雨が降りました。"},
    {"src": "この本は面白いでした。", "tgt": "この本は面白かったです。"},
    {"src": "友達に会うために駅で行った。", "tgt": "友達に会うために駅に行った。"},
    {"src": "日本の食べ物が美味しいと思います。", "tgt": "日本の食べ物は美味しいと思います。"},
    {"src": "静かの部屋で勉強します。", "tgt": "静かな部屋で勉強します。"},
]
print(TEMPLATE.render(text="図書館を勉強しました。", examples=examples[:2]))
```

```python
# 入力に似た例を埋め込みの類似度で選ぶ (kNN few-shot)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-small")
ex_emb = model.encode(["query: " + e["src"] for e in examples], normalize_embeddings=True)

def select(text, k=2):
    sims = ex_emb @ model.encode("query: " + text, normalize_embeddings=True)
    return [examples[i] for i in sims.argsort()[::-1][:k]][::-1]   # 最も似た例を入力の直前に置く

for text in ["この映画は楽しいでした。", "公園を散歩に行きました。", "きれいの花が咲いています。"]:
    print(text, "→ 選ばれた例:", [e["src"] for e in select(text)])
```

- テンプレートを別ファイル (`prompts/gec_v2.j2` など) にしてバージョン管理すると、実験の再現と比較がしやすい。
- 入力に似た例を選ぶと、固定の例より精度が上がることが多い。ただし似た例の答えをそのまま真似る傾向も出る。

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

## 検索した文書を LLM に渡して答えさせる (最小の RAG) {rag_minimal}

LLM が知らない情報 (社内文書、最新情報) について、関連文書を検索してから答えさせたい。

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

name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
llm = AutoModelForCausalLM.from_pretrained(name).eval()

def generate(prompt):
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
```

```python
def retrieve(question, k=2):
    sims = doc_emb @ embedder.encode("query: " + question, normalize_embeddings=True)
    return [(int(i), docs[i]) for i in sims.argsort()[::-1][:k]]

def rag(question):
    hits = retrieve(question)
    context = "\n".join(f"[{i}] {d}" for i, d in hits)
    prompt = (f"以下の資料だけに基づいて質問に答えてください。資料に答えがなければ「わかりません」と答えてください。\n\n"
              f"# 資料\n{context}\n\n# 質問\n{question}")
    return generate(prompt), [i for i, _ in hits]

for q in ["カレーはいくらですか？", "日曜日は何時まで開いていますか？", "クレジットカードで払えますか？"]:
    answer, sources = rag(q)
    print(f"Q: {q}\n  検索なし: {generate(q)[:60]}\n  RAG: {answer} (参照: {sources})")
```

- RAG = 検索 (Retrieval) + 生成 (Generation)。検索なしだと、モデルは知らない店についてもっともらしい嘘 (ハルシネーション) を答える。
- 精度の大半は検索で決まる。まず「正解の文書が上位 k 件に入っているか」を評価し、ハイブリッド検索やリランキングで改善する。
- 参照した文書の番号を一緒に返すと、利用者が根拠を確認できる。

## LLM を 4bit 量子化して少ないメモリで動かす {llm_4bit gpu norun}

7B クラスのモデルを Colab の T4 GPU (16GB) で余裕を持って動かしたい。

```python
!pip install -q bitsandbytes accelerate
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

name = "Qwen/Qwen2.5-7B-Instruct"
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",               # 正規分布を仮定した 4bit 量子化
    bnb_4bit_compute_dtype=torch.bfloat16,   # 計算は 16bit で行う
    bnb_4bit_use_double_quant=True,          # 量子化の定数も量子化して、さらに節約
)
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, quantization_config=quant, device_map="auto")
print(f"メモリ使用量: {model.get_memory_footprint() / 1e9:.1f} GB")   # float16 なら約 15GB → 4bit で約 5.5GB
```

```python
messages = [{"role": "user", "content": "形態素解析とサブワード分割の違いを 3 行で説明してください。"}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
!nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

- 4bit 量子化による精度の低下は、多くのタスクでわずか。「大きなモデルの 4bit」は「小さなモデルの 16bit」より良いことが多い。
- bitsandbytes は NVIDIA GPU 専用。CPU や Mac で動かすなら、GGUF 形式 + llama.cpp (Ollama) を使う。
- 速度は 16bit より遅くなる。メモリに収まるなら `torch_dtype=torch.float16` で普通に読み込むほうが速い。

## LoRA で LLM を少ないメモリでファインチューニングする {lora_finetune gpu norun}

LLM を自分のタスク (ここでは文法誤り訂正) に適応させたい。全パラメータを更新するのは重いので、LoRA で一部だけ学習する。

```python
!pip install -q peft datasets accelerate
```

```python
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="auto")

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])   # アテンションの射影行列に低ランク行列を追加
model = get_peft_model(model, lora)
model.print_trainable_parameters()
```

```python
# 学習データ: 誤りを含む文 → 訂正文 (実際には数千〜数万ペアを用意する)
pairs = [("私は学校を行きます。", "私は学校に行きます。"), ("昨日は雨が降ります。", "昨日は雨が降りました。"),
         ("この本は面白いでした。", "この本は面白かったです。"), ("静かの部屋で勉強します。", "静かな部屋で勉強します。"),
         ("日本語を話すことが上手になりたい。", "日本語を話すのが上手になりたい。"), ("友達を会いました。", "友達に会いました。"),
         ("彼は背が高いだ。", "彼は背が高い。"), ("毎日、日本語が勉強します。", "毎日、日本語を勉強します。")] * 25
INSTRUCTION = "次の文の文法誤りを訂正してください。"

def encode(example):
    messages = [{"role": "user", "content": f"{INSTRUCTION}\n{example['src']}"}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(example["tgt"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    return {"input_ids": prompt_ids + answer_ids,
            "labels": [-100] * len(prompt_ids) + answer_ids}    # プロンプト部分は損失を計算しない

ds = Dataset.from_list([{"src": s, "tgt": t} for s, t in pairs]).map(encode, remove_columns=["src", "tgt"])
```

```python
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

args = TrainingArguments(output_dir="lora-out", num_train_epochs=3, per_device_train_batch_size=8, learning_rate=2e-4,
                         warmup_steps=10, logging_steps=10, save_strategy="no", report_to="none", bf16=True, seed=0)
trainer = Trainer(model=model, args=args, train_dataset=ds,
                  data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))
trainer.train()
```

```python
def correct(text):
    messages = [{"role": "user", "content": f"{INSTRUCTION}\n{text}"}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

for s in ["私は図書館を行きました。", "この映画は楽しいでした。", "きれいの花が咲いています。"]:
    print(s, "→", correct(s))

model.save_pretrained("gec-lora")   # 保存されるのは追加した LoRA の重み (数 MB〜数十 MB) だけ
```

- 学習するパラメータは全体の 1% 未満。元のモデルの重みは固定で、アダプタだけを配布・切り替えできる。
- 読み込み: `PeftModel.from_pretrained(ベースモデル, "gec-lora")`。推論を速くするなら `merge_and_unload()` で重みを統合する。
- 学習率は全体のファインチューニングより高め (1e-4〜3e-4)。`r` は 8〜64 が一般的。
- 4bit 量子化と組み合わせる (QLoRA) と、7B クラスでも T4 で学習できる。`prepare_model_for_kbit_training` を使う。

## Claude API を呼び出す {claude_basic key norun}

Claude を API 経由で使いたい。Colab のシークレット機能で API キーを安全に扱いたい。

```python
!pip install -q anthropic
```

```python
import anthropic
from google.colab import userdata

# 左の 🔑 (シークレット) に ANTHROPIC_API_KEY を登録し、「ノートブックからのアクセス」を有効にしておく
client = anthropic.Anthropic(api_key=userdata.get("ANTHROPIC_API_KEY"))

response = client.beta.messages.create(
    model="claude-opus-5-5",
    max_tokens=16000,
    system="あなたは自然言語処理の専門家です。簡潔に答えてください。",
    messages=[{"role": "user", "content": "BPE と WordPiece の違いを 3 行で説明してください。"}],
    betas=["server-side-fallback-2026-07-01"],
    fallbacks="default",   # 安全フィルタで応答が拒否された場合に、別のモデルで自動的に再実行する
)

if response.stop_reason == "refusal":
    print("応答が拒否されました:", response.stop_details)
else:
    print("".join(block.text for block in response.content if block.type == "text"))
print(response.usage.input_tokens, "入力トークン /", response.usage.output_tokens, "出力トークン")
```

```python
# 対話を続ける: API は状態を持たないので、履歴をすべて送り直す
history = [{"role": "user", "content": "「は」と「が」の違いを一言で。"}]
r1 = client.messages.create(model="claude-opus-5-5", max_tokens=16000, messages=history)
history += [{"role": "assistant", "content": r1.content}, {"role": "user", "content": "例文を 1 つずつ挙げてください。"}]
r2 = client.messages.create(model="claude-opus-5-5", max_tokens=16000, messages=history)
print("".join(b.text for b in r2.content if b.type == "text"))
```

- API キーをノートブックに直接書かない。共有や GitHub への push で漏れる。
- `response.content` はブロックのリスト (テキストのほかに思考ブロックなどが入る)。`type == "text"` のものだけを取り出す。
- `max_tokens` は出力の上限。小さくしすぎると途中で切れる (`stop_reason == "max_tokens"`)。
- 履歴に応答を積むときは、テキストだけでなく `response.content` をそのまま渡す。
- モデル ID と料金は [公式ドキュメント](https://platform.claude.com/docs/en/about-claude/models/overview) で確認する。大量の単純な処理には、より安価な `claude-haiku-5-5` なども選べる。

## Claude の応答をストリーミングで受け取る {claude_streaming key norun}

長い応答を、生成されたそばから表示したい。長い出力でのタイムアウトも避けたい。

```python
!pip install -q anthropic
```

```python
import anthropic
from google.colab import userdata

client = anthropic.Anthropic(api_key=userdata.get("ANTHROPIC_API_KEY"))

with client.messages.stream(
    model="claude-opus-5-5",
    max_tokens=64000,
    messages=[{"role": "user", "content": "自然言語処理の歴史を、1950 年代から現在まで 500 字程度でまとめてください。"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
    final = stream.get_final_message()   # 全体が欲しいときは最後にまとめて取得できる

print("\n---\n", final.stop_reason, final.usage.output_tokens, "出力トークン")
```

- 出力が長くなりうる場合 (`max_tokens` が大きい場合) は、ストリーミングを使うのが基本。非ストリーミングだと HTTP のタイムアウトに当たることがある。
- 逐次表示が不要でも、`stream.get_final_message()` だけ使えば、タイムアウトを避けつつ通常の応答と同じ形で結果を得られる。

## Claude に決まった形式 (JSON スキーマ) で出力させる {claude_structured key norun}

テキストから情報を抽出して、型の決まったデータとして受け取りたい。JSON のパース失敗に悩まされたくない。

```python
!pip install -q anthropic pydantic
```

```python
from typing import Literal
import anthropic
from google.colab import userdata
from pydantic import BaseModel

client = anthropic.Anthropic(api_key=userdata.get("ANTHROPIC_API_KEY"))

class Entity(BaseModel):
    text: str
    type: Literal["人名", "組織名", "地名", "日付", "金額"]

class Extraction(BaseModel):
    entities: list[Entity]
    summary: str
    sentiment: Literal["positive", "negative", "neutral"]

text = "2024年4月、山田太郎氏はトヨタ自動車を退職し、京都市でスタートアップを創業した。創業半年で3億円を調達し、順調な滑り出しを見せている。"
response = client.messages.parse(
    model="claude-opus-5-5",
    max_tokens=16000,
    messages=[{"role": "user", "content": f"次の文章から固有表現を抽出し、一文で要約し、論調を判定してください。\n\n{text}"}],
    output_format=Extraction,
)

if response.stop_reason == "refusal":
    print("応答が拒否されました")
else:
    result = response.parsed_output          # 検証済みの Extraction インスタンス
    for e in result.entities:
        print(e.type, e.text)
    print(result.summary, "/", result.sentiment)
```

- 構造化出力では、スキーマに従う JSON しか生成されないように制約される。`Literal` で選択肢を限定すれば、分類タスクにもそのまま使える。
- 抽出した `text` が原文に実在するかは保証されない。`e.text in text` で検証するとよい。文字位置が必要なら、自分で `text.find()` する。
- スキーマに書けない制約 (最小値・最大値など) は、Pydantic 側の検証で確認する。

## Claude で大量のテキストを安く処理する (Message Batches API) {claude_batch key norun}

数千〜数万件のテキストを LLM で処理したい。急がないので、できるだけ安く済ませたい。

```python
!pip install -q anthropic
```

```python
import time
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from google.colab import userdata

client = anthropic.Anthropic(api_key=userdata.get("ANTHROPIC_API_KEY"))

reviews = {"r001": "料理がとても美味しくて、店員さんも親切でした。", "r002": "1 時間待たされた上に、注文を間違えられた。",
           "r003": "可もなく不可もなく。駅から近いのは助かる。"}
SYSTEM = "レビューの感情を positive / negative / neutral のいずれか 1 語で答えてください。"

batch = client.messages.batches.create(requests=[
    Request(custom_id=rid, params=MessageCreateParamsNonStreaming(
        model="claude-opus-5-5", max_tokens=1024, system=SYSTEM, messages=[{"role": "user", "content": text}]))
    for rid, text in reviews.items()
])
print(batch.id, batch.processing_status)
```

```python
while True:
    batch = client.messages.batches.retrieve(batch.id)
    if batch.processing_status == "ended":
        break
    print("処理中:", batch.request_counts.processing, "件")
    time.sleep(60)

results = {}
for r in client.messages.batches.results(batch.id):   # 結果は順不同で返るので、custom_id で対応付ける
    if r.result.type == "succeeded":
        results[r.custom_id] = "".join(b.text for b in r.result.message.content if b.type == "text").strip()
    else:
        results[r.custom_id] = f"({r.result.type})"
for rid, text in reviews.items():
    print(results[rid], "|", text)
```

- 非同期で処理される代わりに、料金が通常の半額になる。ほとんどのバッチは 1 時間以内に終わる (最大 24 時間)。
- バッチ ID を控えておけば、Colab のセッションが切れても後から結果を取得できる (結果は 29 日間保持される)。
- 共通の長いシステムプロンプトがあるなら、プロンプトキャッシュと併用するとさらに安くなる。

## Claude API のトークン数と料金を事前に見積もる {claude_count_tokens key norun}

データセット全体を API に投げる前に、何トークンになり、いくらかかるのかを見積もりたい。

```python
!pip install -q anthropic
```

```python
import anthropic
from google.colab import userdata

client = anthropic.Anthropic(api_key=userdata.get("ANTHROPIC_API_KEY"))
MODEL = "claude-opus-5-5"
SYSTEM = "次の文の文法誤りを訂正し、訂正後の文だけを出力してください。"

def count(text):
    return client.messages.count_tokens(model=MODEL, system=SYSTEM, messages=[{"role": "user", "content": text}]).input_tokens

for text in ["Natural language processing is fun.", "自然言語処理は楽しい。", "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。"]:
    print(count(text), "トークン |", len(text), "文字 |", text)
```

```python
# データセット全体の見積もり: 一部をサンプリングして平均をとり、件数を掛ける
import random

dataset = ["私は学校を行きます。", "昨日は雨が降ります。", "この本は面白いでした。"] * 1000   # 自分のデータに置き換える
sample = random.Random(0).sample(dataset, 20)
avg_in = sum(count(t) for t in sample) / len(sample)
avg_out = 40                                   # 出力トークン数は、少数を実際に処理して実測する
PRICE_IN, PRICE_OUT = 4.00, 20.00              # 100 万トークンあたりのドル。最新の料金表で確認すること

cost = len(dataset) * (avg_in * PRICE_IN + avg_out * PRICE_OUT) / 1e6
print(f"{len(dataset)} 件 × 入力 {avg_in:.0f} + 出力 {avg_out} トークン ≒ ${cost:.2f} (Batches API なら ${cost / 2:.2f})")
```

- トークン数はモデルによって違う。必ず実際に使うモデルの ID で数える。他社のトークナイザ (tiktoken など) での推定は当てにならない。
- 出力側に思考 (thinking) のトークンも含まれて課金される点に注意。事前に数十件で実測するのが確実。
- トークン数のカウント自体は無料 (レート制限はある)。

## LLM に生成文の品質を採点させる (LLM-as-a-judge) {llm_judge key norun}

要約や翻訳、対話応答の品質を、参照訳との一致ではなく、LLM に評価基準を与えて採点させたい。

```python
!pip install -q anthropic pydantic scipy
```

```python
from typing import Literal
import anthropic
from google.colab import userdata
from pydantic import BaseModel

client = anthropic.Anthropic(api_key=userdata.get("ANTHROPIC_API_KEY"))

class Judgment(BaseModel):
    reasoning: str                    # 先に根拠を書かせてから点数を付けさせる
    faithfulness: Literal[1, 2, 3, 4, 5]
    fluency: Literal[1, 2, 3, 4, 5]

RUBRIC = """あなたは要約の評価者です。元の文章と要約を読み、次の 2 つの観点を 1〜5 の整数で採点してください。
- faithfulness (忠実性): 要約の内容が元の文章と矛盾せず、書かれていない情報を含まないか。5 = 完全に忠実、1 = 重大な誤りがある。
- fluency (流暢性): 日本語として自然で読みやすいか。5 = 完全に自然、1 = 意味が取れない。
要約が短いことや情報が少ないことは減点しないでください。"""

def judge(source, summary):
    r = client.messages.parse(
        model="claude-opus-5-5", max_tokens=16000, system=RUBRIC,
        messages=[{"role": "user", "content": f"# 元の文章\n{source}\n\n# 要約\n{summary}"}],
        output_format=Judgment,
    )
    return r.parsed_output

source = "同社の 2023 年度の売上高は前年比 5% 増の 120 億円だった。一方、原材料費の高騰により営業利益は 8 億円と、前年の 10 億円から減少した。"
for summary in ["売上高は 5% 増の 120 億円だったが、営業利益は 8 億円に減少した。", "売上高、営業利益ともに前年を上回り、好調だった。",
                "売上高 120 億円は 5% 増で営業利益が 8 億円の減少を原材料費だった。"]:
    j = judge(source, summary)
    print(f"忠実性 {j.faithfulness} / 流暢性 {j.fluency} | {summary}\n   {j.reasoning[:80]}")
```

```python
# 評価器そのものを検証する: 人手の評価との相関を必ず確認する
from scipy.stats import spearmanr

human = [5, 1, 3]                      # 人間が付けた忠実性スコア (実際には 100 件以上で確認する)
llm = [5, 1, 4]                        # judge() の faithfulness
print("Spearman:", round(spearmanr(human, llm).statistic, 3))
```

- 採点基準 (ルーブリック) を具体的に書く。観点ごとに分けて採点させ、点数の前に根拠を書かせると安定する。
- LLM の評価には偏りがある (長い出力を好む、自分と同系統のモデルの出力に甘い、提示順に影響される)。2 つの出力を比べさせる場合は、順序を入れ替えて 2 回評価する。
- 人手評価との一致を確認していない LLM 評価は、評価として使ってはいけない。
- 評価されるモデルと評価するモデルは、できれば別系統のものにする。
