# LLM の API（Gemini と Claude）

API で大きなモデルを使う。Colab の組み込み AI、無料枠のある Gemini と有料の Claude で、呼び出し、ストリーミング、JSON スキーマによる構造化出力、トークン数と料金の確認、大量のテキストのバッチ処理を扱う。手元で動かすオープンなモデルは「大規模言語モデル (LLM)」の章を参照。

## Colab の組み込み AI (google.colab.ai) で Gemini を呼び出す {colab_ai norun}

API キーを用意せずに、Colab の中から手軽に Gemini を使いたい。

```python
from google.colab import ai

# 使えるモデルの一覧 (時期によって変わるので、まず確認する)
models = ai.list_models()
print(models)
```

```python
# 1 回の質問に答えさせる。モデルを省略すると、Colab が決めた標準のモデルが使われる
print(ai.generate_text("BPE と WordPiece の違いを 3 行で説明してください。"))
```

```python
# モデルを指定する / ストリーミングで、生成されたそばから表示する
model = next((m for m in models if "flash" in m), models[0])
for chunk in ai.generate_text("自然言語処理の研究テーマを 5 つ、1 行ずつ挙げてください。", model_name=model, stream=True):
    print(chunk or "", end="")
```

```python
# システムプロンプトや対話の履歴は渡せないので、指示や例を 1 つのプロンプトにまとめて書く
reviews = ["画面がきれいで操作も軽快。買ってよかった。", "届いてすぐ壊れた。サポートの対応も遅い。", "値段相応。可もなく不可もなく。"]
prompt = ("次のレビューを「ポジティブ」「ネガティブ」「中立」のどれかに分類し、1 行に 1 つずつ、ラベルだけを出力してください。\n\n"
          + "\n".join(f"{i + 1}. {r}" for i, r in enumerate(reviews)))
print(ai.generate_text(prompt))
```

- `google.colab.ai` は Colab に組み込まれた機能で、API キーも Google Cloud のアカウントも要らない。Colab の外（手元の Python など）では使えない。
- できることは「1 つのプロンプトを送ってテキストを受け取る」だけ。システムプロンプト、対話の履歴、JSON スキーマによる出力の制約、温度などの設定は渡せない。それらが必要なら、次のレシピの Gemini API を使う。
- 使えるモデルと利用の上限は、Colab の利用状況や時期によって変わる。上限に達するとエラーになるので、大量の処理には向かない。
- モデル名は `google/gemini-...` のような形式。`list_models()` の結果から選ぶ。
- 送った内容は Google のサービスで処理される。個人情報や機密の文書は送らない。

## Gemini API を呼び出す（無料枠） {gemini_basic key norun}

Gemini を API で使いたい。システムプロンプト、温度、対話の履歴などを細かく指定したい。Colab のシークレット機能で API キーを安全に扱いたい。

```python
!pip install -q google-genai
```

```python
from google import genai
from google.colab import userdata

# Google AI Studio (aistudio.google.com) で API キーを作り、左の 🔑 (シークレット) に GOOGLE_API_KEY として登録し、
# 「ノートブックからのアクセス」を有効にしておく (シークレットの画面の「Gemini API キーをインポート」からも登録できる)
client = genai.Client(api_key=userdata.get("GOOGLE_API_KEY"))

# 使えるモデルを確認する (モデル名の入れ替わりが早いので、決め打ちせずに一覧から選ぶ)
for m in client.models.list():
    if "generateContent" in (m.supported_actions or []) and "flash" in m.name:
        print(m.name, m.input_token_limit, m.output_token_limit)
```

```python
from google.genai import types

MODEL = "gemini-flash-latest"      # 最新の Flash モデルを指す別名。上の一覧の名前 (models/ を除いた部分) を直接指定してもよい

response = client.models.generate_content(
    model=MODEL,
    contents="BPE と WordPiece の違いを 3 行で説明してください。",
    config=types.GenerateContentConfig(
        system_instruction="あなたは自然言語処理の専門家です。簡潔に答えてください。",
        temperature=0.2,
    ),
)
print(response.text)
u = response.usage_metadata
print(u.prompt_token_count, "入力トークン /", u.candidates_token_count, "出力トークン / 思考", u.thoughts_token_count)
```

```python
# 対話を続ける: chats を使うと、履歴を自動で保持して毎回送ってくれる
chat = client.chats.create(model=MODEL, config=types.GenerateContentConfig(system_instruction="日本語教師として、やさしく答えてください。"))
print(chat.send_message("「は」と「が」の違いを一言で。").text)
print(chat.send_message("例文を 1 つずつ挙げてください。").text)
print(len(chat.get_history()), "件の発話が履歴にある")
```

- API キーをノートブックに直接書かない。共有や GitHub への push で漏れる。
- Gemini API には無料枠があり、Flash 系のモデルは料金なしで使える（2026 年時点）。ただし、1 分あたり・1 日あたりのリクエスト数に上限がある。上限は AI Studio の画面でプロジェクトごとに確認できる。
- 無料枠で送った入力と出力は、Google のモデルの改善に使われることがある。個人情報や機密の文書は送らない（有料枠に切り替えると使われなくなる）。
- 最近の Gemini は答える前に考える（思考）ため、思考のトークンも消費する。単純な処理なら `thinking_config=types.ThinkingConfig(thinking_budget=0)` で思考を止めると速くなる（止められないモデルもある）。
- `response.text` が `None` になることがある（安全フィルタで止められた、出力の上限に達した、など）。`response.candidates[0].finish_reason` で理由を確かめる。

## Gemini の応答をストリーミングで受け取る {gemini_streaming key norun}

長い応答を、生成されたそばから表示したい。

```python
!pip install -q google-genai
```

```python
from google import genai
from google.colab import userdata

client = genai.Client(api_key=userdata.get("GOOGLE_API_KEY"))
MODEL = "gemini-flash-latest"

for chunk in client.models.generate_content_stream(model=MODEL, contents="日本語の形態素解析の歴史を 300 字程度で説明してください。"):
    print(chunk.text or "", end="", flush=True)
```

```python
# 最後のチャンクに使用量が入る。全文も自分でつなげて保存しておく
pieces, last = [], None
for chunk in client.models.generate_content_stream(model=MODEL, contents="「吾輩は猫である」の冒頭を現代風に 3 文で言い換えてください。"):
    pieces.append(chunk.text or "")
    last = chunk
print("".join(pieces))
print("合計トークン:", last.usage_metadata.total_token_count)
```

- 生成に時間がかかる長い出力では、ストリーミングにすると最初の文字がすぐに表示され、待ち時間を短く感じる。チャット画面を作るときは必須。
- 思考 (thinking) をするモデルでは、思考が終わるまで最初のチャンクが届かない。
- 対話 (`client.chats`) でも `chat.send_message_stream()` で同じようにストリーミングできる。

## Gemini に決まった形式（JSON スキーマ）で出力させる {gemini_structured key norun}

LLM の出力を、決まった項目を持つ JSON として確実に受け取りたい。テキストから情報を抜き出して、そのままプログラムで使いたい。

```python
!pip install -q google-genai pydantic
```

```python
from typing import Literal
from google import genai
from google.colab import userdata
from google.genai import types
from pydantic import BaseModel

client = genai.Client(api_key=userdata.get("GOOGLE_API_KEY"))
MODEL = "gemini-flash-latest"

class Entity(BaseModel):
    text: str
    label: Literal["人名", "組織名", "地名", "日付"]

class Extraction(BaseModel):
    entities: list[Entity]
    summary: str

text = "2024 年 4 月、山田花子氏は東京大学を離れ、京都にある理化学研究所の新しいチームに加わった。"
response = client.models.generate_content(
    model=MODEL,
    contents=f"次の文から固有表現を抽出し、1 文で要約してください。\n\n{text}",
    config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=Extraction),
)
result: Extraction = response.parsed          # スキーマに沿って検証済みの Pydantic オブジェクト
for e in result.entities:
    print(e.label, e.text, "(原文にある)" if e.text in text else "(原文にない!)")
print(result.summary)
```

```python
# 分類にも使える: 選択肢を Literal で限定すると、それ以外の値は出力されない
class Sentiment(BaseModel):
    label: Literal["ポジティブ", "ネガティブ", "中立"]
    reason: str

for review in ["画面がきれいで操作も軽快。買ってよかった。", "届いてすぐ壊れた。サポートの対応も遅い。"]:
    r = client.models.generate_content(model=MODEL, contents=f"次のレビューの感情を判定してください。\n\n{review}",
                                       config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=Sentiment))
    print(r.parsed.label, "|", r.parsed.reason)
```

- `response_mime_type="application/json"` と `response_schema` を指定すると、スキーマに従う JSON だけが生成されるように制約される。`response.parsed` で Pydantic のオブジェクトとして受け取れる。
- 抽出した文字列が原文に実在するかは保証されない。`e.text in text` で必ず検証する。文字位置が必要なら、自分で `text.find()` する。
- スキーマで表せない制約（値の範囲など）は、Pydantic の検証で確認する。`response.parsed` が `None` のときは、`response.text` を見て原因を調べる。
- スキーマの項目の説明 (`Field(description=...)`) も、モデルへの指示として使われる。

## Gemini API のトークン数と使用量を確かめる {gemini_count_tokens key norun}

送る前にトークン数を数えて、コンテキスト長や無料枠の上限に収まるかを確かめたい。日本語と英語でトークン数がどれくらい違うかも知りたい。

```python
!pip install -q google-genai
```

```python
from google import genai
from google.colab import userdata

client = genai.Client(api_key=userdata.get("GOOGLE_API_KEY"))
MODEL = "gemini-flash-latest"

samples = {
    "日本語": "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。",
    "英語": "I am a cat. As yet I have no name. I have no idea where I was born.",
}
for lang, s in samples.items():
    n = client.models.count_tokens(model=MODEL, contents=s).total_tokens
    print(f"{lang}: {len(s)} 文字 → {n} トークン ({len(s) / n:.2f} 文字/トークン)")
```

```python
# 実際に生成したときの使用量 (入力・出力・思考) を記録する
response = client.models.generate_content(model=MODEL, contents="上の 2 つの文の違いを 1 文で説明してください。\n" + "\n".join(samples.values()))
u = response.usage_metadata
print({"入力": u.prompt_token_count, "出力": u.candidates_token_count, "思考": u.thoughts_token_count, "合計": u.total_token_count})

# 大量に処理する前に、数件で 1 件あたりのトークン数を測り、全体の量と処理時間を見積もる
n_docs, per_doc = 10_000, u.total_token_count
print(f"{n_docs} 件なら約 {n_docs * per_doc / 1e6:.1f}M トークン")
```

- トークン数はモデルのトークナイザで決まる。他社のトークナイザ（tiktoken など）での推定は当てにならないので、使うモデルで数える。
- 無料枠では料金はかからないが、1 分あたりのトークン数やリクエスト数の上限がある。大量の文書を処理する前に、件数 × 1 件あたりのトークン数で、上限に収まるかを見積もる。
- 思考 (thinking) をするモデルでは、出力とは別に思考のトークンも消費する。単純な処理なら思考を止める (`thinking_budget=0`)。
- トークン数のカウント自体は無料で、生成のリクエスト数にも数えられない（レート制限はある）。

## Gemini で大量のテキストをまとめて処理する (Batch API) {gemini_batch key norun}

数百〜数千件のテキストを LLM で処理したい。1 件ずつリクエストを送る代わりに、まとめて送って、後から結果を受け取りたい。

```python
!pip install -q google-genai
```

```python
import time
from google import genai
from google.colab import userdata

client = genai.Client(api_key=userdata.get("GOOGLE_API_KEY"))
MODEL = "gemini-flash-latest"

reviews = ["画面がきれいで操作も軽快。買ってよかった。", "届いてすぐ壊れた。サポートの対応も遅い。", "値段相応。可もなく不可もなく。",
           "思っていたより小さかったが、使いやすい。", "説明書が分かりにくく、設定に苦労した。"]
requests = [{"contents": [{"role": "user", "parts": [{"text": f"次のレビューを「ポジティブ」「ネガティブ」「中立」のどれかに分類し、ラベルだけを出力してください。\n\n{r}"}]}],
             "metadata": {"key": str(i)}}                  # 結果と入力を対応付けるための番号
            for i, r in enumerate(reviews)]
job = client.batches.create(model=MODEL, src=requests, config={"display_name": "review-sentiment"})
print(job.name, job.state.name)          # ジョブ名を控えておけば、セッションが切れても後から結果を取得できる
```

```python
# 完了を待つ (数分〜最大 24 時間)
while True:
    job = client.batches.get(name=job.name)
    if job.state.name in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
        break
    print(job.state.name); time.sleep(30)
print(job.state.name)
```

```python
for review, r in zip(reviews, job.dest.inlined_responses):
    print((r.response.text.strip() if r.response else f"エラー: {r.error}"), "|", review)
```

- バッチは非同期で処理される代わりに、料金が通常の約半額になる。急がない大量の処理（データのラベル付け、評価、翻訳など）に向く。
- バッチ処理は無料枠では使えないことがある（有料枠の機能として提供されている時期がある）。使えない場合は、1 件ずつ `generate_content` を呼び、レート制限に当たったら待って再試行する。
- 数千件以上なら、リクエストを JSON Lines のファイルにしてアップロードする方法（`client.files.upload` で上げたファイルを `src` に渡す）を使う。
- 結果の順序に頼らず、`metadata` に付けた番号で入力と対応付けるのが安全。

## Claude API を呼び出す {claude_basic key norun}

Claude を API 経由で使いたい（有料。無料で試すなら前の Gemini のレシピ）。Colab のシークレット機能で API キーを安全に扱いたい。

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
- `response.content` はブロックのリスト（テキストのほかに思考ブロックなどが入る）。`type == "text"` のものだけを取り出す。
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

- 出力が長くなりうる場合（`max_tokens` が大きい場合）は、ストリーミングを使うのが基本。非ストリーミングだと HTTP のタイムアウトに当たることがある。
- 逐次表示が不要でも、`stream.get_final_message()` だけ使えば、タイムアウトを避けつつ通常の応答と同じ形で結果を得られる。

## Claude に決まった形式（JSON スキーマ）で出力させる {claude_structured key norun}

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
- スキーマに書けない制約（最小値・最大値など）は、Pydantic 側の検証で確認する。

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

- トークン数はモデルによって違う。必ず実際に使うモデルの ID で数える。他社のトークナイザ（tiktoken など）での推定は当てにならない。
- 出力側に思考 (thinking) のトークンも含まれて課金される点に注意。事前に数十件で実測するのが確実。
- トークン数のカウント自体は無料（レート制限はある）。

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

- 非同期で処理される代わりに、料金が通常の半額になる。ほとんどのバッチは 1 時間以内に終わる（最大 24 時間）。
- バッチ ID を控えておけば、Colab のセッションが切れても後から結果を取得できる（結果は 29 日間保持される）。
- 共通の長いシステムプロンプトがあるなら、プロンプトキャッシュと併用するとさらに安くなる。
