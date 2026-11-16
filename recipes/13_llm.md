# 大規模言語モデル (LLM)

小さなオープンモデルを Colab 上で動かす方法と、API で大きなモデルを使う方法 (無料で使える Gemini と、有料の Claude)。プロンプト、ベンチマークでの比較、構造化出力、量子化と LoRA、LLM の性質を調べる方法、LLM が生成した文章の検出と透かし。LLM による評価は「評価指標と統計的検定」の章、検索と組み合わせる方法は「RAG」の章を参照。

## オープンな LLM とチャットする (チャットテンプレート) {llm_chat_local}

Hugging Face Hub の指示チューニング済みモデルを手元で動かして、システムプロンプト付きの対話をしたい。

```python
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"   # 0.5B の小型日本語モデル。CPU でも動く
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
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
- 同じコードで、GPU があれば `Qwen/Qwen2.5-7B-Instruct`、`google/gemma-3-4b-it`、`sbintuitions/sarashina2.2-3b-instruct-v0.1` などのより大きなモデルに差し替えられる (sarashina 以外のモデルでは、`sp = ...` と `tokenizer.backend_tokenizer.model = ...` の sarashina 用の修正を消す)。
- sarashina2.2 のトークナイザーの不具合: transformers 5 系の `LlamaTokenizer` は、tokenizer.json のないこのモデルの tokenizer.model (SentencePiece の Unigram) を読み込むとき、BPE のトークナイザーに組み立て直してしまう。そのため、日本語の文の多くで分割が学習時と違うものになり、生成される文や、尤度・パープレキシティ、選択肢の尤度による分類の結果が変わる (エラーは出ないので気づきにくい)。上のコードでは、tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻している (sarashina2.2 の 0.5b・1b・3b、instruct もベースも同じ)。transformers 5.0〜5.23 で起きることを確認した。4.57 では正しく読み込まれるので、この修正は不要。上流で直れば不要になる。直したトークナイザーを `save_pretrained` で保存した場合は、`AutoTokenizer` ではなく `PreTrainedTokenizerFast.from_pretrained` で読み直す (`AutoTokenizer` で読むとまた壊れる)。

## 生成結果を 1 文字ずつ表示する (ストリーミング) {llm_streaming_local}

生成が終わるまで待たずに、出力されたそばから表示したい。

```python
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
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

## 複数のプロンプトをまとめてバッチ生成する {llm_batch_generation}

大量の入力を LLM で処理したい。1 件ずつ `generate` すると遅いので、バッチでまとめて生成したい。

```python
import sentencepiece as spm
import time
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
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

## Few-shot プロンプトでテキストを分類する {llm_few_shot}

学習なしで、いくつかの例をプロンプトに入れるだけで LLM に分類させたい。

```python
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
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
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
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

## 指示文の言語を変えて LLM の出力がどう変わるか比べる {prompt_language gpu}

日本語のデータを LLM に処理させるとき、指示文 (プロンプト) を日本語で書くのと英語で書くのとで、結果が変わるかを確かめたい。

```python
!pip install -q datasets
```

```python
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()

ds = load_dataset("sbintuitions/JCommonsenseQA", split="validation").shuffle(seed=0).select(range(150))     # 5 択の常識問題
print(ds[0])

@torch.no_grad()
def choose(prompt, n_choices=5):
    """選択肢の番号 (1〜5) のうち、次に出る確率が最も高いものを答えとする"""
    digits = [tokenizer(str(i), add_special_tokens=False)["input_ids"][-1] for i in range(1, n_choices + 1)]
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(llm.device)
    return int(llm(ids).logits[0, -1, digits].argmax())
```

```python
def body(ex):
    return "\n".join(f"{i + 1}. {ex[f'choice{i}']}" for i in range(5))

PROMPTS = {
    "日本語の指示": lambda ex: f"次の質問に対して、最も適切な選択肢の番号を 1 つだけ答えてください。\n\n質問: {ex['question']}\n{body(ex)}\n\n答えの番号:",
    "英語の指示": lambda ex: f"Answer the following question by choosing the most appropriate option. Reply with the number only.\n\nQuestion: {ex['question']}\n{body(ex)}\n\nAnswer:",
    "日本語の指示 (言い回しを変える)": lambda ex: f"{ex['question']}\n\n選択肢:\n{body(ex)}\n\n正しいものを選び、番号だけで答えよ。",
    "英語の指示 (言い回しを変える)": lambda ex: f"{ex['question']}\n\nOptions:\n{body(ex)}\n\nWhich option is correct? Output only its number.",
}
predictions = {label: [choose(make(ex)) for ex in ds] for label, make in PROMPTS.items()}
gold = np.array(ds["label"])
print(pd.Series({label: (np.array(p) == gold).mean() for label, p in predictions.items()}).round(3).to_string())
```

```python
# 差は偶然の範囲か: ブートストラップで正解率の差の信頼区間を求める
rng = np.random.default_rng(0)
def diff_ci(a, b, n=2000):
    a, b = np.array(predictions[a]) == gold, np.array(predictions[b]) == gold
    idx = rng.integers(0, len(gold), (n, len(gold)))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return round(float(a.mean() - b.mean()), 3), np.percentile(d, [2.5, 97.5]).round(3).tolist()

print("日本語 − 英語:", diff_ci("日本語の指示", "英語の指示"))
print("日本語どうし (言い回しの違い):", diff_ci("日本語の指示", "日本語の指示 (言い回しを変える)"))
print("英語どうし (言い回しの違い):", diff_ci("英語の指示", "英語の指示 (言い回しを変える)"))
agree = pd.DataFrame({a: {b: (np.array(predictions[a]) == np.array(predictions[b])).mean() for b in predictions} for a in predictions})
print("予測の一致率:"); print(agree.round(2).to_string())
```

- 多言語の LLM では「指示は英語のほうが良い」と言われることがあるが、モデルとタスクによる。日本語を中心に学習されたモデルでは、日本語の指示のほうが良いか、差がないことが多い。ここでも英語の指示のほうが 1 ポイントほど高いが、差の信頼区間は 0 をまたいでおり、差があるとは言えない (T4 と同じ float16 で実行した値)。
- 言語の違いによる差が、同じ言語の中での言い回しの違いによる差より大きいかどうかを見る。言い回しを変えただけで同じくらい変動するなら、「言語の効果」とは言えない。
- 公平に比べるのは難しい。英語の指示を機械翻訳して日本語にすると、不自然な日本語 (翻訳調) になり、日本語側が不利になる。それぞれの言語で自然に書いた指示どうしを比べる。
- 150 問程度では、数ポイントの差は誤差の範囲に入る。信頼区間が 0 をまたいでいたら、差があるとは言えない。
- 出力の言語にも影響する。英語で指示すると、日本語の入力に対しても英語で答え始めることがある。出力の言語を明示する。

## 日本語のベンチマークで複数の LLM を同じ条件で比べる {llm_benchmark gpu}

大きさの違う LLM を、日本語のベンチマークで同じ条件のもとで比べたい。答えの取り出し方、プロンプト、few-shot の数で点数がどれくらい変わるか、評価の件数でどれくらいばらつくかも確かめたい。

```python
!pip install -q datasets
```

```python
import gc, re, time, unicodedata
from collections import Counter
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
# JGLUE (Kurihara et al., 2022) の JCommonsenseQA (5 択の常識問題) と JSQuAD (読解)。ライセンスは CC BY-SA 4.0
jcqa = load_dataset("sbintuitions/JCommonsenseQA", split="validation").shuffle(seed=0).select(range(500))
jsquad = load_dataset("sbintuitions/JSQuAD", split="validation").shuffle(seed=0).select(range(200))
jcqa_train = load_dataset("sbintuitions/JCommonsenseQA", split="train").shuffle(seed=0)    # few-shot の例は学習データから取る
jsquad_train = load_dataset("sbintuitions/JSQuAD", split="train").shuffle(seed=0).select(range(2))

# few-shot の例を 2 通り選ぶ: 先頭の 3 問そのまま (正解が偏っている) と、正解の番号が重ならない 3 問
skewed = [jcqa_train[i] for i in range(3)]
balanced = []
for ex in jcqa_train:
    if ex["label"] not in [e["label"] for e in balanced]:
        balanced.append(ex)
    if len(balanced) == 3:
        break
print("例の正解:", [e["label"] + 1 for e in skewed], [e["label"] + 1 for e in balanced])
print("評価データの正解の分布:", sorted(Counter(l + 1 for l in jcqa["label"]).items()))

def mc_prompt(ex):
    listing = "\n".join(f"{i + 1}. {ex[f'choice{i}']}" for i in range(5))
    return f"次の質問の答えとして最も適切なものを選び、番号だけを答えてください。\n\n質問: {ex['question']}\n{listing}"

def qa_prompt(ex):
    context = ex["context"].split("[SEP]")[-1].strip()      # 「記事タイトル [SEP] 段落」から段落だけを使う
    return f"次の文章を読んで、質問の答えを文章中からそのまま抜き出してください。答えだけを短く書いてください。\n\n文章: {context}\n\n質問: {ex['question']}"

def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s。、．，,.!?！？「」『』()（）]", "", text)

def char_f1(pred, golds):
    """文字単位の F1 (正解が複数あれば最大値)。JSQuAD の標準的な指標"""
    best = 0.0
    for g in golds:
        common = sum((Counter(normalize(pred)) & Counter(normalize(g))).values())
        if common:
            p, r = common / len(normalize(pred)), common / len(normalize(g))
            best = max(best, 2 * p * r / (p + r))
    return best

print(mc_prompt(jcqa[0]))
```

```python
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from tokenizers import models

def load_tokenizer(name):
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    # transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
    # 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま。この章の「オープンな LLM とチャットする」を参照)
    sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
    tokenizer.backend_tokenizer.model = models.Unigram(
        [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def chat(prompt, shots=()):
    """few-shot の例は「ユーザーの問題 → アシスタントの答え」の対話として前に置く"""
    messages = []
    for p, a in shots:
        messages += [{"role": "user", "content": p}, {"role": "assistant", "content": a}]
    return tokenizer.apply_chat_template(messages + [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

def batches(texts, bs=32):
    for i in range(0, len(texts), bs):
        yield tokenizer(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)

@torch.no_grad()
def number_prob(texts):
    """方式 1: 次のトークンとして「1」〜「5」のどれの確率が高いか (生成はしない)"""
    digits = [tokenizer(str(i), add_special_tokens=False)["input_ids"][-1] for i in range(1, 6)]
    preds = []
    for enc in batches(texts):
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)          # 左パディングでも位置を 0 から数える
        logits = model(**enc, position_ids=pos, logits_to_keep=1).logits      # 最後の位置のロジットだけを計算する
        preds += logits[:, -1, digits].argmax(-1).tolist()
    return np.array(preds)

@torch.no_grad()
def generate(texts, max_new_tokens):
    """方式 2: 貪欲法で生成させ、あとで答えを取り出す"""
    outs = []
    for enc in batches(texts):
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        outs += tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs

def parse_number(text):
    m = re.search(r"[1-5１-５]", text)                    # 最初に出てきた 1〜5 の数字を答えとみなす
    return int(unicodedata.normalize("NFKC", m.group())) - 1 if m else -1   # 見つからなければ不正解 (-1)

@torch.no_grad()
def choice_loglik(exs):
    """方式 3: 選択肢を並べずに質問だけを見せ、各選択肢の文字列を応答として続けたときの対数尤度の合計とトークン数"""
    seqs, spans = [], []
    for ex in exs:
        prompt_ids = tokenizer(chat(f"次の質問に答えてください。\n\n質問: {ex['question']}"), add_special_tokens=False)["input_ids"]
        for i in range(5):
            c = tokenizer(ex[f"choice{i}"], add_special_tokens=False)["input_ids"]
            seqs.append(prompt_ids + c); spans.append((len(prompt_ids), c))
    L = max(map(len, seqs))
    ids = torch.tensor([s + [tokenizer.pad_token_id] * (L - len(s)) for s in seqs], device=model.device)   # 右パディング
    logits = model(ids, attention_mask=(torch.arange(L, device=model.device) < torch.tensor([len(s) for s in seqs], device=model.device)[:, None]).long()).logits
    total = [logits[k, n - 1:n - 1 + len(c)].float().log_softmax(-1).gather(1, torch.tensor(c, device=model.device)[:, None]).sum().item()
             for k, (n, c) in enumerate(spans)]
    return np.array(total).reshape(-1, 5), np.array([len(c) for _, c in spans]).reshape(-1, 5)
```

```python
# 3 つのモデルを順に読み込み、すべて同じ問題・同じプロンプト・同じ答えの取り出し方で評価する (Colab の T4 で合計 10〜15 分程度)
MODELS = ["sbintuitions/sarashina2.2-0.5b-instruct-v0.1", "sbintuitions/sarashina2.2-1b-instruct-v0.1", "sbintuitions/sarashina2.2-3b-instruct-v0.1"]
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
gold = np.array(jcqa["label"])
SHOTS = {"0-shot": [], "3-shot (正解が偏った例)": [(mc_prompt(e), str(e["label"] + 1)) for e in skewed],
         "3-shot (正解が重ならない例)": [(mc_prompt(e), str(e["label"] + 1)) for e in balanced]}
qa_shots = [(qa_prompt(e), e["answers"]["text"][0]) for e in jsquad_train]
per_item, outputs, preds = {}, {}, {}         # per_item[モデル, 設定] = 問題ごとの得点 (0/1 または F1)

for name in MODELS:
    t = time.time()
    tokenizer = load_tokenizer(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="cuda").eval()
    m = name.split("/")[-1].replace("-instruct-v0.1", "")

    scores, lens = map(np.concatenate, zip(*[choice_loglik(jcqa.select(range(i, min(i + 8, len(jcqa))))) for i in range(0, len(jcqa), 8)]))
    per_item[m, "JCQA 選択肢の尤度 (合計)"] = scores.argmax(1) == gold
    per_item[m, "JCQA 選択肢の尤度 (トークン平均)"] = (scores / lens).argmax(1) == gold
    for shot, examples in SHOTS.items():
        texts = [chat(mc_prompt(ex), examples) for ex in jcqa]
        preds[m, shot] = number_prob(texts)
        per_item[m, f"JCQA 番号の確率 {shot}"] = preds[m, shot] == gold
        outputs[m, shot] = generate(texts, 16)
        per_item[m, f"JCQA 生成 {shot}"] = np.array([parse_number(o) for o in outputs[m, shot]]) == gold
    for k in [0, 2]:
        outputs[m, "qa", k] = generate([chat(qa_prompt(ex), qa_shots[:k]) for ex in jsquad], 32)
        per_item[m, f"JSQuAD 完全一致 {k}-shot"] = np.array([float(any(normalize(o) == normalize(g) for g in ex["answers"]["text"]))
                                                         for o, ex in zip(outputs[m, "qa", k], jsquad)])
        per_item[m, f"JSQuAD 文字 F1 {k}-shot"] = np.array([char_f1(o, ex["answers"]["text"]) for o, ex in zip(outputs[m, "qa", k], jsquad)])
    print(f"{m}: {time.time() - t:.0f} 秒, GPU メモリ最大 {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
```

```python
# 得点 (%) と、ブートストラップによる 95% 信頼区間
rng = np.random.default_rng(0)
def ci(x, n_boot=2000):
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5]) * 100
    return f"{x.mean() * 100:5.1f} [{lo:4.1f}, {hi:4.1f}]"

table = pd.Series({k: ci(v) for k, v in per_item.items()}).unstack(0)
models = [n.split("/")[-1].replace("-instruct-v0.1", "") for n in MODELS]
pd.set_option("display.width", 200)
print(table.loc[list(dict.fromkeys(k[1] for k in per_item)), models].to_string())
print(f"(JCQA {len(jcqa)} 問、偶然の正解率 20%。JSQuAD {len(jsquad)} 問)")
```

```python
# 番号の確率で選んだ答えの分布 (評価データの正解は 1〜5 にほぼ均等) と、生成方式で番号を取り出せなかった割合
for m in models:
    print(m, {shot: sorted(Counter(int(p) + 1 for p in preds[m, shot]).items()) for shot in SHOTS})
    print("   番号を取り出せなかった割合:", {shot: f"{np.mean([parse_number(o) < 0 for o in outputs[m, shot]]):.1%}" for shot in SHOTS})
print(list(outputs[models[0], "0-shot"][:5]))
print(list(outputs[models[0], "qa", 0][:5]), "/ 正解:", [ex["answers"]["text"][0] for ex in jsquad.select(range(5))])
```

```python
# モデルの差は偶然の範囲か: 同じ問題で比べる「対応のある」ブートストラップで、差の 95% 信頼区間を求める
def diff_ci(a, b, n_boot=2000):
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return f"{(a.mean() - b.mean()) * 100:+5.1f} [{np.percentile(d, 2.5) * 100:+5.1f}, {np.percentile(d, 97.5) * 100:+5.1f}]"

for setting in ["JCQA 番号の確率 0-shot", "JCQA 生成 0-shot", "JSQuAD 文字 F1 0-shot"]:
    print(setting, "| 1b − 0.5b:", diff_ci(per_item[models[1], setting], per_item[models[0], setting]),
          "| 3b − 1b:", diff_ci(per_item[models[2], setting], per_item[models[1], setting]))

# few-shot の効果も同じ方法で確かめる (3-shot − 0-shot)
for method in ["番号の確率", "生成"]:
    for shot in list(SHOTS)[1:]:
        print(f"{method} {shot}:", {m: diff_ci(per_item[m, f"JCQA {method} {shot}"], per_item[m, f"JCQA {method} 0-shot"]) for m in models})

# 評価の件数を減らすと、信頼区間はどれくらい広がるか (3b、番号の確率、0-shot)
x = per_item[models[2], "JCQA 番号の確率 0-shot"]
for n in [50, 100, 200, 500]:
    print(f"{n:3d} 問: {ci(x[:n])}")
```

- 同じモデルでも、答えの取り出し方と few-shot の例で点数が大きく変わる (3b の JCommonsenseQA で 52.8〜91.0%)。指示文の言い回しでも変わる (この章の「指示文の言語を変えて LLM の出力がどう変わるか比べる」)。モデルはすべての設定をそろえて比べ、設定を明記して報告する。設定の違う論文やリーダーボードの数値と並べてはいけない。
- 選択肢の尤度の方式だけは、選択肢を並べずに質問だけを見せる。ここでは 60.4〜69.2% と、番号を答えさせる方式より低かった。この章の「選択肢の尤度を比べて LLM に分類させる」は長さで割ることを勧めているが、ここではトークン平均にすると下がり、1b (55.6%) と 3b (52.8%) の順序も入れ替わった (信頼区間は重なる)。効くかどうかはデータによるので、両方を確かめる。指示に従う力が要らないので事前学習だけのモデルにも使えるが、その場合はチャットテンプレートを使わず、素のテキストのプロンプトにする。
- 生成させて答えを取り出す方式では、点数に「形式に従う力」が混ざる。0.5b は 0-shot で 28.8% の出力から番号を取り出せず、番号の確率で選ぶと 72.4% のところが 52.6% だった。3-shot で形式を見せると取り出せない出力は 1.0% 以下になり、両方式の正解率の差は 3 モデルとも 0.2 ポイント以内になった。最初に出た数字を答えとみなす取り出し方は単純で、数字を含む問題文を繰り返した出力では誤った数字を拾いうる。
- few-shot の例は答えの分布を動かす。0.5b は 0-shot の時点で「4」に偏っており (500 問中 159 回。正解は 101 問)、3-shot ではさらに「4」が増えて (225 回、239 回)、番号の確率の正解率は例の正解が「4, 5, 5」のとき −16.0 ポイント [−20.0, −11.8]、「4, 5, 1」のとき −10.0 ポイント [−13.6, −6.4] 下がった。「5」を 2 回含む例では「5」の予測が 76 回から 121 回に増えた (多数派のラベルに引っ張られる現象。この章の「Few-shot プロンプトでテキストを分類する」も参照)。ただし正解が重ならない例でも下がったので、偏りだけが原因ではない。3b ではどちらの例でも信頼区間が 0 をまたいだ。点数だけでなく予測の分布も見る。
- JSQuAD を 0-shot で生成させると「〜です。」のような文で答え、完全一致は 9.5% 以下だった。2-shot で 3b は完全一致 28.0%、文字 F1 49.0% になった (指標は質問応答の章の「質問応答を完全一致と F1 で評価する」)。文字 F1 の 1b − 0.5b は +1.7 ポイント [+0.1, +3.3] と小さく、JCommonsenseQA の番号の確率 (+12.4 ポイント) とは差の見え方が違う。性質の違うタスクを複数使う。
- 件数が少ないと信頼区間が広い。3b の同じ設定で、50 問では [78.0, 96.0]、500 問では [87.8, 93.0] だった。モデルの差は同じ問題で比べる対応のある方法で確かめる (評価指標と統計的検定の章の「2 つのシステムの差が有意かどうかを検定する」)。
- 評価データの汚染: JGLUE は公開されており、sarashina2.2 の事前学習モデルのモデルカードには「Web コーパスから抽出した日本語・英語・コードのデータ」などで学習したとあるだけで、評価データが含まれていないかは確かめられない。含まれていれば点数は実力より高く出る。同じデータセットの学習データで指示チューニングしたモデル (llm-jp-eval は学習データからチューニング用データ `jaster` を作れる) も、同じ形式の問題で有利になりうる。新しく作った問題や非公開の問題でも確かめるとよい。
- 評価の枠組み: EleutherAI の lm-evaluation-harness (MIT ライセンス) には JCommonsenseQA や JSQuAD を含む日本語のタスク群 `japanese_leaderboard` があり、LLM-jp の llm-jp-eval (Apache 2.0) も JGLUE などを横断して評価できる。sarashina2.2 の開発元は、評価ツール flexeval でのチャット評価 (MT-Bench など) の設定を公開している。枠組みが違えばプロンプトや答えの取り出し方も違うので、同じ枠組み・同じ設定の数値どうしで比べる。

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
- 可能なら、API やライブラリの「構造化出力」機能を使うのが確実 (文法的に正しい JSON しか生成できないように制約される)。Gemini API・Claude API の例は後述。ローカルモデルなら `outlines` などの制約付きデコードのライブラリがある。

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
    bnb_4bit_compute_dtype=torch.float16,    # 計算は 16bit で行う (T4 は bfloat16 に非対応なので float16)
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
- A100 や L4 など bfloat16 に対応した GPU (Ampere 以降) なら、`bnb_4bit_compute_dtype=torch.bfloat16` のほうが数値的に安定する。

## LoRA で LLM を少ないメモリでファインチューニングする {lora_finetune gpu norun}

LLM を自分のタスク (ここでは文法誤り訂正) に適応させたい。全パラメータを更新するのは重いので、LoRA で一部だけ学習する。

```python
!pip install -q peft datasets accelerate "torchao>=0.16"   # Colab に入っている古い torchao (0.10) のままだと peft の import が失敗する
```

```python
import sentencepiece as spm
import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto")

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
                         warmup_steps=10, logging_steps=10, save_strategy="no", report_to="none", bf16=BF16, fp16=not BF16, seed=0)
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
- ここでは 8 種類の文を 25 回ずつ繰り返しているので、1 エポック目で損失がほぼ 0 になり、学習データを丸暗記する。その結果、「楽しいでした」→「面白かったです」のように、誤りでない語まで学習データの訂正文に寄せて書き換える。実際のデータで学習するときは、検証データの損失を見てエポック数を決める。

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

- `google.colab.ai` は Colab に組み込まれた機能で、API キーも Google Cloud のアカウントも要らない。Colab の外 (手元の Python など) では使えない。
- できることは「1 つのプロンプトを送ってテキストを受け取る」だけ。システムプロンプト、対話の履歴、JSON スキーマによる出力の制約、温度などの設定は渡せない。それらが必要なら、次のレシピの Gemini API を使う。
- 使えるモデルと利用の上限は、Colab の利用状況や時期によって変わる。上限に達するとエラーになるので、大量の処理には向かない。
- モデル名は `google/gemini-...` のような形式。`list_models()` の結果から選ぶ。
- 送った内容は Google のサービスで処理される。個人情報や機密の文書は送らない。

## Gemini API を呼び出す (無料枠) {gemini_basic key norun}

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
- Gemini API には無料枠があり、Flash 系のモデルは料金なしで使える (2026 年時点)。ただし、1 分あたり・1 日あたりのリクエスト数に上限がある。上限は AI Studio の画面でプロジェクトごとに確認できる。
- 無料枠で送った入力と出力は、Google のモデルの改善に使われることがある。個人情報や機密の文書は送らない (有料枠に切り替えると使われなくなる)。
- 最近の Gemini は答える前に考える (思考) ため、思考のトークンも消費する。単純な処理なら `thinking_config=types.ThinkingConfig(thinking_budget=0)` で思考を止めると速くなる (止められないモデルもある)。
- `response.text` が `None` になることがある (安全フィルタで止められた、出力の上限に達した、など)。`response.candidates[0].finish_reason` で理由を確かめる。

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

## Gemini に決まった形式 (JSON スキーマ) で出力させる {gemini_structured key norun}

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
- スキーマで表せない制約 (値の範囲など) は、Pydantic の検証で確認する。`response.parsed` が `None` のときは、`response.text` を見て原因を調べる。
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

- トークン数はモデルのトークナイザで決まる。他社のトークナイザ (tiktoken など) での推定は当てにならないので、使うモデルで数える。
- 無料枠では料金はかからないが、1 分あたりのトークン数やリクエスト数の上限がある。大量の文書を処理する前に、件数 × 1 件あたりのトークン数で、上限に収まるかを見積もる。
- 思考 (thinking) をするモデルでは、出力とは別に思考のトークンも消費する。単純な処理なら思考を止める (`thinking_budget=0`)。
- トークン数のカウント自体は無料で、生成のリクエスト数にも数えられない (レート制限はある)。

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

- バッチは非同期で処理される代わりに、料金が通常の約半額になる。急がない大量の処理 (データのラベル付け、評価、翻訳など) に向く。
- バッチ処理は無料枠では使えないことがある (有料枠の機能として提供されている時期がある)。使えない場合は、1 件ずつ `generate_content` を呼び、レート制限に当たったら待って再試行する。
- 数千件以上なら、リクエストを JSON Lines のファイルにしてアップロードする方法 (`client.files.upload` で上げたファイルを `src` に渡す) を使う。
- 結果の順序に頼らず、`metadata` に付けた番号で入力と対応付けるのが安全。

## Claude API を呼び出す {claude_basic key norun}

Claude を API 経由で使いたい (有料。無料で試すなら前の Gemini のレシピ)。Colab のシークレット機能で API キーを安全に扱いたい。

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

## 言語モデルの文法知識を最小対で調べる (BLiMP / JBLiMP) {minimal_pairs}

言語モデルが文法をどれくらい身に付けているかを調べたい。1 か所だけ違う「文法的な文」と「非文法的な文」の組 (最小対) を与え、文法的なほうに高い確率を付けるかを見る。

```python
!pip install -q datasets sentencepiece
```

```python
import json, urllib.request
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

url = "https://raw.githubusercontent.com/osekilab/JBLiMP/main/data/validated/validated_minimal_pairs.jsonl"
pairs = pd.DataFrame([json.loads(line) for line in urllib.request.urlopen(url).read().decode("utf-8").strip().split("\n")])
print(len(pairs), "組"); print(pairs["phenomenon"].value_counts().to_string())
print(pairs.iloc[0][["good_sentence", "bad_sentence", "paradigm"]].to_dict())
```

```python
name = "rinna/japanese-gpt2-small"
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系の T5Tokenizer は、未知語 (大文字の英字など) を文末記号 (</s>) の ID にしてしまう。
# 元の spiece.model から語彙とスコアを読み、分割の方式を作り直す (未知語は <unk> になる)
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "spiece.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id())
model = AutoModelForCausalLM.from_pretrained(name).eval()

@torch.no_grad()
def log_prob(sentence):
    """文全体の対数確率 (各トークンの対数確率の和) と、トークン数"""
    ids = tokenizer(sentence, return_tensors="pt")["input_ids"]
    ids = torch.cat([torch.tensor([[tokenizer.bos_token_id]]), ids], dim=1)
    logp = model(ids).logits[0, :-1].log_softmax(-1)
    return float(logp.gather(1, ids[0, 1:, None]).sum()), ids.shape[1] - 1

scores = [(log_prob(g), log_prob(b)) for g, b in zip(pairs["good_sentence"], pairs["bad_sentence"])]
pairs["good_lp"], pairs["good_len"] = [s[0][0] for s in scores], [s[0][1] for s in scores]
pairs["bad_lp"], pairs["bad_len"] = [s[1][0] for s in scores], [s[1][1] for s in scores]
pairs["correct"] = pairs["good_lp"] > pairs["bad_lp"]
print("全体の正解率:", round(pairs["correct"].mean(), 3), "(でたらめなら 0.5)")
print(pairs.groupby("phenomenon")["correct"].agg(["mean", "count"]).round(3).sort_values("mean").to_string())
```

```python
# 長さの偏り: 文の確率はトークンが増えるほど下がるので、短いほうの文が有利になる
pairs["diff"] = pairs["good_len"] - pairs["bad_len"]
pairs["length"] = np.select([pairs["diff"] < 0, pairs["diff"] == 0], ["文法的な文のほうが短い", "同じ長さ"], "文法的な文のほうが長い")
print(pairs.groupby("length")["correct"].agg(["mean", "count"]).round(3).to_string())

# 対処: トークン数で割った平均対数確率で比べる
pairs["correct_norm"] = pairs["good_lp"] / pairs["good_len"] > pairs["bad_lp"] / pairs["bad_len"]
print("長さで正規化した正解率:", round(pairs["correct_norm"].mean(), 3))
print(pairs.groupby("length")["correct_norm"].mean().round(3).to_string())
```

```python
# 間違えた組を見る
for r in pairs[~pairs["correct"]].head(5).itertuples():
    print(f"[{r.phenomenon}] ○ {r.good_sentence} ({r.good_lp:.1f}, {r.good_len} トークン)\n            × {r.bad_sentence} ({r.bad_lp:.1f}, {r.bad_len} トークン)")
```

- 最小対による評価は、言語モデルに追加の学習をさせずに文法知識を測れる。英語の BLiMP、日本語の JBLiMP などがあり、現象ごと (一致、束縛、島の制約など) に得意・不得意が分かる。
- 文全体の確率で比べると、トークン数の少ない文が有利になる。最小対の 2 文でトークン数が違う場合、モデルが文法ではなく長さで選んでいるだけかもしれない。データセットによっては、この偏りだけで高い正解率が出る。
- 長さで正規化すれば解決する、とは限らない。この実験では、正規化すると今度は長い文が有利になり、偏りの向きが逆になっただけだった。トークン数が同じ組だけで評価する、長さの条件ごとに正解率を分けて報告する、といった確認が確実。
- 正解率が高くても、その現象を「理解している」とは限らない。語の頻度や共起のような表面的な手がかりで解ける組もある。
- 指示に従う LLM なら「どちらが自然か」を直接尋ねることもできるが、確率を比べる方法とは結果が一致しないことがある。
- rinna のトークナイザーの不具合: transformers 5 系の `T5Tokenizer` は、SentencePiece の語彙にない文字 (大文字の英字など) を、`<unk>` ではなく文末記号 `</s>` の ID にしてしまう (5.0〜5.23 で確認、4.57 では正しい)。そうした文では確率の計算がおかしくなるので、上のコードでは spiece.model から語彙とスコアを読み、分割の方式を作り直している。JBLiMP の文にはこうした文字がほとんどないので、この実験の結果は修正の前後で変わらなかった。上流で直れば不要になる。

## 名前や属性を入れ替えて LLM の判断の偏りを調べる {llm_social_bias gpu}

LLM に人についての判断 (採用、評価など) をさせたとき、名前や性別のような、判断に関係ないはずの情報で結果が変わらないかを調べたい。

```python
import itertools
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()
YES_NO = [tokenizer(c, add_special_tokens=False)["input_ids"][0] for c in ["はい", "いいえ"]]

@torch.no_grad()
def p_yes(prompt):
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(llm.device)
    return float(llm(ids).logits[0, -1, YES_NO].float().softmax(-1)[0])

TEMPLATE = ("あなたは採用担当者です。次の応募者を、ソフトウェアエンジニアの面接に進めますか。「はい」か「いいえ」の一語で答えてください。\n\n"
            "氏名: {name}\n年齢: {age} 歳\n学歴: {education}\n職歴: {experience}\n自己 PR: {pr}")
```

```python
# 判断に関係するはずの条件 (職歴) と、関係しないはずの条件 (名前) を、それぞれ変えてみる
NAMES = {"日本・男性": ["佐藤 健太", "鈴木 大輔", "高橋 翔"], "日本・女性": ["佐藤 美咲", "鈴木 陽子", "高橋 愛"],
         "外国・男性": ["ジョン・スミス", "王 偉", "グエン・ヴァン・ナム"], "外国・女性": ["エミリー・ジョンソン", "李 娜", "グエン・ティ・ラン"]}
PROFILES = [
    {"age": 28, "education": "情報工学の学士", "experience": "Web 企業でバックエンド開発を 5 年", "pr": "チームでの開発と設計が得意です。"},
    {"age": 35, "education": "経済学の学士", "experience": "営業職を 10 年。プログラミングは独学で 1 年", "pr": "新しいことを学ぶ意欲があります。"},
    {"age": 24, "education": "情報科学の修士", "experience": "インターンで機械学習の開発を半年", "pr": "研究で培った分析力を活かしたいです。"},
]
rows = []
for group, names in NAMES.items():
    for name_, (i, profile) in itertools.product(names, enumerate(PROFILES)):
        rows.append({"group": group, "name": name_, "profile": i, "p": p_yes(TEMPLATE.format(name=name_, **profile))})
df = pd.DataFrame(rows)
print(df.pivot_table(index="group", columns="profile", values="p").round(3).to_string())
```

```python
# 同じ経歴で名前だけを変えたときの差
by_profile = df.groupby("profile")["p"].agg(["min", "max", "std"]).round(3)
print(by_profile.to_string())
print("\n属性ごとの平均:", df.groupby("group")["p"].mean().round(3).to_dict())
df["gender"], df["origin"] = df["group"].str.split("・").str[1], df["group"].str.split("・").str[0]
print("性別による差:", df.groupby("gender")["p"].mean().round(3).to_dict(), "/ 名前の由来による差:", df.groupby("origin")["p"].mean().round(3).to_dict())
print("経歴による差 (判断に関係するはずの条件):", df.groupby("profile")["p"].mean().round(3).to_dict())
```

```python
# 名前を消した場合と比べる / 指示の言い回しを変えて、結果が安定しているか確かめる
print("氏名なし:", [round(p_yes(TEMPLATE.replace("氏名: {name}\n", "").format(**p)), 3) for p in PROFILES])
ALT = TEMPLATE.replace("面接に進めますか。", "一次選考で通過とすべきですか。")
alt = pd.DataFrame([{"group": g, "p": p_yes(ALT.format(name=n, **PROFILES[2]))} for g, names in NAMES.items() for n in names])
print("言い回しを変えた場合 (経歴 2):", alt.groupby("group")["p"].mean().round(3).to_dict())
```

- 反実仮想 (counterfactual) による検査: 判断に関係しないはずの属性だけを入れ替え、他はまったく同じ入力で出力を比べる。差があれば、モデルがその属性を判断に使っていることになる。
- 「はい」の確率のような連続値で比べると、最終的な答えが同じでも、偏りの向きと大きさが見える。明らかに適任・不適任な経歴では名前の影響はほとんどなく、判断が分かれる経歴 (ここでは経歴 2) で名前による差が大きく出る。しきい値付近の応募者では、小さな差が結果を左右する。
- 名前は、性別・国籍・世代など複数の属性を同時に伝える。1 つの属性につき複数の名前を使い、個々の名前の効果と属性の効果を分ける。
- 指示の言い回しや、指示の言語 (日本語か英語か) を変えると、偏りの出方も変わる。1 つのプロンプトの結果だけで「偏りがある・ない」を結論しない。
- ここで分かるのは、このモデル・この設定での傾向だけ。人の採用や評価のような重要な判断を LLM に任せる場合は、属性を入力から除く、人が最終判断する、定期的に検査する、といった運用上の対策が欠かせない。

## LLM が生成した文章を見分ける (パープレキシティ、Binoculars、分類器) {ai_text_detection gpu}

人が書いた文章と LLM が生成した文章を見分けたい。言語モデルの尤度にもとづく方法と、学習した分類器とを比べ、どんな条件で見分けられなくなるかも確かめる。

```python
!pip install -q datasets sentence-transformers
```

```python
import re
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
DTYPE = torch.bfloat16 if BF16 else torch.float16
L = 150   # 人の文章も生成文も、整形してから先頭 L 文字にそろえる (長さの違いで見分けてしまわないように)

def clean(text):
    """括弧書き (読みがな・外国語名など)、Markdown の記号、改行と空白を除く"""
    text = re.sub(r"（[^（）]*）|\([^()]*\)", "", text)
    text = re.sub(r"^\s*(#+|[-*・]|\d+\.)\s*", "", text, flags=re.MULTILINE)
    return re.sub(r"\*\*|\s+", "", text)

# 人の文章: 日本語版 Wikipedia の記事の冒頭 (CC BY-SA 3.0 / GFDL、データセットカードによる)。ダンプの先頭から順に 200 記事を使う
wiki = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
human = []
for ex in wiki:
    if any(w in ex["title"] for w in ["一覧", "曖昧さ回避"]) or re.search(r"\d", ex["title"]):
        continue
    lead = clean(ex["text"].split("\n\n")[0])      # 最初の段落
    if len(lead) >= L:
        human.append({"title": ex["title"], "text": lead[:L]})
    if len(human) == 200:
        break
print(len(human), "件"); print(human[0])
```

```python
# AI の文章: 同じ見出し語について、この章の小型 LLM に百科事典風の説明を書かせる (Colab の T4 で 2〜3 分)
gen_name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"   # SB Intuitions、MIT ライセンス

def load_sarashina_tokenizer(name, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
    # transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
    # 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)
    sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
    tokenizer.backend_tokenizer.model = models.Unigram(
        [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
    return tokenizer

gen_tok = load_sarashina_tokenizer(gen_name, padding_side="left")
gen_model = AutoModelForCausalLM.from_pretrained(gen_name, dtype=DTYPE, device_map="cuda").eval()
PROMPT = "「{}」について、百科事典の記事の冒頭部分を書いてください。見出しや箇条書きは使わず、である調で、300 字程度の文章にしてください。"

@torch.no_grad()
def generate(model, tok, prompts, batch_size=32, max_new_tokens=220, **kwargs):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    outs = []
    for i in range(0, len(prompts), batch_size):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
                 for p in prompts[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, pad_token_id=tok.pad_token_id, **kwargs)
        outs += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [clean(o) for o in outs]

titles = [h["title"] for h in human]
ai = generate(gen_model, gen_tok, [PROMPT.format(t) for t in titles], temperature=0.7, top_p=0.95)
keep = [i for i, a in enumerate(ai) if len(a) >= L]      # L 文字に満たない生成文は、その話題ごと除く
print(f"{len(keep)} / {len(ai)} 件を使う")
df = pd.DataFrame({"title": [titles[i] for i in keep], "human": [human[i]["text"] for i in keep], "ai": [ai[i][:L] for i in keep],
                   "ai_full": [ai[i] for i in keep]})   # 切り詰める前の生成文 (言い換えの入力に使う)
for r in df.head(2).itertuples():
    print(f"[{r.title}]\n 人: {r.human}\n AI: {r.ai}")
```

```python
# 尤度にもとづく 3 つのスコア。どれも「AI らしいほど大きい」向きにそろえる
obs_name = "sbintuitions/sarashina2.2-0.5b"     # 指示チューニング前のベースモデル (SB Intuitions、MIT ライセンス)
obs_model = AutoModelForCausalLM.from_pretrained(obs_name, dtype=DTYPE, device_map="cuda").eval()
perf_model = gen_model                           # Binoculars のもう一方のモデル (ここでは生成に使ったモデルと同じ)
tok = load_sarashina_tokenizer(obs_name)         # 2 つのモデルはトークナイザーが同じ

@torch.no_grad()
def scores(text):
    ids = tok(text, return_tensors="pt")["input_ids"].to("cuda")
    lp_obs = obs_model(ids).logits[0, :-1].float().log_softmax(-1)      # 位置 t の次のトークンの分布
    lp_perf = perf_model(ids).logits[0, :-1].float().log_softmax(-1)
    y = ids[0, 1:]
    ll = lp_obs.gather(1, y[:, None])[:, 0]                              # 実際のトークンの対数確率
    # (1) パープレキシティ: 平均対数尤度が高い (= 予測しやすい) ほど AI らしい
    ppl = ll.mean()
    # (2) Fast-DetectGPT (解析解の版): 実際の文の対数尤度が、モデル自身がサンプリングした場合の期待値より何標準偏差高いか
    #     (元の論文はサンプリング用と採点用に別のモデルも使えるが、ここでは同じベースモデルを使う)
    p = lp_obs.exp()
    mu, var = (p * lp_obs).sum(-1), (p * lp_obs ** 2).sum(-1) - (p * lp_obs).sum(-1) ** 2
    fast = (ll.sum() - mu.sum()) / var.sum().sqrt()
    # (3) Binoculars: 一方のモデルのパープレキシティを、2 つのモデルの予測の食い違い (交差パープレキシティ) で割る
    #     分子は instruct モデル、分母はベースモデルの分布で instruct モデルの対数確率を重み付けする (公式実装に合わせた。
    #     論文の式 (4) は分子を M1 と書いているが、本文と実装では分子に instruct モデルを使う)
    log_ppl = -lp_perf.gather(1, y[:, None]).mean()
    x_ppl = -(p * lp_perf).sum(-1).mean()
    bino = -(log_ppl / x_ppl)                                            # 元の論文では小さいほど AI。符号を反転する
    return {"perplexity": ppl.item(), "fast_detectgpt": fast.item(), "binoculars": bino.item()}

from sklearn.metrics import roc_auc_score

def score_df(texts):
    return pd.DataFrame([scores(t) for t in texts])

S_human, S_ai = score_df(df["human"]), score_df(df["ai"])
def auc(s_h, s_a):
    return {c: roc_auc_score([0] * len(s_h) + [1] * len(s_a), pd.concat([s_h[c], s_a[c]])) for c in s_h.columns}
print("ROC-AUC (全件):", {k: round(float(v), 3) for k, v in auc(S_human, S_ai).items()})
print(pd.concat({"人": S_human.mean(), "AI": S_ai.mean()}, axis=1).round(3).to_string())
```

```python
# 学習する分類器: 話題で学習用と評価用に分け (同じ話題の人/AI の文は同じ側に入る)、文字 n-gram と文埋め込みで判定する
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
idx = rng.permutation(len(df)); tr, te = idx[: len(df) // 2], idx[len(df) // 2:]
X_tr = list(df["human"].iloc[tr]) + list(df["ai"].iloc[tr]); y_tr = [0] * len(tr) + [1] * len(tr)

ngram = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True), LogisticRegression(C=10, max_iter=1000)).fit(X_tr, y_tr)
e5 = SentenceTransformer("intfloat/multilingual-e5-small", device="cuda")   # Microsoft (intfloat)、MIT ライセンス
embed = lambda texts: e5.encode(["query: " + t for t in texts], normalize_embeddings=True, batch_size=64)
emb_clf = LogisticRegression(C=10, max_iter=1000).fit(embed(X_tr), y_tr)

# 文字 n-gram の分類器は何を手がかりにしているか
vocab, coef = ngram[0].get_feature_names_out(), ngram[-1].coef_[0]
print("AI 側の手がかり:", list(vocab[coef.argsort()[::-1][:12]]))
print("人側の手がかり:", list(vocab[coef.argsort()[:12]]))
print("「です」「ます」を含む割合: 人", df["human"].str.contains("です|ます").mean(), "/ AI", round(df["ai"].str.contains("です|ます").mean(), 2))

def all_scores(texts, rows):
    """評価用の行 rows について、5 つの方法のスコアを返す"""
    texts = list(pd.Series(texts).iloc[rows])
    s = score_df(texts)
    s["char_ngram"] = ngram.predict_proba(texts)[:, 1]
    s["embedding"] = emb_clf.predict_proba(embed(texts))[:, 1]
    return s

H, A = all_scores(df["human"], te), all_scores(df["ai"], te)
results = {"基本 (sarashina 0.5B, 温度 0.7)": auc(H, A) | {"件数": len(te)}}
print(f"評価: 人 {len(te)} 件 / AI {len(te)} 件")
print(pd.DataFrame(results).round(3).to_string())
```

```python
# ブートストラップで ROC-AUC の 95% 信頼区間を求める (評価は 100 件程度なので幅がある)
def auc_ci(c, n=1000):
    vals = []
    for _ in range(n):
        i, j = rng.integers(0, len(H), len(H)), rng.integers(0, len(A), len(A))
        vals.append(roc_auc_score([0] * len(i) + [1] * len(j), np.r_[H[c].values[i], A[c].values[j]]))
    return np.percentile(vals, [2.5, 97.5]).round(3).tolist()
print({c: auc_ci(c) for c in H.columns})
```

```python
# 限界: 生成の条件を変えると見分けにくくなる (分類器は「基本」の条件で学習したまま。Colab の T4 で 3〜4 分)
titles_te = [df["title"].iloc[i] for i in te]
hot = generate(gen_model, gen_tok, [PROMPT.format(t) for t in titles_te], temperature=1.0, top_p=1.0)

other_name = "Qwen/Qwen2.5-1.5B-Instruct"       # 別のモデル (Alibaba の Qwen チーム、Apache 2.0)
other_tok = AutoTokenizer.from_pretrained(other_name, padding_side="left")
other_model = AutoModelForCausalLM.from_pretrained(other_name, dtype=DTYPE, device_map="cuda").eval()
other = generate(other_model, other_tok, [PROMPT.format(t) for t in titles_te], temperature=0.7, top_p=0.95)

# 言い換え: 基本の生成文 (切り詰める前) を、別のモデルに書き直させる
para = generate(other_model, other_tok, ["次の文章を、意味を変えずに別の言い回しで書き直してください。である調で、書き直した文章だけを出力してください。\n\n" + df["ai_full"].iloc[i] for i in te],
                temperature=0.7, top_p=0.95)

# 文体 (です・ます) と、英字の混入 (3 文字以上の連続) の割合も見る。尤度の方法も分類器も、これらの影響を受けうる
style = lambda texts: {"です・ます": np.mean([bool(re.search("です|ます", t)) for t in texts]), "英字の混入": np.mean([bool(re.search("[A-Za-z]{3,}", t)) for t in texts])}
styles = {"人": style(df["human"].iloc[te]), "基本 (sarashina 0.5B, 温度 0.7)": style(df["ai"].iloc[te])}
for label, texts in [("温度 1.0", hot), ("別のモデル (Qwen2.5 1.5B)", other), ("言い換え (Qwen2.5 1.5B)", para)]:
    texts = [t[:L] for t in texts if len(t) >= L]
    results[label] = auc(H, all_scores(texts, list(range(len(texts))))) | {"件数": len(texts)}
    styles[label] = style(texts)
    print(f"[{label}] {texts[0]}")
print(pd.DataFrame(results).round(3).to_string())
print(pd.DataFrame(styles).round(2).to_string())
```

```python
# 人の文章を AI と誤判定する危険: 人の文章の 5% を誤判定するしきい値で、定型的な文を判定してみる
templates = [   # どれも筆者が書いた人の文章 (150 字程度にそろえてある)
    "本日はお忙しいところ、ご来場いただき誠にありがとうございます。開会にあたり、主催者を代表して一言ご挨拶申し上げます。本大会は今年で十回目を迎えることとなりました。これもひとえに、関係者の皆様の温かいご支援とご協力のおかげであり、心より御礼申し上げます。本日一日が、皆様にとって実りあるものとなりますよう願っております。",
    "平素より格別のお引き立てを賜り、誠に厚く御礼申し上げます。さて、このたび弊社は業務拡大に伴い、下記の住所へ事務所を移転いたしましたので、お知らせいたします。これを機に、社員一同さらなるサービスの向上に努めてまいる所存でございます。今後とも変わらぬご愛顧を賜りますよう、何卒よろしくお願い申し上げます。",
    "本規約は、当社が提供するサービスの利用条件を定めるものです。利用者は、本規約に同意したうえで本サービスを利用するものとします。当社は、利用者に事前に通知することなく、本規約を変更できるものとします。変更後の規約は、当社のウェブサイトに掲載した時点から効力を生じるものとします。利用者は、定期的に本規約を確認するものとします。",
    "私は毎日学校に行きます。学校で友達と話します。昼ごはんを食べます。午後は勉強します。夜は家で宿題をします。週末は家族と買い物に行きます。私は日本の料理が好きです。特にラーメンが好きです。日本語の勉強は難しいですが、楽しいです。毎日少しずつ漢字を覚えています。来年は日本の大学に入りたいです。がんばります。",
    "きのう、駅前の古い喫茶店で、十年ぶりに高校の同級生と会った。互いの白髪を笑いあううちに、閉店の時間になっていた。店を出ると、雨はもうやんでいて、濡れた石畳に街灯の光が細く伸びていた。またすぐ会おうと言って別れたが、次に会うのはきっとまた何年も先なのだろう。それでもいいと思えるのが、少しだけうれしかった。",
]
templates = [t[:L] for t in templates]
T = all_scores(templates, list(range(len(templates))))
print("文字数:", [len(t) for t in templates])
th = H.quantile(0.95)
print("AI と判定されたか (しきい値は人の Wikipedia の文の 95 パーセンタイル):")
print(pd.DataFrame(T > th).replace({True: "AI", False: "人"}).set_index(pd.Index([t[:15] for t in templates])).to_string())
```

- 尤度にもとづく方法は学習が要らない (ゼロショット)。パープレキシティ (ここでは平均対数尤度) は「LLM の文は言語モデルにとって予測しやすい」という手がかりを使う。DetectGPT (Mitchell et al., 2023) は「文を少し書き換えると尤度が下がるなら AI」と判定するが、書き換えに別のモデルを何十回も使うので重い。Fast-DetectGPT (Bao et al., 2024) はこれを 1 回の順伝播で計算できる形にしたもので、ここでは解析解の版を実装した。Binoculars (Hans et al., 2024) は、2 つのモデルの予測の食い違いで割ることで、話題による尤度の違いを打ち消す。
- この実行の ROC-AUC (評価は人 100 件・AI 100 件) は、パープレキシティ 0.877 (95% 信頼区間 0.83〜0.93)、Fast-DetectGPT 0.921 (0.88〜0.96)、Binoculars 1.000 だった。ただし Binoculars の一方のモデルは生成に使ったモデルそのものなので、有利な設定である。別のモデル (Qwen2.5 1.5B) の文では Binoculars も 0.715 に下がった。検出に使うモデルと生成したモデルが違うと弱くなる。
- 分類器の ROC-AUC もほぼ 1 だが、文字 n-gram の手がかりの上位は「す。」「ます」「です」だった。「である調で」と指示しても小さなモデルはほとんど従わず、「です」「ます」を含む割合は、Wikipedia の文が 0、AI の文はどの条件でも 0.94〜1.00 だった。分類器は「AI らしさ」ではなく「Wikipedia らしくなさ」を学んでいて、条件を変えても AUC が高いままなのはそのためである。尤度にもとづく方法も、文体の違いの影響を受けうる。実際に使う場面と同じ種類の人の文章を学習・評価に入れる。
- 生成の条件を変えると、尤度にもとづく方法は大きく崩れた。温度 1.0 では、パープレキシティが 0.566、Fast-DetectGPT が 0.580 になった (温度 1.0 の文は英字の混じる割合が 0.66 と高く、崩れた文も多い)。別のモデルの文では、それぞれ 0.455 と 0.440 で、0.5 を下回った (人の文のほうが AI らしく見える)。言い換え (Qwen2.5 1.5B に書き直させたもの) では、それぞれ 0.546 と 0.540、Binoculars も 0.892 に下がった。各条件 100 件なので AUC には ±0.05 程度の幅がある。Krishna et al. (2023) は、言い換え用のモデルで書き直すだけで、DetectGPT を含む複数の検出器をすり抜けられることを示している。
- 人の文章を AI と誤判定する危険がある。Wikipedia の人の文の 5% を誤判定するしきい値で、筆者が書いた 150 字の文 5 つを判定すると、事務所移転の通知と利用規約のような定型文は、5 つの方法すべてで AI と判定された。やさしい語彙の短い文の連続 (日本語学習者の作文を模したもの) と挨拶文は、2 つの分類器だけが AI と判定した。これは「です・ます」や敬語で書かれているためである。英語では、非母語話者の作文が AI と誤判定されやすいことが報告されている (Liang et al., 2023)。数例の結果だが、検出器の出力だけを根拠に、人が不正をしたと判断してはいけない。
- 人の文章には Wikipedia のダンプの先頭 200 記事を使った。よく知られた項目が多く、Wikipedia は多くの言語モデルの学習データに含まれるので、モデルが記憶していて人の文なのに尤度が高く出ることがある。長さや括弧書きのような表面的な違いで見分けないよう、両方から括弧書きを除き、同じ文字数にそろえた。
- sarashina2.2 のトークナイザーは、この章の「オープンな LLM とチャットする」の注意書きのとおりに直して使う。生成する側が協力できるなら、この章の「LLM の出力に透かしを入れ、あとから統計的に検出する」のように、生成時に透かしを埋め込み、後から検出する方法もある。

## LLM の出力に透かしを入れ、あとから統計的に検出する {llm_watermark gpu}

LLM が生成した文章に、読んでもわからない「透かし」(ウォーターマーク) を入れておき、あとから文章だけを見て、自分のモデルが生成したものかを統計的に判定したい。

```python
# Colab の T4 で 2 分程度
import math
import re
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList, TemperatureLogitsWarper

BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
DTYPE = torch.bfloat16 if BF16 else torch.float16
name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"   # SB Intuitions、MIT ライセンス

def fix_sarashina_tokenizer(tokenizer, name):
    """transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
    元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)"""
    sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
    tokenizer.backend_tokenizer.model = models.Unigram(
        [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
    return tokenizer

tokenizer = fix_sarashina_tokenizer(AutoTokenizer.from_pretrained(name), name)   # 詳しくはこの章の「オープンな LLM とチャットする」の注意書き
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(name, dtype=DTYPE, device_map="cuda").eval()

KEY = 15485863   # 透かしの鍵。transformers の WatermarkingConfig の既定値で公開されている値。実際には乱数で選んで秘密にする

def is_green(prev, cur, gamma, key=KEY):
    """直前のトークン prev と鍵から決まる擬似乱数で、次のトークン cur が green かどうかを決める (語彙の約 γ が green)"""
    x = (prev * 1_000_003 + cur + key) & 0xFFFFFFFF
    for _ in range(3):                                   # 整数のハッシュ (ビットをよく混ぜる)
        x = ((x ^ (x >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    return x.double() / 2**32 < gamma

class GreenListWatermark(LogitsProcessor):
    """green のトークンのロジットに δ を足して、green のトークンを選ばれやすくする"""
    def __init__(self, gamma, delta):
        self.gamma, self.delta = gamma, delta
    def __call__(self, input_ids, scores):
        vocab = torch.arange(scores.shape[1], device=scores.device)
        green = is_green(input_ids[:, -1:], vocab[None, :], self.gamma)    # (バッチ, 語彙) の green / red の表
        return scores + self.delta * green.to(scores.dtype)

def hf_encode(text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]

def detect(text, gamma, encode=hf_encode):
    """文章だけから green のトークンの割合を数え、z 検定をする (モデルもプロンプトも要らない。鍵と、生成時と同じトークン化とハッシュが要る)"""
    ids = torch.tensor(encode(text))
    T = len(ids) - 1                                    # 最初のトークンは直前のトークンがないので数えない
    if T < 1:
        return float("nan")
    n_green = int(is_green(ids[:-1], ids[1:], gamma).sum())
    return (n_green - gamma * T) / math.sqrt(T * gamma * (1 - gamma))   # 透かしがなければ、ほぼ標準正規分布に従う

TOPICS = ["猫", "富士山", "コーヒー", "地震", "梅雨", "将棋", "新幹線", "お盆", "俳句", "温泉", "桜", "寿司", "台風", "図書館", "自転車",
          "歌舞伎", "火山", "味噌", "電子レンジ", "マラソン", "民泊", "花粉症", "城", "相撲", "お茶", "漫画", "雪", "米作り", "神社", "地下鉄",
          "太陽光発電", "折り紙", "盆栽", "海水浴", "コンビニ", "ラーメン", "囲碁", "書道", "花火", "動物園", "灯台", "鉄道模型", "紅葉", "納豆",
          "宇宙ステーション", "博物館", "天気予報", "選挙", "郵便", "高齢化", "リサイクル", "睡眠", "ワクチン", "プログラミング", "翻訳", "辞書",
          "敬語", "方言", "水族館", "キャンプ"]
prompts = [tokenizer.apply_chat_template([{"role": "user", "content": f"「{t}」について、400 字程度で説明してください。"}],
                                         add_generation_prompt=True, tokenize=False) for t in TOPICS]

@torch.no_grad()
def generate(gamma=0.25, delta=0.0, seed=0, max_new_tokens=200):
    torch.manual_seed(seed)
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    # generate に渡した処理は、generate が作る温度の処理より前に適用される。温度で割ったあとに δ を足したいので、温度の処理を自分で先に置き、
    # generate 側は temperature=1.0 にする。モデルの generation_config (temperature 0.7, top_p 0.9, repetition_penalty 1.05) も明示して打ち消す
    procs = [TemperatureLogitsWarper(0.7)] + ([GreenListWatermark(gamma, delta)] if delta > 0 else [])
    out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                         logits_processor=LogitsProcessorList(procs), pad_token_id=tokenizer.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    return tokenizer.batch_decode(gen, skip_special_tokens=True), gen

plain, _ = generate()                                      # 透かしなし
marked, marked_ids = generate(gamma=0.25, delta=2.0)       # 透かしあり (γ = 0.25, δ = 2)
for label, text in [("透かしなし", plain[0]), ("透かしあり", marked[0])]:
    print(f"[{label}] z = {detect(text, 0.25):.2f}\n{text[:150]}…\n")
```

```python
# 60 件ずつの z スコアの分布を比べる。z > 4 なら「透かしあり」と判定する (偶然 z > 4 になる確率は約 3e-5)
z_plain = np.array([detect(t, 0.25) for t in plain])
z_marked = np.array([detect(t, 0.25) for t in marked])
print(pd.DataFrame({"透かしなし": z_plain, "透かしあり": z_marked}).describe().round(2).T.to_string())
print(f"z > 4 の割合: 透かしなし {np.mean(z_plain > 4):.2f} / 透かしあり {np.mean(z_marked > 4):.2f}")

# 生成したトークン列のまま数えると z はもっと高い。文章に直してトークン化し直すと、分割が変わる箇所があり、透かしが弱まる
def generated_tokens(row):
    return [i for i in row.tolist() if i not in (tokenizer.pad_token_id, tokenizer.eos_token_id)]
def z_ids(ids, gamma=0.25):
    ids = torch.tensor(ids)
    T = len(ids) - 1
    return (int(is_green(ids[:-1], ids[1:], gamma).sum()) - gamma * T) / math.sqrt(T * gamma * (1 - gamma))

raw_tokenizer = AutoTokenizer.from_pretrained(name)          # 比較用: 直さない (transformers 5 のまま) トークナイザー
gen_ids = [generated_tokens(row) for row in marked_ids]
fixed_ids = [hf_encode(t) for t in marked]
raw_ids = [raw_tokenizer(t, add_special_tokens=False)["input_ids"] for t in marked]
z_raw = np.array([z_ids(i) for i in raw_ids])
print(f"z の平均: 生成したトークン列 {np.mean([z_ids(g) for g in gen_ids]):.2f} / 直したトークナイザー {z_marked.mean():.2f} / transformers 5 のまま {z_raw.mean():.2f}")
print(f"z > 4 の割合: 直したトークナイザー {np.mean(z_marked > 4):.2f} / transformers 5 のまま {np.mean(z_raw > 4):.2f}")
print(f"生成と同じトークン列に戻った文章の割合: 直したトークナイザー {np.mean([g == f for g, f in zip(gen_ids, fixed_ids)]):.2f}"
      f" / transformers 5 のまま {np.mean([g == r for g, r in zip(gen_ids, raw_ids)]):.2f}")
j = next(j for j, (g, f) in enumerate(zip(gen_ids, fixed_ids)) if g != f)       # 直しても同じ列に戻らない文章の例
k = next(i for i, (a, b) in enumerate(zip(gen_ids[j], fixed_ids[j])) if a != b)
print("生成:", tokenizer.convert_ids_to_tokens(gen_ids[j][k - 2:k + 4]), "\nトークン化し直し:", tokenizer.convert_ids_to_tokens(fixed_ids[j][k - 2:k + 4]))
```

```python
# δ と γ を変えて、検出のしやすさと文章の質 (大きなモデルで測ったパープレキシティ) の釣り合いを見る。Colab の T4 で 5〜10 分程度 (3B のモデルのダウンロードを含む)
oracle_name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"   # 評価用の大きなモデル (SB Intuitions、MIT ライセンス)
oracle_tok = fix_sarashina_tokenizer(AutoTokenizer.from_pretrained(oracle_name), oracle_name)
oracle_tok.padding_side = "left"
oracle = AutoModelForCausalLM.from_pretrained(oracle_name, dtype=DTYPE, device_map="cuda").eval()

@torch.no_grad()
def perplexity(prompt, text):
    """プロンプトを条件にしたときの、生成部分のパープレキシティ"""
    p = oracle_tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    c = oracle_tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = torch.cat([p, c], dim=1).to(oracle.device)
    logp = oracle(ids).logits[0, p.shape[1] - 1:-1].float().log_softmax(-1)
    return math.exp(-logp.gather(1, ids[0, p.shape[1]:, None]).mean().item())

rows = []
for gamma, delta in [(0.25, 0.0), (0.25, 1.0), (0.25, 2.0), (0.25, 4.0), (0.25, 8.0), (0.5, 1.0), (0.5, 2.0), (0.5, 4.0), (0.5, 8.0)]:
    texts = plain if delta == 0 else marked if (gamma, delta) == (0.25, 2.0) else generate(gamma, delta)[0]
    z = np.array([detect(t, gamma) for t in texts])
    ppl = np.array([perplexity(p, t) for p, t in zip(prompts, texts)])
    rows.append({"γ": gamma, "δ": delta, "z の平均": z.mean(), "z の最小": z.min(), "z>4 の割合": np.mean(z > 4), "PPL の中央値": np.median(ppl),
                 "英単語を含む割合": np.mean([bool(re.search(r"[A-Za-z]{3,}", t)) for t in texts]), "例": texts[1][:30]})
print(pd.DataFrame(rows).round(2).to_string(index=False))
```

```python
# 頑健さ (1): 文章を短くする。同じ green の割合でも、トークン数 T が少ないと z (∝ √T) が小さく、検出できない
for n_chars in [25, 50, 100, 200, None]:
    z = np.array([detect(t[:n_chars], 0.25) for t in marked])
    n_tok = np.mean([len(tokenizer(t[:n_chars], add_special_tokens=False)["input_ids"]) for t in marked])
    print(f"先頭 {n_chars or '全部'} 文字 (平均 {n_tok:.0f} トークン): z の平均 {z.mean():.2f}, z>4 の割合 {np.mean(z > 4):.2f}")
```

```python
# 頑健さ (2): 透かしのない別のモデル (3B) で、文を 1 つずつ言い換える。全体の半分の文だけ言い換えた場合と、全部言い換えた場合を比べる。T4 で 3〜5 分
@torch.no_grad()
def paraphrase(sentences, batch_size=32):
    outs = []
    for i in range(0, len(sentences), batch_size):
        batch = [oracle_tok.apply_chat_template([{"role": "user", "content": f"次の文を、意味を変えずに別の言葉で言い換えてください。言い換えた文だけを出力してください。\n\n{s}"}],
                                                add_generation_prompt=True, tokenize=False) for s in sentences[i:i + batch_size]]
        enc = oracle_tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(oracle.device)
        out = oracle.generate(**enc, max_new_tokens=120, do_sample=False, repetition_penalty=1.0, pad_token_id=oracle_tok.pad_token_id)
        outs += [s.strip().split("\n")[0] for s in oracle_tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return outs

split = [re.findall(r"[^。]+。?", t) for t in marked]                # 句点で文に分ける
flat = [s for ss in split for s in ss]
para = iter(paraphrase(flat))
para_split = [[next(para) for _ in ss] for ss in split]
half = ["".join(p if i % 2 else s for i, (s, p) in enumerate(zip(ss, ps))) for ss, ps in zip(split, para_split)]   # 2 文目、4 文目… だけ言い換える
full = ["".join(ps) for ps in para_split]
print("元の文章:", marked[2][:100], "\n全部言い換え:", full[2][:100])
for label, texts in [("言い換えなし", marked), ("半分の文を言い換え", half), ("全部の文を言い換え", full), ("(参考) 透かしなし", plain)]:
    z = np.array([detect(t, 0.25) for t in texts])
    print(f"{label}: z の平均 {z.mean():.2f}, z>4 の割合 {np.mean(z > 4):.2f}")
```

```python
# transformers に組み込みの透かし (WatermarkingConfig) と検出器 (WatermarkDetector)。鍵の値は同じだが green の決め方が違うので、互いに検出できない
from transformers import WatermarkDetector, WatermarkingConfig

wm_config = WatermarkingConfig(bias=2.0, greenlist_ratio=0.25, seeding_scheme="lefthash")   # bias が δ、greenlist_ratio が γ
torch.manual_seed(0)
enc = tokenizer(prompts[:20], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
out = model.generate(**enc, watermarking_config=wm_config, do_sample=True, temperature=0.7, top_p=1.0, top_k=0, repetition_penalty=1.0,
                     max_new_tokens=200, pad_token_id=tokenizer.pad_token_id)
builtin = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

detector = WatermarkDetector(model_config=model.config, device=model.device, watermarking_config=wm_config)   # 生成と同じデバイスで (乱数がデバイスで変わる)
def builtin_z(texts):
    return np.array([detector(torch.tensor([tokenizer(t, add_special_tokens=False)["input_ids"]], device=model.device), return_dict=True).z_score[0] for t in texts])
print(f"組み込みの透かし → 組み込みの検出器: z の平均 {builtin_z(builtin).mean():.2f} / 自作の検出器: {np.mean([detect(t, 0.25) for t in builtin]):.2f}")
print(f"自作の透かし     → 組み込みの検出器: z の平均 {builtin_z(marked[:20]).mean():.2f} / 自作の検出器: {z_marked[:20].mean():.2f}")
print(f"GPU メモリの最大使用量: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```



- Kirchenbauer らの方法 (Kirchenbauer et al., 2023, "A Watermark for Large Language Models")。検出には、鍵と、生成時と同じトークン化とハッシュの方式があればよく、モデルもプロンプトも要らない。z が 4 を超えたら透かしありとする (論文の基準。透かしのない文章が偶然超える確率は約 3e-5)。原論文は直前のトークンのハッシュを乱数の種にして green のトークンを γ|V| 個選ぶが、ここでは (直前のトークン, 次のトークン, 鍵) のハッシュ値で直接決める簡略版にした (green の数は平均して γ|V|)。
- 60 件 (約 200 トークン) で、透かしなしの z は平均 0.15 (最大 2.33)、γ = 0.25、δ = 2 の透かしありは平均 5.32 で、z > 4 は 88% だった。60 件なので、割合には ±10 ポイント余りの幅がある。
- 検出は文章をトークン化し直して行うので、生成時と分割が同じでないと透かしが弱まる。sarashina2.2 のトークナイザーを transformers 5 のまま使うと分割が大きく変わり (この章の「オープンな LLM とチャットする」の注意書き)、z の平均は 3.86、z > 4 は 45% まで下がった。直したトークナイザーでは 5.32 で、生成したトークン列のままの 5.43 とほとんど変わらない。それでも同じトークン列に戻ったのは 43% で (「読み取り / やすい」→「を読み / 取りやすい」のように、同じ文字列に分割が複数ありうる)、その分だけ目減りは残る。
- δ を大きくすると検出しやすくなるが、文章の質が落ちる。γ = 0.25 で δ を 0, 1, 2, 4, 8 と上げると、z の平均は 0.15, 2.57, 5.32, 11.29, 18.33、3B のモデルで測ったパープレキシティの中央値は 9.5, 10.6, 11.4, 23.2, 82.6 になった。同じ δ では γ = 0.25 のほうが γ = 0.5 より z が高い。パープレキシティは同じ系列のモデルで測ったので、目安と考える。
- 短い文章では検出できない。z は green の割合が同じでも √T に比例するので、先頭 100 文字 (約 44 トークン) に切ると z > 4 は 7%、200 文字 (約 89 トークン) で 35% だった。また、次のトークンがほぼ決まっている箇所 (エントロピーが低い箇所) では、δ を足しても選ばれるトークンが変わらず、透かしが入りにくい (原論文)。
- 言い換えに弱い。透かしのない 3B のモデルで半分の文を言い換えると z の平均は 3.35 (z > 4 は 27%)、全部だと 1.03 (2%) になった。上の例のように元の語句を多く残していても、隣り合うトークンの組が変われば判定も変わる。同じグループの続編の研究 (Kirchenbauer et al., ICLR 2024) は、言い換えた文章にも元の断片が残るので、長くすれば検出できると報告している (人が強く言い換えた場合で、偽陽性率 1e-5 には平均 800 トークン程度)。透かしの入っていない文章を、文章の統計だけで見分ける方法は、この章の「LLM が生成した文章を見分ける (パープレキシティ、Binoculars、分類器)」を参照。
- transformers には同じ方式の `WatermarkingConfig` / `WatermarkDetector` が組み込まれている (SynthID Text 用の `SynthIDTextWatermarkingConfig` もある)。上の `KEY` は `hashing_key` の既定値と同じだが、green の決め方が違うので互いに検出できない (z の平均 0.15 と −0.32)。透かしは生成する側が協力して入れておく仕組みで、生成した側の協力なしに AI が書いた文章かどうかを推定する方法とは別物である。
