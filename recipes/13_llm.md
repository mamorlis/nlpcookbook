# 大規模言語モデル (LLM)

小さなオープンモデルを Colab 上で動かす。チャットとバッチ生成、vLLM による高速な生成、量子化、プロンプトと推論の工夫 (Chain-of-Thought)、構造化出力と制約付きデコーディング、生成した広告文のプログラムによる検証、LoRA・DPO・GRPO による調整。API で大きなモデルを使う方法は「LLM の API（Gemini と Claude）」の章、LLM の比較と性質の調べ方は「LLM の評価と分析」の章、LLM による評価は「評価指標と統計的検定」の章、検索と組み合わせる方法は「RAG」の章を参照。

## オープンな LLM とチャットする（チャットテンプレート） {llm_chat_local}

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

- 0.5B のモデルは日本語としては流暢だが、内容は不正確なことが多い（上の説明も文法的には誤りを含む）。動作確認用と割り切り、実用には大きなモデルを使う。
- モデルごとに特殊トークンの形式が違う。自分で文字列を組み立てず、必ず `apply_chat_template` を使う。
- `add_generation_prompt=True` で「ここからアシスタントの発話」という印が末尾に付く。忘れるとモデルがユーザーの発話の続きを書いてしまう。
- 同じコードで、GPU があれば `Qwen/Qwen2.5-7B-Instruct`、`google/gemma-3-4b-it`、`sbintuitions/sarashina2.2-3b-instruct-v0.1` などのより大きなモデルに差し替えられる（sarashina 以外のモデルでは、`sp = ...` と `tokenizer.backend_tokenizer.model = ...` の sarashina 用の修正を消す）。
- sarashina2.2 のトークナイザーの不具合: transformers 5 系の `LlamaTokenizer` は、tokenizer.json のないこのモデルの tokenizer.model（SentencePiece の Unigram）を読み込むとき、BPE のトークナイザーに組み立て直してしまう。そのため、日本語の文の多くで分割が学習時と違うものになり、生成される文や、尤度・パープレキシティ、選択肢の尤度による分類の結果が変わる（エラーは出ないので気づきにくい）。上のコードでは、tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻している（sarashina2.2 の 0.5b・1b・3b、instruct もベースも同じ）。transformers 5.0〜5.23 で起きることを確認した。4.57 では正しく読み込まれるので、この修正は不要。上流で直れば不要になる。直したトークナイザーを `save_pretrained` で保存した場合は、`AutoTokenizer` ではなく `PreTrainedTokenizerFast.from_pretrained` で読み直す（`AutoTokenizer` で読むとまた壊れる）。

## 生成結果を 1 文字ずつ表示する（ストリーミング） {llm_streaming_local}

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
- バッチ内の長さを揃える（長さでソートする）とパディングが減って効率が良い。
- 数万件以上を本格的に処理するなら、vLLM などの推論エンジン（連続バッチ処理）を使う。この章の「vLLM で大量のプロンプトをまとめて高速に生成する」を参照。

## vLLM で大量のプロンプトをまとめて高速に生成する {vllm_generation gpu}

この章の「複数のプロンプトをまとめてバッチ生成する」よりも多くの入力を、もっと速く処理したい。推論エンジンの vLLM で手元のモデルを動かし、transformers の `generate` と速度と出力を比べたい。

```python
# vLLM 0.34.0 は torch 2.14.0 を指定しているので、Colab の torch 2.14.1 が入れ替わる（Colab で 2〜4 分。pip が依存関係の衝突を表示するが、このレシピは動く）
# 新しいランタイムで、torch を import する前に実行する（すでに import していたらランタイムを再起動する）
!pip install -q vllm==0.34.0

# 依存ライブラリ (nvidia-cutlass-dsl) は .pth ファイルで import の検索パスを足すが、.pth は Python の起動時にしか読まれない。
# このままだと vLLM の起動中に「No module named 'cutlass'」で止まるので、ランタイムを再起動するか、次の 2 行で読み直す
import site
for d in site.getsitepackages() + [site.getusersitepackages()]: site.addsitedir(d)
```

```python
import gc, json, shutil, time
import pandas as pd
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
import vllm
from transformers import AutoModelForCausalLM, AutoTokenizer

print("torch", torch.__version__, "/ vLLM", vllm.__version__)
torch.manual_seed(0)
name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"   # SB Intuitions、MIT ライセンス
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
tokenizer.padding_side = "left"
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()

# この章の「複数のプロンプトをまとめてバッチ生成する」の 8 語を 16 語に増やし、4 種類の質問と組み合わせた 64 件
words = ["形態素解析", "機械翻訳", "固有表現抽出", "要約", "質問応答", "感情分析", "音声認識", "情報検索",
         "構文解析", "単語埋め込み", "言語モデル", "文書分類", "対話システム", "文字認識", "照応解析", "かな漢字変換"]
templates = ["「{}」を一文で説明してください。", "「{}」の具体的な使い道を一つ挙げてください。",
             "「{}」を小学生にもわかるように説明してください。", "「{}」の難しいところは何ですか。簡潔に答えてください。"]
conversations = [[{"role": "user", "content": t.format(w)}] for w in words for t in templates]
prompts = [tokenizer.apply_chat_template(c, add_generation_prompt=True, tokenize=False) for c in conversations]
MAX_NEW = 128
REP = 1.05   # モデルの generation_config.json の値。transformers は自動で使うが、vLLM は SamplingParams で指定しないと使わない

@torch.no_grad()
def hf_generate(batch):
    """貪欲法で生成し、生成したトークン ID の列（パディングと文末記号を除く）を返す"""
    enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, repetition_penalty=REP, pad_token_id=tokenizer.pad_token_id)
    return [[t for t in row if t not in (tokenizer.pad_token_id, tokenizer.eos_token_id)] for row in out[:, enc["input_ids"].shape[1]:].tolist()]

results = {}
def record(label, outs, seconds):
    n_tok = sum(len(o) for o in outs)
    results[label] = {"件数": len(outs), "秒": seconds, "1 件あたりの秒": seconds / len(outs), "出力トークン/秒": n_tok / seconds}

hf_generate(prompts[:2])                                 # 1 回目は準備に時間がかかるので、計測の前に動かしておく
torch.cuda.synchronize(); t = time.perf_counter()
hf_single = [hf_generate([p])[0] for p in prompts[:8]]   # 1 件ずつは時間がかかるので先頭の 8 件だけ
torch.cuda.synchronize(); record("transformers 1 件ずつ", hf_single, time.perf_counter() - t)
t = time.perf_counter()
hf_batch = hf_generate(prompts)                          # 64 件を 1 回のバッチで
torch.cuda.synchronize(); record("transformers バッチ", hf_batch, time.perf_counter() - t)
print(pd.DataFrame(results).T.round(2).to_string())
print(f"GPU メモリ最大 {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

```python
from vllm import LLM, SamplingParams
from vllm.tokenizers import get_tokenizer

def gpu_memory():
    """GPU 全体の使用量 (GB)。Colab では GPU を 1 人で使うので、このノートブックと vLLM のエンジン（子プロセス）の合計になる"""
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9

del model; gc.collect(); torch.cuda.empty_cache()        # transformers のモデルを GPU から外して、vLLM に場所を空ける

# vLLM も transformers の AutoTokenizer でトークナイザーを読むので、同じ不具合が起きる
s = "東京都千代田区の国立国会図書館に行った。"
print("vLLM が既定で読むトークナイザー:", get_tokenizer(name).encode(s, add_special_tokens=False))
print("SentencePiece（学習時と同じ分割）:", sp.encode(s))

# 直したトークナイザーを保存して vLLM に渡す。AutoTokenizer で読み直すと tokenizer_config.json の
# LlamaTokenizer として組み立て直されてまた壊れるので、tokenizer.json をそのまま使うクラスに書き換える
tok_dir = "sarashina_tokenizer"
tokenizer.save_pretrained(tok_dir)
cfg = json.load(open(f"{tok_dir}/tokenizer_config.json"))
cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
json.dump(cfg, open(f"{tok_dir}/tokenizer_config.json", "w"), ensure_ascii=False, indent=1)
shutil.copy(hf_hub_download(name, "config.json"), tok_dir)   # ないと vLLM がこのディレクトリの config.json を探して ERROR を表示する（動作は変わらない）

t = time.perf_counter()
llm = LLM(name, tokenizer=tok_dir,
          dtype="bfloat16" if BF16 else "float16",   # T4 は bfloat16 に非対応なので float16 にする
          max_model_len=2048,                        # 1 件の最大の長さ（プロンプト + 出力）。既定はモデルの上限 (8192)
          gpu_memory_utilization=0.6,                # GPU メモリ全体のうち vLLM が使う割合（モデル、CUDA グラフ、KV キャッシュ）。既定は 0.9
          seed=0)
print(f"起動 {time.perf_counter() - t:.0f} 秒、GPU メモリ {gpu_memory():.1f} GB")   # torch.compile と CUDA グラフの準備をするので時間がかかる（Colab で 2〜5 分）
print("vLLM のトークナイザーの分割が SentencePiece と一致:", llm.get_tokenizer().encode(s, add_special_tokens=False) == sp.encode(s))
```

```python
# 貪欲法 (temperature=0)。max_tokens の既定は 16 なので必ず指定する
greedy = SamplingParams(temperature=0, max_tokens=MAX_NEW, repetition_penalty=REP)
ids = lambda outs: [[t for t in o.outputs[0].token_ids if t != tokenizer.eos_token_id] for o in outs]

llm.chat(conversations[:2], greedy, use_tqdm=False)     # 1 回目は準備に時間がかかるので、計測の前に動かしておく
t = time.perf_counter()
vl_single = [ids(llm.chat([c], greedy, use_tqdm=False))[0] for c in conversations[:8]]
record("vLLM 1 件ずつ", vl_single, time.perf_counter() - t)
t = time.perf_counter()
outs = llm.chat(conversations, greedy, use_tqdm=False)  # llm.chat はチャットテンプレートを当ててから生成する
vl_batch = ids(outs)
record("vLLM まとめて", vl_batch, time.perf_counter() - t)
print(pd.DataFrame(results).T.round(2).to_string())

# 入力: llm.chat が作ったトークン列は、transformers の apply_chat_template と同じか
enc = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
print("\nプロンプトのトークン列が一致:", sum(o.prompt_token_ids == e for o, e in zip(outs, enc)), "/", len(outs))
# 出力: transformers のバッチ生成と vLLM の出力が一致するか。違う場合は何トークン目から分かれたか
def prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
same = [a == b for a, b in zip(hf_batch, vl_batch)]
split = [prefix(a, b) for a, b, s in zip(hf_batch, vl_batch, same) if not s]
print(f"出力が完全に一致: {sum(same)} / {len(same)} 件。一致しない {len(split)} 件で分かれた位置（トークン目）: {sorted(split)}")
print(f"transformers の 1 件ずつとバッチの一致: {sum(a == b for a, b in zip(hf_single, hf_batch))} / 8 件、"
      f"vLLM の 1 件ずつとまとめての一致: {sum(a == b for a, b in zip(vl_single, vl_batch))} / 8 件")
diff = [k for k, s in enumerate(same) if not s]
i = diff[0] if diff else 0
print("\n例:", conversations[i][0]["content"])
print("  transformers:", tokenizer.decode(hf_batch[i]))
print("  vLLM        :", outs[i].outputs[0].text)
```

```python
# 1 つのプロンプトから複数の候補を作る: n で本数、temperature と top_p で多様さ、seed で再現性を決める
sampling = SamplingParams(n=4, temperature=0.8, top_p=0.95, max_tokens=MAX_NEW, repetition_penalty=REP, seed=0)
llm.chat(conversations[:2], sampling, use_tqdm=False)   # 計測の前に動かしておく
t = time.perf_counter()
outs_n = llm.chat(conversations, sampling, use_tqdm=False)
seconds = time.perf_counter() - t
n_tok = sum(len(c.token_ids) for o in outs_n for c in o.outputs)
print(f"64 件 × 4 本 = {sum(len(o.outputs) for o in outs_n)} 本: {seconds:.1f} 秒、{n_tok / seconds:.0f} トークン/秒")
print("4 本がすべて違う文になったプロンプト:", sum(len({c.text for c in o.outputs}) == 4 for o in outs_n), "/", len(outs_n))
outs_n2 = llm.chat(conversations, sampling, use_tqdm=False)   # 同じ設定でもう一度
print("もう一度実行して 4 本とも同じだったプロンプト:",
      sum([c.text for c in a.outputs] == [c.text for c in b.outputs] for a, b in zip(outs_n, outs_n2)), "/", len(outs_n))
print("\n" + conversations[0][0]["content"])
for c in outs_n[0].outputs:
    print(" -", c.text.replace("\n", " ")[:80])
print(f"\nGPU メモリの使用量（このノートブックと vLLM のエンジン）: {gpu_memory():.1f} GB")
```

```python
# enforce_eager=True: CUDA グラフを作らずに 1 演算ずつ実行する。起動は速いが、生成は遅くなる
del llm; gc.collect(); torch.cuda.empty_cache()          # 前のエンジンを止めて GPU を空ける
print(f"止めたあとの GPU メモリ: {gpu_memory():.1f} GB")
t = time.perf_counter()
llm = LLM(name, tokenizer=tok_dir, dtype="bfloat16" if BF16 else "float16", max_model_len=2048,
          gpu_memory_utilization=0.6, enforce_eager=True, seed=0)
startup = time.perf_counter() - t
llm.chat(conversations[:2], greedy, use_tqdm=False)
t = time.perf_counter()
vl_eager = ids(llm.chat(conversations, greedy, use_tqdm=False))
record("vLLM まとめて (enforce_eager)", vl_eager, time.perf_counter() - t)
print(f"起動 {startup:.0f} 秒、GPU メモリ {gpu_memory():.1f} GB")
print(pd.DataFrame(results).T.round(2).to_string())
print("CUDA グラフありとの出力の一致:", sum(a == b for a, b in zip(vl_batch, vl_eager)), "/", len(vl_eager))
```

- vLLM (Kwon et al., 2023) は、KV キャッシュをブロック単位で管理する PagedAttention と、終わった系列の空きにすぐ次の入力を入れる連続バッチ処理 (continuous batching) で、多数の入力を GPU にすき間なく流す推論エンジンである（Apache 2.0 ライセンス）。T4 (compute capability 7.5) は bfloat16 と FlashAttention に対応していないので `dtype="float16"` にし、アテンションは vLLM が自動で Triton の実装 (TRITON_ATTN) を選ぶ。上の出力は float16 と TRITON_ATTN を指定して T4 より速い別の GPU で実行したもので、T4 の実機では確かめていない。
- インストール: vLLM 0.34.0 は torch の版を固定しているので、Colab の torch 2.14.1 が 2.14.0 に入れ替わり、huggingface-hub なども上がる（検証の環境で 120 秒。pip が依存関係の衝突を表示する）。新しいランタイムで最初に実行すれば再起動は要らないが、依存ライブラリが .pth ファイルで足す検索パスを `site.addsitedir` で読み直す必要がある（ないと、エンジンの起動中に `No module named 'cutlass'` で止まった）。一般に、読み込み済みの torch は入れ替えられないので、先に torch を import していたらランタイムを再起動する。
- 速度（同じ 64 件、貪欲法、最大 128 トークン）: transformers は 1 件ずつで 1 件あたり 1.68 秒、64 件のバッチで 2,662 トークン/秒、vLLM は 1 件ずつで 0.30 秒、まとめて 12,059 トークン/秒（transformers のバッチの約 4.5 倍）だった。ただし vLLM の起動に 139 秒かかった（torch.compile と CUDA グラフの準備）ので、数十件を一度だけ処理するなら transformers のバッチ生成で足りる。一般に、件数が多く、出力の長さがばらつくほど vLLM が有利になる。時間は共有の GPU で測った目安で、T4 では遅くなる。
- 出力の一致: `llm.chat` が作ったプロンプトのトークン列は 64 件すべてで `apply_chat_template` と同じだったが、どちらも貪欲法なのに生成した文が完全に一致したのは 55 / 64 件で、残りは 10〜111 トークン目で分かれた（例では「地球」と「場所」）。float16 ではアテンションの実装やバッチの組み方で末尾の桁が変わり、確率がほぼ同じ候補の順位が入れ替わるためで、同じ方式でも 1 件ずつとまとめての一致は vLLM で 6 / 8 件、transformers で 7 / 8 件だった。vLLM に切り替えたら評価をやり直す。
- トークナイザー: vLLM も transformers の `AutoTokenizer` で読むので、sarashina2.2 では同じ不具合が起きる（上の出力の「東京都千代田区の」の分割）。直したトークナイザーを保存し、`tokenizer_config.json` のクラスを `PreTrainedTokenizerFast` に書き換えて `tokenizer=` に渡すと、SentencePiece と一致した。不具合の内容は、この章の「オープンな LLM とチャットする（チャットテンプレート）」の注意書きを参照。モデルは sarashina2.2-0.5b-instruct（SB Intuitions、MIT ライセンス）。
- `SamplingParams`: `max_tokens` の既定は 16 で、指定しないと出力が途中で切れる。`SamplingParams` を渡さないときだけモデルの generation_config.json（temperature 0.7、top_p 0.9、repetition_penalty 1.05）が既定値になるので、渡すときは必要な値をすべて書く。`n=4` では 256 本を 2.6 秒（10,180 トークン/秒）で作り、64 件すべてで 4 本が別の文になった。`seed=0` で同じ呼び出しをもう一度実行すると、この実行では 64 件すべてで同じ 4 本になったが、float16 の計算の揺れで変わることもありうるので、完全な再現は前提にしない。
- GPU メモリ: vLLM は起動時に `gpu_memory_utilization`（GPU 全体に対する割合。既定は 0.9）の分を先に確保し、モデルと CUDA グラフの残りを KV キャッシュにする。ここでは 0.6（T4 で約 9.5GB）にし、transformers のモデルは先に GPU から外した。検証では 24GB の GPU でほぼ同じ量になる 0.38 にし、このノートブックと vLLM のエンジンの分だけを数えると最大 10.1 GB、`del llm` の後は 0.3 GB だった。起動時に「No available memory for the cache blocks」で止まったら、`gpu_memory_utilization` を上げるか `max_model_len` を下げる。
- `enforce_eager=True` にすると、起動は 34 秒と短くなったが、生成は 2,129 トークン/秒と既定（CUDA グラフあり）の約 6 分の 1 になった。確保するメモリは変わらず（どちらも起動の直後に 9.8 GB）、CUDA グラフの分は KV キャッシュに回る。出力の一致は 53 / 64 件だった。試行錯誤の段階など、起動の速さを優先するときに使う。

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
- A100 や L4 など bfloat16 に対応した GPU（Ampere 以降）なら、`bnb_4bit_compute_dtype=torch.bfloat16` のほうが数値的に安定する。

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

- 例があると、出力の「形式」が揃う効果が大きい（ゼロショットでは余計な説明が付きがち）。
- 例の選び方と順序で結果が変わる。ラベルの偏りがあると、その多数派ラベルに引っ張られる。
- 出力は必ず後処理する（前後の空白や句点を除く、想定外の出力を検出する）。次項のように尤度で選ばせる方法もある。

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

- 出力が必ず選択肢のどれかになり、確信度（確率）も得られる。小さなモデルでも安定する。
- モデルは特定のラベルを好む偏りを持つ。内容のない入力でのスコアを差し引く「文脈較正」(Zhao et al., 2021) で補正できる。
- トークン数の多い選択肢は不利になる。長さで割る、あるいは選択肢を「A」「B」「C」のような 1 トークンの記号にして回避する。
- API 経由のモデルでは対数尤度が取れないことが多い。その場合は構造化出力 (enum) で選択肢を制約する（LLM の API の章の「Gemini に決まった形式（JSON スキーマ）で出力させる」「Claude に決まった形式（JSON スキーマ）で出力させる」を参照）。

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

- テンプレートを別ファイル（`prompts/gec_v2.j2` など）にしてバージョン管理すると、実験の再現と比較がしやすい。
- 入力に似た例を選ぶと、固定の例より精度が上がることが多い。ただし似た例の答えをそのまま真似る傾向も出る。

## 指示文の言語を変えて LLM の出力がどう変わるか比べる {prompt_language gpu}

日本語のデータを LLM に処理させるとき、指示文（プロンプト）を日本語で書くのと英語で書くのとで、結果が変わるかを確かめたい。

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

- 多言語の LLM では「指示は英語のほうが良い」と言われることがあるが、モデルとタスクによる。日本語を中心に学習されたモデルでは、日本語の指示のほうが良いか、差がないことが多い。ここでも英語の指示のほうが 1 ポイントほど高いが、差の信頼区間は 0 をまたいでおり、差があるとは言えない（T4 と同じ float16 で実行した値）。
- 言語の違いによる差が、同じ言語の中での言い回しの違いによる差より大きいかどうかを見る。言い回しを変えただけで同じくらい変動するなら、「言語の効果」とは言えない。
- 公平に比べるのは難しい。英語の指示を機械翻訳して日本語にすると、不自然な日本語（翻訳調）になり、日本語側が不利になる。それぞれの言語で自然に書いた指示どうしを比べる。
- 150 問程度では、数ポイントの差は誤差の範囲に入る。信頼区間が 0 をまたいでいたら、差があるとは言えない。信頼区間の求め方は評価指標と統計的検定の章の「ブートストラップ法で評価値の信頼区間を求める」を参照。
- 出力の言語にも影響する。英語で指示すると、日本語の入力に対しても英語で答え始めることがある。出力の言語を明示する。

## 考える過程を書かせて LLM に算数の文章題を解かせる（Chain-of-Thought と self-consistency） {llm_reasoning gpu}

算数の文章題のように何段階かの計算が要る問題で、LLM にいきなり答えさせる場合と、途中の考えを書かせてから答えさせる場合 (Chain-of-Thought) とで、正解率がどう変わるかを確かめたい。何回か解かせて多数決をとる方法 (self-consistency) の効果と、生成するトークン数や時間のコストも測りたい。

```python
!pip install -q datasets
```

```python
import itertools, re, time, unicodedata
from collections import Counter
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
# MGSM (Shi et al., 2022): GSM8K (Cobbe et al., 2021) の小学校レベルの算数の文章題 250 問を、人手で 10 言語に訳したもの。
# 元のリポジトリ (google-research/url-nlp の mgsm/) のライセンスは CC BY 4.0、元の GSM8K は MIT ライセンス
# (Hugging Face のデータセットカードのメタデータは CC BY-SA 4.0。注意書きを参照)
N = 100                                                    # 時間の都合で 250 問のうち先頭の 100 問を使う
mgsm = load_dataset("juletxara/mgsm", "ja", split="test").select(range(N))
print(mgsm[0]["question"], "→", mgsm[0]["answer_number"])

name = "sbintuitions/sarashina2.2-1b-instruct-v0.1"       # MIT ライセンス
tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()
print(model.generation_config)      # 開発元の既定値: 温度 0.7、top_p 0.9、repetition_penalty 1.05
```

```python
# few-shot の例: MGSM に付いている、人手で訳した 8 問の例 (train) の先頭 2 問。答えの部分だけを変えて「直接」と「CoT」の 2 通りにする
# (2 問目の問題文の「難題のコンピューター」は「何台の」の誤りで、元のデータのまま使っている)
exemplars = load_dataset("juletxara/mgsm", "ja", split="train").select(range(2))
INSTRUCTION = {
    "直接": "次の問題に答えてください。答えの数値だけを出力し、途中の考えや説明は書かないでください。",
    "CoT": "次の問題に答えてください。ステップごとに考えて途中の計算を書き、最後に「答えは〇〇です。」と書いてください。",
}
def shot_answer(style, ex):
    if style == "直接":
        return f"答えは{ex['answer_number']}です。"
    return ex["answer"].removeprefix("ステップごとの答え：")      # 例に付いている推論の過程 (「…。答えは11です。」で終わる)

def messages(style, n_shots, question):
    msgs = []
    for ex in exemplars.select(range(n_shots)):     # 例は「ユーザーの問題 → アシスタントの答え」の対話として前に置く
        msgs += [{"role": "user", "content": f"{INSTRUCTION[style]}\n\n問題: {ex['question'].removeprefix('問題：')}"},
                 {"role": "assistant", "content": shot_answer(style, ex)}]
    return msgs + [{"role": "user", "content": f"{INSTRUCTION[style]}\n\n問題: {question}"}]

@torch.no_grad()
def solve(style, n_shots, max_new_tokens, n_samples=1, bs=64):
    """全問を解かせ、問題ごとに n_samples 個の出力と、生成したトークン数・時間を返す。n_samples > 1 ならサンプリング、1 なら貪欲法"""
    texts = [tokenizer.apply_chat_template(messages(style, n_shots, ex["question"]), add_generation_prompt=True, tokenize=False) for ex in mgsm]
    outs, n_tokens, t = [], [], time.time()
    step = max(1, bs // n_samples)
    for i in range(0, len(texts), step):
        enc = tokenizer(texts[i:i + step], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=n_samples > 1, num_return_sequences=n_samples,
                             pad_token_id=tokenizer.pad_token_id)[:, enc["input_ids"].shape[1]:]
        n_tokens += (out != tokenizer.pad_token_id).sum(1).tolist()     # 終了の記号 </s> までを数える
        outs += tokenizer.batch_decode(out, skip_special_tokens=True)
    return {"text": np.array(outs, dtype=object).reshape(N, n_samples), "tokens": np.array(n_tokens).reshape(N, n_samples),
            "sec": time.time() - t, "max_new_tokens": max_new_tokens}

def extract(text):
    """出力から答えの数値を取り出す: 「答え」の後ろ → \\boxed{} の中 → 最後に出てきた数値 の順に探す"""
    text = unicodedata.normalize("NFKC", text).replace(",", "")
    text = re.sub(r"(\d+(?:\.\d+)?)万(\d*)", lambda m: str(round(float(m.group(1)) * 10000) + int(m.group(2) or 0)), text)   # 「3万5000」→ 35000
    for pattern in [r"答え[はが]?\s*[:：]?[^\d\-\n]*(-?\d+(?:\.\d+)?)", r"\\boxed\{[^\d\-]*(-?\d+(?:\.\d+)?)"]:
        found = re.findall(pattern, text)
        if found:
            return float(found[-1])
    found = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(found[-1]) if found else None

gold = np.array(mgsm["answer_number"], dtype=float)
print([extract(s) for s in ["答えは65,000ドルです。", "よって \\boxed{18} です。", "合計は3万ドルです。", "3万5000円", "わかりません"]])
```

```python
# 貪欲法で 1 回ずつ解かせる。「直接」は 16 トークン、CoT は 400 トークンまで (Colab の T4 で 3〜4 分程度)
CONDITIONS = {"直接 (0-shot)": ("直接", 0, 16), "直接 (2-shot)": ("直接", 2, 16), "CoT (0-shot)": ("CoT", 0, 400), "CoT (2-shot)": ("CoT", 2, 400)}
runs = {c: solve(*args) for c, args in CONDITIONS.items()}

def summary(run, answers):
    return {"正解率": (answers == gold).mean(), "答えを取り出せなかった割合": np.isnan(answers).mean(),
            "出力のトークン数 (平均)": run["tokens"].mean(), "上限で打ち切られた割合": (run["tokens"] >= run["max_new_tokens"]).mean(),
            "1 問あたりの秒数": run["sec"] / N}

greedy = {c: np.array([extract(t) for t in r["text"][:, 0]], dtype=float) for c, r in runs.items()}   # 取り出せなければ nan (不正解)
pd.set_option("display.width", 200)
print(pd.DataFrame({c: summary(runs[c], greedy[c]) for c in runs}).round(3).to_string())
print(f"GPU メモリ最大 {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
for c in runs:
    print(f"--- {c} (正解 {gold[1]:g}):", runs[c]["text"][1, 0][-300:])
print("--- 直接 (2-shot) で数値を取り出せなかった出力:", [t for t, a in zip(runs["直接 (2-shot)"]["text"][:, 0], greedy["直接 (2-shot)"]) if np.isnan(a)][:3])
```

```python
# self-consistency: 温度 0.7 でサンプリングして 5 回ずつ解かせ、取り出した答えの多数決をとる (Colab の T4 で 6〜7 分程度)
K = 5
torch.manual_seed(0)
samples = {c: solve(*CONDITIONS[c], n_samples=K) for c in ["直接 (2-shot)", "CoT (0-shot)"]}
answers = {c: [[extract(t) for t in row] for row in r["text"]] for c, r in samples.items()}

def vote(row):
    """答えを取り出せたものだけで多数決。同数なら先に出た答え"""
    row = [a for a in row if a is not None]
    return Counter(row).most_common(1)[0][0] if row else None

def sc_accuracy(c, k):
    """K 個から k 個を選ぶすべての組み合わせで多数決をとり、正解率を平均する (どの k 個を使うかによるばらつきをならす)"""
    combos = list(itertools.combinations(range(K), k))
    return np.mean([[vote([row[j] for j in combo]) == g for combo in combos] for row, g in zip(answers[c], gold)])

table = pd.DataFrame({f"{c} の多数決": {k: sc_accuracy(c, k) for k in range(1, K + 1)} for c in samples})
table.index.name = "回数"
print(table.round(3).to_string())
print({c: f"5 回で 1 問あたり {r['tokens'].sum(1).mean():.0f} トークン、{r['sec'] / N:.2f} 秒" for c, r in samples.items()})
```

```python
# 差は偶然の範囲か: 同じ問題で比べる対応のあるブートストラップで、正解率の差の 95% 信頼区間を求める
rng = np.random.default_rng(0)
correct = {c: greedy[c] == gold for c in runs}
correct["CoT (0-shot) 5 回の多数決"] = np.array([vote(row) == g for row, g in zip(answers["CoT (0-shot)"], gold)])
# サンプリング 1 回の正解は、5 個の標本の正解の平均 (問題ごとに 0〜1 の値) にする。多数決の表の 1 回の行と同じ基準
correct["CoT (0-shot) サンプリング 1 回"] = np.array([np.mean([a == g for a in row]) for row, g in zip(answers["CoT (0-shot)"], gold)])

def diff_ci(a, b, n_boot=2000):
    x, y = correct[a], correct[b]
    idx = rng.integers(0, N, (n_boot, N))
    d = x[idx].mean(axis=1) - y[idx].mean(axis=1)
    print(f"{a} − {b}: {(x.mean() - y.mean()) * 100:+5.1f} ポイント [{np.percentile(d, 2.5) * 100:+5.1f}, {np.percentile(d, 97.5) * 100:+5.1f}]")

diff_ci("CoT (2-shot)", "直接 (2-shot)")          # 同じ 2 問の例で、答えの前に推論を見せるかどうかだけが違う
diff_ci("CoT (0-shot)", "直接 (2-shot)")
diff_ci("CoT (0-shot)", "CoT (2-shot)")
diff_ci("CoT (0-shot) 5 回の多数決", "CoT (0-shot)")
diff_ci("CoT (0-shot) 5 回の多数決", "CoT (0-shot) サンプリング 1 回")

# 票の集まり方は確信度の目安になる: 5 回のうち最も多かった答えが何票だったかごとの、多数決の正解率
top = np.array([max(Counter(a for a in row if a is not None).values(), default=0) for row in answers["CoT (0-shot)"]])
print(pd.DataFrame({"問題数": pd.Series(top).value_counts().sort_index(),
                    "多数決の正解率": pd.Series(correct["CoT (0-shot) 5 回の多数決"]).groupby(top).mean().round(3)}).rename_axis("最多票の数").to_string())
```

- Chain-of-Thought (CoT) は、答えの前に途中の推論を書かせるプロンプトの方法で、Wei et al. (2022) は推論の過程を含む例を few-shot で見せる形で提案した。「ステップごとに考えて」のような指示だけで書かせる方法は zero-shot CoT (Kojima et al., 2022) と呼ばれる。ここでは同じ 2 問の例で答えだけを見せると正解率は 7.0%、推論の過程も見せると 41.0% で、差は +34.0 ポイント [+24.0, +45.0] だった。例なしの CoT (35.0%) も、例ありの直接 (7.0%) より 28.0 ポイント [18.0, 38.0] 高かった。例ありの CoT と例なしの CoT の差（6.0 ポイント [−4.0, +17.0]）は、この件数では偶然の範囲である。
- 指示チューニング済みのモデルは「数値だけを出力」と指示しても説明から書き始め、例なしの直接では 16 トークンの中に数値が出てこない出力が 31% あった。例を 2 問見せると多くは「答えは…です。」で書き始めるようになるが、そのあとに説明を続ける（出力の全部が 16 トークンで打ち切られ、6% は数値が出てこなかった）。答えの取り出し方は、取り出せなかった出力を目で確かめて直す。ここでは「答え」の後ろ、\boxed{} の中、最後に出た数値の順に探し、「3万」「3万5000」のような書き方も数値に直している。
- 多数決 (self-consistency) は Wang et al. (2023) の方法で、サンプリングで複数の推論を生成させ、最終的な答えだけで多数決をとる。例なしの CoT で、サンプリング 1 回では 29.0%（5 回のどれを使うかで平均した値）、5 回の多数決で 36.0% になり、差は +7.0 ポイント [+2.2, +12.0] だった。ただし貪欲法 1 回 (35.0%) との差は +1.0 ポイント [−9.0, +11.0] で、このモデルでは 5 倍のコストをかけても貪欲法を上回ったとはいえない。サンプリングは 1 回ごとには貪欲法より正解率が下がり、多数決はその分を取り戻している。推論を書かせない直接の答えで多数決をとっても 6.6% → 7.0% とほとんど変わらなかった。
- 票の集まり方は確信度の目安になる。5 回とも同じ答えだった 5 問では多数決の正解率が 100%、4 票だった 12 問では 91.7% だったのに対し、最多票が 1（同じ答えが 2 回以上出なかった）の 31 問では 0% だった。票が割れた問題だけ人が確かめる、答えを保留する、といった使い方ができる。
- コスト: 出力のトークン数は、直接の 16 トークン（上限）に対して CoT は平均 171.0 トークン、5 回の多数決では 1 問あたり 930 トークンになる。検証に使った GPU（RTX A5000、1 回のバッチは 64 件、多数決では 12 問 × 5 回の 60 件）での 1 問あたりの時間は、直接 0.03 秒以下、CoT 0.22 秒、5 回の多数決 1.27 秒だった（T4 ではもっとかかる）。例ありの CoT は出力が短い（平均 154.6 トークン）のに 0.32 秒かかった。例の分だけ入力が長くなるためと考えられる（時間の内訳は測っていない）。バッチの中で最も長い出力が終わるまで待つので、時間は出力の長さのばらつきにも左右される。CoT の 1〜2% は 400 トークンの上限で打ち切られた。
- MGSM は 250 問しかなく、ここではさらに先頭の 100 問を使ったので、正解率の差の信頼区間の幅は 10〜21 ポイントあった。GSM8K と MGSM は公開から時間がたっており、sarashina2.2 の事前学習モデルのモデルカードによると、数学の性能を上げるための合成データでも学習している。評価データやその言い換えが学習データに含まれていれば、点数は実力より高く出る（LLM の評価と分析の章の「日本語のベンチマークで複数の LLM を同じ条件で比べる」の注意書きを参照）。データセットのライセンスは、元のリポジトリの mgsm/LICENSE では CC BY 4.0、Hugging Face のデータセットカードのメタデータでは CC BY-SA 4.0 となっている。
- このモデルの generation_config は、サンプリングの既定値を温度 0.7、top_p 0.9 にし、repetition_penalty 1.05 を貪欲法でも掛ける。設定を変えると点数も変わるので、比べるときはそろえる。トークナイザーの修正については、この章の「オープンな LLM とチャットする（チャットテンプレート）」の注意書きを参照。

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
- 可能なら、API やライブラリの「構造化出力」機能を使うのが確実（文法的に正しい JSON しか生成できないように制約される）。Gemini API・Claude API の例は、LLM の API の章の「Gemini に決まった形式（JSON スキーマ）で出力させる」と「Claude に決まった形式（JSON スキーマ）で出力させる」を参照。ローカルモデルなら `outlines` などの制約付きデコードのライブラリがある（この章の「制約付きデコーディングで LLM の出力を JSON スキーマ・正規表現・選択肢に従わせる (outlines)」を参照）。

## 制約付きデコーディングで LLM の出力を JSON スキーマ・正規表現・選択肢に従わせる (outlines) {constrained_decoding gpu}

手元の LLM に JSON や決まった形式で答えさせるとき、生成してから取り出すのではなく、生成の途中で形式に合わないトークンを禁止して、必ず形式どおりの出力を得たい。形式の違反がどれだけ減るか、内容は正しくなるのか、速度はどうかも確かめたい。

```python
!pip install -q outlines==1.3.3 datasets
```

```python
import json, re, time
from typing import Literal
import numpy as np
import pandas as pd
import outlines
import sentencepiece as spm
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from outlines.types import Regex
from pydantic import BaseModel, Field, ValidationError
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
name = "sbintuitions/sarashina2.2-1b-instruct-v0.1"       # SB Intuitions、MIT ライセンス
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()
om = outlines.from_transformers(model, tokenizer)        # outlines (Apache 2.0) のモデル。生成は model.generate に制約の logits_processor を足して行う

def run(output_type, prompts, max_new_tokens):
    """output_type=None なら制約なし。全件を 1 回のバッチで生成し、出力と速度を返す"""
    t = time.perf_counter()
    gen = outlines.Generator(om, output_type)           # ここで形式を受理するオートマトンを作り、語彙の各トークンと対応づける
    t_build = time.perf_counter() - t
    t = time.perf_counter()
    outs = gen.batch(prompts, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed = time.perf_counter() - t
    n_tok = [len(tokenizer(o, add_special_tokens=False)["input_ids"]) for o in outs]
    # バッチ生成のステップ数は、いちばん長い出力のトークン数で決まる。1 ステップあたりの時間で比べる
    return outs, {"準備の秒数": t_build, "1 件あたりの秒数": elapsed / len(prompts),
                  "出力のトークン数（平均）": np.mean(n_tok), "出力のトークン数（最大）": max(n_tok),
                  "1 ステップあたりのミリ秒": elapsed / max(n_tok) * 1000}

rng = np.random.default_rng(0)
def diff_ci(x, y, n_boot=2000):
    """同じ問題で比べる対応のあるブートストラップで、正解率の差 (x − y) と 95% 信頼区間を求める"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    d = x[idx].mean(1) - y[idx].mean(1)
    return f"{(x.mean() - y.mean()) * 100:+.1f} ポイント [{np.percentile(d, 2.5) * 100:+.1f}, {np.percentile(d, 97.5) * 100:+.1f}]"

chat = lambda content: tokenizer.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
```

```python
# 1. JSON スキーマ: この章の「LLM の出力から JSON を取り出して検証する」と同じ課題（レビュー → 感情・点数・観点）。
# 内容の正しさを測れるように、観点は 6 つから選ぶ形にした。レビューと正解は、このレシピのために作った 40 件
ASPECTS = ["料理", "接客", "価格", "雰囲気", "立地", "待ち時間"]
class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    score: int = Field(ge=1, le=5)
    aspects: list[Literal["料理", "接客", "価格", "雰囲気", "立地", "待ち時間"]]

P, N, U = "positive", "negative", "neutral"
data = [
    ("料理がとても美味しくて、店員さんも親切でした。また来ます。", P, "料理 接客"), ("駅から歩いてすぐで、落ち着いた雰囲気のお店でした。", P, "立地 雰囲気"),
    ("この値段でこの味なら大満足です。", P, "価格 料理"), ("注文してから5分で料理が出てきて驚きました。味も申し分ない。", P, "待ち時間 料理"),
    ("スタッフの気配りが行き届いていて、記念日を気持ちよく過ごせました。", P, "接客"), ("窓からの夜景がきれいで、デートにぴったりの雰囲気です。", P, "雰囲気"),
    ("ランチセットが千円以下で、味も量も満足。", P, "価格 料理"), ("予約なしでもすぐに案内してもらえ、店員さんの対応も丁寧でした。", P, "待ち時間 接客"),
    ("焼きたてのパンが絶品。少し高いけれど、それだけの価値はあります。", P, "料理 価格"), ("店内は清潔で明るく、子ども連れでも入りやすかった。", P, "雰囲気"),
    ("会社の近くにあるので重宝しています。日替わり定食がおいしい。", P, "立地 料理"), ("行列ができていたが、回転が速くて10分ほどで入れた。ラーメンも最高。", P, "待ち時間 料理"),
    ("苦手な食材を伝えたら、快く対応してくれました。", P, "接客"), ("静かで落ち着けるので、一人でゆっくり食事を楽しめた。", P, "雰囲気"),
    ("料金は良心的で、店主の人柄も温かい。", P, "価格 接客"),
    ("1 時間待たされた上に、注文を間違えられた。", N, "待ち時間 接客"), ("値段のわりに量が少なくてがっかり。", N, "価格 料理"),
    ("店員の態度が悪く、二度と行きたくない。", N, "接客"), ("スープがぬるく、麺も伸びていた。", N, "料理"),
    ("駅から遠いうえに、看板がなくて迷った。", N, "立地"), ("隣の席の声がうるさく、店内も狭くて落ち着かない。", N, "雰囲気"),
    ("料理が出てくるまで40分もかかった。", N, "待ち時間"), ("この味でこの値段は高すぎる。", N, "料理 価格"),
    ("テーブルがべたべたで、店内も薄暗かった。", N, "雰囲気"), ("呼んでも店員が来ず、料理も冷めていた。", N, "接客 料理"),
    ("刺身が生臭く、食べられなかった。", N, "料理"), ("会計を間違えられたのに謝罪もなかった。", N, "接客"),
    ("駐車場がなく、最寄り駅からも徒歩20分かかる。", N, "立地"), ("予約していたのに30分も待たされ、席も狭かった。", N, "待ち時間 雰囲気"),
    ("メニューの写真と違う料理が出てきて、しかも高い。", N, "料理 価格"),
    ("可もなく不可もなく、といったところ。", U, ""), ("料理は普通。値段も相場どおり。", U, "料理 価格"),
    ("料理はおいしいが、店員の態度は残念だった。", U, "料理 接客"), ("雰囲気は良いが、料理の味は平凡だった。", U, "雰囲気 料理"),
    ("駅前で便利だが、いつも混んでいて待たされる。", U, "立地 待ち時間"), ("特に印象に残らない普通の定食屋。", U, ""),
    ("安いけれど、味はそれなり。", U, "価格 料理"), ("接客は丁寧だが、料理が出てくるのが遅い。", U, "接客 待ち時間"),
    ("量は多いが、味付けが濃すぎる。", U, "料理"), ("店内はおしゃれだが、値段が高め。", U, "雰囲気 価格"),
]
reviews, gold_sent, gold_asp = [d[0] for d in data], [d[1] for d in data], [set(d[2].split()) for d in data]

def extract_json(text):
    """「LLM の出力から JSON を取り出して検証する」と同じ: 前後の説明文やコードブロックがあっても最初の JSON オブジェクトを取り出す"""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find("{")
    if start < 0:
        raise ValueError("JSON が見つかりません")
    return json.JSONDecoder().raw_decode(text[start:])[0]

def parse(text):
    try:
        return Review.model_validate(extract_json(text)), "OK"
    except json.JSONDecodeError:
        return None, "JSON として読めない"
    except ValueError as e:                     # ValidationError も ValueError の一種
        return None, "JSON がない" if "見つかりません" in str(e) else "スキーマに合わない"

SCHEMA = json.dumps(Review.model_json_schema(), ensure_ascii=False)
FULL = ("次の飲食店のレビューを分析し、JSON だけを出力してください。sentiment は感情（positive / negative / neutral）、score は 1〜5 の評価、"
        f"aspects はレビューが触れている観点を {'、'.join(ASPECTS)} から選んだリストです。\nスキーマ: {SCHEMA}\n\nレビュー: ")
SHORT = "次の飲食店のレビューを分析して、JSON で答えてください。\n\nレビュー: "    # スキーマも観点の候補も書かない短い指示

def evaluate(outs):
    parsed = [parse(o) for o in outs]
    ok = [r for r, s in parsed if r]
    tp = sum(len(set(r.aspects) & g) for (r, _), g in zip(parsed, gold_asp) if r)
    n_pred, n_gold = sum(len(set(r.aspects)) for r in ok), sum(len(g) for g in gold_asp)
    return parsed, {"形式の違反": 1 - len(ok) / len(outs),
                    "感情の正解率": np.mean([bool(r) and r.sentiment == g for (r, _), g in zip(parsed, gold_sent)]),   # 違反は不正解として数える
                    "観点の F1": 2 * tp / (n_pred + n_gold),
                    "感情と点数の矛盾": np.mean([r.sentiment == P and r.score <= 2 or r.sentiment == N and r.score >= 4 for r in ok])}

# 各条件を 40 件まとめて生成する（Colab の T4 で 1〜2 分程度）
results, parsed_all = {}, {}     # parsed_all[条件] は（出力, Review か None, 状態）のリスト
for label, prefix, output_type in [("生成して取り出す", FULL, None), ("制約あり", FULL, Review), ("制約あり（短い指示）", SHORT, Review)]:
    outs, speed = run(output_type, [chat(prefix + r) for r in reviews], max_new_tokens=150)
    parsed, metrics = evaluate(outs)
    results[label] = {**metrics, **speed}
    parsed_all[label] = [(o, r, s) for o, (r, s) in zip(outs, parsed)]
pd.set_option("display.width", 200)
print(pd.DataFrame(results).round(3).to_string())
correct = {l: [bool(r) and r.sentiment == g for (_, r, _), g in zip(parsed_all[l], gold_sent)] for l in results}
print("感情の正解率の差（制約あり − 生成して取り出す）:", diff_ci(correct["制約あり"], correct["生成して取り出す"]))
for label in results:
    print(f"--- {label}: 違反の内訳", pd.Series([s for _, _, s in parsed_all[label]]).value_counts().to_dict())
    print("    例:", [o for o, _, s in parsed_all[label] if s != "OK"][:2] or parsed_all[label][0][0])
```

```python
# 感情の誤りの中身: 形式は守れていても、内容が正しいとは限らない
rows = []
for i, (rv, g) in enumerate(zip(reviews, gold_sent)):
    answers = []
    for label in results:
        _, r, status = parsed_all[label][i]
        answers.append(r.sentiment if r else status)    # 違反なら違反の種類を表示する
    rows.append((rv, g, *answers))
table = pd.DataFrame(rows, columns=["レビュー", "正解", *results])
print(table[table["制約あり"] != table["正解"]].to_string(max_colwidth=30))
print(pd.crosstab(table["正解"], table["制約あり"]).to_string())
```

```python
# 2. 選択肢と正規表現
# 選択肢: Literal を渡すと、出力はそのどれか 1 つになる（同じ 40 件の感情だけを答えさせる）
ask = "次の飲食店のレビューの感情を positive、negative、neutral のどれか 1 語で答えてください。\n\nレビュー: "
for label, output_type in [("制約なし", None), ("選択肢", Literal["positive", "negative", "neutral"])]:
    outs, speed = run(output_type, [chat(ask + r) for r in reviews], max_new_tokens=20)
    print(f"{label}: 選択肢どおりの出力 {np.mean([o.strip() in (P, N, U) for o in outs]):.2f} / "
          f"正解率 {np.mean([o.strip() == g for o, g in zip(outs, gold_sent)]):.3f} / 例 {outs[:3]}")

# 正規表現: 文中の日付を YYYY-MM-DD に直させる。日付のない文（後ろの 4 件）を混ぜておく
dates = [("2024年3月5日に予約しました。", "2024-03-05"), ("会議は2023年11月20日の午後に開かれる。", "2023-11-20"),
         ("令和6年4月1日から新しい制度が始まる。", "2024-04-01"), ("平成31年4月30日に退位の儀式が行われた。", "2019-04-30"),
         ("2022年の大晦日に家族で集まった。", "2022-12-31"), ("締め切りは2025年1月10日です。", "2025-01-10"),
         ("昭和64年1月7日は昭和最後の日である。", "1989-01-07"), ("2020年2月29日はうるう日だった。", "2020-02-29"),
         ("明日は雨が降るらしい。", "不明"), ("来週の会議の場所が変わった。", "不明"),
         ("この店の料理は本当においしい。", "不明"), ("毎年春になると桜を見に行く。", "不明")]
ask = "次の文に書かれている日付を YYYY-MM-DD の形式で答えてください。和暦は西暦に直してください。日付が書かれていなければ「不明」と答えてください。\n\n文: "
DATE = r"\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])"
for label, output_type in [("制約なし", None), ("日付だけ", Regex(DATE)), ("日付か「不明」", Regex(f"{DATE}|不明"))]:
    outs, speed = run(output_type, [chat(ask + s) for s, _ in dates], max_new_tokens=60)
    found = [m.group(0) if (m := re.search(f"{DATE}|不明", o)) else None for o in outs]   # 制約なしは出力から取り出す
    print(f"{label}: 形式どおり {np.mean([re.fullmatch(f'{DATE}|不明', o.strip()) is not None for o in outs]):.2f} / "
          f"正解（日付あり 8 件）{sum(f == g for f, (_, g) in zip(found[:8], dates))} / 正解（日付なし 4 件）{sum(f == '不明' for f in found[8:])}")
    print("   ", [f"{s[:8]}→{o.strip()[:20]}" for (s, _), o in zip(dates, outs)])
```

```python
# 3. 強すぎる制約: 算数の文章題で、答えだけの JSON に縛ると考える過程を書けなくなる
# MGSM (Shi et al., 2022) の日本語の先頭 50 問。CC BY 4.0（Hugging Face のメタデータは CC BY-SA 4.0）。
# この章の「考える過程を書かせて LLM に算数の文章題を解かせる（Chain-of-Thought と self-consistency）」と同じデータ
mgsm = load_dataset("juletxara/mgsm", "ja", split="test").select(range(50))
gold = np.array(mgsm["answer_number"], dtype=float)

class Answer(BaseModel):
    answer: int
class ReasonThenAnswer(BaseModel):
    reasoning: str
    answer: int
class AnswerThenReason(BaseModel):
    answer: int
    reasoning: str

THINK = "次の問題に答えてください。ステップごとに考えて途中の計算を書き、最後に答えを書いてください。"
def last_number(text):
    found = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(found[-1]) if found else np.nan

def answer_of(text, schema):
    if schema is None:                        # 制約なしは、出力の最後に出てきた数値を答えとする
        return last_number(text)
    try:
        return float(schema.model_validate_json(text).answer)
    except ValidationError:                   # 上限のトークン数で打ち切られて JSON が閉じていない
        return np.nan

# 4 条件 × 50 問（Colab の T4 で 4〜6 分程度）
math_res, hits = {}, {}
for label, schema in [("制約なし (CoT)", None), ("answer だけ", Answer), ("reasoning → answer", ReasonThenAnswer), ("answer → reasoning", AnswerThenReason)]:
    hint = "" if schema is None else f"\n出力は次の JSON スキーマに従ってください: {json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
    outs, speed = run(schema, [chat(f"{THINK}{hint}\n\n問題: {q}") for q in mgsm["question"]], max_new_tokens=400)
    pred = np.array([answer_of(o, schema) for o in outs])
    hits[label] = pred == gold
    math_res[label] = {"正解率": (pred == gold).mean(), "答えが取れない": np.isnan(pred).mean(), **speed}
    print(f"--- {label}（正解 {gold[1]:g}）:", outs[1][:200].replace("\n", " "))
print(pd.DataFrame(math_res).round(3).to_string())
for label in list(math_res)[1:]:
    print(f"正解率の差（{label} − 制約なし (CoT)）:", diff_ci(hits[label], hits["制約なし (CoT)"]))
print(f"GPU メモリ最大 {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

- 制約付きデコーディングは、生成の各ステップで、形式（JSON スキーマ、正規表現、選択肢）に合わない続きになるトークンの確率を 0 にしてから次のトークンを選ぶ方法である。outlines（dottxt、Apache 2.0）は、形式を正規表現（有限オートマトン）に直し、語彙の各トークンがどの状態から受理されるかを前もって索引にしておく (Willard and Louf, 2023)。transformers のモデルでは `model.generate` に logits_processor を足すだけなので、生成の設定（貪欲法、`max_new_tokens` など）はそのまま使える。ここでは版を 1.3.3 に固定した。裏側の実装は xgrammar (Apache 2.0) や llguidance (MIT) にも切り替えられる。lm-format-enforcer (MIT) も同じ目的のライブラリである。
- JSON（40 件）: 生成してから取り出す方式では 10%（4 件）がスキーマに合わなかった（表示した例は、観点に候補にない「味」「子ども連れfriendly」を書いたもの）。制約ありでは違反は 0 件で、観点の F1 も 0.656 から 0.759 に上がった。感情の正解率は 72.5% と 77.5% で、差の +5.0 ポイント [+0.0, +12.5] は違反を不正解に数えた分にあたり、この件数では偶然と区別しにくい。制約ありの出力は改行のない JSON だけ（平均 16.4 トークン）で、取り出す方式は改行や説明が付いて平均 34.9 トークンだった。
- 制約で守られるのは形式だけである。良い点と悪い点が混じった neutral の 10 件のうち、制約ありは 9 件を negative と答えた（取り出す方式も 7 件が negative、2 件が違反）。指示を短くしてスキーマを書かないと、制約ありでも 40%（16 件）の出力が `score` に 8 や 10 を入れてスキーマに合わず、違反を不正解に数えた感情の正解率は 40% に下がった（形式どおりの 24 件では 16 件が正解）。制約があるので、この 16 件はどれも `score` の範囲だけが違反である。outlines_core 0.2.14 は、JSON スキーマの `minimum` / `maximum`（pydantic の `ge` / `le`）を正規表現に反映しない（`build_regex_from_schema` で確かめた）。制約を掛けても pydantic での検証は残し、範囲を必ず守らせたいなら `Literal[1, 2, 3, 4, 5]` のような列挙にする。指示とスキーマの説明はプロンプトにも書く。
- 選択肢: 感情だけを答えさせると、制約なしでも 97% が選択肢どおりの 1 語で、正解率は制約の有無によらず 80.0% だった。選択肢の制約は、各ステップで許されたトークンのうち最も確率の高いものを選ぶだけで、選択肢の全体の尤度を比べるわけではない（この章の「選択肢の尤度を比べて LLM に分類させる」とは結果が違いうる）。
- 正規表現（12 文）: 制約なしでは「与えられた文に書かれている日付は…」のように説明から始まり、YYYY-MM-DD だけの出力は 0 件、日付のある 8 文から取り出せた正解は 3 件だった。日付の形式だけを許すと 8 文中 7 文が正解になったが、日付のない 4 文すべてに、文に書かれていない日付（2023-05-14 など）を作った。「不明」も許すと 4 文中 3 文で「不明」と答えた（「毎年春になると…」は 2023-04-01）。昭和64年1月7日は、どちらでも形式は正しいが 1 日ずれた 1989-01-08 になった。答えがない場合の逃げ道を形式に入れておかないと、制約がもっともらしい誤りを作らせる。
- 強すぎる制約（MGSM の 50 問）: 自由に考えさせて最後の数値を取ると正解率 36%、答えだけの JSON に縛ると 6%（差 −30.0 ポイント [−44.0, −16.0]）だった。同じく `answer` を先に、`reasoning` を後に置くと 4% で、例の問題のように、先に書いた答え (2) と後の推論 (3) が食い違う。`reasoning` を先に置くと 14% まで戻るが、自由な CoT よりはまだ低かった（−22.0 ポイント [−36.0, −6.0]。原因は調べていない）。形式の制約で推論の課題の性能が下がることは Tam et al. (2024) も報告している。推論が要る課題では、自由に考えさせたあとで別の呼び出しで構造化するか、推論のフィールドを先に置く。
- 速度: バッチ生成では、ステップ数がいちばん長い出力のトークン数で決まるので、表の「1 ステップあたりのミリ秒」で比べる。出力の最大の長さがほぼ同じ MGSM の 2 条件では、`reasoning` → `answer` が 295 ミリ秒、自由な CoT が 61 ミリ秒で、制約ありは 1 ステップが約 5 倍遅かった。各ステップで、バッチの 1 件ずつ索引から許されるトークンのマスクを作り、GPU に送って logits に掛ける処理が加わるためと考えられる（内訳は測っていない）。出力が短い条件はプロンプトの処理などの固定の時間が大きく、1 ステップあたりでは比べにくいが、出力が短くなる分、1 件あたりでは速くなりうる（短い指示の JSON は 0.047 秒、取り出す方式は 0.104 秒）。形式ごとの準備（オートマトンの構築）は 0.3〜0.7 秒だった。時間は共有の GPU で測った目安である。GPU メモリは最大 5.7 GB（T4 と同じ float16 で実行した値）。
- データは、レビュー 40 件と日付の 12 文をこのレシピのために作り、正解も付けた（neutral の判断などには異論がありうる）。MGSM は先頭の 50 問だけなので、信頼区間の幅は 30 ポイント近くある。モデルは sarashina2.2-1b-instruct（SB Intuitions、MIT ライセンス）。API のモデルでは、LLM の API の章の「Gemini に決まった形式（JSON スキーマ）で出力させる」「Claude に決まった形式（JSON スキーマ）で出力させる」の機能を使う。トークナイザーの修正については、この章の「オープンな LLM とチャットする（チャットテンプレート）」の注意書きを参照。

## 検索広告の見出しを LLM で複数作り、文字数やキーワードを自動でチェックして選ぶ {ad_text_generation gpu}

商品やサービスの説明から、検索連動型広告の見出し（半角 30 文字・全角 15 文字まで）を LLM で何本も作り、文字数の超過、重複、キーワードの有無、説明にない数値や誇大な表現を自動でチェックして良いものを選びたい。公開データで、人が作った見出しとの類似度による自動評価も試したい。

```python
!pip install -q rouge-score sacrebleu fugashi unidic-lite
```

```python
import re, unicodedata
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"       # MIT ライセンス
tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()

RULES = ("検索連動型広告の見出しを 1 つ作ってください。\n"
         "- 全角 15 文字 (半角 30 文字) 以内にする\n- 検索キーワードの語をなるべく含める\n"
         "- 説明に書かれていない数値、実績、「No.1」「最安」などの表現は使わない\n- 見出しだけを出力する")

def prompt(kw, desc, shots=()):
    msgs = []
    for s_kw, s_desc, s_title in shots:       # 例は「ユーザーの依頼 → アシスタントの見出し」の対話として前に置く
        msgs += [{"role": "user", "content": f"{RULES}\n\n検索キーワード: {s_kw}\n説明: {s_desc}"}, {"role": "assistant", "content": s_title}]
    msgs.append({"role": "user", "content": f"{RULES}\n\n検索キーワード: {kw}\n説明: {desc}"})
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

def clean(text):
    """1 行目だけを取り、前後の記号 (かぎかっこ、箇条書きの印、番号) を除く"""
    line = text.strip().split("\n")[0].strip()
    line = re.sub(r"^(?:[-・*●]|\d+[.．)）])\s*", "", line)
    line = re.sub(r"[（(\[［【][^）)\]］】]*文字[^）)\]］】]*[）)\]］】]", "", line).strip()   # 「（全角15文字）」のような注記を除く
    return line.strip("「」『』\"' 　")

@torch.no_grad()
def generate(prompts, n=1, bs=32, max_new_tokens=32):
    """プロンプトごとに n 本の見出しを返す。n=1 なら貪欲法、n>1 ならサンプリング (左パディングのバッチ生成は、この章の「複数のプロンプトをまとめてバッチ生成する」と同じ)"""
    outs, step = [], max(1, bs // n)
    for i in range(0, len(prompts), step):
        enc = tokenizer(prompts[i:i + step], return_tensors="pt", padding=True, add_special_tokens=False).to(llm.device)
        sample = dict(do_sample=True, temperature=1.0, top_p=0.95) if n > 1 else dict(do_sample=False)
        out = llm.generate(**enc, max_new_tokens=max_new_tokens, num_return_sequences=n, pad_token_id=tokenizer.pad_token_id, **sample)
        outs += [clean(t) for t in tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return [outs[j * n:(j + 1) * n] for j in range(len(prompts))]

def width(s):
    """全角 (East Asian Width が F か W) を 2、それ以外を 1 と数える。Google 広告の見出しの上限は 30"""
    return sum(2 if unicodedata.east_asian_width(c) in "FW" else 1 for c in s)

def shorten(cands, limit=30, rounds=3):
    """上限を超えた見出しだけを、今の長さを伝えて LLM に短く言い換えさせる (貪欲法)。最大 rounds 回くり返す"""
    cands = list(cands)
    for _ in range(rounds):
        long = [i for i, c in enumerate(cands) if width(c) > limit]
        if not long:
            break
        ask = [tokenizer.apply_chat_template([{"role": "user", "content":
                   f"次の広告の見出しは全角に換算して {width(cands[i]) / 2:g} 文字あり、上限の {limit / 2:g} 文字を超えています。"
                   f"意味を保ったまま {limit / 2:g} 文字以内に短くしてください。新しい情報は加えず、短くした見出しだけを出力してください。\n\n{cands[i]}"}],
                   add_generation_prompt=True, tokenize=False) for i in long]
        for i, (c,) in zip(long, generate(ask)):
            cands[i] = c
    return cands
```

```python
# チェックの関数: キーワード、説明にない数値、誇大になりやすい表現、重複 (文字幅の width は前のセル)
norm = lambda s: unicodedata.normalize("NFKC", s).lower().replace(",", "")
def kw_coverage(text, kw):
    """検索キーワードの語 (空白で区切る) のうち、見出しに含まれる割合"""
    words = norm(kw).split()
    return sum(w in norm(text) for w in words) / len(words) if words else 0.0

def unsupported_numbers(text, source):
    """見出しに出てくる数値のうち、説明にないもの"""
    return [x for x in re.findall(r"\d+(?:\.\d+)?", norm(text)) if x not in re.findall(r"\d+(?:\.\d+)?", norm(source))]

HYPE = r"no\.?1|ナンバーワン|日本一|業界一|世界一|日本初|業界初|世界初|最安|最高|最大|最短|唯一|絶対|必ず|完全|100%|永久|誰でも"
def hype_words(text, source):
    """誇大になりやすい表現のうち、説明に出てこないもの (使ってよいかは根拠しだいなので、人が確かめる印として使う)"""
    return [w for w in re.findall(HYPE, norm(text)) if w not in norm(source)]

def bigrams(s):
    s = norm(re.sub(r"\s", "", s))
    return {s[i:i + 2] for i in range(len(s) - 1)} or {s}
def similar(a, b):
    """文字 2-gram のジャッカード係数"""
    return len(bigrams(a) & bigrams(b)) / len(bigrams(a) | bigrams(b))

def check(cands, kw, source, limit=30):
    rows = []
    for c in cands:
        rows.append({"見出し": c, "幅": width(c), "幅OK": 0 < width(c) <= limit, "KW": round(kw_coverage(c, kw), 2),
                     "説明にない数値": unsupported_numbers(c, source), "誇大表現": hype_words(c, source)})
    df = pd.DataFrame(rows)
    df["要確認"] = df["説明にない数値"].map(bool) | df["誇大表現"].map(bool)
    # 合格した候補どうしの類似度の平均 (多くの候補と内容が共通する、無難な候補ほど高い。最小ベイズリスク復号の考え方)
    ok = df["幅OK"] & ~df["要確認"]
    df["合意"] = [np.mean([similar(c, d) for j, d in enumerate(df["見出し"]) if j != i and ok[j]] or [0]) for i, c in enumerate(df["見出し"])]
    return df

def select(df, k=3, max_sim=0.5):
    """合格した候補を KW の割合 → 合意の順に並べ、すでに選んだものと似すぎるもの (重複) を飛ばして k 本選ぶ"""
    picked = []
    for _, r in df[df["幅OK"] & ~df["要確認"]].sort_values(["KW", "合意"], ascending=False).iterrows():
        if all(similar(r["見出し"], p) < max_sim for p in picked):
            picked.append(r["見出し"])
        if len(picked) == k:
            break
    return picked

# 架空のサービスで試す
kw = "オンライン英会話 初心者"
desc = ("オンライン英会話「ことばラボ」。講師は全員日本語で説明できるので、英語が初めての方も安心。"
        "1 回 25 分、月 8 回で月額 6,980 円 (税込)。無料体験レッスンは 2 回まで。朝 6 時から深夜 1 時まで予約できる。")
torch.manual_seed(0)
cands = generate([prompt(kw, desc)], n=16)[0]
pd.set_option("display.width", 200); pd.set_option("display.unicode.east_asian_width", True)
print("生成した候補の幅:", [width(c) for c in cands], cands[:3])
table = check(shorten(cands), kw, desc)      # 上限を超えたものを短くさせてから、もう一度チェックする
print(table.drop(columns="合意").to_string())
print("選んだ見出し:", select(table))
```

```python
# CAMERA (CyberAgent): 実際の検索広告から作った日本語の広告文生成のデータセット (Mita et al., 2024)。CC BY-NC-SA 4.0 (非商用に限る)
# 入力は検索キーワード (kw) と LP (広告のリンク先のページ) の説明文 (lp_meta_description)。
# テストデータは 1 件に 4 本の参照の見出しがある (title_org は実際の広告、title_ne1〜3 は全角 15 文字以内で人が書き足したもの)
def camera(split):
    path = hf_hub_download("cyberagent/camera", f"without-lp-images/{split}-00000-of-00001.parquet", repo_type="dataset")
    return pd.read_parquet(path).drop(columns="parsed_full_text_annotation")

test = camera("test").sample(120, random_state=0).reset_index(drop=True)      # 時間の都合で 872 件から 120 件
# few-shot の例は架空のものを自分で書く (CAMERA の学習・開発データには、キーワードと LP の内容が合わない組も混じるため)
shots = [("家計簿 アプリ 無料", "レシートを撮るだけで自動で記帳できる家計簿アプリ「マネノート」。基本機能は無料。銀行口座やカードと連携でき、グラフで支出を確認できる。",
          "レシート撮影で家計簿が自動に"),
         ("渋谷 ヨガ 体験", "渋谷駅から徒歩 3 分のヨガスタジオ「アサナ渋谷」。初心者向けのクラスが中心で、体験レッスンは 1,000 円。マットやウェアは無料で借りられる。",
          "渋谷駅3分のヨガ 体験1000円")]
print([(t, width(t)) for *_, t in shots])

# 貪欲法で 1 本と、サンプリングで 8 本ずつ。上限を超えた候補は短くさせる (Colab の T4 で 10 分程度)
prompts = [prompt(k, d, shots) for k, d in zip(test["kw"], test["lp_meta_description"])]
greedy = [c[0] for c in generate(prompts)]
torch.manual_seed(0)
samples = generate(prompts, n=8)
flat = shorten([c for cs in samples for c in cs])                  # 全候補をまとめて短くさせる (バッチにするため)
shortened = [flat[j * 8:(j + 1) * 8] for j in range(len(samples))]
def pick(cands_list):
    checked = [check(c, k, d) for c, k, d in zip(cands_list, test["kw"], test["lp_meta_description"])]
    # 合格がなければ、幅OK → KW の順で最上位のもの (実際には人が書くか、生成し直す)
    return checked, [(select(t, k=1) or [t.sort_values(["幅OK", "KW"], ascending=False)["見出し"].iloc[0]])[0] for t in checked]
checked0, picked0 = pick(samples)
checked, picked = pick(shortened)
print(f"GPU メモリ最大 {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
for i in range(4):
    print(f"KW: {test['kw'][i]} / 実際の広告: {test['title_org'][i]} / 貪欲: {greedy[i]} / 選択: {picked[i]}")
```

```python
# 8 本の候補のチェック結果の集計: 幅の超過、重複、要確認の割合
for label, ch in [("短くする前", checked0), ("短くした後", checked)]:
    a = pd.concat(ch, ignore_index=True)
    print(f"{label}: 幅の超過 {1 - a['幅OK'].mean():.1%}  要確認 {a['要確認'].mean():.1%}  1 本も合格しなかった入力 {np.mean([not select(t, k=1) for t in ch]):.1%}")
allc = pd.concat(checked, ignore_index=True)      # 短くした後の全候補
distinct = np.mean([len(set(map(norm, c))) / len(c) for c in samples])
near_dup = np.mean([np.mean([similar(a, b) >= 0.5 for i, a in enumerate(c) for b in c[i + 1:]]) for c in samples])
print(f"短くする前の {len(allc)} 本: 完全に同じ候補を除くと残る割合 {distinct:.1%}  "
      f"よく似た組 (2-gram の類似度 0.5 以上) の割合 {near_dup:.1%}")
print(f"短くした後: 説明にない数値 {allc['説明にない数値'].map(bool).mean():.1%}  誇大表現 {allc['誇大表現'].map(bool).mean():.1%}")
print("短くした例:", [(a, b) for a, b in zip(samples[0], shortened[0]) if a != b][:3])
print("説明にない数値を含む候補の例:", allc.loc[allc["説明にない数値"].map(bool), "見出し"].head(5).tolist())
print("誇大表現を含む候補の例:", allc.loc[allc["誇大表現"].map(bool), "見出し"].head(5).tolist())
```

```python
# 参照との自動評価: 人が書いた title_ne1〜3 を参照にし、実際の広告 (title_org) も同じ参照で採点して人の目安にする
from fugashi import Tagger
from rouge_score import rouge_scorer
from sacrebleu.metrics import CHRF

tagger = Tagger()
class Tok:
    def tokenize(self, text):
        return [w.surface for w in tagger(text)]
rouge = rouge_scorer.RougeScorer(["rouge1"], tokenizer=Tok())
chrf = CHRF()
refs = test[["title_ne1", "title_ne2", "title_ne3"]].values.tolist()

def scores(hyps):
    return pd.DataFrame({
        "ROUGE-1": [rouge.score_multi(r, h)["rouge1"].fmeasure for r, h in zip(refs, hyps)],      # 参照ごとの F の最大
        "chrF": [chrf.sentence_score(h, r).score / 100 for r, h in zip(refs, hyps)],
        "幅OK": [0 < width(h) <= 30 for h in hyps],
        "KW": [kw_coverage(h, k) for h, k in zip(hyps, test["kw"])],
        "要確認": [bool(unsupported_numbers(h, d) or hype_words(h, d)) for h, d in zip(hyps, test["lp_meta_description"])]})

systems = {"貪欲法 1 本": greedy, "サンプリング 1 本目": [c[0] for c in samples], "8 本からチェックして選ぶ": picked0,
           "8 本を短くしてから選ぶ": picked, "実際の広告 (title_org)": test["title_org"].tolist()}
results = {s: scores(h) for s, h in systems.items()}
print(pd.DataFrame({s: r.mean() for s, r in results.items()}).T.round(3).to_string())

rng = np.random.default_rng(0)
def diff_ci(a, b, col, n_boot=2000):
    """同じ入力で比べる対応のあるブートストラップで、平均の差の 95% 信頼区間"""
    d = (results[a][col] - results[b][col].astype(float)).to_numpy(dtype=float)
    boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    print(f"{a} − {b} ({col}): {d.mean():+.3f} [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]")

for col in ["ROUGE-1", "chrF", "KW"]:
    diff_ci("8 本を短くしてから選ぶ", "貪欲法 1 本", col)
diff_ci("8 本を短くしてから選ぶ", "実際の広告 (title_org)", "ROUGE-1")
```

- 見出しの文字数の上限は、Google 広告のヘルプでは半角 30 文字で、全角 1 文字を半角 2 文字と数える（全角だけなら 15 文字）。ここでは Unicode の East Asian Width が F（全角）か W（広い）の文字を 2 と数えた。「※」「…」のような幅が文脈で変わる文字 (A) を媒体がどう数えるかは確かめていないので、入稿の前に媒体の画面で確かめる。媒体や広告の種類ごとに上限が違うので、`limit` と `width` をそれに合わせる。
- このモデル (3B) では、「全角 15 文字以内」と指示しても文字数はほとんど守られなかった。架空のサービスの例では 16 本すべてが上限を超え（幅 48〜82）、今の長さを伝えて 3 回短くさせた後も 15 本が超えたままだった。CAMERA の 120 件でも、サンプリングした候補の 80.1%、貪欲法の見出しの 71.7% が超えた。最大 3 回短くさせても、上限に収まらない候補が 56.5% 残った。それでも候補を 8 本作っておけば、合格が 1 本もない入力は 41.7% から 11.7% に減り、選んだ見出しの 89.2% が上限に収まった（実際の広告は 94.2%）。文字数は生成のあとに必ずプログラムで数え、超えたものは捨てるか人が直す。上限で切り詰めると、文の途中で切れてしまう。
- 候補は、完全に同じものを除いても 93.0% 残ったが、組の 27.7% は文字 2-gram の類似度が 0.5 以上で、言い回しが少し違うだけのものが多い。`select` のように、すでに選んだ見出しと似すぎるものを飛ばすと、レスポンシブ検索広告のように複数の見出しを入稿するとき、訴求の違うものをそろえやすい。合格した候補の中では、キーワードの語の割合、ほかの候補との類似度の平均（多くの候補と共通する無難な内容ほど高い）の順で並べた。クリック率のような効果は、配信して確かめるしかない。
- 説明にない数値や「No.1」「最安」のような表現は、根拠がなければ景品表示法の優良誤認・有利誤認にあたるおそれがあり、広告主の信用も損なう。「説明にない数値を使わない」と指示しても、短くした後の候補の 2.0% が説明にない数値で、0.2% が誇大になりやすい表現で引っかかった。本当に説明と違う例には、説明の「インターネット割引最大20,000円」を「割引最大15,000円」と書いた候補や、指示の「15文字以内」を書き写した候補があった。一方で、ここでの判定は「説明の文字列に出てこない」という印にすぎず、説明の「30,000円」を「3万」、英語の説明の「213,000 courses and 57 million students」を「21万コース、5,700万人」と書いた候補のような表記の違いも引っかかる。「人気No.1」の 2 本は、説明の「ブランド認知度No.01」を言い換えたもので、表記が違うので文字列の照合では通らなかったが、「認知度」が「人気」に変わって根拠の範囲も変わっている。逆に説明にある「日本初」は通る（説明が正しいかは確かめない）。実際の広告 (title_org) でも 7.5% が引っかかった。LP の説明文 (meta description) はページの一部なので、人が書いた見出しには、ページのほかの部分にある数値も入る。引っかかったものは、広告主の資料と照らして人が確かめる。
- 参照との自動評価では、方法の違いがほとんど見えなかった。人が書いた 3 本の参照に対する ROUGE-1（参照ごとの F の最大）は、どの方法も 0.18〜0.22 で、短くしてから選んだ見出しと貪欲法の差は −0.011 [−0.034, +0.013]（対応のあるブートストラップの 95% 信頼区間。評価指標と統計的検定の章の「ブートストラップ法で評価値の信頼区間を求める」を参照）だった。chrF は −0.013 [−0.027, −0.000] とわずかに下がったが、信頼区間の端が 0 に接する程度である。実際の広告 (0.184) との差も +0.026 [−0.007, +0.059] で、人が書いた見出しも LLM と同じくらい参照と重ならない。広告文は同じ商品でも訴求の仕方がいくつもあるので、参照との重なりは良し悪しの目安にしかならない。CAMERA の論文 (Mita et al., 2024) も、BLEU-4、ROUGE-1、BERTScore のほかに、キーワードを含む割合と文字数の規定を守る割合を指標にし、人手評価との比較では、魅力 (attractiveness) の順位を既存の自動評価指標で再現するのは難しかったと報告している。記事の本文から見出しを生成して実際の見出しと比べる例は、要約と平易化の章の「記事の見出しを生成して実際の見出しと比べる」にある。
- キーワードの語を含む割合は、LLM の見出し (0.80〜0.86) のほうが実際の広告 (0.32) よりずっと高く、実際の広告はキーワードの語をそのまま含まないことが多い。ただし、CAMERA のキーワード (kw) は検索語なので、「は な さくせい めい ほけん」のように分かち書きが崩れたものや、LP や参照と内容が合わないもの（「東京 銀座 バイト」に対して実際の広告が池袋の求人）も混じるので、実際の広告の低さには、キーワードと広告が合わない組の影響も含まれる。件数も 120 件なので、細かな差は読み取らない。
- データは CAMERA（CyberAgent、CC BY-NC-SA 4.0）。非商用に限られ、改変したものも同じライセンスで公開する必要がある。商用の広告文生成のモデルの学習や評価には使えない。ここでは LP の画像を含まない without-lp-images のテストデータ（872 件）の parquet だけを読み込んでいる。モデルは sarashina2.2（SB Intuitions、MIT ライセンス）で、T4 と同じ float16 で実行したとき GPU メモリを最大 10.8 GB 使った。トークナイザーの修正については、この章の「オープンな LLM とチャットする（チャットテンプレート）」の注意書きを参照。

## LoRA で LLM を少ないメモリでファインチューニングする {lora_finetune gpu norun}

LLM を自分のタスク（ここでは文法誤り訂正）に適応させたい。全パラメータを更新するのは重いので、LoRA で一部だけ学習する。

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
- 指示に従うように調整したあと、応答の好みに合わせてさらに調整する方法は、この章の「選好データで LLM を調整する (DPO)」を参照。
- 4bit 量子化と組み合わせる (QLoRA) と、7B クラスでも T4 で学習できる。`prepare_model_for_kbit_training` を使う。
- ここでは 8 種類の文を 25 回ずつ繰り返しているので、1 エポック目で損失がほぼ 0 になり、学習データを丸暗記する。その結果、「楽しいでした」→「面白かったです」のように、誤りでない語まで学習データの訂正文に寄せて書き換える。実際のデータで学習するときは、検証データの損失を見てエポック数を決める。

## 選好データで LLM を調整する (DPO) {dpo gpu}

「良い応答」と「悪い応答」の組（選好データ）から、LLM がより好まれる応答を出すように調整したい。報酬モデルを別に作らずに済む DPO を、LoRA と組み合わせて行う。

```python
!pip install -q trl peft "torchao>=0.16"   # Colab に入っている古い torchao (0.10) のままだと peft の import が失敗する
```

```python
import random
import numpy as np
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

def load_tokenizer(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    # transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
    # 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)
    sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
    tokenizer.backend_tokenizer.model = models.Unigram(
        [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
    return tokenizer

set_seed(0)
name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"   # この章の LoRA のレシピと同じ 0.5B のモデル (MIT License)
tokenizer = load_tokenizer(name)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto")
```

```python
from datasets import Dataset, load_dataset

# 選好データ: llm-jp/hh-rlhf-12k-ja (MIT License)。Anthropic の hh-rlhf (Bai et al., 2022、MIT License) から
# 12,000 件を選び、DeepL で日本語に訳したもの。会話 (conversations) に対して、人が選んだ応答 (chosen) と選ばなかった応答 (rejected) がある
raw = load_dataset("llm-jp/hh-rlhf-12k-ja", split="train")
print(raw.to_pandas()["source"].value_counts())

# 有用さ (helpful-*) の部分だけを使う (harmless-base は有害な依頼とその応答で、不快な内容を含む)。
# 1 往復の会話で、短めのものに絞る。trl の会話形式 (prompt / chosen / rejected にメッセージのリスト) にする
data = [{"prompt": [{"role": "user", "content": r["conversations"][0]["value"]}],
         "chosen": [{"role": "assistant", "content": r["chosen"]}],
         "rejected": [{"role": "assistant", "content": r["rejected"]}]}
        for r in raw
        if r["source"].startswith("helpful") and len(r["conversations"]) == 1 and r["chosen"] != r["rejected"]
        and len(r["conversations"][0]["value"]) <= 200 and max(len(r["chosen"]), len(r["rejected"])) <= 400]
random.Random(0).shuffle(data)
train, test = data[:2000], data[2000:2200]
print(len(data), "件のうち 学習", len(train), "件、評価", len(test), "件")
for ex in test[:2]:
    print("\n質問:", ex["prompt"][0]["content"], "\n選ばれた応答:", ex["chosen"][0]["content"],
          "\n選ばれなかった応答:", ex["rejected"][0]["content"])
print("\n学習データで chosen のほうが長い組の割合:", np.mean([len(ex["chosen"][0]["content"]) > len(ex["rejected"][0]["content"]) for ex in train]))
```

```python
# 応答の対数尤度 log p(応答 | 質問) を、応答部分のトークンについて合計する
def response_logps(model, examples, key, batch_size=16):
    tokenizer.padding_side = "right"
    out = []
    for i in range(0, len(examples), batch_size):
        ids, starts = [], []
        for ex in examples[i:i + batch_size]:
            p = tokenizer(tokenizer.apply_chat_template(ex["prompt"], add_generation_prompt=True, tokenize=False),
                          add_special_tokens=False)["input_ids"]
            r = tokenizer(ex[key][0]["content"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            ids.append(p + r)
            starts.append(len(p))
        n = max(map(len, ids))
        x = torch.tensor([s + [tokenizer.pad_token_id] * (n - len(s)) for s in ids], device=model.device)
        mask = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in ids], device=model.device)
        with torch.no_grad():
            logp = model(input_ids=x, attention_mask=mask).logits[:, :-1].float().log_softmax(-1)
        tok = logp.gather(-1, x[:, 1:, None])[..., 0]
        out += [tok[j, st - 1:len(s) - 1].sum().item() for j, (s, st) in enumerate(zip(ids, starts))]
    return np.array(out)

def generate(model, prompts, max_new_tokens=384, batch_size=25):
    tokenizer.padding_side = "left"
    outs, cut = [], []
    for i in range(0, len(prompts), batch_size):
        inputs = tokenizer.apply_chat_template(prompts[i:i + batch_size], add_generation_prompt=True, return_tensors="pt",
                                               return_dict=True, padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1,
                                 pad_token_id=tokenizer.pad_token_id)
        gen = out[:, inputs["input_ids"].shape[1]:]
        outs += [tokenizer.decode(o, skip_special_tokens=True).strip() for o in gen]
        cut += [o[-1].item() not in (tokenizer.eos_token_id, tokenizer.pad_token_id) for o in gen]   # 長さの上限で打ち切られたか
    return outs, np.array(cut)

# 調整前 (= DPO の参照モデル) の対数尤度と、評価用の質問 100 件への応答
model.eval()
ref_c, ref_r = response_logps(model, test, "chosen"), response_logps(model, test, "rejected")
gen_prompts = [ex["prompt"] for ex in test[:100]]
before, cut_before = generate(model, gen_prompts)   # Colab の T4 で 2〜3 分
print("調整前に、選ばれた応答のほうが尤度が高い割合:", (ref_c > ref_r).mean())
```

```python
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
args = DPOConfig(output_dir="dpo-out", beta=0.1, learning_rate=5e-5, num_train_epochs=1, per_device_train_batch_size=8,
                 warmup_steps=10, max_length=512, logging_steps=25, save_strategy="no", report_to="none",
                 gradient_checkpointing=False, bf16=BF16, fp16=not BF16, seed=0)   # DPOConfig は bf16=True が既定なので明示する
# ref_model を渡さずに peft_config を渡すと、LoRA を外した元のモデルが参照モデルになる (モデルを 2 つ読み込まずに済む)
trainer = DPOTrainer(model=model, args=args, train_dataset=Dataset.from_list(train), processing_class=tokenizer, peft_config=lora)
trainer.model.print_trainable_parameters()
trainer.train()   # Colab の T4 で 10〜15 分程度
for log in trainer.state.log_history:
    if "rewards/accuracies" in log:
        print(f"step {log['step']:3d}  loss {log['loss']:.3f}  報酬の正解率 {log['rewards/accuracies']:.3f}  報酬の差 {log['rewards/margins']:.3f}")
print(f"GPU メモリの最大使用量: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

```python
# 評価データでの選好の正解率。DPO の暗黙の報酬 r = β (log π(応答) − log π_ref(応答)) で、chosen と rejected を比べる
model = trainer.model.eval()
pol_c, pol_r = response_logps(model, test, "chosen"), response_logps(model, test, "rejected")
margin = args.beta * ((pol_c - ref_c) - (pol_r - ref_r))

def rate_ci(hits):   # 割合と、ブートストラップによる 95% 信頼区間
    boot = [hits[i].mean() for i in np.random.default_rng(0).integers(0, len(hits), (2000, len(hits)))]
    return f"{hits.mean():.3f} (95% 信頼区間 {np.percentile(boot, 2.5):.3f}〜{np.percentile(boot, 97.5):.3f}、{len(hits)} 件)"

print(f"評価データで、報酬が chosen > rejected になった割合: {rate_ci(margin > 0)}、報酬の差の平均 {margin.mean():.3f}")
print(f"選ばれた応答のほうが尤度が高い割合: 調整前 {(ref_c > ref_r).mean():.3f} → 調整後 {(pol_c > pol_r).mean():.3f}")
print(f"対数尤度の平均の変化: chosen {np.mean(pol_c - ref_c):+.2f}、rejected {np.mean(pol_r - ref_r):+.2f}")
```

```python
# 同じ質問への応答を、調整の前後で比べる
after, cut_after = generate(model, gen_prompts)
for p, b, a in list(zip(gen_prompts, before, after))[:4]:
    print("質問:", p[0]["content"], "\n[前]", b[:250], "\n[後]", a[:250], "\n")
def n_tok(texts):
    return np.mean([len(tokenizer(t)["input_ids"]) for t in texts])

print(f"応答の平均の長さ: 調整前 {np.mean([len(s) for s in before]):.0f} 文字 ({n_tok(before):.0f} トークン) → "
      f"調整後 {np.mean([len(s) for s in after]):.0f} 文字 ({n_tok(after):.0f} トークン)")
print(f"上限の 384 トークンで打ち切られた割合: 調整前 {cut_before.mean():.2f} → 調整後 {cut_after.mean():.2f}")
```

```python
# 調整前と後の応答を、大きめのモデル (sarashina2.2-3b-instruct、MIT License) に比べさせる (LLM による判定)。
# 応答を A と B に並べ、次のトークンが「A」と「B」になる確率を比べる。位置の偏りを打ち消すため、順番を入れ替えて 2 回聞いて平均する
judge_name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
jtok = load_tokenizer(judge_name)
judge = AutoModelForCausalLM.from_pretrained(judge_name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()
id_a, id_b = [jtok(c, add_special_tokens=False)["input_ids"][-1] for c in ("A", "B")]   # 評価指標と統計的検定の章と同じ取り方
print("判定に使うトークン:", jtok.convert_ids_to_tokens([id_a, id_b]))

def p_first_better(q, x, y):
    def p_a(a, b):
        msg = [{"role": "user", "content": f"次の質問に対する 2 つの応答のうち、より役に立つ応答はどちらですか。A か B の 1 文字で答えてください。\n\n"
                                           f"質問: {q}\n\n応答 A: {a}\n\n応答 B: {b}"}]
        inputs = jtok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(judge.device)
        with torch.no_grad():
            logits = judge(**inputs).logits[0, -1].float()
        return torch.softmax(logits[[id_a, id_b]], -1)[0].item()
    return (p_a(x, y) + 1 - p_a(y, x)) / 2

# まず、判定役が人の選好 (評価データの chosen / rejected) とどれくらい一致するかを確かめる。Colab の T4 で 2〜3 分
agree = np.array([p_first_better(ex["prompt"][0]["content"], ex["chosen"][0]["content"], ex["rejected"][0]["content"]) > 0.5 for ex in test])
print("判定役と人の選好の一致率:", rate_ci(agree))

win = np.array([p_first_better(p[0]["content"], a, b) for p, b, a in zip(gen_prompts, before, after)])
print("調整後の応答が調整前より良いと判定された割合 (勝率):", rate_ci(win > 0.5))
longer = np.array([len(a) > len(b) for b, a in zip(before, after)])   # 判定役が長い応答を好んでいないか
print(f"勝率の内訳: 調整後のほうが長い {longer.sum()} 問で {(win[longer] > 0.5).mean():.3f}、"
      f"長くならなかった (短いか同じ) {(~longer).sum()} 問で {(win[~longer] > 0.5).mean():.3f}")
print(f"GPU メモリの最大使用量: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

- DPO (Direct Preference Optimization; Rafailov et al., 2023) は、選好データから報酬モデルを学習して強化学習 (PPO) で最適化する RLHF の手順を、1 つの分類のような損失に置き換えたもの。参照モデルからのずれを β で抑えながら、chosen の尤度を rejected より相対的に上げる。実際には、この章の「LoRA で LLM を少ないメモリでファインチューニングする」のような指示チューニング (SFT) の後に行う。
- 上の出力では、評価データ（学習に使っていない 200 組）で報酬が chosen > rejected になった割合は 0.680（95% 信頼区間 0.615〜0.745）で、偶然 (0.5) より高い。ただし DPO が上げるのは chosen と rejected の差だけで、対数尤度そのものは chosen も rejected も上がり（平均で +12.82 と +6.95）、「chosen のほうが尤度が高い」割合は 0.480 → 0.485 とほとんど変わらなかった。
- 一方、生成した応答を判定役のモデルに比べさせると、調整後が良いとされたのは 100 問中 0.340（95% 信頼区間 0.250〜0.430）で、調整前より悪いと判定された。応答は短くなり（平均 480 → 435 文字）、長くならなかった 71 問では勝率が 0.268 と特に低い。GPU の計算は実行ごとにわずかに変わることがあり、勝率は小数第 2 位で変わりうるので、100 問の信頼区間の幅で見る。選好の正解率が上がっても、生成が良くなるとは限らない。
- 原因は確かめていないが、学習データの応答は hh-rlhf の論文（2022 年）のモデルの出力を機械翻訳したもので、ここでは 400 文字以下に絞っており、このモデル自身の応答とは文体も長さも違う（方策外のデータ）。調整するモデル自身に応答を複数生成させ、それに人や強いモデルが選好を付けたデータ（方策上のデータ）を使う方法もある。
- 判定役 (3B) と人の選好の一致率は 0.650 程度で、判定はあくまで目安。A と B の順番を入れ替えて平均し、位置の偏りを打ち消している。判定役の信頼性の確かめ方は、評価指標と統計的検定の章の「手元の LLM に生成文を採点させる (LLM-as-a-judge)」を参照。
- データ: hh-rlhf の元のデータセットカードには、選好（報酬）モデルの学習用で、対話エージェントを教師ありで学習するためのものではない（有害なモデルになりうる）と書かれている。DPO は報酬モデルを作らず、対話モデルそのものをこの応答で直接学習するので、この注意が当てはまる。ここで作るモデルは手法を試すためのもので、公開や実用には使わない。harmless-base には差別や暴力などの不快な内容が含まれるので、ここでは使っていない。日本語は DeepL による翻訳で、訳の崩れも混じる（上の 1 件目の「」の対応など）。
- `DPOConfig` は、`TrainingArguments` と違って `bf16=True`、学習率 1e-6、`gradient_checkpointing=True` が既定。T4 では bf16 が使えないので fp16 を明示し、LoRA では学習率を上げる（ここでは 5e-5）。β を大きくすると参照モデルから離れにくくなる。`loss_type="ipo"` などで損失を IPO などに変えられる。trl 1.18.0 で確かめた（版によって引数や既定値が変わることがある）。
- GPU メモリの最大使用量は学習時の 9.2GB で、3B の判定役を読み込んだ後も変わらなかった。sarashina2.2 のトークナイザーの修正 (`load_tokenizer`) については、この章の「オープンな LLM とチャットする（チャットテンプレート）」の注意書きを参照。

## 正解を判定できる課題で LLM を強化学習する (GRPO) {grpo gpu}

算数の文章題のように正解を機械的に判定できる課題で、正解なら報酬を与える関数を書き、LLM が自分で生成した答えから強化学習 (GRPO) で学ばせたい。学習の前後で、学習に使っていない問題の正解率を比べる。

```python
!pip install -q trl==1.18.0 peft==0.21.2 torchao==0.18.0   # Colab に入っている古い torchao (0.10) のままだと peft の import が失敗する
```

```python
import json, re, time, unicodedata
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

set_seed(0)
name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"   # この章の LoRA・DPO のレシピと同じ 0.5B のモデル (MIT License)
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto")

# 学習用: SakanaAI/gsm8k-ja-test_250-1319 (Apache 2.0)。GSM8K (Cobbe et al., 2021、MIT License) のテストの 251 問目以降の
# 1,069 問を GPT-4 で日本語に訳したもの。先頭の 250 問は MGSM と重なるので、開発元があえて除いている
train_raw = load_dataset("json", data_files=hf_hub_download("SakanaAI/gsm8k-ja-test_250-1319", "gsm8k-ja-test_250-1319.jsonl",
                                                            repo_type="dataset"), split="train")
# 評価用: MGSM の日本語（juletxara/mgsm の ja）の 250 問全部。GSM8K のテストの先頭 250 問を人手で訳したもので、学習用の問題とは重ならない
# ライセンスは、元のリポジトリでは CC BY 4.0、データセットカードのメタデータでは CC BY-SA 4.0
test_raw = load_dataset("juletxara/mgsm", "ja", split="test")
print(len(train_raw), len(test_raw))
print(train_raw[1]["question"], "→", train_raw[1]["answer_number"])
```

```python
# 指示: この章の Chain-of-Thought のレシピの「CoT (0-shot)」と同じ。途中の計算を書かせ、最後に答えを決まった形で書かせる
INSTRUCTION = "次の問題に答えてください。ステップごとに考えて途中の計算を書き、最後に「答えは〇〇です。」と書いてください。"
def to_prompt(question):
    return [{"role": "user", "content": f"{INSTRUCTION}\n\n問題: {question}"}]

def normalize(text):
    return unicodedata.normalize("NFKC", text).replace(",", "")

def final_answer(text):
    """決まった形（応答の最後の「答えは〇〇です。」）で書かれた答えを取り出す。形が違えば None"""
    m = re.search(r"答えは\s*[^\d\-\n]{0,5}?(-?\d+(?:\.\d+)?)[^\d\n]{0,10}$", normalize(text).strip())
    return float(m.group(1)) if m else None

def loose_answer(text):
    """形を問わず取り出す（この章の Chain-of-Thought のレシピの取り出し方の簡略版）: 「答え」の後ろ → 最後に出てきた数値"""
    text = normalize(text)
    found = re.findall(r"答え[はが]?\s*[:：]?[^\d\-\n]*(-?\d+(?:\.\d+)?)", text) or re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(found[-1]) if found else None

# 報酬関数: GRPOTrainer は、生成した応答 (completions) と、データセットのほかの列（ここでは answer_number）をキーワード引数で渡す
def correct_reward(completions, answer_number, **kwargs):
    """決まった形で書いた答えが正解なら 1、それ以外は 0"""
    return [1.0 if final_answer(c[0]["content"]) == float(a) else 0.0 for c, a in zip(completions, answer_number)]

def format_reward(completions, **kwargs):
    """正誤に関係なく、応答が「答えは〇〇です。」の形で終わっていれば 0.2"""
    return [0.2 if final_answer(c[0]["content"]) is not None else 0.0 for c in completions]

examples = [[{"content": "3 + 4 = 7 なので、答えは7個です。"}], [{"content": "答えは7個です。よって 3 + 4 = 7"}], [{"content": "答えは8です。"}]]
print(correct_reward(examples, [7, 7, 7]), format_reward(examples))
```

```python
@torch.no_grad()
def evaluate(model, questions, max_new_tokens=320, bs=64):
    """貪欲法で解かせて、応答の文字列のリストと、生成した平均のトークン数を返す"""
    model.eval()
    tokenizer.padding_side = "left"
    outs, n_tok = [], []
    for i in range(0, len(questions), bs):
        enc = tokenizer.apply_chat_template([to_prompt(q) for q in questions[i:i + bs]], add_generation_prompt=True,
                                            return_tensors="pt", return_dict=True, padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        n_tok += (gen != tokenizer.pad_token_id).sum(1).tolist()
        outs += tokenizer.batch_decode(gen, skip_special_tokens=True)
    return outs, np.mean(n_tok)

def scores(outs, gold):
    gold = np.array(gold, dtype=float)
    strict = np.array([final_answer(o) == g for o, g in zip(outs, gold)])
    loose = np.array([loose_answer(o) == g for o, g in zip(outs, gold)])
    return strict, loose, np.mean([final_answer(o) is not None for o in outs])

gold = test_raw["answer_number"]
t = time.time()
before, before_len = evaluate(model, test_raw["question"])   # Colab の T4 で 1〜2 分
b_strict, b_loose, b_fmt = scores(before, gold)
print(f"学習前: 正解率（決まった形の答え）{b_strict.mean():.3f}、正解率（形を問わない）{b_loose.mean():.3f}、"
      f"形を守った割合 {b_fmt:.3f}、平均 {before_len:.0f} トークン（{time.time() - t:.0f} 秒）")
print("--- 学習前の出力の例:\n", before[0])
```

```python
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

train_ds = Dataset.from_list([{"prompt": to_prompt(r["question"]), "answer_number": r["answer_number"]} for r in train_raw])
lora = LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
args = GRPOConfig(
    output_dir="grpo-out", max_steps=40, learning_rate=5e-5, warmup_steps=5,
    num_generations=8,                  # 1 問につき 8 個の応答を生成し、その中で報酬を比べる（グループ）
    per_device_train_batch_size=8, gradient_accumulation_steps=4,   # 1 ステップの更新に 32 応答 = 4 問 × 8 個
    steps_per_generation=8,             # 生成は 64 応答（8 問）をまとめて行い、2 ステップの更新に使う（生成の回数を減らして時間を短くする）
                                        # そのため 2 ステップ目の更新は、1 ステップ前のモデルが生成した応答を使う
    max_completion_length=320, temperature=1.0,
    logging_steps=5, save_strategy="no", report_to="none", seed=0,
    gradient_checkpointing=False, bf16=BF16, fp16=not BF16)   # GRPOConfig は bf16=True と gradient_checkpointing=True が既定なので明示する
trainer = GRPOTrainer(model=model, reward_funcs=[correct_reward, format_reward], args=args,
                      train_dataset=train_ds, processing_class=tokenizer, peft_config=lora)
trainer.model.print_trainable_parameters()
t = time.time()
trainer.train()   # Colab の T4 で 20〜25 分程度
print(f"学習 {time.time() - t:.0f} 秒、GPU メモリの最大使用量 {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

```python
# 報酬の推移（5 ステップごとの平均。1 ステップ = 4 問 × 8 個の応答）
logs = pd.DataFrame([l for l in trainer.state.log_history if "reward" in l])
cols = {"step": "ステップ", "rewards/correct_reward/mean": "正解の報酬", "rewards/format_reward/mean": "形式の報酬",
        "reward_std": "報酬の標準偏差", "completions/mean_length": "応答の長さ", "completions/clipped_ratio": "打ち切りの割合"}
print(logs[list(cols)].rename(columns=cols).round(3).to_string(index=False))
```

```python
# 学習後に、同じ評価用の問題を解かせて比べる
model = trainer.model
t = time.time()
after, after_len = evaluate(model, test_raw["question"])   # Colab の T4 で 1〜2 分
a_strict, a_loose, a_fmt = scores(after, gold)
print(pd.DataFrame({"学習前": [b_strict.mean(), b_loose.mean(), b_fmt, before_len], "学習後": [a_strict.mean(), a_loose.mean(), a_fmt, after_len]},
                   index=["正解率（決まった形の答え）", "正解率（形を問わない）", "形を守った割合", "平均のトークン数"]).round(3).to_string())

def diff_ci(x, y, n_boot=2000):   # 同じ問題で比べる対応のあるブートストラップで、正解率の差の 95% 信頼区間
    idx = np.random.default_rng(0).integers(0, len(x), (n_boot, len(x)))
    d = x[idx].mean(1) - y[idx].mean(1)
    return f"{(x.mean() - y.mean()) * 100:+.1f} ポイント [{np.percentile(d, 2.5) * 100:+.1f}, {np.percentile(d, 97.5) * 100:+.1f}]"

print("学習後 − 学習前（決まった形の答え）:", diff_ci(a_strict, b_strict))
print("学習後 − 学習前（形を問わない）:", diff_ci(a_loose, b_loose))
print(f"GPU メモリの最大使用量 {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
# 問題ごとの入れ替わり（形を問わない正解で数える）
print(pd.crosstab(pd.Series(b_loose, name="学習前"), pd.Series(a_loose, name="学習後")).rename(index=str, columns=str).to_string())

# 出力の例: 学習前は形を守らずに正解していた問題と、学習前は正解で学習後に間違えた問題
for label, hits in [("形が直った", ~b_strict & b_loose & a_strict), ("学習後に間違えた", b_loose & ~a_loose)]:
    i = int(np.flatnonzero(hits)[0])
    print(f"\n=== {label}例: 問題 {i}（正解 {gold[i]:g}）\n[前] {before[i][-250:]}\n[後] {after[i][-250:]}")
```

- GRPO (Group Relative Policy Optimization) は、Shao et al. (2024) が DeepSeekMath の論文で提案した強化学習の方法で、PPO の価値関数のモデルを使わず、同じ問題に対して生成した複数の応答（グループ、ここでは 8 個）の報酬の平均と標準偏差から、各応答がグループの中でどれだけ良かったか（アドバンテージ）を決める。DeepSeek-R1 (2025) は、数学の問題で、決まった形で書かせた最終的な答えの正誤と、形式を守ったかどうかを規則で判定する報酬を使った。ここも同じ考え方で、正解の報酬 (1) と形式の報酬 (0.2) を足している（このように規則で判定できる報酬は、検証できる報酬 (verifiable rewards) と呼ばれる）。グループの 8 個がすべて同じ報酬だとアドバンテージが 0 になって学習に使われないので、学習前に正解がほとんど出ない課題（正解率がほぼ 0）では、正解の報酬からは学べない（形式の報酬で差がつけば、形式だけは学ぶ）。
- この章の「選好データで LLM を調整する (DPO)」との違い: DPO は人が付けた選好の組（chosen と rejected）が要り、その応答は別のモデルの出力（方策外のデータ）だった。GRPO は選好の組も推論の過程の手本も要らず、問題と正解の数値と、それを判定する報酬関数だけで学習できる。応答は学習中のモデル自身がその場で生成する（方策上のデータ）。その代わり、学習のたびに生成するので時間がかかる。正解を機械的に判定できない課題（応答の役に立ち方など）には、判定役のモデル（報酬モデル）が要る。
- 上の出力では、正解率（決まった形の答え）は 0.072 → 0.212（+14.0 ポイント [+8.8, +19.2]）、形を守った割合は 0.200 → 0.944 に上がった。しかし形を問わない正解率は 0.252 → 0.216（−3.6 ポイント [−9.6, +1.6]）で、上がらなかった（信頼区間は 0 をまたぐ）。問題ごとに見ると、30 問が正解から不正解に、21 問が不正解から正解に変わった。この実行で身についたのは「決まった形で答える」ことで、40 ステップ（160 問）の学習では解く力は上がらなかった。報酬の推移でも、正解の報酬の上昇（0.05 前後 → 0.12〜0.26）は形式の報酬の上昇 (0.05 → 0.18) と同時に起きている。1 ステップは 4 問だけなので、正解の報酬は問題の難しさで大きく揺れる。また、GPU の計算は実行ごとにわずかに変わり、それがサンプリングで生成する応答と学習の経過を変えるので、同じ種でも学習後の正解率は実行ごとに変わる。1 回の実行で結論を出さず、種を変えて何回か学習して比べる。
- 応答は短くなった（評価で平均 192 → 130 トークン、学習中に上限の 320 トークンで打ち切られた割合は 0.21 → 0.01〜0.05）。打ち切られた応答は最後の「答えは〇〇です。」まで書けずに報酬が 0 になるので、短い応答が有利になると考えられる（短くなったことが正解率に影響したかどうかは確かめていない）。報酬が形式だけで取れるように設計すると、推論を書かずに答えだけを出すような、報酬は上がるが目的に合わない学習（報酬ハッキング）が起きることがある。報酬だけでなく、応答の長さと出力の例も見る。
- この章の「考える過程を書かせて LLM に算数の文章題を解かせる（Chain-of-Thought と self-consistency）」との関係: 指示文はそのレシピの「CoT (0-shot)」と同じで、推論を書かせてから答えさせる。GRPO は推論の過程には報酬を与えず、最後の答えだけを判定する。推論の書き方は、正解に結びつくかどうかを通じて間接的に学ばれるだけで、上の「形が直った例」でも、学習後の途中の計算は筋が通らないのに最後の答えは正解になっている。そのレシピでは、形の違う答えも取り出せるように取り出し方を工夫したが、ここでは答えの形そのものを LoRA の重みに覚えさせた。正解の判定には、そのレシピの取り出し方より厳しい「決まった形」を使っている。取り出し方が緩いと、途中に出てきた数値がたまたま正解と一致しただけの応答にも報酬を与えてしまう。
- 時間とメモリ: 検証に使った GPU（T4 と同じ float16 の経路）では学習が 497 秒、全体で約 11 分、GPU メモリの最大使用量は 10.5GB だった。T4 では応答の生成が遅いので、この 2〜2.5 倍程度かかると見込んでいる（T4 では測っていない）。応答の生成は `use_vllm=True` で vLLM を使うと速くなるとされるが、Colab の torch などを入れ替えることになるので、ここでは使っていない（確かめていない）。`GRPOConfig` は trl 1.18.0 では `beta=0`（参照モデルとの KL の罰則なし、参照モデルを読み込まない）、学習率 1e-6、`bf16=True`、`gradient_checkpointing=True`、`loss_type="dapo"` が既定で、版によって変わることがある。
- データ: 学習用の SakanaAI/gsm8k-ja-test_250-1319 (Apache 2.0) は、データセットカードによると GPT-4 (gpt-4-0125-preview) で訳したもので、評価用の MGSM と重ならないように GSM8K のテストの 251 問目以降だけを使っている。どちらも GSM8K のテストの問題なので、ここで学習したモデルを GSM8K で評価すると、学習した問題で評価することになる。MGSM のライセンスと、評価データが事前学習に含まれている可能性については、この章の「考える過程を書かせて LLM に算数の文章題を解かせる（Chain-of-Thought と self-consistency）」の注意書きを参照。トークナイザーの修正については、この章の「オープンな LLM とチャットする（チャットテンプレート）」の注意書きを参照。
