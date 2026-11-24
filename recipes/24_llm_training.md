# LLM の学習と調整

小さなオープンモデルを学習・調整して、望む応答を出させる。ベースモデルの LoRA による指示チューニング (SFT)、選好データによる調整 (DPO)、選好データからの報酬モデルの学習、報酬モデルで応答を採点して最良を選ぶ方法 (Best-of-N)、正解を判定できる課題での強化学習 (GRPO)、学習した複数のアダプタを重みの足し引きで合成・調整するタスク算術（モデルマージ）。モデルの動かし方とプロンプトは「大規模言語モデル (LLM)」の章、判定役の LLM による評価は「LLM の評価と分析」の章を前提にする。

## ベースモデルを LoRA で指示チューニングする (SFT) {lora_finetune gpu}

指示チューニング前のベースモデルは、質問を渡しても答えて止まるとは限らない。公開の日本語の指示データで、LoRA により少ないメモリで指示チューニング (SFT) を行い、学習前・学習後・公式の指示チューニング版の応答を同じ指示で比べたい。

```python
!pip install -q trl==1.18.0 peft==0.21.2 torchao==0.18.0   # Colab に入っている古い torchao (0.10) のままだと peft の import が失敗する
```

```python
import json, os, re, time, unicodedata
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

def load_tokenizer(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    # transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
    # 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
    sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
    tokenizer.backend_tokenizer.model = models.Unigram(
        [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
    return tokenizer

set_seed(0)
base_name = "sbintuitions/sarashina2.2-0.5b"                   # 指示チューニング前のベースモデル（SB Intuitions、MIT License）
inst_name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"     # 同じベースモデルから作られた公式の指示チューニング版 (MIT License)
tokenizer = load_tokenizer(base_name)
print("ベースモデルのチャットテンプレート:", tokenizer.chat_template)

# ベースモデルにはチャットテンプレートがないので、公式の指示チューニング版と同じ形式のものを自分で設定する。
# 形式: <|system|>…</s><|user|>…</s><|assistant|>…</s>（<|user|> などは語彙に追加されたトークン）
tokenizer.chat_template = ("{% for m in messages %}{{ '<|' + m['role'] + '|>' + m['content'] + eos_token }}{% endfor %}"
                           "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}")
sample = [{"role": "system", "content": "S"}, {"role": "user", "content": "こんにちは"}, {"role": "assistant", "content": "はい。"}]
official = AutoTokenizer.from_pretrained(inst_name)   # 公式のテンプレートと同じ文字列になるかだけを確かめる
assert tokenizer.apply_chat_template(sample, tokenize=False) == official.apply_chat_template(sample, tokenize=False)
print(tokenizer.apply_chat_template(sample[1:2], tokenize=False, add_generation_prompt=True))

BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
DTYPE = torch.bfloat16 if BF16 else torch.float16
model = AutoModelForCausalLM.from_pretrained(base_name, dtype=DTYPE, device_map="auto")
print(f"パラメータ数: {model.num_parameters() / 1e6:.0f}M（埋め込みと出力層を含む）")
```

```python
from datasets import Dataset, load_dataset

# 指示データ: llm-jp/databricks-dolly-15k-ja (CC BY-SA 3.0)。Databricks の社員が書いた databricks-dolly-15k
# （指示と応答の組、CC BY-SA 3.0）を、LLM-jp が DeepL で日本語に訳したもの。context は Wikipedia から引用した参考文
raw = load_dataset("llm-jp/databricks-dolly-15k-ja", split="train")
print(len(raw), "件", raw.to_pandas()["category"].value_counts().to_dict())

def to_example(r):
    user = r["instruction"] if not r["context"] else f"{r['instruction']}\n\n{r['context']}"
    # trl の「会話形式の prompt-completion」: prompt と completion にメッセージのリストを入れる
    return {"prompt": [{"role": "user", "content": user}], "completion": [{"role": "assistant", "content": r["response"]}]}

MAX_LENGTH = 512
rows = []
too_long = 0
for r in raw:
    if not r["response"].strip():
        continue
    example = to_example(r)
    messages = example["prompt"] + example["completion"]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                        truncation=False, return_dict=False)
    if len(ids) > MAX_LENGTH:
        too_long += 1
    else:
        rows.append(r)
print(f"応答ありのうち {MAX_LENGTH} トークン超過: {too_long} 件、採用候補: {len(rows)} 件")
if len(rows) < 8240:
    raise ValueError("固定の学習・評価分割に必要な 8,240 件に足りません")
rng = np.random.default_rng(0)
rows = [rows[i] for i in rng.permutation(len(rows))]
train_ds = Dataset.from_list([to_example(r) for r in rows[:8000]])
eval_ds = Dataset.from_list([to_example(r) for r in rows[8000:8200]])   # 学習に使わない 200 件（損失の比較用）
heldout = rows[8200:8240]                                               # 学習に使わない 40 件（応答の比較用）
print(len(rows), "件のうち 学習", len(train_ds), "件")
print(train_ds[0])
```

```python
# 自動評価の指示: 決まった形を守ったかを規則で判定できる指示を 6 種類 × 10 問用意する（IFEval のような考え方）
def n_bullets(t):
    return sum(bool(re.match(r"\s*([-・*•]|\d+[.)．、])", l)) for l in t.splitlines()) == 3
def strip_fence(t):
    return re.sub(r"^```(json)?\s*|\s*```$", "", t.strip())
def is_json(t, keys):
    try:
        d = json.loads(strip_fence(t))
        return isinstance(d, dict) and set(d) == set(keys)
    except ValueError:
        return False

CHECKS = {
    "箇条書き 3 つ": ("{}を 3 つ、箇条書きで挙げてください。説明は不要です。", n_bullets,
                   ["夏に食べたい果物", "朝の時間を有効に使う方法", "図書館でできること", "雨の日に家の中でできる遊び", "自転車に乗るときの注意点",
                    "旅行の持ち物", "部屋を片付けるこつ", "野菜を使った料理", "眠れないときに試すこと", "冬に楽しめるスポーツ"]),
    "30 文字以内": ("{}を 30 文字以内で説明してください。", lambda t: 0 < len(t) <= 30,
                 ["虹ができる仕組み", "冷蔵庫の役割", "俳句とは何か", "光合成", "地震が起きたときに最初にすること",
                  "月の満ち欠け", "辞書の使い道", "電子レンジの仕組み", "消しゴムで字が消える理由", "植物に水が必要な理由"]),
    "英語で": ("次の質問に英語で答えてください。{}", lambda t: len(re.findall(r"[A-Za-z]", t)) >= 3 and not re.search(r"[ぁ-んァ-ン一-龥]", t),
              ["日本でいちばん高い山は何ですか。", "1 年は何日ですか。", "好きな季節とその理由を教えてください。", "お茶とコーヒーの違いは何ですか。", "猫はどんな動物ですか。",
               "朝ごはんに何を食べるとよいですか。", "雨はなぜ降るのですか。", "図書館では何ができますか。", "健康のためにできることを教えてください。", "富士山はどこにありますか。"]),
    "はい/いいえ": ("次の質問に「はい」か「いいえ」の一言だけで答えてください。{}", lambda t: re.sub(r"[。.!！\s]", "", t) in ("はい", "いいえ"),
                 ["水は 100 度で沸騰しますか。", "太陽は西から昇りますか。", "1 週間は 7 日ですか。", "ペンギンは空を飛べますか。", "東京は日本の首都ですか。",
                  "氷は水に浮きますか。", "クジラは魚ですか。", "1 年は 12 か月ですか。", "月は自分で光っていますか。", "人間の心臓は 1 つですか。"]),
    "数字だけ": ("次の問いの答えを数字だけで答えてください。{}", lambda t: re.fullmatch(r"-?\d+(\.\d+)?", unicodedata.normalize("NFKC", t).strip(" 。.")) is not None,
               ["12 + 30 は？", "1 ダースは何個？", "1 時間は何分？", "7 × 8 は？", "三角形の内角の和は何度？",
                "1 キロメートルは何メートル？", "100 − 37 は？", "1 日は何時間？", "正方形の辺の数は？", "15 ÷ 3 は？"]),
    "JSON": ("次の文から情報を取り出し、キーが name と age の JSON だけを出力してください。{}", lambda t: is_json(t, ["name", "age"]),
             ["田中さんは 28 歳です。", "佐藤花子は今年 35 歳になった。", "12 歳の鈴木くんがサッカーをしている。", "山本さん（41 歳）は会社員です。", "45 歳の高橋さんが受付にいた。",
              "伊藤さんは 19 歳の大学生です。", "渡辺さん、52 歳。", "中村さんは先月 30 歳になりました。", "7 歳の小林さんは小学生だ。", "加藤さんの年齢は 63 歳です。"]),
}
if_items = [(kind, tmpl.format(x), check) for kind, (tmpl, check, xs) in CHECKS.items() for x in xs]
kinds = [k for k, _, _ in if_items]
if_prompts = [[{"role": "user", "content": p}] for _, p, _ in if_items]
dolly_prompts = [to_example(r)["prompt"] for r in heldout]

@torch.no_grad()
def generate(model, prompts, max_new_tokens=256, batch_size=40):
    """貪欲法で生成し、応答と「</s> を出して止まったか」を返す"""
    model.eval()
    tokenizer.padding_side = "left"
    outs, stopped = [], []
    for i in range(0, len(prompts), batch_size):
        enc = tokenizer.apply_chat_template(prompts[i:i + batch_size], add_generation_prompt=True, return_tensors="pt",
                                            return_dict=True, padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        outs += [tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen]
        stopped += [(g == tokenizer.eos_token_id).any().item() for g in gen]
    return outs, np.array(stopped)

def evaluate(model, label):
    t = time.time()
    if_out, if_stop = generate(model, if_prompts)
    d_out, d_stop = generate(model, dolly_prompts)
    hits = np.array([check(o) for (_, _, check), o in zip(if_items, if_out)])
    print(f"{label}: 指示に従った割合 {hits.mean():.3f}、止まった割合 {np.r_[if_stop, d_stop].mean():.3f}（{time.time() - t:.0f} 秒）")
    return {"if_out": if_out, "hits": hits, "d_out": d_out, "stopped": np.r_[if_stop, d_stop]}

results = {"学習前": evaluate(model, "学習前（ベースモデル）")}   # Colab の T4 で 1〜2 分
for k in ["箇条書き 3 つ", "はい/いいえ", "JSON"]:
    i = kinds.index(k)
    print(f"\n[{if_items[i][1]}]\n→ {results['学習前']['if_out'][i][:200]}")

# 参考: チャットの形式を使わず、ふつうの文章の続きとして書かせる
enc = tokenizer("質問: 日本でいちばん高い山は何ですか。\n答え:", return_tensors="pt").to(model.device)
out = model.generate(**enc, max_new_tokens=60, do_sample=False, repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id)
print("\n[文章の続きとして]", repr(tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)))
```

```python
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])   # すべての線形層に低ランク行列を足す
args = SFTConfig(
    output_dir="sft-out", num_train_epochs=1, per_device_train_batch_size=8, gradient_accumulation_steps=2,
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=10, max_length=MAX_LENGTH,
    completion_only_loss=True,    # prompt-completion 形式のデータでは既定でも True。応答 (completion) の部分だけに損失をかける
    packing=False,                # 複数の例を 1 本の系列に詰めない（下の注意書きを参照）
    logging_steps=50, save_strategy="no", report_to="none", seed=0,
    gradient_checkpointing=False, bf16=BF16, fp16=not BF16)   # SFTConfig は bf16=True と gradient_checkpointing=True が既定なので明示する
trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tokenizer, peft_config=lora)
trainer.model.print_trainable_parameters()

# 損失をかける部分の確認: SFTTrainer が作った labels で、プロンプトの部分は -100（損失を計算しない）になっている
ex = trainer.train_dataset[0]
ids, labels = np.array(ex["input_ids"]), np.array(ex["labels"])
print("損失なし:", repr(tokenizer.decode(ids[labels == -100])[-60:]))
print("損失あり:", repr(tokenizer.decode(ids[labels != -100])))
lengths = np.array([len(x) for x in trainer.train_dataset["input_ids"]])
print(f"トークン数: 平均 {lengths.mean():.0f}、最大 {lengths.max()}")
# 学習と同じくランダムに 8 件ずつまとめたときに、パディングが占める割合（packing をすると減らせる分）
pad = [1 - b.sum() / (len(b) * b.max()) for b in np.array_split(lengths[rng.permutation(len(lengths))], len(lengths) // 8)]
print(f"パディングの割合（ランダムに 8 件ずつ）: {np.mean(pad):.2f}")
```

```python
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

# packing の確認: trl の packing は複数の例を 1 本につなぎ、位置 (position_ids) を例ごとに 0 から振り直す。
# 2 件をつないだときの 2 件目の出力が、2 件目だけを入れたときと同じになるか（前の例を見ていないか）を確かめる
a, b = trainer.train_dataset[0]["input_ids"], trainer.train_dataset[1]["input_ids"]
batch = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id, padding_free=True)([{"input_ids": a}, {"input_ids": b}])
ids, pos = batch["input_ids"].to(model.device), batch["position_ids"].to(model.device)
trainer.model.eval()
with torch.no_grad():
    alone = trainer.model(input_ids=torch.tensor([b], device=model.device)).logits[0].float()
    for label, kw in [("位置を振り直す（学習時と同じくキャッシュなし）", dict(position_ids=pos, use_cache=False)),
                      ("単純につなぐだけ", dict(use_cache=False))]:
        packed = trainer.model(input_ids=ids, **kw).logits[0, len(a):].float()
        print(f"2 件目の logits の最大の差（{label}）: {(packed - alone).abs().max().item():.2f}")
print("アテンションの実装:", model.config._attn_implementation)
```

```python
loss_before = trainer.evaluate()["eval_loss"]   # 学習前（LoRA の B は 0 で初期化されるので、ベースモデルと同じ）
t = time.time()
trainer.train()   # Colab の T4 で 10〜15 分
for log in trainer.state.log_history:
    if "loss" in log:
        print(f"step {log['step']:3d}  loss {log['loss']:.3f}")
print(f"評価データ 200 件の応答部分の損失: 学習前 {loss_before:.3f} → 学習後 {trainer.evaluate()['eval_loss']:.3f}")
print(f"学習 {time.time() - t:.0f} 秒、GPU メモリの最大使用量 {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
# 参考: 全パラメータを Adam で学習すると、重み・勾配・Adam の状態だけで 1 パラメータあたり 16 バイト程度（混合精度の場合）
n_all = sum(p.numel() for p in trainer.model.parameters())
n_train = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
print(f"学習するパラメータ {n_train / 1e6:.1f}M（全体の {n_train / n_all:.2%}）。全パラメータを学習する場合の目安 {n_all * 16 / 1e9:.1f} GB + 活性化")
```

```python
# 学習後のモデルと公式の指示チューニング版を、同じ指示で評価する
model = trainer.model
results["学習後 (SFT)"] = evaluate(model, "学習後（LoRA で SFT）")   # 公式版の読み込みと合わせて Colab の T4 で 1〜3 分
inst = AutoModelForCausalLM.from_pretrained(inst_name, dtype=DTYPE, device_map="auto")
results["公式の instruct"] = evaluate(inst, "公式の指示チューニング版")

def rate_ci(hits):   # 割合と、ブートストラップによる 95% 信頼区間
    boot = hits[np.random.default_rng(0).integers(0, len(hits), (2000, len(hits)))].mean(1)
    return f"{hits.mean():.2f} [{np.percentile(boot, 2.5):.2f}, {np.percentile(boot, 97.5):.2f}]"

table = pd.DataFrame({label: {**{k: r["hits"][[i for i, x in enumerate(kinds) if x == k]].mean() for k in CHECKS},
                              "指示に従った割合（60 問）": rate_ci(r["hits"]), "止まった割合（100 問）": r["stopped"].mean().round(2),
                              "応答の平均の長さ（文字）": round(np.mean([len(o) for o in r["if_out"] + r["d_out"]]))}
                      for label, r in results.items()})
print(table.to_string())
print(f"GPU メモリの最大使用量 {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

```python
# 同じ指示への 3 つの応答を並べる（形を指定した指示 2 問と、学習に使っていない dolly の指示 2 問）
for p, outs in [(if_items[i][1], [r["if_out"][i] for r in results.values()]) for i in [kinds.index("30 文字以内"), kinds.index("数字だけ")]] + \
               [(dolly_prompts[i][0]["content"], [r["d_out"][i] for r in results.values()]) for i in [0, 1]]:
    print("■", p[:150])
    for label, o in zip(results, outs):
        print(f"  [{label}] {o[:150]}")
    print()

model.save_pretrained("sft-lora")   # 保存されるのは追加した LoRA の重みだけ
size = sum(os.path.getsize(f"sft-lora/{f}") for f in os.listdir("sft-lora") if f.endswith(".safetensors"))
print(f"LoRA の重みのファイル: {size / 1e6:.0f} MB")
```

- 指示チューニング (SFT: supervised fine-tuning) は、指示と応答の組で、応答の部分を次のトークンの予測として学習させること。上の出力では、ベースモデルはチャットの形式で渡した指示に英語の文章やプログラムを書き続け、`</s>` を出して止まったのは 100 問中 1 問だった。ふつうの文章の続きとしてなら答えるが、次の質問と答えを自分で作り続ける。8,000 件・1 エポックの SFT のあとは止まった割合が 0.92 になり（公式の指示チューニング版は 0.94）、訓練時の損失は step 50 の 2.439 から step 100 で 2.294 に下がったあとは 2.20〜2.32 付近で推移し（step 500 で 2.281）、学習に使っていない 200 件の応答部分の損失は 2.922 → 2.273 に下がった。なお、ベースモデルのモデルカードには、事前学習の最後の段階で「さまざまな応用タスクの性能を上げるための少量のデータ」で学習したとあり、純粋に Web のテキストだけで学習したモデルではない。
- 決まった形を守ったかの自動評価（60 問）では、指示に従った割合は学習前 0.15、学習後 0.22、公式版 0.20 で、95% 信頼区間が重なり、この問題数では差があるとは言えない。学習前の 0.15 は、何を聞かれても英語を書くので「英語で」の 10 問中 9 問が当たっただけである。学習後は「箇条書き 3 つ」が 0.9 と公式版 (0.2) より高い一方、「JSON」は 0（公式版は 1.0）で、「はい/いいえ」の一言は 3 つとも 0 だった。SFT で身につくのは主に学習データの形と文体で、学習後の応答は短く（平均 92 文字、公式版は 247 文字）、学習データと同じである調になった。dolly には JSON で答える例がほとんどない。「12 + 30」に「15」と答えるように、0.5B のモデルに 8,000 件学習させただけでは中身の正しさは上がらない。決まった形で答えさせたいなら、その形の例を学習データに入れる。中身の良し悪しを比べるには、LLM の評価と分析の章の「手元の LLM に生成文を採点させる (LLM-as-a-judge)」のような判定を使う。
- 損失をかける部分: 会話形式の prompt-completion（`prompt` と `completion` にメッセージのリスト）のデータを渡すと、`SFTTrainer` はプロンプトの部分の `labels` を -100 にして、応答とその後の `</s>` だけに損失をかける（`completion_only_loss` は、この形式では既定で True）。`</s>` を学習するので、答えたあとに止まれるようになる。複数ターンの会話（`messages` の列）で応答だけに損失をかける `assistant_only_loss=True` は、チャットテンプレートに応答の範囲を示す `{% generation %}` のタグが要り、sarashina2.2 の公式のテンプレートにも、ここで設定したテンプレートにもない（trl は、学習用のテンプレートを用意してある一部のモデルでは自動で差し替えるが、sarashina2.2 はその対象ではない）。ベースモデルにはチャットテンプレートがないので、公式版と同じ形式のものを設定した（`<|user|>` などはベースモデルの語彙にも同じ ID で追加されている。特殊トークンの扱いではないので、`skip_special_tokens=True` でも消えない）。こうすると、学習したモデルを公式版と同じ使い方で動かせる。
- packing: トークン数は平均 126（最大 512）で、長さがばらばらな例を 8 件ずつまとめると、上の出力ではトークンの 58% がパディングになる。`packing=True` にすると複数の例を 1 本の系列に詰めてパディングを減らせる。trl は例ごとに位置を 0 から振り直し、transformers 5 は位置の切れ目から例どうしのアテンションを遮る（学習時のようにキャッシュを使わない場合）。上の確認では、位置を振り直したときの 2 件目の logits の差は 0.04（float16 の誤差の程度）、単純につなぐと 16.16 で、前の例が見えてしまう。trl の説明では packing (padding-free) は FlashAttention 2/3 が前提とされているが、この版の transformers では sdpa でも上のとおり遮られた（版によって変わりうる）。ここでは、例ごとに扱えて分かりやすいように packing を使っていない。packing をすると 1 本に複数の例が入り、同じデータでもステップ数が減るので、学習率やステップ数を合わせ直す必要がある（ここでは時間を比べていない）。
- LoRA (Hu et al., 2021) は、元の重みを固定し、線形層に低ランクの行列の積 BA（ランク `r`）を足して、A と B だけを学習する。B は 0 で初期化するので、学習前は元のモデルと同じ出力になる。`lora_alpha / r` が足す量の倍率。`target_modules` はアテンションの 4 つの射影と MLP の 3 つの線形層で、学習するパラメータは 10.1M（全体の 1.25%）、保存される重みは 40MB だった。学習中の GPU メモリの最大使用量は 10.4GB。全パラメータを Adam で学習すると、混合精度では重み・勾配・Adam の状態だけで 1 パラメータあたり 16 バイト程度 (Rajbhandari et al., 2020)、このモデルで 12.8GB に活性化の分が加わり、T4 (16GB) では厳しい。4bit 量子化と組み合わせる (QLoRA) と、もっと大きなモデルも T4 で学習できる（`prepare_model_for_kbit_training` を使う）。
- 読み込みは `PeftModel.from_pretrained(ベースモデル, "sft-lora")`、推論を速くするなら `merge_and_unload()` で重みを統合する。LoRA の学習率は全パラメータの学習より高め（ここでは 2e-4）。`SFTConfig` は trl 1.18.0 では `bf16=True`、`gradient_checkpointing=True`、学習率 2e-5 が既定なので、T4 向けに fp16 を明示している。Colab の T4 GPU では学習が 761 秒（約 13 分）、学習前後の 3 回の評価が各 29〜35 秒で、ノートブック全体の実行は約 17 分だった。
- ベース → SFT → 選好や報酬による調整、という順番がふつうである。この章の「選好データで LLM を調整する (DPO)」と「正解を判定できる課題で LLM を強化学習する (GRPO)」は、公式の指示チューニング版から始めている（その学習方法はモデルカードに書かれていない）。ここで作ったモデルから始めるなら、`merge_and_unload()` で統合したモデルを渡し、トークナイザーはここで設定したチャットテンプレートを持つもの（`tokenizer.save_pretrained` で保存したもの）か公式版のものを使う。ベースモデルのトークナイザーのままではチャットテンプレートがない。SFT は「手本の応答をまねる」学習で、手本より良い応答は学べない。DPO は 2 つの応答のどちらが良いか、GRPO は正解かどうかで、手本のない改善を目指す。
- データ: llm-jp/databricks-dolly-15k-ja (CC BY-SA 3.0) は、Databricks の社員が書いた databricks-dolly-15k を LLM-jp が DeepL で訳したもので、訳の崩れも混じる。元のデータセットカードによると、要約や閉じた QA などの参考文 (context) は Wikipedia から引用したもので、引用の番号（[42] など）が残っていることがある。CC BY-SA は改変したデータを同じライセンスで公開することを求めるので、データを加工して配布するときは注意する。モデルはどちらも MIT License。トークナイザーの修正 (`load_tokenizer`) については、大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」の注意書きを参照。

## 選好データで LLM を調整する (DPO) {dpo gpu}

「良い応答」と「悪い応答」の組（選好データ）から、LLM がより好まれる応答を出すように調整したい。報酬モデルを別に作らずに済む DPO を、LoRA と組み合わせて行う。

```python
!pip install -q trl==1.18.0 peft==0.21.2 torchao==0.18.0   # Colab に入っている古い torchao (0.10) のままだと peft の import が失敗する
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
    # 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
    sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
    tokenizer.backend_tokenizer.model = models.Unigram(
        [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
    return tokenizer

set_seed(0)
name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"   # 0.5B の指示チューニング済みモデル（この章の「ベースモデルを LoRA で指示チューニングする (SFT)」で比べた公式版と同じ、MIT License）
tokenizer = load_tokenizer(name)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto")
```

```python
from datasets import Dataset, load_dataset

# 選好データ: llm-jp/hh-rlhf-12k-ja (MIT License)。Anthropic の hh-rlhf（Bai et al., 2022、MIT License）から
# 12,000 件を選び、DeepL で日本語に訳したもの。会話 (conversations) に対して、人が選んだ応答 (chosen) と選ばなかった応答 (rejected) がある
raw = load_dataset("llm-jp/hh-rlhf-12k-ja", split="train")
print(raw.to_pandas()["source"].value_counts())

# 有用さ (helpful-*) の部分だけを使う（harmless-base は有害な依頼とその応答で、不快な内容を含む）。
# 1 往復の会話で、短めのものに絞る。trl の会話形式（prompt / chosen / rejected にメッセージのリスト）にする
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

# 調整前（= DPO の参照モデル）の対数尤度と、評価用の質問 100 件への応答
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
# ref_model を渡さずに peft_config を渡すと、LoRA を外した元のモデルが参照モデルになる（モデルを 2 つ読み込まずに済む）
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
# 調整前と後の応答を、大きめのモデル（sarashina2.2-3b-instruct、MIT License）に比べさせる（LLM による判定）。
# 応答を A と B に並べ、次のトークンが「A」と「B」になる確率を比べる。位置の偏りを打ち消すため、順番を入れ替えて 2 回聞いて平均する
judge_name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
jtok = load_tokenizer(judge_name)
judge = AutoModelForCausalLM.from_pretrained(judge_name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()
id_a, id_b = [jtok(c, add_special_tokens=False)["input_ids"][-1] for c in ("A", "B")]   # LLM の評価と分析の章の「手元の LLM に生成文を採点させる (LLM-as-a-judge)」と同じ取り方
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

# まず、判定役が人の選好（評価データの chosen / rejected）とどれくらい一致するかを確かめる。Colab の T4 で 2〜3 分
agree = np.array([p_first_better(ex["prompt"][0]["content"], ex["chosen"][0]["content"], ex["rejected"][0]["content"]) > 0.5 for ex in test])
print("判定役と人の選好の一致率:", rate_ci(agree))

win = np.array([p_first_better(p[0]["content"], a, b) for p, b, a in zip(gen_prompts, before, after)])
print("調整後の応答が調整前より良いと判定された割合 (勝率):", rate_ci(win > 0.5))
longer = np.array([len(a) > len(b) for b, a in zip(before, after)])   # 判定役が長い応答を好んでいないか
print(f"勝率の内訳: 調整後のほうが長い {longer.sum()} 問で {(win[longer] > 0.5).mean():.3f}、"
      f"長くならなかった (短いか同じ) {(~longer).sum()} 問で {(win[~longer] > 0.5).mean():.3f}")
print(f"GPU メモリの最大使用量: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

- DPO (Direct Preference Optimization; Rafailov et al., 2023) は、選好データから報酬モデルを学習して強化学習 (PPO) で最適化する RLHF の手順を、1 つの分類のような損失に置き換えたもの。参照モデルからのずれを β で抑えながら、chosen の尤度を rejected より相対的に上げる。実際には、この章の「ベースモデルを LoRA で指示チューニングする (SFT)」のような指示チューニングの後に行う。
- 上の出力では、評価データ（学習に使っていない 200 組）で報酬が chosen > rejected になった割合は 0.680（95% 信頼区間 0.615〜0.745）で、偶然 (0.5) より高い。ただし DPO が上げるのは chosen と rejected の差だけで、対数尤度そのものは chosen も rejected も上がり（平均で +12.80 と +6.95）、「chosen のほうが尤度が高い」割合は 0.480 → 0.485 とほとんど変わらなかった。
- 一方、生成した応答を判定役のモデルに比べさせると、調整後が良いとされたのは 100 問中 0.320（95% 信頼区間 0.230〜0.410）で、調整前より悪いと判定された。応答は短くなり（平均 480 → 445 文字）、長くならなかった 66 問では勝率が 0.288 と、長くなった 34 問 (0.382) より低い。GPU の計算は実行ごとにわずかに変わることがあり、勝率は小数第 2 位で変わりうるので、100 問の信頼区間の幅で見る。選好の正解率が上がっても、生成が良くなるとは限らない。
- 原因は確かめていないが、学習データの応答は hh-rlhf の論文（2022 年）のモデルの出力を機械翻訳したもので、ここでは 400 文字以下に絞っており、このモデル自身の応答とは文体も長さも違う（方策外のデータ）。調整するモデル自身に応答を複数生成させ、それに人や強いモデルが選好を付けたデータ（方策上のデータ）を使う方法もある。
- 判定役 (3B) と人の選好の一致率は 0.650 程度で、判定はあくまで目安。A と B の順番を入れ替えて平均し、位置の偏りを打ち消している。判定役の信頼性の確かめ方は、LLM の評価と分析の章の「手元の LLM に生成文を採点させる (LLM-as-a-judge)」を参照。
- データ: hh-rlhf の元のデータセットカードには、選好（報酬）モデルの学習用で、対話エージェントを教師ありで学習するためのものではない（有害なモデルになりうる）と書かれている。DPO は報酬モデルを作らず、対話モデルそのものをこの応答で直接学習するので、この注意が当てはまる。ここで作るモデルは手法を試すためのもので、公開や実用には使わない。harmless-base には差別や暴力などの不快な内容が含まれるので、ここでは使っていない。日本語は DeepL による翻訳で、訳の崩れも混じる（上の 1 件目の「」の対応など）。
- `DPOConfig` は、`TrainingArguments` と違って `bf16=True`、学習率 1e-6、`gradient_checkpointing=True` が既定。T4 では bf16 が使えないので fp16 を明示し、LoRA では学習率を上げる（ここでは 5e-5）。β を大きくすると参照モデルから離れにくくなる。`loss_type="ipo"` などで損失を IPO などに変えられる。trl 1.18.0 で確かめた（版によって引数や既定値が変わることがある）。
- GPU メモリの最大使用量は学習時の 9.2GB で、3B の判定役を読み込んだ後も変わらなかった。sarashina2.2 のトークナイザーの修正 (`load_tokenizer`) については、大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」の注意書きを参照。

## 選好データから報酬モデルを学習して応答を採点する（Bradley-Terry モデル） {reward_model_train gpu}

「良い応答」と「悪い応答」の組（選好データ）から、質問と応答に対して 1 つの実数スコアを返す報酬モデルを LoRA で学習したい。学習前後での選好の正解率や長さバイアスを確かめ、同じテストデータで公開の 1.7B 報酬モデルや文字数ベースライン・DPO と比べる。

```python
import random, time
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from scipy.stats import pearsonr, spearmanr
from tokenizers import models
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

def load_tokenizer(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    # transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
    # 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
    sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
    tokenizer.backend_tokenizer.model = models.Unigram(
        [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
    return tokenizer

if not torch.cuda.is_available():
    raise RuntimeError("Colab のランタイムを GPU に変更してください")
set_seed(0)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere 以降。T4 では float16 を使う
DTYPE = torch.bfloat16 if BF16 else torch.float16
name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"   # MIT License。DPO・Best-of-N・GRPO と同じ 0.5B モデル
tokenizer = load_tokenizer(name)

# num_labels=1 で読み込み、言語モデルヘッド (lm_head) の代わりにスカラー出力の線形層 score を付ける
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1, dtype=DTYPE, device_map="auto")
model.config.pad_token_id = tokenizer.pad_token_id    # 最後の非パディングトークン位置を特定するために必須
print("分類ヘッド:", model.score, "| pad_token_id:", model.config.pad_token_id)

# パディング有無で最後の非パディングトークンから同じスカラー報酬が出ることを確かめる
sample_ids = tokenizer("質問: おはようございます。", return_tensors="pt")["input_ids"].to(model.device)
padded_ids = torch.cat([sample_ids, torch.full((1, 4), tokenizer.pad_token_id, device=model.device)], dim=1)
padded_mask = torch.cat([torch.ones_like(sample_ids), torch.zeros((1, 4), dtype=torch.long, device=model.device)], dim=1)
with torch.no_grad():
    s_raw = model(input_ids=sample_ids).logits[0, 0].item()
    s_pad = model(input_ids=padded_ids, attention_mask=padded_mask).logits[0, 0].item()
print(f"パディングなし: {s_raw:+.4f} / 右パディングあり: {s_pad:+.4f}（差 {abs(s_raw - s_pad):.4f}）")

# ランダム初期化の score 層も学習対象に含めるため modules_to_save=["score"] を必ず指定する
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    modules_to_save=["score"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()
for pname, param in model.named_parameters():
    if "score" in pname:
        print(f"  {pname}: requires_grad={param.requires_grad}, shape={tuple(param.shape)}")
```

```python
# 選好データ: この章の DPO レシピと完全に同一の絞り込み・乱数シードで train（2,000 組）・test（200 組）を作る
raw = load_dataset("llm-jp/hh-rlhf-12k-ja", split="train")
data = [{"prompt": [{"role": "user", "content": r["conversations"][0]["value"]}],
         "chosen": [{"role": "assistant", "content": r["chosen"]}],
         "rejected": [{"role": "assistant", "content": r["rejected"]}]}
        for r in raw
        if r["source"].startswith("helpful") and len(r["conversations"]) == 1 and r["chosen"] != r["rejected"]
        and len(r["conversations"][0]["value"]) <= 200 and max(len(r["chosen"]), len(r["rejected"])) <= 400]
random.Random(0).shuffle(data)
train, test = data[:2000], data[2000:2200]
print(f"全 {len(data)} 件から 学習 {len(train)} 組・評価 {len(test)} 組（train で chosen が長い割合 {np.mean([len(x['chosen'][0]['content']) > len(x['rejected'][0]['content']) for x in train]):.3f}）")

MAX_LENGTH = 512
def encode_pair(ex):
    c_txt = tokenizer.apply_chat_template(ex["prompt"] + ex["chosen"], tokenize=False)
    r_txt = tokenizer.apply_chat_template(ex["prompt"] + ex["rejected"], tokenize=False)
    return (tokenizer(c_txt, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)["input_ids"],
            tokenizer(r_txt, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)["input_ids"])

train_enc = [encode_pair(ex) for ex in train]
test_enc = [encode_pair(ex) for ex in test]

def collate_pairs(pairs, pad_id):
    """chosen と rejected を 1 つのバッチに並べて右パディングする"""
    seqs = [c for c, _ in pairs] + [r for _, r in pairs]
    n = max(len(s) for s in seqs)
    x = torch.tensor([s + [pad_id] * (n - len(s)) for s in seqs])
    mask = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs])
    return x, mask

@torch.no_grad()
def evaluate_pairs(enc_pairs, bs=16):
    model.eval()
    rc, rr = [], []
    for i in range(0, len(enc_pairs), bs):
        batch = enc_pairs[i:i + bs]
        x, mask = collate_pairs(batch, tokenizer.pad_token_id)
        sc = model(input_ids=x.to(model.device), attention_mask=mask.to(model.device)).logits[:, 0].float()
        m = len(batch)
        rc.extend(sc[:m].tolist())
        rr.extend(sc[m:].tolist())
    return np.array(rc), np.array(rr)

def rate_ci(hits):
    idx = np.random.default_rng(0).integers(0, len(hits), (2000, len(hits)))
    boot = hits[idx].mean(1)
    return f"{hits.mean():.3f}（95% 信頼区間 {np.percentile(boot, 2.5):.3f}〜{np.percentile(boot, 97.5):.3f}）"

rc0, rr0 = evaluate_pairs(test_enc)
print(f"学習前（test {len(test)} 組）: 正解率 {rate_ci(rc0 > rr0)}、平均報酬差 {(rc0 - rr0).mean():+.3f}")

# Bradley-Terry 損失 L = -log sigmoid(r_chosen - r_rejected) で 1 エポック学習
BS, ACCUM, LR = 4, 2, 2e-4
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
total_steps = (len(train_enc) // BS + ACCUM - 1) // ACCUM
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)
order = list(range(len(train_enc)))
random.Random(0).shuffle(order)

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
model.train()
optimizer.zero_grad()
losses = []
for step_idx, start in enumerate(range(0, len(order), BS)):
    batch = [train_enc[i] for i in order[start:start + BS]]
    x, mask = collate_pairs(batch, tokenizer.pad_token_id)
    scores = model(input_ids=x.to(model.device), attention_mask=mask.to(model.device)).logits[:, 0].float()
    m = len(batch)
    loss = -F.logsigmoid(scores[:m] - scores[m:]).mean()
    (loss / ACCUM).backward()
    losses.append(loss.item())
    if (step_idx + 1) % ACCUM == 0 or start + BS >= len(order):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    if (step_idx + 1) % 100 == 0:
        print(f"  batch {step_idx + 1:3d}/{len(order) // BS}  loss {np.mean(losses[-100:]):.3f}")

train_sec = time.time() - t0
rc1, rr1 = evaluate_pairs(test_enc)
print(f"学習 {train_sec:.1f} 秒、GPU メモリの最大使用量 {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
print(f"学習後（test {len(test)} 組）: 正解率 {rate_ci(rc1 > rr1)}、平均報酬差 {(rc1 - rr1).mean():+.3f}")
```

```python
# 長さバイアス診断、文字数ベースライン・DPO（参照値）・Skywork-1.7B との比較、具体例での採点挙動
model.eval()
len_c = np.array([len(ex["chosen"][0]["content"]) for ex in test])
len_r = np.array([len(ex["rejected"][0]["content"]) for ex in test])
len_diff = len_c - len_r
rm_diff = rc1 - rr1
hits_rm = rm_diff > 0
hits_len = len_c > len_r
longer = len_c > len_r

print(f"報酬差と応答文字数差の相関: Pearson r = {pearsonr(rm_diff, len_diff).statistic:.3f}, "
      f"Spearman rho = {spearmanr(rm_diff, len_diff).statistic:.3f}")
print(f"chosen が長い {longer.sum()} 組の正解率: {hits_rm[longer].mean():.3f} / "
      f"長くない {(~longer).sum()} 組の正解率: {hits_rm[~longer].mean():.3f}")

# 公開の 1.7B 報酬モデル (Skywork-Reward-V2-Qwen3-1.7B, Apache 2.0) を同じ test 200 組で評価する
rm_name = "Skywork/Skywork-Reward-V2-Qwen3-1.7B"
sky_tok = AutoTokenizer.from_pretrained(rm_name)
sky_rm = AutoModelForSequenceClassification.from_pretrained(rm_name, dtype=DTYPE, device_map="auto").eval()
t_sky = time.time()
sc_sky, sr_sky = [], []
with torch.no_grad():
    for i in range(0, len(test), 8):
        batch = test[i:i + 8]
        texts = [sky_tok.apply_chat_template(ex["prompt"] + ex["chosen"], tokenize=False) for ex in batch] + \
                [sky_tok.apply_chat_template(ex["prompt"] + ex["rejected"], tokenize=False) for ex in batch]
        enc = sky_tok(texts, return_tensors="pt", padding=True).to(sky_rm.device)
        sc = sky_rm(**enc).logits[:, 0].float()
        m = len(batch)
        sc_sky.extend(sc[:m].tolist())
        sr_sky.extend(sc[m:].tolist())
sc_sky, sr_sky = np.array(sc_sky), np.array(sr_sky)
hits_sky = sc_sky > sr_sky
print(f"\nSkywork-1.7B（{time.time() - t_sky:.1f} 秒）: 正解率 {rate_ci(hits_sky)}、平均報酬差 {(sc_sky - sr_sky).mean():+.3f}")
print(f"文字数ベースライン: 正解率 {rate_ci(hits_len)}")
print(f"DPO（この章の DPO レシピ参照値）: 正解率 0.680（95% 信頼区間 0.615〜0.745）")
print(f"学習した 0.5B 報酬モデルと Skywork-1.7B の選好一致率: {(hits_rm == hits_sky).mean():.3f}、"
      f"報酬差の順位相関: {spearmanr(rm_diff, sc_sky - sr_sky).statistic:.3f}")

@torch.no_grad()
def score_responses(prompt_text, responses):
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt_text}, {"role": "assistant", "content": r}], tokenize=False)
             for r in responses]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False).to(model.device)
    return model(**enc).logits[:, 0].float().cpu().numpy()

# 具体例での採点挙動: Best-of-N レシピと同じりんごの問題・定型締め句の追加・半分削除
q = "りんごが 12 個あります。4 個を友達にあげ、1 個買い足して、自分と兄弟 2 人の 3 人で等しく分けます。1 人何個もらえますか。"
cands = {
    "正しい（日本語）": "12 − 4 = 8 個。1 個買って 8 + 1 = 9 個。3 人で分けるので 9 ÷ 3 = 3。答えは 1 人 3 個です。",
    "誤り（日本語）": "12 − 4 = 8 個。1 個買って 8 + 1 = 9 個。兄弟 2 人で分けるので 9 ÷ 2 = 4.5。答えは 1 人 4 個です。",
    "定型締め句を追加": "12 − 4 = 8 個。1 個買って 8 + 1 = 9 個。3 人で分けるので 9 ÷ 3 = 3。答えは 1 人 3 個です。\n\nほかにもご質問があれば、お気軽にお尋ねください。",
    "後半を削除（半分）": "12 − 4 = 8 個。1 個買って 8 + 1 = 9 個。",
}
print("\nりんごの例での採点:")
for k, s in zip(cands, score_responses(q, list(cands.values()))):
    print(f"  {s:+7.3f}  {k}")

sub = test[:50]
orig_s = rc1[:50]
tail_s = np.array([score_responses(ex["prompt"][0]["content"], [ex["chosen"][0]["content"] + "\n\nほかにもご質問があれば、お気軽にお尋ねください。"])[0] for ex in sub])
half_s = np.array([score_responses(ex["prompt"][0]["content"], [ex["chosen"][0]["content"][:max(1, len(ex["chosen"][0]["content"]) // 2)]])[0] for ex in sub])
print(f"\ntest 先頭 50 件での言い回し操作:")
print(f"  定型締め句を追加: 平均 {(tail_s - orig_s).mean():+.3f}（上昇 {(tail_s > orig_s).mean():.1%}）")
print(f"  後半を半分削除  : 平均 {(half_s - orig_s).mean():+.3f}（上昇 {(half_s > orig_s).mean():.1%}）")
print(f"GPU メモリの最大使用量: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

- Bradley-Terry モデルでは、質問 $x$ に対する 2 つの応答 $y_w$ (chosen) と $y_l$ (rejected) について、$y_w$ が選ばれる確率をシグモイド関数 $\sigma\bigl(r_\theta(x, y_w) - r_\theta(x, y_l)\bigr)$ で表し、負の対数尤度 $\mathcal{L} = -\log \sigma\bigl(r_\theta(x, y_w) - r_\theta(x, y_l)\bigr)$ を最小化してスカラー報酬 $r_\theta(x, y)$ を学習する。上の実行では、学習前は偶然どおり 0.495（95% 信頼区間 0.425〜0.565、平均報酬差 +0.006）だった選好の正解率が、2,000 組・1 エポックの学習（76.5 秒、GPU メモリの最大使用量 7.1 GB）により 0.635（95% 信頼区間 0.570〜0.700、平均報酬差 +0.318）に上がった。訓練損失は batch 100 の 1.518 から batch 500 の 0.619 へ下がった。なお Bradley-Terry 損失は $r_\theta(x, y)$ に定数を足しても差 $r_w - r_l$ が変わらない（平行移動不変）ため、りんごの例での絶対値も学習前の +10.375 付近から学習後は +0.7〜+1.0 付近へ動いている。点数の基準を 0 付近に保ちたい場合は、損失に微小な中心化項 $\lambda(r_w + r_l)^2$ を加えることがある。
- `task_type="SEQ_CLS"` で `modules_to_save=["score"]` と `pad_token_id` 設定が必須な理由: `AutoModelForSequenceClassification.from_pretrained(..., num_labels=1)` は言語モデルヘッドをランダム初期化の線形層 `score`（1280 次元 → 1 スカラー）に置き換える。`modules_to_save=["score"]` を省くと `score` が凍結され (`requires_grad=False`)、LoRA 層だけではランダムな線形結合のまま十分な学習ができない（ここでは学習対象パラメータ約 10.1M、全体の 1.50%）。また `LlamaForSequenceClassification` は各系列の「最後の非パディングトークン」の隠れ状態から報酬を取り出すため、`model.config.pad_token_id` が `None` のままだとバッチサイズ 2 以上で `ValueError` になる。`tokenizer.pad_token_id` (3) を設定した上で右パディングの有無を比べると、どちらも +10.3750（差 0.0000）で一致した。
- 長さバイアスと原因: 学習後の 0.5B 報酬モデルでは、報酬差と応答文字数差の相関が Pearson r = 0.684、Spearman rho = 0.676 と高い。内訳を見ると、`chosen` が長い 108 組では正解率 0.870 なのに対し、長くない 92 組では 0.359 と偶然 (0.5) を下回った。学習データ自体で `chosen` が長い組の割合が 0.589 あり、短い 1 エポックの学習で「長い応答に高い点を付ける」という近道を強く学んだと考えられる。文の後半を半分削る操作でも平均 −0.490（上昇は 8.0%）と下がった一方、末尾に定型締め句を足す操作では平均 −0.070（上昇は 14.0%）となり、この章の Best-of-N レシピで `Skywork-1.7B` が同じ締め句で +0.335（86% 上昇）したのとは逆の挙動を示した。長さバイアスを抑えるには、長さのバランスを揃えた選好データや長さによる正則化・デバイアス手法を使う。
- 公開 1.7B 報酬モデル・文字数ベースライン・DPO との比較: 同じ 200 組で、文字数だけで選んだベースラインの正解率は 0.540（95% 信頼区間 0.475〜0.610）、公開の `Skywork/Skywork-Reward-V2-Qwen3-1.7B` (Apache 2.0) は 0.610（同 0.540〜0.675、平均報酬差 +1.056、推論 6.3 秒）、この章の「選好データで LLM を調整する (DPO)」で同じ分割を学習した DPO の暗黙の報酬は 0.680（同 0.615〜0.745）だった。0.5B 報酬モデルと Skywork-1.7B の選好一致率は 0.585（報酬差の順位相関 0.284）にとどまり、りんごの算数例でも正しい応答 (+0.887) と誤った応答 (+0.980) の正誤を見分けられていない。hh-rlhf-12k-ja の DeepL 翻訳データ 2,000 組だけで学習した 0.5B モデルは対象データの文体や長さに適合しただけで、計算の正誤判定までは学べていない。また GPU の非決定性により同一シードでも実行ごとに正解率は数ポイント前後しうるため、95% 信頼区間の幅で評価する。
- DPO と明示的な報酬モデルの使い分け: DPO（この章の「選好データで LLM を調整する (DPO)」）は別の報酬モデルを学習せずに単一の方策モデルを直接更新できるため手順が簡単だが、暗黙の報酬はその方策と参照モデルの組に結びついており、別のモデルの応答を採点したり学習データ外の応答をふるいにかけたりする用途には使いにくい。明示的な報酬モデルは追加の学習とメモリが要る代わりに、一度学習すればこの章の「報酬モデルで N 個の応答を採点して最良を選ぶ (Best-of-N)」のように推論時の候補選択や、PPO・GRPO などオンライン強化学習の報酬関数（検証器を書けない自由文の課題）、合成選好データのフィルタリングに何度でも再利用できる。
- データ・ライセンス・トークナイザーの注意: データは `llm-jp/hh-rlhf-12k-ja` (MIT License) の `helpful-*` のみを使用しており、`harmless-base` は不快な内容を含むため除外した（DPO レシピと同じ注意が当てはまる）。モデルは `sbintuitions/sarashina2.2-0.5b-instruct-v0.1` (MIT License) で、トークナイザーの Unigram 修正 (`load_tokenizer`) については大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」の注意書きを参照。

## 報酬モデルで N 個の応答を採点して最良を選ぶ (Best-of-N) {reward_best_of_n gpu}

LLM に同じ質問への応答を N 個サンプリングさせ、公開の報酬モデルで採点して最も点の高いものを選びたい (Best-of-N)。N を増やすと品質がどれだけ上がるかを、正解のある課題と別の判定役で確かめ、報酬モデルの偏り（報酬ハッキング）も調べる。選んだ最良と最悪の組を DPO の選好データにしたい。

```python
import itertools, json, random, re, time, unicodedata
from collections import Counter
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, set_seed

def load_tokenizer(name):
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    # transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
    # 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
    sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
    tokenizer.backend_tokenizer.model = models.Unigram(
        [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
    return tokenizer

set_seed(0)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
DTYPE = torch.bfloat16 if BF16 else torch.float16
# 応答を生成するモデル: この章の DPO・GRPO のレシピと同じ 0.5B のモデル (MIT License)
name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"
tokenizer = load_tokenizer(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=DTYPE, device_map="auto").eval()

# 報酬モデル: Skywork-Reward-V2 (Liu et al., 2025) の Qwen3-1.7B 版 (Apache 2.0)。2,600 万組の選好データで学習した
# Bradley-Terry 型の報酬モデルで、質問と応答の会話を入れると 1 つの実数（報酬）を返す。点の絶対値に意味はなく、同じ質問の中で比べる
rm_name = "Skywork/Skywork-Reward-V2-Qwen3-1.7B"
rm_tokenizer = AutoTokenizer.from_pretrained(rm_name)
rm = AutoModelForSequenceClassification.from_pretrained(rm_name, dtype=DTYPE, device_map="auto").eval()

@torch.no_grad()
def reward(question, responses, bs=8):
    """1 つの質問への複数の応答を採点する。モデルカードの推奨に従い、システムプロンプトは付けない"""
    texts = [rm_tokenizer.apply_chat_template([{"role": "user", "content": question}, {"role": "assistant", "content": r}], tokenize=False)
             for r in responses]
    scores = []
    for i in range(0, len(texts), bs):
        enc = rm_tokenizer(texts[i:i + bs], return_tensors="pt", padding=True).to(rm.device)
        scores += rm(**enc).logits[:, 0].float().tolist()
    return np.array(scores)

# 日本語での挙動を確かめる: モデルカードの例（りんごの問題）を日本語に訳し、正しい応答・誤った応答・英語の応答を比べる
q = "りんごが 12 個あります。4 個を友達にあげ、1 個買い足して、自分と兄弟 2 人の 3 人で等しく分けます。1 人何個もらえますか。"
cands = {
    "正しい（日本語）": "12 − 4 = 8 個。1 個買って 8 + 1 = 9 個。3 人で分けるので 9 ÷ 3 = 3。答えは 1 人 3 個です。",
    "誤り（日本語）": "12 − 4 = 8 個。1 個買って 8 + 1 = 9 個。兄弟 2 人で分けるので 9 ÷ 2 = 4.5。答えは 1 人 4 個です。",
    "正しい（英語）": "12 - 4 = 8 apples. After buying 1 more, 8 + 1 = 9. Split among 3 people, 9 / 3 = 3. Each person gets 3 apples.",
    "答えだけ（正しい）": "3 個です。",
    "質問と無関係": "東京は日本の首都で、人口は約 1,400 万人です。",
}
for k, s in zip(cands, reward(q, list(cands.values()))):
    print(f"{s:7.2f}  {k}")
```

```python
# 正解のある課題: MGSM（CC BY 4.0。Hugging Face のメタデータは CC BY-SA 4.0）の日本語の先頭 100 問（大規模言語モデルの章の Chain-of-Thought のレシピと同じ）
# 指示はそのレシピの「CoT (0-shot)」と同じ。温度 0.7・top_p 0.9（開発元の既定値）で 1 問につき N = 8 個サンプリングする。
# 開発元の既定値では repetition_penalty 1.05 も掛かる（貪欲法にも掛かる）
N_MAX, N_Q = 8, 100
mgsm = load_dataset("juletxara/mgsm", "ja", split="test").select(range(N_Q))
gold = np.array(mgsm["answer_number"], dtype=float)
INSTRUCTION = "次の問題に答えてください。ステップごとに考えて途中の計算を書き、最後に「答えは〇〇です。」と書いてください。"
math_q = [f"{INSTRUCTION}\n\n問題: {q}" for q in mgsm["question"]]

@torch.no_grad()
def generate(questions, n, max_new_tokens, bs=64):
    """質問ごとに n 個の応答と、長さの上限で打ち切られたかどうかを返す（n = 1 なら貪欲法、n > 1 ならサンプリング）"""
    outs, cut, step = [], [], max(1, bs // n)
    for i in range(0, len(questions), step):
        enc = tokenizer.apply_chat_template([[{"role": "user", "content": q}] for q in questions[i:i + step]], add_generation_prompt=True,
                                            return_tensors="pt", return_dict=True, padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=n > 1, num_return_sequences=n, pad_token_id=tokenizer.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        outs += tokenizer.batch_decode(gen, skip_special_tokens=True)
        cut += [o[-1].item() not in (tokenizer.eos_token_id, tokenizer.pad_token_id) for o in gen]
    return [outs[j * n:(j + 1) * n] for j in range(len(questions))], np.array(cut).reshape(len(questions), n)

def extract(text):
    """答えの数値を取り出す（大規模言語モデルの章の Chain-of-Thought のレシピの取り出し方の簡略版）: 「答え」の後ろ → 最後に出てきた数値"""
    text = unicodedata.normalize("NFKC", text).replace(",", "")
    found = re.findall(r"答え[はが]?\s*[:：]?[^\d\-\n]*(-?\d+(?:\.\d+)?)", text) or re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(found[-1]) if found else None

t = time.time()
greedy, _ = generate(math_q, 1, 320)
math_out, math_cut = generate(math_q, N_MAX, 320)         # 800 個の応答。貪欲法と合わせて Colab の T4 で 8〜10 分程度
print(f"生成 {time.time() - t:.0f} 秒")
t = time.time()
math_score = np.array([reward(q, outs) for q, outs in zip(math_q, math_out)])    # 100 問 × 8 個
print(f"採点 {time.time() - t:.0f} 秒")
math_ans = [[extract(o) for o in outs] for outs in math_out]
correct = np.array([[a == g for a in row] for row, g in zip(math_ans, gold)])
greedy_correct = np.array([extract(o[0]) == g for o, g in zip(greedy, gold)])
print(f"報酬の平均: 正解の応答 {math_score[correct].mean():.2f}、不正解の応答 {math_score[~correct].mean():.2f}")
print(f"上限の 320 トークンで打ち切られた応答の割合 {math_cut.mean():.3f}（その報酬の平均 {math_score[math_cut].mean():.2f}）")
```

```python
# N を変えて比べる。8 個から n 個を選ぶすべての組み合わせで計算して平均する（どの n 個を使うかによるばらつきをならす）
def vote(answers):
    answers = [a for a in answers if a is not None]
    return Counter(answers).most_common(1)[0][0] if answers else None

def per_question(i, n):
    combos = list(itertools.combinations(range(N_MAX), n))
    return {"ランダムに 1 つ": correct[i].mean(),
            "報酬モデルで最良 (Best-of-N)": np.mean([correct[i, list(c)][np.argmax(math_score[i, list(c)])] for c in combos]),
            "多数決 (self-consistency)": np.mean([vote([math_ans[i][j] for j in c]) == gold[i] for c in combos]),
            "どれか 1 つが正解（上限）": np.mean([correct[i, list(c)].any() for c in combos])}

table = {n: pd.DataFrame([per_question(i, n) for i in range(N_Q)]) for n in [1, 2, 4, 8]}
print(pd.DataFrame({n: d.mean() for n, d in table.items()}).T.rename_axis("N").round(3).to_string())
print(f"貪欲法 1 回: {greedy_correct.mean():.3f}")

def diff_ci(x, y, n_boot=2000):   # 同じ問題で比べる対応のあるブートストラップで、正解率の差の 95% 信頼区間
    idx = np.random.default_rng(0).integers(0, len(x), (n_boot, len(x)))
    d = x[idx].mean(1) - y[idx].mean(1)
    return f"{(x.mean() - y.mean()) * 100:+.1f} ポイント [{np.percentile(d, 2.5) * 100:+.1f}, {np.percentile(d, 97.5) * 100:+.1f}]"

bon8, maj8 = table[8]["報酬モデルで最良 (Best-of-N)"].values, table[8]["多数決 (self-consistency)"].values
print("Best-of-8 − ランダムに 1 つ:", diff_ci(bon8, table[8]["ランダムに 1 つ"].values))
print("Best-of-8 − 貪欲法 1 回:", diff_ci(bon8, greedy_correct.astype(float)))
print("Best-of-8 − 8 個の多数決:", diff_ci(bon8, maj8))
```

```python
# 正解のない課題: 質問への応答の役に立ち方。この章の DPO のレシピと同じ絞り込みと並べ替えで、その学習側の質問 (data[:2000]) の先頭 50 件を使う
# （評価側の data[2000:2200] は使わない。ここで作る選好データを DPO の学習に使っても、評価の質問と重ならない）
# llm-jp/hh-rlhf-12k-ja (MIT License): Anthropic の hh-rlhf (MIT License) の一部を DeepL で日本語に訳したもの。有用さ (helpful-*) の部分だけを使う
raw = load_dataset("llm-jp/hh-rlhf-12k-ja", split="train")
data = [r["conversations"][0]["value"] for r in raw
        if r["source"].startswith("helpful") and len(r["conversations"]) == 1 and r["chosen"] != r["rejected"]
        and len(r["conversations"][0]["value"]) <= 200 and max(len(r["chosen"]), len(r["rejected"])) <= 400]
random.Random(0).shuffle(data)
help_q = data[:50]

t = time.time()
help_out, help_cut = generate(help_q, N_MAX, 384)   # 400 個の応答（上限は DPO のレシピと同じ 384 トークン）。Colab の T4 で 3〜4 分程度
help_score = np.array([reward(q, outs) for q, outs in zip(help_q, help_out)])
print(f"生成と採点 {time.time() - t:.0f} 秒")
best, worst = help_score.argmax(1), help_score.argmin(1)
rows = np.arange(len(help_q))
length = np.array([[len(o) for o in outs] for outs in help_out])
print(f"応答の平均の長さ: 最良 {length[rows, best].mean():.0f} 文字、全体 {length.mean():.0f} 文字、最悪 {length[rows, worst].mean():.0f} 文字")
def mean_rho(score, length, keep):   # 同じ質問の中での報酬と長さの順位相関 (Spearman) の平均。keep の応答だけで計算する
    return np.nanmean([spearmanr(s[k], l[k]).statistic for s, l, k in zip(score, length, keep) if k.sum() >= 3])
math_len = np.array([[len(o) for o in outs] for outs in math_out])
print(f"上限の 384 トークンで打ち切られた応答の割合 {help_cut.mean():.3f}")
print(f"同じ質問の中での報酬と長さの順位相関の平均: {mean_rho(help_score, length, np.ones_like(help_cut)):.3f}"
      f"（打ち切られた応答を除くと {mean_rho(help_score, length, ~help_cut):.3f}）")
print(f"算数の問題では {mean_rho(math_score, math_len, np.ones_like(math_cut)):.3f}（打ち切られた応答を除くと {mean_rho(math_score, math_len, ~math_cut):.3f}）")
i = 0
print("\n質問:", help_q[i])
for j in np.argsort(-help_score[i]):
    print(f"{help_score[i, j]:6.2f}  {help_out[i][j][:80]!r}")
```

```python
# 別の判定役で確かめる: この章の DPO のレシピと同じく sarashina2.2-3b-instruct (MIT License) に 2 つの応答を比べさせる。
# 順番を入れ替えて 2 回聞いて平均する。比べる相手は、報酬モデルを使わない場合（8 個のうち 1 個目 = ランダムに 1 つ）
judge_name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
jtok = load_tokenizer(judge_name)
judge = AutoModelForCausalLM.from_pretrained(judge_name, dtype=DTYPE, device_map="auto").eval()
id_a, id_b = [jtok(c, add_special_tokens=False)["input_ids"][-1] for c in ("A", "B")]

@torch.no_grad()
def p_first_better(q, x, y):
    def p_a(a, b):
        msg = [{"role": "user", "content": f"次の質問に対する 2 つの応答のうち、より役に立つ応答はどちらですか。A か B の 1 文字で答えてください。\n\n"
                                           f"質問: {q}\n\n応答 A: {a}\n\n応答 B: {b}"}]
        inputs = jtok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(judge.device)
        return torch.softmax(judge(**inputs).logits[0, -1].float()[[id_a, id_b]], -1)[0].item()
    return (p_a(x, y) + 1 - p_a(y, x)) / 2

def rate_ci(hits):   # 割合と、ブートストラップによる 95% 信頼区間
    boot = [hits[i].mean() for i in np.random.default_rng(0).integers(0, len(hits), (2000, len(hits)))]
    return f"{hits.mean():.3f}（95% 信頼区間 {np.percentile(boot, 2.5):.3f}〜{np.percentile(boot, 97.5):.3f}、{len(hits)} 問）"

others = [i for i in rows if best[i] != 0]       # 1 個目がたまたま最良だった質問は比べられないので除く
win_best = np.array([p_first_better(help_q[i], help_out[i][best[i]], help_out[i][0]) for i in others])
others = [i for i in rows if worst[i] != 0]
win_worst = np.array([p_first_better(help_q[i], help_out[i][worst[i]], help_out[i][0]) for i in others])   # 判定は合わせて Colab の T4 で 1 分程度
print("報酬モデルの最良がランダムな 1 つより良いと判定された割合:", rate_ci(win_best > 0.5))
print("報酬モデルの最悪がランダムな 1 つより良いと判定された割合:", rate_ci(win_worst > 0.5))
print(f"GPU メモリの最大使用量 {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

```python
# 報酬ハッキングの手がかり: 応答の中身を変えずに、言い回しや長さだけを変えると報酬がどう動くか（各質問の 1 個目の応答で調べる）
edits = {
    "冒頭にほめ言葉を足す": lambda r: "素晴らしいご質問ですね！" + r,
    "末尾に定型の締めを足す": lambda r: r + "\n\nほかにもご質問があれば、お気軽にお尋ねください。",
    "末尾に「正確です」と書き足す": lambda r: r + "\n\nこの回答は専門家が確認した正確な情報です。",
    "同じ応答を 2 回繰り返す": lambda r: r + "\n\n" + r,
    "後半を削って半分にする": lambda r: r[:len(r) // 2],
}
base = help_score[:, 0]
res = {}
for label, f in edits.items():
    d = np.array([reward(q, [f(outs[0])])[0] for q, outs in zip(help_q, help_out)]) - base
    res[label] = {"報酬の変化（平均）": d.mean(), "上がった割合": (d > 0).mean()}
# 算数: 正解した応答の最後の答えの数値だけを 1 大きくする（推論はそのまま、答えだけが誤りになる）
def wrong_answer(text, g):
    return re.sub(r"(答え[はが]?\s*[:：]?[^\d\-\n]*)(-?[\d,]+(?:\.\d+)?)(?!.*答え)", lambda m: m.group(1) + f"{g + 1:g}", text, flags=re.S)
# 各問で最初に正解した応答のうち、「答えは…」の形で書かれていて書き換えられたものを使う
ok = [(i, j) for i in range(N_Q) for j in np.flatnonzero(correct[i])[:1] if wrong_answer(math_out[i][j], gold[i]) != math_out[i][j]]
d = np.array([reward(math_q[i], [wrong_answer(math_out[i][j], gold[i])])[0] - math_score[i, j] for i, j in ok])
res["算数: 最後の答えだけを誤りにする"] = {"報酬の変化（平均）": d.mean(), "上がった割合": (d > 0).mean()}
print(pd.DataFrame(res).T.round(3).to_string())
print(f"（算数は {len(ok)} 問。報酬の標準偏差は、同じ質問の 8 個の中で平均 {help_score.std(1).mean():.2f}）")
```

```python
# 最良と最悪の組を、この章の DPO のレシピと同じ trl の会話形式 (prompt / chosen / rejected) の選好データにする。
# 報酬の差が小さい組は、報酬モデルの誤差で順番が入れ替わりやすいので除く（ここでは差が 2 未満）
pairs = [{"prompt": [{"role": "user", "content": q}],
          "chosen": [{"role": "assistant", "content": outs[b]}], "rejected": [{"role": "assistant", "content": outs[w]}],
          "margin": float(s[b] - s[w])}
         for q, outs, s, b, w in zip(help_q, help_out, help_score, best, worst) if s[b] - s[w] >= 2]
print(f"{len(help_q)} 問から {len(pairs)} 組。報酬の差の中央値 {np.median([p['margin'] for p in pairs]):.2f}")
print(f"chosen のほうが長い組の割合: {np.mean([len(p['chosen'][0]['content']) > len(p['rejected'][0]['content']) for p in pairs]):.3f}")
with open("bon_pairs.jsonl", "w") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
p = max(pairs, key=lambda p: p["margin"])
print("\n質問:", p["prompt"][0]["content"], "\n[chosen]", p["chosen"][0]["content"][:200], "\n[rejected]", p["rejected"][0]["content"][:200])
```

- Best-of-N は、モデルの重みを変えずに推論時の計算を増やして品質を上げる方法で、N 個の候補を生成し、報酬モデルの点が最も高いものを選ぶ（WebGPT (Nakano et al., 2021) などで使われた）。生成と採点が N 倍になるので、時間もほぼ N 倍かかる。検証に使った GPU（T4 と同じ float16 の経路）では全体で約 5 分（算数の 900 個の生成 130 秒、採点 26 秒、質問の 400 個の生成と採点 82 秒）、GPU メモリの最大使用量は 11.9GB だった（0.5B の生成モデル、1.7B の報酬モデル、3B の判定役を同時に載せた状態）。T4 では生成が遅いので、この 2〜2.5 倍程度かかると見込んでいる（T4 では測っていない）。乱数の種を固定しても、GPU の計算の違いでサンプリングの結果が変わることがあり、そのときは表の値も変わる。
- 報酬モデル: Skywork-Reward-V2 は Skywork が 2025 年に公開した報酬モデルの系列（0.6B〜8B の 8 種類）で、Qwen3 から作った 4 種類は Apache 2.0、Llama から作ったものは Llama 3.1・3.2 のライセンスに従う。モデルカードには対応言語の記載がない。最初のセルの 5 件では、日本語の正しい応答 (4.18) が誤った応答 (−1.25) より高く、同じ内容の英語の応答 (2.05) より日本語のほうが高かった。この例では、説明のない「3 個です。」(−2.36) も途中の説明がある応答より低かった（1 問だけの結果である）。MGSM の 800 個の応答でも、正解の応答の平均 5.06 に対して不正解は −1.04 で、日本語でも正誤をある程度見分けている。
- 算数（正解のある課題）: ランダムに 1 つ選ぶと正解率 0.199 のところ、報酬モデルで選ぶと N = 2、4、8 で 0.267、0.321、0.340 に上がり、N = 8 では +14.1 ポイント [+8.3, +20.2] だった。ただし、貪欲法 1 回 (0.260) との差（+8.0 ポイント [−1.0, +17.0]）と、8 個の多数決 (0.310) との差（+3.0 ポイント [−3.0, +9.0]）は、100 問では偶然の範囲である。8 個のどれかが正解だった問題は 0.490 あり、報酬モデルはその 3 割ほどで正解を選び損ねている。N を増やすと上がり方は小さくなる（4 → 8 で +1.9 ポイント）。
- 正解のない課題: 判定役 (3B) は、報酬モデルの最良の応答を、ランダムに選んだ 1 つより良いと 0.667（95% 信頼区間 0.524〜0.810、42 問）で判定し、最悪の応答は 0.244（同 0.133〜0.356、45 問）だった。この判定役と人の選好の一致率は、この章の DPO のレシピでは 0.650 程度だったので、目安にとどまる。また、Best-of-N は候補の中から選ぶだけなので、全部に誤りがあれば最良にも残る（上の例で最も点の高い応答にも「combined (combined)」のような崩れた表現がある）。
- 報酬ハッキング（長さ）: この報酬モデルは、同じ質問の中では長い応答をやや好んだ（報酬と長さの順位相関の平均 0.191。最良の応答の平均 611 文字、全体 569 文字、最悪 538 文字）。上限の 384 トークンで打ち切られた応答は 3.0% で、それを除いても 0.225 だった。ただし、同じ応答を 2 回繰り返して長くすると報酬は平均 −0.720 で、上がったのは 16% だけなので、長ければよいわけではない。後半を削って半分にすると全部で下がった (−4.312) が、文の途中で切れたことへの罰も含まれていると考えられる。算数では逆に長い応答ほど報酬が低い (−0.265)。320 トークンで打ち切られて答えまで届かなかった応答（12.2%、報酬の平均 −4.34）の影響もあるが、それを除いても −0.170 で、原因は確かめていない。
- 報酬ハッキング（言い回し）: 末尾に「ほかにもご質問があれば、お気軽にお尋ねください。」と足すだけで 86% の応答で報酬が上がった（平均 +0.335。同じ質問の 8 個の報酬の標準偏差は平均 1.56 なので、点の近い候補の順位は入れ替わりうる）。冒頭のほめ言葉はほとんど効かず（−0.011、上がったのは 54%）、「専門家が確認した正確な情報です」という自己申告は逆に下がった (−1.124)。算数で最後の答えだけを誤りに書き換えると 31 問すべてで下がった（平均 −6.549）。中身と関係のない言い回しで点が上がるので、N を大きくしたり、この報酬モデルで強化学習したりすると、こうした言い回しが増えると考えられる（確かめていない）。報酬モデルの点を上げすぎると本当の品質から離れていくこと（過最適化）は Gao et al. (2023) が Best-of-N と強化学習の両方で調べている。報酬だけでなく、別の評価（正解率や判定役）と出力の例も見る。
- 選好データ: 報酬の差が 2 以上の 49 組（差の中央値 5.04）を bon_pairs.jsonl に保存した。質問は DPO のレシピの学習側 (`data[:2000]`) から取ったので、そのレシピの `train` を、このファイルから `margin` の列を除いて読み込んだものに置き換えれば、評価の質問と重ならずに DPO に使える（ここでは学習していない）。そのレシピの選好データは別のモデルの応答（方策外のデータ）だったが、これは調整するモデル自身の応答（方策上のデータ）で、人ではなく報酬モデルが選んでいる。そのため、報酬モデルの癖もそのまま選好として学習される。ここでは chosen のほうが長い組が 0.633 あり、学習すると応答が長くなる方向に働きうる。実際には数百問以上で作り、長さの比や言い回しの偏りを確かめる。
- 関係するレシピ: 大規模言語モデルの章の「考える過程を書かせて LLM に算数の文章題を解かせる（Chain-of-Thought と self-consistency）」の多数決は、答えが一致するかだけで選ぶので、答えを機械的に比べられる課題にしか使えない。Best-of-N は報酬モデルの点で選ぶので、自由な文章にも使える。報酬モデル自体を選好データから学習する方法は、この章の「選好データから報酬モデルを学習して応答を採点する（Bradley-Terry モデル）」を参照。この章の「正解を判定できる課題で LLM を強化学習する (GRPO)」は、同じように 1 問に複数の応答を生成し、その報酬の差でモデル自身を学習する。判定役のモデルの信頼性の確かめ方は、LLM の評価と分析の章の「手元の LLM に生成文を採点させる (LLM-as-a-judge)」を参照。トークナイザーの修正 (`load_tokenizer`) については、大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」の注意書きを参照。

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
name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"   # この章の「選好データで LLM を調整する (DPO)」と同じ 0.5B の指示チューニング済みモデル (MIT License)
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto")

# 学習用: SakanaAI/gsm8k-ja-test_250-1319 (Apache 2.0)。GSM8K（Cobbe et al., 2021、MIT License）のテストの 251 問目以降の
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
# 指示: 大規模言語モデルの章の Chain-of-Thought のレシピの「CoT (0-shot)」と同じ。途中の計算を書かせ、最後に答えを決まった形で書かせる
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
    """形を問わず取り出す（大規模言語モデルの章の Chain-of-Thought のレシピの取り出し方の簡略版）: 「答え」の後ろ → 最後に出てきた数値"""
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
- 上の出力では、正解率（決まった形の答え）は 0.072 → 0.248（+17.6 ポイント [+12.4, +22.8]）、形を守った割合は 0.200 → 0.940 に上がった。しかし形を問わない正解率は 0.252 → 0.252（+0.0 ポイント [−5.2, +4.8]）で、上がらなかった（信頼区間は 0 をまたぐ）。問題ごとに見ると、22 問が正解から不正解に、22 問が不正解から正解に変わった。この実行で身についたのは「決まった形で答える」ことで、40 ステップ（160 問）の学習では解く力は上がらなかった。報酬の推移でも、正解の報酬の上昇 (0.06〜0.07 → 0.13〜0.31) は形式の報酬の上昇 (0.04 → 0.17) と同時に起きている。1 ステップは 4 問だけなので、正解の報酬は問題の難しさで大きく揺れる。また、GPU の計算は実行ごとにわずかに変わり、それがサンプリングで生成する応答と学習の経過を変えるので、同じ種でも学習後の正解率は実行ごとに変わる（前の実行では、決まった形の答えの正解率が 0.216、形を問わない正解率が 0.216 だった）。1 回の実行で結論を出さず、種を変えて何回か学習して比べる。
- 応答は短くなった（評価で平均 192 → 157 トークン、学習中に上限の 320 トークンで打ち切られた割合は 0.22 → 0.02〜0.09）。打ち切られた応答は最後の「答えは〇〇です。」まで書けずに報酬が 0 になるので、短い応答が有利になると考えられる（短くなったことが正解率に影響したかどうかは確かめていない）。報酬が形式だけで取れるように設計すると、推論を書かずに答えだけを出すような、報酬は上がるが目的に合わない学習（報酬ハッキング）が起きることがある。報酬だけでなく、応答の長さと出力の例も見る。
- 大規模言語モデルの章の「考える過程を書かせて LLM に算数の文章題を解かせる（Chain-of-Thought と self-consistency）」との関係: 指示文はそのレシピの「CoT (0-shot)」と同じで、推論を書かせてから答えさせる。GRPO は推論の過程には報酬を与えず、最後の答えだけを判定する。推論の書き方は、正解に結びつくかどうかを通じて間接的に学ばれるだけで、途中の計算が正しいかどうかは確かめていない。上の「学習後に間違えた例」では、学習後の応答は「答えは〇〇です。」の形は守っているが、途中でマフィンに使った 4 個を引き忘れ、13 個 × 2 ドル = 26 ドルとしている（文の途中に英語も混じる）。そのレシピでは、形の違う答えも取り出せるように取り出し方を工夫したが、ここでは答えの形そのものを LoRA の重みに覚えさせた。正解の判定には、そのレシピの取り出し方より厳しい「決まった形」を使っている。取り出し方が緩いと、途中に出てきた数値がたまたま正解と一致しただけの応答にも報酬を与えてしまう。
- 時間とメモリ: 検証に使った GPU（T4 と同じ float16 の経路）では学習が 342 秒、全体で約 8 分、GPU メモリの最大使用量は 10.5GB だった（時間は GPU のほかの負荷で変わり、別の実行では学習が 736 秒、全体で約 17 分だった）。T4 では応答の生成が遅いので、この 2〜2.5 倍程度かかると見込んでいる（T4 では測っていない）。応答の生成は `use_vllm=True` で vLLM を使うと速くなるとされるが、Colab の torch などを入れ替えることになるので、ここでは使っていない（確かめていない）。`GRPOConfig` は trl 1.18.0 では `beta=0`（参照モデルとの KL の罰則なし、参照モデルを読み込まない）、学習率 1e-6、`bf16=True`、`gradient_checkpointing=True`、`loss_type="dapo"` が既定で、版によって変わることがある。
- データ: 学習用の SakanaAI/gsm8k-ja-test_250-1319 (Apache 2.0) は、データセットカードによると GPT-4 (gpt-4-0125-preview) で訳したもので、評価用の MGSM と重ならないように GSM8K のテストの 251 問目以降だけを使っている。どちらも GSM8K のテストの問題なので、ここで学習したモデルを GSM8K で評価すると、学習した問題で評価することになる。MGSM のライセンスと、評価データが事前学習に含まれている可能性については、大規模言語モデルの章の「考える過程を書かせて LLM に算数の文章題を解かせる（Chain-of-Thought と self-consistency）」の注意書きを参照。トークナイザーの修正については、大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」の注意書きを参照。

## 学習した複数のアダプタを重みの足し引きで合成・調整する（タスク算術とモデルマージ） {task_arithmetic_merge gpu}

異なる課題向けに学習した複数の LoRA アダプタから重み差分（タスクベクトル）を取り出し、スカラー倍で効果の強弱や打ち消しを調整したり、単一のモデルへ統合（マージ）したりしたい。単純な線形加算で干渉が起きる場合と、TIES-Merging で符号の衝突を解消して合成する場合を比べる。

```python
!pip install -q peft
```

```python
import gc, json, random, re, time
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

def load_tokenizer(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    # transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
    # 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
    sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
    tokenizer.backend_tokenizer.model = models.Unigram(
        [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
    return tokenizer

if not torch.cuda.is_available():
    raise RuntimeError("Colab のランタイムを GPU に変更してください")
set_seed(0)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere 以降。T4 では float16 を使う
DTYPE = torch.bfloat16 if BF16 else torch.float16
MODEL_NAME = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"   # MIT License
tokenizer = load_tokenizer(MODEL_NAME)

# 課題 A: 社内通知の正規化 JSON 抽出（学習・評価で ID、取引先、文型を完全に分離）
TRAIN_A_IDS = ["AX104", "BK209", "CM315", "DL422", "EN530", "FP641", "GR752", "HS863"]
TEST_A_IDS = ["JT910", "KU118", "LV227", "MW336", "NX445", "PY554"]
TRAIN_A_CLIENTS = ["青葉商事", "北星電機", "中央物流", "東都製薬", "富士通信", "明和工業", "新日本食品", "大和重工"]
TEST_A_CLIENTS = ["桜井製作所", "星野メディカル", "南風運輸", "京浜ソフト", "緑川出版", "高田精密"]
A_CATS = [
    ("order", ["特急配送の発注を受領しました", "追加発注の手続きが完了しました", "新規注文の登録を受け付けました", "商品のご注文を確認しました"]),
    ("support", ["稼働障害の調査依頼を受け付けました", "不具合対応のチケットを起票しました", "保守窓口へトラブル相談が入りました", "操作エラーの問い合わせを受領しました"]),
    ("billing", ["月次請求書を発行しました", "利用料金の清算通知を送信しました", "口座振替の引き落とし案内を送りました", "当月分の請求明細を確定しました"]),
]

def build_task_a(n, rng, ids, clients, test_mode=False):
    items = []
    for i in range(n):
        cid = f"{rng.choice(ids)}-{rng.randint(10, 99)}"
        client = rng.choice(clients)
        cat_code, phrases = rng.choice(A_CATS)
        phrase = rng.choice(phrases[2:]) if test_mode else rng.choice(phrases[:2])
        if rng.random() < 0.5:
            k, h = rng.randint(1, 9), rng.randint(1, 9) * 100
            amt, amt_str = k * 10000 + h, f"{k}万{h}円"
        else:
            amt = rng.randint(12, 980) * 100
            amt_str = f"{amt:,}円"
        urgent = rng.choice([True, False])
        urg_txt = rng.choice(["【至急】", "緊急対応が必要です。", "優先度：高。"]) if urgent else rng.choice(["通常進行です。", "急ぎではありません。", "標準処理とします。"])
        m, d = rng.randint(1, 12), rng.randint(1, 28)
        templates = [
            f"{m}月{d}日、{client}（管理番号{cid}）から{phrase}。金額は{amt_str}です。{urg_txt}",
            f"管理番号{cid}：{client}の件、{m}月{d}日に{phrase}（{amt_str}）。{urg_txt}",
            f"【連絡】{client}（ID:{cid}）について{m}月{d}日付で{phrase}。関連金額は{amt_str}、{urg_txt}",
            f"{client}（管理番号:{cid}）：{m}月{d}日付けで{phrase}。対象金額{amt_str}。{urg_txt}",
        ]
        text = templates[2 + (i % 2)] if test_mode else templates[i % 2]
        prompt = f"次の社内通知を所定の正規化 JSON（1 行のみ、余分な説明やコードブロックなし）に変換してください。\n通知: {text}"
        gold = {"code": cid, "client": client, "category": cat_code, "date": f"2026-{m:02d}-{d:02d}", "amount_jpy": amt, "urgent": urgent}
        items.append({"prompt": prompt, "completion": json.dumps(gold, ensure_ascii=False), "gold": gold})
    return items

# 課題 B: 障害報告の丁寧な 3 行箇条書き要約（です・ます調、重要事実の保持）
TRAIN_B_SYS = ["認証基盤", "決済ゲートウェイ", "検索エンジン", "通知キュー", "在庫API", "画像変換基盤"]
TEST_B_SYS = ["配送管理システム", "請求バッチ", "ログ集計基盤", "予約エンジン"]
TRAIN_B_CAUSE = ["メモリ上限の超過", "接続プールの枯渇", "設定値の反映漏れ", "期限設定の誤り", "証明書の期限切れ"]
TEST_B_CAUSE = ["一時記憶の肥大化", "通信帯域の逼迫", "索引更新の競合"]
TRAIN_B_ACT = ["設定の修正と再起動", "上限値の2倍拡張", "旧版への切り戻し", "監視閾値の調整"]
TEST_B_ACT = ["再試行間隔の延長", "並列数の制限", "修正パッチの適用"]
TRAIN_B_DATES = ["11月4日", "11月12日", "11月19日", "12月2日", "12月9日", "12月18日"]
TEST_B_DATES = ["1月15日", "1月23日", "2月6日", "2月14日"]

def build_task_b(n, rng, syss, causes, acts, dates, test_mode=False):
    items = []
    for i in range(n):
        sys_name, cause, act, dt = rng.choice(syss), rng.choice(causes), rng.choice(acts), rng.choice(dates)
        dur, impact = rng.randint(12, 55), rng.randint(15, 320)
        templates = [
            f"{dt}に{sys_name}で{dur}分間の遅延が生じ、約{impact}件の処理が滞った。調査の結果、{cause}が引き金と判明。運用担当が{act}を実施し、現在は平常に戻っている。",
            f"{sys_name}において{dt}、約{impact}件に影響する{dur}分間の応答低下が発生した。原因は{cause}であり、{act}を行って復旧した。",
            f"障害報告：{sys_name}で{dt}に{dur}分間の遅延（影響約{impact}件）が起きた。原因は{cause}であり、{act}を実施して現在は復旧している。",
            f"{dt}、{sys_name}にて{dur}分間の遅延（影響{impact}件）を確認。原因は{cause}で、{act}を行って正常稼働へ戻った。",
        ]
        text = templates[2 + (i % 2)] if test_mode else templates[i % 2]
        prompt = f"次の障害報告を、丁寧な「です・ます調」の箇条書き3行に要約してください。\n報告: {text}"
        comp = (f"- 【発生】{dt}に{sys_name}で{dur}分間の遅延（影響{impact}件）が発生しました。\n"
                f"- 【原因】{cause}が原因でした。\n"
                f"- 【対応】{act}を実施し、現在は復旧しています。")
        items.append({"prompt": prompt, "completion": comp, "facts": [dt, sys_name, str(dur), str(impact), cause, act]})
    return items

rng = random.Random(42)
train_a = build_task_a(160, rng, TRAIN_A_IDS, TRAIN_A_CLIENTS, test_mode=False)
test_a = build_task_a(24, rng, TEST_A_IDS, TEST_A_CLIENTS, test_mode=True)
train_b = build_task_b(160, rng, TRAIN_B_SYS, TRAIN_B_CAUSE, TRAIN_B_ACT, TRAIN_B_DATES, test_mode=False)
test_b = build_task_b(24, rng, TEST_B_SYS, TEST_B_CAUSE, TEST_B_ACT, TEST_B_DATES, test_mode=True)
print(f"課題 A: 学習 {len(train_a)} 件 / 評価 {len(test_a)} 件 | 課題 B: 学習 {len(train_b)} 件 / 評価 {len(test_b)} 件")

def encode_sft(items):
    encoded = []
    for it in items:
        p_ids = tokenizer.apply_chat_template([{"role": "user", "content": it["prompt"]}], add_generation_prompt=True, tokenize=True, return_dict=False)
        c_ids = tokenizer(it["completion"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        encoded.append((p_ids + c_ids, [-100] * len(p_ids) + c_ids))
    return encoded

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def train_lora_delta(train_items, label, seed=0, epochs=2, lr=1e-4, bs=8):
    """LoRA アダプタを学習し、各線形層の重み差分 ΔW = (α/r)(B @ A) を float32 で抽出する"""
    set_seed(seed)
    m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map="auto")
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=TARGET_MODULES))
    enc_data = encode_sft(train_items)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    total_steps = ((len(enc_data) + bs - 1) // bs) * epochs
    sched = get_cosine_schedule_with_warmup(opt, max(1, total_steps // 10), total_steps)
    m.train()
    rng_local, pad_id = random.Random(seed), tokenizer.pad_token_id
    t0 = time.time()
    losses = []
    for ep in range(epochs):
        order = list(range(len(enc_data)))
        rng_local.shuffle(order)
        ep_loss = []
        for i in range(0, len(order), bs):
            batch = [enc_data[j] for j in order[i:i + bs]]
            ml = max(len(x) for x, _ in batch)
            x = torch.tensor([ids + [pad_id] * (ml - len(ids)) for ids, _ in batch], device=m.device)
            y = torch.tensor([lbl + [-100] * (ml - len(lbl)) for _, lbl in batch], device=m.device)
            loss = m(input_ids=x, attention_mask=(x != pad_id).long(), labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            ep_loss.append(loss.item())
        losses.append(float(np.mean(ep_loss)))
    print(f"{label}: 学習時間 {time.time() - t0:.1f} 秒 | エポック別損失 {[round(v, 4) for v in losses]}")
    deltas = {}
    for mod_name, mod in m.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and "default" in mod.lora_A:
            base_key = mod_name.removeprefix("base_model.model.") + ".weight"
            A = mod.lora_A["default"].weight.detach().float().cpu()
            B = mod.lora_B["default"].weight.detach().float().cpu()
            deltas[base_key] = mod.scaling["default"] * (B @ A)
    del m, opt, sched
    gc.collect(); torch.cuda.empty_cache()
    return deltas

tau_a = train_lora_delta(train_a, "アダプタ A (JSON 抽出)", seed=0)
tau_b = train_lora_delta(train_b, "アダプタ B (3 行要約)", seed=1)

# 層・モジュール種別ごとのタスクベクトル統計量（L2 ノルム、コサイン類似度、符号衝突率）
diag_rows = []
for mod_type in TARGET_MODULES:
    keys = [k for k in tau_a if f".{mod_type}." in k]
    va = torch.cat([tau_a[k].flatten() for k in keys])
    vb = torch.cat([tau_b[k].flatten() for k in keys])
    nz = (va != 0) & (vb != 0)
    conflict = ((va[nz] > 0) != (vb[nz] > 0)).float().mean().item()
    diag_rows.append({"モジュール": mod_type, "層数": len(keys), "||τA||_2": va.norm().item(), "||τB||_2": vb.norm().item(),
                      "cos(τA, τB)": torch.nn.functional.cosine_similarity(va[None], vb[None]).item(), "符号衝突率": conflict})
all_a = torch.cat([v.flatten() for v in tau_a.values()])
all_b = torch.cat([v.flatten() for v in tau_b.values()])
nz_all = (all_a != 0) & (all_b != 0)
diag_rows.append({"モジュール": "全対象層合計", "層数": len(tau_a), "||τA||_2": all_a.norm().item(), "||τB||_2": all_b.norm().item(),
                  "cos(τA, τB)": torch.nn.functional.cosine_similarity(all_a[None], all_b[None]).item(),
                  "符号衝突率": ((all_a[nz_all] > 0) != (all_b[nz_all] > 0)).float().mean().item()})
print("\n--- タスクベクトルの層別統計 ---")
print(pd.DataFrame(diag_rows).round(4).to_string(index=False))
```

```python
# スケーリング（タスク算術）とマルチタスク合成（TIES-Merging）の評価
def score_a(pred, gold):
    s = pred.strip()
    if s.startswith("```") or s.endswith("```"):
        return False
    try:
        obj = json.loads(s)
    except Exception:
        return False
    return isinstance(obj, dict) and set(obj.keys()) == set(gold.keys()) and all(obj[k] == gold[k] and type(obj[k]) is type(gold[k]) for k in gold)

POLITE_END = re.compile(r"(です|ます|ました|ません|でした|ください|ましょう)[。.]?$")
def score_b(pred, facts):
    lines = [ln.strip() for ln in pred.strip().splitlines() if ln.strip()]
    prefixes = ("- 【発生】", "- 【原因】", "- 【対応】")
    return (len(lines) == 3 and all(ln.startswith(p) for ln, p in zip(lines, prefixes))
            and all(POLITE_END.search(ln) for ln in lines) and all(f in "\n".join(lines) for f in facts))

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map="auto").eval()
params = dict(base_model.named_parameters())
base_w = {k: params[k].detach().clone() for k in tau_a}

def apply_delta(delta_dict, lam=1.0):
    with torch.no_grad():
        for k, w0 in base_w.items():
            d = delta_dict.get(k)
            params[k].data.copy_(w0 if d is None else (w0.float() + lam * d.to(w0.device)).to(w0.dtype))

@torch.no_grad()
def eval_task(items, fn, bs=24, max_new_tokens=110):
    tokenizer.padding_side = "left"
    prompts = [[{"role": "user", "content": it["prompt"]}] for it in items]
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tokenizer.apply_chat_template(prompts[i:i + bs], add_generation_prompt=True, return_tensors="pt", return_dict=True, padding=True).to(base_model.device)
        gen = base_model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)[:, enc["input_ids"].shape[1]:]
        outs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return np.array([fn(o, it["gold"] if "gold" in it else it["facts"]) for o, it in zip(outs, items)], dtype=float), outs

# 1. タスク算術の係数スケーリング: θ_base + λ * τ_A
scale_rows, scale_samples = [], {}
for lam in [-0.5, 0.0, 0.5, 1.0, 1.5]:
    apply_delta(tau_a, lam=lam)
    ha, oa = eval_task(test_a, score_a)
    hb, _ = eval_task(test_b, score_b)
    scale_rows.append({"λ": lam, "課題 A": ha.mean(), "課題 B": hb.mean()})
    scale_samples[lam] = oa[0].replace("\n", " ")[:68]
print("--- タスク算術スケーリング (θ_base + λ τ_A) ---")
print(pd.DataFrame(scale_rows).round(3).to_string(index=False))

# 2. TIES-Merging (Trim -> Elect Sign -> Disjoint Mean)
def ties_merge(task_vectors, trim_pct=50.0, lam=1.0):
    merged = {}
    for k in task_vectors[0]:
        trimmed = []
        for tv in task_vectors:
            v = tv[k]
            if trim_pct > 0:
                thresh = torch.quantile(v.abs().flatten(), trim_pct / 100.0)
                v = torch.where(v.abs() >= thresh, v, torch.zeros_like(v))
            trimmed.append(v)
        T = torch.stack(trimmed, dim=0)
        elected = torch.sign(T.sum(dim=0))
        agree = (torch.sign(T) == elected.unsqueeze(0)) & (T != 0)
        merged[k] = lam * ((T * agree).sum(dim=0) / agree.sum(dim=0).clamp(min=1))
    return merged

def ci_str(hits, seed=0):
    boot = hits[np.random.default_rng(seed).integers(0, len(hits), (2000, len(hits)))].mean(1)
    return f"{hits.mean():.3f} [{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]"

def avg_ci_str(ha, hb, seed=0):
    rng_b = np.random.default_rng(seed)
    ia = rng_b.integers(0, len(ha), (2000, len(ha)))
    ib = rng_b.integers(0, len(hb), (2000, len(hb)))
    boot = 0.5 * (ha[ia].mean(1) + hb[ib].mean(1))
    return f"{0.5 * (ha.mean() + hb.mean()):.3f} [{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]"

merge_configs = [
    ("ベースモデル (λ=0)", {}),
    ("アダプタ A のみ (τA)", tau_a),
    ("アダプタ B のみ (τB)", tau_b),
    ("単純加算 (τA + τB)", {k: tau_a[k] + tau_b[k] for k in tau_a}),
    ("縮小加算 (0.8(τA + τB))", {k: 0.8 * (tau_a[k] + tau_b[k]) for k in tau_a}),
    ("単純平均 (0.5(τA + τB))", {k: 0.5 * (tau_a[k] + tau_b[k]) for k in tau_a}),
    ("TIES (trim=50%, λ=1.0)", ties_merge([tau_a, tau_b], 50.0, 1.0)),
    ("TIES (trim=70%, λ=1.0)", ties_merge([tau_a, tau_b], 70.0, 1.0)),
]
merge_rows, saved_outs = [], {}
for label, tv in merge_configs:
    apply_delta(tv, 1.0)
    ha, oa = eval_task(test_a, score_a)
    hb, ob = eval_task(test_b, score_b)
    merge_rows.append({"手法": label, "課題 A (95% CI)": ci_str(ha), "課題 B (95% CI)": ci_str(hb), "平均 (95% CI)": avg_ci_str(ha, hb)})
    saved_outs[label] = (oa, ob)
print("\n--- マルチタスク統合の比較 (各 N=24) ---")
print(pd.DataFrame(merge_rows).to_string(index=False))
```

```python
# スケーリングとマージによる出力の変化例を表示し、GPU メモリを解放する
print("--- λ による出力の変化（課題 A 先頭例）---")
for lam in [-0.5, 0.0, 1.0, 1.5]:
    print(f"  λ={lam:+.1f}: {scale_samples[lam]}")

print("\n--- マージ手法ごとの生成例（課題 A[0] / 課題 B[0]）---")
for label in ["ベースモデル (λ=0)", "アダプタ A のみ (τA)", "単純加算 (τA + τB)", "単純平均 (0.5(τA + τB))", "TIES (trim=50%, λ=1.0)"]:
    print(f"[{label}] A: {saved_outs[label][0][0].replace(chr(10), ' ')[:90]}")
for label in ["ベースモデル (λ=0)", "アダプタ B のみ (τB)", "単純加算 (τA + τB)", "TIES (trim=50%, λ=1.0)"]:
    print(f"[{label}] B: {saved_outs[label][1][0].replace(chr(10), ' / ')[:95]}")

print(f"\nGPU メモリの最大使用量: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
del base_model, base_w, params, tau_a, tau_b
gc.collect(); torch.cuda.empty_cache()
```

- 学習とタスクベクトルの性質: アダプタ A（JSON 抽出、学習 10.1 秒、損失 0.3846 → 0.0005）とアダプタ B（3 行要約、学習 9.1 秒、損失 0.541 → 0.0004）を 168 個の線形層から $\Delta W_t = (\alpha/r)(B_t @ A_t)$ として抽出すると、全対象層合計の L2 ノルムは $\|\tau_A\|_2 = 1.8251$、$\|\tau_B\|_2 = 1.8132$ で、うち MLP 層（`gate_proj` が 1.1165 / 1.1128、`up_proj` が 1.1126 / 1.1020）が注意機構の投影層 (0.3913〜0.5954) より大きな変化を担っていた。両アダプタの全体コサイン類似度は 0.0002 とほぼ直交する一方、非ゼロ座標の符号衝突率は全層合計で 0.4999（モジュール種別ごとでも 0.4999〜0.5000）に達し、約半数の座標で逆向きの更新が起きている。
- スケーリング（タスク算術 $\theta_{\text{base}} + \lambda \tau_A$）: 課題 A の正解率は $\lambda = 0.0$ の 0.000（```` ```json ```` で囲み日本語キーを出す）から $\lambda = 0.5$ で 0.583、$\lambda = 1.0$ で 1.000（正規化 JSON のみ出力）へ上昇した。逆に $\lambda = -0.5$ では JSON を作らず「社内通知の所定の正規化 JSON形式に変換するために…」という説明文のみを返して課題能力が打ち消され (0.000)、過大な $\lambda = 1.5$ でも末尾のキーや終了制御が崩れて 0.000 に急落した。なお課題 A 用アダプタの適用で課題 B が解けることはなく、どの $\lambda$ でも課題 B は 0.000 のままだった。
- マルチタスク統合の比較（各 N=24、カッコ内は 95% ブートストラップ信頼区間）: ベースモデルは 0.000 / 0.000（平均 0.000）、単一アダプタ A は 1.000 / 0.000（平均 0.500）、単一アダプタ B は 0.000 / 1.000（平均 0.500）だった。2 つをそのまま足す単純加算 ($\tau_A + \tau_B$) では課題 B は 0.958 [0.875, 1.000] を保つ一方、課題 A が 0.708 [0.541, 0.875] へ低下し（平均 0.833 [0.729, 0.917]）、課題 A の先頭例でも `category` が `"billing"` から `"support"` へ誤反転した。単純平均 ($0.5(\tau_A + \tau_B)$) は課題 B を 1.000 に保ったまま課題 A を 0.083 [0.000, 0.208]（平均 0.542 [0.500, 0.604]）まで大きく落とし、縮小加算 ($0.8(\tau_A + \tau_B)$) は 0.833 / 0.958（平均 0.896 [0.812, 0.979]）だった。
- TIES-Merging の効果とハイパーパラメータ感度: 下位 50% の微小成分を除去 (Trim) して多数派符号を選び (Elect)、同符号成分のみで平均する `TIES (trim=50%, λ=1.0)` は課題 A 0.875 [0.750, 1.000]、課題 B 1.000 [1.000, 1.000]、平均 0.938 [0.875, 1.000] と最高の統合性能を示し、課題 A 先頭例の `"billing"` 誤判定も解消した。ただし密な低ランク積 $B @ A$ に対して `trim=70%` まで削ると有効な小成分まで失われ、課題 A が 0.583 [0.375, 0.792]（平均 0.792 [0.688, 0.896]）に下がった。GPU メモリの最大使用量は 5.96 GB だった（T4 などの float16 環境では実行時間が長くなる点に注意）。
- LoRA マージを安定させるには、PEFT の高水準マージ API の仕様変化に依存せず、`scaling * (B @ A)` を `float32` で明示計算してベース重みへ加算・調整する方法が確実で、任意の線形結合や TIES-Merging をそのまま実装できる。
- 単純平均 ($0.5(\tau_A + \tau_B)$) は課題固有パラメータまで半減させるため、$\lambda = 0.5$ 付近で性能が落ちやすい構造化出力タスクを著しく損なう。一方 TIES-Merging の Disjoint Mean は、トリミング後に片方だけ非ゼロになった座標を 1 で割るため課題固有の大きさを保てる。ただしトリミング率が高すぎると密行列の情報が落ちるため、検証データで `trim`（50% 前後）と全体係数 $\lambda$ を調整する。
- 単一タスクアダプタを過剰適合（エポック数を増やしすぎて損失がほぼ 0 になる状態）させると、$\lambda = 1.0$ 自体が過スケールになりマージ時の干渉が増えやすい。ベースライン・単体アダプタ・統合モデルを同じ評価セットと信頼区間で並べて確認する。
- 単一課題向けの LoRA 学習手順と、モデル `sbintuitions/sarashina2.2-0.5b` の `load_tokenizer` による Unigram 復元については、この章の「ベースモデルを LoRA で指示チューニングする (SFT)」および大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」を参照。
- 純粋な PyTorch ループでの LoRA 学習（`trl` を使わない構成）は、この章の「選好データから報酬モデルを学習して応答を採点する（Bradley-Terry モデル）」も参照。
- タスクベクトルのスケーリングと同じようにモデル出力の形式を厳密に制御したい場合は、大規模言語モデルの章の「制約付きデコーディングで LLM の出力を JSON スキーマ・正規表現・選択肢に従わせる (outlines)」を参照。
