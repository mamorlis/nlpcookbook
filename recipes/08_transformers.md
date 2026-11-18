# Transformers の基本操作

Hugging Face Transformers と Datasets で、事前学習済みモデルを動かすときの定番の操作。fill-mask と NLI、文脈に応じた単語ベクトル、次のトークンの予測と生成のデコード方法、バッチ推論とモデルの保存、データセットの読み込み、BERT の追加の事前学習と小さな GPT の事前学習。言語モデルの確率や内部の表現を調べる方法は「言語モデルの確率と内部を調べる」の章を参照。

## マスクされた単語を BERT に予測させる (fill-mask) {fill_mask}

文中の空欄に入る単語を BERT に予測させたい。モデルが何を知っているかを手軽に調べられる。

```python
!pip install -q fugashi unidic-lite
```

```python
from transformers import pipeline

fill = pipeline("fill-mask", model="tohoku-nlp/bert-base-japanese-v3")
for text in ["日本の首都は[MASK]です。", "私は毎朝[MASK]を飲みます。", "自然言語処理は[MASK]の一分野である。"]:
    print(text)
    for r in fill(text, top_k=5):
        print(f"  {r['token_str']:8}{r['score']:.3f}")
```

```python
# 候補を指定して、その中での確率を比べる
for r in fill("彼は医者なので、病院で[MASK]いる。", targets=["働い", "遊ん", "寝"]):
    print(r["token_str"], round(r["score"], 4))
```

- マスクトークンはモデルによって違う（`[MASK]`、`<mask>` など）。`fill.tokenizer.mask_token` で確認できる。
- 予測できるのは 1 トークンだけ。複数トークンにまたがる語（サブワードに分割される語）は直接は予測できない。

## BERT から文脈に応じた単語ベクトルを取り出す {contextual_word_vectors}

同じ単語でも文脈によって意味が違う（bank = 銀行 / 土手）。文脈を反映した単語ベクトルを取り出して比べたい。

```python
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").eval()

def word_vector(sentence, word):
    words = sentence.split()
    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**enc).last_hidden_state[0]          # (トークン数, 768)
    positions = [i for i, w in enumerate(enc.word_ids()) if w == words.index(word)]
    return hidden[positions].mean(dim=0)                    # サブワードが複数なら平均

sents = ["i deposited money in the bank yesterday", "the bank raised its interest rates",
         "we sat on the bank of the river", "the boat drifted toward the muddy bank"]
vecs = torch.stack([word_vector(s, "bank") for s in sents])
sim = torch.nn.functional.cosine_similarity(vecs[:, None], vecs[None], dim=-1)
for s, row in zip(sents, sim):
    print(" ".join(f"{x:.2f}" for x in row), "|", s)
```

- 金融の bank 同士、川岸の bank 同士の類似度が高くなる。word2vec では 1 つのベクトルに混ざっていた語義が分離される。
- どの層を使うかで性質が変わる。`output_hidden_states=True` で全層を取り出し、最後の 4 層を平均する方法もよく使われる。

## 2 つの文の含意関係を判定する (NLI) {nli}

前提文から仮説文が「言える（含意）」「矛盾する」「どちらでもない（中立）」のどれかを判定したい。

```python
import torch
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast

name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"   # 多言語 NLI モデル
# transformers 5 系の DebertaV2Tokenizer は全角の記号や英数字を [UNK] にしてしまう (issue #49622) ので、tokenizer.json から直接読む
tokenizer = PreTrainedTokenizerFast.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name).eval()
print(model.config.id2label)

def nli(premise, hypothesis):
    enc = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        probs = model(**enc).logits.softmax(-1)[0]
    return {model.config.id2label[i]: round(p.item(), 3) for i, p in enumerate(probs)}

premise = "男性が公園でギターを弾いている。"
for hyp in ["人が楽器を演奏している。", "男性が屋外にいる。", "男性は家で寝ている。", "男性はプロの音楽家だ。", "A man is playing music."]:
    r = nli(premise, hyp)
    print(max(r, key=r.get), r, hyp)
```

```python
# 応用: 要約や生成文が元の文書と矛盾していないかのチェック (事実整合性)
source = "同社の 2023 年度の売上高は前年比 5% 増の 120 億円だった。営業利益は 8 億円で、前年から減少した。"
for claim in ["売上高は増加した。", "営業利益は増加した。", "同社は海外に進出した。"]:
    r = nli(source, claim)
    print(max(r, key=r.get), claim)
```

- NLI は「ゼロショット分類」「事実整合性の評価」「言い換えの検証」などの部品として広く使われる。
- 語彙の重なりが大きいと含意と判定しやすい、といった表面的な手がかりへの依存（アーティファクト）が知られている。過信は禁物。
- 日本語のデータセットには JNLI (JGLUE)、JSICK、JaNLI などがある。
- mDeBERTa のトークナイザーの不具合: transformers 5 系では、`AutoTokenizer` が返す `DebertaV2Tokenizer` が、全角の括弧や英数字（「（」「ＡＢＣ」など）を [UNK] にしてしまう（transformers の issue #49622。5.0〜5.23 で確認、4.57 では正しい）。エラーは出ないまま入力の一部が失われる。リポジトリにある tokenizer.json を `PreTrainedTokenizerFast.from_pretrained` で直接読めば、正しく分割される。`pipeline` で使うときは、こうして読んだトークナイザーを `tokenizer=` に渡す。上流で直れば不要になる。

## 次に来るトークンの確率分布を見る {next_token_probs}

言語モデルが「次に何が来る」と予測しているのか、上位候補と確率を直接見たい。

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

def next_tokens(prefix, k=6):
    ids = tokenizer(tokenizer.bos_token + prefix, add_special_tokens=False, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logits = model(ids).logits[0, -1]            # 最後の位置の出力 = 次のトークンの予測
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, k)
    entropy = -(probs * torch.log2(probs + 1e-12)).sum().item()
    print(f"{prefix} … (エントロピー {entropy:.2f} bit)")
    print("   ", "  ".join(f"{tokenizer.decode(i)!r}:{p:.2f}" for p, i in zip(top.values.tolist(), top.indices.tolist())))

for prefix in ["日本の首都は", "むかしむかし、あるところに", "今日の天気は", "1 + 1 ="]:
    next_tokens(prefix)
```

```python
# 特定の候補同士の確率を比べる (選択式問題を解かせる、など)
def compare(prefix, candidates):
    ids = tokenizer(tokenizer.bos_token + prefix, add_special_tokens=False, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logp = torch.log_softmax(model(ids).logits[0, -1], dim=-1)
    return {c: round(logp[tokenizer(c, add_special_tokens=False)["input_ids"][0]].item(), 2) for c in candidates}

print(compare("富士山があるのは", ["日本", "中国", "アメリカ"]))
```

- 言語モデルの出力は、語彙全体にわたる確率分布。生成とは、この分布から 1 つ選んで追加することの繰り返し。
- エントロピーが低い = 次の語がほぼ決まっている。高い = 候補が多くて迷っている。
- `compare` は各候補の最初の 1 トークンだけを比べている点に注意。複数トークンの候補は、全トークンの対数確率を合計する。

## テキスト生成のデコード方法（貪欲法・ビームサーチ・サンプリング）を比べる {generation_decoding}

言語モデルに文章の続きを生成させたい。`temperature` や `top_p` などの設定で出力がどう変わるかを確かめたい。

```python
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"   # 0.5B パラメータの小さな日本語 LLM (CPU でも動く)
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
model = AutoModelForCausalLM.from_pretrained(name).eval()

messages = [{"role": "user", "content": "猫を主人公にした物語の書き出しを、一文だけ書いてください。"}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

def generate(**kwargs):
    torch.manual_seed(0)
    out = model.generate(**inputs, max_new_tokens=60, pad_token_id=tokenizer.eos_token_id, **kwargs)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

print("貪欲法          :", generate(do_sample=False))
print("ビームサーチ (4):", generate(do_sample=False, num_beams=4, no_repeat_ngram_size=3))
# サンプリング: 3 つのパラメータを毎回すべて明示する (top_p=1.0, top_k=0 が「制限なし」)
print("temperature=0.3 :", generate(do_sample=True, temperature=0.3, top_p=1.0, top_k=0))
print("temperature=1.0 :", generate(do_sample=True, temperature=1.0, top_p=1.0, top_k=0))
print("temperature=1.5 :", generate(do_sample=True, temperature=1.5, top_p=1.0, top_k=0))
print("top_p=0.9       :", generate(do_sample=True, temperature=1.0, top_p=0.9, top_k=0))
print("top_k=20        :", generate(do_sample=True, temperature=1.0, top_p=1.0, top_k=20))
```

- 貪欲法・ビームサーチは決定的で無難。サンプリングは多様だが、temperature を上げすぎると破綻する。
- `top_p`（nucleus サンプリング）は確率の累積が p になるまでの上位トークンだけから選ぶ。`top_k` は上位 k 個から選ぶ。
- 翻訳や要約など正解が定まるタスクはビームサーチ、創作や対話はサンプリング（temperature 0.7 前後 + top_p 0.9 前後）が定番。
- `max_new_tokens` は生成する長さの上限。指定しないと短い既定値で打ち切られる。
- モデルに付属の `generation_config.json` に既定値（temperature や top_p など）が入っていることがある。指定しなかったパラメータにはそれが使われるので、`print(model.generation_config)` で確認する。

## バッチ処理と GPU で推論を高速化する {batch_inference gpu}

数万件のテキストに対してモデルの推論を回したい。1 件ずつ処理すると遅すぎる。

```python
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()

base = ["とても良かった。", "最悪だった。二度と行かない。", "普通でした。可もなく不可もなく、値段相応という印象です。" * 4]
texts = [base[2] if i % 8 == 0 else base[i % 2] for i in range(96)]   # 8 件に 1 件だけ長い文

@torch.inference_mode()
def predict(texts, batch_size, sort_by_length=False):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i])) if sort_by_length else list(range(len(texts)))
    preds = [None] * len(texts)
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        batch = tokenizer([texts[j] for j in idx], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        for j, p in zip(idx, model(**batch).logits.argmax(-1).tolist()):
            preds[j] = model.config.id2label[p]
    return preds

print("device:", device)
predict(texts[:32], batch_size=32)   # 初回は GPU の初期化に時間がかかるので、計測の前に 1 回動かしておく
for label, kwargs in [("1 件ずつ", dict(batch_size=1)), ("バッチ 32", dict(batch_size=32)), ("バッチ 32 + 長さでソート", dict(batch_size=32, sort_by_length=True))]:
    t = time.perf_counter()
    preds = predict(texts, **kwargs)
    print(f"{label:22} {time.perf_counter() - t:5.2f} 秒  {preds[:3]}")
```

- `torch.inference_mode()`（または `no_grad()`）で勾配計算を止める。忘れるとメモリを浪費し、遅くなる。
- 長さの近い文を同じバッチにまとめると、パディングが減って速くなる。結果は元の順番に戻す。
- この 96 件では、検証の GPU で 1 件ずつ 0.31 秒 → バッチ + 長さでソートで 0.04 秒（約 8 倍）。T4 では値が変わる。件数が多いほど、またモデルが大きいほど差は広がる。さらに `model.half()`（半精度）で速度とメモリが改善する。
- CPU では、長さを揃えずにバッチにすると、長い文に合わせたパディングの計算が重く、1 件ずつより遅くなることがある。
- GPU の処理時間を測るときは、最初の 1 回（初期化やカーネルの準備を含む）を計測から外す。
- `pipeline(..., device=0, batch_size=32)` でも同様のバッチ処理ができる。

## モデルのパラメータ数とメモリ使用量を調べる {model_size}

このモデルは Colab の GPU（T4 は 16GB）に載るのか、パラメータ数とメモリ消費の目安を知りたい。

```python
import torch
from transformers import AutoConfig, AutoModel

for name in ["intfloat/multilingual-e5-small", "bert-base-uncased", "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"]:
    model = AutoModel.from_pretrained(name)
    n = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings().weight.numel()
    print(f"{name:46} {n / 1e6:7.1f}M パラメータ (うち埋め込み {emb / n:.0%})  float32 で {model.get_memory_footprint() / 1e9:.2f} GB")
```

```python
# モデルを読み込まずに設定だけ見る
cfg = AutoConfig.from_pretrained("sbintuitions/sarashina2.2-0.5b-instruct-v0.1")
print({k: getattr(cfg, k) for k in ["hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size", "max_position_embeddings"]})

# パラメータ数からメモリの目安を計算する
def memory_gb(n_params, bits):
    return n_params * bits / 8 / 1e9

for label, n in [("0.5B", 0.5e9), ("7B", 7e9), ("70B", 70e9)]:
    print(f"{label:5} float32 {memory_gb(n, 32):6.1f} GB / float16 {memory_gb(n, 16):6.1f} GB / 8bit {memory_gb(n, 8):5.1f} GB / 4bit {memory_gb(n, 4):5.1f} GB")
```

- 推論時のメモリ ≒ パラメータ数 × 1 パラメータのバイト数 + α（活性化や KV キャッシュ）。学習時は勾配とオプティマイザの状態で、さらに 3〜4 倍必要。
- T4 (16GB) なら、float16 で 7B 級がぎりぎり、4bit 量子化なら 13B 級まで載る。
- `torch.cuda.memory_allocated()` や `!nvidia-smi` で実際の使用量を確認できる。

## モデルを保存・読み込みする、Hub にアップロードする {save_load_model}

ファインチューニングしたモデルをファイルに保存し、あとで読み込んで使いたい。Hugging Face Hub で共有したい。

```python
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3, id2label={0: "negative", 1: "neutral", 2: "positive"})

model.save_pretrained("my-model")       # 重み (model.safetensors) と設定 (config.json)
tokenizer.save_pretrained("my-model")   # トークナイザも必ず同じ場所に保存する
for f in sorted(os.listdir("my-model")):
    print(f"{os.path.getsize('my-model/' + f) / 1e6:8.2f} MB  {f}")
```

```python
from transformers import pipeline

clf = pipeline("text-classification", model="my-model")   # ディレクトリ名を渡すだけで読み込める
print(clf("これはテストです。"))                           # (未学習の分類ヘッドなので予測はでたらめ)
print(clf.model.config.id2label)
```

```python
# Hub にアップロードする場合 (書き込み権限のあるトークンが必要)
# from huggingface_hub import login
# login()                                          # または Colab のシークレットに HF_TOKEN を登録
# model.push_to_hub("ユーザー名/my-model", private=True)
# tokenizer.push_to_hub("ユーザー名/my-model", private=True)
```

- `id2label` を設定しておくと、推論結果が `LABEL_0` ではなく意味のあるラベル名で返る。
- Colab のランタイムは切断されるとファイルが消える。Drive にコピーするか Hub に上げておく。
- 重みは safetensors 形式で保存される。古い `pytorch_model.bin` (pickle) と違い、読み込み時に任意コードが実行される心配がない。

## Hub のモデルのキャッシュ場所を確認・変更する {hf_cache}

ダウンロードしたモデルがどこに保存されているか知りたい。ディスクを圧迫しているキャッシュを消したい。Colab で Drive に置いて再ダウンロードを避けたい。

```python
from huggingface_hub import hf_hub_download, scan_cache_dir, snapshot_download

# 1 ファイルだけダウンロード (戻り値はキャッシュ内のパス)
path = hf_hub_download("intfloat/multilingual-e5-small", "config.json")
print(path)

# リポジトリの一部をまとめてダウンロード
local = snapshot_download("intfloat/multilingual-e5-small", allow_patterns=["*.json", "*.txt"])
print(local)
```

```python
info = scan_cache_dir()
print(f"キャッシュ全体: {info.size_on_disk / 1e9:.2f} GB")
for repo in sorted(info.repos, key=lambda r: -r.size_on_disk)[:5]:
    print(f"  {repo.size_on_disk / 1e6:8.1f} MB  {repo.repo_type}  {repo.repo_id}")
```

```python
# 特定のモデルをキャッシュから削除する
repo = next(r for r in info.repos if r.repo_id == "intfloat/multilingual-e5-small")
strategy = info.delete_revisions(*[rev.commit_hash for rev in repo.revisions])
print("解放される容量:", strategy.expected_freed_size_str)
strategy.execute()
```

- 既定のキャッシュ先は `~/.cache/huggingface/hub`。環境変数 `HF_HOME` で変更できる（import より前に設定する）。
- Colab で Drive に置く例: `os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"`。ただし Drive は読み書きが遅く、容量も消費する。
- 個別に指定するなら `from_pretrained(name, cache_dir=...)`。

## Datasets でデータを読み込む・加工する {datasets_basics}

Hugging Face Hub のデータセットを読み込んで、フィルタ・変換・分割などの基本操作をしたい。

```python
!pip install -q datasets
```

```python
from datasets import load_dataset

ds = load_dataset("stanfordnlp/imdb")
print(ds)
print(ds["train"].features)
print(ds["train"][0]["text"][:100], "| label:", ds["train"][0]["label"])
```

```python
train = ds["train"].shuffle(seed=0).select(range(2000))                      # シャッフルして先頭 2000 件
train = train.filter(lambda x: len(x["text"]) < 2000)                         # 条件で絞る
train = train.map(lambda x: {"n_words": len(x["text"].split())})              # 列を追加
train = train.map(lambda b: {"text": [t.replace("<br />", " ") for t in b["text"]]}, batched=True)   # バッチで変換 (速い)
print(train)

split = train.train_test_split(test_size=0.1, stratify_by_column="label", seed=0)   # 層化分割
print({k: len(v) for k, v in split.items()}, "ラベル名:", split["train"].features["label"].names)
```

```python
df = split["train"].to_pandas()                 # pandas へ
print(df.groupby("label")["n_words"].mean())

from datasets import Dataset
mine = Dataset.from_dict({"text": ["良い", "悪い"], "label": [1, 0]})   # 自分のデータから作る
mine.to_json("mine.jsonl", force_ascii=False)                            # JSONL で保存
print(load_dataset("json", data_files="mine.jsonl")["train"][0])
```

- `map` の結果はディスクにキャッシュされ、同じ処理は 2 回目から一瞬で終わる。
- トークナイズのような重い処理は `batched=True` で。`num_proc=2` で並列化もできる。
- 最近の `datasets` は、読み込み用の Python スクリプトを含む古い形式のデータセットに対応していない。Parquet 形式で公開されているものを選ぶ。

## 巨大なデータセットをダウンロードせずにストリーミングで読む {datasets_streaming}

Wikipedia 全体のような数 GB〜数百 GB のデータセットから、先頭の一部だけ、あるいは条件に合うものだけを取り出したい。

```python
!pip install -q datasets
```

```python
from datasets import load_dataset

wiki = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)   # 日本語版 Wikipedia (約 140 万記事)
for article in wiki.take(3):
    print(article["title"], "|", len(article["text"]), "文字 |", article["text"][:40].replace("\n", " "))
```

```python
# 条件に合う記事だけを必要な数だけ集める (全体はダウンロードされない)
nlp_articles = wiki.filter(lambda a: "自然言語処理" in a["text"]).take(5)
print([a["title"] for a in nlp_articles])

# シャッフル (バッファ内での近似的なシャッフル) と変換
sample = wiki.shuffle(seed=0, buffer_size=1000).map(lambda a: {"first": a["text"].split("。")[0]}).take(3)
for a in sample:
    print(a["title"], "→", a["first"][:50])
```

- `streaming=True` だと `IterableDataset` になり、`ds[0]` のような添字アクセスや `len()` は使えない。
- `shuffle` はバッファサイズ分しか混ざらない。完全なランダムサンプルが必要なら、ダウンロードして通常のデータセットとして扱う。
- 取り出したものを保存するなら `Dataset.from_list(list(wiki.take(10000)))`。

## BERT を自分の分野のテキストで追加で事前学習する（ドメイン適応） {mlm_domain_adaptation gpu}

BERT の事前学習のデータとは書き方の違う分野（ここでは旧仮名遣いの近代文学）のテキストで、マスク言語モデルの学習を続けたい。その分野の文の穴埋めがうまくなるか、分類の精度に効くかも確かめる。

```python
!pip install -q fugashi unidic-lite datasets
```

```python
import random
from collections import Counter
import numpy as np
import torch
from datasets import load_dataset

random.seed(0); np.random.seed(0); torch.manual_seed(0)

# 青空文庫の作品のテキストを整形したもの (globis-university/aozorabunko-clean、データセットは CC BY 4.0。
# ここで使う作品は「作品著作権フラグ」が「なし」の、著作権の切れたもの)
ds = load_dataset("globis-university/aozorabunko-clean", split="train")
kyu = [ex for ex in ds if ex["meta"]["文字遣い種別"] == "新字旧仮名" and ex["meta"]["作品著作権フラグ"] == "なし"]
shin = [ex for ex in ds if ex["meta"]["文字遣い種別"] == "新字新仮名" and ex["meta"]["作品著作権フラグ"] == "なし"]

# 作者単位で、追加学習に使う作者 (8 割) と評価だけに使う作者 (2 割) に分ける
authors = sorted({ex["meta"]["人物ID"] for ex in kyu})
random.Random(0).shuffle(authors)
test_authors = set(authors[:len(authors) // 5])
kyu_train = [ex for ex in kyu if ex["meta"]["人物ID"] not in test_authors]
kyu_test = [ex for ex in kyu if ex["meta"]["人物ID"] in test_authors]
print(f"旧仮名の作品: 学習用 {len(kyu_train)} 作品 / 評価用 {len(kyu_test)} 作品 (作者 {len(test_authors)} 人)")

def paragraphs(works, min_len=20):
    return [p.strip() for ex in works for p in ex["text"].split("\n") if len(p.strip()) >= min_len]

print(paragraphs(kyu_test)[:2])
```

```python
from transformers import AutoTokenizer

name = "tohoku-nlp/bert-base-japanese-v3"   # 東北大学の日本語 BERT (Apache 2.0)
tok = AutoTokenizer.from_pretrained(name)
print(tok.tokenize("さうして私はかう思ふのである。"))   # 旧仮名は細かく分かれる

def chunks(texts, n, L=128):
    """段落を順にトークン化してつなぎ、[CLS] + (L-2) トークン + [SEP] の長さ L の列に切る
    (長い段落で「Token indices sequence length is longer than ...」の警告が出るが、L に切るので問題ない)"""
    out, buf = [], []
    for t in texts:
        buf += tok(t, add_special_tokens=False)["input_ids"]
        while len(buf) >= L - 2 and len(out) < n:
            out.append([tok.cls_token_id] + buf[:L - 2] + [tok.sep_token_id]); buf = buf[L - 2:]
        if len(out) == n:
            break
    return out

train_paras = paragraphs(kyu_train)
random.Random(0).shuffle(train_paras)                    # 作品が偏らないよう段落を混ぜる
N_TRAIN = 40000                                          # 約 500 万トークン
train_chunks = chunks(train_paras, N_TRAIN)

# 評価用: 旧仮名 (学習に使っていない作者)、新仮名の文学作品、Wikipedia の 3 種類を 1000 列ずつ
def sample_paras(works, k=3000):
    p = paragraphs(works); random.Random(0).shuffle(p); return p[:k]
wiki = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)   # CC BY-SA 3.0 / GFDL
wiki_paras = [p for a in wiki.take(300) for p in a["text"].split("\n") if len(p) >= 20]
eval_chunks = {"旧仮名 (評価用の作者)": chunks(sample_paras(kyu_test), 1000),
               "新仮名の文学作品": chunks(sample_paras(shin), 1000),
               "Wikipedia": chunks(wiki_paras, 1000)}
print("学習:", len(train_chunks), "列", {k: len(v) for k, v in eval_chunks.items()})
print("[UNK] の割合:", f"{sum(c.count(tok.unk_token_id) for c in train_chunks) / (len(train_chunks) * 128):.4%}")
```

```python
import math
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

# このモデルは単語単位のマスク (whole word masking) で事前学習されている。MeCab の 1 語が「さ ##う」のように
# 複数のサブワードに分かれたときは、まとめて隠す (「さ」だけ見せて「##う」を当てさせると簡単すぎる)
is_sub = torch.zeros(len(tok), dtype=torch.bool)
is_sub[[i for t, i in tok.get_vocab().items() if t.startswith("##")]] = True

def word_mask(ids, p=0.15, generator=None):
    """語の先頭を確率 p で選び、その語のサブワードをすべて隠す位置にする"""
    start = ~is_sub[ids]
    pick = start & (torch.rand(ids.shape, generator=generator) < p) & (ids != tok.cls_token_id) & (ids != tok.sep_token_id)
    m = pick.clone()
    for t in range(1, ids.shape[1]):
        m[:, t] = torch.where(start[:, t], pick[:, t], m[:, t - 1])   # 「##」のサブワードは直前と同じ
    return m

def mask_fixed(ids, seed=0):
    """評価用に、隠す位置を固定する (隠した位置はすべて [MASK] にする)"""
    ids = torch.tensor(ids)
    m = word_mask(ids, generator=torch.Generator().manual_seed(seed))
    return ids.masked_fill(m, tok.mask_token_id), ids.masked_fill(~m, -100)

eval_sets = {k: mask_fixed(v) for k, v in eval_chunks.items()}

@torch.no_grad()
def mlm_eval(model, data, bs=64):
    """隠したトークンの平均損失 (負の対数尤度)、パープレキシティ、1 位の予測が当たった割合"""
    model.eval()
    inp, labels = data
    loss = correct = n = 0
    for i in range(0, len(inp), bs):
        x, y = inp[i:i + bs].to(model.device), labels[i:i + bs].to(model.device)
        logits = model(input_ids=x).logits[y != -100].float()
        loss += F.cross_entropy(logits, y[y != -100], reduction="sum").item()
        correct += (logits.argmax(-1) == y[y != -100]).sum().item(); n += (y != -100).sum().item()
    return {"損失": round(loss / n, 3), "PPL": round(math.exp(loss / n), 1), "1 位の正解率": round(correct / n, 3)}

model = AutoModelForMaskedLM.from_pretrained(name).to("cuda")   # 学習するので float32 で読み、計算だけ半精度にする
before = {k: mlm_eval(model, d) for k, d in eval_sets.items()}
for k, v in before.items():
    print(f"{k:16}", v)
```

```python
from transformers import pipeline

examples = ["雨が降つて[MASK]る。", "彼は何も言は[MASK]かつた。", "私はかう[MASK]ふのである。", "それは美しい[MASK]であつた。",
            "日本の首都は[MASK]です。", "私は毎朝[MASK]を飲みます。"]   # 最後の 2 つはこの章の「マスクされた単語を BERT に予測させる (fill-mask)」と同じ現代文

def show_fill(model):
    fill = pipeline("fill-mask", model=model, tokenizer=tok, device=model.device)
    for text in examples:
        print(f"{text:16}", "  ".join(f"{r['token_str']}({r['score']:.2f})" for r in fill(text, top_k=4)))

show_fill(model)
```

```python
from datasets import Dataset
from transformers import Trainer, TrainingArguments

def collate(examples):
    """追加の事前学習: 語の 15% を隠して当てる。隠す位置は BERT の事前学習と同じく 80% を [MASK]、10% をランダムな語、10% をそのままにする
    (DataCollatorForLanguageModeling の whole_word_mask は fast トークナイザーが必要で、このモデルでは使えない)"""
    ids = torch.tensor([e["input_ids"] for e in examples])
    m = word_mask(ids)
    r = torch.rand(ids.shape)
    inp = ids.masked_fill(m & (r < 0.8), tok.mask_token_id)
    rnd = m & (r >= 0.8) & (r < 0.9)
    inp[rnd] = torch.randint(len(tok), ids.shape)[rnd]
    return {"input_ids": inp, "labels": ids.masked_fill(~m, -100)}

BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
args = TrainingArguments(output_dir="mlm-out", num_train_epochs=1, learning_rate=5e-5, warmup_steps=100, weight_decay=0.01,
                         per_device_train_batch_size=32, bf16=BF16, fp16=not BF16, logging_steps=125,
                         save_strategy="no", report_to="none", seed=0)
trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_dict({"input_ids": train_chunks}),
                  data_collator=collate)
trainer.train()   # 1250 ステップ。T4 で 10 分前後
print(f"GPU メモリの最大使用量: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
trainer.save_model("bert-kyukana"); tok.save_pretrained("bert-kyukana")
```

```python
import pandas as pd

after = {k: mlm_eval(model, d) for k, d in eval_sets.items()}
print(pd.concat({"追加学習の前": pd.DataFrame(before).T, "後": pd.DataFrame(after).T}, axis=1).to_string())
show_fill(model)
```

```python
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, set_seed
from transformers.utils import logging as hf_logging

# 下流のタスク: 旧仮名の作品の段落が、小説 (NDC 913) か随筆・評論 (NDC 914) かを当てる。
# 学習は追加学習に使った作者、評価は評価用の作者の作品から取る (同じ作者や作品が両方に入らない)
def clf_data(works, per_class, seed):
    rng, rows = random.Random(seed), []
    for label, ndc in enumerate(["NDC 913", "NDC 914"]):
        ps = []
        for ex in works:
            if ex["meta"]["分類番号"] == ndc:
                p = paragraphs([ex], 50)
                ps += [(t, ex["meta"]["作品ID"]) for t in rng.sample(p, min(3, len(p)))]   # 1 作品から最大 3 段落
        rng.shuffle(ps)
        rows += [{"text": t, "work": w, "label": label} for t, w in ps[:per_class]]
    enc = tok([r["text"] for r in rows], truncation=True, max_length=128)
    return Dataset.from_dict({k: [r[k] for r in rows] for k in ["text", "work", "label"]} | dict(enc))

hf_logging.set_verbosity_error()   # 読み込みのたびに出る重みの一覧 (分類の層が新しく作られた、など) を表示しない
clf_train, clf_test = clf_data(kyu_train, 500, 0), clf_data(kyu_test, 600, 1)
print("学習", len(clf_train), Counter(clf_train["label"]), "/ 評価", len(clf_test), Counter(clf_test["label"]))

def finetune(path, seed):
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2, id2label={0: "小説", 1: "随筆・評論"},
                                                               label2id={"小説": 0, "随筆・評論": 1})
    if path != name:
        # 保存した MLM のモデルには、分類で使う pooler の層がない。条件をそろえるため、pooler は元のモデルのものを使い、
        # エンコーダー (埋め込みと 12 層) の重みだけを追加学習したものに差し替える
        missing = model.bert.load_state_dict(AutoModelForMaskedLM.from_pretrained(path).bert.state_dict(), strict=False).missing_keys
        assert all(k.startswith("pooler") for k in missing), missing
    args = TrainingArguments(output_dir="clf-out", num_train_epochs=3, learning_rate=3e-5, warmup_steps=10, weight_decay=0.01,
                             per_device_train_batch_size=32, per_device_eval_batch_size=128, bf16=BF16, fp16=not BF16,
                             save_strategy="no", logging_strategy="no", report_to="none", seed=seed)
    trainer = Trainer(model=model, args=args, train_dataset=clf_train.remove_columns(["text", "work"]),
                      data_collator=DataCollatorWithPadding(tok))
    trainer.train()
    return trainer.predict(clf_test.remove_columns(["text", "work"])).predictions.argmax(-1)

del model, trainer; torch.cuda.empty_cache()
y = np.array(clf_test["label"])
correct = {}
for label, path in [("元の BERT", name), ("追加学習した BERT", "bert-kyukana")]:   # T4 で 1 回 40 秒程度 × 6
    correct[label] = np.array([finetune(path, seed) == y for seed in range(3)])   # (シード, 評価の段落)
    print(f"{label:12} 正解率", " ".join(f"{a:.3f}" for a in correct[label].mean(1)), f" 平均 {correct[label].mean():.3f}")
```

```python
# 正解率の差のブートストラップ信頼区間: 同じ作品の段落は似ているので、作品を単位にリサンプリングする
works = np.array(clf_test["work"])
uniq = np.unique(works)
idx = {w: np.where(works == w)[0] for w in uniq}
diff = correct["追加学習した BERT"].mean(0) - correct["元の BERT"].mean(0)   # 段落ごとに、3 シードの平均の差
rng = np.random.default_rng(0)
boot = [np.mean(np.concatenate([diff[idx[w]] for w in rng.choice(uniq, len(uniq))])) for _ in range(2000)]
print(f"正解率の差 {diff.mean():+.3f}  95% 信頼区間 [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]  (作品 {len(uniq)} 件)")
```

- 事前学習済みのモデルを、ラベルのない分野のテキストでマスク言語モデルとして学習し直してから微調整する方法は、ドメイン適応事前学習 (domain-adaptive pretraining) と呼ばれる (Gururangan et al., 2020)。医療・法律・社内文書などでも、テキストを差し替えれば同じ手順で使える。
- この実行（約 500 万トークン、1,250 ステップ）では、学習に使っていない作者の旧仮名の文で、隠した語の損失が 3.356 → 2.928、1 位の正解率が 0.425 → 0.471 に改善した。fill-mask の個々の例はまちまちで、「彼は何も言は[MASK]かつた。」の「な」は 1 位 (0.99) になったが、「雨が降つて[MASK]る。」の「ゐ」は 0.30 → 0.23 と下がった。個々の例よりも、多くの位置で数えた損失や正解率で判断する。
- モデルは CC-100 で 1M ステップ学習したあと Wikipedia（2023 年 1 月の版）でさらに 1M ステップ学習されている（モデルカード）ので、Wikipedia は元の分野にあたる。Wikipedia では損失が 1.711 → 1.745、正解率が 0.667 → 0.642 と少し悪くなり、「私は毎朝[MASK]を飲みます。」の 1 位は「コーヒー」から「酒」に変わった。学習を長く続けたり学習率を大きくしたりすると元の分野の性能が大きく落ちることがあり、破滅的忘却と呼ばれる（機械翻訳の章の「学習済みの翻訳モデルを自分の分野のデータでファインチューニングする」の注意書きも参照）。元の分野のテキストを混ぜる、学習率やステップ数を抑える、といった対策がある（ここでは試していない）。
- 評価の「1 位の正解率」は、この章の「マスクされた単語を BERT に予測させる (fill-mask)」の 1 位の予測が当たる割合を、隠した全トークンで数えたもの。言語モデルの確率と内部を調べるの章の「マスク言語モデルで文の自然さを測る（疑似対数尤度）」は 1 トークンずつ隠して文長の回数だけ推論するが、ここでは語の 15% を同時に隠して 1 回で済ませている。隠す位置を固定しているので前後の比較はできるが、疑似対数尤度の値とは比べられない。
- tohoku-nlp/bert-base-japanese-v3 は、MeCab の 1 語のサブワードをまとめて隠す whole word masking で事前学習されている（モデルカード）。評価と追加学習でもそれにそろえた。`DataCollatorForLanguageModeling(whole_word_mask=True)` は fast トークナイザーの文字位置を使うので、MeCab を使うこのモデルのトークナイザーでは使えず、「##」の付くサブワードを手がかりに自分で書いた。
- 下流の分類（小説か随筆・評論か）では、正解率の平均が 0.824 → 0.828 で、差 +0.004 の 95% 信頼区間（作品 577 件を単位にしたブートストラップ）は [-0.004, +0.012] と 0 をまたいだ。シードによっても 0.01〜0.02 程度動く（元の BERT は 0.812〜0.830）。マスク言語モデルの損失が大きく改善しても、分類の精度が上がるとは限らないので、下流のタスクで、シードを変えた複数回の実行で確かめる。
- 保存した `AutoModelForMaskedLM` には、分類で使う pooler の層が含まれない。そのまま `AutoModelForSequenceClassification` で読むと pooler がランダムに初期化され、元のモデルとの比較が不公平になるので、上のコードでは pooler を元のモデルから取り、エンコーダーだけを差し替えた。
- データは青空文庫（globis-university/aozorabunko-clean、CC BY 4.0。著作権の切れた作品だけを収録）と日本語 Wikipedia（wikimedia/wikipedia、CC BY-SA 3.0 / GFDL）、モデルは東北大学の BERT (Apache 2.0)。数値は A5000 で float16 の経路（`BF16 = False`、T4 と同じ dtype）で実行したもので、GPU メモリの最大使用量は 4.9 GB だった。追加学習したモデルは `bert-kyukana` に保存される。同じランタイムなら、fill-mask の `model=` や疑似対数尤度の `name` に `"bert-kyukana"` を渡せば使える。別のノートブックで使うなら、この章の「モデルを保存・読み込みする、Hub にアップロードする」のように Google ドライブに保存するか Hub に上げる。

## 統制した自前のコーパスで小さな GPT を事前学習する (Trainer) {gpt_from_scratch gpu}

言語獲得や学習データの影響を調べる研究のために、自分で決めたコーパス（日本語版 Wikipedia の決まった約 2 万 2 千記事）だけで 1,400 万パラメータほどの GPT をゼロから事前学習し、学習途中のチェックポイントで、分野の違うテキストのパープレキシティと文法の最小対の正解率がどう変わるかを調べる。

```python
!pip install -q datasets fugashi unidic-lite
```

```python
import json, random
import numpy as np
import torch, transformers, datasets, tokenizers
from datasets import load_dataset

random.seed(0); np.random.seed(0); torch.manual_seed(0)
datasets.disable_progress_bars(); transformers.utils.logging.disable_progress_bar()   # 読み込みや加工の進捗バーを出さない
print("transformers", transformers.__version__, "/ datasets", datasets.__version__, "/ tokenizers", tokenizers.__version__, "/ torch", torch.__version__)

# 学習コーパス: 日本語版 Wikipedia (wikimedia/wikipedia の 20231101.ja。本文は CC BY-SA 3.0 と GFDL)。
# revision でデータセットの版 (コミット) を固定すると、あとで同じ記事を取り出し直せる
WIKI_REV = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
wiki = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True, revision=WIKI_REV)
# 25,000 記事を取り出して短い記事を除き、各記事の先頭 2,000 文字だけを使う (Colab で 1〜2 分)
arts = [a for a in wiki.shuffle(seed=0, buffer_size=10000).take(25000) if len(a["text"]) >= 300]
texts = [a["text"][:2000] for a in arts]
train_texts, val_texts = texts[:-500], texts[-500:]   # 最後の 500 記事は学習に使わず、検証用に取っておく
# 何を学習させたかを記録しておく (研究では、この一覧とコードがあればコーパスを再現できる)
json.dump({"dataset": "wikimedia/wikipedia", "config": "20231101.ja", "revision": WIKI_REV, "max_chars": 2000,
           "train_ids": [a["id"] for a in arts[:-500]], "val_ids": [a["id"] for a in arts[-500:]]},
          open("corpus_manifest.json", "w"))
print(len(train_texts), "記事 (学習) /", len(val_texts), "記事 (検証)", f"学習データ {sum(map(len, train_texts)) / 1e6:.1f}M 文字")

# 分野の違う評価用テキスト: 青空文庫の新字新仮名の作品 (globis-university/aozorabunko-clean、CC BY 4.0。
# 収録作品は著作権の切れたもの)。学習には一切使わない
aozora = load_dataset("globis-university/aozorabunko-clean", split="train", streaming=True,
                      revision="42a9c9c0f1d67e6a5554d9bea4201973dc9b049c")
works = [w for w in aozora.shuffle(seed=0, buffer_size=2000).take(1000) if w["meta"]["文字遣い種別"] == "新字新仮名"]
ood_texts = ["\n".join(p.strip() for p in w["text"].split("\n") if len(p.strip()) >= 20)[:2000] for w in works]
ood_texts = [t for t in ood_texts if len(t) >= 300][:300]
print(len(ood_texts), "作品 (青空文庫、評価用)")
```

```python
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

# トークナイザーも同じコーパスから作る (既存のトークナイザーを使うと、その学習データの影響が入る)。
# GPT-2 と同じバイト単位の BPE で、未知語 ([UNK]) が出ない
bpe = Tokenizer(models.BPE())
bpe.normalizer = normalizers.NFKC()
bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
bpe.decoder = decoders.ByteLevel()
trainer_tok = trainers.BpeTrainer(vocab_size=8000, special_tokens=["<|endoftext|>", "<|pad|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
bpe.train_from_iterator(train_texts[:8000], trainer_tok)        # 語彙は学習用の記事の一部 8,000 記事から作る (1 分以内)
# パディング用のトークンは <|endoftext|> と別にする (注意書きを参照)
tokenizer = PreTrainedTokenizerFast(tokenizer_object=bpe, bos_token="<|endoftext|>", eos_token="<|endoftext|>", pad_token="<|pad|>")
tokenizer.save_pretrained("gpt-ja-tokenizer")
ids = tokenizer("吾輩は猫である。自然言語処理は人工知能の一分野である。")["input_ids"]
print(len(ids), "トークン:", [tokenizer.decode([i]) for i in ids])
```

```python
import itertools
from datasets import Dataset

BLOCK = 256
def tokenize(batch):
    # 記事の終わりに <|endoftext|> を付ける
    return {"input_ids": [x + [tokenizer.eos_token_id] for x in tokenizer(batch["text"])["input_ids"]]}

def group(batch):
    # 記事をつなげて 256 トークンずつのブロックに切る (端数は捨てる。Hugging Face の run_clm.py と同じ方法)
    ids = list(itertools.chain.from_iterable(batch["input_ids"]))
    n = len(ids) // BLOCK * BLOCK
    return {"input_ids": [ids[i:i + BLOCK] for i in range(0, n, BLOCK)]}

def to_blocks(texts):
    ds = Dataset.from_dict({"text": texts}).map(tokenize, batched=True, remove_columns=["text"])
    n_tok = sum(map(len, ds["input_ids"]))
    return ds.map(group, batched=True, batch_size=1000), n_tok / sum(map(len, texts))   # (ブロック, 1 文字あたりのトークン数)

train_ds, _ = to_blocks(train_texts)                 # Colab で 1 分ほど (2 コアに制限して 38 秒)
eval_ds, tok_per_char = {}, {}
eval_ds["wiki"], tok_per_char["wiki"] = to_blocks(val_texts)
eval_ds["aozora"], tok_per_char["aozora"] = to_blocks(ood_texts)
print(f"学習 {len(train_ds)} ブロック ({len(train_ds) * BLOCK / 1e6:.1f}M トークン)  評価", {k: len(v) for k, v in eval_ds.items()}, "ブロック")
print("1 トークンあたりの文字数", {k: round(1 / v, 2) for k, v in tok_per_char.items()})

# 比較の基準: 文脈を見ずに出現頻度だけで予測するモデル (ユニグラム) の損失
counts = np.bincount(np.array(train_ds["input_ids"]).ravel(), minlength=len(tokenizer)) + 1.0
unigram_logp = np.log(counts / counts.sum())
unigram_loss = {k: float(-unigram_logp[np.array(v["input_ids"])].mean()) for k, v in eval_ds.items()}
print("ユニグラムの損失", {k: round(v, 3) for k, v in unigram_loss.items()})
```

```python
import math, time
from transformers import (DataCollatorForLanguageModeling, GPT2Config, GPT2LMHeadModel,
                          Trainer, TrainerCallback, TrainingArguments)

BF16 = torch.cuda.get_device_capability()[0] >= 8    # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
config = GPT2Config(vocab_size=len(tokenizer), n_positions=BLOCK, n_embd=384, n_layer=6, n_head=6,
                    resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,            # データを 2 周ほどしか見ないのでドロップアウトなし
                    bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id)
torch.manual_seed(0)
model = GPT2LMHeadModel(config)                       # from_pretrained ではなく、設定から重みがランダムなモデルを作る
print(f"{model.num_parameters() / 1e6:.1f}M パラメータ")

# 学習の途中経過を調べるステップ (言語獲得の研究では、学習の初期に細かく取る)。このステップで評価し、チェックポイントを保存する
SAVE_AT = [50, 100, 200, 500, 1000, 2000, 3500, 5000]
class EvalAndSaveAt(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in SAVE_AT:
            control.should_evaluate = control.should_save = True
        return control

args = TrainingArguments(
    output_dir="gpt-ckpt", max_steps=5000, per_device_train_batch_size=32, per_device_eval_batch_size=64,
    learning_rate=1e-3, warmup_steps=200, lr_scheduler_type="cosine_with_min_lr", lr_scheduler_kwargs={"min_lr_rate": 0.1},
    weight_decay=0.1, adam_beta2=0.95, max_grad_norm=1.0, bf16=BF16, fp16=not BF16,
    logging_steps=250, eval_strategy="no", save_strategy="no", save_only_model=True,   # 評価と保存は上のコールバックで決める
    report_to="none", seed=0, data_seed=0)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,   # 評価用を辞書で渡すと eval_wiki_loss のように別々に記録される
                  data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),       # labels = input_ids (1 つずらすのはモデルの中で行う)
                  callbacks=[EvalAndSaveAt()])

log = {0: trainer.evaluate()}                        # 学習前 (重みがランダム) の損失
# 健全性の確認: 学習前のモデルはほぼ一様な予測をするので、損失は log (語彙の大きさ) に近いはず
print(f"学習前の損失 {log[0]['eval_wiki_loss']:.3f}  log {len(tokenizer)} = {math.log(len(tokenizer)):.3f}")
trainer.save_model("gpt-ckpt/checkpoint-0")
t0 = time.time()
trainer.train()                                      # T4 で 15〜20 分 (A5000 で 5 分弱から見積もり。最初に出る loss_type=None の警告は無害)
seen = trainer.state.global_step * args.per_device_train_batch_size * BLOCK
print(f"学習 {time.time() - t0:.0f} 秒、{trainer.state.epoch:.1f} エポック (延べ {seen / 1e6:.1f}M トークン、パラメータ数の {seen / model.num_parameters():.1f} 倍)、"
      f"GPU メモリの最大使用量 {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
for h in trainer.state.log_history:               # 評価の結果は評価用のデータごとに別の行で記録されるので、ステップでまとめる
    if any(k.startswith("eval_") for k in h):
        log.setdefault(h["step"], {}).update(h)
train_loss = {h["step"]: h["loss"] for h in trainer.state.log_history if "loss" in h}   # 直前 250 ステップの平均
for s, h in log.items():
    tr = f"{train_loss[s]:.3f}" if s in train_loss else "  -  "
    print(f"step {s:5}  学習 {tr}  検証 (Wikipedia) {h['eval_wiki_loss']:.3f}  青空文庫 {h['eval_aozora_loss']:.3f}")
```

```python
import fugashi

# 文法の最小対を、学習に使っていない検証用の記事の文から自動で作る (各 300 組)
#   助詞: 格助詞「が・を・に」の 1 つを別のものに替える  語順: 隣り合う 2 つの形態素を入れ替える
tagger = fugashi.Tagger()
rng = random.Random(0)
# 「概要」などの見出しが文に混ざらないよう、段落 (行) に分け、「。」で終わる段落だけから文を取り出す
sents = [s.strip() + "。" for t in val_texts for para in t.split("\n") if para.strip().endswith("。")
         for s in para.strip().split("。") if 15 <= len(s.strip()) <= 60]
rng.shuffle(sents)
pairs = {"助詞": [], "語順": []}
for s in sents:
    w = [(m.surface, m.feature.pos1, m.feature.pos2) for m in tagger(s)]
    # pos2 で格助詞に絞る (「雨が降ったが、」の 2 つ目の「が」のような接続助詞は除く)
    cands = [i for i, (x, _, p2) in enumerate(w) if p2 == "格助詞" and x in ("が", "を", "に")]
    if cands and len(pairs["助詞"]) < 300:
        i = rng.choice(cands); new = rng.choice([x for x in ("が", "を", "に") if x != w[i][0]])
        pairs["助詞"].append((s, "".join(x if j != i else new for j, (x, *_) in enumerate(w))))
    cands = [i for i in range(len(w) - 2) if w[i][1] != "補助記号" and w[i + 1][1] != "補助記号" and w[i][0] != w[i + 1][0]]
    if cands and len(pairs["語順"]) < 300:
        i = rng.choice(cands); v = [x for x, *_ in w]; v[i], v[i + 1] = v[i + 1], v[i]
        pairs["語順"].append((s, "".join(v)))
for k, v in pairs.items():
    print(k, len(v), "組  例:", v[0][0], "/", v[0][1])

@torch.no_grad()
def logprob(model, texts, bs=100):
    """文頭に <|endoftext|> を置いたときの、文全体の対数尤度 (各トークンの対数確率の和)"""
    out = []
    for i in range(0, len(texts), bs):
        enc = tokenizer([tokenizer.bos_token + t for t in texts[i:i + bs]], return_tensors="pt", padding=True,
                        return_token_type_ids=False).to(model.device)
        lp = model(**enc).logits[:, :-1].float().log_softmax(-1)
        tgt = enc["input_ids"][:, 1:]
        out += (lp.gather(-1, tgt[..., None])[..., 0] * enc["attention_mask"][:, 1:]).sum(1).tolist()
    return np.array(out)

def score(lp_good, lp_bad):
    """組ごとの得点: 正しい文の対数尤度が高ければ 1、同点なら 0.5、低ければ 0"""
    tie = np.isclose(lp_good, lp_bad, rtol=0, atol=1e-6)
    return np.where(tie, 0.5, (lp_good > lp_bad).astype(float))

# 各チェックポイントで、正しい文の対数尤度が崩した文より高い組の割合を数える (Colab で 1 分ほど)
correct = {}
for s in [0] + SAVE_AT:
    ckpt = GPT2LMHeadModel.from_pretrained(f"gpt-ckpt/checkpoint-{s}").to("cuda").eval()
    correct[s] = {k: score(logprob(ckpt, [g for g, _ in v]), logprob(ckpt, [b for _, b in v])) for k, v in pairs.items()}

# 対照: 文脈を見ないユニグラムでも、トークン数や語の頻度の違いだけで正しい文を選べることがある。
# 語順を入れ替えてもトークンの集合が変わらない組は、ユニグラムでは同点になる (0.5 と数える)
def unigram_lp(texts):
    return np.array([unigram_logp[x].sum() for x in tokenizer(texts)["input_ids"]])
for k, v in pairs.items():
    g, b = unigram_lp([g for g, _ in v]), unigram_lp([b for _, b in v])
    print(f"ユニグラムでの最小対 ({k}) の正解率 {score(g, b).mean():.3f}  (同点 {np.isclose(g, b, rtol=0, atol=1e-6).sum()} 組)")
```

```python
import pandas as pd
import matplotlib.pyplot as plt

# 損失 (1 トークンあたり、自然対数) をトークナイザーによらない 1 文字あたりのビット数にも直す
rows = []
for s in [0] + SAVE_AT:
    r = {"step": s}
    for k in ["wiki", "aozora"]:
        r[f"PPL ({k})"] = math.exp(log[s][f"eval_{k}_loss"])
        r[f"bit/文字 ({k})"] = log[s][f"eval_{k}_loss"] * tok_per_char[k] / math.log(2)
    r |= {f"最小対 ({k})": c.mean() for k, c in correct[s].items()}
    rows.append(r)
df = pd.DataFrame(rows).set_index("step")
print(df.round(3).to_string())
print("ユニグラム: bit/文字", {k: round(unigram_loss[k] * tok_per_char[k] / math.log(2), 2) for k in unigram_loss})

# 最小対の正解率の 95% 信頼区間 (組を単位にしたブートストラップ)。評価は各 300 組なので幅はおよそ ±0.04
brng = np.random.default_rng(0)
def ci(c, n=2000):
    boot = [c[brng.integers(0, len(c), len(c))].mean() for _ in range(n)]
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)
for s in [200, 1000, SAVE_AT[-1]]:
    print(f"step {s:5}", "  ".join(f"{k} {c.mean():.3f} [{lo:.3f}, {hi:.3f}]" for k, c in correct[s].items() for lo, hi in [ci(c)]))

x = np.array(df.index) + 10                           # 対数軸に 0 を置けないので 10 ずらす
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for k, m in [("wiki", "o-"), ("aozora", "s-")]:
    axes[0].plot(x, df[f"bit/文字 ({k})"], m, ms=4, label=f"GPT ({k})")
    axes[0].axhline(unigram_loss[k] * tok_per_char[k] / math.log(2), ls=":", color="C0" if k == "wiki" else "C1", label=f"unigram ({k})")
axes[0].set_xscale("log"); axes[0].set_xlabel("step + 10"); axes[0].set_ylabel("bits / character"); axes[0].legend()
axes[1].plot(x, df["最小対 (助詞)"], "o-", ms=4, label="particle substitution")
axes[1].plot(x, df["最小対 (語順)"], "s-", ms=4, label="adjacent word swap")
axes[1].axhline(0.5, color="gray", ls="--")
axes[1].set_xscale("log"); axes[1].set_xlabel("step + 10"); axes[1].set_ylabel("minimal-pair accuracy"); axes[1].legend()
plt.tight_layout(); plt.show()
```

```python
from transformers import pipeline, set_seed

# チェックポイントごとの生成文 (同じ乱数の種で、temperature 0.8、top_p 0.95 のサンプリング)
for s in [0, 200, 1000, 5000]:
    gen = pipeline("text-generation", model=f"gpt-ckpt/checkpoint-{s}", tokenizer=tokenizer, device=0)
    print(f"--- step {s}")
    for prompt in ["日本の首都は", "私は"]:
        set_seed(0)
        out = gen(tokenizer.bos_token + prompt, max_new_tokens=40, do_sample=True, temperature=0.8, top_p=0.95, top_k=0)
        print("  ", out[0]["generated_text"].replace(tokenizer.bos_token, "").replace("\n", " "))
```

- 公開されている事前学習済みモデルは学習データが膨大で、評価に使う文が学習データに入っていたかどうかも確かめにくい。言語獲得や学習データの影響を調べる研究では、ここのようにコーパスを自分で決めてゼロから学習する。データセットの `revision`、記事の一覧 (`corpus_manifest.json`)、トークナイザー、乱数の種、ライブラリの版を残しておけば、同じ条件で学習し直せる。学習データを操作する実験（ある構文を含む文を除く、分野の比率を変えるなど）は、`train_texts` を差し替えて同じコードで比べる（ここでは試していない）。学習途中のチェックポイントを公開して同じ目的に使えるようにしたモデル群に Pythia (Biderman et al., 2023) がある。
- このコーパスの性質も記録しておく。各記事の先頭 2,000 文字だけを使うので、記事の冒頭（「〜は、…である。」という定義の文や概要）に偏る。ストリーミングの `shuffle` は 10,000 件のバッファの中で混ぜるだけの近似で、Wikipedia 全体から一様に選んだ標本ではない。どちらも記事の一覧があれば後から確かめられるが、論文などでは「無作為に選んだ記事」とは書かない。
- 学習ループは Trainer に任せ、`DataCollatorForLanguageModeling(mlm=False)` が `labels` に `input_ids` の写しを入れる（1 つずらすのはモデルの中）。このコレーターはパディングのトークンと同じ ID のラベルを -100（損失に入れない）にするので、よく見る `pad_token = eos_token` にすると、記事の区切りの `<|endoftext|>` もすべて学習されなくなる。ここでは `<|pad|>` を別に作った。評価と保存のステップはコールバックで指定し、`save_only_model=True` でチェックポイントをモデルの重みだけ（1 つ 54MB、9 つで約 480MB）にした。オプティマイザーの状態がないので、これらのチェックポイントから学習を再開することはできない。
- 5,000 ステップ（2.2 エポック）で、学習に使っていない Wikipedia の記事の PPL は 39.7、学習に一切使っていない青空文庫の作品は 153.9 だった。語彙が Wikipedia から作られているので青空文庫は 1 トークンあたりの文字数が少なく（1.35 文字。Wikipedia は 1.55 文字。語彙 8,000 では「吾輩」「人工知能」も 1 文字ずつに分かれる）、トークンあたりの PPL では比べられない。1 文字あたりのビット数にそろえると Wikipedia 3.42 bit、青空文庫 5.40 bit で、ユニグラム（6.99 bit、8.22 bit）からの下がり方は Wikipedia の方が大きい。学習データの分野の違いがそのまま評価に出る。
- 学習前の損失 9.008 は、語彙 8,000 の一様分布の log 8000 = 8.987 にほぼ等しい。大きく離れていれば、ラベルのずらし方や初期化を疑う。学習の損失（直前 250 ステップの平均）は 2,000 ステップでは検証より 0.033 高かったが、3,500 ステップで 0.059、5,000 ステップで 0.182 低くなった。同じデータを 2 周以上したことによる過学習の始まりで、これ以上ステップを増やすならデータも増やす。学習したトークンは延べ 41.0M（重複しないのは 18.8M）で、パラメータ数 13.8M の約 3 倍しかない。Chinchilla (Hoffmann et al., 2022) は 70B パラメータを 1.4 兆トークン（約 20 倍）で学習しており、それに比べるとデータがかなり少ない。データ量をそろえて比べる研究では、この比も条件として書いておく。エンコーダー・デコーダーをゼロから学習する例は、機械翻訳の章の「Transformer の翻訳モデルをゼロから学習する」にある。
- 文法の最小対の正解率は、最後のチェックポイントで格助詞の置き換え 0.917（95% 信頼区間 [0.883, 0.947]）、隣り合う形態素の入れ替え 0.970 ([0.950, 0.987]) だった。ただし重みがランダムな学習前のモデルでも 0.687 と 0.633、文脈を見ないユニグラムでも 0.640 と 0.635 になる（語順の 300 組のうち 87 組はトークンの集合が同じでユニグラムでは区別できず、0.5 と数えた）。トークン数や語の頻度の違いだけで選べる組があるので、0.5 ではなく、このような対照と比べる。この 1 回の実行（種は 1 つ、各 300 組）では、200 ステップで語順 0.873 ([0.837, 0.910]) に対して助詞 0.760 ([0.713, 0.807]) と、助詞の方が遅れて上がった。2 つの課題は対照の水準も違うので、獲得の順序として主張するには、種を変えた複数の実行と人手で確かめた最小対が要る。
- 最小対は検証用の記事の文から機械的に作ったもので、人手で確かめていない。「が」と「を」を入れ替えても文法的なままの組や、1 組の差が助詞以外の要因で決まる組も混ざりうる。系統的に調べるなら、言語モデルの確率と内部を調べるの章の「言語モデルの文法知識を最小対で調べる (BLiMP / JBLiMP)」のような人手で作った最小対を使う。生成文は、学習前はでたらめなトークンの列、200 ステップと 1,000 ステップでは記号や断片が並ぶだけで、5,000 ステップで「日本の首都は、東京都千代田区にある。」のあとに「概要」のような Wikipedia の見出しが続く文章になった（各チェックポイントで 1 回ずつサンプリングしただけなので、傾向として見る程度にとどめる）。
- 13.8M パラメータで、GPU メモリの最大使用量は 2.6 GB。数値と生成文は、検証のマシン（RTX A5000、T4 と同じ float16 の経路）での 1 回の結果で、学習のセルは 297 秒（うち `trainer.train()` は 285 秒）だった。T4 はこれより遅い。GPU が違うと計算の細部が変わり、生成文などは別のものになりうる。版をそろえるなら `!pip install -q transformers==5.22.1 datasets==4.8.5` のように固定する（検証は transformers 5.22.1、datasets 4.8.5、tokenizers 0.23.3、torch 2.14.1）。
- データは日本語版 Wikipedia（wikimedia/wikipedia の 20231101.ja、データセットカードによれば CC BY-SA 3.0 と GFDL）と青空文庫（globis-university/aozorabunko-clean、CC BY 4.0）。ライブラリは transformers、datasets、tokenizers が Apache 2.0、fugashi が MIT と BSD 3 条項、unidic-lite がパッケージは MIT で、同梱の UniDic 2.1.2 の辞書は GPL、LGPL、BSD のいずれかを選べる（パッケージに同梱の COPYING による）。トークナイザーの学習は、サブワードとトークナイザの章の「Hugging Face tokenizers で BPE / WordPiece を学習する」を参照。仕組みを自分で実装して確かめる版は、付録「仕組みを理解するための実装」の「小さな GPT を日本語のテキストでゼロから事前学習する（学習ループを自分で書く）」にある。
