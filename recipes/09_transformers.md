# Transformers の基本操作

Hugging Face Transformers と Datasets で、事前学習済みモデルを動かすときの定番の操作。

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

- マスクトークンはモデルによって違う (`[MASK]`、`<mask>` など)。`fill.tokenizer.mask_token` で確認できる。
- 予測できるのは 1 トークンだけ。複数トークンにまたがる語 (サブワードに分割される語) は直接は予測できない。

## マスク言語モデルで文の自然さを測る (疑似対数尤度) {mlm_scoring}

BERT のような双方向モデルでも文の自然さを数値化したい。1 トークンずつマスクして、元のトークンの確率を合計する。

```python
!pip install -q fugashi unidic-lite
```

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

name = "tohoku-nlp/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForMaskedLM.from_pretrained(name).eval()

@torch.no_grad()
def pll(text):
    """疑似対数尤度 (pseudo-log-likelihood) と、トークンごとの内訳"""
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    n = len(ids) - 2                                        # [CLS] と [SEP] を除く
    batch = ids.repeat(n, 1)
    pos = torch.arange(1, n + 1)
    batch[torch.arange(n), pos] = tokenizer.mask_token_id   # i 番目の文は i 番目のトークンだけマスク
    logp = model(batch).logits[torch.arange(n), pos].log_softmax(-1)
    scores = logp[torch.arange(n), ids[pos]]
    return scores.sum().item(), [(t, round(v, 1)) for t, v in zip(tokenizer.convert_ids_to_tokens(ids[pos]), scores.tolist())]

for text in ["太郎は図書館で本を読んだ。", "太郎は図書館が本を読んだ。", "太郎は図書館で本を飲んだ。"]:
    total, detail = pll(text)
    print(f"{total:7.2f}  {text}")
    print("        ", detail)
```

- 自己回帰モデルの対数尤度と違い、左右両方の文脈を使って各トークンを評価する。誤りのある位置のスコアが局所的に下がるので、誤り検出に使いやすい。
- 文長分の推論が必要 (文長 n のバッチ 1 回)。長い文や大量の文には重い。
- 確率として正規化されていない「疑似」尤度なので、あくまで相対比較に使う (Salazar et al., 2020)。

## BERT から文脈に応じた単語ベクトルを取り出す {contextual_word_vectors}

同じ単語でも文脈によって意味が違う (bank = 銀行 / 土手)。文脈を反映した単語ベクトルを取り出して比べたい。

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

前提文から仮説文が「言える (含意)」「矛盾する」「どちらでもない (中立)」のどれかを判定したい。

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"   # 多言語 NLI モデル
tokenizer = AutoTokenizer.from_pretrained(name)
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
- 語彙の重なりが大きいと含意と判定しやすい、といった表面的な手がかりへの依存 (アーティファクト) が知られている。過信は禁物。
- 日本語のデータセットには JNLI (JGLUE)、JSICK、JaNLI などがある。

## アテンションの重みをヒートマップで可視化する {attention_viz}

Transformer の自己注意が、どのトークンからどのトークンに向いているのかを図で見たい。

```python
import torch
from transformers import AutoModel, AutoTokenizer

name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name, attn_implementation="eager").eval()   # 重みを取り出すには eager 実装が必要

text = "The animal didn't cross the street because it was too tired."
enc = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    attentions = model(**enc, output_attentions=True).attentions
tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
print(len(attentions), "層", attentions[0].shape, "= (バッチ, ヘッド, トークン, トークン)")
```

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (layer, head) in zip(axes, [(0, 0), (8, 9)]):
    ax.imshow(attentions[layer][0, head], cmap="Blues")
    ax.set_xticks(range(len(tokens))); ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticks(range(len(tokens))); ax.set_yticklabels(tokens)
    ax.set_title(f"layer {layer + 1}, head {head + 1}")
plt.tight_layout(); plt.show()
```

```python
# 「it」がどのトークンに注意を向けているか、全層・全ヘッドの平均と、最も強く向いているヘッド
it = tokens.index("it")
stacked = torch.stack(attentions)[:, 0]                 # (層, ヘッド, トークン, トークン)
avg = stacked[:, :, it].mean(dim=(0, 1))
print("平均:", [(tokens[i], round(avg[i].item(), 2)) for i in avg.argsort(descending=True)[:5]])
animal = tokens.index("animal")
layer, head = divmod(stacked[:, :, it, animal].argmax().item(), stacked.shape[1])
print(f"it → animal が最大のヘッド: layer {layer + 1}, head {head + 1} ({stacked[layer, head, it, animal]:.2f})")
```

- 行が「注意を向ける側」、列が「向けられる側」。各行の和は 1。
- 下位層は隣接トークンや `[SEP]` に向くヘッドが多く、中〜上位層に構文関係や共参照に対応するヘッドが現れる。
- アテンションの重みは、そのまま「モデルの判断根拠」ではない。説明として使うときは注意が必要 (Jain & Wallace, 2019)。

## 次に来るトークンの確率分布を見る {next_token_probs}

言語モデルが「次に何が来る」と予測しているのか、上位候補と確率を直接見たい。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
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

## 言語モデルで文の自然さ (パープレキシティ) を測る {lm_perplexity}

ある文がどれくらい「自然」かを、言語モデルの確率で数値化したい。文法性判定や、生成文の流暢さの評価に使える。

```python
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()

def score(text):
    """(対数尤度の合計, トークン数, パープレキシティ) を返す"""
    ids = tokenizer(tokenizer.bos_token + text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        loss = model(ids, labels=ids).loss.item()   # 1 トークンあたりの平均負対数尤度
    n = ids.shape[1] - 1
    return -loss * n, n, math.exp(loss)

for text in ["吾輩は猫である。名前はまだ無い。", "吾輩は猫がある。名前をまだ無い。", "無いまだは名前。である猫は吾輩。"]:
    ll, n, ppl = score(text)
    print(f"対数尤度 {ll:8.2f}  トークン数 {n:2}  PPL {ppl:8.1f}  {text}")
```

```python
# ミニマルペアによる文法性判定: 正しい文のほうに高い確率を与えられるか
pairs = [("太郎が本を読んだ。", "太郎が本が読んだ。"), ("私は昨日学校に行った。", "私は昨日学校に行く。"),
         ("3 冊の本を買った。", "3 匹の本を買った。"), ("雨が降ったので、試合は中止になった。", "雨が降ったのに、試合は中止になった。")]
for good, bad in pairs:
    print("○" if score(good)[0] > score(bad)[0] else "×", good, "/", bad)
```

- パープレキシティ = exp(1 トークンあたりの平均負対数尤度)。低いほどモデルにとって「予想どおり」の文。
- 長さの違う文を比べるときはパープレキシティ、同じ長さのミニマルペアなら対数尤度の合計で比べる。
- 先頭に BOS トークンを付けないと、最初のトークンの確率が計算されない。
- 値はモデルとトークナイザに依存するので、異なるモデル間でパープレキシティを比較することはできない。

## トークンごとのサプライザルを色で可視化する {surprisal}

文のどの部分が言語モデルにとって「意外」だったのかを見たい。誤りの検出や、読みにくさの分析に使える。

```python
import math
import torch
from IPython.display import HTML
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()

def surprisals(text):
    ids = tokenizer(tokenizer.bos_token + text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logp = torch.log_softmax(model(ids).logits[0, :-1], dim=-1)
    s = -logp[torch.arange(ids.shape[1] - 1), ids[0, 1:]] / math.log(2)   # 各トークンの -log2 P(トークン | 文脈)
    return [(tokenizer.decode(i), v) for i, v in zip(ids[0, 1:].tolist(), s.tolist())]

def show(text):
    spans = []
    for tok, s in surprisals(text):
        alpha = min(s / 20, 1.0)
        spans.append(f"<span title='{s:.1f} bit' style='background: rgba(255, 80, 80, {alpha:.2f}); padding: 2px'>{tok}</span>")
    return "<p style='font-size: 1.2em; line-height: 2'>" + "".join(spans) + "</p>"

texts = ["私は毎朝コーヒーを飲んでから会社に行きます。", "私は毎朝コーヒーを食べてから会社を行きます。", "私は毎朝コーヒーを飲んでから月に行きます。"]
for t in texts:
    print(" ".join(f"{tok}({s:.0f})" for tok, s in surprisals(t)))
HTML("".join(show(t) for t in texts))
```

- サプライザル (−log₂ 確率) は、心理言語学では読み時間と相関することが知られている指標。
- 文法誤り (「コーヒーを食べて」「会社を行き」) や意味的に意外な語 (「月に」) で値が跳ね上がる。
- 文頭のトークンは文脈がないので、いつも値が大きい。

## テキスト生成のデコード方法 (貪欲法・ビームサーチ・サンプリング) を比べる {generation_decoding}

言語モデルに文章の続きを生成させたい。`temperature` や `top_p` などの設定で出力がどう変わるかを確かめたい。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"   # 0.5B パラメータの小さな日本語 LLM (CPU でも動く)
tokenizer = AutoTokenizer.from_pretrained(name)
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
- `top_p` (nucleus サンプリング) は確率の累積が p になるまでの上位トークンだけから選ぶ。`top_k` は上位 k 個から選ぶ。
- 翻訳や要約など正解が定まるタスクはビームサーチ、創作や対話はサンプリング (temperature 0.7 前後 + top_p 0.9 前後) が定番。
- `max_new_tokens` は生成する長さの上限。指定しないと短い既定値で打ち切られる。
- モデルに付属の `generation_config.json` に既定値 (temperature や top_p など) が入っていることがある。指定しなかったパラメータにはそれが使われるので、`print(model.generation_config)` で確認する。

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

- `torch.inference_mode()` (または `no_grad()`) で勾配計算を止める。忘れるとメモリを浪費し、遅くなる。
- 長さの近い文を同じバッチにまとめると、パディングが減って速くなる。結果は元の順番に戻す。
- この 96 件では、GPU で 1 件ずつ 0.2〜0.3 秒 → バッチ + 長さでソートで 0.04 秒 (6〜8 倍)。件数が多いほど、またモデルが大きいほど差は広がる。さらに `model.half()` (半精度) で速度とメモリが改善する。
- CPU では、長さを揃えずにバッチにすると、長い文に合わせたパディングの計算が重く、1 件ずつより遅くなることがある。
- GPU の処理時間を測るときは、最初の 1 回 (初期化やカーネルの準備を含む) を計測から外す。
- `pipeline(..., device=0, batch_size=32)` でも同様のバッチ処理ができる。

## モデルのパラメータ数とメモリ使用量を調べる {model_size}

このモデルは Colab の GPU (T4 は 16GB) に載るのか、パラメータ数とメモリ消費の目安を知りたい。

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

- 推論時のメモリ ≒ パラメータ数 × 1 パラメータのバイト数 + α (活性化や KV キャッシュ)。学習時は勾配とオプティマイザの状態で、さらに 3〜4 倍必要。
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

- 既定のキャッシュ先は `~/.cache/huggingface/hub`。環境変数 `HF_HOME` で変更できる (import より前に設定する)。
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
