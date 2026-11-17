# 言語モデルの確率と内部を調べる

言語モデルが出す確率と、モデルの内部の表現から、モデルが何を知っているかを調べる。アテンションの可視化、文の自然さ（疑似対数尤度、パープレキシティ、サプライザル）、最小対による文法知識の評価、各層にどんな情報が入っているかを調べるプロービングと、どの層で予測が固まるかを見る logit lens、隠れ状態にベクトルを足して生成を変えるアクティベーション・ステアリング。

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

## マスク言語モデルで文の自然さを測る（疑似対数尤度） {mlm_scoring}

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
- 文長分の推論が必要（文長 n のバッチ 1 回）。長い文や大量の文には重い。
- 確率として正規化されていない「疑似」尤度なので、あくまで相対比較に使う (Salazar et al., 2020)。

## 言語モデルで文の自然さ（パープレキシティ）を測る {lm_perplexity}

ある文がどれくらい「自然」かを、言語モデルの確率で数値化したい。文法性判定や、生成文の流暢さの評価に使える。

```python
import math
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
import sentencepiece as spm
import torch
from IPython.display import HTML
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

- サプライザル（−log₂ 確率）は、心理言語学では読み時間と相関することが知られている指標。
- 文法誤り（「コーヒーを食べて」「会社を行き」）や意味的に意外な語（「月」）で値が跳ね上がる。
- 文頭のトークンは文脈がないので、いつも値が大きい。

## 言語モデルの文法知識を最小対で調べる (BLiMP / JBLiMP) {minimal_pairs}

言語モデルが文法をどれくらい身に付けているかを調べたい。1 か所だけ違う「文法的な文」と「非文法的な文」の組（最小対）を与え、文法的なほうに高い確率を付けるかを見る。

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

- 最小対による評価は、言語モデルに追加の学習をさせずに文法知識を測れる。英語の BLiMP、日本語の JBLiMP などがあり、現象ごと（一致、束縛、島の制約など）に得意・不得意が分かる。
- 文全体の確率で比べると、トークン数の少ない文が有利になる。最小対の 2 文でトークン数が違う場合、モデルが文法ではなく長さで選んでいるだけかもしれない。データセットによっては、この偏りだけで高い正解率が出る。
- 長さで正規化すれば解決する、とは限らない。この実験では、正規化すると今度は長い文が有利になり、偏りの向きが逆になっただけだった。トークン数が同じ組だけで評価する、長さの条件ごとに正解率を分けて報告する、といった確認が確実。
- 正解率が高くても、その現象を「理解している」とは限らない。語の頻度や共起のような表面的な手がかりで解ける組もある。
- 指示に従う LLM なら「どちらが自然か」を直接尋ねることもできるが、確率を比べる方法とは結果が一致しないことがある。
- rinna のトークナイザーの不具合: transformers 5 系の `T5Tokenizer` は、SentencePiece の語彙にない文字（大文字の英字など）を、`<unk>` ではなく文末記号 `</s>` の ID にしてしまう（5.0〜5.23 で確認、4.57 では正しい）。そうした文では確率の計算がおかしくなるので、上のコードでは spiece.model から語彙とスコアを読み、分割の方式を作り直している。JBLiMP の文にはこうした文字がほとんどないので、この実験の結果は修正の前後で変わらなかった。上流で直れば不要になる。

## BERT の各層に品詞や構文の情報がどれくらい入っているかを調べる（プロービング） {probing gpu}

BERT の内部の単語ベクトルに、品詞や係り受けの情報がどの層でどれくらい表れているかを、簡単な分類器（プローブ）で調べたい。

```python
!pip install -q fugashi unidic-lite conllu
```

```python
import random
import urllib.request
import numpy as np
import torch
from conllu import parse

random.seed(0); np.random.seed(0); torch.manual_seed(0)

# UD Japanese-GSD (CC BY-SA 4.0、https://github.com/UniversalDependencies/UD_Japanese-GSD)
base = "https://raw.githubusercontent.com/UniversalDependencies/UD_Japanese-GSD/master/ja_gsd-ud-"
def load(split, n=None):
    sents = parse(urllib.request.urlopen(base + split + ".conllu").read().decode("utf-8"))[:n]
    return [[t for t in s if isinstance(t["id"], int)] for s in sents]   # 複数語トークンなどの行を除く

train, test = load("train", 3000), load("test")   # 学習データは先頭 3000 文だけ使う
print(len(train), "文", sum(map(len, train)), "語 (学習) /", len(test), "文", sum(map(len, test)), "語 (評価)")
print([(t["form"], t["upos"], t["deprel"]) for t in test[0]])
```

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
name = "tohoku-nlp/bert-base-japanese-v3"   # Apache 2.0
tokenizer = AutoTokenizer.from_pretrained(name)

def encode(sent):
    """UD の語ごとにサブワードに分け、各語の先頭サブワードの位置を覚えておく"""
    ids, first = [tokenizer.cls_token_id], []
    for t in sent:
        first.append(len(ids))
        ids += tokenizer.encode(t["form"], add_special_tokens=False) or [tokenizer.unk_token_id]
    return ids + [tokenizer.sep_token_id], first

@torch.no_grad()
def features(model, sents, batch_size=32):
    """全層 (埋め込み層 + 12 層) の語ごとのベクトル: (13, 語数, 768)"""
    encoded = [encode(s) for s in sents]
    order = sorted(range(len(sents)), key=lambda i: len(encoded[i][0]))   # 長さの近い文をまとめてパディングを減らす
    out = [None] * len(sents)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        L = max(len(encoded[i][0]) for i in idx)
        ids = torch.zeros(len(idx), L, dtype=torch.long)
        mask = torch.zeros(len(idx), L, dtype=torch.long)
        for j, i in enumerate(idx):
            ids[j, :len(encoded[i][0])] = torch.tensor(encoded[i][0]); mask[j, :len(encoded[i][0])] = 1
        ids, mask = ids.to(device), mask.to(device)
        hs = torch.stack(model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states)
        for j, i in enumerate(idx):
            out[i] = hs[:, j, encoded[i][1]].half()             # 先頭サブワードのベクトル。メモリ節約のため float16 で保存 (1 モデルの学習データ分で約 1.4GB)
    return torch.cat(out, dim=1)

print("最長の文:", max(len(encode(s)[0]) for s in train + test), "サブワード")
bert = AutoModel.from_pretrained(name).to(device).eval()   # 推論だけなので float32 のまま (T4 でも 1GB 未満)
torch.manual_seed(0)
random_bert = AutoModel.from_config(AutoConfig.from_pretrained(name)).to(device).eval()   # 同じ構造で重みがランダムな対照モデル
# GPU で 1〜2 分 (CPU だと 10 分以上かかる)
feats = {m: (features(model, train), features(model, test)) for m, model in [("BERT", bert), ("random BERT", random_bert)]}
print(feats["BERT"][0].shape, f"GPU メモリ {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if device == "cuda" else "")
```

```python
from collections import Counter
import torch.nn.functional as F

y = {task: tuple([t[task] for s in d for t in s] for d in (train, test)) for task in ["upos", "deprel"]}

# 対照タスク (Hewitt & Liang, 2019): 単語の形 (type) ごとにランダムなラベルを 1 つ決め、どこに出てきても同じラベルにする。
# ラベルは正解の品詞の頻度分布から引く。文脈とは無関係なので、当たるとしたら分類器が単語を丸暗記したから
rng = np.random.default_rng(0)
tags, counts = np.unique(y["upos"][0], return_counts=True)
forms = sorted({t["form"] for s in train + test for t in s})
control_map = dict(zip(forms, rng.choice(tags, size=len(forms), p=counts / counts.sum())))
y["control"] = tuple([control_map[t["form"]] for s in d for t in s] for d in (train, test))

def probe(Xtr, ytr, Xte, yte, epochs=20, return_pred=False):
    """線形の分類器 (多クラスのロジスティック回帰) を学習し、評価データの正解率 (と予測) を返す"""
    classes = {c: i for i, c in enumerate(sorted(set(ytr)))}
    ytr = torch.tensor([classes[c] for c in ytr], device=device)
    yte = torch.tensor([classes.get(c, -1) for c in yte], device=device)   # 学習データにないラベルは必ず不正解
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6                                # 標準化
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    torch.manual_seed(0)
    clf = torch.nn.Linear(Xtr.shape[1], len(classes)).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=3e-3, weight_decay=0.01)
    for _ in range(epochs):
        for i in torch.randperm(len(Xtr), device=device).split(1024):
            loss = F.cross_entropy(clf(Xtr[i]), ytr[i])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = clf(Xte).argmax(-1)
    acc = (pred == yte).float().mean().item()
    return (acc, [list(classes)[i] for i in pred.tolist()]) if return_pred else acc

# 2 モデル × 3 タスク × 13 層 = 78 個の分類器。T4 で 3〜5 分
acc = {}
for m, (Ftr, Fte) in feats.items():
    for task in ["upos", "deprel", "control"]:
        acc[m, task] = [probe(Ftr[l].float(), y[task][0], Fte[l].float(), y[task][1]) for l in range(Ftr.shape[0])]
        print(f"{m:12} {task:8}", " ".join(f"{a:.3f}" for a in acc[m, task]))

# 参考: 文脈を見ずに「学習データでその単語に最も多く付いていたラベル」を答える (未知語は全体の最頻ラベル)
def by_form(task):
    c = {}
    for t_, lab in zip((t["form"] for s in train for t in s), y[task][0]):
        c.setdefault(t_, Counter())[lab] += 1
    default = Counter(y[task][0]).most_common(1)[0][0]
    pred = [c[t["form"]].most_common(1)[0][0] if t["form"] in c else default for s in test for t in s]
    return float(np.mean(np.array(pred) == np.array(y[task][1])))
baseline = {task: by_form(task) for task in ["upos", "deprel"]}
print("単語ごとの最頻ラベル:", {k: round(v, 3) for k, v in baseline.items()})
```

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, task in zip(axes, ["upos", "deprel"]):
    ax.plot(acc["BERT", task], "o-", label="BERT")
    ax.plot(acc["random BERT", task], "s--", label="random BERT")
    ax.axhline(baseline[task], color="gray", ls=":", label="most frequent label per word")
    ax.set_title(task.upper() if task == "upos" else "deprel")
    ax.set_xlabel("layer (0 = embeddings)"); ax.set_ylabel("accuracy"); ax.legend()
axes[0].plot(acc["BERT", "control"], "^-", label="BERT, control task"); axes[0].legend()
plt.tight_layout(); plt.show()

# 選択性 (selectivity) = 本来のタスクの正解率 − 対照タスクの正解率
for m in ["BERT", "random BERT"]:
    sel = np.array(acc[m, "upos"]) - np.array(acc[m, "control"])
    print(f"選択性 ({m}, UPOS):", " ".join(f"{s:.3f}" for s in sel))
for task in ["upos", "deprel"]:
    best = int(np.argmax(acc["BERT", task]))
    print(f"{task}: 最高は layer {best} ({acc['BERT', task][best]:.3f})、layer 0 は {acc['BERT', task][0]:.3f}")
```

```python
# 係り受けのラベルごとに、文脈のない層 0 と、評価データで最も良かった層を比べる
# (層の選択に評価データを使っているので、厳密には開発データ ja_gsd-ud-dev で選ぶ)
best = int(np.argmax(acc["BERT", "deprel"]))
Ftr, Fte = feats["BERT"]
gold = np.array(y["deprel"][1])
_, pred0 = probe(Ftr[0].float(), y["deprel"][0], Fte[0].float(), y["deprel"][1], return_pred=True)
_, predb = probe(Ftr[best].float(), y["deprel"][0], Fte[best].float(), y["deprel"][1], return_pred=True)
print(f"評価データで最も良かった層: {best}")
print(f"{'label':12}{'件数':>6}{'layer 0':>9}{f'layer {best}':>9}")
for lab, n in Counter(gold).most_common(12):
    m = gold == lab
    print(f"{lab:12}{n:6}{np.mean(np.array(pred0)[m] == lab):9.3f}{np.mean(np.array(predb)[m] == lab):9.3f}")
```

- プローブ (probe) は、モデルの内部表現を入力にした簡単な分類器で、その情報が表現から取り出せるかを調べる（線形のプローブは Alain & Bengio, 2016。BERT の層ごとの分析は Tenney et al., 2019 など）。単語ベクトルの取り出し方はTransformers の基本操作の章の「BERT から文脈に応じた単語ベクトルを取り出す」と同じく `output_hidden_states=True` で、層 0 は文脈を見ない埋め込み層の出力（位置の情報は含む）。
- この実行では、品詞 (UPOS) は層 0 で 0.883（単語ごとの最頻ラベルの 0.875 とほぼ同じ）から層 1 で 0.960 に上がり、層 4〜5 の 0.971 が最高で、層 12 では 0.950 に下がった。係り受けのラベル (deprel) は層 0 の 0.643 から層 5 の 0.882 まで上がった。評価は 543 文（約 1.3 万語）、シード 1 つの結果なので、層 2〜6 の間の 0.01 未満の差で順位は決めない。
- 層 0 と、評価データで最も良かった層 5 のラベルごとの比較では、`nsubj` (0.161 → 0.857) や `nmod` (0.239 → 0.761) など、同じ名詞でも後ろの助詞や係り先によって決まるラベルが大きく上がった。`case` や `punct` は層 0 で既にほぼ当たる。日本語の品詞は単語の形でほぼ決まるので、文脈の効果は係り受けのラベルのほうが見えやすい。
- 対照タスク (Hewitt & Liang, 2019) の正解率は層 0 の 0.790 から層 12 の 0.650 まで下がり、選択性（品詞の正解率 − 対照タスクの正解率）は 0.093 から約 0.30 に上がった。下の層ほど単語の丸暗記で当てやすい。ただし重みがランダムな BERT でも対照タスクは 0.788 から 0.745 に下がるので、低下の一部は学習の効果ではなく、層を重ねるごとにほかのトークンの情報が混ざることによる。正解率の最高（層 4〜5）と選択性の最高（層 9〜12、0.30 前後）の層がずれた点は、論文の ELMo の結果（1 層目のほうが品詞の正解率はわずかに高いが、2 層目のほうが選択性はかなり高い）と似ている。
- 重みがランダムな BERT でも品詞は 0.78〜0.80 当たるが、対照タスクも 0.75〜0.79 当たり、選択性は 0.01〜0.03 程度しかない。単語の区別が残っていれば分類器が暗記で当てられるということで、「正解率が高い = モデルが品詞を学習した」とは言えない。必ず対照と比べる。
- プロービングで分かるのは「情報が（線形に）取り出せる」ことまでで、モデルが予測にその情報を使っているかは分からない。分類器の表現力を上げれば正解率は上がるが、暗記できる分も増えるので選択性と合わせて見る。結果は学習データの量、分類器の設定、サブワードの代表の選び方（ここでは先頭）にも左右される。この章の「アテンションの重みをヒートマップで可視化する」の重みと同様、相関的な分析であることに注意。
- データは構文解析の章と同じ UD Japanese-GSD（CC BY-SA 4.0。元の文は Google UDT 2.0 のニュース・ブログ）、モデルは東北大学の BERT（Apache 2.0、CC-100 と Wikipedia で学習）。UD の単語単位（短単位）をそのまま 1 語ずつトークナイザに通しているので、モデル本来の分割 (MeCab + unidic-lite) とは少し違う入力になる。

## 言語モデルのどの層で次のトークンの予測が固まるかを見る（logit lens と tuned lens） {logit_lens gpu}

デコーダの言語モデルが「日本の首都は」の次に「東京」を出すとき、その予測は何層目で現れるのかを知りたい。各層の隠れ状態に最終層の正規化と出力層をそのまま当てて読む logit lens と、層ごとに小さな変換を学習してから読む tuned lens で、層ごとの予測を図にする。

```python
!pip install -q tuned-lens nnsight datasets sentencepiece matplotlib-fontja
```

```python
import random
import numpy as np
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(0); np.random.seed(0); torch.manual_seed(0)

# SB Intuitions の Sarashina2.2-0.5B（指示チューニング前のベースモデル。MIT ライセンス）
name = "sbintuitions/sarashina2.2-0.5b"
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)

BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()
n_layers = model.config.num_hidden_layers
print(n_layers, "層", model.model.norm, model.lm_head)

def encode(prompt):
    return tokenizer(tokenizer.bos_token + prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)

@torch.no_grad()
def logit_lens(prompt):
    """各層の隠れ状態を、最終層の正規化 (RMSNorm) と出力層に通したロジット。形は（層 0〜24、位置、語彙数）で、層 0 は埋め込み"""
    ids = encode(prompt)
    out = model(ids, output_hidden_states=True)
    hs = torch.stack(out.hidden_states[:-1])[:, 0]         # 層 0〜23。最後の要素（層 24）は正規化済みなので除く
    logits = model.lm_head(model.model.norm(hs))
    return torch.cat([logits, out.logits]).float(), ids[0]  # 層 24 はモデル本来の出力

def show_top(logits, k=3):
    probs = logits[:, -1].softmax(-1)
    for l, p in enumerate(probs):
        top = p.topk(k)
        print(f"層 {l:2}  " + "  ".join(f"{tokenizer.decode(i)!r}:{v:.2f}" for v, i in zip(top.values.tolist(), top.indices.tolist())))

with torch.no_grad():
    out = model(encode("日本の首都は"), output_hidden_states=True)
print(len(out.hidden_states), "個の隠れ状態。最後の 1 個に出力層だけを当てるとモデルの出力と一致:",
      torch.allclose(model.lm_head(out.hidden_states[-1]).float(), out.logits.float(), atol=1e-2))
logits, ids = logit_lens("日本の首都は")
print([tokenizer.decode(i) for i in ids])
show_top(logits)
```

```python
import matplotlib.pyplot as plt
import matplotlib_fontja  # import するだけで日本語のフォントが使える

# 位置ごと・層ごとの上位 1 位の予測（色はその確率）。各位置の予測は「その次のトークン」。図の「_」は空白
prompt = "日本の首都は東京、フランスの首都は"
logits, ids = logit_lens(prompt)
probs = logits[:, 1:].softmax(-1)                          # 先頭の BOS の位置は除く
top_p, top_i = probs.max(-1)
tokens = [tokenizer.decode(i) for i in ids[1:]]
fig, ax = plt.subplots(figsize=(10, 9))
ax.imshow(top_p.cpu(), cmap="Blues", vmin=0, vmax=1, aspect="auto", origin="lower")
for l in range(top_i.shape[0]):
    for j in range(top_i.shape[1]):
        ax.text(j, l, tokenizer.decode(top_i[l, j]).replace("\n", "\\n").replace(" ", "_").replace("\u3000", "_")[:5], ha="center", va="center", fontsize=8,
                color="white" if top_p[l, j] > 0.6 else "black")
ax.set_xticks(range(len(tokens)), tokens); ax.set_xlabel("入力のトークン（この次を予測する）")
ax.set_yticks(range(0, n_layers + 1, 2)); ax.set_ylabel("層（0 = 埋め込み）")
ax.set_title("logit lens: 各層の上位 1 位の予測")
plt.tight_layout(); plt.show()
```

```python
# 事実の想起・翻訳・計算・言葉の続きの例で、正解のトークンの層ごとの順位と確率を見る
examples = [
    ("事実", "日本の首都は", "東京"), ("事実", "ドイツの首都は", "ベルリン"), ("事実", "日本で一番高い山は", "富士山"),
    ("翻訳", "「りんご」は英語で「", "apple"), ("翻訳", "英語の「dog」は日本語で「", "犬"), ("翻訳", "犬 → dog\n猫 → cat\n水 →", " water"),
    ("計算", "2+3=5\n4+4=8\n3+4=", "7"), ("計算", "26+15=", "4"),          # 26+15=41 の最初の桁
    ("続き", "太陽は東から昇り、西に", "沈む"), ("続き", "月曜日の次は", "火曜日"),
]

def target_id(target):
    ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    assert len(ids) == 1, (target, ids)                    # 1 トークンの語だけを正解にする
    return ids[0]

def rank_prob(logits, tid):
    """最後の位置での正解トークンの順位（1 = 1 位）と確率を層ごとに"""
    last = logits[:, -1]
    return (last > last[:, [tid]]).sum(-1).add(1).cpu().numpy(), last.softmax(-1)[:, tid].cpu().numpy()

ll = {}
for cat, prompt, target in examples:
    ll[prompt] = rank_prob(logit_lens(prompt)[0], target_id(target))
    r, p = ll[prompt]
    first = int(np.argmax(r == 1)) if (r == 1).any() else None
    print(f"{cat} {prompt!r:28} → {target!r:8} 最終層の確率 {p[-1]:.2f}  初めて 1 位になる層 {first}  順位（層 0, 6, 12, 18, 24）: {r[[0, 6, 12, 18, 24]].tolist()}")

colors = {"事実": "tab:blue", "翻訳": "tab:orange", "計算": "tab:green", "続き": "tab:red"}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for cat, prompt, target in examples:
    r, p = ll[prompt]
    label = f"{prompt.splitlines()[-1]}（{target.strip()}）"
    axes[0].plot(r, color=colors[cat], marker=".", label=label); axes[1].plot(p, color=colors[cat], marker=".")
axes[0].set_yscale("log"); axes[0].invert_yaxis(); axes[0].set_ylabel("正解のトークンの順位（対数）")
axes[1].set_ylabel("正解のトークンの確率")
for ax in axes: ax.set_xlabel("層（0 = 埋め込み）")
axes[0].legend(fontsize=8, ncol=2); fig.suptitle("logit lens")
plt.tight_layout(); plt.show()
```

```python
# nnsight（NDIF のライブラリ）でも同じことが書ける。各層のモジュールの出力を取り出したり書き換えたりできる
import nnsight
from nnsight import TransformersModel

lm = TransformersModel(model, tokenizer=tokenizer)        # 読み込み済みのモデルをそのまま包む
with lm.trace(tokenizer.bos_token + "日本の首都は"):
    preds = nnsight.save([])
    for layer in lm.model.layers:                          # transformers 5 では、各層の出力は隠れ状態のテンソル
        preds.append(lm.lm_head(lm.model.norm(layer.output))[0, -1].argmax().save())
nn_top1 = [tokenizer.decode(int(i)) for i in preds]
lens_top1 = [tokenizer.decode(int(i)) for i in logit_lens("日本の首都は")[0][1:, -1].argmax(-1)]
print("nnsight の層 1〜24 の 1 位:", nn_top1)
print("hidden_states と一致した層の数:", sum(a == b for a, b in zip(nn_top1, lens_top1)), "/", n_layers)
```

```python
import time
import warnings
import torch.nn.functional as F
from datasets import load_dataset
from tuned_lens.nn.lenses import TunedLens

# tuned lens の学習データ: 日本語版 Wikipedia（wikimedia/wikipedia の 20231101.ja。本文は CC BY-SA 3.0 と GFDL）の先頭の記事を
# 256 トークンずつに切る（長い記事に偏らないよう 1 記事から最大 4 個）。記事の 20 本に 1 本は評価用に分ける
wiki = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
L, train, val = 256, [], []
for i, article in enumerate(wiki):
    ids = tokenizer(article["text"], add_special_tokens=False)["input_ids"]
    chunks = [[tokenizer.bos_token_id] + ids[s:s + L - 1] for s in range(0, len(ids) - L + 2, L - 1)][:4]
    (val if i % 20 == 0 else train).extend(chunks)
    if len(train) >= 1200 and len(val) >= 50:
        break
train, val = torch.tensor(train), torch.tensor(val[:50])
print("学習", tuple(train.shape), "評価", tuple(val.shape), f"（{i + 1} 記事）")

# 層ごとの「平行移動つきの線形変換」(translator) を、最終層の出力分布との KL ダイバージェンスが小さくなるように学習する。
# 変換は恒等写像から始まり、モデル本体と最終層の正規化・出力層は固定
lens = TunedLens.from_model(model).float().to(model.device)
params = [p for p in lens.parameters() if p.requires_grad]
print(f"translator のパラメータ {sum(p.numel() for p in params) / 1e6:.1f}M（{len(lens)} 層分）")
steps, batch_size = 100, 4
opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 20) * (1 - s / steps))   # 20 ステップのウォームアップ後に線形に減衰
amp_dtype = torch.bfloat16 if BF16 else torch.float16
scaler = torch.amp.GradScaler(enabled=not BF16)            # float16 のときは勾配のアンダーフローを防ぐ
# GradScaler が最初のステップを飛ばすと出る「scheduler の呼び出し順」の警告は、ここでは無害なので表示しない
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step")

def lens_kl(o, l, lens_fn):
    """層 l の隠れ状態をレンズで読んだ分布と、最終層の分布との KL ダイバージェンス（トークンあたり、nat）"""
    with torch.autocast("cuda", dtype=amp_dtype):
        logp = F.log_softmax(lens_fn(o.hidden_states[l].float()).float(), -1)
    return F.kl_div(logp, F.log_softmax(o.logits.float(), -1), log_target=True, reduction="none").sum(-1).mean()

g = torch.Generator().manual_seed(0)
t0 = time.time()
# T4 で 3〜5 分ほど（目安）
for step in range(steps):
    x = train[torch.randint(len(train), (batch_size,), generator=g)].to(model.device)
    with torch.no_grad():
        o = model(x, output_hidden_states=True)
    total = 0.0
    for l in range(n_layers):                              # 層ごとに backward してメモリを抑える（translator は層ごとに独立）
        loss = lens_kl(o, l, lambda h: lens(h, l))
        scaler.scale(loss).backward(); total += loss.item()
    scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
    if step % 20 == 0 or step == steps - 1:
        print(f"step {step:3}  平均 KL {total / n_layers:.3f}  {time.time() - t0:.0f} 秒")
print(f"GPU メモリの最大 {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

```python
# 評価用の記事で、層ごとに「最終層の分布との KL」を logit lens と tuned lens で比べる
kl = {"logit lens": np.zeros(n_layers), "tuned lens": np.zeros(n_layers)}
with torch.no_grad():
    for b in range(0, len(val), batch_size):
        o = model(val[b:b + batch_size].to(model.device), output_hidden_states=True)
        for l in range(n_layers):
            kl["logit lens"][l] += lens_kl(o, l, lens.unembed).item() * len(val[b:b + batch_size]) / len(val)
            kl["tuned lens"][l] += lens_kl(o, l, lambda h: lens(h, l)).item() * len(val[b:b + batch_size]) / len(val)
for k, v in kl.items():
    print(f"{k:10}", " ".join(f"{x:.1f}" for x in v))

@torch.no_grad()
def tuned_lens(prompt):
    ids = encode(prompt)
    out = model(ids, output_hidden_states=True)
    logits = [lens(out.hidden_states[l][0].float(), l) for l in range(n_layers)]
    return torch.cat([torch.stack(logits), out.logits.float()]), ids[0]

tl = {prompt: rank_prob(tuned_lens(prompt)[0], target_id(target)) for _, prompt, target in examples}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for k, v in kl.items():
    axes[0].plot(v, marker=".", label=k)
axes[0].set_yscale("log"); axes[0].set_xlabel("層（0 = 埋め込み）"); axes[0].set_ylabel("最終層の分布との KL（nat、対数）"); axes[0].legend()
for cat, prompt, target in examples:
    axes[1].plot(ll[prompt][0], color=colors[cat], ls=":", alpha=0.6)
    axes[1].plot(tl[prompt][0], color=colors[cat], marker=".")
axes[1].set_yscale("log"); axes[1].invert_yaxis(); axes[1].set_xlabel("層（0 = 埋め込み）"); axes[1].set_ylabel("正解のトークンの順位（対数）")
axes[1].set_title("実線 = tuned lens、点線 = logit lens（色は上の図と同じ）")
plt.tight_layout(); plt.show()

for cat, prompt, target in examples:
    first = {k: (int(np.argmax(d[prompt][0] == 1)) if (d[prompt][0] == 1).any() else None) for k, d in [("logit", ll), ("tuned", tl)]}
    print(f"{cat} {target!r:8} 初めて 1 位になる層  logit lens {first['logit']}  tuned lens {first['tuned']}   "
          f"層 12 の順位  logit lens {ll[prompt][0][12]}  tuned lens {tl[prompt][0][12]}")
show_top(tuned_lens("日本の首都は")[0])
```

- logit lens (nostalgebraist, 2020) は、途中の層の隠れ状態を「最終層の手前にあるもの」とみなして、最終層の正規化と出力層 (unembedding) をそのまま当てて読む。学習は要らない。`output_hidden_states=True` で得られる 25 個（埋め込み + 24 層）のうち、transformers の Llama 系では最後の 1 個が最終層の正規化を通した後の値なので（出力の `True` のとおり）、正規化を 2 回かけないよう層 24 はモデルの出力をそのまま使っている。正解はどれも 1 トークンの語にしてある（「26+15=」は答え 41 の最初の桁の「4」）。
- この実行では、10 例とも正解のトークンが 1 位になったのは層 19〜24 で、確率は層 18 まではほぼ 0 だった。ヒートマップでは、前半の層は入力のトークン自身が 1 位になりやすく（「首都」の位置では層 13 まで「首都」、「東京」の位置では層 6 まで「東京」）、中間の層は「、」や空白（図の「_」）、「である」などのありふれたトークンが多い。「日本の首都は東京、フランスの首都は」の最後の位置では、層 20 で文脈に出てきた「東京」が 1 位になり、層 21 から「パリ」に変わった。
- 素朴な logit lens の限界: 途中の層の隠れ状態が最終層と同じ「座標系」にあるとは限らない。評価用の記事で最終層の分布との KL ダイバージェンスを測ると、層 0 で 73.5、層 12 で 6.3 nat と大きく、前半の層はほとんど読めていない。読めないことは情報がないことを意味しない。Belrose ら (2023) は、BLOOM や OPT では logit lens がもっともらしい予測を取り出せず、多くの層で入力のトークンが 1 位になること、特定の語彙に偏ることを報告している。
- tuned lens（Belrose ら, 2023）は、層ごとに平行移動つきの線形変換 (translator) を、最終層の分布との KL が小さくなるように学習してから読む。ここでは tuned-lens ライブラリ（MIT ライセンス）の `TunedLens` を使い、仕組みが見えるよう学習のループは自分で書いた。学習は 100 ステップ × 4 × 256 トークン（約 10 万トークン）で、論文の初期の設定（SGD、250 ステップ × 2^18 トークン）よりずっと少ない（論文の最新版は、最適化手法を Muon にすると KL がずっと下がるとして Muon を勧めている）。それでも評価用の記事での KL は層 0 で 73.5 → 11.6、層 12 で 6.3 → 2.5 に下がり、「日本の首都は」の前半の層で入力のトークン「は」が 1 位になる現象は消えた。
- 一方、正解が初めて 1 位になる層は、tuned lens でも 10 例中 5 例で 1〜3 層早まっただけで（4 例は同じ、「火曜日」は 1 層遅い）、層 12 での正解の順位が上がったのは 10 例中 8 例だった。例が 10 個、学習が短く 1 回だけの結果なので、「どの層で答えが決まるか」の細かい違いは言えない。ライブラリが公開している学習済みのレンズは GPT-2 や EleutherAI、Meta などの英語中心のモデル向けで、日本語のモデルは自分で学習する。
- この章の「BERT の各層に品詞や構文の情報がどれくらい入っているかを調べる（プロービング）」は、外部のラベル（品詞など）を予測する分類器を層ごとに学習する。logit lens はモデル自身の出力層で語彙の空間に読み出すもので学習が要らず、tuned lens はその中間（目標をモデル自身の最終出力にした線形のプローブ）にあたる。どれも「その層から読み出せる」ことを示すだけで、モデルがその層の情報を使っているかは分からない。確かめるには、層の出力を書き換える介入の実験をする。
- nnsight（Fiotto-Kaufman ら, 2024。MIT ライセンス）は、そのような介入にも使えるライブラリで、`layer.output` を読むだけでなく書き換えることもできる。nnsight 0.8 では `LanguageModel` は非推奨で `TransformersModel` を使う。transformers 5 では層の出力がテンソルなので、古い資料の `layer.output[0]` は不要になる。層の出力を書き換える例は、この章の「隠れ状態にベクトルを足して生成の感情や文体を変える（アクティベーション・ステアリング）」を参照（forward hook で実装している）。
- モデルは SB Intuitions の Sarashina2.2-0.5B（MIT ライセンス。日本語・英語・コードのウェブのデータなど 10 兆トークンで学習したベースモデル）、tuned lens の学習データは日本語版 Wikipedia（CC BY-SA 3.0 と GFDL）。トークナイザーの修正は、大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」の注意書きを参照。T4 と同じ float16 の経路で実行し、GPU メモリの最大は 5.6 GB だった。

## 隠れ状態にベクトルを足して生成の感情や文体を変える（アクティベーション・ステアリング） {activation_steering gpu}

追加の学習をせずに、言語モデルの生成をポジティブな内容やです・ます調に寄せたい。対になる文（肯定的な文と否定的な文など）の隠れ状態の差の平均を「ステアリングベクトル」とし、生成のときに特定の層の出力（残差ストリーム）に足す。足す層と強さを変えて、感情分類器で効果を、パープレキシティと distinct-2 で文の崩れ方を測る。

```python
!pip install -q matplotlib-fontja==1.1.0
```

```python
import random
import numpy as np
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(0); np.random.seed(0); torch.manual_seed(0)

# SB Intuitions の Sarashina2.2-0.5B（指示チューニング前のベースモデル。MIT ライセンス）。この章の logit lens のレシピと同じモデル
name = "sbintuitions/sarashina2.2-0.5b"
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
tokenizer.padding_side = "left"                        # 複数のプロンプトをまとめて生成するので、パディングは左に入れる

BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()
n_layers = model.config.num_hidden_layers

# 対になる文: 感情だけが違い、ほかはなるべく同じにする（自作の例文）
pairs = [
    ("この映画は本当に素晴らしかった。", "この映画は本当にひどかった。"), ("店員さんがとても親切で嬉しかった。", "店員さんがとても無愛想で腹が立った。"),
    ("今日は最高の一日だった。", "今日は最悪の一日だった。"), ("料理はおいしくて大満足です。", "料理はまずくてがっかりです。"),
    ("新しい職場は楽しくて、毎日が充実している。", "新しい職場はつらくて、毎日が苦痛だ。"), ("この本を読んで心が温かくなった。", "この本を読んで気分が悪くなった。"),
    ("旅行は天気にも恵まれて、楽しい思い出になった。", "旅行は雨ばかりで、嫌な思い出になった。"), ("彼の発表はわかりやすくて感動した。", "彼の発表はわかりにくくて退屈だった。"),
    ("買ってよかったと心から思う。", "買わなければよかったと心から思う。"), ("試験に合格して、とても幸せだ。", "試験に落ちて、とても悲しい。"),
    ("友達と過ごす時間は本当に楽しい。", "友達と過ごす時間は本当に面倒だ。"), ("このホテルは清潔で快適だった。", "このホテルは汚くて不快だった。"),
    ("子どもたちの笑顔に元気をもらった。", "子どもたちの騒ぎ声にうんざりした。"), ("サービスが丁寧で、また来たいと思った。", "サービスが雑で、二度と来たくないと思った。"),
    ("春の公園は花が咲いていて気持ちがいい。", "春の公園は人が多くて気分が悪い。"), ("この曲を聴くと幸せな気持ちになる。", "この曲を聴くと憂うつな気持ちになる。"),
    ("引っ越し先の近所の人はみんな優しい。", "引っ越し先の近所の人はみんな冷たい。"), ("久しぶりに会えて、本当に嬉しかった。", "久しぶりに会ったのに、本当に気まずかった。"),
    ("この靴は軽くて歩きやすい。", "この靴は重くて歩きにくい。"), ("プレゼントをもらって、すごく喜んだ。", "プレゼントを忘れられて、すごく落ち込んだ。"),
    ("先生のおかげで勉強が好きになった。", "先生のせいで勉強が嫌いになった。"), ("このアプリは便利で手放せない。", "このアプリは不便で使う気にならない。"),
    ("駅のそばに素敵なカフェを見つけた。", "駅のそばのカフェで嫌な目にあった。"), ("みんなの協力で、イベントは大成功だった。", "準備不足で、イベントは大失敗だった。"),
]
# 方向が文の感情を表しているかを確かめるための別の文（ベクトルの計算には使わない）
test_pos = ["週末の遠足はとても楽しかった。", "この店のケーキは絶品だ。", "新しい部屋は明るくて気に入っている。", "息子が元気に育っていて嬉しい。",
            "親切な人に道を教えてもらって助かった。", "このドラマは毎回わくわくする。", "温泉に入って疲れがすっかり取れた。", "念願のチームに入れて夢のようだ。"]
test_neg = ["週末の遠足は散々だった。", "この店のケーキは食べられたものではない。", "新しい部屋は暗くて気がめいる。", "息子が熱を出して心配でたまらない。",
            "道に迷って、約束に遅れてしまった。", "このドラマは毎回退屈だ。", "満員電車で押されて、くたくたになった。", "選考に落ちて、何もする気が起きない。"]

@torch.no_grad()
def states(texts):
    """各文の隠れ状態の、トークン（先頭の BOS を除く）についての平均（文の数、層 0〜24、次元）と、トークンごとの大きさ（ノルム）の平均（層 0〜24）"""
    means, norms = [], []
    for t in texts:
        ids = tokenizer(tokenizer.bos_token + t, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
        hs = torch.stack(model(ids, output_hidden_states=True).hidden_states)[:, 0, 1:].float()   # （層、トークン、次元）
        means.append(hs.mean(1)); norms.append(hs.norm(dim=-1).mean(1))
    return torch.stack(means), torch.stack(norms).mean(0)

pos, pos_norm = states([p for p, _ in pairs])
neg, neg_norm = states([n for _, n in pairs])
steer_vec = (pos - neg).mean(0)                        # 層ごとのステアリングベクトル（対の差の平均）: （層 0〜24、次元）
h_norm = (pos_norm + neg_norm) / 2                     # 層ごとの隠れ状態の大きさの目安（足す強さの基準にする）
print("隠れ状態の大きさ（層 0, 3, 6, 12, 18, 23）:", [round(h_norm[l].item(), 1) for l in [0, 3, 6, 12, 18, 23]])
print("ベクトルの大きさ / 隠れ状態の大きさ（同じ層）:", [round((steer_vec[l].norm() / h_norm[l]).item(), 2) for l in [0, 3, 6, 12, 18, 23]])
# 上の平均から除いた先頭の BOS の位置は、層によっては大きさがほかの位置とまったく違う
with torch.no_grad():
    ids = tokenizer(tokenizer.bos_token + pairs[0][0], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    hs_bos = model(ids, output_hidden_states=True).hidden_states
print("BOS の位置の隠れ状態の大きさ（層 0, 3, 6, 12, 18, 23）:", [round(hs_bos[l][0, 0].float().norm().item(), 1) for l in [0, 3, 6, 12, 18, 23]])
```

```python
import matplotlib.pyplot as plt
import matplotlib_fontja  # import するだけで日本語のフォントが使える

# 確認 1: ベクトルの方向への射影で、別の文のポジ・ネガを分けられるか（差の平均を方向にした、ごく単純な線形のプローブ）
tp, _ = states(test_pos)
tn, _ = states(test_neg)
acc, gap = [], []
for l in range(n_layers + 1):
    v = steer_vec[l] / steer_vec[l].norm()
    threshold = ((pos[:, l] @ v).mean() + (neg[:, l] @ v).mean()) / 2   # しきい値も対の文から決める
    proj_pos, proj_neg = tp[:, l] @ v, tn[:, l] @ v   # 別の文のポジ・ネガそれぞれの射影
    acc.append(((proj_pos > threshold).float().mean() + (proj_neg <= threshold).float().mean()).item() / 2)
    gap.append(((proj_pos.mean() - proj_neg.mean()) / torch.cat([proj_pos, proj_neg]).std()).item())   # 2 群の平均の差（標準偏差を単位に）
print("別の 16 文の正解率（層 0〜24。層 24 は最終の正規化の後）:", " ".join(f"{a:.2f}" for a in acc))
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(gap, "o-"); ax.set_xlabel("層（0 = 埋め込み）"); ax.set_ylabel("射影の平均の差（標準偏差を単位に）")
ax.set_title("ステアリングベクトルの方向で、別の文のポジ・ネガがどれだけ離れるか")
plt.tight_layout(); plt.show()

# 確認 2: ベクトルを logit lens と同じく最終層の正規化と出力層で読むと、どのトークンを押し上げる方向か
for l in [6, 12, 18]:
    logits = model.lm_head(model.model.norm(steer_vec[l].to(model.dtype))).float()
    up, down = logits.topk(10).indices, (-logits).topk(10).indices
    print(f"層 {l:2}  + 方向: {[tokenizer.decode(i) for i in up]}\n        − 方向: {[tokenizer.decode(i) for i in down]}")
```

```python
import contextlib

@contextlib.contextmanager
def steering(vec, layer, alpha):
    """with の中では、層 layer (1〜23) の出力（残差ストリーム）に、隠れ状態の大きさの alpha 倍の長さにした vec を足す"""
    add = (vec[layer] / vec[layer].norm() * h_norm[layer] * alpha).to(model.dtype)
    def hook(module, inputs, output):
        return output + add                            # transformers 5 の Llama 系では、層の出力は隠れ状態のテンソル。戻り値で置き換わる
    # model.model.layers[l - 1] の出力が hidden_states[l]（層 l）にあたる。プロンプトと生成したトークンのすべての位置に足す
    # （BOS とパディングの位置にも足すが、層 3 以降の BOS の隠れ状態はほかの位置よりずっと大きく（ベクトルを作るセルの出力）、パディングはアテンションで無視されるので、影響は小さいと考えられる）
    handle = model.model.layers[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()                                # 外し忘れると、以後のすべての計算にベクトルが足され続ける

prompts = ["昨日、駅前の新しいレストランに行った。", "先週から新しい仕事を始めた。", "今日は朝から雨が降っていた。", "この前、友達と映画を見に行った。",
           "週末に家族で海へ出かけた。", "最近、新しいスマートフォンを買った。", "今朝、電車の中で", "引っ越してから一か月がたった。",
           "久しぶりに実家に帰った。", "昨日の会議で、", "近所に新しいカフェができた。", "この本の感想を書く。",
           "今年の夏休みは", "兄から手紙が届いた。", "駅まで歩いていると、", "新しい先生が来た。",
           "昼ごはんにラーメンを食べた。", "午後から図書館で勉強した。", "初めて飛行機に乗った。", "今日の天気は"]
N_SAMPLES = 4                                          # プロンプトごとに 4 本、計 80 本を生成する

@torch.no_grad()
def generate(max_new_tokens=40, seed=0):
    enc = tokenizer([tokenizer.bos_token + p for p in prompts], add_special_tokens=False, padding=True, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)                            # どの条件でも同じ乱数で生成し、違いをステアリングだけにする
    out = model.generate(**enc, do_sample=True, temperature=0.8, top_p=0.9, max_new_tokens=max_new_tokens,
                         num_return_sequences=N_SAMPLES, pad_token_id=tokenizer.pad_token_id)
    return [tokenizer.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]

base = generate()
with steering(steer_vec, 12, 0.25):
    steered = generate()
for i in [0, 12, 40]:
    print(f"[{prompts[i // N_SAMPLES]}]\n  なし: {base[i]!r}\n  あり: {steered[i]!r}")
```

```python
import time
import pandas as pd
from transformers import pipeline

# 評価 1: 感情分類器（テキスト分類の章の「日本語の感情分析（ポジネガ判定）をする」と同じ多言語モデル。Apache 2.0）。
# スコアは P(positive) − P(negative) で、−1〜+1
clf = pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", device=model.device, top_k=None)

def sentiment(texts):
    out = clf([t if t.strip() else "。" for t in texts], truncation=True, batch_size=64)   # 空の文字列は分類器に渡せないので「。」に置き換える
    scores = []
    for r in out:
        d = {x["label"]: x["score"] for x in r}         # ラベルごとの確率
        scores.append(d["positive"] - d["negative"])
    return np.array(scores)

# 評価 2: 生成した続きの、ステアリングなしのモデルでのパープレキシティ（プロンプトを条件にした続きの部分だけ）
ptexts = [p for p in prompts for _ in range(N_SAMPLES)]
@torch.no_grad()
def perplexity(texts):
    nll, n = 0.0, 0
    for p, t in zip(ptexts, texts):
        a = tokenizer(tokenizer.bos_token + p, add_special_tokens=False)["input_ids"]
        b = tokenizer(t, add_special_tokens=False)["input_ids"]
        if not b:
            continue
        ids = torch.tensor([a + b], device=model.device)
        labels = ids.clone(); labels[:, :len(a)] = -100                # プロンプトの部分は数えない
        nll += model(ids, labels=labels).loss.float().item() * len(b); n += len(b)
    return float(np.exp(nll / n))

# 評価 3: 繰り返しの少なさ（distinct-2 = 異なるトークンの 2-gram の数 / 2-gram の総数）。同じ語の繰り返しはパープレキシティが低くなるので、別に測る
def distinct2(texts):
    grams = [tuple(ids[i:i + 2]) for t in texts for ids in [tokenizer(t, add_special_tokens=False)["input_ids"]] for i in range(len(ids) - 1)]
    return len(set(grams)) / max(len(grams), 1)

def evaluate(texts):
    s = sentiment(texts)
    return {"sentiment": s.mean(), "positive_rate": (s > 0).mean(), "ppl": perplexity(texts), "distinct2": distinct2(texts)}

# 層 7 つ × 強さ 7 つ = 49 条件 × 80 本。T4 で 5〜10 分ほど（目安）
layers, alphas = [3, 6, 9, 12, 15, 18, 21], [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0]
t0 = time.time()
texts = {(0, 0.0): base}
for layer in layers:
    for alpha in alphas:
        with steering(steer_vec, layer, alpha):
            texts[layer, alpha] = generate()
print(f"生成 {time.time() - t0:.0f} 秒")
results = pd.DataFrame([{"layer": l, "alpha": a, **evaluate(t)} for (l, a), t in texts.items()])
print(f"評価まで {time.time() - t0:.0f} 秒、GPU メモリの最大 {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
b = results.iloc[0]
print(f"ステアリングなし: 感情スコア {b.sentiment:.3f}  ポジティブの割合 {b.positive_rate:.2f}  PPL {b.ppl:.1f}  distinct-2 {b.distinct2:.3f}")
for col in ["sentiment", "ppl", "distinct2"]:
    print(f"\n{col}（行 = 層、列 = alpha）\n" + results.iloc[1:].pivot(index="layer", columns="alpha", values=col).round(2).to_string())
```

```python
grid = sorted(alphas + [0.0])                          # 横軸は等間隔に並べる。alpha 0 はどの層もステアリングなしの値
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, col, label in zip(axes, ["sentiment", "ppl", "distinct2"], ["感情スコア (P(positive) − P(negative))", "パープレキシティ（対数）", "distinct-2"]):
    for layer in layers:
        r = results[results.layer == layer].set_index("alpha")[col]
        ax.plot(range(len(grid)), [b[col] if a == 0 else r[a] for a in grid], marker=".", label=f"層 {layer}")
    ax.axhline(b[col], color="gray", ls=":", label="ステアリングなし")
    ax.set_xticks(range(len(grid)), [str(a) for a in grid])
    ax.set_xlabel("強さ alpha（隠れ状態の大きさに対する倍率。負はネガティブの向き）"); ax.set_title(label)
axes[1].set_yscale("log"); axes[0].legend(fontsize=8)
plt.tight_layout(); plt.show()
```

```python
# 効果と崩れのバランスが良い条件: PPL がなしの 2 倍以内、distinct-2 がなしの 0.9 倍以上のうち、感情スコアが最も高いもの
ok = results[(results.alpha > 0) & (results.ppl < 2 * b.ppl) & (results.distinct2 > 0.9 * b.distinct2)]
best = ok.loc[ok.sentiment.idxmax()]
L = int(best.layer)
print(f"条件を満たすもののうち感情スコアが最も高い: 層 {L}、alpha {best.alpha}（感情スコア {best.sentiment:.3f}、PPL {best.ppl:.1f}、distinct-2 {best.distinct2:.3f}）")

# 条件を選んだのと同じ生成で差を測ると、たまたま高く出た条件を選んだぶん大きめに出る。乱数の種を変えて生成し直して測る
new_base = generate(seed=1)
with steering(steer_vec, L, best.alpha):
    new_steered = generate(seed=1)
# 差の信頼区間: プロンプト単位で復元抽出するブートストラップ（同じプロンプトの 4 本は独立ではないので）
s0 = sentiment(new_base).reshape(len(prompts), N_SAMPLES).mean(1)
s1 = sentiment(new_steered).reshape(len(prompts), N_SAMPLES).mean(1)
rng = np.random.default_rng(0)
boot = [(s1[i] - s0[i]).mean() for i in rng.integers(0, len(prompts), (2000, len(prompts)))]
print(f"生成し直した 80 本: なし {s0.mean():.3f}、あり {s1.mean():.3f}、差 {np.mean(s1 - s0):+.3f}  95% 信頼区間 [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]")

# 同じ層で強さを変えたときの生成例（同じプロンプト、同じ乱数）
for alpha in [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
    t = texts[(0, 0.0) if alpha == 0 else (L, alpha)]
    print(f"alpha {alpha:+.1f}: " + " / ".join(repr(t[i][:45]) for i in [0, 20]))
```

```python
# 文体も同じ方法で変えられる: です・ます調と、だ・である調の対からベクトルを作り、です・ます調の文の割合を数える
style_pairs = [("今日は雨が降っている。", "今日は雨が降っています。"), ("この店のラーメンはおいしい。", "この店のラーメンはおいしいです。"),
               ("明日は早く起きるつもりだ。", "明日は早く起きるつもりです。"), ("駅までは歩いて十分ほどだ。", "駅までは歩いて十分ほどです。"),
               ("昨日は友達と映画を見た。", "昨日は友達と映画を見ました。"), ("この問題はとても難しい。", "この問題はとても難しいです。"),
               ("週末は家でゆっくり過ごした。", "週末は家でゆっくり過ごしました。"), ("会議は午後三時から始まる。", "会議は午後三時から始まります。"),
               ("彼は毎朝ジョギングをしている。", "彼は毎朝ジョギングをしています。"), ("その本はまだ読んでいない。", "その本はまだ読んでいません。"),
               ("夏休みに北海道へ行く予定だ。", "夏休みに北海道へ行く予定です。"), ("この部屋は少し寒い。", "この部屋は少し寒いです。")]
plain, _ = states([a for a, _ in style_pairs])
polite, _ = states([b for _, b in style_pairs])
style_vec = (polite - plain).mean(0)                   # です・ます調の向き

def polite_rate(texts):
    """文末（。で区切った各文）が「です」「ます」「でした」「ました」「ません」などで終わる割合"""
    sents = [s.strip() for t in texts for s in t.split("。")[:-1] if s.strip()]
    return np.mean([s.endswith(("です", "ます", "でした", "ました", "ません", "でしょう", "ましょう")) for s in sents]) if sents else float("nan")

print(f"ステアリングなし: です・ます調の文の割合 {polite_rate(base):.2f}  PPL {perplexity(base):.1f}  distinct-2 {distinct2(base):.3f}")
for alpha in [0.25, 0.5, 1.0]:
    with steering(style_vec, L, alpha):
        t = generate()
    print(f"層 {L}、alpha {alpha}: です・ます調の文の割合 {polite_rate(t):.2f}  PPL {perplexity(t):.1f}  distinct-2 {distinct2(t):.3f}  例: {t[0][:50]!r}")
```

- アクティベーション・ステアリングは、追加の学習をせずに、推論中の隠れ状態にベクトルを足して出力を変える方法である。ActAdd（Turner ら, 2023）は「Love」と「Hate」のような 1 対のプロンプトの隠れ状態の差を、ある層のプロンプトの先頭の位置に足す（GPT-2-XL などで実験）。CAA（Rimsky ら, ACL 2024。arXiv の版の第 1 著者名は Nina Panickssery）は、多数の対（A/B の選択問題の答えの記号の位置）の差の平均を Llama 2 Chat の残差ストリームに、プロンプトより後のすべての位置で足す。ここでは CAA と同じく 24 対の差の平均を使い（各文の隠れ状態はトークンで平均）、プロンプトも含むすべての位置に足した。強さは、層によって隠れ状態の大きさが違う（層 3 で 14.1、層 23 で 47.8）ので、その層の隠れ状態の大きさに対する倍率 alpha で表した。差の平均そのものの長さは隠れ状態の 0.10〜0.19 倍である。
- この実行（プロンプト 20 個 × 4 本 = 80 本、40 トークン）では、alpha 0.25 で、ステアリングなしで 0.208 の感情スコアが層によって 0.32〜0.46 になり、おおむね上がった（最も小さい層 6 の +0.11 は、後で示す信頼区間の幅（±0.11 ほど）から見てばらつきの範囲かもしれない）。PPL（4.6〜6.5、なしは 4.9）と distinct-2（0.53〜0.65、なしは 0.62）は大きくは崩れなかった。逆向きの alpha −0.25 では −0.01〜0.07 に下がった。PPL が 2 倍以内で distinct-2 が 0.9 倍以上という条件で最も感情スコアが高かったのは層 3、alpha 0.5（0.493）だった。同じ生成で選んだ条件の差は大きめに出るので、乱数の種を変えて 80 本ずつ生成し直すと、なし 0.151、あり 0.439 で、差は +0.288（プロンプト単位のブートストラップの 95% 信頼区間 [+0.170, +0.400]）だった。ただし対の文もプロンプトも 1 組で、層と強さの探索は乱数の種 1 つの結果なので、層による 0.1 程度の違いで「この層が最適」とは言えない。CAA でも最適な層はモデルと課題ごとに探している。
- 強すぎると文が崩れる。alpha 1〜2 では「また、素敵な思い出をありがとうございます」の繰り返しや改行だけの出力が増え、alpha 2 の distinct-2 は 0.05〜0.28 まで下がった。alpha −1 の PPL は 11.5〜436 に上がった。一方、同じ語や改行の繰り返しは PPL がなしと同じくらいか、それより低くなる。層 6 の alpha 1 は PPL 2.27（distinct-2 0.09）、層 21 の alpha 2 は PPL 5.1 でなしとほぼ同じなのに distinct-2 は 0.05 だった。PPL だけでは崩れを見落とす。分類器も崩れた文を誤判定し、層 3 の alpha −1（ネガティブの向き）は感情スコアが +0.45 になった。自動評価の数値だけで条件を選ばず、生成例を必ず読む。
- この章の「BERT の各層に品詞や構文の情報がどれくらい入っているかを調べる（プロービング）」との関係: 差の平均の方向は、最も単純な線形のプローブにあたる。計算に使っていない 16 文を射影で分けると、正解率は層 0〜10 で 0.81〜0.94、層 11〜23 で 1.00（最終の正規化の後の層 24 では 0.88）、2 群の離れ方は層 14 前後が最大だった（16 文だけなので目安）。足して効いたのは、正解率が最も高い層だけではなかった（正解率 0.88 の層 3 でも感情は変わった。ただし層 3 に足したベクトルは残差ストリームを通って後ろの層にも届く）。プロービングは「情報が読み出せる」ことまでしか示さないが、ステアリングはその方向を書き換えると出力が変わることを示す介入の実験になる。
- この章の「言語モデルのどの層で次のトークンの予測が固まるかを見る（logit lens と tuned lens）」と同じく、ベクトルを最終層の正規化と出力層で読むと、+ の向きは「嬉しい」「感謝」「素敵な」や英語の warmth、enjoyed、− の向きは「ひどい」「酷い」や worse を押し上げる方向だった。足したベクトルは最終層まで残差ストリームを通って、こうした語の確率を直接上げていると考えられる。
- 文体も同じ方法で変わった。だ・である調とです・ます調の 12 対から作ったベクトルを層 3 に足すと、です・ます調で終わる文の割合が 0.25 から 0.78 (alpha 0.25)、0.85 (alpha 0.5) に上がった。ただし alpha 0.5 ですでに「レストランに行きました。」のような同じ文の繰り返しが出始め（distinct-2 はなしの 0.62 から 0.33 に下がり、PPL 3.9 はなしの 4.9 より低い）、alpha 1 では文が崩れた（distinct-2 0.11）。この実行で繰り返しがまだ少ないのは alpha 0.25 (distinct-2 0.55) までだった。層は感情の実験で選んだものをそのまま使い、文末の判定も簡単な文字列の比較である。
- 0.5B のベースモデルは、ステアリングなしでも英訳の練習や問題集のような続きをよく出し、多言語の感情分類器はそうした文もポジティブ寄りに判定する（なしでもポジティブの割合が 0.78）。指示に従う LLM なら、まずプロンプトで指示するほうが簡単で確実である。ステアリングは、プロンプトでは変えにくい性質を調べたり、内部の表現の役割を確かめたりするのに向く。forward hook は外し忘れると以後の計算すべてに効くので、with で囲んで必ず外す（この章の logit lens のレシピの nnsight でも同じ介入が書ける）。
- モデルは SB Intuitions の Sarashina2.2-0.5B（MIT ライセンス）、感情分類器は lxyuan/distilbert-base-multilingual-cased-sentiments-student (Apache 2.0)、対の文とプロンプトは自作。トークナイザーの修正は、大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」の注意書きを参照。T4 と同じ float16 の経路で実行し、GPU メモリの最大は 2.9 GB だった。
