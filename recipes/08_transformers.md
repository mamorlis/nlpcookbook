# Transformers の基本操作

Hugging Face Transformers と Datasets で、事前学習済みモデルを動かすときの定番の操作。言語モデルの確率で文の自然さや文法知識を測る方法、BERT の追加の事前学習と小さな GPT の事前学習、モデルの内部にどんな情報があるかを調べるプロービングと、どの層で予測が固まるかを見る logit lens も扱う。

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
- 評価の「1 位の正解率」は、この章の「マスクされた単語を BERT に予測させる (fill-mask)」の 1 位の予測が当たる割合を、隠した全トークンで数えたもの。この章の「マスク言語モデルで文の自然さを測る（疑似対数尤度）」は 1 トークンずつ隠して文長の回数だけ推論するが、ここでは語の 15% を同時に隠して 1 回で済ませている。隠す位置を固定しているので前後の比較はできるが、疑似対数尤度の値とは比べられない。
- tohoku-nlp/bert-base-japanese-v3 は、MeCab の 1 語のサブワードをまとめて隠す whole word masking で事前学習されている（モデルカード）。評価と追加学習でもそれにそろえた。`DataCollatorForLanguageModeling(whole_word_mask=True)` は fast トークナイザーの文字位置を使うので、MeCab を使うこのモデルのトークナイザーでは使えず、「##」の付くサブワードを手がかりに自分で書いた。
- 下流の分類（小説か随筆・評論か）では、正解率の平均が 0.824 → 0.828 で、差 +0.004 の 95% 信頼区間（作品 577 件を単位にしたブートストラップ）は [-0.004, +0.012] と 0 をまたいだ。シードによっても 0.01〜0.02 程度動く（元の BERT は 0.812〜0.830）。マスク言語モデルの損失が大きく改善しても、分類の精度が上がるとは限らないので、下流のタスクで、シードを変えた複数回の実行で確かめる。
- 保存した `AutoModelForMaskedLM` には、分類で使う pooler の層が含まれない。そのまま `AutoModelForSequenceClassification` で読むと pooler がランダムに初期化され、元のモデルとの比較が不公平になるので、上のコードでは pooler を元のモデルから取り、エンコーダーだけを差し替えた。
- データは青空文庫（globis-university/aozorabunko-clean、CC BY 4.0。著作権の切れた作品だけを収録）と日本語 Wikipedia（wikimedia/wikipedia、CC BY-SA 3.0 / GFDL）、モデルは東北大学の BERT (Apache 2.0)。数値は A5000 で float16 の経路（`BF16 = False`、T4 と同じ dtype）で実行したもので、GPU メモリの最大使用量は 4.9 GB だった。追加学習したモデルは `bert-kyukana` に保存される。同じランタイムなら、fill-mask の `model=` や疑似対数尤度の `name` に `"bert-kyukana"` を渡せば使える。別のノートブックで使うなら、この章の「モデルを保存・読み込みする、Hub にアップロードする」のように Google ドライブに保存するか Hub に上げる。

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
- 最小対は検証用の記事の文から機械的に作ったもので、人手で確かめていない。「が」と「を」を入れ替えても文法的なままの組や、1 組の差が助詞以外の要因で決まる組も混ざりうる。系統的に調べるなら、この章の「言語モデルの文法知識を最小対で調べる (BLiMP / JBLiMP)」のような人手で作った最小対を使う。生成文は、学習前はでたらめなトークンの列、200 ステップと 1,000 ステップでは記号や断片が並ぶだけで、5,000 ステップで「日本の首都は、東京都千代田区にある。」のあとに「概要」のような Wikipedia の見出しが続く文章になった（各チェックポイントで 1 回ずつサンプリングしただけなので、傾向として見る程度にとどめる）。
- 13.8M パラメータで、GPU メモリの最大使用量は 2.6 GB。数値と生成文は、検証のマシン（RTX A5000、T4 と同じ float16 の経路）での 1 回の結果で、学習のセルは 297 秒（うち `trainer.train()` は 285 秒）だった。T4 はこれより遅い。GPU が違うと計算の細部が変わり、生成文などは別のものになりうる。版をそろえるなら `!pip install -q transformers==5.22.1 datasets==4.8.5` のように固定する（検証は transformers 5.22.1、datasets 4.8.5、tokenizers 0.23.3、torch 2.14.1）。
- データは日本語版 Wikipedia（wikimedia/wikipedia の 20231101.ja、データセットカードによれば CC BY-SA 3.0 と GFDL）と青空文庫（globis-university/aozorabunko-clean、CC BY 4.0）。ライブラリは transformers、datasets、tokenizers が Apache 2.0、fugashi が MIT と BSD 3 条項、unidic-lite がパッケージは MIT で、同梱の UniDic 2.1.2 の辞書は GPL、LGPL、BSD のいずれかを選べる（パッケージに同梱の COPYING による）。トークナイザーの学習は、サブワードとトークナイザの章の「Hugging Face tokenizers で BPE / WordPiece を学習する」を参照。仕組みを自分で実装して確かめる版は、付録「仕組みを理解するための実装」の「小さな GPT を日本語のテキストでゼロから事前学習する（学習ループを自分で書く）」にある。

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
- この 96 件では、GPU で 1 件ずつ 0.2〜0.3 秒 → バッチ + 長さでソートで 0.04 秒（6〜8 倍）。件数が多いほど、またモデルが大きいほど差は広がる。さらに `model.half()`（半精度）で速度とメモリが改善する。
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

- プローブ (probe) は、モデルの内部表現を入力にした簡単な分類器で、その情報が表現から取り出せるかを調べる（線形のプローブは Alain & Bengio, 2016。BERT の層ごとの分析は Tenney et al., 2019 など）。単語ベクトルの取り出し方はこの章の「BERT から文脈に応じた単語ベクトルを取り出す」と同じく `output_hidden_states=True` で、層 0 は文脈を見ない埋め込み層の出力（位置の情報は含む）。
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
- nnsight（Fiotto-Kaufman ら, 2024。MIT ライセンス）は、そのような介入にも使えるライブラリで、`layer.output` を読むだけでなく書き換えることもできる。nnsight 0.8 では `LanguageModel` は非推奨で `TransformersModel` を使う。transformers 5 では層の出力がテンソルなので、古い資料の `layer.output[0]` は不要になる。
- モデルは SB Intuitions の Sarashina2.2-0.5B（MIT ライセンス。日本語・英語・コードのウェブのデータなど 10 兆トークンで学習したベースモデル）、tuned lens の学習データは日本語版 Wikipedia（CC BY-SA 3.0 と GFDL）。トークナイザーの修正は、大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」の注意書きを参照。T4 と同じ float16 の経路で実行し、GPU メモリの最大は 5.6 GB だった。
