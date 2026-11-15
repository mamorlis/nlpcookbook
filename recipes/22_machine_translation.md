# 機械翻訳

翻訳モデルを動かす、対訳データを用意する、モデルを学習・微調整する、訳文を評価する。

## 対訳コーパスを取得して中身を確認する {parallel_corpus}

機械翻訳の実験用に、日英の対訳データを手に入れて、品質や長さの分布を確認したい。

```python
!pip install -q datasets
```

```python
from datasets import load_dataset

ds = load_dataset("Helsinki-NLP/opus-100", "en-ja")     # OPUS から抽出した 100 言語の対訳 (日英は 100 万文対)
print(ds)
for ex in ds["train"].select(range(5)):
    print(ex["translation"]["ja"], "|", ex["translation"]["en"])
```

```python
import pandas as pd

sample = ds["train"].shuffle(seed=0).select(range(20000))
df = pd.DataFrame(sample["translation"])
df["len_ja"], df["len_en"] = df["ja"].str.len(), df["en"].str.split().str.len()
df["ratio"] = df["len_ja"] / df["len_en"].clip(lower=1)
print(df[["len_ja", "len_en", "ratio"]].describe().round(1))

# 対訳として怪しいものを機械的に除く: 空、長すぎ、長さの比が極端、原文と訳文が同じ、日本語を含まない
ok = ((df["len_ja"] > 0) & (df["len_en"] > 0) & (df["len_en"] <= 100) & df["ratio"].between(0.5, 8)
      & (df["ja"] != df["en"]) & df["ja"].str.contains(r"[ぁ-んァ-ヶ一-龥]"))
print(f"残る割合: {ok.mean():.1%}")
print(df[~ok].sample(5, random_state=0)[["ja", "en"]].to_string(index=False))
```

- OPUS の日英は映画字幕 (OpenSubtitles) 由来が多く、口語的で短く、ノイズや意訳も多い。データの出どころと分野を必ず確認する。
- 日英の定番データ: JParaCrawl (Web、大規模)、ASPEC (科学技術論文、要申請)、KFTT (京都関連の Wikipedia)、JESC (字幕)、WMT の news タスク。
- 長さの比によるフィルタは最低限の処理。さらに、言語判定、文埋め込みの類似度 (LaBSE など) によるフィルタが有効。

## 多言語翻訳モデル NLLB で翻訳する {translation_nllb}

日本語と英語、あるいは 200 言語の任意の組み合わせで機械翻訳をしたい。API を使わず手元で動かしたい。

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSeq2SeqLM.from_pretrained(name).eval()

def translate(texts, src, tgt, num_beams=4):
    tokenizer.src_lang = src
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        out = model.generate(**batch, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt),
                             num_beams=num_beams, max_new_tokens=256)
    return tokenizer.batch_decode(out, skip_special_tokens=True)

ja = ["吾輩は猫である。名前はまだ無い。", "機械翻訳の品質は、ここ数年で大きく向上した。"]
for s, t in zip(ja, translate(ja, "jpn_Jpan", "eng_Latn")):
    print(s, "→", t)
```

```python
en = ["Natural language processing is a subfield of artificial intelligence.", "Could you tell me how to get to the station?"]
for tgt in ["jpn_Jpan", "zho_Hans", "kor_Hang", "fra_Latn"]:
    print(tgt, translate(en, "eng_Latn", tgt))
```

- 言語は FLORES-200 のコード (`jpn_Jpan`、`eng_Latn`、`zho_Hans` など) で指定する。出力言語は、デコーダの最初のトークンを強制することで決まる。
- 文単位で学習されたモデルなので、長い文章は文に分割してから 1 文ずつ訳す。
- 精度を上げるなら `facebook/nllb-200-1.3B` や `nllb-200-3.3B` (GPU 推奨)。NLLB のライセンスは CC-BY-NC (非商用) である点に注意。
- 日英・英日に特化するなら、LLM に翻訳させるほうが流暢なことも多い (LLM の章を参照)。

## Transformer の翻訳モデルをゼロから学習する {nmt_from_scratch gpu norun}

対訳データだけから、ニューラル機械翻訳モデルを自分で学習したい。サブワード分割、モデルの定義、学習、ビームサーチ、BLEU による評価までを一通り行う。

```python
!pip install -q datasets sentencepiece sacrebleu
```

```python
import random
import re
from datasets import load_dataset

ds = load_dataset("Helsinki-NLP/opus-100", "en-ja")

def clean(pairs, max_len=60):
    out = []
    for p in pairs:
        ja, en = p["ja"].strip(), p["en"].strip()
        if not (0 < len(ja) <= max_len and 0 < len(en.split()) <= 30):
            continue
        if not re.search(r"[ぁ-んァ-ヶ一-龥]", ja) or re.search(r"[ぁ-んァ-ヶ一-龥]", en):   # 日本語側に日本語がない・英語側に日本語が混ざる対を除く
            continue
        out.append((ja, en))
    return out

random.seed(0)
train = clean(ds["train"]["translation"][:300000])
valid = clean(ds["validation"]["translation"])[:500]
test = clean(ds["test"]["translation"])[:1000]
print(len(train), len(valid), len(test), train[0])
```

```python
import sentencepiece as spm

with open("train.txt", "w", encoding="utf-8") as f:
    for ja, en in train:
        f.write(ja + "\n" + en + "\n")
# 日英で語彙を共有する 1 つのサブワードモデル。pad=0, unk=1, bos=2, eos=3
spm.SentencePieceTrainer.train(input="train.txt", model_prefix="spm", vocab_size=16000, character_coverage=0.9995,
                               pad_id=0, unk_id=1, bos_id=2, eos_id=3, input_sentence_size=600000, shuffle_input_sentence=True, minloglevel=2)
sp = spm.SentencePieceProcessor(model_file="spm.model")
print(sp.encode("吾輩は猫である。", out_type=str), sp.encode("I am a cat.", out_type=str))

MAX_TOKENS = 64
def encode(pairs):
    return [(sp.encode(ja)[:MAX_TOKENS - 1] + [3], sp.encode(en)[:MAX_TOKENS - 1] + [3]) for ja, en in pairs]   # 末尾に EOS
train_ids, valid_ids = encode(train), encode(valid)
```

```python
import math
import torch
from transformers import MarianConfig, MarianMTModel

device = "cuda" if torch.cuda.is_available() else "cpu"
config = MarianConfig(
    vocab_size=16000, d_model=256, encoder_layers=4, decoder_layers=4, encoder_attention_heads=4, decoder_attention_heads=4,
    encoder_ffn_dim=1024, decoder_ffn_dim=1024, dropout=0.1, max_position_embeddings=128,
    pad_token_id=0, eos_token_id=3, decoder_start_token_id=0, share_encoder_decoder_embeddings=True,
)
model = MarianMTModel(config).to(device)          # 事前学習なし。重みはランダムに初期化される
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M パラメータ")

def batches(data, batch_size, shuffle):
    order = sorted(range(len(data)), key=lambda i: len(data[i][0]))      # 長さの近い文をまとめてパディングを減らす
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        random.shuffle(chunks)
    for chunk in chunks:
        src = torch.nn.utils.rnn.pad_sequence([torch.tensor(data[i][0]) for i in chunk], batch_first=True, padding_value=0)
        tgt = torch.nn.utils.rnn.pad_sequence([torch.tensor(data[i][1]) for i in chunk], batch_first=True, padding_value=-100)   # -100 は損失で無視される
        yield src.to(device), tgt.to(device)
```

```python
EPOCHS, BATCH, PEAK_LR, WARMUP = 6, 128, 7e-4, 2000
optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.98), weight_decay=0.01)
schedule = lambda step: min((step + 1) / WARMUP, math.sqrt(WARMUP / (step + 1)))       # 線形に上げてから、ステップ数の平方根に反比例して下げる
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
scaler = torch.amp.GradScaler(enabled=device == "cuda")
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

step = 0
for epoch in range(EPOCHS):
    model.train()
    for src, tgt in batches(train_ids, BATCH, shuffle=True):
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            logits = model(input_ids=src, attention_mask=src != 0, labels=tgt).logits      # labels を渡すと、右に 1 つずらしたものがデコーダの入力になる
            loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        step += 1
        if step % 500 == 0:
            print(f"epoch {epoch + 1} step {step} loss {loss.item():.3f} lr {scheduler.get_last_lr()[0]:.2e}")
    model.eval()
    with torch.no_grad():
        val = [criterion(model(input_ids=s, attention_mask=s != 0, labels=t).logits.flatten(0, 1), t.flatten()).item() for s, t in batches(valid_ids, 256, False)]
    print(f"== epoch {epoch + 1}: 検証損失 {sum(val) / len(val):.3f}")
```

```python
import sacrebleu

@torch.no_grad()
def translate(sentences, num_beams=4, batch_size=64):
    model.eval()
    outputs = []
    for i in range(0, len(sentences), batch_size):
        ids = [sp.encode(s)[:MAX_TOKENS - 1] + [3] for s in sentences[i:i + batch_size]]
        src = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in ids], batch_first=True, padding_value=0).to(device)
        out = model.generate(input_ids=src, attention_mask=src != 0, num_beams=num_beams, max_new_tokens=MAX_TOKENS)
        outputs += [sp.decode([t for t in row.tolist() if t > 3]) for row in out]
    return outputs

hyps = translate([ja for ja, _ in test])
print(sacrebleu.corpus_bleu(hyps, [[en for _, en in test]]))
for (ja, en), hyp in list(zip(test, hyps))[:6]:
    print(f"{ja}\n   訳: {hyp}\n   参照: {en}")
```

- 流れは「対訳データの掃除 → サブワード分割 → モデルの定義 → 学習 → ビームサーチで翻訳 → BLEU で評価」。事前学習済みモデルを使わなくても、数十万文対あれば翻訳らしいものが出てくる。
- T4 GPU で 6 エポックが 15〜20 分程度。モデルは約 2,000 万パラメータの小さな Transformer で、字幕のような短い文なら BLEU は 1 桁後半〜10 台前半が目安 (この数値は Colab での実測ではない)。
- 学習率のウォームアップ (最初は小さく、徐々に上げる) がないと、Transformer の学習は発散しやすい。ラベル平滑化と勾配クリッピングも定番の設定。
- 検証損失が下がらなくなったら学習を止める。データが少ないのに層を増やしても、過学習するだけで良くならない。
- 本格的に学習するなら、データを増やし (JParaCrawl など数千万文対)、逆翻訳で単言語データを活用し、fairseq や Marian、OpenNMT などの専用ツールを使う。

## 学習済みの翻訳モデルを自分の分野のデータでファインチューニングする {mt_finetune gpu norun}

公開されている翻訳モデルを、自分の分野 (ここでは映画やドラマの字幕) の対訳データで追加学習して、その分野での訳質を上げたい。

```python
!pip install -q datasets sentencepiece sacremoses sacrebleu accelerate
```

```python
import re
from datasets import Dataset, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "Helsinki-NLP/opus-mt-ja-en"        # 日英の Marian モデル (約 300MB)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

ds = load_dataset("Helsinki-NLP/opus-100", "en-ja")
def clean(pairs):
    return [p for p in pairs if 0 < len(p["ja"]) <= 80 and 0 < len(p["en"].split()) <= 40 and re.search(r"[ぁ-んァ-ヶ一-龥]", p["ja"])]

train = Dataset.from_list(clean(ds["train"]["translation"][:60000])).shuffle(seed=0).select(range(30000))
test = Dataset.from_list(clean(ds["test"]["translation"])).select(range(500))

def preprocess(batch):
    return tokenizer(batch["ja"], text_target=batch["en"], truncation=True, max_length=64)   # text_target が訳文側 (labels になる)

train_enc = train.map(preprocess, batched=True, remove_columns=["ja", "en"])
test_enc = test.map(preprocess, batched=True, remove_columns=["ja", "en"])
```

```python
import sacrebleu
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def translate(model, sentences, batch_size=64):
    model.to(device).eval()
    out = []
    for i in range(0, len(sentences), batch_size):
        batch = tokenizer(sentences[i:i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        out += tokenizer.batch_decode(model.generate(**batch, num_beams=4, max_new_tokens=64), skip_special_tokens=True)
    return out

before = translate(model, test["ja"])
print("微調整前:", sacrebleu.corpus_bleu(before, [test["en"]]))
```

```python
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    output_dir="mt-finetuned", num_train_epochs=1, learning_rate=3e-5, warmup_steps=100, weight_decay=0.01,
    per_device_train_batch_size=32, fp16=True, logging_steps=100, save_strategy="no", report_to="none", seed=0,
)
trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_enc, data_collator=DataCollatorForSeq2Seq(tokenizer, model=model))
trainer.train()
```

```python
after = translate(model, test["ja"])
print("微調整前:", sacrebleu.corpus_bleu(before, [test["en"]]))
print("微調整後:", sacrebleu.corpus_bleu(after, [test["en"]]))
for ja, en, b, a in list(zip(test["ja"], test["en"], before, after))[:6]:
    print(f"{ja}\n   前: {b}\n   後: {a}\n   参照: {en}")

model.save_pretrained("mt-finetuned"); tokenizer.save_pretrained("mt-finetuned")
```

- ゼロから学習するよりはるかに少ないデータと時間で、特定の分野の文体や用語に合わせられる。T4 GPU で 3 万文対・1 エポックが 5 分程度。
- 学習率は小さめ (1e-5〜5e-5) にする。大きすぎると、元のモデルが持っていた一般的な翻訳能力が壊れる (破滅的忘却)。
- 微調整後は、対象の分野だけでなく、一般的な文のテストセットでも評価する。分野内で上がって分野外で下がるのが典型的な副作用。
- BLEU の差が小さいときは、評価の章の検定 (ブートストラップ) で有意かどうかを確かめる。訳文を実際に見比べることも欠かせない。
- NLLB のような多言語モデルや、LLM を LoRA で微調整する方法 (LLM の章) でも同じ考え方が使える。

## 言語をまたいで意味の近い文を対応付ける (対訳文の抽出) {cross_lingual_similarity}

日本語と英語の文集合から、互いに翻訳になっている文のペアを見つけたい (対訳コーパスの構築)。

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/LaBSE")

ja = ["猫がソファの上で寝ている。", "明日は雨が降るでしょう。", "この論文は新しい翻訳手法を提案する。", "駅までの道を教えてください。", "私はコーヒーより紅茶が好きだ。"]
en = ["It will probably rain tomorrow.", "I prefer tea to coffee.", "Could you tell me the way to the station?",
      "This paper proposes a novel translation method.", "The stock market crashed.", "A cat is sleeping on the sofa."]

sim = model.similarity(model.encode(ja, normalize_embeddings=True), model.encode(en, normalize_embeddings=True)).numpy()
for i, row in enumerate(sim):
    j = row.argmax()
    print(f"{row[j]:.2f}  {ja[i]}  ⇔  {en[j]}")
```

```python
# 相互最近傍 (双方向で互いに 1 位) かつしきい値以上のペアだけを採用すると、誤対応が減る
best_en, best_ja = sim.argmax(axis=1), sim.argmax(axis=0)
pairs = [(ja[i], en[j], sim[i, j]) for i, j in enumerate(best_en) if best_ja[j] == i and sim[i, j] > 0.6]
print(len(pairs), "ペア")
```

- LaBSE は 109 言語の対訳文で学習された、対訳文検索向けの文埋め込みモデル。
- 大規模な対訳抽出では、近傍との類似度の比を使う margin-based スコアリングが標準的。

## 機械翻訳を BLEU で評価する (sacreBLEU) {bleu}

翻訳システムの出力を、参照訳との n-gram の一致度 (BLEU) で評価したい。日本語の出力も正しく評価したい。

```python
!pip install -q "sacrebleu[ja]"
```

```python
from sacrebleu.metrics import BLEU

hyps = ["猫がマットの上に座っている。", "彼は昨日東京へ行った。", "この本はとても面白いです。"]
refs = ["猫はマットの上に座っています。", "彼は昨日、東京に行きました。", "この本は非常に面白い。"]

bleu = BLEU(tokenize="ja-mecab")           # 日本語は MeCab で分割してから n-gram を数える
result = bleu.corpus_score(hyps, [refs])   # 参照は「参照セットのリスト」で渡す (複数参照に対応するため)
print(result)
print(bleu.get_signature())                # 論文にはこの署名を載せる
```

```python
# トークナイズの違いでスコアは大きく変わる
for tok in ["ja-mecab", "char", "13a"]:
    print(f"{tok:10} {BLEU(tokenize=tok).corpus_score(hyps, [refs]).score:.2f}")

# 英語 (デフォルトの 13a トークナイザ)。参照訳が 2 セットある場合
hyps_en = ["The cat sat on the mat.", "He went to Tokyo yesterday."]
refs_en = [["The cat is sitting on the mat.", "He went to Tokyo yesterday."], ["A cat sat on the mat.", "Yesterday he traveled to Tokyo."]]
print(BLEU().corpus_score(hyps_en, refs_en))
```

```python
# 文単位の BLEU (平滑化が必要。コーパス単位の値とは性質が違う)
sent_bleu = BLEU(tokenize="ja-mecab", effective_order=True)
for h, r in zip(hyps, refs):
    print(f"{sent_bleu.sentence_score(h, [r]).score:5.1f}  {h}")
```

- BLEU は、モデルの出力をデトークナイズした「普通の文」に対して計算する。自前で分かち書きした文字列を渡してはいけない。
- 日本語を既定の `13a` で評価すると、文全体が 1〜数トークンになり、ほぼ意味のない値になる。必ず `ja-mecab` を指定する。
- BLEU はコーパス全体の n-gram を集計する指標。文ごとの BLEU の平均とは一致しない。
- 異なるトークナイズ・異なるテストセットの BLEU 同士は比較できない。署名を必ず報告する。

## chrF と TER で翻訳を評価する {chrf_ter}

BLEU 以外の自動評価指標も併せて報告したい。単語分割に依存しない指標が欲しい。

```python
!pip install -q "sacrebleu[ja]"
```

```python
from sacrebleu.metrics import BLEU, CHRF, TER

hyps = ["猫がマットの上に座っている。", "彼は昨日東京へ行った。", "この本はとても面白いです。"]
refs = ["猫はマットの上に座っています。", "彼は昨日、東京に行きました。", "この本は非常に面白い。"]

chrf = CHRF()                 # 文字 n-gram の F スコア (既定は 6-gram まで、β=2)
chrfpp = CHRF(word_order=2)   # chrF++: 単語 1-gram, 2-gram も加える
ter = TER(asian_support=True, normalized=True)   # 翻訳編集率: 参照に直すための編集回数 (低いほど良い)

print(chrf.corpus_score(hyps, [refs]), "|", chrf.get_signature())
print(chrfpp.corpus_score(hyps, [refs]))
print(ter.corpus_score(hyps, [refs]))
```

```python
# 形態が豊かな言語や表記ゆれでは、chrF のほうが BLEU より人手評価との相関が高い傾向がある
hyp_en, ref_en = ["He goes to schools every days."], ["He goes to school every day."]
print(BLEU().corpus_score(hyp_en, [ref_en]).score, CHRF().corpus_score(hyp_en, [ref_en]).score)
```

- chrF は文字単位なので、日本語でも単語分割の方法に左右されない。語尾だけ違う語にも部分点が入る。
- いずれも表層の一致を見る指標で、言い換えは評価できない。意味を見るには BERTScore や COMET などの学習ベースの指標を併用する。

## COMET で翻訳の品質を評価する {comet gpu norun}

機械翻訳の評価で現在標準的に使われている、人手評価に基づいて学習されたニューラル評価指標を使いたい。

```python
!pip install -q unbabel-comet
```

```python
from comet import download_model, load_from_checkpoint

model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))   # 原文・訳文・参照訳を使うモデル (約 2.3GB)

data = [
    {"src": "吾輩は猫である。名前はまだ無い。", "mt": "I am a cat. I don't have a name yet.", "ref": "I am a cat. As yet I have no name."},
    {"src": "吾輩は猫である。名前はまだ無い。", "mt": "I am a dog. My name is not here.", "ref": "I am a cat. As yet I have no name."},
    {"src": "彼は昨日東京に行った。", "mt": "He went to Tokyo yesterday.", "ref": "Yesterday he went to Tokyo."},
]
out = model.predict(data, batch_size=8, gpus=1)
for d, s in zip(data, out.scores):
    print(f"{s:.3f}  {d['mt']}")
print("システムスコア:", round(out.system_score, 4))
```

```python
# 参照訳なしで評価する品質推定 (QE) モデル。利用規約への同意と Hugging Face へのログインが必要
# from huggingface_hub import login; login()
# qe = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))
# print(qe.predict([{"src": "彼は昨日東京に行った。", "mt": "He went to Tokyo yesterday."}], batch_size=8, gpus=1).scores)
```

- BLEU と違い、原文も入力に使う。意味が合っていれば表現が違っても高く評価され、人手評価との相関が高い。
- スコアはモデルごとに尺度が違う。異なる COMET モデルの値を比べてはいけない。モデル名を必ず報告する。
- Colab では、インストール時に依存ライブラリのバージョンが入れ替わることがある。エラーが出たらランタイムを再起動する。
