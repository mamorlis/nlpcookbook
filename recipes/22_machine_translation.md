# 機械翻訳

翻訳モデルを動かす、対訳データを用意する、訳文を評価する。

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
