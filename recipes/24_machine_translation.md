# 機械翻訳

対訳データを用意する、翻訳モデルを動かす、訳文を評価する、モデルを学習・微調整する、訳し方を制御する。

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
!pip install -q unbabel-comet "setuptools<81"   # comet が使う torchmetrics は、新しい setuptools で削除された pkg_resources を読み込む
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

## 参照訳なしで翻訳の品質を推定する (品質推定) {mt_quality_estimation}

正解の訳 (参照訳) がない状況で、機械翻訳の出力がどれくらい良いかを見積もりたい。

```python
!pip install -q datasets sentence-transformers
```

```python
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.stats import kendalltau, spearmanr

# WMT 2022 の人手評価 (原文、機械訳、参照訳、人が付けた品質スコア)。英日翻訳の部分を使う
df = load_dataset("RicardoRei/wmt-sqm-human-evaluation", split="train").to_pandas()
df = df[(df["lp"] == "en-ja") & (df["src"].str.len().between(20, 200))].sample(600, random_state=0).reset_index(drop=True)
print(len(df), "件 / システム数", df["system"].nunique(), "/ スコアの分布", df["score"].describe().round(2).to_dict())
for r in df.sort_values("score").iloc[[0, 300, -1]].itertuples():
    print(f"[{r.score:.2f}] {r.src[:60]}\n        → {r.mt[:60]}")
```

```python
from sentence_transformers import SentenceTransformer

# 方法 1: 原文と訳文の、多言語埋め込みによる類似度
embedder = SentenceTransformer("intfloat/multilingual-e5-small")
e_src = embedder.encode(["query: " + s for s in df["src"]], normalize_embeddings=True, batch_size=32)
e_mt = embedder.encode(["query: " + s for s in df["mt"]], normalize_embeddings=True, batch_size=32)
df["埋め込みの類似度"] = (e_src * e_mt).sum(axis=1)

# 方法 2: 長さの比 (極端に短い・長い訳は怪しい、という素朴なベースライン)
ratio = df["mt"].str.len() / df["src"].str.len()
df["長さの比"] = -np.abs(np.log(ratio / ratio.median()))
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 方法 3: LLM に「正確で自然な訳か」を尋ね、「はい」と答える確率をスコアにする
name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
llm = AutoModelForCausalLM.from_pretrained(name).eval()
YES_NO = [tokenizer(c, add_special_tokens=False)["input_ids"][0] for c in ["はい", "いいえ"]]

@torch.no_grad()
def p_good(src, mt):
    prompt = ("次の「英語の原文」を日本語に翻訳した「訳文」を評価してください。原文の意味を過不足なく伝えていて、日本語として自然なら「はい」、"
              f"誤訳・訳抜け・不自然な表現があれば「いいえ」と、一語で答えてください。\n\n英語の原文: {src}\n訳文: {mt}")
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    return float(llm(ids).logits[0, -1, YES_NO].softmax(-1)[0])

sub = df.head(150).copy()                      # LLM は 1 件あたり数秒かかるので、150 件だけで比べる
sub["LLM (はいの確率)"] = sub["llm"] = [p_good(s, m) for s, m in zip(sub["src"], sub["mt"])]
```

```python
rows = []
for name in ["長さの比", "埋め込みの類似度", "LLM (はいの確率)"]:
    rows.append({"方法": name, "Spearman": spearmanr(sub["score"], sub[name]).statistic, "Kendall": kendalltau(sub["score"], sub[name]).statistic})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("600 件での埋め込みの類似度:", round(spearmanr(df["score"], df["埋め込みの類似度"]).statistic, 3))

# LLM が低く、人が高く評価した例 / その逆
sub["差"] = sub["LLM (はいの確率)"].rank() - sub["score"].rank()
for r in pd.concat([sub.sort_values("差").head(2), sub.sort_values("差").tail(2)]).itertuples():
    print(f"人 {r.score:.0f} / LLM {r.llm:.2f} | {r.src[:50]} → {r.mt[:50]}")
```

```python
# はっきり壊れた訳なら見つけられるか: 別の文の訳にすり替えたもの、後半を切り落としたものを混ぜて、元の訳と区別できるかを調べる
from sklearn.metrics import roc_auc_score

good = sub.head(60)
swapped = good["mt"].shift(1, fill_value=good["mt"].iloc[-1]).tolist()          # 別の原文の訳
truncated = [m[:len(m) // 2] for m in good["mt"]]                               # 訳抜け
def embed_score(srcs, mts):
    a = embedder.encode(["query: " + s for s in srcs], normalize_embeddings=True)
    b = embedder.encode(["query: " + s for s in mts], normalize_embeddings=True)
    return (a * b).sum(axis=1)

for label, corrupted in [("別の文の訳にすり替え", swapped), ("後半を切り落とす", truncated)]:
    y = [1] * len(good) + [0] * len(good)
    emb = np.concatenate([embed_score(good["src"], good["mt"]), embed_score(good["src"], corrupted)])
    llm_s = list(good["llm"]) + [p_good(s, m) for s, m in zip(good["src"], corrupted)]
    print(f"{label}: 元の訳と区別できる度合い (AUC) 埋め込み {roc_auc_score(y, emb):.3f} / LLM {roc_auc_score(y, llm_s):.3f}")
```

- 品質推定 (quality estimation, QE) は、参照訳を使わずに原文と訳文だけから品質を予測する。翻訳結果をそのまま公開してよいか、人が後編集すべきかを振り分けるのに使われる。
- この評価データは、最近の高品質な翻訳システムの出力で、ほとんどが 80 点以上。その中の細かい優劣を当てるのは非常に難しく、ここで試した 3 つの方法は、どれも人の評価とほとんど相関しなかった。
- 一方、別の文の訳や訳抜けのような、はっきり壊れた訳は区別できる。簡易な方法は「明らかな失敗をはじく」用途に使い、細かい品質の順位付けには使わない。
- 大きな LLM を使った判定は、専用の品質推定モデルに匹敵することが報告されている。小さなモデルで同じ結果は期待できない。
- 専用のモデルとしては、人手評価で学習された CometKiwi が代表的 (利用にはライセンスへの同意が必要)。大量の文を安定して評価するなら、まずこれを検討する。
- 文単位の人手評価はそもそも雑音が大きい。相関係数は 0.3〜0.5 程度でも「使える」とされることが多く、高い値を期待しすぎない。

## 文埋め込みから翻訳の評価指標を自分で学習する {mt_metric_regression}

BLEU のような文字列の一致ではなく、意味の近さで翻訳を評価したい。人手評価のデータから、評価指標そのものを学習する。

```python
!pip install -q datasets sentence-transformers "sacrebleu[ja]"
```

```python
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.stats import kendalltau

df = load_dataset("RicardoRei/wmt-sqm-human-evaluation", split="train").to_pandas()
df = df[(df["lp"] == "en-ja") & (df["mt"].str.len().between(10, 150))].sample(6000, random_state=0).reset_index(drop=True)

# 同じ原文の訳が学習と評価の両方に入らないよう、原文単位で分ける
sources = df["src"].drop_duplicates().sample(frac=1, random_state=0)
test_src = set(sources[:len(sources) // 5])
train, test = df[~df["src"].isin(test_src)].reset_index(drop=True), df[df["src"].isin(test_src)].reset_index(drop=True)
print(len(train), "件で学習 /", len(test), "件で評価")
```

```python
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("intfloat/multilingual-e5-small")
encode = lambda texts: embedder.encode(["query: " + t for t in texts], normalize_embeddings=True, batch_size=64)

def features(d):
    """訳文 h と参照訳 r の埋め込みから、[h, r, |h − r|, h × r] を特徴量にする"""
    h, r = encode(d["mt"].tolist()), encode(d["ref"].tolist())
    return np.hstack([h, r, np.abs(h - r), h * r]), (h * r).sum(axis=1)

X_train, _ = features(train)
X_test, cos_test = features(test)
print(X_train.shape)
```

```python
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# 人手のスコアを予測する回帰モデル (RBF カーネルのサポートベクター回帰)。線形の回帰とも比べる
regressor = make_pipeline(StandardScaler(), SVR(C=10, epsilon=1.0)).fit(X_train, train["score"])
learned = regressor.predict(X_test)
linear = make_pipeline(StandardScaler(), Ridge(alpha=100.0)).fit(X_train, train["score"]).predict(X_test)
```

```python
import sacrebleu

# 比較対象: 文字列の一致に基づく指標と、学習なしの埋め込み類似度
bleu = [sacrebleu.sentence_bleu(h, [r], tokenize="ja-mecab").score for h, r in zip(test["mt"], test["ref"])]
chrf = [sacrebleu.sentence_chrf(h, [r]).score for h, r in zip(test["mt"], test["ref"])]

rows = [{"指標": name, "Kendall τ": kendalltau(test["score"], s).statistic, "Pearson": np.corrcoef(test["score"], s)[0, 1]}
        for name, s in [("文単位の BLEU", bleu), ("chrF", chrf), ("埋め込みのコサイン類似度 (学習なし)", cos_test), ("学習した回帰モデル (線形)", linear), ("学習した回帰モデル (SVR)", learned)]]
print(pd.DataFrame(rows).round(3).to_string(index=False))

# 文字列はあまり一致しないが、学習した指標が高く評価した例
test["chrF"], test["learned"] = chrf, learned
for r in test[test["chrF"] < 25].sort_values("learned", ascending=False).head(3).itertuples():
    print(f"人 {r.score:.2f} / chrF {r.chrF:.0f} / 学習 {r.learned:.2f}\n   訳: {r.mt[:70]}\n   参照: {r.ref[:70]}")
```

- 訳文と参照訳をそれぞれ文埋め込みにし、その組み合わせから人手のスコアを回帰する。RUSE という指標の考え方で、のちの BLEURT や COMET (学習型の評価指標) につながった。
- 特徴量に差の絶対値と要素ごとの積を入れるのは、2 つの文の関係を学習するときの定番 (自然言語推論のモデルと同じ)。
- 回帰モデルの選び方で結果が大きく変わる。この実験では、線形の回帰は学習なしのコサイン類似度にも負け、非線形の SVR にして初めて上回った。高次元の特徴量に対して学習データが数千件しかないので、過学習しやすい。
- 相関の絶対値は低い (Kendall の τ で 0.2 前後)。最近の翻訳システムの出力は質が高くて差が小さく、文単位の人手評価の雑音も大きいため。
- 学習型の指標は、学習データと同じ言語対・同じ年の評価では強いが、分野や翻訳システムの種類が変わると当てにならなくなることがある。原文単位で分割して、見たことのない文で評価する。
- 文単位の BLEU は人手評価との相関が低い。文単位で比べたいときは、chrF か学習型の指標を使う。
- 実用では、自分で学習するより公開されている COMET を使う (この章の COMET のレシピ)。ここでの目的は、学習型の指標が何をしているかを理解すること。

## Moses で統計的機械翻訳システムを作る {moses_smt}

ニューラル以前の主流だったフレーズベース統計的機械翻訳 (SMT) を、対訳コーパスから自分で組み立てたい。単語アライメント、フレーズテーブル、言語モデル、デコードまでを一通り動かす。

```python
# Moses 本体と、単語アライメントの fast_align をビルドする (Colab で 15 分ほどかかる)
!apt-get install -y -q build-essential cmake libboost-all-dev zlib1g-dev libbz2-dev liblzma-dev > /dev/null 2>&1
!git clone -q --depth 1 https://github.com/moses-smt/mosesdecoder
!cd mosesdecoder && ./bjam -j4 -q > ../bjam.log 2>&1; tail -n 1 ../bjam.log
!git clone -q --depth 1 https://github.com/clab/fast_align
!mkdir -p fast_align/build && cd fast_align/build && cmake .. > /dev/null 2>&1 && make -j4 > /dev/null 2>&1
!ls mosesdecoder/bin/moses mosesdecoder/bin/lmplz fast_align/build/fast_align fast_align/build/atools
```

```python
!pip install -q datasets fugashi unidic-lite sacremoses sacrebleu
```

```python
import os
from datasets import load_dataset
from fugashi import Tagger
from sacremoses import MosesTokenizer

os.makedirs("data", exist_ok=True)
ds = load_dataset("Helsinki-NLP/opus-100", "en-ja")       # 映画・ドラマの字幕が中心の対訳データ
tagger, mt = Tagger("-Owakati"), MosesTokenizer(lang="en")

def tok_ja(s):
    s = s.translate(str.maketrans("|<>[]　", "｜＜＞［］ "))     # Moses は | < > [ ] を特別な文字として扱うので、全角に置き換える
    return tagger.parse(s).strip()

def tok_en(s):
    return mt.tokenize(s.lower(), escape=True, return_str=True)    # 小文字にそろえて単語に分ける

def write(split, n, prefix, max_len=40):
    count = 0
    with open(f"data/{prefix}.ja", "w") as f_ja, open(f"data/{prefix}.en", "w") as f_en, open(f"data/{prefix}.raw.en", "w") as f_raw:
        for ex in ds[split]:
            ja, en = ex["translation"]["ja"].strip(), ex["translation"]["en"].strip()
            if not ja or not en or len(ja) > 60:
                continue
            tj, te = tok_ja(ja), tok_en(en)
            if not (1 <= len(tj.split()) <= max_len and 1 <= len(te.split()) <= max_len):     # 長すぎる文はアライメントの質を下げるので除く
                continue
            f_ja.write(tj + "\n"); f_en.write(te + "\n"); f_raw.write(en.lower() + "\n")
            count += 1
            if count >= n:
                break
    print(prefix, count, "文対")

write("train", 50000, "train")
write("test", 500, "test")
!head -n 3 data/train.ja data/train.en
```

```python
# 1. 単語アライメント: 日→英と英→日の両方向で求め、grow-diag-final-and で 1 つにまとめる
!paste -d '\t' data/train.ja data/train.en | sed 's/\t/ ||| /' > data/train.ja-en
!fast_align/build/fast_align -i data/train.ja-en -d -o -v > data/fwd.align 2> /dev/null
!fast_align/build/fast_align -i data/train.ja-en -d -o -v -r > data/rev.align 2> /dev/null
!fast_align/build/atools -i data/fwd.align -j data/rev.align -c grow-diag-final-and > data/aligned.grow-diag-final-and
!head -n 3 data/aligned.grow-diag-final-and

# 2. 言語モデル: 訳文側 (英語) の 3-gram を KenLM で学習する
!mosesdecoder/bin/lmplz -o 3 --discount_fallback < data/train.en > data/lm.arpa 2> /dev/null
!mosesdecoder/bin/build_binary data/lm.arpa data/lm.blm > /dev/null 2>&1
```

```python
# 3. 翻訳モデル: アライメントからフレーズ対を抽出し、翻訳確率と並べ替えモデルを推定する (手順 4〜9)
!mkdir -p train/model && cp data/aligned.grow-diag-final-and train/model/
!mosesdecoder/scripts/training/train-model.perl --root-dir train --corpus data/train --f ja --e en \
    --alignment grow-diag-final-and --reordering msd-bidirectional-fe --lm 0:3:$PWD/data/lm.blm:8 \
    --first-step 4 --last-step 9 --cores 4 > train.log 2>&1
!ls -lh train/model/

# フレーズテーブルの中身: 「原言語のフレーズ ||| 目的言語のフレーズ ||| 4 つの確率 ||| アライメント ||| 頻度」
import gzip
from collections import defaultdict

table = defaultdict(list)
with gzip.open("train/model/phrase-table.gz", "rt") as f:
    for line in f:
        src, tgt, scores = line.split(" ||| ")[:3]
        if src in ("ありがとう", "分から ない", "お 願い"):
            table[src].append((float(scores.split()[2]), tgt))      # 3 番目のスコアが P(英語 | 日本語)
for src, cands in table.items():
    print(src, "→", [(tgt, round(p, 3)) for p, tgt in sorted(cands, reverse=True)[:6]])
```

```python
# 4. デコード: テストデータを翻訳して BLEU を測る
!mosesdecoder/bin/moses -f train/model/moses.ini -threads 4 < data/test.ja > data/test.out 2> decode.log

import sacrebleu
from sacremoses import MosesDetokenizer

md = MosesDetokenizer(lang="en")
hyps = [md.detokenize(line.split()) for line in open("data/test.out")]
refs = [line.strip() for line in open("data/test.raw.en")]
print(sacrebleu.corpus_bleu(hyps, [refs]))
for src, hyp, ref in list(zip(open("data/test.ja"), hyps, refs))[:8]:
    print(f"{src.strip()}\n   訳: {hyp}\n   参照: {ref}")
```

```python
# 翻訳の過程を見る: どのフレーズ対を使って訳したか (-t でフレーズの対応を出力する)
!echo "ありがとう 。 本当 に 助かり まし た 。" | mosesdecoder/bin/moses -f train/model/moses.ini -t 2> /dev/null
!echo "彼 は どこ に 行っ た の か 分から ない 。" | mosesdecoder/bin/moses -f train/model/moses.ini -t 2> /dev/null
```

- フレーズベース SMT は「翻訳モデル (フレーズ対とその確率)」「言語モデル (訳文の自然さ)」「並べ替えモデル」を別々に学習し、デコーダがそれらのスコアの重み付き和が最大になる訳を探す。ニューラル機械翻訳と違い、どのフレーズ対を使ったかを直接たどれる。
- 5 万文対・チューニングなしで、字幕データでの BLEU は 5 前後 (実測は 5.4〜5.5。実行する環境によって単語アライメントの結果がわずかに変わり、値も少し揺れる)。語順が大きく違う日英は、フレーズベースの手法が特に苦手とする言語対。
- 学習データにない単語 (未知語) は、訳されずにそのまま出力される。データが少ないと未知語が多くなる。
- ここでは省略したが、本来は開発データで各モデルの重みを調整する (MERT。`scripts/training/mert-moses.pl`)。BLEU が数ポイント変わるので、きちんと比較するなら必須。
- 単語アライメントには GIZA++ が標準的に使われてきたが、fast_align のほうがはるかに速く、質も大きくは変わらない。`train-model.perl` の手順 1〜3 (GIZA++ によるアライメント) を飛ばして、手順 4 から実行している。
- ビルドに時間がかかる (4 コアで約 6 分、Colab の 2 コアではその 2〜3 倍)。うまくいかないときは `bjam.log` と `train.log` を確認する。
- 現在の実用システムはほぼニューラルに置き換わったが、フレーズテーブルは対訳辞書の抽出や訳語の分析に、単語アライメントは用語の対応付けや注釈の投影に、今も使われている。

## Transformer の翻訳モデルをゼロから学習する {nmt_from_scratch gpu norun}

対訳データだけから、ニューラル機械翻訳モデルを自分で学習したい。サブワード分割、モデルの定義、学習、ビームサーチ、BLEU による評価までを一通り行う。

```python
!pip install -q sentencepiece sacrebleu
```

```python
import gzip
import random
import re
import urllib.request

# 田中コーパス (EDRDG 配布版、CC BY)。Tatoeba の日英対訳で、"A: 日本語<TAB>English#ID=..." の行が 1 対
raw = gzip.decompress(urllib.request.urlopen("https://www.edrdg.org/pub/Nihongo/examples.utf.gz").read()).decode("utf-8")
pairs = [m.groups() for m in re.finditer(r"^A: (.+?)\t(.+?)#ID=", raw, re.M)]

# 日本語・英語のどちらかが同じ対は 1 つだけ残す (学習とテストに同じ文が入るのを防ぐ)
seen_ja, seen_en, uniq = set(), set(), []
for ja, en in sorted(pairs):
    if ja not in seen_ja and en not in seen_en:
        seen_ja.add(ja); seen_en.add(en); uniq.append((ja, en))
random.seed(0)
random.shuffle(uniq)
test, valid, train = uniq[:1000], uniq[1000:1500], uniq[1500:]
print(len(pairs), "→", len(uniq), "対 /", len(train), len(valid), len(test), train[0])
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
EPOCHS, BATCH, PEAK_LR, WARMUP = 12, 128, 7e-4, 2000
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

- 流れは「対訳データの掃除 → サブワード分割 → モデルの定義 → 学習 → ビームサーチで翻訳 → BLEU で評価」。事前学習済みモデルを使わなくても、十数万文対の短い文なら翻訳らしいものが出てくる。
- モデルは約 1,150 万パラメータの小さな Transformer。RTX A5000 で 12 エポックが約 6 分 (T4 ならその 2〜3 倍)。
- 重複を除かずにランダムに分けると、学習データと同じ文がテストに入り、BLEU が約 21 と高めに出た。Tatoeba には同じ文や言い換えの対が多いので、分割の前に重複を除く。
- データの質が結果を大きく左右する。田中コーパスは短く対訳のずれが少ない文が多く、BLEU は 12 エポックで約 18 だった。同じモデルを、字幕のような会話文が多い opus-100 (29 万文対) で学習すると、BLEU は 3〜5 にとどまった。
- EDRDG 版は Tatoeba から定期的に更新されるので、実行する日によってデータの件数や BLEU の値が少し変わる。固定の分割で比べたいなら、田中コーパスから 5 万文対を抜き出した [small_parallel_enja](https://github.com/odashi/small_parallel_enja) を使う (英語は小文字化・トークン化済み)。
- 学習率のウォームアップ (最初は小さく、徐々に上げる) がないと、Transformer の学習は発散しやすい。ラベル平滑化と勾配クリッピングも定番の設定。
- 検証損失が下がらなくなったら学習を止める。データが少ないのに層を増やしても、過学習するだけで良くならない。
- 本格的に学習するなら、データを増やし (JParaCrawl など数千万文対)、逆翻訳で単言語データを活用し、fairseq や Marian、OpenNMT などの専用ツールを使う。

## 学習済みの翻訳モデルを自分の分野のデータでファインチューニングする {mt_finetune gpu norun}

公開されている翻訳モデルを、自分の分野の対訳データで追加学習して、その分野での訳質を上げたい。ここでは、田中コーパス (Tatoeba の日英対訳。辞書や教科書の例文のような短い文) を「自分の分野」に見立てる。

```python
!pip install -q datasets sentencepiece sacremoses sacrebleu accelerate
```

```python
import gzip
import random
import re
import urllib.request
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "Helsinki-NLP/opus-mt-ja-en"        # 日英の Marian モデル (約 300MB)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 田中コーパス (EDRDG 配布版、CC BY)。日本語・英語のどちらかが同じ対は 1 つだけ残してから分ける
raw = gzip.decompress(urllib.request.urlopen("https://www.edrdg.org/pub/Nihongo/examples.utf.gz").read()).decode("utf-8")
pairs = [m.groups() for m in re.finditer(r"^A: (.+?)\t(.+?)#ID=", raw, re.M)]
seen_ja, seen_en, uniq = set(), set(), []
for ja, en in sorted(pairs):
    if ja not in seen_ja and en not in seen_en:
        seen_ja.add(ja); seen_en.add(en); uniq.append((ja, en))
random.seed(0)
random.shuffle(uniq)
test = Dataset.from_list([{"ja": ja, "en": en} for ja, en in uniq[:500]])
train = Dataset.from_list([{"ja": ja, "en": en} for ja, en in uniq[1500:31500]])   # 3 万文対だけ使う

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

- ゼロから学習するよりはるかに少ないデータと時間で、特定の分野の文体や用語に合わせられる。RTX A5000 で 3 万文対・1 エポックが 1 分弱 (T4 ならその 2〜3 倍)。
- 実測の BLEU は 39.9 → 43.8。「I've」→「I have」のように短縮形を避けるなど、田中コーパスの英訳の文体に寄る。2 エポックにしても、全 13.8 万文対で学習しても 43.3〜43.6 で、それ以上は上がらなかった。
- 微調整前から BLEU が高いのは、`opus-mt-ja-en` が Tatoeba を含む OPUS のデータで学習されていて、テスト文をすでに見ている可能性があるため。分野が元のモデルの学習データと重なっていると、微調整の効果は出にくい (字幕が多い opus-100 で同じことをすると、BLEU は 17.6 → 17.5 で上がらなかった)。微調整の前に、元のモデルがその分野をどれだけ訳せるかを必ず測る。
- 学習率は小さめ (1e-5〜5e-5) にする。大きすぎると、元のモデルが持っていた一般的な翻訳能力が壊れる (破滅的忘却)。
- 微調整後は、対象の分野だけでなく、一般的な文のテストセットでも評価する。分野内で上がって分野外で下がるのが典型的な副作用。
- BLEU の差が小さいときは、評価の章の検定 (ブートストラップ) で有意かどうかを確かめる。訳文を実際に見比べることも欠かせない。
- NLLB のような多言語モデルや、LLM を LoRA で微調整する方法 (LLM の章) でも同じ考え方が使える。

## 逆翻訳で擬似対訳データを作る {back_translation}

対訳データは少ないが、訳文側の言語だけのテキスト (単言語データ) は大量にある。これを使って翻訳モデルの学習データを増やしたい。

```python
!pip install -q datasets sentencepiece sacremoses sacrebleu
```

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 英日翻訳モデルを強化したい場合: 日本語の単言語データを「日 → 英」のモデルで英語に訳し、(機械訳の英語, 人が書いた日本語) の組を作る
name = "Helsinki-NLP/opus-mt-ja-en"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSeq2SeqLM.from_pretrained(name).eval()

@torch.no_grad()
def translate(sentences, batch_size=32, **kwargs):
    out = []
    for i in range(0, len(sentences), batch_size):
        enc = tokenizer(sentences[i:i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=128)
        out += tokenizer.batch_decode(model.generate(**enc, max_new_tokens=128, **kwargs), skip_special_tokens=True)
    return out

mono = ["吾輩は猫である。名前はまだ無い。", "国境の長いトンネルを抜けると雪国であった。", "今日は朝から雨が降っていて、洗濯物が乾かない。",
        "新しいスマートフォンを買ったが、まだ使い方がよく分からない。", "この店のラーメンは、スープが濃厚でとてもおいしい。",
        "明日の会議は午後 3 時に変更になりました。", "彼女は大学で自然言語処理を研究している。", "駅前に大きな書店ができたので、帰りに寄ってみた。"]
pseudo_src = translate(mono, num_beams=4)
for en, ja in zip(pseudo_src, mono):
    print(f"{en}\n    ← {ja}")
```

```python
# 逆翻訳の作り方を変える: ビームサーチ (最も確からしい訳) / サンプリング (多様だが雑音が多い)
torch.manual_seed(0)
for ja in mono[2:5]:
    print(ja)
    print("   ビーム    :", translate([ja], num_beams=4)[0])
    for _ in range(3):
        print("   サンプリング:", translate([ja], do_sample=True, top_k=0, temperature=1.0)[0])
```

```python
# まとまった量を作って保存する。擬似データであることが分かる印 (タグ) を原言語側の先頭に付けておく
from datasets import load_dataset

ds = load_dataset("Helsinki-NLP/opus-100", "en-ja", split="test")
mono_ja = [ex["translation"]["ja"] for ex in ds.select(range(200)) if 5 <= len(ex["translation"]["ja"]) <= 60]
bt_en = translate(mono_ja, num_beams=4)
with open("backtranslated.tsv", "w", encoding="utf-8") as f:
    for en, ja in zip(bt_en, mono_ja):
        f.write(f"<BT> {en}\t{ja}\n")
print(len(mono_ja), "文対を保存")
!head -n 5 backtranslated.tsv

# 逆翻訳の質の目安: このデータにはたまたま人手の英訳があるので、それと比べる
import sacrebleu
refs = [ex["translation"]["en"] for ex in ds.select(range(200)) if 5 <= len(ex["translation"]["ja"]) <= 60]
print(sacrebleu.corpus_bleu(bt_en, [refs]))
```

- 逆翻訳 (back-translation) は、訳文側が「人の書いた自然な文」で、原文側が「機械訳」の対訳データを作る。訳文側の質が保たれるので、原文側に多少の誤訳があっても、翻訳モデルの流暢さが上がる。
- 作った擬似対訳は、本物の対訳データと混ぜて学習に使う。比率は 1:1 から始めて、検証データで調整するのが一般的。
- 擬似データの先頭に `<BT>` のようなタグを付けておくと、モデルが本物と擬似データを区別して学習できる (tagged back-translation)。
- データが大量にあるときは、ビームサーチよりサンプリングで作った逆翻訳のほうが効果が高いと報告されている。原文側が多様になり、学習の信号が増えるため。データが少ないときは、ビームサーチのほうが安全。
- 逆翻訳に使うモデル自体の質が低いと、効果は小さい。次項のフィルタリングで質の低い組を捨てる方法がある。

## 擬似対訳データを往復翻訳の一致度でふるいにかける {pseudo_parallel_filter}

逆翻訳で作った擬似対訳には、誤訳や意味の抜けが混ざる。質の低い組を自動的に取り除きたい。

```python
!pip install -q datasets sentencepiece sacremoses sacrebleu sentence-transformers
```

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def load(name, **kwargs):
    return AutoTokenizer.from_pretrained(name, **kwargs), AutoModelForSeq2SeqLM.from_pretrained(name).eval()

ja_en = load("Helsinki-NLP/opus-mt-ja-en")                                       # 逆翻訳用 (日 → 英)
en_ja = load("facebook/nllb-200-distilled-600M", src_lang="eng_Latn")            # 訳し戻し用 (英 → 日)

@torch.no_grad()
def translate(pair, sentences, batch_size=16, **kwargs):
    tokenizer, model = pair
    out = []
    for i in range(0, len(sentences), batch_size):
        enc = tokenizer(sentences[i:i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=128)
        out += tokenizer.batch_decode(model.generate(**enc, num_beams=4, max_new_tokens=128, **kwargs), skip_special_tokens=True)
    return out

ds = load_dataset("Helsinki-NLP/opus-100", "en-ja", split="test")
mono_ja = [ex["translation"]["ja"] for ex in ds if 8 <= len(ex["translation"]["ja"]) <= 60][:200]
pseudo_en = translate(ja_en, mono_ja)            # 逆翻訳: 日 → 英
round_trip = translate(en_ja, pseudo_en, forced_bos_token_id=en_ja[0].convert_tokens_to_ids("jpn_Jpan"))     # もう一度訳し戻す: 英 → 日
print(mono_ja[0], "→", pseudo_en[0], "→", round_trip[0])
```

```python
import pandas as pd
import sacrebleu
from sentence_transformers import SentenceTransformer

# 指標 1: 元の日本語と、往復翻訳して戻ってきた日本語の一致度 (文単位の chrF)
chrf = [sacrebleu.sentence_chrf(rt, [ja]).score for rt, ja in zip(round_trip, mono_ja)]

# 指標 2: 日本語と、逆翻訳した英語の、多言語埋め込みによる類似度 (翻訳し直す必要がない)
embedder = SentenceTransformer("intfloat/multilingual-e5-small")
e_ja = embedder.encode(["query: " + s for s in mono_ja], normalize_embeddings=True)
e_en = embedder.encode(["query: " + s for s in pseudo_en], normalize_embeddings=True)
cos = (e_ja * e_en).sum(axis=1)

df = pd.DataFrame({"ja": mono_ja, "pseudo_en": pseudo_en, "round_trip": round_trip, "chrF": chrf, "cos": cos})
print(df[["chrF", "cos"]].describe().round(3).T)
print("2 つの指標の相関:", round(df["chrF"].corr(df["cos"], method="spearman"), 3))
pd.set_option("display.max_colwidth", 60); pd.set_option("display.width", 250)
print("--- 往復翻訳の一致度が低い組 (捨てる候補)")
print(df.sort_values("chrF").head(8)[["ja", "pseudo_en", "chrF", "cos"]].to_string(index=False))
print("--- 一致度が高い組")
print(df.sort_values("chrF").tail(5)[["ja", "pseudo_en", "chrF", "cos"]].to_string(index=False))
```

```python
# ふるいの効果を確かめる: 人手の英訳に対する逆翻訳の質 (chrF) が、残した組で上がっているか
refs = [ex["translation"]["en"] for ex in ds if 8 <= len(ex["translation"]["ja"]) <= 60][:200]
df["quality"] = [sacrebleu.sentence_chrf(h, [r]).score for h, r in zip(pseudo_en, refs)]
for metric in ["chrF", "cos"]:
    ranked = df.sort_values(metric, ascending=False)
    print(metric, {f"上位 {int(r * 100)}%": round(float(ranked.head(int(len(df) * r))["quality"].mean()), 1) for r in [0.25, 0.5, 0.75, 1.0]})

# 埋め込みの類似度が高すぎる組は、訳されずに原文がそのまま残っている (コピー) ことが多い
print(df[df["cos"] > 0.97][["ja", "pseudo_en", "cos"]].head(5).to_string(index=False))
```

- 質の高い擬似対訳なら、訳し戻したときに元の文に近いものが返ってくるはず、という考え方。一致度の低い組は、逆翻訳の時点で意味が壊れている可能性が高い。
- 一致度の上位だけを学習に使う。しきい値を高くすると質は上がるが量が減るので、検証データの翻訳精度で決める。
- 往復翻訳は、2 つの翻訳モデルを通すので時間がかかる。多言語の文埋め込みの類似度なら 1 回の計算で済み、対訳コーパスからの雑音の除去にも同じように使える。この実験では、埋め込みの類似度で選んだほうが、残した組の質が高かった。
- 埋め込みの類似度は、訳されずに原文がそのまま残った組 (コピー) に最高点を付けてしまう。類似度が高すぎる組や、原文と訳文の文字列がほぼ同じ組は、別に取り除く。
- 往復で元に戻るからといって、中間の訳が正しいとは限らない (同じ誤りを往復で打ち消すことがある)。逆に、正しい訳でも言い換えで戻ってくると一致度は下がる。あくまで目安として使う。
- 短い文や定型文ばかりが高得点で残りやすい。残したデータの長さや内容の分布が偏っていないかを確かめる。

## タグを付けて訳文の文体を制御する (丁寧さの制御) {controlled_translation gpu norun}

同じ原文を、敬体 (です・ます) と常体 (だ・である) に訳し分けたい。原文の先頭に付けたタグで、訳文の文体を指定できるようにする。

```python
!pip install -q datasets sentencepiece sacremoses accelerate
```

```python
import re
from datasets import Dataset, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

POLITE = re.compile(r"(です|ます|でした|ました|ません|でしょう|ください|ましょう)(か|よ|ね|わ)?[。！？!?]*$")
PLAIN = re.compile(r"(だ|である|だった|ない|た|る|う|い)(よ|ね|な|ぞ|わ|か)?[。！？!?]*$")

def style(ja):
    """訳文の文末から文体を判定する (規則による自動ラベル付け)"""
    return "<polite>" if POLITE.search(ja) else "<plain>" if PLAIN.search(ja) else None

ds = load_dataset("Helsinki-NLP/opus-100", "en-ja")
rows = []
for ex in ds["train"].select(range(150000)):
    en, ja = ex["translation"]["en"].strip(), ex["translation"]["ja"].strip()
    if 10 <= len(ja) <= 50 and 3 <= len(en.split()) <= 20 and style(ja):
        rows.append({"en": f"{style(ja)} {en}", "ja": ja, "style": style(ja)})
train = Dataset.from_list(rows).shuffle(seed=0)
polite = train.filter(lambda r: r["style"] == "<polite>").select(range(3000))
plain = train.filter(lambda r: r["style"] == "<plain>").select(range(3000))
from datasets import concatenate_datasets
train = concatenate_datasets([polite, plain]).shuffle(seed=0)      # 敬体と常体を同数にする
print(train[0], train[1])
```

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(name, src_lang="eng_Latn", tgt_lang="jpn_Jpan")
model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device)
JA = tokenizer.convert_tokens_to_ids("jpn_Jpan")
tokenizer.add_tokens(["<polite>", "<plain>"], special_tokens=True)       # タグを新しいトークンとして追加し、埋め込みの行を増やす
model.resize_token_embeddings(len(tokenizer))

@torch.no_grad()
def translate(sentences, tag=None, batch_size=32):
    model.eval()
    out = []
    for i in range(0, len(sentences), batch_size):
        batch = [f"{tag} {s}" if tag else s for s in sentences[i:i + batch_size]]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64).to(model.device)
        out += tokenizer.batch_decode(model.generate(**enc, num_beams=4, max_new_tokens=64, forced_bos_token_id=JA), skip_special_tokens=True)
    return out

tests = ["I will go to Tokyo tomorrow.", "This book is very interesting.", "Do you know where the station is?", "He did not come to the party.",
         "We need to finish this work by Friday.", "It was raining when I left the house."]
before = translate(tests)
print(before)
```

```python
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

enc = train.map(lambda b: tokenizer(b["en"], text_target=b["ja"], truncation=True, max_length=64), batched=True, remove_columns=["en", "ja", "style"])
args = Seq2SeqTrainingArguments(output_dir="mt-style", num_train_epochs=1, learning_rate=5e-5, warmup_steps=20,
                                per_device_train_batch_size=32, fp16=True, logging_steps=50, save_strategy="no", report_to="none", seed=0)
Seq2SeqTrainer(model=model, args=args, train_dataset=enc, data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)).train()
```

```python
for en, b, p, q in zip(tests, before, translate(tests, "<polite>"), translate(tests, "<plain>")):
    print(f"{en}\n   学習前: {b}\n   <polite>: {p}\n   <plain> : {q}")

# 指定どおりの文体になった割合を、学習に使っていない文で測る
held_out = [ex["translation"]["en"] for ex in ds["test"] if 3 <= len(ex["translation"]["en"].split()) <= 20][:200]
for tag in ["<polite>", "<plain>"]:
    outputs = translate(held_out, tag)
    print(tag, "指定どおりの割合:", round(sum(style(o) == tag for o in outputs) / len(outputs), 3),
          "/ 判定できない文:", sum(style(o) is None for o in outputs))
```

- 学習データの訳文を規則で分類し、その結果をタグとして原文の先頭に付けて学習する (サイドコンストレイント)。モデルは「タグと訳文の文体の対応」を学ぶので、翻訳時にタグを変えるだけで文体を切り替えられる。
- 対訳データに文体の注釈は要らない。訳文から自動的に判定できる性質なら、何でも制御の対象にできる (丁寧さ、態、長さ、分野など)。
- 6,000 文対・1 エポックなら T4 GPU で数分。新しく足したタグの埋め込みは乱数から始まるので、データが少ないと効きが不完全になる。指定どおりになった割合を必ず測る。
- 文体を変えると、訳の内容まで変わってしまうことがある。制御の成功率だけでなく、BLEU などで翻訳の質が落ちていないかも確かめる。
- LLM による翻訳なら、プロンプトで「です・ます調で」と指示するだけで同じことができる。専用の翻訳モデルを使う場面 (速度、コスト、機密性) では、タグによる制御が今も有効。

## 漢字を部品に分解して日本語と中国語で共有する {subcharacter_decomposition}

日本語と中国語は漢字を共有しているが、字体が違う (経 / 经、験 / 验) ために別の文字として扱われてしまう。漢字を構成要素に分解して、共通部分を取り出したい。

```python
!pip install -q sentencepiece datasets
!wget -q https://raw.githubusercontent.com/cjkvi/cjkvi-ids/master/ids.txt
```

```python
import re

# IDS (漢字構成記述列): 漢字を「構成の型を表す記号 (⿰ 左右、⿱ 上下など) + 部品」で表したデータ
ids = {}
for line in open("ids.txt", encoding="utf-8"):
    if line.startswith("U+"):
        _, char, *descriptions = line.rstrip("\n").split("\t")
        ids[char] = re.sub(r"\[[A-Z]+\]", "", descriptions[0])         # 地域ごとの字形の違いがある場合は最初のものを使う
print(len(ids), "字")
for c in "語験経樹鬱":
    print(c, ids[c])
```

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def decompose(char, depth=1):
    """depth 段だけ分解する。分解できない字 (それ自体が部品) はそのまま返す"""
    d = ids.get(char, char)
    if depth == 0 or d == char:
        return char
    return "".join(decompose(c, depth - 1) if c in ids and c != char else c for c in d)

def decompose_text(text, depth=1):
    return "".join(decompose(c, depth) if "一" <= c <= "鿿" else c for c in text)

for word in ["経験", "言語", "機械翻訳"]:
    print(word, "→ 1 段:", decompose_text(word, 1), "/ 2 段:", decompose_text(word, 2), "/ 3 段:", decompose_text(word, 3))
```

```python
# 日本語の字体と中国語 (簡体字) の字体で、共有される部品がどれだけ増えるか
STRUCTURE = set("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")
pairs = [("経験", "经验"), ("図書館", "图书馆"), ("機械", "机械"), ("発音", "发音"), ("説明", "说明"), ("電話", "电话"), ("銀行", "银行"), ("読書", "读书")]

def overlap(a, b):
    a, b = set(a) - STRUCTURE, set(b) - STRUCTURE
    return len(a & b) / len(a | b)

for ja, zh in pairs:
    row = [f"{overlap(decompose_text(ja, d), decompose_text(zh, d)):.2f}" for d in (0, 1, 2, 3)]
    print(f"{ja} / {zh}: 文字のまま {row[0]} → 1 段 {row[1]} → 2 段 {row[2]} → 3 段 {row[3]} | {decompose_text(ja, 2)} / {decompose_text(zh, 2)}")
```

```python
# 分解した文字列でサブワード分割を学習し、日本語と中国語で共有される語彙の割合を調べる
import sentencepiece as spm
from datasets import load_dataset

# 日本語と中国語のテキスト (対訳である必要はない)
ja = [ex["translation"]["ja"] for ex in load_dataset("Helsinki-NLP/opus-100", "en-ja", split="train").select(range(30000))]
zh = [ex["translation"]["zh"] for ex in load_dataset("Helsinki-NLP/opus-100", "en-zh", split="train").select(range(30000))]
for name, depth in [("char", 0), ("ideograph", 2)]:
    with open(f"{name}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(decompose_text(s, depth) for s in ja + zh))
    spm.SentencePieceTrainer.train(input=f"{name}.txt", model_prefix=name, vocab_size=8000, character_coverage=0.9995, minloglevel=2)

for name, depth in [("char", 0), ("ideograph", 2)]:
    sp = spm.SentencePieceProcessor(model_file=f"{name}.model")
    vocab = {sp.id_to_piece(i) for i in range(sp.get_piece_size())}
    share = lambda sents: {p for s in sents[:5000] for p in sp.encode(decompose_text(s, depth), out_type=str)}
    v_ja, v_zh = share(ja), share(zh)
    print(f"{name}: 日本語で使われる語彙 {len(v_ja)} / 中国語 {len(v_zh)} / 共通 {len(v_ja & v_zh)} ({len(v_ja & v_zh) / len(v_ja | v_zh):.0%})")
    print("   ", sp.encode(decompose_text("経験を説明する", depth), out_type=str), sp.encode(decompose_text("说明经验", depth), out_type=str))
```

- 漢字は、偏や旁などの部品の組み合わせでできている。日本の新字体と中国の簡体字は字体が違っても、部品まで分解すると共通部分が見つかることがある (説 / 说 はどちらも「兑」を含む)。
- 分解した文字列でサブワード分割を学習すれば、日本語と中国語で語彙の共有が増えると期待できる。ただし、この実験では共通の語彙の割合は増えなかった (22% → 21%)。部品の並びが長くなって別々のサブワードに切られること、簡体字の部品が別の字であることが原因で、分解するだけでは足りない。
- 対訳データが少ない言語対のニューラル機械翻訳では、部首や筆画への分解で翻訳精度が上がったという報告がある。効果は分解の深さやデータの量に左右されるので、翻訳の精度で確かめる。
- 分解の深さは調整が必要。浅いと共有が増えず、深すぎると (筆画まで) 文字列が長くなって、意味の手がかりも失われる。
- IDS には、Unicode に登録されていない部品を表す丸数字 (①、② など) が含まれる。別々の部品が同じ記号になるので、深く分解しすぎると情報が失われる。
- 簡体字は部品そのものが簡略化されている (訁 → 讠) ので、分解しても一致しないことが多い。字体の対応表 (異体字データ、OpenCC など) で先に字体をそろえる方法と比べて選ぶ。
- 構成の型を表す記号 (⿰ など) を残すと元の字に戻せるが、系列は長くなる。翻訳の入力側だけ分解する、という使い方もある。

## 1 つの原文から多様な訳を複数出す {diverse_translation}

1 つの原文に対して、言い回しの違う訳を複数出したい。ビームサーチの上位候補が似たものばかりになる問題にも対処したい。

```python
!pip install -q sentencepiece sacremoses sacrebleu
```

```python
import itertools
import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(name, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(name).eval()
JA = tokenizer.convert_tokens_to_ids("jpn_Jpan")

@torch.no_grad()
def generate(text, n=5, **kwargs):
    enc = tokenizer(text, return_tensors="pt")
    out = model.generate(**enc, max_new_tokens=64, num_return_sequences=n, forced_bos_token_id=JA, **kwargs)
    return tokenizer.batch_decode(out, skip_special_tokens=True)

def diversity(candidates):
    """候補どうしの似ていなさ: 1 − (候補の組の文字 n-gram 一致度の平均)。大きいほど多様"""
    pairs = list(itertools.combinations(candidates, 2))
    return 1 - sum(sacrebleu.sentence_chrf(a, [b]).score for a, b in pairs) / len(pairs) / 100

text = "I am sorry, but I cannot attend the meeting tomorrow because I have a doctor's appointment."
methods = {
    "ビームサーチの上位 5 件": dict(num_beams=5),
    "ビーム幅を広げる (20)": dict(num_beams=20),
    "サンプリング (温度 1.0)": dict(do_sample=True, top_k=0, temperature=1.0),
    "サンプリング (top-p 0.9)": dict(do_sample=True, top_p=0.9),
    "サンプリング (温度 0.7)": dict(do_sample=True, top_k=0, temperature=0.7),
}
torch.manual_seed(0)
for name_, kwargs in methods.items():
    cands = generate(text, **kwargs)
    print(f"--- {name_} (多様性 {diversity(cands):.2f}, 異なる訳 {len(set(cands))} 種類)")
    for c in cands:
        print("   ", c)
```

```python
# 訳の先頭を指定して、違う出だしの訳を作る (デコーダの入力を途中まで与える)
@torch.no_grad()
def translate_with_prefix(text, prefix):
    enc = tokenizer(text, return_tensors="pt")
    prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"]
    start = torch.tensor([[model.config.decoder_start_token_id, JA]])          # NLLB のデコーダは「開始記号 + 言語タグ」から始まる
    out = model.generate(**enc, decoder_input_ids=torch.cat([start, prefix_ids], dim=1), num_beams=4, max_new_tokens=64)
    return tokenizer.decode(out[0], skip_special_tokens=True)

for prefix in ["申し訳", "すみません", "明日", "病院", "医者"]:
    print(f"[{prefix}] {translate_with_prefix(text, prefix)}")
```

```python
# 多様性と質の両方を測る: 多様にするほど、訳の質 (参照訳との一致) は下がりやすい
from datasets import load_dataset

ds = load_dataset("Helsinki-NLP/opus-100", "en-ja", split="test")
sample = [(ex["translation"]["en"], ex["translation"]["ja"]) for ex in ds if 5 <= len(ex["translation"]["en"].split()) <= 15][:30]
torch.manual_seed(0)
for name_, kwargs in methods.items():
    div, quality, best = [], [], []
    for en, ja in sample:
        cands = generate(en, **kwargs)
        scores = [sacrebleu.sentence_chrf(c, [ja]).score for c in cands]
        div.append(diversity(cands)); quality.append(sum(scores) / len(scores)); best.append(max(scores))
    print(f"{name_}: 多様性 {sum(div) / len(div):.2f} / 平均の質 (chrF) {sum(quality) / len(quality):.1f} / 最も良い候補の質 {sum(best) / len(best):.1f}")
```

- ビームサーチの上位候補は、文末や助詞が少し違うだけの、ほとんど同じ訳になりやすい。ビーム幅を広げても、多様性はあまり増えない。
- サンプリングは多様な訳を出すが、温度が高いと誤訳や不自然な訳が混ざる。NLLB のような多言語モデルでは、温度 1.0 のサンプリングで他の言語の文字 (韓国語、ラテン文字の単語など) まで混ざった。温度や top-p で、多様性と質の釣り合いを調整する。
- 訳の出だしを指定すると、語順や構文の違う訳を狙って作れる。翻訳の候補を人に提示する場面 (翻訳支援) で便利。
- 複数の候補のうち最も良いものの質は、ほどほどに多様な設定 (ここでは温度 0.7) で最も高く、多様にしすぎると下がった。候補を出してから品質推定やリランキングで 1 つを選ぶ、という使い方ができる。
- 順方向と逆方向 (文末から生成する) のモデルの出力を組み合わせる、言い換えモデルを通す、といった方法でも多様な訳が得られる。

## 古文を現代語に訳す {classical_to_modern}

古文を現代語に訳したい。LLM に訳させ、品詞分解の情報を与えると訳が良くなるかも試す。

```python
!pip install -q sacrebleu
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
llm = AutoModelForCausalLM.from_pretrained(name).eval()

@torch.no_grad()
def chat(prompt, max_new_tokens=160):
    ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
    out = llm.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0]

# (古文, 参考にする現代語訳) の組
data = [
    ("今は昔、竹取の翁といふ者ありけり。", "今となっては昔のことだが、竹取の翁という者がいた。"),
    ("野山にまじりて竹を取りつつ、よろづのことに使ひけり。", "野山に分け入って竹を取っては、いろいろなことに使っていた。"),
    ("春はあけぼの。やうやう白くなりゆく山ぎは、少し明かりて、紫だちたる雲の細くたなびきたる。", "春は明け方がよい。だんだん白くなっていく山ぎわが少し明るくなって、紫がかった雲が細くたなびいているのがよい。"),
    ("つれづれなるままに、日暮らし、硯に向かひて、心にうつりゆくよしなしごとを、そこはかとなく書きつくれば、あやしうこそものぐるほしけれ。",
     "することもなく退屈なのにまかせて、一日中硯に向かって、心に浮かんでは消えていくとりとめのないことを、何となく書きつけていると、妙に気持ちが高ぶってくる。"),
    ("行く河の流れは絶えずして、しかももとの水にあらず。", "流れていく川の流れは絶えることがなく、それでいて、もとの水ではない。"),
    ("男もすなる日記といふものを、女もしてみむとてするなり。", "男も書くという日記というものを、女の私も書いてみようと思って書くのである。"),
]

PLAIN = "次の古文を、意味を変えずに現代語に訳してください。訳文だけを 1 行で出力してください。\n\n古文: {}"
FEWSHOT = ("古文を現代語に訳します。訳文だけを 1 行で出力してください。\n\n古文: いづれの御時にか、女御、更衣あまたさぶらひたまひけるなかに\n現代語訳: どの帝の御代であったか、女御や更衣が大勢お仕えしていらっしゃった中に\n\n"
           "古文: 月日は百代の過客にして、行きかふ年もまた旅人なり。\n現代語訳: 月日は永遠の旅人であり、過ぎ去ってはやって来る年もまた旅人である。\n\n古文: {}\n現代語訳:")
for src, ref in data:
    print(f"{src}\n   訳 (指示のみ): {chat(PLAIN.format(src))}\n   訳 (例を示す): {chat(FEWSHOT.format(src))}\n   参考: {ref}")
```

```python
import sacrebleu

for label, prompt in [("指示のみ", PLAIN), ("例を示す", FEWSHOT)]:
    hyps = [chat(prompt.format(src)) for src, _ in data]
    print(label, "chrF =", round(sacrebleu.corpus_chrf(hyps, [[ref for _, ref in data]]).score, 1))
print("訳さずにそのまま出した場合 chrF =", round(sacrebleu.corpus_chrf([src for src, _ in data], [[ref for _, ref in data]]).score, 1))
```

```python
# 誤訳しやすい語 (古今異義語) の語釈を与えてから訳させる
GLOSSARY = {"あやし": "不思議だ、妙だ (現代語の「怪しい」とは違う)", "ものぐるほし": "気持ちが高ぶる、正気を失ったようだ", "つれづれなり": "することがなく退屈だ",
            "よしなしごと": "とりとめのないこと", "すなる": "する + 伝聞の「なり」= するという", "やうやう": "だんだん、しだいに"}
src = data[3][0]
notes = "\n".join(f"- {k}: {v}" for k, v in GLOSSARY.items() if k[:3] in src)
print(notes)
print(chat(f"次の古文を現代語に訳してください。語釈を参考にし、訳文だけを 1 行で出力してください。\n\n語釈:\n{notes}\n\n古文: {src}"))
```

- 古文から現代語への翻訳は、同じ言語の中での「時代をまたぐ翻訳」。語順はほぼ同じなので、難しさは語彙 (古今異義語)、助動詞の意味、省略された主語や敬語の解釈にある。
- LLM は有名な作品の冒頭なら訳を覚えていることが多い。知られていない文での性能は、それより低いと考える。評価には、モデルが見たことのなさそうな文を使う。
- この実験では、例を示しても訳は良くならなかった (chrF は「指示のみ」が 33.5〜35.5、「例を示す」が 33.8。6 文しかないので、この程度の差は実行する環境による揺れの範囲)。2 文目では、例として示した源氏物語の訳をそのまま出力する失敗も出た。一方、語釈を与えると「ものぐるほし」が正しく「気持ちが高ぶる」と訳された。訳文に英単語 (thoughts) が混ざる失敗も出ている。
- 小さなモデルは、古語を現代語の似た語と取り違える (「あやし」を「怪しい」と訳す)。語釈や品詞分解の結果 (形態素解析の章の古文のレシピ) をプロンプトに入れると、誤訳を減らせることがある。
- 対訳データがあれば、この章のファインチューニングのレシピがそのまま使える。現代語の単語ベクトルを古文に適応させる、古文の単言語データで逆翻訳する、といった工夫が研究されている。
- 参考訳は 1 つの解釈にすぎず、直訳か意訳かでも大きく変わる。自動評価の数値は目安にとどめる。
