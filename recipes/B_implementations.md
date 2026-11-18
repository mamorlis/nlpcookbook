# 仕組みを理解するための実装

ライブラリを使えば数行で済む処理を、あえて自分で実装して仕組みを確かめるレシピ。単語分割と品詞タグ付け（系列ラベリング、HMM、ビタビアルゴリズム）、BPE、n-gram 言語モデル、カウントベースの単語ベクトル、GPT の事前学習の学習ループ、SimCSE、ラベル伝播、LLM の透かし、構文解析のアルゴリズム、論理式による意味表現、適合率・再現率・F1、転置索引と BM25、規則によるチャットボット、翻訳の評価指標の学習、スペル訂正。実務で使うときのライブラリ版がある場合は、各レシピの注意書きに本文のレシピを挙げた。

## 単語分割を文字単位の系列ラベリングとして解く {char_segmentation}

日本語の単語分割を「各文字が単語の先頭 (B) か途中 (I) か」を当てる問題として、CRF で学習したい。

```python
!pip install -q sklearn-crfsuite fugashi unidic-lite regex
```

```python
import io, re, urllib.request, zipfile
from fugashi import Tagger

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

# MeCab の分割結果を正解とみなす (本来は人手で分割したコーパスを使う)
tagger = Tagger()
text = aozora("https://www.aozora.gr.jp/cards/000148/files/752_ruby_2438.zip")
sents = [[w.surface for w in tagger(s)] for s in re.split(r"(?<=。)|\n", text) if s and s.strip()]
train, test = sents[:2200], sents[2200:]

def to_chars_tags(words):
    chars = [c for w in words for c in w]
    tags = [t for w in words for t in ["B"] + ["I"] * (len(w) - 1)]
    return chars, tags
```

```python
import regex
import sklearn_crfsuite

def char_type(c):
    for name in ["Han", "Hiragana", "Katakana", "Latin", "Nd"]:
        if regex.match(rf"\p{{{name}}}", c):
            return name
    return "Other"

def features(chars, i):
    f = {"bias": 1.0}
    for d in range(-2, 3):          # 前後 2 文字の文字と文字種
        if 0 <= i + d < len(chars):
            f[f"c{d}"] = chars[i + d]
            f[f"t{d}"] = char_type(chars[i + d])
    for d in (-1, 0):               # 文字 bigram と文字種の変わり目
        if 0 <= i + d and i + d + 1 < len(chars):
            f[f"bi{d}"] = chars[i + d] + chars[i + d + 1]
            f[f"tt{d}"] = char_type(chars[i + d]) + ">" + char_type(chars[i + d + 1])
    return f

def dataset(sents):
    X, y = [], []
    for words in sents:
        chars, tags = to_chars_tags(words)
        X.append([features(chars, i) for i in range(len(chars))]); y.append(tags)
    return X, y

X_train, y_train = dataset(train)
X_test, y_test = dataset(test)
crf = sklearn_crfsuite.CRF(c1=0.1, c2=0.1, max_iterations=60).fit(X_train, y_train)
```

```python
def segment(sentence):
    chars = list(sentence)
    tags = crf.predict_single([features(chars, i) for i in range(len(chars))])
    return "".join((" " if t == "B" and i else "") + c for i, (c, t) in enumerate(zip(chars, tags))).split()

def word_f1(gold_sents, pred_sents):
    """単語単位の F1: 開始位置と終了位置が両方一致した単語だけを正解とする"""
    tp = n_gold = n_pred = 0
    for gold, pred in zip(gold_sents, pred_sents):
        spans = lambda ws: {(sum(map(len, ws[:i])), sum(map(len, ws[:i + 1]))) for i in range(len(ws))}
        g, p = spans(gold), spans(pred)
        tp += len(g & p); n_gold += len(g); n_pred += len(p)
    prec, rec = tp / n_pred, tp / n_gold
    return 2 * prec * rec / (prec + rec)

pred = [segment("".join(ws)) for ws in test]
print("単語 F1:", round(word_f1(test, pred), 4))
print(segment("親譲りの無鉄砲で小供の時から損ばかりしている。"))
print(segment("最新のスマートフォンで自然言語処理を勉強する。"))
```

- 辞書を使わず、文字の並びだけから単語の境界を学習する。学習データにない分野の語では誤りが増えやすい。ただし、この例の 2 文目は正しく分割できている。CRF はラベル列全体をまとめて決めるが、日本語の形態素解析と文分割の章の「KyTea で単語分割と読みの推定をする」は境界ごとに独立に判定し（点予測）、辞書も素性に使う。
- 中国語の単語分割では、B/M/E/S（先頭・中間・末尾・単独）の 4 タグが標準的。

## HMM で品詞タガーを作る {hmm_tagger}

隠れマルコフモデル (HMM) による古典的な品詞タグ付けを学習・評価して、ベースラインと比べたい。

```python
import nltk
nltk.download("treebank", quiet=True)
nltk.download("universal_tagset", quiet=True)
from nltk.corpus import treebank
from nltk.tag import DefaultTagger, UnigramTagger, hmm, tnt

sents = list(treebank.tagged_sents(tagset="universal"))
train, test = sents[:3500], sents[3500:]
print(len(train), len(test), train[0][:6])

baseline = UnigramTagger(train, backoff=DefaultTagger("NOUN"))   # 各単語に最頻の品詞を付けるだけ
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(train, estimator=lambda fd, bins: nltk.LidstoneProbDist(fd, 0.01, bins))

tnt_tagger = tnt.TnT(unk=DefaultTagger("NOUN"), Trained=True)   # trigram HMM + 未知語は名詞とみなす
tnt_tagger.train(train)

print("ユニグラム (最頻タグ):   ", round(baseline.accuracy(test), 4))
print("HMM (bigram):            ", round(tagger.accuracy(test), 4))
print("TnT (trigram + 未知語処理):", round(tnt_tagger.accuracy(test), 4))
```

```python
for sent in ["They can fish .", "I saw her duck .", "Time flies like an arrow ."]:
    print(tagger.tag(sent.split()))
```

```python
# 未知語と既知語に分けて正解率を見る
vocab = {w for s in train for w, _ in s}
known = unknown = known_ok = unknown_ok = 0
for s in test:
    for (w, gold), (_, pred) in zip(s, tagger.tag([w for w, _ in s])):
        if w in vocab:
            known += 1; known_ok += gold == pred
        else:
            unknown += 1; unknown_ok += gold == pred
print(f"既知語 {known_ok / known:.3f} ({known} 語) / 未知語 {unknown_ok / unknown:.3f} ({unknown} 語)")
```

- 「各単語の最頻タグ」だけで 9 割を超える。品詞タグ付けの難しさは、残りの曖昧な語と未知語にある。
- 素朴な HMM は未知語の出力確率をうまく見積もれず、未知語でほぼ当たらない。そのせいで全体ではユニグラムのベースラインに負けることすらある。
- HMM は単語の綴りを見ないのが弱点。接尾辞や大文字などの特徴を使える CRF やニューラルモデルが有利な理由。

## ビタビアルゴリズムを実装して最適なラベル列を求める {viterbi}

HMM や CRF の復号に使われるビタビアルゴリズムを、自分で実装して動きを確かめたい。

```python
import numpy as np

def viterbi(emission, transition, start):
    """emission:（系列長, ラベル数）, transition:（ラベル数, ラベル数）, start:（ラベル数,）の対数スコア"""
    T, K = emission.shape
    score = start + emission[0]
    back = np.zeros((T, K), dtype=int)
    for t in range(1, T):
        cand = score[:, None] + transition      # cand[i, j] = 直前が i で今が j のときのスコア
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + emission[t]
    path = [int(score.argmax())]
    for t in range(T - 1, 0, -1):               # 後ろからたどる
        path.append(int(back[t, path[-1]]))
    return path[::-1], float(score.max())

labels = ["名詞", "動詞", "助詞"]
words = ["猫", "が", "魚", "を", "食べる"]
log = lambda p: np.log(np.array(p) + 1e-12)
start = log([0.7, 0.2, 0.1])
transition = log([[0.2, 0.2, 0.6],     # 名詞 → 名詞 / 動詞 / 助詞
                  [0.3, 0.3, 0.4],     # 動詞 →
                  [0.6, 0.39, 0.01]])  # 助詞 →
emission = log([[0.9, 0.05, 0.05],     # 猫
                [0.5, 0.1, 0.4],       # が（単独では「蛾」の名詞の確率が高い）
                [0.9, 0.05, 0.05],     # 魚
                [0.1, 0.1, 0.8],       # を
                [0.1, 0.85, 0.05]])    # 食べる

path, score = viterbi(emission, transition, start)
print([f"{w}/{labels[i]}" for w, i in zip(words, path)], round(score, 3))
```

```python
# 貪欲法（各位置で独立に最大のラベルを選ぶ）との比較、および総当たりとの一致の確認
from itertools import product

greedy = emission.argmax(axis=1).tolist()
def total(p):
    return start[p[0]] + emission[0, p[0]] + sum(transition[a, b] + emission[t, b] for t, (a, b) in enumerate(zip(p, p[1:]), 1))
brute = max(product(range(3), repeat=5), key=total)
print("貪欲法:", [labels[i] for i in greedy], round(float(total(greedy)), 3))
print("総当たり:", [labels[i] for i in brute], round(float(total(brute)), 3), " ビタビと一致:", list(brute) == path)
```

- 「が」は単独で見ると名詞（蛾）の確率が高いので、貪欲法は「猫/名詞 が/名詞」と誤る。ビタビは「名詞 → 名詞」の遷移が起こりにくいことも考えて「が/助詞」を選び、スコアも高くなる。
- 計算量は O(系列長 × ラベル数²)。総当たりの O(ラベル数^系列長) を動的計画法で避けている。
- 遷移スコアに「O → I- は −∞」のような制約を入れれば、ニューラルモデルの出力に対しても正しい BIO 系列だけを出力させられる。

## BPE を 30 行で自作して仕組みを理解する {bpe_scratch}

Byte Pair Encoding が何をしているのか、最小限の実装で確かめたい。

```python
from collections import Counter

def learn_bpe(words, num_merges):
    # 各単語を文字列に分解し、語末記号 </w> を付ける
    vocab = Counter(tuple(w) + ("</w>",) for w in words)
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[a, b] += freq
        if not pairs:
            break
        best = max(pairs, key=pairs.get)      # 最頻の隣接ペアを 1 つの記号に併合
        merges.append(best)
        new_vocab = Counter()
        for symbols, freq in vocab.items():
            out, i = [], 0
            while i < len(symbols):
                if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == best:
                    out.append(symbols[i] + symbols[i + 1]); i += 2
                else:
                    out.append(symbols[i]); i += 1
            new_vocab[tuple(out)] += freq
        vocab = new_vocab
    return merges

def apply_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:                        # 学習した順に併合を適用
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols[i:i + 2] = [a + b]
            else:
                i += 1
    return symbols

corpus = "low low low low low lower lower newest newest newest newest newest newest widest widest widest".split()
merges = learn_bpe(corpus, 10)
print(merges)
for w in ["lowest", "newer", "wider"]:
    print(w, "→", apply_bpe(w, merges))
```

- 未知語 `lowest` が既知の部品 `low` + `est</w>` に分解される。これがサブワードで未知語がなくなる理由。

## n-gram 言語モデルを自分で実装してパープレキシティを測り、文を生成する {ngram_lm_scratch}
古典的な n-gram 言語モデルを学習して、文の「それらしさ」を数値化したい。文も生成してみたい。

```python
!pip install -q fugashi unidic-lite
```

```python
import io, re, urllib.request, zipfile
from fugashi import Tagger

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

tagger = Tagger()
text = aozora("https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip")
sents = [[w.surface for w in tagger(s)] for s in re.split(r"(?<=。)|\n", text) if s and s.strip()]
train, test = sents[:-300], sents[-300:]
print(len(train), len(test))
```

```python
from nltk.lm import KneserNeyInterpolated, Laplace, Vocabulary
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.util import ngrams

n = 3
models = {}
for name, cls in [("Laplace", Laplace), ("Kneser-Ney", KneserNeyInterpolated)]:
    data, words = padded_everygram_pipeline(n, train)
    lm = cls(n, vocabulary=Vocabulary(words, unk_cutoff=2))  # 1 回しか出ない語は <UNK> として学習
    lm.fit(data)
    models[name] = lm

def perplexity(lm, sentences):
    grams = [g for s in sentences for g in ngrams(pad_both_ends(lm.vocab.lookup(s), n=n), n)]
    return lm.perplexity(grams)

for name, lm in models.items():
    print(f"{name:11} テスト PPL = {perplexity(lm, test[:100]):.1f}")
```

```python
lm = models["Kneser-Ney"]
for s in ["吾輩は猫である。", "猫は吾輩である。", "である猫吾輩は。"]:
    print(f"{perplexity(lm, [[w.surface for w in tagger(s)]]):10.1f}  {s}")

for seed in range(3):
    out = lm.generate(30, text_seed=["<s>", "<s>"], random_seed=seed)
    print("".join(w for w in out if w not in ("<s>", "</s>")))
```

- 未知語があると確率 0 (PPL = ∞) になるので、低頻度語を `<UNK>` に置き換えて学習する (`unk_cutoff`)。
- スムージングで未知の n-gram の確率を 0 にしないのが肝。加算 (Laplace) より Kneser-Ney のほうがずっと良い（PPL が低い）。
- 本格的に使うなら KenLM（高速・省メモリ）。ニューラル言語モデルのパープレキシティは「言語モデルの確率と内部を調べる」の章を参照。
- 実務でライブラリを使って同じことをする版は、コーパスの統計と可視化の章の「KenLM で n-gram 言語モデルを作り、パープレキシティでコーパスのノイズを見つける」。

## PPMI 行列の SVD で単語ベクトルを作る（カウントベース） {ppmi_svd}

ニューラルネットを使わずに、共起行列から単語ベクトルを作りたい。word2vec が何をしているかの理解にも役立つ。

```python
import nltk
nltk.download("brown", quiet=True)
from collections import Counter
from nltk.corpus import brown
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

sents = [[w.lower() for w in s if w.isalpha()] for s in brown.sents()]
freq = Counter(w for s in sents for w in s)
vocab = [w for w, c in freq.most_common(8000)]
idx = {w: i for i, w in enumerate(vocab)}

window, cooc = 4, Counter()
for s in sents:
    ids = [idx[w] for w in s if w in idx]
    for i, a in enumerate(ids):
        for b in ids[max(0, i - window):i]:
            cooc[a, b] += 1
            cooc[b, a] += 1
print(len(cooc), "組の共起")
```

```python
rows, cols = zip(*cooc.keys())
C = csr_matrix((list(cooc.values()), (rows, cols)), shape=(len(vocab), len(vocab)), dtype=np.float64)
total, row_sum = C.sum(), np.asarray(C.sum(axis=1)).ravel()

coo = C.tocoo()
pmi = np.log(coo.data * total / (row_sum[coo.row] * row_sum[coo.col]))
ppmi = csr_matrix((np.maximum(pmi, 0), (coo.row, coo.col)), shape=C.shape)   # 負の値は 0 に

U, S, _ = svds(ppmi, k=100)
vecs = U * np.sqrt(S)
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9

def most_similar(word, topn=6):
    sims = vecs @ vecs[idx[word]]
    return [vocab[i] for i in np.argsort(-sims)[1:topn + 1]]

for w in ["money", "school", "monday", "walked"]:
    print(w, most_similar(w))
```

- skip-gram with negative sampling は、（シフトした）PMI 行列の行列分解を暗黙に行っていることが知られている (Levy & Goldberg, 2014)。

## 小さな GPT を日本語のテキストでゼロから事前学習する（学習ループを自分で書く） {gpt_scratch gpu}
事前学習済みのモデルを使うだけでなく、言語モデルがテキストから何を学んでいくのかを自分の手で確かめたい。日本語版 Wikipedia の一部でトークナイザーを作り、1,400 万パラメータほどの GPT をゼロから学習して、損失・パープレキシティと生成される文章の変化を見る。

```python
!pip install -q datasets
```

```python
import random
import numpy as np
import torch
from datasets import load_dataset

random.seed(0); np.random.seed(0); torch.manual_seed(0)

# 日本語版 Wikipedia（wikimedia/wikipedia の 20231101.ja。本文は CC BY-SA 3.0 と GFDL）
# ストリーミングで 25,000 記事を取り出し、短い記事を除いて、各記事の先頭 2,000 文字だけを使う（1 分ほど）
wiki = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
texts = [a["text"][:2000] for a in wiki.shuffle(seed=0, buffer_size=10000).take(25000) if len(a["text"]) >= 300]
train_texts, val_texts = texts[:-500], texts[-500:]   # 最後の 500 記事を検証用に取っておく
print(len(train_texts), "記事 (学習) /", len(val_texts), "記事 (検証)", f"学習データ {sum(map(len, train_texts)) / 1e6:.1f}M 文字")
print(train_texts[0][:100])
```

```python
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

# GPT-2 と同じバイト単位の BPE。どんな文字もバイト列に戻せるので、未知語 ([UNK]) が出ない
bpe = Tokenizer(models.BPE())
bpe.normalizer = normalizers.NFKC()                                   # 全角英数字などを半角にそろえる
bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
bpe.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=8000, special_tokens=["<|endoftext|>"],
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
bpe.train_from_iterator(train_texts[:8000], trainer)                  # 語彙は学習データの一部 8,000 記事から作る（Colab で 1 分以内）
tokenizer = PreTrainedTokenizerFast(tokenizer_object=bpe, bos_token="<|endoftext|>", eos_token="<|endoftext|>")

ids = tokenizer("吾輩は猫である。自然言語処理は人工知能の一分野である。")["input_ids"]
print(len(ids), "トークン:", [tokenizer.decode([i]) for i in ids])
```

```python
# 全記事をトークン ID にし、記事の間に <|endoftext|> を挟んでつなげ、256 トークンずつのブロックに切る
BLOCK = 256
def to_blocks(texts):
    ids = np.concatenate([x + [tokenizer.eos_token_id] for x in tokenizer(texts)["input_ids"]])
    n = len(ids) // BLOCK * BLOCK
    return torch.tensor(ids[:n], dtype=torch.long).view(-1, BLOCK), len(ids)

train_data, n_train = to_blocks(train_texts)
val_data, n_val = to_blocks(val_texts)
val_chars = sum(len(t) for t in val_texts)
print(f"学習 {n_train / 1e6:.1f}M トークン ({train_data.shape[0]} ブロック) / 検証 {n_val / 1e3:.0f}K トークン")
print(f"1 トークンあたり {val_chars / n_val:.2f} 文字")

# 比較の基準: 文脈を見ずにトークンの出現頻度だけで予測するモデル（ユニグラム）のパープレキシティ
counts = torch.bincount(train_data.flatten(), minlength=len(tokenizer)).float() + 1
unigram_loss = -(counts / counts.sum()).log()[val_data.flatten()].mean().item()
print(f"ユニグラムのパープレキシティ: {np.exp(unigram_loss):.0f} (語彙 {len(tokenizer)})")
```

```python
from transformers import GPT2Config, GPT2LMHeadModel

device = "cuda"
BF16 = torch.cuda.get_device_capability()[0] >= 8    # T4 は bf16 が使えないので float16 + 損失のスケーリング
config = GPT2Config(vocab_size=len(tokenizer), n_positions=BLOCK, n_embd=384, n_layer=6, n_head=6,
                    resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,           # データを 2 周ほどしか見ないのでドロップアウトなし
                    bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
torch.manual_seed(0)
model = GPT2LMHeadModel(config).to(device)            # from_pretrained ではなく、設定から重みがランダムなモデルを作る
n_params = sum(p.numel() for p in model.parameters())
n_emb = model.transformer.wte.weight.numel() + model.transformer.wpe.weight.numel()
print(f"{n_params / 1e6:.1f}M パラメータ (うち埋め込み {n_emb / 1e6:.1f}M。出力層は入力の埋め込みと重みを共有)")
```

```python
import math, time

STEPS, BATCH, LR, WARMUP = 5000, 32, 1e-3, 200
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / WARMUP, 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / STEPS))))
scaler = torch.amp.GradScaler(enabled=not BF16)
amp = dict(device_type="cuda", dtype=torch.bfloat16 if BF16 else torch.float16)

@torch.no_grad()
def evaluate(data, max_blocks=None):
    """検証データの 1 トークンあたりの平均損失（負対数尤度、自然対数）"""
    model.eval()
    losses = [model(input_ids=x.to(device), labels=x.to(device)).loss.item() for x in data[:max_blocks].split(64)]
    model.train()
    return float(np.mean(losses))

prompts = ["日本の首都は", "猫は"]
SAMPLING = dict(do_sample=True, temperature=0.8, top_p=0.95, top_k=0)
@torch.no_grad()
def sample(prompt, seed=0, **kwargs):
    """プロンプトの続きを 40 トークン生成する（kwargs は generate に渡すデコードの設定）"""
    model.eval()
    torch.manual_seed(seed)                      # 生成用の乱数（データの順番は下の data_gen で別に管理する）
    ids = tokenizer(tokenizer.bos_token + prompt, return_tensors="pt", return_token_type_ids=False).to(device)
    out = model.generate(**ids, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id, **kwargs)
    model.train()
    return tokenizer.decode(out[0, 1:], skip_special_tokens=True).replace("\n", " ")

log, samples, recent = [], {}, []        # log:（ステップ, 検証の損失, 直前の区間の学習の損失の平均, 経過秒）
step, t0 = 0, time.time()
data_gen = torch.Generator().manual_seed(0)   # データの順番専用の乱数。sample() の manual_seed の影響を受けない
model.train()
# Colab の T4 で 10〜15 分程度（検証の GPU での時間の 2〜3 倍として見積もった目安。T4 では測っていない。最初に出る loss_type=None の警告は無害）
while step < STEPS:
    for idx in torch.randperm(len(train_data), generator=data_gen).split(BATCH):          # 1 周（エポック）ごとにブロックの順番をシャッフル
        if step % 250 == 0 or step in (50, 100):
            train_loss = torch.stack(recent).mean().item() if recent else float("nan"); recent = []
            log.append((step, evaluate(val_data, 200), train_loss, time.time() - t0))   # 検証は先頭 200 ブロック（約 5 万トークン）で概算
            print(f"step {step:5}  学習の損失 {train_loss:.3f}  検証の損失 {log[-1][1]:.3f}  検証 PPL {math.exp(log[-1][1]):7.1f}")
        if step in (0, 100, 500, 1000, 2500):
            samples[step] = [sample(p, **SAMPLING) for p in prompts]
        if step == STEPS:
            break
        x = train_data[idx].to(device)
        with torch.autocast(**amp):
            loss = model(input_ids=x, labels=x).loss
        recent.append(loss.detach())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        step += 1
samples[STEPS] = [sample(p, **SAMPLING) for p in prompts]
val_loss = evaluate(val_data)
print(f"学習 {step} ステップ ({step * BATCH / len(train_data):.1f} エポック)、GPU メモリ {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
print(f"検証データ全体: 損失 {val_loss:.3f}  PPL {math.exp(val_loss):.1f}  1 文字あたり {val_loss * n_val / val_chars / math.log(2):.2f} bit")   # 文字数は正規化前の元の文字数
```

```python
import matplotlib.pyplot as plt

steps, losses, train_losses, _ = zip(*log)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(steps, losses, "o-", ms=3, label="GPT (validation)")
axes[0].plot(steps, train_losses, "-", alpha=0.6, label="GPT (train, mean of interval)")
axes[0].axhline(unigram_loss, color="gray", ls=":", label="unigram")
axes[0].axhline(math.log(len(tokenizer)), color="gray", ls="--", label="uniform (log V)")
axes[0].set_xlabel("step"); axes[0].set_ylabel("loss (nats / token)"); axes[0].legend()
axes[1].plot(steps, np.exp(losses), "o-", ms=3)
axes[1].axhline(np.exp(unigram_loss), color="gray", ls=":")
axes[1].set_yscale("log"); axes[1].set_xlabel("step"); axes[1].set_ylabel("perplexity (log scale)")
plt.tight_layout(); plt.show()
```

```python
# 学習の途中で生成した文章（同じ乱数の種で、temperature 0.8、top_p 0.95 のサンプリング）
val_at = {s: v for s, v, _, _ in log}          # ステップ → 検証の損失（先頭 200 ブロック）
for s, outs in samples.items():
    print(f"--- step {s} (検証 PPL {math.exp(val_at[s]):.0f})")
    for o in outs:
        print("  ", o)
```

```python
# 言語モデルの確率と内部を調べるの章の「言語モデルで文の自然さ（パープレキシティ）を測る」と同じ文で、自分で学習したモデルの確率を見る
@torch.no_grad()
def score(text):
    ids = tokenizer(tokenizer.bos_token + text, return_tensors="pt")["input_ids"].to(device)
    loss = model(ids, labels=ids).loss.item()
    n = ids.shape[1] - 1
    return -loss * n, n, math.exp(loss)

model.eval()
for text in ["吾輩は猫である。名前はまだ無い。", "吾輩は猫がある。名前をまだ無い。", "無いまだは名前。である猫は吾輩。"]:
    ll, n, ppl = score(text)
    print(f"対数尤度 {ll:8.2f}  トークン数 {n:2}  PPL {ppl:8.1f}  {text}")
pairs = [("太郎が本を読んだ。", "太郎が本が読んだ。"), ("私は昨日学校に行った。", "私は昨日学校に行く。"),
         ("3 冊の本を買った。", "3 匹の本を買った。"), ("雨が降ったので、試合は中止になった。", "雨が降ったのに、試合は中止になった。")]
for good, bad in pairs:
    print("○" if score(good)[0] > score(bad)[0] else "×", good, "/", bad)

# Transformers の基本操作の章の「テキスト生成のデコード方法（貪欲法・ビームサーチ・サンプリング）を比べる」と同じく、デコードの方法で出力が変わる
for label, kw in [("貪欲法", dict(do_sample=False)), ("貪欲法 + no_repeat_ngram_size=3", dict(do_sample=False, no_repeat_ngram_size=3)),
                  ("サンプリング (temperature 0.8)", SAMPLING)]:
    print(f"{label:30}", sample("日本の首都は", **kw))
```

- 言語モデルの事前学習は「次のトークンを当てる」分類の繰り返し。`labels=input_ids` を渡すと、1 つずらした交差エントロピー（言語モデルの確率と内部を調べるの章の「言語モデルで文の自然さ（パープレキシティ）を測る」と同じ平均負対数尤度）が返る。重みがランダムな学習前のモデルの検証の損失は 9.070 (PPL 8,692) で、語彙 8,000 から一様に選ぶ場合の log 8000 ≈ 8.99 にほぼ等しい。
- 50 ステップで検証 PPL が 1,687 と出現頻度だけのユニグラム (1,842) を下回り、5,000 ステップ（2.2 エポック）で検証データ全体の PPL は 41.9 になった（学習中の表示は先頭 200 ブロックでの概算）。学習の損失（直前 250 ステップの平均）は 1 周目を終えた直後の 2,500 ステップで検証の損失を下回り、3 周目に入った 4,582 ステップの後は、差が 4,500 ステップの 0.03 から最後の 0.14（学習 3.570、検証 3.709）に開いた。同じデータを繰り返し見ると過学習が進むので、さらに下げるにはデータとステップ数を一緒に増やす必要がある（この実験では確かめていない）。
- 生成文の変化: 学習前はでたらめなトークンの列（バイトの切れ端が `�` になる）、100 ステップでは「関連項目」「概」のような頻出の断片と空白が並ぶだけ、1,000 ステップで「東京都出身。父は…」のような人物記事の書式が現れる。5,000 ステップでは「日本の首都は、福岡県北九州市西区に本部を置く日本のプロサッカーリーグで、サッカー団体である。」と文の形は整うが、内容は誤っている。この規模では、知識より先に書式や文の形を覚える。
- バイト単位の BPE（GPT-2 と同じ方式）で語彙 8,000 にすると、1 トークンあたり 1.55 文字で、「吾輩」「人工知能」は 1 文字ずつに分かれた（トークナイザーの学習はサブワードとトークナイザの章の「Hugging Face tokenizers で BPE / WordPiece を学習する」も参照）。トークンあたりの PPL はトークナイザーで変わるので、分割の違うモデルと比べるときは 1 文字あたりのビット数（ここでは 3.47 bit）にそろえる。
- 言語モデルの確率と内部を調べるの章の「言語モデルで文の自然さ（パープレキシティ）を測る」と同じ 3 文では、PPL が正しい文 208.5、助詞を誤った文 396.7、語順を崩した文 1,373.6 の順になった。最小対は 4 組中 3 組で正しい文を選び、「昨日〜行った / 行く」の時制は外した。4 組では何も言えないので、系統的に調べるなら言語モデルの確率と内部を調べるの章の「言語モデルの文法知識を最小対で調べる (BLiMP / JBLiMP)」の方法を使う。
- 貪欲法では「東京都港区にある、」を繰り返すループに陥った。`no_repeat_ngram_size=3` で同じ並びの繰り返しは止まるが、「東京都港区と東京都港区の境界」のような形を変えた繰り返しは残る。小さなモデルほど繰り返しに陥りやすく、Transformers の基本操作の章の「テキスト生成のデコード方法（貪欲法・ビームサーチ・サンプリング）を比べる」のサンプリングの設定が効いてくる。
- 13.8M パラメータ（うち埋め込み 3.2M）、学習中の GPU メモリは 2.7 GB。検証の GPU（T4 より速いもの。T4 と同じ float16 の経路）で学習のセルは約 4 分（1 ステップ約 50ms）で、T4 ではこの 2〜3 倍かかると見込む。ここに書いた数値と生成文は検証の GPU での 1 回の結果で、GPU が違うと計算の細部が変わり、生成文や最小対の結果は別のものになりうる。
- 学習したトークンは延べ 4,100 万（重複しないのは 1,880 万）で、パラメータ数の約 3 倍（重複なしでは約 1.4 倍）。Chinchilla (Hoffmann et al., 2022) は 70B パラメータを 1.4 兆トークン（約 20 倍）で学習しており、それに比べるとデータがかなり少ない。エンコーダー・デコーダーをゼロから学習する例は、機械翻訳の章の「Transformer の翻訳モデルをゼロから学習する」にある。
- データは日本語版 Wikipedia（wikimedia/wikipedia の 20231101.ja、データセットカードによれば本文は GFDL と CC BY-SA 3.0）。読み込みはTransformers の基本操作の章の「巨大なデータセットをダウンロードせずにストリーミングで読む」と同じストリーミングで、`shuffle` はバッファ内の近似的なシャッフル。各記事の先頭 2,000 文字だけを使っているので、記事の冒頭（定義文）に偏っている。
- 実務でライブラリを使って同じことをする版は、Transformers の基本操作の章の「統制した自前のコーパスで小さな GPT を事前学習する (Trainer)」。

## SimCSE の損失と学習ループを自分で書いて、素の BERT から文埋め込みモデルを作る {simcse_scratch gpu}
文埋め込み用に学習されていない素の BERT から、ラベルなしの文だけで文埋め込みモデルを作りたい。同じ文をドロップアウトの違う 2 回で埋め込んで正例にする対照学習（教師なし SimCSE）を、損失から自分で書く。

```python
!pip install -q fugashi unidic-lite datasets
```

```python
import json, random, re, urllib.request
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging

logging.set_verbosity_error()   # モデルを読むたびに出る「使わない重み」の一覧を出さない
random.seed(0); np.random.seed(0); torch.manual_seed(0)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
DTYPE = torch.bfloat16 if BF16 else torch.float16

# 学習データ: 日本語 Wikipedia（wikimedia/wikipedia の 20231101.ja、CC BY-SA 3.0 / GFDL）の先頭の記事から文を切り出す
# streaming=True なので全体（約 4GB）はダウンロードしない
wiki = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
sents = []
for article in wiki:
    for line in article["text"].split("\n"):
        sents += [s for s in re.findall(r"[^。]+。", line) if 20 <= len(s) <= 100]
    if len(sents) >= 40000:
        break
random.shuffle(sents)
print(len(sents), "文", sents[:3])

# 評価データ: JGLUE の JSTS v1.3 のテストセット（1589 組、0〜5 の類似度、CC BY-SA 4.0）
url = "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jsts-v1.3/test-v1.3.json"
jsts = [json.loads(l) for l in urllib.request.urlopen(url).read().decode("utf-8").splitlines()]
print(len(jsts), "組", jsts[1])
```

```python
name = "tohoku-nlp/bert-base-japanese-v3"   # 東北大の日本語 BERT (Apache 2.0)。MLM で事前学習しただけで、文埋め込み用の学習はしていない
tokenizer = AutoTokenizer.from_pretrained(name)

def embed(model, texts, max_length=64):
    """平均プーリング（文埋め込みとクラスタリングの章の「BERT の出力から文ベクトルを自分で作る (mean pooling)」と同じ）"""
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=DTYPE):
        hidden = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    return (hidden.float() * mask).sum(1) / mask.sum(1)

@torch.no_grad()
def jsts_cos(model):
    model.eval()   # 評価ではドロップアウトを切る
    e1 = torch.cat([embed(model, [d["sentence1"] for d in jsts[i:i + 256]]) for i in range(0, len(jsts), 256)])
    e2 = torch.cat([embed(model, [d["sentence2"] for d in jsts[i:i + 256]]) for i in range(0, len(jsts), 256)])
    return F.cosine_similarity(e1, e2).cpu().numpy()

gold = np.array([d["label"] for d in jsts])
jsts_spearman = lambda model: spearmanr(jsts_cos(model), gold).statistic   # 文埋め込みとクラスタリングの章の「文の意味的類似度 (STS) でモデルを評価する」と同じ指標

bert = AutoModel.from_pretrained(name).to("cuda")
print(f"学習前の BERT (平均プーリング): JSTS Spearman = {jsts_spearman(bert):.4f}")
```

```python
def simcse_loss(z1, z2, temperature=0.05, z3=None):
    """InfoNCE。z1[i] と z2[i] が正例、z2[j] (j≠i) はすべて負例。z3 を渡すと z3[j] も負例（ハードネガティブ）に加える"""
    z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    sim = z1 @ z2.T                                         # (N, N) のコサイン類似度。対角が正例
    if z3 is not None:
        sim = torch.cat([sim, z1 @ F.normalize(z3, dim=-1).T], dim=1)   # (N, 2N)
    labels = torch.arange(len(z1), device=z1.device)        # i 行目の正解は i 列目
    return F.cross_entropy(sim / temperature, labels)       # 「N 個（2N 個）の候補から正例を当てる」分類の損失

def train(examples, batch_size=64, temperature=0.05, lr=3e-5, supervised=False, seed=0):
    torch.manual_seed(seed)
    model = AutoModel.from_pretrained(name).to("cuda").train()   # train() でドロップアウト (p=0.1) が有効になる
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = len(examples) // batch_size
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / steps)   # 学習率を線形に 0 まで下げる
    scaler = torch.amp.GradScaler(enabled=not BF16)                          # float16 の勾配のアンダーフロー対策
    for s in range(steps):
        batch = examples[s * batch_size:(s + 1) * batch_size]
        if supervised:   # （前提, 含意, 矛盾）の三つ組を 1 回の順伝播で埋め込む
            z = embed(model, [a for a, _, _ in batch] + [p for _, p, _ in batch] + [n for _, _, n in batch], max_length=32)
            z1, z2, z3 = z.chunk(3)
        else:            # 同じ文を 2 回並べる。ドロップアウトのマスクが行ごとに違うので、2 つの埋め込みは少しずれる
            z1, z2 = embed(model, batch + batch, max_length=32).chunk(2)   # 学習は 32 トークンで切る（論文と同じ）
            z3 = None
        loss = simcse_loss(z1, z2, temperature, z3)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if s % 100 == 0 or s == steps - 1:
            print(f"step {s:4d}  loss {loss.item():.3f}")
    return model

# 12800 文、バッチ 64 で 200 ステップ。乱数の種を変えて 3 回学習する（Colab の T4 で 2〜4 分の見込み）
scores = []
for seed in range(3):
    simcse = train(sents[:12800], seed=seed)
    scores.append(jsts_spearman(simcse))
    print(f"教師なし SimCSE (seed={seed}): JSTS Spearman = {scores[-1]:.4f}")
print(f"平均 {np.mean(scores):.4f} (最小 {min(scores):.4f}、最大 {max(scores):.4f})")
```

```python
# 学習前後で、文の組の類似度がどう変わるか（simcse は上で最後に学習した seed=2 のモデル）
pairs = [("猫がソファの上で寝ている。", "ソファで猫が眠っています。"),
         ("猫がソファの上で寝ている。", "犬が公園を走り回っている。"),
         ("猫がソファの上で寝ている。", "株価が大きく値下がりした。")]
bert.eval(); simcse.eval()   # ドロップアウトを切る
with torch.no_grad():
    for a, b in pairs:
        r = [F.cosine_similarity(embed(m, [a]), embed(m, [b])).item() for m in (bert, simcse)]
        print(f"学習前 {r[0]:.3f}  学習後 {r[1]:.3f}  {a} / {b}")

# ランダムな文の組の類似度の平均: 素の BERT はどの文同士も似ている（異方性）
with torch.no_grad():
    for label, m in [("学習前", bert), ("学習後", simcse)]:
        e = F.normalize(embed(m, sents[-512:]), dim=-1)
        print(f"{label}: ランダムな文の組のコサイン類似度の平均 = {(e @ e.T)[~torch.eye(512, dtype=bool)].mean():.3f}")
```

```python
# 温度とバッチサイズを変えてみる（それぞれ 12800 文、seed=0。温度 0.05・バッチ 64 は上の seed=0 と同じ設定。Colab の T4 で 2〜4 分の見込み）
for t, b in [(0.01, 64), (1.0, 64), (0.05, 256)]:
    print(f"温度 {t:<4} バッチ {b:<3}: JSTS Spearman = {jsts_spearman(train(sents[:12800], batch_size=b, temperature=t)):.4f}")
```

```python
# 教師あり SimCSE: JSNLI（SNLI の日本語訳、京都大学、CC BY-SA 4.0）の「含意」を正例、「矛盾」をハードネガティブにする
import io, zipfile
url = "https://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi?down=https://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JSNLI/jsnli_1.1.zip&name=JSNLI.zip"
with urllib.request.urlopen(url) as r:
    z = zipfile.ZipFile(io.BytesIO(r.read()))
rows = [l.split("\t") for l in z.read("jsnli_1.1/train_w_filtering.tsv").decode("utf-8").splitlines()]
unspace = lambda s: re.sub(r"(?<![A-Za-z0-9]) | (?![A-Za-z0-9])", "", s.strip())   # 単語区切りの空白を取る
hyp = {}
for label, premise, hypothesis in rows:
    hyp.setdefault(unspace(premise), {}).setdefault(label, unspace(hypothesis))
triples = [(p, h["entailment"], h["contradiction"]) for p, h in hyp.items() if "entailment" in h and "contradiction" in h]
random.shuffle(triples)
print(len(triples), "組", triples[0])

# 16000 組、バッチ 64 で 250 ステップ（ダウンロードを含めて Colab の T4 で 2〜3 分の見込み）
sup = train(triples[:16000], lr=5e-5, supervised=True)
print(f"教師あり SimCSE: JSTS Spearman = {jsts_spearman(sup):.4f}")
```

```python
# 差がばらつきの範囲かどうか: 評価の 1589 組をブートストラップで再標本化して、Spearman の差の 95% 信頼区間を求める
cos = {"学習前": jsts_cos(bert), "教師なし": jsts_cos(simcse), "教師あり": jsts_cos(sup)}
idx = np.random.default_rng(0).integers(0, len(gold), (1000, len(gold)))
for a, b in [("学習前", "教師なし"), ("学習前", "教師あり")]:
    d = [spearmanr(cos[b][i], gold[i]).statistic - spearmanr(cos[a][i], gold[i]).statistic for i in idx]
    print(f"{b} − {a}: {np.mean(d):+.4f}  95% 信頼区間 [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
```

```python
# 自作の損失は、Sentence Transformers の MultipleNegativesRankingLoss（scale=20 = 1/温度 0.05）と同じ値になる
from sentence_transformers.losses import MultipleNegativesRankingLoss

z1, z2, z3 = torch.randn(3, 8, 768)
mnrl = MultipleNegativesRankingLoss(model=None)
print("自作:", simcse_loss(z1, z2).item(), " MNRL:", mnrl.compute_loss_from_embeddings([z1, z2], None).item())
print("自作 (ハードネガティブあり):", simcse_loss(z1, z2, z3=z3).item(), " MNRL:", mnrl.compute_loss_from_embeddings([z1, z2, z3], None).item())
print(f"最大メモリ {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
```

- 以下の数値は、検証の GPU で float16 の経路（`BF16 = False`、T4 と同じ dtype）で実行したもの（最大メモリ 10.1GB）。学習の最初に出る `lr_scheduler.step()` の警告は、float16 の GradScaler が勾配のオーバーフローでそのステップを飛ばしたときに出るもので、害はない。
- SimCSE は Gao, Yao, Chen (EMNLP 2021) の手法。教師なしでは、同じ文を 2 回エンコードしたときのドロップアウト (p=0.1) のマスクの違いだけがデータ拡張になり、バッチ内の他の文が負例になる。論文の教師なしは英語 Wikipedia の 100 万文、バッチ 64、学習率 3e-5、温度 0.05 で、[CLS] に MLP を載せて学習し、評価時は MLP を外す。教師ありは SNLI と MNLI（論文の記載で 31.4 万組）、バッチ 512、学習率 5e-5、3 エポック、[CLS] + MLP である。このレシピは平均プーリングで MLP を使わず、教師なしは 1 万 2800 文、教師ありはバッチ 64 で 1 万 6000 組（1 エポック未満）に縮めた。論文は STS-B の開発セットで最良のチェックポイントを選んでいるが、ここでは選んでいない（教師なしの損失は 100 ステップで 0.001 まで下がっている）。
- JSTS のテストセット（1589 組）の Spearman は、学習前の平均プーリングで 0.7536、教師なし SimCSE で 0.756〜0.763（乱数の種 3 つ、平均 0.760）になり、学習前をわずかに上回った。ただし seed=2 と学習前の差 (+0.0065) のブートストラップの 95% 信頼区間（−0.007〜+0.020）は 0 をまたぎ、この規模ではばらつきと区別できない。種を変えてもデータの順序は同じで、変わるのはドロップアウトの乱数だけなので、この幅は学習のばらつきを小さめに見せている可能性がある。日本語 SimCSE の報告（Tsukagoshi ら 2023）は、約 100 万文で学習し、ハイパーパラメータとチェックポイントを開発セット (JSICK) で選んで、同じ BERT で JSTS の検証セット（このレシピのテストセットとは別）が 74.09（学習前、平均プーリング）から 78.70（5 回の平均）に上がったと報告している。
- 類似度の値の分布は大きく変わる。ランダムな文の組のコサイン類似度の平均は 0.834 から 0.177 に、無関係な「猫が…」と「株価が…」は 0.755 から 0.119 に下がった。文埋め込みとクラスタリングの章の「BERT の出力から文ベクトルを自分で作る (mean pooling)」で、素の BERT の平均ベクトルは類似度計算にあまり向かないとしたのはこの点で、順位 (Spearman) はそこそこでも、どの文の組も類似度が高く出る（異方性）ので、値をそのまま使ったりしきい値を決めたりするのに向かない。JSTS は短い画像の説明文の組なので、単語の重なりで判断しやすく、素の BERT に有利な面もある。
- 温度を 1.0 にすると、損失は 3.2 付近で下げ止まり（ln 64 ≈ 4.16 に近いまま）、Spearman は 0.7246 と学習前より悪くなった。温度 0.01 (0.7675) とバッチ 256 (0.7711) は温度 0.05・バッチ 64 の seed=0 (0.7556) を上回ったが、どれも種 1 つだけで、種による幅も 0.008 あるので優劣は言えない。同じ種でも GPU の演算の非決定性で値が変わることがある。学習率は 3e-5 のまま変えていない。論文の付録は、学習率を合わせればバッチサイズにはあまり敏感でない、としている。
- 教師あり SimCSE（JSNLI の含意を正例、矛盾をハードネガティブ）は 0.8256 で、学習前との差は +0.072（95% 信頼区間 +0.054〜+0.090。評価の組を復元抽出するブートストラップで求めた。評価指標と統計的検定の章の「ブートストラップ法で評価値の信頼区間を求める」を参照）だった。NLI のラベルが「同じ意味か」の教師信号になるので、論文でも Tsukagoshi らでも教師ありは教師なしより高い。JSNLI は SNLI を機械翻訳して自動で絞り込んだものである。また SNLI の元は Flickr30k の画像の説明文、JSTS は YJ Captions（MS COCO の画像の日本語説明文）で、分野が近いことも効いている可能性がある。
- 検索の章の「検索用の埋め込みモデルを自分のデータで微調整する」で使う `MultipleNegativesRankingLoss` は、既定（コサイン類似度、scale=20 = 温度 0.05）でこの `simcse_loss` と同じ値になる（最後のセル）。Sentence Transformers の説明でも、既定の設定は InfoNCE や SimCSE の損失とも呼ばれるとしている。違うのは正例の組の作り方だけで、検索の章は（クエリ, 文書）、教師なし SimCSE は（文, 同じ文）、教師ありは（前提, 含意）に矛盾を足した三つ組である。Sentence Transformers で教師なし SimCSE をするなら、anchor と positive に同じ文を入れて学習すればよい。
- データとモデルの出典とライセンス: 日本語 Wikipedia（wikimedia/wikipedia、CC BY-SA 3.0 / GFDL）、JGLUE の JSTS（ヤフー、CC BY-SA 4.0）、JSNLI（京都大学、CC BY-SA 4.0）、tohoku-nlp/bert-base-japanese-v3 (Apache 2.0)。ハイパーパラメータや学習の打ち切りを決めるときは、テストセットではなく JSTS の学習セットなど別の開発データで選ぶ。
- 実務でライブラリを使って同じことをする版は、文埋め込みとクラスタリングの章の「文埋め込み用でない BERT を対照学習で文埋め込みモデルにする（SimCSE、Sentence Transformers）」。

## ラベル伝播と相互 k 近傍グラフを自分で実装して、少数のラベルを広げる {label_propagation_scratch}
ラベル付きの文書がごく少ししかないとき、ラベルのない大量の文書も使って分類の精度を上げたい。

```python
!pip install -q scikit-learn fugashi unidic-lite
```

```python
import io, re, tarfile, urllib.request
import numpy as np
import pandas as pd
from fugashi import Tagger
from sklearn.feature_extraction.text import TfidfVectorizer

def load_livedoor():
    url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
    with urllib.request.urlopen(url) as r:
        tar = tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")
    rows = []
    for m in tar.getmembers():
        parts = m.name.split("/")
        if m.isfile() and len(parts) == 3 and parts[2] not in ("LICENSE.txt", "README.txt"):
            lines = tar.extractfile(m).read().decode("utf-8").split("\n")
            rows.append({"label": parts[1], "text": lines[2] + "\n" + "\n".join(lines[3:])[:600]})
    return pd.DataFrame(rows)

df = load_livedoor().sample(3000, random_state=0).reset_index(drop=True)
tagger = Tagger()
tokenize = lambda t: [w.surface for w in tagger(t) if w.feature.pos1 in ("名詞", "動詞", "形容詞")]
X = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, min_df=3, max_df=0.5, sublinear_tf=True).fit_transform(df["text"])
y = pd.factorize(df["label"])[0]
print(X.shape, df["label"].nunique(), "クラス")
```

```python
import scipy.sparse as sp

S = (X @ X.T).toarray()                    # TF-IDF は正規化済みなので、内積がコサイン類似度
np.fill_diagonal(S, 0)

def knn_graph(S, k, mutual):
    """各文書から類似度の高い k 個に辺を張る。mutual=True なら、互いに k 近傍に入っている組だけを残す"""
    n = len(S)
    idx = np.argsort(-S, axis=1)[:, :k]
    A = sp.csr_matrix((np.ones(n * k), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    A = A.multiply(A.T) if mutual else A.maximum(A.T)
    return sp.csr_matrix(A.multiply(S))    # 辺の重みは類似度

for k in [10, 30]:
    for mutual in [False, True]:
        A = knn_graph(S, k, mutual)
        degree = np.asarray((A > 0).sum(axis=1)).ravel()
        same = (y[A.nonzero()[0]] == y[A.nonzero()[1]]).mean()
        print(f"k={k} {'相互 k 近傍' if mutual else 'k 近傍'}: 辺 {A.nnz // 2} / 次数の最大 {degree.max()} / 孤立した文書 {(degree == 0).sum()} / 同じクラスどうしを結ぶ辺の割合 {same:.3f}")
```

```python
def label_propagation(A, y, labeled, alpha=0.9, iterations=50):
    """ラベル拡散法: 隣の文書のスコアを受け取ることを繰り返す（alpha は隣からもらう割合）"""
    d = np.asarray(A.sum(axis=1)).ravel()
    d_inv = np.where(d > 0, 1 / np.sqrt(np.maximum(d, 1e-12)), 0)
    W = sp.diags(d_inv) @ A @ sp.diags(d_inv)
    Y = np.zeros((len(y), y.max() + 1)); Y[labeled, y[labeled]] = 1
    F = Y.copy()
    for _ in range(iterations):
        F = alpha * (W @ F) + (1 - alpha) * Y
    return F.argmax(axis=1)

from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)
results = []
for per_class in [2, 5, 20]:
    for trial in range(5):
        labeled = np.concatenate([rng.choice(np.where(y == c)[0], per_class, replace=False) for c in range(y.max() + 1)])
        rest = np.setdiff1d(np.arange(len(y)), labeled)
        row = {"1 クラスあたりのラベル数": per_class,
               "ロジスティック回帰 (ラベル付きのみ)": (LogisticRegression(C=10, max_iter=1000).fit(X[labeled], y[labeled]).predict(X[rest]) == y[rest]).mean()}
        for name, mutual in [("ラベル伝播 (k 近傍)", False), ("ラベル伝播 (相互 k 近傍)", True)]:
            row[name] = (label_propagation(knn_graph(S, 30, mutual), y, labeled)[rest] == y[rest]).mean()
        results.append(row)
pd.DataFrame(results).groupby("1 クラスあたりのラベル数").mean().round(3)
```

```python
# k 近傍グラフでは、多くの文書の近傍に入る「ハブ」ができる。ハブの辺の多くは一方向（相手の k 近傍にハブが入っているだけ）で、
# 一方向の辺は、互いに近傍である辺より別のクラスどうしを結びやすい。相互 k 近傍グラフは一方向の辺を捨てる
A = knn_graph(S, 30, mutual=False)
M = knn_graph(S, 30, mutual=True)
rows, cols = sp.triu(A).nonzero()
kept = np.asarray(M[rows, cols]).ravel() > 0
same = y[rows] == y[cols]
degree = np.asarray((A > 0).sum(axis=1)).ravel()
print(f"同じクラスどうしを結ぶ辺の割合: 互いに近傍の辺 {same[kept].mean():.3f} / 一方向の辺 {same[~kept].mean():.3f} / 次数の中央値 {np.median(degree):.0f}")

own = np.argsort(-S, axis=1)[:, :30]           # 各文書自身の k 近傍
for h in np.argsort(-degree)[:5]:
    neighbors = A[h].nonzero()[1]
    one_way = np.setdiff1d(neighbors, own[h])   # ハブを k 近傍に入れているが、ハブの k 近傍には入らない文書
    print(f"次数 {degree[h]} / クラス {df['label'][h]} / 同じクラスの割合: 自身の k 近傍 {np.mean(y[own[h]] == y[h]):.2f}, 一方向の近傍 {len(one_way)} 件 {np.mean(y[one_way] == y[h]):.2f}"
          f" / 別のクラスの近傍 {(y[neighbors] != y[h]).sum()} 件 | {df['text'][h].splitlines()[0][:30]}")
```

- グラフに基づく半教師あり学習は、「似ている文書は同じラベルを持つ」という仮定で、少数のラベルをグラフの辺に沿って広げる。ラベルが極端に少ないときに効果が大きい。
- 結果はグラフの作り方に強く左右される。普通の k 近傍グラフでは、多くの文書の近傍に入る「ハブ」ができる。この実験のハブは、近傍の大半が同じクラスだが、次数が大きいぶん、別のクラスの文書も数十件つながり、ラベルの通り道になる。ハブの辺の多くは片方の k 近傍にだけ入る一方向の辺で、一方向の辺は、互いに近傍である辺より別のクラスどうしを結ぶ割合が高い。
- 相互 k 近傍グラフ（互いに近傍である組だけを結ぶ）はハブができず、同じクラスどうしを結ぶ辺の割合が高い。その代わり、どこにもつながらない孤立点ができる。この実験では、分類の正解率の差は小さく、ラベル数によって優劣が入れ替わった。
- ラベルが増えると、普通の教師あり学習との差は縮まり、1 クラス 20 件では逆転している。数十件以上のラベルがあるなら、まずは事前学習済みモデルのファインチューニングや、埋め込み + 分類器（テキスト分類の章の「文埋め込みを特徴量にして少ない学習データで分類する」）を試す。
- 類似度に文埋め込みを使えば、同じ方法をそのまま適用できる。高次元の埋め込みでもハブは生じる（ハブネス問題）。
- 少数の例から語の集合を広げるブートストラッピングでも、ハブ（一般的な語）を通って無関係な語に広がる「意味ドリフト」が起きる（語の意味と意味の変化の章の「ブートストラッピングの意味ドリフトを観察して抑える」）。
- 実務でライブラリを使って同じことをする版は、テキスト分類の章の「少数のラベルをグラフで広げて分類する（scikit-learn の LabelSpreading）」。

## LLM の出力に入れる透かし（green リスト）を自分で実装し、統計的に検出する {llm_watermark_scratch gpu}
LLM が生成した文章に、読んでもわからない「透かし」（ウォーターマーク）を入れておき、あとから文章だけを見て、自分のモデルが生成したものかを統計的に判定したい。

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
    元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）"""
    sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
    tokenizer.backend_tokenizer.model = models.Unigram(
        [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
    return tokenizer

tokenizer = fix_sarashina_tokenizer(AutoTokenizer.from_pretrained(name), name)   # 詳しくは大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」の注意書き
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(name, dtype=DTYPE, device_map="cuda").eval()

KEY = 15485863   # 透かしの鍵。transformers の WatermarkingConfig の既定値で公開されている値。実際には乱数で選んで秘密にする

def is_green(prev, cur, gamma, key=KEY):
    """直前のトークン prev と鍵から決まる擬似乱数で、次のトークン cur が green かどうかを決める（語彙の約 γ が green）"""
    x = (prev * 1_000_003 + cur + key) & 0xFFFFFFFF
    for _ in range(3):                                   # 整数のハッシュ（ビットをよく混ぜる）
        x = ((x ^ (x >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    return x.double() / 2**32 < gamma

class GreenListWatermark(LogitsProcessor):
    """green のトークンのロジットに δ を足して、green のトークンを選ばれやすくする"""
    def __init__(self, gamma, delta):
        self.gamma, self.delta = gamma, delta
    def __call__(self, input_ids, scores):
        vocab = torch.arange(scores.shape[1], device=scores.device)
        green = is_green(input_ids[:, -1:], vocab[None, :], self.gamma)    # （バッチ, 語彙）の green / red の表
        return scores + self.delta * green.to(scores.dtype)

def hf_encode(text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]

def detect(text, gamma, encode=hf_encode):
    """文章だけから green のトークンの割合を数え、z 検定をする（モデルもプロンプトも要らない。鍵と、生成時と同じトークン化とハッシュが要る）"""
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
# 60 件ずつの z スコアの分布を比べる。z > 4 なら「透かしあり」と判定する（偶然 z > 4 になる確率は約 3e-5）
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

raw_tokenizer = AutoTokenizer.from_pretrained(name)          # 比較用: 直さない（transformers 5 のまま）トークナイザー
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
# δ と γ を変えて、検出のしやすさと文章の質（大きなモデルで測ったパープレキシティ）の釣り合いを見る。Colab の T4 で 5〜10 分程度（3B のモデルのダウンロードを含む）
oracle_name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"   # 評価用の大きなモデル（SB Intuitions、MIT ライセンス）
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

detector = WatermarkDetector(model_config=model.config, device=model.device, watermarking_config=wm_config)   # 生成と同じデバイスで（乱数がデバイスで変わる）
def builtin_z(texts):
    return np.array([detector(torch.tensor([tokenizer(t, add_special_tokens=False)["input_ids"]], device=model.device), return_dict=True).z_score[0] for t in texts])
print(f"組み込みの透かし → 組み込みの検出器: z の平均 {builtin_z(builtin).mean():.2f} / 自作の検出器: {np.mean([detect(t, 0.25) for t in builtin]):.2f}")
print(f"自作の透かし     → 組み込みの検出器: z の平均 {builtin_z(marked[:20]).mean():.2f} / 自作の検出器: {z_marked[:20].mean():.2f}")
print(f"GPU メモリの最大使用量: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```



- Kirchenbauer らの方法 (Kirchenbauer et al., 2023, "A Watermark for Large Language Models")。検出には、鍵と、生成時と同じトークン化とハッシュの方式があればよく、モデルもプロンプトも要らない。z が 4 を超えたら透かしありとする（論文の基準。透かしのない文章が偶然超える確率は約 3e-5）。原論文は直前のトークンのハッシュを乱数の種にして green のトークンを γ|V| 個選ぶが、ここでは（直前のトークン, 次のトークン, 鍵）のハッシュ値で直接決める簡略版にした（green の数は平均して γ|V|）。
- 60 件（約 200 トークン）で、透かしなしの z は平均 0.15（最大 2.33）、γ = 0.25、δ = 2 の透かしありは平均 5.32 で、z > 4 は 88% だった。60 件なので、割合には ±10 ポイント余りの幅がある。
- 検出は文章をトークン化し直して行うので、生成時と分割が同じでないと透かしが弱まる。sarashina2.2 のトークナイザーを transformers 5 のまま使うと分割が大きく変わり（大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」の注意書き）、z の平均は 3.86、z > 4 は 45% まで下がった。直したトークナイザーでは 5.32 で、生成したトークン列のままの 5.43 とほとんど変わらない。それでも同じトークン列に戻ったのは 43% で（「読み取り / やすい」→「を読み / 取りやすい」のように、同じ文字列に分割が複数ありうる）、その分だけ目減りは残る。
- δ を大きくすると検出しやすくなるが、文章の質が落ちる。γ = 0.25 で δ を 0, 1, 2, 4, 8 と上げると、z の平均は 0.15, 2.57, 5.32, 11.29, 18.33、3B のモデルで測ったパープレキシティの中央値は 9.5, 10.6, 11.4, 23.2, 82.6 になった。同じ δ では γ = 0.25 のほうが γ = 0.5 より z が高い。パープレキシティは同じ系列のモデルで測ったので、目安と考える。
- 短い文章では検出できない。z は green の割合が同じでも √T に比例するので、先頭 100 文字（約 44 トークン）に切ると z > 4 は 7%、200 文字（約 89 トークン）で 35% だった。また、次のトークンがほぼ決まっている箇所（エントロピーが低い箇所）では、δ を足しても選ばれるトークンが変わらず、透かしが入りにくい（原論文）。
- 言い換えに弱い。透かしのない 3B のモデルで半分の文を言い換えると z の平均は 3.35（z > 4 は 27%）、全部だと 1.03 (2%) になった。上の例のように元の語句を多く残していても、隣り合うトークンの組が変われば判定も変わる。同じグループの続編の研究 (Kirchenbauer et al., ICLR 2024) は、言い換えた文章にも元の断片が残るので、長くすれば検出できると報告している（人が強く言い換えた場合で、偽陽性率 1e-5 には平均 800 トークン程度）。透かしの入っていない文章を、文章の統計だけで見分ける方法は、LLM の評価と分析の章の「LLM が生成した文章を見分ける（パープレキシティ、Binoculars、分類器）」を参照。
- transformers には同じ方式の `WatermarkingConfig` / `WatermarkDetector` が組み込まれている（SynthID Text 用の `SynthIDTextWatermarkingConfig` もある）。上の `KEY` は `hashing_key` の既定値と同じだが、green の決め方が違うので互いに検出できない（z の平均 0.15 と −0.32）。透かしは生成する側が協力して入れておく仕組みで、生成した側の協力なしに AI が書いた文章かどうかを推定する方法とは別物である。
- 実務でライブラリを使って同じことをする版は、LLM の評価と分析の章の「LLM の出力に透かしを入れて検出する（transformers の WatermarkingConfig と SynthID）」。

## 遷移型の依存構造解析の動きを追う (arc-standard) {transition_parsing}

スタックとバッファを使って、左から 1 語ずつ読みながら依存構造木を組み立てる「遷移型解析」の仕組みを、手を動かして理解したい。

```python
def oracle(words, heads):
    """正解の木 (heads[i] = i 番目の語の主辞。0 が ROOT、語は 1 始まり) を作るための遷移の列を求める"""
    n = len(words)
    n_children = [sum(1 for h in heads if h == i) for i in range(n + 1)]   # まだ付けていない子の数
    stack, buffer, steps, arcs = [0], list(range(1, n + 1)), [], []
    name = lambda i: "ROOT" if i == 0 else words[i - 1]
    while buffer or len(stack) > 1:
        action = "SHIFT"
        if len(stack) >= 2:
            s1, s0 = stack[-2], stack[-1]
            if s1 != 0 and heads[s1 - 1] == s0 and n_children[s1] == 0:
                action = "LEFT-ARC"       # スタックの 2 番目が、一番上の語に係る
            elif heads[s0 - 1] == s1 and n_children[s0] == 0:
                action = "RIGHT-ARC"      # スタックの一番上が、2 番目の語に係る
        if action == "SHIFT" and not buffer:
            raise ValueError("この木は arc-standard では作れません (交差する係り受けがある)")
        steps.append((action, [name(i) for i in stack], [name(i) for i in buffer]))
        if action == "SHIFT":
            stack.append(buffer.pop(0))
        elif action == "LEFT-ARC":
            arcs.append((name(s0), name(s1))); n_children[s0] -= 1; del stack[-2]
        else:
            arcs.append((name(s1), name(s0))); n_children[s1] -= 1; stack.pop()
    return steps, arcs

words = ["太郎", "が", "赤い", "本", "を", "読ん", "だ"]
heads = [6, 1, 4, 6, 4, 0, 6]          # 太郎→読ん, が→太郎, 赤い→本, 本→読ん, を→本, 読ん→ROOT, だ→読ん
steps, arcs = oracle(words, heads)
for action, stack, buffer in steps:
    print(f"{action:10} スタック: {' '.join(stack):24} バッファ: {' '.join(buffer)}")
print("作られた係り受け (主辞 → 従属語):", arcs)
```

```python
# 交差する係り受け (非交差性の違反) があるかを調べる
def is_projective(heads):
    arcs = [(min(i, h), max(i, h)) for i, h in enumerate(heads, 1)]
    return not any(a < c < b < d for a, b in arcs for c, d in arcs)

print(is_projective(heads))
# "A hearing is scheduled on the issue today": on(5) が hearing(2) に係り、is scheduled をまたぐ
en_words = ["A", "hearing", "is", "scheduled", "on", "the", "issue", "today"]
en_heads = [2, 4, 4, 0, 7, 7, 2, 4]
print(is_projective(en_heads))
try:
    oracle(en_words, en_heads)
except ValueError as e:
    print(e)
```

- 長さ n の文は、ちょうど 2n 回の遷移（SHIFT が n 回、ARC が n 回）で解析できる。各時点でどの遷移を選ぶかを分類器で予測するのが遷移型パーザで、spaCy の parser もこの方式。
- 学習時は、ここで求めた「正解の遷移列」を教師データにする（スタックとバッファの状態 → 次の遷移）。
- 従属語をすべて付け終えてからでないと RIGHT-ARC できない（`n_children` の確認）。早く付けてしまうと、その語の子を付けられなくなる。
- 素の arc-standard は交差する係り受けを作れない。日本語の係り受けはほぼ交差しないが、チェコ語やドイツ語などでは無視できない頻度で現れるので、SWAP 遷移の追加などの拡張が使われる。

## スコアが最大の依存構造木を求める（最大全域木） {mst_decoding}

「どの語がどの語に係りそうか」のスコアが全ペアについて与えられたとき、全体として最も良い木を選びたい。グラフに基づく解析の復号を理解したい。

```python
import numpy as np

words = ["ROOT", "John", "saw", "Mary"]
# score[h, d] = 語 h が語 d の主辞である、というスコア (実際のパーザではニューラルネットが出力する)
score = np.array([
    [-np.inf,  9, 10,  9],    # ROOT → John / saw / Mary
    [-np.inf, -np.inf, 20,  3],    # John →
    [-np.inf, 30, -np.inf, 30],    # saw  →
    [-np.inf, 11,  0, -np.inf],    # Mary →
])

# 貪欲法: 各語について、スコアが最大の主辞を独立に選ぶ
greedy = {words[d]: words[int(score[:, d].argmax())] for d in range(1, len(words))}
print("貪欲法:", greedy)   # John → saw → John の循環ができて、木にならない
```

```python
import networkx as nx

def mst(score):
    """Chu-Liu/Edmonds のアルゴリズムで、ROOT (0 番) を根とする最大全域有向木を求める"""
    G = nx.DiGraph()
    n = len(score)
    for h in range(n):
        for d in range(1, n):
            if h != d and np.isfinite(score[h, d]):
                G.add_edge(h, d, weight=float(score[h, d]))
    tree = nx.maximum_spanning_arborescence(G)
    heads = {d: h for h, d in tree.edges}
    return [heads[d] for d in range(1, n)], sum(score[heads[d], d] for d in range(1, n))

heads, total = mst(score)
print("最大全域木:", {words[d]: words[h] for d, h in enumerate(heads, 1)}, "合計スコア:", total)
```

```python
# 総当たりで確かめる: すべての主辞の割り当てのうち、木になっているものの中で最大のもの
from itertools import product

def is_tree(heads):
    for d in range(1, len(heads) + 1):
        seen, cur = set(), d
        while cur != 0:
            if cur in seen:
                return False
            seen.add(cur); cur = heads[cur - 1]
    return True

best = max((h for h in product(range(4), repeat=3) if all(h[d - 1] != d for d in range(1, 4)) and is_tree(h)),
           key=lambda h: sum(score[h[d - 1], d] for d in range(1, 4)))
print("総当たり:", list(best), "一致:", list(best) == heads)
```

- グラフに基づく解析は「係り受けのスコアを付ける」と「スコア最大の木を探す」の 2 段階。前者を biaffine 注意機構で計算するのが、Stanza や GiNZA（Transformer 版）などで使われている現在の標準的な方式。
- 最大全域木は交差する係り受けも扱える。交差を許さない場合は Eisner のアルゴリズム（動的計画法、O(n³)）を使う。
- ROOT から出る辺が複数になりうる点に注意。「ROOT の子は 1 つ」という UD の制約を守らせるには追加の処理が要る。
- 遷移型は局所的な判断の積み重ねで速く (O(n))、グラフ型は文全体で最適化する。長距離の係り受けはグラフ型が有利とされる。

## 文脈自由文法を書いて構文解析する (NLTK) {cfg_parse}

自分で書いた小さな文法で文を解析し、構文木と構造的曖昧性を確かめたい（授業の演習向け）。

```python
import nltk

grammar = nltk.CFG.fromstring("""
    S   -> NP VP
    NP  -> Det N | Det N PP | 'I'
    VP  -> V NP | VP PP
    PP  -> P NP
    Det -> 'an' | 'my' | 'the'
    N   -> 'elephant' | 'pajamas' | 'telescope' | 'man'
    V   -> 'shot' | 'saw'
    P   -> 'in' | 'with'
""")
parser = nltk.ChartParser(grammar)

sentence = "I shot an elephant in my pajamas".split()
trees = list(parser.parse(sentence))
print(len(trees), "通りの解析")
for tree in trees:
    tree.pretty_print()
```

```python
# 確率文脈自由文法 (PCFG) で最も確からしい木を選ぶ
pcfg = nltk.PCFG.fromstring("""
    S   -> NP VP [1.0]
    NP  -> Det N [0.5] | Det N PP [0.2] | 'I' [0.3]
    VP  -> V NP [0.6] | VP PP [0.4]
    PP  -> P NP [1.0]
    Det -> 'an' [0.5] | 'my' [0.5]
    N   -> 'elephant' [0.5] | 'pajamas' [0.5]
    V   -> 'shot' [1.0]
    P   -> 'in' [1.0]
""")
best = next(nltk.ViterbiParser(pcfg).parse(sentence))
print(best.prob())
best.pretty_print()
```

- 「パジャマを着た象を撃った / パジャマ姿で象を撃った」という前置詞句付加の曖昧性の古典的な例。

## 組合せ範疇文法 (CCG) で文を導出する (NLTK) {ccg_nltk}

単語に「どんな項を取って何になるか」という範疇を割り当て、少数の組合せ規則だけで文を導出する CCG を試したい。

```python
from nltk.ccg import chart, lexicon

lex = lexicon.fromstring(r"""
    :- S, NP, N
    I => NP
    John => NP
    Mary => NP
    books => NP
    saw => (S\NP)/NP
    likes => (S\NP)/NP
    hates => (S\NP)/NP
    the => NP/N
    man => N
    telescope => N
    with => ((S\NP)\(S\NP))/NP
    with => (N\N)/NP
    and => var\.,var/.,var
""")

# 関数適用だけを使う: X/Y Y → X (前方適用 >)、Y X\Y → X (後方適用 <)
parser = chart.CCGChartParser(lex, chart.ApplicationRuleSet)
chart.printCCGDerivation(next(parser.parse("I saw the man".split())))
```

```python
# 構造的曖昧性: with の範疇が 2 つあるので、導出も 2 通りになる
parses = list(parser.parse("I saw the man with the telescope".split()))
print(len(parses), "通り")
for p in parses:
    chart.printCCGDerivation(p)
```

```python
# 関数合成と型繰り上げを加えると、「John likes and Mary hates」のような構成素でないまとまりを等位接続できる
print("関数適用のみ:", len(list(parser.parse("John likes and Mary hates books".split()))), "通り")
full = chart.CCGChartParser(lex, chart.DefaultRuleSet)
parses = list(full.parse("John likes and Mary hates books".split()))
print("合成・型繰り上げあり:", len(parses), "通り")
chart.printCCGDerivation(parses[0])
```

- `(S\NP)/NP` は「右に NP を取り、次に左に NP を取ると S になる」= 他動詞。文法の情報のほとんどが辞書（語彙項目）に入っていて、規則は言語によらない少数のものだけ、というのが CCG の特徴。
- 範疇と意味表現（ラムダ式）を対応させると、導出と同時に論理式が組み上がる。この付録の「文を一階述語論理の式に変換して真偽を確かめる (NLTK)」と同じ考え方。
- 規則を増やすと、意味が同じ導出が大量に生まれる（擬似的曖昧性）。実用的なパーザは正規形の制約でこれを抑える。
- 実用的な CCG パーザには、英語と日本語に対応した depccg がある。日本語 CCGbank や、CCG 導出から論理式を作って推論する ccg2lambda と組み合わせて使われる。

## 文を一階述語論理の式に変換して真偽を確かめる (NLTK) {logic_semantics}

「すべての少女が犬を追いかける」のような文の意味を、論理式として厳密に表したい。モデル（世界の状態）の上で真偽を計算し、推論もしたい。

```python
import nltk
nltk.download("book_grammars", quiet=True)

# 各規則に意味表現 (ラムダ式) が付いた素性文法。構文木を下から合成して、文全体の論理式を得る
parser = nltk.load_parser("grammars/book_grammars/simple-sem.fcfg", trace=0)
for sent in ["Irene walks", "a dog barks", "Angus chases a dog", "every girl chases a dog", "Angus sees Irene"]:
    for tree in parser.parse(sent.split()):
        print(f"{sent:26} {tree.label()['SEM']}")
```

```python
# 文法の中身を見る: 限定詞や動詞に対応するラムダ式
print(nltk.data.load("grammars/book_grammars/simple-sem.fcfg", format="text")[430:1300])
```

```python
# モデル (個体の集合と、述語の解釈) を定義して、式の真偽を評価する
val = nltk.Valuation.fromstring("""
    angus => a
    irene => i
    girl => {i, j}
    boy => {a}
    dog => {d1, d2}
    walk => {i, a}
    bark => {d1}
    chase => {(i, d1), (j, d2), (a, d1)}
    see => {(a, i)}
""")
model = nltk.Model(val.domain, val)
g = nltk.Assignment(val.domain)

for sent in ["Irene walks", "every girl chases a dog", "every girl walks", "a dog barks", "Irene sees Angus"]:
    sem = next(parser.parse(sent.split())).label()["SEM"]
    print(f"{model.evaluate(str(sem), g)!s:6} {sent:26} {sem}")
```

```python
# 量化子のスコープの曖昧性: 「どの少女も (それぞれ別の) 犬を追う」と「ある 1 匹の犬を全員が追う」
read = nltk.sem.Expression.fromstring
wide_every = read("all x.(girl(x) -> exists y.(dog(y) & chase(x,y)))")
wide_some = read("exists y.(dog(y) & all x.(girl(x) -> chase(x,y)))")
print("∀ が広いスコープ:", model.evaluate(str(wide_every), g), " ∃ が広いスコープ:", model.evaluate(str(wide_some), g))

# 定理証明器で含意関係を確かめる (モデルによらず、論理的に必ず成り立つか)
prover = nltk.ResolutionProver()
print("∃∀ ならば ∀∃:", prover.prove(wide_every, [wide_some]))
print("∀∃ ならば ∃∀:", prover.prove(wide_some, [wide_every]))
premises = [read("all x.(dog(x) -> animal(x))"), read("all x.(animal(x) -> mortal(x))"), read("dog(pochi)")]
print("ポチは死すべきものである:", prover.prove(read("mortal(pochi)"), premises))
```

- 構成性の原理: 文の意味は、部分の意味とその組み合わせ方から決まる。ラムダ計算は、その「組み合わせ」を関数適用として表す道具。
- `every girl chases a dog` には 2 通りの読みがあるが、この文法が出すのは片方だけ。両方を得るには、Cooper storage などスコープを後から決める仕組みが要る。
- 手書きの文法が扱える文はごく一部。広い範囲の文を論理式にするには、CCG パーザに基づく ccg2lambda や、AMR などの意味表現へのニューラルなパージングが使われる。
- 自然言語推論を「埋め込みの類似度」ではなく「証明」で行うアプローチは、否定や数量を含む推論に強い。

## 適合率・再現率・F1 を自分で計算する（micro と macro） {prf_scratch}
ライブラリに頼らず、適合率・再現率・F1 を定義から計算して、micro 平均と macro 平均の違いを理解したい。

```python
from collections import Counter

gold = ["A", "A", "A", "A", "A", "A", "B", "B", "B", "C"]
pred = ["A", "A", "A", "A", "A", "B", "B", "B", "A", "A"]

labels = sorted(set(gold) | set(pred))
tp, fp, fn = Counter(), Counter(), Counter()
for g, p in zip(gold, pred):
    if g == p:
        tp[g] += 1
    else:
        fp[p] += 1     # p と予測したが違った
        fn[g] += 1     # g を見逃した

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f

per_class = {}
for c in labels:
    per_class[c] = prf(tp[c], fp[c], fn[c])
    print(f"{c}: P = {per_class[c][0]:.3f}  R = {per_class[c][1]:.3f}  F1 = {per_class[c][2]:.3f}  (TP {tp[c]}, FP {fp[c]}, FN {fn[c]})")

macro_f1 = sum(f for _, _, f in per_class.values()) / len(labels)        # クラスごとの F1 の単純平均
micro = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))         # 全クラスの TP/FP/FN を合算してから計算
print(f"macro-F1 = {macro_f1:.3f}   micro-F1 = {micro[2]:.3f}   正解率 = {sum(tp.values()) / len(gold):.3f}")
```

```python
from sklearn.metrics import precision_recall_fscore_support

p, r, f, _ = precision_recall_fscore_support(gold, pred, average="macro", zero_division=0)
print("scikit-learn の macro:", round(p, 3), round(r, 3), round(f, 3))

# 抽出タスク (固有表現抽出など、正解と予測が「集合」の場合) の F1
gold_set, pred_set = {("山田", "PER"), ("京都", "LOC"), ("ソニー", "ORG")}, {("山田", "PER"), ("京都", "ORG"), ("東京", "LOC")}
n = len(gold_set & pred_set)
print("抽出の P/R/F1:", [round(x, 3) for x in prf(n, len(pred_set) - n, len(gold_set) - n)])
```

- 単一ラベルの多クラス分類では、micro-F1 は正解率と一致する。
- macro-F1 は少数クラス（ここでは一度も当たっていない C）の影響を強く受ける。クラスの偏りがあるときに、モデルの弱点が見える。
- 「macro-F1」には「クラスごとの F1 の平均」と「macro 適合率と macro 再現率の調和平均」の 2 つの流儀があり、値が違う。前者が一般的。
- 実務でライブラリを使って同じことをする版は、評価指標と統計的検定の章の「scikit-learn で適合率・再現率・F1 を集計する（micro / macro 平均と抽出の評価）」。

## 転置索引を自作してブール検索・フレーズ検索をする {inverted_index_scratch}
検索エンジンの基本である転置索引を自分で作り、AND / OR / NOT や語順どおりのフレーズで検索したい。

```python
!pip install -q fugashi unidic-lite
```

```python
from collections import defaultdict
from fugashi import Tagger

tagger = Tagger()
docs = [
    "東京都立大学で自然言語処理の研究をしている。",
    "自然言語処理では機械翻訳の研究が盛んだ。",
    "東京で機械学習の勉強会が開かれた。",
    "京都大学は言語処理の研究で知られる。",
    "機械翻訳は自然な言語を処理する技術の一つである。",
    "東京と京都を結ぶ新幹線に乗った。",
]

def tokenize(text):
    """（出現位置, 単語）の列。位置は助詞などを除く前の語の列で数える（除いた語の分も位置を空ける）"""
    return [(pos, w.surface) for pos, w in enumerate(tagger(text))
            if w.feature.pos1 not in ("助詞", "助動詞", "補助記号")]

# 転置索引: 単語 → {文書番号: [出現位置, ...]}
index = defaultdict(dict)
for doc_id, doc in enumerate(docs):
    for pos, term in tokenize(doc):
        index[term].setdefault(doc_id, []).append(pos)

print(len(index), "語")
for term in ["東京", "研究", "処理"]:
    print(term, dict(index[term]))
```

```python
ALL = set(range(len(docs)))
posting = lambda term: set(index.get(term, {}))

def AND(*terms): return set.intersection(*[posting(t) for t in terms])
def OR(*terms): return set.union(*[posting(t) for t in terms])
def NOT(term): return ALL - posting(term)

def phrase(*terms):
    """語がこの順番で隣り合って出現する文書（位置情報を使う）"""
    hits = set()
    for doc_id in AND(*terms):
        if any(all(p + i in index[t][doc_id] for i, t in enumerate(terms)) for p in index[terms[0]][doc_id]):
            hits.add(doc_id)
    return hits

queries = {
    "東京 AND 研究": AND("東京", "研究"),
    "機械 AND (翻訳 OR 学習)": posting("機械") & OR("翻訳", "学習"),
    "研究 AND NOT 東京": posting("研究") & NOT("東京"),
    'フレーズ "自然 言語 処理"': phrase("自然", "言語", "処理"),
    "自然 AND 言語 AND 処理": AND("自然", "言語", "処理"),
}
for q, result in queries.items():
    print(f"{q}\n   → {sorted(result)} {[docs[i][:18] for i in sorted(result)]}")
```

- 転置索引は「この単語を含む文書はどれか」を即座に引ける構造。全文書を走査せずに済むので、数億件の文書でも高速に検索できる。
- 出現位置まで持っておくと、フレーズ検索や「近くに出現する」という条件の検索ができる。最後の 2 つの結果の違い（語順どおりに並んでいるか）がその効果。文書 4 の「自然な言語を処理する」は 3 語とも含むが、隣り合っていないのでフレーズ検索では外れる。位置は助詞などを除く前の語の列で数える。除いてから数えると「自然 言語 処理」が隣り合っているように見えてしまう。
- 日本語では単語の区切り方が検索結果を左右する。「東京都立大学」が 1 語になる辞書だと「東京」で検索しても当たらない。対策は、検索の章の「単語に区切らず文字 n-gram で検索する（表記ゆれに強い検索）」。
- 実用の検索エンジン（Lucene、Elasticsearch、Tantivy）も、この構造を圧縮し、スコア計算と組み合わせたもの。
- 実務でライブラリを使って同じことをする版は、検索の章の「SQLite の FTS5 で日本語の文書をブール検索・フレーズ検索する」。

## BM25 を自分で実装してパラメータの働きを確かめる {bm25_scratch}

キーワード検索の標準的なスコアである BM25 が、何を計算しているのかを理解したい。パラメータ k1 と b を変えると何が起きるかを見たい。

```python
!pip install -q rank_bm25 fugashi unidic-lite
```

```python
import math
from collections import Counter
import numpy as np
from fugashi import Tagger

tagger = Tagger()
tok = lambda s: [w.surface for w in tagger(s) if w.feature.pos1 not in ("助詞", "助動詞", "補助記号")]

docs = [
    "猫は可愛い。",
    "猫と犬はどちらも人気のあるペットで、猫は室内で、犬は散歩が必要なので屋外でも飼われることが多い。",
    "猫、猫、猫。猫が大好きだ。",
    "犬の散歩は毎日の習慣である。",
    "ペットを飼うと生活が豊かになると言われている。",
]
tokenized = [tok(d) for d in docs]
N, avgdl = len(docs), np.mean([len(d) for d in tokenized])
df = Counter(t for d in tokenized for t in set(d))

def bm25(query, k1=1.5, b=0.75):
    scores = []
    for d in tokenized:
        tf, total = Counter(d), 0.0
        for t in tok(query):
            if t not in tf:
                continue
            idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)           # 珍しい語ほど重い
            norm = tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))   # 出現回数は頭打ち、長い文書は割り引く
            total += idf * norm
        scores.append(total)
    return np.array(scores)

for q in ["猫", "猫 ペット"]:
    print(q, bm25(q).round(2).tolist())
```

```python
# k1: 出現回数をどこまで重視するか / b: 文書の長さでどこまで割り引くか
print("文書の長さ:", [len(d) for d in tokenized])
for k1, b in [(1.5, 0.75), (0.0, 0.75), (10.0, 0.75), (1.5, 0.0), (1.5, 1.0)]:
    s = bm25("猫", k1, b)
    print(f"k1={k1:4} b={b:4}  スコア {s[:3].round(2).tolist()}  順位 {np.argsort(-s)[:3].tolist()}")
```

```python
# ライブラリの実装と順位が一致することを確かめる (IDF の式に細かい違いがあるので値は少し違う)
from rank_bm25 import BM25Okapi

lib = BM25Okapi(tokenized, k1=1.5, b=0.75).get_scores(tok("猫 ペット"))
mine = bm25("猫 ペット")
print(lib.round(2).tolist(), np.argsort(-lib).tolist())
print(mine.round(2).tolist(), np.argsort(-mine).tolist())
```

- BM25 = 「語の珍しさ (IDF)」×「その文書での出現回数（頭打ちあり）」を、クエリの語について足したもの。
- k1 を 0 にすると出現回数を無視する（含むか含まないかだけ）。大きくすると、同じ語を繰り返す文書ほど高くなる。
- b を 0 にすると文書の長さを無視し、1 にすると長さに比例して割り引く。長い文書は語を多く含みやすいので、その有利さを打ち消している。
- 既定値（k1=1.2〜2.0、b=0.75）は多くの場合そのままでよい。タイトルのような短い文書ばかりなら b を小さくする。

## 規則だけでチャットボットを作る（ELIZA 方式） {eliza_rule}

機械学習を使わずに、パターンと応答テンプレートだけで対話するボットを作りたい。対話システムの原点を体験したい。

```python
import random
import re

RULES = [
    (r"(.+)(?:が|は)(?:つらい|辛い|しんどい|苦しい)", ["{0}がつらいのですね。いつ頃からですか？", "{0}について、もう少し聞かせてください。"]),
    (r"(.+)(?:が|を)(?:心配|不安)", ["{0}の何がいちばん心配ですか？", "{0}が不安なのですね。"]),
    (r"(?:私|わたし|僕|ぼく)は(.+)(?:と思う|と思います)", ["なぜ{0}と思うのですか？", "本当に{0}でしょうか？"]),
    (r"(.+)したい", ["{0}したいのですね。何がそれを妨げていますか？", "{0}できたら、どんな気持ちになりそうですか？"]),
    (r"(母|父|家族|上司|先生|友達|友人)", ["{0}との関係について教えてください。", "{0}はあなたにとってどんな存在ですか？"]),
    (r"(?:いつも|毎日|ずっと)", ["「いつも」ですか。例外はありませんでしたか？"]),
    (r"(?:はい|うん|そうです)", ["なるほど。続けてください。"]),
    (r"(?:いいえ|いや|違い)", ["そうではないのですね。では、どうなのでしょう？"]),
    (r"(?:ありがとう|さようなら|またね)", ["お話しできてよかったです。またいつでもどうぞ。"]),
]
DEFAULT = ["もう少し詳しく話してもらえますか？", "それについてどう感じていますか？", "なるほど。それで？"]

def respond(text, rng):
    for pattern, templates in RULES:           # 上から順に照合し、最初に合った規則を使う
        m = re.search(pattern, text)
        if m:
            groups = [re.sub(r"^(?:でも|けど|最近|実は|正直)[、,]?\s*", "", g).strip("、。 ") for g in m.groups()]
            return rng.choice(templates).format(*groups)
    return rng.choice(DEFAULT)

rng = random.Random(0)
for user in ["最近、仕事がつらいです。", "上司に毎日怒られています。", "私は自分が悪いと思う。", "転職したい。", "でも家族が心配で。", "今日は天気がいいですね。", "ありがとう。"]:
    print(f"あなた: {user}\nボット: {respond(user, rng)}")
```

- 1966 年の ELIZA と同じ仕組み。相手の言葉を拾って質問で返すだけだが、「聞いてもらえている」感じが出る（ELIZA 効果）。
- 規則に合う入力には確実に意図どおりの応答を返せる。予約受付や定型的な案内では、今でも規則ベースの部分が残っている。
- 規則の順序が重要。特殊な規則を上に、一般的な規則を下に置く。どれにも合わないときの応答を必ず用意する。

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

- 訳文と参照訳をそれぞれ文埋め込みにし、その組み合わせから人手のスコアを回帰する。RUSE という指標の考え方で、のちの BLEURT や COMET（学習型の評価指標）につながった。
- 特徴量に差の絶対値と要素ごとの積を入れるのは、2 つの文の関係を学習するときの定番（自然言語推論のモデルと同じ）。
- 回帰モデルの選び方で結果が大きく変わる。この実験では、線形の回帰は学習なしのコサイン類似度にも負け、非線形の SVR にして初めて上回った。高次元の特徴量に対して学習データが数千件しかないので、過学習しやすい。
- 相関の絶対値は低い（Kendall の τ で 0.2 前後）。最近の翻訳システムの出力は質が高くて差が小さく、文単位の人手評価の雑音も大きいため。
- 学習型の指標は、学習データと同じ言語対・同じ年の評価では強いが、分野や翻訳システムの種類が変わると当てにならなくなることがある。原文単位で分割して、見たことのない文で評価する。
- 文単位の BLEU は人手評価との相関が低い。文単位で比べたいときは、chrF か学習型の指標を使う。
- 実用では、自分で学習するより公開されている COMET を使う（機械翻訳の章の「COMET で翻訳の品質を評価する」）。ここでの目的は、学習型の指標が何をしているかを理解すること。

## 編集距離と単語の頻度で英語のスペル誤りを訂正する（Norvig 方式を自分で実装） {spelling_correction_scratch}
英語のつづりの誤りを見つけて訂正したい。編集距離と単語の頻度による古典的な方法を作り、文脈を使う方法と比べる（ライブラリで手軽に直すだけなら、文法誤り訂正と言語教育の章の「英語のスペルミスをライブラリで直す (pyspellchecker)」）。

```python
!pip install -q symspellpy
!wget -q https://norvig.com/big.txt https://norvig.com/spell-testset1.txt
```

```python
import re
from collections import Counter

WORDS = Counter(re.findall(r"[a-z]+", open("big.txt", encoding="utf-8").read().lower()))     # 約 100 万語の英文から単語の頻度を数える
TOTAL = sum(WORDS.values())
LETTERS = "abcdefghijklmnopqrstuvwxyz"

def edits1(word):
    """編集距離 1 の文字列をすべて作る (削除・隣接文字の入れ替え・置換・挿入)"""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits if b for c in LETTERS]
    inserts = [a + c + b for a, b in splits for c in LETTERS]
    return set(deletes + transposes + replaces + inserts)

def known(words):
    return {w for w in words if w in WORDS}

def candidates(word):
    """辞書にある語を、編集距離の小さい順に探す"""
    return known([word]) or known(edits1(word)) or known(e2 for e1 in edits1(word) for e2 in edits1(e1)) or {word}

def correct(word):
    return max(candidates(word), key=WORDS.get)

for w in ["speling", "korrectud", "bycycle", "inconvient", "recieve", "definately", "their", "teh"]:
    print(f"{w} → {correct(w)}  (候補: {sorted(candidates(w), key=WORDS.get, reverse=True)[:5]})")
```

```python
# 誤りと正解の組のテストセットで正解率を測る
tests = [(wrong, right) for line in open("spell-testset1.txt") for right, wrongs in [line.strip().split(":")] for wrong in wrongs.split()]

def accuracy(corrector, tests):
    result = [corrector(wrong) == right for wrong, right in tests]
    return sum(result) / len(result)

print(len(tests), "組 / 正解率:", round(accuracy(correct, tests), 3), "/ 正解の語が辞書にない割合:", round(sum(r not in WORDS for _, r in tests) / len(tests), 3))
errors = [(w, correct(w), r) for w, r in tests if correct(w) != r][:8]
print("誤りの例 (入力, 出力, 正解):", errors)
```

```python
# 高速なライブラリ: SymSpell (削除だけで候補を引く索引を作っておくので、非常に速い)
import time
from symspellpy import SymSpell, Verbosity

sym = SymSpell(max_dictionary_edit_distance=2)
for word, count in WORDS.items():
    sym.create_dictionary_entry(word, count)

def correct_symspell(word):
    hits = sym.lookup(word, Verbosity.TOP, max_edit_distance=2)
    return hits[0].term if hits else word

for name, f in [("自作", correct), ("SymSpell", correct_symspell)]:
    t = time.perf_counter(); acc = accuracy(f, tests)
    print(f"{name}: 正解率 {acc:.3f} / {len(tests) / (time.perf_counter() - t):.0f} 語/秒")
```

```python
# 文脈を使う: 候補のうち、マスク言語モデルが文脈に合うと判断したものを選ぶ (辞書にある語への打ち間違いにも対応できる)
import math
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
mlm = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").eval()

@torch.no_grad()
def correct_in_context(sentence, position, penalty=2.0):
    words = sentence.split()
    word = words[position].lower()
    near = known(edits1(word))
    far = known(e2 for e1 in edits1(word) for e2 in edits1(e1)) - near      # peace → piece のように、実在語どうしの取り違えは編集距離 2 のことが多い
    distance = {word: 0, **{c: 1 for c in near - {word}}, **{c: 2 for c in far - {word}}}
    enc = tokenizer(" ".join(words[:position] + [tokenizer.mask_token] + words[position + 1:]), return_tensors="pt")
    pos = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero()[0, 0]
    logp = mlm(**enc).logits[0, pos].log_softmax(-1)
    def score(c):
        ids = tokenizer(c, add_special_tokens=False)["input_ids"]
        context = float(logp[ids[0]]) if len(ids) == 1 else -20.0               # 1 トークンで表せない語は文脈のスコアを使わない
        return context - penalty * distance[c]                                   # 元の語から離れるほどペナルティ
    return max(distance, key=score)

for sentence, position in [("I want a peace of cake .", 3), ("He will except the offer .", 2), ("She lives in a quite village .", 4),
                           ("The whether is nice today .", 1), ("Their is a cat on the table .", 0), ("I need to buy some bred .", 5),
                           ("I like to read books .", 4), ("The cat sat on the mat .", 1)]:
    print(sentence, "→", sentence.split()[position], "⇒", correct_in_context(sentence, position), "(単語だけで訂正:", correct(sentence.split()[position].lower()) + ")")
```

- 古典的なスペル訂正は「編集距離が近い語のうち、最も頻度の高い語を選ぶ」だけで、かなりの誤りを直せる。Peter Norvig の有名な 20 数行の実装がこれにあたる。
- 弱点は 3 つ: 辞書にない語は正解にできない、頻度だけで選ぶので文脈に合わない語を選ぶことがある、そして実在する別の語になってしまった誤り (peace / piece) には気付けない。
- 文脈を使うと、実在する語への打ち間違い (real-word error) も直せる（peace → piece、whether → weather）。こうした取り違えは編集距離 2 のことが多いので、候補を距離 2 まで広げる。正しい語まで書き換えてしまう危険が増えるので、元の語から離れるほどペナルティを付け、正しい文（books、cat）が変わらないことも確かめる。
- 学習者のスペル誤りは、打ち間違いとは傾向が違う（発音に引きずられる、母語の影響を受ける）。編集操作ごとの起こりやすさを学習者のデータから推定すると、精度が上がる。
- 日本語には分かち書きがなく、スペル誤りにあたるのは誤変換や脱字。日本語入力の章と、文法誤り検出のレシピを参照。
- 実務でライブラリを使って同じことをする版は、文法誤り訂正と言語教育の章の「symspellpy とマスク言語モデルで英語のスペル誤りを文脈に合わせて直す」。
