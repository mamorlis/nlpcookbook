# サブワードとトークナイザ

SentencePiece や Hugging Face のトークナイザを学習・利用し、モデルに入れる ID 列を作る。

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

## SentencePiece でサブワードモデルを学習する {sentencepiece_train}

自分のコーパスからサブワード分割モデルを作りたい。日本語でも事前の単語分割なしで学習できる。

```python
!pip install -q sentencepiece
```

```python
import io, re, urllib.request, zipfile

# 学習用コーパス: 青空文庫「吾輩は猫である」
url = "https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip"
with urllib.request.urlopen(url) as r:
    z = zipfile.ZipFile(io.BytesIO(r.read()))
text = z.read(z.namelist()[0]).decode("cp932")
text = re.split(r"-{50,}", text)[2].split("底本：")[0]          # ヘッダと奥付を除く
text = re.sub(r"《.+?》|［＃.+?］|｜", "", text)                 # ルビ・注記を除く
lines = [s for s in re.split(r"(?<=。)|\r\n", text) if s and s.strip()]
with open("corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(s.strip() for s in lines))
print(len(lines), "文")
```

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt", model_prefix="neko", vocab_size=4000,
    model_type="unigram",        # "bpe" も選べる
    character_coverage=0.9995,   # 日本語・中国語は 0.9995、英語などは 1.0
    minloglevel=1,
)
sp = spm.SentencePieceProcessor(model_file="neko.model")
s = "吾輩は猫である。名前はまだ無い。"
print(sp.encode(s, out_type=str))
print(sp.encode(s))
print(sp.decode(sp.encode(s)))
```

- `▁` (U+2581) は空白 (単語境界) を表す。デコードすると元の文に完全に戻る (可逆) のが SentencePiece の特徴。
- 語彙サイズは、翻訳なら 8k〜32k、小さなコーパスなら 4k〜8k が目安。

## サブワード正則化: 分割をサンプリングしてデータ拡張する {subword_regularization}

同じ文を毎回違う分割でモデルに見せて、頑健性を上げたい (subword regularization / BPE-dropout)。

```python
!pip install -q sentencepiece
```

```python
import sentencepiece as spm

import io, re, urllib.request, zipfile

# 青空文庫「吾輩は猫である」でモデルを学習 (前のレシピと同じ)
url = "https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip"
with urllib.request.urlopen(url) as r:
    z = zipfile.ZipFile(io.BytesIO(r.read()))
text = z.read(z.namelist()[0]).decode("cp932")
text = re.split(r"-{50,}", text)[2].split("底本：")[0]
text = re.sub(r"《.+?》|［＃.+?］|｜", "", text)
with open("corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(s.strip() for s in re.split(r"(?<=。)|\r\n", text) if s and s.strip()))

spm.SentencePieceTrainer.train(input="corpus.txt", model_prefix="neko", vocab_size=4000, model_type="unigram", minloglevel=2)
sp = spm.SentencePieceProcessor(model_file="neko.model")
```

```python
text = "吾輩は猫である。名前はまだ無い。"
print("最尤:", sp.encode(text, out_type=str))
for _ in range(5):
    print("標本:", sp.encode(text, out_type=str, enable_sampling=True, alpha=0.2, nbest_size=-1))
```

```python
# 確率の高い順に N 個の分割候補
for pieces in sp.nbest_encode(text, nbest_size=3, out_type=str):
    print(pieces)
```

- `alpha` が小さいほど多様な (確率の低い) 分割が出る。`nbest_size=-1` は全候補からサンプリング。
- 学習時だけサンプリングし、評価時は最尤分割を使う。

## Hugging Face tokenizers で BPE / WordPiece を学習する {tokenizers_train}

Transformers でそのまま使えるトークナイザを、自分のコーパスから高速に学習したい。

```python
import nltk
nltk.download("gutenberg", quiet=True)
from nltk.corpus import gutenberg

lines = [l for l in gutenberg.raw("austen-emma.txt").split("\n") if l.strip()]
print(len(lines), "行")
```

```python
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

tok = Tokenizer(models.BPE(unk_token="[UNK]"))
tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=3000, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"], show_progress=False)
tok.train_from_iterator(lines, trainer)

enc = tok.encode("Emma Woodhouse was unquestionably handsome.")
print(enc.tokens)
print(enc.ids)
print(tok.decode(enc.ids))
```

```python
# Transformers のトークナイザとして保存・読み込み
from transformers import AutoTokenizer, PreTrainedTokenizerFast

fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]")
fast.save_pretrained("my_tokenizer")
print(AutoTokenizer.from_pretrained("my_tokenizer")("Hello Emma!"))
```

- WordPiece にするなら `models.WordPiece` + `trainers.WordPieceTrainer` + `pre_tokenizers.Whitespace()` に差し替える。
- 既存モデルと同じ方式で作り直すだけなら `old_tokenizer.train_new_from_iterator(lines, vocab_size)` が最短。

## 事前学習モデルのトークナイザを使う (日本語 BERT) {hf_tokenizer_basics}

BERT などのモデルに入力するために、テキストをトークン ID 列に変換したい。逆に ID からテキストに戻したい。

```python
!pip install -q fugashi unidic-lite
```

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-v3")
text = "自然言語処理のクックブックを作っています。"

print(tokenizer.tokenize(text))                 # サブワード列
enc = tokenizer(text)                           # モデル入力 (dict)
print(enc["input_ids"])
print(tokenizer.convert_ids_to_tokens(enc["input_ids"]))  # [CLS] と [SEP] が付く
print(tokenizer.decode(enc["input_ids"]))
print(tokenizer.decode(enc["input_ids"], skip_special_tokens=True))
```

```python
print("語彙サイズ:", tokenizer.vocab_size)
print("特殊トークン:", tokenizer.special_tokens_map)
print("最大長:", tokenizer.model_max_length)
```

- 東北大 BERT は MeCab で単語分割してから WordPiece をかけるので `fugashi` と `unidic-lite` が必要。
- `##` は「前のトークンに続く」印。`decode` すると日本語でもトークン間に空白が入る (次々項を参照)。

## パディングと切り詰めをしてバッチのテンソルを作る {padding_truncation}

長さの違う複数の文を、モデルに入れられる同じ長さのテンソルにまとめたい。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
texts = ["短い文。", "これは少し長めの文で、トークン数が多くなります。", "This is an English sentence."]

batch = tokenizer(texts, padding=True, truncation=True, max_length=12, return_tensors="pt")
print(batch["input_ids"])
print(batch["attention_mask"])  # 1 = 本物のトークン、0 = パディング
```

```python
for name, kwargs in {
    "padding=True (バッチ内最長に合わせる)": dict(padding=True),
    "padding='max_length'": dict(padding="max_length", max_length=20),
    "truncation のみ": dict(truncation=True, max_length=8),
}.items():
    print(name, [len(x) for x in tokenizer(texts, **kwargs)["input_ids"]])
```

- 学習時は `padding=True` を tokenizer ではなく `DataCollatorWithPadding` に任せると、バッチごとに最短で済んで速い (動的パディング)。
- 生成モデル (GPT 系) は左パディングが必要: `tokenizer.padding_side = "left"`。

## 文ペアを 1 つの入力にまとめる {sentence_pair}

含意関係認識や質問応答のように、2 つの文をまとめてモデルに入力したい。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
enc = tokenizer("猫がマットの上で寝ている。", "動物が眠っている。")
print(tokenizer.decode(enc["input_ids"]))
print(enc["token_type_ids"])  # 0 = 1 文目, 1 = 2 文目
```

```python
# バッチの場合は 1 文目のリストと 2 文目のリストを渡す。長い方 (ここでは文書側) だけを切り詰める
questions = ["日本の首都は？", "富士山の高さは？"]
contexts = ["日本の首都は東京である。" * 20, "富士山は標高 3776 メートルの山である。" * 20]
batch = tokenizer(questions, contexts, truncation="only_second", max_length=32, padding=True)
print([len(x) for x in batch["input_ids"]])
print(tokenizer.decode(batch["input_ids"][0]))
```

- RoBERTa 系には `token_type_ids` がなく、`</s></s>` で区切るだけ。`tokenizer(a, b)` と書いておけばモデルに合った形式になる。

## 日本語 BERT のデコード結果から余分な空白を消す {decode_ja_spaces}

日本語 BERT の `decode()` はトークン間に空白を入れてしまう。元の文のように連結したい。

```python
!pip install -q fugashi unidic-lite
```

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-v3")
ids = tokenizer("東京スカイツリーは634メートルです。")["input_ids"]
print(tokenizer.decode(ids, skip_special_tokens=True))

def decode_ja(ids):
    tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
    return "".join(t[2:] if t.startswith("##") else t for t in tokens)

print(decode_ja(ids))
```

- 単純に `.replace(" ", "")` すると英単語間の空白も消える。厳密に原文へ戻したい場合は、オフセットを返すトークナイザ (次項) を使う。

## トークンと元テキストの文字位置を対応付ける (offset mapping) {offset_mapping}

各トークンが原文の何文字目から何文字目に対応するかを知りたい (固有表現抽出や抽出型 QA の後処理)。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
text = "山田さんは東京都立大学で教えています。"
enc = tokenizer(text, return_offsets_mapping=True)

for tok, (s, e) in zip(enc.tokens(), enc["offset_mapping"]):
    print(f"{tok:10} ({s:2},{e:2}) {text[s:e]!r}")
```

```python
# 文字位置 → トークン位置 (例: 「東京都立大学」= 5〜11 文字目のトークン範囲)
start_char, end_char = text.index("東京都立大学"), text.index("東京都立大学") + 6
start_tok, end_tok = enc.char_to_token(start_char), enc.char_to_token(end_char - 1)
print(start_tok, end_tok, enc.tokens()[start_tok:end_tok + 1])
```

- `return_offsets_mapping` は高速 (Rust 実装) トークナイザでのみ使える。`tokenizer.is_fast` で確認できる。
- 特殊トークンのオフセットは `(0, 0)` になる。

## 単語単位のラベルをサブワードに揃える (word_ids) {word_ids}

単語ごとに付けたラベル (品詞・固有表現タグ) を、サブワード分割後のトークン列に対応させたい。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
words = ["Hanako", "Yamada", "works", "at", "Hokkaido", "University", "."]
labels = ["B-PER", "I-PER", "O", "O", "B-ORG", "I-ORG", "O"]

enc = tokenizer(words, is_split_into_words=True)
print(enc.tokens())
print(enc.word_ids())
```

```python
# 各単語の先頭サブワードにだけラベルを付け、残りと特殊トークンは -100 (損失計算で無視される)
label2id = {l: i for i, l in enumerate(sorted(set(labels)))}
aligned, prev = [], None
for wid in enc.word_ids():
    if wid is None or wid == prev:
        aligned.append(-100)
    else:
        aligned.append(label2id[labels[wid]])
    prev = wid
for tok, lab in zip(enc.tokens(), aligned):
    print(f"{tok:12}{lab}")
```

- 予測時も同じ `word_ids()` を使って、各単語の先頭サブワードの予測だけを取り出す。

## 語彙に新しいトークン・特殊トークンを追加する {add_tokens}

`<URL>` のような特殊トークンや、ドメイン固有の用語を 1 トークンとして扱えるようにしたい。

```python
from transformers import AutoModel, AutoTokenizer

name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name)

text = "check <URL> and ask <USER> about covid19 vaccine"
print(tokenizer.tokenize(text))
```

```python
n1 = tokenizer.add_special_tokens({"additional_special_tokens": ["<URL>", "<USER>"]})  # 分割も小文字化もされない
n2 = tokenizer.add_tokens(["covid19"])                                                 # 通常の語として追加
print("追加数:", n1, n2, "語彙サイズ:", len(tokenizer))
print(tokenizer.tokenize(text))

model.resize_token_embeddings(len(tokenizer))  # 埋め込み行列を広げる (忘れると範囲外アクセスで落ちる)
print(model.get_input_embeddings().weight.shape)
```

- 追加したトークンの埋め込みはランダム初期化なので、ファインチューニングで学習させる必要がある。
- `tokenizer.vocab_size` は追加前のサイズのまま。追加後のサイズは `len(tokenizer)`。

## 未知語 ([UNK]) がどれくらい出るか調べる {unk_rate}

使おうとしているモデルのトークナイザが、手元のテキストをきちんと表現できているか確認したい。

```python
!pip install -q fugashi unidic-lite
```

```python
from transformers import AutoTokenizer

texts = ["𠮷野家で牛丼を食べた😋", "髙橋さんと﨑山さんが来た。", "한국어와 日本語 and English", "ｷﾀ━━━(ﾟ∀ﾟ)━━━!!", "普通の日本語の文です。"]

for name in ["tohoku-nlp/bert-base-japanese-v3", "bert-base-multilingual-cased", "xlm-roberta-base"]:
    tok = AutoTokenizer.from_pretrained(name)
    total, unk = 0, 0
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        total += len(ids)
        unk += sum(i == tok.unk_token_id for i in ids)
    print(f"{name:36} UNK 率 {unk / total:.1%} ({unk}/{total})")
    print("   ", tok.tokenize(texts[0]))
```

- WordPiece は語彙にない文字を含む単語をまるごと `[UNK]` にする。SentencePiece 系 (XLM-R) は文字単位にフォールバックするので UNK が出にくい。
- UNK が多い場合は、NFKC 正規化や異体字の置換などの前処理を検討する。

## トークン長の分布を調べて max_length を決める {token_length_dist}

データセットのトークン長がどのくらいかを調べて、切り詰め長 (`max_length`) を根拠をもって決めたい。

```python
import numpy as np
import nltk
nltk.download("movie_reviews", quiet=True)
from nltk.corpus import movie_reviews
from transformers import AutoTokenizer

texts = [movie_reviews.raw(f) for f in movie_reviews.fileids()[:500]]
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
lengths = np.array([len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]])

print("平均", lengths.mean().round(1), "中央値", np.median(lengths), "最大", lengths.max())
for p in [50, 90, 95, 99]:
    print(f"{p} パーセンタイル: {np.percentile(lengths, p):.0f}")
for L in [128, 256, 512]:
    print(f"max_length={L}: {(lengths <= L).mean():.1%} が切り詰めなしで収まる")
```

```python
import matplotlib.pyplot as plt

plt.hist(lengths, bins=50)
plt.axvline(512, color="red", linestyle="--", label="512")
plt.xlabel("tokens"); plt.ylabel("documents"); plt.legend(); plt.show()
```

- 95 パーセンタイルあたりを `max_length` にするのが定番。長い文書が多いなら、先頭 + 末尾を残す・分割して集約するなどを検討する。

## トークン数を数える (LLM のコンテキスト長・料金の見積もり) {count_tokens}

テキストが LLM の何トークンになるかを数えたい。日本語と英語でトークン数がどれくらい違うかも知りたい。

```python
!pip install -q tiktoken
```

```python
import tiktoken

enc = tiktoken.get_encoding("o200k_base")  # GPT-4o 系の語彙
for text in ["Natural language processing is fun.", "自然言語処理は楽しい。", "吾輩は猫である。名前はまだ無い。"]:
    ids = enc.encode(text)
    print(len(text), "文字", len(ids), "トークン", [enc.decode([i]) for i in ids])
```

```python
# オープンなモデルは Hugging Face のトークナイザで数える
from transformers import AutoTokenizer

text = "自然言語処理は楽しい。吾輩は猫である。名前はまだ無い。"
for name in ["Qwen/Qwen2.5-0.5B-Instruct", "xlm-roberta-base", "google/byt5-small"]:
    tok = AutoTokenizer.from_pretrained(name)
    print(f"{name:30} {len(tok(text, add_special_tokens=False)['input_ids'])} トークン")
```

- 1 文字が複数トークンに分かれると、トークン単体では表示できず `�` になる (上の「輩」)。
- 日本語は「1 文字 ≒ 1 トークン前後」、英語は「1 単語 ≒ 1.3 トークン」が大まかな目安。モデルによって大きく違うので実測する。
- API 型のモデルは各社が公式のトークン数カウント API を提供している。厳密な見積もりにはそちらを使う。
