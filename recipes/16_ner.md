# 固有表現抽出

人名・組織名・地名などの固有表現を見つける。学習済みモデルの利用、辞書やパターンによる抽出、個人情報のマスクや音声認識の結果、商品の属性抽出への応用、BIO タグの扱いと評価、CRF や BERT による学習。固有表現抽出や系列ラベリングの背景については、姉妹編『[自然言語処理の教科書](https://github.com/mamorlis/nlpbook)』の第 3 章も参照。

## 日本語の固有表現（人名・地名・組織名など）を抽出する {ner_ginza}

日本語の文章から人名、地名、組織名、日付、金額などを抜き出したい。

```python
!pip install -q ginza ja_ginza
```

```python
import spacy
from spacy import displacy

nlp = spacy.load("ja_ginza")
text = "2024年4月、山田太郎さんはトヨタ自動車を退職し、京都市でスタートアップを創業して3億円を調達した。"
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_, ent.start_char, ent.end_char, sep="\t")
displacy.render(doc, style="ent", jupyter=True)
```

```python
# GiNZA のラベルは関根の拡張固有表現階層 (約 200 種類) で細かいので、必要に応じて粗いラベルにまとめる
COARSE = {"Person": "PERSON", "Company": "ORG", "City": "LOC", "Province": "LOC", "Country": "LOC",
          "Date": "DATE", "Money": "MONEY"}
print([(ent.text, COARSE.get(ent.label_, "MISC")) for ent in doc.ents])
```

- spaCy 公式モデル `ja_core_news_md` は OntoNotes 風の 22 ラベル (PERSON, ORG, GPE, DATE …)。

## 英語の固有表現を抽出して色付きで表示する {ner_spacy_en}

英文から人名・組織名・地名・日付などを抜き出して、ハイライト表示したい。

```python
!python -m spacy download en_core_web_sm -q
```

```python
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
text = ("Apple is looking at buying a U.K. startup for $1 billion. "
        "Tim Cook met Prime Minister Rishi Sunak in London on March 3, 2024.")
doc = nlp(text)
for ent in doc.ents:
    print(f"{ent.text:22}{ent.label_:10}{spacy.explain(ent.label_)}")
displacy.render(doc, style="ent", jupyter=True)
```

```python
# 特定のラベルだけ・色を変えて表示
displacy.render(doc, style="ent", jupyter=True,
                options={"ents": ["PERSON", "ORG"], "colors": {"PERSON": "#ffd166", "ORG": "#8ecae6"}})
```

```python
# 文書集合から固有表現の頻度を集計
from collections import Counter

docs = nlp.pipe(["Google and Microsoft compete in AI.", "Microsoft invested in OpenAI.", "Sam Altman leads OpenAI."])
print(Counter((e.text, e.label_) for d in docs for e in d.ents).most_common())
```

- 精度が必要なら Transformer ベースの `en_core_web_trf` を使う（遅いが高精度）。

## Transformers の pipeline で日本語の固有表現抽出をする {ner_transformers_ja}

Hugging Face Hub にあるファインチューニング済みモデルで、日本語の固有表現抽出をしたい。

```python
!pip install -q fugashi unidic-lite
```

```python
from transformers import pipeline

ner = pipeline("token-classification", model="llm-book/bert-base-japanese-v3-ner-wikipedia-dataset", aggregation_strategy="simple")
text = "鈴木一郎は1973年に愛知県で生まれ、オリックスとシアトル・マリナーズで活躍した。"
for e in ner(text):
    print(f"{e['word'].replace(' ', ''):\u3000<12}{e['entity_group']:\u3000<8}{e['score']:.3f}")
```

```python
# aggregation_strategy を指定しないとサブワード単位の生の予測 (BIO タグ) が返る
raw = pipeline("token-classification", model="llm-book/bert-base-japanese-v3-ner-wikipedia-dataset")
print([(e["word"], e["entity"]) for e in raw("鈴木一郎は愛知県で生まれた。")])
```

- `aggregation_strategy="simple"` でサブワードを結合し、B-/I- タグをまとめて 1 つの固有表現にしてくれる。
- 日本語 BERT のトークナイザは文字位置を返さないので `start`/`end` は `None` になる。`word` の空白を除いて使う。
- このモデルのラベルは 8 種類: 人名、法人名、政治的組織名、その他の組織名、地名、施設名、製品名、イベント名（ストックマーク社の Wikipedia NER データセット）。
- 多言語モデルなら `tsmatz/xlm-roberta-ner-japanese` などがあり、こちらは `start`/`end` が得られる。

## 学習なしで好きなラベルの固有表現を抽出する (GLiNER) {gliner}

「病名」「薬品名」「役職」のような独自のラベルで固有表現抽出をしたいが、学習データがない。

```python
!pip install -q gliner fugashi unidic-lite
```

```python
from gliner import GLiNER
from huggingface_hub import hf_hub_download
from sentencepiece import sentencepiece_model_pb2
from tokenizers import Regex, normalizers

model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
# 中身は mDeBERTa。transformers 5 系の DebertaV2Tokenizer は文字の正規化 (全角→半角など) を組み込まずに作られ、
# 全角の記号や英数字を [UNK] にしてしまう (issue #49622)。GLiNER の中のトークナイザーに、元の spm.model にある正規化の規則を戻す
proto = sentencepiece_model_pb2.ModelProto()
proto.ParseFromString(open(hf_hub_download(model.config.model_name, "spm.model"), "rb").read())
model.data_processor.transformer_tokenizer.backend_tokenizer.normalizer = normalizers.Sequence([
    normalizers.Strip(), normalizers.Precompiled(proto.normalizer_spec.precompiled_charsmap), normalizers.Replace(Regex(" {2,}"), " ")])

text = "Dr. Sarah Chen, CTO of Anthropic, presented the Claude model at NeurIPS 2024 in Vancouver."
for e in model.predict_entities(text, ["person", "job title", "company", "product", "conference", "city"], threshold=0.4):
    print(f"{e['text']:14}{e['label']:12}{e['score']:.2f}")
```

```python
# 日本語は空白で区切られていないので、分かち書きしてから渡す
from fugashi import Tagger

wakati = Tagger("-Owakati")
text = "田中医師は、インフルエンザと診断された患者にタミフルを5日分処方し、慶應義塾大学病院への紹介状を書いた。"
labels = ["人名", "病名", "薬品名", "医療機関", "期間"]
for e in model.predict_entities(wakati.parse(text).strip(), labels, threshold=0.5):
    print(f"{e['text'].replace(' ', ''):\u3000<10}{e['label']:\u3000<6}{e['score']:.2f}")
```

- ラベル名を自然言語で与えるだけで抽出できる小型のエンコーダモデル。CPU でも動く。
- 内部で空白区切りの単語を単位にしているため、日本語や中国語は分かち書きが必要（そのまま渡すと文全体が 1 語扱いになる）。
- ラベル名の言い回しで結果が変わる。`threshold` を下げると再現率が上がる。
- LLM に抽出させる方法と比べて、高速で、原文中の位置 (`start`/`end`) が確実に得られるのが利点。
- GLiNER はトークナイザーを内部で読み込み、元のモデル (`microsoft/mdeberta-v3-base`) には `tokenizer.json` もないので、`tokenizer.json` から読み直す方法（Transformers の基本操作の章の「2 つの文の含意関係を判定する (NLI)」の注意書きを参照）が使えない。そこで、読み込んだあとのトークナイザーの正規化の部分だけを差し替えている。上流 (transformers) で直れば不要になる。

## 辞書やルールで固有表現を追加する (EntityRuler) {entity_ruler}

統計モデルが取りこぼす製品名・専門用語などを、手持ちの辞書で確実に固有表現として認識させたい。

```python
!pip install -q ginza ja_ginza
```

```python
import spacy

nlp = spacy.load("ja_ginza")
text = "山田研究室では、GECToRとBARTを使って文法誤り訂正の実験をした。"
print("追加前:", [(e.text, e.label_) for e in nlp(text).ents])

ruler = nlp.add_pipe("entity_ruler", before="ner")  # ner の前に置くと辞書が優先される
ruler.add_patterns([
    {"label": "MODEL", "pattern": "GECToR"},
    {"label": "MODEL", "pattern": "BART"},
    {"label": "TASK", "pattern": [{"ORTH": "文法"}, {"ORTH": "誤り"}, {"ORTH": "訂正"}]},  # トークン列のパターン
])
print("追加後:", [(e.text, e.label_) for e in nlp(text).ents])
```

```python
# パターンを JSONL に保存・読み込み
ruler.to_disk("patterns.jsonl")
print(open("patterns.jsonl", encoding="utf-8").read())
```

- 文字列パターンはトークナイザの分割と一致する必要がある。合わないときは、トークン列を確認してトークンパターンで書く。
- `after="ner"` に置くと統計モデルの結果が優先され、重ならない部分だけ辞書で補う動作になる。

## 品詞や原形のパターンで表現を抽出する (spaCy Matcher) {spacy_matcher}

「形容詞 + 名詞」「〜することができる」のように、品詞や原形を使ったパターンに合う箇所を見つけたい。正規表現の単語版。

```python
!pip install -q ginza ja_ginza
!python -m spacy download en_core_web_sm -q
```

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
matcher.add("ADJ_NOUN", [[{"POS": "ADJ", "OP": "+"}, {"POS": "NOUN", "OP": "+"}]], greedy="LONGEST")
matcher.add("BE_GOING_TO", [[{"LEMMA": "be"}, {"LOWER": "going"}, {"LOWER": "to"}, {"POS": "VERB"}]])

doc = nlp("Large language models are going to change natural language processing. She was going to buy a new red car.")
for match_id, start, end in matcher(doc):
    print(nlp.vocab.strings[match_id], "|", doc[start:end].text)
```

```python
# 日本語: 「〜ことができる」（可能表現）と「サ変名詞 + する」
nlp_ja = spacy.load("ja_ginza")
m = Matcher(nlp_ja.vocab)
# GiNZA では「要約する」の「する」は POS が AUX になるので、POS ではなく TAG（動詞-…）で動詞を指定する
# サ変名詞（「要約」など）があれば含める ("OP": "?")。重なる候補からは最長のものを残す
m.add("KOTO_GA_DEKIRU", [[{"TAG": {"REGEX": "サ変可能"}, "OP": "?"}, {"TAG": {"REGEX": "^動詞"}}, {"ORTH": "こと"},
                          {"ORTH": {"IN": ["が", "も", "は"]}}, {"LEMMA": "できる"}]], greedy="LONGEST")
m.add("SAHEN", [[{"POS": {"IN": ["NOUN", "VERB"]}, "TAG": {"REGEX": "サ変可能"}}, {"LEMMA": "する"}]])

doc = nlp_ja("このモデルは長い文章を要約することができる。日本語を翻訳することもできるし、質問に回答する。")
for match_id, start, end in m(doc):
    print(nlp_ja.vocab.strings[match_id], "|", doc[start:end].text)
```

- 使える属性は `ORTH` `LOWER` `LEMMA` `POS` `TAG` `DEP` `ENT_TYPE` `IS_DIGIT` など。`OP` は `?` `*` `+` `!`。
- 単語列の辞書（数万語）を高速に引くなら `PhraseMatcher`、係り受けパターンなら `DependencyMatcher`。

## 文字位置のアノテーションを BIO タグ列に変換する {spans_to_bio}

「何文字目から何文字目が人名」という形式（doccano や brat の出力）のアノテーションを、トークンごとの BIO タグに変換したい。

```python
!pip install -q fugashi unidic-lite
```

```python
from fugashi import Tagger

tagger = Tagger()

def tokenize_with_offsets(text):
    pos, out = 0, []
    for w in tagger(text):
        start = text.index(w.surface, pos)
        pos = start + len(w.surface)
        out.append((w.surface, start, pos))
    return out

def spans_to_bio(text, spans):
    """spans: [(開始文字位置, 終了文字位置, ラベル), ...]"""
    tokens = tokenize_with_offsets(text)
    tags = ["O"] * len(tokens)
    for s, e, label in spans:
        inside = [i for i, (_, ts, te) in enumerate(tokens) if ts < e and te > s]   # スパンと重なるトークン
        for n, i in enumerate(inside):
            tags[i] = ("B-" if n == 0 else "I-") + label
        if inside and (tokens[inside[0]][1] != s or tokens[inside[-1]][2] != e):
            print(f"  警告: 境界がトークンと合わない: {text[s:e]!r} → {''.join(tokens[i][0] for i in inside)!r}")
    return [t for t, _, _ in tokens], tags

# doccano の JSONL 形式: {"text": ..., "label": [[start, end, label], ...]}
record = {"text": "山田太郎は2020年に京都大学を卒業し、東京都で働いている。",
          "label": [[0, 4, "PER"], [5, 10, "DATE"], [11, 15, "ORG"], [20, 22, "LOC"]]}
tokens, tags = spans_to_bio(record["text"], record["label"])
for tok, tag in zip(tokens, tags):
    print(f"{tok}\t{tag}")
```

```python
# アノテーションの境界がトークンの境界と合わない例
tokens, tags = spans_to_bio("東京都立大学の教授に会った。", [(0, 3, "LOC")])
print(list(zip(tokens, tags)))
```

- 「東京」だけがアノテーションされているのに、トークナイザが「東京都」を 1 語にする、といった境界のずれが必ず起きる。件数を数えて、多ければ文字単位のタグ付けやサブワードの利用を検討する。
- I- で始まる系列を許す IOB1、末尾と単独を区別する BIOES (BILOU) などの変種がある。データセットがどれかを確認する。

## BIO タグ列から固有表現のスパンを取り出す {bio_to_spans}

モデルが出力した BIO タグ列を、「どこからどこまでが何のエンティティか」という形に戻したい。不正なタグ列にも対処したい。

```python
def bio_to_spans(tags):
    """[(開始トークン位置, 終了トークン位置 (含まない), ラベル), ...] を返す。B- なしで始まる I- は新しいスパンの開始とみなす"""
    spans, start, label = [], None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, _, lab = tag.partition("-")
        if start is not None and (prefix != "I" or lab != label):   # 今のスパンが終わる
            spans.append((start, i, label))
            start, label = None, None
        if prefix == "B" or (prefix == "I" and start is None):      # 新しいスパンが始まる
            start, label = i, lab
    return spans

tokens = ["山田", "太郎", "は", "京都", "大学", "を", "卒業", "し", "、", "東京", "都", "で", "働く"]
tags = ["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "O", "O", "O", "O", "B-LOC", "I-LOC", "O", "O"]
for s, e, label in bio_to_spans(tags):
    print(label, "".join(tokens[s:e]))
```

```python
# モデルの出力にありがちな不正な系列
bad = ["I-PER", "I-PER", "O", "B-ORG", "I-LOC", "O", "B-PER", "B-PER"]
print(bio_to_spans(bad))

def repair(tags):
    """不正な I- を B- に直して、正しい BIO 系列にする"""
    out, prev = [], "O"
    for tag in tags:
        if tag.startswith("I-") and prev[2:] != tag[2:]:
            tag = "B-" + tag[2:]
        out.append(tag)
        prev = tag
    return out

print(repair(bad))
```

- 不正な系列の扱い方（捨てる / 直す）で評価値が変わる。seqeval のデフォルトは、ここと同じく「I- で始まっても新しいスパンとみなす」寛容な方式。
- 根本的に防ぐには、CRF 層や、遷移制約付きのビタビ復号を使う。

## 固有表現抽出をエンティティ単位で評価する (seqeval) {seqeval}

系列ラベリングの結果を、トークン単位の正解率ではなく、エンティティ単位の適合率・再現率・F1 で評価したい。

```python
!pip install -q seqeval
```

```python
from seqeval.metrics import accuracy_score, classification_report, f1_score
from seqeval.scheme import IOB2

y_true = [["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "O", "O"], ["B-LOC", "O", "O", "B-PER", "O"], ["O", "O", "B-ORG", "I-ORG", "I-ORG"]]
y_pred = [["B-PER", "I-PER", "O", "B-ORG", "O", "O", "O"], ["B-LOC", "O", "O", "B-PER", "O"], ["O", "O", "B-LOC", "I-LOC", "I-LOC"]]

print("トークン単位の正解率:", round(accuracy_score(y_true, y_pred), 3))
print("エンティティ単位の F1:", round(f1_score(y_true, y_pred), 3))
print(classification_report(y_true, y_pred, digits=3))
```

```python
# 厳密モード: 不正な系列 (B- なしの I-) をエンティティとして数えない
bad_pred = [["I-PER", "I-PER", "O", "B-ORG", "I-ORG", "O", "O"], ["B-LOC", "O", "O", "B-PER", "O"], ["O", "O", "B-ORG", "I-ORG", "I-ORG"]]
print("default:", f1_score(y_true, bad_pred), " strict:", f1_score(y_true, bad_pred, mode="strict", scheme=IOB2))
```

- エンティティ単位では、範囲とラベルの両方が完全に一致したときだけ正解。境界が 1 トークンずれても、偽陽性 1 + 偽陰性 1 になる。
- O タグが大半を占めるので、トークン単位の正解率は高く出すぎて指標にならない。
- 論文で報告されるのは通常 micro 平均の F1。

## CRF で固有表現抽出器を学習する {crf_tagger}

ニューラルネットを使わずに、手作りの特徴量と条件付き確率場 (CRF) で系列ラベリングをしたい。CPU で数十秒で学習できる。

```python
!pip install -q sklearn-crfsuite seqeval
```

```python
import nltk
nltk.download("conll2002", quiet=True)
from nltk.corpus import conll2002

train = list(conll2002.iob_sents("esp.train"))   # スペイン語の固有表現データ: (単語, 品詞, BIO タグ)
test = list(conll2002.iob_sents("esp.testb"))
print(len(train), len(test), train[0][:6])

def word_features(sent, i):
    w, pos = sent[i][0], sent[i][1]
    f = {"bias": 1.0, "lower": w.lower(), "suffix3": w[-3:], "suffix2": w[-2:], "is_upper": w.isupper(),
         "is_title": w.istitle(), "is_digit": w.isdigit(), "pos": pos}
    for d in (-1, 1):   # 前後 1 語の特徴
        if 0 <= i + d < len(sent):
            w2, pos2 = sent[i + d][0], sent[i + d][1]
            f.update({f"{d}:lower": w2.lower(), f"{d}:is_title": w2.istitle(), f"{d}:pos": pos2})
        else:
            f["BOS" if d == -1 else "EOS"] = True
    return f

X_train = [[word_features(s, i) for i in range(len(s))] for s in train]
y_train = [[t for _, _, t in s] for s in train]
X_test = [[word_features(s, i) for i in range(len(s))] for s in test]
y_test = [[t for _, _, t in s] for s in test]
```

```python
import sklearn_crfsuite
from seqeval.metrics import classification_report

crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
crf.fit(X_train, y_train)
print(classification_report(y_test, crf.predict(X_test), digits=3))
```

```python
# 学習された遷移の重み: O → I-xxx のような不正な遷移には大きな負の重みが付く
from collections import Counter

trans = Counter(crf.transition_features_)
print("起きやすい遷移:", [(a, b, round(w, 1)) for (a, b), w in trans.most_common(5)])
print("起きにくい遷移:", [(a, b, round(w, 1)) for (a, b), w in trans.most_common()[-5:]])
```

- CRF はラベル間の遷移も学習するので、B- なしの I- のような不正な系列を出しにくい。
- 特徴量を自分で設計できる・学習が速い・挙動を解釈できるのが利点。少量データや計算資源が限られる場面では今でも有力。

## BERT をファインチューニングして日本語の固有表現抽出器を作る {bert_ner_finetune gpu}

日本語の固有表現抽出モデルを、文字位置のアノテーション付きデータから学習したい。

```python
!pip install -q datasets seqeval accelerate
```

```python
import json, urllib.request
from datasets import Dataset, DatasetDict

# ストックマーク社の Wikipedia 固有表現データセット (CC BY-SA 3.0)
url = "https://raw.githubusercontent.com/stockmarkteam/ner-wikipedia-dataset/main/ner.json"
data = json.load(urllib.request.urlopen(url))
print(len(data), data[0])

types = sorted({e["type"] for d in data for e in d["entities"]})
labels = ["O"] + [f"{p}-{t}" for t in types for p in ("B", "I")]
label2id = {l: i for i, l in enumerate(labels)}
ds = Dataset.from_list([{"text": d["text"], "starts": [e["span"][0] for e in d["entities"]],
                         "ends": [e["span"][1] for e in d["entities"]], "types": [e["type"] for e in d["entities"]]} for d in data])
# テストを先に確保し、残りから開発用を取り分ける（学習・開発・テストは約 70・10・20%）
holdout = ds.train_test_split(test_size=0.2, seed=0)
train_dev = holdout["train"].train_test_split(test_size=0.125, seed=0)
ds = DatasetDict(train=train_dev["train"], dev=train_dev["test"], test=holdout["test"])
print(ds)
```

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

model_name = "xlm-roberta-base"   # 文字位置 (offset mapping) を返せる高速トークナイザを持つモデルを使う
tokenizer = AutoTokenizer.from_pretrained(model_name)

def encode(example):
    enc = tokenizer(example["text"], truncation=True, max_length=256, return_offsets_mapping=True)
    tags = []
    for tok, (s, e) in zip(enc.tokens(), enc["offset_mapping"]):
        if s == e:                                  # 特殊トークンは損失計算から除外
            tags.append(-100)
            continue
        tag = "O"
        if tok == "▁":                              # 単独の「▁」は次の文字と同じ位置を返すので、常に O にする
            tags.append(label2id[tag])
            continue
        for start, end, typ in zip(example["starts"], example["ends"], example["types"]):
            if s < end and e > start:               # トークンがスパンと重なる
                tag = ("B-" if s <= start else "I-") + typ
                break
        tags.append(label2id[tag])
    enc["labels"] = tags
    enc.pop("offset_mapping")
    return enc

tokenized = ds.map(encode, remove_columns=["text", "starts", "ends", "types"])
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label2id)
```

```python
import numpy as np
from seqeval.metrics import classification_report, f1_score
from transformers import Trainer, TrainingArguments

def to_tags(logits, gold):
    pred = np.argmax(logits, axis=-1)
    y_true = [[labels[g] for g in row if g != -100] for row in gold]
    y_pred = [[labels[p] for p, g in zip(prow, grow) if g != -100] for prow, grow in zip(pred, gold)]
    return y_true, y_pred

def compute_metrics(eval_pred):
    y_true, y_pred = to_tags(*eval_pred)
    return {"f1": f1_score(y_true, y_pred)}

args = TrainingArguments(
    output_dir="ner-out", num_train_epochs=5, learning_rate=5e-5, warmup_steps=60, weight_decay=0.01,
    per_device_train_batch_size=32, per_device_eval_batch_size=64,
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="f1",
    fp16=True, logging_steps=50, report_to="none", seed=0,
)
trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["dev"],
                  data_collator=DataCollatorForTokenClassification(tokenizer), compute_metrics=compute_metrics)
trainer.train()
# 開発セットの F1 が最も高かったモデルで、テストセットを 1 回だけ評価する
out = trainer.predict(tokenized["test"])
print(classification_report(*to_tags(out.predictions, out.label_ids), digits=3))
```

```python
from transformers import pipeline

ner = pipeline("token-classification", model=trainer.model, tokenizer=tokenizer, aggregation_strategy="simple", device=0)
text = "ソニーグループの吉田憲一郎会長は、東京都港区の本社で新型プレイステーションを発表した。"
for e in ner(text):
    print(text[e["start"]:e["end"]], e["entity_group"], round(float(e["score"]), 2))
```

- データを学習・開発・テスト（約 70・10・20%）に 3 分割し、各エポックの後に開発セットのエンティティ F1 を計算して最良のチェックポイント（ここでは 5 エポック目、開発 F1 0.812）を復元し、テストセットは最後に 1 回だけ評価した。
- 検証環境の L4 GPU では 5 エポックの学習に約 2 分かかり、テストセットでは micro F1 が 0.825、macro F1 が 0.805（人名 0.907、地名 0.864、法人名 0.844 など）だった。
- ここではサブワードすべてにラベルを付けている（先頭だけに付けて残りを -100 にする流儀もある）。評価は本来、単語や文字の単位に戻して行うのが厳密。
- XLM-R のトークナイザは、文頭や空白の後に単独の `▁` トークンを置き、その文字位置を次の文字と同じ `(0, 1)` のように返す。そのまま位置で判定すると `▁` と次のトークンの両方が `B-` になり、推論時に「ソ」と「ソニーグループ」のように同じ固有表現が二重に出る。
- 日本語特化の BERT (`tohoku-nlp/bert-base-japanese-v3`) のほうが精度は高いが、トークナイザが文字位置を返さないので、位置合わせを自分で実装する必要がある。

## 個人情報をマスクする（仮名化） {pii_mask}

テキスト中の人名・地名・電話番号・メールアドレスなどを伏せ字にして、データを共有できる形にしたい。

```python
!pip install -q ginza ja_ginza
```

```python
import re
import spacy

nlp = spacy.load("ja_ginza")

REGEX = {
    "EMAIL": r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
    "PHONE": r"(?<!\d)0\d{1,4}-\d{1,4}-\d{4}(?!\d)",
    "POSTAL": r"(?<![\d-])〒?\d{3}-\d{4}(?![\d-])",
}
NER_LABELS = {"Person": "PERSON", "City": "LOCATION", "Province": "LOCATION", "Company": "ORG", "School": "ORG"}

def mask(text):
    spans = []
    for label, pat in REGEX.items():
        spans += [(m.start(), m.end(), label) for m in re.finditer(pat, text)]
    for ent in nlp(text).ents:
        if ent.label_ in NER_LABELS:
            spans.append((ent.start_char, ent.end_char, NER_LABELS[ent.label_]))
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    out, pos = [], 0
    for start, end, label in spans:
        if start < pos:      # 重なりは先に始まる (長い) ほうを優先
            continue
        out += [text[pos:start], f"<{label}>"]
        pos = end
    return "".join(out) + text[pos:]

text = "山田太郎(yamada@example.com、090-1234-5678)は横浜市在住で、ソニーに勤務している。佐藤花子は東京大学の学生です。連絡先は〒191-0065、会員番号1234-5678-9012-3456。"
print(mask(text))
```

- NER の再現率は 100% ではない。公開前には人手の確認が必要。再現率を優先するなら複数のモデルの和集合をとる。
- 単にマスクすると文脈が失われる。「山田太郎 → 人物A」のように同じ実体に同じ仮名を割り当てると、後段の分析に使いやすい。
- 郵便番号や電話番号の正規表現には `(?<![\d-])...(?![\d-])` や `(?<!\d)...(?!\d)` のような否定後読み・先読み境界を入れ、カード番号や会員番号（例: `1234-5678-9012-3456` の一部）など長い番号列への誤マッチを防ぐ。
- 英語を含む多言語なら Microsoft Presidio (`presidio-analyzer`) が定番。

## 音声認識の結果から固有表現を抽出する {asr_ner}

会議の録音やコールセンターの通話のような音声から、人名・地名・組織名を取り出したい。音声認識の誤りが固有表現抽出にどう影響するかも確かめる。

```python
!pip install -q gTTS ginza ja-ginza jiwer
!apt-get install -y -q ffmpeg > /dev/null 2>&1
```

```python
from gtts import gTTS

# 手元に音声がないので、ここでは合成音声で試す (実際には録音した音声ファイルを使う)
texts = [
    "山田太郎さんは来週の月曜日に大阪の本社で開かれる会議に出席します。",
    "京都大学の研究チームは、理化学研究所と共同で新しい実験を行いました。",
    "ソニーとホンダは、電気自動車を開発する新会社を東京に設立すると発表した。",
    "総理大臣は明日、ワシントンでアメリカの大統領と会談する予定です。",
    "鈴木花子選手は、パリで行われた大会で金メダルを獲得しました。",
    "北海道の函館市では、昨日から大雪が続いており、JR 北海道は一部の列車を運休しています。",
]
for i, t in enumerate(texts):
    gTTS(t, lang="ja").save(f"audio{i}.mp3")
!ls -la audio*.mp3 | head -n 3
```

```python
import torch
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", generate_kwargs={"language": "japanese", "task": "transcribe"})
transcripts = [asr(f"audio{i}.mp3")["text"].strip() for i in range(len(texts))]
for ref, hyp in zip(texts, transcripts):
    print(f"正解: {ref}\n認識: {hyp}")
```

```python
import jiwer
import spacy

nlp = spacy.load("ja_ginza")
normalize = lambda s: s.replace(" ", "").replace("、", "").replace("。", "")
print("文字誤り率 (CER):", round(jiwer.cer([normalize(t) for t in texts], [normalize(t) for t in transcripts]), 3))

def entities(text):
    return {(e.text, e.label_) for e in nlp(text).ents}

tp = fp = fn = 0
for ref, hyp in zip(texts, transcripts):
    gold, pred = entities(ref), entities(hyp)          # 正解の書き起こしから取り出した固有表現を「正解」と見なす
    tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
    if gold != pred:
        print(f"書き起こしから: {sorted(gold - pred)}\n認識結果から  : {sorted(pred - gold)}")
p, r = tp / (tp + fp), tp / (tp + fn)
print(f"固有表現の一致: 適合率 {p:.2f} / 再現率 {r:.2f} / F1 {2 * p * r / (p + r):.2f}")
```

```python
# 表記のゆれを吸収して比べる: 読み (カタカナ) が同じなら一致と見なす
def readings(text):
    doc = nlp(text)
    return {("".join(t.morph.get("Reading")[0] if t.morph.get("Reading") else t.text for t in e), e.label_) for e in doc.ents}

tp = fp = fn = 0
for ref, hyp in zip(texts, transcripts):
    gold, pred = readings(ref), readings(hyp)
    tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
p, r = tp / (tp + fp), tp / (tp + fn)
print(f"読みで比べた場合: 適合率 {p:.2f} / 再現率 {r:.2f} / F1 {2 * p * r / (p + r):.2f}")
```

- 音声認識の結果には、句読点がない（またはモデルが推定したもの）、同音異義語の誤り（人名の漢字が違う）、数字や英字の表記のゆれ、といった書き言葉との違いがある。書き言葉で学習した固有表現抽出器は、その分だけ精度が落ちる。
- 固有名詞は音声認識が最も間違えやすい部分。特に人名は、音が合っていても漢字が違うことが多い。読みで照合する、候補の辞書と突き合わせる、といった後処理が有効。
- 読みで照合しても、この例では結果が変わらなかった。「ホンダ」が「本田」になると、表記だけでなく種類の判定まで（会社 → 人名）変わってしまうため。音声認識の誤りは、固有表現の範囲と種類の両方に波及する。
- 全体の文字誤り率が低くても、固有表現に限ると誤りが多いことがある。目的が固有表現なら、固有表現だけの誤り率を測る。
- ここで使ったのは合成音声で、実際の会話音声（雑音、言いよどみ、複数話者）よりはるかに認識しやすい。実データでの性能は、必ず実データで測る。
- 音声認識と固有表現抽出を別々に行う方法（ここでの方法）のほかに、音声から直接固有表現を出すモデルや、認識結果の誤りを LLM で訂正してから抽出する方法がある。

## EC の商品名からブランド・色・サイズ・素材・容量を取り出す（辞書と正規表現、BERT、LLM） {product_attribute_extraction gpu}

通販サイトの商品名から、ブランド・色・サイズ・素材・容量といった属性の値を取り出し、「Mサイズ」と「M」、「ブラック」と「黒」のような表記の違いをそろえて、商品マスタの値と突き合わせたい。辞書と正規表現、BERT のトークン分類、LLM による JSON 抽出を、属性ごとの適合率・再現率で比べる。

```python
import random, re, unicodedata
import numpy as np
import pandas as pd

ATTRS = ["brand", "color", "size", "material", "capacity"]
# 評価用の商品名（このレシピのために作ったもの。実在のブランド名や商品名・型番も含むが、属性の組み合わせは実際の商品の仕様とは限らない）
# [表記|属性] が正解。表記と正規化した値が違うときは [表記|属性=正規化した値] と書く
TEST = """\
【送料無料】[NIKE|brand=ナイキ] エア ズーム ペガサス ランニングシューズ メンズ [ブラック|color=黒] [26.5cm|size]
[ユニクロ|brand] クルーネックT 半袖 [ホワイト|color=白] [Mサイズ|size=M] [綿|material]60% [ポリエステル|material]40%
[サーモス|brand] 真空断熱ケータイマグ [500ml|capacity] [ステンレス|material] [ネイビー|color=紺] JNL-506
[THERMOS|brand=サーモス] 真空断熱タンブラー [0.4L|capacity=400ml] [シルバー|color=銀]
[象印|brand] [ステンレス|material]マグ シームレスせん [480mL|capacity=480ml] [ピンク|color]
[アイリスオーヤマ|brand] 電気ケトル [1.0L|capacity=1000ml] [ホワイト|color=白] 空焚き防止
[タイガー|brand] 魔法瓶 [ステンレス|material]ボトル [600ml|capacity] [マットブラック|color=黒]
[ワークマン|brand] 防水防寒ジャケット [LL|size=XL] [ブラック|color=黒]
[ニトリ|brand] [綿|material]100% ボックスシーツ [シングル|size] [グレー|color=灰]
【楽天1位】[無印良品|brand] [麻|material]平織 ベッドシーツ [セミダブル|size] [ネイビー|color=紺]
[エレコム|brand] USBメモリ [64GB|capacity] USB3.2 [ブルー|color=青]
[SanDisk|brand=サンディスク] microSDXCカード [128GB|capacity] UHS-I
[バッファロー|brand] 外付けHDD [2TB|capacity] [黒|color] テレビ録画対応
[Anker|brand=アンカー] モバイルバッテリー [10000mAh|capacity] [ブラック|color=黒]
[アディダス|brand] スタンスミス スニーカー [ホワイト|color=白]×[グリーン|color=緑] [25.0cm|size=25cm]
[New Balance|brand=ニューバランス] 996 スニーカー [グレー|color=灰] [27cm|size]
[コールマン|brand] アウトドアチェア [スチール|material]フレーム [グリーン|color=緑]
[山善|brand] 扇風機 リビング扇 [ホワイト|color=白] リモコン付き
[ロジクール|brand] ワイヤレスマウス 静音 [グラファイト|color=灰]
【まとめ買い】天然水 [2L|capacity=2000ml]×9本
[パナソニック|brand] ドライヤー ナノケア [ディープネイビー|color=紺]
[カシオ|brand] 腕時計 スタンダード [ステンレス|material]バンド [シルバー|color=銀]
メンズ ビジネスシューズ [本革|material=革] [ブラウン|color=茶] [26.0cm|size=26cm] 幅広 3E
レディース [ウール|material]100% ニットカーディガン [ベージュ|color] [フリーサイズ|size=フリー]
綿棒 200本入り 抗菌 個包装
ホワイトニング 歯磨き粉 薬用 [90g|capacity]
L字デスク [木|material]製 幅120cm [ブラウン|color=茶]
麻婆豆腐の素 中辛 [150g|capacity]
[PUレザー|material=合成皮革] トートバッグ レディース 大容量 A4対応 [キャメル|color=茶]
[STANLEY|brand=スタンレー] クラシック真空ボトル [1.0L|capacity=1000ml] [グリーン|color=緑]
[mont-bell|brand=モンベル] ダウンジャケット [ナイロン|material] [XL|size] [レッド|color=赤]
キッズ Tシャツ [130cm|size] [イエロー|color=黄] [コットン|material=綿]100%
[ユニクロ|brand] ウルトラライトダウン [ネイビー|color=紺] [L|size]
[無印良品|brand] [ポリプロピレン|material]収納ケース 引出式 深型
[Dyson|brand=ダイソン] コードレス掃除機 V8 [シルバー|color=銀]
[ナイキ|brand] ドライフィット ハーフパンツ [S|size] [ブラック|color=黒]/[ホワイト|color=白]
[アイリスオーヤマ|brand] 布団乾燥機 カラリエ [ピンク|color]
[象印|brand] 炊飯器 [5.5合|capacity] 極め炊き [黒|color]
[タイガー|brand] 電気ポット [2.2L|capacity=2200ml] 蒸気レス [ブラウン|color=茶]
国産 [ヒノキ|material] まな板 [L|size]
[エレコム|brand] スマホケース iPhone15 [シリコン|material] [ブラック|color=黒]
レインコート 自転車 [LL|size=XL] 男女兼用 [カーキ|color]
[ニトリ|brand] 遮光カーテン 幅100×丈178cm 2枚組 [ベージュ|color]
[パナソニック|brand] 充電池 エネループ 単3形 4本パック [1900mAh|capacity]
【2個セット】[ガラス|material]製 保存容器 [800ml|capacity] 耐熱
[カシオ|brand] 電子辞書 エクスワード [ホワイト|color=白]
メンズ チノパン ストレッチ [W82|size] [ベージュ|color]
[THERMOS|brand=サーモス] フライパン [26cm|size] IH対応
[アディダス|brand] スポーツバッグ [ポリエステル|material] [30L|capacity=30000ml] [ネイビー|color=紺]
[ワークマン|brand] 作業用 手袋 [M|size] 滑り止め [グレー|color=灰]
[Coleman|brand=コールマン] ソフトクーラー [15L|capacity=15000ml] [ブルー|color=青]
[ロジクール|brand] キーボード 日本語配列 [オフホワイト|color=白]
子供用 長靴 [18cm|size] [レッド|color=赤] [ゴム|material]製
[UNIQLO|brand=ユニクロ] ヒートテック タイツ [XXL|size] [黒|color]
[アンカー|brand] USB充電器 65W [ホワイト|color=白]
[山善|brand] こたつ 長方形 [天然木|material=木] 105×75cm
[無印良品|brand] [オーガニックコットン|material=綿] オックスシャツ [S|size] [サックスブルー|color=青]
[象印|brand] 水筒 [0.36L|capacity=360ml] [ステンレス|material]マグ [ミントブルー|color=青]
USB-C ハブ 7in1 [スペースグレー|color=灰] [アルミ|material]
[ナイキ|brand] リュック [25L|capacity=25000ml] [ブラック|color=黒]
大きいサイズ メンズ ポロシャツ [3L|size=XXL] [ネイビー|color=紺] [綿|material]100%"""

MARK = re.compile(r"\[([^\[\]|]+)\|(\w+)(?:=([^\]]+))?\]")

def parse(line):
    """[表記|属性=値] の印を外して、(本文, [(開始, 終了, 属性, 正規化した値), ...]) を返す"""
    text, spans, pos = "", [], 0
    for m in MARK.finditer(line):
        text += line[pos:m.start()]
        spans.append((len(text), len(text) + len(m[1]), m[2], m[3] or m[1]))
        text += m[1]
        pos = m.end()
    return text + line[pos:], spans

test = [parse(line) for line in TEST.split("\n")]
print(len(test), "件。属性ごとの正解の数:", pd.Series([a for _, g in test for *_, a, _ in g]).value_counts().to_dict())
print("表記と正規化した値が違う正解:", sum(s != c for line in TEST.split("\n") for s, _, c in MARK.findall(line) if c), "個")
print(test[1])
```

```python
# 値の正規化: 商品マスタの値（左）と、その別表記（右）。辞書による抽出にも同じ表を使う
BRANDS = {"ナイキ": ["NIKE"], "アディダス": ["adidas"], "ユニクロ": ["UNIQLO"], "無印良品": ["MUJI"], "サーモス": ["THERMOS"],
          "象印": ["ZOJIRUSHI"], "タイガー": ["TIGER"], "アイリスオーヤマ": ["IRIS OHYAMA"], "パナソニック": ["Panasonic"],
          "エレコム": ["ELECOM"], "サンディスク": ["SanDisk"], "コールマン": ["Coleman"], "アンカー": ["Anker"],
          "ニューバランス": ["New Balance"]}
COLORS = {"黒": ["ブラック", "BLACK", "黒色"], "白": ["ホワイト", "WHITE", "白色"], "赤": ["レッド", "RED"], "青": ["ブルー", "BLUE"],
          "紺": ["ネイビー", "NAVY"], "灰": ["グレー", "グレイ", "GRAY"], "茶": ["ブラウン", "BROWN"], "緑": ["グリーン", "GREEN"],
          "黄": ["イエロー", "YELLOW"], "銀": ["シルバー", "SILVER"], "ピンク": ["PINK"], "ベージュ": [], "カーキ": []}
MATERIALS = {"綿": ["コットン", "cotton"], "革": ["本革", "レザー", "牛革"], "合成皮革": ["合皮", "PUレザー", "フェイクレザー"],
             "麻": ["リネン"], "ウール": ["羊毛"], "木": ["天然木", "木製"], "ステンレス": [], "ポリエステル": [], "ナイロン": [],
             "ガラス": [], "シリコン": ["シリコーン"]}
TABLES = {"brand": BRANDS, "color": COLORS, "material": MATERIALS}
ALIAS = {attr: {unicodedata.normalize("NFKC", a).lower(): canon for canon, aliases in table.items() for a in [canon, *aliases]}
         for attr, table in TABLES.items()}
SIZE_ALIAS = {"LL": "XL", "2L": "XL", "3L": "XXL", "4L": "XXXL", "F": "フリー"}
UNITS = {"ml": ("ml", 1), "ミリリットル": ("ml", 1), "l": ("ml", 1000), "リットル": ("ml", 1000), "g": ("g", 1), "kg": ("g", 1000),
         "gb": ("GB", 1), "tb": ("TB", 1), "mah": ("mAh", 1), "合": ("合", 1)}

def normalize(attr, value):
    v = unicodedata.normalize("NFKC", str(value)).strip()   # 全角英数字→半角、「ℓ」→「l」など
    if attr in ALIAS:
        alias = ALIAS[attr]
        if v.lower() in alias:
            return alias[v.lower()]
        hits = [a for a in alias if a in v.lower()]            # 「マットブラック」→「ブラック」→「黒」
        return alias[max(hits, key=len)] if hits else v
    if attr == "size":
        v = re.sub(r"サイズ|\s", "", v.upper())
        m = re.fullmatch(r"(\d+(?:\.\d+)?)CM", v)
        return f"{float(m[1]):g}cm" if m else SIZE_ALIAS.get(v, v)
    if attr == "capacity":
        m = re.search(r"(\d+(?:\.\d+)?)\s*(mah|ml|ミリリットル|リットル|l|kg|g|gb|tb|合)", v.lower())
        if m:
            unit, k = UNITS[m[2]]
            return f"{float(m[1]) * k:g}{unit}"
    return v

for attr, v in [("size", "Mサイズ"), ("size", "Ｍ"), ("size", "3L"), ("size", "25.0cm"), ("color", "マットブラック"),
                ("color", "BLACK"), ("material", "PUレザー"), ("capacity", "1.0L"), ("capacity", "480mL"), ("capacity", "５００ｍｌ")]:
    print(f"{attr:9}{v:10} → {normalize(attr, v)}")
```

```python
def evaluate(preds, name, use_norm=True):
    """preds: 商品ごとの [(属性, 値), ...]。(属性, 正規化した値) の集合で正解と比べ、属性ごとの適合率・再現率・F1 を返す"""
    counts = {a: np.zeros((len(test), 3)) for a in ATTRS}      # 商品ごとの (正しい予測, 誤った予測, 見落とし)
    for i, ((text, gold), pred) in enumerate(zip(test, preds)):
        g = {(a, c) for *_, a, c in gold}
        p = {(a, normalize(a, v) if use_norm else str(v)) for a, v in pred if a in ATTRS and str(v).strip()}
        for a in ATTRS:
            ga, pa = {x for x in g if x[0] == a}, {x for x in p if x[0] == a}
            counts[a][i] = len(ga & pa), len(pa - ga), len(ga - pa)
    def prf(tp, fp, fn):
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        return p, r, 2 * p * r / max(p + r, 1e-9)
    rows = {a: prf(*counts[a].sum(0)) for a in ATTRS}
    total = sum(counts.values())
    rows["全体 (micro)"] = prf(*total.sum(0))
    rng = np.random.default_rng(0)                            # 商品を復元抽出して micro F1 の 95% 信頼区間を出す
    boot = [prf(*total[rng.integers(0, len(test), len(test))].sum(0))[2] for _ in range(1000)]
    df = pd.DataFrame(rows, index=["P", "R", "F1"]).T.round(2)
    df.columns = pd.MultiIndex.from_product([[name], df.columns])
    return df, np.percentile(boot, [2.5, 97.5])
```

```python
# 方法 1: 辞書と正規表現
def dict_regex(table_alias):
    words = sorted(table_alias, key=len, reverse=True)       # 長い語を先に（「PUレザー」を「レザー」より優先）
    return re.compile("|".join(re.escape(w) for w in words), re.IGNORECASE)

DICT_RE = {attr: dict_regex(alias) for attr, alias in ALIAS.items()}
SIZE_RE = re.compile(r"(?<![A-Za-z0-9.])(?:XXL|XL|LL|[2-4]L|S|M|L|フリー)(?:サイズ)?(?![A-Za-z0-9])|\d{2,3}(?:\.\d)?cm|W\d{2}")
CAPACITY_RE = re.compile(r"\d+(?:\.\d+)?(?:mAh|ml|mL|GB|TB|kg|g|L|合)(?![A-Za-z])")

def extract_rules(text):
    text = unicodedata.normalize("NFKC", text)
    out = [(attr, m[0]) for attr, pat in DICT_RE.items() for m in pat.finditer(text)]
    out += [("size", m[0]) for m in SIZE_RE.finditer(text)] + [("capacity", m[0]) for m in CAPACITY_RE.finditer(text)]
    return out

rule_preds = [extract_rules(t) for t, _ in test]
for (t, _), p in list(zip(test, rule_preds))[24:29]:
    print(t, "→", p)
results = {}
for use_norm in (False, True):
    df, ci = evaluate(rule_preds, "辞書+正規表現", use_norm)
    print(f"正規化{'あり' if use_norm else 'なし'}: micro F1 {df.iloc[-1, 2]:.2f} (95% CI {ci[0]:.2f}〜{ci[1]:.2f})")
    if use_norm:
        results["辞書+正規表現"] = (df, ci)
df
```

```python
# 方法 2 の準備: BERT の学習データをテンプレートから自動で作る（人手のアノテーションなし）
rng = random.Random(0)
KANA = list("アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン")

def rand_brand():
    """辞書のブランドに加えて、ランダムな名前も混ぜる（辞書にないブランドにも対応できるように）"""
    if rng.random() < 0.5:
        canon = rng.choice(list(BRANDS))
        return rng.choice([canon, *BRANDS[canon]])
    if rng.random() < 0.6:
        return "".join(rng.choices(KANA, k=rng.randint(2, 5))) + rng.choice(["", "", "ー", "ン"])
    return "".join(rng.choices("abcdefghijklmnoprstuvwxyz", k=rng.randint(3, 7))).capitalize()

def pick(table):
    canon = rng.choice(list(table))
    return rng.choice([canon, *table[canon]])

SLOTS = {
    "color": lambda: rng.choice(["", "", "マット", "ダーク", "ライト"]) + pick(COLORS),
    "size": lambda: rng.choice(["S", "M", "L", "XL", "LL", "3L", "XXL"]) + rng.choice(["", "", "サイズ"]),
    "shoe": lambda: f"{rng.choice(range(22, 30))}{rng.choice(['', '.0', '.5'])}cm",
    "bed": lambda: rng.choice(["シングル", "セミダブル", "ダブル"]),
    "material": lambda: pick(MATERIALS),
    "ml": lambda: rng.choice([f"{rng.choice([300, 350, 480, 500, 600, 750, 800])}{rng.choice(['ml', 'mL'])}",
                              f"{rng.choice(['0.5', '1.0', '1.2', '1.5', '2', '2.2'])}L"]),
    "gb": lambda: f"{rng.choice([16, 32, 64, 128, 256, 512])}GB" if rng.random() < 0.8 else f"{rng.choice([1, 2, 4])}TB",
    "mah": lambda: f"{rng.choice([5000, 10000, 20000])}mAh",
}
LABEL = {"shoe": "size", "bed": "size", "ml": "capacity", "gb": "capacity", "mah": "capacity"}
CATEGORIES = [
    (["Tシャツ", "パーカー", "ジャケット", "シャツ", "パンツ", "ニット", "ポロシャツ", "スカート"], ["color", "size", "material"]),
    (["スニーカー", "ランニングシューズ", "サンダル", "ブーツ"], ["color", "shoe"]),
    (["水筒", "タンブラー", "マグボトル", "電気ケトル"], ["ml", "color", "material"]),
    (["USBメモリ", "SDカード", "外付けSSD"], ["gb", "color"]),
    (["モバイルバッテリー"], ["mah", "color"]),
    (["シーツ", "布団カバー", "枕カバー"], ["color", "bed", "material"]),
    (["ワイヤレスイヤホン", "電動歯ブラシ", "デスクライト", "財布", "ショルダーバッグ", "スマートウォッチ", "加湿器"], ["color", "material"]),
]
NOISE = ["", "", "", "【送料無料】", "【公式】", "ポイント10倍 ", "2024年モデル ", "【あす楽】"]
# 属性ではない語（宣伝文句や機能）も混ぜておかないと、属性以外の語まで属性として取り出すようになる
EXTRA = ["メンズ", "レディース", "キッズ", "大容量", "軽量", "撥水", "保温", "保冷", "速乾", "UVカット", "おしゃれ", "ギフト", "人気",
         "新作", "限定", "日本製", "洗える", "折りたたみ", "コンパクト", "3枚セット", "母の日", "プレゼント", "通勤", "アウトドア"]

def make_example():
    nouns, slots = rng.choice(CATEGORIES)
    parts = [(rng.choice(NOISE), None)]
    if rng.random() < 0.85:
        parts += [(rand_brand(), "brand"), (" ", None)]
    parts += [(rng.choice(nouns), None)]
    for slot in rng.sample(slots, len(slots)):
        if rng.random() < 0.8:
            value, label = SLOTS[slot](), LABEL.get(slot, slot)
            parts += [(" ", None), (value, label)]
            if label == "material":
                parts += [(rng.choice(["", "100%", "素材", "製"]), None)]
    for _ in range(rng.choice([0, 1, 1, 2, 3])):
        gaps = [k for k, (s, _) in enumerate(parts) if s == " "] + [len(parts)]
        parts.insert(rng.choice(gaps), (f" {rng.choice(EXTRA)} ", None))
    text, spans = "", []
    for s, label in parts:
        if not label and text.endswith(" "):
            s = s.lstrip()                                     # 空白が続かないようにする
        if label:
            spans.append((len(text), len(text) + len(s), label))
        text += s
    return text.rstrip(), spans

train = [make_example() for _ in range(3000)]
for t, s in train[:3]:
    print(t, [(t[a:b], l) for a, b, l in s])
```

```python
import torch
from datasets import Dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          Trainer, TrainingArguments, pipeline)

torch.manual_seed(0)
BF16 = torch.cuda.get_device_capability()[0] >= 8
labels = ["O"] + [f"{p}-{a}" for a in ATTRS for p in ("B", "I")]
label2id = {l: i for i, l in enumerate(labels)}
model_name = "xlm-roberta-base"   # MIT ライセンス。文字位置 (offset mapping) を返す高速トークナイザーがある
tokenizer = AutoTokenizer.from_pretrained(model_name)

def encode(example):
    enc = tokenizer(example["text"], truncation=True, max_length=96, return_offsets_mapping=True)
    tags = []
    for tok, (s, e) in zip(enc.tokens(), enc["offset_mapping"]):
        tag = "O"
        if s == e:
            tags.append(-100)                                # 特殊トークンは損失の計算から除く
            continue
        if tok != "▁":                                       # 単独の「▁」は次の文字と同じ位置を返すので O にする（この章の「BERT をファインチューニングして日本語の固有表現抽出器を作る」を参照）
            for start, end, attr in zip(example["starts"], example["ends"], example["attrs"]):
                if s < end and e > start:
                    tag = ("B-" if s <= start else "I-") + attr
                    break
        tags.append(label2id[tag])
    enc["labels"] = tags
    enc.pop("offset_mapping")
    return enc

ds = Dataset.from_list([{"text": t, "starts": [a for a, _, _ in s], "ends": [b for _, b, _ in s], "attrs": [l for *_, l in s]}
                        for t, s in train]).map(encode, remove_columns=["text", "starts", "ends", "attrs"])
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(labels),
                                                        id2label=dict(enumerate(labels)), label2id=label2id)
args = TrainingArguments(output_dir="attr-out", num_train_epochs=3, learning_rate=5e-5, warmup_steps=30,
                         per_device_train_batch_size=32, save_strategy="no", logging_steps=50, report_to="none",
                         seed=0, bf16=BF16, fp16=not BF16)
trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=DataCollatorForTokenClassification(tokenizer))
trainer.train()   # T4 で 1〜2 分（出力の train_runtime は検証の GPU での値）
```

```python
# 方法 2: BERT (XLM-R) のトークン分類で、評価用の商品名から属性を取り出す
ner = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=0)
raw = ner([t for t, _ in test])
bert_preds = [[(e["entity_group"], t[e["start"]:e["end"]].strip()) for e in ents] for (t, _), ents in zip(test, raw)]
for (t, _), ents in list(zip(test, raw))[24:29]:
    print(t, "→", [(e["entity_group"], t[e["start"]:e["end"]].strip(), round(float(e["score"]), 2)) for e in ents])
results["BERT"] = evaluate(bert_preds, "BERT")
print(f"micro F1 95% CI: {results['BERT'][1][0]:.2f}〜{results['BERT'][1][1]:.2f}")
# 確信度 (score) の低いものを捨てると、適合率が上がり再現率が下がる。閾値は本来、評価データとは別の開発データで決める
for th in [0.5, 0.7, 0.9]:
    kept = [[(a, v) for (a, v), e in zip(p, ents) if e["score"] >= th] for p, ents in zip(bert_preds, raw)]
    print(f"score >= {th}:", dict(zip(["P", "R", "F1"], evaluate(kept, "BERT")[0].iloc[-1])))
results["BERT"][0]
```

```python
# 方法 3: LLM に JSON で答えさせる（sarashina2.2-3b-instruct、MIT ライセンス）
import gc, json
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM

del trainer, model, ner
gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
llm_tok = AutoTokenizer.from_pretrained(name, padding_side="left")
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
llm_tok.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()

INSTRUCTION = ("次の商品名から、ブランド (brand)・色 (color)・サイズ (size)・素材 (material)・容量 (capacity) を取り出し、"
               "JSON だけを出力してください。説明は書かないでください。値は商品名に書かれているとおりの文字列のリストにし、"
               "書かれていない属性は空のリストにしてください。size は衣類・靴・寝具などのサイズ表記 (M、XL、26.5cm、シングルなど) で、"
               "幅や丈などの寸法は含めません。capacity は容量・内容量・記憶容量 (500ml、2L、90g、64GB、10000mAh など) です。")
def as_json(text, spans):
    return json.dumps({a: [text[s:e] for s, e, l in spans if l == a] for a in ATTRS}, ensure_ascii=False)

shots = [train[i] for i in (0, 1, 2)]                         # few-shot の例は学習データ（テンプレート）から取る
def prompt(text):
    messages = []
    for t, s in shots:
        messages += [{"role": "user", "content": f"{INSTRUCTION}\n商品名: {t}"}, {"role": "assistant", "content": as_json(t, s)}]
    messages.append({"role": "user", "content": f"{INSTRUCTION}\n商品名: {text}"})
    return llm_tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

def extract_json(text):
    start = text.find("{")
    try:
        return json.JSONDecoder().raw_decode(text[start:])[0] if start >= 0 else {}
    except json.JSONDecodeError:
        return {}

outputs = []
for i in range(0, len(test), 16):                             # T4 で 1〜2 分（モデルのダウンロードは別）。VRAM は 10GB 程度
    enc = llm_tok([prompt(t) for t, _ in test[i:i + 16]], return_tensors="pt", padding=True, add_special_tokens=False).to(llm.device)
    with torch.no_grad():
        out = llm.generate(**enc, max_new_tokens=120, do_sample=False, pad_token_id=llm_tok.pad_token_id)
    outputs += llm_tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"VRAM の最大使用量: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

llm_preds = []
for out in outputs:
    obj = extract_json(out)
    llm_preds.append([(a, v) for a in ATTRS for v in (obj.get(a) or []) if isinstance(v, (str, int, float))])
print("JSON として読めなかった出力:", sum(extract_json(o) == {} for o in outputs))
print("商品名に書かれていない値:", [(a, v) for (t, _), p in zip(test, llm_preds) for a, v in p if str(v) not in t])
for (t, _), o in list(zip(test, outputs))[24:29]:
    print(t, "→", o.strip())
results["LLM"] = evaluate(llm_preds, "LLM")
print(f"micro F1 95% CI: {results['LLM'][1][0]:.2f}〜{results['LLM'][1][1]:.2f}")
results["LLM"][0]
```

```python
# 3 つの方法を並べる。正規化しない（取り出した文字列のまま商品マスタの値と比べる）場合の F1 も見る
summary = pd.concat([df for df, _ in results.values()], axis=1)
n_gold = pd.Series([a for _, g in test for *_, a, _ in g]).value_counts()
summary[("", "正解数")] = [n_gold.get(a, n_gold.sum()) for a in summary.index]
for name, preds in [("辞書+正規表現", rule_preds), ("BERT", bert_preds), ("LLM", llm_preds)]:
    print(f"{name:8} micro F1  正規化なし {evaluate(preds, name, False)[0].iloc[-1, 2]:.2f}"
          f" → あり {evaluate(preds, name)[0].iloc[-1, 2]:.2f}")
summary
```

```python
# 方法ごとの誤りを見る: 辞書にないブランド、紛らわしい語（綿棒、L字、幅120cm）など
def diff(i, preds):
    g = {(a, c) for *_, a, c in test[i][1]}
    p = {(a, normalize(a, v)) for a, v in preds[i]}
    return sorted(p - g), sorted(g - p)

for i in [7, 18, 24, 26, 34, 60]:
    print(test[i][0])
    for name, preds in [("辞書", rule_preds), ("BERT", bert_preds), ("LLM", llm_preds)]:
        fp, fn = diff(i, preds)
        print(f"  {name:4} 誤検出 {fp}  見落とし {fn}")
```

- 評価に使った 61 件の商品名と正解は、このレシピのために作ったものである。実在のブランド名や商品名・型番も含むが、属性の組み合わせは実際の商品の仕様とは限らない。正解は属性ごとに 21〜52 件しかなく、micro F1 は辞書と正規表現が 0.89（95% 信頼区間 0.84〜0.92）、LLM が 0.89 (0.84〜0.93) で、この件数では差は見られない（区間がほぼ重なる）。BERT の 0.73 (0.67〜0.79) とは区間が重ならない。実際の商品名に数百件以上の正解を付けて比べること。
- 辞書（正規化の表）はこの評価データを見ながら作り、BERT の学習データのテンプレート（と、そこから取った LLM の few-shot の例）にも同じ表を使ったので、辞書と BERT に有利な比較である。それでも辞書は表にないブランド（45 件中 14 件）を取れず、ブランドの再現率は 0.69 だった。実データでは、辞書にない値の割合がそのまま再現率の上限になる。
- 正解 166 個のうち 76 個は、商品名の表記と商品マスタの値が違う（「ブラック」と「黒」、「1.0L」と「1000ml」など）。このため、正規化しないで取り出した文字列のまま比べると、micro F1 はどの方法も 0.42〜0.49 にとどまる。逆に、表に別表記がないと正しく取り出しても一致しない（BERT と LLM は「Dyson」を取り出したが、表に「ダイソン」の別表記がないので不一致になった）。
- 辞書と正規表現の誤検出は、部分一致によるもの（「綿棒」の「綿」、「L字デスク」の「L」、「幅120cm」、サイズの「3L」を容量 3000ml と読む）である。前後の文字を見る規則を足して防ぐことになり、規則は増え続ける。
- BERT はテンプレートから自動で作った 3000 件だけで学習しており、人手のアノテーションはない。ランダムな名前をブランドとして混ぜたので、辞書にないブランドもかなり取れる（ブランドの再現率 0.91）。一方で適合率は 0.62 と低い。テンプレートでは先頭がほぼブランドなので、先頭の見慣れない語（「綿棒」の「綿」、「麻婆豆腐」、「大きいサイズ」）をブランドにし、「A4対応」を「A」と「4」に分けるような断片も出す。確信度 0.9 未満を捨てると F1 は 0.82 になったが、この閾値は評価データで選んだもので楽観的である。実データに少しでも正解を付けて学習に加えるのが効く。
- LLM (sarashina2.2-3b-instruct) は学習なしで、属性の定義を書いた指示と 3 つの例だけで辞書と同程度になった。弱いのはサイズの範囲（寸法や容量との区別）で、「幅120cm」「90g」「レディース」「A4対応」をサイズとして返した（サイズの適合率 0.65）。指示しても説明文を付け足したり、値の中に「〜は商品名に記載がないため不明」のような文を書いたりするので、JSON の取り出しと検証（大規模言語モデルの章の「LLM の出力から JSON を取り出して検証する」を参照）や、値が商品名に含まれるかの確認が要る。
- 実務では、辞書と正規表現で確実なものを取り、BERT や LLM で辞書にない値を補い、最後にどの方法の値も同じ正規化を通して商品マスタと照合する、という組み合わせが多い。学習なしで補う方法には、この章の「学習なしで好きなラベルの固有表現を抽出する (GLiNER)」もある。新しく見つかった値は、人が確かめて辞書に足していく。
- sarashina2.2 の動かし方とトークナイザーの修正については、大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」を参照。
