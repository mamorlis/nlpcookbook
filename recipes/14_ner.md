# 固有表現抽出

人名・組織名・地名などの固有表現を見つける。辞書やパターンによる抽出、個人情報のマスクも扱う。

## 日本語の固有表現 (人名・地名・組織名など) を抽出する {ner_ginza}

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

- 精度が必要なら Transformer ベースの `en_core_web_trf` を使う (遅いが高精度)。

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
- このモデルのラベルは 8 種類: 人名、法人名、政治的組織名、その他の組織名、地名、施設名、製品名、イベント名 (ストックマーク社の Wikipedia NER データセット)。
- 多言語モデルなら `tsmatz/xlm-roberta-ner-japanese` などがあり、こちらは `start`/`end` が得られる。

## 学習なしで好きなラベルの固有表現を抽出する (GLiNER) {gliner}

「病名」「薬品名」「役職」のような独自のラベルで固有表現抽出をしたいが、学習データがない。

```python
!pip install -q gliner fugashi unidic-lite
```

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

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
- 内部で空白区切りの単語を単位にしているため、日本語や中国語は分かち書きが必要 (そのまま渡すと文全体が 1 語扱いになる)。
- ラベル名の言い回しで結果が変わる。`threshold` を下げると再現率が上がる。
- LLM に抽出させる方法と比べて、高速で、原文中の位置 (`start`/`end`) が確実に得られるのが利点。

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
# 日本語: 「〜ことができる」(可能表現) と「サ変名詞 + する」
nlp_ja = spacy.load("ja_ginza")
m = Matcher(nlp_ja.vocab)
m.add("KOTO_GA_DEKIRU", [[{"POS": "VERB"}, {"ORTH": "こと"}, {"ORTH": {"IN": ["が", "も", "は"]}}, {"LEMMA": "できる"}]])
m.add("SAHEN", [[{"POS": {"IN": ["NOUN", "VERB"]}, "TAG": {"REGEX": "サ変可能"}}, {"LEMMA": "する"}]])

doc = nlp_ja("このモデルは長い文章を要約することができる。日本語を翻訳することもできるし、質問に回答する。")
for match_id, start, end in m(doc):
    print(nlp_ja.vocab.strings[match_id], "|", doc[start:end].text)
```

- 使える属性は `ORTH` `LOWER` `LEMMA` `POS` `TAG` `DEP` `ENT_TYPE` `IS_DIGIT` など。`OP` は `?` `*` `+` `!`。
- 単語列の辞書 (数万語) を高速に引くなら `PhraseMatcher`、係り受けパターンなら `DependencyMatcher`。

## 個人情報をマスクする (仮名化) {pii_mask}

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
    "PHONE": r"0\d{1,4}-\d{1,4}-\d{4}",
    "POSTAL": r"〒?\d{3}-\d{4}",
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

text = "山田太郎(yamada@example.com、090-1234-5678)は横浜市在住で、ソニーに勤務している。佐藤花子は東京大学の学生です。"
print(mask(text))
```

- NER の再現率は 100% ではない。公開前には人手の確認が必要。再現率を優先するなら複数のモデルの和集合をとる。
- 単にマスクすると文脈が失われる。「山田太郎 → 人物A」のように同じ実体に同じ仮名を割り当てると、後段の分析に使いやすい。
- 英語を含む多言語なら Microsoft Presidio (`presidio-analyzer`) が定番。
