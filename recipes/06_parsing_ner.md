# 構文解析と固有表現抽出

係り受け解析で文の構造を取り出し、人名・組織名などの固有表現を見つける。

## GiNZA で日本語の依存構造解析 (係り受け解析) をする {ginza_dependency}

日本語の文について、どの単語がどの単語に係っているか (主語・目的語などの関係) を解析したい。

```python
!pip install -q ginza ja_ginza
```

```python
import spacy

nlp = spacy.load("ja_ginza")
doc = nlp("太郎は図書館で花子に借りた本を返した。")
for t in doc:
    print(f"{t.i:2} {t.text:5} {t.lemma_:5} {t.pos_:6} {t.dep_:8} → {t.head.i:2} {t.head.text}")
```

```python
root = [t for t in doc if t.dep_ == "ROOT"][0]
print("述語 (ROOT):", root.text)
print("直接の子:", [(c.text, c.dep_) for c in root.children])
print("「本」の部分木:", "".join(t.text for t in doc[6].subtree))
print(spacy.explain("nsubj"), "/", spacy.explain("obl"), "/", spacy.explain("acl"))
```

- ラベルは Universal Dependencies (UD)。日本語でも英語でも同じ体系で扱える。
- より高精度な Transformer 版は `pip install ja_ginza_electra` して `spacy.load("ja_ginza_electra")`。
- spaCy 公式の `ja_core_news_sm/md/lg/trf` も同じ使い方ができる。

## 依存構造木を図で表示する (displaCy) {displacy_dep}

係り受けの解析結果を、矢印付きの図で確認したい。

```python
!pip install -q ginza ja_ginza
!python -m spacy download en_core_web_sm -q
```

```python
import spacy
from spacy import displacy

nlp = spacy.load("ja_ginza")
doc = nlp("太郎は花子に美しい花束を渡した。")
displacy.render(doc, style="dep", jupyter=True, options={"compact": True, "distance": 90})
```

```python
nlp_en = spacy.load("en_core_web_sm")
doc_en = nlp_en("The quick brown fox jumps over the lazy dog.")
displacy.render(doc_en, style="dep", jupyter=True, options={"distance": 100, "collapse_punct": True})
```

```python
# SVG ファイルに保存 (論文やスライド用)
svg = displacy.render(doc, style="dep", jupyter=False, options={"compact": True, "distance": 90})
with open("dep.svg", "w", encoding="utf-8") as f:
    f.write(svg)
```

- Colab 以外 (スクリプト) では `displacy.serve(doc)` でブラウザ表示できる。

## 文節に区切って文節単位の係り受けを得る {bunsetsu}

日本語の伝統的な単位である「文節」で区切り、「どの文節がどの文節に係るか」を知りたい (CaboCha 相当)。

```python
!pip install -q ginza ja_ginza
```

```python
import ginza
import spacy

nlp = spacy.load("ja_ginza")
doc = nlp("太郎は図書館で花子に借りた本を昨日返した。")

for sent in doc.sents:
    for span in ginza.bunsetu_spans(sent):
        head = ginza.bunsetu_span(span.root.head)
        target = "(文末)" if head.start == span.start else head.text
        print(f"{span.text:　<6} → {target:　<6} 主辞: {ginza.phrase(span.root)}")
```

```python
print("文節:", [s.text for s in ginza.bunsetu_spans(doc)])
print("主辞句:", [(s.text, s.label_) for s in ginza.bunsetu_phrase_spans(doc)])
```

- 文節 = 自立語 (+ 付属語)。UD の単語単位の係り受けより粗いが、日本語の直感に合いやすく、係り受けの交差も少ない。

## 「誰が・何を・どうした」の三つ組を抜き出す {svo_extract}

文から 主語-目的語-述語 の組を取り出して、簡単な情報抽出や知識グラフの素材にしたい。

```python
!pip install -q ginza ja_ginza
```

```python
import ginza
import spacy

nlp = spacy.load("ja_ginza")

def phrase(token):
    """トークンを含む文節から、後ろに付く助詞などを除いた部分を返す"""
    return "".join(t.text for t in ginza.bunsetu_span(token) if t.i <= token.i)

def triples(text):
    out = []
    for sent in nlp(text).sents:
        for verb in sent:
            if verb.pos_ not in ("VERB", "ADJ"):
                continue
            args = {c.dep_: phrase(c) for c in verb.children if c.dep_ in ("nsubj", "obj", "iobj", "obl")}
            if "nsubj" in args:
                out.append((args.get("nsubj"), args.get("obj") or args.get("obl"), verb.lemma_))
    return out

for s in ["OpenAI が新しい言語モデルを発表した。", "山田教授は学生に自然言語処理を教えている。",
          "猫がネズミを追いかけ、犬が猫を追いかけた。", "政府は来年度の予算案を閣議で決定した。"]:
    print(s, "→", triples(s))
```

- 日本語は主語の省略が多く、「は」で示される主題が必ずしも主語ではない。本格的には述語項構造解析やゼロ照応解析が必要。
- 受身・使役では格と意味役割がずれる (「ネズミが猫に追いかけられた」)。

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

## 名詞句を抽出する {noun_chunks}

文から名詞句 (修飾語を含む名詞のまとまり) を取り出して、キーフレーズ候補や検索語にしたい。

```python
!pip install -q ginza ja_ginza
!python -m spacy download en_core_web_sm -q
```

```python
import spacy

nlp_en = spacy.load("en_core_web_sm")
doc = nlp_en("The new pretrained language model achieved impressive results on several machine translation benchmarks.")
for chunk in doc.noun_chunks:
    print(f"{chunk.text:42} 主辞: {chunk.root.text:12} 関係: {chunk.root.dep_:6} → {chunk.root.head.text}")
```

```python
nlp_ja = spacy.load("ja_ginza")
doc = nlp_ja("新しい事前学習済み言語モデルが、複数の機械翻訳ベンチマークで最高性能を達成した。")
print([c.text for c in doc.noun_chunks])
```

- `noun_chunks` は入れ子にならない「基本名詞句」を返す。前置詞句や関係節まで含めた大きい句は `token.subtree` で取る。

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

## Stanza で多言語の形態素・依存構造解析をする {stanza}

日本語・英語以外も含めて、70 以上の言語を同じ API で解析したい (Universal Dependencies 準拠)。

```python
!pip install -q stanza
```

```python
import stanza

stanza.download("ja", verbose=False)
nlp = stanza.Pipeline("ja", processors="tokenize,pos,lemma,depparse", verbose=False)
doc = nlp("太郎は花子に本を渡した。")
for sent in doc.sentences:
    for w in sent.words:
        head = sent.words[w.head - 1].text if w.head > 0 else "ROOT"
        print(f"{w.id:2} {w.text:4} {w.lemma:4} {w.upos:6} {w.deprel:8} → {head}")
```

```python
# 言語を変えても同じコードで動く
for lang, text in [("de", "Der Hund jagt die Katze."), ("ko", "나는 학교에 갑니다.")]:
    stanza.download(lang, verbose=False)
    p = stanza.Pipeline(lang, processors="tokenize,pos,lemma,depparse", verbose=False)
    print(lang, [(w.text, w.upos, w.deprel) for s in p(text).sentences for w in s.words])
```

- `w.head` は 1 始まりの単語番号で、0 が ROOT。
- GPU があれば自動で使う。CPU に固定するなら `stanza.Pipeline(..., use_gpu=False)`。

## 文脈自由文法を書いて構文解析する (NLTK) {cfg_parse}

自分で書いた小さな文法で文を解析し、構文木と構造的曖昧性を確かめたい (授業の演習向け)。

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

## 解析結果を CoNLL-U 形式で保存・読み込みする {conllu}

依存構造解析の結果を、標準的な CoNLL-U 形式でファイルに保存したり、UD のツリーバンクを読み込んだりしたい。

```python
!pip install -q ginza ja_ginza conllu
```

```python
import spacy

nlp = spacy.load("ja_ginza")

def to_conllu(doc):
    lines = []
    for sid, sent in enumerate(doc.sents, 1):
        lines += [f"# sent_id = {sid}", f"# text = {sent.text}"]
        for i, t in enumerate(sent, 1):
            head = 0 if t.dep_ == "ROOT" else t.head.i - sent.start + 1
            feats = str(t.morph) or "_"
            misc = "_" if t.whitespace_ else "SpaceAfter=No"
            lines.append("\t".join([str(i), t.text, t.lemma_, t.pos_, t.tag_, feats, str(head), t.dep_.lower() if t.dep_ == "ROOT" else t.dep_, "_", misc]))
        lines.append("")
    return "\n".join(lines) + "\n"

text = to_conllu(nlp("太郎は本を読んだ。花子は寝ていた。"))
with open("parsed.conllu", "w", encoding="utf-8") as f:
    f.write(text)
print(text)
```

```python
from conllu import parse

with open("parsed.conllu", encoding="utf-8") as f:
    sentences = parse(f.read())
for sent in sentences:
    print(sent.metadata["text"], [(t["form"], t["upos"], t["head"], t["deprel"]) for t in sent])
print(sentences[0].to_tree().token["form"], "が ROOT")
```

- 10 列: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC。列はタブ区切り、文は空行区切り。
- GiNZA はコマンドラインでも `ginza -f conllu < input.txt` で CoNLL-U を出力できる。

## spaCy で大量の文書を高速に処理する {spacy_pipe}

数万件のテキストを spaCy で解析したいが、1 件ずつ `nlp(text)` すると遅い。

```python
!python -m spacy download en_core_web_sm -q
```

```python
import time
import spacy

texts = ["Apple is looking at buying a U.K. startup for $1 billion.", "The quick brown fox jumps over the lazy dog."] * 1000

def timeit(label, fn):
    t = time.perf_counter()
    result = fn()
    print(f"{label:38} {time.perf_counter() - t:5.1f} 秒")
    return result

nlp = spacy.load("en_core_web_sm")
_ = timeit("1 件ずつ nlp(text)", lambda: [nlp(t) for t in texts])
_ = timeit("nlp.pipe (バッチ処理)", lambda: list(nlp.pipe(texts, batch_size=256)))

nlp_light = spacy.load("en_core_web_sm", exclude=["parser", "ner", "lemmatizer"])  # 品詞だけ欲しい場合
_ = timeit("不要な要素を除外 + pipe", lambda: list(nlp_light.pipe(texts, batch_size=256)))

nlp_tok = spacy.blank("en")  # トークナイズだけ
_ = timeit("トークナイザのみ", lambda: list(nlp_tok.pipe(texts)))
```

```python
# メタデータを一緒に流す (as_tuples) と、ID との対応を保ったまま処理できる
data = [("Tokyo is the capital of Japan.", {"id": 1}), ("Paris is in France.", {"id": 2})]
for doc, ctx in nlp.pipe(data, as_tuples=True):
    print(ctx["id"], [(e.text, e.label_) for e in doc.ents])
```

- `nlp.pipe(texts, n_process=2)` でマルチプロセス化もできる (Transformer モデル + GPU の場合は使わない)。
- 文分割だけ必要なら `nlp.add_pipe("sentencizer")` を入れた `spacy.blank` が最速。

## 依存距離と構文木の深さで文の複雑さを測る {dep_distance}

文の構造的な複雑さを数値化したい (可読性の評価、学習者作文の分析、言語間の比較)。

```python
!pip install -q ginza ja_ginza
```

```python
import spacy

nlp = spacy.load("ja_ginza")

def depth(token):
    return 0 if token.head.i == token.i else 1 + depth(token.head)

def complexity(text):
    rows = []
    for sent in nlp(text).sents:
        toks = [t for t in sent if t.pos_ != "PUNCT"]
        dists = [abs(t.i - t.head.i) for t in toks if t.head.i != t.i]
        rows.append({"文": sent.text[:22], "語数": len(toks),
                     "平均依存距離": round(sum(dists) / max(1, len(dists)), 2),
                     "最大依存距離": max(dists, default=0),
                     "木の深さ": max(depth(t) for t in toks)})
    return rows

import pandas as pd

text = ("猫が寝ている。"
        "太郎が昨日図書館で借りた本を花子が読んでいる。"
        "私が先週友人から勧められて書店で買った小説を弟が勝手に持ち出して学校で友達に貸してしまった。")
pd.DataFrame(complexity(text))
```

- 平均依存距離 (MDD) は、係り元と係り先が離れているほど大きくなり、ワーキングメモリの負荷の指標とされる。
- 日本語は主辞後置型なので、長い連体修飾節があると依存距離が大きくなる。
