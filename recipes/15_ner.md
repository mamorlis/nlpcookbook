# 固有表現抽出

人名・組織名・地名などの固有表現を見つける。学習済みモデルの利用、辞書やパターンによる抽出、個人情報のマスクや音声認識の結果への応用、BIO タグの扱いと評価、CRF や BERT による学習。

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
- 内部で空白区切りの単語を単位にしているため、日本語や中国語は分かち書きが必要 (そのまま渡すと文全体が 1 語扱いになる)。
- ラベル名の言い回しで結果が変わる。`threshold` を下げると再現率が上がる。
- LLM に抽出させる方法と比べて、高速で、原文中の位置 (`start`/`end`) が確実に得られるのが利点。
- GLiNER はトークナイザーを内部で読み込み、元のモデル (`microsoft/mdeberta-v3-base`) には `tokenizer.json` もないので、`tokenizer.json` から読み直す方法 (Transformers の基本操作の章の「2 つの文の含意関係を判定する (NLI)」の注意書きを参照) が使えない。そこで、読み込んだあとのトークナイザーの正規化の部分だけを差し替えている。上流 (transformers) で直れば不要になる。

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

- 音声認識の結果には、句読点がない (またはモデルが推定したもの)、同音異義語の誤り (人名の漢字が違う)、数字や英字の表記のゆれ、といった書き言葉との違いがある。書き言葉で学習した固有表現抽出器は、その分だけ精度が落ちる。
- 固有名詞は音声認識が最も間違えやすい部分。特に人名は、音が合っていても漢字が違うことが多い。読みで照合する、候補の辞書と突き合わせる、といった後処理が有効。
- 読みで照合しても、この例では結果が変わらなかった。「ホンダ」が「本田」になると、表記だけでなく種類の判定まで (会社 → 人名) 変わってしまうため。音声認識の誤りは、固有表現の範囲と種類の両方に波及する。
- 全体の文字誤り率が低くても、固有表現に限ると誤りが多いことがある。目的が固有表現なら、固有表現だけの誤り率を測る。
- ここで使ったのは合成音声で、実際の会話音声 (雑音、言いよどみ、複数話者) よりはるかに認識しやすい。実データでの性能は、必ず実データで測る。
- 音声認識と固有表現抽出を別々に行う方法 (ここでの方法) のほかに、音声から直接固有表現を出すモデルや、認識結果の誤りを LLM で訂正してから抽出する方法がある。

## 文字位置のアノテーションを BIO タグ列に変換する {spans_to_bio}

「何文字目から何文字目が人名」という形式 (doccano や brat の出力) のアノテーションを、トークンごとの BIO タグに変換したい。

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

- 不正な系列の扱い方 (捨てる / 直す) で評価値が変わる。seqeval のデフォルトは、ここと同じく「I- で始まっても新しいスパンとみなす」寛容な方式。
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

## BERT をファインチューニングして日本語の固有表現抽出器を作る {bert_ner_finetune gpu norun}

日本語の固有表現抽出モデルを、文字位置のアノテーション付きデータから学習したい。

```python
!pip install -q datasets seqeval accelerate
```

```python
import json, urllib.request
from datasets import Dataset

# ストックマーク社の Wikipedia 固有表現データセット (CC BY-SA 3.0)
url = "https://raw.githubusercontent.com/stockmarkteam/ner-wikipedia-dataset/main/ner.json"
data = json.load(urllib.request.urlopen(url))
print(len(data), data[0])

types = sorted({e["type"] for d in data for e in d["entities"]})
labels = ["O"] + [f"{p}-{t}" for t in types for p in ("B", "I")]
label2id = {l: i for i, l in enumerate(labels)}
ds = Dataset.from_list([{"text": d["text"], "starts": [e["span"][0] for e in d["entities"]],
                         "ends": [e["span"][1] for e in d["entities"]], "types": [e["type"] for e in d["entities"]]} for d in data])
ds = ds.train_test_split(test_size=0.2, seed=0)
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
    per_device_train_batch_size=32, per_device_eval_batch_size=64, eval_strategy="epoch", save_strategy="no",
    fp16=True, logging_steps=50, report_to="none", seed=0,
)
trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["test"],
                  data_collator=DataCollatorForTokenClassification(tokenizer), compute_metrics=compute_metrics)
trainer.train()
out = trainer.predict(tokenized["test"])
print(classification_report(*to_tags(out.predictions, out.label_ids), digits=3))
```

```python
from transformers import pipeline

ner = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=0)
text = "ソニーグループの吉田憲一郎会長は、東京都港区の本社で新型プレイステーションを発表した。"
for e in ner(text):
    print(text[e["start"]:e["end"]], e["entity_group"], round(float(e["score"]), 2))
```

- T4 GPU で 5 エポック約 5 分、エンティティ単位の F1 は 0.84 前後。
- ここではサブワードすべてにラベルを付けている (先頭だけに付けて残りを -100 にする流儀もある)。評価は本来、単語や文字の単位に戻して行うのが厳密。
- XLM-R のトークナイザは、文頭や空白の後に単独の `▁` トークンを置き、その文字位置を次の文字と同じ `(0, 1)` のように返す。そのまま位置で判定すると `▁` と次のトークンの両方が `B-` になり、推論時に「ソ」と「ソニーグループ」のように同じ固有表現が二重に出る。
- 日本語特化の BERT (`tohoku-nlp/bert-base-japanese-v3`) のほうが精度は高いが、トークナイザが文字位置を返さないので、位置合わせを自分で実装する必要がある。
