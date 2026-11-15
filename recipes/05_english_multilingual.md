# 英語・多言語の前処理

英語を中心に、空白で区切られる言語や中国語などの基本的な前処理。

## 英文を単語と文に分割する (NLTK) {nltk_tokenize}

英語のテキストを文に区切り、さらに単語 (トークン) に区切りたい。`split()` だと句読点がくっついてしまう。

```python
import nltk
nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize

text = "Dr. Smith didn't go to Washington D.C. on Jan. 5th. He stayed home, i.e., he was sick! Isn't that sad?"
for s in sent_tokenize(text):
    print(s)
print(text.split()[:8])
print(word_tokenize(text)[:10])
```

```python
# SNS テキスト向け: 顔文字・ハッシュタグ・メンションを壊さない
from nltk.tokenize import TweetTokenizer

tt = TweetTokenizer(strip_handles=True, reduce_len=True)
print(tt.tokenize("@nlp_bot This is sooooooo coooool!!! :-) #NLProc https://example.com"))
```

- `word_tokenize` は Penn Treebank 方式で `didn't` を `did` + `n't` に分ける。
- 新しい NLTK では `punkt` ではなく `punkt_tab` をダウンロードする。

## 略語に強いルールベースの文分割 (pySBD、多言語) {pysbd}

略語や小数点、引用符が多い文章を正確に文分割したい。日本語や中国語など 20 以上の言語で使いたい。

```python
!pip install -q pysbd
```

```python
import pysbd

seg = pysbd.Segmenter(language="en", clean=False)
text = "My name is Jonas E. Smith. Please turn to p. 55. The U.S. Drug Enforcement Administration (DEA) says hello. I paid $3.50 for it at 5 p.m. yesterday."
for s in seg.segment(text):
    print("-", s)
```

```python
seg_ja = pysbd.Segmenter(language="ja", clean=False)
print(seg_ja.segment("これはペンです。それは「マーカーです。」と彼は言った。値段は1.5倍！本当？"))

# 文字オフセット付き
for span in pysbd.Segmenter(language="en", clean=False, char_span=True).segment("Hello World. My name is Jonas."):
    print(span.start, span.end, repr(span.sent))
```

## Moses 方式でトークナイズ・デトークナイズする {sacremoses}

機械翻訳の前処理で定番の Moses トークナイザを使いたい。トークン列を自然な文に戻したい。

```python
!pip install -q sacremoses
```

```python
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer

mt, md = MosesTokenizer(lang="en"), MosesDetokenizer(lang="en")
text = "Hello, world! It's a \"test\" (isn't it?) — costing $3.50, i.e. cheap."
tokens = mt.tokenize(text, escape=False)
print(tokens)
print(mt.tokenize(text, return_str=True))  # escape=True (デフォルト) だと &quot; などにエスケープされる
print(md.detokenize(tokens))
```

```python
# 句読点や引用符の表記ゆれを正規化
mpn = MosesPunctNormalizer(lang="en")
print(mpn.normalize("«Bonjour» , he said…  “Curly quotes”  and ‘single’ ."))
```

- モデルの出力 (トークン列) を評価する前に、必ずデトークナイズしてから sacreBLEU に渡す。

## 大文字・小文字を正しく無視して比較する (casefold) {casefold}

大文字と小文字を区別せずに文字列を比較・検索したい。`lower()` では足りないケースがある。

```python
pairs = [("Straße", "STRASSE"), ("İstanbul", "istanbul"), ("ΣΊΣΥΦΟΣ", "σίσυφος"), ("ＮＬＰ", "nlp"), ("Apple", "APPLE")]
for a, b in pairs:
    print(f"{a:10} {b:10} lower: {a.lower() == b.lower()!s:6} casefold: {a.casefold() == b.casefold()}")
```

```python
import unicodedata

def canonical(s):
    # 正規化 → casefold → 再度正規化 (casefold で互換文字が出ることがあるため)
    return unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", s).casefold())

for a, b in pairs:
    print(a, b, canonical(a) == canonical(b))
```

- `casefold()` は比較専用の強い小文字化 (ß → ss)。表示用には使わない。
- トルコ語の `İ/ı` のように言語依存の規則は Python 標準では扱えない (PyICU などが必要)。

## 英語の短縮形を展開する (I'm → I am) {contractions}

`don't` `I'll` `gonna` などの短縮形を元の形に戻して、表記を統一したい。

```python
!pip install -q contractions
```

```python
import contractions

for s in ["I'm sure you'll love it, but they don't.", "She's been there. He's happy.", "y'all gonna wanna see this", "It's John's book."]:
    print(contractions.fix(s))
```

- `'s` は is / has / 所有格の曖昧性があり、機械的な展開では区別できない。

## アクセント記号を外す・非 ASCII 文字を ASCII にする {unidecode}

`café` と `cafe` を同一視したい。人名などを ASCII のみで表記したい。

```python
!pip install -q unidecode
```

```python
import unicodedata
from unidecode import unidecode

def strip_accents(s):
    # NFD で分解して結合文字 (Mn) を捨てる。アクセント以外は変えない
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

for s in ["Crème brûlée à la café", "Dvořák, Gödel, Erdős, Søren, Łódź", "Москва", "北京", "東京", "ガ"]:
    print(f"{s:36} {strip_accents(s):36} {unidecode(s)}")
```

- `strip_accents` は濁点も結合文字として落とす (ガ→カ) ので、日本語が混ざるテキストでは対象をラテン文字に限定する。
- `unidecode` は漢字を中国語読みにするため、日本語には使えない (東京 → Dong Jing)。

## 語幹を取り出す (ステミング) {stemming}

`running` `runs` `ran` のような語形変化を雑にまとめて、検索や頻度集計で同じ語として扱いたい。

```python
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer

words = ["running", "runs", "ran", "easily", "studies", "studying", "university", "universe", "organization", "better"]
porter, snowball, lancaster = PorterStemmer(), SnowballStemmer("english"), LancasterStemmer()
print(f"{'word':14}{'Porter':12}{'Snowball':12}{'Lancaster'}")
for w in words:
    print(f"{w:14}{porter.stem(w):12}{snowball.stem(w):12}{lancaster.stem(w)}")
```

```python
# 英語以外
print(SnowballStemmer.languages)
print(SnowballStemmer("german").stem("Bücherregale"), SnowballStemmer("spanish").stem("corriendo"))
```

- ステミングは規則で語尾を削るだけなので、`ran` はそのまま、`university` と `universe` が同じになるなど粗い。正確さが要るなら次のレンマ化を使う。

## 単語を辞書の見出し語に戻す (レンマ化) {lemmatize_spacy}

`was` → `be`、`mice` → `mouse` のように、品詞を考慮して正しい原形に戻したい。

```python
!python -m spacy download en_core_web_sm -q
```

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The striped bats were hanging on their feet and ate best fishes. I saw her duck.")
for token in doc:
    print(f"{token.text:10}{token.lemma_:10}{token.pos_}")
```

```python
# NLTK の WordNet レンマタイザ (品詞を渡さないと名詞扱いになる)
import nltk
nltk.download("wordnet", quiet=True)
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()
print(wnl.lemmatize("better"), wnl.lemmatize("better", pos="a"), wnl.lemmatize("were", pos="v"), wnl.lemmatize("feet"))
```

- 大量に処理するときは `nlp.pipe(texts, batch_size=256)` を使い、不要な要素は `spacy.load(..., disable=["parser", "ner"])` で止めると速い。

## 英語のストップワードを除去する {stopwords_en}

`the` `is` `of` のような機能語を取り除いて、内容語だけにしたい。

```python
import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop = set(stopwords.words("english"))
print(len(stop), sorted(stop)[:15])

text = "This is not a good example of what we should be doing in the lab."
tokens = [w for w in word_tokenize(text.lower()) if w.isalpha() and w not in stop]
print(tokens)
```

```python
# ライブラリによってリストが違う
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from spacy.lang.en.stop_words import STOP_WORDS

print(len(stop), len(ENGLISH_STOP_WORDS), len(STOP_WORDS))
print(stopwords.fileids()[:10], "...")  # NLTK は多言語対応
```

- `not` や `no` もストップワードに入っている。感情分析では否定が消えて意味が逆になるので、`stop - {"not", "no"}` のように残す。
- BERT などのニューラルモデルにはストップワード除去は不要 (むしろ有害)。

## 英語テキストを定番の手順でクリーニングする {clean_text_en}

古典的な機械学習 (BoW / TF-IDF) に入れる前の、小文字化・記号除去・空白整理をひとまとめにしたい。

```python
import html
import re
import string
import unicodedata

def clean(text, lower=True, remove_punct=True, remove_digits=False):
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", " ", text)                 # HTML タグ
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URL
    text = re.sub(r"\S+@\S+", " ", text)                 # メールアドレス
    if lower:
        text = text.lower()
    if remove_digits:
        text = re.sub(r"\d+", " ", text)
    if remove_punct:
        text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    return re.sub(r"\s+", " ", text).strip()

raw = "<p>Check out https://example.com !!!  It&#39;s   GREAT — e-mail me at foo@bar.com (ASAP), 100%.</p>"
print(clean(raw))
print(clean(raw, lower=False, remove_punct=False))
```

- 記号を「削除」ではなく「空白に置換」するのがコツ (`e-mail` が `email` ではなく `e mail` になるが、`word,word` がくっつく事故を防げる)。
- 事前学習済みモデルを使う場合は、こうした前処理はせずに生のテキストを渡す。

## 英語の品詞タグ付けをする {pos_tag_en}

英文の各単語に品詞 (名詞・動詞・形容詞…) を付けたい。

```python
import nltk
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("universal_tagset", quiet=True)

tokens = nltk.word_tokenize("Time flies like an arrow; fruit flies like a banana.")
print(nltk.pos_tag(tokens))                       # Penn Treebank タグ
print(nltk.pos_tag(tokens, tagset="universal"))   # 粗い共通タグ
```

```python
!python -m spacy download en_core_web_sm -q
```

```python
import spacy

nlp = spacy.load("en_core_web_sm")
for t in nlp("Time flies like an arrow; fruit flies like a banana."):
    print(f"{t.text:8}{t.pos_:7}{t.tag_:5}{spacy.explain(t.tag_)}")
```

- `pos_` は Universal Dependencies の粗い品詞、`tag_` は Penn Treebank の細かい品詞。

## WordNet で同義語・上位語・類似度を調べる (日本語も) {wordnet}

単語の同義語や上位概念を辞書から引きたい。日本語 WordNet も使いたい。

```python
import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)  # 多言語 WordNet (NLTK のバージョンによって 1.4 か 2.0 を使う)
nltk.download("omw-2.0", quiet=True)
from nltk.corpus import wordnet as wn

for syn in wn.synsets("bank")[:4]:
    print(syn.name(), syn.lemma_names(), "-", syn.definition())
```

```python
dog, cat, car = wn.synset("dog.n.01"), wn.synset("cat.n.01"), wn.synset("car.n.01")
print("上位語:", [s.name() for s in dog.hypernyms()])
print("下位語:", [s.name() for s in dog.hyponyms()][:5])
print("上位語をたどる:", " > ".join(s.lemma_names()[0] for s in dog.hypernym_paths()[0]))
print("類似度 dog-cat:", round(dog.wup_similarity(cat), 3), " dog-car:", round(dog.wup_similarity(car), 3))
```

```python
# 日本語 WordNet (Open Multilingual Wordnet 経由)
for syn in wn.synsets("犬", lang="jpn"):
    print(syn.name(), syn.lemma_names("jpn"), "|", syn.lemma_names()[:3])
print(sorted({l for s in wn.synsets("美しい", lang="jpn") for l in s.lemma_names("jpn")})[:15])
```

- 同義語によるデータ拡張 (EDA) や、語義曖昧性解消のベースライン (Lesk) に使える。

## 英語のスペルミスを訂正する {spellcheck}

タイプミスを含む英文から誤りを見つけ、修正候補を出したい。

```python
!pip install -q pyspellchecker
```

```python
import re
from spellchecker import SpellChecker

spell = SpellChecker()
text = "I recieved a wonderfull mesage from my freind yestarday about langauge procesing"
words = re.findall(r"[a-z']+", text.lower())
for w in spell.unknown(words):
    print(w, "→", spell.correction(w), sorted(spell.candidates(w) or [])[:5])
```

```python
def correct(text):
    return re.sub(r"[A-Za-z']+", lambda m: spell.correction(m.group()) or m.group() if m.group().lower() in spell.unknown([m.group()]) else m.group(), text)

print(correct(text))
spell.word_frequency.load_words(["nlp", "tokenizer"])  # 専門用語を辞書に追加
print(spell.unknown(["nlp", "tokenizer", "transformr"]))
```

- 編集距離と単語頻度に基づく Norvig 方式。文脈は見ないので `their/there` のような実在語の誤りは直せない。文脈を見るには文法誤り訂正の章を参照。

## 英単語の単数形・複数形・冠詞を変換する (inflect) {inflect}

テンプレートから英文を生成するとき、数に応じて複数形にしたり `a/an` を正しく付けたりしたい。

```python
!pip install -q inflect
```

```python
import inflect

p = inflect.engine()
for w in ["cat", "child", "analysis", "corpus", "mouse", "sheep", "criterion"]:
    print(f"{w:10} → {p.plural(w)}")
print(p.singular_noun("corpora"), p.singular_noun("mice"), p.singular_noun("cat"))  # 単数形を渡すと False
print(p.a("apple"), "/", p.a("university"), "/", p.a("hour"), "/", p.a("NLP model"))
print(p.ordinal(23), "/", p.number_to_words(1234), "/", p.join(["apples", "bananas", "cherries"]))
```

```python
for n in [0, 1, 2]:
    print(f"Found {p.no('error', n)} in {n} {p.plural('file', n)}.")
```

## 数値を単語で読み上げる形にする (num2words) {num2words}

「42」を `forty-two` や「四十二」に、序数や年号の読みに変換したい (音声合成の前処理、テキスト正規化)。

```python
!pip install -q num2words
```

```python
from num2words import num2words

print(num2words(1234567))
print(num2words(42, to="ordinal"), num2words(42, to="ordinal_num"))
print(num2words(1999, to="year"))
print(num2words(12.5))
for lang in ["ja", "fr", "de", "es", "ko"]:
    print(lang, num2words(2026, lang=lang))
```

```python
import re

def spell_out(text, lang="en"):
    return re.sub(r"\d+(?:\.\d+)?", lambda m: num2words(float(m.group()) if "." in m.group() else int(m.group()), lang=lang), text)

print(spell_out("I have 3 cats and 12 dogs, which cost 99.5 dollars."))
```

## 英文の読みやすさ (リーダビリティ) を測る {readability_en}

英文の難易度を、学年レベルなどの指標で数値化したい (教材の選定、平易化の評価)。

```python
!pip install -q textstat
```

```python
import textstat

easy = "The cat sat on the mat. It was a sunny day. The cat was happy."
hard = ("The proliferation of large-scale pretrained language models has precipitated a fundamental "
        "reconsideration of evaluation methodologies within computational linguistics.")

for name, text in [("easy", easy), ("hard", hard)]:
    print(name)
    print("  Flesch Reading Ease  :", round(textstat.flesch_reading_ease(text), 1), "(高いほど易しい)")
    print("  Flesch-Kincaid Grade :", round(textstat.flesch_kincaid_grade(text), 1), "(米国の学年)")
    print("  Gunning Fog          :", round(textstat.gunning_fog(text), 1))
    print("  音節数 / 単語数 / 文数:", textstat.syllable_count(text), textstat.lexicon_count(text), textstat.sentence_count(text))
```

- どの指標も「文の長さ」と「単語の長さ (音節数)」の組み合わせ。短い文章では値が不安定になる。

## テキストが何語で書かれているか判定する (langdetect) {langdetect}

多言語が混ざったデータから日本語の行だけを取り出したい、言語ごとに処理を分けたい。

```python
!pip install -q langdetect
```

```python
from langdetect import DetectorFactory, detect, detect_langs

DetectorFactory.seed = 0  # 結果を再現可能にする (デフォルトは実行ごとに揺れる)

texts = ["これは日本語の文です。", "This is an English sentence.", "这是一个中文句子。", "이것은 한국어 문장입니다.",
         "Ceci est une phrase en français.", "Das ist ein deutscher Satz.", "OK", "東京大学"]
for t in texts:
    print(detect(t), [str(p) for p in detect_langs(t)][:2], t, sep="\t")
```

- 短い文字列 (「OK」「東京大学」) は誤判定しやすい。漢字だけだと日本語か中国語か区別できない。
- 数字や記号しかない文字列は例外 `LangDetectException` になるので、`try` で囲む。

## 高精度な言語判定と、混在テキストの言語区間検出 (lingua) {lingua}

短いテキストでも精度よく言語判定したい。1 つの文に複数言語が混ざっている箇所も検出したい。

```python
!pip install -q lingua-language-detector
```

```python
from lingua import Language, LanguageDetectorBuilder

# 候補の言語を絞ると速く・正確になる
detector = LanguageDetectorBuilder.from_languages(
    Language.JAPANESE, Language.ENGLISH, Language.CHINESE, Language.KOREAN, Language.FRENCH, Language.GERMAN
).build()

for t in ["これは日本語の文です。", "languages are awesome", "这是一个中文句子。", "Das ist gut"]:
    lang = detector.detect_language_of(t)
    conf = detector.compute_language_confidence_values(t)[:2]
    print(lang.name, [(c.language.name, round(c.value, 2)) for c in conf], t)
```

```python
# 混在テキスト
mixed = "Parlez-vous français? Ich spreche Französisch nur ein bisschen. A little bit is better than nothing."
for r in detector.detect_multiple_languages_of(mixed):
    print(r.language.name, "|", mixed[r.start_index:r.end_index])
```

- 全言語を対象にするなら `LanguageDetectorBuilder.from_all_languages()` (メモリを 1GB 程度使う)。

## 中国語を単語分割する (jieba) {jieba}

中国語の文を単語に区切りたい。品詞やキーワードも取りたい。

```python
!pip install -q jieba
```

```python
import jieba
import jieba.posseg as pseg

text = "我来到北京清华大学学习自然语言处理"
print(jieba.lcut(text))                 # 精確モード (デフォルト)
print(jieba.lcut(text, cut_all=True))   # 全モード (可能な語をすべて出す)
print(jieba.lcut_for_search(text))      # 検索エンジン向け (長い語をさらに分割)
print([(w.word, w.flag) for w in pseg.cut(text)])
```

```python
import jieba.analyse

doc = "自然语言处理是人工智能和语言学领域的分支学科。此领域探讨如何处理及运用自然语言，包括自然语言理解和自然语言生成。"
print(jieba.analyse.extract_tags(doc, topK=5, withWeight=True))  # TF-IDF によるキーワード抽出
jieba.add_word("大语言模型")  # 単語を辞書に追加
print(jieba.lcut("大语言模型很强大"))
```

## 中国語の簡体字と繁体字を変換する (OpenCC) {opencc}

簡体字と繁体字を相互に変換したい。台湾・香港の語彙や字体の違いも吸収したい。

```python
!pip install -q opencc-python-reimplemented
```

```python
from opencc import OpenCC

print(OpenCC("s2t").convert("自然语言处理与机器学习"))    # 簡体字 → 繁体字
print(OpenCC("t2s").convert("自然語言處理與機器學習"))    # 繁体字 → 簡体字
print(OpenCC("s2twp").convert("计算机软件和鼠标"))        # 台湾の語彙に (软件→軟體、鼠标→滑鼠)
print(OpenCC("s2hk").convert("自然语言处理的应用里面"))  # 香港の字体に
print(OpenCC("tw2sp").convert("電腦軟體和滑鼠"))          # 台湾の語彙 → 大陸の語彙・簡体字
```

- 文字単位の置換ではなく語単位の辞書変換なので、「发」→「發/髮」のような 1 対多の対応も文脈で解決される。
- 設定名は `s2t` `t2s` `s2tw` `s2twp` `tw2s` `tw2sp` `s2hk` `hk2s` など。
