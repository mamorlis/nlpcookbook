# 日本語の形態素解析

MeCab (fugashi)・Sudachi・Janome で単語に区切り、品詞や原形、読みを取り出す。

## MeCab (fugashi + UniDic) で形態素解析する {mecab_fugashi}

日本語の文を単語に区切って、品詞・原形・読みを得たい。Colab では pip だけで入る fugashi が手軽。

```python
!pip install -q fugashi unidic-lite
```

```python
from fugashi import Tagger

tagger = Tagger()
for word in tagger("国境の長いトンネルを抜けると雪国であった。"):
    f = word.feature
    print(word.surface, f.pos1, f.pos2, f.lemma, f.kana, sep="\t")
```

```python
# MeCab そのままの出力が欲しいとき
print(tagger.parse("すもももももももものうち"))
```

- `unidic-lite` は軽量版の UniDic。最新のフル版は `pip install unidic && python -m unidic download` (約 1GB)。
- UniDic は短単位なので「雪国」「であった」などが細かく切れる。長めの単位が欲しければ Sudachi の C モードか IPA 辞書を使う。
- `lemma` は語彙素 (「だ」「有る」「トンネル-tunnel」のように代表表記になる)。活用を戻しただけの形が欲しければ `orthBase` を使う。

## 分かち書きする {wakati}

文をスペース区切りの単語列にしたい (word2vec の学習や BM25 の前処理など)。

```python
!pip install -q fugashi unidic-lite
```

```python
from fugashi import Tagger

wakati = Tagger("-Owakati")
print(wakati.parse("自然言語処理の小ネタを集めたクックブックを作りたい。"))
```

```python
# リストで欲しいとき
tagger = Tagger()

def tokenize(text):
    return [w.surface for w in tagger(text)]

print(tokenize("自然言語処理の小ネタを集めたクックブックを作りたい。"))
```

## 品詞で絞って内容語の原形だけ取り出す {content_words}

助詞や記号を捨てて、名詞・動詞・形容詞の原形だけを取り出したい (頻度集計やトピックモデルの前処理)。

```python
!pip install -q fugashi unidic-lite
```

```python
from fugashi import Tagger

tagger = Tagger()
KEEP = {"名詞", "動詞", "形容詞", "形状詞", "副詞"}

def content_words(text):
    words = []
    for w in tagger(text):
        f = w.feature
        if f.pos1 not in KEEP:
            continue
        if f.pos2 in ("非自立可能", "数詞"):   # 「する」「いる」「こと」や数字を除く
            continue
        words.append(f.orthBase or w.surface)  # 未知語は orthBase が None
    return words

print(content_words("昨日、美しい桜の花が静かに散っていくのを3人で眺めていました。"))
```

- UniDic の `非自立可能` は「する・いる・ある・こと・もの」など機能語的な語をまとめて落とせて便利。
- IPA 辞書の場合は `pos1` が「非自立」「接尾」「代名詞」「数」のものを除く。

## 漢字の読みを取得してふりがな (ルビ) を付ける {furigana}

文の読みをカタカナ/ひらがなで得たい。HTML の `<ruby>` でふりがなを表示したい。

```python
!pip install -q fugashi unidic-lite jaconv regex
```

```python
from fugashi import Tagger
import jaconv
import regex

tagger = Tagger()
text = "吾輩は猫である。名前はまだ無い。"
print("".join(w.feature.kana or w.surface for w in tagger(text)))                 # 書字形の読み
print("".join(w.feature.pron or w.surface for w in tagger(text)))                 # 発音 (ワガハイワ…)
print(jaconv.kata2hira("".join(w.feature.kana or w.surface for w in tagger(text))))
```

```python
from IPython.display import HTML

def ruby(text):
    out = []
    for w in tagger(text):
        if regex.search(r"\p{Han}", w.surface) and w.feature.kana:
            out.append(f"<ruby>{w.surface}<rt>{jaconv.kata2hira(w.feature.kana)}</rt></ruby>")
        else:
            out.append(w.surface)
    return "".join(out)

HTML(f"<p style='font-size:1.4em; line-height:2.2'>{ruby(text)}</p>")
```

- 送り仮名ごとルビが付く (「無い」全体に「ない」)。漢字部分だけに付けたい場合は、表層と読みの末尾の共通ひらがなを削る。

## IPA 辞書で MeCab を使う {mecab_ipadic}

過去の研究や既存のコードと合わせるため、UniDic ではなく IPA 辞書で解析したい。

```python
!pip install -q fugashi ipadic
```

```python
from fugashi import GenericTagger
import ipadic

tagger = GenericTagger(ipadic.MECAB_ARGS)
for w in tagger("国境の長いトンネルを抜けると雪国であった。"):
    # IPA 辞書の素性: 品詞, 細分類1-3, 活用型, 活用形, 原形, 読み, 発音
    f = w.feature
    print(w.surface, f[0], f[1], f[6], f[7] if len(f) > 7 else "", sep="\t")
```

- `GenericTagger` は素性に名前が付かないので添字でアクセスする。未知語は読みがなく素性が 7 個になる点に注意。
- IPA 辞書は 2007 年で更新が止まっている。新しく始めるなら UniDic か Sudachi がおすすめ。

## Sudachi で分割の粒度 (A/B/C) を切り替える {sudachi_split}

「国立国会図書館」を 1 語にするか「国立/国会/図書/館」にするか、用途に応じて粒度を変えたい。

```python
!pip install -q sudachipy sudachidict_core
```

```python
from sudachipy import Dictionary, SplitMode

tokenizer = Dictionary().create()
text = "国立国会図書館で選挙管理委員会の資料を読んだ"
for mode in [SplitMode.A, SplitMode.B, SplitMode.C]:
    print(mode, [m.surface() for m in tokenizer.tokenize(text, mode)])
```

```python
for m in tokenizer.tokenize("国立国会図書館で資料を読んだ", SplitMode.C):
    print(m.surface(), m.part_of_speech()[:2], m.dictionary_form(), m.reading_form(), sep="\t")
```

- A は UniDic 短単位相当、C は固有表現相当。検索なら A と C の両方を索引に入れると再現率と適合率を両立しやすい。
- 語彙を増やすには `sudachidict_full` を入れて `Dictionary(dict="full")`。

## Sudachi の正規化形で表記ゆれを吸収する {sudachi_normalize}

「附属/付属」「シュミレーション/シミュレーション」「打込む/打ち込む」のような表記ゆれを同じ語として扱いたい。

```python
!pip install -q sudachipy sudachidict_core
```

```python
from sudachipy import Dictionary, SplitMode

tokenizer = Dictionary().create()

def normalize(text):
    return [m.normalized_form() for m in tokenizer.tokenize(text, SplitMode.C)]

for s in ["附属病院でシュミレーションを行なう", "付属病院でシミュレーションを行う", "データを打込む", "空缶をすてる", "かつ丼とカツ丼"]:
    print(s, "→", normalize(s))
```

- `dictionary_form()` は活用を戻すだけ、`normalized_form()` は表記ゆれまで統一する。頻度集計や検索では後者が便利。

## Janome で形態素解析する (Pure Python) {janome}

C 拡張なしで動く解析器を使いたい。フィルタを組み合わせた前処理パイプラインも作りたい。

```python
!pip install -q janome
```

```python
from janome.tokenizer import Tokenizer

t = Tokenizer()
for token in t.tokenize("すもももももももものうち"):
    print(token.surface, token.part_of_speech, token.base_form, token.reading, sep="\t")
print(list(t.tokenize("分かち書きもできます", wakati=True)))
```

```python
# Analyzer: 文字フィルタ → トークナイザ → トークンフィルタ のパイプライン
from janome.analyzer import Analyzer
from janome.charfilter import UnicodeNormalizeCharFilter, RegexReplaceCharFilter
from janome.tokenfilter import POSKeepFilter, LowerCaseFilter, ExtractAttributeFilter

analyzer = Analyzer(
    char_filters=[UnicodeNormalizeCharFilter(), RegexReplaceCharFilter(r"https?://\S+", "")],
    tokenizer=Tokenizer(),
    token_filters=[POSKeepFilter(["名詞", "動詞", "形容詞"]), LowerCaseFilter(), ExtractAttributeFilter("base_form")],
)
print(list(analyzer.analyze("Ｊａｎｏｍｅは遅いけれど手軽に使えて嬉しい https://example.com")))
```

- 辞書は IPA 辞書。速度は MeCab の 10 分の 1 程度なので、大規模データには向かない。

## Janome にユーザー辞書を追加する {janome_userdic}

新語や専門用語が細かく分割されてしまうので、辞書に単語を追加して 1 語として扱いたい。

```python
!pip install -q janome
```

```python
from janome.tokenizer import Tokenizer

text = "大規模言語モデルと検索拡張生成について東京スカイツリーで話した"
print(list(Tokenizer().tokenize(text, wakati=True)))
```

```python
# 簡易辞書形式: 表層形,品詞,読み
with open("userdic.csv", "w", encoding="utf-8") as f:
    f.write("大規模言語モデル,名詞,ダイキボゲンゴモデル\n")
    f.write("検索拡張生成,名詞,ケンサクカクチョウセイセイ\n")
    f.write("東京スカイツリー,名詞,トウキョウスカイツリー\n")

t = Tokenizer("userdic.csv", udic_type="simpledic", udic_enc="utf8")
print(list(t.tokenize(text, wakati=True)))
```

- MeCab や Sudachi のユーザー辞書はコンパイルが必要。手早く試すなら Janome の簡易辞書が楽。
- 辞書を作らずに済ませるなら、次の「複合名詞を抽出する」のように名詞の連続を後処理でつなげる手もある。

## 連続する名詞をつなげて複合名詞を抽出する {compound_noun}

「自然/言語/処理」のように分割された専門用語を、名詞の連続としてまとめて取り出したい。

```python
!pip install -q fugashi unidic-lite
```

```python
from collections import Counter
from fugashi import Tagger

tagger = Tagger()

def compound_nouns(text, min_len=2):
    result, buf = [], []
    for w in list(tagger(text)) + [None]:
        is_noun = w is not None and (w.feature.pos1 in ("名詞", "接頭辞", "接尾辞")) and w.feature.pos2 != "数詞"
        if is_noun:
            buf.append(w.surface)
        else:
            if len(buf) >= min_len:
                result.append("".join(buf))
            buf = []
    return result

text = "深層学習に基づく自然言語処理では、大規模言語モデルの事前学習と文法誤り訂正への応用が研究されている。自然言語処理の応用は広い。"
print(Counter(compound_nouns(text)).most_common())
```

- 専門用語抽出の最も単純なベースライン。頻度や連接の多様性でスコア付けすると用語抽出らしくなる。

## 日本語のストップワードを除去する {stopwords_ja}

「これ」「する」「ある」のように、どの文書にも出てきて特徴にならない語を取り除きたい。

```python
!pip install -q fugashi unidic-lite
```

```python
from fugashi import Tagger
from spacy.lang.ja.stop_words import STOP_WORDS  # spaCy 同梱のリスト (モデル不要)

print(len(STOP_WORDS), sorted(STOP_WORDS)[:30])
```

```python
tagger = Tagger()
MY_STOP = STOP_WORDS | {"思う", "言う", "行う"}  # タスクに合わせて追加

def tokenize(text):
    words = [w.feature.orthBase or w.surface for w in tagger(text) if w.feature.pos1 not in ("助詞", "助動詞", "補助記号")]
    return [w for w in words if w not in MY_STOP]

print(tokenize("これは私が昨日行った実験の結果であると思います。"))
```

- 品詞フィルタ (助詞・助動詞・記号を落とす) とストップワードリストの併用が基本。
- コーパスから自動で作る方法もある: 文書頻度が極端に高い語を落とす (`TfidfVectorizer(max_df=0.8)`)。

## 単語 n-gram を作る {word_ngram}

単語 bigram / trigram を数えて、よく出る言い回しを調べたい。

```python
!pip install -q fugashi unidic-lite
```

```python
from collections import Counter
from fugashi import Tagger

tagger = Tagger()

def ngrams(tokens, n):
    return list(zip(*[tokens[i:] for i in range(n)]))

text = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。吾輩はここで始めて人間というものを見た。"
tokens = [w.surface for w in tagger(text)]
print(ngrams(tokens, 2)[:5])
print(Counter(ngrams(tokens, 2)).most_common(3))
```

```python
# 文をまたがないようにし、文頭・文末記号を付ける
def sent_ngrams(sentence, n=2):
    tokens = ["<s>"] * (n - 1) + [w.surface for w in tagger(sentence)] + ["</s>"]
    return ngrams(tokens, n)

print(sent_ngrams("名前はまだ無い。"))
```

- NLTK にも `nltk.ngrams(tokens, n)` や `nltk.lm.preprocessing.pad_both_ends` がある。

## 動詞・形容詞の活用型と活用形を調べる {conjugation}

用言がどの活用形で出現しているかを知りたい (敬語・文体の分析、学習者作文の誤り分析など)。

```python
!pip install -q fugashi unidic-lite
```

```python
from fugashi import Tagger

tagger = Tagger()
for w in tagger("食べられなかったので、早く帰りましょうと言われた。"):
    f = w.feature
    if f.cType and f.cType != "*":
        print(w.surface, f.pos1, f.orthBase, f.cType, f.cForm, sep="\t")
```

```python
# 応用: 文末が「です・ます」体かどうかの判定
def is_polite(sentence):
    lemmas = [w.feature.lemma for w in tagger(sentence) if w.feature.pos1 == "助動詞"]
    return any(l in ("です", "ます") for l in lemmas)

for s in ["明日は雨が降るでしょう。", "明日は雨が降るだろう。", "ご飯を食べました。", "ご飯を食べた。"]:
    print(is_polite(s), s)
```

## トークンの文字位置 (オフセット) を取得する {token_offsets}

各単語が元の文の何文字目から何文字目にあるかを知りたい (アノテーションとの対応付け、ハイライト表示など)。

```python
!pip install -q sudachipy sudachidict_core fugashi unidic-lite
```

```python
from sudachipy import Dictionary, SplitMode

tokenizer = Dictionary().create()
text = "東京都の 小池知事が会見した。"
for m in tokenizer.tokenize(text, SplitMode.C):
    print(m.begin(), m.end(), repr(m.surface()), repr(text[m.begin():m.end()]))
```

```python
# MeCab (fugashi) はオフセットを返さないので、表層形を前から順に探して復元する
from fugashi import Tagger

tagger = Tagger()

def with_offsets(text):
    pos, out = 0, []
    for w in tagger(text):
        start = text.index(w.surface, pos)
        pos = start + len(w.surface)
        out.append((start, pos, w.surface))
    return out

print(with_offsets(text))
```

- Sudachi は空白もトークンとして返すので、足し合わせると元の文に戻る。MeCab は空白を捨てる。
- 正規化をかけてから解析するとオフセットがずれる。先に正規化するなら、正規化後の文字列を「原文」として扱う。

## DataFrame の大量テキストをまとめて分かち書きする {tokenize_pandas}

数万件のテキスト列を持つ DataFrame を、進捗を見ながら効率よくトークン化したい。

```python
!pip install -q fugashi unidic-lite
```

```python
import pandas as pd
from fugashi import Tagger
from tqdm.auto import tqdm

df = pd.DataFrame({"text": ["今日は良い天気ですね。", "自然言語処理を勉強しています。", "明日は雨が降るらしい。"] * 10000})
tagger = Tagger("-Owakati")  # Tagger は 1 回だけ作って使い回す (毎回作ると非常に遅い)

tqdm.pandas()
df["wakati"] = df["text"].progress_apply(lambda s: tagger.parse(s).strip())
df.head(3)
```

```python
# 複数コアで並列化 (プロセスごとに Tagger を作る)
from joblib import Parallel, delayed
import numpy as np

def work(texts):
    t = Tagger("-Owakati")
    return [t.parse(s).strip() for s in texts]

chunks = np.array_split(df["text"].to_numpy(), 8)
results = Parallel(n_jobs=2)(delayed(work)(c) for c in chunks)
df["wakati2"] = [s for chunk in results for s in chunk]
print((df["wakati"] == df["wakati2"]).all())
```

- Colab の無料枠は 2 コアなので並列化の効果は 2 倍弱。MeCab は 1 コアでも毎秒数万文は処理できる。

## 解析器・辞書による分割の違いを比べる {tokenizer_compare}

MeCab (UniDic / IPA 辞書)、Sudachi、Janome で同じ文がどう分割されるかを見比べたい。

```python
!pip install -q fugashi unidic-lite ipadic sudachipy sudachidict_core janome
```

```python
from fugashi import Tagger, GenericTagger
import ipadic
from sudachipy import Dictionary, SplitMode
from janome.tokenizer import Tokenizer

unidic = Tagger()
ipa = GenericTagger(ipadic.MECAB_ARGS)
sudachi = Dictionary().create()
janome = Tokenizer()

tokenizers = {
    "MeCab+UniDic": lambda s: [w.surface for w in unidic(s)],
    "MeCab+IPAdic": lambda s: [w.surface for w in ipa(s)],
    "Sudachi A": lambda s: [m.surface() for m in sudachi.tokenize(s, SplitMode.A)],
    "Sudachi C": lambda s: [m.surface() for m in sudachi.tokenize(s, SplitMode.C)],
    "Janome": lambda s: list(janome.tokenize(s, wakati=True)),
}

for s in ["外国人参政権について話し合った。", "東京都立大学で機械翻訳を研究しています。", "食べさせられたくなかった。"]:
    print(s)
    for name, f in tokenizers.items():
        print(f"  {name:13}", " | ".join(f(s)))
```

- 単語の単位は辞書の設計思想で決まる。評価データや事前学習モデルが使っている辞書に合わせるのが鉄則。

## MeCab の N-best 解を得る {mecab_nbest}

曖昧な文に対して、最も確からしい分割だけでなく 2 位以下の候補も見たい。

```python
!pip install -q fugashi unidic-lite
```

```python
from fugashi import Tagger

tagger = Tagger("-Owakati")

def nbest(text, n=20):
    # 品詞だけが違う解は分かち書きでは同じになるので、重複を除く
    return list(dict.fromkeys(s.strip() for s in tagger.nbest(text, n).split("\n")))

for text in ["くるまでまつ", "ここではきものを脱ぐ"]:
    for i, cand in enumerate(nbest(text), 1):
        print(i, cand)
```

- 「車で待つ / 来るまで待つ」「ここで履物を / ここでは着物を」のような曖昧性の例を作るのにも使える。

## 辞書にない未知語を見つける {unknown_words}

コーパス中の未知語 (辞書に載っていない語) を洗い出して、辞書追加の候補にしたい。

```python
!pip install -q fugashi unidic-lite
```

```python
from collections import Counter
from fugashi import Tagger

tagger = Tagger()
texts = ["ChatGPTでバズった投稿をリポストした。", "推し活でエモい写真をインスタにあげた。", "ググってもわからんかった。"]

unk = Counter()
for text in texts:
    for w in tagger(text):
        if w.is_unk:
            unk[w.surface] += 1
        print(f"{w.surface}{'*' if w.is_unk else ''}", end=" ")
    print()
print(unk.most_common())
```

- `*` が未知語。未知語でなくても誤って細かく分割されている語 (「バズった」「エモい」「インスタ」) は検出できない点に注意。

## 日本語の単語を読みの五十音順に並べる {sort_by_reading}

漢字を含む単語リストを、文字コード順ではなく読みの五十音順でソートしたい (索引や用語集の作成)。

```python
!pip install -q fugashi unidic-lite jaconv
```

```python
from fugashi import Tagger
import jaconv

tagger = Tagger()

def reading(word):
    kana = "".join(w.feature.kana or w.surface for w in tagger(word))
    return jaconv.kata2hira(kana)

words = ["機械翻訳", "形態素解析", "言語モデル", "アテンション", "構文解析", "埋め込み", "要約", "分かち書き"]
print(sorted(words))  # 文字コード順 (漢字が読みと無関係に並ぶ)
for w in sorted(words, key=reading):
    print(reading(w), w, sep="\t")
```

- 固有名詞や専門用語は読みを誤ることがあるので、最終的には人手の確認が必要。
