# 日本語の形態素解析と文分割

MeCab (fugashi)・Sudachi・Janome・Juman++・KyTea で単語に区切り、品詞や原形、読みを取り出す。NEologd のような辞書の追加、文への分割、単語分割や品詞タグ付けの仕組みも扱う。

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

## NEologd 辞書で新語・固有名詞をまとまりとして解析する {mecab_neologd}

人名・作品名・組織名や新語が IPA 辞書では細かく切れてしまうので、1 語として扱いたい。ウェブ上の語彙から新語・固有表現を IPA 辞書に追加した MeCab 用のシステム辞書 mecab-ipadic-NEologd を使う。

```python
# Colab で 5 分ほどかかる (辞書のダウンロードとビルド。できる辞書は 900MB ほど)
!apt-get install -y -q mecab libmecab-dev mecab-ipadic-utf8 > /dev/null
!git clone -q --depth 1 https://github.com/neologd/mecab-ipadic-neologd
!cd mecab-ipadic-neologd && ./bin/install-mecab-ipadic-neologd -n -y -u > ../neologd_install.log 2>&1; tail -n 3 ../neologd_install.log
!pip install -q fugashi
# インストール先は mecab-config --dicdir の下の mecab-ipadic-neologd
!ls -lh $(mecab-config --dicdir)/mecab-ipadic-neologd/sys.dic
!echo "米津玄師の新曲" | mecab -d $(mecab-config --dicdir)/mecab-ipadic-neologd
```

```python
import subprocess
from fugashi import GenericTagger

dicdir = subprocess.run(["mecab-config", "--dicdir"], capture_output=True, text=True).stdout.strip()
NEOLOGD = f"{dicdir}/mecab-ipadic-neologd"
# fugashi は既定で /usr/local/etc/mecabrc を探すので、apt の MeCab の設定ファイルを -r で渡す
ipa = GenericTagger("-r /etc/mecabrc")                  # apt の IPA 辞書 (mecabrc の既定の辞書)
neo = GenericTagger(f"-r /etc/mecabrc -d {NEOLOGD}")    # NEologd

sents = [
    "米津玄師の新曲が鬼滅の刃の主題歌に決まった。",
    "大谷翔平と藤井聡太が話題になった。",
    "きゃりーぱみゅぱみゅがNHK紅白歌合戦に出場した。",
    "君の名は。を観てからインスタ映えする場所を巡った。",
    "逃げるは恥だが役に立つの恋ダンスが流行った。",
    "ほぼ日刊イトイ新聞とメルカリとZOZOTOWNを比べる。",
]
for s in sents:
    print("IPA    :", " | ".join(w.surface for w in ipa(s)))
    print("NEologd:", " | ".join(w.surface for w in neo(s)))
    print()
```

```python
# NEologd で固有名詞になった語の品詞 (細分類) と読み。素性は IPA 辞書と同じ並び
for s in sents:
    for w in neo(s):
        f = w.feature
        if f[1] == "固有名詞":
            print(w.surface, ",".join(f[:4]), f[7] if len(f) > 7 else "(読みなし: 未知語)", sep="\t")
```

```python
# 収録されていない新しい語 (2020 年 9 月より後に広まった語など) は、IPA 辞書と同じく切れるか未知語になる
for s in ["推しの子のアニメを観た。", "ChatGPTと生成AIの話をした。"]:
    for w in neo(s):
        f = w.feature
        print(w.surface, ",".join(f[:4]), f[7] if len(f) > 7 else "(読みなし: 未知語)", sep="\t")
    print()
```

```python
# 長い固有名詞が 1 語になると、その一部の語で検索したときに見つからなくなる (再現率が下がる)
docs = [
    "東京大学大学院情報理工学系研究科の共同研究。",
    "東京大学の入試が行われた。",
    "新型コロナウイルス感染症の影響でテレワークが広がった。",
    "コロナ禍でオンライン授業が増えた。",
    "コロナの影響で休校になった。",
]
for name, tagger in [("IPA", ipa), ("NEologd", neo)]:
    index = [set(w.surface for w in tagger(d)) for d in docs]
    for q in ["東京大学", "コロナ"]:
        print(f"{name:8}{q}: {[i for i, toks in enumerate(index) if q in toks]}")
print(" | ".join(w.surface for w in neo(docs[0])))
print(" | ".join(w.surface for w in neo(docs[2])))

# 対策の一例: 検索用の索引には両方の辞書のトークンを入れる (README も IPA 辞書との併用を勧めている)
index = [set(w.surface for w in ipa(d)) | set(w.surface for w in neo(d)) for d in docs]
for q in ["東京大学", "コロナ", "新型コロナウイルス感染症"]:
    print(f"両方    {q}: {[i for i, toks in enumerate(index) if q in toks]}")
```

- mecab-ipadic-NEologd は、はてなキーワード・郵便番号データ・駅名・ニュース記事・ウェブの文書などから集めた新語・固有表現を IPA 辞書 (mecab-ipadic-2.7.0-20070801) に追加したシステム辞書 (佐藤敏紀氏ら、Apache License 2.0)。IPA 辞書と同じ素性の並びで読みも付く。apt の mecab-ipadic-utf8 と pip の `ipadic` (前のレシピ「IPA 辞書で MeCab を使う」) は同じ版の IPA 辞書。NEologd は IPA 辞書の語の読みや生起コストにもパッチを当てるので、IPA 辞書との差がすべて新語によるものとは限らない。
- **更新は止まっている。** README には「少なくとも毎週 2 回更新」とあるが、master の最後のコミットは 2020 年 9 月 14 日 (シードのデータは 2020-09-10 版) で、それ以降の語は入っていない (2026 年 11 月時点で、リポジトリはアーカイブされていない)。上の例では、収録されていない「推しの子」が「推し / の子」に切れて「の子」が人名になり、「ChatGPT」は読みのない未知語になる。
- 固有名詞の細分類は粗く、誤りもある (上の例では「ほぼ日刊イトイ新聞」が「人名」)。README の「欠点」にも、分類が不十分なこと、固有表現ではない語も固有表現として登録されていること、読みの誤りがあることが書かれている。入力は、NEologd の正規化ルールを実装した neologdn (文字列処理と正規化の章の「neologdn で日本語の表記ゆれを正規化する」) で正規化してから解析するとよい。
- 長い固有名詞が 1 語になるので、トークン単位で「東京大学」「コロナ」と検索しても「東京大学大学院情報理工学系研究科」「新型コロナウイルス感染症」は見つからない。README も、ウェブの文書の解析では標準の IPA 辞書との併用を勧めている。単語の単位を評価データや学習済みモデルに合わせる必要がある場合は使わない。
- インストーラーは元の IPA 辞書の tar.gz を自分でダウンロードし (SHA-1 で照合)、パッチを当ててビルドする。最初のダウンロード先 osdn.net は検証時 (2026 年 11 月) 名前解決できず、SourceForge から取得された。`-n` は GitHub の最新版に更新してからインストール、`-y` はテスト結果の確認を省略、`-u` はインストール先が自分のディレクトリでなくても sudo を使わずに `make install` する (付けないと Colab では `sudo make install` を通る)。インストール先は `mecab-config --dicdir` の下 (Colab では `/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd`) で、`-p ディレクトリ` で変えられる。
- README の目安では、標準のインストールに空きメモリが最低 2GB (推奨 6GB) 必要。手元の検証では、ビルド中の最大のプロセスのメモリ使用量 (最大常駐サイズ) は約 1.6GB、できた `sys.dic` は `ls -lh` で 866M (README では約 910MB) だった。`--ignore_adverb` などの `--ignore_*` で一部の辞書を外すと小さくなり (README によれば全部指定して約 300MB 減る)、`-a` では標準では入らない低頻度の日時・数量表現などもすべて入る。Colab のランタイムを作り直すとインストールはやり直しになる。

## 古文を形態素解析する (中古和文 UniDic) {classical_japanese}

源氏物語や枕草子のような古文を単語に分け、品詞・活用・読みを調べたい。現代語用の辞書ではうまく解析できないので、古文用の辞書を使う。

```python
!pip install -q fugashi unidic-lite
!wget -q https://clrd.ninjal.ac.jp/unidic_archive/2512/unidic-chuko-v202512.zip && unzip -q -o unidic-chuko-v202512.zip -d chuko
!find chuko -name "dicrc" | head -n 1
```

```python
import glob, os
from fugashi import GenericTagger, Tagger

dicdir = os.path.dirname(glob.glob("chuko/**/dicrc", recursive=True)[0])
classical = GenericTagger(f'-d "{dicdir}" -r "{dicdir}/dicrc"')      # 中古和文 UniDic (平安時代の仮名文学向け)
modern = Tagger()                                                    # 現代語の UniDic (unidic-lite)

text = "いづれの御時にか、女御、更衣あまたさぶらひたまひけるなかに、いとやむごとなき際にはあらぬが、すぐれて時めきたまふありけり。"
print("中古和文 UniDic:", " | ".join(w.surface for w in classical(text)))
print("現代語 UniDic :", " | ".join(w.surface for w in modern(text)))
```

```python
# 素性の並びを確認する (UniDic 共通の形式: 品詞 4 階層, 活用型, 活用形, 語彙素読み, 語彙素, ...)
for w in classical("春はあけぼの。やうやう白くなりゆく山ぎは、少し明かりて、紫だちたる雲の細くたなびきたる。"):
    f = w.feature
    print(f"{w.surface}\t{f[0]}-{f[1]}\t{f[4]}\t{f[5]}\t語彙素: {f[7]} ({f[6]})")
```

```python
# 助動詞を取り出して、意味を調べる手がかりにする
def auxiliaries(text):
    return [(w.surface, w.feature[7], w.feature[5]) for w in classical(text) if w.feature[0] == "助動詞"]

for s in ["竹取の翁といふ者ありけり。", "花の色は移りにけりないたづらに", "男もすなる日記といふものを、女もしてみむとてするなり。", "行く河の流れは絶えずして、しかももとの水にあらず。"]:
    print(s, "→", auxiliaries(s))
```

```python
# 現代語の辞書で解析すると、どこが違ってくるか
import difflib

s = "いづれの御時にか、女御、更衣あまたさぶらひたまひけるなかに"
a, b = [w.surface for w in classical(s)], [w.surface for w in modern(s)]
print("中古和文:", a); print("現代語  :", b)
print("一致する単語の割合:", round(difflib.SequenceMatcher(None, a, b).ratio(), 2))
# 区切りが同じでも、品詞や語彙素の判定が違うことがある (伝聞の「なる」を、現代語の辞書は断定の「なり」とする)
s2 = "男もすなる日記といふものを"
print("中古和文:", [(w.surface, w.feature[7]) for w in classical(s2)][:5])
print("現代語  :", [(w.surface, w.feature.lemma) for w in modern(s2)][:5])
```

- 古文は、語彙も活用も仮名遣いも現代語と違う。現代語用の辞書では、助動詞 (けり、なり、む) や歴史的仮名遣い (いふ、やうやう) を正しく扱えない。
- 国立国語研究所が、時代・文体別の UniDic を公開している。平安時代の仮名文学には「中古和文」、明治の文語文には「近代文語」、ほかに上代語、中世の説話や狂言、近世の洒落本向けなどがある。対象のテキストに合うものを選ぶ。
- 辞書の形式は現代語の UniDic と共通なので、MeCab (fugashi) の辞書を差し替えるだけで使える。語彙素 (辞書の見出し) でまとめれば、表記のゆれをそろえて数えられる。
- 活用形や助動詞の語彙素が分かるので、古文の学習支援 (品詞分解)、文体の計量的な分析、時代による語の変化の調査に使える。
- 辞書と違う時代・文体のテキスト (漢文訓読体、和漢混交文、近世の口語) では精度が落ちる。解析結果を抜き取って確かめる。

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
- 辞書を作らずに済ませるなら、この章の「連続する名詞をつなげて複合名詞を抽出する」のように名詞の連続を後処理でつなげる手もある。

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

## Juman++ で形態素解析する (代表表記・カテゴリ付き) {jumanpp}

単語の区切りと品詞に加えて、「りんご/林檎/リンゴ」をまとめる代表表記や、「人」「場所」のような意味カテゴリも欲しい。京都大学の研究室が開発した、RNN 言語モデルで単語列のもっともらしさを評価する形態素解析器 Juman++ を使う。

```python
# ダウンロード (約 300MB)・展開・ビルドで、Colab (2 コア) では 10 分ほどかかる見込み
!wget -q https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc4/jumanpp-2.0.0-rc4.tar.xz
!tar xf jumanpp-2.0.0-rc4.tar.xz
!cd jumanpp-2.0.0-rc4 && mkdir -p bld && cd bld && cmake .. -DCMAKE_BUILD_TYPE=Release -DJPP_ENABLE_TESTS=OFF -DCMAKE_INSTALL_PREFIX=/usr/local > /dev/null && make install -j2 > /dev/null 2>&1
!pip install -q rhoknp fugashi unidic-lite
!jumanpp --version
```

```python
import subprocess

# 生の出力 (JUMAN 形式): 表層形 読み 原形 品詞 品詞ID 細分類 ID 活用型 ID 活用形 ID 意味情報
r = subprocess.run(["jumanpp"], input="外国人参政権について話し合った。\n", capture_output=True, text=True)
print(r.stdout)
```

```python
from rhoknp import Jumanpp

# skip_sanity_check=True: 起動時の確認 (空の文を送る) を省く (下の注意書きを参照)
# 別の場所に入れたときは Jumanpp("/path/to/bin/jumanpp", skip_sanity_check=True)。options=["--beam", "10"] でオプションも渡せる
jumanpp = Jumanpp(skip_sanity_check=True)

def analyze(text, keys=("代表表記", "カテゴリ", "ドメイン")):
    for m in jumanpp.apply_to_sentence(text).morphemes:
        sem = {k: v for k, v in m.semantics.items() if keys is None or k in keys}
        print(m.text, m.reading, m.lemma, m.pos, m.subpos, m.conjtype, m.conjform, sem, sep="\t")

analyze("国境の長いトンネルを抜けると雪国であった。")
print()
analyze("スマホでググった。", keys=None)   # 意味情報をすべて表示
print()

# 生の出力の @ 行 (同形異義語) は homographs に入る
for m in jumanpp.apply_to_sentence("外国人参政権について話し合った。").morphemes:
    if m.homographs:
        print(m.text, m.canon, [h.canon for h in m.homographs])
```

```python
# 代表表記 (canon) で表記ゆれをまとめる。代表表記がない語 (自動獲得語など) は canon が None なので表層形で代用する
from collections import Counter

texts = ["りんごを食べた。", "林檎を食べた。", "リンゴを食べた。", "子どもと子供とこどもが遊ぶ。"]
surf, canon = Counter(), Counter()
for t in texts:
    for m in jumanpp.apply_to_sentence(t).morphemes:
        if m.pos == "名詞":
            surf[m.text] += 1
            canon[m.canon or m.text] += 1
print("表層形:  ", surf.most_common())
print("代表表記:", canon.most_common())
```

```python
# MeCab + UniDic (fugashi) と、分割と品詞を比べる (例文は「解析器・辞書による分割の違いを比べる」と同じ)
from fugashi import Tagger

unidic = Tagger()
for s in ["外国人参政権について話し合った。", "東京都立大学で機械翻訳を研究しています。", "食べさせられたくなかった。"]:
    print(s)
    print("  Juman++     ", " | ".join(f"{m.text}/{m.pos}" for m in jumanpp.apply_to_sentence(s).morphemes))
    print("  MeCab+UniDic", " | ".join(f"{w.surface}/{w.feature.pos1}" for w in unidic(s)))
```

- 品詞体系と辞書は JUMAN のもので、UniDic とは品詞名も単位も違う。「であった」は判定詞 1 語、「させ」「られ」「たく」「なかった」は接尾辞になる (UniDic では助動詞や形容詞で、「なかっ/た」と切れる)。構文解析の章の「KWJA で文節・基本句の係り受けと並列構造を解析する」の KWJA も同じ品詞体系で、結果は rhoknp で同じように扱える。
- 「外国人参政権」は、Juman++ では「外国/人/参政/権」、MeCab + UniDic では「外国/人参/政権」になった。Juman++ は単語列の意味的なもっともらしさを RNN 言語モデルで考慮する (README から張られたウェブデモもこの文を例にしている)。比べた 3 文は、この章の「解析器・辞書による分割の違いを比べる」と同じ例文なので、Sudachi や Janome の結果とも見比べられる。
- 意味情報には、代表表記、カテゴリ (人、場所-施設、抽象物など)、ドメイン (政治、料理・食事など)、自他動詞や反義語の対応が入る。代表表記は「Sudachi の正規化形で表記ゆれを吸収する」の正規化形と同じ目的に使え、「りんご/林檎/リンゴ」は「林檎/りんご」に、「子ども/子供/こども」は「子供/こども」にまとまった。
- 生の出力の `@` で始まる行は、読みも品詞も同じで代表表記だけが違う候補 (同形異義語)。「スマホ」のような自動獲得語は代表表記がなく (`canon` が None)、意味情報に「Wikipediaリダイレクト:スマートフォン」が入った。
- Juman++ (v2.0.0-rc4) は、最初の入力行が空行だと異常終了 (セグメンテーション違反) することがある (毎回ではない)。rhoknp (1.8.2) の `Jumanpp()` は起動時の確認で空の文を送るので、異常終了した回は「failed to start Juman++」という警告が出る (解析は再起動したプロセスで行われる)。`skip_sanity_check=True` で確認を省いたときは、最初に空文字列を渡さない。subprocess で直接渡すときも先頭に空行を置かない。
- 配布物 (約 300MB) に学習済みのモデルが含まれるので、GitHub のソースではなくリリースの tar.xz を使う。ライセンスは Apache 2.0。最新版は 2023 年の v2.0.0-rc4。

## 日本語の文を句点で分割する (括弧内は分割しない) {sentence_split_ja}

「。」「！」「？」で文に区切りたいが、鉤括弧の中の句点では切りたくない。

```python
import re

def split_sentences(text):
    sentences, buf, depth = [], "", 0
    pairs = {"「": "」", "『": "』", "（": "）", "(": ")"}
    closers = set(pairs.values())
    for i, ch in enumerate(text):
        buf += ch
        if ch in pairs:
            depth += 1
        elif ch in closers:
            depth = max(0, depth - 1)
        elif depth == 0 and ch in "。！？!?\n":
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt not in "。！？!?」』）)":  # 「！？」などの連続は最後で切る
                sentences.append(buf.strip())
                buf = ""
    if buf.strip():
        sentences.append(buf.strip())
    return [s for s in sentences if s]

text = "彼は「もう帰る。明日も早いから」と言った。本当ですか！？ はい（たぶん。）そうです。最後は句点なし"
for s in split_sentences(text):
    print("-", s)
```

```python
# 括弧を気にしなくてよいなら 1 行で書ける
print(re.findall(r"[^。！？]+[。！？]*", "今日は晴れ。明日は雨？そうかも！"))
```

## 日本語の文分割をライブラリで行って比べる {sentence_split_libs}

日本語の文章を文に区切りたい。括弧の中の句点、改行、絵文字や顔文字で終わる文など、単純な規則では失敗する場合も扱いたい。

```python
!pip install -q ja-sentence-segmenter pysbd ginza ja_ginza
```

```python
import functools
import re
import pysbd
import spacy
from ja_sentence_segmenter.common.pipeline import make_pipeline
from ja_sentence_segmenter.concatenate.simple_concatenator import concatenate_matching
from ja_sentence_segmenter.normalize.neologd_normalizer import normalize
from ja_sentence_segmenter.split.simple_splitter import split_newline, split_punctuation

# 1. 正規表現だけ
def split_regex(text):
    return [s for s in re.split(r"(?<=[。！？!?])|\n", text) if s and s.strip()]

# 2. ja-sentence-segmenter: 正規化 → 改行で分割 → 句読点で分割 (括弧内は分割しない) → 分割しすぎた箇所を連結、を組み合わせる
split_punc = functools.partial(split_punctuation, punctuations=r"。!?")
concat_tail = functools.partial(concatenate_matching, former_matching_rule=r"^(?P<result>.+)(て)$", remove_former_matched=False)
jss = make_pipeline(normalize, split_newline, concat_tail, split_punc)

# 3. pySBD: 多言語対応の規則ベース  4. GiNZA: 係り受け解析の結果から文境界を決める
sbd = pysbd.Segmenter(language="ja", clean=False)
nlp = spacy.load("ja_ginza")

splitters = {
    "正規表現": split_regex,
    "ja-sentence-segmenter": lambda t: list(jss(t)),
    "pySBD": lambda t: [s.strip() for s in sbd.segment(t)],
    "GiNZA": lambda t: [s.text.strip() for s in nlp(t.replace("\n", "")).sents],
}

text = ("彼は「もう帰る。明日も早いから」と言った。本当ですか！？\n"
        "はい（たぶん。）そうです。今日は楽しかった😊また行こうね(^^)\n"
        "値段は1.5倍になった。詳しくはhttps://example.com/a.htmlを見て\n"
        "ください。最後は句点なし")
for name, f in splitters.items():
    sents = f(text)
    print(f"[{name}] {len(sents)} 文")
    for s in sents:
        print("   ", s)
```

```python
# 大量の文章を処理するときの速度
import time

long_text = "今日は良い天気です。散歩に行きましょう！どこへ行きますか？「公園がいい。近いから」と彼は言った。" * 200
for name, f in splitters.items():
    t = time.perf_counter()
    n = len(f(long_text))
    print(f"{name:22} {n:5} 文  {time.perf_counter() - t:6.2f} 秒")
```

- 単純な正規表現は、括弧内の句点で切ってしまい、絵文字や顔文字で終わる文は切れない。小数点や URL の中のピリオドは、全角の句点だけを区切りにすれば避けられる。
- ja-sentence-segmenter は、分割と連結の規則を自分で組み替えられる。行の途中で折り返された文章 (メールや PDF 由来のテキスト) をつなぎ直す処理も書ける。
- GiNZA は句点のない文末 (絵文字、顔文字の後ろ) も区切れるが、解析を伴うので 2 桁以上遅い。括弧内で切ってしまうこともある。
- SNS や口コミのような崩れた文章向けには、学習ベースの bunkai がある (執筆時点では依存ライブラリ emoji の新しい版と互換性がなく、そのままでは動かなかった)。
- 文分割の誤りは後段のすべての処理に響く。自分のデータの数十文で、どこで切れているかを必ず目で確かめる。

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

- 辞書を使わず、文字の並びだけから単語の境界を学習する。学習データにない分野の語 (2 文目) では誤りが増える。CRF はラベル列全体をまとめて決めるが、次のレシピの KyTea は境界ごとに独立に判定し (点予測)、辞書も素性に使う。
- 中国語の単語分割では、B/M/E/S (先頭・中間・末尾・単独) の 4 タグが標準的。

## KyTea で単語分割と読みの推定をする {kytea}

辞書の照合ではなく、文字の並びから「ここで単語が切れるか」を判定する解析器で、単語分割と品詞・読みを得たい。辞書にない語の読みも推定できる KyTea (京都テキスト解析ツールキット) を使う。

```python
!pip install -q kytea fugashi unidic-lite
# 学習済みの日本語モデル (gzip で約 38MB、展開後 約 120MB) はソースのリポジトリに同梱されている。pip のパッケージには含まれない
!wget -q https://github.com/neubig/kytea/raw/master/data/model.bin.gz && gunzip -f model.bin.gz && ls -lh model.bin
```

```python
import Mykytea

kytea = Mykytea.Mykytea("-model model.bin")   # オプションはコマンド版の kytea と同じ書き方で渡す
s = "太郎は図書館で借りた本を花子に渡した。"
print(" ".join(kytea.getWS(s)))      # 単語分割だけ
print(kytea.getTagsToString(s))      # 単語/品詞/読み

# getTags は単語ごとに [品詞の候補, 読みの候補] を返す (各候補は (タグ, スコア))
for w in kytea.getTags(s):
    (pos, pos_score), = w.tag[0]
    (yomi, yomi_score), = w.tag[1]
    print(w.surface, pos, yomi, round(pos_score, 2), round(yomi_score, 2), sep="\t")
```

```python
# getAllTags は 2 位以下の候補とスコアも返す (-tagmax で候補数を変えられる。既定は 3)。読みは文脈で変わる
for s in ["日本語の本を今日読んだ。", "今日は日本語の本を読んだ。"]:
    for w in kytea.getAllTags(s):
        if w.surface == "日本":
            print(s, "日本 の読みの候補:", [(t, round(sc, 2)) for t, sc in w.tag[1]])

# -notags は分割だけ、-notag 1 は 1 番目のタグ (品詞) を推定せず読みだけを付ける
print(Mykytea.Mykytea("-model model.bin -notags").getTagsToString("日本語の本を今日読んだ。"))
print(Mykytea.Mykytea("-model model.bin -notag 1").getTagsToString("日本語の本を今日読んだ。"))
```

```python
# MeCab + UniDic (fugashi) と分割の単位を比べる
from fugashi import Tagger

tagger = Tagger()
for s in ["食べさせられたくなかった。", "大規模言語モデルを用いた検索拡張生成の論文を読んだ。", "ググってもわからんかった。"]:
    print("KyTea  :", " | ".join(kytea.getWS(s)))
    print("UniDic :", " | ".join(w.surface for w in tagger(s)))
```

```python
# UniDic (unidic-lite) で未知語になったり細かく分かれたりする語を含む文 (最後の文は「辞書にない未知語を見つける」と同じ)。
# UniDic は未知語に読みを付けられないが、KyTea は自分の辞書にない語 (「鬼滅」「ジーピーティー」) も文字の並びから区切りと読みを推定する
for s in ["鬼滅の刃の竈門炭治郎が好きだ。", "チャットジーピーティーに聞いた。", "彼はスマホをポチポチいじっていた。", "推し活でエモい写真をインスタにあげた。"]:
    print("KyTea  :", kytea.getTagsToString(s))
    print("UniDic :", " ".join(f"{w.surface}/{'(未知語)' if w.is_unk else w.feature.kana or w.surface}" for w in tagger(s)))
```

```python
# 速度の比較 (10,000 文)
import time

texts = ["国境の長いトンネルを抜けると雪国であった。", "自然言語処理の小ネタを集めたクックブックを作りたい。"] * 5000
for name, f in [("KyTea (分割のみ)", Mykytea.Mykytea("-model model.bin -notags").getWS),
                ("KyTea (品詞・読み)", kytea.getTags),
                ("MeCab + UniDic", tagger)]:
    t = time.perf_counter()
    for x in texts:
        f(x)
    print(f"{name:16} {time.perf_counter() - t:5.2f} 秒")
```

- KyTea は Graham Neubig らが開発した解析器。文字と文字の間ごとに単語境界かどうかを、周りの文字 n-gram・文字種・辞書との一致を素性とする線形分類器 (線形 SVM やロジスティック回帰。学習は LIBLINEAR) で 1 か所ずつ独立に判定し、単語の品詞と読みも個別に推定する (点予測)。前のレシピの CRF はラベル列全体をまとめて決めるが、KyTea は隣の判定結果を使わないので、一部の境界や読みだけを付けた「部分的アノテーション」のデータでも学習できる。
- 同梱のモデルは現代日本語書き言葉均衡コーパス (BCCWJ) や UniDic などで学習されたもので、活用語尾を語幹と別の単語にする (「渡 し た」「読 ん だ」「た く な かっ た」)。MeCab + UniDic の短単位 (「読ん」「たく」「なかっ」) より細かい。構文解析の章の「Ckylark で日本語の句構造解析をする」の日本語モデルは、この分割を前提にしている。
- 読みはひらがなで、単語の表記に対応する部分だけが付く (「渡」→「わた」)。送り仮名が別の単語になるので、この章の「漢字の読みを取得してふりがな (ルビ) を付ける」(送り仮名ごとルビが付く) と違い、漢字の部分だけにルビを付けやすい。辞書で読みの候補が 1 つしかない語はスコアが 100 になり、「日本」のように候補が複数ある語は文脈で結果が変わる (「にほん」と「にっぽん」)。
- 辞書にない語の推定は誤ることもある (「推し活」は「推 (お) / し活」に分割された)。Python のバインディングからは未知語かどうかは分からない。コマンド版の `kytea -unktag '[UNK]'` で同じ文を試すと、「鬼滅」「ジーピーティー」「し活」「エモ」「ポチポチ」に印が付き、「スマホ」「インスタ」は KyTea の辞書にある語だった。
- 検証環境で 10,000 文にかかった時間は、品詞・読みありが 0.8〜0.9 秒、分割だけ (`-notags`) が約 0.15 秒、MeCab + UniDic が 0.1 秒ほど。モデルの読み込みに約 2 秒かかるので、`Mykytea` は 1 回だけ作って使い回す。
- pip の `kytea` (Mykytea-python) は KyTea のライブラリを含むのでビルドは不要だが、モデルは含まない。モデルは研究・商用での利用は許可されているが無断の再配布は禁じられているので、各自が作者のリポジトリから取得する。コマンド版や学習用の `train-kytea` (`-part` で部分的アノテーション、`-dict` で辞書を与える) が要るなら、構文解析の章の「Ckylark で日本語の句構造解析をする」のようにソースからビルドする。ソースは Apache License 2.0 で、最後のリリースは 0.4.7 (2014 年)、リポジトリの最後の更新は 2020 年。

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
    """emission: (系列長, ラベル数), transition: (ラベル数, ラベル数), start: (ラベル数,) の対数スコア"""
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
                [0.3, 0.1, 0.6],       # が (「蛾」かもしれない)
                [0.9, 0.05, 0.05],     # 魚
                [0.1, 0.1, 0.8],       # を
                [0.1, 0.85, 0.05]])    # 食べる

path, score = viterbi(emission, transition, start)
print([f"{w}/{labels[i]}" for w, i in zip(words, path)], round(score, 3))
```

```python
# 貪欲法 (各位置で独立に最大のラベルを選ぶ) との比較、および総当たりとの一致の確認
from itertools import product

greedy = emission.argmax(axis=1).tolist()
def total(p):
    return start[p[0]] + emission[0, p[0]] + sum(transition[a, b] + emission[t, b] for t, (a, b) in enumerate(zip(p, p[1:]), 1))
brute = max(product(range(3), repeat=5), key=total)
print("貪欲法:", [labels[i] for i in greedy], " 総当たり:", [labels[i] for i in brute], " 一致:", list(brute) == path)
```

- 計算量は O(系列長 × ラベル数²)。総当たりの O(ラベル数^系列長) を動的計画法で避けている。
- 遷移スコアに「O → I- は −∞」のような制約を入れれば、ニューラルモデルの出力に対しても正しい BIO 系列だけを出力させられる。
