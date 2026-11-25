# 日本語の形態素解析と文分割

MeCab (fugashi)・Sudachi・Janome・Juman++・KyTea で単語に区切り、品詞や原形、読みを取り出す。FLR や C-value による専門用語の候補の抽出、NEologd や MeCab・Sudachi・Janome へのユーザー辞書の追加、解析器の比べ方、KyTea のモデルの学習による分野への適応、pyopenjtalk による発音・モーラ数・アクセント、文への分割と句読点の復元も扱う。単語分割や品詞タグ付けの仕組み（系列ラベリング、HMM、ビタビアルゴリズム）を自分で実装するレシピは、付録の「仕組みを理解するための実装」にある。日本語の形態素解析や辞書の構築・運用については、姉妹編『[自然言語処理の教科書](https://github.com/mamorlis/nlpbook)』の第 3 章・第 5 章も参照。

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

- `unidic-lite` は軽量版の UniDic。最新のフル版は `pip install unidic && python -m unidic download`（約 1GB）。
- UniDic は短単位なので「雪国」「であった」などが細かく切れる。長めの単位が欲しければ Sudachi の C モードか IPA 辞書を使う。
- `lemma` は語彙素（「だ」「有る」「トンネル-tunnel」のように代表表記になる）。活用を戻しただけの形が欲しければ `orthBase` を使う。

## 分かち書きする {wakati}

文をスペース区切りの単語列にしたい（word2vec の学習や BM25 の前処理など）。

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

助詞や記号を捨てて、名詞・動詞・形容詞の原形だけを取り出したい（頻度集計やトピックモデルの前処理）。

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

## 漢字の読みを取得してふりがな（ルビ）を付ける {furigana}

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

- 送り仮名ごとルビが付く（「無い」全体に「ない」）。漢字部分だけに付けたい場合は、表層と読みの末尾の共通ひらがなを削る。

## 動詞・形容詞の活用型と活用形を調べる {conjugation}

用言がどの活用形で出現しているかを知りたい（敬語・文体の分析、学習者作文の誤り分析など）。

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

## トークンの文字位置（オフセット）を取得する {token_offsets}

各単語が元の文の何文字目から何文字目にあるかを知りたい（アノテーションとの対応付け、ハイライト表示など）。

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

コーパス中の未知語（辞書に載っていない語）を洗い出して、辞書追加の候補にしたい。

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

- `*` が未知語。未知語でなくても誤って細かく分割されている語（「バズった」「エモい」「インスタ」）は検出できない点に注意。

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

- 専門用語抽出の最も単純なベースライン。頻度や連接の多様性でスコア付けすると用語抽出らしくなる（この章の「専門用語の候補を FLR と C-value でスコア付けして抽出する」）。

## 専門用語の候補を FLR と C-value でスコア付けして抽出する {term_extraction}

専門的な文書から、その分野の用語らしい語（単名詞と複合名詞）を、頻度だけでなく「どれだけ多様な語と結び付くか」や「より長い語の一部としてだけ出ていないか」も考えてスコア付けし、上位から取り出したい。名詞の連続を候補にして、中川・森の FLR と Frantzi らの C-value を自分で実装する。題材は青空文庫の中谷宇吉郎『雪』（1938 年）。

```python
!pip install -q fugashi unidic-lite
```

```python
import requests
from bs4 import BeautifulSoup

# 青空文庫の中谷宇吉郎『雪』の XHTML 版（Shift_JIS）。ルビ（読み）と入力者の注記を除いて本文だけにする
url = "https://www.aozora.gr.jp/cards/001569/files/52468_49669.html"
soup = BeautifulSoup(requests.get(url, timeout=60).content.decode("cp932"), "html.parser")
for tag in soup.select("rp, rt, span.notes"):
    tag.decompose()
text = soup.select_one("div.main_text").get_text()
lines = [l.strip() for l in text.splitlines() if l.strip()]
print(len(text), "文字、", len(lines), "行")
print(next(l for l in lines if len(l) > 50)[:100])
```

```python
from collections import Counter
from fugashi import Tagger

tagger = Tagger()
STOP = {"こと", "もの", "ところ"}   # unidic-lite では普通名詞-一般になる形式名詞

def is_part(w):
    """複合名詞を作る語か（名詞・接頭辞・接尾辞。数詞、「的」、副詞的・助数詞的な名詞は除く）"""
    f = w.feature
    return (f.pos1 in ("名詞", "接頭辞", "接尾辞") and f.pos2 not in ("数詞", "形状詞的")
            and f.pos3 not in ("副詞可能", "助数詞可能") and w.surface not in STOP)

def noun_seqs(line):
    """名詞の連続を (語, 品詞) のタプルで返す。先頭の接尾辞と末尾の接頭辞は外す"""
    seqs, buf = [], []
    for w in list(tagger(line)) + [None]:
        if w is not None and is_part(w):
            buf.append((w.surface, w.feature.pos1))
            continue
        while buf and buf[0][1] == "接尾辞":
            buf.pop(0)
        while buf and buf[-1][1] == "接頭辞":
            buf.pop()
        if buf:
            seqs.append(tuple(buf))
        buf = []
    return seqs

raw = [s for l in lines for s in noun_seqs(l)]
POS = {w: p for s in raw for w, p in s}
seqs = [tuple(w for w, _ in s) for s in raw]      # 出現ごとの候補（語のタプル）
freq = Counter(seqs)                               # 候補がそのままの形（より長い名詞連続の一部でなく）で出た回数
print("候補の出現:", len(seqs), " 異なり:", len(freq), " うち複合名詞:", sum(len(s) > 1 for s in freq))
```

```python
import math
from collections import defaultdict

# FLR（中川・森）: 単名詞 N の左右に連接する単名詞の異なり数 LN(N), RN(N) から、候補 CN = N1…NL の
#   LR(CN) = (Π (LN(Ni)+1)(RN(Ni)+1)) ^ (1/2L)、FLR(CN) = f(CN) × LR(CN)（f は単独での出現回数）
left, right = defaultdict(set), defaultdict(set)
for s in freq:
    for a, b in zip(s, s[1:]):
        right[a].add(b)
        left[b].add(a)

def lr(s):
    return math.prod((len(left[n]) + 1) * (len(right[n]) + 1) for n in s) ** (1 / (2 * len(s)))

flr = {s: freq[s] * lr(s) for s in freq}
for n in ["結晶", "雪", "研究"]:
    examples = [("".join(s), f) for s, f in freq.most_common() if n in s and len(s) > 1][:4]
    print(f"{n}: 左 {len(left[n])} 種、右 {len(right[n])} 種、LR {lr((n,)):.1f}、単独 {freq[(n,)]} 回、例 {examples}")

# C-value（Frantzi ら）: 2 語以上の候補 a について
#   C(a) = log2|a| × f(a)                          （a がより長い候補に含まれないとき）
#   C(a) = log2|a| × (f(a) − t(a) / c(a))          （t: より長い候補の中での出現回数、c: その長い候補の種類数）
# f(a) はより長い候補の一部としての出現も数える。長い候補から順に、単独の出現回数を部分列に配る
def spans(s):
    for i in range(len(s)):
        for j in range(i + 2, len(s) + 1):
            if (i, j) != (0, len(s)) and POS[s[i]] != "接尾辞" and POS[s[j - 1]] != "接頭辞":
                yield s[i:j]

sub = Counter()
for s in seqs:
    if len(s) > 1:
        sub[s] += 1
        sub.update(spans(s))
t, c, cvalue = Counter(), Counter(), {}
for a in sorted(sub, key=len, reverse=True):
    cvalue[a] = math.log2(len(a)) * (sub[a] - (t[a] / c[a] if c[a] else 0))
    for b in spans(a):
        t[b] += sub[a] - t[a]
        c[b] += 1
print("C-value を付けた候補:", len(cvalue))
```

```python
import pandas as pd

def top(score, k=20, multi=False):
    items = [s for s in sorted(score, key=score.get, reverse=True) if not multi or len(s) > 1]
    return [f"{''.join(s)} ({score[s]:.0f})" for s in items[:k]]

pd.DataFrame({"頻度（複合名詞）": top(freq, multi=True), "FLR": top(flr), "FLR（複合名詞）": top(flr, multi=True),
              "C-value": top(cvalue)}, index=range(1, 21))
```

- 候補は、この章の「連続する名詞をつなげて複合名詞を抽出する」と同じく名詞の連続（UniDic の短単位をつないだもの）。そこに「こと・もの」などの形式名詞、副詞的に使う名詞（時・上・ため）、助数詞的な名詞（度・年）、数詞、「的」を加えないという条件を足した。「雪の結晶」のように助詞をはさむ用語は取れない。
- FLR は、多くの異なる語と複合語を作る単名詞を重要な構成要素とみなし、その語を含む候補を上げる。「結晶」は左に 16 種、右に 19 種の語が付き（「側面結晶」「平板結晶」「結晶形」「結晶質」など）LR が 18.4、「研究」は左 3 種・右 9 種で 6.3 だった。『雪』では単名詞の「結晶」「雪」が飛び抜けて高く、複合名詞だけで見ると「水蒸気」「樹枝状」「角板」「雪華」「人工雪」が上位に来た。頻度に掛け算するので、頻度の高い一般語（「研究」「図」「写真」）も上位に残る。
- C-value は、長い候補ほど重み (log2|a|) を付け、より長い候補の一部として出た回数も数えたうえで、それを含む長い候補の種類数で割った分を割り引く。決まった長い候補の中にしか出ない部分列は低くなり、いろいろな長い候補に入る語は上がる（「六角板」「小角板」「角板状」に入る「角板」は、単独の頻度 34 より高い 45 になった）。log2 1 = 0 なので単名詞にはスコアが付かず、2 語以上の候補の順位付けに向く。『雪』では長い候補が少ないので、上位は複合名詞の頻度順に近く、3 語の「顕微鏡写真」「雪華図説」は頻度より上がった。
- どちらも正解のない教師なしの順位付けなので、上位を人が見て用語集や辞書の候補にする使い方が基本。分野の文書の頻度を一般の文書（新聞や Wikipedia など）の頻度と比べる方法（TF-IDF や対数尤度比）を組み合わせると、「研究」「写真」のような一般語を下げられる。取り出した用語を MeCab のユーザー辞書に入れるには、この章の「MeCab (fugashi + UniDic) にユーザー辞書を追加する」を参照。
- FLR と C-value をまとめて計算する Python 版の termextract もあるが、PyPI には登録されていない（検証時に `pip install termextract` では入らなかった）。ここでは定義どおりに数十行で実装した。FLR の LR は、ここでは連接の異なり数で数えた（延べ数で数える流儀もある）。
- 『雪』（中谷宇吉郎、1938 年）は著作権の保護期間が満了しており、青空文庫の本文は自由に使える（入力・校正は青空文庫のボランティア）。

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

- 品詞フィルタ（助詞・助動詞・記号を落とす）とストップワードリストの併用が基本。
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

漢字を含む単語リストを、文字コード順ではなく読みの五十音順でソートしたい（索引や用語集の作成）。

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
!apt-get install -y -q mecab libmecab-dev mecab-ipadic-utf8 > /dev/null
!pip install -q fugashi
```

```python
# Colab で 5 分ほどかかる（辞書のダウンロードとビルド。できる辞書は 900MB ほど）
!git clone -q --depth 1 https://github.com/neologd/mecab-ipadic-neologd
!cd mecab-ipadic-neologd && ./bin/install-mecab-ipadic-neologd -n -y -u > ../neologd_install.log 2>&1
# ログから、元の IPA 辞書をどこから取得したかと、インストール先を確かめる
!grep -E "Unable to access|Try to download|Make install to" neologd_install.log
# インストール先は mecab-config --dicdir の下の mecab-ipadic-neologd
!du -h --apparent-size $(mecab-config --dicdir)/mecab-ipadic-neologd/sys.dic
!echo "米津玄師の新曲" | mecab -d $(mecab-config --dicdir)/mecab-ipadic-neologd
```

```python
import subprocess
from fugashi import GenericTagger

dicdir = subprocess.run(["mecab-config", "--dicdir"], capture_output=True, text=True).stdout.strip()
NEOLOGD = f"{dicdir}/mecab-ipadic-neologd"
# fugashi は既定で /usr/local/etc/mecabrc を探すので、apt の MeCab の設定ファイルを -r で渡す
ipa = GenericTagger("-r /etc/mecabrc")                  # apt の IPA 辞書（mecabrc の既定の辞書）
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
# NEologd で固有名詞になった語の品詞（細分類）と読み。素性は IPA 辞書と同じ並び
for s in sents:
    for w in neo(s):
        f = w.feature
        if f[1] == "固有名詞":
            print(w.surface, ",".join(f[:4]), f[7] if len(f) > 7 else "(読みなし: 未知語)", sep="\t")
```

```python
# 収録されていない新しい語（2020 年 9 月より後に広まった語など）は、IPA 辞書と同じく切れるか未知語になる
for s in ["推しの子のアニメを観た。", "ChatGPTと生成AIの話をした。"]:
    for w in neo(s):
        f = w.feature
        print(w.surface, ",".join(f[:4]), f[7] if len(f) > 7 else "(読みなし: 未知語)", sep="\t")
    print()
```

```python
# 長い固有名詞が 1 語になると、その一部の語で検索したときに見つからなくなる（再現率が下がる）
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

# 対策の一例: 検索用の索引には両方の辞書のトークンを入れる（README も IPA 辞書との併用を勧めている）
index = [set(w.surface for w in ipa(d)) | set(w.surface for w in neo(d)) for d in docs]
for q in ["東京大学", "コロナ", "新型コロナウイルス感染症"]:
    print(f"両方    {q}: {[i for i, toks in enumerate(index) if q in toks]}")
```

- mecab-ipadic-NEologd は、はてなキーワード・郵便番号データ・駅名・ニュース記事・ウェブの文書などから集めた新語・固有表現を IPA 辞書 (mecab-ipadic-2.7.0-20070801) に追加したシステム辞書（佐藤敏紀氏ら、Apache License 2.0）。IPA 辞書と同じ素性の並びで読みも付く。apt の mecab-ipadic-utf8 と pip の `ipadic`（前のレシピ「IPA 辞書で MeCab を使う」）は同じ版の IPA 辞書。NEologd は IPA 辞書の語の読みや生起コストにもパッチを当てるので、IPA 辞書との差がすべて新語によるものとは限らない。
- **更新は止まっている。** README には「少なくとも毎週 2 回更新」とあるが、master の最後のコミットは 2020 年 9 月 14 日（シードのデータは 2020-09-10 版）で、それ以降の語は入っていない（2026 年 11 月時点で、リポジトリはアーカイブされていない）。上の例では、収録されていない「推しの子」が「推し / の子」に切れて「の子」が人名になり、「ChatGPT」は読みのない未知語になる。
- 固有名詞の細分類は粗く、誤りもある（上の例では「ほぼ日刊イトイ新聞」が「人名」）。README の「欠点」にも、分類が不十分なこと、固有表現ではない語も固有表現として登録されていること、読みの誤りがあることが書かれている。入力は、NEologd の正規化ルールを実装した neologdn（文字列処理と正規化の章の「neologdn で日本語の表記ゆれを正規化する」）で正規化してから解析するとよい。
- 長い固有名詞が 1 語になるので、トークン単位で「東京大学」「コロナ」と検索しても「東京大学大学院情報理工学系研究科」「新型コロナウイルス感染症」は見つからない。README も、ウェブの文書の解析では標準の IPA 辞書との併用を勧めている。単語の単位を評価データや学習済みモデルに合わせる必要がある場合は使わない。
- インストーラーは元の IPA 辞書の tar.gz を自分でダウンロードし（SHA-1 で照合）、パッチを当ててビルドする。検証時（2026 年 11 月）は、最初のダウンロード先 osdn.net が名前解決できず、次の Google ドライブからも正しいファイルが得られず（SHA-1 が合わない）、3 番目の SourceForge から取得された（上の出力の `Try to download from` の行）。`-n` は GitHub の最新版に更新してからインストール、`-y` はテスト結果の確認を省略、`-u` はインストール先が自分のディレクトリでなくても sudo を使わずに `make install` する（付けないと Colab では `sudo make install` を通る）。インストール先は `mecab-config --dicdir` の下（Colab では `/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd`）で、`-p ディレクトリ` で変えられる。
- README の目安では、標準のインストールに空きメモリが最低 2GB（推奨 6GB）必要。手元の検証では、ビルド中の最大のプロセスのメモリ使用量（最大常駐サイズ）は約 1.6GB、できた `sys.dic` は上の出力のとおり 866M（README では約 910MB）だった。`--ignore_adverb` などの `--ignore_*` で一部の辞書を外すと小さくなり（README によれば全部指定して約 300MB 減る）、`-a` では標準では入らない低頻度の日時・数量表現などもすべて入る。Colab のランタイムを作り直すとインストールはやり直しになる。

## MeCab (fugashi + UniDic) にユーザー辞書を追加する {mecab_userdic}

UniDic で細かく切れてしまう専門用語や固有名詞を、MeCab のユーザー辞書に登録して 1 語にしたい。CSV に表層形・左右文脈 ID・生起コスト・品詞・読みを書き、`mecab-dict-index` に当たるコンパイラで `user.dic` にして `Tagger('-u user.dic')` で読み込む。fugashi の wheel にはコンパイラが同梱されているので、apt で MeCab を入れなくてよい。

```python
!pip install -q fugashi unidic-lite
```

```python
from fugashi import Tagger

tagger = Tagger()
# 各語の左文脈 ID (%phl)・右文脈 ID (%phr)・生起コスト (%pw) と品詞を出す出力形式
info = Tagger(r'-O "" -F "%m\t%phl\t%phr\t%pw\t%f[0],%f[1],%f[2]\n" -U "%m\t%phl\t%phr\t%pw\t未知語\n"')

sents = ["大規模言語モデルで検索拡張生成を試す。", "外国人参政権について話し合った。",
         "形態素解析にフガシを使う。", "都立大に行く。"]
for s in sents:
    print(" | ".join(w.surface for w in tagger(s)))
print()
print(info.parse("大規模言語モデルを東大で試す"))
```

```python
import csv, subprocess, sys
import unidic_lite

def ref_entry(surface, kana, ref, cost=None):
    """似た既存語 ref から左右文脈 ID・生起コスト・品詞を借りて、ユーザー辞書の CSV の 1 行を作る"""
    lid, rid, ref_cost = info.parse(ref).split("\n")[0].split("\t")[1:4]
    f = tagger(ref)[0].feature
    # unidic-lite の素性は 26 列（dicrc の先頭に一覧がある）。品詞・活用・語種は ref のものを使い、表記と読みを差し替える。
    # 語彙素読み・発音・アクセントなどは簡単のため読みか * で埋める
    feats = [f.pos1, f.pos2, f.pos3, f.pos4, f.cType, f.cForm,
             kana, surface, surface, kana, surface, kana, f.goshu, "*", "*", "*", "*",
             kana, kana, kana, kana, "*", "*", "*", "*", "*"]
    return [surface, lid, rid, str(ref_cost if cost is None else cost)] + feats

def build_userdic(rows, name):
    """CSV を書き、fugashi 同梱のコンパイラ（mecab-dict-index と同じもの）で name.dic にする"""
    with open(f"{name}.csv", "w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerows(rows)
    # build_dictionary は失敗するとプロセスごと終了するので、カーネルを巻き込まないよう別プロセスで呼ぶ
    code = "import sys, fugashi; fugashi.build_dictionary('mecab-dict-index ' + ' '.join(sys.argv[1:]))"
    r = subprocess.run([sys.executable, "-c", code, "-d", unidic_lite.DICDIR, "-u", f"{name}.dic",
                        "-f", "utf-8", "-t", "utf-8", f"{name}.csv"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return Tagger(f"-u {name}.dic")

entries = [  # (表層形, 読み, 品詞・ID・コストを借りる既存語)
    ("大規模言語モデル", "ダイキボゲンゴモデル", "言語"),
    ("検索拡張生成", "ケンサクカクチョウセイセイ", "生成"),
    ("参政権", "サンセイケン", "政権"),
    ("フガシ", "フガシ", "グーグル"),
    ("都立大", "トリツダイ", "東大"),
]
rows = [ref_entry(*e) for e in entries]
for row in rows:
    print(",".join(row[:10]), "...", f"({len(row)} 列)")
user = build_userdic(rows, "user")
print()
for s in sents:
    print("前:", " | ".join(w.surface for w in tagger(s)))
    print("後:", " | ".join(w.surface for w in user(s)))
print()
for w in user("形態素解析にフガシを使う。"):
    f = w.feature
    print(w.surface, f.pos1, f.pos2, f.pos3, f.kana, w.is_unk, sep="\t")
```

```python
# 「都立大」の生起コストだけを変えて作り直し、ほかの文の分割が崩れないかを見る（コストが小さいほど 1 語として選ばれやすい）
checks = ["都立大に行く。", "都立大の学生", "都立大学に行く。", "東京都立大学に行く。"]
ref_cost = int(ref_entry("都立大", "トリツダイ", "東大")[3])
for cost in [0, ref_cost, 11000, 14000]:
    t = build_userdic([ref_entry("都立大", "トリツダイ", "東大", cost)], f"toritsu_{cost}")
    print(f"{cost:>6}", " / ".join(" ".join(w.surface for w in t(s)) for s in checks))
```

```python
# 「都立大」のコストを 11000 にして辞書を作り直し、期待どおりに分割されるかを単体テスト（回帰テスト）で確かめる
rows = [ref_entry(*e) for e in entries[:-1]] + [ref_entry("都立大", "トリツダイ", "東大", 11000)]
user = build_userdic(rows, "user")
cases = [
    ("大規模言語モデルを作る", ["大規模言語モデル", "を", "作る"]),
    ("検索拡張生成で答える", ["検索拡張生成", "で", "答える"]),
    ("外国人参政権の議論", ["外国", "人", "参政権", "の", "議論"]),
    ("フガシで解析する", ["フガシ", "で", "解析", "する"]),
    ("都立大に行く", ["都立大", "に", "行く"]),
    ("都立大学に行く", ["都立", "大学", "に", "行く"]),          # 登録した語が他の語の一部を奪わない
    ("東京都立大学の言語モデル", ["東京", "都立", "大学", "の", "言語", "モデル"]),
]
for sent, expected in cases:
    got = [w.surface for w in user(sent)]
    assert got == expected, f"{sent}: {got}"
print(f"単体テスト: {len(cases)}/{len(cases)} 件合格")
```

- MeCab のユーザー辞書の CSV は「表層形,左文脈 ID,右文脈 ID,生起コスト,素性…」の形で、素性の列はシステム辞書と同じ並び（unidic-lite は 26 列、IPA 辞書は 9 列）にする。左右文脈 ID は前後の語とのつながりやすさ（連接コスト）を引くための番号で、UniDic では品詞だけでなく語種でも分かれる（上の出力では、漢語の普通名詞「言語」が 5146、外来語の「モデル」が 5144、固有名詞の「東大」が 4786）。ID は辞書ごとに違うので、同じ品詞・語種の既存語から借りるのが確実で、システム辞書を変えたら取り直して作り直す。ID の列を空にすると、`mecab-dict-index` が辞書の `left-id.def`・`right-id.def`・`rewrite.def` を使って素性から自動で付ける（手元で試すと、上と同じ 4786 や 5146 が付いた）。
- 生起コストは小さいほどその語が選ばれやすい。長い複合語は、元の分割の語のコストと連接コストの合計に勝てばよいので、似た語のコストを借りれば多くの場合 1 語になる（「大規模言語モデル」「検索拡張生成」「参政権」）。一方、短い語はほかの語の一部を奪いやすい。上の例では、「東大」から借りたコスト（8591）でも「都立大学」が「都立大 / 学」に、0 では「東京都立大学」まで「東京 / 都立大 / 学」に崩れ、14000 では「都立大」自体が 1 語にならなかった。11000 なら 4 文とも期待どおりで、このように「変わってほしい文」と「変わってほしくない文」を並べて、両方を満たすコストを探す。`mecab-dict-index` には学習済みのモデルからコストを推定する `-m`・`-a` もあるが、unidic-lite にはモデルのファイルが含まれていない。
- 「都立大学生」のような文は「都立大 / 学生」とも「都立大学 / 生」とも読めて、辞書だけでは決まらない（手元で試すと、コスト 11000 では前者、12000 では後者になった）。語を足したら、ここの単体テストのように期待する分割を並べた文で回帰テストを続ける。
- `fugashi.build_dictionary` は fugashi の wheel に同梱の libmecab の `mecab-dict-index` をそのまま呼ぶ関数で、コマンドの `fugashi-build-dict` からも使える（引数は `mecab-dict-index` と同じ）。エラー時には Python の例外にならずプロセスが終わるので、ノートブックでは別プロセスで呼んでいる。apt の `mecab-utils` や MeCab のソースからのビルドは要らない（検証はコンテナの中で root にならずに行った）。
- 使い分け: 数語〜数千語を UniDic の単位と品詞体系のまま足したいならこのレシピ。表記ゆれも正規化形でまとめたいなら、この章の「Sudachi にユーザー辞書をコンパイルして追加する」（18 列の CSV と `sudachipy ubuild`、分割の単位 A/B/C も指定できる）。コンパイルせずに手早く試すなら「Janome にユーザー辞書を追加する」（表層形・品詞・読みの 3 列の簡易辞書）。ウェブ上の新語・固有名詞を大量にまとめて入れたいなら「NEologd 辞書で新語・固有名詞をまとまりとして解析する」（IPA 辞書に追加したシステム辞書。2020 年で更新が止まっている）。登録する候補は、この章の「辞書にない未知語を見つける」や「専門用語の候補を FLR と C-value でスコア付けして抽出する」で集められる。
- unidic-lite（UniDic 2.1.2）のライセンスは BSD・LGPL・GPL のいずれか。fugashi は MIT ライセンス。

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
- ユーザー辞書を作って表記ゆれや専門用語を登録する方法は、次の「Sudachi にユーザー辞書をコンパイルして追加する」を参照。

## Sudachi にユーザー辞書をコンパイルして追加する {sudachi_userdic}

システム辞書で細かく分割されてしまう専門用語や、カタカナ表記から英語表記へ揃えたい表記ゆれを、ユーザー辞書（18 列の CSV）に登録して `sudachipy ubuild` でコンパイルし、分割と正規化形を単体テストで確かめたい。

```python
!pip install -q sudachipy sudachidict_core
```

```python
from sudachipy import Dictionary, SplitMode

sys_tokenizer = Dictionary().create()
test_sentences = [
    "自然言語処理では検索拡張生成やラベルバイアス問題が話題になる。",
    "形態素解析にスダチパイやフガシを使う。",
    "翻訳の品質推定にコメットキューイーを用いる。",
]

def show_tokens(tok, sentences):
    for s in sentences:
        items = [f"{m.surface()}({m.part_of_speech()[0]}-{m.part_of_speech()[1]}|{m.normalized_form()})"
                 for m in tok.tokenize(s, SplitMode.C)]
        print("入力:", s)
        print("  →", " / ".join(items))

print("=== システム辞書のみ（登録前） ===")
show_tokens(sys_tokenizer, test_sentences)
```

```python
import json, shutil, subprocess, sys, tempfile
from pathlib import Path
import sudachidict_core, sudachipy

# （表層形、品詞大分類、品詞中分類、読み、正規化形）
user_entries = [
    ("ラベルバイアス問題", "名詞", "普通名詞", "ラベルバイアスモンダイ", "ラベルバイアス問題"),
    ("検索拡張生成", "名詞", "普通名詞", "ケンサクカクチョウセイセイ", "検索拡張生成"),
    ("スダチパイ", "名詞", "固有名詞", "スダチパイ", "SudachiPy"),
    ("フガシ", "名詞", "固有名詞", "フガシ", "fugashi"),
    ("コメットキューイー", "名詞", "固有名詞", "コメットキューイー", "COMET-QE"),
]

# Sudachi ユーザー辞書の 18 列 CSV 形式を作成して sudachipy ubuild で user.dic にコンパイルする
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    csv_path, dic_path, cfg_path = work / "user.csv", work / "user.dic", work / "sudachi.json"
    rows = [
        # 左連接ID, 右連接ID は名詞の標準的な値 (4785)、コストは既存語より優先されるよう小さめの値 (4000) にする
        ",".join([surf, "4785", "4785", "4000", surf, pos1, pos2, "一般", "*", "*", "*", rd, norm, "*", "A", "*", "*", "*"])
        for surf, pos1, pos2, rd, norm in user_entries
    ]
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    sys_dic = Path(sudachidict_core.__file__).parent / "resources" / "system.dic"
    cand_bin = Path(sys.executable).parent / "sudachipy"
    sudachipy_bin = str(cand_bin) if cand_bin.exists() else (shutil.which("sudachipy") or "sudachipy")
    subprocess.run([sudachipy_bin, "ubuild", "-s", str(sys_dic), "-o", str(dic_path), str(csv_path)], check=True)

    # 既定の sudachi.json を基に systemDict と userDict を指定する
    base_cfg = json.loads((Path(sudachipy.__file__).parent / "resources" / "sudachi.json").read_text(encoding="utf-8"))
    base_cfg["systemDict"] = str(sys_dic)
    base_cfg["userDict"] = [str(dic_path)]
    if "characterDefinitionFile" in base_cfg and not Path(base_cfg["characterDefinitionFile"]).is_absolute():
        base_cfg["characterDefinitionFile"] = str(Path(sudachipy.__file__).parent / "resources" / base_cfg["characterDefinitionFile"])
    cfg_path.write_text(json.dumps(base_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    user_tokenizer = Dictionary(config_path=str(cfg_path)).create()
    print("=== ユーザー辞書適用後 ===")
    show_tokens(user_tokenizer, test_sentences)

    # 単体テスト（回帰テスト）: 期待どおりの分割と正規化形になっているか確認する
    cases = [
        ("ラベルバイアス問題を学ぶ", "ラベルバイアス問題", "ラベルバイアス問題"),
        ("検索拡張生成の精度を上げる", "検索拡張生成", "検索拡張生成"),
        ("形態素解析にスダチパイを使う", "スダチパイ", "SudachiPy"),
        ("ラッパーのフガシを試す", "フガシ", "fugashi"),
        ("コメットキューイーで評価する", "コメットキューイー", "COMET-QE"),
    ]
    for sent, expected_surface, expected_norm in cases:
        matched = [m for m in user_tokenizer.tokenize(sent, SplitMode.C) if m.surface() == expected_surface]
        assert matched, f"分割失敗: {sent!r} に {expected_surface!r} が見つかりません"
        assert matched[0].normalized_form() == expected_norm, f"正規化不一致: {matched[0].normalized_form()}"
    print(f"\n単体テスト: {len(cases)}/{len(cases)} 件合格")
```

- Sudachi のユーザー辞書は 18 列の CSV で表層形・連接コスト ID・生起コスト・品詞・読み・正規化形・分割タイプ（`A`/`B`/`C` のどの単位で登録するか）などを指定し、`sudachipy ubuild` でバイナリの `user.dic` に変換してから設定 JSON の `userDict` に登録する。
- 名詞として追加する場合は左・右連接 ID に標準的な名詞の ID (`4785`) を流用でき、生起コストを小さめ（`4000` など）にするとシステム辞書の細かな分割（「ラベル / バイアス / 問題」「検索 / 拡張 / 生成」）より 1 語としての分割が選ばれやすくなる。
- 表記ゆれ（「スダチパイ」→ `SudachiPy`、「フガシ」→ `fugashi`）を正規化形欄に書いておくと、`m.normalized_form()` で表記を統一できる。辞書に語を足したときは、想定外の文で誤分割が起きないかも含めて単体テストで継続的に検証する。
- 設定不要で手早く簡易 CSV を試したい場合は、この章の「Janome にユーザー辞書を追加する」も使える。

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

MeCab（UniDic / IPA 辞書）、Sudachi、Janome で同じ文がどう分割されるかを見比べたい。

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

## Juman++ で形態素解析する（代表表記・カテゴリ付き） {jumanpp}

単語の区切りと品詞に加えて、「りんご/林檎/リンゴ」をまとめる代表表記や、「人」「場所」のような意味カテゴリも欲しい。京都大学の研究室が開発した、RNN 言語モデルで単語列のもっともらしさを評価する形態素解析器 Juman++ を使う。

```python
# ダウンロード（約 300MB）・展開・ビルドで、Colab（2 コア）では 10 分ほどかかる見込み
!wget -q https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc4/jumanpp-2.0.0-rc4.tar.xz
!tar xf jumanpp-2.0.0-rc4.tar.xz
!cd jumanpp-2.0.0-rc4 && mkdir -p bld && cd bld && cmake .. -DCMAKE_BUILD_TYPE=Release -DJPP_ENABLE_TESTS=OFF -DCMAKE_INSTALL_PREFIX=/usr/local > /dev/null 2>&1 && make install -j2 > /dev/null 2>&1
!jumanpp --version
```

```python
!pip install -q rhoknp==1.8.2 fugashi unidic-lite
```

```python
import subprocess

# 生の出力（JUMAN 形式）: 表層形 読み 原形 品詞 品詞ID 細分類 ID 活用型 ID 活用形 ID 意味情報
r = subprocess.run(["jumanpp"], input="外国人参政権について話し合った。\n", capture_output=True, text=True)
print(r.stdout)
```

```python
from rhoknp import Jumanpp

# skip_sanity_check=True: 起動時の確認（空の文を送る）を省く（下の注意書きを参照）
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

# 生の出力の @ 行（同形異義語）は homographs に入る
for m in jumanpp.apply_to_sentence("外国人参政権について話し合った。").morphemes:
    if m.homographs:
        print(m.text, m.canon, [h.canon for h in m.homographs])
```

```python
# 代表表記 (canon) で表記ゆれをまとめる。代表表記がない語（自動獲得語など）は canon が None なので表層形で代用する
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
# MeCab + UniDic (fugashi) と、分割と品詞を比べる（例文はこの章の「解析器・辞書による分割の違いを比べる」と同じ）
from fugashi import Tagger

unidic = Tagger()
for s in ["外国人参政権について話し合った。", "東京都立大学で機械翻訳を研究しています。", "食べさせられたくなかった。"]:
    print(s)
    print("  Juman++     ", " | ".join(f"{m.text}/{m.pos}" for m in jumanpp.apply_to_sentence(s).morphemes))
    print("  MeCab+UniDic", " | ".join(f"{w.surface}/{w.feature.pos1}" for w in unidic(s)))
```

- 品詞体系と辞書は JUMAN のもので、UniDic とは品詞名も単位も違う。「であった」は判定詞 1 語、「させ」「られ」「たく」「なかった」は接尾辞になる（UniDic では助動詞や形容詞で、「なかっ/た」と切れる）。構文解析の章の「KWJA で文節・基本句の係り受けと並列構造を解析する」の KWJA も同じ品詞体系で、結果は rhoknp で同じように扱える。
- 「外国人参政権」は、Juman++ では「外国/人/参政/権」、MeCab + UniDic では「外国/人参/政権」になった。Juman++ は単語列の意味的なもっともらしさを RNN 言語モデルで考慮する（README から張られたウェブデモもこの文を例にしている）。比べた 3 文は、この章の「解析器・辞書による分割の違いを比べる」と同じ例文なので、Sudachi や Janome の結果とも見比べられる。
- 意味情報には、代表表記、カテゴリ（人、場所-施設、抽象物など）、ドメイン（政治、料理・食事など）、自他動詞や反義語の対応が入る。代表表記は「Sudachi の正規化形で表記ゆれを吸収する」の正規化形と同じ目的に使え、「りんご/林檎/リンゴ」は「林檎/りんご」に、「子ども/子供/こども」は「子供/こども」にまとまった。
- 生の出力の `@` で始まる行は、読みも品詞も同じで代表表記だけが違う候補（同形異義語）。「スマホ」のような自動獲得語は代表表記がなく（`canon` が None）、意味情報に「Wikipediaリダイレクト:スマートフォン」が入った。
- Juman++ (v2.0.0-rc4) は、最初の入力行が空行だと異常終了（セグメンテーション違反）することがある（毎回ではない）。rhoknp (1.8.2) の `Jumanpp()` は起動時の確認で空の文を送るので、異常終了した回は「failed to start Juman++」という警告が出る（解析は再起動したプロセスで行われる）。`skip_sanity_check=True` で確認を省いたときは、最初に空文字列を渡さない。subprocess で直接渡すときも先頭に空行を置かない。
- 配布物（約 300MB）に学習済みのモデルが含まれるので、GitHub のソースではなくリリースの tar.xz を使う。ライセンスは Apache 2.0。最新版は 2023 年の v2.0.0-rc4。

## KyTea で単語分割と読みの推定をする {kytea}

辞書の照合ではなく、文字の並びから「ここで単語が切れるか」を判定する解析器で、単語分割と品詞・読みを得たい。辞書にない語の読みも推定できる KyTea（京都テキスト解析ツールキット）を使う。

```python
!pip install -q kytea fugashi unidic-lite
```

```python
# 学習済みの日本語モデル（gzip で約 38MB、展開後 約 120MB）はソースのリポジトリに同梱されている。pip のパッケージには含まれない
!wget -q https://github.com/neubig/kytea/raw/master/data/model.bin.gz && gunzip -f model.bin.gz && du -h model.bin
```

```python
import Mykytea

kytea = Mykytea.Mykytea("-model model.bin")   # オプションはコマンド版の kytea と同じ書き方で渡す
s = "太郎は図書館で借りた本を花子に渡した。"
print(" ".join(kytea.getWS(s)))      # 単語分割だけ
print(kytea.getTagsToString(s))      # 単語/品詞/読み

# getTags は単語ごとに [品詞の候補, 読みの候補] を返す（各候補は (タグ, スコア)）
for w in kytea.getTags(s):
    (pos, pos_score), = w.tag[0]
    (yomi, yomi_score), = w.tag[1]
    print(w.surface, pos, yomi, round(pos_score, 2), round(yomi_score, 2), sep="\t")
```

```python
# getAllTags は 2 位以下の候補とスコアも返す（-tagmax で候補数を変えられる。既定は 3）。読みは文脈で変わる
for s in ["日本語の本を今日読んだ。", "今日は日本語の本を読んだ。"]:
    for w in kytea.getAllTags(s):
        if w.surface == "日本":
            print(s, "日本 の読みの候補:", [(t, round(sc, 2)) for t, sc in w.tag[1]])

# -notags は分割だけ、-notag 1 は 1 番目のタグ（品詞）を推定せず読みだけを付ける
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
# UniDic (unidic-lite) で未知語になったり細かく分かれたりする語を含む文（最後の文はこの章の「辞書にない未知語を見つける」と同じ）。
# UniDic は未知語に読みを付けられないが、KyTea は自分の辞書にない語（「鬼滅」「ジーピーティー」）も文字の並びから区切りと読みを推定する
for s in ["鬼滅の刃の竈門炭治郎が好きだ。", "チャットジーピーティーに聞いた。", "彼はスマホをポチポチいじっていた。", "推し活でエモい写真をインスタにあげた。"]:
    print("KyTea  :", kytea.getTagsToString(s))
    print("UniDic :", " ".join(f"{w.surface}/{'(未知語)' if w.is_unk else w.feature.kana or w.surface}" for w in tagger(s)))
```

```python
# 速度の比較（10,000 文）
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

- KyTea は Graham Neubig らが開発した解析器。文字と文字の間ごとに単語境界かどうかを、周りの文字 n-gram・文字種・辞書との一致を素性とする線形分類器（線形 SVM やロジスティック回帰。学習は LIBLINEAR）で 1 か所ずつ独立に判定し、単語の品詞と読みも個別に推定する（点予測）。付録「仕組みを理解するための実装」の「単語分割を文字単位の系列ラベリングとして解く」の CRF はラベル列全体をまとめて決めるが、KyTea は隣の判定結果を使わないので、一部の境界や読みだけを付けた「部分的アノテーション」のデータでも学習できる。
- 同梱のモデルは現代日本語書き言葉均衡コーパス (BCCWJ) や UniDic などで学習されたもので、活用語尾を語幹と別の単語にする（「渡 し た」「読 ん だ」「た く な かっ た」）。MeCab + UniDic の短単位（「読ん」「たく」「なかっ」）より細かい。構文解析の章の「Ckylark で日本語の句構造解析をする」の日本語モデルは、この分割を前提にしている。
- 読みはひらがなで、単語の表記に対応する部分だけが付く（「渡」→「わた」）。送り仮名が別の単語になるので、この章の「漢字の読みを取得してふりがな（ルビ）を付ける」（送り仮名ごとルビが付く）と違い、漢字の部分だけにルビを付けやすい。辞書で読みの候補が 1 つしかない語はスコアが 100 になり、「日本」のように候補が複数ある語は文脈で結果が変わる（「にほん」と「にっぽん」）。
- 辞書にない語の推定は誤ることもある（「推し活」は「推（お）/ し活」に分割された）。Python のバインディングからは未知語かどうかは分からない。コマンド版の `kytea -unktag '[UNK]'` で同じ文を試すと、「鬼滅」「ジーピーティー」「し活」「エモ」「ポチポチ」に印が付き、「スマホ」「インスタ」は KyTea の辞書にある語だった。
- 検証環境で 10,000 文にかかった時間は、品詞・読みありが 0.8〜0.9 秒、分割だけ (`-notags`) が約 0.15 秒、MeCab + UniDic が 0.1 秒ほど。モデルの読み込みに約 2 秒かかるので、`Mykytea` は 1 回だけ作って使い回す。
- pip の `kytea` (Mykytea-python) は KyTea のライブラリを含むのでビルドは不要だが、モデルは含まない。モデルは研究・商用での利用は許可されているが無断の再配布は禁じられているので、各自が作者のリポジトリから取得する。コマンド版や学習用の `train-kytea`（`-part` で部分的アノテーション、`-dict` で辞書を与える）が要るなら、構文解析の章の「Ckylark で日本語の句構造解析をする」のようにソースからビルドする。ソースは Apache License 2.0 で、最後のリリースは 0.4.7（2014 年）、リポジトリの最後の更新は 2020 年。

## KyTea の単語分割モデルを学習して分野に合わせる（辞書・部分的アノテーション） {kytea_train}

医療のような専門分野の文章で、薬剤名や交ぜ書きの用語（「せん妄」「じょく瘡」）の分割を誤る。人手で分割したコーパスから単語分割のモデルを自分で学習し、分野の語の辞書と、誤りやすい箇所だけに境界を付けたデータを加えて直したい。KyTea の学習ツール train-kytea を使う。

```python
!pip install -q kytea==0.2.0 conllu==6.1.1
```

```python
# pip の kytea には学習ツールが入っていないので、KyTea 0.4.7 のリリース（約 39MB。学習済みの日本語モデルも入っている）をビルドする。
# リリースは configure 済みなので autoreconf も apt-get も要らない。--disable-shared で静的にリンクし、make install せずに使う。
# Colab（2 コア）では、ダウンロードとビルドで 3 分ほどの見込み
!wget -q https://www.phontron.com/kytea/download/kytea-0.4.7.tar.gz && tar xzf kytea-0.4.7.tar.gz
# ビルドのログは make.log に書き、失敗したときだけ末尾を表示する
!cd kytea-0.4.7 && ./configure -q --disable-shared > /dev/null && (make -j2 > make.log 2>&1 && echo "ビルド完了" || tail -20 make.log)
!kytea-0.4.7/src/bin/train-kytea --help 2>&1 | grep -E "^  -(tok|part|dict|notags) "
```

```python
import subprocess
import urllib.request
import Mykytea
from conllu import parse

TRAIN = "kytea-0.4.7/src/bin/train-kytea"

# UD Japanese-GSD（CC BY-SA 4.0、https://github.com/UniversalDependencies/UD_Japanese-GSD）。人手で短単位に分割されている
# 数値を再現できるように、リリース r2.19 のタグに固定する（master は UD のリリースごとに変わる）
base = "https://raw.githubusercontent.com/UniversalDependencies/UD_Japanese-GSD/r2.19/ja_gsd-ud-"
def load(split):
    sents = parse(urllib.request.urlopen(base + split + ".conllu").read().decode("utf-8"))
    return [[t["form"] for t in s if isinstance(t["id"], int)] for s in sents]   # 複数語トークンなどの行を除く

def ok(w):
    return not set(w) & set("/& \\")   # KyTea の区切り記号（タグの "/"、候補の "&"）や空白を含む語は使わない

train, test = load("train"), load("test")
train = [s for s in train if all(map(ok, s))]
print(len(train), "文", sum(map(len, train)), "語（学習）/", len(test), "文（評価）")

# 学習データは 1 行 1 文で、単語を空白で区切る（-tok の形式。品詞や読みは付けない）
with open("gsd.tok", "w", encoding="utf-8") as f:
    f.writelines(" ".join(s) + "\n" for s in train)

def train_kytea(model, *args):
    """train-kytea で単語分割だけのモデルを学習する（-notags: 品詞・読みは学習しない）"""
    r = subprocess.run([TRAIN, "-tok", "gsd.tok", *args, "-notags", "-model", model], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return Mykytea.Mykytea(f"-model {model} -notags")

base_model = train_kytea("gsd.model")   # データのダウンロードと学習で、Colab で 2 分ほどの見込み
default = Mykytea.Mykytea("-model kytea-0.4.7/data/model.bin -notags")   # KyTea に同梱のモデル
```

```python
def spans(words):
    out, p = set(), 0
    for w in words:
        out.add((p, p + len(w))); p += len(w)
    return out

def word_f1(gold_sents, pred_sents):
    """単語単位の F1: 開始位置と終了位置が両方一致した単語だけを正解とする"""
    tp = n_gold = n_pred = 0
    for gold, pred in zip(gold_sents, pred_sents):
        g, p = spans(gold), spans(pred)
        tp += len(g & p); n_gold += len(g); n_pred += len(p)
    prec, rec = tp / n_pred, tp / n_gold
    return round(2 * prec * rec / (prec + rec), 4)

def segment_all(kytea, sents):
    return [list(kytea.getWS("".join(s))) for s in sents]

print("GSD の評価データの単語 F1")
print("  同梱のモデル  :", word_f1(test, segment_all(default, test)))
print("  GSD で学習    :", word_f1(test, segment_all(base_model, test)))
for s in ["新しい産業構造を作らなければいけない。", "しばらくの完全休養とリハビリが必要とのことでした。"]:
    print("同梱:", " ".join(default.getWS(s)))
    print("GSD :", " ".join(base_model.getWS(s)))
```

```python
import re

# 分野の語の一覧（ここでは医療の語。実務では用語集や薬剤の一覧から作る）
terms = """せん妄 じょく瘡 誤えん 膿瘍 てんかん 嚥下 褥瘡 喀痰 耐性 術後
アセトアミノフェン ロキソプロフェン バラシクロビル レベチラセタム セフトリアキソン クロピドグレル
ビソプロロール ファモチジン メトホルミン プレドニゾロン サルブタモール レボフロキサシン アモキシシリン""".split()

# 分野の文（分割していない生の文）。部分的アノテーションでは、用語の前後と内部の境界だけを付ける
domain_raw = """高齢者はせん妄を起こしやすい。
せん妄の原因として脱水が考えられる。
じょく瘡の深さを観察する。
寝たきりの患者はじょく瘡ができやすい。
誤えんを防ぐために姿勢を工夫する。
誤えんの有無を確認した。
てんかんの診断で紹介された。
肝膿瘍の治療は長期に及ぶ。
ロキソプロフェンを食後に服用する。
アセトアミノフェンは解熱にも使われる。
バラシクロビルの内服を終了した。
セフトリアキソンの投与を終了した。
クロピドグレルの内服を再開した。
ビソプロロールで脈拍が落ち着いた。
喀痰の量が減った。
耐性菌の対策を強化する。
嚥下の訓練を続けている。""".split()

# 評価用の分野の文（上とは別の文。人手で短単位に分割した正解）
domain_test = [s.split() for s in """夜間 に せん妄 が 見 られ た ため 、 家族 に 付き添い を 依頼 し た 。
術後 せん妄 の 有無 を 毎日 評価 する 。
じょく瘡 の 予防 に 体位 変換 を 行う 。
仙骨 部 の じょく瘡 は 改善 し て いる 。
誤えん の リスク が 高い ため 、 食事 形態 を 変更 し た 。
誤えん 性 肺炎 で 入院 し た 。
てんかん 発作 が 起き た とき の 対応 を 説明 し た 。
肝 膿瘍 に 対し て 抗菌 薬 を 投与 し た 。
腹部 エコー で 膿瘍 を 認め た 。
発熱 に は アセトアミノフェン を 使用 する 。
腰痛 に 対し て ロキソプロフェン を 処方 し た 。
ロキソプロフェン で 胃 の 不快 感 が 出 た 。
バラシクロビル を 腎 機能 に 合わせ て 減量 する 。
レベチラセタム の 内服 を 継続 する 。
尿路 感染 症 に セフトリアキソン を 点滴 する 。
手術 の 前 に クロピドグレル を 中止 し た 。
脈拍 が 速い ため ビソプロロール を 増量 し た 。
ファモチジン は 腎 機能 が 低い と 減量 が 必要 だ 。
血糖 の 管理 に メトホルミン を 用いる 。
プレドニゾロン の 副 作用 に 注意 する 。
喀痰 から 耐性 菌 が 検出 さ れ た 。
嚥下 の 評価 を 行い 、 食事 を 再開 し た 。
褥瘡 の 状態 を 毎週 記録 する 。
吸入 に は サルブタモール を 用いる 。
レボフロキサシン と アモキシシリン の 併用 は 避ける 。""".split("\n")]

def partial_annotation(sent, terms):
    """文字の間に "|"（境界あり）、"-"（境界なし）、" "（不明）を挟む。用語の出現箇所だけ境界を決める"""
    b = [" "] * (len(sent) - 1)          # b[i] は sent[i] と sent[i+1] の間
    for t in terms:
        for m in re.finditer(re.escape(t), sent):
            i, j = m.start(), m.end()
            if i > 0: b[i - 1] = "|"
            if j < len(sent): b[j - 1] = "|"
            for k in range(i, j - 1): b[k] = "-"
    return "".join(c + x for c, x in zip(sent, b)) + sent[-1]

with open("domain.part", "w", encoding="utf-8") as f:
    f.writelines(partial_annotation(s, terms) + "\n" for s in domain_raw)
print(open("domain.part", encoding="utf-8").read()[:120])
print("評価:", len(domain_test), "文", sum(map(len, domain_test)), "語（うち分野の語",
      sum(w in terms for s in domain_test for w in s), "回）")
```

```python
from collections import Counter

# 辞書は 1 行 1 語（"語/読み" の形で読みも書ける）。3 通り作る
freq = Counter(w for s in train for w in s)
dicts = {
    "terms.dict": terms,                                           # 分野の語だけ
    "all+terms.dict": [w for w in freq if ok(w)] + terms,          # GSD の学習データの全語彙 + 分野の語
    "freq2+terms.dict": [w for w, n in freq.items() if n >= 2 and ok(w)] + terms,   # 2 回以上出た語 + 分野の語
}
for name, words in dicts.items():
    with open(name, "w", encoding="utf-8") as f:
        f.writelines(w + "\n" for w in dict.fromkeys(words))
    print(name, len(set(words)), "語")

def term_acc(gold_sents, pred_sents):
    """正解の中の分野の語のうち、ちょうど 1 語として切り出せた割合"""
    hit = n = 0
    for gold, pred in zip(gold_sents, pred_sents):
        p, pos = spans(pred), 0
        for w in gold:
            if w in terms:
                n += 1; hit += (pos, pos + len(w)) in p
            pos += len(w)
    return f"{hit}/{n}"
```

```python
import pandas as pd

# 学習の設定ごとにモデルを作って比べる（Colab で全部で 4 分ほどの見込み）
settings = {
    "GSD のみ": [],
    "+ 辞書（分野の語だけ）": ["-dict", "terms.dict"],
    "+ 辞書（全語彙 + 分野の語）": ["-dict", "all+terms.dict"],
    "+ 辞書（2 回以上の語 + 分野の語）": ["-dict", "freq2+terms.dict"],
    "+ 部分的アノテーション": ["-part", "domain.part"],
    "+ 辞書（2 回以上の語 + 分野の語）+ 部分的アノテーション": ["-dict", "freq2+terms.dict", "-part", "domain.part"],
}
rows, models = [], {}
for i, (name, args) in enumerate(settings.items()):
    if args:
        k = train_kytea(f"model{i}.bin", *args)
    else:
        k = base_model   # 「GSD のみ」は学習済みのモデルを使い回す
    models[name] = k
    pred = segment_all(k, domain_test)
    rows.append({"設定": name, "分野の F1": word_f1(domain_test, pred), "分野の語": term_acc(domain_test, pred),
                 "誤った文": sum(p != g for p, g in zip(pred, domain_test)),
                 "GSD の F1": word_f1(test, segment_all(k, test))})
pd.DataFrame(rows).set_index("設定")
```

```python
# 誤りのあった文を、GSD だけのモデルと最後のモデルで見比べる
before, after = models["GSD のみ"], models[list(settings)[-1]]
for gold in domain_test:
    s = "".join(gold)
    b, a = list(before.getWS(s)), list(after.getWS(s))
    if b != gold or a != gold:
        print("GSD のみ:", " ".join(b))
        print("適応後  :", " ".join(a), "" if a == gold else "  ← まだ誤り（正解: " + " ".join(gold) + "）")

# 学習データの全語彙を辞書にしたモデルの出力
for s in ["じょく瘡の予防に体位変換を行う。", "腹部エコーで膿瘍を認めた。", "新しい産業構造を作らなければいけない。"]:
    print("全語彙の辞書:", " ".join(models["+ 辞書（全語彙 + 分野の語）"].getWS(s)))
```

- 同梱のモデルは GSD の評価データで単語 F1 が 0.8333、GSD で学習したモデルは 0.9715 だった。同梱のモデルは活用語尾を切り離す（「新し い」「作 ら な けれ ば」「で し た」）ので、この差には精度だけでなく単位の基準の違いが大きく効いている。評価データや後段の処理と同じ基準で分割したコーパスで学習する。GSD は BCCWJ の短単位（数詞だけはまとめる）で人手で分割されている。
- この章の「KyTea で単語分割と読みの推定をする」の注意書きのとおり KyTea は点予測なので、一部の境界だけを決めた部分的アノテーションでも学習できる（付録「仕組みを理解するための実装」の「単語分割を文字単位の系列ラベリングとして解く」の CRF では、すべての文字にラベルの付いた文が要る）。ここでは分野の文 17 文の用語の前後と内部だけに境界を付けて加え、評価の 25 文で分野の語を正しく切り出せた数が 17/28 から 25/28 になり、GSD の F1 はほぼ変わらなかった（0.9715 と 0.9714）。
- 辞書は、学習データの中で辞書の語と一致した箇所から辞書の素性の重みを学習する（KyTea のソースの `wsDictionaryFeatures` は `-dict` で与えた語だけを使う）。分野の語だけの辞書は GSD の文にほとんど現れないので、結果は辞書なしと同じになった。学習データの全語彙を辞書に入れると、分野の語は 28/28 になったが GSD の F1 が 0.9048 に下がり、辞書にない語を辞書にある短い語に切った（「体 位」「エコ ー」）。学習データでは正解の語が必ず辞書にあるので、辞書を信じすぎるようになったと考えられる。2 回以上出た語に絞ると、分野の語 27/28、GSD の F1 0.9748 になり、分野の語をほぼ切り出しつつ GSD の F1 も辞書なしより少し高かった。学習データとは別に作られた、同じ単位の一般の辞書に分野の語を足す方法も考えられるが、ここでは試していない。
- 辞書と部分的アノテーションを両方使うと、分野の語は 28/28、誤りのある文は 25 文中 1 文になった。残った誤りの「肺 炎」は、GSD の学習データに一度も出てこず、分野の語の一覧にも入れていない語。評価の文と用語の一覧はこのレシピのために作った小さなもの（25 文、247 語）で、評価に出る用語をすべて辞書に入れているので、結果は楽観的。設定の間の 1〜2 語の差は、このデータ量ではばらつきの範囲と考える。
- リリースの tar.gz は configure 済みなので、構文解析の章の「Ckylark で日本語の句構造解析をする」（GitHub のソースから autoreconf でビルドする）と違って apt-get が要らない。`--disable-shared` で静的にリンクすれば、`make install` や `ldconfig` をせずに `src/bin/train-kytea` を使える。学習したモデルは pip の `kytea` (0.2.0) でそのまま読み込める。tar.gz の `data/model.bin` は、この章の「KyTea で単語分割と読みの推定をする」でダウンロードするモデルと同じファイル（展開後のハッシュが一致した）。
- 検証では、何度実行しても表の数値は同じだった（GSD は r2.19。master も同じ内容だった）。KyTea のソースは Apache License 2.0、同梱のモデルの利用条件は「KyTea で単語分割と読みの推定をする」の注意書きを参照。GSD は CC BY-SA 4.0 なので、学習したモデルを配布するときは表示と継承の条件の扱いを確かめる。用語の一覧と医療の文はこのレシピのために作った架空の例。

## 発音・モーラ数・アクセントを取り出し、5-7-5 や読みの違いを調べる (pyopenjtalk) {mora_accent}

読みの先にある、文の発音（「今日は」→「キョーワ」）、モーラ（拍）の数、アクセント句ごとのアクセント核を取り出したい。音声合成システム Open JTalk のテキスト処理部分を Python から使える pyopenjtalk で求め、俳句・川柳の 5-7-5 の判定と、同じ表記で読みの違う語の読み分けに使う。

```python
# pyopenjtalk は PyPI にソースだけがあり、インストール時にビルドする（Colab で 2〜5 分ほど）
!pip install -q pyopenjtalk==0.4.1 fugashi==1.5.2 unidic-lite==1.0.8 kytea
# KyTea の学習済みモデル（この章の「KyTea で単語分割と読みの推定をする」と同じ）
!wget -q https://github.com/neubig/kytea/raw/master/data/model.bin.gz && gunzip -f model.bin.gz
```

```python
import pandas as pd
import pyopenjtalk

pd.set_option("display.unicode.east_asian_width", True)

# 初回の呼び出しで Open JTalk の辞書（約 23MB）を GitHub からダウンロードする
text = "今日はいい天気ですね。"
print(pyopenjtalk.g2p(text, kana=True))   # 発音形（カタカナ）
print(pyopenjtalk.g2p(text))              # 音素列（大文字の母音は無声化）

# run_frontend は単語ごとの解析結果を返す。orig は原形、read は読み、pron は発音、mora_size はモーラ数、
# acc はアクセント核の位置、chain_flag が 1 なら前の単語と同じアクセント句
cols = ["string", "pos", "orig", "read", "pron", "mora_size", "acc", "chain_flag"]
pd.DataFrame(pyopenjtalk.run_frontend(text))[cols]
```

```python
import re

def split_mora(kana):
    """カタカナをモーラに分ける。小書きの「ャュョァィゥェォヮ」は前の文字と合わせて 1 モーラ、「ッ」「ン」「ー」は 1 モーラ"""
    return re.findall(r".[ァィゥェォャュョヮ]?", kana)

def accent_phrases(text):
    """アクセント句ごとに、単語・発音・モーラ数・アクセント核（下がり目の直前のモーラの位置。0 は下がらない平板型）を返す"""
    phrases = []
    for f in pyopenjtalk.run_frontend(text):
        if f["mora_size"] == 0:                    # 句読点などの記号
            continue
        pron = f["pron"].replace("’", "")          # ’ は直前の母音の無声化の印
        if f["chain_flag"] == 1 and phrases:       # 前の単語と同じアクセント句
            phrases[-1]["words"].append(f["string"])
            phrases[-1]["pron"] += pron
        else:                                      # 句の先頭の単語の acc が句全体のアクセント核
            phrases.append({"words": [f["string"]], "pron": pron, "acc": f["acc"]})
    for p in phrases:
        mora, acc = split_mora(p["pron"]), p["acc"]
        p["mora"] = len(mora)
        p["marked"] = "".join(m + ("＼" if i == acc else "") for i, m in enumerate(mora, 1))
        p["HL"] = pitch(len(mora), acc)
    return phrases

def pitch(n, acc):
    """東京方言の高低: 1 モーラ目と 2 モーラ目は高さが違い、核の後ろで下がる"""
    if acc == 1:
        return "H" + "L" * (n - 1)
    high = (acc or n) - 1                          # 平板型 (0) は最後まで高い
    return "L" + "H" * high + "L" * (n - 1 - high)

for s in ["今日はいい天気ですね。", "東京特許許可局", "ちょっと待ってください。", "新しいスマートフォンを買いに行く。"]:
    print(s)
    for p in accent_phrases(s):
        print(f"  {'/'.join(p['words']):16} {p['marked']:14} {p['mora']} モーラ  核 {p['acc']}  {p['HL']}")
```

```python
# 確認: 音声合成用のフルコンテキストラベルの /F:f1_f2 はアクセント句のモーラ数と核の位置。
# ラベルでは平板型 (0) の核を句のモーラ数で表す（Open JTalk の jpcommon_label.c）ので、そろえて比べる
def phrases_from_labels(text):
    out, prev = [], None
    for lab in pyopenjtalk.extract_fullcontext(text):
        phone = re.search(r"-(.+?)\+", lab).group(1)
        if phone in ("sil", "pau"):                # 無音・ポーズで句が切れる
            prev = None
            continue
        f = re.search(r"/F:([^/]+)", lab).group(1)
        if f != prev:
            out.append(tuple(int(x) for x in re.match(r"(\d+)_(\d+)", f).groups()))
            prev = f
    return out

for s in ["今日はいい天気ですね。", "東京特許許可局", "ちょっと待ってください。", "新しいスマートフォンを買いに行く。"]:
    mine = [(p["mora"], p["acc"] or p["mora"]) for p in accent_phrases(s)]
    print(s, mine, mine == phrases_from_labels(s))
```

```python
# 読みが同じでアクセントで区別する語。UniDic (unidic-lite) の aType は単語単位のアクセント型
from fugashi import Tagger

tagger = Tagger()
rows = []
for s in ["橋を渡る。", "箸を使う。", "端に寄る。", "雨が降る。", "飴をなめる。"]:
    word = next(iter(tagger(s)))
    p = accent_phrases(s)[0]
    rows.append({"文": s, "語": word.surface, "UniDic aType": word.feature.aType,
                 "pyopenjtalk の句": p["marked"], "句の核": p["acc"], "高低": p["HL"]})
print("UniDic aType:", [(w.surface, w.feature.aType) for w in tagger("東京特許許可局")])
pd.DataFrame(rows)
```

```python
# 応用 1: 俳句・川柳の 5-7-5 を判定する。俳句の「音」はモーラで数える（拗音は 1、促音・撥音・長音も 1）
def nodes(text):
    return [(f["string"], f["pron"].replace("’", ""), f["mora_size"])
            for f in pyopenjtalk.run_frontend(text) if f["mora_size"] > 0]

def split_575(text, pattern=(5, 7, 5)):
    """単語の切れ目で 5・7・5 モーラに分けられれば 3 つの句を返す。分けられなければ理由を返す"""
    ns = nodes(text)
    total = sum(m for _, _, m in ns)
    if total != sum(pattern):
        return f"{total} モーラ"
    parts, cur, cum, goal = [], [], 0, 0
    for surface, pron, m in ns:
        cur.append((surface, pron))
        cum += m
        if cum > goal + pattern[len(parts)]:
            return f"単語「{surface}」が句の切れ目をまたぐ"
        if cum == goal + pattern[len(parts)]:
            parts.append(("".join(s for s, _ in cur), "".join(p for _, p in cur)))
            cur, goal = [], cum
    return parts

haiku = [  # 作者の没後 100 年以上たったパブリックドメインの句
    ("古池や蛙飛び込む水の音", "松尾芭蕉"),
    ("閑さや岩にしみ入る蝉の声", "松尾芭蕉"),
    ("菜の花や月は東に日は西に", "与謝蕪村"),
    ("柿くへば鐘が鳴るなり法隆寺", "正岡子規"),
    ("やせ蛙まけるな一茶是に有", "小林一茶"),
] + [(s, "自作") for s in [
    "今日もまた電車が遅れ汗をかく",
    "学校のチャイムが鳴ってほっとする",
    "新しいスマートフォンを買いに行く",
    "朝からアルバイトをして眠くなる",
    "自然言語処理の小ネタを集めたい",
]]
rows = []
for text, author in haiku:
    pron = "".join(p for _, p, _ in nodes(text))
    r = split_575(text)
    rows.append({"句": text, "作": author, "発音": pron, "文字数": len(pron), "モーラ数": len(split_mora(pron)),
                 "判定": " / ".join(p for p, _ in r) if isinstance(r, list) else "× " + r})
pd.DataFrame(rows)
```

```python
# 句の切れ目が分かっているとき（空白区切り）は句ごとに数える。字余りも分かる
def count_parts(text):
    return [(part, sum(m for _, _, m in nodes(part))) for part in text.split()]

print(count_parts("雀の子 そこのけそこのけ 御馬が通る"))   # 小林一茶（パブリックドメイン）
print(count_parts("ちょっと待って ショートケーキが 売り切れだ"))  # 自作
```

```python
# 応用 2: 同じ表記で読みの違う語を、MeCab + UniDic、KyTea、pyopenjtalk で読み分けられるか比べる（例文は自作）
import Mykytea

kytea = Mykytea.Mykytea("-model model.bin")

def hira2kata(s):
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in s)

def span_reading(tokens, sent, target):
    """(表層, 読み) の列から、target の文字範囲に重なる単語の読みをつなげる"""
    # 表層をつなぐと元の文になると仮定している。pyopenjtalk は表層を書き換えることがあり
    # （「く」→「九」など）、長さが変わると文字位置がずれるので、写して使うときは注意する
    i = sent.index(target)
    j, pos, out = i + len(target), 0, []
    for surface, yomi in tokens:
        if pos < j and pos + len(surface) > i:
            out.append(yomi)
        pos += len(surface)
    return "".join(out)

readers = {
    "MeCab+UniDic": lambda s: [(w.surface, w.feature.kana or w.surface) for w in tagger(s)],
    "KyTea": lambda s: [(w.surface, hira2kata(w.tag[1][0][0])) for w in kytea.getTags(s)],
    "pyopenjtalk": lambda s: [(f["string"], f["read"]) for f in pyopenjtalk.run_frontend(s)],
}
cases = [  # (文, 対象の語, 正しい読み)
    ("今日は朝から雨が降っている。", "今日", "キョウ"),
    ("今日では誰もがスマートフォンを持っている。", "今日", "コンニチ"),
    ("魚を市場で買って帰った。", "市場", "イチバ"),
    ("株式市場が大きく混乱した。", "市場", "シジョウ"),
    ("彼は料理が上手だ。", "上手", "ジョウズ"),
    ("相手の方が一枚上手だった。", "上手", "ウワテ"),
    ("役者が舞台の上手から登場する。", "上手", "カミテ"),
    ("生物の授業で細胞を学んだ。", "生物", "セイブツ"),
    ("生物は早めに食べてください。", "生物", "ナマモノ"),
    ("一日に三回薬を飲む。", "一日", "イチニチ"),
    ("四月一日は入学式だ。", "一日", "ツイタチ"),
    ("人気のある店に並ぶ。", "人気", "ニンキ"),
    ("人気のない夜道を歩く。", "人気", "ヒトケ"),
    ("この辛いカレーは汗が出る。", "辛い", "カライ"),
    ("友達との別れが辛い。", "辛い", "ツライ"),
    ("この店は大人気だ。", "大人気", "ダイニンキ"),
    ("大人気ない態度をとった。", "大人気", "オトナゲ"),
]
rows = []
for sent, target, gold in cases:
    row = {"文": sent, "語": target, "正解": gold}
    for name, f in readers.items():
        y = span_reading(f(sent), sent, target)
        row[name] = ("○ " if y.startswith(gold) else "× ") + y
    rows.append(row)
df = pd.DataFrame(rows)
print({name: f"{df[name].str.startswith('○').sum()}/{len(df)}" for name in readers})
df
```

- pyopenjtalk は山本龍一氏による Open JTalk（名古屋工業大学の HTS Working Group が開発した日本語音声合成システム）の Python ラッパーで、MIT ライセンス。Open JTalk 本体は修正 BSD ライセンス。初回に自動でダウンロードされる辞書 (open_jtalk_dic_utf_8-1.11) は mecab-naist-jdic を元にしたもので、同梱の COPYING によれば奈良先端科学技術大学院大学と UniDic コンソーシアムの修正 BSD ライセンス。ここでは `run_frontend`（テキスト処理）と `extract_fullcontext`（ラベル）だけを使い、音声合成はしない。品詞は IPA 辞書の体系で、「いい天気」の「いい」が動詞「いう」と解析されるような誤りもある。
- この章の「漢字の読みを取得してふりがな（ルビ）を付ける」と「KyTea で単語分割と読みの推定をする」は単語の読みまでを出す。pyopenjtalk はその先の、助詞の「は」→「ワ」や長音（「キョウ」→「キョー」）を反映した発音、無声化、アクセント句への区切りとアクセント核を出す。アクセント核は単語ごとの辞書のアクセント型に結合の規則を当てて句ごとに決まる。UniDic の `aType` は単語単位で「東京」は 0 型だが、pyopenjtalk では「東京特許許可局」が 1 つの句（11 モーラ、核 9）になった。
- 「橋」「箸」「端」は UniDic の aType でも pyopenjtalk でも 2・1・0 型で、助詞まで含めた句で見ると「ハシ＼ヲ」「ハ＼シヲ」「ハシニ」と違いが出る（尾高型の「橋」と平板型の「端」は、単語だけでは高低が同じになる）。アクセントは辞書と規則による東京方言の推定である。
- 5-7-5 の判定では、発音のカタカナの文字数は小書きの「ョ」「ャ」などの分だけモーラ数より多い（「今日もまた電車が遅れ汗をかく」は 19 文字で 17 モーラ）。判定は読みの推定に左右され、芭蕉の「閑さや」は「ヒマサヤ」、一茶の「是に有」は「ゼニユー」と読まれて、どちらも 16 モーラになった（正しくは「シズカサヤ」「コレニアリ」）。合計が 17 モーラでも、単語が句の切れ目をまたぐ句（句またがり）は × になる（「朝からアルバイトをして眠くなる」）。子規の「柿くへば」は旧仮名遣いの「くへ」が「九」「へ」と解析されたが、発音は「カキクエバ」で 17 モーラになった。古典や旧仮名遣いの句は、判定の前に読みを人が確かめる（古文の解析は、この章の「古文を形態素解析する（中古和文 UniDic）」を参照）。
- 読み分けは自作の 17 例で、正解は MeCab + UniDic が 10、KyTea と pyopenjtalk が 11 だった。例が少ないので、優劣ではなく誤り方の違いとして見る。「市場（イチバ）」「生物（ナマモノ）」「人気（ヒトケ）」は 3 つとも誤り、「舞台の上手（カミテ）」は UniDic だけ、「一枚上手（ウワテ）」「大人気ない」は pyopenjtalk だけ、「今日では（コンニチ）」「辛いカレー（カライ）」は KyTea だけが当てた。読みの候補とスコアを見たいときは、KyTea の `getAllTags` を使う（「KyTea で単語分割と読みの推定をする」を参照）。
- 決まった語の読みやアクセントを直したいときは、pyopenjtalk のユーザー辞書（`mecab_dict_index` で CSV から作り、`update_global_jtalk_with_user_dict` で読み込む）を使う。文脈で読みが変わる語は、辞書の登録では直らない。
- pyopenjtalk は PyPI に wheel がなく、インストール時に C++ のコードをビルドする（検証では 2 コアでインストールのセル全体が約 3 分）。派生版の pyopenjtalk-plus は、ビルド済みの wheel と独自に修正した辞書を同梱し、同じ `import pyopenjtalk` で使えると README にある（MIT ライセンス）。両方を同じ環境に入れると衝突するので、どちらか一方を入れる。
- pyopenjtalk の `tts` で音声合成まで行い、LLM の答えを読み上げる例は、対話システムの章の「音声で話せる対話システムを作る（音声認識・LLM・音声合成）」を参照。

## 古文を形態素解析する（中古和文 UniDic） {classical_japanese}

源氏物語や枕草子のような古文を単語に分け、品詞・活用・読みを調べたい。現代語用の辞書ではうまく解析できないので、古文用の辞書を使う。

```python
!pip install -q fugashi unidic-lite
```

```python
!wget -q https://clrd.ninjal.ac.jp/unidic_archive/2512/unidic-chuko-v202512.zip && unzip -q -o unidic-chuko-v202512.zip -d chuko
!find chuko -name "dicrc" | head -n 1
```

```python
import glob, os
from fugashi import GenericTagger, Tagger

dicdir = os.path.dirname(glob.glob("chuko/**/dicrc", recursive=True)[0])
classical = GenericTagger(f'-d "{dicdir}" -r "{dicdir}/dicrc"')      # 中古和文 UniDic（平安時代の仮名文学向け）
modern = Tagger()                                                    # 現代語の UniDic (unidic-lite)

text = "いづれの御時にか、女御、更衣あまたさぶらひたまひけるなかに、いとやむごとなき際にはあらぬが、すぐれて時めきたまふありけり。"
print("中古和文 UniDic:", " | ".join(w.surface for w in classical(text)))
print("現代語 UniDic :", " | ".join(w.surface for w in modern(text)))
```

```python
# 素性の並びを確認する（UniDic 共通の形式: 品詞 4 階層, 活用型, 活用形, 語彙素読み, 語彙素, ...）
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
# 区切りが同じでも、品詞や語彙素の判定が違うことがある（伝聞の「なる」を、現代語の辞書は断定の「なり」とする）
s2 = "男もすなる日記といふものを"
print("中古和文:", [(w.surface, w.feature[7]) for w in classical(s2)][:5])
print("現代語  :", [(w.surface, w.feature.lemma) for w in modern(s2)][:5])
```

- 古文は、語彙も活用も仮名遣いも現代語と違う。現代語用の辞書では、助動詞（けり、なり、む）や歴史的仮名遣い（いふ、やうやう）を正しく扱えない。
- 国立国語研究所が、時代・文体別の UniDic を公開している。平安時代の仮名文学には「中古和文」、明治の文語文には「近代文語」、ほかに上代語、中世の説話や狂言、近世の洒落本向けなどがある。対象のテキストに合うものを選ぶ。
- 辞書の形式は現代語の UniDic と共通なので、MeCab (fugashi) の辞書を差し替えるだけで使える。語彙素（辞書の見出し）でまとめれば、表記のゆれをそろえて数えられる。
- 活用形や助動詞の語彙素が分かるので、古文の学習支援（品詞分解）、文体の計量的な分析、時代による語の変化の調査に使える。
- 辞書と違う時代・文体のテキスト（漢文訓読体、和漢混交文、近世の口語）では精度が落ちる。解析結果を抜き取って確かめる。

## 日本語の文を句点で分割する（括弧内は分割しない） {sentence_split_ja}

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

- 単純な規則では、括弧内の句点だけでなく「モーニング娘。」のように句点を含む固有名詞や、略語・小数点などでも誤って文を切ってしまうことがある。
- 改行・絵文字・小数点・URL などにも対応した既存ライブラリの比較は、次のレシピ「日本語の文分割をライブラリで行って比べる」を参照。

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
- ja-sentence-segmenter は、分割と連結の規則を自分で組み替えられる。行の途中で折り返された文章（メールや PDF 由来のテキスト）をつなぎ直す処理も書ける。
- ja-sentence-segmenter と pySBD は、「本当ですか！？」を「本当ですか！」と「？」の 2 文に分けてしまう。pySBD は、折り返された「見て」と「ください。」もつながない。
- GiNZA は句点のない文末（絵文字、顔文字の後ろ）も区切れるが、解析を伴うので 2 桁以上遅い。「はい（たぶん。）そうです。」は、閉じかっこの後ろで切って 2 文にした。
- SNS や口コミのような崩れた文章向けには、学習ベースの bunkai がある（執筆時点では依存ライブラリ emoji の新しい版と互換性がなく、そのままでは動かなかった）。
- 文分割の誤りは後段のすべての処理に響く。自分のデータの数十文で、どこで切れているかを必ず目で確かめる。

## 句読点のない文章に読点と句点を付ける（句読点の復元） {punctuation_restoration gpu}

音声認識の書き起こしのように句読点のない日本語の文章に、読点「、」と句点「。」を付けたい。句読点を消した文章から学習データを作り、文字単位の BERT で「各文字の後ろに何が来るか」を当てる。

```python
!pip install -q fugashi unidic-lite
```

```python
import random, re
from datasets import load_dataset

random.seed(0)
PUNCT = {"、": 1, "。": 2}          # ラベル: 0 = なし、1 = 後ろに読点、2 = 後ろに句点
NAMES = ["なし", "読点", "句点"]
WIN = 200                           # 1 件あたりの文字数（句読点を除く）

def strip_punct(text):
    """句読点を消した文字列と、各文字の後ろに付いていた句読点のラベルを返す"""
    chars, labels = [], []
    for c in text:
        if c in PUNCT:
            if labels:
                labels[-1] = PUNCT[c]
        else:
            chars.append(c); labels.append(0)
    return "".join(chars), labels

# 日本語 Wikipedia（wikimedia/wikipedia の 20231101.ja、CC BY-SA 3.0 / GFDL）をストリーミングで読む
# 記事の本文の段落をつなぎ、句読点を消してから WIN 文字ずつに切る（切れ目は文の境界とは限らない）
wiki = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
train, test, test_doc = [], [], []      # test_doc: 評価用の各件がどの記事から来たか
for i, article in enumerate(wiki):
    if any(w in article["title"] for w in ["一覧", "曖昧さ回避"]):
        continue
    paras = [l.strip() for l in article["text"].split("\n")
             if len(l) >= 40 and l.strip().endswith("。") and not re.search(r"[，．]", l)]   # 「，．」を使う記事は除く
    chars, labels = strip_punct("".join(paras))
    dest = test if i % 10 == 0 else train            # 記事単位で学習用と評価用に分ける
    for s in range(0, min(len(chars), 4 * WIN) - WIN + 1, WIN):   # 1 記事から最大 4 件
        dest.append((chars[s:s + WIN], labels[s:s + WIN]))
        if dest is test:
            test_doc.append(i)
    if len(train) >= 20000 and len(test) >= 1000:
        break
train, test, test_doc = train[:20000], test[:1000], test_doc[:1000]
print(len(train), len(test))
print(test[0][0][:80])
print("読点の割合:", round(sum(l.count(1) for _, l in train) / (WIN * len(train)), 3),
      " 句点の割合:", round(sum(l.count(2) for _, l in train) / (WIN * len(train)), 3))
```

```python
import torch, unicodedata
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          PrinterCallback, Trainer, TrainingArguments, set_seed)

set_seed(0)
# 東北大の文字単位の日本語 BERT (Apache 2.0)。語彙は 1 文字ずつなので、文字とトークンが 1 対 1 に対応する
model_name = "tohoku-nlp/bert-base-japanese-char-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
vocab = tokenizer.get_vocab()

def encode(chars):
    # tokenizer() を通すと MeCab で単語に切ってから文字に分けるが、ここでは 1 文字ずつ直接 ID にして、文字とラベルの位置をそろえる。
    # 語彙は NFKC 正規化済み（全角の「（」「Ａ」「１」はなく、半角の「(」「A」「1」がある）なので、1 文字ずつ NFKC にしてから引く。
    # NFKC で複数の文字になるもの（「㍿」など）や語彙にない文字は [UNK]
    return ([tokenizer.cls_token_id] + [vocab.get(unicodedata.normalize("NFKC", c), tokenizer.unk_token_id) for c in chars]
            + [tokenizer.sep_token_id])

def to_features(data):
    return [{"input_ids": encode(c), "labels": [-100] + l + [-100]} for c, l in data]   # [CLS] と [SEP] は損失から除く

model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=3, id2label=dict(enumerate(NAMES)), label2id={n: i for i, n in enumerate(NAMES)})
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
args = TrainingArguments(
    output_dir="punct-out", num_train_epochs=1, learning_rate=5e-5, warmup_steps=30, weight_decay=0.01,
    per_device_train_batch_size=32, lr_scheduler_type="linear", bf16=BF16, fp16=not BF16,
    logging_steps=100, save_strategy="no", report_to="none", seed=0, disable_tqdm=True)
trainer = Trainer(model=model, args=args, train_dataset=to_features(train),
                  data_collator=DataCollatorForTokenClassification(tokenizer))
trainer.remove_callback(PrinterCallback)   # 学習中のログ（所要時間を含む）は出さず、損失だけを後で表示する
trainer.train()   # T4 で 5 分前後の見込み
for log in trainer.state.log_history:
    if "loss" in log:
        print(f"ステップ {log['step']}: 損失 {log['loss']:.4f}")
```

```python
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

model.eval()

@torch.no_grad()
def predict(texts, batch_size=64):
    """句読点のない文字列（各 WIN 文字以下）ごとに、各文字の後ろのラベルの列を返す"""
    out = []
    for b in range(0, len(texts), batch_size):
        batch = DataCollatorForTokenClassification(tokenizer)([{"input_ids": encode(t)} for t in texts[b:b + batch_size]])
        logits = model(**{k: v.to(model.device) for k, v in batch.items()}).logits.float()
        out += [row[1:len(t) + 1].argmax(-1).tolist() for t, row in zip(texts[b:b + batch_size], logits)]
    return out

def report(gold, pred, title):
    g, p = sum(gold, []), sum(pred, [])
    prec, rec, f1, n = precision_recall_fscore_support(g, p, labels=[1, 2], zero_division=0)
    print(title)
    for k, name in enumerate(["読点", "句点"]):
        print(f"  {name}: 適合率 {prec[k]:.3f}  再現率 {rec[k]:.3f}  F1 {f1[k]:.3f}  (正解 {n[k]} 個)")

def restore(chars, labels):
    return "".join(c + ("", "、", "。")[l] for c, l in zip(chars, labels))

pred = predict([c for c, _ in test])
report([l for _, l in test], pred, f"BERT (評価 {len(test)} 件、{WIN * len(test)} 文字)")
ids = sum((encode(c)[1:-1] for c, _ in test), [])
print(f"評価データで [UNK] になった文字の割合: {ids.count(tokenizer.unk_token_id) / len(ids):.4f}")
print("正解:", restore(*test[1])[:120])
print("予測:", restore(test[1][0], pred[1])[:120])
```

```python
# 書き起こし風の話し言葉に付けてみる。長い文章は WIN 文字ずつに切って予測する
def punctuate(text):
    text = re.sub(r"[\s、。]", "", text)
    pieces = [text[s:s + WIN] for s in range(0, len(text), WIN)]
    return "".join(restore(p, l) for p, l in zip(pieces, predict(pieces)))

speech = ("えーと今日はですね新しい製品の発表についてお話ししたいと思いますまず価格ですが"
          "従来モデルと同じに据え置きました次に発売日ですが来月の十日を予定しております"
          "ご質問があればあとでまとめてお受けします")
restored = punctuate(speech)
print(restored)
# 句点が付けば文に分けられる。ここでは簡単のため句点だけで分ける（括弧の扱いはこの章の「日本語の文を句点で分割する（括弧内は分割しない）」を参照）
for s in re.findall(r"[^。]+。?", restored):
    print("-", s)
```

```python
# LLM に句読点を付けさせて比べる（sarashina2.2-3b-instruct、MIT ライセンス）。評価用の先頭 100 件だけを使う
import difflib
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM

name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
tok = AutoTokenizer.from_pretrained(name, padding_side="left")
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tok.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()

INST = "次の文章に読点「、」と句点「。」を付けてください。文字の追加・削除・変更はせず、句読点を付けた文章だけを出力してください。\n\n"
example = train[0]   # 1-shot の例（学習データから）
def messages(text):
    return [{"role": "user", "content": INST + example[0]}, {"role": "assistant", "content": restore(*example)},
            {"role": "user", "content": INST + text}]

@torch.no_grad()
def llm_punctuate(texts, batch_size=10):
    outs = []
    for b in range(0, len(texts), batch_size):
        enc = tok.apply_chat_template([messages(t) for t in texts[b:b + batch_size]], add_generation_prompt=True,
                                      padding=True, return_tensors="pt", return_dict=True).to(llm.device)
        gen = llm.generate(**enc, max_new_tokens=300, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in outs]

def project(src, out):
    """LLM の出力の句読点を、元の文字列 src の文字位置に写す（LLM が文字を変えた部分は句読点なし扱い）"""
    o_chars, o_labels = strip_punct(out)
    sm = difflib.SequenceMatcher(None, src, o_chars, autojunk=False)
    labels = [0] * len(src)
    for a, b, n in sm.get_matching_blocks():
        labels[a:a + n] = o_labels[b:b + n]
    changes = [(src[i1:i2], o_chars[j1:j2]) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]
    return labels, changes

sub = test[:100]
outs = llm_punctuate([c for c, _ in sub])   # T4 では 3〜4 分かかる見込み（モデルのダウンロードは別）
proj = [project(c, o) for (c, _), o in zip(sub, outs)]
print("句読点以外の文字を変えなかった出力:", sum(not ch for _, ch in proj), "/", len(sub))
print("LLM が変えた箇所の例 (元 → 出力):", [ch for _, chs in proj[:5] for ch in chs[:2]])
report([l for _, l in sub], [l for l, _ in proj], "LLM (1-shot、100 件)")
report([l for _, l in sub], pred[:100], "BERT (同じ 100 件)")
print("LLM:", outs[1][:120])
print(f"GPU メモリの最大使用量: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

```python
# 100 件は少ないので、ブートストラップで F1 の差の 95% 信頼区間を求める。
# 同じ記事の断片どうしは似ているので、断片ではなく記事を単位に復元抽出する
from sklearn.metrics import f1_score

gold100, bert100, llm100 = [l for _, l in sub], pred[:100], [l for l, _ in proj]
docs = sorted(set(test_doc[:len(sub)]))
by_doc = {d: [j for j in range(len(sub)) if test_doc[j] == d] for d in docs}
print("100 件のもとの記事の数:", len(docs))
rng = np.random.default_rng(0)
diffs = {1: [], 2: []}
for _ in range(1000):
    idx = [j for d in rng.choice(docs, len(docs)) for j in by_doc[d]]
    g = sum((gold100[i] for i in idx), [])
    fb = f1_score(g, sum((bert100[i] for i in idx), []), labels=[1, 2], average=None, zero_division=0)
    fl = f1_score(g, sum((llm100[i] for i in idx), []), labels=[1, 2], average=None, zero_division=0)
    for k in (1, 2):
        diffs[k].append(fb[k - 1] - fl[k - 1])
for k in (1, 2):
    lo, hi = np.percentile(diffs[k], [2.5, 97.5])
    print(f"{NAMES[k]}の F1 の差 (BERT − LLM): 95% 信頼区間 [{lo:.3f}, {hi:.3f}]")
```

- 句読点を消すだけで、正解付きの学習データがいくらでも作れる（自己教師あり）。付録「仕組みを理解するための実装」の「単語分割を文字単位の系列ラベリングとして解く」と同じく、文字ごとのラベル付けとして解いている。語彙が 1 文字単位の BERT を使うと、トークンと文字の位置合わせが要らない。ただしこの BERT の語彙は NFKC 正規化済みで、全角の「（」「１」などを含まないので、1 文字ずつ NFKC にしてから ID を引いている（評価データで [UNK] になった文字は 0.05%）。
- 評価用の 1000 件（20 万文字）で、句点は F1 0.952（適合率 0.946、再現率 0.958）と高く、読点は F1 0.790（適合率 0.829、再現率 0.754）と低かった（T4 と同じ float16 で学習・推論したときの値。GPU での学習は完全には再現しないので、実行するたびに小数第 3 位が少し変わる）。読点の位置は書き手によって揺れ、付けても付けなくてもよい箇所が多い（上の例でも「最後に、」の読点を落とした）。元の文章の読点を唯一の正解として数えるので、読点の数値は実用上の質より低めに出ることがある。
- 学習も評価も書き言葉の Wikipedia なので、フィラー（「えーと」）や言い直しを含む実際の書き起こしでの精度は確かめていない。上の話し言葉の例では文末 4 か所すべてに句点が付いたが、「まず価格ですが」の後ろには読点が付かず、「思い、ます」のように語の途中に読点を付ける誤りもあった（この例の結果は実行ごとに変わることがある）。また、この BERT は事前学習に 2023 年 1 月の日本語 Wikipedia を使っているので、評価用の記事を句読点付きで見ている可能性がある。実際に使うときは、句読点の付いた話し言葉のテキスト（字幕や議事録など）を学習データに加え、自分のデータで評価する。
- 句点を復元すれば、この章の「日本語の文を句点で分割する（括弧内は分割しない）」や「日本語の文分割をライブラリで行って比べる」の方法で文に分けられる。句点の誤りはそのまま文分割の誤りになる。長い文章は 200 文字ずつ重なりなしに切っているので、切れ目の前後の文字は文脈が足りない。重なりを持たせて切り、中央部分の予測だけを使うとよい。
- LLM（sarashina2.2-3b-instruct、1-shot）は、同じ 100 件で読点 F1 0.663、句点 F1 0.808 と BERT（0.791、0.957）より低く、F1 の差のブートストラップ 95% 信頼区間（100 件のもとの 27 記事を単位に復元抽出。読点 [0.093, 0.167]、句点 [0.109, 0.189]）も 0 を含まなかった。句読点以外の文字を変えなかった出力は 100 件中 1 件だけで、文の途中で切れた断片の続きを書き足したり、記号を足したりした。そのため出力を difflib で元の文字列に対応づけてから評価している。より大きな LLM や API のモデルでの結果は確かめていない。
- GPU メモリの最大使用量は、BERT の学習と 3B の LLM の推論を合わせて 9.0 GB で、T4 (16GB) に収まる。
- データとモデルの出典とライセンス: 日本語 Wikipedia（wikimedia/wikipedia、CC BY-SA 3.0 / GFDL）、tohoku-nlp/bert-base-japanese-char-v3（東北大学、Apache 2.0）、sbintuitions/sarashina2.2-3b-instruct-v0.1（SB Intuitions、MIT）。sarashina のトークナイザーを直すコードについては、大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」の注意書きを参照。
