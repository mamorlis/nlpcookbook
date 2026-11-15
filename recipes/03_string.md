# 文字列処理と正規化

形態素解析やモデルに入れる前の、文字レベルの下ごしらえ。

## Unicode 正規化 (NFKC) で全角英数・半角カナを揃える {nfkc}

全角英数字を半角に、半角カナを全角に、①や㍻のような互換文字を標準的な形に、まとめて揃えたい。

```python
import unicodedata

text = "ＮＬＰは楽しい！　ﾊﾝｶｸｶﾅ と ①②③ と ㍻ と ｶﾞ"
print(unicodedata.normalize("NFKC", text))
```

```python
# 4 つの正規化形式の違い
s = "が"  # 1 文字
for form in ["NFC", "NFD", "NFKC", "NFKD"]:
    t = unicodedata.normalize(form, s)
    print(form, len(t), [f"U+{ord(c):04X}" for c in t])
```

- 迷ったら **NFKC**。検索・分類の前処理の定番。
- NFD/NFKD は濁点が分離する (macOS のファイル名で遭遇しがち)。`"が" != "が"` になったら NFC に戻す。
- NFKC は全角チルダ「～」→「~」など記号も変えるので、原文を保持したい場合は別に持っておく。

## 全角⇔半角を種類ごとに変換する {zenhan}

NFKC では一括変換になってしまうので、「数字だけ半角に」「カナだけ全角に」のように種類を選んで変換したい。

```python
!pip install -q jaconv
```

```python
import jaconv

text = "ＡＢＣ１２３ｱｲｳ アイウ abc123"
print(jaconv.z2h(text, kana=False, ascii=True, digit=True))   # 全角英数 → 半角
print(jaconv.z2h(text, kana=True, ascii=False, digit=False))  # 全角カナ → 半角カナ
print(jaconv.h2z(text, kana=True, ascii=False, digit=False))  # 半角カナ → 全角カナ
print(jaconv.h2z(text, kana=True, ascii=True, digit=True))    # 全部全角
```

## ひらがな⇔カタカナを変換する {hira_kata}

読みの比較や検索のために、ひらがなとカタカナを相互に変換したい。

```python
# ライブラリなし: ひらがな (U+3041-3096) とカタカナ (U+30A1-30F6) は 0x60 ずれているだけ
HIRA2KATA = {i: i + 0x60 for i in range(0x3041, 0x3097)}
KATA2HIRA = {v: k for k, v in HIRA2KATA.items()}

print("しぜんげんごしょり".translate(HIRA2KATA))
print("シゼンゲンゴショリ".translate(KATA2HIRA))
```

```python
!pip install -q jaconv
```

```python
import jaconv

print(jaconv.hira2kata("しぜんげんごしょり"))
print(jaconv.kata2hira("シゼンゲンゴショリ"))
print(jaconv.hira2hkata("しぜんげんご"))  # 半角カナへ
```

## 文字種 (漢字・ひらがな・カタカナ・英数字) を判定する {char_type}

文字列に漢字が含まれるか、カタカナだけでできているか、などを判定したい。

```python
!pip install -q regex
```

```python
import regex  # 標準の re と違い \p{Han} などの Unicode プロパティが使える

text = "東京タワーは333mで、とても高いです。"
patterns = {
    "漢字": r"\p{Han}+",
    "ひらがな": r"\p{Hiragana}+",
    "カタカナ": r"[\p{Katakana}ー]+",
    "英字": r"\p{Latin}+",
    "数字": r"\p{Nd}+",
}
for name, pat in patterns.items():
    print(name, regex.findall(pat, text))
```

```python
def is_all_katakana(s):
    return regex.fullmatch(r"[\p{Katakana}ー]+", s) is not None

def has_kanji(s):
    return regex.search(r"\p{Han}", s) is not None

print(is_all_katakana("コンピューター"), is_all_katakana("コンピュータ室"))
print(has_kanji("ひらがなだけ"), has_kanji("漢字あり"))
```

- 標準の `re` だけでやるなら `[一-龥々]` `[ぁ-ん]` `[ァ-ヶー]` で概ね足りるが、拡張漢字は漏れる。

## neologdn で日本語の表記ゆれを正規化する {neologdn}

長音の連続、半角カナ、不要な空白など、Web テキストによくある表記ゆれをまとめて直したい。

```python
!pip install -q neologdn
```

```python
import neologdn

print(neologdn.normalize("ﾊﾝｶｸｶﾅ　と　全角　ス ペ ー ス"))
print(neologdn.normalize("スーーーーパーーー"))
print(neologdn.normalize("うまああああい", repeat=2))  # 同じ文字の繰り返しを 2 回までに
print(neologdn.normalize("Ｃｏｋｅ ｚｅｒｏ　を　飲む"))
```

- mecab-ipadic-NEologd の正規化ルールの実装。日本語文字間の空白は消え、英単語間の空白は残る。
- NFKC とは挙動が異なる (全角記号を残すなど) ので、どちらを使うかはタスク次第。

## 絵文字を検出・除去・テキスト化する {emoji}

SNS テキストから絵文字を取り除いたり、逆に絵文字だけ取り出したり、名前に置換したりしたい。

```python
!pip install -q emoji
```

```python
import emoji

text = "今日は楽しかった😊🎉 また行きたい👨‍👩‍👧"
print(emoji.replace_emoji(text, replace=""))              # 除去
print([e["emoji"] for e in emoji.emoji_list(text)])       # 抽出
print(emoji.emoji_count(text))                            # 個数
print(emoji.demojize(text, language="ja"))                # 名前に置換
```

- 絵文字は感情の手がかりなので、感情分析では消さずに `demojize` で残す手もある。

## 見た目どおりの文字数を数える (書記素クラスタ) {grapheme}

`len()` だと絵文字や結合文字で文字数がずれる。人間が見る「1 文字」単位で数えたり切ったりしたい。

```python
!pip install -q regex
```

```python
import regex

text = "👨‍👩‍👧と🇯🇵でがんばる"
graphemes = regex.findall(r"\X", text)  # \X = 拡張書記素クラスタ
print(len(text), "←len()")
print(len(graphemes), "←書記素数")
print(graphemes)
print("先頭 3 文字:", "".join(graphemes[:3]))
```

- 文字数制限での切り詰めは書記素単位で行わないと、絵文字が壊れて文字化けする。

## 文字化けを修復する (ftfy) {ftfy}

`æ—¥æœ¬èªž` のような mojibake を元に戻したい。

```python
!pip install -q ftfy
```

```python
import ftfy

broken = "日本語".encode("utf-8").decode("latin-1")
print(broken)
print(ftfy.fix_text(broken))
print(ftfy.fix_text("The Mona Lisa doesnâ€™t have eyebrows."))
print(ftfy.fix_text("&lt;b&gt;HTML 実体参照&lt;/b&gt;"))
```

```python
# 何が起きたのかを説明させる
fixed, steps = ftfy.fix_and_explain("doesnâ€™t")
print(fixed, steps)
```

## 文字コードを判定して UTF-8 に変換する {encoding_detect}

Shift_JIS か EUC-JP か分からない古いテキストファイルを読みたい。

```python
!pip install -q charset-normalizer
```

```python
from charset_normalizer import from_bytes

text = "これは文字コードの分からない日本語のテキストファイルです。自然言語処理では昔のコーパスを扱うときによく遭遇します。"
for enc in ["shift_jis", "euc_jp", "utf-8", "iso2022_jp"]:
    raw = text.encode(enc)
    best = from_bytes(raw).best()
    print(f"{enc:10} → 判定: {best.encoding:12} 復元OK: {str(best) == text}")
```

```python
# ファイルを読むときの定番パターン
from pathlib import Path

Path("sjis.txt").write_bytes(text.encode("cp932"))
raw = Path("sjis.txt").read_bytes()
decoded = str(from_bytes(raw).best())
Path("utf8.txt").write_text(decoded, encoding="utf-8")
print(decoded[:20])
```

- Windows 由来の Shift_JIS は `cp932` で読むのが安全 (①や髙などの機種依存文字を含むため)。
- 短い文字列は判定を誤りやすい。分かっているなら明示的に `encoding=` を指定する。

## URL・メンション・ハッシュタグを除去する {remove_url}

SNS やウェブのテキストから URL や @メンションなどのノイズを取り除きたい。

```python
import re

text = "新しい論文出ました！ https://example.com/paper?id=1 @hanako さんありがとう #NLP #自然言語処理 RT @foo: すごい"

URL = re.compile(r"https?://[\w/:%#$&?()~.=+\-]+")
MENTION = re.compile(r"@\w+:?")
HASHTAG = re.compile(r"[#＃](\w+)")

t = URL.sub("", text)
t = MENTION.sub("", t)
print(HASHTAG.sub("", t))        # ハッシュタグごと消す
print(HASHTAG.sub(r"\1", t))     # # だけ消して語は残す
print(re.sub(r"\s+", " ", HASHTAG.sub(r"\1", t)).strip())  # 余分な空白も整理
```

- 日本語は URL の直後に空白なしで本文が続くことがあるので、URL のパターンは ASCII 文字に限定しておく。
- 消す代わりに `<URL>` `<USER>` のような特殊トークンに置換すると、情報を残せる。

## HTML タグと実体参照を取り除く {strip_html}

HTML からタグを除いて本文テキストだけにしたい。

```python
import html
from bs4 import BeautifulSoup

doc = """<html><head><style>p{color:red}</style><script>alert(1)</script></head>
<body><h1>見出し</h1><p>これは<b>太字</b>&amp;<a href="#">リンク</a>です。<br>改行&nbsp;あり。</p></body></html>"""

soup = BeautifulSoup(doc, "html.parser")
for tag in soup(["script", "style"]):  # 中身ごと捨てる
    tag.decompose()
for br in soup("br"):                  # 改行タグは改行文字に
    br.replace_with("\n")
for block in soup(["h1", "h2", "h3", "p", "li", "div"]):  # ブロック要素の後ろで改行
    block.append("\n")
print(soup.get_text().strip())
```

```python
# タグのない文字列の実体参照だけ戻す
print(html.unescape("A &amp; B &lt; C &#12354; &hearts;"))
```

- 正規表現 `<[^>]+>` で消す方法は `<script>` の中身が残るなど事故が多い。パーサを使う。
- ニュース記事などの「本文だけ」が欲しい場合は、データ収集の章の trafilatura を参照。

## 漢数字⇔アラビア数字を変換する {kansuji}

「三千二百万」を 32000000 に、あるいはその逆に変換したい。

```python
!pip install -q kanjize
```

```python
from kanjize import kanji2number, number2kanji

for s in ["三千二百万", "十二", "一億二千三百四十五万六千七百八十九", "2億5000万"]:
    print(s, "→", kanji2number(s))
print(number2kanji(123456789))
```

```python
# 文中の漢数字を置換する
import re

def replace_kansuji(text):
    return re.sub(r"[〇一二三四五六七八九十百千万億兆]+",
                  lambda m: str(kanji2number(m.group())), text)

print(replace_kansuji("参加者は三百二十人で、予算は五千万円です。"))
```

- 「一人」「三日月」「四国」のような語の一部まで変換されるので、形態素解析で数詞と判定されたものだけ変換するのが安全。

## 和暦⇔西暦を変換する {wareki}

「令和5年」「平成元年」を西暦に、西暦を和暦に変換したい。

```python
import re
from datetime import date

ERAS = [("令和", date(2019, 5, 1)), ("平成", date(1989, 1, 8)), ("昭和", date(1926, 12, 25)),
        ("大正", date(1912, 7, 30)), ("明治", date(1868, 1, 25))]

def wareki_to_seireki(text):
    def repl(m):
        start = dict(ERAS)[m.group(1)].year
        n = 1 if m.group(2) == "元" else int(m.group(2))
        return f"{start + n - 1}年"
    return re.sub(r"(令和|平成|昭和|大正|明治)(元|\d+)年", repl, text)

def to_wareki(d):
    for era, start in ERAS:
        if d >= start:
            n = d.year - start.year + 1
            return f"{era}{'元' if n == 1 else n}年{d.month}月{d.day}日"
    raise ValueError("明治より前は未対応")

print(wareki_to_seireki("平成元年に生まれ、令和5年に卒業した。"))
print(to_wareki(date(2019, 4, 30)), to_wareki(date(2019, 5, 1)), to_wareki(date(2026, 11, 13)))
```

- 数字が全角・漢数字の場合は、先に NFKC 正規化と漢数字変換をかけておく。

## 全角を 2 として表示幅を数える・揃える {display_width}

等幅フォントでの表示幅 (全角 2、半角 1) を数えて、日本語を含む表をきれいに揃えたい。

```python
import unicodedata

def width(s):
    return sum(2 if unicodedata.east_asian_width(c) in "FWA" else 1 for c in s)

def ljust(s, n):
    return s + " " * max(0, n - width(s))

rows = [("単語", "品詞", "頻度"), ("自然言語処理", "名詞", "120"), ("NLP", "名詞", "85"), ("ﾊﾝｶｸ", "名詞", "3")]
for a, b, c in rows:
    print(ljust(a, 14) + ljust(b, 6) + c)
```

- `east_asian_width` は F (全角)・W (広い)・A (曖昧)・H (半角)・Na (狭い)・N (中立) を返す。A (ギリシャ文字や罫線など) を 2 とするかは環境次第。
- より厳密には `wcwidth` パッケージを使う。

## 繰り返し文字を縮約する (「すごーーーい」「wwww」) {squeeze_repeats}

SNS テキストの誇張表現を縮めて、語彙のばらつきを減らしたい。

```python
import re

def squeeze(text, max_repeat=2):
    # 同じ 1 文字が max_repeat 回を超えて続いたら max_repeat 回に
    text = re.sub(r"(.)\1{%d,}" % max_repeat, lambda m: m.group(1) * max_repeat, text)
    # 「ww」「笑笑」のような 2 文字以上の繰り返しにも対応
    text = re.sub(r"(..+?)\1{%d,}" % max_repeat, lambda m: m.group(1) * max_repeat, text)
    return text

for s in ["すごーーーーーい！！！！！", "草wwwwwwwww", "やばいやばいやばいやばい", "ははははは", "1000000 円"]:
    print(s, "→", squeeze(s))
```

- 数字の `000000` も縮むので、数字は対象外にする (`(\D)\1{2,}`) などタスクに合わせて調整する。

## 漢字かな交じり文をローマ字・ひらがなにする {romaji}

日本語の文をローマ字 (ヘボン式) や、ひらがなだけの文に変換したい。

```python
!pip install -q pykakasi
```

```python
import pykakasi

kks = pykakasi.kakasi()
result = kks.convert("自然言語処理は楽しい学問です")
for item in result:
    print(item["orig"], item["hira"], item["kana"], item["hepburn"], sep="\t")

print(" ".join(item["hepburn"] for item in result))
print("".join(item["hira"] for item in result))
```

- 辞書ベースなので文脈による読み分け (「今日」きょう/こんにち) は苦手。精度が要るなら形態素解析器の読みを使う (形態素解析の章を参照)。

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

## 文字 n-gram を作る {char_ngram}

単語分割に頼らずに、文字 bigram / trigram を特徴量や類似度計算に使いたい。

```python
def char_ngrams(text, n=2, pad=False):
    if pad:
        text = "^" * (n - 1) + text + "$" * (n - 1)
    return [text[i:i + n] for i in range(len(text) - n + 1)]

print(char_ngrams("自然言語処理", 2))
print(char_ngrams("自然言語処理", 3))
print(char_ngrams("言語", 2, pad=True))
```

```python
# 文字 bigram の Jaccard 係数で表記ゆれに強い類似度を測る
def jaccard(a, b, n=2):
    A, B = set(char_ngrams(a, n)), set(char_ngrams(b, n))
    return len(A & B) / len(A | B)

print(jaccard("コンピューター", "コンピュータ"))
print(jaccard("コンピューター", "インターネット"))
```

- scikit-learn なら `CountVectorizer(analyzer="char", ngram_range=(2, 3))` で同じことができる。

## 日付・時刻の表現を datetime に変換する {date_parse}

「2024年3月5日」「3日前」「明日の15時」のような表現を機械的に扱える日付にしたい。

```python
!pip install -q dateparser
```

```python
from datetime import datetime
import dateparser

base = datetime(2026, 11, 13, 12, 0)
settings = {"RELATIVE_BASE": base, "PREFER_DATES_FROM": "past"}
for s in ["2024年3月5日", "令和6年3月5日", "3日前", "昨日", "明日の15時", "2週間後", "March 5, 2024", "5 mars 2024"]:
    print(f"{s:14} → {dateparser.parse(s, languages=['ja', 'en', 'fr'], settings=settings)}")
```

```python
# 文中から日付表現を探す
from dateparser.search import search_dates

print(search_dates("締切は2026年12月1日で、発表は2027年3月10日です。", languages=["ja"]))
```

- 解釈できない表現は `None` が返る。和暦などは前処理 (和暦→西暦のレシピ) と組み合わせる。

## 電話番号・郵便番号・メールアドレスを正規表現で抜き出す {regex_patterns}

テキストから定型的な文字列を抽出したり、個人情報としてマスクしたりしたい。

```python
import re
import unicodedata

PATTERNS = {
    "郵便番号": r"(?<![\d-])\d{3}-\d{4}(?![\d-])",
    "電話番号": r"(?<!\d)0\d{1,4}-\d{1,4}-\d{4}(?!\d)",
    "メール": r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
    "金額": r"(?:¥|￥)?\d{1,3}(?:,\d{3})+円?|\d+円",
}

text = "〒１９１-００６５ 東京都日野市。電話 042-585-0000、携帯 090-1234-5678、連絡先 taro.yamada+nlp@example.ac.jp、参加費 3,000円"
text = unicodedata.normalize("NFKC", text)  # 先に全角を半角へ
for name, pat in PATTERNS.items():
    print(name, re.findall(pat, text))
```

```python
def mask(text):
    for name, pat in PATTERNS.items():
        if name != "金額":
            text = re.sub(pat, f"<{name}>", text)
    return text

print(mask(text))
```

- 氏名や住所は正規表現では取れない。固有表現抽出の章を参照。

## テキストの言語 (文字種) の割合を調べる {script_ratio}

コーパスに日本語以外の行やゴミが混ざっていないか、文字種の割合でざっくりフィルタしたい。

```python
!pip install -q regex
```

```python
import regex

def script_ratio(text):
    chars = regex.findall(r"\S", text)
    scripts = {"かな": r"[\p{Hiragana}\p{Katakana}ー]", "漢字": r"\p{Han}", "ラテン": r"\p{Latin}",
               "ハングル": r"\p{Hangul}", "数字": r"\p{N}", "記号": r"[\p{P}\p{S}]"}
    return {k: round(len(regex.findall(p, text)) / max(1, len(chars)), 2) for k, p in scripts.items()}

lines = ["自然言語処理は面白い。", "This is an English sentence.", "这是中文句子。", "한국어 문장입니다.", "★☆★☆ 12345 !!!"]
for line in lines:
    r = script_ratio(line)
    is_ja = r["かな"] > 0.1  # かなを含めば日本語とみなす簡易判定
    print(f"{'日本語' if is_ja else '------'}  {line}  {r}")
```

- 中国語と日本語は漢字だけでは区別できないが、「かなを含むか」でほぼ判別できる。
- きちんとした言語判定は英語・多言語の前処理の章 (言語判定のレシピ) を参照。

## ゼロ幅文字・制御文字・異体字セレクタを取り除く {invisible_chars}

見た目は同じなのに文字列が一致しない原因になる、目に見えない文字を検出・除去したい。

```python
import unicodedata

text = "自然​言語﻿処理 と　葛\U000e0100飾区\x07"
for c in text:
    cat = unicodedata.category(c)
    if cat[0] in "CZM":
        print(f"U+{ord(c):04X} {cat} {unicodedata.name(c, '(no name)')}")
```

```python
def remove_invisible(text):
    out = []
    for c in text:
        cat = unicodedata.category(c)
        if cat in ("Cc", "Cf") and c not in "\n\t":   # 制御文字・書式文字 (ゼロ幅スペース, BOM など)
            continue
        if 0xFE00 <= ord(c) <= 0xFE0F or 0xE0100 <= ord(c) <= 0xE01EF:  # 異体字セレクタ
            continue
        out.append(" " if cat == "Zs" else c)          # 各種スペースを半角スペースに
    return "".join(out)

clean = remove_invisible(text)
print(repr(clean), len(text), "→", len(clean))
```

- `Cf` を消すと絵文字の ZWJ (U+200D) も消えて家族の絵文字などが分解される。絵文字を残したいなら U+200D は除外する。

## 編集距離とあいまい文字列マッチング (RapidFuzz) {edit_distance}

表記ゆれやタイプミスを含む文字列同士の近さを測りたい。候補リストから一番近いものを選びたい。

```python
!pip install -q rapidfuzz
```

```python
from rapidfuzz import fuzz, process
from rapidfuzz.distance import DamerauLevenshtein, JaroWinkler, Levenshtein

for a, b in [("コンピューター", "コンピュータ"), ("自然言語処理", "自然原語処理"), ("recieve", "receive"), ("東京都立大学", "首都大学東京")]:
    print(f"{a} / {b}: 編集距離 {Levenshtein.distance(a, b)}  正規化類似度 {Levenshtein.normalized_similarity(a, b):.2f}"
          f"  転置込み {DamerauLevenshtein.distance(a, b)}  Jaro-Winkler {JaroWinkler.similarity(a, b):.2f}")
print(Levenshtein.editops("recieve", "receive"))
```

```python
# 候補の中から近いものを探す
universities = ["東京大学", "京都大学", "東京工業大学", "東京都立大学", "一橋大学", "大阪大学", "東京科学大学", "東北大学"]
for q in ["東京都率大学", "ひとつばし大学", "東工大"]:
    print(q, process.extract(q, universities, scorer=fuzz.WRatio, limit=2))
```

```python
# 単語単位の編集距離 (リストを渡す)
ref, hyp = "the cat sat on the mat".split(), "the cat sit on mat".split()
print(Levenshtein.distance(ref, hyp), Levenshtein.editops(ref, hyp))
```

- 「東工大」のような略称や「ひとつばし」のような読みの違いは、文字の編集距離では拾えない。読みに変換してから比べるか、別名辞書を使う。
- RapidFuzz は C++ 実装で、数百万件の突き合わせにも使える (`process.cdist` で行列計算)。

## MinHash と LSH で大量文書からほぼ重複を見つける {minhash_lsh}

数十万件の文書から、コピペや微修正による「ほぼ同じ文書」を、全ペア比較せずに高速に見つけたい。

```python
!pip install -q datasketch
```

```python
from datasketch import MinHash, MinHashLSH

def shingles(text, n=3):
    text = "".join(text.split())
    return {text[i:i + n] for i in range(len(text) - n + 1)}

def minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for s in shingles(text):
        m.update(s.encode("utf-8"))
    return m

docs = {
    "a": "自然言語処理は、人間が日常的に使っている言語をコンピュータに処理させる一連の技術である。",
    "b": "自然言語処理とは、人間が日常的に使っている言語をコンピューターに処理させる技術のことである。",
    "c": "機械学習は、データから規則性を学習してタスクを解くアルゴリズムの研究分野である。",
    "d": "自然言語処理は、人間が日常的に使っている言語をコンピュータに処理させる一連の技術である！",
}
mh = {k: minhash(v) for k, v in docs.items()}

def jaccard(a, b):
    A, B = shingles(docs[a]), shingles(docs[b])
    return len(A & B) / len(A | B)

for x, y in [("a", "b"), ("a", "c"), ("a", "d")]:
    print(f"{x}-{y}: 真の Jaccard = {jaccard(x, y):.2f}  MinHash による推定 = {mh[x].jaccard(mh[y]):.2f}")
```

```python
lsh = MinHashLSH(threshold=0.5, num_perm=128)
for k, m in mh.items():
    lsh.insert(k, m)
for k in docs:
    print(k, "と似ている:", sorted(set(lsh.query(mh[k])) - {k}))
```

- MinHash は集合の Jaccard 係数を固定長の署名で近似する。LSH は署名が似たものだけを同じバケットに入れ、候補を絞る。
- 大規模な事前学習コーパスの重複除去で標準的に使われている。意味的な重複 (言い換え) は拾えないので、それには埋め込みを使う。

## 巨大な辞書の語をテキストから一括で探す (Aho-Corasick) {aho_corasick}

数万〜数百万語の辞書 (商品名、病名、NG ワードなど) に載っている語を、長いテキストから高速にすべて見つけたい。

```python
!pip install -q pyahocorasick
```

```python
import ahocorasick

terms = {"東京": "LOC", "東京都": "LOC", "京都": "LOC", "東京都立大学": "ORG", "自然言語処理": "TERM", "言語": "TERM", "機械翻訳": "TERM"}
A = ahocorasick.Automaton()
for term, label in terms.items():
    A.add_word(term, (term, label))
A.make_automaton()

text = "東京都立大学では自然言語処理と機械翻訳を研究している。京都にも行った。"
for end, (term, label) in A.iter(text):          # 重なりも含めてすべての一致
    print(end - len(term) + 1, end + 1, term, label)
```

```python
# 最長一致だけ残す (重なったら長い語を優先)
for end, (term, label) in A.iter_long(text):
    print(end - len(term) + 1, end + 1, term, label)
```

```python
# 正規表現の | 連結と速度を比べる
import random, re, time

random.seed(0)
vocab = ["".join(random.choices("あいうえおかきくけこさしすせそ", k=random.randint(3, 6))) for _ in range(50000)]
big = "".join(random.choices("あいうえおかきくけこさしすせそ", k=200000))
B = ahocorasick.Automaton()
for w in vocab:
    B.add_word(w, w)
B.make_automaton()

t = time.perf_counter(); n1 = sum(1 for _ in B.iter(big)); t1 = time.perf_counter() - t
t = time.perf_counter(); n2 = sum(1 for w in vocab[:500] if w in big); t2 = time.perf_counter() - t
print(f"Aho-Corasick (5 万語): {t1:.3f} 秒 / 単純な in 判定 (500 語だけ): {t2:.3f} 秒")
```

- テキストを 1 回走査するだけで、辞書の語数に関係なくすべての一致を列挙できる。
- 「東京都」の中の「京都」のような誤一致は最長一致でも完全には防げない。形態素境界と照合すると精度が上がる。
