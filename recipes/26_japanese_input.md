# 日本語入力 (かな漢字変換)

ローマ字からかなへの変換、辞書とラティス、ビタビアルゴリズムによるかな漢字変換、候補の提示、予測入力、履歴の学習、ニューラルかな漢字変換、評価まで、日本語入力システムの部品を作る。

## ローマ字をかなに変換する {romaji_to_kana}

キーボードから打たれたローマ字の列を、ひらがなに変換したい。促音 (っ) や撥音 (ん) の規則も正しく扱いたい。

```python
!pip install -q jaconv
```

```python
# 変換表: 長い綴りを優先して照合する (最長一致)
TABLE = {
    "a": "あ", "i": "い", "u": "う", "e": "え", "o": "お",
    "ka": "か", "ki": "き", "ku": "く", "ke": "け", "ko": "こ", "sa": "さ", "si": "し", "shi": "し", "su": "す", "se": "せ", "so": "そ",
    "ta": "た", "ti": "ち", "chi": "ち", "tu": "つ", "tsu": "つ", "te": "て", "to": "と", "na": "な", "ni": "に", "nu": "ぬ", "ne": "ね", "no": "の",
    "ha": "は", "hi": "ひ", "hu": "ふ", "fu": "ふ", "he": "へ", "ho": "ほ", "ma": "ま", "mi": "み", "mu": "む", "me": "め", "mo": "も",
    "ya": "や", "yu": "ゆ", "yo": "よ", "ra": "ら", "ri": "り", "ru": "る", "re": "れ", "ro": "ろ", "wa": "わ", "wo": "を", "nn": "ん",
    "ga": "が", "gi": "ぎ", "gu": "ぐ", "ge": "げ", "go": "ご", "za": "ざ", "zi": "じ", "ji": "じ", "zu": "ず", "ze": "ぜ", "zo": "ぞ",
    "da": "だ", "di": "ぢ", "du": "づ", "de": "で", "do": "ど", "ba": "ば", "bi": "び", "bu": "ぶ", "be": "べ", "bo": "ぼ",
    "pa": "ぱ", "pi": "ぴ", "pu": "ぷ", "pe": "ぺ", "po": "ぽ", "-": "ー", ",": "、", ".": "。",
    "la": "ぁ", "li": "ぃ", "lu": "ぅ", "le": "ぇ", "lo": "ぉ", "xtu": "っ", "ltu": "っ",
}
for c, kana in [("k", "き"), ("s", "し"), ("t", "ち"), ("n", "に"), ("h", "ひ"), ("m", "み"), ("r", "り"), ("g", "ぎ"), ("z", "じ"), ("b", "び"), ("p", "ぴ"), ("d", "ぢ")]:
    for v, small in [("a", "ゃ"), ("u", "ゅ"), ("o", "ょ")]:
        TABLE[c + "y" + v] = kana + small                      # kya → きゃ など
for a, kana in [("sh", "し"), ("ch", "ち"), ("j", "じ")]:
    for v, small in [("a", "ゃ"), ("u", "ゅ"), ("o", "ょ")]:
        TABLE[a + v] = kana + small                            # sha → しゃ、ja → じゃ など
MAX_LEN = max(map(len, TABLE))

def romaji_to_kana(text):
    text, out, i = text.lower(), [], 0
    while i < len(text):
        # 促音: 同じ子音が続く (kk, tt, ss ...)。ただし nn は「ん」。ヘボン式の tch (matcha) も促音
        if i + 1 < len(text) and (text[i] == text[i + 1] and text[i] in "kstpgzdbhfmrwcj" or text[i:i + 3] == "tch"):
            out.append("っ"); i += 1; continue
        for n in range(MAX_LEN, 0, -1):                        # 最長一致
            if text[i:i + n] in TABLE:
                out.append(TABLE[text[i:i + n]]); i += n; break
        else:
            if text[i] == "n" and (i + 1 == len(text) or text[i + 1] not in "aiueoy"):
                out.append("ん")                               # 母音や y が続かない n は「ん」
            else:
                out.append(text[i])                            # 変換できない文字はそのまま (入力途中の子音など)
            i += 1
    return "".join(out)

for s in ["kyouhaiitenkidesu", "gakkouniitta", "shinbunwoyomu", "konnnichiha", "kinen", "kinnen", "matcha", "chottomatte", "ko-hi-", "nihongonyuuryoku", "sizengengosyori", "k"]:
    print(f"{s:20} → {romaji_to_kana(s)}")
```

```python
# ライブラリを使う場合 / 逆方向 (かな → ローマ字)
import jaconv

for s in ["kyouhaiitenkidesu", "gakkouniitta", "kinen", "kinnen", "chottomatte"]:
    print(f"{s:20} → {jaconv.alphabet2kana(s)}")
print(jaconv.kana2alphabet("にほんごにゅうりょく"), "/", jaconv.kata2alphabet("カナカンジヘンカン"))
```

- 1 つのかなに複数の綴り (si / shi、tu / tsu、zi / ji) があり、利用者によって違う。実際の入力システムは、ヘボン式・訓令式・独自の綴りをすべて受け付ける表を持つ。
- 「ん」の扱いが難しい。`kinen` は「きねん」、`kinnen` は「きんえん」、`konnichiha` と `konnnichiha` は利用者の習慣で分かれる。次の文字を見ないと決められない。
- 入力途中の未確定の子音 (最後の `k`) は、変換せずに残しておく。1 文字打つたびに先頭から変換し直せば、表示は常に正しくなる。
- 表を差し替えれば、かな入力や AZIK などの拡張ローマ字入力にも同じ仕組みで対応できる。

## 打ち間違いを許してローマ字をかなに変換する {romaji_fuzzy}

ローマ字入力で、キーの打ち間違いや文字の抜けがあっても、意図した単語を推定して変換したい。日本語を学び始めた人の、つづりの誤りにも対応したい。

```python
!pip install -q jaconv fugashi unidic-lite
```

```python
import heapq, io, re, urllib.request, zipfile
from collections import Counter
import jaconv
from fugashi import Tagger

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text)

# 単語の読み (かな) をローマ字にした辞書を作る: ローマ字 → {(かな, 表記): 頻度}
tagger = Tagger()
base = "https://www.aozora.gr.jp/cards/000148/files/"
words = Counter()
for f in ["789_ruby_5639.zip", "752_ruby_2438.zip", "773_ruby_5968.zip"]:
    for w in tagger(aozora(base + f)):
        if w.feature.kana and w.feature.pos1 in ("名詞", "動詞", "形容詞", "副詞", "形状詞") and re.fullmatch(r"[ァ-ヶー]+", w.feature.kana):
            words[jaconv.kata2hira(w.feature.kana), w.surface] += 1
lexicon = {}
for (kana, surface), n in words.items():
    if n >= 3 and len(kana) >= 2:
        lexicon.setdefault(jaconv.kana2alphabet(kana), Counter())[kana, surface] += n
print(len(lexicon), "種類のローマ字つづり", list(lexicon.items())[:3])
```

```python
QWERTY = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
NEIGHBORS = {c: set() for row in QWERTY for c in row}
for r, row in enumerate(QWERTY):
    for i, c in enumerate(row):
        NEIGHBORS[c] |= {row[j] for j in (i - 1, i + 1) if 0 <= j < len(row)}
        for other in (r - 1, r + 1):
            if 0 <= other < len(QWERTY):
                NEIGHBORS[c] |= {QWERTY[other][j] for j in (i - 1, i, i + 1) if 0 <= j < len(QWERTY[other])}

def cost_substitute(a, b):
    """置換のコスト: キーボードで隣のキーなら安く、母音どうしの取り違えも少し安くする"""
    if a == b:
        return 0.0
    if b in NEIGHBORS.get(a, ()):
        return 0.5
    return 0.8 if a in "aiueo" and b in "aiueo" else 1.0

def weighted_distance(typed, target, limit=2.5):
    """重み付き編集距離 (挿入・削除は 1、隣接文字の入れ替えは 0.5)。limit を超えたら打ち切る"""
    prev2, prev = None, [j * 1.0 for j in range(len(target) + 1)]
    for i in range(1, len(typed) + 1):
        cur = [i * 1.0]
        for j in range(1, len(target) + 1):
            best = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost_substitute(typed[i - 1], target[j - 1]))
            if i > 1 and j > 1 and typed[i - 1] == target[j - 2] and typed[i - 2] == target[j - 1]:
                best = min(best, prev2[j - 2] + 0.5)
            cur.append(best)
        if min(cur) > limit:
            return float("inf")
        prev2, prev = prev, cur
    return prev[-1]

print(weighted_distance("gakkou", "gakkou"), weighted_distance("gakkpu", "gakkou"), weighted_distance("gakkuo", "gakkou"), weighted_distance("gakou", "gakkou"), weighted_distance("gakkau", "gakkou"))
```

```python
import math

def convert(typed, k=5):
    """距離が小さく、頻度が高い語を上位にする (スコア = 距離 − 頻度の対数の重み付き)"""
    scored = []
    for romaji, entries in lexicon.items():
        if abs(len(romaji) - len(typed)) <= 2:
            d = weighted_distance(typed, romaji)
            if d < float("inf"):
                for (kana, surface), n in entries.items():
                    scored.append((d - 0.15 * math.log(n), d, romaji, kana, surface))
    return [(surface, kana, romaji, d) for _, d, romaji, kana, surface in heapq.nsmallest(k, scored)]

for typed in ["gakkou", "gakkpu", "gakou", "sennsei", "sensie", "tomodati", "tomodachi", "benkyo", "kyuo", "arigatou", "muzukasii", "muzukashi"]:
    print(f"{typed:10} → {[(s, k) for s, k, _, d in convert(typed)][:4]}")
```

```python
# 学習者に多いつづりの誤り (長音・促音・撥音の抜け) を、編集距離とは別に規則で候補にする
def learner_variants(typed):
    """長音の「u」や促音の子音の重なりが抜けた形から、補った候補を作る"""
    out = {typed}
    for i in range(len(typed)):
        if typed[i] in "ou":
            out.add(typed[:i + 1] + "u" + typed[i + 1:])                      # benkyo → benkyou
        if typed[i] in "kstp" and i > 0 and typed[i - 1] in "aiueo":
            out.add(typed[:i] + typed[i] + typed[i:])                         # gako → gakko
        if typed[i] == "n" and i + 1 < len(typed) and typed[i + 1] in "aiueoy":
            out.add(typed[:i] + "n" + typed[i:])                              # konichiwa → konnichiwa
    return out

def learner_variants_2(typed):
    """規則を 2 回まで重ねて適用する (gako → gakko → gakkou のように、誤りが 2 つある場合)"""
    return {v2 for v1 in learner_variants(typed) for v2 in learner_variants(v1)}

for typed in ["benkyo", "gako", "kino", "ryoko", "kekon", "sensei"]:
    hits = Counter()
    for v in learner_variants_2(typed):
        for (kana, surface), n in lexicon.get(v, {}).items():
            hits[surface, kana, v] += n
    print(typed, "→", hits.most_common(4))
```

- 打ち間違いには傾向がある: 隣のキーを押す、2 文字の順序が入れ替わる、同じ文字を打ち損ねる・重ねる。操作ごとにコストを変えた編集距離を使うと、ただの編集距離より当たりやすくなる。
- 候補の順位は、「つづりの近さ」と「単語の出やすさ (頻度)」の両方で決める。雑音のある通信路モデルという考え方で、スペル訂正と同じ枠組みになる。
- 日本語学習者は、打ち間違いではなく、発音の聞き取りに由来する誤りをする (長音・促音・撥音の抜け、清音と濁音の取り違え)。キーボードの配置とは関係がないので、別の規則や重みを用意する。
- 辞書のすべての語と距離を計算するのは遅い。実用では、トライ木の上で編集距離を計算する、先頭の数文字で候補を絞る、SymSpell のような索引を使う、といった高速化をする。
- ここでは単語 1 つを対象にした。文全体では、かな漢字変換のラティスに「誤りを含む読み」の候補も加え、言語モデルでまとめて選ぶ。

## 読みから表記を引く辞書をコーパスから作る {kkc_dictionary}

かな漢字変換のための「読み → 表記の候補」の辞書を作りたい。入力の先頭から、辞書にある読みをすべて高速に見つけたい。

```python
!pip install -q fugashi unidic-lite jaconv marisa-trie
```

```python
import io, re, urllib.request, zipfile
from collections import Counter, defaultdict
import jaconv
from fugashi import Tagger

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

# コーパスを形態素解析して、(表記, 読み) の列にする。読みは形態素解析器が付けたものを使う
tagger = Tagger()
base = "https://www.aozora.gr.jp/cards/000148/files/"
sentences = []
for f in ["789_ruby_5639.zip", "752_ruby_2438.zip", "773_ruby_5968.zip"]:      # 吾輩は猫である・坊っちゃん・こころ
    for s in re.split(r"(?<=。)|\n", aozora(base + f)):
        if s.strip():
            sentences.append([(w.surface, jaconv.kata2hira(w.feature.kana or w.surface)) for w in tagger(s.strip())])
print(len(sentences), "文", sum(map(len, sentences)), "語")
print(sentences[2])
```

```python
# 辞書: 読み → {表記: 出現回数}
dic = defaultdict(Counter)
for sent in sentences:
    for surface, reading in sent:
        dic[reading][surface] += 1
print(len(dic), "種類の読み")
for reading in ["こうしょう", "かんじ", "きく", "は", "かける"]:
    print(reading, dic[reading].most_common(6))

ambiguity = Counter(len(v) for v in dic.values())
print("表記の候補が 1 つだけの読み:", ambiguity[1], "/ 2 つ以上:", sum(c for n, c in ambiguity.items() if n > 1))
```

```python
# トライ木に入れて、入力の各位置から始まる辞書の読みをすべて列挙する (共通接頭辞検索)
import marisa_trie

trie = marisa_trie.Trie(dic.keys())
text = "わがはいはねこである"
for i in range(len(text)):
    print(f"{i:2} {text[i:]:　<10} {trie.prefixes(text[i:])}")
```

```python
import time

t = time.perf_counter()
n = sum(len(trie.prefixes(text[i:])) for _ in range(10000) for i in range(len(text)))
print(f"共通接頭辞検索 {10000 * len(text)} 回: {time.perf_counter() - t:.2f} 秒")
trie.save("reading.trie")
```

- かな漢字変換の辞書は「読みをキーにして表記の候補を引く」もの。形態素解析の辞書 (表記をキーにして読みや品詞を引く) の逆向きにあたる。
- 同じ読みに多くの表記がある (同音異義語) ことが、かな漢字変換の難しさの中心。どれを選ぶかは、頻度と前後の文脈で決める。
- 共通接頭辞検索は「この位置から始まる単語の候補をすべて挙げる」操作で、次項のラティス (単語の候補のグラフ) を作るのに使う。トライ木なら入力の長さに比例した時間で済む。
- 実用の辞書は数十万〜数百万語。Mozc の辞書、SKK 辞書、Sudachi や UniDic の語彙に読みを付けたものなどが元になる。

## 単語の頻度だけでかな漢字変換をする (ビタビアルゴリズム) {kkc_unigram}

ひらがなの列を、漢字かな交じりの文に変換したい。最も単純な方法として、単語の出現頻度だけを使って、全体として最も確からしい単語の並びを選ぶ。

```python
!pip install -q fugashi unidic-lite jaconv marisa-trie
```

```python
import io, math, re, urllib.request, zipfile
from collections import Counter, defaultdict
import jaconv
import marisa_trie
from fugashi import Tagger

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

tagger = Tagger()
base = "https://www.aozora.gr.jp/cards/000148/files/"
sentences = []
for f in ["789_ruby_5639.zip", "752_ruby_2438.zip", "773_ruby_5968.zip"]:
    for s in re.split(r"(?<=。)|\n", aozora(base + f)):
        if s.strip():
            sentences.append([(w.surface, jaconv.kata2hira(w.feature.kana or w.surface)) for w in tagger(s.strip())])

dic = defaultdict(Counter)                    # 読み → {表記: 回数}
for sent in sentences:
    for surface, reading in sent:
        dic[reading][surface] += 1
total = sum(sum(c.values()) for c in dic.values())
trie = marisa_trie.Trie(dic.keys())
```

```python
UNKNOWN_COST = 20.0        # 辞書にない 1 文字をそのまま通すときのコスト

def lattice(kana):
    """各位置から始まる単語の候補 (終了位置, 表記, コスト) を列挙する。コスト = −log 確率"""
    edges = defaultdict(list)
    for i in range(len(kana)):
        for reading in trie.prefixes(kana[i:]):
            for surface, count in dic[reading].items():
                edges[i].append((i + len(reading), surface, -math.log(count / total)))
        edges[i].append((i + 1, kana[i], UNKNOWN_COST))      # 未知語: 1 文字をひらがなのまま出す
    return edges

def convert(kana):
    """ビタビアルゴリズム: 各位置までの最小コストを前から順に求め、最後から経路をたどる"""
    edges = lattice(kana)
    best = [(math.inf, None, None)] * (len(kana) + 1)        # (コスト, 直前の位置, 表記)
    best[0] = (0.0, None, None)
    for i in range(len(kana)):
        if best[i][0] == math.inf:
            continue
        for j, surface, cost in edges[i]:
            if best[i][0] + cost < best[j][0]:
                best[j] = (best[i][0] + cost, i, surface)
    words, pos = [], len(kana)
    while pos > 0:
        _, prev, surface = best[pos]
        words.append(surface); pos = prev
    return words[::-1], best[len(kana)][0]

for kana in ["わがはいはねこである", "なまえはまだない", "きょうはいいてんきだ", "がっこうへいく", "せんせいにてがみをかく", "はしのうえではしをおとした", "きしゃがきしゃできしゃした"]:
    words, cost = convert(kana)
    print(f"{kana} → {'|'.join(words)}  (コスト {cost:.1f})")
```

```python
# ラティスの中身を見る: 「はし」の位置にどんな候補があるか
edges = lattice("はしをわたる")
for i in [0, 3]:
    print(i, sorted([(s, round(c, 1), j) for j, s, c in edges[i]], key=lambda x: x[1])[:8])
```

- 入力の各位置から始まる単語の候補を並べたグラフをラティスと呼ぶ。変換は、ラティスの左端から右端までの最もコストの低い経路を探す問題になる。
- 単語の頻度だけ (unigram) だと、同音異義語は文脈に関係なく、常に最も頻度の高い表記になる (「きしゃ」はすべて「汽車」)。
- 短い単語を細かくつなぐ経路と、長い単語 1 つの経路のどちらが選ばれるかは、コストの合計だけで決まる。頻度の高い助詞をつないだ「は|し」が「橋」や「箸」に勝ったり、「は|まだ」より「浜田」が選ばれたりするのは、単語の並びの自然さを見ていないため。
- コーパスが明治の小説なので、現代の語 (「天気」はあっても「スマホ」はない) や文体には弱い。辞書とコーパスが変換の質を決める。

## 前後の単語を考慮してかな漢字変換をする (bigram 言語モデル) {kkc_bigram}

同音異義語を文脈に合わせて選びたい。直前の単語との並びやすさ (bigram) を使って変換する。

```python
!pip install -q fugashi unidic-lite jaconv marisa-trie
```

```python
import io, math, re, urllib.request, zipfile
from collections import Counter, defaultdict
import jaconv
import marisa_trie
from fugashi import Tagger

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

tagger = Tagger()
base = "https://www.aozora.gr.jp/cards/000148/files/"
sentences = []
for f in ["789_ruby_5639.zip", "752_ruby_2438.zip", "773_ruby_5968.zip"]:
    for s in re.split(r"(?<=。)|\n", aozora(base + f)):
        if s.strip():
            sentences.append([(w.surface, jaconv.kata2hira(w.feature.kana or w.surface)) for w in tagger(s.strip())])
train, test = sentences[:-300], sentences[-300:]

dic, unigram, bigram = defaultdict(set), Counter(), Counter()
for sent in train:
    prev = "<s>"
    unigram[prev] += 1
    for surface, reading in sent:
        dic[reading].add(surface)
        unigram[surface] += 1
        bigram[prev, surface] += 1
        prev = surface
    bigram[prev, "</s>"] += 1
total, trie = sum(unigram.values()), marisa_trie.Trie(dic.keys())
print(len(train), "文で学習 /", len(unigram), "語 /", len(bigram), "種類の bigram")
```

```python
LAMBDA, UNKNOWN_COST = 0.8, 20.0

def cost(prev, word, lam=LAMBDA):
    """−log P(word | prev)。bigram と unigram を混ぜる (線形補間) ことで、見たことのない並びにも確率を与える"""
    p_uni = unigram.get(word, 0) / total
    p_bi = bigram.get((prev, word), 0) / unigram[prev] if unigram.get(prev) else 0.0
    p = lam * p_bi + (1 - lam) * p_uni
    return -math.log(p) if p > 0 else UNKNOWN_COST

def convert(kana, lam=LAMBDA):
    # best[位置][その位置で終わる単語] = (コスト, 直前の位置, 直前の単語)。直前の単語ごとに最小コストを覚えておく
    best = [dict() for _ in range(len(kana) + 1)]
    best[0]["<s>"] = (0.0, None, None)
    for i in range(len(kana)):
        if not best[i]:
            continue
        candidates = [(i + len(r), s) for r in trie.prefixes(kana[i:]) for s in dic[r]] + [(i + 1, kana[i])]
        for j, word in candidates:
            for prev, (c, _, _) in best[i].items():
                new = c + cost(prev, word, lam)
                if word not in best[j] or new < best[j][word][0]:
                    best[j][word] = (new, i, prev)
    word = min(best[-1], key=lambda w: best[-1][w][0] + cost(w, "</s>", lam))
    words, pos = [], len(kana)
    while pos > 0:
        _, prev_pos, prev = best[pos][word]
        words.append(word); pos, word = prev_pos, prev
    return words[::-1]

for kana in ["あめがふってきた", "かみをきった", "ひをつける", "はをみがく", "かぜをひく", "きがつく", "はながさく", "はしでたべる", "はしをわたる", "くすりがきく"]:
    print(f"{kana} → unigram: {''.join(convert(kana, lam=0.0))} / bigram: {''.join(convert(kana))}")
```

```python
# 学習に使っていない 300 文で、文がまるごと正しく変換できた割合を比べる
def sentence_accuracy(lam, n=300):
    ok = 0
    for sent in test[:n]:
        kana, gold = "".join(r for _, r in sent), "".join(s for s, _ in sent)
        ok += "".join(convert(kana, lam)) == gold
    return ok / n

for lam in [0.0, 0.5, 0.8, 0.95]:
    print(f"lambda = {lam}: 文正解率 {sentence_accuracy(lam):.3f}")
```

- bigram では、「髪を → 切っ」「火を → つける」「風邪を → 引く」のように、直前の単語との並びやすさで表記が選ばれる。unigram では区別できなかった同音異義語が、文脈で決まるようになる。
- 直前の 1 語しか見ないので、「はしを わたる」の「はし」は後ろの「渡る」を手がかりにできず「箸」になる。学習データにない並び (「薬が効く」) も当てられない。
- ビタビアルゴリズムの状態が「位置」から「位置と、そこで終わる単語」の組に増える。計算量は候補数の 2 乗に比例するが、文が短いので実用上は問題ない。
- 学習データにない単語の並びの確率が 0 にならないよう、unigram と混ぜる (平滑化)。混ぜる割合は、学習に使っていないデータで決める。
- 実用の変換エンジン (Mozc など) は、単語そのものではなく品詞や単語クラスの bigram を使う。単語 bigram より少ないデータで学習でき、メモリも小さくて済む。
- 文正解率は厳しい指標 (1 か所でも違えば不正解)。部分的な正しさも測る指標は、この章の評価のレシピを参照。

## 変換候補を複数出す (N-best) {kkc_nbest}

第 1 候補が間違っていたときに、利用者が選び直せるよう、変換候補を確からしい順に複数出したい。

```python
!pip install -q fugashi unidic-lite jaconv marisa-trie
```

```python
import heapq, io, math, re, urllib.request, zipfile
from collections import Counter, defaultdict
import jaconv
import marisa_trie
from fugashi import Tagger

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

tagger = Tagger()
base = "https://www.aozora.gr.jp/cards/000148/files/"
dic, unigram, bigram = defaultdict(set), Counter(), Counter()
for f in ["789_ruby_5639.zip", "752_ruby_2438.zip", "773_ruby_5968.zip"]:
    for s in re.split(r"(?<=。)|\n", aozora(base + f)):
        if not s.strip():
            continue
        prev = "<s>"; unigram[prev] += 1
        for w in tagger(s.strip()):
            surface, reading = w.surface, jaconv.kata2hira(w.feature.kana or w.surface)
            dic[reading].add(surface); unigram[surface] += 1; bigram[prev, surface] += 1; prev = surface
        bigram[prev, "</s>"] += 1
total, trie = sum(unigram.values()), marisa_trie.Trie(dic.keys())

def cost(prev, word, lam=0.8):
    p = lam * (bigram.get((prev, word), 0) / unigram[prev] if unigram.get(prev) else 0.0) + (1 - lam) * unigram.get(word, 0) / total
    return -math.log(p) if p > 0 else 20.0
```

```python
def nbest(kana, n=5, beam=30):
    """ビームサーチ: 各位置で、コストの低い順に beam 個の途中結果 (コスト, 単語列) を残しながら進む"""
    beams = [[] for _ in range(len(kana) + 1)]
    beams[0] = [(0.0, ("<s>",))]
    for i in range(len(kana)):
        beams[i] = heapq.nsmallest(beam, beams[i])
        candidates = [(i + len(r), s) for r in trie.prefixes(kana[i:]) for s in dic[r]] + [(i + 1, kana[i])]
        for c, words in beams[i]:
            for j, word in candidates:
                beams[j].append((c + cost(words[-1], word), words + (word,)))
    final = sorted((c + cost(words[-1], "</s>"), words[1:]) for c, words in beams[-1])
    results, seen = [], set()
    for c, words in final:                       # 区切り方だけが違って、表記が同じものは 1 つにまとめる
        text = "".join(words)
        if text not in seen:
            seen.add(text); results.append((text, round(c, 1), "|".join(words)))
        if len(results) == n:
            break
    return results

for kana in ["かみをきった", "じしんがある", "きかいがない", "はなしをきく"]:
    print(kana)
    for text, c, seg in nbest(kana):
        print(f"   {c:6.1f}  {text}  ({seg})")
```

```python
# 文節ごとに候補を選び直す操作のために: 文の一部分だけについて候補を出す
def candidates_for(kana, left="<s>", n=6):
    """読み全体が 1 語になる候補を、直前の単語 left との並びやすさの順に返す"""
    return sorted(dic.get(kana, {kana}), key=lambda s: cost(left, s))[:n]

print("「が」の後ろの「きく」:", candidates_for("きく", left="が"))
print("「を」の後ろの「きく」:", candidates_for("きく", left="を"))
print("文頭の「かみ」:", candidates_for("かみ"))
print("「の」の後ろの「かみ」:", candidates_for("かみ", left="の"))
```

- 最良の 1 つだけを求めるビタビアルゴリズムと違い、途中結果を複数残しながら進めると、上位 N 個の変換結果が得られる。厳密な N-best には、後ろ向きの A* 探索を組み合わせる方法が使われる。
- 表記が同じで区切り方だけが違う候補は、利用者から見れば同じなので、まとめて 1 つにする。
- 実際の入力システムでは、文全体の候補を並べるより、文節ごとに候補を出して選び直せるようにするほうが使いやすい。選び直した結果は、次の文節の候補の順序にも影響する。
- 候補の一覧には、ひらがな・カタカナのままの表記も必ず入れておく。辞書にない語を入力する手段がなくなってしまう。

## かな漢字変換の精度を評価する {kkc_eval}

かな漢字変換の出力が、正解の文とどれくらい合っているかを数値で測り、複数の変換器を比べたい。

```python
!pip install -q fugashi unidic-lite jaconv marisa-trie
```

```python
import io, math, re, urllib.request, zipfile
from collections import Counter, defaultdict
import jaconv
import marisa_trie
from fugashi import Tagger

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

tagger = Tagger()
base = "https://www.aozora.gr.jp/cards/000148/files/"
sentences = []
for f in ["789_ruby_5639.zip", "752_ruby_2438.zip", "773_ruby_5968.zip"]:
    for s in re.split(r"(?<=。)|\n", aozora(base + f)):
        if s.strip():
            sentences.append([(w.surface, jaconv.kata2hira(w.feature.kana or w.surface)) for w in tagger(s.strip())])
train, test = sentences[:-200], [s for s in sentences[-200:] if 5 <= sum(len(r) for _, r in s) <= 60]

dic, unigram, bigram = defaultdict(set), Counter(), Counter()
for sent in train:
    prev = "<s>"; unigram[prev] += 1
    for surface, reading in sent:
        dic[reading].add(surface); unigram[surface] += 1; bigram[prev, surface] += 1; prev = surface
    bigram[prev, "</s>"] += 1
total, trie = sum(unigram.values()), marisa_trie.Trie(dic.keys())

def cost(prev, word, lam):
    p = lam * (bigram.get((prev, word), 0) / unigram[prev] if unigram.get(prev) else 0.0) + (1 - lam) * unigram.get(word, 0) / total
    return -math.log(p) if p > 0 else 20.0

def convert(kana, lam=0.8):
    best = [dict() for _ in range(len(kana) + 1)]
    best[0]["<s>"] = (0.0, None, None)
    for i in range(len(kana)):
        for j, word in [(i + len(r), s) for r in trie.prefixes(kana[i:]) for s in dic[r]] + [(i + 1, kana[i])]:
            for prev, (c, _, _) in best[i].items():
                new = c + cost(prev, word, lam)
                if word not in best[j] or new < best[j][word][0]:
                    best[j][word] = (new, i, prev)
    word = min(best[-1], key=lambda w: best[-1][w][0] + cost(w, "</s>", lam))
    words, pos = [], len(kana)
    while pos > 0:
        _, prev_pos, prev = best[pos][word]
        words.append(word); pos, word = prev_pos, prev
    return "".join(words[::-1])
```

```python
def lcs(a, b):
    """最長共通部分列の長さ (文字単位)"""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]

def evaluate(systems, data):
    rows = {}
    for name, f in systems.items():
        exact = match = n_gold = n_out = 0
        for sent in data:
            kana, gold = "".join(r for _, r in sent), "".join(s for s, _ in sent)
            out = f(kana)
            exact += out == gold
            match += lcs(out, gold); n_gold += len(gold); n_out += len(out)
        p, r = match / n_out, match / n_gold
        rows[name] = {"文正解率": exact / len(data), "適合率": p, "再現率": r, "F 値": 2 * p * r / (p + r)}
    return rows

import pandas as pd

systems = {"変換しない (ひらがなのまま)": lambda k: k, "単語 unigram": lambda k: convert(k, 0.0), "単語 bigram": lambda k: convert(k, 0.8)}
print(len(test), "文で評価")
pd.DataFrame(evaluate(systems, test)).T.round(3)
```

```python
# 誤りの中身を見る: どの読みを、どう間違えたか
import difflib

errors = Counter()
for sent in test:
    kana, gold = "".join(r for _, r in sent), "".join(s for s, _ in sent)
    out = convert(kana)
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, gold, out, autojunk=False).get_opcodes():
        if op == "replace":
            errors[gold[i1:i2], out[j1:j2]] += 1
print("多い誤り (正解 → 出力):", errors.most_common(12))
```

- かな漢字変換の評価には、文がまるごと合った割合 (文正解率) と、最長共通部分列に基づく文字単位の適合率・再現率・F 値がよく使われる。文正解率は厳しく、F 値は部分的な正しさも反映する。
- 「変換しない」場合でも、ひらがなの部分は一致するので F 値はかなり高く出る。必ずこのベースラインと比べる。
- 正解と違っても誤りとは限らない (「出来る / できる」「事 / こと」のような表記のゆれ)。誤りの一覧を見て、本当の誤りと表記のゆれを分けて考える。
- このテストは学習データと同じ作家の文章なので、数値は高めに出る。分野の違うテキスト (現代の Web 文書など) では大きく下がる。評価データの分野は、想定する利用場面に合わせる。

## 入力の途中で続きを予測する (予測入力) {predictive_input}

数文字入力しただけで、入力しようとしている単語や、次に来そうな単語を候補として出したい (スマートフォンの入力のように)。

```python
!pip install -q fugashi unidic-lite jaconv marisa-trie
```

```python
import io, re, urllib.request, zipfile
from collections import Counter, defaultdict
import jaconv
import marisa_trie
from fugashi import Tagger

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

tagger = Tagger()
base = "https://www.aozora.gr.jp/cards/000148/files/"
word_count, next_word, phrase_count = Counter(), defaultdict(Counter), Counter()
for f in ["789_ruby_5639.zip", "752_ruby_2438.zip", "773_ruby_5968.zip"]:
    for s in re.split(r"(?<=。)|\n", aozora(base + f)):
        words = [(w.surface, jaconv.kata2hira(w.feature.kana or w.surface), w.feature.pos1) for w in tagger(s.strip())]
        for i, (surface, reading, pos) in enumerate(words):
            if pos not in ("補助記号", "空白"):
                word_count[reading, surface] += 1
            if i + 1 < len(words):
                next_word[surface][words[i + 1][0]] += 1
            if i + 2 < len(words) and pos in ("名詞", "動詞", "形容詞"):     # 内容語から始まる 3 語の言い回しも候補にする
                phrase_count["".join(r for _, r, _ in words[i:i + 3]), "".join(s for s, _, _ in words[i:i + 3])] += 1

# 読みの前方一致で引けるように、「読み\t表記」をキーにしたトライ木を作る
entries = {f"{r}\t{s}": c for (r, s), c in list(word_count.items()) + [(k, c) for k, c in phrase_count.items() if c >= 5]}
trie = marisa_trie.Trie(entries.keys())
print(len(entries), "件")
```

```python
def predict(prefix, k=6):
    """入力中の読みで前方一致する語句を、頻度の高い順に出す"""
    hits = [(entries[key], key.split("\t")) for key in trie.keys(prefix)]
    hits = [(c, r, s) for c, (r, s) in hits if r != prefix or s != prefix]
    return [s for c, r, s in sorted(hits, reverse=True)[:k]]

for prefix in ["わ", "わが", "せんせ", "にんげ", "おも", "しか", "かんが"]:
    print(f"{prefix} → {predict(prefix)}")
```

```python
def predict_next(previous, k=6):
    """直前に確定した単語の次に来やすい単語を出す (何も入力しなくても候補が出る)"""
    return [w for w, _ in next_word[previous].most_common(k)]

for w in ["吾輩", "先生", "東京", "しかし", "考え"]:
    print(f"{w} の次 → {predict_next(w)}")

# 入力の手間がどれだけ減るか: 候補の上位 3 件に目的の語が出るまでに、何文字打つ必要があるか
def keystrokes(reading, surface, top=3):
    for n in range(1, len(reading) + 1):
        if surface in predict(reading[:n], top):
            return n
    return len(reading)

sample = [(r, s) for (r, s), c in word_count.most_common(2000) if len(r) >= 3][:300]
saved = sum(len(r) - keystrokes(r, s) for r, s in sample) / sum(len(r) for r, _ in sample)
print(f"頻出 300 語で、打鍵数を {saved:.0%} 減らせる")
```

- 予測入力は「読みの前方一致 + 頻度順」が基本。トライ木なら、入力中の文字列で始まる候補を高速に列挙できる。
- 単語だけでなく、よく使う言い回し (複数の単語のまとまり) も候補にすると、打鍵数がさらに減る。
- 直前の単語の次に来やすい単語を出すと、何も打たなくても候補が出る。ここでは bigram を使ったが、ニューラル言語モデルを使うと、より長い文脈を踏まえた予測ができる。
- 利用者ごとの履歴 (最近入力した語を上位に出す) が、実際の使い勝手を大きく左右する。この章の学習のレシピを参照。
- 予測候補に個人情報や不適切な語が出ないようにする配慮も必要。履歴の学習では、パスワード欄などの入力を記録しない。

## 利用者の選択を学習して変換結果を変える {kkc_user_history}

一度選び直した変換は、次から最初に出てほしい。利用者ごとの入力履歴を使って、変換の優先順位を変えたい。

```python
!pip install -q fugashi unidic-lite jaconv marisa-trie
```

```python
import io, math, re, urllib.request, zipfile
from collections import Counter, defaultdict
import jaconv
import marisa_trie
from fugashi import Tagger

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

tagger = Tagger()
base = "https://www.aozora.gr.jp/cards/000148/files/"
dic, unigram, bigram = defaultdict(set), Counter(), Counter()
for f in ["789_ruby_5639.zip", "752_ruby_2438.zip", "773_ruby_5968.zip"]:
    for s in re.split(r"(?<=。)|\n", aozora(base + f)):
        if not s.strip():
            continue
        prev = "<s>"; unigram[prev] += 1
        for w in tagger(s.strip()):
            surface, reading = w.surface, jaconv.kata2hira(w.feature.kana or w.surface)
            dic[reading].add(surface); unigram[surface] += 1; bigram[prev, surface] += 1; prev = surface
        bigram[prev, "</s>"] += 1
total = sum(unigram.values())
```

```python
class Converter:
    """bigram によるかな漢字変換 + 利用者の履歴 (確定した単語と、その並び) による補正"""
    def __init__(self, history_weight=3.0):
        self.user_dic = defaultdict(set)          # 利用者が登録した語: 読み → 表記
        self.user_uni, self.user_bi = Counter(), Counter()
        self.history_weight = history_weight
        self.rebuild()

    def rebuild(self):
        self.trie = marisa_trie.Trie(set(dic) | set(self.user_dic))

    def cost(self, prev, word, lam=0.8):
        p = lam * (bigram.get((prev, word), 0) / unigram[prev] if unigram.get(prev) else 0.0) + (1 - lam) * unigram.get(word, 0) / total
        base = -math.log(p) if p > 0 else 20.0
        # 履歴にある単語・並びは、使った回数に応じてコストを下げる (回数の対数に比例)
        bonus = self.history_weight * (math.log1p(self.user_uni[word]) + math.log1p(self.user_bi[prev, word]))
        return max(base - bonus, 0.1)

    def convert(self, kana):
        best = [dict() for _ in range(len(kana) + 1)]
        best[0]["<s>"] = (0.0, None, None)
        for i in range(len(kana)):
            cands = [(i + len(r), s) for r in self.trie.prefixes(kana[i:]) for s in dic.get(r, set()) | self.user_dic.get(r, set())] + [(i + 1, kana[i])]
            for j, word in cands:
                for prev, (c, _, _) in best[i].items():
                    new = c + self.cost(prev, word)
                    if word not in best[j] or new < best[j][word][0]:
                        best[j][word] = (new, i, prev)
        word = min(best[-1], key=lambda w: best[-1][w][0] + self.cost(w, "</s>"))
        words, pos = [], len(kana)
        while pos > 0:
            _, prev_pos, prev = best[pos][word]
            words.append(word); pos, word = prev_pos, prev
        return words[::-1]

    def commit(self, words):
        """利用者が確定した単語列を履歴に記録する"""
        for prev, word in zip(["<s>"] + words, words):
            self.user_uni[word] += 1; self.user_bi[prev, word] += 1

    def forget(self):
        """履歴をすべて消す (登録した単語は残す)"""
        self.user_uni.clear(); self.user_bi.clear()

    def register(self, reading, surface):
        """単語登録: 辞書にない語を追加する"""
        self.user_dic[reading].add(surface); self.user_uni[surface] += 1; self.rebuild()

conv = Converter()
tests = ["きかいがない", "きかいをうかがう", "よいきかいだ"]
print("学習前:", ["".join(conv.convert(k)) for k in tests])
for n in range(1, 4):
    conv.commit(["機械", "が", "ない"])             # 利用者が「機会」を「機械」に選び直して確定した
    print(f"{n} 回確定した後:", ["".join(conv.convert(k)) for k in tests])
```

```python
# 単語登録: 辞書にない固有名詞や新しい語
tests = ["すまほをみる", "ちゃっとぼっととはなす"]
print("登録前:", ["".join(conv.convert(k)) for k in tests])
conv.register("すまほ", "スマホ")
conv.register("ちゃっとぼっと", "チャットボット")
print("登録後:", ["".join(conv.convert(k)) for k in tests])
```

```python
# 履歴の重みが強すぎると、関係のない文脈まで巻き込んで変わってしまう
for weight in [1.0, 3.0, 10.0]:
    c = Converter(history_weight=weight)
    for _ in range(3):
        c.commit(["機械", "が", "ない"])
    print(f"重み {weight}:", ["".join(c.convert(k)) for k in ["きかいがない", "きかいをうかがう", "よいきかいだ"]])

conv.forget()
print("履歴を消した後:", "".join(conv.convert("きかいがない")))
```

- 履歴の学習には、単語の頻度 (この語をよく使う) と、単語の並び (この語の後ろではこの語) の 2 種類がある。並びまで覚えると、文脈に応じた学習になる。
- 学習が強すぎると、一度選んだ表記が関係のない文脈でも出るようになる (過学習)。回数の対数に比例させる、古い履歴の重みを徐々に下げる、といった調整をする。
- 「学習してほしくない誤変換」を確定してしまうこともある。履歴を取り消す手段 (直前の学習を忘れる、履歴の消去) が必要になる。
- 履歴は個人情報のかたまりなので、端末の中だけで扱い、外に送らないのが基本。

## 学習済みのニューラルモデルでかな漢字変換をする (zenz) {kkc_neural}

大量のテキストで学習されたニューラルかな漢字変換モデルを使って、文脈を踏まえた高精度な変換をしたい。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Miwa-Keita/zenz-v2.5-small"   # かな漢字変換用に学習された GPT-2 型のモデル (約 9,000 万パラメータ)。CPU でも動く
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()

@torch.no_grad()
def convert(kana, n=1, num_beams=None):
    """入力は「 + カタカナの読み + 」の形で与え、続きとして変換結果を生成させる"""
    prompt = "" + kana + ""
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    out = model.generate(**enc, max_new_tokens=len(kana) + 10, do_sample=False, num_beams=num_beams or max(n, 1),
                         num_return_sequences=n, pad_token_id=tokenizer.eos_token_id)
    results = [tokenizer.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]
    return results[0] if n == 1 else results

for kana in ["キョウハイイテンキデス", "シゼンゲンゴショリノケンキュウヲスル", "ハシノウエデハシヲオトシタ", "キシャノキシャガキシャデキシャシタ",
             "カレハイシャニイシャリョウヲセイキュウシタ", "スマホデドウガヲミル", "コノセンセイハジンコウチノウノセンモンカダ"]:
    print(f"{kana} → {convert(kana)}")
```

```python
# 候補を複数出す (ビームサーチ)
for kana in ["コウエンニイク", "カミヲキル", "アツイヒガツヅク"]:
    print(kana, convert(kana, n=4))
```

```python
# ひらがなの入力はカタカナに直して渡す / 速度を測る
import time
import jaconv

text = "わたしはまいあさこーひーをのんでからだいがくにいきます"
t = time.perf_counter()
print(convert(jaconv.hira2kata(text)), f"({time.perf_counter() - t:.2f} 秒)")
```

- zenz は、日本語入力アプリ azooKey のために作られたモデル。読みを条件として、変換後の文を 1 文字ずつ生成する。大量のテキストから学習しているので、現代の語彙や長い文脈に強い。
- 単語 bigram のモデル (この章の bigram のレシピ) と違い、文全体を見て同音異義語を選べる (「医者に慰謝料を請求した」)。ただし万能ではなく、「きしゃのきしゃがきしゃできしゃした」のような同音異義語の言葉遊びは、すべて同じ表記になってしまう。
- 生成モデルなので、読みと対応しない文字を出力することがある。実用のシステムでは、従来型の変換結果を候補として作り、ニューラルモデルで検証・並べ替えをする構成 (azooKey の Zenzai) がとられている。
- 入力のたびにモデルを動かすので、応答の速さが課題になる。小さなモデル、量子化、入力途中の計算結果の再利用などで対処する。

## Web API でかな漢字変換をする (Google CGI API for Japanese Input) {kkc_web_api}

自分で辞書やモデルを用意せずに、手軽にかな漢字変換の結果と候補を得たい。

```python
import json
import urllib.parse
import urllib.request

def transliterate(hiragana):
    """文節ごとに [読み, [候補, ...]] のリストが返る"""
    url = "https://www.google.com/transliterate?" + urllib.parse.urlencode({"langpair": "ja-Hira|ja", "text": hiragana})
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

result = transliterate("きょうはいいてんきですね")
for reading, candidates in result:
    print(reading, candidates)
print("第 1 候補をつなげる:", "".join(c[0] for _, c in result))
```

```python
for text in ["はしのうえではしをおとした", "しぜんげんごしょりのけんきゅうをする", "きしゃのきしゃがきしゃできしゃした", "すまほでどうがをみる"]:
    r = transliterate(text)
    print("".join(c[0] for _, c in r), "|", " ".join(reading for reading, _ in r))

# 文節の区切りを指定する: 区切りたい位置にカンマを入れる
print([(reading, c[:3]) for reading, c in transliterate("ここではきものをぬぐ")])
print([(reading, c[:3]) for reading, c in transliterate("ここで,はきものを,ぬぐ")])
print([(reading, c[:3]) for reading, c in transliterate("くるまでまつ")])
print([(reading, c[:3]) for reading, c in transliterate("くる,までまつ")])
```

- 文節に区切ったうえで、文節ごとの候補を返す。文節の区切りが意図と違うときは、読みにカンマを入れて区切りを指定できる。
- 手軽だが、外部のサーバーに入力を送ることになる。個人情報や機密を含む文章には使わない。利用規約と呼び出しの頻度にも注意する。
- 評価用のデータを作る、自作の変換器と結果を比べる、といった用途に便利。
- ネットワークにつながっていないと使えず、応答に時間もかかるので、キー入力のたびに呼ぶ用途には向かない。
