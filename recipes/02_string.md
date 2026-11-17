# 文字列処理と正規化

形態素解析やモデルに入れる前の、文字レベルの下ごしらえ。日付や住所のような表記ゆれの多い表現の正規化、旧字体・歴史的仮名遣いの新字新仮名への変換、会社名や住所の名寄せも扱う。

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

- Windows 由来の Shift_JIS は `cp932` で読むのが安全（①や髙などの機種依存文字を含むため）。
- 短い文字列は判定を誤りやすい。分かっているなら明示的に `encoding=` を指定する。

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
- NFD/NFKD は濁点が分離する（macOS のファイル名で遭遇しがち）。`"が" != "が"` になったら NFC に戻す。
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
- NFKC とは挙動が異なる（全角記号を残すなど）ので、どちらを使うかはタスク次第。

## 旧字体・歴史的仮名遣いの文章を新字体・現代仮名遣いに直す {kyujitai_kana gpu}

「舊字體」「言ふ」「ゐる」のような旧字旧仮名の文章を、「旧字体」「言う」「いる」の新字新仮名に直して、現代語用の解析器や検索で扱いやすくしたい。対応表と規則で直す方法と、手元の LLM に直させる方法を、青空文庫の同じ作品の新字新仮名版と比べて評価する。

```python
!pip install -q fugashi==1.5.2 unidic-lite==1.0.8 rapidfuzz==3.14.6
# kyujipy は実行に要らない build や twine も依存に持つので、--no-deps で入れて表の読み込みに要る cson だけを足す
!pip install -q --no-deps kyujipy==0.6.8 cson==0.8 speg==0.3
```

```python
import csv, io, re, urllib.request, zipfile
from rapidfuzz.distance import Levenshtein

def fetch_zip(url):
    with urllib.request.urlopen(url) as r:
        return zipfile.ZipFile(io.BytesIO(r.read()))

# 青空文庫の「公開中 作家別作品一覧拡充版」（書誌データは CC BY 4.0）から、同じ作品の旧字旧仮名版と新字新仮名版を探す
z = fetch_zip("https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip")
versions = {}
for r in csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8-sig")):
    if r["役割フラグ"] == "著者" and r["作品著作権フラグ"] == "なし":
        versions.setdefault((r["姓"] + r["名"], r["作品名"]), {}).setdefault(r["文字遣い種別"], r)
both = [k for k, v in versions.items() if "旧字旧仮名" in v and "新字新仮名" in v]
print("旧字旧仮名版と新字新仮名版がそろった作品名:", len(both))     # 副題の違う連作（捕物帳など）は 1 つに数える

def jis_x0213(plane, row, cell):
    """JIS X 0213 の面区点番号を文字にする（EUC-JIS-2004 のバイト列にして復号する）"""
    b = bytes([0xA0 + row, 0xA0 + cell])
    return (b if plane == 1 else b"\x8f" + b).decode("euc_jis_2004")

def gaiji(m):
    """外字の注記を文字に戻す。例: ※［＃「言＋墟のつくり」、第4水準2-88-74］→ 譃"""
    note = m.group(1)
    if u := re.search(r"U\+([0-9A-Fa-f]{4,5})", note):
        return chr(int(u.group(1), 16))
    if j := re.search(r"([12])-(\d{1,2})-(\d{1,2})］?$", note):
        return jis_x0213(*map(int, j.groups()))
    return "〓"   # 面区点番号のない外字（字形の説明とページ数だけのもの）

def aozora_paragraphs(url):
    """データの収集と入出力の章の「青空文庫から作品をダウンロードして本文を取り出す」に、外字の復元と段落への分割を足したもの"""
    z = fetch_zip(url)
    raw = z.read([n for n in z.namelist() if n.endswith(".txt")][0]).decode("cp932")
    parts = re.split(r"-{50,}", raw)
    body = (parts[2] if len(parts) >= 3 else raw).split("底本：")[0]
    body = re.sub(r"※［＃(.+?)］", gaiji, body)       # JIS X 0208 にない旧字は「※」と注記で書かれている
    body = re.sub(r"《.+?》|［＃.+?］|｜", "", body)    # ルビ、入力者注、ルビの開始位置の記号
    return [p.strip("　 ") for p in body.replace("\r\n", "\n").split("\n") if p.strip("　 ")]

print(aozora_paragraphs(versions[("芥川竜之介", "羅生門")]["旧字旧仮名"]["テキストファイルURL"])[1][:60])
```

```python
def align(src, tgt, key=lambda s: s):
    """2 つの版の段落の対応をとる（1 対 1 か読み飛ばし。似ていない段落どうしは組にしない）"""
    a = [key(s) for s in src]
    n, m = len(a), len(tgt)
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = {}
    for i in range(n + 1):
        for j in range(m + 1):
            cands = []
            if i and j:
                cands.append((score[i - 1][j - 1] + Levenshtein.normalized_similarity(a[i - 1], tgt[j - 1]) - 0.3, (i - 1, j - 1)))
            if i:
                cands.append((score[i - 1][j], (i - 1, j)))
            if j:
                cands.append((score[i][j - 1], (i, j - 1)))
            if cands:
                score[i][j], back[i, j] = max(cands)
    pairs, (i, j) = [], (n, m)
    while i or j:
        pi, pj = back[i, j]
        if (pi, pj) == (i - 1, j - 1):
            pairs.append((src[i - 1], tgt[j - 1]))
        i, j = pi, pj
    return pairs[::-1]

# 規則を作るときに誤りを見た作品 (dev) と、評価だけに使う作品 (test) を分ける
WORKS = {"dev": [("太宰治", "列車"), ("太宰治", "ダス・ゲマイネ"), ("堀辰雄", "窓"), ("堀辰雄", "鳥料理"),
                 ("永井荷風", "虫の声"), ("伊藤左千夫", "奈々子")],
         "test": [("芥川竜之介", "羅生門"), ("芥川竜之介", "藪の中"), ("芥川竜之介", "蜜柑"), ("中島敦", "山月記"),
                  ("中島敦", "名人伝"), ("梶井基次郎", "檸檬"), ("夏目漱石", "変な音"), ("国木田独歩", "画の悲み"),
                  ("森鴎外", "寒山拾得"), ("森鴎外", "高瀬舟")]}
```

```python
from kyujipy import BasicConverter, KyujitaiConverter
import unicodedata

# 旧字→新字の対応表: kyujipy (MIT)。常用漢字表・人名用漢字表・Unicode の Standardized Variants・kyujitai.js をもとにした 1 字ずつの表
bc = BasicConverter()
def shinjitai(t):
    """字体だけを新字にする。「辨・瓣・辯」→「弁」は kyujipy では書きかえの表にあるので、ここで足す"""
    return bc.kyujitai_to_shinjitai(t).translate(str.maketrans("辨瓣辯", "弁弁弁"))

print(shinjitai("舊字體の國語を學び、圓い燈を點す"))
# KyujitaiConverter は同音の漢字による書きかえ（輿論→世論、編輯→編集）までするので、語を変えたくないときは使わない
print(shinjitai("輿論と編輯"), "/", KyujitaiConverter().kyujitai_to_shinjitai("輿論と編輯"))
print(unicodedata.normalize("NFKC", "舊字體の國語"))     # NFKC では旧字体は変わらない

data = []
for group, works in WORKS.items():
    for author, title in works:
        v = versions[(author, title)]
        src, tgt = aozora_paragraphs(v["旧字旧仮名"]["テキストファイルURL"]), aozora_paragraphs(v["新字新仮名"]["テキストファイルURL"])
        pairs = align(src, tgt, key=shinjitai)     # 漢字だけ新字にしてから似ている段落を探す
        data += [{"group": group, "work": title, "kyu": a, "shin": b} for a, b in pairs]
        print(f"{group:4} {author} {title}: 段落 {len(src)} / {len(tgt)} → 対応 {len(pairs)}")
print(data[0]["kyu"][:50]); print(data[0]["shin"][:50])
```

```python
import fugashi

tagger = fugashi.Tagger()     # UniDic (unidic-lite) は歴史的仮名遣いの語も持っていて、語形 (form) に現代の発音の形が入っている
for w in tagger("さうして私はかう思ふのである。雨が降つてゐる。"):
    print(w.surface, w.feature.pos1, w.feature.kana, w.feature.form, sep="\t")
```

```python
K2H = {i: i - 0x60 for i in range(0x30A1, 0x30F7)}
VOICE = dict(zip("かきくけこさしすせそたちつてとはひふへほカキクケコサシスセソタチツテトハヒフヘホ",
                 "がぎぐげござじずぜぞだぢづでどばびぶべぼガギグゲゴザジズゼゾダヂヅデドバビブベボ"))
UNVOICE = {v: k for k, v in VOICE.items()}
A2O = dict(zip("あかがさざただなばぱまやらわは", "おこごそぞとどのぼぽもよろおお"))

def odoriji(t):
    """踊り字を開く: こゝろ→こころ、づゝ→づつ、たゞ→ただ、うと／＼→うとうと（くの字点は直前の 2 文字の繰り返しとみなす）"""
    t = re.sub(r"(.)(.)／″＼", lambda m: m.group(1) + m.group(2) + VOICE.get(m.group(1), m.group(1)) + m.group(2), t)
    t = re.sub(r"(..)／＼", r"\1\1", t)
    t = re.sub(r"(.)[ゝヽ]", lambda m: m.group(1) + UNVOICE.get(m.group(1), m.group(1)), t)     # ゝ は清音で繰り返す
    return re.sub(r"(.)[ゞヾ]", lambda m: m.group(1) + VOICE.get(UNVOICE.get(m.group(1), m.group(1)), m.group(1)), t)

def kana_simple(t):
    """形態素を見ずに、文字の並びだけで置き換える"""
    t = t.translate(str.maketrans("ゐゑぢづ", "いえじず"))
    t = re.sub("(?<=[ぁ-ゖ])[はひふへほ]", lambda m: "わいうえお"["はひふへほ".index(m.group())], t)   # 語中・語末のハ行
    t = re.sub("([あかがさざただなばぱまやらわ])う", lambda m: A2O[m.group(1)] + "う", t)         # さう→そう、だらう→だろう
    return re.sub("つ(?=[たて])", "っ", t)                                                    # 言つた→言った

def rewrite(surface, kana, form):
    """表層形の仮名の部分だけを語形の仮名に置き換える（思ふ → 思う、言ひ → 言い）。漢字の部分は読みごとそのまま"""
    segs = re.findall(r"[ぁ-ゖ]+|[^ぁ-ゖ]+", surface)
    is_kana = [bool(re.match("[ぁ-ゖ]", s)) for s in segs]
    m = re.fullmatch("".join(f"({re.escape(s)})" if k else "(.+?)" for s, k in zip(segs, is_kana)), kana.translate(K2H))
    if not m:
        return surface
    m2 = re.fullmatch("".join("(.*?)" if k else re.escape(r) for r, k in zip(m.groups(), is_kana)), form.translate(K2H))
    if not m2:
        return surface
    new = iter(m2.groups())
    return "".join(next(new) if k else s for s, k in zip(segs, is_kana))

KEEP = "\uE000"     # 促音にしない「つ」の目印（私用領域の文字）

def standalone(prev):
    """直前が文頭・記号か、名詞などに付いた助詞 1 つ（彼は/かつて）なら、独立した語とみなす（向/かつて、通り/か/かつて は除く）"""
    pos = [p.feature.pos1 for p in prev[-2:]]
    if prev and prev[-1].surface.endswith("か"):     # 「かかつて」はたいてい「掛かって」
        return False
    return not pos or pos[-1] == "補助記号" or (pos[-1] == "助詞" and (len(pos) == 1 or pos[0] not in ("助詞", "補助記号")))

def kana_morph(t):
    """形態素解析の結果を使って置き換える。助詞の「は・へ・を」や語頭のハ行（母、はは）はそのまま残る"""
    out, prev = [], []
    for w in tagger(t):
        s, f = w.surface, w.feature
        changed = False
        if not w.is_unk and f.kana not in ("", "*") and f.form not in ("", "*") and f.kana != f.form and re.search("[ぁ-ゖ]", s):
            s2 = rewrite(s, f.kana, f.form)     # 辞書にある語: 言ふ→言う、ゐる→いる、てふてふ→ちょうちょう
            changed, s = s2 != s, s2
        if not changed:
            s = re.sub("([あかがさざただなばぱまやらわ])う", lambda m: A2O[m.group(1)] + "う", s)   # 1 語の中の「a 段+う」
            if s in ("ひ", "ふ", "へ", "ほ") and f.pos1 != "助詞" and out and re.search(r"[一-鿿々]$", out[-1]):
                s = "いうえお"["ひふへほ".index(s)]        # 漢字の直後の、助詞でない 1 文字は送り仮名とみなす（工合ひ）
        if s == "う" and f.pos1 in ("助動詞", "感動詞") and out:   # 推量・意志の「う」の前: 書か/う→書こう、ませ/う→ましょう
            out[-1] = re.sub("([あかがさざただなばぱまやらわ])$", lambda m: A2O[m.group(1)], out[-1])
            out[-1] = re.sub("せ$", "しょ", out[-1])
        if (f.lemma == "嘗て" and standalone(prev)) or (re.match("つ[たて]", s) and f.lemma in ("拙い", "伝える", "伝わる")):
            s = s.replace("つ", KEEP, 1)     # 副詞の「かつて」と「つたない」などの「つ」は促音にしない
        out.append(w.white_space + s)
        prev.append(w)
    t = "".join(out).translate(str.maketrans("ゐゑヰヱ", "いえイエ"))
    # 促音: 持つて→持って、なかつた→なかった。旧仮名では語の区切りが崩れやすい（なか/つた）ので文字の並びで直す
    return re.sub(r"(?<=[ぁ-ゖ一-鿿々])つ(?=[たて])", "っ", t).replace(KEEP, "つ")

SYSTEMS = {"変換なし": lambda t: t,
           "漢字だけ": lambda t: shinjitai(odoriji(t)),
           "漢字+文字の規則": lambda t: kana_simple(shinjitai(odoriji(t))),
           "漢字+形態素の規則": lambda t: kana_morph(shinjitai(odoriji(t)))}

for s in ["はゝは川へ行つた。", "これはあひるです。", "道がつゞく。", "一本づゝ。", "さうでせう。", "彼はかつて海へ向かつて歩いた。",
          "人にしたがふ。", "思ひ出してゐた。", "つい忘れてしまふのだ。", "言ひつたへ。"]:
    print(s, "→", SYSTEMS["漢字+文字の規則"](s), "/", SYSTEMS["漢字+形態素の規則"](s))
```

```python
import pandas as pd

for d in data:
    for name, fn in SYSTEMS.items():
        d[name] = fn(d["kyu"])

def cer(rows, name):
    """文字誤り率: 新字新仮名版を正解とした編集距離の合計 ÷ 正解の文字数"""
    return sum(Levenshtein.distance(d[name], d["shin"]) for d in rows) / sum(len(d["shin"]) for d in rows)

df = pd.DataFrame(data)
table = pd.DataFrame({(g, w): {name: cer(df[(df.group == g) & (df.work == w)].to_dict("records"), name) for name in SYSTEMS}
                      for g, w in df[["group", "work"]].drop_duplicates().itertuples(index=False)}).T
print(table.round(3).to_string())
print(pd.DataFrame({g: {name: cer(df[df.group == g].to_dict("records"), name) for name in SYSTEMS} for g in WORKS})
      .T.assign(段落数=df.group.value_counts()).round(3).to_string())
```

```python
import difflib
from collections import Counter

# test の作品で、形態素の規則の出力と新字新仮名版とで違うところ（前後 2 文字を付けて数える）
diffs = Counter()
for d in df[df.group == "test"].to_dict("records"):
    h, y = d["漢字+形態素の規則"], d["shin"]
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, h, y, autojunk=False).get_opcodes():
        if op != "equal":
            diffs[(h[max(0, i1 - 2):i2 + 1], y[max(0, j1 - 2):j2 + 1])] += 1
for (h, y), c in diffs.most_common(25):
    print(c, h, "→", y)
```

```python
# 手元の LLM に直させて比べる（GPU 向けのセル。T4 で 5〜10 分。初回はモデルの約 8GB のダウンロードが加わる）
import random, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen3-4B-Instruct-2507"     # Alibaba の Qwen チーム、Apache 2.0
tok = AutoTokenizer.from_pretrained(name, padding_side="left")
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()

INSTRUCTION = ("次の文章の旧字体を新字体に、歴史的仮名遣いを現代仮名遣いに直してください。"
               "語の選び方、漢字と仮名の使い分け、句読点は変えないでください。直した文章だけを出力してください。")
EXAMPLE = ("其の男は「さうでせう。」と云つて、舊い家へ歸つて行つた。", "其の男は「そうでしょう。」と云って、旧い家へ帰って行った。")

def messages(text):
    return [{"role": "user", "content": f"{INSTRUCTION}\n\n{EXAMPLE[0]}"}, {"role": "assistant", "content": EXAMPLE[1]},
            {"role": "user", "content": f"{INSTRUCTION}\n\n{text}"}]

@torch.no_grad()
def llm_convert(texts, batch_size=16):
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        prompts = [tok.apply_chat_template(messages(t), add_generation_prompt=True, tokenize=False) for t in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(llm.device)
        gen = llm.generate(**enc, max_new_tokens=int(max(map(len, batch)) * 1.5) + 20, do_sample=False)
        outs += [s.strip() for s in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return outs

random.seed(0)
sample = random.sample([d for d in data if d["group"] == "test" and len(d["kyu"]) <= 300], 120)
t0 = time.time()
for d, out in zip(sample, llm_convert([d["kyu"] for d in sample])):
    d["LLM"] = out
print(f"{len(sample)} 段落、{time.time() - t0:.0f} 秒、GPU メモリの最大 {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

names = list(SYSTEMS) + ["LLM"]
def leftovers(t):
    """直し残し: 表で新字にできる旧字、ゐ・ゑ、「た・て」の前の大きい「つ」の数"""
    return (sum(shinjitai(c) != c for c in t), len(re.findall("[ゐゑ]", t)), len(re.findall("(?<=[ぁ-ゖ一-鿿])つ(?=[たて])", t)))
print(pd.DataFrame({n: {"CER": cer(sample, n), **dict(zip(["旧字", "ゐ・ゑ", "つ+た・て"], np.sum([leftovers(d[n]) for d in sample], axis=0)))}
                    for n in names + ["shin"]}).T.round(3).to_string())     # shin は正解（新字新仮名版）

# 段落を選び直すブートストラップで、CER の差の 95% 信頼区間を求める
rng = np.random.default_rng(0)
err = {n: np.array([Levenshtein.distance(d[n], d["shin"]) for d in sample]) for n in names}
length = np.array([len(d["shin"]) for d in sample])
idx = rng.integers(0, len(sample), (2000, len(sample)))
for a, b in [("LLM", "漢字+形態素の規則"), ("LLM", "漢字だけ")]:
    diff = err[a][idx].sum(1) / length[idx].sum(1) - err[b][idx].sum(1) / length[idx].sum(1)
    print(f"{a} − {b}: {cer(sample, a) - cer(sample, b):+.3f}  95% CI {np.percentile(diff, [2.5, 97.5]).round(3)}")
```

```python
# LLM と規則の出力を並べる: 規則より LLM の CER が大きい段落
worst = sorted(sample, key=lambda d: Levenshtein.distance(d["LLM"], d["shin"]) - Levenshtein.distance(d["漢字+形態素の規則"], d["shin"]))[-3:]
for d in worst:
    print("原文:", d["kyu"][:80]); print("規則:", d["漢字+形態素の規則"][:80]); print("LLM :", d["LLM"][:80]); print("正解:", d["shin"][:80]); print()
```

- 青空文庫の著作権の切れた作品は、「青空文庫収録ファイルの取り扱い規準」で複製・再配布・翻案が自由とされ、著作権法の範囲で旧かな・旧漢字を現代表記にあらためる書き換えもできるとされている（入力者などの情報を残すことと、書き換えの履歴を書くことが望まれている）。作家別作品一覧の CSV は CC BY 4.0。kyujipy と fugashi は MIT、unidic-lite はパッケージが MIT で、同梱の UniDic 2.1.2 は GPL、LGPL、BSD から選べる。Qwen3-4B-Instruct-2507 は Apache 2.0。
- 「正解」の新字新仮名版は底本が違い、仮名遣い以外の書き換え（「云う」→「言う」、「居る」→「いる」、「唯」→「ただ」、「ます。」」→「ます」」）も含むので、CER は 0 にならない。test の 10 作品の CER は、変換なしで 0.137、漢字だけで 0.109、文字の規則を足して 0.085、形態素の規則を足して 0.064 だった。残った差の上位はこうした書き換えが多いが、規則の誤り（「てしもう」「はひっ」「さつき」「やつと」）も混ざっている。dev の「虫の声」「奈々子」は 0.154 残り、底本どうしの違いが大きい。
- 文字の並びだけの規則では、語頭や助詞のハ行と語中のハ行の区別がつかず、「はゝは」が「はわわ」、「これは」が「これわ」、「あひる」が「あいる」、「つゞく」が「つずく」、「したがふ」が「したごう」、副詞の「かつて」が「かって」になった。形態素解析で語の切れ目と品詞が分かれば、助詞の「は・へ・を」や語頭の「は」（母、はは）、副詞の「かつて」を残せる。
- 形態素の規則は、UniDic の語形 (form) に入っている現代の形（イウ、イル、ソウ）に表層形の仮名を合わせ、辞書で拾えない「書か/う」「なかつた」などを少しの規則で補ったもの（補う規則は dev の作品の誤りだけを見て作った）。旧仮名では語の区切りが崩れやすく、「人にしたがふ」は「し/た/がふ」と区切られて残り、「思ひ出」も残り、「つい忘れてしまふ」は語形の選び方で「しもう」になり、「言ひつたへ」は「つた」を「蔦」とみて「言いったへ」になった。
- 漢字は字体だけを 1 字ずつ変える BasicConverter を使った。KyujitaiConverter は「輿論→世論」「編輯→編集」のような同音の漢字による書きかえもするので、語を変えたくないときには向かない。表にない異体字は変わらず（上の例の「迂囘」）、test の 120 段落の新字新仮名版にも表で変わる字が 13 字残っていて、そこでは表の変換が正解と食い違う。「舊」のような統合漢字の旧字体は NFKC では変わらない。
- 青空文庫のテキストは Shift_JIS (JIS X 0208) なので、それにない字は「※［＃「言＋墟のつくり」、第4水準2-88-74］」のような注記になっている。面区点番号があれば EUC-JIS-2004 のコーデックで文字に戻せる。データの収集と入出力の章の「青空文庫から作品をダウンロードして本文を取り出す」の関数は注記を消すので、旧字の多い作品ではこのように戻してから使う。
- LLM の比較のセルだけが GPU 向けで、それより前のセルは CPU で動く。Qwen3-4B-Instruct-2507 に 1 例を付けて直させると、test の 120 段落の CER は 0.103 で、漢字だけ新字にしたもの (0.108) との差 −0.006 は 95% 信頼区間 [−0.011, 0.000] の上限が 0 にかかり、形態素の規則 (0.064) より 0.039 悪かった。旧字 435 字のうち 160 字、ゐ・ゑ 84 字のうち 16 字が残り、上の例では最初の数語だけ直してあとは原文のまま写していた。GPU メモリの最大は 9.8 GB（T4 と同じ float16 の値。bf16 の GPU では出力が変わりうる）。

## 繰り返し文字を縮約する（「すごーーーい」「wwww」） {squeeze_repeats}

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
- ニュース記事などの「本文だけ」が欲しい場合は、データの収集と入出力の章の「Web ページから本文だけを抽出する (trafilatura)」を参照。

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

## 文字種（漢字・ひらがな・カタカナ・英数字）を判定する {char_type}

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

## テキストの言語（文字種）の割合を調べる {script_ratio}

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
- きちんとした言語判定は英語・多言語の前処理の章（言語判定のレシピ）を参照。

## 見た目どおりの文字数を数える（書記素クラスタ） {grapheme}

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

## 全角を 2 として表示幅を数える・揃える {display_width}

等幅フォントでの表示幅（全角 2、半角 1）を数えて、日本語を含む表をきれいに揃えたい。

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

- `east_asian_width` は F（全角）・W（広い）・A（曖昧）・H（半角）・Na（狭い）・N（中立）を返す。A（ギリシャ文字や罫線など）を 2 とするかは環境次第。
- より厳密には `wcwidth` パッケージを使う。

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

- 解釈できない表現は `None` が返る。和暦などは前処理（和暦→西暦のレシピ）と組み合わせる。

## 日本の住所の表記ゆれを正規化して要素に分ける {address_normalize}

「東京都千代田区丸の内1丁目9番1号」「千代田区丸の内一丁目9-1」「東京都千代田区丸ノ内１－９－１」のような住所の表記ゆれをそろえ、都道府県・市区町村・町丁目・番地に分けたい。

```python
!pip install -q kanjize normalize-japanese-addresses jageocoder
```

```python
# まず辞書を使わない簡易な方法: NFKC → 漢数字を算用数字に → 「丁目・番・号」をハイフンに → 正規表現で分割
# 漢数字の変換はこの章の「漢数字⇔アラビア数字を変換する」と同じ kanjize (MIT) を使う
import re
import unicodedata
from kanjize import kanji2number, number2kanji

KANSUJI = "〇一二三四五六七八九十百千"
DASH = "-‐‑‒–—―−ー－"  # ハイフンに見える文字 (長音「ー」やマイナス「−」も混ざる)

def simple_normalize(addr):
    s = unicodedata.normalize("NFKC", addr).replace(" ", "")
    # 漢数字は「丁目・番・号・の・ハイフン」の直前にあるものだけ変換する (「八王子」「六本木」「一番町」を守る)
    s = re.sub(rf"[{KANSUJI}]+(?=丁目|番(?!町)|号|の[\d{KANSUJI}]|[{DASH}]|$)", lambda m: str(kanji2number(m.group())), s)
    s = re.sub(rf"(?<=\d)[{DASH}](?=\d)", "-", s)
    has_chome = "丁目" in s
    s = re.sub(r"(?<=\d)(丁目|番地|番|の)(?=\d)", "-", s)
    s = re.sub(r"(?<=\d)(丁目|番地|番|号)", "", s)
    m = re.match(r"(東京都|北海道|(?:京都|大阪)府|.{2,3}県)?((?:.+?郡)?.+?[市区町村](?:.+?区)?)?(\D*)([\d-]*)(.*)", s)
    pref, city, town, nums, rest = (g or "" for g in m.groups())
    nums = [n for n in nums.split("-") if n]
    if nums and (has_chome or len(nums) >= 3):  # 丁目と書いていなくても、数字が 3 つなら先頭を丁目とみなす (推測)
        town += number2kanji(int(nums.pop(0))) + "丁目"
    return {"pref": pref, "city": city, "town": town, "addr": "-".join(nums), "rest": rest}

for a in ["東京都千代田区丸の内1丁目9番1号", "千代田区丸の内一丁目9-1", "東京都千代田区丸ノ内１－９－１",
          "東京都港区六本木六丁目10番1号", "東京都八王子市元本郷町三丁目24番1号", "東京都千代田区丸の内一の九の一",
          "東京都千代田区一番町10-1"]:
    print(a, "→", simple_normalize(a))
```

```python
# 簡易な方法が誤る例
for a in ["三重県四日市市諏訪町1-5",                 # 「四日市」で市が終わったと判断する
          "長野県大町市大町3887",                    # 「大町」で町が終わったと判断する
          "京都市中京区寺町通御池上る上本能寺前町488",  # 通り名が町名に混ざる。都道府県も補えない
          "北海道札幌市西区24-2-2-3-3",              # 「二十四軒二条二丁目」の略記は辞書がないと読めない
          "東京都千代田区霞ヶ関1-3-1"]:               # 正しくは「霞が関」だが、それは辞書がないと分からない
    print(a, "→", simple_normalize(a))
```

```python
# 方法 1: normalize-japanese-addresses (Geolonia の JavaScript 版を Python に移植したもの、MIT)
# 町丁目の一覧を Geolonia 住所データの API (https://geolonia.github.io/japanese-addresses/api/ja) から都度取得する
# 住所データ: Geolonia 住所データ (CC BY 4.0。国土交通省の位置参照情報と日本郵便の郵便番号データをもとに作成)
from normalize_japanese_addresses import normalize

ADDRS = ["東京都千代田区丸の内1丁目9番1号", "千代田区丸の内一丁目9-1", "東京都千代田区丸ノ内１－９－１",
         "三重県四日市市諏訪町1-5", "長野県大町市大町3887", "京都市中京区寺町通御池上る上本能寺前町488",
         "北海道札幌市西区24-2-2-3-3", "東京都千代田区霞ヶ関1-3-1", "大阪府大阪市北区梅田3-1-1 グランフロント大阪"]
for a in ADDRS:
    r = normalize(a)
    print(f"level={r['level']} {r['pref']} | {r['city']} | {r['town']} | {r['addr']}   ← {a}")
```

```python
# 方法 2: jageocoder (東京大学 CSIS のアドレスマッチングで使われている DAMS の Python 移植、MIT)
# 住所辞書は別に入れる。ここでは街区レベルの全国版 (zip で 351MB、展開後 1.5GB 程度) を使う
# 辞書: 株式会社情報試作室「jageocoder 用住所データベース (街区レベル)」2025-04-23 版
#   国土交通省「位置参照情報」(令和5年)、Geolonia 住所データ、デジタル庁「アドレス・ベース・レジストリ」などを加工したもの
#   商用・非商用とも利用可。複製するときは同梱の README.md を同じ場所に置き、成果物を公開するときは出典を書く (README.md の利用条件)
# 2026-04-17 版 (20260417/v2/) もあるが、README と latest/v2/ の一覧が指す 2025-04-23 版を使う
# Colab で数分程度 (ダウンロードと展開)
import os
import urllib.request
import jageocoder

URL = "https://www.info-proto.com/static/jageocoder/20250423/v2/gaiku_all_20250423_v22.zip"
if not os.path.exists("jageocoder_db"):
    urllib.request.urlretrieve(URL, "dic.zip")
    jageocoder.install_dictionary("dic.zip", db_dir="jageocoder_db", skip_confirmation=True)
    os.remove("dic.zip")
jageocoder.init(db_dir="jageocoder_db")  # 辞書を入れずに公開デモサーバを使うなら jageocoder.init(url="https://jageocoder.info-proto.com/jsonrpc")
print(jageocoder.installed_dictionary_readme("jageocoder_db")[:200])  # 利用条件を必ず読む
```

```python
# レベル (1: 都道府県, 2: 郡, 3: 市・特別区, 4: 政令市の区, 5: 大字・町, 6: 字・丁目, 7: 街区・地番) で要素に分ける
def jageo_normalize(addr):
    results = jageocoder.searchNode(addr)
    if not results:
        return None
    res = results[0]  # 候補が複数あるときは同点のものが並ぶが、ここでは先頭だけを使う (都道府県などで絞るとよい)
    parts = {"pref": "", "city": "", "town": "", "addr": ""}
    for node in res.node.get_parent_list():
        key = "pref" if node.level == 1 else "city" if node.level <= 4 else "town" if node.level <= 6 else "addr"
        parts[key] += re.sub(r"(番地|番|号)$", "", node.name) if key == "addr" else node.name
    # 街区レベルの辞書には住居番号 (「1号」) がないので、残りの先頭の数字を足す
    rest = unicodedata.normalize("NFKC", addr[len(res.matched):])
    if parts["addr"] and (m := re.match(r"(\d+)号?", rest)):
        parts["addr"] += "-" + m.group(1)
        rest = rest[m.end():]
    parts["rest"] = rest  # 辞書で照合できなかった残り
    parts["n_cand"] = len(results)
    return parts

for a in ADDRS:
    r = jageo_normalize(a)
    if r is None:
        print("(照合できない)   ←", a)
        continue
    print(f"{r['pref']} | {r['city']} | {r['town']} | {r['addr']} | 残り={r['rest']!r}   ← {a}")
```

```python
# 3 つの方法がそろって、あるいはどれかが失敗する例
import pandas as pd

def fmt(d):
    return "|".join(d[k] for k in ["pref", "city", "town", "addr"]) if d else "(なし)"

HARD = ["東京千代田区丸の内1-9-1",      # 「都」の抜け
        "東京都千代田区丸野内1-9-1",    # 誤字
        "東京都千代田区丸の内壱丁目9番1号",  # 「壱」のような改ざん防止用の数字
        "東京都千代田区丸の内9丁目1番",  # 存在しない丁目
        "埼玉県浦和市高砂3-15-1",       # 合併前の市名 (現在はさいたま市浦和区)
        "北区梅田3-1-1",                # 都道府県なしの「北区」
        "中央区本町1-1-1"]              # 都道府県なしで、候補が複数ある
rows = []
for a in HARD:
    j, n = jageo_normalize(a), normalize(a)
    rows.append({"入力": a, "簡易": fmt(simple_normalize(a)), "NJA": f"{fmt(n)} (level={n['level']})",
                 "jageocoder": fmt(j) + (f" 残り={j['rest']}" if j and j["rest"] else "")
                               + (f" (候補{j['n_cand']})" if j and j["n_cand"] > 1 else "")})
pd.set_option("display.max_colwidth", None)
pd.DataFrame(rows)
```

- 簡易な方法で揃うのは全角・漢数字・ハイフンまで。市区町村の境目（「四日市」で切れた）や正式な表記（「霞ヶ関」が残った）は辞書がないと決められない。地名の漢数字（「一番町」）は規則で守るので漏れはありうる。
- normalize-japanese-addresses は都道府県を補い（京都府）、通り名を除き、「24-2-2-3-3」を「二十四軒二条二丁目」と読めた。分けるのは町丁目までで、番地と建物名は addr に残る。
- normalize-japanese-addresses は町丁目の一覧を API から取る（住所全体は送らない）。API の最終更新は 2023 年 11 月で（2026 年 11 月に確認）、新しい町名はない可能性がある。
- jageocoder は手元の辞書で照合し、合併前の「浦和市」を「さいたま市浦和区」に直せた（normalize-japanese-addresses は level=1）。「東京千代田区」と「24-2-2-3-3」は照合できなかった。
- 「北区」「中央区」だけだと、normalize-japanese-addresses は東京都と決め打ちし、jageocoder は候補をすべて返す（中央区は 2 つ）。このコードは先頭の候補だけを使うので、都道府県などで絞る。
- 街区レベルの辞書には住居番号（「1号」）がない。住居番号まで入った版 (jukyo_all) は zip で 4.5GB ある。公開デモサーバも使えるが、リクエスト数に上限があり、住所がサーバに送られる。
- 誤字（「丸野内」）、「壱丁目」、存在しない丁目は 3 つとも失敗した。辞書を使う 2 つは町丁目が空か途中で止まるので気づけるが、簡易な方法は「丸の内九丁目」のようなもっともらしい誤りを返す。
- 誤字は、この章の「編集距離とあいまい文字列マッチング (RapidFuzz)」で町丁目の一覧から近いものを探すとよい。文章から住所の範囲を見つけるのは固有表現抽出の章を参照。

## 漢字かな交じり文をローマ字・ひらがなにする {romaji}

日本語の文をローマ字（ヘボン式）や、ひらがなだけの文に変換したい。

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

- 辞書ベースなので文脈による読み分け（「今日」きょう/こんにち）は苦手。精度が要るなら形態素解析器の読みを使う（日本語の形態素解析と文分割の章を参照）。

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
- RapidFuzz は C++ 実装で、数百万件の突き合わせにも使える（`process.cdist` で行列計算）。

## 取引先の会社名・住所・代表者名の表記ゆれを名寄せする（エンティティ解決） {entity_resolution}

「株式会社さくら物産」「㈱さくら物産」「さくら物産 Co., Ltd.」のように表記の違う 2 つの名簿のレコードが、同じ会社を指しているかを判定したい。

```python
!pip install -q rapidfuzz kanjize
```

```python
# 架空の取引先データを作る (会社名・住所・代表者名。社名や住所は組み合わせで作ったもので、実在の会社とは関係ない)
# A: 社内の取引先マスタ 240 社。B: 別の部署から来た名簿 300 件で、うち 200 件は A のどれかと同じ会社、100 件は A にない会社
import random
import pandas as pd

random.seed(0)
STEMS = [("さくら", "Sakura"), ("富士", "Fuji"), ("大和", "Yamato"), ("東洋", "Toyo"), ("北斗", "Hokuto"),
         ("みどり", "Midori"), ("青葉", "Aoba"), ("朝日", "Asahi"), ("太平", "Taihei"), ("若葉", "Wakaba"),
         ("丸山", "Maruyama"), ("三協", "Sankyo"), ("日進", "Nisshin"), ("協和", "Kyowa"), ("昭和", "Showa"),
         ("サンライズ", "Sunrise"), ("アルファ", "Alpha"), ("グリーン", "Green"), ("ミライ", "Mirai"), ("スター", "Star")]
BIZ = [("物産", "Bussan"), ("商事", "Shoji"), ("精機", "Seiki"), ("電機", "Denki"), ("工業", "Kogyo"),
       ("運輸", "Unyu"), ("建設", "Kensetsu"), ("食品", "Foods"), ("化学", "Kagaku"), ("製作所", "Seisakusho"),
       ("産業", "Sangyo"), ("電子", "Denshi"), ("印刷", "Printing"), ("興産", "Kosan"), ("システムズ", "Systems")]
PLACES = [("東京都", "千代田区", ["神田錦町", "内神田", "麹町"]), ("東京都", "港区", ["芝浦", "新橋", "赤坂"]),
          ("大阪府", "大阪市北区", ["梅田", "天満", "中之島"]), ("愛知県", "名古屋市中区", ["栄", "丸の内", "錦"]),
          ("福岡県", "福岡市博多区", ["博多駅前", "中洲", "祇園町"]), ("神奈川県", "横浜市西区", ["高島", "北幸", "南幸"]),
          ("広島県", "広島市中区", ["紙屋町", "大手町", "八丁堀"]), ("埼玉県", "さいたま市大宮区", ["桜木町", "大門町", "宮町"]),
          ("兵庫県", "神戸市中央区", ["三宮町", "元町通", "栄町通"]), ("静岡県", "静岡市葵区", ["紺屋町", "呉服町", "追手町"])]
SURNAMES = ["高橋", "斎藤", "渡辺", "山崎", "吉田", "沢田", "浜田", "佐藤", "鈴木", "田中", "伊藤", "小林", "中村", "加藤"]
GIVEN = ["一郎", "誠", "健太", "裕子", "美穂", "直樹", "恵", "大輔", "隆", "由美"]

def new_company(stem, biz, place=None):
    place = random.randrange(len(PLACES)) if place is None else place
    return {"stem": stem, "biz": biz, "legal": random.choices(["株式会社", "有限会社", "合同会社"], [8, 1, 1])[0],
            "place": place, "town": random.choice(PLACES[place][2]),
            "num": (random.randint(1, 5), random.randint(1, 20), random.randint(1, 30)),
            "rep": (random.choice(SURNAMES), random.choice(GIVEN))}

def render_master(c):  # マスタはきれいな表記
    pref, city, _ = PLACES[c["place"]]
    core = c["stem"][0] + c["biz"][0]
    ch, ban, go = c["num"]
    return {"name": c["legal"] + core if random.random() < 0.6 else core + c["legal"],
            "address": f"{pref}{city}{c['town']}{ch}丁目{ban}番{go}号", "rep": "".join(c["rep"])}

KAN = "〇一二三四五六七八九"
ZEN = str.maketrans("0123456789-", "０１２３４５６７８９－")
HANKANA = {"サンライズ": "ｻﾝﾗｲｽﾞ", "アルファ": "ｱﾙﾌｧ", "グリーン": "ｸﾞﾘｰﾝ", "ミライ": "ﾐﾗｲ", "スター": "ｽﾀｰ", "システムズ": "ｼｽﾃﾑｽﾞ"}
TYPO = {"精": "清", "機": "器", "電": "雷", "建": "健", "設": "説", "化": "花", "物": "仏", "業": "葉", "進": "新",
        "和": "話", "洋": "羊", "産": "生", "事": "時"}
VARIANT = {"高": "髙", "斎": "齋", "辺": "邊", "崎": "﨑", "吉": "𠮷", "沢": "澤", "浜": "濱"}
FORMS = {"株式会社": ["株式会社{}", "{}株式会社", "(株){}", "㈱{}", "{}（株）", "{} Co., Ltd.", "{}"],
         "有限会社": ["有限会社{}", "{}有限会社", "(有){}", "{}㈲", "{}"],
         "合同会社": ["合同会社{}", "{}合同会社", "{} LLC", "{}"]}
EN_FORM = {"株式会社": "Co., Ltd.", "有限会社": "Ltd.", "合同会社": "LLC"}

def render_noisy(c):  # 名簿 B は表記がゆれている
    stem, biz = c["stem"][0], c["biz"][0]
    if random.random() < 0.2:
        stem, biz = HANKANA.get(stem, stem), HANKANA.get(biz, biz)
    core = stem + (" " if random.random() < 0.15 else "") + biz
    if random.random() < 0.1 and (pos := [i for i, ch in enumerate(core) if ch in TYPO]):
        i = random.choice(pos)
        core = core[:i] + TYPO[core[i]] + core[i + 1:]  # 誤字
    name = random.choice(FORMS[c["legal"]]).format(core)
    if random.random() < 0.08:
        name = f"{c['stem'][1]} {c['biz'][1]} {EN_FORM[c['legal']]}"  # 英文社名
    pref, city, towns = PLACES[c["place"]]
    town, (ch, ban, go) = c["town"], c["num"]
    if random.random() < 0.08:  # 移転前の古い住所。半分は別の市区
        if random.random() < 0.5:
            pref, city, towns = random.choice(PLACES)
        town, ch, ban, go = random.choice(towns), random.randint(1, 5), random.randint(1, 20), random.randint(1, 30)
    elif random.random() < 0.05:
        go += 1  # 号の誤り
    nums = random.choice([f"{ch}丁目{ban}番{go}号", f"{ch}-{ban}-{go}", f"{KAN[ch]}丁目{ban}-{go}", f"{ch}丁目{ban}番地{go}",
                          f"{ch}-{ban}-{go}".translate(ZEN)])
    address = ("" if random.random() < 0.3 else pref) + city + town + nums
    if random.random() < 0.2:
        address += f" {random.choice(['第一', '中央', 'サン', '駅前'])}ビル{random.randint(2, 9)}F"
    sur, given = c["rep"]
    if random.random() < 0.3:
        sur = "".join(VARIANT.get(x, x) for x in sur)  # 異体字
    rep = "" if random.random() < 0.15 else sur + random.choice(["", " ", "　"]) + given
    return {"name": name, "address": address, "rep": rep}

combos = [(s, b) for s in STEMS for b in BIZ]
random.shuffle(combos)
companies = [new_company(s, b) for s, b in combos[:240]]
A = pd.DataFrame([{"a_id": f"A{i:03d}"} | render_master(c) for i, c in enumerate(companies)])

rows = [render_noisy(companies[i]) | {"true": f"A{i:03d}", "kind": "同じ会社"} for i in random.sample(range(240), 200)]
for s, b in combos[240:270]:  # 同じ語幹の会社と同じ市区にある、名前の似た別会社 (さくら物産に対するさくら商事)
    near = random.choice([c for c in companies if c["stem"] == s])
    rows.append(render_noisy(new_company(s, b, near["place"])) | {"true": None, "kind": "似た名前の別会社"})
for s, b in combos[270:]:
    rows.append(render_noisy(new_company(s, b)) | {"true": None, "kind": "A にない会社"})
for i in random.sample(range(240), 40):  # 同名の別会社。半分は同じ市区にある
    c = companies[i]
    rows.append(render_noisy(new_company(c["stem"], c["biz"], c["place"] if random.random() < 0.5 else None))
                | {"true": None, "kind": "同名の別会社"})
random.shuffle(rows)
B = pd.DataFrame(rows)
B.insert(0, "b_id", [f"B{i:03d}" for i in range(len(B))])
B["split"] = ["dev" if i % 2 == 0 else "test" for i in range(len(B))]  # しきい値は dev で決め、test で評価する
print(len(A), len(B), B.kind.value_counts().to_dict())
pd.set_option("display.max_colwidth", None)
pd.concat([A.head(3), B[["b_id", "name", "address", "rep", "true", "kind"]].head(8)])
```

```python
# 正規化: NFKC (この章の「Unicode 正規化 (NFKC) で全角英数・半角カナを揃える」) → 法人格の除去、住所は丁目・番・号をハイフンに
# 住所の正規化はこの章の「日本の住所の表記ゆれを正規化して要素に分ける」の簡易な方法を縮めたもの
import re
import unicodedata
from kanjize import kanji2number

# 英語の法人格は前後が英字でないときだけ除く ("prince" の "inc" などを消さない。\b は日本語の直後で働かないので使わない)
LEGAL_RE = re.compile(r"株式会社|有限会社|合同会社|\((?:株|有|同)\)|(?<![a-z])(?:co\.?,?\s*ltd|ltd|inc|llc|corporation)(?![a-z])\.?")
ITAIJI = str.maketrans({"髙": "高", "齋": "斎", "邊": "辺", "邉": "辺", "﨑": "崎", "𠮷": "吉", "澤": "沢", "濱": "浜"})  # NFKC では揃わない
PREF_RE = re.compile(r"^(東京都|北海道|(?:京都|大阪)府|.{2,3}県)")
CITY_RE = re.compile(r"^(.+?市.+?区|.+?[市区町村])")

def strip_legal(s):  # 法人格を除き、空白を詰める前の社名 (埋め込みに使う)
    s = unicodedata.normalize("NFKC", s).lower()  # ㈱ → (株)、（株） → (株)、全角英数 → 半角、半角カナ → 全角
    return re.sub(r"\s+", " ", LEGAL_RE.sub("", s)).strip(" ,.")

def norm_name(s):
    return re.sub(r"[\W_]+", "", strip_legal(s))  # 空白と記号を除く

def norm_person(s):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s)).translate(ITAIJI)

def norm_addr(s):
    s = re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))
    s = re.sub(r"[一二三四五六七八九十]+(?=丁目)", lambda m: str(kanji2number(m.group())), s)
    s = re.sub(r"(?<=\d)[‐−ー―](?=\d)", "-", s)
    s = re.sub(r"(?<=\d)(丁目|番地|番)(?=\d)", "-", s)
    s = re.match(r"(.*?\d+(?:-\d+)*)", s).group(1) if re.search(r"\d", s) else s  # 番地より後ろ (建物名) を落とす
    s = PREF_RE.sub("", s)  # 都道府県は抜けていることが多いので比べない
    m = CITY_RE.match(s)
    return s, (m.group(1) if m else "")

for df in (A, B):
    df["n_name"] = df.name.map(norm_name)
    df["e_name"] = df.name.map(strip_legal)
    df["n_rep"] = df.rep.map(norm_person)
    df[["n_addr", "city"]] = df.address.map(norm_addr).tolist()
B[["name", "n_name", "address", "n_addr", "city", "rep", "n_rep"]].head(8)
```

```python
# ブロッキング: 全ペア (240 × 300) を比べる代わりに、同じキーを持つペアだけを候補にする
from collections import defaultdict

KEYS = {"市区": lambda r: r.city, "社名の先頭 2 文字": lambda r: r.n_name[:2]}

def block(key_names):
    index = defaultdict(set)
    for r in A.itertuples():
        for k in key_names:
            index[k, KEYS[k](r)].add(r.a_id)
    return {r.b_id: set().union(*(index[k, KEYS[k](r)] for k in key_names)) for r in B.itertuples()}

true_pairs = {(r.b_id, r.true) for r in B.itertuples() if r.true}
for keys in [["市区"], ["社名の先頭 2 文字"], ["市区", "社名の先頭 2 文字"]]:
    cands = block(keys)
    n = sum(map(len, cands.values()))
    found = sum((b, a) in true_pairs for b, s in cands.items() for a in s)
    print(f"{' or '.join(keys):24} 候補 {n:5} ペア ({n / (len(A) * len(B)):.1%})  正解ペアの網羅率 {found}/{len(true_pairs)}")
```

```python
# 候補ペアごとに、社名・住所・代表者名の類似度を RapidFuzz で求める (この章の「編集距離とあいまい文字列マッチング (RapidFuzz)」)
from rapidfuzz import fuzz

cands = block(["市区", "社名の先頭 2 文字"])
pairs = pd.DataFrame([(b, a) for b, s in cands.items() for a in sorted(s)], columns=["b_id", "a_id"])
pairs = pairs.merge(B.add_suffix("_b").rename(columns={"b_id_b": "b_id"}), on="b_id").merge(
    A.add_suffix("_a").rename(columns={"a_id_a": "a_id"}), on="a_id")
pairs["label"] = pairs.a_id == pairs.true_b
pairs["name_fz"] = [fuzz.ratio(x, y) / 100 for x, y in zip(pairs.n_name_b, pairs.n_name_a)]
pairs["addr_fz"] = [fuzz.ratio(x, y) / 100 for x, y in zip(pairs.n_addr_b, pairs.n_addr_a)]
pairs["rep_fz"] = [fuzz.ratio(x, y) / 100 if x and y else 0.5 for x, y in zip(pairs.n_rep_b, pairs.n_rep_a)]  # 空欄は 0.5
pairs.groupby("label")[["name_fz", "addr_fz", "rep_fz"]].describe().T.loc[(slice(None), ["mean", "25%", "50%", "75%"]), :].round(2)
```

```python
# 社名の意味的な近さを埋め込みでも求める (文埋め込みとクラスタリングの章と同じ multilingual-e5-small、MIT ライセンス)
# Colab の CPU で数分 (ライブラリの読み込みとモデルのダウンロードを含む)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
names = sorted(set(A.e_name) | set(B.e_name))
vec = dict(zip(names, model.encode(["query: " + n for n in names], normalize_embeddings=True, batch_size=64)))
pairs["name_emb"] = [float(vec[x] @ vec[y]) for x, y in zip(pairs.e_name_b, pairs.e_name_a)]

examples = [("sakura bussan", "さくら物産"), ("fuji seiki", "富士精機"), ("yamato unyu", "大和運輸"), ("富士清機", "富士精機"),
            ("さくら物産", "さくら商事"), ("さくら物産", "みどり物産"), ("サンライズ印刷", "sunrise printing")]
for x, y in examples:
    ex = model.encode([f"query: {x}", f"query: {y}"], normalize_embeddings=True)
    print(f"{x:16} / {y:10}  RapidFuzz {fuzz.ratio(norm_name(x), norm_name(y)) / 100:.2f}  埋め込み {float(ex[0] @ ex[1]):.2f}")
```

```python
# 類似度を組み合わせて「同じ会社である確率」にする (ロジスティック回帰)。重みは dev の候補ペアで学習する
import numpy as np
from sklearn.linear_model import LogisticRegression

FEATS = {"社名 (RapidFuzz)": ["name_fz"], "社名 (埋め込み)": ["name_emb"], "社名+住所+代表者": ["name_fz", "addr_fz", "rep_fz"],
         "社名+住所+代表者+埋め込み": ["name_fz", "addr_fz", "rep_fz", "name_emb"]}
dev = pairs[pairs.split_b == "dev"]
best_th = {}  # 特徴量ごとの、dev で F1 が最大になるしきい値
for name, cols in FEATS.items():
    clf = LogisticRegression().fit(dev[cols], dev.label)
    pairs[name] = clf.predict_proba(pairs[cols])[:, 1]
    print(f"{name:24} 重み", dict(zip(cols, clf.coef_[0].round(1).tolist())))

def outcomes(score, th, split):
    """B の各レコードについて最も確率の高い候補を 1 つ選び、しきい値以上なら「同じ会社」と判定する。
    レコードごとに (判定したか, 正しい相手か, 本当は A にあるか) を返す"""
    p = pairs[pairs.split_b == split].sort_values(score, ascending=False).drop_duplicates("b_id").set_index("b_id")
    b = B[B.split == split].set_index("b_id")
    pred = (p[score] >= th).reindex(b.index, fill_value=False)
    correct = pred & (p.a_id.reindex(b.index) == b.true)
    return np.c_[pred, correct, b.true.notna()].astype(int)  # ブロッキングで漏れた正解も「本当は A にある」に入る

def prf(o):
    n_pred, tp, n_true = o.sum(0)
    prec, rec = tp / max(n_pred, 1), tp / n_true
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)

def evaluate(score, th, split):
    return prf(outcomes(score, th, split))

def thresholds(score):  # dev の各レコードの最上位候補のスコアをすべてしきい値の候補にする
    return np.unique(pairs[pairs.split_b == "dev"].groupby("b_id")[score].max())

rng = np.random.default_rng(0)
table = []
for name in FEATS:
    th = best_th[name] = max(thresholds(name), key=lambda t: evaluate(name, t, "dev")[2])  # dev で F1 が最大になるしきい値
    o = outcomes(name, th, "test")
    f1s = [prf(o[i])[2] for i in rng.integers(0, len(o), (1000, len(o)))]  # test のレコードを復元抽出して F1 の 95% 区間
    table.append([name, th, *evaluate(name, th, "dev"), *prf(o), f"{np.percentile(f1s, 2.5):.3f}–{np.percentile(f1s, 97.5):.3f}"])
pd.DataFrame(table, columns=["特徴量", "しきい値", "dev 適合率", "dev 再現率", "dev F1", "test 適合率", "test 再現率", "test F1",
                            "test F1 の 95% 区間"]).round(3)
```

```python
# しきい値を動かすと適合率と再現率が入れ替わる。実務では 2 つのしきい値で「自動で統合」「人が確認」「別の会社」に分けることが多い
best = "社名+住所+代表者+埋め込み"
print(pd.DataFrame([(t, *evaluate(best, t, "dev")) for t in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]],
                   columns=["しきい値", "適合率", "再現率", "F1"]).round(3).to_string(index=False))

ths = thresholds(best)
hi = min(t for t in ths if evaluate(best, t, "dev")[0] == 1.0)  # dev で誤統合が 0 になる最小のしきい値
lo = max(t for t in ths if evaluate(best, t, "dev")[1] == evaluate(best, ths[0], "dev")[1])  # dev の再現率を下げない最大のしきい値
p = pairs[pairs.split_b == "test"].sort_values(best, ascending=False).drop_duplicates("b_id")
auto, review = p[p[best] >= hi], p[(p[best] >= lo) & (p[best] < hi)]
print(f"\n自動で統合 (確率 ≥ {hi:.3f}): {len(auto)} 件、うち誤り {int((auto.a_id != auto.true_b).sum())} 件")
print(f"人が確認 ({lo:.3f} ≤ 確率 < {hi:.3f}): {len(review)} 件、うち同じ会社 {int((review.a_id == review.true_b).sum())} 件")
print(f"別の会社とみなす: {len(B[B.split == 'test']) - len(auto) - len(review)} 件")
```

```python
# test での誤りを見る (dev で F1 最大のしきい値)
th = best_th[best]
p = pairs[pairs.split_b == "test"].sort_values(best, ascending=False).drop_duplicates("b_id")
fp = p[(p[best] >= th) & (p.a_id != p.true_b)]
fn = B[(B.split == "test") & B.true.notna() & ~B.b_id.isin(p[(p[best] >= th) & (p.a_id == p.true_b)].b_id)]
cols = ["name_b", "name_a", "address_b", "address_a", "rep_b", "rep_a", "kind_b", best]
print("誤って統合したもの")
display(fp[cols].round(2))
print("見逃したもの")
display(fn.merge(A, left_on="true", right_on="a_id", suffixes=("_b", "_a"))[["name_b", "name_a", "address_b", "address_a", "rep_b", "rep_a"]])
en = p[p.name_b.str.match(r"[A-Za-z]")]  # 英文社名のレコードは何で拾えたか
print("英文社名のレコード")
display(en[["name_b", "name_a", "kind_b", "name_fz", "name_emb", "addr_fz", "rep_fz", best]].round(2))
```

- 名寄せは「正規化 → ブロッキングで候補を絞る → 項目ごとの類似度 → 組み合わせてしきい値で判定 → 評価」の順に作る。ブロッキングのキーは 1 つだと漏れる。市区だけでは別の市区へ移転した 9 件を、社名の先頭 2 文字だけでは英文社名や先頭の誤字（大和→大話）の 16 件を落とした。両方の和を取ると全ペアの 14.1% に絞っても、正解ペアは 1 件も落ちなかった。ただし移転や誤字はこちらで決めた割合で入れたもので、実データでは住所の欠落や両方のキーがずれるレコードもある。
- 社名だけでは同名の別会社と区別できない（test の F1 0.827、95% 区間 0.762–0.879）。住所と代表者名を加えると 0.952 (0.915–0.979) になった。残った誤統合 5 件は、どれも同じ市区にある同名・似た名前の別会社である。見逃し 4 件は、英文社名と、代表者名が空欄で住所も違うレコードだった。
- 埋め込みは英文社名と日本語の社名を近づける（上の例で 0.79〜0.89）が、別の会社の「さくら物産 / みどり物産」も 0.88、「さくら物産 / さくら商事」は 0.96 になる。test の英文社名でも、同じ会社の 0.82〜0.90 に対し、別の会社の「Kyowa Unyu / 朝日運輸」が 0.78 で、差は小さい。埋め込みだけでは F1 0.784 で、加えても 0.952 のままだった。英文社名の同じ会社は確率が 0.12〜0.17 にとどまり、代表者名が空欄の 2 件は 0.01 になった。英文社名が多い名簿では、読みのローマ字に揃える方法（この章の「漢字かな交じり文をローマ字・ひらがなにする」）も試すとよい。
- しきい値は用途で決める。dev で誤統合が 0 になる確率 0.643 以上を自動で統合すると、test では 82 件を誤りなく統合し、0.046〜0.643 の 32 件（うち同じ会社 10 件）が人の確認に回った。残る 36 件にも、確率 0.01 の英文社名の同じ会社が 2 件入っている。確率は候補ペアの正例と負例の比率に左右されるので、ブロッキングを変えたら学習し直す。
- 法人格を取り除いたので「株式会社さくら物産」と「有限会社さくら物産」も同じになる。法人番号（国税庁が法人に指定する 13 桁の番号）が両方の名簿にあれば、それで突き合わせるのが確実である。
- 髙・齋・﨑などの異体字は NFKC では揃わないので、対応表で寄せた。人名の異体字を同一視してよいかは用途によって違う。
- このデータは自分で作ったもので、test は 150 件（うち同じ会社 94 件）しかない。実務ではしきい値を実データの数百ペアの人手のラベルで決める。大きな名簿には、Fellegi–Sunter のモデルでラベルなしでも重みを推定できる Splink（MIT ライセンス）などのライブラリがある。

## 巨大な辞書の語をテキストから一括で探す (Aho-Corasick) {aho_corasick}

数万〜数百万語の辞書（商品名、病名、NG ワードなど）に載っている語を、長いテキストから高速にすべて見つけたい。

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
