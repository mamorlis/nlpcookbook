# 文法誤り訂正と言語教育

学習者の作文を訂正する、訂正箇所を取り出して分類・評価する、教材づくりに使う。

## 2 つの文の違いを色付きで表示する {diff_highlight}

訂正前と訂正後の文を比べて、どこが削除され、どこが追加されたのかを一目で分かるように表示したい。

```python
import difflib
from IPython.display import HTML

def diff_html(src, tgt):
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, src, tgt, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(src[i1:i2])
        if op in ("delete", "replace"):
            out.append(f"<del style='background:#ffd7d5; color:#82071e'>{src[i1:i2]}</del>")
        if op in ("insert", "replace"):
            out.append(f"<ins style='background:#ccffd8; color:#116329; text-decoration:none'>{tgt[j1:j2]}</ins>")
    return "".join(out)

pairs = [("私は昨日学校を行きました。", "私は昨日学校に行きました。"),
         ("この本はとても面白いでした。", "この本はとても面白かったです。"),
         ("I has went to school yesterday and buyed a apple.", "I went to school yesterday and bought an apple.")]
HTML("<br>".join(f"<span style='font-size:1.2em; line-height:2'>{diff_html(s, t)}</span>" for s, t in pairs))
```

```python
# 端末やログ向けのテキスト表示
def diff_text(src, tgt):
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, src, tgt, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(src[i1:i2])
        else:
            out.append(f"[{src[i1:i2]}→{tgt[j1:j2]}]")
    return "".join(out)

for s, t in pairs:
    print(diff_text(s, t))
```

- 文字単位の差分は細かすぎることがある (「面白い**でし**た」→「面白**かっ**た**です**」)。単語単位で見たいときは次項のように分かち書きしてから比べる。
- `autojunk=False` を付けないと、長い文で頻出文字が無視されて差分が粗くなることがある。

## 訂正前後の文から編集 (置換・挿入・削除) を取り出す {edit_extraction}

誤り文と訂正文の組から、「どの単語がどの単語に直されたか」という編集の一覧を作りたい。誤り傾向の集計に使える。

```python
!pip install -q fugashi unidic-lite
```

```python
import difflib
from collections import Counter
from fugashi import Tagger

tagger = Tagger()

def tokenize(text):
    return [(w.surface, w.feature.pos1) for w in tagger(text)]

def extract_edits(src, tgt):
    s, t = tokenize(src), tokenize(tgt)
    sm = difflib.SequenceMatcher(None, [w for w, _ in s], [w for w, _ in t], autojunk=False)
    edits = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        o, c = "".join(w for w, _ in s[i1:i2]), "".join(w for w, _ in t[j1:j2])
        pos = (s[i1:i2] or t[j1:j2])[0][1]                       # 編集箇所の先頭の品詞
        kind = {"replace": "置換", "delete": "余剰", "insert": "脱落"}[op]
        edits.append({"type": f"{kind}:{pos}", "orig": o, "corr": c, "span": (i1, i2)})
    return edits

pairs = [("私は昨日学校を行きました。", "私は昨日学校に行きました。"), ("友達と映画を見に行くました。", "友達と映画を見に行きました。"),
         ("日本の食べ物はおいしいと思うます。", "日本の食べ物はおいしいと思います。"), ("私は本読むのが好きです。", "私は本を読むのが好きです。"),
         ("彼は学生のです。", "彼は学生です。"), ("図書館を勉強します。", "図書館で勉強します。")]
stats = Counter()
for src, tgt in pairs:
    edits = extract_edits(src, tgt)
    stats.update(e["type"] for e in edits)
    print(src, "→", [(e["type"], e["orig"], e["corr"]) for e in edits])
print(stats.most_common())
```

- 「置換:助詞」が多ければ助詞の指導を重点的に、といった分析ができる。
- 語順の入れ替えや、複数の誤りが隣り合う場合は、差分アルゴリズムだけでは人間の直感と違う対応付けになることがある。英語では、言語学的な情報を使って対応付ける ERRANT が標準 (次項)。

## ERRANT で英語の訂正を誤りタイプ付きの編集に変換する {errant}

英語の誤り文と訂正文から編集を抽出し、「動詞の時制」「冠詞」「綴り」などの誤りタイプを自動で付けたい。

```python
!pip install -q errant
!python -m spacy download en_core_web_sm -q
```

```python
import errant

annotator = errant.load("en")
orig = annotator.parse("I has went to school yesterday and buyed a apple , but it were not delicious .")
cor = annotator.parse("I went to school yesterday and bought an apple , but it was not delicious .")
edits = annotator.annotate(orig, cor)
for e in edits:
    print(f"{e.o_start:2}-{e.o_end:2}  {e.o_str or '∅':8} → {e.c_str or '∅':8} {e.type}")
```

```python
# M2 形式 (GEC の標準的なアノテーション形式) で出力
print("S", orig.text)
for e in edits:
    print(e.to_m2())
```

- タイプは「操作:品詞:詳細」。操作は M (Missing、脱落)、U (Unnecessary、余剰)、R (Replacement、置換)。
- 入力は空白でトークン化済みの文を想定している。句読点の前に空白を入れる。
- ERRANT は BEA-2019 Shared Task の公式評価ツールでもある。コマンドラインでは `errant_parallel` で M2 を作り、`errant_compare` でスコアを計算する。

## M2 形式のファイルを読み書きする {m2_format}

CoNLL-2014 や BEA-2019 などの GEC データセットで使われる M2 形式を読み込み、訂正文を復元したい。

```python
m2 = """S I has went to school yesterday and buyed a apple .
A 1 2|||U:VERB:TENSE||||||REQUIRED|||-NONE-|||0
A 7 8|||R:VERB:INFL|||bought|||REQUIRED|||-NONE-|||0
A 8 9|||R:DET|||an|||REQUIRED|||-NONE-|||0
A 1 3|||R:VERB:TENSE|||went|||REQUIRED|||-NONE-|||1
A 7 8|||R:VERB:INFL|||bought|||REQUIRED|||-NONE-|||1
A 8 9|||R:DET|||an|||REQUIRED|||-NONE-|||1

S This sentence is correct .
A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0
"""

def read_m2(text):
    """[(原文のトークン列, {アノテータ ID: [(開始, 終了, タイプ, 訂正), ...]}), ...]"""
    data = []
    for block in text.strip().split("\n\n"):
        lines = block.split("\n")
        tokens = lines[0][2:].split()
        edits = {}
        for line in lines[1:]:
            span, typ, corr, _, _, annotator = line[2:].split("|||")
            start, end = map(int, span.split())
            edits.setdefault(int(annotator), [])
            if typ != "noop":
                edits[int(annotator)].append((start, end, typ, corr))
        data.append((tokens, edits))
    return data

def apply_edits(tokens, edits):
    out = list(tokens)
    for start, end, _, corr in sorted(edits, reverse=True):     # 後ろから適用すれば位置がずれない
        out[start:end] = corr.split() if corr and corr != "-NONE-" else []
    return " ".join(out)

for tokens, edits in read_m2(m2):
    print("原文:", " ".join(tokens))
    for annotator, es in edits.items():
        print(f"  アノテータ {annotator}: {apply_edits(tokens, es)}   {[(s, e, t) for s, e, t, _ in es]}")
```

- `S` 行が原文、`A` 行が 1 つの編集 (`開始 終了|||タイプ|||訂正|||必須か|||コメント|||アノテータ ID`)。位置はトークン単位で、終了は含まない。
- 訂正が不要な文は `noop` の行が 1 つ付く。アノテータが複数いる場合は、同じ文に対して ID の違う編集の集合が並ぶ。

## 文法誤り訂正を編集単位の F0.5 で評価する {gec_f05}

GEC システムの出力を、「正しい編集をどれだけ提案できたか」で評価したい。再現率より適合率を重視する F0.5 を使う。

```python
!pip install -q errant
!python -m spacy download en_core_web_sm -q
```

```python
import errant

annotator = errant.load("en")

def edit_set(src, tgt):
    o, c = annotator.parse(src), annotator.parse(tgt)
    return {(e.o_start, e.o_end, e.c_str) for e in annotator.annotate(o, c)}

def gec_score(sources, hypotheses, references, beta=0.5):
    tp = fp = fn = 0
    for src, hyp, ref in zip(sources, hypotheses, references):
        h, r = edit_set(src, hyp), edit_set(src, ref)
        tp += len(h & r); fp += len(h - r); fn += len(r - h)
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = (1 + beta ** 2) * p * r / (beta ** 2 * p + r) if p + r else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "P": round(p, 3), "R": round(r, 3), f"F{beta}": round(f, 3)}

src = ["I has went to school yesterday .", "She like apples very much .", "He go to the park every days ."]
ref = ["I went to school yesterday .", "She likes apples very much .", "He goes to the park every day ."]
systems = {
    "何もしない": src,
    "慎重 (1 か所だけ直す)": ["I went to school yesterday .", "She like apples very much .", "He go to the park every days ."],
    "積極的 (直しすぎ)": ["I went to the school yesterday .", "She likes apples a lot .", "He goes to a park every day ."],
    "完全": ref,
}
for name, hyp in systems.items():
    print(f"{name:20}", gec_score(src, hyp, ref))
```

- F0.5 は適合率を再現率の 2 倍重視する。間違った訂正を学習者に見せる害のほうが、見逃しより大きいという考え方。
- 正しい訂正は 1 通りではない。参照が 1 つだけだと、妥当な別解 (「very much → a lot」) が誤りとして数えられる。複数参照の場合は、文ごとに最も F 値が高くなる参照を選ぶ。
- 流暢性まで含めた訂正の評価には GLEU や、参照なしの指標 (IMPARA、SOME など) が使われる。

## 学習済みモデルで英語の文法誤りを訂正する {gec_model_en}

英語の文法誤り訂正を、公開されているファインチューニング済みモデルですぐに試したい。

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

name = "vennify/t5-base-grammar-correction"   # JFLEG で学習した T5
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSeq2SeqLM.from_pretrained(name).eval()

def correct(sentences, num_beams=5):
    batch = tokenizer(["grammar: " + s for s in sentences], return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        out = model.generate(**batch, num_beams=num_beams, max_new_tokens=128)
    return tokenizer.batch_decode(out, skip_special_tokens=True)

sentences = ["I has went to school yesterday and buyed a apple.", "She don't likes swimming in the see.",
             "Despite of the rain, we enjoyed.", "This sentence is already correct.", "He suggested me to go there."]
for s, c in zip(sentences, correct(sentences)):
    print(f"{s}\n  → {c}")
```

```python
# 上位 3 つの訂正候補を見る
batch = tokenizer(["grammar: He suggested me to go there."], return_tensors="pt")
with torch.no_grad():
    out = model.generate(**batch, num_beams=5, num_return_sequences=3, max_new_tokens=64)
print(tokenizer.batch_decode(out, skip_special_tokens=True))
```

- T5 は入力の先頭にタスクを表す接頭辞 (`grammar: `) を付ける約束で学習されている。付け忘れると訂正されない。
- 系列変換型は書き換えすぎる (正しい文まで変える) ことがある。編集タグを予測する方式 (GECToR) は高速で、過剰訂正が少ない。
- 現在は LLM による訂正が流暢性の面で強いが、必要以上に書き換える傾向がある。「最小限の訂正」を求める評価 (CoNLL-2014、BEA-2019) では専用モデルが有利なことも多い。

## 規則で誤りを注入して GEC の擬似学習データを作る {synthetic_errors}

文法誤り訂正の学習データが足りない。正しい文に人工的な誤りを入れて、(誤り文, 正しい文) のペアを大量に作りたい。

```python
!pip install -q fugashi unidic-lite
```

```python
import random
from fugashi import Tagger

tagger = Tagger()
PARTICLES = ["が", "を", "に", "で", "は", "と", "へ", "の"]

def corrupt(sentence, rng, p=0.15):
    """学習者に多い誤りを模して、助詞の置換・脱落・余剰と、活用語尾の誤りを確率的に入れる"""
    out = []
    for w in tagger(sentence):
        surface, pos1, pos2 = w.surface, w.feature.pos1, w.feature.pos2
        if pos1 == "助詞" and pos2 in ("格助詞", "係助詞") and surface in PARTICLES and rng.random() < p:
            r = rng.random()
            if r < 0.6:
                out.append(rng.choice([x for x in PARTICLES if x != surface]))   # 置換
            elif r < 0.8:
                pass                                                              # 脱落
            else:
                out += [surface, rng.choice(PARTICLES)]                           # 余剰
        elif pos1 == "動詞" and w.feature.cForm and w.feature.cForm.startswith("連用形") and rng.random() < p:
            out.append(w.feature.orthBase or surface)                             # 活用させずに辞書形のままにする
        else:
            out.append(surface)
    return "".join(out)

rng = random.Random(0)
clean = ["私は昨日友達と映画を見に行きました。", "図書館で日本語の本を読みます。", "母が作った料理はとてもおいしいです。", "来週の月曜日に東京へ出張します。"]
for s in clean:
    for _ in range(2):
        noisy = corrupt(s, rng, p=0.3)
        if noisy != s:
            print(f"{noisy}  →  {s}")
```

- 擬似データで事前学習してから実データでファインチューニングするのが、GEC の定番の手順。
- 誤りの種類と割合は、実際の学習者コーパスでの分布に合わせるほど効果が高い。ここの規則は最小限の例。
- 逆翻訳 (正しい文 → 誤り文を生成するモデル) や、LLM に誤りを書かせる方法もある。

## 日本語の文章の読みやすさを測る {readability_ja}

日本語の文章の難しさを、文の長さ・漢字の割合・語種などの簡単な指標で数値化したい (教材選び、やさしい日本語の確認)。

```python
!pip install -q fugashi unidic-lite regex
```

```python
import re
import pandas as pd
import regex
from fugashi import Tagger

tagger = Tagger()

def readability(text):
    sents = [s for s in re.split(r"(?<=[。！？])", text) if s.strip()]
    words = [(w.surface, w.feature.pos1, w.feature.goshu) for s in sents for w in tagger(s) if w.feature.pos1 not in ("補助記号", "空白")]
    chars = regex.findall(r"\S", text)
    content = [w for w in words if w[1] in ("名詞", "動詞", "形容詞", "形状詞", "副詞")]
    return {
        "文数": len(sents),
        "平均文長 (語)": round(len(words) / len(sents), 1),
        "漢字率": round(len(regex.findall(r"\p{Han}", text)) / len(chars), 2),
        "カタカナ率": round(len(regex.findall(r"\p{Katakana}", text)) / len(chars), 2),
        "漢語率": round(sum(w[2] == "漢" for w in content) / len(content), 2),
        "和語率": round(sum(w[2] == "和" for w in content) / len(content), 2),
        "読点/文": round(text.count("、") / len(sents), 1),
    }

texts = {
    "やさしい日本語": "じしんが おきました。たてものの そとに でて ください。エレベーターは つかわないで ください。ちかくの がっこうに にげて ください。",
    "新聞": "政府は 12 日、来年度予算案を閣議決定した。一般会計の歳出総額は過去最大を更新し、社会保障費と防衛費の増加が全体を押し上げた。",
    "法律": "この法律は、個人情報の適正な取扱いに関し、基本理念及び政府による基本方針の作成その他の個人情報の保護に関する施策の基本となる事項を定め、国及び地方公共団体の責務等を明らかにすることを目的とする。",
}
pd.DataFrame({k: readability(v) for k, v in texts.items()})
```

- 文が長く、漢語 (音読みの語) が多いほど難しく感じられる。UniDic の語種 (`goshu`) で和語・漢語・外来語を区別できる。
- 学習者向けには、語彙の難易度 (日本語能力試験の級別語彙など) や、係り受けの距離も重要な手がかりになる。
- 英語の Flesch-Kincaid のような定番の式は日本語にはない。日本語の公開ツールとしては jReadability などがある。

## 単語の頻度から語彙の難しさを推定する (wordfreq) {word_frequency_level}

文中のどの単語が「珍しい (難しい)」のかを、大規模コーパスでの出現頻度から見積もりたい。

```python
!pip install -q "wordfreq[cjk]" fugashi unidic-lite
```

```python
from wordfreq import top_n_list, zipf_frequency

# Zipf 尺度: 10 億語あたりの出現回数の常用対数。7 = 極めて高頻度、3 以下 = 低頻度
for w in ["the", "language", "linguistics", "morphology", "agglutinative"]:
    print(f"{w:15}{zipf_frequency(w, 'en'):.2f}")
for w in ["猫", "言語", "翻訳", "形態素", "膠着語"]:
    print(f"{w}\t{zipf_frequency(w, 'ja'):.2f}")
print(top_n_list("ja", 15))
```

```python
from fugashi import Tagger

tagger = Tagger()

def difficult_words(text, lang="ja", threshold=4.0):
    """内容語のうち、Zipf 頻度がしきい値未満のものを返す"""
    out = []
    for w in tagger(text):
        if w.feature.pos1 in ("名詞", "動詞", "形容詞", "形状詞", "副詞"):
            base = w.feature.orthBase or w.surface
            z = zipf_frequency(base, lang)
            if z < threshold:
                out.append((base, round(z, 2)))
    return out

for text in ["猫が庭で寝ている。", "当該事案の瑕疵について遺漏なく精査されたい。", "形態素解析器が膠着語の接辞を分離する。"]:
    print(text, difficult_words(text))
```

- wordfreq は Wikipedia、字幕、Web、SNS など複数の資源を統合した頻度表で、40 以上の言語に対応している (データは 2021 年までで更新終了)。
- 頻度は難しさの近似にすぎない。高頻度でも多義で難しい語 (「かける」「とる」) や、学習者の母語によって易しい語 (漢字圏の学習者にとっての漢語) がある。
- 日本語の教育語彙としては、日本語教育語彙表や旧日本語能力試験の出題基準などが使われる。

## 穴埋め問題 (クローズテスト) を自動で作る {cloze_generation}

文章から語を抜いて穴埋め問題を作り、もっともらしい誤答選択肢も自動で用意したい。

```python
!pip install -q fugashi unidic-lite
```

```python
import random
import torch
from fugashi import Tagger
from transformers import AutoModelForMaskedLM, AutoTokenizer

tagger = Tagger()
name = "tohoku-nlp/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForMaskedLM.from_pretrained(name).eval()

def make_cloze(sentence, target_pos=("名詞", "動詞", "形容詞"), n_distractors=3, seed=0):
    rng = random.Random(seed)
    words = [(w.surface, w.feature.pos1, w.feature.pos2) for w in tagger(sentence)]
    candidates = [i for i, (s, p1, p2) in enumerate(words) if p1 in target_pos and p2 not in ("非自立可能", "数詞") and len(tokenizer.tokenize(s)) == 1]
    if not candidates:
        return None
    i = rng.choice(candidates)
    answer, pos = words[i][0], words[i][1]
    masked = "".join(tokenizer.mask_token if j == i else s for j, (s, _, _) in enumerate(words))

    enc = tokenizer(masked, return_tensors="pt")
    mask_pos = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero()[0, 0]
    with torch.no_grad():
        probs = model(**enc).logits[0, mask_pos].softmax(-1)
    distractors = []
    for tok_id in probs.argsort(descending=True)[5:200].tolist():     # 上位すぎる語は「別の正解」になりうるので避ける
        tok = tokenizer.convert_ids_to_tokens(tok_id)
        analysis = list(tagger(tok))
        if len(analysis) == 1 and analysis[0].feature.pos1 == pos and tok != answer and not tok.startswith("##"):
            distractors.append(tok)
        if len(distractors) == n_distractors:
            break
    choices = distractors + [answer]
    rng.shuffle(choices)
    return {"question": masked.replace(tokenizer.mask_token, "（　　）"), "choices": choices, "answer": answer}

for s in ["太郎は図書館で本を借りた。", "雨が降ったので、試合は中止になった。", "この料理はとても辛いので、水をたくさん飲んだ。"]:
    q = make_cloze(s)
    print(q["question"], " / ".join(q["choices"]), f"(正解: {q['answer']})")
```

- 誤答選択肢は「同じ品詞で、文脈にそこそこ合うが、正解ではない語」が良い。マスク言語モデルの中位の候補はその性質に近い。
- 自動生成した選択肢には、正解になりうるものが混ざる。出題前に人間が確認する。
- 特定の文法項目 (助詞、活用) を問いたいときは、`target_pos` を変える。
