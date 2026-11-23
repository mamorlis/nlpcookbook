# 文法誤り訂正と文章校正

学習者の作文を訂正する、訂正箇所を取り出して分類・評価する（ERRANT、M2 形式、F0.5、参照なし評価）、日本語の文章の校正すべき箇所を指摘する、スペルミスを直す。リーダビリティ測定・語彙平易化・問題生成・自動採点は「言語教育・リーダビリティ・自動採点」の章を参照。文法誤り訂正や言語処理の応用については、姉妹編『[自然言語処理の教科書](https://github.com/mamorlis/nlpbook)』の第 3 章も参照。

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

- 文字単位の差分は細かすぎることがある（「面白い**でし**た」→「面白**かっ**た**です**」）。単語単位で見たいときは次項のように分かち書きしてから比べる。
- `autojunk=False` を付けないと、長い文で頻出文字が無視されて差分が粗くなることがある。

## 訂正前後の文から編集（置換・挿入・削除）を取り出す {edit_extraction}

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
- 語順の入れ替えや、複数の誤りが隣り合う場合は、差分アルゴリズムだけでは人間の直感と違う対応付けになることがある。英語では、言語学的な情報を使って対応付ける ERRANT が標準（次項）。
- 動的計画法によるトークンアライメントから連続編集を M2 / ERRANT 形式の編集スパンへ統合し、マルチリファレンス $F_{0.5}$ で評価する演習は、姉妹編『[自然言語処理の教科書](https://github.com/mamorlis/nlpbook)』第 3 章の [編集距離と n-gram 言語モデルによる誤り検出・訂正](https://github.com/mamorlis/nlpbook/blob/main/notebooks/ch3_edit_distance_lm.ipynb)（第 6 節）も参照。

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

- タイプは「操作:品詞:詳細」。操作は M（Missing、脱落）、U（Unnecessary、余剰）、R（Replacement、置換）。
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
- 複数アノテータの M2 編集スパンを復元・統合してマルチリファレンス Best-Annotator $F_{0.5}$ を算出する演習は、姉妹編『[自然言語処理の教科書](https://github.com/mamorlis/nlpbook)』第 3 章の [編集距離と n-gram 言語モデルによる誤り検出・訂正](https://github.com/mamorlis/nlpbook/blob/main/notebooks/ch3_edit_distance_lm.ipynb)（第 6 節）も参照。

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
- 正しい訂正は 1 通りではない。参照が 1 つだけだと、妥当な別解（「very much → a lot」）が誤りとして数えられる。複数参照の場合は、文ごとに最も F 値が高くなる参照を選ぶ。
- 流暢性まで含めた訂正の評価には GLEU や、参照なしの指標（IMPARA、SOME など）が使われる。
- Token Accuracy の罠（何もしないベースラインが約 90% になる問題）、$F_1$ との比較による過剰訂正ペナルティ、および 1 参照 vs 3 参照 Best-Annotator $F_{0.5}$ の比較演習は、姉妹編『[自然言語処理の教科書](https://github.com/mamorlis/nlpbook)』第 3 章の [編集距離と n-gram 言語モデルによる誤り検出・訂正](https://github.com/mamorlis/nlpbook/blob/main/notebooks/ch3_edit_distance_lm.ipynb)（第 6 節）も参照。

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
- 系列変換型は書き換えすぎる（正しい文まで変える）ことがある。逆に、直しきれないこともある。この出力では 2 文目が「She doesn't likes swimming in the see.」となり、likes と see の誤りが残った。編集タグを予測する方式 (GECToR) は高速で、過剰訂正が少ない。
- 現在は LLM による訂正が流暢性の面で強いが、必要以上に書き換える傾向がある。「最小限の訂正」を求める評価（CoNLL-2014、BEA-2019）では専用モデルが有利なことも多い。

## 規則で誤りを注入して GEC の擬似学習データを作る {synthetic_errors}

文法誤り訂正の学習データが足りない。正しい文に人工的な誤りを入れて、（誤り文, 正しい文）のペアを大量に作りたい。

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
- 逆翻訳（正しい文 → 誤り文を生成するモデル）や、LLM に誤りを書かせる方法もある。

## LLM で日本語の文法誤りを訂正して評価する {gec_ja_llm gpu}

日本語学習者の書いた文の誤りを訂正したい。LLM（GPU を使う）に訂正させ、参照訂正と比べて評価する。

```python
!pip install -q datasets
```

```python
import difflib, re
import sentencepiece as spm
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる（大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」を参照）。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()

@torch.no_grad()
def chat(prompt, max_new_tokens=80):
    ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to(llm.device)
    out = llm.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0]

def clean(text):
    """LLM が付け足しがちな「訂正後の文:」のような見出しや、末尾の注釈（「（誤りはありません）」）を取り除く"""
    text = re.sub(r"^(訂正後の文|訂正文|書き直した文|文)\s*[:：]\s*", "", text.strip())
    return re.sub(r"[（(][^（）()]*[）)]$", "", text).strip().strip("「」")

MINIMAL = ("次の文は日本語学習者が書いたものです。文法の誤りだけを、できるだけ少ない変更で訂正してください。"
           "誤りがなければそのまま出力してください。訂正後の文だけを 1 行で出力し、説明は書かないでください。\n\n文: {}")
FLUENT = "次の文は日本語学習者が書いたものです。自然で流暢な日本語に書き直してください。書き直した文だけを 1 行で出力してください。\n\n文: {}"

for s in ["図書館で本が借りました。", "私は毎日学校を行きます。", "昨日は友達と映画を見ます。", "この料理はとてもおいしいでした。", "私は東京に住んでいます。"]:
    print(f"{s}\n   最小限の訂正: {clean(chat(MINIMAL.format(s)))}\n   流暢な書き直し: {clean(chat(FLUENT.format(s)))}")
```

```python
# 訂正を編集（どこをどう変えたか）の集合にして、参照訂正と比べる
def edits(source, target):
    """文字単位の差分から（開始位置, 終了位置, 置き換える文字列）の集合を作る"""
    ops = difflib.SequenceMatcher(None, source, target, autojunk=False).get_opcodes()
    return {(i1, i2, target[j1:j2]) for op, i1, i2, j1, j2 in ops if op != "equal"}

def evaluate(system, n=100):
    ds = load_dataset("ronantakizawa/jfleg-japanese", split="validation").select(range(n))
    tp = fp = fn = exact = unchanged = 0
    examples = []
    for ex in ds:
        hyp = system(ex["sentence"])
        hyp_edits = edits(ex["sentence"], hyp)
        # 参照訂正が複数あるので、最も F0.5 が高くなるものと比べる
        best = max((edits(ex["sentence"], ref) for ref in ex["corrections"]),
                   key=lambda ref: (1.25 * len(hyp_edits & ref) / (len(hyp_edits) + 0.25 * len(ref) + 1e-9), -len(ref)))
        tp += len(hyp_edits & best); fp += len(hyp_edits - best); fn += len(best - hyp_edits)
        exact += hyp in ex["corrections"]; unchanged += hyp == ex["sentence"]
        examples.append((ex["sentence"], hyp, ex["corrections"][0]))
    p, r = tp / (tp + fp + 1e-9), tp / (tp + fn + 1e-9)
    return {"適合率": p, "再現率": r, "F0.5": 1.25 * p * r / (0.25 * p + r + 1e-9), "参照と完全一致": exact / n, "変更なし": unchanged / n}, examples

import pandas as pd

# 100 文 × 2 通りの指示で生成する。T4 で 10 分ほど
results, outputs = {}, {}
for label, system in [("何もしない", lambda s: s), ("LLM (最小限の訂正)", lambda s: clean(chat(MINIMAL.format(s)))), ("LLM (流暢な書き直し)", lambda s: clean(chat(FLUENT.format(s))))]:
    results[label], outputs[label] = evaluate(system)
pd.DataFrame(results).T.round(3)
```

```python
for src, hyp, ref in outputs["LLM (最小限の訂正)"][:8]:
    print(f"{'○' if hyp == ref else '　'} {src}\n     → {hyp}\n     参照: {ref}")
```

- 「誤りだけを最小限に直す」のか「自然な文に書き直す」のかで、出力も評価も大きく変わる。指示で明確にする。学習者へのフィードバックには前者、文章の清書には後者が向く。
- 出力の整形が評価を左右する。小さな LLM は「訂正後の文:」のような見出しや「（誤りはありません）」のような注釈を付けたり、助詞を重ねて（「駅のの近く」）新しい誤りを作ったりする。見出しや注釈は機械的に取り除き、新しい誤りはそのまま評価に反映させる。訂正文に続けて説明を書き出す出力（「…見に行きました。」が正しい文です。…」）は `clean` では取り除けないので、必要なら規則を足す。
- 意味を変えてしまう訂正も出る（「レストランが夕食を食べました」→「レストランは夕食を食べませんでした」）。
- LLM は直しすぎる傾向がある（正しい部分まで言い換える）。編集単位の適合率が下がるので、F0.5 のように適合率を重視する指標では不利になる。この実験では、流暢な書き直しは再現率が高い (0.519) が適合率が低く (0.418)、F0.5 は 0.435 で、最小限の訂正の 0.441 をわずかに下回った（T4 と同じ float16 で実行した値。100 文なので、この程度の差は誤差の範囲）。逆に、最小限の訂正は直すべき文を変えずに残すことがある（「公園に友達と遊びました。」）。
- 参照訂正は 1 つとは限らない。複数の参照と比べないと、正しい訂正を誤りと数えてしまう。それでも参照にない正しい訂正は残る。
- ここでは文字単位の差分で編集を作った。きちんと評価するなら、形態素単位で編集を取り出す（この章の ERRANT のレシピの日本語版にあたる処理）。
- このデータは JFLEG にならって作られた公開データで、文が短く誤りも単純。実際の学習者の作文（Lang-8 コーパス、TEC-JL など）はもっと難しい。

## 参照訂正なしで訂正文の質を評価する {gec_reference_free}

参照訂正（正解）がなくても、訂正システムの出力が「文法的か」「流暢か」「元の意味を保っているか」を自動で評価したい。

```python
!pip install -q sentencepiece sentence-transformers datasets
```

```python
import math
import numpy as np
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

lm_name = "rinna/japanese-gpt2-small"
lm_tok = AutoTokenizer.from_pretrained(lm_name, use_fast=False)
# transformers 5 系の T5Tokenizer は未知語 (大文字の英字など) を文末記号 (</s>) の ID にしてしまう。
# 元の spiece.model から語彙とスコアを読み、未知語の ID を正しく扱う Unigram に差し替える
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(lm_name, "spiece.model"))
lm_tok.backend_tokenizer.model = models.Unigram([(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id())
lm = AutoModelForCausalLM.from_pretrained(lm_name).eval()
embedder = SentenceTransformer("intfloat/multilingual-e5-small")

@torch.no_grad()
def fluency(sentence):
    """言語モデルによる 1 トークンあたりの対数尤度 (高いほど自然)。長さで割って、短い文が有利にならないようにする"""
    ids = lm_tok(sentence, return_tensors="pt")["input_ids"]
    return -float(lm(ids, labels=ids).loss)

def meaning(source, hypothesis):
    """元の文と訂正文の文埋め込みの類似度 (意味が保たれているか)"""
    a, b = embedder.encode(["query: " + source, "query: " + hypothesis], normalize_embeddings=True)
    return float(a @ b)

source = "私は毎日学校を行きますが、昨日は病気ので休みました。"
candidates = {
    "元の文 (訂正なし)": source,
    "正しい訂正": "私は毎日学校に行きますが、昨日は病気なので休みました。",
    "一部だけ訂正": "私は毎日学校に行きますが、昨日は病気ので休みました。",
    "流暢だが意味が違う": "私は毎日会社に行きますが、明日は休みます。",
    "意味は同じだが大きく書き換え": "毎日通学していますが、昨日は体調不良のため欠席しました。",
    "壊れた出力": "私は毎日学校行きますに昨日が、病気休みましたなので。",
}
for label, hyp in candidates.items():
    print(f"{label}: 流暢さ {fluency(hyp):.2f} / 意味の保存 {meaning(source, hyp):.3f}")
```

```python
# 流暢さの「改善幅」と意味の保存を組み合わせて 1 つのスコアにする
def score(source, hypothesis, weight=0.5):
    gain = 1 / (1 + math.exp(-(fluency(hypothesis) - fluency(source)) * 2))        # 流暢さが上がれば 0.5 より大きく、下がれば小さく
    return weight * gain + (1 - weight) * meaning(source, hypothesis)

for label, hyp in sorted(candidates.items(), key=lambda x: -score(source, x[1])):
    print(f"{score(source, hyp):.3f}  {label}")
```

```python
# データで確かめる: 誤りを含む文より、参照訂正のほうが高いスコアになる割合
from datasets import load_dataset

ds = load_dataset("ronantakizawa/jfleg-japanese", split="validation").select(range(200))
wins = [fluency(ex["corrections"][0]) > fluency(ex["sentence"]) for ex in ds]
print("参照訂正のほうが流暢と判定された割合:", round(np.mean(wins), 3))
miss = [(ex["sentence"], ex["corrections"][0]) for ex, w in zip(ds, wins) if not w][:5]
for s, c in miss:
    print(f"   逆転した例: {s} ({fluency(s):.2f}) → {c} ({fluency(c):.2f})")
```

- 参照訂正と比べる評価（M2、ERRANT）は、参照にない正しい訂正を誤りと数えてしまう。参照なしの評価は、訂正文そのものの質を直接見るので、この問題がない。
- 見る観点は 3 つ: 文法性（誤りが残っていないか）、流暢性（自然な文か）、意味の保存（元の文の内容を変えていないか）。1 つだけだと簡単にだませる（流暢さだけなら、無関係な自然な文が最高点になる）。
- 言語モデルの尤度は、頻度の高い語や言い回しに甘い。正しくても珍しい表現は低く、誤りを含んでいてもありふれた表現は高くなる。
- 人手評価に合うように各観点のモデルを学習した指標に SOME、訂正前後の文から品質を推定する指標に IMPARA がある。本格的に使うなら、これらの公開モデルを検討する。
- 文埋め込みの類似度は、助詞 1 つの違いのような小さな変更にはほとんど反応しない。「意味が大きく変わっていないか」の確認として使う。

## LLM に訂正の良し悪しを判定させて人手評価と比べる（メタ評価） {gec_llm_evaluator gpu}

複数の訂正システムの出力のうちどれが良いかを LLM に判定させたい。その判定が人の評価とどれくらい合うかを確かめる（評価指標の評価 = メタ評価）。

```python
import itertools
import numpy as np
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from scipy.stats import kendalltau, spearmanr
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる（大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」を参照）。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()
AB = [tokenizer(c, add_special_tokens=False)["input_ids"][-1] for c in ["A", "B"]]

@torch.no_grad()
def prefer_a(source, a, b):
    prompt = ("日本語学習者が書いた「元の文」に対する 2 つの訂正 A と B があります。文法的に正しく、自然で、元の文の意味を変えていないほうを選び、"
              f"A か B の 1 文字だけで答えてください。\n\n元の文: {source}\n訂正 A: {a}\n訂正 B: {b}")
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(llm.device)
    return float(llm(ids).logits[0, -1, AB].float().softmax(-1)[0])

def compare(source, x, y):
    """提示順を入れ替えて 2 回聞き、平均する（x のほうが良い確率）"""
    return (prefer_a(source, x, y) + 1 - prefer_a(source, y, x)) / 2
```

```python
# 元の文ごとに、質の違う 4 つの訂正と、人が付けた順位（1 が最良。同順位あり）を用意する
data = [
    ("私は毎日学校を行きます。", [("私は毎日学校に行きます。", 1), ("私は毎日学校へ通っています。", 2), ("私は毎日学校を行きます。", 3), ("私は毎日会社に行きます。", 4)]),
    ("昨日は友達と映画を見ます。", [("昨日は友達と映画を見ました。", 1), ("昨日、友人と一緒に映画を鑑賞しました。", 2), ("昨日は友達と映画を見ます。", 3), ("昨日は友達に映画が見ました。", 4)]),
    ("この料理はとてもおいしいでした。", [("この料理はとてもおいしかったです。", 1), ("この料理はとてもおいしいです。", 2), ("この料理はとてもおいしいでした。", 3), ("この料理はとてもまずかったです。", 4)]),
    ("図書館で本が借りました。", [("図書館で本を借りました。", 1), ("図書館から本を借りてきました。", 2), ("図書館で本が借りました。", 3), ("図書館に本を貸しました。", 4)]),
    ("日本に来たばかりので、日本語がまだ下手です。", [("日本に来たばかりなので、日本語がまだ下手です。", 1), ("来日して間もないため、日本語はまだ上手ではありません。", 2),
                                                 ("日本に来たばかりので、日本語がまだ下手です。", 3), ("日本に来たばかりなので、日本語がもう上手です。", 4)]),
    ("彼は私にプレゼントをもらいました。", [("彼は私にプレゼントをくれました。", 1), ("私は彼にプレゼントをもらいました。", 1), ("彼は私にプレゼントをもらいました。", 3), ("彼は私のプレゼントを捨てました。", 4)]),
]

sentence_tau, all_llm, all_human = [], [], []
for source, candidates in data:
    texts, human = [c for c, _ in candidates], [r for _, r in candidates]
    wins = np.zeros(len(texts))
    for i, j in itertools.combinations(range(len(texts)), 2):
        p = compare(source, texts[i], texts[j])
        wins[i] += p; wins[j] += 1 - p               # 総当たりの「勝ち数」（確率の和）をスコアにする
    tau = kendalltau(-wins, human).statistic
    sentence_tau.append(tau); all_llm += list(-wins); all_human += human
    print(f"{source}  τ = {tau:.2f}")
    for k in np.argsort(-wins):
        print(f"    LLM {wins[k]:.2f} / 人の順位 {human[k]} | {texts[k]}")
print("文ごとの Kendall τ の平均:", round(float(np.mean(sentence_tau)), 3), "/ 全体の Spearman:", round(spearmanr(all_llm, all_human).statistic, 3))
```

```python
# システム単位のメタ評価: 訂正の種類を「システム」と見なし、全文での平均スコアの順位が人の順位と合うか
systems = ["最小限の正しい訂正", "流暢な書き直し", "訂正なし", "意味を変えた訂正"]
llm_mean = -np.array(all_llm).reshape(len(data), 4).mean(axis=0)
human_mean = np.array(all_human).reshape(len(data), 4).mean(axis=0)
for k in np.argsort(-llm_mean):
    print(f"{systems[k]}: LLM の平均勝ち数 {llm_mean[k]:.2f} / 人の平均順位 {human_mean[k]:.2f}")
print("システム単位の Spearman:", round(spearmanr(-llm_mean, human_mean).statistic, 3))
```

- メタ評価は「評価指標がどれくらい人の評価と合うか」を測る。文単位（同じ元の文に対する訂正の順位）とシステム単位（全体の平均での順位）の 2 つの粒度があり、結果が食い違うことがある。この実験では、文ごとの Kendall τ の平均が 0.42、全体の Spearman が 0.65、システム単位の Spearman が 0.8 だった（T4 と同じ float16 で実行した値）。3B のモデルは「訂正なし」を「流暢な書き直し」より高く評価している。
- 総当たりの一対比較は、1〜5 の採点をさせるより小さなモデルでも安定する。ただし比較の回数は候補数の 2 乗で増える。
- 大規模な LLM による評価は、文法誤り訂正でも従来の自動評価指標より人手評価に近いと報告されている。流暢な書き直しを正当に評価できる点が、編集の一致を数える指標との違い。
- 人手評価のデータそのものにも設計上の問題がありうる（評価した時代のシステムが古い、文単位と編集単位の評価が混ざる）。メタ評価用のデータとして SEEDA が公開されている。
- ここの 6 文は手作りの動作確認用。指標を採用するかどうかは、実際のシステム出力と人手評価の組（数百文以上）で判断する。

## マスク言語モデルで誤りのありそうな箇所を見つける（文法誤り検出） {ged_mlm}

訂正まではしなくてよいので、文のどこに誤りがありそうかを指摘したい。誤りの注釈付きデータを使わずに、学習済みの言語モデルだけで行う。

```python
!pip install -q fugashi unidic-lite datasets
```

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

name = "tohoku-nlp/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForMaskedLM.from_pretrained(name).eval()

@torch.no_grad()
def token_scores(sentence):
    """各トークンを 1 つずつ [MASK] に置き換え、元のトークンが予測される確率と、最も有力な候補を調べる"""
    ids = tokenizer(sentence, return_tensors="pt")["input_ids"][0]
    positions = range(1, len(ids) - 1)
    batch = ids.repeat(len(positions), 1)
    for row, pos in enumerate(positions):
        batch[row, pos] = tokenizer.mask_token_id
    probs = model(input_ids=batch).logits.softmax(-1)
    out = []
    for row, pos in enumerate(positions):
        p = probs[row, pos]
        out.append((tokenizer.convert_ids_to_tokens(int(ids[pos])), float(p[ids[pos]]), tokenizer.convert_ids_to_tokens(int(p.argmax()))))
    return out

def detect(sentence, threshold=0.01):
    marked = []
    for token, p, best in token_scores(sentence):
        token = token.replace("##", "")
        marked.append(f"【{token}→{best.replace('##', '')}?】" if p < threshold else token)
    return "".join(marked)

for s in ["図書館で本が借りました。", "私は毎日学校を行きます。", "昨日は友達と映画を見ます。", "この料理はとてもおいしいでした。",
          "彼は日本語が上手に話せます。", "私は東京に住んでいます。", "明日は雨が降るでしょう。"]:
    print(detect(s))
```

```python
# トークンごとの確率を見る
for token, p, best in token_scores("図書館で本が借りました。"):
    print(f"{token}\t{p:.4f}\t最有力: {best}")
```

```python
# 誤りを含む文と訂正済みの文で、検出される箇所の数を比べる (検出の当たり外れの目安)
from datasets import load_dataset

ds = load_dataset("ronantakizawa/jfleg-japanese", split="validation").select(range(150))

def n_flagged(sentence, threshold):
    return sum(p < threshold for _, p, _ in token_scores(sentence))

for threshold in [0.001, 0.01, 0.05]:
    wrong = [n_flagged(ex["sentence"], threshold) > 0 for ex in ds]
    right = [n_flagged(ex["corrections"][0], threshold) > 0 for ex in ds]
    tp, fp = sum(wrong), sum(right)
    print(f"しきい値 {threshold}: 誤りのある文で指摘あり {tp / len(ds):.2f} / 正しい文で指摘あり (誤検出) {fp / len(ds):.2f} / 適合率 {tp / (tp + fp):.2f}")
```

- マスク言語モデルは「この位置にこの語が来る確率」を出せる。確率が極端に低い語は、文脈に合っていない、つまり誤りの可能性がある。誤りの注釈データなしで使える。
- 助詞の誤り（「本が借りました」）のように、周りの語から正解がほぼ決まる誤りは見つけやすい。内容語の選び間違いや、語の抜け・余分は見つけにくい。
- 珍しい固有名詞や専門用語も確率が低くなるので、誤りと区別できない。しきい値を下げると見逃しが増え、上げると誤検出が増える。
- 誤りの位置を正確に指すとは限らない（誤った助詞ではなく、その後ろの動詞の確率が下がることがある）。「このあたりが怪しい」という手がかりとして使う。
- 高い精度が必要なら、誤りの注釈付きデータで系列ラベリング（各トークンが正しいか誤りか）を学習する。固有表現抽出の章の BERT のファインチューニングと同じ方法が使える。

## 訂正にフィードバックコメントを付ける {feedback_comment gpu}

学習者に、誤りを直した結果だけでなく、「なぜ誤りなのか」「どう直せばよいか」の説明を返したい。

```python
import difflib
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる（大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」の注意書きを参照）。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()

@torch.no_grad()
def chat(prompt, max_new_tokens=160):
    ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to(llm.device)
    out = llm.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def diff(source, corrected):
    """訂正箇所を「元 → 訂正後」の組として取り出す"""
    ops = difflib.SequenceMatcher(None, source, corrected, autojunk=False).get_opcodes()
    return [(source[i1:i2] or "(なし)", corrected[j1:j2] or "(削除)") for op, i1, i2, j1, j2 in ops if op != "equal"]

pairs = [("私は毎日学校を行きます。", "私は毎日学校に行きます。"), ("この料理はとてもおいしいでした。", "この料理はとてもおいしかったです。"),
         ("日本に来たばかりので、日本語がまだ下手です。", "日本に来たばかりなので、日本語がまだ下手です。"), ("彼は私にプレゼントをもらいました。", "彼は私にプレゼントをくれました。")]
for s, c in pairs:
    print(s, "→", c, diff(s, c))
```

```python
PROMPT = """あなたは日本語教師です。日本語学習者の文と、その訂正を示します。
訂正箇所について、学習者向けのフィードバックを書いてください。
- なぜ元の表現が誤りなのかを、文法の規則として 1〜2 文で説明する
- 専門用語はできるだけ使わず、やさしい日本語で書く
- 同じ規則を使った正しい例文を 1 つ挙げる
- あいさつや前置きは書かず、全体で 150 字以内にする

学習者の文: {source}
訂正後の文: {corrected}
訂正箇所: {edits}"""

for s, c in pairs:
    edits = "、".join(f"「{a}」→「{b}」" for a, b in diff(s, c))
    print(f"■ {s} → {c}\n{chat(PROMPT.format(source=s, corrected=c, edits=edits))}\n")
```

```python
# 訂正と説明を別々に行う理由: LLM に一度に任せると、説明が訂正と食い違うことがある。説明に訂正後の表現が含まれているかを機械的に確かめる
def check(source, corrected, comment):
    return {"訂正後の表現に触れている": all(b in comment for a, b in diff(source, corrected) if b != "(削除)"),
            "長すぎない": len(comment) <= 300}

s, c = pairs[0]
comment = chat(PROMPT.format(source=s, corrected=c, edits="、".join(f"「{a}」→「{b}」" for a, b in diff(s, c))))
print(check(s, c, comment))
```

- フィードバックコメント生成は、訂正（何を直すか）とは別のタスクとして研究されている。学習効果のためには、答えを示すだけでなく、規則に気付かせる説明が重要とされる。
- 訂正箇所を差分で明示してから説明させると、どの誤りについての説明かがぶれにくい。誤りが複数あるときは、1 か所ずつ説明させる。
- 長さの指示は守られにくい。この実験でも、3 つ目の説明は「150 字以内」の指示を大きく超えて 400 字近くになり、「recentness」のような英語まで混ざった。1 つ目も 150 字をわずかに超えている（下の点検は上限を 300 字にしているので通る）。生成後に長さを点検し、超えたら要約させ直す、といった後処理を用意する。
- 説明が訂正と食い違うこともある。2 つ目の説明は、訂正後の文（「おいしかったです」）に反して「「ました」を使うのが正解」と書いている。
- 小さな LLM は、もっともらしいが誤った文法説明を書くことがある（存在しない規則を作る、助詞の働きを取り違える）。学習者にそのまま見せる前に、教師による確認か、誤りの種類ごとに用意した説明文（テンプレート）との併用を検討する。
- 誤りの種類が分かっていれば（助詞の選択、時制、活用など）、種類ごとに人が書いた説明を引いてくる方法が最も安全。誤りタイプの判定には、この章の ERRANT のレシピの考え方が使える。
- 学習者のレベルに合わせて説明の言葉を変える（初級者には母語で、など）ことも、LLM への指示で行える。

## 日本語の文章の校正すべき箇所を指摘する（表記ゆれ・冗長な表現・ら抜き言葉） {proofreading_ja gpu}

報告書やメールの下書きから、表記ゆれ、冗長な言い回し、ら抜き言葉、二重敬語、長すぎる文、同じ助詞の連続を見つけて、書き直しの案と一緒に指摘したい。形態素解析による規則、textlint、手元の LLM の 3 つを比べる。

```python
!pip install -q fugashi unidic-lite
# textlint (MIT) は Node.js の校正ツール。Colab には Node.js が入っているので、作業ディレクトリに入れて Python から呼ぶ
!npm install --silent --no-audit --no-fund textlint@15.8.0 textlint-rule-preset-ja-technical-writing@12.0.2 @textlint-ja/textlint-rule-no-synonyms@1.3.0 sudachi-synonyms-dictionary@20.0.0
```

```python
import re
from collections import defaultdict
from functools import lru_cache
from fugashi import Tagger

tagger = Tagger()

def split_sentences(text):
    return [s for s in re.split(r"(?<=[。！？])", text) if s.strip()]

REDUNDANT = [(r"することができ", "「できる」で足りる"), (r"することが可能", "「できる」で足りる"),
             (r"まず最初に|一番最初", "「まず」か「最初に」のどちらかにする"), (r"であると言え", "「である」で足りる"),
             (r"(?<!を)を行[いうえっわ]", "「〜する」にする")]
HONORIFIC = {"仰る", "為さる", "いらっしゃる", "召し上がる", "下さる"}     # UniDic の語彙素 (lemma) で書く

@lru_cache(maxsize=None)     # 同じ語を何度も解析しないように結果を覚えておく
def is_ichidan(lemma):
    """一段動詞かカ変動詞か（ら抜きが起きるのはこれだけ。「帰れる」のような五段動詞の可能形は正しい）"""
    ws = tagger(lemma)
    return len(ws) == 1 and ws[0].feature.cType.startswith(("上一段", "下一段", "カ行変格"))

def rule_check(text):
    """（種類, 箇所, 書き直しの案）のリストを返す"""
    issues = []
    for s in split_sentences(text):
        ws = list(tagger(s))
        for pat, advice in REDUNDANT:
            issues += [("冗長", m.group(), advice) for m in re.finditer(pat, s)]
        for i, w in enumerate(ws):
            f = w.feature
            # ら抜き (1): UniDic は「起きれる」「見れる」を 1 語（書字形 起きれる、語彙素 起きる）として登録している
            if f.pos1 == "動詞" and f.lemma and f.orthBase == f.lemma[:-1] + "れる" and f.lemma.endswith("る") and is_ichidan(f.lemma):
                issues.append(("ら抜き", w.surface, f"「{f.lemma[:-1]}られる」の形にする"))
            # ら抜き (2): 「食べ」（一段の未然形）+ 助動詞「れる」と解析される場合
            if f.pos1 == "動詞" and f.cForm.startswith("未然") and f.cType.startswith(("上一段", "下一段", "カ行変格")) \
                    and i + 1 < len(ws) and ws[i + 1].feature.lemma == "れる":
                issues.append(("ら抜き", w.surface + ws[i + 1].surface, f"「{f.lemma[:-1]}られる」の形にする"))
            # 二重敬語: 尊敬語（おっしゃる、召し上がる、お〜になる、ご覧になる）に尊敬の「れる・られる」を重ねる
            if f.pos1 == "助動詞" and f.lemma in ("れる", "られる") and i > 0:
                prev = ws[i - 1].feature
                start = None                       # 尊敬語の始まりの位置（指摘する箇所の先頭）
                if prev.lemma in HONORIFIC:
                    start = i - 1
                elif prev.lemma == "成る" and i >= 3 and ws[i - 2].surface == "に":
                    # 「ご覧になる」か「お/ご + 語 + になる」
                    starts = [j for j in range(max(0, i - 6), i - 2)
                              if ws[j].feature.lemma == "御覧" or (ws[j].feature.pos1 == "接頭辞" and ws[j].surface in ("お", "ご"))]
                    start = starts[-1] if starts else None
                if start is not None:
                    issues.append(("二重敬語", "".join(x.surface for x in ws[start:i + 1]), "「れる・られる」を取る"))
        # 長すぎる文: 80 字を超えるか、読点が 4 つ以上
        if len(s) > 80 or s.count("、") >= 4:
            issues.append(("長文", s[:12] + "…", f"{len(s)} 字・読点 {s.count('、')} 個。2 文に分ける"))
        # 助詞の連続: 読点で区切った範囲に同じ助詞（が・を・に・で・は）が 2 回、または格助詞の「の」が 3 回以上
        for clause in s.split("、"):
            ps = [w for w in tagger(clause) if w.feature.pos1 == "助詞"]
            for p in ("が", "を", "に", "で", "は"):
                if sum(w.surface == p for w in ps) >= 2:
                    issues.append(("助詞連続", clause[:12] + "…", f"「{p}」が 2 回。語順や言い方を変える"))
            if sum(w.surface == "の" and w.feature.pos2 == "格助詞" for w in ps) >= 3:
                issues.append(("助詞連続", clause[:12] + "…", "「の」が 3 回以上。言い換える"))
    # 表記ゆれ: 文書の中で、同じ語彙素 (lemma) が違う書字形 (orthBase) で書かれている
    forms = defaultdict(set)
    for w in tagger(text):
        if w.feature.pos1 in ("名詞", "動詞", "形容詞", "副詞") and w.feature.lemma:
            forms[(w.feature.lemma, w.feature.pos1)].add(w.feature.orthBase or w.surface)
    issues += [("表記ゆれ", "/".join(sorted(fs)), "どれかにそろえる") for fs in forms.values() if len(fs) > 1]
    return issues

# 動作確認用に手作りした文書（1〜2 文）と、含まれる問題の種類（正解）。空集合は「問題なし」
DOCS = [
    ("このツールを使えば、作業を自動化することができます。", {"冗長"}),
    ("会議では、まず最初に前回の議事録を確認します。", {"冗長"}),
    ("この問題は、設定を変えるだけで解決することが可能です。", {"冗長"}),
    ("各部署ごとに担当者を一人決めてください。", {"冗長"}),
    ("頭痛が痛いので、今日は早めに帰ります。", {"冗長"}),
    ("この件について、来週検討を行いたいと思います。", {"冗長"}),
    ("後で後悔しないように、よく確認しよう。", {"冗長"}),
    ("朝は6時に起きれるようになった。", {"ら抜き"}),
    ("明日のパーティーには来れますか。", {"ら抜き"}),
    ("この店では新鮮な刺身が食べれる。", {"ら抜き"}),
    ("その映画はもう見れない。", {"ら抜き"}),
    ("虫が入るので窓を開けれない。", {"ら抜き"}),
    ("社長がそのようにおっしゃられました。", {"二重敬語"}),
    ("先生はもう資料をご覧になられましたか。", {"二重敬語"}),
    ("お客様が料理を召し上がられています。", {"二重敬語"}),
    ("ご注文の品はお揃いになられましたでしょうか。", {"二重敬語"}),
    ("部長が資料をお読みになられた。", {"二重敬語"}),
    ("子供の数が減っている。子どもの遊び場も少なくなった。", {"表記ゆれ"}),
    ("結果が分かったら連絡します。わからない点は聞いてください。", {"表記ゆれ"}),
    ("設定ファイルをサーバに置く。次にサーバーを再起動する。", {"表記ゆれ"}),
    ("申込書を出してください。締め切りを過ぎた申し込みは受け付けません。", {"表記ゆれ"}),
    ("ウェブサイトを更新した。Webサイトのアドレスは変わらない。", {"表記ゆれ"}),
    ("打ち合わせは明日です。打合せの資料を準備します。", {"表記ゆれ"}),
    ("私の兄の友人の会社の社長に会った。", {"助詞連続"}),
    ("この案は予算は少ないが効果は大きい。", {"助詞連続"}),
    ("データの分析の結果の報告は来週です。", {"助詞連続"}),
    ("本研究では近年急速に普及している大規模言語モデルを用いて日本語の学習者が書いた作文に含まれる文法的な誤りを自動的に検出し"
     "その誤りの種類に応じた説明を生成する手法を提案し実際の作文による評価実験で有効性を検証した。", {"長文"}),
    ("今回の改修では、画面の配色、ボタンの配置、入力欄の大きさ、エラーメッセージの文言、ヘルプの内容を見直した。", {"長文"}),
    ("次回の打ち合わせでは昨年度の売上実績と今年度の目標を比較したうえで来期に向けた営業戦略の方向性について"
     "各担当者から意見を出してもらい最終的な方針を決める予定です。", {"長文"}),
    ("先生がおっしゃられたとおり、この方法で問題を解決することができます。", {"二重敬語", "冗長"}),
    ("明日は6時に起きれるので、子どもを駅まで送る。子供はきっと喜ぶだろう。", {"ら抜き", "表記ゆれ"}),
    # 以下は問題のない文（規則が誤検出しやすい形を含む）
    ("この服はまだ着られる。", set()), ("駅から歩いて帰れる距離だ。", set()), ("この道は狭くて車が通れない。", set()),
    ("この部屋からは富士山が見られる。", set()), ("部長は明日の会議にいらっしゃいます。", set()), ("先生がおっしゃったとおりです。", set()),
    ("東京に住んでいる友人に手紙を送った。", set()), ("彼は私が書いた本が好きだ。", set()), ("忙しいときは昼食を抜くことがある。", set()),
    ("新しい制度は来月から始まるが、詳しい手続きは担当者から改めて連絡する予定である。", set()),
    ("この方法なら、誰でも簡単に設定できます。", set()), ("会議は午後3時から始まります。資料は事前に配布します。", set()),
]
def show(text, gold, pred, issues, width=(20, 30)):
    """1 文書を 1 行で表示する。正解と合わなければ × と正解の種類も出す"""
    head = text if len(text) <= 14 else text[:14] + "…"
    mark = "○" if pred == gold else f"×［正解 {'・'.join(sorted(gold)) or 'なし'}］"
    found = " / ".join(f"{c}「{span[:width[0]]}」→ {advice[:width[1]]}" for c, span, advice in issues)
    print(f"{mark} {head}  {found or '指摘なし'}")

print(len(DOCS), "文書")
rule_pred = []
for text, gold in DOCS:
    issues = rule_check(text)
    rule_pred.append({c for c, _, _ in issues})
    show(text, gold, rule_pred[-1], issues)
```

```python
# textlint を Python から呼ぶ。文書ごとにファイルに書き、まとめて 1 回で検査して JSON で受け取る
import json, os, subprocess
from pathlib import Path

config = {"rules": {"preset-ja-technical-writing": {"sentence-length": {"max": 80}},   # 文の長さの上限を規則とそろえる（既定は 100 字）
                    "@textlint-ja/no-synonyms": True}}
Path(".textlintrc.json").write_text(json.dumps(config))
Path("docs").mkdir(exist_ok=True)
paths = []
for i, (text, _) in enumerate(DOCS):
    paths.append(f"docs/{i:02d}.txt")
    Path(paths[-1]).write_text(text + "\n")
r = subprocess.run(["npx", "textlint", "--format", "json", *paths], capture_output=True, text=True)
out = r.stdout
if not out:      # インストールの失敗などで textlint が動かなかったときは、エラーを表示する
    print(r.stderr)

# 規則の名前を、この表の種類に対応づける（二重敬語の規則はプリセットにない。ほかの規則の指摘は「その他」）
TL_CAT = {"ja-no-redundant-expression": "冗長", "no-dropping-the-ra": "ら抜き", "no-doubled-joshi": "助詞連続",
          "sentence-length": "長文", "max-ten": "長文", "no-synonyms": "表記ゆれ"}
tl_msgs = {p: [] for p in paths}
for f in json.loads(out):
    tl_msgs[os.path.relpath(f["filePath"])] = [(m["ruleId"].split("/")[-1], m["message"].split("\n")[0]) for m in f["messages"]]
tl_pred = [{TL_CAT[r] for r, _ in tl_msgs[p] if r in TL_CAT} for p in paths]
for (text, gold), p in zip(DOCS, paths):
    if tl_msgs[p]:
        print(text[:24], f"正解 {sorted(gold)}")
        for r, m in tl_msgs[p]:
            print(f"      [{TL_CAT.get(r, 'その他')}] {r}: {m[:60]}")
```

```python
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"     # MIT ライセンス
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる（注意書きを参照）。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()

@torch.no_grad()
def chat(prompt, max_new_tokens=200):
    ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to(llm.device)
    out = llm.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

CATS = ["表記ゆれ", "冗長", "ら抜き", "二重敬語", "長文", "助詞連続"]
PROMPT = """あなたは日本語の文章の校正者です。次の文章に、下の 6 種類の問題があれば指摘してください。
- 表記ゆれ: 同じ語が違う書き方で書かれている (例: 子供と子ども)
- 冗長: 削っても意味が変わらない言い回し (例: することができる → できる)
- ら抜き: 「見られる」を「見れる」とするような、可能の「られる」の「ら」が抜けた形
- 二重敬語: 敬語を重ねた形 (例: おっしゃられる → おっしゃる)
- 長文: 長すぎて読みにくい文
- 助詞連続: 同じ助詞が 1 つの文の中で続いている (例: 私の兄の友人の会社)
問題 1 つにつき 1 行で「種類|問題の箇所|書き直しの案」の形で出力してください。問題がなければ「なし」とだけ出力してください。説明は書かないでください。

文章: {}"""

def llm_check(text):
    """「種類|箇所|案」の行だけを取り出す。箇所が「なし」の行や、後に続く説明の文は捨てる"""
    issues = []
    for line in chat(PROMPT.format(text)).splitlines():
        f = [x.strip() for x in line.split("|")]
        if len(f) >= 2 and f[0] in CATS and f[1] not in ("", "なし"):
            issues.append((f[0], f[1], f[2] if len(f) > 2 else ""))
    return issues

# 43 文書の生成は T4 で 3 分ほど（初回はモデルのダウンロードに数分かかる）
llm_pred = []
for text, gold in DOCS:
    issues = llm_check(text)
    llm_pred.append({c for c, _, _ in issues})
    show(text, gold, llm_pred[-1], issues, width=(20, 25))
```

```python
# 文書 × 種類の単位で、指摘の適合率・再現率を比べる
import pandas as pd

def prf(preds, cat):
    tp = sum(cat in p and cat in g for p, (_, g) in zip(preds, DOCS))
    fp = sum(cat in p and cat not in g for p, (_, g) in zip(preds, DOCS))
    fn = sum(cat not in p and cat in g for p, (_, g) in zip(preds, DOCS))
    return f"{tp}/{tp + fp}" if tp + fp else "-", f"{tp}/{tp + fn}"

COLS = ("適合 (正しい指摘/指摘)", "再現 (見つけた/正解)")
table = {sys_name: {(cat, col): v for cat in CATS for col, v in zip(COLS, prf(preds, cat))}
         for sys_name, preds in [("規則", rule_pred), ("textlint", tl_pred), ("LLM", llm_pred)]}
pd.DataFrame(table)
```

```python
# 誤検出（正解にない指摘）と見逃し（正解にあるのに指摘なし）を種類ごとにまとめて並べる
for sys_name, preds in [("規則", rule_pred), ("textlint", tl_pred), ("LLM", llm_pred)]:
    print(f"■ {sys_name}")
    for kind in ("誤検出", "見逃し"):
        for cat in CATS:
            hits = [text[:10] for p, (text, gold) in zip(preds, DOCS)
                    if cat in (p - gold if kind == "誤検出" else gold - p)]
            if hits:
                print(f"   {kind} {cat} {len(hits)} 件: {' / '.join(hits)}")
```

- 文法誤り訂正（この章の「LLM で日本語の文法誤りを訂正して評価する」など）は、誤りを含む文を正しい文 1 つに書き換える。校正支援は、文法的には正しくても読みにくい箇所に印を付けて理由と書き直しの案を示し、直すかどうかは書き手に任せる。そのため評価も、参照文との一致ではなく、指摘の正しさ（適合率）と見逃しの少なさ（再現率）で見る。
- 規則は、ら抜き・二重敬語・長文のように形で決まる問題に強い。UniDic は「起きれる」を語彙素「起きる」の別の書字形として登録しているので、語彙素と書字形を比べればら抜きも表記ゆれも見つかる。一方、パターンにない言い回し（「各部署ごと」「頭痛が痛い」「後で後悔」）は見逃し、助詞連続は 7 文書で指摘して正しいのは 3 文書だった（「東京に住んでいる友人に」のような自然な重なりや、読点のない長い文を誤検出）。規則と例文を同時に作ったので、規則の成績は実際より良く見える。
- 語彙素の違う「ウェブサイト/Webサイト」は、規則でも textlint でも見つからなかった（LLM は見つけた）。日本語の形態素解析と文分割の章の「Sudachi の正規化形で表記ゆれを吸収する」の正規化形を使えば拾える可能性があるが、ここでは確かめていない。
- textlint は、規則をプラグインとして組み合わせるテキストの校正ツール (Node.js) で、日本語の規則が多く公開されている。ここでは技術文書向けのプリセットと、Sudachi 同義語辞書を使う表記ゆれの規則を `subprocess` で呼んだ。冗長・ら抜き・長文・表記ゆれの指摘はすべて正しかったが、表記ゆれは 2/7 しか見つからず、二重敬語の規則はプリセットにない (0/6)。助詞連続は 3 文書で指摘して 2 文書が誤検出で、README のとおり連体化の「の」の重なりは許すので「私の兄の友人の会社の」は見逃した。
- LLM（Sarashina2.2 3B、T4 と同じ float16）は、ら抜き 5/6、二重敬語 4/6（指摘した 4 文書はすべて正しい）で、書き直しの案も出せた。表記ゆれは 7 文書すべてで指摘したが（規則 6/7、textlint 2/7）、正解にない 12 文書でも指摘した（「「案」と「案」」など）。冗長は 2/8 で、うち 1 文書は「だけで」を挙げた的外れな指摘。助詞連続は 0/3 で、正解にない 7 文書で指摘した。長文は 2/3 を見つけたが、書き直しの案は英語になるなど崩れていた。
- LLM の指摘は、種類が合っていても中身が誤っていることがある。表記ゆれとして「「結果」と「分かる」」を挙げたり、「受け付けません」をら抜きとして「受け付けられません」に直したり、敬語を落とす案（「お読みになられた」→「読んだ」）を出したりした。評価は文書単位の「その種類を指摘したか」で箇所までは照合していないので、これらも正しい指摘に数えている。種類ごとの正解は 3〜8 文書しかなく、1〜2 件の差は誤差の範囲。
- 実務では、規則（textlint や自作の規則）で機械的に検出できるものを先に潰し、LLM には意味の重複や言い換えの提案のような規則で書きにくい部分を任せる、という分担が現実的。LLM の指摘はそのまま採用せず、書き手が判断する候補として出す。
- ツールとモデルのライセンス: textlint と使った規則 (MIT)、Sudachi 同義語辞書（sudachi-synonyms-dictionary、Apache-2.0）、sbintuitions/sarashina2.2-3b-instruct-v0.1（SB Intuitions、MIT）。sarashina のトークナイザーを直すコードについては、大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」の注意書きを参照。

## 英語のスペルミスをライブラリで直す (pyspellchecker) {spellcheck}

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

- 編集距離と単語頻度に基づく Norvig 方式。文脈は見ないので `their/there` のような実在語の誤りは直せない。文脈を見る方法は、この章の「symspellpy とマスク言語モデルで英語のスペル誤りを文脈に合わせて直す」を参照。

## symspellpy とマスク言語モデルで英語のスペル誤りを文脈に合わせて直す {spelling_correction}

英文のつづりの誤りを、辞書にない語 (recieved) だけでなく、実在する別の語になってしまった誤り (a peace of cake) も含めて直したい。速い辞書引きで候補を出し、文脈に合う語をマスク言語モデルで選ぶ（1 語ずつ手軽に直すだけなら、この章の「英語のスペルミスをライブラリで直す (pyspellchecker)」）。

```python
!pip install -q symspellpy==6.10.2 pyspellchecker==0.9.1
# 評価用: Peter Norvig が Birkbeck spelling error corpus (Roger Mitton) から取り出した (誤り, 正解) の組。ライセンスの記載はない
!wget -nc -q https://norvig.com/spell-testset1.txt
```

```python
import importlib.resources
from symspellpy import SymSpell, Verbosity

# 同梱の英語の辞書 (約 8 万語と頻度)。SymSpell の README によると、Google Books Ngram (CC BY 3.0) と SCOWL の両方にある語から作られている
sym = SymSpell(max_dictionary_edit_distance=2)
res = importlib.resources.files("symspellpy")
sym.load_dictionary(str(res / "frequency_dictionary_en_82_765.txt"), term_index=0, count_index=1)
sym.load_bigram_dictionary(str(res / "frequency_bigramdictionary_en_243_342.txt"), term_index=0, count_index=2)

# 1 語ずつ: 編集距離が最も小さい候補を、頻度の高い順に返す
for w in ["speling", "recieve", "definately", "inconvient", "teh", "peace"]:
    print(w, "→", [(h.term, h.distance, h.count) for h in sym.lookup(w, Verbosity.CLOSEST, max_edit_distance=2)[:4]])

# 文全体: 語の分割・結合の誤り (whereis, th elove) もまとめて直す
for s in ["whereis th elove hehad dated forImuch of thepast", "I recieved a wonderfull mesage from my freind"]:
    print(s, "\n →", sym.lookup_compound(s, max_edit_distance=2)[0].term)

# 空白のない文字列を単語に分ける
print(sym.word_segmentation("thequickbrownfoxjumpsoverthelazydog").corrected_string)
```

```python
# 1 語ずつの訂正の正解率と速さを pyspellchecker と比べる
import time
from spellchecker import SpellChecker

tests = [(wrong, right) for line in open("spell-testset1.txt") for right, wrongs in [line.strip().split(":")] for wrong in wrongs.split()]
spell = SpellChecker()

def by_symspell(w):
    hits = sym.lookup(w, Verbosity.TOP, max_edit_distance=2)
    return hits[0].term if hits else w

def by_pyspellchecker(w):
    return spell.correction(w) or w

print(len(tests), "組 / 正解の語が symspellpy の辞書にない割合:", round(sum(r not in sym.words for _, r in tests) / len(tests), 3))
for name, f in [("symspellpy", by_symspell), ("pyspellchecker", by_pyspellchecker)]:
    t = time.perf_counter()
    out = [f(w) for w, _ in tests]
    sec = time.perf_counter() - t
    acc = sum(o == r for o, (_, r) in zip(out, tests)) / len(tests)
    print(f"{name}: 正解率 {acc:.3f} / {len(tests) / sec:.0f} 語/秒")
    print("   誤りの例 (入力, 出力, 正解):", [(w, o, r) for o, (w, r) in zip(out, tests) if o != r][:8])
```

```python
# 文脈を使う: symspellpy の候補 (元の語も含む) を、マスク言語モデルで文脈に合う順に並べ替える
import re
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

torch.manual_seed(0)
name = "distilbert-base-uncased"     # Apache-2.0。小文字にしてから扱う
tokenizer = AutoTokenizer.from_pretrained(name)
mlm = AutoModelForMaskedLM.from_pretrained(name).eval()

def candidates(word, k=15):
    """編集距離 2 までの辞書の語のうち、頻度の高い k 語 (短い語は距離 1 まで)。辞書にある元の語は必ず残す"""
    hits = [h for h in sym.lookup(word, Verbosity.ALL, max_edit_distance=2) if h.distance <= 1 or len(word) >= 4]
    cands = {h.term: h.distance for h in sorted(hits, key=lambda h: -h.count)[:k]}
    if word in sym.words:
        cands[word] = 0
    return cands

@torch.no_grad()
def context_scores(words, i, cands):
    """i 番目の語を各候補に置き換えたときの、その語の対数確率 (複数トークンの語は、自分と後ろのトークンを隠して 1 トークンずつ予測して足す)"""
    left = tokenizer(" ".join(words[:i]), add_special_tokens=False)["input_ids"] if i else []
    right = tokenizer(" ".join(words[i + 1:]), add_special_tokens=False)["input_ids"]
    seqs, meta = [], []
    for c in cands:
        ids = tokenizer(c, add_special_tokens=False)["input_ids"]
        for j in range(len(ids)):
            span = ids[:j] + [tokenizer.mask_token_id] * (len(ids) - j)
            seqs.append([tokenizer.cls_token_id] + left + span + right + [tokenizer.sep_token_id])
            meta.append((c, 1 + len(left) + j, ids[j]))
    batch = tokenizer.pad({"input_ids": seqs}, return_tensors="pt")
    logp = mlm(**batch).logits.log_softmax(-1)
    scores = dict.fromkeys(cands, 0.0)
    for row, (c, pos, tok) in enumerate(meta):
        scores[c] += float(logp[row, pos, tok])
    return scores

def correct_sentence(sentence, penalty=2.0, margin=2.0):
    """左から 1 語ずつ直す。元の語から離れるほど減点し、辞書にある語は差が margin を超えたときだけ書き換える"""
    words = sentence.lower().split()
    for i, w in enumerate(words):
        if not re.fullmatch(r"[a-z]+", w):
            continue
        cands = candidates(w)
        if not cands or list(cands) == [w]:
            continue
        ctx = context_scores(words, i, cands)
        total = {c: ctx[c] - penalty * d for c, d in cands.items()}
        best = max(total, key=total.get)
        if best != w and (w not in cands or total[best] > total[w] + margin):
            words[i] = best
    return " ".join(words)

for s in ["I want a peace of cake .", "The whether is nice today .", "Their is a cat on the table .", "I like to read books ."]:
    print(s, "→", correct_sentence(s))
```

```python
# 手作りの文で比べる: 実在語の誤り (19 文)、辞書にない語の誤り (4 文)、誤りのない文 (12 文)
# Colab の CPU で 1〜3 分
data = """I want a peace of cake . || I want a piece of cake .
He will except the offer . || He will accept the offer .
She lives in a quite village . || She lives in a quiet village .
The whether is nice today . || The weather is nice today .
Their is a cat on the table . || There is a cat on the table .
I need to buy some bred . || I need to buy some bread .
Can you here me now ? || Can you hear me now ?
I lost my keys and could not find then . || I lost my keys and could not find them .
She is taller then her brother . || She is taller than her brother .
We swam in the see . || We swam in the sea .
Please turn of the light . || Please turn off the light .
He new the answer . || He knew the answer .
I have to brake the news to him . || I have to break the news to him .
It is to cold to swim . || It is too cold to swim .
I will meat you at the station . || I will meet you at the station .
She road her bike to school . || She rode her bike to school .
My farther is a doctor . || My father is a doctor .
We had a grate time at the party . || We had a great time at the party .
He has a dog and to cats . || He has a dog and two cats .
I recieved a wonderfull mesage from my freind . || I received a wonderful message from my friend .
The goverment anounced a new polcy . || The government announced a new policy .
She is a very inteligent studnet . || She is a very intelligent student .
I beleive this is the rigth answer . || I believe this is the right answer .
I like to read books . || I like to read books .
The cat sat on the mat . || The cat sat on the mat .
She bought a new car last week . || She bought a new car last week .
They went to the park after lunch . || They went to the park after lunch .
He plays the piano very well . || He plays the piano very well .
We are going to visit our grandparents . || We are going to visit our grandparents .
The weather was cold , so we stayed home . || The weather was cold , so we stayed home .
I think their idea is better than ours . || I think their idea is better than ours .
Can you hear the birds singing ? || Can you hear the birds singing ?
Please put the box over there . || Please put the box over there .
He accepted the job offer . || He accepted the job offer .
The quiet library is near the station . || The quiet library is near the station ."""
pairs = [line.split(" || ") for line in data.split("\n")]
kinds = ["実在語の誤り"] * 19 + ["辞書にない語の誤り"] * 4 + ["誤りなし"] * 12   # data の文の順
known = set(sym.words)

def words_of(s):
    """記号を除いた小文字の語の列 (lookup_compound は記号を落とすので、記号を除いて比べる)"""
    return re.findall(r"[a-z]+", s.lower())

def is_word(w):
    return re.fullmatch(r"[a-z]+", w) is not None

def word_by_word_symspell(s):
    """文脈を見ずに、symspellpy の辞書にない語だけを 1 語ずつ直す"""
    return " ".join(by_symspell(w) if is_word(w) and w not in known else w for w in s.lower().split())

def word_by_word_pyspellchecker(s):
    """文脈を見ずに、pyspellchecker の辞書にない語だけを 1 語ずつ直す"""
    return " ".join(by_pyspellchecker(w) if is_word(w) and spell.unknown([w]) else w for w in s.lower().split())

def compound(s):
    return sym.lookup_compound(" ".join(words_of(s)), max_edit_distance=2)[0].term

systems = {"pyspellchecker (1 語ずつ)": word_by_word_pyspellchecker,
           "symspellpy (1 語ずつ)": word_by_word_symspell,
           "symspellpy の lookup_compound": compound,
           "symspellpy + マスク言語モデル": correct_sentence}

for sys_name, f in systems.items():
    t = time.perf_counter()
    n_err = {"実在語の誤り": 0, "辞書にない語の誤り": 0}
    fixed = {"実在語の誤り": 0, "辞書にない語の誤り": 0}
    miscorrected = wrong = 0
    failures = []
    for (src, ref), kind in zip(pairs, kinds):
        s, r, out = words_of(src), words_of(ref), words_of(f(src))
        if len(out) != len(s):
            # 語の数が変わった (分割・結合した) 文は、誤りを直せず、正しい語を 1 語書き換えたと数える
            if kind in n_err:
                n_err[kind] += sum(a != b for a, b in zip(s, r))
            wrong += 1
            failures.append(" ".join(out))
            continue
        for a, o, b in zip(s, out, r):
            if a != b:
                n_err[kind] += 1
                fixed[kind] += (o == b)
                miscorrected += (o != b and o != a)
            elif o != b:
                wrong += 1                      # 正しい語を書き換えた
        if out != r:
            failures.append(" ".join(out))
    sec = time.perf_counter() - t
    detail = " / ".join(f"{k} {fixed[k]}/{n_err[k]}" for k in n_err)
    print(f"{sys_name}: 直せた {sum(fixed.values())}/{sum(n_err.values())} 語 ({detail}) / 別の誤った語にした {miscorrected} / 正しい語を書き換えた {wrong} / {sec:.1f} 秒")
    print("   直しきれなかった文の例:", failures[:4])
```

- この章の「英語のスペルミスをライブラリで直す (pyspellchecker)」は、編集距離と頻度で 1 語ずつ直す方法で、文脈を見ない。このレシピは候補を出す部分を symspellpy にし、選ぶ部分にマスク言語モデルを使って、実在する別の語になってしまった誤り（real-word error。peace → piece、whether → weather）も直す。
- symspellpy は Wolf Garbe の SymSpell (C#) を Python に移したもので、MIT ライセンス。辞書の語から文字を削除した形をあらかじめ索引にしておき、入力からも削除した形を作って引くので、候補を速く引ける。同梱の辞書は SymSpell の README によると Google Books Ngram (CC BY 3.0) と SCOWL から作られている。`lookup_compound` は語の分割・結合の誤り（whereis、th elove）もまとめて直せる。
- Norvig のテストセット（270 組、1 語ずつ）では、正解率は symspellpy が 0.744、pyspellchecker が 0.763 で、差は 5 語ほどなのでほぼ同じと見てよい。速さは大きく違い、この実行では symspellpy が毎秒約 1.5 万語、pyspellchecker が毎秒 16 語だった（pyspellchecker は編集距離 2 の文字列をすべて作って辞書と照合する）。大量の文書を処理するなら symspellpy がよい。誤りの例を見ると、失敗の傾向も 2 つで似ている。正解より近い語があればそちらを選び（contende → contend、symspellpy の juse → use）、同じ距離なら頻度の高い語を選ぶ（localy → local、symspellpy の proplen → people）。入力が実在語 (contended → contented) だと直せない。
- 手作りの 35 文では、文脈を見ない 3 つの方法（lookup_compound も含む）はどれも、辞書にない語の誤り 11 語はすべて直せたが、実在語の誤り 19 語は 1 つも直せなかった。マスク言語モデルで選ぶと 30 語中 26 語（実在語の誤り 15/19、辞書にない語の誤り 11/11）を直し、誤りのない 12 文は 1 語も書き換えなかった。ただし文の数が少なく、減点 (`penalty`) としきい値 (`margin`) はこの 35 文を見ながら決めた値なので、自分のデータで確かめてから使う。
- 直せなかった 4 語（bred → red、here → help、meat → get、road → had）は、どれも文としては自然だが意味の違う語に書き換えている。マスク言語モデルは「その位置に来やすい語」を選ぶので、よく使われる語に引き寄せられる。`margin` を大きくすると書き換えが減り、見逃しが増える。教育の場面では、自動で書き換えずに候補として示すほうが安全である。
- 文脈を使う訂正は遅い。この実行では 2 コアの CPU で 35 文に 38 秒（1 文 1 秒ほど）かかり、symspellpy だけなら 0.1 秒以下だった。大量に処理するなら GPU を使う、取り違えやすい語の組（their / there、then / than など）だけを調べる、といった工夫がいる。
- 小文字にしてから扱うので、大文字・小文字の情報は失われる。辞書にない語は、編集距離 2 以内に候補があれば必ずどれかに書き換えるので、固有名詞や専門用語は `sym.create_dictionary_entry("tokenizer", 1000)` のように辞書に足しておく。学習者のつづりの誤りは打ち間違いと傾向が違う（発音に引きずられる、母語の影響を受ける）ので、学習者の作文で確かめる。日本語では、スペル誤りにあたるのは誤変換や脱字である。日本語入力の章と、この章の「マスク言語モデルで誤りのありそうな箇所を見つける（文法誤り検出）」を参照。
- 仕組みを自分で実装して確かめる版（編集距離 1・2 の候補を作り、頻度で選ぶ Norvig 方式）は、付録「仕組みを理解するための実装」の「編集距離と単語の頻度で英語のスペル誤りを訂正する（Norvig 方式を自分で実装）」にある。
