# 文法誤り訂正と言語教育

学習者の作文を訂正する、訂正箇所を取り出して分類・評価する、スペルミスを直す、文章の読みやすさや語彙の難しさを測る、教材づくりに使う。

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

## LLM で日本語の文法誤りを訂正して評価する {gec_ja_llm gpu}

日本語学習者の書いた文の誤りを訂正したい。LLM (GPU を使う) に訂正させ、参照訂正と比べて評価する。

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
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる (LLM の章の「オープンな LLM とチャットする」を参照)。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)
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
    """LLM が付け足しがちな「訂正後の文:」のような見出しや、末尾の注釈 (「（誤りはありません）」) を取り除く"""
    text = re.sub(r"^(訂正後の文|訂正文|書き直した文|文)\s*[:：]\s*", "", text.strip())
    return re.sub(r"[（(][^（）()]*[）)]$", "", text).strip().strip("「」")

MINIMAL = ("次の文は日本語学習者が書いたものです。文法の誤りだけを、できるだけ少ない変更で訂正してください。"
           "誤りがなければそのまま出力してください。訂正後の文だけを 1 行で出力し、説明は書かないでください。\n\n文: {}")
FLUENT = "次の文は日本語学習者が書いたものです。自然で流暢な日本語に書き直してください。書き直した文だけを 1 行で出力してください。\n\n文: {}"

for s in ["図書館で本が借りました。", "私は毎日学校を行きます。", "昨日は友達と映画を見ます。", "この料理はとてもおいしいでした。", "私は東京に住んでいます。"]:
    print(f"{s}\n   最小限の訂正: {clean(chat(MINIMAL.format(s)))}\n   流暢な書き直し: {clean(chat(FLUENT.format(s)))}")
```

```python
# 訂正を編集 (どこをどう変えたか) の集合にして、参照訂正と比べる
def edits(source, target):
    """文字単位の差分から (開始位置, 終了位置, 置き換える文字列) の集合を作る"""
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

# 100 文 × 2 通りの指示で生成する。T4 で 5 分ほど
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
- 出力の整形が評価を左右する。小さな LLM は「訂正後の文:」のような見出しや「（誤りはありません）」のような注釈を付けたり、助詞を重ねて (「駅のの近く」) 新しい誤りを作ったりする。見出しや注釈は機械的に取り除き、新しい誤りはそのまま評価に反映させる。訂正文に続けて説明を書き出す出力 (「…見に行きました。」が正しい文です。…」) は `clean` では取り除けないので、必要なら規則を足す。
- 意味を変えてしまう訂正も出る (「レストランが夕食を食べました」→「レストランは夕食を食べませんでした」)。
- LLM は直しすぎる傾向がある (正しい部分まで言い換える)。編集単位の適合率が下がるので、F0.5 のように適合率を重視する指標では不利になる。この実験では、流暢な書き直しは再現率が高い (0.519) が適合率が低く (0.418)、F0.5 は 0.435 で、最小限の訂正の 0.441 をわずかに下回った (T4 と同じ float16 で実行した値。100 文なので、この程度の差は誤差の範囲)。逆に、最小限の訂正は直すべき文を変えずに残すことがある (「公園に友達と遊びました。」)。
- 参照訂正は 1 つとは限らない。複数の参照と比べないと、正しい訂正を誤りと数えてしまう。それでも参照にない正しい訂正は残る。
- ここでは文字単位の差分で編集を作った。きちんと評価するなら、形態素単位で編集を取り出す (この章の ERRANT のレシピの日本語版にあたる処理)。
- このデータは JFLEG にならって作られた公開データで、文が短く誤りも単純。実際の学習者の作文 (Lang-8 コーパス、TEC-JL など) はもっと難しい。

## 参照訂正なしで訂正文の質を評価する {gec_reference_free}

参照訂正 (正解) がなくても、訂正システムの出力が「文法的か」「流暢か」「元の意味を保っているか」を自動で評価したい。

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

- 参照訂正と比べる評価 (M2、ERRANT) は、参照にない正しい訂正を誤りと数えてしまう。参照なしの評価は、訂正文そのものの質を直接見るので、この問題がない。
- 見る観点は 3 つ: 文法性 (誤りが残っていないか)、流暢性 (自然な文か)、意味の保存 (元の文の内容を変えていないか)。1 つだけだと簡単にだませる (流暢さだけなら、無関係な自然な文が最高点になる)。
- 言語モデルの尤度は、頻度の高い語や言い回しに甘い。正しくても珍しい表現は低く、誤りを含んでいてもありふれた表現は高くなる。
- 人手評価に合うように各観点のモデルを学習した指標に SOME、訂正前後の文から品質を推定する指標に IMPARA がある。本格的に使うなら、これらの公開モデルを検討する。
- 文埋め込みの類似度は、助詞 1 つの違いのような小さな変更にはほとんど反応しない。「意味が大きく変わっていないか」の確認として使う。

## LLM に訂正の良し悪しを判定させて人手評価と比べる (メタ評価) {gec_llm_evaluator gpu}

複数の訂正システムの出力のうちどれが良いかを LLM に判定させたい。その判定が人の評価とどれくらい合うかを確かめる (評価指標の評価 = メタ評価)。

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
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる (LLM の章の「オープンな LLM とチャットする」を参照)。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)
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
    """提示順を入れ替えて 2 回聞き、平均する (x のほうが良い確率)"""
    return (prefer_a(source, x, y) + 1 - prefer_a(source, y, x)) / 2
```

```python
# 元の文ごとに、質の違う 4 つの訂正と、人が付けた順位 (1 が最良。同順位あり) を用意する
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
        wins[i] += p; wins[j] += 1 - p               # 総当たりの「勝ち数」(確率の和) をスコアにする
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

- メタ評価は「評価指標がどれくらい人の評価と合うか」を測る。文単位 (同じ元の文に対する訂正の順位) とシステム単位 (全体の平均での順位) の 2 つの粒度があり、結果が食い違うことがある。この実験では、文ごとの Kendall τ の平均が 0.42、全体の Spearman が 0.65、システム単位の Spearman が 0.8 だった (T4 と同じ float16 で実行した値)。3B のモデルは「訂正なし」を「流暢な書き直し」より高く評価している。
- 総当たりの一対比較は、1〜5 の採点をさせるより小さなモデルでも安定する。ただし比較の回数は候補数の 2 乗で増える。
- 大規模な LLM による評価は、文法誤り訂正でも従来の自動評価指標より人手評価に近いと報告されている。流暢な書き直しを正当に評価できる点が、編集の一致を数える指標との違い。
- 人手評価のデータそのものにも設計上の問題がありうる (評価した時代のシステムが古い、文単位と編集単位の評価が混ざる)。メタ評価用のデータとして SEEDA が公開されている。
- ここの 6 文は手作りの動作確認用。指標を採用するかどうかは、実際のシステム出力と人手評価の組 (数百文以上) で判断する。

## マスク言語モデルで誤りのありそうな箇所を見つける (文法誤り検出) {ged_mlm}

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
- 助詞の誤り (「本が借りました」) のように、周りの語から正解がほぼ決まる誤りは見つけやすい。内容語の選び間違いや、語の抜け・余分は見つけにくい。
- 珍しい固有名詞や専門用語も確率が低くなるので、誤りと区別できない。しきい値を下げると見逃しが増え、上げると誤検出が増える。
- 誤りの位置を正確に指すとは限らない (誤った助詞ではなく、その後ろの動詞の確率が下がることがある)。「このあたりが怪しい」という手がかりとして使う。
- 高い精度が必要なら、誤りの注釈付きデータで系列ラベリング (各トークンが正しいか誤りか) を学習する。固有表現抽出の章の BERT のファインチューニングと同じ方法が使える。

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

- 編集距離と単語頻度に基づく Norvig 方式。文脈は見ないので `their/there` のような実在語の誤りは直せない。文脈を見る方法は、この章の「英語のスペル誤りを訂正する」を参照。

## 英語のスペル誤りを訂正する {spelling_correction}

英語のつづりの誤りを見つけて訂正したい。編集距離と単語の頻度による古典的な方法を作り、文脈を使う方法と比べる (ライブラリで手軽に直すだけなら、この章の「英語のスペルミスをライブラリで直す (pyspellchecker)」)。

```python
!pip install -q symspellpy
!wget -q https://norvig.com/big.txt https://norvig.com/spell-testset1.txt
```

```python
import re
from collections import Counter

WORDS = Counter(re.findall(r"[a-z]+", open("big.txt", encoding="utf-8").read().lower()))     # 約 100 万語の英文から単語の頻度を数える
TOTAL = sum(WORDS.values())
LETTERS = "abcdefghijklmnopqrstuvwxyz"

def edits1(word):
    """編集距離 1 の文字列をすべて作る (削除・隣接文字の入れ替え・置換・挿入)"""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits if b for c in LETTERS]
    inserts = [a + c + b for a, b in splits for c in LETTERS]
    return set(deletes + transposes + replaces + inserts)

def known(words):
    return {w for w in words if w in WORDS}

def candidates(word):
    """辞書にある語を、編集距離の小さい順に探す"""
    return known([word]) or known(edits1(word)) or known(e2 for e1 in edits1(word) for e2 in edits1(e1)) or {word}

def correct(word):
    return max(candidates(word), key=WORDS.get)

for w in ["speling", "korrectud", "bycycle", "inconvient", "recieve", "definately", "their", "teh"]:
    print(f"{w} → {correct(w)}  (候補: {sorted(candidates(w), key=WORDS.get, reverse=True)[:5]})")
```

```python
# 誤りと正解の組のテストセットで正解率を測る
tests = [(wrong, right) for line in open("spell-testset1.txt") for right, wrongs in [line.strip().split(":")] for wrong in wrongs.split()]

def accuracy(corrector, tests):
    result = [corrector(wrong) == right for wrong, right in tests]
    return sum(result) / len(result)

print(len(tests), "組 / 正解率:", round(accuracy(correct, tests), 3), "/ 正解の語が辞書にない割合:", round(sum(r not in WORDS for _, r in tests) / len(tests), 3))
errors = [(w, correct(w), r) for w, r in tests if correct(w) != r][:8]
print("誤りの例 (入力, 出力, 正解):", errors)
```

```python
# 高速なライブラリ: SymSpell (削除だけで候補を引く索引を作っておくので、非常に速い)
import time
from symspellpy import SymSpell, Verbosity

sym = SymSpell(max_dictionary_edit_distance=2)
for word, count in WORDS.items():
    sym.create_dictionary_entry(word, count)

def correct_symspell(word):
    hits = sym.lookup(word, Verbosity.TOP, max_edit_distance=2)
    return hits[0].term if hits else word

for name, f in [("自作", correct), ("SymSpell", correct_symspell)]:
    t = time.perf_counter(); acc = accuracy(f, tests)
    print(f"{name}: 正解率 {acc:.3f} / {len(tests) / (time.perf_counter() - t):.0f} 語/秒")
```

```python
# 文脈を使う: 候補のうち、マスク言語モデルが文脈に合うと判断したものを選ぶ (辞書にある語への打ち間違いにも対応できる)
import math
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
mlm = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").eval()

@torch.no_grad()
def correct_in_context(sentence, position, penalty=2.0):
    words = sentence.split()
    word = words[position].lower()
    near = known(edits1(word))
    far = known(e2 for e1 in edits1(word) for e2 in edits1(e1)) - near      # peace → piece のように、実在語どうしの取り違えは編集距離 2 のことが多い
    distance = {word: 0, **{c: 1 for c in near - {word}}, **{c: 2 for c in far - {word}}}
    enc = tokenizer(" ".join(words[:position] + [tokenizer.mask_token] + words[position + 1:]), return_tensors="pt")
    pos = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero()[0, 0]
    logp = mlm(**enc).logits[0, pos].log_softmax(-1)
    def score(c):
        ids = tokenizer(c, add_special_tokens=False)["input_ids"]
        context = float(logp[ids[0]]) if len(ids) == 1 else -20.0               # 1 トークンで表せない語は文脈のスコアを使わない
        return context - penalty * distance[c]                                   # 元の語から離れるほどペナルティ
    return max(distance, key=score)

for sentence, position in [("I want a peace of cake .", 3), ("He will except the offer .", 2), ("She lives in a quite village .", 4),
                           ("The whether is nice today .", 1), ("Their is a cat on the table .", 0), ("I need to buy some bred .", 5),
                           ("I like to read books .", 4), ("The cat sat on the mat .", 1)]:
    print(sentence, "→", sentence.split()[position], "⇒", correct_in_context(sentence, position), "(単語だけで訂正:", correct(sentence.split()[position].lower()) + ")")
```

- 古典的なスペル訂正は「編集距離が近い語のうち、最も頻度の高い語を選ぶ」だけで、かなりの誤りを直せる。Peter Norvig の有名な 20 数行の実装がこれにあたる。
- 弱点は 3 つ: 辞書にない語は正解にできない、頻度だけで選ぶので文脈に合わない語を選ぶことがある、そして実在する別の語になってしまった誤り (peace / piece) には気付けない。
- 文脈を使うと、実在する語への打ち間違い (real-word error) も直せる (peace → piece、whether → weather)。こうした取り違えは編集距離 2 のことが多いので、候補を距離 2 まで広げる。正しい語まで書き換えてしまう危険が増えるので、元の語から離れるほどペナルティを付け、正しい文 (books、cat) が変わらないことも確かめる。
- 学習者のスペル誤りは、打ち間違いとは傾向が違う (発音に引きずられる、母語の影響を受ける)。編集操作ごとの起こりやすさを学習者のデータから推定すると、精度が上がる。
- 日本語には分かち書きがなく、スペル誤りにあたるのは誤変換や脱字。日本語入力の章と、文法誤り検出のレシピを参照。

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
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる (LLM の章の「オープンな LLM とチャットする」を参照)。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)
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

- フィードバックコメント生成は、訂正 (何を直すか) とは別のタスクとして研究されている。学習効果のためには、答えを示すだけでなく、規則に気付かせる説明が重要とされる。
- 訂正箇所を差分で明示してから説明させると、どの誤りについての説明かがぶれにくい。誤りが複数あるときは、1 か所ずつ説明させる。
- 長さの指示は守られにくい。この実験でも、3 つ目の説明は「150 字以内」の指示を大きく超えて 400 字近くになり、「recentness」のような英語まで混ざった。1 つ目も 150 字をわずかに超えている (下の点検は上限を 300 字にしているので通る)。生成後に長さを点検し、超えたら要約させ直す、といった後処理を用意する。
- 説明が訂正と食い違うこともある。2 つ目の説明は、訂正後の文 (「おいしかったです」) に反して「「ました」を使うのが正解」と書いている。
- 小さな LLM は、もっともらしいが誤った文法説明を書くことがある (存在しない規則を作る、助詞の働きを取り違える)。学習者にそのまま見せる前に、教師による確認か、誤りの種類ごとに用意した説明文 (テンプレート) との併用を検討する。
- 誤りの種類が分かっていれば (助詞の選択、時制、活用など)、種類ごとに人が書いた説明を引いてくる方法が最も安全。誤りタイプの判定には、この章の ERRANT のレシピの考え方が使える。
- 学習者のレベルに合わせて説明の言葉を変える (初級者には母語で、など) ことも、LLM への指示で行える。

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
- 英語の Flesch-Kincaid (この章の「英文の読みやすさ (リーダビリティ) を測る」) のような定番の式は日本語にはない。日本語の公開ツールとしては jReadability などがある。

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

## 難しい単語を見つけてやさしい単語に言い換える (語彙平易化) {lexical_simplification}

文章の中の難しい単語を見つけ、文脈に合うやさしい単語に置き換えたい。

```python
!pip install -q fugashi unidic-lite wordfreq
```

```python
import torch
from fugashi import Tagger
from transformers import AutoModelForMaskedLM, AutoTokenizer
from wordfreq import zipf_frequency

tagger = Tagger()
name = "tohoku-nlp/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForMaskedLM.from_pretrained(name).eval()

def complex_words(sentence, threshold=4.0):
    """1. 難しい語の特定: 内容語のうち、頻度 (Zipf 尺度) がしきい値より低いものを難しい語とする"""
    out = []
    for w in tagger(sentence):
        if w.feature.pos1 in ("名詞", "動詞", "形容詞", "形状詞", "副詞") and w.feature.pos2 not in ("固有名詞", "数詞", "非自立可能"):
            z = zipf_frequency(w.feature.lemma or w.surface, "ja")
            if z < threshold:
                out.append((w.surface, round(z, 2)))
    return out

sentence = "政府は景気の低迷を打開するため、新たな施策を講じると表明した。"
print(complex_words(sentence))
```

```python
@torch.no_grad()
def candidates(sentence, target, k=30):
    """2. 言い換え候補の生成: 「元の文 [SEP] 対象語をマスクした文」を与え、マスクに入る語を予測させる (元の文を見せると、意味の近い語が出やすい)"""
    masked = sentence.replace(target, tokenizer.mask_token, 1)
    enc = tokenizer(sentence, masked, return_tensors="pt")
    pos = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero()[0, 0]
    probs = model(**enc).logits[0, pos].softmax(-1)
    top = probs.topk(k)
    return [(tokenizer.convert_ids_to_tokens(int(i)), float(p)) for p, i in zip(top.values, top.indices)]

def simplify_word(sentence, target, k=30, min_prob=0.02):
    """3. 候補の選別と順位付け: 元の語と同じ品詞で、元の語より頻度が高く (やさしく)、文脈に合う確率が一定以上の語だけを残す"""
    target_word = next(w for w in tagger(target))
    target_pos, target_zipf = target_word.feature.pos1, zipf_frequency(target_word.feature.lemma or target, "ja")
    ranked = []
    for cand, p in candidates(sentence, target, k):
        if cand.startswith("##") or cand == target or cand in target or target in cand:
            continue
        w = next(iter(tagger(cand)), None)
        z = zipf_frequency(cand, "ja")
        if w is not None and len(tagger(cand)) == 1 and w.feature.pos1 == target_pos and z > target_zipf and p >= min_prob:
            ranked.append((cand, round(z, 2), round(p, 3)))
    return sorted(ranked, key=lambda x: -x[2])[:5]                 # 文脈に合う確率の高い順

for target, z in complex_words(sentence):
    print(f"{target} (頻度 {z}) →", simplify_word(sentence, target))
```

```python
def simplify(sentence, threshold=4.0):
    out = sentence
    for target, _ in complex_words(sentence, threshold):
        cands = simplify_word(out, target)
        if cands:
            out = out.replace(target, cands[0][0], 1)
    return out

for s in [sentence, "医師は患者に対し、速やかに治療を開始するよう勧告した。", "当該施設の利用に際しては、事前に申請を行う必要がある。",
          "彼は困難な状況に直面しても、決して諦めなかった。"]:
    print(f"{s}\n   → {simplify(s)}")
```

- 語彙平易化は 3 段階に分けて考える: 難しい語の特定 (complex word identification)、言い換え候補の生成、候補の選別と順位付け。
- 単語の難しさは、頻度でかなり近似できる。日本語教育では、語彙の級 (旧日本語能力試験の級など) や学習者コーパスでの頻度も使われる。誰にとって難しいか (子ども、学習者、専門外の人) で基準は変わる。
- マスク言語モデルは文脈に合う語を出すが、意味が同じとは限らない (反対語や、単に文脈に合うだけの別の語が出る)。確率の低い候補を捨て、良い候補がなければ言い換えない、という慎重な設定にする。それでも意味のずれは残るので、結果は人が確認する。
- 活用語は、候補の語を元の語と同じ活用形にそろえる必要がある。ここでは形の合う候補だけを残す簡易な方法をとっている。
- LLM に「この語をやさしく言い換えて」と頼む方法が、現在は最も精度が高い。その場合も、候補が本当にやさしいか (頻度) と、意味が保たれているかの確認は別に行う。
- 単語単位の言い換えでは、文の構造の難しさは解消しない。文全体の平易化は、要約と平易化の章を参照。

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

## 作文を自動で採点する {essay_scoring}

学習者の作文に、人が付けるのと近い点数を自動で付けたい。特徴量による回帰と文埋め込みによる回帰を比べ、採点者間の一致度と同じ指標 (QWK) で評価する。

```python
!pip install -q datasets sentence-transformers
```

```python
import re
import numpy as np
import pandas as pd
from datasets import load_dataset

df = load_dataset("nlpatunt/D_ASAP-AES", split="train").to_pandas()      # ASAP: 米国の生徒が書いた英語の作文と、人が付けた点数
print(df.columns.tolist(), len(df))
print(df.groupby("essay_set")["domain1_score"].agg(["count", "min", "max", "mean"]).round(2))
```

```python
essays = df[df["essay_set"] == 1].dropna(subset=["domain1_score"]).sample(frac=1, random_state=0).reset_index(drop=True)     # 課題 1 (意見文、2〜12 点)
essays["score"] = essays["domain1_score"].astype(int)
train, test = essays.iloc[:1000], essays.iloc[1000:]
print(len(train), "件で学習 /", len(test), "件で評価")

def handcrafted(text):
    words = re.findall(r"[A-Za-z']+", text.lower())
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    return [len(words), np.log(len(words) + 1), len(set(words)) / (len(words) + 1), np.mean([len(w) for w in words] or [0]),
            len(sentences), len(words) / (len(sentences) + 1), text.count(","), sum(len(w) >= 8 for w in words) / (len(words) + 1)]

names = ["語数", "log 語数", "異なり語の割合", "平均語長", "文数", "平均文長", "コンマの数", "長い語の割合"]
X_train, X_test = np.array([handcrafted(t) for t in train["essay"]]), np.array([handcrafted(t) for t in test["essay"]])
print(pd.Series({n: np.corrcoef(X_train[:, i], train["score"])[0, 1] for i, n in enumerate(names)}).round(3).to_string())
```

```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def qwk(y_true, y_pred):
    """重み付き κ 係数 (quadratic weighted kappa): 点数の差が大きい食い違いほど重く罰する。作文採点の標準的な指標"""
    y_pred = np.clip(np.rint(y_pred), y_true.min(), y_true.max()).astype(int)
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
E_train = embedder.encode(train["essay"].tolist(), batch_size=32, normalize_embeddings=True)
E_test = embedder.encode(test["essay"].tolist(), batch_size=32, normalize_embeddings=True)

y_train, y_test = train["score"].to_numpy(), test["score"].to_numpy()
models = {
    "語数だけ": (X_train[:, :2], X_test[:, :2]),
    "手作りの特徴量": (X_train, X_test),
    "文埋め込み": (E_train, E_test),
    "特徴量 + 文埋め込み": (np.hstack([X_train, E_train * 10]), np.hstack([X_test, E_test * 10])),
}
for label, (a, b) in models.items():
    pred = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(a, y_train).predict(b)
    print(f"{label}: QWK = {qwk(y_test, pred):.3f} / 平均絶対誤差 = {np.abs(pred - y_test).mean():.2f}")

human = essays.dropna(subset=["rater1_domain1", "rater2_domain1"])
print("参考: 人の採点者 2 人の間の QWK =", round(cohen_kappa_score(human["rater1_domain1"].astype(int), human["rater2_domain1"].astype(int), weights="quadratic"), 3))
```

- 作文の長さだけで、点数のかなりの部分が説明できてしまう。新しい手法は、必ず「語数だけ」のベースラインと比べる。長さだけを見るモデルは、意味のない長文に高得点を付けてしまう。
- この課題の点数は 2 人の採点者の点数の合計なので、1 人の採点者どうしの一致度 (QWK 0.70) より、モデルと合計点の一致度のほうが高く出る。人と同等以上に見えても、比べている対象が違う点に注意する。
- 文埋め込みは、この課題では手作りの特徴量より大きく劣った。一般的な文埋め込みは文章の話題を表すもので、文章の巧拙はあまり反映しない。
- 評価には QWK (重み付き κ) を使う。人の採点者どうしの一致度と同じ物差しなので、「人と同じくらい一致している」かどうかを直接比べられる。
- 課題 (プロンプト) ごとに採点基準も点数の範囲も違う。ある課題で学習したモデルは、別の課題にはそのまま使えない。
- 自動採点は、内容の正しさや論理の一貫性をほとんど見ていない。重要な判定に使うときは、人の採点と併用し、極端な入力 (丸写し、無関係な文章) への挙動を確かめる。
- 日本語学習者の作文では、文法誤りの数、語彙の難易度 (この章の語彙の難しさのレシピ)、接続表現の使い方などが特徴量としてよく使われる。観点別 (内容・構成・言語) に採点する方法もある。
