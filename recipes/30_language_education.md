# 言語教育・リーダビリティ・自動採点

文章の読みやすさ（リーダビリティ）や語彙の難しさを測る方法、目標の難易度 (CEFR) に合わせた語彙平易化、穴埋め問題や多肢選択式の問題の作成、総合点および観点別ルーブリックに基づく作文の自動採点を扱う。学習者の文法誤り訂正やスペル訂正は「文法誤り訂正と文章校正」の章を参照。教育応用を含む日本語処理や順序尺度・自動採点の評価については、姉妹編『[自然言語処理の教科書](https://github.com/mamorlis/nlpbook)』の第 2 章・第 3 章も参照。

## 英文の読みやすさ（リーダビリティ）を測る {readability_en}

英文の難易度を、学年レベルなどの指標で数値化したい（教材の選定、平易化の評価）。

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

- どの指標も「文の長さ」と「単語の長さ（音節数）」の組み合わせ。短い文章では値が不安定になる。

## 日本語の文章の読みやすさを測る {readability_ja}

日本語の文章の難しさを、文の長さ・漢字の割合・語種などの簡単な指標で数値化したい（教材選び、やさしい日本語の確認）。

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

- 文が長く、漢語（音読みの語）が多いほど難しく感じられる。UniDic の語種 (`goshu`) で和語・漢語・外来語を区別できる。
- 学習者向けには、語彙の難易度（日本語能力試験の級別語彙など）や、係り受けの距離も重要な手がかりになる。
- 英語の Flesch-Kincaid（この章の「英文の読みやすさ（リーダビリティ）を測る」）のような定番の式は日本語にはない。日本語の公開ツールとしては jReadability などがある。

## 単語の頻度から語彙の難しさを推定する (wordfreq) {word_frequency_level}

文中のどの単語が「珍しい（難しい）」のかを、大規模コーパスでの出現頻度から見積もりたい。

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

- wordfreq は Wikipedia、字幕、Web、SNS など複数の資源を統合した頻度表で、40 以上の言語に対応している（データは 2021 年までで更新終了）。
- 頻度は難しさの近似にすぎない。高頻度でも多義で難しい語（「かける」「とる」）や、学習者の母語によって易しい語（漢字圏の学習者にとっての漢語）がある。
- 日本語の教育語彙としては、日本語教育語彙表や旧日本語能力試験の出題基準などが使われる。

## 目標の難易度（CEFR レベル）に合わせて語彙を平易化する {cefr_simplification}

TSAR のような難易度制御付きの語彙平易化を、目標レベル（`B1/B2` 向け vs より易しい `A1/A2` 向け）を切り替えて行いたい。`wordfreq` の Zipf 頻度で難語を特定し、マスク言語モデルで文脈に合う置換候補を生成したうえで、品詞一致・目標レベルの頻度範囲・文埋め込みと NLI による意味保持で候補を選び、文章全体の語彙レベルとリーダビリティの変化を確かめる。

```python
!pip install -q wordfreq textstat sentence-transformers
```

```python
import re
import nltk
import numpy as np
import pandas as pd
import textstat
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer
from wordfreq import zipf_frequency

nltk.download("averaged_perceptron_tagger_eng", quiet=True)

mlm_name = "distilbert-base-uncased"
mlm_tok = AutoTokenizer.from_pretrained(mlm_name)
mlm = AutoModelForMaskedLM.from_pretrained(mlm_name).eval()

sim_model = SentenceTransformer("all-MiniLM-L6-v2")

nli_name = "cross-encoder/nli-deberta-v3-xsmall"
nli_tok = AutoTokenizer.from_pretrained(nli_name)
nli = AutoModelForSequenceClassification.from_pretrained(nli_name).eval()
```

```python
sentences = [
    "The committee attempted to ameliorate the adverse consequences of the economic downturn.",
    "Smartphones have become ubiquitous in contemporary society.",
    "Researchers must substantiate their hypotheses with empirical evidence.",
    "The new policy will mitigate the financial burden on low-income families.",
    "Careful planning can alleviate severe traffic congestion during peak hours.",
    "The scientist sought to elucidate the complex mechanism behind the phenomenon.",
]

# CEFR 目標レベルごとの頻度範囲（Zipf 尺度: 値が大きいほど高頻度・平易）と意味保持のしきい値
CEFR_TARGETS = {
    "B1/B2": {"trigger_zipf": 4.2, "min_zipf": 4.2, "max_zipf": 5.0, "min_sent_sim": 0.85, "min_nli": 0.65},
    "A1/A2": {"trigger_zipf": 4.6, "min_zipf": 4.9, "max_zipf": 8.0, "min_sent_sim": 0.80, "min_nli": 0.55},
}

def detokenize(tokens):
    return re.sub(r"\s+([.,!?;:])", r"\1", " ".join(tokens))

@torch.no_grad()
def bi_entailment(orig, cands):
    """原文と候補文の双方向の含意確率の最小値（反意語や無関係な語へのすり替わりを防ぐ）"""
    fwd = nli(**nli_tok([orig] * len(cands), cands, padding=True, truncation=True, return_tensors="pt")).logits.softmax(-1)[:, 1]
    bwd = nli(**nli_tok(cands, [orig] * len(cands), padding=True, truncation=True, return_tensors="pt")).logits.softmax(-1)[:, 1]
    return torch.minimum(fwd, bwd).numpy()

@torch.no_grad()
def simplify_to_cefr(sent, cfg):
    tokens = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*'?[A-Za-z]*|[.,!?;:]", sent)
    tagged = nltk.pos_tag(tokens)
    orig_emb = sim_model.encode([sent], normalize_embeddings=True, show_progress_bar=False)[0]
    out_tokens = list(tokens)
    edits = []
    for i, (w, tag) in enumerate(tagged):
        if tag[:2] not in ("NN", "VB", "JJ", "RB") or not w.isalpha() or w[0].isupper():
            continue
        z_orig = zipf_frequency(w, "en")
        if z_orig >= cfg["trigger_zipf"]:
            continue
        cur_sent = detokenize(out_tokens)
        masked_sent = detokenize(out_tokens[:i] + [mlm_tok.mask_token] + out_tokens[i + 1:])
        enc = mlm_tok(cur_sent, masked_sent, return_tensors="pt")
        pos = (enc["input_ids"][0] == mlm_tok.mask_token_id).nonzero()[0, 0]
        top = mlm(**enc).logits[0, pos].softmax(-1).topk(80)

        used = {t.lower() for j, t in enumerate(out_tokens) if j != i and t.isalpha()}
        cands = []
        for idx, p in zip(top.indices.tolist(), top.values.tolist()):
            cw = mlm_tok.convert_ids_to_tokens(idx)
            if not cw.isalpha() or len(cw) < 2 or cw == w.lower() or cw in used or p < 1e-4:
                continue
            zc = zipf_frequency(cw, "en")
            if not (cfg["min_zipf"] <= zc <= cfg["max_zipf"] and zc > z_orig + 0.3):
                continue
            cand_tokens = out_tokens[:i] + [cw] + out_tokens[i + 1:]
            if nltk.pos_tag(cand_tokens)[i][1] != tag:
                continue
            cands.append((cw, zc, p, detokenize(cand_tokens)))
        if not cands:
            continue
        cs_embs = sim_model.encode([c[3] for c in cands], normalize_embeddings=True, show_progress_bar=False)
        s_sims = cs_embs @ orig_emb
        shortlist = sorted(
            [(cw, zc, p, cs, float(ss)) for (cw, zc, p, cs), ss in zip(cands, s_sims) if ss >= cfg["min_sent_sim"]],
            key=lambda x: -(0.7 * x[4] + 0.3 * min(x[2] * 10, 1.0)),
        )[:10]
        if not shortlist:
            continue
        nli_scores = bi_entailment(cur_sent, [x[3] for x in shortlist])
        valid = [(cw, zc, p, ss, float(ns)) for (cw, zc, p, _, ss), ns in zip(shortlist, nli_scores) if ns >= cfg["min_nli"]]
        if valid:
            best = max(valid, key=lambda x: 0.5 * x[4] + 0.4 * x[3] + 0.1 * min(x[2] * 10, 1.0))
            out_tokens[i] = best[0]
            edits.append((w, round(z_orig, 2), best[0], round(best[1], 2), round(best[4], 2)))
    return detokenize(out_tokens), edits
```

```python
def corpus_metrics(texts):
    words = [w for t in texts for w in re.findall(r"[A-Za-z]+", t) if zipf_frequency(w, "en") > 0]
    return {
        "平均 Zipf 頻度": float(np.mean([zipf_frequency(w, "en") for w in words])),
        "低頻度語 (<4.5) 割合": float(np.mean([zipf_frequency(w, "en") < 4.5 for w in words])),
        "Flesch-Kincaid Grade": float(np.mean([textstat.flesch_kincaid_grade(t) for t in texts])),
    }

results = {"原文": sentences}
for level, cfg in CEFR_TARGETS.items():
    outs = []
    print(f"=== {level} ===")
    for s in sentences:
        out, edits = simplify_to_cefr(s, cfg)
        outs.append(out)
        print(f"  {out}")
        if edits:
            print("    置換:", ", ".join(f"{o}({zo})→{c}({zc}, NLI={ns})" for o, zo, c, zc, ns in edits))
    results[level] = outs

summary = pd.DataFrame({k: corpus_metrics(v) for k, v in results.items()}).T.round(2)
print("\n=== レベル別の語彙頻度とリーダビリティの変化 ===")
print(summary.to_string())
```

- 難語のしきい値 (`trigger_zipf`) と置換候補の頻度範囲 (`min_zipf`〜`max_zipf`) を目標 CEFR レベルに合わせて変えることで、`B1/B2` では *address / negative / confirm / theories / reduce* のように中級語彙へ、`A1/A2` では *tried / bad / results / common / support / lower* のように基礎語彙へ段階的に平易化され、平均 Zipf 頻度と Flesch-Kincaid Grade が単調に易しくなる。
- マスク言語モデルの確率と文埋め込みの類似度だけでは、同じ文脈に収まる反意語（例: *mitigate* → *increase*）を通すことがある。原文と候補文の双方向の NLI 含意確率を組み合わせることで意味の反転を弾ける。
- 語彙の置換だけでは文構造や連語（コロケーション）の不自然さ（例: *severe traffic* の後ろの *peak hours* → *high hours*）が残ることがあり、単語単位の言い換えの限界も見える。文全体の書き換えや日本語の平易化については、要約と平易化の章の「難しい単語を見つけてやさしい単語に言い換える（語彙平易化）」「文章をやさしい日本語に書き換える（テキスト平易化）」も参照。

## 穴埋め問題（クローズテスト）を自動で作る {cloze_generation}

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
- 特定の文法項目（助詞、活用）を問いたいときは、`target_pos` を変える。

## 文章から内容理解を問う多肢選択式の問題を LLM で作り、自動で検証して選ぶ {quiz_generation gpu}

説明文を読んだ人が内容を理解できたかを確かめる 4 択の問題を、LLM で自動で作りたい。作った問題のうち、本文を読めば正解でき、読まなくても当てられず、正解が 1 つに決まるものだけを残す。

```python
!pip install -q datasets==4.8.5
```

```python
import json, random, re
import sentencepiece as spm
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(0); torch.manual_seed(0)
# 本文: JSQuAD（JGLUE、CC BY-SA 4.0）の検証セットの段落。元は日本語 Wikipedia の記事。記事 1 つにつき段落 1 つ（250〜450 字）を選ぶ
ds = load_dataset("sbintuitions/JSQuAD", split="validation")
# 教材の例に向かない政治的な題材（政党、選挙、政治体制、紛争、植民地支配、民族・宗教の対立など）の記事は除く
POLITICAL = {"日本共産党", "ルガンスク人民共和国", "コンゴ共和国", "ラオス", "国際連合平和維持活動", "労働基本権",
             "オランダ", "ポルトガル", "グスタフ・マーラー", "同君連合", "出入国管理及び難民認定法"}
by_title = {}
for c in dict.fromkeys(ds["context"]):
    title, text = [s.strip() for s in c.split("[SEP]", 1)]
    if 250 <= len(text) <= 450 and title not in POLITICAL:
        by_title.setdefault(title, text)
titles = random.Random(0).sample(sorted(by_title), 23)
passages = [by_title[t] for t in titles]
print(len(by_title), "記事から", len(titles), "記事:", titles)

# 問題を作る LLM: sarashina2.2-3b-instruct（SB Intuitions、MIT）
name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
tokenizer.padding_side = "left"           # バッチ生成では左パディング
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()

PROMPT = """次の文章の内容を理解できているかを確かめる 4 択の問題を 3 問作ってください。
- 文章を読めば答えが 1 つに決まり、文章を読まないと答えられない問題にする
- 誤答は、文章に出てくる語句などを使ったもっともらしいものにするが、文章から見て明らかに誤りであるものにする
- 次の形の JSON の配列だけを出力する
[{{"question": "問題文", "answer": "正解", "distractors": ["誤答1", "誤答2", "誤答3"]}}, ...]

# 文章
{passage}"""

@torch.no_grad()
def generate(prompts, max_new_tokens=512, batch_size=24):
    outs = []
    for i in range(0, len(prompts), batch_size):
        texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
                 for p in prompts[i:i + batch_size]]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(llm.device)
        out = llm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        outs += tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs

raw = generate([PROMPT.format(passage=p) for p in passages])     # 23 記事 × 3 問。T4 で 3〜5 分（モデルのダウンロードは別）
print(f"GPU メモリの最大 {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
print(titles[0], "\n", passages[0], "\n\n", raw[0])
```

```python
# 1. 形式の検査: JSON として読めるか、選択肢が 4 つで重複がないか
def parse(out):
    """出力から {"question": ... } のオブジェクトを 1 つずつ取り出す（配列の閉じ忘れなどがあっても、読める問題は拾う）。
    1 問も取り出せなかった出力は、3 問分が壊れていても None の 1 件として返す"""
    items, dec = [], json.JSONDecoder()
    for m in re.finditer(r'\{\s*"question"', out):
        try:
            items.append(dec.raw_decode(out[m.start():])[0])
        except json.JSONDecodeError:
            items.append(None)
    return items or [None]

def format_error(item):
    fields_ok = (isinstance(item, dict) and all(isinstance(item.get(k), str) for k in ("question", "answer"))
                 and isinstance(item.get("distractors"), list) and all(isinstance(x, str) for x in item["distractors"]))
    if not fields_ok:
        return "JSON が壊れている"
    if len(item["distractors"]) != 3:
        return f"誤答が {len(item['distractors'])} 個"
    if len({c.strip() for c in [item["answer"], *item["distractors"]]} - {""}) < 4:
        return "選択肢が重複・空"
    return None

quiz = []
for pi, out in enumerate(raw):
    for item in parse(out):
        q = {"passage": pi, "reason": format_error(item)}
        if q["reason"] is None:
            q["question"] = item["question"].strip()
            q["choices"] = [c.strip() for c in [item["answer"], *item["distractors"]]]   # choices[0] が正解
        quiz.append(q)
print("取り出した問題", len(quiz), "/ 形式の検査に通った問題", sum(q["reason"] is None for q in quiz))
for pi in sorted({q["passage"] for q in quiz if q["reason"]})[:2]:
    print(f"--- 形式の誤り ({titles[pi]}): {[q['reason'] for q in quiz if q['passage'] == pi and q['reason']]}\n{raw[pi][:400]}")
```

```python
import numpy as np

# 2. 内容の検査には、問題を作ったのとは別の LLM を使う: Qwen3-4B-Instruct-2507（Alibaba の Qwen チーム、Apache 2.0）
del llm; torch.cuda.empty_cache()        # T4 の 15GB に 2 つのモデルは載らないので、作る側を消してから読み込む
name = "Qwen/Qwen3-4B-Instruct-2507"
tok = AutoTokenizer.from_pretrained(name)
tok.padding_side = "left"
judge_llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()
digit_ids = [tok(str(i), add_special_tokens=False)["input_ids"][0] for i in range(1, 5)]         # 「1」〜「4」のトークン
yes_no_ids = [tok(w, add_special_tokens=False)["input_ids"][0] for w in ("はい", "いいえ")]      # 最初のトークン（このモデルでは「は」と「いい」）

@torch.no_grad()
def next_token_probs(prompts, ids, batch_size=16):
    """各プロンプトについて、アシスタントの最初のトークンが ids のどれになるかの確率（ids の中で正規化）"""
    res = []
    for i in range(0, len(prompts), batch_size):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False) for p in prompts[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(judge_llm.device)
        res.append(judge_llm(**enc).logits[:, -1, ids].float().softmax(-1).cpu())
    return torch.cat(res).numpy()

def solve(qs, with_passage):
    """4 択を解かせ、各選択肢を選ぶ確率を返す。位置の偏りを打ち消すため、選択肢を 4 通りに回して並べ、確率を平均する"""
    prompts = []
    for q in qs:
        for r in range(4):
            rotated = q["choices"][r:] + q["choices"][:r]
            listing = "\n".join(f"{i}. {c}" for i, c in enumerate(rotated, 1))
            context = f"次の文章を読んで、質問の答えとして最も適切なものを選び、番号だけを答えてください。\n\n# 文章\n{passages[q['passage']]}\n\n" \
                if with_passage else "次の質問の答えとして最も適切なものを選び、番号だけを答えてください。\n\n"
            prompts.append(f"{context}# 質問\n{q['question']}\n{listing}")
    p = next_token_probs(prompts, digit_ids).reshape(len(qs), 4, 4)        # [問題, 回し方 r, 表示位置 k]
    return np.stack([sum(np.roll(p[n, r], r) for r in range(4)) / 4 for n in range(len(qs))])   # 表示位置 k は元の選択肢 (k + r) % 4

JUDGE = """次の文章だけを根拠にして判断してください。質問に対する答えとして、次の選択肢は正しいですか。「はい」か「いいえ」だけで答えてください。

# 文章
{passage}

# 質問
{question}

# 選択肢
{choice}"""

def judge_each(qs):
    """選択肢を 1 つずつ本文と照らし、「正しい」と判定される確率を返す"""
    prompts = [JUDGE.format(passage=passages[q["passage"]], question=q["question"], choice=c) for q in qs for c in q["choices"]]
    return next_token_probs(prompts, yes_no_ids)[:, 0].reshape(len(qs), 4)

# 捨てる条件。pw は本文ありで各選択肢を選ぶ確率、pn は本文なしの確率、j は各選択肢が本文から見て正しいと判定される確率（いずれも [0] が正解）
THRESHOLD = 0.5
CHECKS = {
    "本文があっても正解できない": lambda pw, pn, j: pw.argmax() != 0,
    "正解が本文から見て正しくない": lambda pw, pn, j: j[0] < THRESHOLD,
    "誤答にも正しいものがある": lambda pw, pn, j: (j[1:] >= THRESHOLD).any(),
    "本文なしで当たる": lambda pw, pn, j: pn[0] >= THRESHOLD,
}
ok = [q for q in quiz if q["reason"] is None]
P_with, P_without, J = solve(ok, True), solve(ok, False), judge_each(ok)      # 37 問の検査で T4 で 2〜3 分ほど
for q, pw, pn, j in zip(ok, P_with, P_without, J):
    q.update(p_with=pw[0], p_without=pn[0], judge=j)
    q["failed"] = [reason for reason, bad in CHECKS.items() if bad(pw, pn, j)]
    q["reason"] = q["failed"][0] if q["failed"] else None      # 最初に引っかかった検査を捨てた理由とする
print(f"GPU メモリの最大 {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

```python
import pandas as pd

# 3. 捨てた理由と残る割合。左は最初に引っかかった検査、右は検査ごとに引っかかった数（1 問が複数に引っかかることがある）
kept = [q for q in quiz if q["reason"] is None]
print(pd.DataFrame({"捨てた理由": pd.Series([q["reason"] or "採用" for q in quiz]).value_counts(),
                    "検査ごと": pd.Series([r for q in ok for r in q["failed"]]).value_counts()}).fillna(0).astype(int).to_string())
print(f"採用 {len(kept)} / {len(quiz)} 問 ({len(kept) / len(quiz):.0%})、問題が 1 つ以上残った記事 {len({q['passage'] for q in kept})} / {len(titles)}")
# 選択肢の形の手がかり: 正解がいちばん長い選択肢である割合（でたらめなら 25%）
longest = lambda qs: np.mean([max(range(4), key=lambda i: len(q["choices"][i])) == 0 for q in qs])
print(f"正解がいちばん長い選択肢: 形式に通った問題 {longest(ok):.0%} / 採用した問題 {longest(kept):.0%}")

def show(q):
    print(f"[{titles[q['passage']]}] {q['question']}\n   正解: {q['choices'][0]} / 誤答: {' / '.join(q['choices'][1:])}\n"
          f"   本文あり {q['p_with']:.2f} / 本文なし {q['p_without']:.2f} / 各選択肢が正しい確率 {[round(float(x), 2) for x in q['judge']]}")

for reason in ["本文なしで当たる", "誤答にも正しいものがある", "正解が本文から見て正しくない", "本文があっても正解できない"]:
    print(f"\n=== 捨てた例: {reason}")
    for q in [q for q in ok if reason in q["failed"]][:2]:
        show(q)
```

```python
# 4. 残った問題を人が見て確かめる（出題前には必ず本文と照らして読む）
for q in kept:
    show(q)
```

- この章の「穴埋め問題（クローズテスト）を自動で作る」は文から語を抜く語彙の問題を作る。質問応答の章の「文書から質問と答えの組を自動で作る（質問生成）」は、文書中の語句を答えに指定して質問を作り、抽出型 QA のモデルで解き直して指定した答えが返るものを残す（選択肢はない）。ここの「本文があっても正解できない」の検査は、その往復検査を 4 択に当てはめたものである。内容の理解を問う多肢選択式では、それに加えて誤答の質（本文を読まなくても当てられないか）と正解が 1 つに決まるかが問題の良し悪しを決めるので、その検査が中心になる。4 択を解かせる部分は、質問応答の章の「多肢選択式の問題を解く」と同じく番号のトークンの確率を比べる方法で、選択肢を 4 通りに回して確率を平均し、位置の偏りを打ち消している。
- この実行では、23 記事の出力から 67 件を取り出し、形式の検査に通ったのは 37 問、最後に残ったのは 13 問（19%、問題が残った記事は 9）だった。いちばん多く引っかかったのは「本文なしで当たる」で、37 問中 22 問。「この文章で「応力」という言葉が指すものは何か」の正解が「応力テンソル」のように問題文の語から推せる問題や、「曜日を読み上げるとき、なぜ「火」「土」が引き伸ばされるか」の正解「他の2モーラ1フットに合わせるため」に対して、誤答が「特別な意味を強調するため」のように的外れな問題（どちらも本文なしで 1.00）がここで落ちる。本文は Wikipedia なので、検査する LLM が記事の内容を覚えていて解ける問題もここに含まれる。世界知識で解ける問題を出題してよいかは目的しだいで、しきい値で調整する。形式の誤りは、誤答を 4 つ書いたもの（23 問。正解と同じ「作業を行う従業員の安全」を誤答にも入れたものもあった）、JSON が壊れたもの（6 件）、誤答が 2 つのもの（1 問）。大規模言語モデルの章の「LLM の出力から JSON を取り出して検証する」のように誤りを伝えて作り直させるか、「制約付きデコーディングで LLM の出力を JSON スキーマ・正規表現・選択肢に従わせる (outlines)」で形を強制すると減らせる。
- 正解の一意性は、選択肢を 1 つずつ本文と照らして「はい」「いいえ」の確率で判定した。「打楽器が重要な役割を果たすジャンルはいくつ挙げられているか」のように数え方で答えが変わる問題は、すべての選択肢が正しいと判定されて落ち、本文にない「426グルデン」を正解にした問題 (0.19) も落ちた。「グーテンベルクがフストに返済を命じられた最終的な金額」の正解「2,026グルデン」は 0.48 で、しきい値のすぐ下で落ちた。本文で 2,026 グルデンはフストが要求した額で、裁判所が命じた額としては書かれていない。しきい値の 0.5 は仮の値で、判定の精度を人の判定と比べて測ってはいない。NLI で照らすこともできるが（Transformers の基本操作の章の「2 つの文の含意関係を判定する (NLI)」）、質問と選択肢を 1 つの文に書き換える必要があり、数値の取り違えにも弱い。
- 残った 13 問を本文と照らして人が読んだところ、正解が正しく誤答が誤りだとはっきりしていたのは 12 問だった。「GNSS の観測データで何を計算できるか」は、本文の「点間の基線長」が正解だが、誤答の「観測点の座標」も相対測位で求められると考える人がいそうである。笑福亭松鶴の考えを問う問題は、正解を「真打制度はお客様が決めるべきもの」と言い換えているが、本文は「真打にふさわしいかどうかはお客様が決めること」で、言い換えで意味が少しずれている。また、ほとんどの問題は本文の語句をそのまま正解にしたもので（天体暦の記事の 3 問は、どれも本文に出てくる名称を問うだけ）、言い換えや推論を要する問題は少なかった。自動の検査は明らかに悪い問題を減らすための足切りで、出題の前には人が読む。
- 選択肢の形の手がかりは残る。正解がいちばん長い選択肢である割合は、形式に通った問題で 78%、採用した問題でも 77% だった（でたらめなら 25%）。「本文なしで当たる」の検査はこれを取り除けていない。誤答の長さや詳しさを正解にそろえるよう指示するか、長さの偏りを検査に加える。
- 問題を作る LLM (sarashina2.2-3b-instruct) と検査する LLM (Qwen3-4B-Instruct-2507) を分けたのは、自分で作った問題を自分で解くと、作ったときと同じ思い込みで正解してしまうおそれがあるため。T4 の 15GB に 2 つは同時に載らないので、作る側を消してから読み込んでいる。GPU メモリの最大は 11.0 GB だった（T4 と同じ float16 で実行した値）。2 つのモデルで合わせて約 15GB をダウンロードする。
- 本文は JSQuAD（JGLUE、ヤフー、CC BY-SA 4.0。元は日本語 Wikipedia）の段落で、作った問題を本文と一緒に配布するなら CC BY-SA 4.0 に従う。sarashina2.2 は MIT、Qwen3-4B-Instruct-2507 は Apache 2.0。sarashina2.2 のトークナイザーの修正は、大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」の注意書きを参照。

## 作文を自動で採点する {essay_scoring}

学習者の作文に、人が付けるのと近い点数を自動で付けたい。手作りの特徴量による回帰と文埋め込みによる回帰（および両者の結合）を比べ、四捨五入と分位点による境界推定の違いも確かめながら、採点者間の一致度と同じ指標 (QWK) で評価する。

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
essays = df[df["essay_set"] == 1].dropna(subset=["domain1_score"]).sample(frac=1, random_state=0).reset_index(drop=True)     # 課題 1（意見文、2〜12 点）
essays["score"] = essays["domain1_score"].astype(int)
print(len(essays), "件の作文を 5 分割交差検証で評価")

def handcrafted(text):
    words = re.findall(r"[A-Za-z']+", text.lower())
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    return [len(words), np.log(len(words) + 1), len(set(words)) / (len(words) + 1), np.mean([len(w) for w in words] or [0]),
            len(sentences), len(words) / (len(sentences) + 1), text.count(","), sum(len(w) >= 8 for w in words) / (len(words) + 1)]

names = ["語数", "log 語数", "異なり語の割合", "平均語長", "文数", "平均文長", "コンマの数", "長い語の割合"]
X = np.array([handcrafted(t) for t in essays["essay"]])
y = essays["score"].to_numpy()
print(pd.Series({n: np.corrcoef(X[:, i], y)[0, 1] for i, n in enumerate(names)}).round(3).to_string())
```

```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

def qwk(y_true, y_pred):
    """重み付き κ 係数 (quadratic weighted kappa): 点数の差が大きい食い違いほど重く罰する。作文採点の標準的な指標"""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
E = embedder.encode(essays["essay"].tolist(), batch_size=32, normalize_embeddings=True, show_progress_bar=False)

alphas = np.logspace(-1, 3, 9)
cv = KFold(5, shuffle=True, random_state=0)
labels = ["語数だけ", "手作りの特徴量", "文埋め込み", "特徴量 + 文埋め込み"]
oof_cont = {k: np.empty(len(y)) for k in labels}
oof_cut = {k: np.empty(len(y), dtype=int) for k in labels}
s_min, s_max = int(y.min()), int(y.max())

# 手作りの特徴量 (X) だけを訓練データで標準化し、L2 正規化済みの文埋め込み (E) はそのまま結合して重みを保つ
for tr, te in cv.split(X):
    X_train, X_test, E_train, E_test, y_train = X[tr], X[te], E[tr], E[te], y[tr]
    scaler = StandardScaler().fit(X_train)
    Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)
    qs = [(y_train <= k).mean() for k in range(int(y_train.min()), int(y_train.max()))]
    feats = {
        "語数だけ": (Xtr_s[:, :2], Xte_s[:, :2]),
        "手作りの特徴量": (Xtr_s, Xte_s),
        "文埋め込み": (E_train, E_test),
        "特徴量 + 文埋め込み": (np.hstack([Xtr_s, E_train]), np.hstack([Xte_s, E_test])),
    }
    for label, (a, b) in feats.items():
        reg = RidgeCV(alphas=alphas).fit(a, y_train)
        pred_test = reg.predict(b)
        oof_cont[label][te] = pred_test
        # 回帰の予測は平均の側に縮むため、訓練データの交差検証予測から正解分布に合う分位点境界を決める
        pred_train = cross_val_predict(RidgeCV(alphas=alphas), a, y_train, cv=cv)
        cuts = np.quantile(pred_train, qs)
        oof_cut[label][te] = np.digitize(pred_test, cuts) + int(y_train.min())

rows = []
for label in labels:
    pred_rint = np.clip(np.rint(oof_cont[label]), s_min, s_max).astype(int)
    rows.append({
        "特徴量": label,
        "QWK（四捨五入）": qwk(y, pred_rint),
        "QWK（分位点）": qwk(y, oof_cut[label]),
        "平均絶対誤差": np.abs(oof_cont[label] - y).mean(),
        "標準偏差（連続）": oof_cont[label].std(),
        "標準偏差（分位点）": oof_cut[label].std(),
    })
print(pd.DataFrame(rows).set_index("特徴量").round(3).to_string())
print(f"正解の標準偏差 = {y.std():.3f}")

human = essays.dropna(subset=["rater1_domain1", "rater2_domain1"])
print("参考: 人の採点者 2 人の間の QWK =", round(qwk(human["rater1_domain1"].astype(int), human["rater2_domain1"].astype(int)), 3))
```

- 作文の長さだけで、点数のかなりの部分が説明できてしまう。新しい手法は、必ず「語数だけ」のベースライン (QWK 0.788) と比べる。長さだけを見るモデルは、意味のない長文に高得点を付けてしまう。
- この課題の点数は 2 人の採点者の点数の合計なので、1 人の採点者どうしの一致度 (QWK 0.704) より、モデルと合計点の一致度のほうが高く出る。人と同等以上に見えても、比べている対象が違う点に注意する。
- 特徴量と文埋め込みを混ぜるときは、スケールの揃え方に注意する。手作りの特徴量 (X) と L2 正規化済みの文埋め込み (E) をまとめて `StandardScaler` に掛けると、384 次元の埋め込みすべての分散が 1 に引き伸ばされて（合計分散が約 384 対 8 になり）、手作りの特徴量が埋もれてしまう（さらに列単位の標準化なので、結合前に定数倍しても打ち消される）。ここでは `X_train` だけを `StandardScaler` で標準化し、`E` は L2 正規化したまま結合して `RidgeCV` で正則化の強さを選ぶことで、「特徴量 + 文埋め込み」（QWK 0.839、平均絶対誤差 0.602）が「手作りの特徴量」（QWK 0.834、平均絶対誤差 0.608）と「文埋め込み」（QWK 0.636、平均絶対誤差 0.905）の両方を上回った。
- 回帰の予測値は平均の側へ縮むため（たとえば「文埋め込み」では連続予測の標準偏差が 0.980 と正解の 1.503 よりかなり小さい）、単純に `np.rint` で四捨五入すると中央の点数に偏りやすい。訓練データ内での交差検証予測を使い、訓練データの正解分布と同じ累積割合の分位点を `np.quantile` で求めて `np.digitize` で離散化すると、予測の散らばりが正解分布に近づき、すべての手法で QWK が上がった（特に縮みの大きい「文埋め込み」では 0.590 から 0.636 へ +0.046 改善）。
- 評価には QWK（重み付き κ）を使う。人の採点者どうしの一致度と同じ物差しなので、「人と同じくらい一致している」かどうかを直接比べられる。
- 課題（プロンプト）ごとに採点基準も点数の範囲も違う。ある課題で学習したモデルは、別の課題にはそのまま使えない。
- 自動採点は、内容の正しさや論理の一貫性をほとんど見ていない。重要な判定に使うときは、人の採点と併用し、極端な入力（丸写し、無関係な文章）への挙動を確かめる。
- 日本語学習者の作文では、文法誤りの数、語彙の難易度（この章の語彙の難しさのレシピ）、接続表現の使い方などが特徴量としてよく使われる。観点別（内容・構成・言語）に採点する方法は、次の「ルーブリック（観点別の採点基準）に基づいて作文を採点する」を参照。

## ルーブリック（観点別の採点基準）に基づいて作文を採点する {rubric_scoring}

総合点 1 スカラーだけでなく、「内容・論点」「構成・論理展開」「表現・語彙」「文法・表記」といった複数観点のルーブリックに基づいて作文を採点し、観点ごとの強みと改善点のフィードバックまで生成したい。ASAP データセットのうち 4 観点のアナリティック採点が付いた課題 7 を用いて、観点別採点と総合点予測を比較する。

```python
!pip install -q datasets sentence-transformers wordfreq
```

```python
import re
import numpy as np
import pandas as pd
from datasets import load_dataset
from wordfreq import zipf_frequency

df = load_dataset("nlpatunt/D_ASAP-AES", split="train").to_pandas()
es7 = df[df["essay_set"] == 7].dropna(subset=["domain1_score", "rater1_trait1", "rater2_trait1"]).reset_index(drop=True)

# ASAP 課題 7 の 4 観点ルーブリック（各採点者 0〜3 点、2 人の合計 0〜6 点。4 観点の合計が domain1_score = 0〜24 点）
TRAITS = [
    ("trait1", "内容・論点 (Ideas)"),
    ("trait2", "構成・展開 (Organization)"),
    ("trait3", "表現・語彙 (Style)"),
    ("trait4", "文法・表記 (Conventions)"),
]
for k, _ in TRAITS:
    es7[k] = (es7[f"rater1_{k}"] + es7[f"rater2_{k}"]).astype(int)
es7["total"] = es7["domain1_score"].astype(int)
print(f"課題 7 の作文数: {len(es7)} 件 / 総合点の範囲: {es7['total'].min()}〜{es7['total'].max()}")
```

```python
# 観点ごとの特徴量を設計する: 分量・段落/接続表現・語彙の豊かさと頻度・綴り/句読点の正確さ
CONNECTIVES = {
    "however", "therefore", "moreover", "furthermore", "although", "because", "since",
    "while", "then", "next", "finally", "first", "second", "consequently", "meanwhile", "instead",
}

def rubric_features(text):
    words = re.findall(r"[A-Za-z']+", text)
    lower = [w.lower() for w in words]
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    paras = [p.strip() for p in text.splitlines() if p.strip()]
    nw = max(len(lower), 1)
    zipfs = [zipf_frequency(w, "en") for w in lower if w.isalpha()]
    return {
        "log_words": np.log1p(len(words)),
        "n_sents": len(sents),
        "avg_sent_len": len(words) / max(len(sents), 1),
        "n_paras": len(paras),
        "conn_rate": sum(w in CONNECTIVES for w in lower) / max(len(sents), 1),
        "ttr": len(set(lower)) / nw,
        "avg_word_len": float(np.mean([len(w) for w in lower] or [0.0])),
        "rare_rate": sum(0 < z < 4.5 for z in zipfs) / max(len(zipfs), 1),
        "oov_rate": sum(z == 0 for z in zipfs) / max(len(zipfs), 1),
        "comma_rate": text.count(",") / max(len(sents), 1),
        "cap_start_rate": sum(bool(re.match(r"^[A-Z@]", s)) for s in sents) / max(len(sents), 1),
    }

feat_df = pd.DataFrame([rubric_features(t) for t in es7["essay"]])
corr_table = pd.DataFrame({name: [feat_df[c].corr(es7[k]) for c in feat_df.columns] for k, name in TRAITS}, index=feat_df.columns)
print("=== 各特徴量と人手評点（観点別）の相関係数 ===")
print(corr_table.round(3).to_string())
```

```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

embedder = SentenceTransformer("all-MiniLM-L6-v2")
E = embedder.encode(es7["essay"].tolist(), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
X = feat_df.to_numpy()
cv = KFold(5, shuffle=True, random_state=0)
alphas = np.logspace(-1, 3, 9)
qwk = lambda yt, yp: cohen_kappa_score(yt, yp, weights="quadratic")

oof_len = {k: np.empty(len(es7), dtype=int) for k, _ in TRAITS}
oof_feat = {k: np.empty(len(es7), dtype=int) for k, _ in TRAITS}
oof_all = {k: np.empty(len(es7), dtype=int) for k, _ in TRAITS}
oof_all_cont = {k: np.empty(len(es7)) for k, _ in TRAITS}
oof_direct = np.empty(len(es7), dtype=int)

for tr, te in cv.split(X):
    scaler = StandardScaler().fit(X[tr])
    Xtr_s, Xte_s = scaler.transform(X[tr]), scaler.transform(X[te])
    Xtr_all, Xte_all = np.hstack([Xtr_s, E[tr]]), np.hstack([Xte_s, E[te]])
    for k, _ in TRAITS:
        y_tr = es7[k].to_numpy()[tr]
        oof_len[k][te] = np.clip(np.rint(RidgeCV(alphas=alphas).fit(Xtr_s[:, :1], y_tr).predict(Xte_s[:, :1])), 0, 6).astype(int)
        oof_feat[k][te] = np.clip(np.rint(RidgeCV(alphas=alphas).fit(Xtr_s, y_tr).predict(Xte_s)), 0, 6).astype(int)
        pred = RidgeCV(alphas=alphas).fit(Xtr_all, y_tr).predict(Xte_all)
        oof_all_cont[k][te] = pred
        oof_all[k][te] = np.clip(np.rint(pred), 0, 6).astype(int)
    oof_direct[te] = np.clip(np.rint(RidgeCV(alphas=alphas).fit(Xtr_all, es7["total"].to_numpy()[tr]).predict(Xte_all)), 0, 24).astype(int)

rows = []
for k, name in TRAITS:
    rows.append({
        "観点": name,
        "語数のみ": qwk(es7[k], oof_len[k]),
        "設計特徴量": qwk(es7[k], oof_feat[k]),
        "特徴量+埋め込み": qwk(es7[k], oof_all[k]),
        "採点者間QWK": qwk(es7[f"rater1_{k}"].astype(int), es7[f"rater2_{k}"].astype(int)),
    })
print("\n=== 観点別の採点精度 (QWK) ===")
print(pd.DataFrame(rows).set_index("観点").round(3).to_string())

sum_pred = np.clip(np.rint(sum(oof_all_cont[k] for k, _ in TRAITS)), 0, 24).astype(int)
human_tot = qwk(es7["rater1_domain1"].astype(int), es7["rater2_domain1"].astype(int))
print(f"\n総合点 QWK: 直接予測={qwk(es7['total'], oof_direct):.3f} / 4 観点合算={qwk(es7['total'], sum_pred):.3f} (参考: 採点者間={human_tot:.3f})")
```

```python
# 予測された観点別スコアに基づくフィードバック生成（強みと改善すべき観点を提示）
es7["pred_sum"] = sum_pred
for k, _ in TRAITS:
    es7[f"pred_{k}"] = oof_all[k]
es7["gap"] = es7["pred_trait1"] - es7["pred_trait4"]

ADVICE = {
    "trait1": ("具体的な体験や描写がよく書けています。", "話題に沿った具体例や背景の説明をもう少し補いましょう。"),
    "trait2": ("話の流れが整理され、まとまりがあります。", "場面の切り替わりや接続表現を意識して段落構成を整えましょう。"),
    "trait3": ("状況が伝わる表現や語彙の工夫が見られます。", "同じ語の繰り返しを避け、具体的な形容詞・動詞を増やしましょう。"),
    "trait4": ("句読点や綴り・文法の誤りが少なく読みやすいです。", "文の区切り（ピリオド）やスペル・大文字小文字を見直しましょう。"),
}

samples = pd.concat([es7.nlargest(1, "gap"), es7.nsmallest(1, "gap"), es7.nlargest(1, "pred_sum")])
for _, row in samples.iterrows():
    breakdown = ", ".join(f"{name.split()[0]}={row[f'pred_{k}']}/6(正解{row[k]})" for k, name in TRAITS)
    best_k = max(TRAITS, key=lambda x: row[f"pred_{x[0]}"])[0]
    weak_k = min(TRAITS, key=lambda x: row[f"pred_{x[0]}"])[0]
    print(f"\n[総合予測 {row['pred_sum']}/24 (正解 {row['total']})] {breakdown}")
    print("  冒頭:", row["essay"][:90].replace("\n", " "))
    print("  強み:", ADVICE[best_k][0])
    print("  改善:", "すべての観点で十分な基準を満たしています。" if row[f"pred_{weak_k}"] >= 5 else ADVICE[weak_k][1])
```

- 観点別に予測すると、単なる「語数だけ」のベースラインに対して、文埋め込みを加えたモデルは「内容・論点 (Ideas)」で特に大きく伸び (0.554 → 0.757)、綴りミス率 (`oov_rate`) や平均文長（句読点の欠落で長くなる）などを含む設計特徴量は「表現・語彙 (Style)」(0.588 → 0.707) と「文法・表記 (Conventions)」(0.489 → 0.660) を大きく改善する。
- 4 観点の連続予測を合算した総合点 QWK (0.813) は、総合点を直接予測した QWK (0.814) とほぼ同等でありながら、「内容は書けているが句読点・綴りが弱い」「文法は正確だが具体例が不足している」といった学習者ごとの指導ポイントを定量的に示せる。
- 前のレシピ「作文を自動で採点する」と同様、モデルが比べている正解は 2 人の採点者の合計点（0〜6 点・0〜24 点）であるため、1 人の採点者同士の一致度より見かけ上の QWK が高く出る点に注意する。また、文脈の誤りや論理の一貫性を細かく指摘するには、文法誤り訂正と文章校正の章の「訂正にフィードバックコメントを付ける」のような生成モデルとの併用が有効。
