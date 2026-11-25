# 意味解析

「誰が何をどうした」「それは何を指すか」「2 つの文はどうつながるか」を取り出す。構文の一歩先の解析。関係の抽出では、学習済みモデルの利用と、関係分類器の学習を扱う。章の後半では、含意関係認識の分類器の学習・仮説文だけで学習したモデルで分かるデータの偏りを扱い、最後にこれらの部品を組み合わせてニュースから出来事を抜き出して時間順に並べるイベント抽出（小さな LLM による JSON 抽出との比較）に取り組む。語義の判定は「語の意味と意味の変化」の章、文を一階述語論理の式に変換するレシピは付録の「仕組みを理解するための実装」にある。開発サイクルの設計や遠距離教師あり学習による関係抽出データの構築については、姉妹編『[自然言語処理の教科書](https://github.com/mamorlis/nlpbook)』の第 1 章も参照。

## 日本語の述語項構造解析をする (KWJA) {pas_kwja}

各述語について、ガ格・ヲ格・ニ格に当たる語を求めたい。省略された主語や目的語（ゼロ代名詞）も補いたい。

```python
!pip install -q kwja
```

```python
import subprocess
from rhoknp import Document

def kwja(text, model_size="base"):
    """KWJA を実行して、解析結果を rhoknp の Document として返す (初回はモデルのダウンロードに数分かかる)"""
    result = subprocess.run(["kwja", "--text", text, "--tasks", "char,word", "--model-size", model_size],
                            capture_output=True, text=True, check=True)
    return Document.from_knp(result.stdout)

doc = kwja("太郎は書店で新しい辞書を買った。家に帰ってから、花子に見せた。")
for sent in doc.sentences:
    print(sent.text)
    for bp in sent.base_phrases:
        args = bp.pas.get_all_arguments(relax=False) if bp.pas else {}
        if args:
            print("  述語:", bp.text, {case: [str(a) for a in arguments] for case, arguments in args.items()})
```

```python
# 項の種類を見る: 係り受けがある項 / 文内・文間のゼロ照応 / 文章の外を指す外界照応 (著者・読者・不特定の人)
from rhoknp.cohesion import EndophoraArgument, ExophoraArgument

for bp in doc.base_phrases:
    if not bp.pas:
        continue
    for case, arguments in bp.pas.get_all_arguments(relax=False).items():
        for a in arguments:
            if isinstance(a, EndophoraArgument):
                where = "同じ文" if a.base_phrase.sentence.sid == bp.sentence.sid else "前の文"
                print(f"{bp.text} の{case}格 = {a.base_phrase.text} ({a.type.value}, {where})")
            elif isinstance(a, ExophoraArgument):
                print(f"{bp.text} の{case}格 = {a.exophora_referent.text} (外界照応)")
```

- 2 文目の「帰って」「見せた」には主語が書かれていないが、前の文の「太郎」がガ格として補われる。これがゼロ照応解析。
- KWJA は京都大学で開発された統合的な日本語解析器（BERT 系のエンコーダを使う）。形態素解析から、固有表現、係り受け、述語項構造、共参照、談話関係までを一度に行う。
- `--model-size` は tiny / base / large。base 以上は CPU だと 1 文に数秒かかる。大量に処理するなら GPU を使う。
- インストール時に Transformers のバージョンが入れ替わることがある。import でエラーになったらランタイムを再起動する。

## FrameNet と PropBank で述語の意味フレームと意味役割を調べる {framenet_propbank}

動詞 `buy` がどんな参加者（買い手、商品、売り手…）を取るのかを、言語資源から引きたい。意味役割付与のラベル体系を理解したい。

```python
import nltk
nltk.download("framenet_v17", quiet=True)
nltk.download("propbank", quiet=True)
from nltk.corpus import framenet as fn

frame = fn.frame("Commerce_buy")
print(frame.name, "-", frame.definition[:160])
print("中核的なフレーム要素:", [name for name, fe in frame.FE.items() if fe.coreType == "Core"])
print("周辺的なフレーム要素:", [name for name, fe in frame.FE.items() if fe.coreType != "Core"][:8])
print("このフレームを喚起する語:", sorted(frame.lexUnit)[:8])
```

```python
# 単語からフレームを探す / フレーム間の関係をたどる
print([f.name for f in fn.frames_by_lemma(r"(?i)\bsell\b")])
print([f"{r.superFrameName} → {r.subFrameName} ({r.type.name})" for r in frame.frameRelations][:4])

# 注釈付きの例文: どの部分がどのフレーム要素か
lu = [l for l in fn.lus(r"(?i)^buy\.v$") if l.frame.name == "Commerce_buy"][0]
for ex in lu.exemplars[:3]:
    print(ex.text.strip())
    print("   ", [(ex.text[s:e], role) for s, e, role in ex.FE[0]])
```

```python
# PropBank: 動詞の語義 (roleset) ごとに、番号付きの項 (Arg0, Arg1, ...) を定義する
from nltk.corpus import propbank

for roleset_id in ["buy.01", "sell.01", "give.01"]:
    rs = propbank.roleset(roleset_id)
    print(roleset_id, rs.attrib["name"], {f"Arg{r.attrib['n']}": r.attrib["descr"] for r in rs.findall("roles/role")})
```

- FrameNet は「場面」（フレーム）を単位に、場面ごとの役割（Buyer、Goods、Seller）を定義する。`buy` と `sell` は同じ商取引の場面を別の視点から述べたものとして関連付けられる。
- PropBank は動詞ごとに Arg0（動作主的な項）、Arg1（対象的な項）… と番号を振る。ラベルが少なく一貫しているので、意味役割付与の学習データとして広く使われる。
- 日本語には、日本語フレームネット、京都大学格フレーム、NPCMJ の PropBank 風アノテーションなどがある。

## 日本語の共参照解析をする (KWJA) {coref_kwja}

「彼」「その会社」「同社」が、文章中のどの語と同じものを指しているのかを知りたい。

```python
!pip install -q kwja
```

```python
import subprocess
from rhoknp import Document

def kwja(text, model_size="base"):
    result = subprocess.run(["kwja", "--text", text, "--tasks", "char,word", "--model-size", model_size],
                            capture_output=True, text=True, check=True)
    return Document.from_knp(result.stdout)

text = "山田社長は新製品を発表した。彼はこの製品が同社の主力になると述べた。山田氏によると、発売は来年の春だという。"
doc = kwja(text)

# 同じものを指す基本句をまとめる (共参照クラスタ)
clusters, seen = [], set()
for bp in doc.base_phrases:
    if bp.global_index in seen:
        continue
    group = [bp] + list(bp.get_coreferents())
    if len(group) > 1:
        seen.update(g.global_index for g in group)
        clusters.append(sorted(group, key=lambda g: g.global_index))
for i, group in enumerate(clusters):
    print(i, [(g.text, f"{g.sentence.sid.split('-')[-1]} 文目") for g in group])
```

```python
# 固有表現と、各基本句の意味的な素性もまとめて得られる
print([(ne.text, ne.category.value) for ne in doc.named_entities])
for bp in doc.sentences[0].base_phrases:
    print(bp.text, [k for k in bp.features if k in ("体言", "用言", "SM-主体", "時制", "節-主辞")], bp.features.get("時制", ""))
```

- 日本語は代名詞を使わずに省略することが多い。省略された項の補完は、前項の述語項構造解析（ゼロ照応）で扱われる。共参照解析は、書かれている語同士の同一性を扱う。
- 「同社」のように、指示先が文中にはっきり書かれていない場合（ここでは「山田社長」の会社）は解決できない。橋渡し照応と呼ばれる、より難しい問題。
- 精度は文体に依存する。ニュースや Wikipedia のような整った文に強く、会話や SNS では落ちる。

## 英語の共参照解析をして代名詞を先行詞に置き換える (Stanza) {coref_en}

英文中の `he` `it` `the company` が何を指しているかを解決したい。情報抽出や要約の前処理として、代名詞を具体的な名前に置き換えたい。

```python
!pip install -q stanza
```

```python
import stanza

stanza.download("en", processors="tokenize,coref", verbose=False)
nlp = stanza.Pipeline("en", processors="tokenize,coref", verbose=False)   # 多言語の XLM-R large を使うモデル (約 2GB)

text = ("Barack Obama was born in Hawaii. He was elected president in 2008, and his wife Michelle joined him in Washington. "
        "Apple unveiled a new phone on Tuesday. The company said it would go on sale next month.")
doc = nlp(text)

for chain in doc.coref:
    mentions = [" ".join(w.text for w in doc.sentences[m.sentence].words[m.start_word:m.end_word]) for m in chain.mentions]
    if len(mentions) > 1:
        print(f"{chain.representative_text!r:16} ← {mentions}")
```

```python
PRONOUNS = {"he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their"}
POSSESSIVE = {"his", "her", "its", "their"}

def resolve(doc):
    """代名詞の言及を、その連鎖の代表的な言及 (最も具体的な表現) に置き換える"""
    replace = {}
    for chain in doc.coref:
        for m in chain.mentions:
            words = doc.sentences[m.sentence].words[m.start_word:m.end_word]
            if len(words) == 1 and words[0].text.lower() in PRONOUNS:
                suffix = "'s" if words[0].text.lower() in POSSESSIVE else ""
                replace[(m.sentence, m.start_word)] = chain.representative_text + suffix
    out = []
    for si, sent in enumerate(doc.sentences):
        tokens = [replace.get((si, wi), w.text) for wi, w in enumerate(sent.words)]
        out.append(" ".join(tokens).replace(" .", ".").replace(" ,", ","))
    return " ".join(out)

print(resolve(doc))
```

- 「Apple」と「The company」のように、代名詞だけでなく名詞句同士（言い換え）の同一性も解決される。`it` は Apple ではなく、発売される「a new phone」に正しく結び付けられ、そのまま置き換えられている。
- `her` は目的格と所有格が同形なので、この簡易な置換では区別できない。品詞解析 (`pos`) を併用すると判別できる。
- 置換後の文は不自然になることがある。人が読むためではなく、関係抽出や検索の前処理として使う。
- Stanza の共参照モデルは英語のほか、ドイツ語、フランス語、スペイン語など多数の言語に対応している（日本語は未対応なので、前項の KWJA を使う）。

## 代名詞が何を指すかを常識で判断させる (Winograd Schema Challenge) {winograd_ja gpu}

「狙撃兵はテロリストを撃った。彼は悪漢だったからだ。」の「彼」が誰を指すかのように、文法だけでは決まらず、常識が必要な照応の問題をモデルに解かせたい。

```python
import re, urllib.request
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

url = "https://raw.githubusercontent.com/ku-nlp/Winograd-Schema-Challenge-Ja/master/test.txt"
blocks = urllib.request.urlopen(url).read().decode("utf-8").strip().split("\n\t\t\n")
problems = []
for block in blocks:
    lines = [line.split("\t") for line in block.split("\n")]
    if len(lines) >= 4:
        candidates = [c.strip() for c in re.split(r"[、,，]", lines[2][1]) if c.strip()]
        problems.append({"text": lines[0][1], "pronoun": lines[1][1], "candidates": candidates, "answer": lines[3][1].strip()})
problems = [p for p in problems if len(p["candidates"]) == 2 and p["answer"] in p["candidates"]]
print(len(problems), "問"); print(problems[0]); print(problems[1])
```

```python
name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()
ONE_TWO = [tokenizer(c, add_special_tokens=False)["input_ids"][-1] for c in ["1", "2"]]

@torch.no_grad()
def p_first(text, pronoun, a, b):
    prompt = (f"次の文の「{pronoun}」が指しているのはどちらですか。番号だけで答えてください。\n\n文: {text}\n1. {a}\n2. {b}")
    chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    ids = tokenizer(chat, add_special_tokens=False, return_tensors="pt")["input_ids"].to(llm.device)
    return float(llm(ids).logits[0, -1, ONE_TWO].float().softmax(-1)[0])

def solve(p):
    """選択肢の順序を入れ替えて 2 回聞き、平均する（1 番目を選びやすい、といった偏りを打ち消す）"""
    a, b = p["candidates"]
    score = (p_first(p["text"], p["pronoun"], a, b) + 1 - p_first(p["text"], p["pronoun"], b, a)) / 2
    return a if score > 0.5 else b

sample = problems[:200]
df = pd.DataFrame(sample)
df["prediction"] = [solve(p) for p in sample]   # 1 問に 2 回ずつ計算する。T4 なら 200 問で 1〜2 分
df["correct"] = df["prediction"] == df["answer"]
print("正解率:", round(df["correct"].mean(), 3), "(でたらめなら 0.5)")
```

```python
# このデータは 2 文で 1 組（同じ登場人物で、1 語か 2 語だけ違い、答えが入れ替わる）。組の両方に正解できた割合を見る
pair_ok = [df["correct"][i] and df["correct"][i + 1] for i in range(0, len(df) - 1, 2)]
print("組の両方に正解:", round(np.mean(pair_ok), 3), "/ 常に同じ側を答えているだけの組:",
      round(np.mean([df["prediction"][i] == df["prediction"][i + 1] for i in range(0, len(df) - 1, 2)]), 3))
print("先に出てくる候補を選んだ割合:", round(np.mean([p == c[0] for p, c in zip(df["prediction"], df["candidates"])]), 3))
for r in df[~df["correct"]].head(5).itertuples():
    print(f"× {r.text}\n     「{r.pronoun}」→ 予測: {r.prediction} / 正解: {r.answer}")
```

- Winograd Schema Challenge は、文法的にはどちらの候補も先行詞になれるが、常識を使えば一方に決まる問題を集めたもの。1 語を変えると答えが入れ替わる 2 文が組になっている。
- 組になっていることが重要。片方だけ正解しても、「主語を選びやすい」「先に出たほうを選びやすい」といった癖で当たっただけかもしれない。組の両方に正解した割合のほうが、実力をよく表す。
- 日本語版 (JWSC) は英語からの翻訳で作られている。日本語ではゼロ代名詞が自然な場面でも「彼」「それ」が使われるので、やや不自然な文が含まれる。
- 事前学習済みモデルの登場で、この種の問題の正解率は大きく上がった。ただし、Web 上の類似文を覚えているだけの可能性や、データの偏りを使っている可能性が指摘されており、改良版（WinoGrande など）が作られている。
- 照応解析の一般的な方法は、この章の「日本語の共参照解析をする (KWJA)」を参照。
- sarashina2.2 の動かし方とトークナイザーの修正については、大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」を参照。

## 文から関係の三つ組を抽出する (REBEL) {relation_extraction}

文章から「（夏目漱石, 職業, 小説家）」のような（主語, 関係, 目的語）の三つ組を取り出して、知識グラフを作りたい。

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

name = "Babelscape/rebel-large"   # Wikipedia と Wikidata の対応から学習した、関係抽出専用の系列変換モデル (BART)
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSeq2SeqLM.from_pretrained(name).eval()

def parse(decoded):
    """モデルの出力 '<triplet> 主語 <subj> 目的語 <obj> 関係 <subj> 目的語 <obj> 関係 <triplet> ...' を三つ組のリストにする"""
    triples, head, tail, rel, state = [], "", "", "", None
    for tok in decoded.replace("<s>", "").replace("</s>", "").replace("<pad>", "").split():
        if tok == "<triplet>":
            if head and tail and rel:
                triples.append((head.strip(), rel.strip(), tail.strip()))
            head, tail, rel, state = "", "", "", "head"
        elif tok == "<subj>":
            if head and tail and rel:
                triples.append((head.strip(), rel.strip(), tail.strip()))
            tail, rel, state = "", "", "tail"
        elif tok == "<obj>":
            rel, state = "", "rel"
        elif state == "head":
            head += " " + tok
        elif state == "tail":
            tail += " " + tok
        elif state == "rel":
            rel += " " + tok
    if head and tail and rel:
        triples.append((head.strip(), rel.strip(), tail.strip()))
    return triples

def extract(text, num_beams=4, num_return_sequences=3):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = model.generate(**enc, max_length=200, num_beams=num_beams, num_return_sequences=num_return_sequences)
    triples = []
    for seq in tokenizer.batch_decode(out, skip_special_tokens=False):   # 上位の候補を合わせて再現率を上げる
        triples += [t for t in parse(seq) if t not in triples]
    return triples

texts = ["Natsume Soseki was a Japanese novelist born in Tokyo in 1867. He wrote the novel Kokoro.",
         "Toyota Motor Corporation is a Japanese automobile manufacturer headquartered in Toyota City, Aichi. It was founded by Kiichiro Toyoda in 1937."]
for text in texts:
    print(text)
    for t in extract(text):
        print("   ", t)
```

```python
import networkx as nx

G = nx.DiGraph()
for text in texts:
    for head, rel, tail in extract(text):
        G.add_edge(head, tail, relation=rel)
print(G.number_of_nodes(), "ノード", G.number_of_edges(), "エッジ")
print([(h, t, d["relation"]) for h, t, d in G.edges(data=True) if "Toyota" in h][:5])
```

- 関係の種類は Wikidata のプロパティ（約 220 種類）に限られる。それ以外の関係や、ドメイン固有の関係は取れない。
- 文に書かれていない関係を「知識」から出力してしまうことがある。抽出した主語・目的語が原文に出現するかで検証するとよい。
- 多言語版の mREBEL もあるが、日本語の精度は英語に及ばない。日本語では、係り受けと固有表現を組み合わせた規則（構文解析の章の三つ組抽出）や、この章の最初の KWJA による述語項構造解析が現実的な選択肢になる。

## 2 つのエンティティの関係を分類する BERT を学習する（エンティティの印、SemEval-2010 Task 8） {relation_classification gpu}

文中の 2 つの名詞の間の関係（「原因と結果」「部分と全体」「中身と容器」など）を、自分のラベルで分類するモデルを学習したい。エンティティを `<e1>…</e1>` などの印で囲み、BERT をファインチューニングする。

```python
import re
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

# SemEval-2010 Task 8（Hendrickx ら, 2010）。CC BY 3.0。学習 8000 文、評価 2717 文、名詞の組ごとに 9 種類の関係 × 方向 2 つ + Other
ds = load_dataset("SemEvalWorkshop/sem_eval_2010_task_8")
labels = ds["train"].features["relation"].names
print(len(labels), "ラベル:", labels[:4], "...", labels[-1])
for ex in ds["train"].select(range(3)):
    print(labels[ex["relation"]], "|", ex["sentence"])
counts = pd.DataFrame({sp: pd.Series([labels[r] for r in ds[sp]["relation"]]).value_counts() for sp in ["train", "test"]})
counts = counts.sort_values("test", ascending=False)   # ラベルごとの件数（学習と評価）
print(counts.head(3).to_string(), "\n...\n", counts.tail(3).to_string(), sep="")
```

```python
from torch import nn
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

model_name = "google-bert/bert-base-uncased"   # Apache-2.0
MARKERS = ["<e1>", "</e1>", "<e2>", "</e2>"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_tokens(MARKERS, special_tokens=True)   # 印を 1 トークンとして語彙に足す（分割も小文字化もされない）
print(tokenizer.tokenize("The <e1>song</e1> was composed for a famous Brazilian <e2>musician</e2>."))

def encode(batch, marker):
    if marker:
        texts = batch["sentence"]
    else:   # 印なし: 印を消して文だけを入れる（どの 2 語の関係を聞いているのか、モデルには分からない）
        texts = [re.sub(r"</?e[12]>", "", s) for s in batch["sentence"]]
    enc = tokenizer(texts, truncation=True, max_length=128)   # 印が 128 トークンより後ろにあると切れて見つからず、下の index で止まる。長い文では増やす
    ids = tokenizer.convert_tokens_to_ids(["<e1>", "<e2>"])
    # <e1> と <e2> の位置（印なしのときは [CLS] の位置 0 にしておく。使わない）
    enc["e1_pos"] = [x.index(ids[0]) if marker else 0 for x in enc["input_ids"]]
    enc["e2_pos"] = [x.index(ids[1]) if marker else 0 for x in enc["input_ids"]]
    enc["labels"] = batch["relation"]
    return enc

class RelationClassifier(nn.Module):
    """BERT + 線形層。pool="cls" は [CLS] のベクトル、"entity" は <e1> と <e2> の位置のベクトルをつないで分類する"""
    def __init__(self, pool):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.resize_token_embeddings(len(tokenizer))   # 足した印の埋め込みを用意する（乱数で初期化され、学習で決まる）
        self.pool = pool
        hidden = self.bert.config.hidden_size * (2 if pool == "entity" else 1)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, len(labels)))

    def forward(self, input_ids, attention_mask, e1_pos, e2_pos, labels=None, token_type_ids=None):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pool == "entity":
            rows = torch.arange(len(h), device=h.device)
            vec = torch.cat([h[rows, e1_pos], h[rows, e2_pos]], dim=-1)
        else:
            vec = h[:, 0]
        logits = self.head(vec)
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}
```

```python
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う

def train(marker, pool, epochs=4):
    set_seed(0)
    data = ds.map(lambda b: encode(b, marker), batched=True, remove_columns=["sentence", "relation"])
    model = RelationClassifier(pool)
    args = TrainingArguments(
        output_dir="re-out", num_train_epochs=epochs, learning_rate=5e-5, warmup_steps=100, weight_decay=0.01,
        per_device_train_batch_size=32, per_device_eval_batch_size=128, save_strategy="no",
        bf16=BF16, fp16=not BF16, logging_steps=100, report_to="none", seed=0)
    trainer = Trainer(model=model, args=args, train_dataset=data["train"], data_collator=DataCollatorWithPadding(tokenizer))
    trainer.train()
    pred = trainer.predict(data["test"]).predictions.argmax(-1)
    return model, pred

gold = np.array(ds["test"]["relation"])
results = {}
for name, marker, pool in [("印なし + [CLS]", False, "cls"), ("印あり + [CLS]", True, "cls"), ("印あり + 印の位置", True, "entity")]:
    model, pred = train(marker, pool)   # T4 で 1 つあたり 4〜6 分の見込み
    results[name] = pred
    print(name, "正解率:", round(float((pred == gold).mean()), 3))
print(f"GPU メモリの最大: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

```python
from sklearn.metrics import classification_report

rel_type = np.array([l.split("(")[0] for l in labels])   # "Cause-Effect(e1,e2)" → "Cause-Effect"
TYPES = [t for t in dict.fromkeys(rel_type) if t != "Other"]

def official_f1(gold, pred):
    """SemEval-2010 Task 8 の公式のスコアラー（Perl）と同じ計算: 9 種類の関係ごとの F1 の平均（Other は平均に入れない）。
    方向を間違えたら不正解とし、精度は「その関係と予測した数」、再現率は「正解がその関係の数」で割る"""
    f1s = []
    for t in TYPES:
        tp = np.sum((pred == gold) & (rel_type[gold] == t))
        p, r = tp / max(np.sum(rel_type[pred] == t), 1), tp / max(np.sum(rel_type[gold] == t), 1)
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))

rng = np.random.default_rng(0)
boot = [rng.integers(0, len(gold), len(gold)) for _ in range(1000)]   # 評価データを復元抽出で作り直す（ブートストラップ）
for name, pred in results.items():
    scores = [official_f1(gold[i], pred[i]) for i in boot]
    print(f"{name}: 公式の F1 {official_f1(gold, pred):.3f}（95% 信頼区間 {np.percentile(scores, 2.5):.3f}〜{np.percentile(scores, 97.5):.3f}）")
a, b = results["印あり + [CLS]"], results["印あり + 印の位置"]
diff = [official_f1(gold[i], b[i]) - official_f1(gold[i], a[i]) for i in boot]   # 同じ復元抽出で差をとる（対応のあるブートストラップ）
print(f"印の位置 − [CLS] の差: {official_f1(gold, b) - official_f1(gold, a):+.3f}（95% 信頼区間 {np.percentile(diff, 2.5):+.3f}〜{np.percentile(diff, 97.5):+.3f}）\n")
pred = results["印あり + 印の位置"]
print(classification_report(gold, pred, labels=range(len(labels)), target_names=labels, digits=3, zero_division=0))
```

```python
# 方向をまとめた 10 種類で混同行列を見る。対角線の数には、方向を間違えたものも含む
short = {t: "".join(w[0] for w in t.split("-")) for t in dict.fromkeys(rel_type)}   # "Cause-Effect" → "CE"、"Other" → "O"
cm = pd.crosstab(pd.Series([short[t] for t in rel_type[gold]], name="正解"), pd.Series([short[t] for t in rel_type[pred]], name="予測"))
print(cm.to_string())
same_type = rel_type[gold] == rel_type[pred]
print("関係の種類は合っていて方向だけ違う:", int(np.sum(same_type & (gold != pred))), "件 /",
      "Other を関係ありと予測:", int(np.sum((rel_type[gold] == "Other") & (rel_type[pred] != "Other"))), "件 /",
      "関係ありを Other と予測:", int(np.sum((rel_type[gold] != "Other") & (rel_type[pred] == "Other"))), "件")
```

```python
# 新しい文で予測する。model はループで最後に学習したもの（印あり + 印の位置）。SemEval では先に出る名詞が常に e1 なので、同じ関係でも語順によって向きが変わる
model.eval()

@torch.no_grad()
def classify(sentence):
    enc = encode({"sentence": [sentence], "relation": [0]}, marker=True)
    batch = {k: torch.tensor(v).to(model.bert.device) for k, v in enc.items() if k != "labels"}
    prob = model(**batch)["logits"][0].float().softmax(-1)
    return labels[int(prob.argmax())], round(float(prob.max()), 2)

for s in ["The <e1>fire</e1> was caused by a faulty <e2>heater</e2>.",
          "A faulty <e1>heater</e1> caused the <e2>fire</e2>.",
          "She put the <e1>letters</e1> into an old wooden <e2>box</e2>.",
          "The <e1>engine</e1> of the <e2>car</e2> was replaced last week.",
          "The <e1>girl</e1> sat quietly next to the <e2>window</e2>."]:
    print(classify(s), s)
```

- 関係分類は、文と 2 つのエンティティが与えられたときに、その組の関係のラベルを 1 つ選ぶ文分類である。この章の「文から関係の三つ組を抽出する (REBEL)」は、学習済みのモデルが Wikidata の関係を文から生成するだけで、関係の種類を自分では決められない。自分の分野の関係（薬と副作用、部品と製品など）を扱うには、注釈を付けたデータでこのような分類器を学習する。
- 実務では、固有表現抽出の章の「BERT をファインチューニングして日本語の固有表現抽出器を作る」で文中のエンティティを見つけ、その組ごとに印を付けてこの分類器にかけるパイプラインにすることが多い。組の大半は関係なしになるので、Other (no_relation) の例を十分に学習させる必要がある。BERT に分類の層を付けて Trainer で学習する流れは、テキスト分類の章の「BERT をファインチューニングして文書分類する (Trainer)」と同じで、ここでは印の位置のベクトルを取り出すために自作のモデルを渡している。
- 公式の F1 は、印なし 0.810、印あり + [CLS] 0.888、印あり + 印の位置 0.891 になった。印の効果ははっきりしている。印なしでも 0.81 になるのは、SemEval の文は短く、関係を聞く名詞の組が 1 つで、先に出る名詞が常に e1 だからと考えられる。印の位置のベクトルを使う方法は Baldini Soares ら (2019) が最もよかったと報告しているが、ここでは [CLS] との差 +0.003 の信頼区間（-0.005〜+0.011）が 0 をまたいでおり、ばらつきの範囲である。
- ラベルは方向付きで、`Cause-Effect(e1,e2)` は e1 が原因、`(e2,e1)` は e2 が原因を表す。上の例でも、受け身の文と能動の文で向きが入れ替わった。関係の種類は合っていて方向だけを間違えたのは 7 件と少ない。ただし `Entity-Destination(e2,e1)` は学習データにも評価データにも 1 件しかなく、評価データの 1 件も当たっていない。
- 公式の評価は、Other を除く 9 種類の関係の F1 のマクロ平均で、方向を間違えると不正解になる。Other は「9 種類のどれでもない」の寄せ集めで、クラスごとの F1 は 0.664 にとどまり、関係ありのラベル（0.76〜0.94。評価データが 1 件だけの `Entity-Destination(e2,e1)` を除く）より明らかに低い。混同行列でも、Other を関係ありと予測した 166 件と、関係ありを Other と予測した 125 件が誤りの大半を占める。
- T4 では、3 つの学習で合わせて 12〜18 分ほどかかる見込みで、GPU メモリの最大は 3.7 GB である。ここに示した値は、T4 と同じ float16 で学習したときのものである（bfloat16 が使える GPU では `BF16` が True になり、値が少し変わる。同じ GPU でも、実行のたびに F1 が 0.003 程度ぶれる）。ここでは学習データ全体で学習し、評価データで 1 回だけ測っている。学習率やエポック数を選ぶなら、学習データの一部を開発データとして分ける。
- 日本語では、ライセンスが明確で手軽に使える関係分類の注釈付きデータが見当たらない。日本語で作るなら、同じ形式（印を付けた文とラベル）で自分のデータに注釈を付け、`tohoku-nlp/bert-base-japanese-v3` などの日本語のモデルに替える。

## 固有表現を Wikidata のエンティティに結び付ける（エンティティリンキング） {entity_linking}

文中の「アップル」が企業なのか果物なのか、どのエンティティなのかを特定して、知識ベースの ID に結び付けたい。

```python
import time
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

API = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "nlp-cookbook-example/0.1 (educational notebook)"}
model = SentenceTransformer("intfloat/multilingual-e5-small")

def candidates(mention, lang="ja", limit=8):
    """表層形から候補のエンティティを検索する（候補生成）"""
    params = {"action": "wbsearchentities", "search": mention, "language": lang, "uselang": lang, "format": "json", "limit": limit}
    r = requests.get(API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    time.sleep(1)  # API を連続して呼ぶときは間隔を空ける
    hits = r.json()["search"]
    return [{"id": h["id"], "label": h.get("label", ""), "description": h.get("description", "")} for h in hits]

def link(mention, context, lang="ja"):
    """候補の説明文と文脈の類似度に検索順位を加味して、最もふさわしいエンティティを選ぶ（曖昧性解消）"""
    cands = [c for c in candidates(mention, lang) if c["description"]]
    if not cands:
        return None
    emb = model.encode([f"passage: {c['label']}: {c['description']}" for c in cands], normalize_embeddings=True)
    sims = emb @ model.encode("query: " + context, normalize_embeddings=True)
    scores = sims - 0.01 * np.arange(len(cands))  # 検索順位（人気度）が 1 つ下がるごとに 0.01 引く
    best = int(scores.argmax())
    return {**cands[best], "score": round(float(scores[best]), 3), "url": f"https://www.wikidata.org/wiki/{cands[best]['id']}"}

for mention, context in [("アップル", "アップルは新しい iPhone を発表した。"), ("アップル", "アップルパイを焼くために、アップルを 3 個買った。"),
                         ("ジャガー", "ジャガーは南米の密林に生息するネコ科の動物だ。"), ("ジャガー", "彼はジャガーの新型セダンを購入した。")]:
    print(mention, "|", context, "\n   →", link(mention, context))
```

1 つ目と 3・4 つ目は正しく結び付く。2 つ目の果物の「アップル」は、Wikidata での日本語名が「リンゴ」なので、検索で果物のエンティティ (Q89) が候補に入らず、別のもの（映画など）が選ばれてしまう。候補生成で正解が漏れると後段では取り返せない例である。

```python
# 結び付けたエンティティの情報を取得する（別名、分類、他言語のラベル）
def entity(qid, lang="ja"):
    params = {"action": "wbgetentities", "ids": qid, "languages": f"{lang}|en", "props": "labels|aliases|claims", "format": "json"}
    e = requests.get(API, params=params, headers=HEADERS, timeout=30).json()["entities"][qid]
    instance_of = [c["mainsnak"]["datavalue"]["value"]["id"] for c in e["claims"].get("P31", [])]
    return {"ja": e["labels"].get("ja", {}).get("value"), "en": e["labels"].get("en", {}).get("value"),
            "aliases": [a["value"] for a in e["aliases"].get(lang, [])][:5], "instance_of": instance_of}

print(entity("Q312"))
```

- エンティティリンキングは「候補生成」と「曖昧性解消」の 2 段階で行うのが基本。候補生成で正解が漏れると、後段では取り返せない。
- 候補の人気度（その表層形がどのエンティティを指すことが多いか）は非常に強い手がかり。検索 API の返す順位がほぼそれに当たるので、類似度と組み合わせるとよい。上のコードでも、類似度だけで選ぶと 1 つ目は「Apple II」になってしまうため、順位に応じて点を引いている。
- 知識ベースにないエンティティ (NIL) の扱いも重要。類似度が低い場合は「該当なし」とする。ただし e5 の類似度は 0.8 前後に固まりやすいので、しきい値は開発用のデータで決める。
- API を連続して呼ぶときは間隔を空ける。大量に処理するなら、Wikidata のダンプや、学習済みのエンティティリンキングモデル（mGENRE、BLINK、ReFinED など）を使う。

## 日本語の時間表現を抽出して正規化する (ja-timex) {timex_ja}

「来週の金曜日」「3 日前」「午後 3 時半」といった表現を見つけて、計算機で扱える日付や時間の形式に直したい。

```python
!pip install -q ja-timex
```

```python
import pendulum
from ja_timex import TimexParser

text = "彼は2008年4月に入社し、週に3回、30分間のジョギングを10年以上続けている。来週の金曜日の午後3時半から会議があり、一昨日届いた資料を使う。"
parser = TimexParser(reference=pendulum.datetime(2026, 11, 14, tz="Asia/Tokyo"))   # 「来週」「一昨日」の基準になる日
for t in parser.parse(text):
    print(f"{t.text:10} {t.type:9} value={t.value:12} mod={t.mod or '-'}")
```

```python
# TIMEX3 形式のタグとして出力 (TimeML。時間情報のアノテーションの標準形式)
for t in parser.parse("明日の朝9時に東京を出発して、3時間かけて京都に行く。"):
    print(t.to_tag())
```

- TIMEX3 の 4 つの型: DATE（日付）、TIME（時刻）、DURATION（期間の長さ）、SET（頻度・繰り返し）。
- 「来週」「一昨日」のような相対的な表現は、「基準から 1 週間後」「2 日前」という長さと向き (`mod`) で表される。具体的な日付にするには、文書の作成日などの基準日から自分で計算する。
- 「来週の金曜日」は「来週」と「金曜日」の 2 つに分かれる。複数の表現を組み合わせた解釈は自分で行う必要がある。
- 規則ベースなので高速で、挙動が予測できる。一方、「先の大戦」「バブルの頃」のような表現や、文脈に依存する表現（「その翌日」）は扱えない。
- 英語には HeidelTime や SUTime、多言語には `dateparser`（文字列処理と正規化の章を参照）がある。

## 否定・モダリティ・時制を取り出す {negation_modality_ja}

「行かなかったかもしれない」が、肯定なのか否定なのか、事実なのか推量なのかを判定したい。情報抽出で「起きていないこと」を事実として拾わないために。

```python
!pip install -q fugashi unidic-lite
```

```python
import re
from fugashi import Tagger

tagger = Tagger()

MODALITY = [  # (ラベル, 正規表現)。長い表現を先に照合する
    ("推量", r"かもしれない|かもしれません|かも知れない|だろう|でしょう|と思われる|らしい|ようだ|みたいだ|はずだ"),
    ("伝聞", r"そうだ$|そうです$|とのことだ|という$|といいます$|と言われている"),
    ("義務", r"なければならない|なければなりません|べきだ|べきです|ねばならない|ないといけない"),
    ("許可", r"てもよい|てもいい|ても構わない|てもかまわない"),
    ("禁止", r"てはいけない|てはならない|てはいけません|べきではない"),
    ("願望", r"たい$|たいです$|たかった$|てほしい|てほしかった"),
    ("意志", r"つもりだ|つもりです|よう$|ましょう$|ことにする|ことにした"),
    ("依頼", r"てください|てくれ$|ていただけますか|てもらえますか"),
    ("疑問", r"か$|かな$|かしら$|だろうか$|でしょうか$"),
]

def analyze(sentence):
    s = sentence.rstrip("。．！？!?")
    past = any(w.feature.pos1 == "助動詞" and w.feature.lemma == "た" for w in tagger(s))
    modality = [label for label, pat in MODALITY if re.search(pat, s)]
    core = s
    for label, pat in MODALITY:                       # モダリティ表現を除いた「命題」部分で否定を数える
        core = re.sub(pat, "", core)
    negations = sum(1 for w in tagger(core)
                    if (w.feature.pos1 == "助動詞" and w.feature.lemma in ("ない", "ぬ", "ず", "ん", "まい"))
                    or (w.feature.pos1 == "形容詞" and w.feature.lemma == "無い"))
    polarity = "否定" if negations % 2 == 1 else ("肯定 (二重否定)" if negations else "肯定")
    return {"極性": polarity, "時制": "過去" if past else "非過去", "モダリティ": modality or ["断定"]}

for s in ["彼は会議に出席した。", "彼は会議に出席しなかった。", "彼は会議に出席しないかもしれない。", "明日は雨が降るらしい。",
          "レポートを提出しなければならない。", "ここで写真を撮ってはいけない。", "行かないわけにはいかなかった。", "一度も失敗しなかった人はいないだろう。",
          "もっと早く相談してほしかった。", "本当に彼がやったのだろうか。"]:
    print(s, analyze(s))
```

- 「なければならない」「てはいけない」の「ない」は、否定ではなく義務・禁止の一部。先にモダリティ表現を取り除いてから否定を数えるのがポイント。
- この規則は文末の述語しか見ていない。「来なかった人が合格した」のような従属節の否定、「全員は来なかった」のような否定の作用域（スコープ）は扱えない。
- 事実性（その出来事が実際に起きたと書き手が述べているか）の判定は、否定・モダリティ・時制・条件節（「もし〜なら」）の組み合わせで決まる。医療文書の所見抽出（「〜を認めない」）などで実用上とても重要。
- この章の前半で扱った KWJA も、基本句の素性として時制やモダリティの情報を出力する。

## 2 つの文の論理的な関係を接続表現から推定する（談話関係） {discourse_relation}

2 つの文が「原因と結果」「逆接」「例示」などのどの関係にあるかを推定したい。注釈付きデータを使わず、接続表現の予測として解く。

```python
!pip install -q fugashi unidic-lite datasets
```

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

name = "tohoku-nlp/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForMaskedLM.from_pretrained(name).eval()

# 談話関係ごとの代表的な接続表現
CONNECTIVES = {"原因・理由": ["だから", "そのため", "したがって"], "逆接": ["しかし", "ところが", "だが"], "例示": ["たとえば", "例えば"],
               "付加": ["また", "さらに", "そして"], "言い換え": ["つまり", "すなわち"]}
print({c: tokenizer.tokenize(c) for cs in CONNECTIVES.values() for c in cs})      # 「だから」は「だ / から」のように 2 トークンになるものがある

@torch.no_grad()
def connective_score(first, second, connective):
    """接続表現を入れた文で、その接続表現のトークンを 1 つずつ [MASK] にして予測させ、対数確率の平均をとる (疑似対数尤度)"""
    enc = tokenizer(f"{first}{connective}、{second}", return_tensors="pt")
    ids = enc["input_ids"][0]
    conn_ids = tokenizer(connective, add_special_tokens=False)["input_ids"]
    start = next(i for i in range(len(ids)) if ids[i:i + len(conn_ids)].tolist() == conn_ids)
    batch = ids.repeat(len(conn_ids), 1)
    for k in range(len(conn_ids)):
        batch[k, start + k] = tokenizer.mask_token_id
    logp = model(input_ids=batch).logits.log_softmax(-1)
    return float(sum(logp[k, start + k, conn_ids[k]] for k in range(len(conn_ids)))) / len(conn_ids)     # トークン数で割って、長い表現が不利にならないようにする

def relation_scores(first, second):
    """関係ごとに、その関係の接続表現のうち最も自然なもののスコアをとる"""
    return {rel: max(connective_score(first, second, c) for c in cs) for rel, cs in CONNECTIVES.items()}

def relation(first, second, prior=None):
    """スコアを確率の形に直す。prior を与えると、関係ごとの「どんな文の組でも出やすい度合い」を差し引く"""
    best = relation_scores(first, second)
    if prior:
        best = {rel: v - prior[rel] for rel, v in best.items()}
    z = torch.tensor(list(best.values())).softmax(0)
    return {rel: round(float(p), 3) for rel, p in sorted(zip(best, z), key=lambda x: -x[1])}

examples = [("昨夜は大雪が降った。", "今朝は電車が大幅に遅れた。"), ("彼は毎日遅くまで練習した。", "試合には勝てなかった。"),
            ("この地域には多くの果物がある。", "りんごやぶどうが有名だ。"), ("この店は料理がおいしい。", "値段も手ごろだ。"),
            ("彼は約束の時間に来なかった。", "約束を破ったのだ。"), ("新しい薬の効果が確認された。", "副作用も報告されている。")]
for a, b in examples:
    print(a, "→", b, "\n    ", relation(a, b))
```

```python
# 明示的な接続表現のある文の組をコーパスから集めて、正解率を測る (接続表現を隠して当てさせる)
import re
from datasets import load_dataset

ds = load_dataset("globis-university/aozorabunko-clean", split="train")
c2r = {c: rel for rel, cs in CONNECTIVES.items() for c in cs}
pattern = re.compile("([^。「」\n]{12,50}。)(" + "|".join(c2r) + ")、([^。「」\n]{12,50}。)")
data = []
for ex in ds.select(range(3000)):
    if ex["meta"]["文字遣い種別"] == "新字新仮名":
        data += [(m.group(1), m.group(3), c2r[m.group(2)]) for m in pattern.finditer(ex["text"][:20000])]
import random
random.seed(0); random.shuffle(data)
by_rel = {}
for a, b, rel in data:
    by_rel.setdefault(rel, []).append((a, b, rel))
test = [x for rel, items in by_rel.items() for x in items[:60]]          # 関係ごとに最大 60 組
calibration = [x for rel, items in by_rel.items() for x in items[60:72]]  # 補正用に、評価に使わない組を関係ごとに 12 組
print({rel: len(items) for rel, items in by_rel.items()}, "→ 評価に", len(test), "組 / 補正に", len(calibration), "組")

import numpy as np
import pandas as pd

# 補正: 関係ごとのスコアの平均 (= その関係の接続表現が、文の中身と関係なく選ばれやすい度合い) を求める
cal_scores = [relation_scores(a, b) for a, b, _ in calibration]
prior = {rel: float(np.mean([s[rel] for s in cal_scores])) for rel in CONNECTIVES}
print("関係ごとの平均スコア:", {rel: round(v, 2) for rel, v in prior.items()})

gold = [rel for _, _, rel in test]
scores = [relation_scores(a, b) for a, b, _ in test]
for label, p in [("補正なし", None), ("補正あり", prior)]:
    pred = [max(s, key=lambda rel: s[rel] - (p[rel] if p else 0)) for s in scores]
    print(f"{label}: 正解率 {sum(x == g for x, g in zip(pred, gold)) / len(test):.3f}")
print("最も多い関係を常に答えた場合:", round(max(pd.Series(gold).value_counts()) / len(test), 3))
print(pd.crosstab(pd.Series(gold, name="正解"), pd.Series(pred, name="予測 (補正あり)")).to_string())
```

- 談話関係の解析には、接続表現が書かれている場合（明示的）と、書かれていない場合（暗黙的）がある。暗黙的な関係の推定は、今も難しい課題。
- 「接続表現を隠して当てる」問題は、テキストから自動的に大量に作れる。これで学習・評価したモデルを暗黙的な関係の推定に転用する方法は、古くから使われている。
- ただし、書き手が接続表現を省略する場面と、明示する場面は性質が違う。明示的な例での成績が、暗黙的な関係でもそのまま出るとは限らない。
- 接続表現は 1 トークンとは限らない（「だから」は「だ / から」）。トークンを 1 つずつ隠して予測させる疑似対数尤度で、長さの違う表現を比べられるようにしている。ただし、表現によってスコアの出やすさが大きく違い、補正しないと特定の関係（ここでは原因・理由）ばかりが選ばれる。評価に使わない少数の組で関係ごとの平均スコアを求め、差し引いてから比べる。
- 「また」「そして」のような付加の表現はコーパスでは最も多い（3325 組）が、モデルの予測はそこに偏っていない。補正後はむしろ例示の予測が多く（120 件）、原因・理由の 24 組と言い換えの 20 組が例示と予測されている。正解率だけでなく、混同行列や関係ごとの再現率も見る。
- 注釈付きのデータとしては、英語の Penn Discourse Treebank、日本語の京都大学ウェブ文書リードコーパスの談話関係注釈がある。精度が必要なら、これらで分類器を学習する。

## 文ペアを分類する BERT を学習し、仮説文だけでどこまで当たるかを調べる（含意関係認識、JNLI） {nli_hypothesis_only gpu}

前提文と仮説文の組を「含意・矛盾・中立」に分類する BERT を、日本語の JNLI でファインチューニングしたい。あわせて、前提文を見せずに仮説文だけで学習したモデル (hypothesis-only) がどこまで当たるかを調べ、データの作り方に由来する偏りを確かめる。

```python
!pip install -q fugashi unidic-lite
```

```python
import pandas as pd

# JGLUE の JNLI v1.3（ヤフーと早稲田大学、CC BY-SA 4.0）。画像キャプション (YJ Captions) の文を前提文にして、仮説文をクラウドソーシングで書いたもの
gh = "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jnli-v1.3/"
data = {s: pd.read_json(gh + f"{s}-v1.3.json", lines=True) for s in ["train", "valid", "test"]}
LABELS = ["entailment", "contradiction", "neutral"]
for s, d in data.items():
    print(f"{s:5} {len(d):6} 組  ラベルの割合:", d["label"].value_counts(normalize=True).round(3).to_dict())
pd.set_option("display.width", 200)
print(data["train"][["sentence1", "sentence2", "label"]].head(4).to_string(index=False))
```

```python
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          PrinterCallback, Trainer, TrainingArguments, set_seed)
from transformers.utils import logging

import datasets
logging.set_verbosity_error()   # 分類用の層を新しく初期化したという報告を出さない
logging.disable_progress_bar(); datasets.disable_progress_bars()   # 重みの読み込みと map の進み具合のバーを出さない
datasets.logging.set_verbosity_error()   # map に渡す関数をキャッシュ用にハッシュできないという警告を出さない
model_name = "tohoku-nlp/bert-base-japanese-v3"   # Apache 2.0
tokenizer = AutoTokenizer.from_pretrained(model_name)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う

# 2 文を渡すと [CLS] 前提文 [SEP] 仮説文 [SEP] の 1 列になり、token_type_ids で 1 文目 (0) と 2 文目 (1) を区別する。
# transformers 5 系のこのトークナイザー (BertJapaneseTokenizer) は、既定では token_type_ids を返さない（全部 0 とみなされる）ので、明示して返させる
enc = tokenizer("男性がギターを弾いている。", "人が楽器を演奏している。", return_token_type_ids=True)
print(tokenizer.convert_ids_to_tokens(enc["input_ids"]))
print(enc["token_type_ids"])

def encode(df, inputs):
    """inputs: "pair" は前提文と仮説文の組、"hypothesis" は仮説文だけ、"premise" は前提文だけ"""
    ds = Dataset.from_dict({"p": list(df["sentence1"]), "h": list(df["sentence2"]), "label": [LABELS.index(l) for l in df["label"]]})
    def tok(b):
        if inputs == "pair":
            return tokenizer(b["p"], b["h"], truncation=True, max_length=128, return_token_type_ids=True)
        return tokenizer(b["h"] if inputs == "hypothesis" else b["p"], truncation=True, max_length=64)
    return ds.map(tok, batched=True, remove_columns=["p", "h"])

def train(inputs, epochs=2):
    set_seed(0)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
    args = TrainingArguments(output_dir="out", num_train_epochs=epochs, learning_rate=3e-5, weight_decay=0.01, warmup_steps=0.1,
                             per_device_train_batch_size=32, per_device_eval_batch_size=128, save_strategy="no",
                             logging_strategy="no", report_to="none", bf16=BF16, fp16=not BF16, seed=0)
    trainer = Trainer(model=model, args=args, train_dataset=encode(data["train"], inputs), data_collator=DataCollatorWithPadding(tokenizer))
    trainer.remove_callback(PrinterCallback)   # 学習の終わりに train_runtime などの辞書を print しない
    trainer.train()
    return trainer
```

```python
import time

gold = np.array([LABELS.index(l) for l in data["test"]["label"]])
print("多数派（中立）を答え続けたとき: 正解率", round(float((gold == 2).mean()), 3))
preds, trainers = {}, {}
for inputs, title in [("pair", "前提文 + 仮説文"), ("hypothesis", "仮説文だけ"), ("premise", "前提文だけ")]:
    t = time.time()
    trainers[inputs] = train(inputs)
    preds[inputs] = trainers[inputs].predict(encode(data["test"], inputs)).predictions.argmax(-1)
    print(f"{title:10} 正解率 {accuracy_score(gold, preds[inputs]):.3f}  macro-F1 {f1_score(gold, preds[inputs], average='macro'):.3f}"
          f"  （学習と予測に {time.time() - t:.0f} 秒）")
print(f"GPU メモリの最大: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

```python
# 仮説文だけで当たる例（易しい例）と当たらない例（難しい例）に評価データを分けて、組で学習したモデルの正解率を比べる
easy = preds["hypothesis"] == gold
for name, mask in [("仮説文だけで当たる", easy), ("仮説文だけでは外れる", ~easy)]:
    print(f"{name}: {mask.sum():4} 組  組で学習したモデルの正解率 {(preds['pair'][mask] == gold[mask]).mean():.3f}")
print(pd.crosstab(pd.Series([LABELS[g] for g in gold], name="正解"),
                  pd.Series([LABELS[p] for p in preds["hypothesis"]], name="仮説文だけのモデルの予測")).to_string())
```

```python
# 偏りの正体を探す: 仮説文に出てくる語のうち、特定のラベルに偏るもの（学習データで 50 回以上出る語）
from collections import Counter
from fugashi import Tagger

tagger = Tagger()
train_df = data["train"]
words = [set((w.feature.lemma or w.surface).split("-")[0] for w in tagger(h)) for h in train_df["sentence2"]]   # 「サーフィン-surfing」→「サーフィン」
prior = train_df["label"].value_counts(normalize=True)
count, by_label = Counter(), {l: Counter() for l in LABELS}
for ws, label in zip(words, train_df["label"]):
    count.update(ws); by_label[label].update(ws)
rows = [(w, l, by_label[l][w] / count[w], count[w]) for w in count if count[w] >= 50 for l in LABELS]
table = pd.DataFrame(rows, columns=["語", "ラベル", "割合", "出現数"])
table["全体の割合との比"] = table["割合"] / table["ラベル"].map(prior)
for l in LABELS:
    top = table[table["ラベル"] == l].nlargest(6, "全体の割合との比")
    print(f"{l:13} (全体 {prior[l]:.2f}):", ", ".join(f"{r.語}({r.割合:.2f}, {r.出現数})" for r in top.itertuples()))

# 前提文と仮説文の文字の重なり（仮説文の文字のうち、前提文にも出てくるものの割合）。ラベルごとの平均
overlap = [len(set(h) & set(p)) / len(set(h)) for p, h in zip(train_df["sentence1"], train_df["sentence2"])]
print("文字の重なり:", pd.Series(overlap).groupby(train_df["label"]).mean().round(3).to_dict())

# 文体: JNLI の文はほとんどが です・ます調。そうでない仮説文のラベルの割合
polite = train_df["sentence2"].str.contains("(?:ます|です)。$")
print(f"仮説文が です・ます調でない組: {(~polite).sum()} 組 ({(~polite).mean():.2f})  ラベルの割合:",
      train_df["label"][~polite].value_counts(normalize=True).round(2).to_dict())
```

```python
# 同じ仮説文に別の前提文を付けて、2 つのモデルの予測を比べる。仮説文だけのモデルは前提文を見ないので、予測は変わらない
def predict(trainer, inputs, pairs):
    df = pd.DataFrame(pairs, columns=["sentence1", "sentence2"]).assign(label="neutral")
    return [LABELS[i] for i in trainer.predict(encode(df, inputs)).predictions.argmax(-1)]

# JNLI に合わせて です・ます調で書く。最後の 2 組は、同じ内容を常体（〜ている）にしたもの
pairs = [("男性が公園でギターを弾いています。", "男性が楽器を演奏しています。"),
         ("男性が部屋で本を読んでいます。", "男性が楽器を演奏しています。"),
         ("犬が芝生の上を走っています。", "犬は寝ていません。"),
         ("犬が芝生の上で寝ています。", "犬は寝ていません。"),
         ("女性がキッチンで料理をしています。", "女性が家族のために夕食を作っています。"),
         ("女性が夕食を作り、家族が待っています。", "女性が家族のために夕食を作っています。"),
         ("男性が公園でギターを弾いている。", "男性が楽器を演奏している。"),
         ("女性が夕食を作り、家族が待っている。", "女性が家族のために夕食を作っている。")]
for (p, h), a, b in zip(pairs, predict(trainers["pair"], "pair", pairs), predict(trainers["hypothesis"], "hypothesis", pairs)):
    print(f"組: {a:13} 仮説文だけ: {b:13} {p} → {h}")
```

- 含意関係認識（自然言語推論、NLI）は、2 つの文の組を 3 クラスに分ける文ペアの分類である。トークナイザーに 2 文を渡せば `[CLS] 前提文 [SEP] 仮説文 [SEP]` の 1 列になり、あとはテキスト分類の章の「BERT をファインチューニングして文書分類する (Trainer)」（記事の題名と本文の組を分類する）と同じ流れで学習できる。学習済みの NLI モデルで推論するだけなら、Transformers の章の「2 つの文の含意関係を判定する (NLI)」を使う。ここでは自分で学習し、入力を変えたモデルと比べることで、データの性質を調べている。
- `tohoku-nlp/bert-base-japanese-v3` のトークナイザー (BertJapaneseTokenizer) は、transformers 5 系では既定で `token_type_ids` を返さない（`model_input_names` に入っていない）。そのまま学習すると、前提文と仮説文の区別が [SEP] だけになる。`return_token_type_ids=True` を付けて返させる。
- 結果は、組で学習したモデルが正解率 0.881 (macro-F1 0.842)、仮説文だけが 0.576 (0.454)、前提文だけが 0.572 (0.396) で、多数派の中立を答え続けると 0.544 だった。仮説文だけで多数派を上回り、macro-F1 では差がはっきりする。ただし英語の SNLI で報告されている差（多数派 0.34 に対して仮説文だけで 0.67 前後、Gururangan ら 2018、Poliak ら 2018）よりずっと小さく、JNLI の仮説文の偏りは比較的弱い。前提文だけでも同じくらい当たるのは、JNLI では同じ 2 文が前後を入れ替えて別の組としても入っており（データの先頭の 4 組を参照）、前提文と仮説文の役割がはっきり分かれていないためと考えられる。
- 仮説文だけのモデルが当てた評価データ 1444 組では、組で学習したモデルの正解率は 0.939、外した 1064 組では 0.802 だった。仮説文だけで解ける組は易しい組で、評価データ全体の正解率にはその分の「下駄」が含まれる。Gururangan ら (2018) は、この方法で評価データを易しい組と難しい組に分け、難しい組で報告することを提案している。
- 偏りの手がかりとしては、否定（「ず」「ない」「無い」）や「だけ」を含む仮説文が矛盾に、「複数」や色・位置の語を含む仮説文が中立に偏る。クラウドソーシングで「矛盾する文を書いて」と頼むと否定を使いやすい、といった書き方の癖がそのまま手がかりになる。文字の重なりも、中立だけが低い（0.57、含意と矛盾は 0.78）。前提文にない情報を足すと中立になるからである。
- 文体の偏りもある。JNLI の文のほとんどは です・ます調だが、そうでない仮説文（学習データの 8%）は 54% が矛盾である。最後のセルでも、同じ内容を常体（〜ている）にした組は、組で学習したモデルでも矛盾と判定された。自分の入力を分類するときは、学習データの文体に合わせるか、学習データに文体の違う例を入れる。
- 学習時間は、検証の GPU で 3 つのモデルを合わせて約 4 分（学習データ 20,073 組、2 エポック、系列長は組が 128、1 文が 64）、GPU メモリの最大は 5.1 GB だった。T4 ではその 2〜3 倍の見込み。ここでは学習率やエポック数を調整しておらず、評価データで 1 回だけ測っている。調整するなら valid を使う。同じ設定でも実行のたびに正解率が 0.005 程度ぶれる。
- JNLI の入手: 公式の配布は GitHub の yahoojapan/JGLUE で、v1.3 には train・valid・test がそろっている（JGLUE の論文の時点では test は非公開だった）。HF の `shunk031/JGLUE` は読み込みスクリプト (JGLUE.py) の形式で、`trust_remote_code=True` が必要だった。datasets 4 系は読み込みスクリプトを実行しないので、そのままでは読めない。ここでは公式の JSON Lines を直接読んでいる。
- データは JNLI v1.3（ヤフーと早稲田大学、CC BY-SA 4.0）、モデルは `tohoku-nlp/bert-base-japanese-v3` (Apache 2.0)。JNLI の別の使い方（英語のデータから日本語への言語間転移）は、テキスト分類の章の「英語のデータだけで学習して日本語を分類する（ゼロショットの言語間転移）」にある。

## ニュースから出来事（誰が・いつ・どこで・何をした）を抜き出して時間順に並べる（イベント抽出） {event_extraction gpu}

ニュース記事から「誰が・何を・どこで・いつ・どうした」の組を出来事ごとに取り出し、起きた順に並べたい。この章の部品（述語項構造、時間表現の正規化、否定・モダリティ）を組み合わせる方法と、小さなオープン LLM に JSON で抜き出させる方法を、手作りの小さな正解で比べる。

```python
!pip install -q kwja ja-timex
```

```python
import re
import pandas as pd

# 架空のニュース 5 本（このレシピのために作ったもの）。日付は記事の掲載日で、「19日」「昨年」「来月」を解釈する基準になる
DOCS = [("2026-11-20", "みなと市の山本市長は19日、市役所で記者会見を開き、新しい図書館が2028年に開館する予定だと発表した。市は昨年、駅前の土地を購入していた。"),
        ("2026-11-21", "さくら電機の大阪工場で19日夜、火災が起きた。けが人はいなかった。同社は20日、来月にも生産を再開する見通しだと明らかにした。"),
        ("2026-11-22", "東西大学の研究チームは、新しい電池を開発したと21日に発表した。チームは2030年までに実用化を目指す。ただ、量産の方法はまだ決まっていない。"),
        ("2026-11-23", "北川選手は22日、東京で行われたマラソン大会で優勝した。北川選手は昨年の大会では途中で棄権していた。来年はアメリカで練習するかもしれない。"),
        ("2026-11-24", "会社員の男が23日、詐欺の疑いで県警に逮捕された。男は先月、高齢の女性から現金200万円をだまし取った疑いがある。その翌日には別の女性にも電話をかけていたという。")]

# 正解: (述語を見つける正規表現, 誰が, 何を, どこで, いつ, 事実性)。「|」は別解、「-」は書かれていない、「*」は採点しない
# いつ: 記事の日付から計算して YYYY / YYYY-MM / YYYY-MM-DD にする。事実性: 事実 / 否定 / 未確定（予定・推量・目標）
GOLD = [[("開[きくいけ]", "山本市長", "記者会見", "市役所", "2026-11-19", "事実"),
         ("開館", "図書館", "-", "-", "2028", "未確定"),
         ("発表", "山本市長", "*", "市役所|-", "2026-11-19", "事実"),     # 「何を」は「〜と」の引用（内容）なので採点しない
         ("購入", "市|みなと市", "土地|駅前の土地", "駅前|-", "2025", "事実")],
        [("起き", "火災", "-", "大阪工場|工場", "2026-11-19", "事実"),
         ("い[なる]", "けが人", "-", "大阪工場|工場|-", "2026-11-19|-", "否定"),
         ("再開", "さくら電機|同社", "生産", "大阪工場|工場|-", "2026-12", "未確定"),
         ("明らか", "さくら電機|同社", "*", "-", "2026-11-20", "事実")],
        [("開発", "研究チーム|チーム", "電池|新しい電池", "-", "-", "事実"),
         ("発表", "研究チーム|チーム", "*", "-", "2026-11-21", "事実"),
         ("目指", "研究チーム|チーム", "実用化", "-", "2030", "未確定"),
         ("決ま", "方法|量産の方法", "-", "-", "-", "否定")],
        [("行わ", "マラソン大会|大会", "-", "東京", "2026-11-22", "事実"),
         ("優勝", "北川選手", "-", "マラソン大会|東京", "2026-11-22", "事実"),
         ("棄権", "北川選手", "-", "大会|昨年の大会|-", "2025", "事実"),
         ("練習", "北川選手", "-", "アメリカ", "2027", "未確定")],
        [("逮捕", "県警", "男|会社員の男", "-", "2026-11-23", "事実"),             # 受け身: 逮捕したのは県警、されたのは男
         ("だまし取", "男|会社員の男", "現金|現金200万円", "-", "2026-10", "未確定"),  # 「疑いがある」なので確定していない
         ("かけ", "男|会社員の男", "電話", "-", "2026-10|-", "未確定")]]           # 「その翌日」は日付にできない。「という」は伝聞
SLOTS = ["who", "what", "where", "when", "fact"]

def match(gold, value, slot):
    """正解の別解のどれかに合えば True。時は完全一致、それ以外は 2 文字以上なら部分一致も認める（「市長」と「山本市長」）"""
    if gold == "*":
        return None
    v = "" if value is None else re.sub(r"\s", "", str(value))
    v = "" if v in ("null", "None", "なし", "不明") else v
    for g in gold.split("|"):
        if g == "-":
            if not v:
                return True
        elif v == g or (slot != "when" and len(v) >= 2 and (v in g or g in v)):
            return True
    return False

def evaluate(pred_docs):
    """pred_docs: 記事ごとの出来事のリスト。出来事は {"pred", "who", "what", "where", "when", "fact"} の辞書。
    正解の出来事ごとに、述語が合う予測を 1 つ探して対応づけ、項ごとに採点する（見つからなければその項はすべて誤り）"""
    found, hits, n = 0, {s: 0 for s in SLOTS}, {s: 0 for s in SLOTS}
    for gold, preds in zip(GOLD, pred_docs):
        used = set()
        for pat, *values in gold:
            j = next((j for j, p in enumerate(preds) if j not in used and re.search(pat, p["pred"] or "")), None)
            if j is not None:
                used.add(j); found += 1
            for slot, g in zip(SLOTS, values):
                ok = match(g, preds[j][slot] if j is not None else None, slot) if g != "*" else None
                if ok is not None:
                    n[slot] += 1; hits[slot] += bool(ok and j is not None)
    n_gold, n_pred = sum(map(len, GOLD)), sum(map(len, pred_docs))
    return {"出来事の再現率": f"{found}/{n_gold}", "出来事の適合率": f"{found}/{n_pred}",
            **{f"{s} の一致率": round(hits[s] / n[s], 2) for s in SLOTS}}

print(sum(map(len, GOLD)), "個の出来事。例:", GOLD[1][2])
```

```python
# 方法 1: 部品を組み合わせる。KWJA の述語項構造（ガ格 = 誰が、ヲ格 = 何を、デ格 = どこで、時間格 = いつ）と、
# 用言の素性（否定表現・時制・モダリティ）で事実性を決め、ja-timex で時間表現を記事の日付から正規化する
import pathlib, subprocess
import pendulum
from ja_timex import TimexParser
from rhoknp import Document

for i, (_, text) in enumerate(DOCS):
    pathlib.Path(f"doc{i}.txt").write_text(text + "\n")
args = [a for i in range(len(DOCS)) for a in ("--filename", f"doc{i}.txt")]   # 1 回の起動でまとめて解析する（モデルの読み込みは 1 回）
out = subprocess.run(["kwja", *args, "--tasks", "char,word", "--model-size", "base"], capture_output=True, text=True, check=True).stdout
sents = {}                                        # S-ID の末尾は「記事の番号-文の番号」
for block in re.findall(r"# S-ID:.*?\nEOS\n", out + "\n", flags=re.S):
    doc_no = int(block.split()[1].split("-")[-2])
    sents.setdefault(doc_no, []).append(block)
kdocs = [Document.from_knp("".join(sents[i])) for i in range(len(DOCS))]

def content(bp):
    """項の基本句を含む文節から、助詞や記号を除いた部分（「山本市長は」→「山本市長」）。文章の外を指す項 (不特定:物 など) は空"""
    return "".join(m.text for m in bp.phrase.morphemes if m.pos not in ("助詞", "特殊", "判定詞"))

def normalize_time(t, ref):
    """ja-timex の値を、記事の日付 ref を基準に YYYY / YYYY-MM / YYYY-MM-DD にする"""
    if t.type == "DATE":
        y, m, d = (t.value[:10].split("-") + ["XX", "XX"])[:3]
        y = str(ref.year) if y == "XXXX" else y
        if d != "XX":
            return f"{y}-{f'{ref.month:02d}' if m == 'XX' else m}-{d}"
        return f"{y}-{m}" if m != "XX" else (y if t.value[:4] != "XXXX" else None)
    if t.type == "DURATION" and t.mod in ("BEFORE", "AFTER", "NOW"):        # 「昨年」= 1 年前、「来月」= 1 か月後
        n, unit = int(t.value[1:-1] or 0), t.value[-1]
        n = -n if t.mod == "BEFORE" else (0 if t.mod == "NOW" else n)
        if unit == "Y":
            return str(ref.add(years=n).year)
        if unit == "M":
            return ref.add(months=n).format("YYYY-MM")
        if unit == "D":
            return ref.add(days=n).format("YYYY-MM-DD")
    return None

def first_time(text, ref):
    times = [normalize_time(t, ref) for t in TimexParser(reference=ref).parse(text)]
    return next((x for x in times if x), None)

def last_time(text, ref):
    times = [normalize_time(t, ref) for t in TimexParser(reference=ref).parse(text)]
    return next((x for x in reversed(times) if x), None)

def factuality(bp):
    if "否定表現" in bp.features:
        return "否定"
    if bp.features.get("時制") == "非過去" or any(k.startswith("モダリティ-") for k in bp.features):
        return "未確定"                      # 「開館する予定だ」「再開する見通し」「練習するかもしれない」
    return "事実"

def pipeline(doc, date):
    ref = pendulum.parse(date, tz="Asia/Tokyo")
    events = []
    for sent in doc.sentences:
        for k, bp in enumerate(sent.base_phrases):
            # 動詞の述語で、節の主辞になっているものを出来事とする（「新しい」のような形容詞や「予定だ」は除く）
            if bp.features.get("用言") != "動" or "節-主辞" not in bp.features or not bp.pas:
                continue
            arg = {case: a[0] for case, a in bp.pas.get_all_arguments(relax=False).items() if a}
            text_of = lambda case: content(arg[case].base_phrase) if case in arg and hasattr(arg[case], "base_phrase") else None
            where = text_of("デ")
            if where is None and "ニ" in arg and hasattr(arg["ニ"], "base_phrase"):     # ニ格は地名のときだけ場所とみなす
                ne = [n for n in sent.named_entities if n.category.value == "LOCATION" and arg["ニ"].base_phrase.head in n.morphemes]
                where = text_of("ニ") if ne else None
            # いつ: 時間格の文節を正規化する。時間格がなければ、同じ文で述語より前にある最後の時間表現を使う
            when = first_time(arg["時間"].base_phrase.phrase.text, ref) if "時間" in arg and hasattr(arg["時間"], "base_phrase") else None
            if when is None:
                when = last_time("".join(b.text for b in sent.base_phrases[:k]), ref)
            events.append({"pred": bp.text.rstrip("。、"), "who": text_of("ガ"), "what": text_of("ヲ"), "where": where,
                           "when": when, "fact": factuality(bp)})
    return events

pipe_events = [pipeline(doc, date) for doc, (date, _) in zip(kdocs, DOCS)]
for i, events in enumerate(pipe_events):
    for e in events:
        print(i, e)
print(evaluate(pipe_events))
```

```python
# 方法 2: 小さなオープン LLM に JSON で抜き出させる（Qwen3-4B-Instruct-2507、Apache 2.0）
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
logging.disable_progress_bar()   # 重みの読み込みの進み具合のバーを出さない

name = "Qwen/Qwen3-4B-Instruct-2507"
tok = AutoTokenizer.from_pretrained(name)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()

INSTRUCTION = """次のニュース記事から出来事を抜き出し、JSON だけを出力してください。
- 出来事は動詞 1 つにつき 1 つとし、記事に出てくる順に並べる。
- 各出来事は {"predicate": 動詞（記事の表記のまま）, "who": 動作や変化の主体, "what": 動作の対象（「〜を」に当たるもの）, "where": 場所, "when": 時, "factuality": 事実性}。
- when は記事の日付を基準に YYYY、YYYY-MM、YYYY-MM-DD のどれかに直す。
- factuality は「事実」「否定」「未確定」（予定・推量・目標など、まだ起きていないこと）のどれか。
- 書かれていない項目は null。主語が省略されていれば、文脈から補う。
出力の形: {"events": [...]}"""
# 形式を示す例を 1 つ付ける（評価の記事とは別に作ったもの）
EXAMPLE = ("記事の日付: 2026-10-05\n記事: 青空鉄道は4日、新しい駅を来年3月に開業すると発表した。工事は先月始まった。",
           {"events": [{"predicate": "開業する", "who": "青空鉄道", "what": "駅", "where": None, "when": "2027-03", "factuality": "未確定"},
                       {"predicate": "発表した", "who": "青空鉄道", "what": None, "where": None, "when": "2026-10-04", "factuality": "事実"},
                       {"predicate": "始まった", "who": "工事", "what": None, "where": None, "when": "2026-09", "factuality": "事実"}]})

@torch.no_grad()
def llm_extract(date, text):
    messages = [{"role": "user", "content": f"{INSTRUCTION}\n\n{EXAMPLE[0]}"},
                {"role": "assistant", "content": json.dumps(EXAMPLE[1], ensure_ascii=False)},
                {"role": "user", "content": f"{INSTRUCTION}\n\n記事の日付: {date}\n記事: {text}"}]
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(llm.device)
    out = llm.generate(**ids, max_new_tokens=600, do_sample=False)
    raw = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        events = json.JSONDecoder().raw_decode(raw[raw.find("{"):])[0]["events"]
    except (ValueError, KeyError):
        print("JSON として読めない:", raw[:200]); return []
    fact = lambda f: "否定" if "否定" in str(f) else ("事実" if "事実" in str(f) else "未確定")
    return [{"pred": e.get("predicate"), "who": e.get("who"), "what": e.get("what"), "where": e.get("where"),
             "when": e.get("when"), "fact": fact(e.get("factuality"))} for e in events]

llm_events = [llm_extract(date, text) for date, text in DOCS]
for i, events in enumerate(llm_events):
    for e in events:
        print(i, e)
print(f"GPU メモリの最大: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
pd.DataFrame({"部品の組み合わせ": evaluate(pipe_events), "LLM (JSON)": evaluate(llm_events)}).T
```

```python
# 時間順に並べる。いつが分かった出来事だけを日付の文字列で並べる（粒度が違うと、「2026-11」は「2026-11-19」より前に来る）
def timeline(pred_docs):
    rows = [(e["when"], i, e["pred"], e["who"], e["fact"]) for i, events in enumerate(pred_docs) for e in events if e["when"]]
    return [f"{w:10} [{i}] {who or '（不明）'}が {p}（{f}）" for w, i, p, who, f in sorted(rows, key=lambda r: str(r[0]))]

for title, events in [("部品の組み合わせ", pipe_events), ("LLM", llm_events)]:
    print(f"--- {title}"); print("\n".join(timeline(events)))
```

- イベント抽出は、出来事を表す語（述語）と、その参加者・時・場所を組にして取り出す課題である。この章の「文から関係の三つ組を抽出する (REBEL)」が 2 つのエンティティの間の関係を取るのに対し、ここでは述語を中心に、誰が・何を・どこで・いつを 1 つの組にする。方法 1 は、この章の「日本語の述語項構造解析をする (KWJA)」の格（ガ格・ヲ格・デ格・時間格）を項に当て、「日本語の時間表現を抽出して正規化する (ja-timex)」で時を日付に直し、事実性は KWJA が用言に付ける素性（否定表現・時制・モダリティ）から決めている。「否定・モダリティ・時制を取り出す」の規則を、述語から節の終わりまでの文字列にかけてもよい。
- 手作りの 19 個の出来事では、部品の組み合わせは出来事を 19 個すべて見つけ（余分が 1 個）、項の一致率は 0.89〜0.95 だった。LLM (Qwen3-4B-Instruct-2507) は 14 個しか見つけず、項の一致率は 0.53〜0.74 だった。ただし、記事も正解も筆者が作った 19 個だけで、「誰が = ガ格」「どこで = デ格」という決め方も KWJA の出力に近い。整った短いニュース文は解析器に有利でもある。一般的な優劣の結論ではなく、どこで間違えるかを見るための比較である。
- LLM は出来事を 5 個落とした。「〜と発表した」「〜と明らかにした」の 3 か所では、引用の中と外の述語のどちらか一方しか出さず、ほかに「いなかった」「行われた」も落とした。また、日付を細かくしすぎた（「昨年の大会」を 2025-11-22、「先月」を 2026-10-23）。「開発した」に発表の日付を付けた。「その翌日」を 2026-11-24 にしたのは根拠がない。事実性も、「疑いがある」「〜という」を事実とし、「決まっていない」を未確定とした。
- 部品の組み合わせの誤りは、どれも部品の限界によるものである。受け身の「男が県警に逮捕された」では、KWJA の格は表層の格なので、ガ格の「男」が「誰が」になる（逮捕したのは県警）。受け身・使役では格を入れ替える必要がある。「その翌日」は ja-timex が記事の日付の翌日と解釈し、2026-11-25（記事より後）になって、時系列でも逮捕の後に並んでしまった。前の出来事を基準にする時間表現は、その出来事の日付から計算しなければならない。「だまし取った疑いがある」は、述語自身の素性だけを見るので事実になり、「ある」が余分な出来事として拾われた。デ格は手段や原因も表すので、「途中」「疑い」が場所になった。
- 時系列は、日付の文字列で並べただけである。粒度の違う日付（「2026-10」と「2026-10-23」）の前後や、日付のない出来事の前後（「開発した」は「発表した」より前）は決められない。出来事の間の前後関係を直接推定する課題は時間関係の認識と呼ばれ、英語の TimeBank や TempEval、日本語の BCCWJ-TimeBank などのデータがある。
- 実用では、両方を組み合わせるとよい。解析器で述語と格を漏れなく取り出し、受け身の扱いや、時の基準の解決、事実性の判定のような文脈が必要なところを LLM に確かめさせる。LLM の出力は、大規模言語モデルの章の「LLM の出力から JSON を取り出して検証する」のように、形式と値（日付の書式、事実性のラベル）を検証してから使う。
- 検証の GPU では、KWJA のセルが 1 分弱、LLM のセルがモデルの読み込みを含めて 1 分余り（初回のモデルのダウンロードは別）で、LLM の GPU メモリの最大は 8.2 GB だった。`kwja` を入れると transformers などのバージョンが入れ替わるので、すでに transformers を読み込んでいたらランタイムを再起動する。
- 記事は架空のもの。KWJA と ja-timex は MIT ライセンス、Qwen3-4B-Instruct-2507 は Apache 2.0。
