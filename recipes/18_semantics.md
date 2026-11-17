# 意味解析

「誰が何をどうした」「それは何を指すか」「2 つの文はどうつながるか」を取り出す。構文の一歩先の解析。語義の判定は「語の意味と意味の変化」の章、文を一階述語論理の式に変換するレシピは付録の「仕組みを理解するための実装」にある。

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

- 「Apple … The company … it」のように、代名詞だけでなく名詞句同士（言い換え）の同一性も解決される。
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
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)
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
    """選択肢の順序を入れ替えて 2 回聞き、平均する (1 番目を選びやすい、といった偏りを打ち消す)"""
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
# このデータは 2 文で 1 組 (同じ登場人物で、1 語か 2 語だけ違い、答えが入れ替わる)。組の両方に正解できた割合を見る
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

## 固有表現を Wikidata のエンティティに結び付ける（エンティティリンキング） {entity_linking norun}

文中の「アップル」が企業なのか果物なのか、どのエンティティなのかを特定して、知識ベースの ID に結び付けたい。

```python
import requests
from sentence_transformers import SentenceTransformer

API = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "nlp-cookbook-example/0.1 (educational notebook)"}
model = SentenceTransformer("intfloat/multilingual-e5-small")

def candidates(mention, lang="ja", limit=8):
    """表層形から候補のエンティティを検索する (候補生成)"""
    params = {"action": "wbsearchentities", "search": mention, "language": lang, "uselang": lang, "format": "json", "limit": limit}
    hits = requests.get(API, params=params, headers=HEADERS, timeout=30).json()["search"]
    return [{"id": h["id"], "label": h.get("label", ""), "description": h.get("description", "")} for h in hits]

def link(mention, context, lang="ja"):
    """候補の説明文と文脈の類似度で、最もふさわしいエンティティを選ぶ (曖昧性解消)"""
    cands = [c for c in candidates(mention, lang) if c["description"]]
    if not cands:
        return None
    emb = model.encode([f"passage: {c['label']}: {c['description']}" for c in cands], normalize_embeddings=True)
    sims = emb @ model.encode("query: " + context, normalize_embeddings=True)
    best = int(sims.argmax())
    return {**cands[best], "score": round(float(sims[best]), 3), "url": f"https://www.wikidata.org/wiki/{cands[best]['id']}"}

for mention, context in [("アップル", "アップルは新しい iPhone を発表した。"), ("アップル", "アップルパイを焼くために、アップルを 3 個買った。"),
                         ("ジャガー", "ジャガーは南米の密林に生息するネコ科の動物だ。"), ("ジャガー", "彼はジャガーの新型セダンを購入した。")]:
    print(mention, "|", context, "\n   →", link(mention, context))
```

```python
# 結び付けたエンティティの情報を取得する (別名、分類、他言語のラベル)
def entity(qid, lang="ja"):
    params = {"action": "wbgetentities", "ids": qid, "languages": f"{lang}|en", "props": "labels|aliases|claims", "format": "json"}
    e = requests.get(API, params=params, headers=HEADERS, timeout=30).json()["entities"][qid]
    instance_of = [c["mainsnak"]["datavalue"]["value"]["id"] for c in e["claims"].get("P31", [])]
    return {"ja": e["labels"].get("ja", {}).get("value"), "en": e["labels"].get("en", {}).get("value"),
            "aliases": [a["value"] for a in e["aliases"].get(lang, [])][:5], "instance_of": instance_of}

print(entity("Q312"))
```

- エンティティリンキングは「候補生成」と「曖昧性解消」の 2 段階で行うのが基本。候補生成で正解が漏れると、後段では取り返せない。
- 候補の人気度（その表層形がどのエンティティを指すことが多いか）は非常に強い手がかり。検索 API の返す順位がほぼそれに当たるので、類似度と組み合わせるとよい。
- 知識ベースにないエンティティ (NIL) の扱いも重要。類似度が低い場合は「該当なし」とする。
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
- 「また」「そして」のような付加の表現は頻度が高く、モデルの予測もそこに偏る。関係ごとに事前確率で割る、関係ごとの再現率を見る、といった補正を検討する。
- 注釈付きのデータとしては、英語の Penn Discourse Treebank、日本語の京都大学ウェブ文書リードコーパスの談話関係注釈がある。精度が必要なら、これらで分類器を学習する。
