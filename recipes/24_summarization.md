# 要約と平易化

文章を短くまとめる、やさしく書き換える。抽出型と生成型の要約、3 行要約、見出し生成、テキスト平易化、文体の変換と、その評価。

## 重要な文を選んで抽出型要約を作る (LexRank) {extractive_summary}

長い文章から、内容を代表する文をいくつか抜き出して要約にしたい。生成モデルと違い、原文にない内容が混ざらない。

```python
!pip install -q fugashi unidic-lite
```

```python
import re
import networkx as nx
import numpy as np
from fugashi import Tagger
from sklearn.feature_extraction.text import TfidfVectorizer

tagger = Tagger()
tok = lambda s: [w.feature.orthBase or w.surface for w in tagger(s) if w.feature.pos1 in ("名詞", "動詞", "形容詞") and w.feature.pos2 != "非自立可能"]

def lexrank(text, n=3, threshold=0.1):
    sents = [s.strip() for s in re.split(r"(?<=[。！？])", text) if s.strip()]
    X = TfidfVectorizer(tokenizer=tok, token_pattern=None, lowercase=False).fit_transform(sents)
    sim = (X @ X.T).toarray()                      # 文同士のコサイン類似度
    np.fill_diagonal(sim, 0)
    sim[sim < threshold] = 0
    scores = nx.pagerank(nx.from_numpy_array(sim), weight="weight")   # 多くの文と似ている文 = 中心的な文
    top = sorted(sorted(scores, key=scores.get, reverse=True)[:n])    # スコア上位を、元の順番で並べる
    return [sents[i] for i in top], [round(scores[i], 3) for i in range(len(sents))]

text = ("自然言語処理は、人間が日常的に使っている言語をコンピュータに処理させる技術である。"
        "その歴史は 1950 年代の機械翻訳の研究にさかのぼる。"
        "初期の機械翻訳は人手で書かれた規則に基づいていたが、十分な精度は得られなかった。"
        "1990 年代になると、大量の対訳データから統計的に翻訳を学習する統計的機械翻訳が主流となった。"
        "2010 年代半ばには、ニューラルネットワークを用いたニューラル機械翻訳が登場し、翻訳の精度は飛躍的に向上した。"
        "現在では、機械翻訳に限らず、自然言語処理のほとんどのタスクでニューラルネットワークが使われている。"
        "特に、大量のテキストで事前学習された大規模言語モデルは、少ない追加学習で多様なタスクに適用できる。"
        "ところで、昨日の夕食はカレーだった。")
summary, scores = lexrank(text)
print(scores)
for s in summary:
    print("・", s)
```

- 文をノード、類似度を辺の重みとするグラフを作り、PageRank で中心的な文を選ぶ。無関係な文 (最後の文) はどの文とも似ていないのでスコアが低い。
- 類似度の計算を TF-IDF から文埋め込みに替えると、言い換えにも強くなる。
- ニュース記事では「先頭の 3 文を取る」(Lead-3) が非常に強いベースラインになる。要約手法を評価するときは必ず比較する (この章の「先頭の数文を要約とする (Lead-3) ベースラインと比べる」を参照)。
- 英語を含む各種の抽出型要約アルゴリズムは `sumy` にまとまっている。

## 要約を ROUGE で評価する (日本語対応) {rouge_ja}

要約の出力を、参照要約との単語の重なり (ROUGE-1/2/L) で評価したい。日本語でも正しく計算したい。

```python
!pip install -q rouge-score fugashi unidic-lite
```

```python
from fugashi import Tagger
from rouge_score import rouge_scorer

class MecabTokenizer:
    """rouge-score の既定のトークナイザは英数字以外を捨ててしまうので、日本語用のものを渡す"""
    def __init__(self):
        self.tagger = Tagger("-Owakati")
    def tokenize(self, text):
        return self.tagger.parse(text).split()

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=MecabTokenizer())
ref = "政府は来年度予算案を閣議決定し、一般会計の総額は過去最大となった。"
for hyp in ["政府は来年度予算案を決定し、総額は過去最大となった。", "来年度の予算は過去最大の規模である。", "今日は天気が良い。"]:
    s = scorer.score(ref, hyp)    # 引数の順番は (参照, 出力)
    print(f"R1 {s['rouge1'].fmeasure:.3f}  R2 {s['rouge2'].fmeasure:.3f}  RL {s['rougeL'].fmeasure:.3f}  {hyp}")
```

```python
# 既定のトークナイザに日本語を渡すと、何も残らず 0 になる
default = rouge_scorer.RougeScorer(["rouge1"])
print(default.score(ref, "政府は来年度予算案を決定し、総額は過去最大となった。"))

# コーパス全体では、文書ごとの F 値を平均して報告するのが一般的
import numpy as np
refs = [ref, "新しい言語モデルが発表され、性能が大幅に向上した。"]
hyps = ["政府は予算案を決定した。", "新しいモデルが発表された。"]
print(round(float(np.mean([scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)])), 3))
```

- ROUGE-1/2 は単語 1-gram / 2-gram の重なり、ROUGE-L は最長共通部分列。再現率・適合率・F 値のどれを報告しているかを明記する。
- 日本語で「ROUGE が 0 になる」「異常に高い/低い」ときは、ほぼトークナイザの問題。

## 先頭の数文を要約とする (Lead-3) ベースラインと比べる {lead3_baseline}

凝った手法と比べる前に、「先頭の 3 文を取るだけ」という最も単純な方法でどこまでいけるかを確かめたい。

```python
!pip install -q rouge-score fugashi unidic-lite networkx
```

```python
import io, re, tarfile, urllib.request
import pandas as pd

def load_livedoor():
    url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
    with urllib.request.urlopen(url) as r:
        tar = tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")
    rows = []
    for m in tar.getmembers():
        parts = m.name.split("/")
        if m.isfile() and len(parts) == 3 and parts[2] != "LICENSE.txt":
            lines = tar.extractfile(m).read().decode("utf-8").split("\n")
            rows.append({"label": parts[1], "title": lines[2], "text": "\n".join(lines[3:])})
    return pd.DataFrame(rows)

def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[。！？])|\n+", text) if s and len(s.strip()) > 5]

df = load_livedoor()
df = df[df["label"].isin(["topic-news", "sports-watch", "it-life-hack"])].sample(150, random_state=0).reset_index(drop=True)
df["sents"] = df["text"].map(split_sentences)
df = df[df["sents"].map(len) >= 8].reset_index(drop=True)
print(len(df), "記事 / 平均", round(df["sents"].map(len).mean(), 1), "文")
```

```python
import random
import networkx as nx
import numpy as np
from fugashi import Tagger
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer

tagger = Tagger()
content = lambda s: [w.surface for w in tagger(s) if w.feature.pos1 in ("名詞", "動詞", "形容詞")]

class Tok:
    def tokenize(self, text):
        return [w.surface for w in tagger(text)]

def lead(sents, n=3):
    return sents[:n]

def random_pick(sents, n=3, seed=0):
    return [sents[i] for i in sorted(random.Random(seed).sample(range(len(sents)), n))]

def lexrank(sents, n=3):
    X = TfidfVectorizer(tokenizer=content, token_pattern=None, lowercase=False).fit_transform(sents)
    sim = (X @ X.T).toarray(); np.fill_diagonal(sim, 0)
    scores = nx.pagerank(nx.from_numpy_array(sim), weight="weight")
    return [sents[i] for i in sorted(sorted(scores, key=scores.get, reverse=True)[:n])]

# 人手の要約がないので、記事のタイトルを「参照要約」の代わりに使う (タイトルの語をどれだけ拾えたか = ROUGE の再現率)
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], tokenizer=Tok())
rows = []
for name, method in [("ランダムな 3 文", random_pick), ("Lead-3 (先頭 3 文)", lead), ("LexRank", lexrank)]:
    scores = [scorer.score(title, "".join(method(sents))) for title, sents in zip(df["title"], df["sents"])]
    rows.append({"手法": name, "ROUGE-1 再現率": np.mean([s["rouge1"].recall for s in scores]), "ROUGE-2 再現率": np.mean([s["rouge2"].recall for s in scores]),
                 "平均文字数": np.mean([len("".join(method(s))) for s in df["sents"]])})
pd.DataFrame(rows).round(3)
```

```python
i = 0
print("タイトル:", df["title"][i])
print("Lead-3 :", " ".join(lead(df["sents"][i])))
print("LexRank:", " ".join(lexrank(df["sents"][i])))
```

- ニュース記事は重要なことを最初に書く (逆ピラミッド型) ので、Lead-3 は非常に強い。凝った手法が Lead-3 に勝てないことは珍しくない。
- 比べている LexRank と、日本語のトークナイザを渡した ROUGE の使い方は、この章の「重要な文を選んで抽出型要約を作る (LexRank)」と「要約を ROUGE で評価する (日本語対応)」で説明した。
- 新しい要約手法を評価するときは、必ず Lead-3 とランダムを並べて報告する。
- ここではタイトルを参照の代わりにした簡易な評価。本来は人手で書いた参照要約 (日本語なら livedoor の 3 行要約、XL-Sum の日本語部分、WikiLingua など) を使う。
- 小説、論文、議事録のように、重要な内容が先頭にあるとは限らない文書では、Lead の優位は小さくなる。

## LLM で文章を要約する (長さと観点を指定する) {llm_summary gpu}

文章の内容を、自分の言葉でまとめ直した要約 (生成型要約) が欲しい。長さや、何に注目するかを指定したい。

```python
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"   # 3B の日本語モデル。GPU (T4 など) で 1 要約あたり数秒〜10 秒ほど (VRAM 約 7GB)
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる (LLM の章の「オープンな LLM とチャットする」を参照)。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()

def chat(prompt, max_new_tokens=150):
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to(llm.device)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

text = ("市は 14 日、来年 4 月から市内の全小中学校で給食費を無償化すると発表した。対象は約 1 万 2 千人で、年間の費用は約 6 億円を見込む。"
        "財源には、ふるさと納税の寄付金と、公共施設の統廃合で生じる削減分を充てる。"
        "市長は記者会見で「子育て世帯の負担を減らし、若い世代に選ばれるまちにしたい」と述べた。"
        "一方、市議会の一部からは「恒久的な財源の裏付けが弱い」との指摘が出ている。"
        "市は来月の議会に関連する条例案を提出する方針で、可決されれば県内の市では 3 例目となる。"
        "保護者からは歓迎の声が上がる一方、食材費の高騰で給食の質が下がらないか心配する意見もある。")
print(len(text), "文字")
print(chat(f"次の文章を要約してください。\n\n{text}"))
```

```python
# 長さを指定する
for instruction in ["1 文 (40 字以内) で要約してください。", "100 字程度で要約してください。"]:
    summary = chat(f"次の文章を{instruction}要約だけを出力してください。\n\n{text}")
    print(f"[{instruction}] ({len(summary)} 字) {summary}")
```

```python
# 観点や読み手を指定する
for focus in ["財源と費用に注目して、2 文で要約してください。", "賛成と反対の意見を 1 文ずつにまとめてください。", "小学生にもわかる言葉で、2 文で要約してください。"]:
    print(f"[{focus}]\n   {chat(focus + chr(10) + chr(10) + text)}")
```

- 生成型要約は、文をつなぎ直したり言い換えたりできるので、抽出型より短く自然にまとまる。そのかわり、原文にないことを書く (ハルシネーション) おそれがある。数値と固有名詞は必ず原文と照らし合わせる (上の最初の出力でも、原文の「来年 4 月」が具体的な年に書き換えられている)。
- 文字数の指定は目安にしかならない。厳密な上限が必要なら、生成後に文字数を確かめて、超えていたら「もっと短く」と再生成する。
- 「誰のための、何のための要約か」を書くと質が上がる。同じ文章でも、目的が違えば残すべき情報が違う。
- 小さいモデルは、長い入力の後半を見落としやすい。長い文書は、この章の「長い文書を分割して要約する」を参照。

## ニュース記事を 3 行に要約する {three_line_summary}

記事を「3 つの箇条書き」にまとめたい。形式が崩れた出力を検出して、決まった形に整えたい。

```python
import re
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
llm = AutoModelForCausalLM.from_pretrained(name).eval()

def chat(messages, max_new_tokens=160):
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

INSTRUCTION = ("次の記事を 3 行に要約してください。\n"
               "- 必ず 3 行で、各行は「・」で始める\n- 1 行は 35 字以内の 1 文にする\n- 記事に書かれていないことは書かない\n- 3 行以外の文章 (前置きや補足) は出力しない")
EXAMPLE_IN = ("気象庁は 10 日、関東甲信地方が梅雨入りしたとみられると発表した。平年より 3 日遅く、昨年より 2 日早い。"
              "前線の影響で、11 日にかけて太平洋側を中心に大雨となるおそれがある。同庁は土砂災害や低い土地の浸水に警戒を呼びかけている。")
EXAMPLE_OUT = "・関東甲信地方が梅雨入りしたとみられる\n・平年より 3 日遅く、昨年より 2 日早い\n・11 日にかけて太平洋側で大雨のおそれ"

def parse(output):
    """「・」で始まる行だけを取り出す。番号付きや「-」で書かれても拾う"""
    lines = [re.sub(r"^\s*(?:[・\-*●]|\d+[.．)])\s*", "", l).strip() for l in output.splitlines() if re.match(r"^\s*(?:[・\-*●]|\d+[.．)])", l)]
    return [l for l in lines if l]

def three_lines(text, max_retries=1):
    messages = [{"role": "user", "content": f"{INSTRUCTION}\n\n{EXAMPLE_IN}"}, {"role": "assistant", "content": EXAMPLE_OUT},   # 出力例を 1 つ見せる
                {"role": "user", "content": f"{INSTRUCTION}\n\n{text}"}]
    for attempt in range(max_retries + 1):
        output = chat(messages)
        lines = parse(output)
        if len(lines) == 3:
            return lines, attempt
        messages += [{"role": "assistant", "content": output}, {"role": "user", "content": "形式が違います。「・」で始まる 3 行だけを出力し直してください。"}]
    return (lines + ["(要約できませんでした)"] * 3)[:3], attempt      # それでも駄目なら、先頭 3 行に切り詰める・埋める

articles = [
    ("市は 14 日、来年 4 月から市内の全小中学校で給食費を無償化すると発表した。対象は約 1 万 2 千人で、年間の費用は約 6 億円を見込む。"
     "財源には、ふるさと納税の寄付金と、公共施設の統廃合で生じる削減分を充てる。市長は記者会見で「子育て世帯の負担を減らし、若い世代に選ばれるまちにしたい」と述べた。"
     "一方、市議会の一部からは「恒久的な財源の裏付けが弱い」との指摘が出ている。市は来月の議会に関連する条例案を提出する方針だ。"),
    ("大学の研究チームは、スマートフォンのカメラで撮影した葉の画像から、稲の病気を判定する人工知能を開発した。約 5 万枚の画像で学習させ、8 種類の病気を 9 割以上の精度で見分けられるという。"
     "農家が専門家に相談する前に、その場で病気の見当を付けられるのが利点だ。チームは来年度、県内の農協と協力して実証実験を行い、3 年後の実用化を目指す。"
     "課題は、照明や撮影角度による精度のばらつきで、今後は撮影方法の案内機能も加える予定だ。"),
]
for text in articles:
    lines, retries = three_lines(text)
    print(f"(再試行 {retries} 回)")
    for l in lines:
        print(f"  ・{l}  ({len(l)} 字)")
```

- 形式の指定は、箇条書きの条件として具体的に書き、出力例を 1 つ対話の形で見せると安定する。
- それでも形式が崩れることはある。「解析して検証 → 駄目なら再生成 → 最後は機械的に整形」と、必ず決まった形で返る仕組みにしておく。
- 1 つ目の記事では、原文に「市」としか書かれていないのに「横浜市」と書いている。「記事に書かれていないことは書かない」と指示しても、小さいモデルは固有名詞を作ってしまうことがある。
- 3 行要約は、1 行目に最も重要な事実、2 行目以降に補足、という構成が読みやすい。順序も指示できる。
- 文字数の上限は守られないことが多い。厳密に必要なら、行ごとに検査して長い行だけ言い換えさせる。

## 長い文書を分割して要約する {long_document_summary gpu}

モデルに一度に入らない (あるいは入れると後半を読み飛ばす) 長い文書を要約したい。分割して要約し、それをまとめ直す。

```python
import io, re, urllib.request, zipfile
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる (LLM の章の「オープンな LLM とチャットする」を参照)。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()

def chat(prompt, max_new_tokens=160):
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to(llm.device)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

text = aozora("https://www.aozora.gr.jp/cards/000035/files/1567_ruby_4948.zip")   # 走れメロス (約 1 万字)
sents = [s.strip() for s in re.split(r"(?<=[。！？」])(?![」。！？])|\n+", text) if s and s.strip()]

def make_chunks(sents, max_chars=2500):
    chunks, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) > max_chars:
            chunks.append(cur); cur = ""
        cur += s
    return chunks + [cur]

chunks = make_chunks(sents)
print(len(text), "文字 →", len(chunks), "分割", [len(c) for c in chunks])
```

```python
# 1. 分割ごとに要約する (map)。T4 なら 4 分割で 30 秒〜1 分ほど
partial = []
for i, chunk in enumerate(chunks, 1):
    s = chat(f"次は小説の一部です。この部分で起きた出来事を、登場人物の名前を入れて 2〜3 文で要約してください。要約だけを出力してください。\n\n{chunk}", max_new_tokens=140)
    partial.append(s)
    print(f"[{i}] {s}")
```

```python
# 2. 部分要約をつなげて、全体の要約を作る (reduce)
joined = "\n".join(f"{i}. {s}" for i, s in enumerate(partial, 1))
final = chat(f"次は、ある小説を前から順に分けて要約したものです。これらをまとめて、物語全体のあらすじを 4 文で書いてください。あらすじだけを出力してください。\n\n{joined}", max_new_tokens=220)
print(final)
```

- 「分割して要約 → 要約を要約」は map-reduce 型と呼ばれる。分割ごとの処理は独立なので、並列化もできる。
- 部分要約の段階で落とした情報は、最終要約には戻らない。固有名詞や数値を残すよう、部分要約のプロンプトで指示する。
- 3B クラスのモデルでは、部分要約に話の取り違えや人名の誤りが混ざることがある (上の出力でも、短刀を持って王城に入り捕らえられたメロスが「王に短刀を差し出した」ことになり、最終要約では「短刀を差し出して抵抗する」に変わっている。妹の結婚式を翌日に繰り上げる場面は、部分要約の段階で「延期」と取り違え、最終要約にもそのまま残った)。誤りは最終要約にそのまま引き継がれるので、重要な用途ではより大きなモデルを使うか、次項の検査と組み合わせる。
- 前の部分の要約を次の部分の入力に加えていく方法 (逐次的な洗練) だと、話のつながりは保ちやすいが、並列化はできず、誤りが後ろに伝わる。
- 長い入力をそのまま読めるモデルでも、文書が長くなるほど中ほどの内容を見落としやすい。分割は今でも有効な方法。

## 要約が原文と食い違っていないか検査する {summary_faithfulness}

生成した要約に、原文にないことや、原文と矛盾することが書かれていないかを自動で確かめたい。

```python
import re
import unicodedata
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast

source = ("市は 14 日、来年 4 月から市内の全小中学校で給食費を無償化すると発表した。対象は約 1 万 2 千人で、年間の費用は約 6 億円を見込む。"
          "財源には、ふるさと納税の寄付金と、公共施設の統廃合で生じる削減分を充てる。"
          "市長は記者会見で「子育て世帯の負担を減らし、若い世代に選ばれるまちにしたい」と述べた。"
          "一方、市議会の一部からは「恒久的な財源の裏付けが弱い」との指摘が出ている。")
summaries = {
    "忠実": "市は来年 4 月から小中学校の給食費を無償化する。費用は年間約 6 億円を見込む。市議会の一部からは財源を不安視する声がある。",
    "矛盾あり": "市は来年 4 月から小中学校の給食費を無償化する。市議会は全会一致でこれを歓迎している。",
    "付け足しあり": "市は来年 4 月から小中学校の給食費を無償化する。隣の市でも同様の制度が始まっている。",
    "数値の誤り": "市は来年 4 月から小中学校の給食費を無償化する。費用は年間約 60 億円で、対象は約 1 万 2 千人である。",
}

nli_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
# transformers 5 系の DebertaV2Tokenizer は全角の記号や英数字を [UNK] にしてしまう (issue #49622) ので、tokenizer.json から直接読む
tok = PreTrainedTokenizerFast.from_pretrained(nli_name)
nli = AutoModelForSequenceClassification.from_pretrained(nli_name).eval()

def nli_probs(premise, hypotheses):
    with torch.no_grad():
        return nli(**tok([premise] * len(hypotheses), hypotheses, return_tensors="pt", padding=True, truncation=True)).logits.softmax(-1)

rows = []
for label, summary in summaries.items():
    sents = [s for s in re.split(r"(?<=。)", summary) if s.strip()]
    for sent, p in zip(sents, nli_probs(source, sents)):
        verdict = "含意" if p[0] > 0.5 else "矛盾" if p[2] > 0.5 else "中立 (原文にない)"
        rows.append({"要約": label, "文": sent, "含意": round(p[0].item(), 2), "矛盾": round(p[2].item(), 2), "判定": verdict})
pd.set_option("display.max_colwidth", 60)
pd.DataFrame(rows)
```

```python
# 数値と固有名詞は NLI が見逃しやすいので、原文に同じ表記があるかを別に確かめる
def numbers(text):
    text = unicodedata.normalize("NFKC", text)
    return set(re.findall(r"\d+(?:\.\d+)?\s*(?:万|億|千)?\s*(?:円|人|日|月|年|例目|%)?", text.replace(" ", "")))

src_nums = numbers(source)
for label, summary in summaries.items():
    missing = [n for n in numbers(summary) if n not in src_nums]
    print(f"{label}: 原文にない数値表現 = {missing}")
```

- 要約の各文を仮説、原文を前提として含意関係を判定する。含意なら裏付けあり、矛盾なら誤り、中立なら原文にない付け足し。
- NLI は桁違いの数値 (6 億と 60 億) のような誤りを見逃すことがある。数値・日付・固有名詞は、文字列として原文に存在するかを別に検査する。
- 原文が長いと NLI モデルの入力長を超える。原文を段落に分け、要約文ごとに最も含意の高い段落を採用する。
- 忠実であることと、良い要約であることは別。重要な内容を落としていないか (網羅性) は、参照要約との比較や人手で確かめる。

## 記事の見出しを生成して実際の見出しと比べる {headline_generation gpu}

記事の本文から、内容を一言で表す見出し (タイトル) を作りたい。極端に短い要約として、生成結果を実際の見出しと比べたい。

```python
!pip install -q rouge-score fugashi unidic-lite
```

```python
import io, tarfile, urllib.request
import pandas as pd
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_livedoor():
    url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
    with urllib.request.urlopen(url) as r:
        tar = tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")
    rows = []
    for m in tar.getmembers():
        parts = m.name.split("/")
        if m.isfile() and len(parts) == 3 and parts[2] != "LICENSE.txt":
            lines = tar.extractfile(m).read().decode("utf-8").split("\n")
            rows.append({"label": parts[1], "title": lines[2], "text": "\n".join(lines[3:])})
    return pd.DataFrame(rows)

df = load_livedoor()
df = df[df["label"].isin(["topic-news", "sports-watch"])].sample(12, random_state=1).reset_index(drop=True)

name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる (LLM の章の「オープンな LLM とチャットする」を参照)。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()

def headline(text):
    prompt = ("次のニュース記事に、30 字以内の見出しを 1 つ付けてください。体言止めにし、記事にない情報は入れないでください。見出しだけを出力してください。\n\n" + text[:600])
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to(llm.device)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip("「」『』")

df["generated"] = [headline(t) for t in df["text"]]   # T4 なら 12 記事で 1 分ほど
df["lead"] = [t.strip().split("。")[0][:30] for t in df["text"]]      # ベースライン: 本文の最初の文の先頭 30 字
for _, r in df.head(5).iterrows():
    print(f"実際: {r['title']}\n生成: {r['generated']}\n先頭: {r['lead']}\n")
```

```python
import numpy as np
from fugashi import Tagger
from rouge_score import rouge_scorer

tagger = Tagger()

class Tok:
    def tokenize(self, text):
        return [w.surface for w in tagger(text)]

scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], tokenizer=Tok())
for col, label in [("generated", "LLM の見出し"), ("lead", "本文の先頭 30 字")]:
    scores = [scorer.score(t, g) for t, g in zip(df["title"], df[col])]
    print(f"{label:12} ROUGE-1 F = {np.mean([s['rouge1'].fmeasure for s in scores]):.3f}  ROUGE-L F = {np.mean([s['rougeL'].fmeasure for s in scores]):.3f}  平均 {df[col].str.len().mean():.0f} 字")
```

- 見出し生成は、要約の中でも最も圧縮率が高いタスク。何を残すかの選択がそのまま品質になる。
- 実際の見出しは、読者の興味を引くための表現 (煽り、伏せ字、疑問形) を含むことが多く、内容の要約とは限らない。ROUGE が低くても、見出しとして悪いとは言えない。
- 12 記事では平均値のばらつきが大きい。手法を比べるなら、数百件以上で評価し、信頼区間を付ける (評価指標と統計的検定の章を参照)。
- 「記事にない情報は入れない」と指示しても、固有名詞や数字を作ってしまうことがある。公開する見出しに使うなら、人の確認が必要。

## 文章をやさしい日本語に書き換える (テキスト平易化) {text_simplification}

難しい文章を、子どもや日本語を学んでいる人にも読める、やさしい文章に書き換えたい。本当にやさしくなったか、意味が保たれているかも確かめたい。

```python
!pip install -q fugashi unidic-lite regex "wordfreq[cjk]"
```

```python
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
llm = AutoModelForCausalLM.from_pretrained(name).eval()

def chat(messages, max_new_tokens=200):
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

RULES = ("次の文章を「やさしい日本語」に書き換えてください。\n"
         "- 1 文を短くする (1 文に 1 つのことだけを書く)\n- 難しい言葉は、簡単な言葉に言い換える\n- 敬語や受け身、二重否定を使わない\n"
         "- 文末は「です」「ます」にする\n- 大事な情報 (日時、場所、してほしいこと) は残す\n- 書き換えた文章だけを出力する")
SHOT_IN = "台風の接近に伴い、本日午後より公共交通機関に大幅な遅延が生じる見込みであるため、不要不急の外出は控えられたい。"
SHOT_OUT = "台風が近づいています。今日の午後から、電車やバスがとても遅れます。大事な用事がない人は、外に出ないでください。"

def simplify(text):
    return chat([{"role": "user", "content": f"{RULES}\n\n{SHOT_IN}"}, {"role": "assistant", "content": SHOT_OUT},
                 {"role": "user", "content": f"{RULES}\n\n{text}"}])

texts = ["本日未明に発生した地震の影響により、当該地域では断水が継続しており、復旧の見通しは立っていないことから、給水所の利用を検討されたい。",
         "転入届は、新住所に居住を開始した日から 14 日以内に、本人確認書類を持参のうえ、市役所の窓口にて手続きを行う必要がある。",
         "発熱や咳などの症状が認められる場合には、医療機関を受診する前に、あらかじめ電話で相談することが推奨される。"]
simplified = [simplify(t) for t in texts]
for src, out in zip(texts, simplified):
    print(f"原文: {src}\n平易: {out}\n")
```

```python
# 1. 本当にやさしくなったか: 文の長さ、漢字の割合、語彙の頻度 (よく使われる語ほど Zipf 値が高い) で比べる
import re
import pandas as pd
import regex
from fugashi import Tagger
from wordfreq import zipf_frequency

tagger = Tagger()

def difficulty(text):
    sents = [s for s in re.split(r"(?<=[。！？])", text) if s.strip()]
    words = [(w.surface, w.feature.pos1, w.feature.orthBase or w.surface) for s in sents for w in tagger(s) if w.feature.pos1 not in ("補助記号", "空白")]
    content = [base for _, pos, base in words if pos in ("名詞", "動詞", "形容詞", "形状詞", "副詞")]
    chars = regex.findall(r"\S", text)
    return {"文数": len(sents), "平均文長 (語)": round(len(words) / len(sents), 1),
            "漢字率": round(len(regex.findall(r"\p{Han}", text)) / len(chars), 2),
            "語の頻度 (平均)": round(sum(zipf_frequency(w, "ja") for w in content) / len(content), 2),
            "低頻度語の割合": round(sum(zipf_frequency(w, "ja") < 4 for w in content) / len(content), 2)}

rows = []
for i, (src, out) in enumerate(zip(texts, simplified), 1):
    rows += [{"文章": f"{i} 原文", **difficulty(src)}, {"文章": f"{i} 平易", **difficulty(out)}]
pd.DataFrame(rows)
```

```python
# 2. 意味が保たれているか: 平易化した文章の内容が原文から言えるか (含意) と、原文の要点が残っているか (逆向きの含意)
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast

nli_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
# transformers 5 系の DebertaV2Tokenizer は全角の記号や英数字を [UNK] にしてしまう (issue #49622) ので、tokenizer.json から直接読む
nli_tok = PreTrainedTokenizerFast.from_pretrained(nli_name)
nli = AutoModelForSequenceClassification.from_pretrained(nli_name).eval()

def entail(premise, hypothesis):
    with torch.no_grad():
        return nli(**nli_tok(premise, hypothesis, return_tensors="pt", truncation=True)).logits.softmax(-1)[0, 0].item()

for src, out in zip(texts, simplified):
    out_sents = [s for s in re.split(r"(?<=[。！？])", out) if s.strip()]
    added = min(entail(src, s) for s in out_sents)          # 低いと、原文にないことを書き足している
    kept = entail(out, src)                                  # 低いと、原文の情報が落ちている
    print(f"付け足しなし {added:.2f} / 情報の保持 {kept:.2f} | {out[:40]}...")
```

- テキスト平易化は「意味を保ったまま、やさしく書き換える」タスク。評価も、やさしさ・意味の保持・自然さの 3 つを分けて見る。上の 2 つ目の文章では、原文にない「手続きを忘れないようにしてください」を書き足したため、「付け足しなし」が 0.00 になった。
- やさしい日本語には、1 文を短く、漢語より和語、敬語や受け身を避ける、といった具体的な指針がある。それを条件として列挙し、書き換えの例を 1 つ見せると従いやすい。
- 短く書き換える過程で、期限や持ち物のような大事な情報が落ちやすい。行政や防災の文章では、落ちていないかの確認が欠かせない。
- 参照となる平易文がある場合は SARI (追加・削除・保持の適否を測る指標) がよく使われる。日本語のデータには SNOW T15/T23 (やさしい日本語対訳コーパス)、JADES などがある。

## 文体を変換する (です・ます調、だ・である調、敬語) {style_transfer_ja gpu}

文章の文体を、です・ます調とだ・である調の間で揃え直したい。くだけた文を丁寧な文に言い換えたい。活用形を使った規則と LLM で変換し、文体が変わったか、意味が保たれているか、余計な書き換えがないかを測って比べる。

```python
!pip install -q fugashi unidic-lite
```

```python
# 例文はこのレシピのために作ったもの。常体と敬体の対 30 組 (両方向の変換の正解として使う) と、くだけた文 12 文と敬語の言い換えの一例
PAIRS = [  # (だ・である調, です・ます調)
    ("明日は雨が降る。", "明日は雨が降ります。"),
    ("昨日は図書館で本を読んだ。", "昨日は図書館で本を読みました。"),
    ("彼はまだ駅に着いていない。", "彼はまだ駅に着いていません。"),
    ("その店は日曜日に開かなかった。", "その店は日曜日に開きませんでした。"),
    ("この問題は難しい。", "この問題は難しいです。"),
    ("昨日の試験は易しかった。", "昨日の試験は易しかったです。"),
    ("この部屋はあまり広くない。", "この部屋はあまり広くありません。"),
    ("父は高校の教師だ。", "父は高校の教師です。"),
    ("会議の資料は先週のものだった。", "会議の資料は先週のものでした。"),
    ("それは私の傘ではない。", "それは私の傘ではありません。"),
    ("当時、この町に駅はなかった。", "当時、この町に駅はありませんでした。"),
    ("少し休憩しよう。", "少し休憩しましょう。"),
    ("午後には晴れるだろう。", "午後には晴れるでしょう。"),
    ("締め切りは来週の金曜日である。", "締め切りは来週の金曜日です。"),
    ("窓を開けてもいいか。", "窓を開けてもいいですか。"),
    ("毎朝、駅まで歩いて通っている。", "毎朝、駅まで歩いて通っています。"),
    ("財布を家に忘れてしまった。", "財布を家に忘れてしまいました。"),
    ("来年は海外で働きたい。", "来年は海外で働きたいです。"),
    ("書類は明日までに提出しなければならない。", "書類は明日までに提出しなければなりません。"),
    ("電車が遅れたので、会議に間に合わなかった。", "電車が遅れたので、会議に間に合いませんでした。"),
    ("値段は高いが、品質はとても良い。", "値段は高いですが、品質はとても良いです。"),
    ("先生は「明日は休みだ」と言った。", "先生は「明日は休みだ」と言いました。"),
    ("駅の近くに新しいパン屋ができた。毎朝、行列ができている。", "駅の近くに新しいパン屋ができました。毎朝、行列ができています。"),
    ("彼が来るかどうかは分からない。", "彼が来るかどうかは分かりません。"),
    ("この薬は食後に飲むことになっている。", "この薬は食後に飲むことになっています。"),
    ("彼女は来月、大阪に引っ越すそうだ。", "彼女は来月、大阪に引っ越すそうです。"),
    ("道が混んでいるのは、事故があったからだ。", "道が混んでいるのは、事故があったからです。"),
    ("彼は昨日、熱があったのだ。", "彼は昨日、熱があったのです。"),
    ("明日、もう一度来る。", "明日、もう一度来ます。"),
    ("その映画はまだ見ていないが、評判は聞いている。", "その映画はまだ見ていませんが、評判は聞いています。"),
]
CASUAL = [  # (くだけた文, 敬語の言い換えの一例。正解は 1 つではない)
    ("ちょっと遅れるかも。", "少し遅れるかもしれません。"),
    ("この資料、もう見た？", "この資料はもうご覧になりましたか。"),
    ("明日の会議、出られる？", "明日の会議に出席できますか。"),
    ("それ、すごくいいじゃん。", "それはとても良いですね。"),
    ("今、手が離せないから、あとで電話する。", "今、手が離せないので、後ほどお電話します。"),
    ("悪いけど、ペン貸してくれない？", "申し訳ありませんが、ペンを貸していただけませんか。"),
    ("駅で待ってるね。", "駅でお待ちしています。"),
    ("その件はもう部長に言っといた。", "その件はもう部長に申し上げておきました。"),
    ("資料、メールで送っちゃった。", "資料をメールでお送りしました。"),
    ("何時ごろ来る？", "何時ごろいらっしゃいますか。"),
    ("雨だから、今日はやめとこう。", "雨なので、今日はやめておきましょう。"),
    ("了解、すぐやる。", "承知しました。すぐに取りかかります。"),
]

import re
from fugashi import Tagger

tagger = Tagger()
def split_sents(text):
    return [s for s in re.split(r"(?<=[。！？])", text) if s.strip()]

CASUAL_LEMMAS = {"てる", "でる", "ちゃう", "じゃう", "とく", "どく", "じゃん"}

def ending_style(sentence):
    """文末の文体の判定器: 文末の述語の部分 (最後の自立語から後ろ) に「です」「ます」があれば敬体"""
    ws = [w for w in tagger(sentence) if w.feature.pos1 != "補助記号" and w.surface not in ("「", "」")]
    while ws and ws[-1].feature.pos2 == "終助詞":
        ws.pop()
    tail = []
    for w in reversed(ws):
        if w.feature.pos2 == "接続助詞":               # 「〜ましたので、行く」の前半は見ない
            break
        tail.append(w)
        if w.feature.pos1 in ("名詞", "代名詞", "形状詞", "副詞") or w.feature.pos2 == "一般":
            break
    if any(w.feature.lemma in ("です", "ます") or w.feature.orthBase in ("ください", "ござる") for w in tail):
        return "敬体"
    return "体言止め" if ws and ws[-1].feature.pos1 in ("名詞", "代名詞") else "常体"

def has_polite(text):
    """「」の外に「です」「ます」が 1 つでもあるか (常体にしたい文章に、文中も含めて敬体が残っていないかを見る)"""
    depth, found = 0, False
    for w in tagger(text):
        depth += (w.surface == "「") - (w.surface == "」")
        found |= depth == 0 and w.feature.lemma in ("です", "ます")
    return found

def is_casual(text):
    """くだけた表現 (縮約形、「じゃん」「けど」、疑問の「？」) が残っているか"""
    ws = list(tagger(text))
    return any(w.feature.lemma in CASUAL_LEMMAS or w.surface in ("じゃ", "けど", "？") for w in ws)

# 判定器そのものの確認: 正解の文を正しく判定できるか
plain_ok = sum(ending_style(s) == "常体" for p, _ in PAIRS for s in split_sents(p))
polite_ok = sum(ending_style(s) == "敬体" for _, q in PAIRS + CASUAL for s in split_sents(q))
print(f"常体の文 → 常体: {plain_ok}/{sum(len(split_sents(p)) for p, _ in PAIRS)}  敬体の文 → 敬体: {polite_ok}/{sum(len(split_sents(q)) for _, q in PAIRS + CASUAL)}")
```

```python
# 1. 規則による変換: 形態素解析の活用型・活用形を見て、述語の末尾の語を作り直す
ROW = {"カ": "かきくけこ", "ガ": "がぎぐげご", "サ": "さしすせそ", "タ": "たちつてと", "ナ": "なにぬねの",
       "バ": "ばびぶべぼ", "マ": "まみむめも", "ラ": "らりるれろ", "ワ": "わいうえお"}
TA = {"カ": "いた", "ガ": "いだ", "サ": "した", "タ": "った", "ラ": "った", "ワ": "った", "ナ": "んだ", "バ": "んだ", "マ": "んだ"}

def conj(w, form):
    """動詞 (と「れる・られる」などの助動詞) の基本形と活用型から、ます形・ない形・意志形・た形の語幹を作る"""
    base, ctype = w.feature.orthBase, w.feature.cType
    stem = base[:-1]
    if ctype.startswith("五段"):
        row = ROW[ctype[3]]                                    # "五段-カ行" の「カ」
        if form == "ta":
            return stem + ("った" if w.feature.lemma == "行く" else TA[ctype[3]])
        if base in ("いらっしゃる", "おっしゃる", "くださる", "なさる", "ござる"):
            row = row.replace("り", "い")                       # いらっしゃいます
        return stem + {"renyou": row[1], "mizen": row[0], "ishi": row[4] + "う"}[form]
    if ctype.startswith(("上一段", "下一段", "助動詞-")):        # 「られる」「させる」も一段と同じ
        return stem + {"renyou": "", "mizen": "", "ishi": "よう", "ta": "た"}[form]
    if ctype == "サ行変格":
        return base[:-2] + {"renyou": "し", "mizen": "し", "ishi": "しよう", "ta": "した"}[form]
    if ctype == "カ行変格":
        k = "来" if base.startswith("来") else {"renyou": "き", "mizen": "こ", "ishi": "こ", "ta": "き"}[form]
        return k + {"renyou": "", "mizen": "", "ishi": "よう", "ta": "た"}[form]
    raise ValueError(f"未対応の活用型: {base} {ctype}")

def pos(w, pos1, lemma=None):
    return w.feature.pos1 == pos1 and (lemma is None or w.feature.lemma == lemma)

def verbish(w):
    return pos(w, "動詞") or pos(w, "助動詞") and w.feature.cType in ("助動詞-レル", "助動詞-ラレル", "助動詞-セル", "助動詞-サセル")   # 「出られる」の「られる」

def to_polite_end(t):
    """述語の末尾の語の列 t を常体 → 敬体に。(置き換える末尾の語数, 新しい文字列) を返す。0 語なら後ろに足す"""
    L, P, f = t[-1], (t[-2] if len(t) > 1 else None), t[-1].feature
    if pos(L, "助動詞", "だ"):
        return {"終止形-一般": (1, "です"), "意志推量形": (1, "でしょう")}.get(f.cForm)
    if pos(L, "助動詞", "た") and P is not None:
        if pos(P, "助動詞", "だ"):
            return 2, "でした"
        if P.feature.orthBase == "ある" and len(t) > 2 and pos(t[-3], "助動詞", "だ"):
            return 3, "でした"                                  # であった
        if pos(P, "助動詞", "ない") and len(t) > 2 and verbish(t[-3]):
            return 3, conj(t[-3], "renyou") + "ませんでした"
        if pos(P, "形容詞") and P.feature.orthBase == "ない" and P.feature.pos2 == "非自立可能":
            return 2, "ありませんでした"                         # ではなかった、広くなかった
        if pos(P, "形容詞") or pos(P, "助動詞", "たい"):
            return 0, "です"                                     # 易しかったです
        if verbish(P):
            return 2, conj(P, "renyou") + "ました"
        return None
    if pos(L, "助動詞", "ない") and P is not None and verbish(P):
        return 2, conj(P, "renyou") + "ません"
    if pos(L, "形容詞") and f.orthBase == "ない" and f.pos2 == "非自立可能" and f.cForm.startswith("終止形"):
        return 1, "ありません"
    if verbish(L):
        if f.orthBase == "ある" and P is not None and pos(P, "助動詞", "だ"):
            return 2, "でしょう" if f.cForm == "意志推量形" else "です"     # であろう、である
        if f.cForm.startswith("終止形"):
            return 1, conj(L, "renyou") + "ます"
        if f.cForm == "意志推量形":
            return 1, conj(L, "renyou") + "ましょう"
        return None
    if (pos(L, "形容詞") or pos(L, "助動詞", "たい") or pos(L, "助動詞", "らしい")) and f.cForm.startswith("終止形"):
        return 0, "です"
    return None

def to_plain_end(t):
    """述語の末尾の語の列 t を敬体 → 常体に"""
    s = [w.feature.lemma for w in t[-4:]]
    L, P = t[-1], (t[-2] if len(t) > 1 else None)
    def verb(k):
        return t[-k - 1] if len(t) > k and verbish(t[-k - 1]) else None
    if s[-4:] == ["ます", "ず", "です", "た"] and (v := verb(4)):          # ませんでした
        return (5, "なかった") if v.feature.orthBase == "ある" else (5, conj(v, "mizen") + "なかった")
    if s[-2:] == ["ます", "ず"] and (v := verb(2)):                       # ません (ありません → ない)
        return (3, "ない") if v.feature.orthBase == "ある" else (3, conj(v, "mizen") + "ない")
    if s[-2:] == ["ます", "た"] and (v := verb(2)):                       # ました
        return 3, conj(v, "ta")
    if pos(L, "助動詞", "ます") and (v := verb(1)):
        if L.feature.cForm.startswith("終止形"):
            return 2, v.feature.orthBase
        if L.feature.cForm == "意志推量形":
            return 2, conj(v, "ishi")
    if s[-2:] == ["です", "た"]:
        return 2, "だった"
    if pos(L, "助動詞", "です"):
        if L.feature.cForm == "意志推量形":
            return 1, "だろう"
        if P is not None and (pos(P, "形容詞") or pos(P, "助動詞") and P.feature.lemma != "だ"):
            return 1, ""                                          # 高いです → 高い
        return 1, "だ"
    return None

def convert(text, target):
    """target="polite" (です・ます調) か "plain" (だ・である調) へ。文末と「が・けど」の前 (常体へは接続助詞の前すべて) を変える。「」の中は変えない"""
    words = list(tagger(text))
    pieces = [w.white_space + w.surface for w in words]
    depth, ends = 0, []
    for i, w in enumerate(words):
        depth += (w.surface == "「") - (w.surface == "」")
        nxt = words[i + 1] if i + 1 < len(words) else None
        if depth or w.feature.pos1 in ("補助記号", "助詞"):
            continue
        if nxt is None or nxt.feature.pos2 in ("句点", "終助詞") or (
                nxt.feature.pos2 == "接続助詞" and (target == "plain" or nxt.surface in ("が", "けど", "けれど"))):
            ends.append(i)
    for i in reversed(ends):                                      # 後ろから置き換えれば、前の位置はずれない
        r = (to_polite_end if target == "polite" else to_plain_end)(words[:i + 1])
        if r:
            k, new = r
            if k == 0:
                pieces[i] += new
            else:
                pieces[i + 1 - k:i + 1] = [words[i + 1 - k].white_space + new]
    return "".join(pieces)

LEX = {"ちょっと": "少し", "すごく": "とても", "あとで": "後ほど", "やっぱり": "やはり"}

def casual_to_polite(text):
    """縮約形を戻し、語を置き換えてから敬体にする。尊敬語・謙譲語や、省かれた助詞は扱わない"""
    out, ws = [], list(tagger(text))
    for i, w in enumerate(ws):
        f, s = w.feature, w.surface
        if f.lemma in ("てる", "でる"):
            s = s[0] + "い" + s[1:]                                  # 待ってる → 待っている
        elif f.lemma in ("ちゃう", "じゃう"):
            s = ("て" if s[0] == "ち" else "で") + "しま" + s[2:]   # 送っちゃった → 送ってしまった
        elif f.lemma in ("とく", "どく"):
            s = ("て" if s[0] == "と" else "で") + "お" + s[1:]     # 言っといた → 言っておいた
        elif f.lemma == "じゃん":
            s = "ね"
        elif s == "じゃ" and f.pos1 == "助動詞":
            s = "では"
        elif s in ("けど", "けれど") and f.pos2 == "接続助詞":
            s = "が"
        elif s == "から" and f.pos2 == "接続助詞":
            s = "ので"
            if out and out[-1].strip() == "だ":
                out[-1] = out[-1].replace("だ", "な")                # 雨だから → 雨なので
        elif s == "悪い" and i + 1 < len(ws) and ws[i + 1].surface in ("けど", "けれど"):
            s = "申し訳ない"                                         # 「悪いけど」に限る (「天気が悪い」は変えない)
        elif s == "も" and i >= 1 and ws[i - 1].surface == "か" and (i + 1 == len(ws) or ws[i + 1].feature.pos1 == "補助記号"):
            s = "もしれない"                                         # 遅れるかも → 遅れるかもしれない
        s = LEX.get(s, s)
        out.append(w.white_space + s)
    text = "".join(out)
    question = text.endswith("？")
    text = convert(text.replace("？", "。"), "polite")
    if question and not text.endswith("か。"):
        text = text[:-1] + "か。"
    return text

rule = {"→敬体": [convert(p, "polite") for p, _ in PAIRS], "→常体": [convert(q, "plain") for _, q in PAIRS],
        "くだけ→敬語": [casual_to_polite(c) for c, _ in CASUAL]}
for k in rule:
    print(k, rule[k][:3])
```

```python
# 2. 規則を作るときに見ていない文でも試す: 青空文庫の 2 作品 (どちらも著作権の保護期間が終わっている)
# 新美南吉「ごん狐」(です・ます調で語られる) の敬体の文 → 常体、芥川龍之介「羅生門」(だ・である調) の常体の文 → 敬体
# 「」を含む文は除く (何文もある台詞の途中の文は残る。羅生門の老婆の台詞など)
import io, random, urllib.request, zipfile

def aozora(url):
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("cp932")
    text = re.split(r"-{50,}", text)[2].split("底本：")[0]
    return re.sub(r"《.+?》|［＃.+?］|｜", "", text).replace("\r\n", "\n").strip()

def pick(url, style, n=50):
    sents = [s.strip() for s in re.split(r"(?<=[。？！])|\n", aozora(url)) if s.strip()]
    sents = [s for s in sents if "「" not in s and "」" not in s and 10 <= len(s) <= 60 and s.endswith("。")
             and ending_style(s) == style]                       # もとから目標の文体の文は、変換しなくても成功になるので除く
    return random.Random(0).sample(sents, n)

gon = pick("https://www.aozora.gr.jp/cards/000121/files/628_ruby_649.zip", "敬体")
rasho = pick("https://www.aozora.gr.jp/cards/000879/files/127_ruby_150.zip", "常体")
print(len(gon), gon[:2], len(rasho), rasho[:2])
rule["ごん狐→常体"] = [convert(s, "plain") for s in gon]
rule["羅生門→敬体"] = [convert(s, "polite") for s in rasho]
```

```python
# 3. LLM による変換 (この章の「LLM で文章を要約する (長さと観点を指定する)」と同じモデル)。読み込みを含めて T4 で 3〜5 分ほど
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

set_seed(0)
name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる (大規模言語モデルの章の「オープンな LLM とチャットする (チャットテンプレート)」を参照)。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す (特殊トークンやチャットテンプレートはそのまま)
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()

PROMPT = {
    "polite": ("次の文を、です・ます調 (敬体) に書き換えてください。文末の表現だけを変え、ほかの言葉や語順は変えないでください。"
               "「」の中はそのままにしてください。書き換えた文だけを出力してください。", "駅前の店は朝九時に開く。", "駅前の店は朝九時に開きます。"),
    "plain": ("次の文を、だ・である調 (常体) に書き換えてください。文末の表現だけを変え、ほかの言葉や語順は変えないでください。"
              "「」の中はそのままにしてください。書き換えた文だけを出力してください。", "この道は夜になると暗くなります。", "この道は夜になると暗くなる。"),
    "keigo": ("次のくだけた文を、職場で目上の人に話すときの丁寧な言い方 (敬語) に書き換えてください。内容は変えないでください。"
              "書き換えた文だけを出力してください。", "その本、もう読んだ？", "その本はもうお読みになりましたか。"),
}

def llm_convert(texts, task, batch_size=16):
    rules, shot_in, shot_out = PROMPT[task]
    outs = []
    for i in range(0, len(texts), batch_size):
        convs = [[{"role": "user", "content": f"{rules}\n\n{shot_in}"}, {"role": "assistant", "content": shot_out},
                  {"role": "user", "content": f"{rules}\n\n{t}"}] for t in texts[i:i + batch_size]]
        enc = tokenizer.apply_chat_template(convs, add_generation_prompt=True, return_tensors="pt", return_dict=True, padding=True).to(llm.device)
        with torch.no_grad():
            out = llm.generate(**enc, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        outs += [o.strip().split("\n")[0] for o in tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return outs

llm_out = {"→敬体": llm_convert([p for p, _ in PAIRS], "polite"), "→常体": llm_convert([q for _, q in PAIRS], "plain"),
           "くだけ→敬語": llm_convert([c for c, _ in CASUAL], "keigo"),
           "ごん狐→常体": llm_convert(gon, "plain"), "羅生門→敬体": llm_convert(rasho, "polite")}
print(f"GPU メモリの最大使用量: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
for k in llm_out:
    print(k, llm_out[k][:3])
```

```python
# 4. 評価: 文体の成功率、正解との一致、内容語の保持と追加、正解との編集距離
import pandas as pd

STOP = {"有る", "居る", "為る", "成る", "無い", "下さる", "仕舞う", "置く", "事", "物", "の", "良い"}

def content(text):
    """内容語の語彙素 (表記の揺れ「分かる/わかる」は同じ語彙素になる)。補助的な動詞や形式名詞は除く"""
    return {w.feature.lemma for w in tagger(text) if w.feature.pos1 in ("名詞", "動詞", "形容詞", "副詞", "形状詞") and w.feature.lemma not in STOP}

def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]

TASKS = {  # 課題: (入力, 正解 (なければ None), 目標の文体)
    "→敬体": ([p for p, _ in PAIRS], [[q] for _, q in PAIRS], "敬体"),
    "→常体": ([q for _, q in PAIRS], [[p, p.replace("である", "だ")] for p, _ in PAIRS], "常体"),   # 「である」「だ」どちらも正解
    "くだけ→敬語": ([c for c, _ in CASUAL], None, "敬体"),
    "ごん狐→常体": (gon, None, "常体"),
    "羅生門→敬体": (rasho, None, "敬体"),
}
# 尊敬語・謙譲語の目印 (簡単な正規表現で、すべては拾えない)。「申し訳」と「〜ておく」(ておいた、でおく) の「お」は除く
KEIGO = re.compile(r"ご覧|いらっしゃ|おっしゃ|なさ|いただ|申し(?!訳)|伺|参り|存じ|おり(ま|ませ)|いたし"
                   r"|(?<![てで])お[^\s。、]{1,4}(し|にな|ください)|(?<=[てで])お(?![いかきくけこ])[^\s。、]{1,4}(し|にな|ください)"
                   r"|ご[一-龥]{1,3}(し|にな|さ|ください)")

rows, details = [], []
for method, outputs in [("規則", rule), ("LLM", llm_out)]:
    for task, (srcs, refs, target) in TASKS.items():
        for j, (src, out) in enumerate(zip(srcs, outputs[task])):
            ok = all(ending_style(s) == target for s in split_sents(out))     # 敬体へ: どの文の文末も敬体
            if target == "常体":
                ok = ok and not has_polite(out)                               # 常体へ: 文中の「〜ましたが」なども残っていない
            if task == "くだけ→敬語":
                ok = ok and not is_casual(out)
            cs, co = content(src), content(out)
            details.append({"方法": method, "課題": task, "入力": src, "出力": out, "成功": ok,
                            "一致": float("nan") if refs is None else float(out in refs[j]),
                            "保持": len(cs & co) / max(len(cs), 1), "追加": len(co - cs) / max(len(co), 1),
                            "距離": float("nan") if refs is None else min(edit_distance(out, r) for r in refs[j]),
                            "敬語の語": bool(KEIGO.search(out))})
d = pd.DataFrame(details)
summary = d.groupby(["課題", "方法"], sort=False).agg(件数=("成功", "size"), 文体の成功率=("成功", "mean"), 正解と一致=("一致", "mean"),
                                                   内容語の保持=("保持", "mean"), 内容語の追加=("追加", "mean"), 正解との距離=("距離", "mean"),
                                                   敬語の語を含む=("敬語の語", "mean"))
summary.round(2)
```

```python
# 5. 失敗と書き換えすぎの例を読む
pd.set_option("display.max_colwidth", 60)
print("■ 規則: 目標の文体にならなかった文")
print(d[(d.方法 == "規則") & ~d.成功][["課題", "入力", "出力"]].to_string(index=False))
print("■ 規則: 判定器は成功としたが、読むと誤っている文 (老婆の古風な台詞)")
print(d[(d.方法 == "規則") & d.成功 & d.入力.str.contains("であろ。|ぞよ。")][["入力", "出力"]].to_string(index=False))
llm_d = d[(d.方法 == "LLM") & (d.課題 != "くだけ→敬語")]
print(f"\n■ LLM: 目標の文体にならなかった文 ({(~llm_d.成功).sum()} 件のうち先頭 10 件)")
print(llm_d[~llm_d.成功][["課題", "入力", "出力"]].head(10).to_string(index=False))
changed = llm_d[llm_d.成功 & ((llm_d.保持 < 1) | (llm_d.追加 > 0))]
print(f"\n■ LLM: 文体は変わったが、内容語を足した・落とした文 ({len(changed)} 件のうち先頭 6 件)")
print(changed[["課題", "入力", "出力"]].head(6).to_string(index=False))
print("\n■ くだけた文 → 敬語")
for (c, ref), r_out, l_out in zip(CASUAL, rule["くだけ→敬語"], llm_out["くだけ→敬語"]):
    print(f"{c}\n   規則: {r_out}\n   LLM : {l_out}\n   例  : {ref}")
```

- 文体の変換 (style transfer) は、内容を保ったまま書き方だけを変える書き換え。この章の「文章をやさしい日本語に書き換える (テキスト平易化)」と同じく、文体が変わったか、意味が保たれているか、余計な書き換えがないかを分けて測る。です・ます調とだ・である調の変換は文末だけが変わるはずなので、平易化のような含意 (NLI) の判定でなく、内容語の一致と正解との編集距離で書き換えすぎを見つけられる。平易化のプロンプトの「文末は「です」「ます」にする」が守られたかも、上の判定器で確かめられる。
- 規則による方法は、形態素解析の活用型と活用形 (日本語の形態素解析と文分割の章の「動詞・形容詞の活用型と活用形を調べる」) から「行った → 行きました」「広くない → 広くありません」を作る。作例では両方向とも 30/30 が正解と一致したが、規則を作りながら見ていた文なので楽観的な数字である。見ていない青空文庫の文 (変換が要る文だけを選んだ) では、ごん狐 → 常体が 50/50、羅生門 → 敬体が 49/50 で、失敗は古い否定の「ぬ」だった。判定器は成功としたが、老婆の台詞の「であろ → です」(推量が落ちた) や「これだぞよ → これですぞよ」のような誤りもある。
- LLM は「文末の表現だけを変え」と指示しても書き換えすぎる。作例で正解と一致したのは →敬体 23/30、→常体 22/30 で、「から → ため」「分かりません → 不明だ」「歩いて通っています → 歩いている」(「通う」が落ちた) のように、文体は変わったのに内容語が増減した文が 23 あった。
- LLM は変換し損ねも多い。ごん狐 → 常体は 41/50 で、失敗した 9 文のほとんどは文末が「〜ました」「〜ます」のまま、表記の変更 (「だまって → 黙って」) や誤り (「まきついた → まきつくった」) だけが入っていた。「（すでにだ・である調になっていますので、変更は不要です。）」と説明を付け足した文もある。規則なら、文中の「〜ますと」「〜ましたが」もまとめて変えられる。
- くだけた文 → 敬語では、規則は丁寧語と縮約形 (「待ってる」「言っといた」) の復元しかできず、尊敬語・謙譲語は 0/12、省かれた助詞も戻らない。LLM は目印の正規表現で数えて 10/12 で尊敬語・謙譲語を使ったが (「ご出席されますか」のように適否が分かれる表現もある)、内容語の保持は 0.39 で、「遅れる → 遅刻する」「やめとこう → 中止にいたしましょうか」のように語や意図まで変わる。敬語では「見る → ご覧になる」のように語が替わるのが正しいこともあるので、内容語の一致では測れず、含意の判定や人の確認が要る。
- 判定器は正解の文をすべて正しく判定した (常体 31/31、敬体 44/44) が、文末しか見ず、規則と同じ形態素解析器を使うので、規則の誤り (上の「であろ」) を見逃す。くだけた表現の判定 (is_casual) も縮約形や「じゃん」「けど」「？」しか見ず、「かも」や省かれた助詞は拾わないので、規則のくだけ → 敬語の成功率 1.00 は甘い。尊敬語・謙譲語の目印も簡単な正規表現で、すべては拾えない。機械翻訳の章の「タグを付けて訳文の文体を制御する (丁寧さの制御)」も、正規表現の判定器で文体の成功率を測っている。
- 評価は 30〜50 文、1 モデル、貪欲法での結果で、数文の差はばらつきの範囲と考える。実際には、規則で変換して判定器が失敗とした文だけを LLM に回す、LLM の出力を判定器と内容語の一致で検査する、といった組み合わせが考えられる (試していない)。例文はこのレシピのための作例、青空文庫の 2 作品は著者の没年 (1943 年、1927 年) から著作権の保護期間が終わっている。sarashina2.2 は SB Intuitions のモデルで MIT ライセンス。
