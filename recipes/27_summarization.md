# 要約と平易化

文章を短くまとめる、やさしく書き換える。抽出型と生成型の要約、3 行要約、話題の切れ目での区切り、口コミからの意見の要約、議事録の作成、規約の新旧の比較、見出し生成（T5 のファインチューニングを含む）、テキスト平易化と語彙平易化、文体の変換と、その評価。要約・生成タスクの理論や背景については、姉妹編『[自然言語処理の教科書](https://github.com/mamorlis/nlpbook)』の第 4 章も参照。

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

- 文をノード、類似度を辺の重みとするグラフを作り、PageRank で中心的な文を選ぶ。無関係な文（最後の文）はどの文とも似ていないのでスコアが低い。
- 類似度の計算を TF-IDF から文埋め込みに替えると、言い換えにも強くなる。
- ニュース記事では「先頭の 3 文を取る」(Lead-3) が非常に強いベースラインになる。要約手法を評価するときは必ず比較する（この章の「先頭の数文を要約とする (Lead-3) ベースラインと比べる」を参照）。
- 英語を含む各種の抽出型要約アルゴリズムは `sumy` にまとまっている。

## 要約を ROUGE で評価する（日本語対応） {rouge_ja}

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

- ニュース記事は重要なことを最初に書く（逆ピラミッド型）ので、Lead-3 は非常に強い。凝った手法が Lead-3 に勝てないことは珍しくない。
- 比べている LexRank と、日本語のトークナイザを渡した ROUGE の使い方は、この章の「重要な文を選んで抽出型要約を作る (LexRank)」と「要約を ROUGE で評価する（日本語対応）」で説明した。
- 新しい要約手法を評価するときは、必ず Lead-3 とランダムを並べて報告する。
- ここではタイトルを参照の代わりにした簡易な評価。本来は人手で書いた参照要約（日本語なら livedoor の 3 行要約、XL-Sum の日本語部分、WikiLingua など）を使う。
- 小説、論文、議事録のように、重要な内容が先頭にあるとは限らない文書では、Lead の優位は小さくなる。

## LLM で文章を要約する（長さと観点を指定する） {llm_summary gpu}

文章の内容を、自分の言葉でまとめ直した要約（生成型要約）が欲しい。長さや、何に注目するかを指定したい。

```python
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"   # 3B の日本語モデル。GPU（T4 など）で 1 要約あたり数秒〜10 秒ほど（VRAM 約 7GB）
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる（大規模言語モデル (LLM) の章の「オープンな LLM とチャットする（チャットテンプレート）」を参照）。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
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

- 生成型要約は、文をつなぎ直したり言い換えたりできるので、抽出型より短く自然にまとまる。そのかわり、原文にないことを書く（ハルシネーション）おそれがある。数値と固有名詞は必ず原文と照らし合わせる（上の最初の出力でも、原文の「来年 4 月」が具体的な年に書き換えられている）。
- 文字数の指定は目安にしかならない。厳密な上限が必要なら、生成後に文字数を確かめて、超えていたら「もっと短く」と再生成する。
- 「誰のための、何のための要約か」を書くと質が上がる。同じ文章でも、目的が違えば残すべき情報が違う。
- 小さいモデルは、長い入力の後半を見落としやすい。長い文書は、この章の「長い文書を分割して要約する」を参照。

## ニュース記事を 3 行に要約する {three_line_summary gpu}

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
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる（大規模言語モデル (LLM) の章の「オープンな LLM とチャットする（チャットテンプレート）」を参照）。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()

def chat(messages, max_new_tokens=160):
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(llm.device)
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
- この例では、NLI は「費用は年間約 60 億円…」の文を矛盾 0.95 と判定し、桁違いの数値（6 億と 60 億）の誤りを見つけた。ただし、数値の誤りをいつも見つけられるとは限らない。数値・日付・固有名詞は、文字列として原文に存在するかを別に検査する。
- 原文が長いと NLI モデルの入力長を超える。原文を段落に分け、要約文ごとに最も含意の高い段落を採用する。
- 忠実であることと、良い要約であることは別。重要な内容を落としていないか（網羅性）は、参照要約との比較や人手で確かめる。

## 長い文書を分割して要約する {long_document_summary gpu}

モデルに一度に入らない（あるいは入れると後半を読み飛ばす）長い文書を要約したい。分割して要約し、それをまとめ直す。

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
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる（大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」を参照）。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
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

text = aozora("https://www.aozora.gr.jp/cards/000035/files/1567_ruby_4948.zip")   # 走れメロス（約 1 万字）
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
- 3B クラスのモデルでは、部分要約に話の取り違えや人名の誤りが混ざることがある（上の出力でも、短刀を持って王城に入り捕らえられたメロスが「王に短刀を差し出した」ことになり、最終要約では「短刀を差し出して抵抗する」に変わっている。妹の結婚式を翌日に繰り上げる場面は、部分要約の段階で「延期」と取り違え、最終要約にもそのまま残った）。誤りは最終要約にそのまま引き継がれるので、重要な用途ではより大きなモデルを使うか、この章の「要約が原文と食い違っていないか検査する」と組み合わせる。
- 前の部分の要約を次の部分の入力に加えていく方法（逐次的な洗練）だと、話のつながりは保ちやすいが、並列化はできず、誤りが後ろに伝わる。
- 長い入力をそのまま読めるモデルでも、文書が長くなるほど中ほどの内容を見落としやすい。分割は今でも有効な方法。

## 大量の口コミから観点ごとの賛否をまとめる（意見の要約） {opinion_summarization gpu}

1 つのホテルについての数百件の口コミから、「立地は 9 割が高評価、部屋の狭さへの不満が多い」のように、観点ごとの賛否の割合と代表的な意見をまとめた要約を作りたい。要約が多数派の意見を正しく伝えているか、少数の強い苦情を誇張していないかも確かめたい。

```python
import numpy as np
import pandas as pd

# 架空のビジネスホテルの口コミ（このレシピのために作ったもの）。観点ごとに言及する確率と肯定の確率を決めて文を組み合わせるので、正解の割合がわかる
ASPECTS = {   # 値は、言及する確率と、言及したときに肯定である確率
    "立地": (0.5, 0.9), "部屋の広さ": (0.4, 0.2), "清潔さ": (0.35, 0.93), "朝食": (0.4, 0.7), "接客": (0.35, 0.55), "価格": (0.3, 0.75)}
PHRASES = {
    ("立地", "肯定"): ["駅から歩いて 3 分ほどで、とても便利でした。", "駅前なので、雨の日でも移動が楽でした。", "近くにコンビニや飲食店が多く、立地は申し分ありません。",
                      "空港バスの乗り場が目の前で助かりました。", "観光の拠点にちょうどよい場所です。", "繁華街まで歩いて行けるのがよかったです。"],
    ("立地", "否定"): ["駅から少し歩くうえ、坂道がきつかったです。", "周りに食事ができる店が少なく、不便でした。", "入口がわかりにくく、地図を見ても迷いました。"],
    ("部屋の広さ", "肯定"): ["部屋は思ったより広く、スーツケースも広げられました。", "部屋がゆったりしていて、くつろげました。", "ビジネスホテルにしては部屋が広めでした。"],
    ("部屋の広さ", "否定"): ["部屋が狭く、スーツケースを広げる場所がありませんでした。", "ベッドと机だけでいっぱいの、かなり狭い部屋でした。", "二人で泊まるには部屋が窮屈でした。",
                          "部屋は狭めで、荷物の置き場に困りました。", "写真で見るより部屋が小さく感じました。", "部屋が狭いうえに窓も小さく、圧迫感がありました。"],
    ("清潔さ", "肯定"): ["部屋は清潔で、掃除が行き届いていました。", "水回りもきれいで、気持ちよく使えました。", "シーツもタオルも清潔でした。",
                        "古い建物ですが、清掃はしっかりしていました。", "ほこり一つなく、きれいな部屋でした。"],
    ("清潔さ", "否定"): ["夜中に部屋でゴキブリが出ました。本当に最悪で、二度と泊まりません。", "浴室にカビがあり、とても不潔でした。信じられません。",
                        "ベッドに前の客の髪の毛が残っていて、ぞっとしました。"],   # 少数だが強い言葉の苦情
    ("朝食", "肯定"): ["朝食のバイキングは品数が多く、満足しました。", "朝食の焼きたてのパンがおいしかったです。", "地元の食材を使った朝食がよかったです。",
                      "朝食は無料なのに品数が多く、得をした気分でした。"],
    ("朝食", "否定"): ["朝食は品数が少なく、少し残念でした。", "朝食会場が混んでいて、30 分ほど待たされました。", "朝食の料理が冷めていました。"],
    ("接客", "肯定"): ["フロントの方の対応がとても丁寧でした。", "チェックインのとき、スタッフが笑顔で迎えてくれました。", "荷物を早めに預かってもらえて助かりました。",
                      "周辺のおすすめの店を親切に教えてくれました。"],
    ("接客", "否定"): ["フロントの対応がそっけなく、感じがよくありませんでした。", "チェックインに 20 分以上かかり、説明も不十分でした。",
                      "電話で問い合わせたら、たらい回しにされました。", "スタッフの言葉づかいが少し気になりました。"],
    ("価格", "肯定"): ["料金が手ごろで、コストパフォーマンスがよいです。", "平日は安く泊まれるので、出張で毎回使っています。", "値段のわりに設備が整っていました。"],
    ("価格", "否定"): ["週末は料金が高く、割高に感じました。", "駐車場が有料で、思ったより高くつきました。", "この内容でこの料金は、少し高いと思います。"],
}
OPENERS = ["出張で 1 泊しました。", "家族旅行で利用しました。", "友人と 2 泊しました。", "", "", ""]

def make_reviews(n, seed=0):
    rng = np.random.default_rng(seed)
    reviews, gold = [], []
    while len(reviews) < n:
        labels = {a: ("肯定" if rng.random() < p_pos else "否定") for a, (p_mention, p_pos) in ASPECTS.items() if rng.random() < p_mention}
        if not labels:
            continue
        body = "".join(rng.choice(PHRASES[(a, labels[a])]) for a in rng.permutation(list(labels)))
        pos = sum(v == "肯定" for v in labels.values())
        closer = "また利用したいです。" if pos > len(labels) / 2 else "次は別のホテルにします。" if pos < len(labels) / 2 else ""
        reviews.append(rng.choice(OPENERS) + body + closer)
        gold.append(labels)
    return reviews, gold

reviews, gold = make_reviews(200)
gold_df = pd.DataFrame([{"review": i, "aspect": a, "polarity": p} for i, g in enumerate(gold) for a, p in g.items()])
truth = gold_df.groupby("aspect")["polarity"].value_counts().unstack(fill_value=0).reindex(list(ASPECTS))
truth["肯定率"] = (truth["肯定"] / (truth["肯定"] + truth["否定"])).round(3)
print(len(reviews), "件、", sum(map(len, reviews)), "字")
print(truth)
for r in reviews[:3]:
    print("-", r)
```

```python
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"   # この章のほかのレシピと同じ 3B の日本語モデル（MIT ライセンス）。初回はダウンロード（約 6.4GB）を含めて数分
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる（大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」を参照）。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
tokenizer.padding_side = "left"      # 複数の入力をまとめて生成するときは左に詰める
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()

def chat_batch(messages_list, max_new_tokens=400):
    texts = [tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages_list]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(llm.device)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return [tokenizer.decode(o[inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]

chat = lambda prompt, max_new_tokens=400: chat_batch([[{"role": "user", "content": prompt}]], max_new_tokens)[0]
print("文脈長の上限:", llm.config.max_position_embeddings, "トークン")
```


```python
# 方法 1: 全部の口コミを 1 回で LLM に渡し、割合と代表的な意見を書かせる。指示は長い入力の後ろに置く
FORMAT = "\n".join(f"- {a}: 肯定 ○%、否定 ○%。代表的な意見" for a in ASPECTS) + "\nまとめ: 全体の評価を 1〜2 文で"
def one_shot_prompt(revs):
    joined = "\n".join(f"[{i}] {r}" for i, r in enumerate(revs, 1))
    return (f"次は、あるビジネスホテルの口コミ {len(revs)} 件です。\n\n{joined}\n\n上の口コミ {len(revs)} 件を要約してください。"
            f"観点ごとに、その観点に触れた口コミのうち肯定的なものと否定的なものの割合と、代表的な意見を書いてください。次の形式で、○ には数字を入れてください。\n{FORMAT}")

for n in [200, 300]:
    print(f"口コミ {n} 件のプロンプト:", len(tokenizer(one_shot_prompt(make_reviews(n)[0]))["input_ids"]), "トークン")

# 口コミの並び順だけを変えて 3 回作る。T4 で 1 回 30 秒〜1 分ほど
one_shot = {}
for seed in [0, 1, 2]:
    order = np.random.default_rng(seed).permutation(len(reviews)) if seed else np.arange(len(reviews))
    one_shot[f"一括（順序 {seed}）"] = out = chat(one_shot_prompt([reviews[i] for i in order]))
    print(f"--- 一括（順序 {seed}）\n{out}")
```

```python
import json
import re

# 方法 2 の段階 1: 口コミを 1 件ずつ LLM に読ませ、観点・評価・根拠の文を JSON で抜き出す
INSTRUCTION = (f"次のホテルの口コミが、観点（{'、'.join(ASPECTS)}）のどれについて、肯定と否定のどちらの評価をしているかを JSON のリストで答えてください。"
               "触れていない観点は含めないでください。根拠には口コミの文をそのまま抜き出してください。JSON 以外は出力しないでください。")
EXAMPLE_IN = "駅から近くて便利でした。ただ、部屋は清掃が行き届いていませんでした。朝食はまあまあです。"
EXAMPLE_OUT = json.dumps([{"観点": "立地", "評価": "肯定", "根拠": "駅から近くて便利でした。"},
                          {"観点": "清潔さ", "評価": "否定", "根拠": "部屋は清掃が行き届いていませんでした。"}], ensure_ascii=False)

def messages(review):
    return [{"role": "user", "content": f"{INSTRUCTION}\n\n口コミ: {EXAMPLE_IN}"}, {"role": "assistant", "content": EXAMPLE_OUT},
            {"role": "user", "content": f"{INSTRUCTION}\n\n口コミ: {review}"}]

def validate(output, review):
    """観点と評価が決められた値の項目だけを残す。根拠が口コミにそのまま出てくるかを ok に記録する。JSON が読めなければ None"""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", output.strip(), flags=re.MULTILINE)
    try:
        items, _ = json.JSONDecoder().raw_decode(text[text.index("["):])
    except ValueError:
        return None
    items = [x for x in items if isinstance(x, dict) and x.get("観点") in ASPECTS and x.get("評価") in ("肯定", "否定")]
    for x in items:
        ev = str(x.get("根拠", "")).strip().rstrip("。．.")
        x["ok"] = len(ev) >= 5 and ev in review
    return items

# 16 件ずつまとめて生成する。T4 で 200 件に 3〜5 分ほど
outputs = []
for i in range(0, len(reviews), 16):
    outputs += chat_batch([messages(r) for r in reviews[i:i + 16]], max_new_tokens=300)
extracted = [validate(o, r) for o, r in zip(outputs, reviews)]
failed = [i for i, e in enumerate(extracted) if e is None]
print("JSON が読めなかった口コミ:", len(failed), "例:", [outputs[i][:120] for i in failed[:2]])
# 読めなかったものは、誤りを伝えて 1 回だけ出し直させる
retry = chat_batch([messages(reviews[i]) + [{"role": "assistant", "content": outputs[i]},
                    {"role": "user", "content": "出力が JSON として読めません。例と同じ形の JSON だけを出力し直してください。"}] for i in failed], max_new_tokens=300) if failed else []
for i, o in zip(failed, retry):
    extracted[i] = validate(o, reviews[i])
print("出し直しても読めなかった口コミ:", sum(e is None for e in extracted))

pred_df = pd.DataFrame([{"review": i, "aspect": x["観点"], "polarity": x["評価"], "evidence": str(x.get("根拠", "")).strip(), "ok": x["ok"]}
                        for i, e in enumerate(extracted) for x in (e or [])]).drop_duplicates(["review", "aspect"])
print("根拠が口コミにそのまま出てこない項目:", int((~pred_df["ok"]).sum()), "/", len(pred_df))

# 正解と比べる（実際の口コミでは、数十件に人が正解を付けて確かめる）
m = gold_df.merge(pred_df, on=["review", "aspect"], how="outer", suffixes=("_gold", "_pred"), indicator=True)
both = m[m["_merge"] == "both"]
print(f"観点の検出: 再現率 {len(both) / len(gold_df):.3f}  適合率 {len(both) / len(pred_df):.3f}  "
      f"| 検出できた観点の極性の正解率 {(both['polarity_gold'] == both['polarity_pred']).mean():.3f}")
errors = m[(m["_merge"] != "both") | (m["polarity_gold"] != m["polarity_pred"])]
print(errors["_merge"].map({"left_only": "見落とし", "right_only": "余分", "both": "極性の誤り"}).value_counts().to_dict(),
      "見落とした観点:", errors[errors["_merge"] == "left_only"].groupby("aspect")["polarity_gold"].value_counts().to_dict())
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from statsmodels.stats.proportion import proportion_confint

# 段階 2: 集計はコードで行う。割合には 95% の信頼区間（下限〜上限）を付け、代表的な意見は、根拠の文のうちグループの中心に近いものを選ぶ
def representative(texts):
    if len(texts) <= 1:
        return texts[0] if texts else "なし"
    X = TfidfVectorizer(analyzer="char", ngram_range=(2, 3)).fit_transform(texts)
    return texts[int(np.argmax(X @ np.asarray(X.mean(axis=0)).T))]

rows = []
for a in ASPECTS:
    d = pred_df[pred_df["aspect"] == a]
    pos, n = int((d["polarity"] == "肯定").sum()), len(d)
    lo, hi = proportion_confint(pos, n, method="wilson")
    ev = d[d["ok"]]      # 代表的な意見には、口コミにそのまま出てくる根拠だけを使う
    rows.append({"観点": a, "言及": n, "肯定": pos, "否定": n - pos, "肯定率": round(pos / n, 3), "下限": round(lo, 2), "上限": round(hi, 2),
                 "肯定の例": representative(ev[ev["polarity"] == "肯定"]["evidence"].tolist()),
                 "否定の例": representative(ev[ev["polarity"] == "否定"]["evidence"].tolist())})
table = pd.DataFrame(rows).set_index("観点")
print(table[["言及", "肯定", "否定", "肯定率", "下限", "上限"]])
print(table[["肯定の例", "否定の例"]].to_string())

# 段階 3: 集計表だけを LLM に渡して文章にする。数字はコードが計算したものを使わせ、件数も添える
# 多数派かどうかもコードで判定して渡す。信頼区間が 0.5 をまたぐなら「賛否が分かれる」とする
verdict = lambda r: "肯定が多数" if r["下限"] > 0.5 else "否定が多数" if r["上限"] < 0.5 else "賛否が分かれる"
digest = "\n".join(f"- {a}: 言及 {r['言及']} 件、肯定 {r['肯定']} 件 ({r['肯定率']:.0%})、否定 {r['否定']} 件 ({1 - r['肯定率']:.0%})、{verdict(r)}。"
                   f"肯定の例「{r['肯定の例']}」 否定の例「{r['否定の例']}」" for a, r in table.iterrows())
staged_prompt = (f"次は、あるビジネスホテルの口コミ {len(reviews)} 件を観点ごとに集計した結果です。\n\n{digest}\n\n"
                 "この集計だけをもとに、口コミの要約を書いてください。割合は集計の値をそのまま使ってください。"
                 "件数の少ない意見は「一部」「少数」などと書き、多数派の意見のように書かないでください。まとめでは、各観点の「肯定が多数」「否定が多数」「賛否が分かれる」の判定に従ってください。"
                 f"次の形式で、○ には数字を入れてください。\n{FORMAT}")
staged = chat(staged_prompt)
print(f"--- 段階的\n{staged}")
```

```python
# 要約の主張を検査する: 「立地については」のような観点の書き出しから次の観点までを 1 区切りとし、その中で最初に肯定か否定の語と同じ句にある割合を読み取る
MARK = r"(立地|部屋の広さ|広さ|清潔さ|朝食|接客|価格)(?:に関しては|については|は|[:：])"
PCT, POS, NEG = r"(\d+(?:\.\d+)?)\s*[%％]", "肯定|満足|好意|好評|高く評価|良い|よい", "否定|不満|苦情"

def claims(summary):
    found, marks = {}, list(re.finditer(MARK, summary))
    for k, mk in enumerate(marks):
        a = "部屋の広さ" if mk.group(1) == "広さ" else mk.group(1)
        segment = summary[mk.start():marks[k + 1].start() if k + 1 < len(marks) else len(summary)]
        for clause in re.split(r"[。、\n]", segment):     # 「肯定 ○%、否定 ○%」のように並ぶので、読点でも区切る
            if a in found:
                break
            for words, sign in [(POS, 1), (NEG, -1)]:   # 「肯定 ○%」「○% の口コミが満足」など。否定の割合なら 1 から引く
                if m := re.search(rf"(?:{words}).*?{PCT}", clause) or re.search(rf"{PCT}.*?(?:{words})", clause):
                    v = float(m.group(1)) / 100
                    found[a] = (v if sign > 0 else 1 - v, clause.strip())
                    break
    return found

def side(p):   # 6 割以上を多数派とみなす
    return "肯定が多数" if p >= 0.6 else "否定が多数" if p <= 0.4 else "賛否が分かれる"

summaries = {**one_shot, "段階的": staged}
parsed = {k: claims(s) for k, s in summaries.items()}
check = pd.DataFrame({"正解": truth["肯定率"], "段階 1 の集計": table["肯定率"],
                      **{k: pd.Series({a: v for a, (v, _) in c.items()}, dtype=float) for k, c in parsed.items()}}).reindex(list(ASPECTS))
print("要約が主張する肯定率（NaN は割合を読み取れなかった観点）")
print(check.round(2))
for k in check.columns[1:]:
    ok = check[k].notna()
    print(f"{k:12} 読み取れた観点 {ok.sum()}  正解との差の平均 {(check[k] - check['正解'])[ok].abs().mean():.3f}  "
          f"多数派の向きが合う観点 {sum(side(p) == side(t) for p, t in zip(check[k][ok], check['正解'][ok]))}")
print("\n読み取った句（一括（順序 0））:")
for a, (v, s) in parsed["一括（順序 0）"].items():
    print(f"  {a}: {v:.0%} ← {s}")

# 少数の強い苦情（清潔さの否定）に、要約がどう触れているか
complaint = re.compile(r"ゴキブリ|カビ|髪の毛|不潔|害虫")
print(f"\n清潔さの否定は正解で {1 - truth.loc['清潔さ', '肯定率']:.0%}")
for k, s in summaries.items():
    print(f"{k}: {[l.strip()[:80] for l in re.split(r'(?<=。)|\n', s) if complaint.search(l)]}")
```

```python
# 割合の数字が正しくても、まとめの文で多数派を言い違えることがある。「まとめ」「総じて」「全体」で始まる最初の文から最後までをまとめの部分とし、
# 観点と評価の言葉が同じ句にあれば、その観点についての主張とみなして検査する
TOPIC = {"立地": "立地", "部屋の広さ": "部屋|広さ", "清潔さ": "清潔", "朝食": "朝食", "接客": "接客|スタッフ", "価格": "価格|料金"}
STANCE = [("賛否が分かれる", "賛否|両論|分かれ|半々|ばらつき"), ("否定が多数", "否定的な意見が多|不満を持つ声が多|不満が多|改善の余地|不評"),
          ("肯定が多数", "肯定的|好評|高評価|評価が高|評価は高|満足")]

def overall_claims(summary):
    found, in_summary = [], False
    for sent in re.split(r"(?<=。)|\n", summary):
        in_summary = in_summary or bool(re.match(r"\s*(?:まとめ|総じて|全体)", sent))
        if not in_summary:
            continue
        for clause in re.split(r"おり、|ですが、|ものの、|一方、?|が、", sent):
            stance = next((label for label, pat in STANCE if re.search(pat, clause)), None)
            for a, key in TOPIC.items():   # 「価格を除けば」「価格以外は」の観点は除く
                if stance and re.search(key, clause) and not re.search(rf"(?:{key})(?:を除|以外)", clause):
                    found.append({"観点": a, "主張": stance, "正解": side(truth.loc[a, "肯定率"]), "句": clause.strip()[:30]})
    return pd.DataFrame(found, columns=["観点", "主張", "正解", "句"])

for k, s in summaries.items():
    c = overall_claims(s)
    print(f"--- {k}: 合う {int((c['主張'] == c['正解']).sum())} / 主張 {len(c)}")
    if (c["主張"] != c["正解"]).any():
        print(c[c["主張"] != c["正解"]].to_string(index=False))

# 段階的な方法では、集計の判定と食い違う主張を伝えて、1 回だけ書き直させる
# 多数派の基準は 2 つある。上の検査は、正解の肯定率と 6 割の基準（side）で採点する（要約が書く割合には件数がなく、区間を作れないため）。
# 書き直しの指示は、LLM に渡した集計の信頼区間（verdict）で判定する。実際の口コミには正解がないので、使えるのは verdict のほうである
c = overall_claims(staged)
wrong = [f"「{r.観点}」を「{r.主張}」と書いていますが、集計では「{verdict(table.loc[r.観点])}」です。" for r in c.itertuples() if r.主張 != verdict(table.loc[r.観点])]
if wrong:
    staged = chat_batch([[{"role": "user", "content": staged_prompt}, {"role": "assistant", "content": staged},
                          {"role": "user", "content": "まとめに集計と合わない記述があります。" + "".join(wrong) + "形式はそのままで、全体を書き直してください。"}]])[0]
    c = overall_claims(staged)
    print(f"\n--- 書き直した段階的な要約: 合う {int((c['主張'] == c['正解']).sum())} / 主張 {len(c)}、割合 {[f'{a} {v:.0%}' for a, (v, _) in claims(staged).items()]}")
    print(staged.splitlines()[-1])
    print(c[c["主張"] != c["正解"]].to_string(index=False))
print(f"GPU メモリの最大使用量: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

```python
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast

# 参考: この章の「要約が原文と食い違っていないか検査する」の NLI で、割合の主張を確かめられるか試す
nli_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
# transformers 5 系の DebertaV2Tokenizer は全角の記号や英数字を [UNK] にしてしまう (issue #49622) ので、tokenizer.json から直接読む
nli_tok = PreTrainedTokenizerFast.from_pretrained(nli_name)
nli = AutoModelForSequenceClassification.from_pretrained(nli_name).eval()
pairs = [("接客は肯定的な意見と否定的な意見がほぼ半々である。", "接客は肯定的な意見が多い。"),
         ("接客は肯定的な意見と否定的な意見がほぼ半々である。", "接客は賛否が分かれる。"),
         ("清潔さは肯定的な意見が大多数である。", "清潔さには不満が多い。"),
         ("清潔さは肯定的な意見が大多数である。", "清潔さについては改善の余地があると感じる宿泊客もいる。")]
with torch.no_grad():
    for premise, hypothesis in pairs:
        p = nli(**nli_tok(premise, hypothesis, return_tensors="pt")).logits.softmax(-1)[0]
        print(f"含意 {p[0]:.2f} 中立 {p[1]:.2f} 矛盾 {p[2]:.2f} | {premise} → {hypothesis}")
```

- 全部を 1 回で渡す方法は、口コミ 200 件（13,508 字）で 7,045 トークンになり、このモデルの文脈長 8,192 トークンに出力の分を残して入るぎりぎりである（300 件では 10,623 トークンで入らない）。出てきた割合は「約 60%」のような 10% 刻みの値で、数えた値とは考えにくい。並び順だけを変えた 3 回で、立地は 3 回とも 60%（正解 89%）、価格は 30〜40%（正解 67%）、接客は 70% と 50%（正解 50%）で、正解との差の平均は 0.15〜0.20 だった。順序 0 では「部屋の清潔さや広さ」と 2 つの観点をまとめて書き、清潔さの割合は読み取れなかった。
- 少数の強い苦情（清潔さの否定は正解で 10%）について、一括の 3 回のどれもが苦情の中身（ゴキブリ、カビなど）には触れなかった。そのかわり順序 0 は、清潔さを部屋の広さとまとめて「否定的な意見が約 70%」とし、まとめでは肯定が 9 割の清潔さや 7 割の朝食まで「改善の余地がある」とした。順序 1 は肯定が 7 割近い価格を「不満を持つ声が多かった」とし、順序 2 は賛否が半々の接客を「概ね好評」とした。少数意見の中身を落とすことと、全体の評価をゆがめることの両方が起きる。
- 段階的な方法の抽出は、観点の再現率 0.920、適合率 1.000、極性の正解率 1.000 で（JSON が読めなかった 18 件は、出し直しで 3 件に減った）、集計した肯定率と正解の差は平均 0.022 だった。ただし、件数は少ないが見落としは偏っていて、立地の否定は 11 件中 5 件を見落とした（肯定は 90 件中 11 件）ため、立地の肯定率は 0.89 が 0.93 になった。口コミは定型の文を組み合わせた架空のものなので、皮肉や比較、暗黙の評価を含む実際の口コミでは精度はこれより下がると考えられる。数十件に人が正解を付けて、見落としの偏りも確かめる。
- 段階的な方法の要約は、集計表の割合と代表的な意見（表の「肯定の例」「否定の例」）をそのまま写し、ゴキブリの苦情は「少数」と書いた。しかし、まとめの文では、集計で否定が多数の部屋の広さと肯定が多数の価格を「賛否両論」とし、「価格を除けば概ね高評価」と集計と逆のことも書いた（「〜を除けば」の観点は検査で除外しているので、この誤りは検出されていない）。誤りを伝えて書き直させると、部屋の広さは「否定的な意見が majority を占め」と直ったが（英単語が混ざり、検査の語に合わないので数に入っていない）、価格は直らなかった。さらに、書き足した「ただし」の文で、賛否が分かれる接客について「改善の余地があるかもしれません」と、新しい言い過ぎを加えた（合う 5 / 主張 7）。書き直させるたびに検査し直す。重要な用途では、観点ごとの行と多数派の判定はコードのテンプレートで作り、LLM には代表的な意見の言い換えだけを任せる。
- 割合には件数と信頼区間を添える。接客の肯定率 0.46 は 95% 区間が 0.35〜0.58 で 0.5 をまたぐので、「否定が多い」ではなく「賛否が分かれる」と書くのが正しい。口コミが数十件しかない店や商品では、区間はさらに広がる。
- 要約の検査は、割合の表現と評価の語を正規表現で読み取る簡単なもので、「約 40% の口コミでは部屋が狭く」のような書き方は読み取れない（表の NaN）。この章の「要約が原文と食い違っていないか検査する」の NLI は、4 組のうち 3 組（「賛否が分かれる」は含意、「不満が多い」は矛盾、「改善の余地があると感じる宿泊客もいる」は中立）は妥当に判定したが、「ほぼ半々」から「肯定的な意見が多い」を含意 0.98 と判定し、この例では割合の言い違いを見分けられなかった。割合の主張は、数えた結果と突き合わせて確かめる。
- 段階的な方法は、この章の「長い文書を分割して要約する」の map-reduce と同じ形だが、まとめる段階を LLM ではなくコードの集計にしている。部分要約を LLM にまとめさせると件数の情報が失われるので、割合を伝える要約では、数える部分をコードに任せる。
- 正解の割合がわかるように、口コミはこのレシピのために作った（テキスト分類の章の「観点ごとにポジティブ・ネガティブを判定する（アスペクトベース感情分析）」の chABSA は有価証券報告書の文なので使っていない）。実際の口コミでは、楽天トラベルのレビューや、その約 7 万文に 7 項目の観点と極性を付けたコーパスがあるが、国立情報学研究所の IDR を通じて、国内の大学と公的研究機関の研究者に学術研究の目的に限って提供されている。Amazon の多言語レビューコーパス (MARC) は配布が終了し、ライセンスも非商用の学術研究に限られていた。モデルは sarashina2.2-3b-instruct（SB Intuitions、MIT ライセンス）と mDeBERTa-v3-base-mnli-xnli（MIT ライセンス）。数値は 検証の GPU で float16 の経路（`BF16 = False`、T4 と同じ dtype）で実行したもので、GPU メモリの最大使用量は 8.9 GB だった。トークナイザーの修正については、大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」の注意書きを参照。

## 長い文章を話題の切れ目で区切る（TextTiling と文埋め込み） {text_segmentation}

長い文章や会議の書き起こしを、話題が変わるところで区切りたい。単語の重なりで見る TextTiling と、文埋め込みの類似度の谷で見る方法を、正解の切れ目があるデータで Pk と WindowDiff を使って比べる。

```python
!pip install -q fugashi==1.5.2 unidic-lite==1.0.8
```

```python
import io, re, tarfile, urllib.request
import numpy as np
import pandas as pd

def load_livedoor():
    """livedoor ニュースコーパス（株式会社ロンウイット配布）。記事は CC BY-ND 2.1 JP（表示・改変禁止）"""
    url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
    with urllib.request.urlopen(url) as r:
        tar = tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")
    rows = []
    for m in tar.getmembers():
        parts = m.name.split("/")
        if m.isfile() and len(parts) == 3 and parts[2] != "LICENSE.txt":
            lines = tar.extractfile(m).read().decode("utf-8").split("\n")
            rows.append({"label": parts[1], "title": lines[2].strip(), "text": "\n".join(lines[3:]).strip()})
    return pd.DataFrame(rows)

def split_sentences(text):
    sents = [s.strip() for s in re.split(r"(?<=[。！？])|\n+", text) if s and len(s.strip()) > 5]
    return [s for s in sents if "http" not in s and not s.startswith(("■", "【関連"))]   # リンクや関連記事の行を除く

df = load_livedoor()
df["sents"] = df["text"].map(split_sentences)
df = df[df["sents"].map(len).between(5, 15)].reset_index(drop=True)   # 5〜15 文の記事だけを使う

def make_docs(df, n_docs, n_articles=4, same_label=False, seed=0):
    """記事を n_articles 本つなげて 1 文書にし、記事の境界を正解の切れ目にする。
    same_label=False なら隣り合う記事のカテゴリが必ず違い、True なら 1 つのカテゴリの記事だけを使う。
    正解は文ごとの "0"/"1" の列で、"1" はその文の後ろで話題が変わることを表す"""
    rng = np.random.default_rng(seed)
    cats = df["label"].unique()
    docs = []
    for _ in range(n_docs):
        if same_label:
            labels = [rng.choice(cats)] * n_articles
        else:
            labels = [rng.choice(cats)]
            while len(labels) < n_articles:
                labels.append(rng.choice([c for c in cats if c != labels[-1]]))
        idx = []
        for c in labels:   # カテゴリごとに、まだ使っていない記事を 1 本選ぶ
            idx.append(rng.choice(df.index[(df["label"] == c) & ~df.index.isin(idx)]))
        arts = df.loc[idx]
        sents, ref = [], ""
        for s in arts["sents"]:
            sents += s
            ref += "0" * (len(s) - 1) + "1"
        docs.append({"sents": sents, "ref": ref[:-1] + "0", "labels": list(arts["label"])})
    return docs

# 調整用 (dev) と評価用 (test) で記事が重ならないように分ける
rng = np.random.default_rng(0)
is_dev = rng.random(len(df)) < 0.3
dev = make_docs(df[is_dev], 10, seed=1) + make_docs(df[is_dev], 10, same_label=True, seed=4)   # 両方の種類を 10 文書ずつ
test_mixed = make_docs(df[~is_dev], 30, seed=2)                    # 隣り合う記事のカテゴリが違う（切れ目が見つけやすい）
test_same = make_docs(df[~is_dev], 30, same_label=True, seed=3)    # 同じカテゴリの記事だけをつなげる（難しい）
print(len(df), "記事 / 1 文書の平均", round(np.mean([len(d["sents"]) for d in test_mixed]), 1), "文 / 埋め込む文の数",
      sum(len(d["sents"]) for d in dev + test_mixed + test_same))
print(test_mixed[0]["labels"])
```

```python
from fugashi import Tagger
from nltk.tokenize.texttiling import TextTilingTokenizer

tagger = Tagger()
vocab = {}

def to_ascii(sent):
    """内容語の原形を、英小文字だけの記号 (a, b, ..., aa, ab, ...) に置き換える。
    NLTK の TextTiling は a-z 以外の文字を消してから単語を数えるので、日本語のままでは単語が 1 つも残らない"""
    out = []
    for w in tagger(sent):
        f = w.feature
        if f.pos1 in ("名詞", "動詞", "形容詞") and f.pos2 not in ("非自立可能", "数詞"):
            i = vocab.setdefault(f.lemma or w.surface, len(vocab))
            code = ""
            while True:
                code = chr(97 + i % 26) + code
                i = i // 26 - 1
                if i < 0:
                    break
            out.append(code)
    return " ".join(out)

def pick_boundaries(depth, threshold=None, n_boundaries=None, min_gap=3):
    """depth[i] は i 番目の文の後ろの谷の深さ。深い順に、threshold 以上のものを全部、または n_boundaries 個だけ選ぶ。
    すでに選んだ切れ目から min_gap 文以内のものは選ばない"""
    chosen = []
    for i in np.argsort(-depth):
        if (n_boundaries is None and depth[i] < threshold) or (n_boundaries is not None and len(chosen) >= n_boundaries):
            break
        if all(abs(i - j) >= min_gap for j in chosen):
            chosen.append(i)
    return "".join("1" if i in chosen else "0" for i in range(len(depth) + 1))

def texttiling(sents, w=20, k=4, n_boundaries=None):
    """TextTiling で切れ目を求め、文ごとの "0"/"1" の列で返す。n_boundaries を与えると、谷の深い順にその数だけ選ぶ"""
    codes = [to_ascii(s) for s in sents]
    ends = np.cumsum([len(c.split()) for c in codes])     # 各文の終わりが何語目か
    # stopwords を渡さないと英語のストップワードを読みに行く（内容語だけにしたので不要）。
    # demo_mode=True で、w 語ずつの擬似文の間（ギャップ）ごとの谷の深さと、切れ目かどうかの判定を受け取る
    tt = TextTilingTokenizer(w=w, k=k, stopwords=[], demo_mode=True)
    try:
        _, _, depth_gap, is_boundary = tt.tokenize("\n\n".join(codes))
    except (ValueError, ZeroDivisionError):      # 文書が短すぎるとき
        return "0" * len(sents)
    # ギャップ g は (g + 1) * w 語目の後ろ。いちばん近い文の終わりに対応づける
    depth, auto = np.zeros(len(sents) - 1), np.zeros(len(sents) - 1)
    for g, (dep, b) in enumerate(zip(depth_gap, is_boundary)):
        j = int(np.argmin(np.abs(ends - (g + 1) * w)))
        if j < len(sents) - 1:
            depth[j], auto[j] = max(depth[j], dep), max(auto[j], b)
    if n_boundaries is None:
        return "".join("1" if x else "0" for x in auto) + "0"
    return pick_boundaries(depth, n_boundaries=n_boundaries)

print(test_mixed[0]["ref"])
print(texttiling(test_mixed[0]["sents"]))
```

```python
from sentence_transformers import SentenceTransformer

# 日本語の文埋め込みモデル Ruri v3 の最小のもの (Apache 2.0)。"トピック: " は話題を捉えるためのプレフィックス（モデルカードによる）
model = SentenceTransformer("cl-nagoya/ruri-v3-30m")

def embed(docs):
    for d in docs:
        d["emb"] = model.encode(["トピック: " + s for s in d["sents"]], normalize_embeddings=True, batch_size=64, show_progress_bar=False)

def gap_similarity(emb, window=3):
    """各文の後ろの切れ目について、前の window 文と後ろの window 文の平均ベクトルのコサイン類似度"""
    sims = []
    for i in range(len(emb) - 1):
        a, b = emb[max(0, i - window + 1):i + 1].mean(0), emb[i + 1:i + 1 + window].mean(0)
        sims.append(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    return np.array(sims)

def depth_scores(sims):
    """谷の深さ: 左右それぞれ、類似度が上がり続ける間たどった山の高さと、谷の底との差の和（TextTiling と同じ考え方）"""
    depth = []
    for i, s in enumerate(sims):
        l = i
        while l > 0 and sims[l - 1] >= sims[l]:
            l -= 1
        r = i
        while r < len(sims) - 1 and sims[r + 1] >= sims[r]:
            r += 1
        depth.append(sims[l] + sims[r] - 2 * s)
    return np.array(depth)

def embedding_seg(emb, window=3, threshold=0.1, n_boundaries=None, min_gap=3):
    return pick_boundaries(depth_scores(gap_similarity(emb, window)), threshold, n_boundaries, min_gap)

# Colab の CPU で 2 分ほど
for docs in (dev, test_mixed, test_same):
    embed(docs)
print(test_mixed[0]["ref"])
print(embedding_seg(test_mixed[0]["emb"]))
```

```python
from nltk.metrics.segmentation import pk, windowdiff

def evaluate(docs, segment):
    """Pk と WindowDiff の平均（小さいほどよい）と、Pk の標準誤差。窓の幅 k は、正解の区間の平均の長さの半分"""
    rows = []
    for d in docs:
        ref, hyp = d["ref"], segment(d)
        k = max(2, round(len(ref) / (ref.count("1") + 1) / 2))
        rows.append({"Pk": pk(ref, hyp, k), "WindowDiff": windowdiff(ref, hyp, k), "切れ目の数": hyp.count("1")})
    rows = pd.DataFrame(rows)
    return pd.Series({"Pk": rows["Pk"].mean(), "Pk の標準誤差": rows["Pk"].sem(), "WindowDiff": rows["WindowDiff"].mean(), "切れ目の数": rows["切れ目の数"].mean()})

# 窓の大きさとしきい値は、調整用の文書 (dev) で WindowDiff が最小になるものを選ぶ
def tune(segment, grid):
    return min(grid, key=lambda p: evaluate(dev, lambda d: segment(d, *p))["WindowDiff"])

n_ref = lambda d: d["ref"].count("1")
best_tt = tune(lambda d, w, k: texttiling(d["sents"], w, k), [(w, k) for w in (5, 10, 20, 30) for k in (2, 4, 6, 10)])
best_tt_n = tune(lambda d, w, k: texttiling(d["sents"], w, k, n_ref(d)), [(w, k) for w in (5, 10, 20, 30) for k in (2, 4, 6, 10)])
best_emb = tune(lambda d, win, th: embedding_seg(d["emb"], win, th), [(win, th) for win in (1, 2, 3, 5, 7, 10) for th in (0.05, 0.1, 0.15, 0.2, 0.3)])
best_emb_n = tune(lambda d, win: embedding_seg(d["emb"], win, n_boundaries=n_ref(d)), [(win,) for win in (1, 2, 3, 5, 7, 10)])
print("TextTiling (w, k):", best_tt, "/ 数を与えるとき", best_tt_n)
print("文埋め込み (window, threshold):", best_emb, "/ 数を与えるとき window", best_emb_n)
```

```python
def fixed_length(n, every=10):
    return "".join("1" if (i + 1) % every == 0 and i < n - 1 else "0" for i in range(n))

def random_seg(n, n_boundaries, rng):
    chosen = set(rng.choice(n - 1, n_boundaries, replace=False))
    return "".join("1" if i in chosen else "0" for i in range(n))

rng = np.random.default_rng(0)   # ランダムな区切りの乱数（種を固定）
methods = {
    "切れ目なし": lambda d: "0" * len(d["ref"]),
    "10 文ごとに機械的に区切る": lambda d: fixed_length(len(d["ref"])),
    "ランダム（正解と同じ数）": lambda d: random_seg(len(d["ref"]), n_ref(d), rng),
    "TextTiling": lambda d: texttiling(d["sents"], *best_tt),
    "TextTiling（切れ目の数を与える）": lambda d: texttiling(d["sents"], *best_tt_n, n_boundaries=n_ref(d)),
    "文埋め込みの谷": lambda d: embedding_seg(d["emb"], *best_emb),
    "文埋め込みの谷（切れ目の数を与える）": lambda d: embedding_seg(d["emb"], *best_emb_n, n_boundaries=n_ref(d)),
}
res = pd.DataFrame({(cond, name): evaluate(docs, f) for name, f in methods.items()
                    for cond, docs in [("別カテゴリ", test_mixed), ("同じカテゴリ", test_same)]}).T
print("正解の切れ目の数: 1 文書に 3")

# 2 つの方法を同じ文書で比べる: 文書ごとの Pk の差（TextTiling − 文埋め込み）の平均と標準誤差（どちらも切れ目の数を与える）
for cond, docs in [("別カテゴリ", test_mixed), ("同じカテゴリ", test_same)]:
    diff = []
    for d in docs:
        ref = d["ref"]
        k = max(2, round(len(ref) / (ref.count("1") + 1) / 2))
        diff.append(pk(ref, methods["TextTiling（切れ目の数を与える）"](d), k) - pk(ref, methods["文埋め込みの谷（切れ目の数を与える）"](d), k))
    print(f"{cond}: Pk の差 {np.mean(diff):.3f} ± {pd.Series(diff).sem():.3f}（標準誤差）、文埋め込みのほうがよい文書 {sum(x > 0 for x in diff)} / 同じ {sum(x == 0 for x in diff)} / 悪い {sum(x < 0 for x in diff)}")
res.unstack(0).reindex(methods).round(3)
```

```python
# 会議の書き起こしに使う例（架空の短い会議）。発言を 1 単位にし、話題の数が分かっているとして切れ目の数を与える
meeting = [
    ("田中", "では定例会議を始めます。まず来月の展示会の準備状況からお願いします。"),
    ("高橋", "ブースの場所は決まりました。入口に近い角の区画です。"),
    ("鈴木", "展示用のパネルは来週中に印刷会社へ入稿します。"),
    ("田中", "当日の説明員は何人必要ですか。"),
    ("高橋", "午前と午後で 2 人ずつ、合計 4 人いれば回せます。"),
    ("田中", "次に、新しい勤怠システムへの移行について。"),
    ("佐藤", "来月 1 日から新システムに切り替えます。古いシステムの打刻データは今月末で締めます。"),
    ("伊藤", "有給休暇の残日数は自動で引き継がれますか。"),
    ("佐藤", "はい、残日数は移行されます。申請中の休暇だけは出し直してもらいます。"),
    ("田中", "最後に、オフィスの引っ越しの件です。"),
    ("鈴木", "新しいオフィスの内装工事が少し遅れていて、入居は 2 週間後ろにずれそうです。"),
    ("伊藤", "それなら段ボールの配布も遅らせたほうがいいですね。"),
    ("鈴木", "荷造りは入居日の 1 週間前から始めてもらう予定です。"),
]
emb = model.encode(["トピック: " + u for _, u in meeting], normalize_embeddings=True)
hyp = embedding_seg(emb, window=2, n_boundaries=2, min_gap=2)
for (who, u), b in zip(meeting, hyp):
    print(f"{who}: {u}")
    if b == "1":
        print("―――― 話題の切れ目 ――――")
```

- 評価データは、livedoor ニュースコーパスの 5〜15 文の記事を 4 本ずつつなげ、記事の境界を正解の切れ目にして作った（隣り合う記事のカテゴリが違う 30 文書と、同じカテゴリの記事だけの 30 文書。1 文書は 4 区間で切れ目は 3 つ。前者は平均 47.1 文）。調整用の文書も両方の種類を 10 文書ずつにし、評価用とは記事が重ならないようにした。手軽に作れるが、記事の境界は話題の変わり目として分かりやすく、実際の文書や会議では切れ目がもっとあいまいなことがある。記事は CC BY-ND 2.1 JP（表示・改変禁止）なので、つなげた評価データは配布せず、手元で作り直す。
- Pk は、k 文離れた 2 つの文が同じ区間に入るかどうかの判定を誤る割合、WindowDiff は、幅 k の窓の中の切れ目の数が正解と食い違う割合で、どちらも 0 が最良（Pk は Beeferman ら 1999、WindowDiff は Pevzner と Hearst 2002 の提案）。上の表では切れ目を 1 つも置かなくても Pk が 0.415〜0.419 になり、10 文ごとに機械的に区切ると、かえって悪くなる (0.464〜0.516)。必ずこうしたベースラインと並べる。
- NLTK の `TextTilingTokenizer`（Hearst 1997 の TextTiling の実装）は、a-z 以外の文字を消してから単語を数えるので、日本語をそのまま渡すと単語が 1 つも残らない。ここでは形態素解析で内容語の原形を取り出し、英小文字だけの記号に置き換えて渡した。また、切れ目を段落の位置に戻す処理では、最初のギャップ以外の切れ目が w 語分前にずれる（nltk 3.9.1 のソースで確認）ので、`demo_mode=True` で各ギャップの判定と谷の深さを受け取り、文の位置への対応づけは自分で行った。
- 結果: 切れ目の数を自動で決める設定では、文埋め込みの谷（Pk 0.102 / 0.193、隣り合うカテゴリが違う / 同じカテゴリ）が TextTiling (0.300 / 0.303) より大幅によかった。切れ目の数を与えると TextTiling は 0.213 / 0.217 まで改善するが、文埋め込み (0.106 / 0.146) との差は残り、同じ文書どうしで比べた Pk の差は 0.107 ± 0.029 / 0.071 ± 0.026（平均 ± 標準誤差）で、30 文書中 23 / 19 文書で文埋め込みのほうがよかった。文埋め込みは同じカテゴリの文書で悪くなるが、TextTiling はどちらでもほとんど変わらない。
- 文埋め込みの方法は、前後の window 文の平均ベクトルの類似度が落ち込む「谷」を切れ目にする。単語が重ならない言い換えでも話題の近さを捉えられる。窓の大きさとしきい値は文書の種類で変わるので調整用の文書で選んだ（window は自動の設定で 7、数を与える設定で 5。TextTiling の (w, k) は (30, 4) と (10, 10) で、どちらも調べた範囲の端なので、範囲を広げるとよくなる余地がある）。自動の設定では切れ目がやや多め（平均 4.1 / 3.7 個。正解は 3 個）になった。話題の数（議題の数など）が分かっているなら、深い谷から順にその数だけ選べばしきい値は要らない。
- 最後の架空の会議では、発言を 1 単位にし、切れ目の数 (2) を与えると、議題が変わる 2 か所で区切れた。ただし 1 例だけの確認である。発言は短く、窓に入る情報が少ないので、実際の書き起こしでは切れ目が前後にずれることがある前提で、区切った結果を人が確かめる。
- LangChain の `langchain-experimental` の `SemanticChunker` も、隣り合う文（前後 `buffer_size` 文を含めたもの）の埋め込みの距離が大きいところで切る同じ考え方の実装である。既定の文の区切りは英語の句読点 (`(?<=[.?!])\s+`) なので、日本語では `sentence_split_regex` を変える。`number_of_chunks` でチャンクの数の目安を与えることもできるが、距離のパーセンタイルに換算してしきい値にするので、ちょうどその数になるとは限らない（0.4.2 のソースで確認）。
- 議事録を作るときは、話題ごとに分けてから要約すると、長い書き起こしをそのまま渡すより論点が混ざりにくい（この章の「会議の書き起こしから議事録を作る（決定事項・ToDo を JSON で抜き出す）」「長い文書を分割して要約する」を参照）。RAG のチャンク分割でも、文字数で切る代わりに話題の切れ目で切れば、1 つのチャンクに別の話題が混ざりにくくなる（RAG の章の「長い文書をチャンクに分割する」を参照）。

## 会議の書き起こしから議事録を作る（決定事項・ToDo を JSON で抜き出す） {meeting_minutes gpu}

話者付きの会議の書き起こしから、決定事項・ToDo（担当者と期限）・論点を JSON で抜き出し、議事録にまとめたい。書き起こしにない担当者や期限を作っていないかも確かめたい。

```python
import re
import unicodedata

# 架空の会議の書き起こし（このレシピのために作ったもの）。実際には文字起こしの結果を（話者, 発言）の列にして使う
transcript = [
    ("田中", "それでは、学習アプリ「まなぶん」のリリース準備会議を始めます。今日は、リリース日、対応する端末、料金、残っている作業の分担を決めたいと思います。"),
    ("佐藤", "開発の状況から報告します。主な機能はほぼできていますが、パスワードを再設定するとまれにログインできなくなる不具合が残っています。"),
    ("田中", "その不具合は、リリースまでに直せそうですか。"),
    ("佐藤", "原因の見当はついているので、11 月 28 日までには直せると思います。私が担当します。"),
    ("伊藤", "品質保証の立場から言うと、その修正のあとに回帰テストを 1 週間はやりたいです。"),
    ("田中", "そうすると、当初予定していた 12 月 3 日のリリースは厳しいですね。"),
    ("高橋", "営業としては、年内にはどうしても出したいです。12 月の第 2 週に、展示会で発表の場がありますので。"),
    ("佐藤", "修正が 28 日、テストが 1 週間だとすると、12 月 5 日ごろにテストが終わります。"),
    ("田中", "では、リリース日は 12 月 10 日にしましょう。展示会にも間に合います。皆さん、よろしいですか。"),
    ("鈴木", "異論ありません。"),
    ("伊藤", "それなら大丈夫です。"),
    ("田中", "リリース日は 12 月 10 日で決定とします。伊藤さん、テスト計画を作っていただけますか。"),
    ("伊藤", "はい、11 月 25 日までにテスト計画をまとめて共有します。"),
    ("田中", "次に、対応する端末です。iOS と Android を同時に出すかどうか。"),
    ("佐藤", "Android 版は、通知の機能がまだ不安定です。同時に出すと、テストの手間が倍になります。"),
    ("高橋", "お客様の半分くらいは Android を使っているので、できれば同時がいいのですが。"),
    ("伊藤", "正直なところ、両方を 1 週間でテストするのは無理です。"),
    ("田中", "初回のリリースは iOS 版だけにして、Android 版は来年 2 月を目標にしましょう。高橋さん、お客様への説明をお願いできますか。"),
    ("高橋", "わかりました。主なお客様 5 社には、私から直接説明します。"),
    ("田中", "続いて料金です。無料版と有料版を分ける方針は前回決めましたが、有料版の月額をいくらにするか。"),
    ("高橋", "競合が月額 800 円なので、500 円にして安さを打ち出したいです。"),
    ("鈴木", "ただ、500 円だと、サーバーの費用を考えると赤字にならないでしょうか。"),
    ("佐藤", "利用者が 1 万人を超えるまでは、500 円だと赤字になる計算です。"),
    ("田中", "料金は今日は決めきれませんね。経理の意見も聞いてから、次回の会議で決めます。高橋さん、競合の料金表をまとめておいてもらえますか。"),
    ("高橋", "はい、まとめておきます。"),
    ("田中", "次はデザインです。鈴木さん、アイコンはどうなっていますか。"),
    ("鈴木", "今の案は社内の評判があまりよくないので、作り直そうと思っています。来週の月曜までに、新しい案を 3 つ出します。"),
    ("田中", "お願いします。ストアに載せる紹介用の画像も必要ですね。"),
    ("鈴木", "画像はアイコンが決まってから作ります。"),
    ("田中", "それから、プレスリリースの文案も誰かが書かないといけません。"),
    ("高橋", "営業部でも書けますが、広報に頼んだほうがいいかもしれません。"),
    ("田中", "担当は広報と相談してから決めましょう。"),
    ("伊藤", "一つ気になっているのですが、利用者の学習記録を海外のサーバーに保存する点について、個人情報の扱いを法務に確認しましたか。"),
    ("佐藤", "まだです。利用規約の文面にも関わりますね。"),
    ("田中", "それは確認が必要です。佐藤さん、法務への確認をお願いします。リリースに関わるので、11 月 30 日までに結果を教えてください。"),
    ("佐藤", "承知しました。"),
    ("伊藤", "すみません、先ほどのテスト計画ですが、ログインの修正方針を見てから作りたいので、期限を 11 月 27 日に延ばしてもいいでしょうか。"),
    ("田中", "はい、27 日で構いません。"),
    ("田中", "最後に確認します。リリースは 12 月 10 日、初回は iOS 版のみ、料金は次回決定です。次回の会議は 12 月 1 日です。お疲れさまでした。"),
]
lines = [f"[{i}] {who}: {text}" for i, (who, text) in enumerate(transcript, 1)]   # 発言番号を付けて、根拠として指させるようにする
speakers = sorted({who for who, _ in transcript})

def make_chunks(lines, max_chars=600):
    """発言の途中で切らないように、発言単位で max_chars 文字ずつにまとめる"""
    chunks, cur = [], []
    for line in lines:
        if cur and sum(map(len, cur)) + len(line) > max_chars:
            chunks.append(cur); cur = []
        cur.append(line)
    return chunks + [cur]

chunks = make_chunks(lines)
print(sum(map(len, lines)), "文字 /", len(lines), "発言 / 参加者", speakers)
print(len(chunks), "分割:", [(c[0].split("]")[0] + "]", c[-1].split("]")[0] + "]", sum(map(len, c))) for c in chunks])
```

```python
import json
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ValidationError, field_validator
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"   # 3B の日本語モデル（MIT ライセンス）。VRAM 約 7GB
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる（大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」を参照）。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()

def chat(messages, max_new_tokens=700):
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(llm.device)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

class Item(BaseModel):
    content: str
    evidence: list[int]          # 根拠の発言番号

class Todo(BaseModel):
    task: str
    owner: str | None            # 発言に担当者がなければ null
    due: str | None              # 発言に期限がなければ null
    evidence: list[int]

    @field_validator("owner", "due")
    @classmethod
    def none_if_unknown(cls, v):    # 「未定」「なし」などと文字で書いてきたものも null として扱う
        return None if v is None or re.match(r"\s*(未定|なし|特になし|不明|null)", v) else v

class Minutes(BaseModel):
    decisions: list[Item]
    todos: list[Todo]
    issues: list[Item]           # 結論が出ていない論点

INSTRUCTION = """次は会議の書き起こしの一部です。各行は「[発言番号] 話者: 発言」です。
決定事項 (decisions)、ToDo (todos)、結論が出ていない論点 (issues) を、例と同じ形の JSON で抜き出してください。
- 誰かが頼まれて引き受けたこと、自分からやると言ったこと、担当が決まっていないがやる必要があることは ToDo にする
- evidence には根拠となる発言番号を入れる
- ToDo の owner (担当者) と due (期限) は、発言の中で明示されたときだけ書く。明示されていなければ null にする。推測で補わない
- 該当するものがなければ空のリスト [] にする
- JSON 以外は出力しない"""
EXAMPLE_IN = """[1] 山田: 新しい名刺のデザインは A 案に決めます。
[2] 木村: 印刷会社への発注は私がやります。金曜までに済ませます。
[3] 山田: 配布の方法は、また今度考えましょう。社内報に載せる記事も、誰かに書いてもらわないと。"""
EXAMPLE_OUT = json.dumps({
    "decisions": [{"content": "名刺のデザインを A 案にする", "evidence": [1]}],
    "todos": [{"task": "印刷会社に発注する", "owner": "木村", "due": "金曜", "evidence": [2]},
              {"task": "社内報の記事を書く", "owner": None, "due": None, "evidence": [3]}],
    "issues": [{"content": "名刺の配布方法", "evidence": [3]}]}, ensure_ascii=False)

def extract_json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    obj, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
    return obj

def extract(chunk, max_retries=2):
    """（抜き出した結果, 出し直した回数）を返す。出し直しを使い切っても検証を通らなければ（None, 回数）を返す"""
    messages = [{"role": "user", "content": f"{INSTRUCTION}\n\n{EXAMPLE_IN}"}, {"role": "assistant", "content": EXAMPLE_OUT},
                {"role": "user", "content": f"{INSTRUCTION}\n\n" + "\n".join(chunk)}]
    for retry in range(max_retries + 1):
        out = chat(messages)
        try:
            return Minutes.model_validate(extract_json(out)), retry
        except (ValueError, ValidationError) as e:   # 壊れた JSON は、エラーを伝えて出し直させる
            messages += [{"role": "assistant", "content": out}, {"role": "user", "content": f"出力が不正です: {str(e)[:200]}\n例と同じ形の JSON だけを出力し直してください。"}]
    return None, max_retries

# 1. 分割ごとに抜き出す (map)。T4 なら、モデルのダウンロード（約 7GB）を除いて数分ほど
parts = []
for k, chunk in enumerate(chunks, 1):
    m, retry = extract(chunk)
    if m is None:    # 黙って空の結果にすると、その分割の決定事項や ToDo が消えたことに気づけない
        print(f"--- 分割 {k}: 出し直し {retry} 回でも有効な JSON が得られなかった。この分割は手で確かめる")
        continue
    parts.append(m)
    print(f"--- 分割 {k} (出し直し {retry} 回)")
    for t in m.todos:
        print("  ToDo:", t.model_dump())
```

```python
import pandas as pd

# 2. 分割ごとの結果を 1 つにまとめる (reduce)。ToDo は根拠の発言番号の順に並べるだけで、LLM には渡さない
decisions = [d for m in parts for d in m.decisions]
issues = [d for m in parts for d in m.issues]
todos = sorted([t for m in parts for t in m.todos], key=lambda t: min(t.evidence or [0]))

# 3. 抜き出した担当者と期限が、根拠の発言に本当に書かれているかを機械的に確かめる
norm = lambda s: unicodedata.normalize("NFKC", s).replace(" ", "")

def check_todo(todo):
    problems = []
    ev = [transcript[i - 1] for i in todo.evidence if 1 <= i <= len(transcript)]
    if len(ev) < len(todo.evidence) or not ev:
        problems.append("発言番号が不正")
    ev_text = norm(" ".join(f"{who} {text}" for who, text in ev))
    if todo.owner is not None:
        owner = norm(todo.owner).removesuffix("さん")
        if owner not in speakers:
            problems.append("担当者が参加者にいない")
        elif owner not in ev_text:            # 根拠の発言の話者か本文に、担当者の名前が出てくるか
            problems.append("担当者が根拠の発言にない")
    if todo.due is not None:
        due = norm(todo.due)
        nums = re.findall(r"\d+", due)
        if due not in ev_text and not (nums and all(n in re.findall(r"\d+", ev_text) for n in nums)):
            problems.append("期限が根拠の発言にない")
    return problems

# 検査が働くことを、わざと作った誤り（書き起こしにない期限、参加者でない担当者）で確かめる
fakes = [Todo(task="プレスリリースの文案を書く", owner="高橋", due="12 月 1 日", evidence=[30, 31]),
         Todo(task="法務に確認する", owner="山本", due="11 月 30 日", evidence=[35])]
for t in fakes:
    print(t.task, t.owner, t.due, "→", check_todo(t))

rows = [{"ToDo": t.task, "担当": t.owner, "期限": t.due, "根拠": t.evidence, "検査": "、".join(check_todo(t)) or "OK"} for t in todos]
pd.set_option("display.max_colwidth", 40)
pd.DataFrame(rows)
```

```python
# 4. 書き起こしを作った人（ここではレシピの作者）が付けた正解と比べる。task に含まれる語で対応を取る
gold = [("不具合|パスワード", "佐藤", "11月28日"), ("テスト計画", "伊藤", "11月27日"), ("説明", "高橋", None), ("料金表", "高橋", None),
        ("アイコン", "鈴木", "来週の月曜"), ("法務|個人情報", "佐藤", "11月30日"), ("プレスリリース", None, None)]
def same_due(a, b):
    """期限が同じか。どちらも None なら同じ。文字列が同じか、含まれる数字の並びが同じ（「11/27」と「11月27日」）なら同じとみなす"""
    if a is None or b is None:
        return a is None and b is None
    nums_a, nums_b = re.findall(r"\d+", norm(a)), re.findall(r"\d+", norm(b))
    return norm(a) == norm(b) or (nums_a != [] and nums_a == nums_b)

for key, owner, due in gold:
    found = [t for t in todos if re.search(key, t.task)]
    if not found:
        print(f"{key:<14} 見落とし")
    for t in found:
        print(f"{key:<14} 担当 {'○' if t.owner == owner else '×'} ({t.owner})  期限 {'○' if same_due(t.due, due) else '×'} ({t.due})  {t.task}")
print("正解にない ToDo:", [t.task for t in todos if not any(re.search(key, t.task) for key, _, _ in gold)])
```

```python
# 5. 議事録の形に整え、全体の要約を付ける。要約の入力は書き起こしではなく、抜き出した JSON にする
def bullet(items):
    return "\n".join(f"- {d.content} (発言 {', '.join(map(str, d.evidence))})" for d in items) or "- なし"

digest = json.dumps({"decisions": [d.content for d in decisions], "todos": [f"{t.task} (担当: {t.owner or '未定'}、期限: {t.due or '未定'})" for t in todos],
                     "issues": [d.content for d in issues]}, ensure_ascii=False)
summary = chat([{"role": "user", "content": f"次は会議で決まったこと、ToDo、残った論点の一覧です。会議の内容を 3 文で要約してください。一覧にないことは書かないでください。\n\n{digest}"}], max_new_tokens=200)

def todo_line(t):
    line = f"- {t.task} / 担当: {t.owner or '未定'} / 期限: {t.due or '未定'}"
    problems = check_todo(t)
    if problems:     # 検査に引っかかったものには印を付けて、人が確かめるようにする
        line += f"  ※要確認 ({'、'.join(problems)})"
    return line

todo_lines = "\n".join(todo_line(t) for t in todos)
print(f"# 議事録: リリース準備会議\n参加者: {'、'.join(speakers)}\n\n## 要約\n{summary}\n\n## 決定事項\n{bullet(decisions)}\n\n## ToDo\n{todo_lines}\n\n## 論点 (未決)\n{bullet(issues)}")
```

- 書き起こしに発言番号を付け、抜き出した項目ごとに根拠の発言番号 (evidence) を答えさせる。人が確かめるときにすぐ原文に戻れるうえ、担当者や期限がその発言に本当に書かれているかを機械的に検査できる。JSON の取り出しと、壊れていたら出し直させる方法は、大規模言語モデルの章の「LLM の出力から JSON を取り出して検証する」と同じ（上の出力でも、分割 1 は 1 回目の出力が検証を通らず出し直している）。出し直しを使い切った分割は、空の結果にせずに知らせる。
- 書き起こしを発言の途中で切らないように分割し、分割ごとに抜き出してからまとめる（map-reduce。この章の「長い文書を分割して要約する」と同じ考え方）。議事録では、ある分割で決まった期限が後の分割で変わることがある。上の出力では、伊藤さんのテスト計画は [13] の 11 月 25 日が [37]〜[38] で 27 日に延びており、ToDo には期限なし [12]・11/25 [13]・11/27 [37] の 3 件が並んだ。期限なしの 1 件は、依頼 [12] が分割 1 の末尾に、引き受けと期限 [13] が分割 2 の先頭に分かれたために出た重複である（前の分割の末尾の数発言を重ねて入れると減らせる）。つなげただけでは重複や古い期限が残る（要約も 11 月 25 日と書いている）ので、同じ内容の ToDo は根拠の発言番号を見比べて人が見直す。
- 担当者と期限の検査は、担当者が参加者にいるか、担当者の名前と期限（の数字）が根拠の発言に出てくるかを確かめるだけの簡単なものである。上の出力では引っかかる ToDo はなく、わざと作った誤り（参加者でない「山本」、書き起こしにない「12 月 1 日」）は検出できた。一方、[31] の話者が高橋さんなので「プレスリリースの担当: 高橋」という誤りは通ってしまう。根拠の発言番号そのものの正しさ、「来週の月曜」のような相対的な日付、仕事の内容 (task) までは確かめられない。
- 検査は「書いたものに根拠があるか」しか見ないので、見落としや、決定でないものを決定とする誤りは見つからない。正解と比べると、7 件の ToDo のうち、法務への確認 ([35]) と担当未定のプレスリリースの文案 ([30]〜[32]) を ToDo として拾えなかった（どちらも論点としては出ているが、法務への確認は [35] で担当と期限まで決まっているので、ToDo から落ちたのは痛い）。逆に、誰も引き受けていない「Android 版のテスト戦略を決定する」を ToDo にしている（[18] で初回は iOS 版だけと決まっている。プロンプトの「担当が決まっていないがやる必要があることは ToDo にする」が、こうした項目を出しやすくしている可能性がある）。また、高橋さんの提案にすぎない [21] を「有料版の月額を 500 円とする」という決定事項にしている（実際は [24] で次回に持ち越し）。重要な会議では、決定事項は根拠の発言を人が読んで確かめるか、より大きなモデルを使う。
- 要約は書き起こし全体ではなく、抜き出した一覧から作っている。入力は短くて済むが、一覧の誤りはそのまま要約に入る（上の出力の要約も「有料版の月額を 500 円とすることが決定された」と書いている）。上の出力では「3 文で」と指示しても 4 文になっている。要約と原文の照合は、この章の「要約が原文と食い違っていないか検査する」を参照。
- 書き起こしはこのレシピのために作った架空の会議である。公開されている会議録には国立国会図書館の国会会議録検索システム（API あり）があるが、その利用条件によると、発言の著作権は原則として個々の発言者にあり、政治上の演説や陳述の利用、情報解析などの権利制限規定に当たる場合を除いて許諾が必要とされているので、使い方に応じて確かめる。社内の会議の書き起こしを外部の API に送るときは、社内の規程も確かめる。

## 規約の新旧を条項ごとに比べ、改定の要点をまとめる {contract_diff gpu}

利用規約や契約書の新旧 2 つの版を比べて、追加・削除・変更された条項を一覧にし、変更の要点を短くまとめたい。条番号がずれても正しく対応づけ、「できる」が「しなければならない」に変わるような小さな変更も見落とさないようにしたい。

```python
!pip install -q fugashi unidic-lite
```

```python
import re
import unicodedata
import pandas as pd

# 説明のために作った架空の規約（実在のサービスの規約ではない）。新版は条番号を全角で書いてある
OLD = """サンプル会員サービス利用規約 (2025 年 4 月 1 日版)
第1条（適用）
本規約は、当社が提供するサンプル会員サービス（以下「本サービス」といいます。）の利用に関する条件を定めるものです。
第2条（会員登録）
本サービスの利用を希望する者は、当社の定める方法により会員登録を申請することができます。登録は、当社が申請を承認した時点で完了します。
第3条（料金及び支払方法）
会員は、本サービスの利用料金として月額500円（税込）を、当社が指定する方法により支払うものとします。
第4条（禁止事項）
会員は、本サービスの利用にあたり、次の行為をしてはなりません。
(1) 法令に違反する行為
(2) 他の会員の情報を収集する行為
(3) 本サービスの運営を妨げる行為
第5条（パスワードの管理）
会員は、自己の責任においてパスワードを管理するものとします。パスワードが第三者に知られたおそれがある場合、会員は当社に通知することができます。
第6条（本サービスの変更）
当社は、会員に事前に通知することにより、本サービスの内容を変更することができます。
第7条（退会）
会員は、当社所定の方法により届け出ることにより、いつでも退会することができます。会員が第4条に違反した場合、当社はその会員を退会させることができます。
第8条（ポイント）
当社は、本サービスの利用に応じて会員にポイントを付与します。ポイントの有効期限は、付与日から1年間とします。
第9条（免責）
当社は、本サービスに関して会員に生じた損害について、当社に故意又は重大な過失がある場合を除き、責任を負いません。
第10条（準拠法及び管轄）
本規約には日本法を適用し、本サービスに関する紛争については、東京地方裁判所を第一審の専属的合意管轄裁判所とします。
"""

NEW = """サンプル会員サービス利用規約 (2026 年 4 月 1 日版)
第１条（適用）
本規約は、当社が提供するサンプル会員サービス（以下「本サービス」といいます。）の利用に関する条件を定めるものです。
第２条（定義）
本規約において「会員」とは、第３条に基づき会員登録が完了した者をいい、「投稿データ」とは、会員が本サービスに投稿した文章、画像その他の情報をいいます。
第３条（会員登録）
本サービスの利用を希望する者は、当社の定める方法により会員登録を申請することができます。登録は、当社が申請を承認した時点で完了します。
第４条（料金及び支払方法）
会員は、本サービスの利用料金として月額550円（税込）を、当社が指定する方法により支払うものとします。
第５条（禁止事項）
会員は、本サービスの利用にあたり、次の行為をしてはなりません。
(1) 法令に違反する行為
(2) 他の会員の情報を収集する行為
(3) 本サービスの運営を妨げる行為
(4) 本サービスで提供される情報を、当社の許可なく機械学習に利用する行為
第６条（パスワードの管理）
会員は、自己の責任においてパスワードを管理するものとします。パスワードが第三者に知られたおそれがある場合、会員は当社に通知しなければなりません。
第７条（本サービスの変更）
当社は、会員に事前に通知することにより、本サービスの内容を変更することができます。
第８条（退会）
会員は、当社所定の方法により届け出ることにより、いつでも退会することができます。会員が第５条に違反した場合、当社はその会員を退会させることができます。
第９条（免責事項）
当社は、本サービスに関して会員に生じた損害について、当社に故意がある場合を除き、責任を負いません。
第１０条（準拠法及び管轄）
本規約には日本法を適用し、本サービスに関する紛争については、東京地方裁判所を第一審の専属的合意管轄裁判所とします。
"""

def parse_articles(doc):
    """「第N条（見出し）」の行で区切り、条番号・見出し・本文の一覧にする。全角の数字や括弧は NFKC で半角にそろえる"""
    doc = unicodedata.normalize("NFKC", doc)
    parts = re.split(r"^第(\d+)条\s*\((.+?)\)\s*$", doc, flags=re.M)
    return [{"no": int(n), "title": t, "body": "\n".join(l.strip() for l in b.strip().splitlines())}
            for n, t, b in zip(parts[1::3], parts[2::3], parts[3::3])]

old, new = parse_articles(OLD), parse_articles(NEW)
print(len(old), "条 →", len(new), "条")

# 素朴な方法: 同じ条番号どうしを比べる
old_by_no, new_by_no = {a["no"]: a for a in old}, {a["no"]: a for a in new}
pd.DataFrame([{"条": n, "旧": old_by_no.get(n, {}).get("title"), "新": new_by_no.get(n, {}).get("title"),
               "本文が同じか": old_by_no.get(n, {}).get("body") == new_by_no.get(n, {}).get("body")} for n in sorted(old_by_no | new_by_no)])
```

```python
import difflib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

emb_model = SentenceTransformer("intfloat/multilingual-e5-small")   # MIT ライセンス

def mask_refs(s):
    return re.sub(r"第\d+条", "第N条", s)        # 対応づけの段階では、条番号への参照を伏せる

def view(a):
    return f"{a['title']}\n{mask_refs(a['body'])}"

E_old = emb_model.encode(["query: " + view(a) for a in old], normalize_embeddings=True)
E_new = emb_model.encode(["query: " + view(a) for a in new], normalize_embeddings=True)
emb_sim = E_old @ E_new.T                       # 意味の近さ（言い換えや見出しの変更に強い）
char_sim = np.array([[difflib.SequenceMatcher(None, view(a), view(b), autojunk=False).ratio() for b in new] for a in old])   # 文字列の重なり
score = (emb_sim + char_sim) / 2

# 割当問題として解き、スコアの合計が最大になる 1 対 1 の組を選ぶ（ハンガリアン法などで解ける問題）
rows, cols = linear_sum_assignment(-score)
THRESHOLD = 0.75                                # これより低い組は対応なし（削除と新設）とみなす
pairs = [(i, j) for i, j in zip(rows, cols) if score[i, j] >= THRESHOLD]
# 対応した組と、それ以外の組の類似度の差を、3 つの尺度で比べる
is_pair = np.zeros_like(score, dtype=bool)
for i, j in pairs:
    is_pair[i, j] = True
for label, m in [("文埋め込み", emb_sim), ("文字列", char_sim), ("平均", score)]:
    print(f"{label}: 対応した組の最小 {m[is_pair].min():.2f} / それ以外の最大 {m[~is_pair].max():.2f}")
pd.DataFrame(score.round(2), index=[f"旧{a['no']} {a['title']}" for a in old], columns=[f"新{a['no']}" for a in new])
```

```python
from collections import Counter
from fugashi import Tagger

tagger = Tagger()
o2n = {old[i]["no"]: new[j]["no"] for i, j in pairs}

def remap_refs(s):
    """旧版の「第N条」を、対応する新版の条番号に読み替える（番号の繰り下げだけによる差分を消す）"""
    return re.sub(r"第(\d+)条", lambda m: f"第{o2n.get(int(m.group(1)), m.group(1))}条", s)

def word_diff(a, b):
    """形態素の単位で差分を取り、削除を [-…-]、追加を {+…+} で示す"""
    wa, wb = [w.white_space + w.surface for w in tagger(a)], [w.white_space + w.surface for w in tagger(b)]
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes():
        if op == "equal":
            out.append("".join(wa[i1:i2]))
        else:
            out.append(("[-" + "".join(wa[i1:i2]) + "-]") * (i2 > i1) + ("{+" + "".join(wb[j1:j2]) + "+}") * (j2 > j1))
    return "".join(out)

# 権利や義務の強さを決める語句と、数値。文全体の類似度ではほとんど変わらないので、別に数えて比べる
KEY_PATTERNS = {"義務": r"しなければな(?:らない|りません)", "禁止": r"してはな(?:らない|りません)", "できる(権利・任意)": r"ことができ(?:る|ます)",
                "努力義務": r"努め(?:る|ます)", "免責": r"責任を負(?:わない|いません)", "故意": r"故意", "重大な過失": r"重大な過失",
                "事前に": r"事前に", "直ちに": r"直ちに", "いつでも": r"いつでも"}

def key_terms(s):
    c = Counter({k: len(re.findall(p, s)) for k, p in KEY_PATTERNS.items()})
    c.update(re.findall(r"\d+(?:\.\d+)?(?:円|%|日|か月|年間|年|時間|件)", mask_refs(s)))
    return +c

def term_changes(a, b):
    ca, cb = key_terms(a), key_terms(b)
    return ", ".join([f"-{k}" for k in (ca - cb)] + [f"+{k}" for k in (cb - ca)])

def row(old_no, new_no, status, title, old_text, new_text, terms="", diff=""):
    return {"旧": old_no, "新": new_no, "区分": status, "見出し": title, "要注意の語句": terms, "差分": diff,
            "old_text": old_text, "new_text": new_text}

report = []
for i, j in sorted(pairs, key=lambda p: p[1]):
    a, b = old[i], new[j]
    a_body = remap_refs(a["body"])
    title = b["title"] if a["title"] == b["title"] else f"{a['title']}→{b['title']}"
    if (a["title"], a_body) == (b["title"], b["body"]):
        report.append(row(a["no"], b["no"], "変更なし", title, a_body, b["body"]))
    else:
        report.append(row(a["no"], b["no"], "変更", title, a_body, b["body"],
                          term_changes(a_body, b["body"]), word_diff(a_body, b["body"])))
matched_old, matched_new = {i for i, _ in pairs}, {j for _, j in pairs}
report += [row(None, b["no"], "新設", b["title"], "", b["body"]) for j, b in enumerate(new) if j not in matched_new]
report += [row(a["no"], None, "削除", a["title"], a["body"], "") for i, a in enumerate(old) if i not in matched_old]
report = pd.DataFrame(report).astype({"旧": "Int64", "新": "Int64"})

# 変更のあった条の差分は、切れないように全文を表示する
for _, r in report[report["区分"] == "変更"].iterrows():
    print(f"旧{r['旧']}→新{r['新']} {r['見出し']}\n{r['差分']}\n")
report.drop(columns=["差分", "old_text", "new_text"])
```

```python
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"   # SB Intuitions、MIT ライセンス。初回はダウンロード（約 6.4GB）を含めて数分。VRAM 約 7GB
tokenizer = AutoTokenizer.from_pretrained(name)
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
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

def summarize_change(r):
    if r["区分"] == "新設":
        return chat(f"利用規約の改定で、次の条項が新しく設けられました。何が定められたかを 1 文で書いてください。\n\n（{r['見出し']}）{r['new_text']}")
    if r["区分"] == "削除":
        return chat(f"利用規約の改定で、次の条項が削除されました。何がなくなったかを 1 文で書いてください。\n\n（{r['見出し']}）{r['old_text']}")
    return chat("利用規約の条項が改定されました。旧版と新版、差分 ([-削除-] {+追加+})、機械的に検出した語句の変化を示します。"
                "何がどう変わったかを 1 文で、会員にとって有利か不利かを 1 文で書いてください。差分に表れた変更は、小さなものでも省かないでください。\n\n"
                f"旧版: {r['old_text']}\n新版: {r['new_text']}\n差分: {r['差分']}\n語句の変化: {r['要注意の語句'] or 'なし'}")

changed = report[report["区分"] != "変更なし"].copy()
changed["要点"] = [summarize_change(r) for _, r in changed.iterrows()]   # 6 条ぶんの生成。モデルの読み込みと合わせて、検証の GPU で約 15 秒（ダウンロード済みのとき）。T4 ではその 2〜3 倍が目安

def no(x):
    return "-" if pd.isna(x) else x

for _, r in changed.iterrows():
    print(f"[{r['区分']}] 旧{no(r['旧'])}→新{no(r['新'])} {r['見出し']}  {r['要注意の語句']}")
    print(f"   {r['要点']}")
```

```python
# 条ごとの要点から、改定全体の要約を作る
items = "\n".join(f"- {r['区分']} ({r['見出し']}): {r['要点']}" for _, r in changed.iterrows())
print(chat(f"利用規約の改定内容を条項ごとにまとめたものです。会員への影響が大きい順に、5 項目以内の箇条書きにしてください。各項目は 1 文にしてください。\n\n{items}", max_new_tokens=400))
```

- 条番号だけで対応づけると、条項を 1 つ新設しただけで、新設と削除の間にある条（ここでは第2〜9条）がすべて「変更」に見える（最初の表。第10条は削除で番号のずれが打ち消されて一致した）。見出しと本文の類似度で対応づけ、`scipy.optimize.linear_sum_assignment` で割当問題を解いて 1 対 1 の組を選ぶと、番号のずれ、新設、削除を区別できる。
- 類似度は、文埋め込みと文字列の重なり (difflib) の平均にした。multilingual-e5-small のコサイン類似度は無関係な条どうしでも高く、上の出力では対応した組の最小 0.98 に対しそれ以外の最大が 0.94 だった。平均を取ると 0.90 と 0.66 に開くので、しきい値（ここでは 0.75）を置きやすい。見出しや言い回しを大きく書き換えた条では値が下がるので、しきい値は表を見て決める。
- 本文中の「第4条に違反した場合」のような条番号の参照は、番号が繰り下がっただけでも差分になる。対応づけでは伏せ、比べるときは旧版の番号を新版の番号に読み替えると、実質的な変更だけが残る（上の退会の条は「変更なし」になった）。
- 「することができます」→「しなければなりません」は文全体の類似度にほとんど表れない（パスワードの条を含め、対応した組の文埋め込みはすべて 0.98 以上）が、任意の通知が義務に変わる重要な改定である。変更の有無は文字列の一致で判定し、義務・禁止・免責などの語句と数値の数の変化を「要注意の語句」の列に出しておく。ただし、号の追加のように語句の数が変わらない変更はこの列に表れない（上の禁止事項の条）。差分の全文も確かめる。
- LLM には差分と語句の変化も渡し、どこが変わったかを先に示しておく。それでも細部は落ちる。上の出力では、免責の条の要点が旧版を「重大な過失がない限り」と説明して「故意又は」を落とし、全体の要約にもその誤りが引き継がれた。全体の要約には「月額料金が50円から増額され」という崩れた表現も混ざり、「各項目は 1 文」の指示も守られていない。有利・不利の判断もモデルの意見にすぎないので、最終的な判断は差分の一覧で行う。
- 実際の規約では、PDF や HTML から条の区切りを正しく取り出すのに手間がかかることが多い（このレシピでは試していない）。項や号（「2」「(1)」）の単位まで分けると、どこが変わったかをさらに細かく示せる。
- データとモデルの出典とライセンス: 新旧の規約は説明のために作った架空のもの。intfloat/multilingual-e5-small (MIT)、sbintuitions/sarashina2.2-3b-instruct-v0.1（SB Intuitions、MIT）。sarashina のトークナイザーを直すコードについては、大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」の注意書きを参照。

## 記事の見出しを生成して実際の見出しと比べる {headline_generation gpu}

記事の本文から、内容を一言で表す見出し（タイトル）を作りたい。極端に短い要約として、生成結果を実際の見出しと比べたい。

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
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる（大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」を参照）。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
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
- 実際の見出しは、読者の興味を引くための表現（煽り、伏せ字、疑問形）を含むことが多く、内容の要約とは限らない。ROUGE が低くても、見出しとして悪いとは言えない。
- 12 記事では平均値のばらつきが大きい。手法を比べるなら、数百件以上で評価し、信頼区間を付ける（評価指標と統計的検定の章を参照）。
- 「記事にない情報は入れない」と指示しても、固有名詞や数字を作ってしまうことがある。公開する見出しに使うなら、人の確認が必要。

## 日本語の T5 をファインチューニングして見出しを生成する (Seq2SeqTrainer) {t5_finetune gpu}

記事の本文から見出しを作るような「文から文へ」の課題を、数千件の正解例で系列変換モデル（エンコーダ・デコーダ型の T5）に学習させたい。LLM にプロンプトで頼むより小さく速いモデルで、データの書き方（見出しの文体）に合わせた出力を得たい。

```python
!pip install -q rouge-score fugashi unidic-lite
```

```python
import io, tarfile, urllib.request
import pandas as pd

def load_livedoor():
    """livedoor ニュースコーパス（株式会社ロンウイット配布）。記事は CC BY-ND 2.1 JP（表示・改変禁止）"""
    url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
    with urllib.request.urlopen(url) as r:
        tar = tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")
    rows = []
    for m in tar.getmembers():
        parts = m.name.split("/")
        if m.isfile() and len(parts) == 3 and parts[2] != "LICENSE.txt":
            lines = tar.extractfile(m).read().decode("utf-8").split("\n")
            rows.append({"label": parts[1], "title": lines[2].strip(), "text": "\n".join(lines[3:]).strip()})
    return pd.DataFrame(rows)

df = load_livedoor()
# この章の「記事の見出しを生成して実際の見出しと比べる」と同じ 12 記事（LLM との比較用）。学習には使わない
same12 = df[df["label"].isin(["topic-news", "sports-watch"])].sample(12, random_state=1)
rest = df.drop(same12.index).sample(frac=1, random_state=0).reset_index(drop=True)
test, valid, train = rest[:300], rest[300:500], rest[500:4500]       # 評価 300、検証 200、学習 4,000 記事
print(len(df), "記事 → 学習", len(train), "/ 検証", len(valid), "/ 評価", len(test))
print(train[["label", "title"]].head(3).to_string())
```

```python
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from tokenizers import decoders, models
from transformers import AutoTokenizer, PreTrainedTokenizerFast

name = "sonoisa/t5-base-japanese-v1.1"     # 日本語の T5（v1.1 の構造、約 2.5 億パラメータ）。CC BY-SA 4.0
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "spiece.model"))

def load_tokenizer(fix=True):
    tokenizer = AutoTokenizer.from_pretrained(name)
    if fix:
        # transformers 5 系の T5Tokenizer は byte_fallback を落とし、語彙にない文字（大文字の英字の一部など）を <unk> にしてしまう。
        # 元の spiece.model から語彙とスコアを読んで Unigram を組み立て直し、デコード側にもバイト列を文字に戻す処理を入れる
        tokenizer.backend_tokenizer.model = models.Unigram(
            [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
        tokenizer.backend_tokenizer.decoder = decoders.Sequence(
            [decoders.ByteFallback(), decoders.Metaspace(replacement="▁", prepend_scheme="always", split=True)])
    return tokenizer

# SentencePiece 本体と分割・復元が一致するかを、評価用の記事の見出しと本文の先頭で確かめる
texts = test["title"].tolist() + [t[:200] for t in test["text"]]
for fix in [False, True]:
    tok = load_tokenizer(fix)
    enc_ok = sum(tok(s, add_special_tokens=False)["input_ids"] == sp.encode(s) for s in texts)
    dec_ok = sum(tok.decode(sp.encode(s)) == sp.decode(sp.encode(s)) for s in texts)
    print(f"修正{'後' if fix else '前'}: 分割が一致 {enc_ok}/{len(texts)}  復元が一致 {dec_ok}/{len(texts)}")
    s = "【Sports Watch】KDDIのiPhone 5"
    print("   ", tok.tokenize(s), "→", tok.decode(tok(s)["input_ids"], skip_special_tokens=True))
tokenizer = load_tokenizer()

# 修正後のトークナイザーを保存して読み直すときは、読み込みのクラスで結果が変わる
tokenizer.save_pretrained("t5-headline-tokenizer")
for label, cls in [("AutoTokenizer", AutoTokenizer), ("PreTrainedTokenizerFast", PreTrainedTokenizerFast)]:
    tok = cls.from_pretrained("t5-headline-tokenizer")
    print(f"{label} で読み直す: 分割が一致 {sum(tok(s, add_special_tokens=False)['input_ids'] == sp.encode(s) for s in texts)}/{len(texts)}")
```

```python
import torch
from transformers import AutoModelForSeq2SeqLM, set_seed

set_seed(0)
# 学習するので float32 のまま読み込む。「tied weights … NOT tie them」の警告が出るが、このモデルは入力と出力の埋め込みを別々に持つ設定なので無視してよい
model = AutoModelForSeq2SeqLM.from_pretrained(name).to("cuda")
print(f"{model.num_parameters() / 1e6:.0f}M パラメータ")
SRC_LEN, TGT_LEN = 256, 64        # 本文は先頭 256 トークン（見出しに要る情報はたいてい冒頭にある）、見出しは 64 トークンまで

def generate(texts, batch_size=32, **kwargs):
    model.eval()
    outputs = []
    for i in range(0, len(texts), batch_size):
        x = tokenizer(list(texts[i:i + batch_size]), max_length=SRC_LEN, truncation=True, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**x, max_new_tokens=TGT_LEN, **kwargs)
        outputs += tokenizer.batch_decode(out, skip_special_tokens=True)
    return outputs

samples = test.head(3)
before = generate(samples["text"], num_beams=4)
for t, g in zip(samples["title"], before):
    print(f"実際: {t}\n学習前: {g!r}\n")

# T5 v1.1 の構造は float16 で計算すると値があふれる。損失を float16 と float32 で比べる
x = tokenizer(list(samples["text"]), max_length=SRC_LEN, truncation=True, padding=True, return_tensors="pt").to(model.device)
x["labels"] = tokenizer(text_target=list(samples["title"]), max_length=TGT_LEN, truncation=True, padding=True, return_tensors="pt")["input_ids"].to(model.device)
x["labels"][x["labels"] == tokenizer.pad_token_id] = -100
with torch.no_grad():
    loss32 = model(**x).loss.item()
    with torch.autocast("cuda", dtype=torch.float16):
        loss16 = model(**x).loss.item()
print(f"損失 float32: {loss32:.3f}  float16 (autocast): {loss16}")
```

```python
import time
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

def encode(batch):
    x = tokenizer(batch["text"], max_length=SRC_LEN, truncation=True)
    x["labels"] = tokenizer(text_target=batch["title"], max_length=TGT_LEN, truncation=True)["input_ids"]
    return x

to_ds = lambda d: Dataset.from_pandas(d[["text", "title"]], preserve_index=False).map(encode, batched=True, remove_columns=["text", "title"])
train_ds, valid_ds = to_ds(train), to_ds(valid)

BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 だと損失が nan になるので float32 で学習する
args = Seq2SeqTrainingArguments(
    output_dir="t5-headline", num_train_epochs=1, per_device_train_batch_size=16, per_device_eval_batch_size=32,
    learning_rate=3e-4, warmup_steps=25, lr_scheduler_type="linear", weight_decay=0.01,
    eval_strategy="steps", eval_steps=50, logging_steps=50, save_strategy="no", report_to="none",
    bf16=BF16, seed=0, disable_tqdm=True)
trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=valid_ds,
                         data_collator=DataCollatorForSeq2Seq(tokenizer, model=model), processing_class=tokenizer)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
trainer.train()        # 4,000 記事 × 1 エポック（250 ステップ）。T4 (float32) では 10 分前後の見込み
print(f"学習 {time.time() - t0:.0f} 秒、GPU メモリの最大 {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

curve = pd.DataFrame([{"学習した記事数": h["step"] * 16, "検証の損失": round(h["eval_loss"], 3)} for h in trainer.state.log_history if "eval_loss" in h])
print(curve.to_string(index=False))
```

```python
import numpy as np
from fugashi import Tagger
from rouge_score import rouge_scorer

tagger = Tagger()

class Tok:
    def tokenize(self, text):
        return [w.surface for w in tagger(text)]

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=Tok())

for t, b, g in zip(samples["title"], before, generate(samples["text"], num_beams=4)):
    print(f"実際: {t}\n学習前: {b!r}\n学習後: {g}\n")

def evaluate(refs, hyps):
    s = [scorer.score(r, h) for r, h in zip(refs, hyps)]
    return {"ROUGE-1": np.mean([x["rouge1"].fmeasure for x in s]), "ROUGE-2": np.mean([x["rouge2"].fmeasure for x in s]),
            "ROUGE-L": np.mean([x["rougeL"].fmeasure for x in s]), "平均字数": np.mean([len(h) for h in hyps])}

def diff_ci(refs, hyps_a, hyps_b, n_boot=1000):
    """記事ごとの ROUGE-L の差 (b − a) を、同じ記事の組で再標本化して 95% 信頼区間を求める（対応のあるブートストラップ）"""
    d = np.array([scorer.score(r, b)["rougeL"].fmeasure - scorer.score(r, a)["rougeL"].fmeasure for r, a, b in zip(refs, hyps_a, hyps_b)])
    boot = np.random.default_rng(0).choice(d, (n_boot, len(d))).mean(1)
    return f"{d.mean():+.3f}（{np.percentile(boot, 2.5):+.3f}〜{np.percentile(boot, 97.5):+.3f}）"

settings = {"貪欲法": dict(num_beams=1),
            "ビーム 4": dict(num_beams=4),
            "ビーム 4 + 3-gram の繰り返し禁止": dict(num_beams=4, no_repeat_ngram_size=3),
            "サンプリング (top_p=0.9)": dict(do_sample=True, top_p=0.9)}
rows, outputs = [], {}
for label, kw in settings.items():       # 評価 300 記事 × 4 通り。T4 では合わせて 3〜4 分の見込み
    set_seed(0)
    t0 = time.time()
    outputs[label] = generate(test["text"], **kw)
    rows.append({"デコード": label, **evaluate(test["title"], outputs[label]),
                 "ROUGE-L の差（対 貪欲法、95%CI）": "" if label == "貪欲法" else diff_ci(test["title"], outputs["貪欲法"], outputs[label]), "秒": time.time() - t0})
print("実際の見出しの平均字数", round(test["title"].str.len().mean(), 1))
pd.DataFrame(rows).round(3)
```

```python
import re

# この章の「記事の見出しを生成して実際の見出しと比べる」と同じ 12 記事で、本文の先頭 30 字（同じレシピのベースライン）と比べる
lead = [t.strip().split("。")[0][:30] for t in same12["text"]]
t5 = generate(same12["text"], num_beams=4, no_repeat_ngram_size=3)
untag = lambda xs: [re.sub(r"【[^】]*】", "", x).strip() for x in xs]     # 【Sports Watch】のような、連載やカテゴリの決まった表記を除く
print(pd.DataFrame([{"手法": "T5（ファインチューニング）", **evaluate(same12["title"], t5)},
                    {"手法": "T5（【】の表記を除いて評価）", **evaluate(untag(same12["title"]), untag(t5))},
                    {"手法": "本文の先頭 30 字", **evaluate(same12["title"], lead)}]).round(3).to_string(index=False))
for t, g in list(zip(same12["title"], t5))[:3]:
    print(f"実際: {t}\nT5  : {g}")

# 評価 300 記事で、【】の表記を含む見出しの割合と、それが実際の見出しと一致した割合
tag = lambda x: re.findall(r"【[^】]*】", x)
hyps = outputs["ビーム 4 + 3-gram の繰り返し禁止"]
print(f"【】を含む: 実際 {np.mean([bool(tag(t)) for t in test['title']]):.0%}  T5 {np.mean([bool(tag(h)) for h in hyps]):.0%}  "
      f"T5 が付けた【】が実際の見出しにもある {np.mean([set(tag(h)) <= set(tag(t)) for t, h in zip(test['title'], hyps) if tag(h)]):.0%}")
```


- 使い分け: この章の「記事の見出しを生成して実際の見出しと比べる」は、指示に従う LLM (3B) にプロンプトで頼むので学習データが要らない。T5 は数千件の正解例と学習の時間が要るかわりに、モデルが小さく（248M パラメータ）、データの見出しの書き方まで真似る。同じ 12 記事で、上の T5 は ROUGE-1 F 0.438（【】の表記を除くと 0.415）、そのレシピの LLM はノートブックの出力で 0.332 だった。ただし LLM には「30 字以内」と指示しているので平均 27 字と短く、入力も本文の先頭 600 字で、条件がそろっていない。12 記事ではばらつきも大きく、再実行で T5 の値は 0.39〜0.44 の間で変わった。
- LLM の学習と調整の章の「ベースモデルを LoRA で指示チューニングする (SFT)」は、LLM の重みを固定して一部だけを学習する。T5 は全パラメータを学習するが、モデルが小さいので T4 の float32 でも収まる（この学習で GPU メモリの最大 10.4GB）。入力と出力の形が決まった変換を大量に処理するなら系列変換モデル、指示を変えながら色々な課題に使うなら LLM、と考えるとよい。
- 学習前のモデルは事前学習（文中の穴埋め）しかしていないので、本文の断片を繰り返すだけで見出しにならない。v1.1 の構造の T5 は float16 で計算すると値があふれ、上の出力でも損失が nan になった。T4 では float32 で学習し（`fp16=True` にしない）、bfloat16 が使える GPU (A100, L4) では `bf16=True` にする。
- データの量と時間の目安: 4,000 記事を 1 エポック学習して、手元の GPU（float32、ほかの処理と共有）で 259 秒だった。検証の損失は 800 記事で 2.605、1,600 記事で 2.348、4,000 記事で 2.206 と、増やすほど下がり方が小さくなる（1 回の学習の途中の値なので目安）。時間を減らすなら、学習する記事数か `SRC_LEN` を減らす。
- デコードの設定: ビーム幅 4 のビーム探索は、貪欲法より ROUGE-L が 0.017 高い（95% 信頼区間 +0.002〜+0.033）。3-gram の繰り返しを禁止しても ROUGE-L は同じで、平均字数が 37.6 字から 35.5 字に縮んだ（実際の見出しは平均 36.7 字）。貪欲法は短め（31.9 字）で、ビームの半分ほどの時間で済む。サンプリングは ROUGE-L が 0.033 低いが、毎回違う候補が出るので、複数の案から人が選ぶ使い方に向く。
- 学習データの書き方をそのまま覚える: 【Sports Watch】のような連載やカテゴリの表記は実際の見出しの 33% にしかないのに、T5 は 65% に付け、付けた表記のうち実際の見出しにもあったのは 17% だった（上の浅田真央の例も、実際の見出しには表記がない）。不要なら、学習データの見出しから表記を除いておくか、生成後に取り除く。
- トークナイザーの不具合: transformers 5 系の T5Tokenizer は、このモデルの SentencePiece の byte_fallback を落とすので、語彙にない文字が `<unk>` になり、デコードで消える（上の「【ports atch】」）。修正すると、600 文で分割も復元も SentencePiece と一致した。ほかのモデル（rinna など）の直し方と違い、デコーダーにもバイト列を文字に戻す処理が要る。保存したトークナイザーは、上の出力のとおり AutoTokenizer で読むと修正が失われるので、`PreTrainedTokenizerFast.from_pretrained` で読み直す。
- データとモデルの出典とライセンス: livedoor ニュースコーパスの記事は CC BY-ND 2.1 JP（表示・改変禁止）。sonoisa/t5-base-japanese-v1.1 は、日本語の Wikipedia（2022 年 6 月 27 日時点）、OSCAR、CC-100 で事前学習した T5 で、CC BY-SA 4.0。モデルカードは Common Crawl の利用規約も守るよう求めている。学習したモデルを配布するなら同じライセンスを継承する必要があり、改変禁止の記事で学習したモデルや生成した見出しを公開するときは、扱いを確かめる。

## 文章をやさしい日本語に書き換える（テキスト平易化） {text_simplification gpu}

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
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる（大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」を参照）。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "tokenizer.model"))
tokenizer.backend_tokenizer.model = models.Unigram(
    [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())], unk_id=sp.unk_id(), byte_fallback=True)
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="auto").eval()

def chat(messages, max_new_tokens=200):
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(llm.device)
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
# 1. 本当にやさしくなったか: 文の長さ、漢字の割合、語彙の頻度（よく使われる語ほど Zipf 値が高い）で比べる
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
# 2. 意味が保たれているか: 平易化した文章の内容が原文から言えるか（含意）と、原文の要点が残っているか（逆向きの含意）
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
- 参照となる平易文がある場合は SARI（追加・削除・保持の適否を測る指標）がよく使われる。日本語のデータには SNOW T15/T23（やさしい日本語対訳コーパス）、JADES などがある。

## 難しい単語を見つけてやさしい単語に言い換える（語彙平易化） {lexical_simplification}

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
- 単語の難しさは、頻度でかなり近似できる。日本語教育では、語彙の級（旧日本語能力試験の級など）や学習者コーパスでの頻度も使われる。誰にとって難しいか（子ども、学習者、専門外の人）で基準は変わる。
- マスク言語モデルは文脈に合う語を出すが、意味が同じとは限らない（反対語や、単に文脈に合うだけの別の語が出る）。確率の低い候補を捨て、良い候補がなければ言い換えない、という慎重な設定にする。それでも意味のずれは残るので、結果は人が確認する。
- 活用語は、候補の語を元の語と同じ活用形にそろえる必要がある。ここでは形の合う候補だけを残す簡易な方法をとっている。
- LLM に「この語をやさしく言い換えて」と頼む方法が、現在は最も精度が高い。その場合も、候補が本当にやさしいか（頻度）と、意味が保たれているかの確認は別に行う。
- 単語単位の言い換えでは、文の構造の難しさは解消しない。文全体の平易化は、この章の「文章をやさしい日本語に書き換える（テキスト平易化）」を参照。

## 文体を変換する（です・ます調、だ・である調、敬語） {style_transfer_ja gpu}

文章の文体を、です・ます調とだ・である調の間で揃え直したい。くだけた文を丁寧な文に言い換えたい。活用形を使った規則と LLM で変換し、文体が変わったか、意味が保たれているか、余計な書き換えがないかを測って比べる。

```python
!pip install -q fugashi unidic-lite
```

```python
# 例文はこのレシピのために作ったもの。常体と敬体の対 30 組（両方向の変換の正解として使う）と、くだけた文 12 文と敬語の言い換えの一例
PAIRS = [  # （だ・である調, です・ます調）
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
CASUAL = [  # （くだけた文, 敬語の言い換えの一例。正解は 1 つではない）
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
    """文末の文体の判定器: 文末の述語の部分（最後の自立語から後ろ）に「です」「ます」があれば敬体"""
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
    """「」の外に「です」「ます」が 1 つでもあるか（常体にしたい文章に、文中も含めて敬体が残っていないかを見る）"""
    depth, found = 0, False
    for w in tagger(text):
        depth += (w.surface == "「") - (w.surface == "」")
        found |= depth == 0 and w.feature.lemma in ("です", "ます")
    return found

def is_casual(text):
    """くだけた表現（縮約形、「じゃん」「けど」、疑問の「？」）が残っているか"""
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
    """動詞（と「れる・られる」などの助動詞）の基本形と活用型から、ます形・ない形・意志形・た形の語幹を作る"""
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
    """述語の末尾の語の列 t を常体 → 敬体に。（置き換える末尾の語数, 新しい文字列）を返す。0 語なら後ろに足す"""
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
    if s[-2:] == ["ます", "ず"] and (v := verb(2)):                       # ません（ありません → ない）
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
    """target="polite"（です・ます調）か "plain"（だ・である調）へ。文末と「が・けど」の前（常体へは接続助詞の前すべて）を変える。「」の中は変えない"""
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
            s = "申し訳ない"                                         # 「悪いけど」に限る（「天気が悪い」は変えない）
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
# 2. 規則を作るときに見ていない文でも試す: 青空文庫の 2 作品（どちらも著作権の保護期間が終わっている）
# 新美南吉「ごん狐」（です・ます調で語られる）の敬体の文 → 常体、芥川龍之介「羅生門」（だ・である調）の常体の文 → 敬体
# 「」を含む文は除く（何文もある台詞の途中の文は残る。羅生門の老婆の台詞など）
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
# 3. LLM による変換（この章の「LLM で文章を要約する（長さと観点を指定する）」と同じモデル）。読み込みを含めて T4 で 3〜5 分ほど
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from tokenizers import models
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

set_seed(0)
name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
# transformers 5 系は、このモデルの SentencePiece (Unigram) を BPE に変換してしまい、学習時と違う分割になる（大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」を参照）。
# 元の tokenizer.model から語彙とスコアを読み、分割の方式だけを Unigram に戻す（特殊トークンやチャットテンプレートはそのまま）
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
    """内容語の語彙素（表記の揺れ「分かる/わかる」は同じ語彙素になる）。補助的な動詞や形式名詞は除く"""
    return {w.feature.lemma for w in tagger(text) if w.feature.pos1 in ("名詞", "動詞", "形容詞", "副詞", "形状詞") and w.feature.lemma not in STOP}

def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]

TASKS = {  # 課題: （入力, 正解（なければ None）, 目標の文体）
    "→敬体": ([p for p, _ in PAIRS], [[q] for _, q in PAIRS], "敬体"),
    "→常体": ([q for _, q in PAIRS], [[p, p.replace("である", "だ")] for p, _ in PAIRS], "常体"),   # 「である」「だ」どちらも正解
    "くだけ→敬語": ([c for c, _ in CASUAL], None, "敬体"),
    "ごん狐→常体": (gon, None, "常体"),
    "羅生門→敬体": (rasho, None, "敬体"),
}
# 尊敬語・謙譲語の目印（簡単な正規表現で、すべては拾えない）。「申し訳」と「〜ておく」（ておいた、でおく）の「お」は除く
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
```

```python
# 6. くだけた文 → 敬語の全 12 文を、規則・LLM・言い換えの一例で並べる
print("■ くだけた文 → 敬語")
for (c, ref), r_out, l_out in zip(CASUAL, rule["くだけ→敬語"], llm_out["くだけ→敬語"]):
    print(f"{c}\n   規則: {r_out}\n   LLM : {l_out}\n   例  : {ref}")
```

- 文体の変換 (style transfer) は、内容を保ったまま書き方だけを変える書き換え。この章の「文章をやさしい日本語に書き換える（テキスト平易化）」と同じく、文体が変わったか、意味が保たれているか、余計な書き換えがないかを分けて測る。です・ます調とだ・である調の変換は文末だけが変わるはずなので、平易化のような含意 (NLI) の判定でなく、内容語の一致と正解との編集距離で書き換えすぎを見つけられる。平易化のプロンプトの「文末は「です」「ます」にする」が守られたかも、上の判定器で確かめられる。
- 規則による方法は、形態素解析の活用型と活用形（日本語の形態素解析と文分割の章の「動詞・形容詞の活用型と活用形を調べる」）から「行った → 行きました」「広くない → 広くありません」を作る。作例では両方向とも 30/30 が正解と一致したが、規則を作りながら見ていた文なので楽観的な数字である。見ていない青空文庫の文（変換が要る文だけを選んだ）では、ごん狐 → 常体が 50/50、羅生門 → 敬体が 49/50 で、失敗は古い否定の「ぬ」だった。判定器は成功としたが、老婆の台詞の「であろ → です」（推量が落ちた）や「これだぞよ → これですぞよ」のような誤りもある。
- LLM は「文末の表現だけを変え」と指示しても書き換えすぎる。作例で正解と一致したのは →敬体 23/30、→常体 22/30 で、「から → ため」「分かりません → 不明だ」「歩いて通っています → 歩いている」（「通う」が落ちた）のように、文体は変わったのに内容語が増減した文が 23 あった。
- LLM は変換し損ねも多い。ごん狐 → 常体は 41/50 で、失敗した 9 文のほとんどは文末が「〜ました」「〜ます」のまま、表記の変更（「だまって → 黙って」）や誤り（「まきついた → まきつくった」）だけが入っていた。「（すでにだ・である調になっていますので、変更は不要です。）」と説明を付け足した文もある。規則なら、文中の「〜ますと」「〜ましたが」もまとめて変えられる。
- くだけた文 → 敬語では、規則は丁寧語と縮約形（「待ってる」「言っといた」）の復元しかできず、尊敬語・謙譲語は 0/12、省かれた助詞も戻らない。LLM は目印の正規表現で数えて 10/12 で尊敬語・謙譲語を使ったが（「ご出席されますか」のように適否が分かれる表現もある）、内容語の保持は 0.39 で、「遅れる → 遅刻する」「やめとこう → 中止にいたしましょうか」のように語や意図まで変わる。敬語では「見る → ご覧になる」のように語が替わるのが正しいこともあるので、内容語の一致では測れず、含意の判定や人の確認が要る。
- 判定器は正解の文をすべて正しく判定した（常体 31/31、敬体 44/44）が、文末しか見ず、規則と同じ形態素解析器を使うので、規則の誤り（上の「であろ」）を見逃す。くだけた表現の判定 (is_casual) も縮約形や「じゃん」「けど」「？」しか見ず、「かも」や省かれた助詞は拾わないので、規則のくだけ → 敬語の成功率 1.00 は甘い。尊敬語・謙譲語の目印も簡単な正規表現で、すべては拾えない。機械翻訳の章の「タグを付けて訳文の文体を制御する（丁寧さの制御）」も、正規表現の判定器で文体の成功率を測っている。
- 評価は 30〜50 文、1 モデル、貪欲法での結果で、数文の差はばらつきの範囲と考える。実際には、規則で変換して判定器が失敗とした文だけを LLM に回す、LLM の出力を判定器と内容語の一致で検査する、といった組み合わせが考えられる（試していない）。例文はこのレシピのための作例、青空文庫の 2 作品は著者の没年（1943 年、1927 年）から著作権の保護期間が終わっている。sarashina2.2 は SB Intuitions のモデルで MIT ライセンス。
