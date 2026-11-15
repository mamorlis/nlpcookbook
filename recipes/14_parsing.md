# 構文解析

係り受け (依存構造) と句構造の解析。文の骨組みを取り出し、評価し、解析の仕組みを理解する。

## GiNZA で日本語の依存構造解析 (係り受け解析) をする {ginza_dependency}

日本語の文について、どの単語がどの単語に係っているか (主語・目的語などの関係) を解析したい。

```python
!pip install -q ginza ja_ginza
```

```python
import spacy

nlp = spacy.load("ja_ginza")
doc = nlp("太郎は図書館で花子に借りた本を返した。")
for t in doc:
    print(f"{t.i:2} {t.text:5} {t.lemma_:5} {t.pos_:6} {t.dep_:8} → {t.head.i:2} {t.head.text}")
```

```python
root = [t for t in doc if t.dep_ == "ROOT"][0]
print("述語 (ROOT):", root.text)
print("直接の子:", [(c.text, c.dep_) for c in root.children])
print("「本」の部分木:", "".join(t.text for t in doc[6].subtree))
print(spacy.explain("nsubj"), "/", spacy.explain("obl"), "/", spacy.explain("acl"))
```

- ラベルは Universal Dependencies (UD)。日本語でも英語でも同じ体系で扱える。
- より高精度な Transformer 版は `pip install ja_ginza_electra` して `spacy.load("ja_ginza_electra")`。
- spaCy 公式の `ja_core_news_sm/md/lg/trf` も同じ使い方ができる。

## 依存構造木を図で表示する (displaCy) {displacy_dep}

係り受けの解析結果を、矢印付きの図で確認したい。

```python
!pip install -q ginza ja_ginza
!python -m spacy download en_core_web_sm -q
```

```python
import spacy
from spacy import displacy

nlp = spacy.load("ja_ginza")
doc = nlp("太郎は花子に美しい花束を渡した。")
displacy.render(doc, style="dep", jupyter=True, options={"compact": True, "distance": 90})
```

```python
nlp_en = spacy.load("en_core_web_sm")
doc_en = nlp_en("The quick brown fox jumps over the lazy dog.")
displacy.render(doc_en, style="dep", jupyter=True, options={"distance": 100, "collapse_punct": True})
```

```python
# SVG ファイルに保存 (論文やスライド用)
svg = displacy.render(doc, style="dep", jupyter=False, options={"compact": True, "distance": 90})
with open("dep.svg", "w", encoding="utf-8") as f:
    f.write(svg)
```

- Colab 以外 (スクリプト) では `displacy.serve(doc)` でブラウザ表示できる。

## 文節に区切って文節単位の係り受けを得る {bunsetsu}

日本語の伝統的な単位である「文節」で区切り、「どの文節がどの文節に係るか」を知りたい (CaboCha 相当)。

```python
!pip install -q ginza ja_ginza
```

```python
import ginza
import spacy

nlp = spacy.load("ja_ginza")
doc = nlp("太郎は図書館で花子に借りた本を昨日返した。")

for sent in doc.sents:
    for span in ginza.bunsetu_spans(sent):
        head = ginza.bunsetu_span(span.root.head)
        target = "(文末)" if head.start == span.start else head.text
        print(f"{span.text:　<6} → {target:　<6} 主辞: {ginza.phrase(span.root)}")
```

```python
print("文節:", [s.text for s in ginza.bunsetu_spans(doc)])
print("主辞句:", [(s.text, s.label_) for s in ginza.bunsetu_phrase_spans(doc)])
```

- 文節 = 自立語 (+ 付属語)。UD の単語単位の係り受けより粗いが、日本語の直感に合いやすく、係り受けの交差も少ない。

## KWJA で文節・基本句の係り受けと並列構造を解析する {kwja_dependency}

京大コーパスの基準 (KNP 形式) で、文節単位・基本句単位の係り受けを得たい。並列や同格の関係も区別したい。

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

doc = kwja("太郎と花子は、昨日京都で買った抹茶とお菓子を先生に渡した。")
DEP = {"D": "係り受け", "P": "並列", "A": "同格", "I": "部分並列"}
print("文節:")
for p in doc.phrases:
    print(f"  {p.text:　<8} → {p.parent.text if p.parent else '(文末)':　<8} {DEP[p.dep_type.value]}")
```

```python
print("基本句 (文節をさらに細かくした、係り受けと述語項構造の単位):")
for bp in doc.base_phrases:
    print(f"  {bp.text:　<6} → {bp.parent.text if bp.parent else '(文末)':　<6} {bp.dep_type.value}  主辞: {bp.head.text}")

print([(m.text, m.reading, m.lemma, m.pos, m.subpos) for m in doc.morphemes][:6])
```

- 「太郎と → 花子は」「抹茶と → お菓子を」は、普通の係り受け (D) ではなく並列 (P) として区別される。UD (GiNZA) では並列も `nmod` になってしまうので、並列構造を扱いたいときはこちらが便利。
- 解析結果は KNP 形式。`rhoknp` で文 → 文節 → 基本句 → 形態素の階層をたどれる。
- 形態素は JUMAN の品詞体系・辞書に基づく。MeCab + UniDic とは単語の区切りも品詞名も違う点に注意。
- インストール時に Transformers のバージョンが入れ替わることがある。import でエラーになったらランタイムを再起動する。

## Stanza で多言語の形態素・依存構造解析をする {stanza}

日本語・英語以外も含めて、70 以上の言語を同じ API で解析したい (Universal Dependencies 準拠)。

```python
!pip install -q stanza
```

```python
import stanza

stanza.download("ja", verbose=False)
nlp = stanza.Pipeline("ja", processors="tokenize,pos,lemma,depparse", verbose=False)
doc = nlp("太郎は花子に本を渡した。")
for sent in doc.sentences:
    for w in sent.words:
        head = sent.words[w.head - 1].text if w.head > 0 else "ROOT"
        print(f"{w.id:2} {w.text:4} {w.lemma:4} {w.upos:6} {w.deprel:8} → {head}")
```

```python
# 言語を変えても同じコードで動く
for lang, text in [("de", "Der Hund jagt die Katze."), ("ko", "나는 학교에 갑니다.")]:
    stanza.download(lang, verbose=False)
    p = stanza.Pipeline(lang, processors="tokenize,pos,lemma,depparse", verbose=False)
    print(lang, [(w.text, w.upos, w.deprel) for s in p(text).sentences for w in s.words])
```

- `w.head` は 1 始まりの単語番号で、0 が ROOT。
- GPU があれば自動で使う。CPU に固定するなら `stanza.Pipeline(..., use_gpu=False)`。

## 係り受けのパターンで特定の構文を探す (DependencyMatcher) {dependency_matcher}

連体修飾節、受身、使役、並列のような特定の構文を含む箇所を、係り受けの形を指定してコーパスから取り出したい。

```python
!pip install -q ginza ja_ginza
!python -m spacy download en_core_web_sm -q
```

```python
import spacy
from spacy.matcher import DependencyMatcher

nlp = spacy.load("ja_ginza")
matcher = DependencyMatcher(nlp.vocab)

# 各パターンは「基準のトークン」と、そこからの関係 (> は直接の子、< は直接の親) で書く
matcher.add("連体修飾節", [[
    {"RIGHT_ID": "noun", "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}},
    {"LEFT_ID": "noun", "REL_OP": ">", "RIGHT_ID": "pred", "RIGHT_ATTRS": {"DEP": "acl"}},
]])
matcher.add("受身", [[
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "aux", "RIGHT_ATTRS": {"LEMMA": {"IN": ["れる", "られる"]}}},
]])
matcher.add("使役", [[
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "aux", "RIGHT_ATTRS": {"LEMMA": {"IN": ["せる", "させる"]}}},
]])
matcher.add("名詞の並列", [[
    {"RIGHT_ID": "second", "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}},
    {"LEFT_ID": "second", "REL_OP": ">", "RIGHT_ID": "first", "RIGHT_ATTRS": {"DEP": "nmod"}},
    {"LEFT_ID": "first", "REL_OP": ">", "RIGHT_ID": "particle", "RIGHT_ATTRS": {"ORTH": {"IN": ["と", "や", "および", "または"]}}},
]])

texts = ["太郎が昨日図書館で借りた本を花子が読んでいる。", "太郎は先生に褒められた。", "母は子供に野菜を食べさせた。",
         "りんごとみかんを買った。", "弟に大切なケーキを食べられた。", "走っている犬や寝ている猫を見た。"]
for doc in nlp.pipe(texts):
    print(doc.text)
    for match_id, ids in matcher(doc):
        print("   ", nlp.vocab.strings[match_id], [doc[i].text for i in ids])
```

```python
# 英語: 受動態と関係節
nlp_en = spacy.load("en_core_web_sm")
m = DependencyMatcher(nlp_en.vocab)
m.add("passive", [[
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subj", "RIGHT_ATTRS": {"DEP": "nsubjpass"}},
]])
m.add("relative clause", [[
    {"RIGHT_ID": "noun", "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}},
    {"LEFT_ID": "noun", "REL_OP": ">", "RIGHT_ID": "verb", "RIGHT_ATTRS": {"DEP": "relcl"}},
]])
doc = nlp_en("The paper that she wrote was rejected by the reviewers. The model was trained on data which we collected.")
for match_id, ids in m(doc):
    print(nlp_en.vocab.strings[match_id], [doc[i].text for i in ids])
```

- `REL_OP` には `>` (直接の子)、`<` (直接の親)、`>>` (子孫)、`<<` (祖先)、`.` (直後)、`;` (直前) などがある。
- 日本語の UD では、名詞の並列は `conj` ではなく「と」「や」の付いた `nmod` として表される。パターンを書く前に、狙った構文が実際にどう解析されるかを確認する。
- 「られる」は受身のほかに可能・尊敬・自発の用法があり、係り受けの形だけでは区別できない。
- 表層の語順で書くなら `Matcher`、係り受けの形で書くなら `DependencyMatcher`。語順が自由な日本語では後者が向いている場面が多い。

## 「誰が・何を・どうした」の三つ組を抜き出す {svo_extract}

文から 主語-目的語-述語 の組を取り出して、簡単な情報抽出や知識グラフの素材にしたい。

```python
!pip install -q ginza ja_ginza
```

```python
import ginza
import spacy

nlp = spacy.load("ja_ginza")

def phrase(token):
    """トークンを含む文節から、後ろに付く助詞などを除いた部分を返す"""
    return "".join(t.text for t in ginza.bunsetu_span(token) if t.i <= token.i)

def triples(text):
    out = []
    for sent in nlp(text).sents:
        for verb in sent:
            if verb.pos_ not in ("VERB", "ADJ"):
                continue
            args = {c.dep_: phrase(c) for c in verb.children if c.dep_ in ("nsubj", "obj", "iobj", "obl")}
            if "nsubj" in args:
                out.append((args.get("nsubj"), args.get("obj") or args.get("obl"), verb.lemma_))
    return out

for s in ["OpenAI が新しい言語モデルを発表した。", "山田教授は学生に自然言語処理を教えている。",
          "猫がネズミを追いかけ、犬が猫を追いかけた。", "政府は来年度の予算案を閣議で決定した。"]:
    print(s, "→", triples(s))
```

- 日本語は主語の省略が多く、「は」で示される主題が必ずしも主語ではない。本格的には述語項構造解析やゼロ照応解析が必要。
- 受身・使役では格と意味役割がずれる (「ネズミが猫に追いかけられた」)。

## 名詞句を抽出する {noun_chunks}

文から名詞句 (修飾語を含む名詞のまとまり) を取り出して、キーフレーズ候補や検索語にしたい。

```python
!pip install -q ginza ja_ginza
!python -m spacy download en_core_web_sm -q
```

```python
import spacy

nlp_en = spacy.load("en_core_web_sm")
doc = nlp_en("The new pretrained language model achieved impressive results on several machine translation benchmarks.")
for chunk in doc.noun_chunks:
    print(f"{chunk.text:42} 主辞: {chunk.root.text:12} 関係: {chunk.root.dep_:6} → {chunk.root.head.text}")
```

```python
nlp_ja = spacy.load("ja_ginza")
doc = nlp_ja("新しい事前学習済み言語モデルが、複数の機械翻訳ベンチマークで最高性能を達成した。")
print([c.text for c in doc.noun_chunks])
```

- `noun_chunks` は入れ子にならない「基本名詞句」を返す。前置詞句や関係節まで含めた大きい句は `token.subtree` で取る。

## 依存距離と構文木の深さで文の複雑さを測る {dep_distance}

文の構造的な複雑さを数値化したい (可読性の評価、学習者作文の分析、言語間の比較)。

```python
!pip install -q ginza ja_ginza
```

```python
import spacy

nlp = spacy.load("ja_ginza")

def depth(token):
    return 0 if token.head.i == token.i else 1 + depth(token.head)

def complexity(text):
    rows = []
    for sent in nlp(text).sents:
        toks = [t for t in sent if t.pos_ != "PUNCT"]
        dists = [abs(t.i - t.head.i) for t in toks if t.head.i != t.i]
        rows.append({"文": sent.text[:22], "語数": len(toks),
                     "平均依存距離": round(sum(dists) / max(1, len(dists)), 2),
                     "最大依存距離": max(dists, default=0),
                     "木の深さ": max(depth(t) for t in toks)})
    return rows

import pandas as pd

text = ("猫が寝ている。"
        "太郎が昨日図書館で借りた本を花子が読んでいる。"
        "私が先週友人から勧められて書店で買った小説を弟が勝手に持ち出して学校で友達に貸してしまった。")
pd.DataFrame(complexity(text))
```

- 平均依存距離 (MDD) は、係り元と係り先が離れているほど大きくなり、ワーキングメモリの負荷の指標とされる。
- 日本語は主辞後置型なので、長い連体修飾節があると依存距離が大きくなる。

## 解析結果を CoNLL-U 形式で保存・読み込みする {conllu}

依存構造解析の結果を、標準的な CoNLL-U 形式でファイルに保存したり、UD のツリーバンクを読み込んだりしたい。

```python
!pip install -q ginza ja_ginza conllu
```

```python
import spacy

nlp = spacy.load("ja_ginza")

def to_conllu(doc):
    lines = []
    for sid, sent in enumerate(doc.sents, 1):
        lines += [f"# sent_id = {sid}", f"# text = {sent.text}"]
        for i, t in enumerate(sent, 1):
            head = 0 if t.dep_ == "ROOT" else t.head.i - sent.start + 1
            feats = str(t.morph) or "_"
            misc = "_" if t.whitespace_ else "SpaceAfter=No"
            lines.append("\t".join([str(i), t.text, t.lemma_, t.pos_, t.tag_, feats, str(head), t.dep_.lower() if t.dep_ == "ROOT" else t.dep_, "_", misc]))
        lines.append("")
    return "\n".join(lines) + "\n"

text = to_conllu(nlp("太郎は本を読んだ。花子は寝ていた。"))
with open("parsed.conllu", "w", encoding="utf-8") as f:
    f.write(text)
print(text)
```

```python
from conllu import parse

with open("parsed.conllu", encoding="utf-8") as f:
    sentences = parse(f.read())
for sent in sentences:
    print(sent.metadata["text"], [(t["form"], t["upos"], t["head"], t["deprel"]) for t in sent])
print(sentences[0].to_tree().token["form"], "が ROOT")
```

- 10 列: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC。列はタブ区切り、文は空行区切り。
- GiNZA はコマンドラインでも `ginza -f conllu < input.txt` で CoNLL-U を出力できる。

## 依存構造解析の精度を UAS と LAS で評価する {dep_eval}

依存構造パーザの出力を正解のツリーバンクと比べて、係り先の正解率 (UAS) と、ラベルまで含めた正解率 (LAS) を計算したい。

```python
!pip install -q stanza conllu
```

```python
import urllib.request
import stanza
from conllu import parse

url = "https://raw.githubusercontent.com/UniversalDependencies/UD_Japanese-GSD/master/ja_gsd-ud-test.conllu"
gold = parse(urllib.request.urlopen(url).read().decode("utf-8"))[:150]     # UD Japanese-GSD のテストデータの先頭 150 文
gold = [[t for t in sent if isinstance(t["id"], int)] for sent in gold]    # 複数語トークンなどの行を除く
print(len(gold), "文", sum(map(len, gold)), "語")

# 評価では単語分割を正解に揃える (分割が違うと、係り受けを 1 対 1 で比べられない)
stanza.download("ja", verbose=False)
nlp = stanza.Pipeline("ja", processors="tokenize,pos,lemma,depparse", tokenize_pretokenized=True, verbose=False)
pred = nlp([[t["form"] for t in sent] for sent in gold])
```

```python
from collections import Counter

total = uas = las = 0
by_rel, by_rel_ok = Counter(), Counter()
errors = Counter()
for g_sent, p_sent in zip(gold, pred.sentences):
    for g, p in zip(g_sent, p_sent.words):
        if g["upos"] == "PUNCT":           # 句読点を除いて評価する流儀
            continue
        total += 1
        head_ok = g["head"] == p.head
        uas += head_ok
        las += head_ok and g["deprel"] == p.deprel
        by_rel[g["deprel"]] += 1
        by_rel_ok[g["deprel"]] += head_ok and g["deprel"] == p.deprel
        if head_ok and g["deprel"] != p.deprel:
            errors[g["deprel"], p.deprel] += 1
print(f"UAS = {uas / total:.4f}  LAS = {las / total:.4f}  ({total} 語)")
print("関係ごとの LAS:")
for rel, n in by_rel.most_common(12):
    print(f"  {rel:12} {by_rel_ok[rel] / n:.3f} ({n})")
print("係り先は合っているがラベルが違う:", errors.most_common(5))
```

```python
# 文単位の完全一致率と、文の長さごとの精度
exact, buckets = 0, {}
for g_sent, p_sent in zip(gold, pred.sentences):
    ok = [g["head"] == p.head for g, p in zip(g_sent, p_sent.words) if g["upos"] != "PUNCT"]
    exact += all(ok)
    key = "〜10 語" if len(ok) <= 10 else "11〜30 語" if len(ok) <= 30 else "31 語〜"
    buckets.setdefault(key, []).extend(ok)
print(f"文単位の完全一致率: {exact / len(gold):.3f}")
print({k: round(sum(v) / len(v), 3) for k, v in sorted(buckets.items())})
```

- UAS (Unlabeled Attachment Score) は係り先が合っている語の割合、LAS (Labeled Attachment Score) は係り先とラベルの両方が合っている語の割合。
- 句読点を含めるかどうかで値が変わる。CoNLL の公式評価は句読点も含めるので、比較するときは流儀を揃える。
- 生のテキストから解析した場合は、単語分割の誤りも含めて評価する必要がある。その場合は、文字位置で単語を対応付ける CoNLL 2018 Shared Task の公式スクリプト (`conll18_ud_eval.py`) を使う。
- 日本語では `case` (助詞) や `aux` (助動詞) のように、ほぼ機械的に決まる関係が全体の精度を押し上げる。内容語の係り受け (`nsubj`、`obl`、`acl` など) を個別に見るのが大事。

## 遷移型の依存構造解析の動きを追う (arc-standard) {transition_parsing}

スタックとバッファを使って、左から 1 語ずつ読みながら依存構造木を組み立てる「遷移型解析」の仕組みを、手を動かして理解したい。

```python
def oracle(words, heads):
    """正解の木 (heads[i] = i 番目の語の主辞。0 が ROOT、語は 1 始まり) を作るための遷移の列を求める"""
    n = len(words)
    n_children = [sum(1 for h in heads if h == i) for i in range(n + 1)]   # まだ付けていない子の数
    stack, buffer, steps, arcs = [0], list(range(1, n + 1)), [], []
    name = lambda i: "ROOT" if i == 0 else words[i - 1]
    while buffer or len(stack) > 1:
        action = "SHIFT"
        if len(stack) >= 2:
            s1, s0 = stack[-2], stack[-1]
            if s1 != 0 and heads[s1 - 1] == s0 and n_children[s1] == 0:
                action = "LEFT-ARC"       # スタックの 2 番目が、一番上の語に係る
            elif heads[s0 - 1] == s1 and n_children[s0] == 0:
                action = "RIGHT-ARC"      # スタックの一番上が、2 番目の語に係る
        if action == "SHIFT" and not buffer:
            raise ValueError("この木は arc-standard では作れません (交差する係り受けがある)")
        steps.append((action, [name(i) for i in stack], [name(i) for i in buffer]))
        if action == "SHIFT":
            stack.append(buffer.pop(0))
        elif action == "LEFT-ARC":
            arcs.append((name(s0), name(s1))); n_children[s0] -= 1; del stack[-2]
        else:
            arcs.append((name(s1), name(s0))); n_children[s1] -= 1; stack.pop()
    return steps, arcs

words = ["太郎", "が", "赤い", "本", "を", "読ん", "だ"]
heads = [6, 1, 4, 6, 4, 0, 6]          # 太郎→読ん, が→太郎, 赤い→本, 本→読ん, を→本, 読ん→ROOT, だ→読ん
steps, arcs = oracle(words, heads)
for action, stack, buffer in steps:
    print(f"{action:10} スタック: {' '.join(stack):24} バッファ: {' '.join(buffer)}")
print("作られた係り受け (主辞 → 従属語):", arcs)
```

```python
# 交差する係り受け (非交差性の違反) があるかを調べる
def is_projective(heads):
    arcs = [(min(i, h), max(i, h)) for i, h in enumerate(heads, 1)]
    return not any(a < c < b < d for a, b in arcs for c, d in arcs)

print(is_projective(heads))
# "A hearing is scheduled on the issue today": on(5) が hearing(2) に係り、is scheduled をまたぐ
en_words = ["A", "hearing", "is", "scheduled", "on", "the", "issue", "today"]
en_heads = [2, 4, 4, 0, 7, 7, 2, 4]
print(is_projective(en_heads))
try:
    oracle(en_words, en_heads)
except ValueError as e:
    print(e)
```

- 長さ n の文は、ちょうど 2n 回の遷移 (SHIFT が n 回、ARC が n 回) で解析できる。各時点でどの遷移を選ぶかを分類器で予測するのが遷移型パーザで、spaCy の parser もこの方式。
- 学習時は、ここで求めた「正解の遷移列」を教師データにする (スタックとバッファの状態 → 次の遷移)。
- 従属語をすべて付け終えてからでないと RIGHT-ARC できない (`n_children` の確認)。早く付けてしまうと、その語の子を付けられなくなる。
- 素の arc-standard は交差する係り受けを作れない。日本語の係り受けはほぼ交差しないが、チェコ語やドイツ語などでは無視できない頻度で現れるので、SWAP 遷移の追加などの拡張が使われる。

## スコアが最大の依存構造木を求める (最大全域木) {mst_decoding}

「どの語がどの語に係りそうか」のスコアが全ペアについて与えられたとき、全体として最も良い木を選びたい。グラフに基づく解析の復号を理解したい。

```python
import numpy as np

words = ["ROOT", "John", "saw", "Mary"]
# score[h, d] = 語 h が語 d の主辞である、というスコア (実際のパーザではニューラルネットが出力する)
score = np.array([
    [-np.inf,  9, 10,  9],    # ROOT → John / saw / Mary
    [-np.inf, -np.inf, 20,  3],    # John →
    [-np.inf, 30, -np.inf, 30],    # saw  →
    [-np.inf, 11,  0, -np.inf],    # Mary →
])

# 貪欲法: 各語について、スコアが最大の主辞を独立に選ぶ
greedy = {words[d]: words[int(score[:, d].argmax())] for d in range(1, len(words))}
print("貪欲法:", greedy)   # John → saw → John の循環ができて、木にならない
```

```python
import networkx as nx

def mst(score):
    """Chu-Liu/Edmonds のアルゴリズムで、ROOT (0 番) を根とする最大全域有向木を求める"""
    G = nx.DiGraph()
    n = len(score)
    for h in range(n):
        for d in range(1, n):
            if h != d and np.isfinite(score[h, d]):
                G.add_edge(h, d, weight=float(score[h, d]))
    tree = nx.maximum_spanning_arborescence(G)
    heads = {d: h for h, d in tree.edges}
    return [heads[d] for d in range(1, n)], sum(score[heads[d], d] for d in range(1, n))

heads, total = mst(score)
print("最大全域木:", {words[d]: words[h] for d, h in enumerate(heads, 1)}, "合計スコア:", total)
```

```python
# 総当たりで確かめる: すべての主辞の割り当てのうち、木になっているものの中で最大のもの
from itertools import product

def is_tree(heads):
    for d in range(1, len(heads) + 1):
        seen, cur = set(), d
        while cur != 0:
            if cur in seen:
                return False
            seen.add(cur); cur = heads[cur - 1]
    return True

best = max((h for h in product(range(4), repeat=3) if all(h[d - 1] != d for d in range(1, 4)) and is_tree(h)),
           key=lambda h: sum(score[h[d - 1], d] for d in range(1, 4)))
print("総当たり:", list(best), "一致:", list(best) == heads)
```

- グラフに基づく解析は「係り受けのスコアを付ける」と「スコア最大の木を探す」の 2 段階。前者を biaffine 注意機構で計算するのが、Stanza や GiNZA (Transformer 版) などで使われている現在の標準的な方式。
- 最大全域木は交差する係り受けも扱える。交差を許さない場合は Eisner のアルゴリズム (動的計画法、O(n³)) を使う。
- ROOT から出る辺が複数になりうる点に注意。「ROOT の子は 1 つ」という UD の制約を守らせるには追加の処理が要る。
- 遷移型は局所的な判断の積み重ねで速く (O(n))、グラフ型は文全体で最適化する。長距離の係り受けはグラフ型が有利とされる。

## spaCy で依存構造パーザを自分で学習する {spacy_train_parser}

ツリーバンクから依存構造パーザを学習して、精度を確かめたい。自分の分野のデータで学習し直すための手順を知りたい。

```python
!pip install -q ginza ja_ginza
```

```python
import urllib.request

base = "https://raw.githubusercontent.com/UniversalDependencies/UD_Japanese-GSD/master/ja_gsd-ud-"
for split, n in [("train", 1500), ("dev", 200)]:        # 短時間で終わるように一部だけ使う (全体は train 7050 文)
    sents = urllib.request.urlopen(base + split + ".conllu").read().decode("utf-8").strip().split("\n\n")
    with open(f"{split}.conllu", "w", encoding="utf-8") as f:
        f.write("\n\n".join(sents[:n]) + "\n\n")
    print(split, len(sents), "文中", n, "文を使用")
```

```python
# 1. CoNLL-U を spaCy のバイナリ形式に変換  2. 設定ファイルを生成  3. 学習
!python -m spacy convert train.conllu . --converter conllu -n 1 | tail -1
!python -m spacy convert dev.conllu . --converter conllu -n 1 | tail -1
!python -m spacy init config config.cfg --lang ja --pipeline parser --optimize efficiency | tail -1
!python -m spacy train config.cfg --output model --paths.train train.spacy --paths.dev dev.spacy --training.max_steps 600 --training.eval_frequency 200
```

```python
import spacy

nlp = spacy.load("model/model-best")
doc = nlp("太郎は図書館で花子に借りた本を返した。")
for t in doc:
    print(f"{t.i:2} {t.text:4} {t.dep_:8} → {t.head.text}")
```

```python
# 学習済みの GiNZA と、同じ開発データで比べる
!python -m spacy evaluate model/model-best dev.spacy | grep -E "UAS|LAS"
!python -m spacy evaluate ja_ginza dev.spacy | grep -E "UAS|LAS"
```

- 1500 文・600 ステップ (CPU で 1〜2 分) でも UAS は 80% 台後半になる。全データで十分に学習すれば 90% を超える。
- GiNZA のほうが低く出るのは、GiNZA が別のツリーバンク (UD Japanese-BCCWJ) の基準で学習されていて、単語分割やラベルの付け方が GSD と少し違うため。パーザの優劣ではなく、「評価データと同じ基準で学習したか」の差が出ている。
- `--optimize accuracy` と事前学習済みの Transformer を使う設定 (`--gpu-id 0`、`--pipeline transformer,parser`) にすると精度が大きく上がる。
- 単語分割は spaCy の日本語トークナイザ (SudachiPy) が行う。ツリーバンクの分割と合わない語は学習・評価で対応が取れないので、分割基準の一致が重要。
- 自分の分野のデータを数百文アノテーションして追加学習すると、分野適応の効果が大きい。その場合は既存モデルの重みから始める (`[initialize]` や `source` の設定)。

## 文脈自由文法を書いて構文解析する (NLTK) {cfg_parse}

自分で書いた小さな文法で文を解析し、構文木と構造的曖昧性を確かめたい (授業の演習向け)。

```python
import nltk

grammar = nltk.CFG.fromstring("""
    S   -> NP VP
    NP  -> Det N | Det N PP | 'I'
    VP  -> V NP | VP PP
    PP  -> P NP
    Det -> 'an' | 'my' | 'the'
    N   -> 'elephant' | 'pajamas' | 'telescope' | 'man'
    V   -> 'shot' | 'saw'
    P   -> 'in' | 'with'
""")
parser = nltk.ChartParser(grammar)

sentence = "I shot an elephant in my pajamas".split()
trees = list(parser.parse(sentence))
print(len(trees), "通りの解析")
for tree in trees:
    tree.pretty_print()
```

```python
# 確率文脈自由文法 (PCFG) で最も確からしい木を選ぶ
pcfg = nltk.PCFG.fromstring("""
    S   -> NP VP [1.0]
    NP  -> Det N [0.5] | Det N PP [0.2] | 'I' [0.3]
    VP  -> V NP [0.6] | VP PP [0.4]
    PP  -> P NP [1.0]
    Det -> 'an' [0.5] | 'my' [0.5]
    N   -> 'elephant' [0.5] | 'pajamas' [0.5]
    V   -> 'shot' [1.0]
    P   -> 'in' [1.0]
""")
best = next(nltk.ViterbiParser(pcfg).parse(sentence))
print(best.prob())
best.pretty_print()
```

- 「パジャマを着た象を撃った / パジャマ姿で象を撃った」という前置詞句付加の曖昧性の古典的な例。

## 構文木を括弧表記で読み書きし、操作する (NLTK Tree) {tree_bracket}

Penn Treebank 形式の括弧付き構文木を読み込んで、表示したり、部分木を取り出したり、文法規則を数えたりしたい。

```python
from nltk import Tree

s = "(S (NP (DT The) (JJ quick) (NN fox)) (VP (VBD jumped) (PP (IN over) (NP (DT the) (JJ lazy) (NN dog)))) (. .))"
tree = Tree.fromstring(s)
tree.pretty_print()
print("単語:", tree.leaves())
print("品詞:", tree.pos())
print("高さ:", tree.height(), " ラベル:", tree.label())
```

```python
# 部分木の取り出し
for sub in tree.subtrees(lambda t: t.label() == "NP"):
    print("NP:", " ".join(sub.leaves()))
print("PP の位置:", [pos for pos in tree.treepositions() if isinstance(tree[pos], Tree) and tree[pos].label() == "PP"])
print(tree[1, 1])                       # 位置を指定してアクセス (VP の 2 番目の子)

# 木から文法規則 (生成規則) を取り出す
for prod in tree.productions()[:6]:
    print(" ", prod)
```

```python
# ツリーバンクから PCFG を推定する
import nltk
nltk.download("treebank", quiet=True)
from nltk.corpus import treebank

productions = [p for t in treebank.parsed_sents()[:500] for p in t.productions()]
grammar = nltk.induce_pcfg(nltk.Nonterminal("S"), productions)
print(len(grammar.productions()), "規則")
for p in sorted(grammar.productions(lhs=nltk.Nonterminal("NP")), key=lambda p: -p.prob())[:5]:
    print(f"  {p.prob():.3f}  {p}")
```

```python
# 木の変形: チョムスキー標準形 (二分木) への変換と、括弧表記への書き戻し
t = Tree.fromstring("(NP (DT the) (JJ big) (JJ red) (NN ball))")
t.chomsky_normal_form()
print(t)
print(" ".join(str(tree).split()))      # 1 行の括弧表記に戻す
```

- 括弧表記は `(ラベル 子1 子2 ...)` の入れ子。句構造のツリーバンク (Penn Treebank、京都大学の NPCMJ、Stanza の構文解析結果など) はほぼこの形式で配布されている。
- CKY アルゴリズムなど多くの構文解析アルゴリズムは二分木を前提にするので、学習前に二分化し、評価前に元に戻す。
- 句構造解析の評価は、ラベル付き括弧の適合率・再現率 (PARSEVAL、`evalb`) で行う。

## 学習済みモデルで句構造解析をする (Stanza) {constituency_stanza}

文を名詞句・動詞句などの入れ子構造に解析したい。手書きの文法ではなく、ツリーバンクで学習したパーザを使いたい。

```python
!pip install -q stanza
```

```python
import stanza
from nltk import Tree

stanza.download("en", processors="tokenize,pos,constituency", verbose=False)
nlp = stanza.Pipeline("en", processors="tokenize,pos,constituency", verbose=False)

doc = nlp("The quick brown fox jumps over the lazy dog. I saw the man with the telescope.")
for sent in doc.sentences:
    print(sent.constituency)                                  # 括弧表記
    Tree.fromstring(str(sent.constituency)).pretty_print()    # NLTK の木に変換して表示
```

```python
# 名詞句と前置詞句を取り出す
def phrases(node, label):
    found = [" ".join(node.leaf_labels())] if node.label == label else []
    for child in node.children:
        found += phrases(child, label)
    return found

tree = doc.sentences[1].constituency
print("NP:", phrases(tree, "NP"))
print("PP:", phrases(tree, "PP"))
```

```python
# 日本語の句構造解析
stanza.download("ja", processors="tokenize,pos,constituency", verbose=False)
nlp_ja = stanza.Pipeline("ja", processors="tokenize,pos,constituency", verbose=False)
tree = nlp_ja("太郎は図書館で借りた本を花子に渡した。").sentences[0].constituency
print(tree)
Tree.fromstring(str(tree)).pretty_print()
```

- 英語のモデルは Penn Treebank 形式のラベル (NP、VP、PP、SBAR …) を出力する。日本語のモデルは ALT (Asian Language Treebank) で学習されており、ラベル体系が違う。
- 「with the telescope」が動詞句と名詞句のどちらに付くかは、パーザが 1 つの解釈を選ぶ。手書き文法のレシピのように、すべての解釈は列挙されない。
- 依存構造と句構造は相互に変換できる情報を多く含むが、等価ではない。語と語の関係を見たいなら依存構造、句の範囲 (スパン) が欲しいなら句構造が扱いやすい。

## 組合せ範疇文法 (CCG) で文を導出する (NLTK) {ccg_nltk}

単語に「どんな項を取って何になるか」という範疇を割り当て、少数の組合せ規則だけで文を導出する CCG を試したい。

```python
from nltk.ccg import chart, lexicon

lex = lexicon.fromstring(r"""
    :- S, NP, N
    I => NP
    John => NP
    Mary => NP
    books => NP
    saw => (S\NP)/NP
    likes => (S\NP)/NP
    hates => (S\NP)/NP
    the => NP/N
    man => N
    telescope => N
    with => ((S\NP)\(S\NP))/NP
    with => (N\N)/NP
    and => var\.,var/.,var
""")

# 関数適用だけを使う: X/Y Y → X (前方適用 >)、Y X\Y → X (後方適用 <)
parser = chart.CCGChartParser(lex, chart.ApplicationRuleSet)
chart.printCCGDerivation(next(parser.parse("I saw the man".split())))
```

```python
# 構造的曖昧性: with の範疇が 2 つあるので、導出も 2 通りになる
parses = list(parser.parse("I saw the man with the telescope".split()))
print(len(parses), "通り")
for p in parses:
    chart.printCCGDerivation(p)
```

```python
# 関数合成と型繰り上げを加えると、「John likes and Mary hates」のような構成素でないまとまりを等位接続できる
print("関数適用のみ:", len(list(parser.parse("John likes and Mary hates books".split()))), "通り")
full = chart.CCGChartParser(lex, chart.DefaultRuleSet)
parses = list(full.parse("John likes and Mary hates books".split()))
print("合成・型繰り上げあり:", len(parses), "通り")
chart.printCCGDerivation(parses[0])
```

- `(S\NP)/NP` は「右に NP を取り、次に左に NP を取ると S になる」= 他動詞。文法の情報のほとんどが辞書 (語彙項目) に入っていて、規則は言語によらない少数のものだけ、というのが CCG の特徴。
- 範疇と意味表現 (ラムダ式) を対応させると、導出と同時に論理式が組み上がる。意味解析の章の「一階述語論理の式に変換する」レシピと同じ考え方。
- 規則を増やすと、意味が同じ導出が大量に生まれる (擬似的曖昧性)。実用的なパーザは正規形の制約でこれを抑える。
- 実用的な CCG パーザには、英語と日本語に対応した depccg がある。日本語 CCGbank や、CCG 導出から論理式を作って推論する ccg2lambda と組み合わせて使われる。

## spaCy で大量の文書を高速に処理する {spacy_pipe}

数万件のテキストを spaCy で解析したいが、1 件ずつ `nlp(text)` すると遅い。

```python
!python -m spacy download en_core_web_sm -q
```

```python
import time
import spacy

texts = ["Apple is looking at buying a U.K. startup for $1 billion.", "The quick brown fox jumps over the lazy dog."] * 1000

def timeit(label, fn):
    t = time.perf_counter()
    result = fn()
    print(f"{label:38} {time.perf_counter() - t:5.1f} 秒")
    return result

nlp = spacy.load("en_core_web_sm")
_ = timeit("1 件ずつ nlp(text)", lambda: [nlp(t) for t in texts])
_ = timeit("nlp.pipe (バッチ処理)", lambda: list(nlp.pipe(texts, batch_size=256)))

nlp_light = spacy.load("en_core_web_sm", exclude=["parser", "ner", "lemmatizer"])  # 品詞だけ欲しい場合
_ = timeit("不要な要素を除外 + pipe", lambda: list(nlp_light.pipe(texts, batch_size=256)))

nlp_tok = spacy.blank("en")  # トークナイズだけ
_ = timeit("トークナイザのみ", lambda: list(nlp_tok.pipe(texts)))
```

```python
# メタデータを一緒に流す (as_tuples) と、ID との対応を保ったまま処理できる
data = [("Tokyo is the capital of Japan.", {"id": 1}), ("Paris is in France.", {"id": 2})]
for doc, ctx in nlp.pipe(data, as_tuples=True):
    print(ctx["id"], [(e.text, e.label_) for e in doc.ents])
```

- `nlp.pipe(texts, n_process=2)` でマルチプロセス化もできる (Transformer モデル + GPU の場合は使わない)。
- 文分割だけ必要なら `nlp.add_pipe("sentencizer")` を入れた `spacy.blank` が最速。
