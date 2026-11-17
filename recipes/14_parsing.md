# 構文解析

係り受け（依存構造）と句構造の解析。文の骨組みを取り出し、評価し、パーザを学習する。解析のアルゴリズム（遷移型、最大全域木、文脈自由文法、組合せ範疇文法）を自分で動かすレシピは、付録の「仕組みを理解するための実装」にある。

## GiNZA で日本語の依存構造解析（係り受け解析）をする {ginza_dependency}

日本語の文について、どの単語がどの単語に係っているか（主語・目的語などの関係）を解析したい。

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

- Colab 以外（スクリプト）では `displacy.serve(doc)` でブラウザ表示できる。

## 文節に区切って文節単位の係り受けを得る {bunsetsu}

日本語の伝統的な単位である「文節」で区切り、「どの文節がどの文節に係るか」を知りたい（CaboCha 相当）。

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

- 文節 = 自立語（+ 付属語）。UD の単語単位の係り受けより粗いが、日本語の直感に合いやすく、係り受けの交差も少ない。

## CaboCha で文節単位の係り受けを解析する {cabocha}

日本語の係り受け解析器として長く使われてきた CaboCha で、文を文節に区切り、どの文節がどの文節に係るかを解析したい。前のレシピの GiNZA の結果とも比べる。

```python
# Colab で 5 分ほど (CRF++ と CaboCha をソースからビルドする。CaboCha の make の中で、IPA・JUMAN・UniDic 用のモデルファイルも作られる)
!apt-get install -y -q automake mecab libmecab-dev mecab-ipadic-utf8 > /dev/null
!git clone -q --depth 1 https://github.com/taku910/crfpp
!git clone -q --depth 1 https://github.com/taku910/cabocha
# CRF++: git 版には winmain.h がないので、それを読み込む行を消してからビルドする
!cd crfpp && sed -i '/winmain.h/d' crf_learn.cpp crf_test.cpp && ./configure -q > /dev/null 2>&1 && make -j4 > /dev/null 2>&1 && make install > /dev/null && ldconfig
# CaboCha: git 版にない install-sh などを automake から補い、文字コードを UTF-8 にしてビルドする
!cd cabocha && cp /usr/share/automake-*/install-sh /usr/share/automake-*/missing /usr/share/automake-*/depcomp . && ./configure -q --with-charset=UTF8 --enable-utf8-only > /dev/null 2>&1 && make -j4 > /dev/null 2>&1 && make install > /dev/null && ldconfig
# Python バインディング: 付属の CaboCha.py は Python 3.12 で廃止された imp を使うので、そこを通らないようにしてから入れる
!sed -i 's/(2,6,0)/(9,0,0)/' cabocha/python/CaboCha.py && pip install -q ./cabocha/python
!pip install -q ginza ja_ginza
!cabocha --version
```

```python
# 木の表示 (-f0、既定)。各文節の右の D から線をたどった先が係り先
!echo "太郎は図書館で花子に借りた本を昨日返した。" | cabocha -f0
# 格子形式 (-f1)。「* 文節番号 係り先D 主辞/機能語 スコア」の行の下に、MeCab (IPA 辞書) の解析結果が続く
!echo "太郎は図書館で花子に借りた本を昨日返した。" | cabocha -f1
```

```python
import CaboCha

parser = CaboCha.Parser()  # コマンドと同じオプションを文字列で渡せる (例: CaboCha.Parser("-n 1"))

def cabocha_chunks(text):
    """文節ごとに (文節の文字列, 係り先の文節番号, 主辞の形態素, 機能語の形態素) を返す。係り先 -1 は文末"""
    tree = parser.parse(text)
    out = []
    for i in range(tree.chunk_size()):
        c = tree.chunk(i)
        toks = [tree.token(c.token_pos + j) for j in range(c.token_size)]
        out.append(("".join(t.surface for t in toks), c.link,
                    toks[c.head_pos].surface, toks[c.func_pos].surface))
    return out

text = "太郎は図書館で花子に借りた本を昨日返した。"
chunks = cabocha_chunks(text)
for i, (surface, link, head, func) in enumerate(chunks):
    target = chunks[link][0] if link >= 0 else "(文末)"
    # 「昨日」のように付属語がない文節では、機能語の位置 (func_pos) が主辞と同じになる
    print(f"{i} {surface:　<6} → {target:　<6} 主辞: {head}  機能語: {func}")
```

```python
import ginza
import spacy

nlp = spacy.load("ja_ginza")

def ginza_pairs(text):
    """前のレシピと同じ方法で、GiNZA の文節と係り先の組を返す"""
    pairs = []
    for span in ginza.bunsetu_spans(nlp(text)):
        head = ginza.bunsetu_span(span.root.head)
        pairs.append((span.text, "(文末)" if head.start == span.start else head.text))
    return pairs

def cabocha_pairs(text):
    chunks = cabocha_chunks(text)
    return [(s, chunks[link][0] if link >= 0 else "(文末)") for s, link, _, _ in chunks]

for t in [text, "彼は赤い帽子をかぶった女の子に道を教えてあげた。", "吾輩は猫である。名前はまだ無い。"]:
    print(t)
    for name, pairs in [("CaboCha", cabocha_pairs(t)), ("GiNZA", ginza_pairs(t))]:
        print(f"  {name:8}", "  ".join(f"{s}→{h}" for s, h in pairs))
```

```python
# 固有表現 (-n 1)。IREX の定義の固有表現が、木の表示では <PERSON> などのタグで、形態素ごとには IOB2 形式のタグで得られる
!echo "昨日、山田太郎さんは東京大学でトヨタ自動車の社長と会った。" | cabocha -n 1
ne_parser = CaboCha.Parser("-n 1")  # parse の結果はパーザの中にあるので、パーザを変数に入れておく
ne_tree = ne_parser.parse("昨日、山田太郎さんは東京大学でトヨタ自動車の社長と会った。")
print([(ne_tree.token(i).surface, ne_tree.token(i).ne) for i in range(ne_tree.token_size())
       if ne_tree.token(i).ne != "O"])
```

- CaboCha は工藤拓（奈良先端大・松本研究室）による、Support Vector Machines (SVM) に基づく日本語係り受け解析器。方式は論文「チャンキングの段階適用による日本語係り受け解析」（Cascaded Chunking、工藤・松本 2002）。付属のモデルでは、係り受けは 2 次の多項式カーネルの SVM、文節の区切りと固有表現は CRF++ の CRF で解析する（CRF++ が必要なのはこのため）。最後のリリースは 0.69（2015 年 1 月）。GitHub (taku910/cabocha) では、2015 年 1 月より後にソースは変わっておらず、説明のページ（index.html など）が追加・修正されただけ。
- 形態素解析には MeCab を使い、品詞体系と辞書（IPA・JUMAN・UniDic）に合ったモデルが必要。ここでは既定の IPA 辞書 (mecab-ipadic-utf8) と IPA モデルを使う。ドキュメントによると、IPA モデルは京都大学テキストコーパス 4.0 の形態素の部分を MeCab で自動解析し直したものから学習していて、コーパスから直接学習した JUMAN モデルより精度が落ちる。
- 付属のモデルは毎日新聞の記事から学習されていて、そのまま使えるのは研究目的・個人利用に限られる（プログラム自体は LGPL か BSD ライセンス）。自分のコーパスで `cabocha-learn` を使って学習し直したモデルにはこの制限はない。
- configure の文字コードの既定は EUC-JP で、モデル (`/usr/local/lib/cabocha/model/*.model`) は make の中でテキスト形式からその文字コードで作られる。ここではドキュメントに従い、UTF-8 の IPA 辞書に合わせて `--with-charset=UTF8 --enable-utf8-only` を指定した。使うモデルや既定の設定は `/usr/local/etc/cabocharc` に書かれる。
- 最初の文では「太郎は」の係り先だけが違い、CaboCha は文末の「返した。」、GiNZA は「借りた」になった。主題の「は」は連体修飾節（ここでは「借りた本」の「借りた」）の中には入りにくいので、文末の述語に係るとする CaboCha の解析が普通の基準に合う。「借りた」の動作主も太郎であることは、係り受けではなく格解析や省略解析で扱う問題になる（この章の「Juman++ と KNP で係り受けと格関係を解析する」を参照）。
- 文節の単位も少し違う。「教えてあげた。」は CaboCha では 1 文節だが、GiNZA では「教えて」と「あげた。」の 2 文節になり、ほかの文節は「あげた。」や「教えて」に係る。GiNZA は単語単位の係り受け (UD) を解析してから文節をまとめていて、補助動詞の「あげる」が述語の中心 (ROOT) になるため。
- CaboCha は渡された文字列（1 行）を 1 文として扱う。「吾輩は猫である。名前はまだ無い。」をそのまま渡すと、「猫である。」が次の文の「無い。」に係ってしまう。文が複数あるときは、先に文に分割してから 1 文ずつ渡す。
- Python バインディングがビルドできない環境では、`subprocess.run(["cabocha", "-f1"], input=text, ...)` の出力を読めばよい。`*` で始まる行が文節の区切りで、2 列目が文節番号、3 列目が係り先（末尾の D を除く）になる。
- Python バインディングの `parse` が返す木はパーザの中のデータを指す。パーザを変数に入れずに `CaboCha.Parser("-n 1").parse(...)` と書くと、パーザが消えた後に木を読むことになり、カーネルが落ちた。
- 格子形式の最後の数値は係りやすさのスコアで、ドキュメントでは大きいほど係りやすいとされるが、実際の精度との関係は調べられていないとも書かれている。最初の文では、文末の「返した。」に係る「太郎は」「本を」「昨日」のスコアがどれも -1.474801 で、正しい係り先なのに負になった。文末の文節への係りは同じ値になることがあるので、スコアを確信度として使うのは避ける。

## KWJA で文節・基本句の係り受けと並列構造を解析する {kwja_dependency}

京大コーパスの基準（KNP 形式）で、文節単位・基本句単位の係り受けを得たい。並列や同格の関係も区別したい。

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

## Juman++ と KNP で係り受けと格関係を解析する {knp}

京大の構文・格解析システム KNP で、文節・基本句の係り受けに加えて、述語ごとの格関係（誰が・何を・誰に）を得たい。KNP は形態素解析器 Juman++ の出力を入力にとるので、両方をソースからビルドする。

```python
# Colab で 10〜20 分ほどかかる (KNP 辞書 2.8GB のダウンロードと展開、Juman++ と KNP のビルド。ダウンロード速度で変わる)
!apt-get install -y -q libtool > /dev/null
!wget -q https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc4/jumanpp-2.0.0-rc4.tar.xz
!tar xf jumanpp-2.0.0-rc4.tar.xz && mkdir -p jumanpp-2.0.0-rc4/bld
# テストなどを省き、jumanpp 本体だけをビルドしてインストールする
!cd jumanpp-2.0.0-rc4/bld && cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1 && make -j2 jumanpp_v2 > /dev/null 2>&1 && cmake --install . > /dev/null
!git clone -q --depth 1 https://github.com/ku-nlp/knp
!wget -q http://lotus.kuee.kyoto-u.ac.jp/nl-resource/knp/dict/latest/knp-dict-latest-bin.zip
!unzip -q knp-dict-latest-bin.zip && cp -ars `pwd`/dict-bin/* knp/dict
!cd knp && ./autogen.sh > /dev/null 2>&1 && ./configure > /dev/null && make -j2 > /dev/null 2>&1 && make install > /dev/null 2>&1
# 辞書はインストール先にコピーされるので、ダウンロードと展開したもの (約 10GB) は消してよい
!rm -rf knp-dict-latest-bin.zip dict-bin jumanpp-2.0.0-rc4.tar.xz
!pip install -q rhoknp kwja
!jumanpp -v && du -sh /usr/local/share/knp/dict
!echo "太郎は花子に本を渡した。" | jumanpp | knp  # 既定は木構造の表示
```

```python
from rhoknp import KNP, Jumanpp

# executable="/path/to/jumanpp" のようにバイナリの場所も指定できる。
# rhoknp は起動確認で空行を送るが、Juman++ rc4 は先頭が空行だと異常終了することがあり、そのとき警告が出るので、確認は省く
jumanpp = Jumanpp(skip_sanity_check=True)
knp = KNP(jumanpp=jumanpp)      # KNP(executable=...) も同様。既定のオプションは ["-tab"] (KNP 形式で出力)
sent = knp.apply_to_sentence("太郎と花子は、昨日京都で買った抹茶とお菓子を先生に渡した。")

DEP = {"D": "係り受け", "P": "並列", "A": "同格", "I": "部分並列"}
print("文節:")
for p in sent.phrases:
    print(f"  {p.text:　<8} → {p.parent.text if p.parent else '(文末)':　<8} {DEP[p.dep_type.value]}")
print([(m.text, m.reading, m.lemma, m.pos, m.subpos) for m in sent.morphemes][:6])
```

```python
# 格解析の結果: 述語の基本句ごとに、格フレームの ID と、各格に割り当てられた項
ARG = {"C": "表層の形から格が決まる (格助詞・時間表現など)", "N": "格が表層に現れない (提題・連体修飾など)",
       "O": "省略", "E": "外界"}

def arg_text(a):
    """項の表記 (文中の基本句なら その表記、「不特定:人」のような外界の項なら その名前)"""
    return a.base_phrase.text if hasattr(a, "base_phrase") else str(a)

for pas in sent.pas_list:
    print(f"{pas.predicate.text}  (格フレーム {pas.predicate.cfid})")
    print("  素性:", pas.predicate.base_phrase.features["格解析結果"])
    for case, args in pas.get_all_arguments(relax=False).items():
        for a in args:
            print(f"  {case}格: {arg_text(a):　<6} {a.type.value}: {ARG.get(a.type.value, '')}")

# 基本句の素性 (feature)。KNP が付けた情報は辞書として取り出せる
for bp in sent.base_phrases:
    keys = [k for k in ("用言", "体言", "係", "解析格", "時制", "SM-人") if k in bp.features]
    print(f"  {bp.text:　<6}", {k: bp.features[k] for k in keys})

import time
t = time.time()
for _ in range(20):
    knp.apply_to_sentence(sent.text)
print(f"1 文あたり {(time.time() - t) / 20 * 1000:.0f} ミリ秒 (Juman++ と KNP の合計)")
```

```python
# 同じ文を KWJA (前のレシピ) で解析して比べる。出力はどちらも KNP 形式なので rhoknp で同じように扱える
import subprocess
from rhoknp import Document

out = subprocess.run(["kwja", "--text", sent.text, "--tasks", "char,word", "--model-size", "base"],
                     capture_output=True, text=True, check=True).stdout
kw = Document.from_knp(out).sentences[0]

# 文節の区切りが両者で同じであることを前提に、文節を順に並べて比べる
assert [p.text for p in sent.phrases] == [q.text for q in kw.phrases]
print(f"{'文節':　<8} {'KNP':　<10} {'KWJA':　<10}")
for p, q in zip(sent.phrases, kw.phrases):
    a = (p.parent.text if p.parent else "(文末)") + p.dep_type.value
    b = (q.parent.text if q.parent else "(文末)") + q.dep_type.value
    print(f"{p.text:　<8} {a:　<10} {b:　<10}", "" if a == b else "← 違う")

def args_of(pas):
    return {case: [arg_text(a) for a in v] for case, v in pas.get_all_arguments(relax=False).items()}

# 述語は基本句の番号で対応づける (基本句の区切りも同じであることを確かめておく)
assert [bp.text for bp in sent.base_phrases] == [bp.text for bp in kw.base_phrases]
knp_pas = {pas.predicate.base_phrase.index: pas for pas in sent.pas_list}
kw_pas = {pas.predicate.base_phrase.index: pas for pas in kw.pas_list}
for i in sorted(knp_pas.keys() | kw_pas.keys()):
    print(sent.base_phrases[i].text)
    print("  KNP :", args_of(knp_pas[i]) if i in knp_pas else "-")
    print("  KWJA:", args_of(kw_pas[i]) if i in kw_pas else "-")
```

- KNP は、Juman++（または JUMAN）の形態素解析結果を入力にとり、文節・基本句の係り受け、格関係、照応関係を出力する。人手で書いたルール（`rule/` の下）で形態素・文節に素性を付けて係り受けの候補を作り、並列構造は前後の文節列の類似度で検出する。候補は、ウェブテキストから自動構築した大規模な格フレームに基づく確率で評価する（README とマニュアル `doc/manual.pdf` による）。マニュアルによると、出力するのは最もよい 1 つの解だけである。
- この文では、文節の係り先と種類 (D/P) は KNP と KWJA で全く同じになった。「太郎と → 花子は」「抹茶と → お菓子を」が並列 (P)、「京都で」「昨日」は「買った」に係る。
- 格解析の違い: KNP は並列の両方（「渡した」のガ格に「太郎と」「花子は、」、ヲ格に「抹茶と」「お菓子を」）を項に入れるが、KWJA は並列の一方だけを項にした（「渡した」のガ格は「太郎と」、ヲ格は「抹茶と」、「買った」のヲ格は「お菓子を」）。また「買った」のガ格は、ガ格になる文節が「買った」に係っていないため KNP では空き（`格解析結果` の `ガ/U/-`）だが、KWJA は係り受けのない「花子は、」をガ格として出した。KNP の照応解析 (`-anaphora`) は試していない。
- `格解析結果` の各項は `格/種類/表記/基本句番号/...` の形。種類は、C が「述語に係る格要素で、表層の形から格が決まるもの」（格助詞の付いた「先生に」のほか、時間格の「昨日」も C）、N が「格が表層に現れないもの」（提題の「花子は、」や、連体修飾で係る「お菓子を」（「買った」のヲ格）など）、U が「割り当てなし」。並列の前の要素（「太郎と」「抹茶と」）は後ろの要素と同じ種類になっている。rhoknp は U 以外を `BasePhrase.pas` (`Sentence.pas_list`) に読み込む。格フレームの ID (`渡す/わたす:動1`) は、1 つの述語に用法ごとに複数ある格フレームのどれを選んだかを表す。
- 基本句の素性には、係りの形（`係:未格` など）、格解析で決まった格 (`解析格:ガ`)、意味素性 (`SM-人`)、時制などが入る。素性の一覧は KNP の `doc/knp_feature.pdf` にある。
- 出力は KWJA と同じ KNP 形式で、rhoknp で同じように読める。形態素は JUMAN の品詞体系（「名詞/人名」「助詞/副助詞」「特殊/読点」など）。コマンドラインの `knp` は既定で木構造を表示し、rhoknp は `-tab`（表形式）で呼ぶ。
- 速さ: rhoknp は Juman++ と KNP のプロセスを起動したまま使い回すので、この文では 1 文あたり 0.15〜0.2 秒ほどだった（上の計測。CPU 2 コアで 2 回計って 152 ミリ秒と 184 ミリ秒）。KWJA はコマンドを呼ぶたびにモデルを読み込み、GPU なし（CPU 2 コア）では初回（モデルのダウンロードを含む）に 50 秒ほどかかった。
- `pip install kwja` で、ほかのパッケージとの依存関係の衝突の警告 (ERROR: pip's dependency resolver ...) が出るが、ここでは無視してよい。KWJA はコマンドとして別プロセスで動かしているので、ノートブックの再起動も要らない（KWJA を Python から import して使う場合は、前のレシピの注意のとおり再起動が必要になることがある）。
- ビルドの手間: KNP は GitHub にリリースがなく、git から autogen.sh（libtool が必要）でビルドする。辞書（格フレームなど）は別配布の 2.8GB の zip で、インストール先に 7GB ほどを使う。Juman++ rc4 は、先頭の入力が空行だと異常終了することがある（毎回ではない）。rhoknp の起動確認は空行を送るので、そのとき「failed to start Juman++」の警告が出る。これを避けるため `skip_sanity_check=True` を付ける。

## Stanza で多言語の形態素・依存構造解析をする {stanza}

日本語・英語以外も含めて、70 以上の言語を同じ API で解析したい（Universal Dependencies 準拠）。

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

- `REL_OP` には `>`（直接の子）、`<`（直接の親）、`>>`（子孫）、`<<`（祖先）、`.`（直後）、`;`（直前）などがある。
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
- 受身・使役では格と意味役割がずれる（「ネズミが猫に追いかけられた」）。

## 名詞句を抽出する {noun_chunks}

文から名詞句（修飾語を含む名詞のまとまり）を取り出して、キーフレーズ候補や検索語にしたい。

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

文の構造的な複雑さを数値化したい（可読性の評価、学習者作文の分析、言語間の比較）。

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

- `nlp.pipe(texts, n_process=2)` でマルチプロセス化もできる（Transformer モデル + GPU の場合は使わない）。
- 文分割だけ必要なら `nlp.add_pipe("sentencizer")` を入れた `spacy.blank` が最速。

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
- 日本語では `case`（助詞）や `aux`（助動詞）のように、ほぼ機械的に決まる関係が全体の精度を押し上げる。内容語の係り受け（`nsubj`、`obl`、`acl` など）を個別に見るのが大事。

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

- 1500 文・600 ステップ（CPU で 1〜2 分）でも UAS は 80% 台後半になる。全データで十分に学習すれば 90% を超える。
- GiNZA のほうが低く出るのは、GiNZA が別のツリーバンク (UD Japanese-BCCWJ) の基準で学習されていて、単語分割やラベルの付け方が GSD と少し違うため。パーザの優劣ではなく、「評価データと同じ基準で学習したか」の差が出ている。
- `--optimize accuracy` と事前学習済みの Transformer を使う設定（`--gpu-id 0`、`--pipeline transformer,parser`）にすると精度が大きく上がる。
- 単語分割は spaCy の日本語トークナイザ (SudachiPy) が行う。ツリーバンクの分割と合わない語は学習・評価で対応が取れないので、分割基準の一致が重要。
- 自分の分野のデータを数百文アノテーションして追加学習すると、分野適応の効果が大きい。その場合は既存モデルの重みから始める（`[initialize]` や `source` の設定）。

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

- 括弧表記は `(ラベル 子1 子2 ...)` の入れ子。句構造のツリーバンク（Penn Treebank、京都大学の NPCMJ、Stanza の構文解析結果など）はほぼこの形式で配布されている。
- CKY アルゴリズムなど多くの構文解析アルゴリズムは二分木を前提にするので、学習前に二分化し、評価前に元に戻す。
- 句構造解析の評価は、ラベル付き括弧の適合率・再現率（PARSEVAL、`evalb`）で行う。

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

- 英語のモデルは Penn Treebank 形式のラベル（NP、VP、PP、SBAR …）を出力する。日本語のモデルは ALT (Asian Language Treebank) で学習されており、ラベル体系が違う。
- 「with the telescope」が動詞句と名詞句のどちらに付くかは、パーザが 1 つの解釈を選ぶ。手書き文法のレシピのように、すべての解釈は列挙されない。
- 依存構造と句構造は相互に変換できる情報を多く含むが、等価ではない。語と語の関係を見たいなら依存構造、句の範囲（スパン）が欲しいなら句構造が扱いやすい。

## Ckylark で日本語の句構造解析をする {ckylark}

日本語の文を「名詞句」「動詞句」のような句の入れ子構造（句構造木）に解析したい。日本語と英語の学習済みモデルが付属する PCFG-LA パーザ Ckylark を使う。日本語モデルは KyTea 方式の単語分割を前提にしているので、KyTea も一緒にビルドする。

```python
!apt-get install -y -q autoconf automake libtool libboost-iostreams-dev libboost-program-options-dev > /dev/null
!git clone -q --depth 1 https://github.com/odashi/ckylark
!git clone -q --depth 1 https://github.com/neubig/kytea
!cd ckylark && autoreconf -i > /dev/null 2>&1 && ./configure -q > /dev/null && make -j4 > /dev/null 2>&1
!cd kytea && autoreconf -i > /dev/null 2>&1 && ./configure -q > /dev/null && make -j4 > /dev/null 2>&1 && make install > /dev/null 2>&1 && ldconfig
!pip install -q fugashi unidic-lite
!ls ckylark/data/*.grammar.gz && echo 'This is a pen .' | ckylark/src/bin/ckylark --model ckylark/data/wsj
```

```python
import subprocess
from nltk import Tree

CKYLARK = "ckylark/src/bin/ckylark"
# 半角英数字・記号を全角にする(日本語モデルの語彙は全角表記)
ZEN = str.maketrans({chr(c): chr(c + 0xFEE0) for c in range(0x21, 0x7F)})

def kytea_segment(sentences):
    text = "\n".join(s.translate(ZEN) for s in sentences) + "\n"
    r = subprocess.run(["kytea", "-notags"], input=text, capture_output=True, text=True)
    return r.stdout.strip().split("\n")

def ckylark_parse(segmented, model="ckylark/data/jdc"):
    # モデル読み込みに数秒かかるので、文はまとめて 1 回で渡す(1 行 1 文、空白区切り)
    r = subprocess.run([CKYLARK, "--model", model], input="\n".join(segmented) + "\n",
                       capture_output=True, text=True)
    return [Tree.fromstring(line)[0] for line in r.stdout.strip().split("\n")]  # 外側の空ラベルを外す

sents = ["太郎は図書館で借りた本を花子に渡した。", "私は昨日友達と映画を見に行きました。"]
seg = kytea_segment(sents)
trees = ckylark_parse(seg)
for s, t in zip(seg, trees):
    print(s)
    print(t)
trees[0].pretty_print()
```

```python
# 「名詞句 + 助詞」の句(助詞をP, 助詞にP など)から、助詞の前の部分を項として取り出す
def arguments(tree):
    out = []
    for st in tree.subtrees(lambda t: t.label().startswith("助詞") and t.label().endswith("P")):
        head, particle = st[0], st[-1]
        if isinstance(head, Tree) and head.label() in ("名詞", "名詞P", "代名詞", "接尾辞P"):
            out.append(("".join(head.leaves()), "".join(particle.leaves())))
    return out

for t in trees:
    print(arguments(t))

# 比較: MeCab + UniDic(fugashi)で分割すると、活用語尾の切り方が違うため品詞の推定が崩れる
import fugashi
tagger = fugashi.Tagger()
uni = " ".join(w.surface for w in tagger(sents[1]))
r = subprocess.run([CKYLARK, "--model", "ckylark/data/jdc", "--output-format", "postag"],
                   input=uni + "\n", capture_output=True, text=True)
print(r.stdout.strip())
```

```python
# 英語は WSJ(Penn Treebank)モデル。入力はトークン化済み(句読点も空白で区切る)
en = ckylark_parse(["The quick brown fox jumped over the lazy dog near the river ."],
                   model="ckylark/data/wsj")[0]
en.pretty_print()
print([" ".join(np.leaves()) for np in en.subtrees(lambda t: t.label() == "NP")])
```

- PCFG-LA（潜在注釈付き確率文脈自由文法）は、「NP」などの記号を自動的に細分化した文法で CKY 解析を行う手法で、Berkeley Parser と同じ方式。Ckylark は、未知語があっても解析が失敗しにくいよう工夫されている。
- 日本語モデル (`--model ckylark/data/jdc`) は、KyTea で分割した「日本語単語依存構造コーパス」で学習されており、「渡 し た」「行 き ま し た」のように動詞の語幹と活用語尾を分ける分割を前提にしている。MeCab + UniDic の分割（「渡し た」）を渡しても解析はできるが、品詞と構造が崩れやすいので、KyTea で分割する。語彙は全角英数字なので、半角は全角に直してから渡す。
- 日本語モデルのラベルは「名詞」「動詞」「語尾」「助動詞」「補助記号」などの品詞と、その句である「名詞P」「動詞P」、さらに助詞ごとの「助詞はP」「助詞をP」「助詞にP」など。英語モデルは Penn Treebank の NP、VP、PP など。出力は 1 文 1 行の S 式で、一番外側にラベルのない括弧が付く（`--add-root-tag` で ROOT にできる）。
- 係り先の誤りは普通に起きる（「太郎は図書館で借りた本を花子に渡した。」で、「図書館で」が「借りた」ではなく「渡した」に係る、など）。N-best 出力はなく、得られるのは最良の 1 つの木だけ。
- モデルの読み込みに日本語で約 2 秒、英語で約 5 秒かかるので、文は 1 回の呼び出しにまとめて渡す。入力中の半角括弧は -LRB- / -RRB- に置き換わる。
- 主なオプション: `--output-format postag`（品詞列だけを出す）、`--add-root-tag`、`--binarize`（二分木で出力）、`--force-generate`（解析に失敗しても平らな木を出す）、`--prune-threshold`（枝刈りの強さ。速度と精度の調整）。
- 2015 年ごろの研究用ツールで、ビルド（Ckylark と KyTea で 2 分ほど）が必要。インストールだけで済ませたい場合や、係り受けも欲しい場合は、前のレシピの Stanza も検討する。
