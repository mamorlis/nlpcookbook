# 画像と言語

画像とテキストを結びつける。CLIP による画像とテキストの埋め込み、視覚言語モデル (VLM) による説明文の生成と質問応答、キャプションの評価、OCR、PDF の読解、請求書やレシートからの項目の抽出、ページ画像を検索するマルチモーダル RAG。VLM を使うレシピは GPU ランタイム (T4) を想定する。

## CLIP で画像と日本語の文を同じ空間に埋め込み、検索・分類する {clip_ja}

画像とテキストを同じベクトル空間に置いて、学習データなしに日本語のラベルで画像を分類したり、日本語の文で画像を検索したりしたい。

```python
!pip install -q matplotlib-fontja
```

```python
import io, re, time
import requests
from PIL import Image

# Wikimedia Commons の画像 (CC BY / CC BY-SA / FAL / パブリックドメイン。ファイルごとの作者・出典・ライセンスは下で表示する)
FILES = {
    "犬": ["Alaskan Malamute R Bartz.jpg", "Beagle sleeping in Hautmont (DSC07212).jpg",
           "Dalmatian fetching a stick.jpg", "2011-04-02 a Shiba Inu in Japan.jpg"],
    "猫": ["Cat sleeping in the Ngọc Sơn Temple in Hanoi, 20240204 1309 5747.jpg", "Golden tabby and white kitten n01.jpg",
           "Colossus the Cat 2.JPG", "Бишкек, кот в траве на Элебесова.jpg"],
    "鳥": ["Black-naped Monarch 0A2A8267.jpg", "Cedar waxwing in Green-Wood Cemetery (55650).jpg",
           "Grey-crowned Babblers 1605.jpg", "Sarcelle élégante DSCF2036.jpg"],
    "車": ["00 car, 40. ADAC Stormarn Rallye, Luetjensee (TR256763).jpg", "1939 Delage D6 Grand Prix, front right (Brooklyn).jpg",
           "18. ADAC Stormarn Classic, Zarpen (TR257255).jpg", "18. ADAC Stormarn Classic, Luetjensee (P1045457).jpg"],
    "電車": ["20160814 fomer JR train Yangon Circular Railway 0228.jpg", "Ankara asv2021-10 img75 Yenimahalle metro.jpg",
             "Coradia ZZe-03, Gare de l'Agha 2023.jpg", "Flirt, Susteren (LRM 20220311 101307).jpg"],
    "花": ["Abelmoschus sagittifolius 01.JPG", "Adenium obesum 02.JPG", "Alppiasteri 1.jpg", "Anthemis April 2010-1.jpg"],
    "ラーメン": ["2013.02.27.ramen.jpg", "Bikkuri-rahmen.jpg", "Butter Corn Ramen.JPG",
                 "Free-photo-of-a-bowl-of-ramen-with-eggs-and-meat.jpg"],
    "寿司": ["HSY- Sushi, Imitation Crab.jpg", "HSY- Sushi, Tai.jpg", "Nigiri Sushi (26505096851).jpg",
             "Nigirizushi at Luckiefun's Redi close up in April 2025.jpg"],
}
COUNT = ["2020-09-22 Shiba Inu running.jpg", "2020-09-22 two Shiba Inus 01.jpg", "2020-09-22 three Shiba Inus.jpg"]  # 柴犬が 1・2・3 匹

UA = {"User-Agent": "nlpcookbook-clip-recipe/0.1"}   # Wikimedia は User-Agent のないアクセスを拒否する

def commons_images(names, width=330):
    """Commons の API で縮小画像の URL・ファイルページの URL・ライセンス・作者を 1 回で取得し、画像を順にダウンロードする"""
    titles = ["File:" + n for n in names]
    r = requests.get("https://commons.wikimedia.org/w/api.php", headers=UA, timeout=30, params={
        "action": "query", "titles": "|".join(titles), "prop": "imageinfo", "iiprop": "url|extmetadata",
        "iiurlwidth": width, "format": "json", "formatversion": 2}).json()
    norm = {n["from"]: n["to"] for n in r["query"].get("normalized", [])}
    info = {p["title"]: p["imageinfo"][0] for p in r["query"]["pages"]}
    items = []
    for t in titles:
        ii = info[norm.get(t, t)]
        meta = ii["extmetadata"]
        img = Image.open(io.BytesIO(requests.get(ii["thumburl"], headers=UA, timeout=30).content)).convert("RGB")
        artist = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip().split("\n")[0]
        items.append({"title": t[5:], "image": img, "license": meta["LicenseShortName"]["value"], "artist": artist,
                      "url": ii["descriptionurl"]})
        time.sleep(0.2)
    return items

labels = list(FILES)
items = commons_images([n for ns in FILES.values() for n in ns] + COUNT)
data, count_items = items[:-3], items[-3:]
for it, y in zip(data, [y for y in labels for _ in FILES[y]]):
    it["label"] = y
for it in items:
    print(f"{it.get('label', '数')} | {it['artist']} | {it['license']} | {it['url']}")   # 作者・出典・ライセンス
```

```python
import matplotlib.pyplot as plt
import matplotlib_fontja

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for ax, it in zip(axes.flat, data):
    ax.imshow(it["image"]); ax.set_title(it["label"], fontsize=10); ax.axis("off")
plt.tight_layout(); plt.show()
```

```python
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

device = "cuda" if torch.cuda.is_available() else "cpu"
name = "line-corporation/clip-japanese-base"   # LY Corporation の日本語 CLIP (Apache-2.0)。画像側 Eva02-B、テキスト側 BERT
rev = "77a62f8af977acd73ec2f927fb73fdaeb13af7d1"      # リポジトリのコミットを固定する (中のコードを実行するので、すべての読み込みに渡す)
tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, revision=rev)
processor = AutoImageProcessor.from_pretrained(name, trust_remote_code=True, revision=rev)
# モデルカードの AutoModel.from_pretrained(name, trust_remote_code=True) は transformers 5 系ではエラーになるので、
# リポジトリ内のクラスを取り出して重みを直接読み込む
CLYPModel = get_class_from_dynamic_module("modeling_clyp.CLYPModel", name, revision=rev)  # modeling_clyp.py を取得してクラスを返す
model = CLYPModel(AutoConfig.from_pretrained(name, trust_remote_code=True, revision=rev))  # config から重みが乱数のモデルを作る
print(model.load_state_dict(load_file(hf_hub_download(name, "model.safetensors", revision=rev))))  # All keys matched と出れば全部の重みが入った
model = model.to(device).eval()                      # 重みは float32 のまま使う (約 0.8GB で、T4 にも CPU にも載る)

@torch.no_grad()
def embed_images(images, batch_size=16):
    out = [model.get_image_features(**processor(images[i:i + batch_size], return_tensors="pt").to(device))
           for i in range(0, len(images), batch_size)]
    return F.normalize(torch.cat(out), dim=-1)     # 長さ 1 に正規化 → 内積 = コサイン類似度

@torch.no_grad()
def embed_texts(texts):
    return F.normalize(model.get_text_features(**tokenizer(texts).to(device)), dim=-1)

img_emb = embed_images([it["image"] for it in data])
temperature = model.itc_loss_fn.temperature.item()
print(img_emb.shape, "温度:", temperature)
```

```python
# (1) ゼロショット分類: ラベルを文にして埋め込み、画像に最も近い文のラベルを答えにする
gold = torch.tensor([labels.index(it["label"]) for it in data], device=device)
# 正解にならない紛らわしい候補も混ぜて、難しくした版も試す
distractors = ["オオカミ", "キツネ", "ウサギ", "バス", "トラック", "飛行機", "木", "うどん", "そば", "パスタ", "おにぎり", "刺身", "ケーキ"]
templates = ["{}", "{}の写真", "これは{}の写真です。", "{}が写っている画像", "{}のイラスト"]

def classify(templates, cands):
    # テンプレートが複数なら、ラベルごとに埋め込みを平均する (プロンプトのアンサンブル)
    t = torch.stack([embed_texts([tpl.format(y) for y in cands]) for tpl in templates]).mean(0)
    return img_emb @ F.normalize(t, dim=-1).T

for tpl in templates + [templates[:4]]:
    tpls = tpl if isinstance(tpl, list) else [tpl]
    sims = classify(tpls, labels)
    others = sims.masked_fill(F.one_hot(gold, len(labels)).bool(), -1).max(1).values
    margin = (sims[range(len(data)), gold] - others).mean()         # 正解ラベルと、次に近いラベルとの類似度の差
    sims2 = classify(tpls, labels + distractors)
    pred2 = sims2.argmax(1)
    wrong = [f"{it['title'][:20]}: {it['label']}→{(labels + distractors)[p]}" for it, p, g in zip(data, pred2.tolist(), gold.tolist()) if p != g]
    print(f"{' / '.join(tpls)[:42]:42} 8 択: 正解率 {(sims.argmax(1) == gold).float().mean():.3f} 差 {margin:.3f} | "
          f"21 択: 正解率 {(pred2 == gold).float().mean():.3f} 誤り {wrong}")

probs = (classify(["{}の写真"], labels) / temperature).softmax(1)   # 温度で割って softmax すると「確率」になる
print("1 枚目の画像:", {y: round(p, 3) for y, p in zip(labels, probs[0].tolist())})
```

```python
# 細かい区別: 犬種 (柴犬・ビーグル・ダルメシアン・アラスカン・マラミュート) を、似た犬種も混ぜた候補から選ばせる
breeds = ["柴犬", "秋田犬", "ビーグル", "バセット・ハウンド", "ダルメシアン", "アラスカン・マラミュート", "シベリアン・ハスキー"]
dogs = [it for it in data if it["label"] == "犬"]
sims = img_emb[[data.index(d) for d in dogs]] @ embed_texts([f"{b}の写真" for b in breeds]).T
for d, row in zip(dogs, (sims / temperature).softmax(1)):
    top = row.topk(3)
    print(f"{d['title'][:35]:35}", "  ".join(f"{breeds[i]} {p:.2f}" for p, i in zip(top.values.tolist(), top.indices.tolist())))
```

```python
# (2) 日本語の文で画像を検索する: 文の埋め込みと全画像の埋め込みのコサイン類似度で並べる
queries = ["雪の積もった山で犬が立っている", "眠っている動物", "赤いスポーツカー", "地下鉄の駅のホーム", "黄色い花", "ゆで卵がのった麺料理"]
q_emb = embed_texts(queries)
sims = q_emb @ img_emb.T
fig, axes = plt.subplots(len(queries), 4, figsize=(12, 2.6 * len(queries)))
for row, q, s in zip(axes, queries, sims):
    top = s.topk(3)
    row[0].text(0, 0.5, q, fontsize=12, va="center"); row[0].axis("off")
    for ax, v, i in zip(row[1:], top.values.tolist(), top.indices.tolist()):
        ax.imshow(data[i]["image"]); ax.set_title(f"{v:.3f} ({data[i]['label']})", fontsize=10); ax.axis("off")
plt.tight_layout(); plt.show()
```

```python
# (3) 類似度行列: 8 枚の画像と、それぞれを説明する 8 つの文。対角線 (正しい組) が最も大きくなるように学習されている
idx = [labels.index(y) * 4 for y in labels]          # 各クラスの 1 枚目
captions = ["雪山に立つ犬", "赤い敷物の上で寝ている猫", "枝にとまった青い鳥", "郊外の道路を走る銀色の車",
            "駅に停まっている列車", "赤い花と緑の葉", "チャーシューとメンマがのったラーメン", "カニカマのにぎり寿司"]
S = img_emb[idx] @ embed_texts(captions).T          # 行: 画像、列: 文
fig, axes = plt.subplots(1, 9, figsize=(16, 2.2))
for ax, i in zip(axes, idx):
    ax.imshow(data[i]["image"]); ax.axis("off")
axes[-1].axis("off")
plt.show()
fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(S.cpu(), cmap="viridis")
ax.set_xticks(range(8), captions, rotation=60, ha="right"); ax.set_yticks(range(8), [data[i]["label"] for i in idx])
for r in range(8):
    for c in range(8):
        ax.text(c, r, f"{S[r, c]:.2f}", ha="center", va="center", color="white" if S[r, c] < S.max() * 0.8 else "black", fontsize=8)
plt.tight_layout(); plt.show()

# CLIP の学習の損失 (InfoNCE): 類似度を温度で割り、行 (画像→文) と列 (文→画像) の両方向でクロスエントロピーを取る。
# バッチ内の正しい組が正例、それ以外の組がすべて負例になる
target = torch.arange(8, device=device)
for t in [1.0, 0.1, temperature]:
    logits = S / t
    loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
    print(f"温度 {t:<5} 損失 {loss:.3f}  正しい組の確率 (画像→文) の平均 {logits.softmax(1).diag().mean():.3f}")
```

```python
# 苦手なこと (a) 数: 柴犬が 1・2・3 匹の写真を「N 匹の犬」の文で分類する
nums = ["一匹の犬", "二匹の犬", "三匹の犬", "四匹の犬"]
sims = embed_images([it["image"] for it in count_items]) @ embed_texts(nums).T
for true_n, row in zip([1, 2, 3], (sims / temperature).softmax(1)):
    print(f"実際は {true_n} 匹:", "  ".join(f"{n} {p:.2f}" for n, p in zip(nums, row.tolist())))

# 苦手なこと (b) 位置関係: 犬と猫の写真を左右に並べた画像を作り、「左に犬、右に猫」と逆の文のどちらに近いかを見る
dog, cat = data[0]["image"].resize((224, 224)), data[5]["image"].resize((224, 224))
pairs = {"犬が左": (dog, cat), "猫が左": (cat, dog)}
canvas = []
for left, right in pairs.values():
    c = Image.new("RGB", (448, 224)); c.paste(left, (0, 0)); c.paste(right, (224, 0)); canvas.append(c)
texts = ["左に犬、右に猫がいる写真", "左に猫、右に犬がいる写真"]
sims = embed_images(canvas) @ embed_texts(texts).T
for k, row in zip(pairs, sims.tolist()):
    print(k, {t: round(v, 4) for t, v in zip(texts, row)})
```

```python
# 比較: 英語の CLIP (OpenAI) に、英語のラベルと日本語のラベルで同じ分類をさせる
from transformers import CLIPModel, CLIPProcessor

en_name = "openai/clip-vit-base-patch32"
en_model = CLIPModel.from_pretrained(en_name).to(device).eval()
en_proc = CLIPProcessor.from_pretrained(en_name)
en_labels = ["dog", "cat", "bird", "car", "train", "flower", "ramen", "sushi"]
en_distractors = ["wolf", "fox", "rabbit", "bus", "truck", "airplane", "tree", "udon", "soba", "pasta", "onigiri", "sashimi", "cake"]
for name_, cands, tpl in [("英語ラベル", en_labels + en_distractors, "a photo of a {}."), ("日本語ラベル", labels + distractors, "{}の写真")]:
    for n in [8, len(cands)]:
        inputs = en_proc(text=[tpl.format(y) for y in cands[:n]], images=[it["image"] for it in data], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            pred = en_model(**inputs).logits_per_image.argmax(1)    # 画像 × ラベルの類似度 (温度で割ったもの)
        wrong = [f"{it['label']}→{cands[p]}" for it, p, g in zip(data, pred.tolist(), gold.tolist()) if p != g]
        print(f"{name_} {n} 択: 正解率 {(pred == gold).float().mean():.3f}  誤り {wrong[:8]}{' ...' if len(wrong) > 8 else ''}")
if device == "cuda":
    print(f"GPU メモリの最大使用量: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

- CLIP は、画像のエンコーダとテキストのエンコーダを、画像とそのキャプションの組で対照学習したモデル。`line-corporation/clip-japanese-base` (Apache-2.0) は、Web から集めた約 10 億組で学習されている（モデルカードによる）。パラメータは画像側 8600 万、テキスト側 1 億 1000 万（モデルカードの表では 100M）で、重みは約 0.8GB。float32 のままで、GPU メモリは英語の CLIP と合わせて最大 1.50GB だった。CPU を 2 コアに制限しても、ダウンロード後なら全体が 1 分あまりで終わった。
- ゼロショット分類は「ラベルを文にして埋め込み、画像にいちばん近い文を選ぶ」だけで、分類器の学習は要らない。この 32 枚の 8 択では、どのテンプレートでも全問正解した。テンプレートの違いは、正解と次点のラベルの類似度の差（0.102〜0.113、4 つの平均で 0.117）に表れる程度だった。紛らわしい候補を加えた 21 択では、どのテンプレートでもマラミュートの写真 (Alaskan Malamute R Bartz.jpg) を「オオカミ」と誤った。「これは{}の写真です。」と「{}が写っている画像」では、さらに玉子の握り寿司 (Nigiri Sushi (26505096851).jpg) を「おにぎり」と誤った。32 枚では 1 枚が 3% にあたるので、テンプレートの良し悪しを決めるにはもっと多くの評価画像が要る。モデルのリポジトリには、ImageNet の評価に使った日本語のテンプレート集 (`ja-imagenet-1k-templates.txt`) もある。
- 検索の上位は意図どおりだった（「雪の積もった山で犬が立っている」→ 雪の上のマラミュート、「赤いスポーツカー」→ 赤いクーペ、「地下鉄の駅のホーム」→ 地下鉄）。「黄色い花」の 3 位には、黄色いコーンがのったラーメンが入った（色が手がかりになったと考えられる）。上位 1 件のコサイン類似度でも 0.29〜0.36 程度なので、値そのものではなく順位で使い、「該当なし」にするしきい値はデータで決める。画像の埋め込みは前もって計算しておき、枚数が多ければ検索の章の「FAISS で大量のベクトルから高速に近傍検索する」の方法で探す。
- 類似度行列では、正しい組（対角線）が 0.29〜0.38、それ以外の組は 0.17 以下だった。CLIP の学習の損失 (InfoNCE) は、この行列を温度で割り、画像→文と文→画像の両方向でクロスエントロピーを取ったもの。バッチ内の正しい組が正例、残りの組がすべて負例になる。同じ 8 組では、温度 1 で 1.842（当て推量の log 8 ≈ 2.08 に近い）、0.1 で 0.398、学習済みの 0.01 で 0.000 になった。温度が小さいほど、類似度のわずかな差が確率の大きな差になる。検索の章の「検索用の埋め込みモデルを自分のデータで微調整する」で使った `MultipleNegativesRankingLoss` も同じ仕組みで、クエリと文書の組を画像とキャプションの組に置き換えたものが CLIP にあたる（sentence-transformers の既定は類似度を 20 倍、つまり温度 0.05 で、クエリ→文書の片方向）。
- 細かい区別、数、位置関係は苦手（OpenAI の CLIP のモデルカードも、細かい分類と物の数を数えることを苦手なこととして挙げている）。マラミュートの写真は、シベリアン・ハスキーが 0.64、マラミュートが 0.30 で、ハスキーと答えた。柴犬が 1 匹と 2 匹 (0.62) の写真は当てたが、3 匹の写真は「四匹の犬」が 0.84 だった。犬と猫を左右に並べた画像は、並べ方を逆にしても 2 枚とも「左に猫、右に犬」の文のほうに近く、差は 0.01 程度だった。文全体と画像全体を 1 本ずつのベクトルにまとめるので、語順や配置の情報は残りにくいと考えられる。数や位置を問いたいなら、この章の「視覚言語モデルで画像を説明し、画像について質問に答える (VQA)」を試す。
- 英語の CLIP（`openai/clip-vit-base-patch32`、2021 年）のライセンスは、GitHub の openai/CLIP では MIT で、Hugging Face のカードには書かれていない。英語のラベルでは、8 択で全問正解、21 択で 0.875 だった（犬→wolf、ラーメン 2 枚→soba、寿司→sashimi）。日本語のラベルでは 8 択 0.938、21 択 0.781 に下がった。モデルカードにも、英語以外の言語では学習も評価もしていないとある。なお、英語の CLIP のプロセッサは画像の中央を正方形に切り抜くが、LINE のものは長辺を 224 に縮めて余白を黒で埋める。
- 学習データの偏りはモデルに入る。OpenAI はモデルカードで、学習データはインターネットから集めたもので先進国や若い男性の利用者の側に偏っていると書いている。人の写真の分類で人種や性別による差が出たことも報告している。英語の CLIP が 21 択でラーメンを soba と答えたのは、日本の料理が学習データに少ない例かもしれない。LINE のモデルカードには、日本特有の料理・花・施設・名所の分類（Recruit のデータセット）で、rinna の日本語 CLIP（`rinna/japanese-clip-vit-b-16`。使うには GitHub の専用パッケージが要る）の 0.54 に対して 0.89 という評価が載っている。
- モデルカードどおりの `AutoModel.from_pretrained(name, trust_remote_code=True)` は、Colab の transformers 5 系では `all_tied_weights_keys` がないというエラーで止まる。そこで上では、クラスを取り出して重みを直接読み込んでいる。この回避策は、`trust_remote_code` の確認なしにリポジトリのコードを取得して実行する。そのため `get_class_from_dynamic_module`、`AutoConfig`、`hf_hub_download`、トークナイザ、プロセッサのすべてに `revision=` で同じコミットを渡し、中身を確かめた版に固定する。画像は Wikimedia Commons の縮小版で、出力されたノートブックや図には画像が含まれる。共有するときは、表示した作者・出典（ファイルページの URL）・ライセンスを併記する。CC BY-SA の画像を改変したものを公開するなら、同じライセンスにする。Commons の API は、User-Agent のないアクセスを拒否し、短時間に繰り返すと 429 を返す。

## 視覚言語モデルで画像を説明し、画像について質問に答える (VQA) {vlm_vqa gpu}

画像を入力して、日本語で説明文を作らせたり、画像についての質問（何が写っているか、いくつあるか、何色か、どこにあるか、何と書いてあるか）に答えさせたりしたい。視覚言語モデル (VLM) の Qwen3-VL-4B-Instruct を transformers で動かし、得意なことと苦手なことを確かめ、日本語の VQA データで正解との一致を測る。

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# 初回はモデル (約 8.3GB) のダウンロードに数分かかる
name = "Qwen/Qwen3-VL-4B-Instruct"   # Apache 2.0。画像エンコーダ (ViT) + 射影 + 4B の LLM
processor = AutoProcessor.from_pretrained(name)   # 画像の前処理 (リサイズ・パッチ分割) とトークナイザをまとめたもの
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="cuda").eval()

count = lambda m: sum(p.numel() for p in m.parameters()) / 1e9
print(type(model).__name__, f"画像エンコーダ {count(model.model.visual):.2f}B / LLM {count(model.model.language_model):.2f}B")
print(f"{model.dtype}, GPU メモリ: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
print(model.model.visual.merger)   # 画像の特徴を LLM の埋め込み (2560 次元) に写す射影
```

```python
import io
import requests
from datasets import load_dataset
from PIL import Image, ImageOps

# 日本語の VQA データ JA-VG-VQA-500 (SakanaAI、CC BY 4.0)。Visual Genome (Krishna et al., 2017) の画像に
# 日本語の質問と答えを付けた JA-VG-VQA (Shimizu et al., 2018) から 500 枚。画像は Visual Genome 経由の Flickr の写真で、
# 写真ごとのライセンスはデータセットのライセンスとは別 (vqa[i]["flickr_id"] で元の写真をたどれる)
# parquet を直接読むと分割名は "train" になるが、中身はテスト分割の 500 件
vqa = load_dataset("parquet", data_files="hf://datasets/SakanaAI/JA-VG-VQA-500/data/test-00000-of-00001.parquet", split="train")
food = vqa[31]["image"].convert("RGB")

# 浜寺公園駅 (大阪府堺市、南海本線) の駅名標。Wikimedia Commons、作者 Tatsuya tetsudo、CC0 (パブリックドメイン)
url = "https://upload.wikimedia.org/wikipedia/commons/e/ed/%E6%B5%9C%E5%AF%BA%E5%85%AC%E5%9C%92%E9%A7%85_%E9%A7%85%E5%90%8D%E6%A8%99.jpg"
res = requests.get(url, headers={"User-Agent": "nlpcookbook-example/0.1"}, timeout=60)   # Wikimedia は User-Agent がないと断られる
sign = ImageOps.exif_transpose(Image.open(io.BytesIO(res.content))).convert("RGB")
print(food.size, sign.size)
display(food, sign.resize((sign.width // 8, sign.height // 8)))
```

```python
import time

def ask(image, text, max_new_tokens=128, max_image_tokens=1024):
    """画像 1 枚と質問文を 1 つのメッセージに入れて生成する。max_image_tokens で画像のトークン数の上限を決める"""
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
    # 1 トークン = 32×32 画素 (16×16 のパッチを 2×2 まとめる)。画素数をこの範囲に収まるようにリサイズする
    size = {"shortest_edge": 32 * 32 * 16, "longest_edge": 32 * 32 * max_image_tokens}
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                           return_tensors="pt", processor_kwargs={"size": size}).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# テンプレートの中身: 画像は <|image_pad|> の位置に、画像のトークン数だけ展開される
messages = [{"role": "user", "content": [{"type": "image", "image": food}, {"type": "text", "text": "説明してください。"}]}]
print(processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
print("グリッド (時間, 縦, 横) =", inputs["image_grid_thw"].tolist(), "→ 画像のトークン数",
      (inputs["input_ids"] == model.config.image_token_id).sum().item(), "/ 全体", inputs["input_ids"].shape[1])

for prompt in ["この画像を日本語で一文で説明してください。", "この画像を日本語で詳しく説明してください。"]:
    t = time.perf_counter()
    caption = ask(food, prompt, max_new_tokens=400)
    print(f"\n## {prompt} ({len(caption)} 文字, {time.perf_counter() - t:.1f} 秒)\n{caption}")
```

```python
# 物の種類・数・色・位置について質問する。データに付いている人手の答えと並べる (最後の 2 問は自分で足した質問)
qas = [(qa["question"], qa["answer"]) for qa in vqa[31]["qas"]] + [("人参は何本ありますか？", "6本 (目視で数えた)"), ("スプーンは何色ですか？", "(スプーンは写っていない)")]
for q, gold in qas:
    print(f"Q: {q}\n  正解: {gold}\n  VLM : {ask(food, q + ' 短く答えてください。', max_new_tokens=60)}")
```

```python
# 画像の中の文字を読む・位置関係を答える・画像の外の知識を問う
for q in ["駅名標に書かれている文字をすべて書き出してください。",
          "この駅の名前は何ですか？",
          "左向きの矢印の先にある隣の駅は？", "右向きの矢印の先にある隣の駅は？",
          "この駅は何県にありますか？短く答えてください。",
          "この駅は何県にありますか？画像から判断できなければ「わからない」と答えてください。"]:
    print(f"Q: {q}\nA: {ask(sign, q, max_new_tokens=150)}\n")
```

```python
# 数を数える: 正解がわかるように、赤い丸を n 個描いた画像を作って数えさせる
import random
from PIL import ImageDraw

def circles(n, seed=0, size=(640, 480), r=28):
    rng = random.Random(seed)
    img = Image.new("RGB", size, "white"); draw = ImageDraw.Draw(img); centers = []
    while len(centers) < n:
        x, y = rng.randrange(r + 5, size[0] - r - 5), rng.randrange(r + 5, size[1] - r - 5)
        if all((x - a) ** 2 + (y - b) ** 2 > (2 * r + 10) ** 2 for a, b in centers):   # 重ならないように置く
            centers.append((x, y)); draw.ellipse([x - r, y - r, x + r, y + r], fill="red")
    return img

for n in [3, 5, 8, 12, 17, 25]:
    print(f"正解 {n:2d} → VLM {ask(circles(n, seed=n), '赤い丸はいくつありますか？数字だけで答えてください。', max_new_tokens=10)}")
display(circles(25, seed=25).resize((320, 240)))
```

```python
# 画像のトークン数 (解像度) と、速度・GPU メモリ・文字の読み取りの関係
# 駅名標の元画像は 3072×4080 画素。既定の上限 (16384 トークン) では約 1.2 万トークンになる
for max_tokens in [64, 256, 1024, 4096, 16384]:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    size = {"shortest_edge": 32 * 32 * 16, "longest_edge": 32 * 32 * max_tokens}
    n = processor.image_processor(images=[sign], size=size)["image_grid_thw"].prod().item() // 4   # 2×2 のパッチで 1 トークン
    torch.cuda.synchronize(); t = time.perf_counter()
    answer = ask(sign, "駅名標に書かれている漢字の駅名を、左の駅・この駅・右の駅の順に書いてください。", max_new_tokens=40, max_image_tokens=max_tokens)
    torch.cuda.synchronize()
    print(f"上限 {max_tokens:5d} → 画像 {n:5d} トークン, {time.perf_counter() - t:5.1f} 秒, "
          f"最大メモリ {torch.cuda.max_memory_allocated() / 1e9:.1f} GB: {answer!r}")
```

```python
# JA-VG-VQA-500 の 10 枚 × 4 問 = 40 問で、人手の答えとの一致を測る (バッチでまとめて生成)
import re
import unicodedata

items = [(r["image"].convert("RGB"), qa["question"], qa["answer"]) for r in vqa.select(range(10)) for qa in r["qas"][:4]]
processor.tokenizer.padding_side = "left"   # 生成のバッチ処理では左パディング

def ask_batch(batch, max_new_tokens=24):
    msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": f"質問に単語か短い句で答えてください。\n質問: {q}"}]}]
            for im, q in batch]
    inputs = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                                           processor_kwargs={"padding": True, "size": {"shortest_edge": 32 * 32 * 16, "longest_edge": 32 * 32 * 256}}).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return [s.strip() for s in processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)]

torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
preds = [p for i in range(0, len(items), 8) for p in ask_batch([(im, q) for im, q, _ in items[i:i + 8]])]
print(f"{len(items)} 問 {time.perf_counter() - t:.1f} 秒, 最大メモリ {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

```python
import pandas as pd

KANSUJI = str.maketrans("〇一二三四五六七八九", "0123456789")
def norm1(s):   # 全角半角 (NFKC)・漢数字・文末の「です」「います」・句読点をそろえる
    s = unicodedata.normalize("NFKC", s).translate(KANSUJI)
    return re.sub(r"(が?(います|あります)|です)?[。.!?！？\s]*$", "", s).replace(" ", "")
def norm2(s):   # さらに助数詞と「色」を落とす (「3つ」「3」「3頭」、「黒色」「黒」を同じとみなす)
    return re.sub(r"(?<=\d)(つ|個|台|頭|匹|羽|人|本|枚|種類)$|色$", "", norm1(s))

golds = [g for _, _, g in items]
scores = {
    "完全一致 (そのまま)": [p == g for p, g in zip(preds, golds)],
    "完全一致 (正規化1)": [norm1(p) == norm1(g) for p, g in zip(preds, golds)],
    "完全一致 (正規化2)": [norm2(p) == norm2(g) for p, g in zip(preds, golds)],
    "部分一致 (正規化2)": [norm2(g) in norm2(p) or norm2(p) in norm2(g) for p, g in zip(preds, golds)],
}
print({k: round(sum(v) / len(v), 2) for k, v in scores.items()})
df = pd.DataFrame({"質問": [q for _, q, _ in items], "正解": golds, "VLM": preds, "一致(正規化2)": scores["完全一致 (正規化2)"],
                   "部分一致": scores["部分一致 (正規化2)"]})
df[~df["一致(正規化2)"]]   # 一致しなかったものを見る: 本当の誤りと、言い換え・表記ゆれが混ざっている
```

- 以下の数値と出力は、A5000 で float16 の経路（`BF16 = False`、T4 と同じ dtype）で実行したもの。VLM の大まかな構成は「画像エンコーダ (ViT) → 射影 → LLM」で、画像を 16×16 画素のパッチに切って ViT で特徴にし、2×2 のパッチをまとめて射影（上の `merger`）で LLM の埋め込み（2560 次元）に写す。これは簡略化した説明で、Qwen3-VL は ViT の最終層だけでなく複数の層の特徴を使う（DeepStack、モデルカードによる）。パラメータは画像エンコーダ 0.42B、LLM 4.02B である。
- 呼び方はテキストの LLM（大規模言語モデルの章の「オープンな LLM とチャットする（チャットテンプレート）」）とほぼ同じで、`content` を `{"type": "image"}` と `{"type": "text"}` のリストにするだけである。テンプレートの文字列では画像は `<|image_pad|>` 1 個だが、トークン化すると画像のトークン数だけ並ぶ。500×375 画素の料理の画像は 24×32 のパッチで 192 トークンになり、入力 207 トークンのほとんどを占めた。
- 画像のトークン数はおよそ「画素数 ÷ (32×32)」で、`processor_kwargs={"size": {"shortest_edge": 最小画素数, "longest_edge": 最大画素数}}` で範囲を決める（以前の版の `min_pixels` / `max_pixels` に当たる）。3072×4080 の駅名標は既定の上限のままだと 12288 トークンになり、上の実行では 1 回の質問に 13.3 秒、最大メモリ 12.6GB だった。上限 1024 トークン（972 トークン）では 1.6 秒、9.2GB で同じ駅名を答えた。T4 では数倍かかる見込みである（未測定）。減らしすぎると、54 トークンでは隣の駅を「東寺公園」と作り、逆に 4015 トークンでは「浜」を簡体字の「滨」と書いた。トークン数を増やせば必ず良くなるわけではない。
- 説明文は「一文で」（40 文字）と「詳しく」（400 トークンの上限で途中で打ち切られた）で長さが大きく違い、「詳しく」では「おそらくピーナッツまたはホワイトビーン」のような推測が増える。説明文では左下の器を「人参」と書いたのに、「人参は何本ありますか」には「0」と答えた。同じ画像でも聞き方で答えが食い違う。
- 画像の外の知識を問うと、もっともらしい誤り（ハルシネーション）が出る。駅名は正しく読んだのに「大阪市営地下鉄の千日前線、大阪市東淀川区」と誤った説明を付け足し（実際は南海本線、堺市西区）、都道府県には「わからなければそう答えて」と指示しても理由まで作って「千葉県」と答えた。写っていないスプーンの色には「なし」と答えた。文字の読み取り自体は得意だが、「諏訪ノ森」を「観訪ノ森」と読み違えた（誤り率の測り方は評価指標と統計的検定の章の「音声認識や OCR の誤り率 (WER / CER) を計算する」）。
- 数と位置関係は苦手である。赤い丸は 3 個と 8 個は正しく数えたが、5 個を 4、12 個を 10、17 個を 15、25 個を 20 と少なめに答えた。料理の写真では「右手前」の器（ブロッコリー）を「人参」「オレンジ色」と答えた。数を正確に知りたいなら物体検出のモデルで数えるほうが確実である。
- VQA の評価は、言い換えと表記ゆれのせいで完全一致では低く出る。40 問で、そのままの完全一致 0.25 が、全角半角・漢数字・文末をそろえると 0.40、助数詞と「色」を落とすと 0.53、部分一致では 0.60 になった。部分一致で増えた 3 件（晴れ / 晴れている、地面 / 柵の中の地面の上、黒い / 黒色）はどれも妥当だが、（仮の例だが）正解が「白」で答えが「白くない」でも部分一致では一致になる。残った不一致には、正しい言い換え（ゼブラ / 縞馬、駐車料金機 / パーキングメーター）、粗い言い換えに近いもの（紺色・水色 → 青）、本当の誤り（鴨 3 羽 → 2匹、象 2 頭 → 3、ベッド 3 つ → 2つ、サングラス → 風船）、答えなかったもの（建物の国 → 不明）が混ざる。数値だけでなく不一致の一覧を必ず目で見る。言い換えまで判定したいなら LLM に判定させる（RAG の章の「RAG の回答の正しさを LLM に判定させる」、LLM の評価と分析の章の LLM-as-a-judge）。
- 生成は `do_sample=False`（貪欲法）にしている（モデルカードの推奨はサンプリング）。float16 では重みだけで 8.9GB、上の実行で測った最大は 12.6GB（1.2 万トークンの画像のとき）、画像を 1024 トークン以下にした質問とバッチ評価では 9.5GB 以下だった。この範囲なら T4 (16GB) に載るが、画像を大きくしたり枚数やバッチを増やしたりすると増える。さらに小さい GPU では同じ系列の `Qwen/Qwen3-VL-2B-Instruct` に替えられる。
- リポジトリに保存したノートブックには、料理の写真（JA-VG-VQA の Flickr の写真。写真ごとのライセンスがデータセットとは別）の表示を載せていない。Colab で実行すると表示される。

## 画像の説明文を評価する（BLEU・CIDEr・CLIPScore） {caption_eval}

画像の説明文（キャプション）を、参照のキャプションとの一致（BLEU・CIDEr）と、画像そのものとの一致 (CLIPScore) で自動評価したい。それぞれの指標がどんな誤りに強く、どんな誤りに弱いかを、日本語のキャプションで確かめたい。この章の「視覚言語モデルで画像を説明し、画像について質問に答える (VQA)」で作った説明文も、この方法で評価できる。

```python
# pycocoevalcap は METEOR や SPICE の Java の部品も含むので、ダウンロードが約 100MB ある (ここでは CIDEr だけを使う)
!pip install -q "sacrebleu[ja]" pycocoevalcap
```

```python
# STAIR Captions: MS COCO の画像 1 枚に日本語のキャプションを 5 つ付けたデータ (キャプションは CC BY 4.0)
# 画像は COCO (Flickr) の写真で、データセットのライセンスとは別に 1 枚ごとにライセンスが違う (license 列。非営利のみ・改変禁止のものも含む)
# ここでは画像を表示せず、評価に使うだけにする
# 検証セットの最初のファイルから 300 枚 (100 枚ずつの行グループ 3 つ、約 50MB) だけを読む
import io
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image

path = "datasets/shunk031/STAIR-Captions/v1.2.0/validation-00000-of-00014.parquet"
with HfFileSystem().open(path, block_size=8 << 20) as f:
    rows = pq.ParquetFile(f).read_row_groups([0, 1, 2], columns=["image_id", "image", "annotations"]).to_pylist()

images = [Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB") for r in rows]
captions = [r["annotations"]["caption"] for r in rows]
print(len(images), "枚、キャプションの数:", {len(c) for c in captions})
for c in captions[0]:
    print(" ", c)
```

```python
# 1 つ目のキャプションを「候補」(システムの出力の代わり)、残り 4 つを参照にする
import random

cand = [c[0] for c in captions]
refs = [c[1:] for c in captions]

# わざと誤りを入れた候補を作る: 種類ごとの置き換えの表。文中で最初に見つかった 1 か所だけを置き換える
EDITS = {
    "人物": {"男性": "女性", "女性": "男性", "男の子": "女の子", "女の子": "男の子", "少年": "少女", "少女": "少年"},
    "物": {"犬": "猫", "猫": "犬", "牛": "馬", "馬": "牛", "電車": "バス", "バス": "電車", "自転車": "バイク", "バイク": "自転車",
          "テーブル": "ベッド", "ベッド": "テーブル", "ピザ": "ケーキ", "ケーキ": "ピザ", "キッチン": "浴室", "トイレ": "冷蔵庫",
          "キリン": "ゾウ", "象": "キリン", "ゾウ": "キリン", "飛行機": "船", "皿": "箱"},
    "色": {"赤い": "青い", "青い": "赤い", "白い": "黒い", "黒い": "白い", "白と黒": "赤と青", "茶色": "緑色", "緑": "赤"},
    "位置・数": {"の上に": "の下に", "の下に": "の上に", "の前": "の後ろ", "の横": "の上", "左": "右", "右": "左",
               "一人": "三人", "二人": "一人", "一頭": "三頭", "二頭": "一頭", "1人": "3人", "2人": "1人", "たくさんの": "一つの"},
    "言い換え (意味は同じ)": {"男性": "男の人", "女性": "女の人", "少年": "男の子", "少女": "女の子", "子供": "子ども",
                      "パソコン": "コンピューター", "バイク": "オートバイ", "道路": "道", "机": "テーブル", "沢山": "たくさん"},
}

def replace_first(s, table):
    hits = [(s.find(k), -len(k), k) for k in table if k in s]
    if not hits:
        return None
    i, _, k = min(hits)
    return s[:i] + table[k] + s[i + len(k):]

variants = {kind: [replace_first(s, table) for s in cand] for kind, table in EDITS.items()}
rng = random.Random(0)
perm = list(range(len(cand)))
rng.shuffle(perm)
variants["別の画像"] = [cand[j] if j != i else cand[(i + 1) % len(cand)] for i, j in enumerate(perm)]   # 別の画像のキャプション
other = variants["別の画像"]

for kind, vs in variants.items():
    i = next(i for i, v in enumerate(vs) if v is not None)
    print(f"{kind} ({sum(v is not None for v in vs)} 枚): {cand[i]} → {vs[i]}")
```

```python
# BLEU と CIDEr は、日本語を単語に分割してから n-gram を数える。sacreBLEU の ja-mecab と同じ分割を CIDEr にも使う
from sacrebleu.metrics import BLEU
from sacrebleu.tokenizers.tokenizer_ja_mecab import TokenizerJaMecab
from pycocoevalcap.cider.cider import Cider   # CIDEr-D (n=1〜4、長さの罰則と頻度の上限つき、10 倍した値)

mecab = TokenizerJaMecab()
print(mecab(cand[0]))

def bleu_corpus(hyps, refsets, tokenize="ja-mecab"):
    # 参照は「参照セットのリスト」に組み替える。zip は、画像ごとに参照の数が違うと最も少ない数に切りそろえてしまうので注意
    return BLEU(tokenize=tokenize).corpus_score(hyps, list(map(list, zip(*refsets)))).score

sent_bleu = BLEU(tokenize="ja-mecab", effective_order=True)
def bleu_sent(hyps, refsets):
    return [sent_bleu.sentence_score(h, r).score for h, r in zip(hyps, refsets)]

def cider(hyps, refsets, tok=mecab):
    gts = {i: [tok(r) for r in rs] for i, rs in enumerate(refsets)}
    res = {i: [tok(h)] for i, h in enumerate(hyps)}
    mean, per_image = Cider().compute_score(gts, res)
    return mean, per_image

print(f"人のキャプション (参照 4 つ): BLEU = {bleu_corpus(cand, refs):.1f}  CIDEr = {cider(cand, refs)[0]:.3f}")
refs2 = [r[:2] for r in refs]
print(f"人のキャプション (参照 2 つ): BLEU = {bleu_corpus(cand, refs2):.1f}  CIDEr = {cider(cand, refs2)[0]:.3f}")
```

```python
# 単語に分割しないとどうなるか。13a (英語向け) では日本語の文全体がほぼ 1 トークンになる
for name, tok in [("ja-mecab", "ja-mecab"), ("char", "char"), ("13a", "13a")]:
    print(f"BLEU ({name:8}) 人 = {bleu_corpus(cand, refs, tok):5.1f}  別の画像 = {bleu_corpus(other, refs, tok):5.1f}")
for name, tok in [("MeCab", mecab), ("文字", lambda s: " ".join(s)), ("分割なし", lambda s: s)]:
    print(f"CIDEr ({name:4}) 人 = {cider(cand, refs, tok)[0]:.3f}  別の画像 = {cider(other, refs, tok)[0]:.3f}")
```

```python
# CLIPScore (Hessel et al., 2021): 画像とキャプションの埋め込みのコサイン類似度 c から 2.5 × max(c, 0)。参照が要らない
# (元の論文は英語の CLIP で、文の前に "A photo depicts" を付けている。ここでは日本語の CLIP に文をそのまま入れる)
# RefCLIPScore: CLIPScore と「候補と参照の埋め込みの類似度の最大値」の調和平均。参照も使う
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

device = "cuda" if torch.cuda.is_available() else "cpu"
name = "line-corporation/clip-japanese-base"   # LY Corporation の日本語 CLIP (Apache-2.0)。この章の CLIP のレシピと同じモデルと読み込み方
rev = "77a62f8af977acd73ec2f927fb73fdaeb13af7d1"      # リポジトリのコミットを固定する (中のコードを実行するので、すべての読み込みに渡す)
clip_tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, revision=rev)
clip_proc = AutoImageProcessor.from_pretrained(name, trust_remote_code=True, revision=rev)
# モデルカードの AutoModel.from_pretrained(name, trust_remote_code=True) は transformers 5 系ではエラーになるので、
# リポジトリ内のクラスを取り出して重みを直接読み込む
CLYPModel = get_class_from_dynamic_module("modeling_clyp.CLYPModel", name, revision=rev)  # modeling_clyp.py を取得してクラスを返す
clip = CLYPModel(AutoConfig.from_pretrained(name, trust_remote_code=True, revision=rev))  # config から重みが乱数のモデルを作る
print(clip.load_state_dict(load_file(hf_hub_download(name, "model.safetensors", revision=rev))))  # All keys matched と出れば全部の重みが入った
clip = clip.to(device).eval()                        # 重みは float32 のまま使う (約 0.8GB で、T4 にも CPU にも載る)

@torch.no_grad()
def embed_images(imgs, bs=32):
    out = [clip.get_image_features(**clip_proc(imgs[i:i + bs], return_tensors="pt").to(device)) for i in range(0, len(imgs), bs)]
    return torch.nn.functional.normalize(torch.cat(out).float(), dim=-1).cpu().numpy()

@torch.no_grad()
def embed_texts(texts, bs=64):
    out = [clip.get_text_features(**clip_tok(texts[i:i + bs]).to(device)) for i in range(0, len(texts), bs)]
    return torch.nn.functional.normalize(torch.cat(out).float(), dim=-1).cpu().numpy()

img_emb = embed_images(images)   # CPU (2 コア) では、モデルの読み込みを含めて 2〜4 分程度
ref_emb = [embed_texts(rs) for rs in refs]

def clip_score(hyps, idx=None):
    """idx: 各候補に対応する画像の番号 (省略時は 0, 1, 2, ...)"""
    idx = list(range(len(hyps))) if idx is None else idx
    return 2.5 * np.maximum((embed_texts(hyps) * img_emb[idx]).sum(-1), 0)

def ref_clip_score(hyps, idx=None):
    idx = list(range(len(hyps))) if idx is None else idx
    h = embed_texts(hyps)
    cs = 2.5 * np.maximum((h * img_emb[idx]).sum(-1), 0)
    rs = np.array([max(0, (ref_emb[i] @ v).max()) for i, v in zip(idx, h)])
    return 2 * cs * rs / np.maximum(cs + rs, 1e-8)   # 両方 0 のときの 0 除算を避ける

print(f"CLIPScore 人のキャプション = {clip_score(cand).mean():.3f}  別の画像 = {clip_score(other).mean():.3f}")
```

```python
# 誤りの種類ごとに、置き換えができた画像だけを集めて、元の候補と誤りを入れた候補を比べる
import pandas as pd

def evaluate(hyps, idx):
    h, r = [hyps[i] for i in idx], [refs[i] for i in idx]
    each = dict(文BLEU=np.array(bleu_sent(h, r)), CIDEr=cider(h, r)[1], CLIPScore=clip_score(h, idx), RefCLIPScore=ref_clip_score(h, idx))
    return {"BLEU": bleu_corpus(h, r), **{k: v.mean() for k, v in each.items()}}, each

ratio, wins = [], []
for kind, hyps in variants.items():
    idx = [i for i, s in enumerate(hyps) if s is not None]
    base, base_each = evaluate(cand, idx)     # 同じ画像の集合で、元の (人の) 候補も評価する
    test, test_each = evaluate(hyps, idx)
    ratio.append({"種類": kind, "枚数": len(idx), **{k: f"{test[k] / base[k]:.0%}" for k in base}})
    # 1 枚ごとに「元の候補のほうが高い」割合。意味が変わる誤りなら 100% に近いほど、誤りをよく見分けている
    # 値が変わらない (同点) 場合は括弧内に割合を示す。置き換えた語が参照にも出てこなければ、BLEU は変わらないことが多い
    wins.append({"種類": kind, **{k: f"{(base_each[k] > test_each[k]).mean():.0%} ({(np.isclose(base_each[k], test_each[k])).mean():.0%})"
                                for k in base_each}})
print("誤りを入れた候補の値 ÷ 元の候補の値")
display(pd.DataFrame(ratio))
print("1 枚ごとに、元の候補のほうが高く評価された割合 (括弧内は同点の割合)")
display(pd.DataFrame(wins))
```

```python
# CIDEr は参照の中でよく出る n-gram を IDF で割り引く。参照から求めた IDF の例
from collections import Counter

df = Counter(g for rs in refs for g in {w for r in rs for w in mecab(r).split()})
N = len(refs)
for w in ["が", "て", "いる", "の", "男性", "テーブル", "キリン", "パソコン"]:
    print(f"{w:8} 出てくる画像の数 = {df[w]:3}  IDF = {np.log(N / max(1, df[w])):.2f}")

# どの画像にも当てはまりそうな「ありがちな」候補を全画像に出したとき
for generic in ["男性が立っている", "テーブルの上に置いてある"]:
    g = [generic] * len(cand)
    print(f"{generic}: BLEU = {bleu_corpus(g, refs):.1f}  CIDEr = {cider(g, refs)[0]:.3f}  CLIPScore = {clip_score(g).mean():.3f}")
```

- 人が書いたキャプションを同じ画像の別の人のキャプションで評価しても、BLEU は 25.3、CIDEr は 0.806 にとどまった（300 枚、参照 4 つ）。システムの値は、この「人同士の一致」を目安に読む。参照を 2 つに減らすと BLEU は 17.1 に下がった（どれかの参照と一致すればよいので、参照が多いほど一致しやすい）が、CIDEr は参照ごとの類似度の平均なので 0.822 とほとんど変わらなかった。参照の数が違う結果同士は比べない。
- 日本語は単語に分割してから数える。英語向けの `13a` では BLEU が 0.0、分割しない CIDEr は 0.002 になった。文字単位にすると BLEU 39.2・CIDEr 1.108 と MeCab で分割したとき（25.3・0.806）より高く出るので、分割の方法が違う値も比べられない（機械翻訳の章の「機械翻訳を BLEU で評価する (sacreBLEU)」の署名を参照）。pycocoevalcap の分割器 (PTBTokenizer) は英語用で Java も要るので使わず、MeCab で分割した文字列を `Cider` に渡している。
- n-gram の一致の指標は言い換えに弱い。意味が同じ言い換え（「バイク → オートバイ」など）でも、BLEU は元の 86%、CIDEr は 88% に下がり、文単位の BLEU では 70% の画像で元の候補のほうが高かった。CLIPScore は 101% で、元のほうが高いのは 48%（ほぼ半々）だった。言い換えに強い参照ありの指標には、評価指標と統計的検定の章の「BERTScore で意味的な近さを評価する」もある。
- CLIPScore は細かい誤りに鈍い。物の入れ替えは 92% の画像で見分けたが、位置・数の誤り（「の上に → の下に」「2人 → 1人」など）では値が元の 97% のままで、見分けたのは 70% だった。参照と画像の両方を使う RefCLIPScore は、物の入れ替えで 98%、位置・数で 80% と、CLIPScore より見分けた。
- 色の入れ替えでは、文単位の BLEU の 48% が同点だった。元の語（「赤い → 青い」の「赤い」）も新しい語（「青い」）も参照に出てこなければ、一致する n-gram は変わらないからである。CIDEr はコサイン類似度で、候補の n-gram のベクトルの長さで割るので、参照と一致しない語でも、元の語と新しい語の IDF が違えば値が変わる。そのため同点は 0% だったが、色を入れ替えたほうが高くなった画像も 17% あった（元のほうが高いのは 83%）。誤りを入れれば必ず下がるわけではない。
- CIDEr は、多くの画像の参照に出てくる n-gram を IDF で割り引く（「が」は 0.00、「男性」は 1.99、「パソコン」は 3.76）。どの画像にも当てはまりそうな「男性が立っている」を全画像に出すと、BLEU は人の値の約 23% (5.9) なのに、CIDEr は約 4〜5% (0.036) まで下がった。IDF は評価する画像の集合の参照から求めるので、集合が変われば値も変わる（上の表では、誤りの種類ごとに集合が違うので、元の候補との比で示した）。
- CLIPScore は参照なしで使える反面、値は 0〜2.5 のうち下のほうに集まる（人のキャプションで 0.899、コサイン類似度にして約 0.36）。また、ありがちな候補にも 0.413 が付き、別の画像のキャプション (0.250) より高かった。係数の 2.5 は元の論文（英語の CLIP）で使われた値で、モデルが変われば値の範囲も変わる。違うモデルの CLIPScore 同士は比べない。CLIP の仕組みは、この章の「CLIP で画像と日本語の文を同じ空間に埋め込み、検索・分類する」を参照。
- ここでは 300 枚（誤りの種類ごとに 39〜123 枚）の小さな実験で、置き換えも表に基づく機械的なもの。システムを比べるときは、評価指標と統計的検定の章の「ブートストラップ法で評価値の信頼区間を求める」で信頼区間を求め、人手評価との相関は「自動評価指標と人手評価の相関を調べる」で確かめる。なお STAIR Captions の検証セットは、clip-japanese-base のモデルカードで検索の評価に使われているデータでもある。

## 画像の中の日本語を読み取り (OCR)、文字誤り率で比べる {ocr_ja gpu}

画像に写った日本語の文字を読み取りたい。従来型の OCR (Tesseract)、日本語に特化した OCR (manga-ocr)、視覚言語モデル（VLM、Qwen3-VL）を、条件を変えた画像の文字誤り率 (CER) で比べる。

```python
# Tesseract 本体 (Colab には入っている) と日本語 (横書き・縦書き) のモデル、日本語フォント (Noto CJK)
!apt-get install -y -q tesseract-ocr tesseract-ocr-jpn tesseract-ocr-jpn-vert fonts-noto-cjk > /dev/null
!pip install -q pytesseract jiwer manga-ocr
```

```python
# 正解が分かるように、PIL で日本語の文を描いた画像を自分で作る (条件: フォント、文字の大きさ、傾き、ノイズ、ぼけ、背景の模様、縦書き)
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SANS = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"    # fonts-noto-cjk のゴシック体
SERIF = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"  # 明朝体

def render(text, size=32, font=SANS, vertical=False, angle=0, noise=0, scale=1, pattern=False, pad=24):
    f = ImageFont.truetype(font, size)
    lines, step = text.split("\n"), int(size * 1.6)
    if vertical:
        w, h = len(lines) * step + 2 * pad, max(len(l) for l in lines) * int(size * 1.05) + 2 * pad
    else:
        w, h = max(int(f.getlength(l)) for l in lines) + 2 * pad, len(lines) * step + 2 * pad
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    if pattern:                                     # 斜めの縞模様の背景
        for x in range(-h, w, 12):
            d.line([(x, 0), (x + h, h)], fill=170, width=3)
    for i, l in enumerate(lines):
        if vertical:                                # 縦書き: 右の列から左へ。vert で括弧や長音記号を縦書き用の字形にする
            d.text((w - pad - (i + 1) * step + (step - size) // 2, pad), l, font=f, fill=0, direction="ttb", features=["vert"])
        else:
            d.text((pad, pad + i * step), l, font=f, fill=0)
    if angle:
        img = img.rotate(angle, expand=True, fillcolor=255, resample=Image.BICUBIC)
    if scale != 1:                                  # 一度縮小してから戻す (解像度の低い画像)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR).resize(img.size, Image.BILINEAR)
    if noise:                                       # 少しぼかしてからガウスノイズを足す
        a = np.asarray(img.filter(ImageFilter.GaussianBlur(1)), dtype=float)
        img = Image.fromarray((a + np.random.default_rng(0).normal(0, noise, a.shape)).clip(0, 255).astype("uint8"))
    return img.convert("RGB")

TEXTS = ["クックブック食堂は月曜日が定休日です。\n営業時間は11時から15時までとなります。",
         "ロボットのカメラは工場の入口に設置された。\n二人のエンジニアが力を合わせて調整した。",   # ロ/口、エ/工、ニ/二、カ/力
         "「ＡＩ」と「人工知能」は同じ意味ですか？\n価格は3,980円（税込）、残りは2個です。"]       # 全角・半角、記号
HARD = ["口ロ工エ力カ二ニ夕タ卜ト八ハ",                        # 漢字とカタカナの組 (意味のない並び)
        "森鷗外の旧居の近くに髙橋さんと山﨑さんが住む。",       # 異体字 (鷗、髙、﨑)
        "型番B8-0O1l、価格¥1,280（約9.5%引き）"]               # 英数字と記号 (0/O、1/l)
CONDS = {"基準": {}, "明朝体": {"font": SERIF}, "小さい文字": {"size": 10}, "傾き5度": {"angle": 5},
         "ノイズ": {"noise": 60}, "低解像度": {"scale": 0.3}, "背景の模様": {"pattern": True}, "縦書き": {"vertical": True}}
data = [(c, t, render(t, **kw)) for c, kw in CONDS.items() for t in TEXTS]
data += [("紛らわしい文字", t, render(t)) for t in HARD]
print(len(data), "枚")
for c, t, img in data[2::3]:
    print(c)
    display(img)
```

```python
# (1) 従来型の OCR: Tesseract。横書きは jpn、縦書きは jpn_vert のモデルを使う
import time
import pytesseract

def tesseract(img, vertical=False):
    if vertical:
        return pytesseract.image_to_string(img, lang="jpn_vert", config="--psm 5")   # psm 5: 縦書きの 1 ブロック
    return pytesseract.image_to_string(img, lang="jpn", config="--psm 6")            # psm 6: 横書きの 1 ブロック

results, seconds = {}, {}
t = time.time()
results["Tesseract"] = [tesseract(img, c == "縦書き") for c, _, img in data]
seconds["Tesseract"] = (time.time() - t) / len(data)
for i in [0, 12, 21]:     # 基準、ノイズ、縦書き
    print(repr(results["Tesseract"][i]))
```

```python
# (2) 日本語に特化した OCR: manga-ocr (漫画の吹き出しなど、短い文字のまとまりを 1 回で読む。縦書きも横書きもそのまま読める)
from manga_ocr import MangaOcr

mocr = MangaOcr()   # 初回はモデル (kha-white/manga-ocr-base, 約 450MB) をダウンロードする
t = time.time()
results["manga-ocr"] = [mocr(img) for _, _, img in data]
seconds["manga-ocr"] = (time.time() - t) / len(data)
print(repr(results["manga-ocr"][0]))
```

```python
# (3) 視覚言語モデル (VLM) に「書き写して」と頼む: Qwen3-VL-4B-Instruct (Apache-2.0)
# 初回はモデル (約 9GB) のダウンロードに数分かかる。float16 で GPU メモリを約 9GB 使うので T4 (16GB) に載る
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

name = "Qwen/Qwen3-VL-4B-Instruct"
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
processor = AutoProcessor.from_pretrained(name)
model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="cuda").eval()

def vlm_ocr(img, prompt="画像の中の文字を、改行も含めてそのまま書き写してください。書き写した文字以外は出力しないでください。"):
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

t = time.time()
results["Qwen3-VL"] = [vlm_ocr(img) for _, _, img in data]
seconds["Qwen3-VL"] = (time.time() - t) / len(data)
for i in [0, 2]:
    print(repr(results["Qwen3-VL"][i]))
print(f"GPU メモリの最大使用量 (manga-ocr を含む): {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

```python
# 正規化 (NFKC で全角・半角をそろえ、空白と改行を除く) してから、条件ごとに文字誤り率 (CER) を計算する
import unicodedata
import jiwer
import pandas as pd

def normalize(s):
    return "".join(unicodedata.normalize("NFKC", s).split())

rows = []
for method, outs in results.items():
    for (c, ref, _), hyp in zip(data, outs):
        rows.append({"条件": c, "手法": method, "CER": jiwer.cer(normalize(ref), normalize(hyp) or " "),
                     "CER (正規化なし)": jiwer.cer(ref, hyp.strip() or " ")})
df = pd.DataFrame(rows)
table = df.pivot_table(index="条件", columns="手法", values="CER", sort=False)
table.loc["平均"] = table.mean()
print(table.round(3).to_string())
print("\n正規化しない場合の平均 CER:", df.groupby("手法")["CER (正規化なし)"].mean().round(3).to_dict())
print("1 枚あたりの秒数:", {k: round(v, 2) for k, v in seconds.items()})
```

```python
# どんな誤りが多いか: 文字の置換 (正解 → 出力) を数え、誤りの例を並べる
from collections import Counter

def substitutions(ref, hyp):
    ref, hyp = normalize(ref), normalize(hyp)
    out = jiwer.process_characters(ref, hyp or " ")
    pairs = []
    for ch in out.alignments[0]:
        if ch.type == "substitute":
            pairs += [f"{ref[ch.ref_start_idx + k]}→{hyp[ch.hyp_start_idx + k]}" for k in range(ch.ref_end_idx - ch.ref_start_idx)]
    return pairs

for method, outs in results.items():
    pairs = Counter(p for (_, ref, _), hyp in zip(data, outs) for p in substitutions(ref, hyp))
    print(method, pairs.most_common(12))

for i in [8, 23, 25, 26]:     # 小さい文字、縦書き、紛らわしい文字 (異体字、英数字)
    c, ref, _ = data[i]
    print(f"\n[{c}] 正解: {ref!r}")
    for method, outs in results.items():
        print(f"  {method:10s} {outs[i].strip()!r}")
```

```python
# 縦書きの画像を横書きのモデル (jpn) で読むと、行を横に読んでしまい、読み順が崩れる
c, ref, img = data[21]
print(repr(pytesseract.image_to_string(img, lang="jpn", config="--psm 6")[:60]))
```

```python
# VLM は書かれていない文字を補ったり、誤字を直したりすることがある。正解とは「画像に書かれているとおり」の文字列
def masked(text, start, n, size=32):     # text[start:start+n] を白い四角で消す
    img = render(text, size=size)
    f = ImageFont.truetype(SANS, size)
    x0, x1 = 24 + f.getlength(text[:start]), 24 + f.getlength(text[:start + n])
    ImageDraw.Draw(img).rectangle([x0, 20, x1, 24 + size * 1.3], fill="white")
    return img

demo = {"誤字 (ごさいました)": render("本日はご来店いただき、誠にありがとうごさいました。"),
        "誤字 (シュミレーション)": render("明日は避難訓練のシュミレーションを行います。"),
        "重複 (からから)": render("会議は午後3時からからです。"),
        "1 字を消す (回)": masked("急がば回れ、急いては事を仕損じる。", 3, 1)}
for k, img in demo.items():
    print(f"[{k}]\n  Tesseract {tesseract(img).strip()!r}\n  manga-ocr {mocr(img)!r}\n  Qwen3-VL  {vlm_ocr(img)!r}")
```

- 表の値は、条件ごとに 1 枚ずつ計算した CER の平均。評価指標と統計的検定の章の「音声認識や OCR の誤り率 (WER / CER) を計算する」のように jiwer に文の一覧をまとめて渡すと、全体の文字数で割った CER になり、値が少し違う。1 条件は 3 枚（約 70 字）しかないので、1 字の誤りで CER が約 0.014 動く（基準の 0.008 と低解像度の 0.016 の差は 1 字ぶん）。小さな差で手法の優劣を決めない。
- 正規化の影響: Tesseract は縦書きで字の間に空白を入れ、括弧を半角にする。manga-ocr はライブラリの後処理で空白を除き、英数字を全角にする。Qwen3-VL も「A I」のように空白を入れる。正規化しないと平均 CER は Tesseract 0.397、manga-ocr 0.253、Qwen3-VL 0.037 で、NFKC と空白の除去をすると 0.280、0.190、0.019 になった。ここでは記号の誤りも見たいので句読点は残した。正規化の方針で値が変わるので、そろえて明記する。
- 従来型の OCR は、レイアウト解析（ブロックや行を見つける。`--psm` で指定する）と、行ごとの文字認識（Tesseract 4 以降は LSTM）を順に行う。内部で二値化（大津の方法）してから文字を探すので（公式ドキュメント「Improving the quality of the output」）、背景の模様（縞を N や S と読んだ）やノイズ（空の出力）で読めなくなり、小さい文字や傾きでも誤りが増えた。VLM は画像全体を見て、読み順も含めて文字列を生成する。そのぶん重く、1 枚あたり Tesseract は 0.18 秒 (CPU)、Qwen3-VL は 1.17 秒 (GPU) かかった（検証に使った 32 コアの CPU と RTX A5000 での値で、実行ごとに 2〜3 割変わる。Colab の T4 では確かめていない）。
- 誤りの種類（上の出力より）。似た字形: Tesseract は「ＡＩ」の I を T やエと読み、堂→登、意→斉と取り違えた。Qwen3-VL は意味のない並びの「力カ二ニ夕タ卜ト八ハ」で、漢字の力・二・夕・卜・八をカタカナにした（口ロ工エは区別できた）。記号: Tesseract は ¥ をバックスラッシュ (`\`) に、「9.5%」を「9.59%」に、「3,980」を「3,.980」や「3980」にした。異体字: 髙→高、﨑→崎は Tesseract と Qwen3-VL の両方が常用の字形に置き換えた（鷗は Qwen3-VL だけが正しく、Tesseract は「暗」）。NFKC の正規化では、これらは同じ字にならない。
- 縦書きは、Tesseract では縦書き用のモデル `jpn_vert` と `--psm 5` を指定すると読み順も正しく読めた。横書きのモデル `jpn` で読むと、2 列を横に並べて読んでしまう（「営 ク / 業 ツ ...」）。manga-ocr と Qwen3-VL は、指定しなくても縦書きを右の列から読んだ。
- VLM は、書かれていない文字をもっともらしく直すことがある。誤字の「ごさいました」を「ございました」に、「シュミレーション」を「シミュレーション」に直して出力した（Tesseract は書かれたとおり。manga-ocr も「ございました」に直した）。一方、「からから」の重複や、消した「回」は補わなかったので、いつ直すかは予測しにくい。誤字の検出や史料の翻刻のように、書かれたとおりに読みたい用途では、この書き換えが誤りになる。manga-ocr（kha-white/manga-ocr-base、Apache-2.0）は漫画の吹き出しを主な対象とするモデル。入力を 224×224 に縮小する (preprocessor_config) ので、横に長い 2 行の画像は不利だと考えられ、「人工知能」を「人工知識」と読むような、もっともらしい別の語を出す誤りが目立った。
- この結果は、1 種類のフォントで描いた短い文の合成画像に限ったもの。Qwen3-VL の CER がほぼ 0 でも、VLM が完璧に読めるということではない。段組み、表、長いページ、手書き、写真の中の文字は試していない。長いページでは、この `max_new_tokens=200` だと出力が途中で切れる。対象と同じ種類の画像を集めて正解を付け、同じ方法で CER を測ってから選ぶ。PDF に文字のデータがあるなら、データの収集と入出力の章の「PDF からテキストを抽出する (PyMuPDF)」のように抽出するほうが正確で速い（そのレシピの注意書きにある PyMuPDF の `get_textpage_ocr()` は、中で Tesseract を呼び出す）。
- 数値はすべて float16 で実行した値。Qwen3-VL-4B-Instruct (Apache-2.0) のモデルカードは、OCR が 32 言語に対応し、ぼけや傾きに強いとしている。GPU メモリの最大使用量は 9.4GB（manga-ocr のモデルも GPU に載せたままの値）。bfloat16 で実行したときも、CER の表、出力、メモリの使用量は同じだった。apt の `tesseract-ocr-jpn` は tessdata_fast（速度重視の整数版モデル）で、精度を優先する tessdata_best も公開されている（ここでは比べていない）。

## PDF のページを画像にして視覚言語モデルで読む（表とグラフ） {pdf_vlm gpu}

表やグラフを含む PDF を読み取りたい。テキストを抽出するだけ（データの収集と入出力の章の「PDF からテキストを抽出する (PyMuPDF)」）では表の行と列の対応やグラフの値が失われるので、ページを画像にして視覚言語モデル (VLM) の Qwen3-VL-4B-Instruct に読ませ、テキストだけを渡した場合と比べる。

```python
!pip install -q pymupdf matplotlib-fontja
```

```python
import urllib.request
import pymupdf  # PyMuPDF (import fitz と同じ)

# 出典:「統計トピックスNo.148 我が国のこどもの数」（総務省統計局）（https://www.stat.go.jp/data/jinsui/topics/topi1480.html）を加工して作成
# 利用条件: 公共データ利用規約 第1.0版 (PDL1.0)。CC BY 4.0 と互換
urllib.request.urlretrieve("https://www.stat.go.jp/data/jinsui/topics/pdf/topics148.pdf", "topics148.pdf")
pub = pymupdf.open("topics148.pdf")[1]      # 2 ページ目: 表 1、図 1 (棒グラフ)、表 2 (見出しが 2 段の表)
text = pub.get_text()
print(len(text), "文字")
print(text[:420].replace("\n", " | "))      # 表 2 の部分。見出しの語順が崩れ、数値は 1 列に並ぶ

# スキャンした PDF を模して、ページの画像だけを貼った PDF を作ると、テキスト層がないので何も取れない
scan = pymupdf.open()
scan.new_page(width=pub.rect.width, height=pub.rect.height).insert_image(pub.rect, pixmap=pub.get_pixmap(dpi=100))
print("画像だけの PDF から抽出したテキスト:", repr(scan[0].get_text()))
```

```python
# 正解の分かる PDF を自分で作る (架空のデータ)。表 1 は数値を書くが、図 1 の折れ線には値を書き込まない
import matplotlib.pyplot as plt
import matplotlib_fontja

plt.rcParams["pdf.fonttype"] = 42          # フォントを TrueType で埋め込む (テキストとして抽出できる)
years = ["2019", "2020", "2021", "2022", "2023"]
table = {"来館者数(千人)": [412, 198, 265, 331, 356], "貸出冊数(千冊)": [1520, 1105, 1283, 1347, 1398], "電子書籍(千件)": [12, 48, 61, 70, 77]}
series = {"一般書": [820, 560, 640, 690, 700], "児童書": [430, 380, 450, 470, 520], "雑誌": [270, 165, 193, 187, 178]}

fig = plt.figure(figsize=(8.27, 11.69))    # A4
fig.text(0.08, 0.95, "架空市立図書館の利用状況 (架空のデータ)", fontsize=16, weight="bold")
fig.text(0.08, 0.91, "2020年度は臨時休館の影響で来館者数が大きく減少した。電子書籍の貸出は毎年増えている。", fontsize=10)
ax = fig.add_axes([0.08, 0.62, 0.84, 0.25]); ax.axis("off"); ax.set_title("表1 年度別の利用状況", fontsize=12)
t = ax.table(cellText=[[y] + [str(v[i]) for v in table.values()] for i, y in enumerate(years)],
             colLabels=["年度"] + list(table), loc="center", cellLoc="center")
t.scale(1, 1.8); t.set_fontsize(11)
ax2 = fig.add_axes([0.12, 0.12, 0.8, 0.38])
for label, v in series.items():
    ax2.plot(years, v, marker="o", label=label)
ax2.set_title("図1 分類別の貸出冊数 (千冊)", fontsize=12); ax2.set_ylim(0, 900); ax2.grid(alpha=0.4); ax2.legend()
fig.savefig("report.pdf"); plt.close(fig)

rep = pymupdf.open("report.pdf")[0]
print(rep.get_text().replace("\n", " | "))  # 表の数値と軸の目盛りは取れるが、折れ線の値はどこにもない
```

```python
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

name = "Qwen/Qwen3-VL-4B-Instruct"         # Apache-2.0。重みは約 8.9GB (初回はダウンロードに数分かかる)
BF16 = torch.cuda.get_device_capability()[0] >= 8   # bfloat16 は Ampere 以降。T4 では float16
processor = AutoProcessor.from_pretrained(name)
model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="cuda").eval()

def page_image(page, dpi):
    pix = page.get_pixmap(dpi=dpi)          # 解像度を dpi で指定して画像にする (72 dpi で 1pt = 1 画素)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

def ask(question, image=None, text=None, max_new_tokens=64):
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    if text is not None:                     # 比較用: 同じモデルに抽出したテキストだけを渡す
        content.append({"type": "text", "text": "次は PDF のページから抽出したテキストです。\n" + text})
    content.append({"type": "text", "text": question})
    inputs = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                           tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    n_image = int((inputs["input_ids"] == model.config.image_token_id).sum())
    return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip(), n_image

img = page_image(pub, 100)
md, n_image = ask("このページの内容を Markdown に変換してください。表は Markdown の表にしてください。", image=img, max_new_tokens=1500)
print(img.size, "画素 →", n_image, "画像トークン")
print(md)
```

```python
import re

# Markdown への変換では図が画像への参照に置き換わるので、グラフの値は明示的に表にさせる
print(ask("図1の棒グラフの値を読み取って、Markdown の表にしてください。", image=page_image(pub, 100), max_new_tokens=300)[0])

# 自作 PDF の折れ線グラフ (値は書き込んでいない) を表にさせ、正解との差を測る
for dpi in [100, 150]:
    ans = ask("図1の折れ線グラフについて、各分類・各年度の値を読み取って Markdown の表にしてください。",
              image=page_image(rep, dpi), max_new_tokens=400)[0]
    rows = {r[0]: r[1:] for line in ans.splitlines() if re.match(r"\|\s*20\d\d", line)
            for r in [[int(x) for x in re.findall(r"\d+", line)]]}
    if dpi == 100:
        print(ans)
    errors = [abs(rows[int(y)][j] - v[i]) for i, y in enumerate(years) for j, v in enumerate(series.values())]
    print(f"{dpi} dpi: 15 個の値の平均絶対誤差 {sum(errors) / len(errors):.1f} 千冊、最大 {max(errors)} 千冊")
```

```python
import unicodedata

QA = [  # (ページ, 質問, 正解)。統計局の資料の正解はページに書かれた値
    ("pub", "表1で、2025年4月1日現在の女のこどもの数は何万人ですか。", "666"),
    ("pub", "表1で、総人口(男女計)の対前年増減数は何万人ですか。", "-54"),
    ("pub", "表2で、9～11歳の男のこどもの数は何万人ですか。", "152"),
    ("pub", "表2で、未就学の乳幼児(0～5歳)の女の数は何万人ですか。", "222"),
    ("pub", "表2で、3～5歳のこどもが総人口に占める割合は何％ですか。", "2.0"),
    ("pub", "図1で、こどもの数が最も多い年齢階級はどれですか。", "12～14"),
    ("rep", "表1で、2021年度の貸出冊数は何千冊ですか。", "1283"),
    ("rep", "表1で、来館者数が最も少ない年度はどれですか。", "2020"),
    ("rep", "図1で、2020年度の一般書の貸出冊数はおよそ何千冊ですか。", "560"),
    ("rep", "図1で、2023年度の児童書の貸出冊数はおよそ何千冊ですか。", "520"),
    ("rep", "図1で、雑誌の貸出冊数が最も多い年度はどれですか。", "2019"),
    ("rep", "図1で、2019年度より2023年度のほうが貸出冊数が多い分類はどれですか。", "児童書"),
]
pages = {"pub": pub, "rep": rep}

def is_correct(answer, gold, question):
    a = unicodedata.normalize("NFKC", answer).replace(",", "").replace("~", "～").replace("−", "-")
    if not re.fullmatch(r"-?[\d.]+", gold):
        return a.startswith(gold)            # 「12～14歳」「2020年度」は可、「一般書、児童書」のような列挙は不可
    nums = re.findall(r"-?\d+(?:\.\d+)?", a)
    if not nums:
        return False
    tol = 0.05 * abs(float(gold)) if "およそ" in question else 0   # グラフから読む値は ±5% まで正解とする
    return abs(float(nums[0]) - float(gold)) <= tol

def evaluate(mode, dpi=100, verbose=True):
    score, wrong = 0, []
    for pg, q, gold in QA:
        q2 = q + "答えだけを短く答えてください。"
        ans = ask(q2, text=pages[pg].get_text())[0] if mode == "text" else ask(q2, image=page_image(pages[pg], dpi))[0]
        ok = is_correct(ans, gold, q); score += ok
        if not ok:
            wrong.append(f"{gold}→{ans[:8]!r}")
        if verbose:
            print(f"{'○' if ok else '×'} 正解 {gold:>6} | {ans[:30]!r} | {q}")
    return score, wrong

for mode in ["text", "image"]:
    print(f"--- {mode}: {evaluate(mode)[0]} / {len(QA)}")
```

```python
import time

# 解像度を変えて、画像トークン数・正答数・1 問あたりの時間・VRAM の最大使用量を比べる
for dpi in [50, 72, 100, 150, 200]:
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
    score, wrong = evaluate("image", dpi, verbose=False)
    sec = (time.perf_counter() - t) / len(QA)
    n_image = ask("", image=page_image(pub, dpi), max_new_tokens=1)[1]
    print(f"{dpi:3d} dpi: {page_image(pub, dpi).size} 画素, {n_image:4d} 画像トークン, 正答 {score:2d}/{len(QA)}, "
          f"{sec:.2f} 秒/問, 最大 {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print("     誤答 (正解→出力):", ", ".join(wrong))
```

- 出力の値は float16（T4 と同じ）で実行したもの。bfloat16（Ampere 以降の GPU）でも大部分は同じで、違うところは括弧内に書く。
- テキストを抽出すると、表 2 の 2 段の見出しが「こどもの未就学の乳幼児（０～５歳）| 小学生（６～11歳）| 中学生 | 数 | …」のように崩れ、どの列の値かが分からなくなる。同じモデルにこのテキストだけを渡すと、表 2 の「9～11 歳の男」を 118、「0～5 歳の女」を 104 と答えた（正解は 152、222）。図 1 の棒グラフの値 (213〜309) はテキストに含まれているのに、最も多い年齢階級を「6～8歳」と答えた。一方、自作 PDF の表 1 のような見出しが 1 段の表は、テキストだけでも正しく答えた。失われるのは、入れ子の見出しやグラフのように位置で意味が決まる情報である。
- 値を書き込んでいない折れ線グラフは、テキストには軸の目盛りしか残らない。テキストだけでは 2020 年度の一般書も 2023 年度の児童書も「300」と答えた（正解は 560、520）。画像を渡すと 550 と 520（bfloat16 では 570 と 520）になった。グラフ全体を表にさせると、15 個の値の誤差は 100 dpi で平均 2.5 千冊（bfloat16 では 4.2 千冊）、最大 10 千冊だった。グラフから読んだ値は目盛りの細かさに応じた誤差を含む推定値で、元のデータの代わりにはならない。
- 画像を渡しても誤りはある。表 2 の「9～11 歳の男」には男女計の行の 296 を、「3～5 歳の割合」には 1 列左の 1.7 を答えた。Markdown に変換した表 2 も、2 段目の見出しが 7 列しかなく、9 列のデータとずれている。図 1 は `![図1](https://example.com/image.png)` という実在しない URL への参照に置き換わり、値を表にさせると値が列見出しに入った空の表になった。出力はもっともらしい形をしていても、元のページと突き合わせて確かめる。結合したセルのある表は、セルごとに質問するなどの工夫が要る。
- この例では、「2019 年度より 2023 年度のほうが多い分類」（正解は児童書）に 50、100、200 dpi で「一般書」と答えた。1 つの値を読む質問より間違えやすかった。
- 画像トークン数は画素数にほぼ比例する（約 32×32 画素で 1 トークン）。A4 のページで 50 dpi なら 234、100 dpi なら 962、200 dpi なら 3796 だった。正答は 50 dpi で 4/12、72 dpi で 7/12、100 dpi で 9/12、150 dpi と 200 dpi で 10/12。1 問あたりの時間は 0.20 秒から 1.30 秒に、VRAM の最大使用量は 9.0GB から 10.0GB に増えた（重みだけで約 8.9GB）。T4 (16GB) に収まる。時間は RTX A5000 で測ったもので、T4 では長くなると見込まれる（T4 では測っていない）。このページでは 100〜150 dpi が釣り合いがよい。bfloat16 では 150 dpi の正答が 9/12 だった。数値の精度が違うだけで 1 問変わるので、12 問での 1 問の差は誤差の範囲である。
- スキャンした PDF にはテキスト層がなく、`get_text()` は空文字列を返す（上の画像だけの PDF）。ページを画像にして VLM に渡すか、この章の「画像の中の日本語を読み取り (OCR)、文字誤り率で比べる」の方法を使う。長い文書はページごとに処理する。100 dpi でも 1 ページで約 1000 トークンになる。ページごとの Markdown を RAG の章の「長い文書をチャンクに分割する」で分割すれば、RAG に使える。ページの画像のまま検索する方法は、この章の「PDF のページを画像のまま検索して VLM に答えさせる（マルチモーダル RAG）」を参照。
- Qwen3-VL-4B-Instruct のライセンスは Apache-2.0（モデルカード）。古い transformers では読み込めない（検証時の Colab の transformers 5.22.1 では動いた）。モデルカードはサンプリング（temperature=0.7 など）を推奨しているが、ここでは再現性のために貪欲法 (`do_sample=False`) にした。

## 請求書やレシートの画像から、日付・金額・取引先・明細を JSON で取り出す {document_field_extraction gpu}

請求書やレシートの画像から、取引先・日付・明細・小計・消費税・合計を JSON で取り出して、会計のデータとして扱いたい。正解が分かるように PIL で書類の画像を作り、視覚言語モデル (VLM) に画像から直接 JSON を出させる方法と、OCR (Tesseract) で読んだテキストを LLM で JSON にする方法を、項目ごとの正解率で比べる。明細の合計と小計が合うかのような検算で、誤りをどこまで見つけられるかも確かめる。

```python
# 所要時間の目安: T4 の Colab で 10〜15 分程度と見込む（モデルのダウンロード約 9GB を含む。T4 では未測定で、検証の GPU での時間の 2〜3 倍として見積もった）
# Tesseract 本体（Colab には入っている）と日本語のモデル、日本語フォント (Noto CJK)
!apt-get install -y -q tesseract-ocr tesseract-ocr-jpn fonts-noto-cjk > /dev/null
!pip install -q pytesseract
```

```python
# 正解の分かる請求書とレシートの画像を PIL で作る（会社名・店名・住所・電話番号はすべて架空）
import random
from datetime import date, timedelta
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SANS = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
font = lambda size, bold=False: ImageFont.truetype(BOLD if bold else SANS, size)

VENDORS = {"invoice": ["株式会社クックブック商事", "合同会社ことば工房", "有限会社みなと印刷", "株式会社さくらシステム"],
           "receipt": ["ひまわり文具店", "青空ベーカリー", "やまびこ青果", "喫茶こもれび"]}
CUSTOMERS = ["株式会社テスト物産", "架空市立図書館", "合同会社サンプル企画", "株式会社れい電機"]   # 請求書の宛先（取引先と紛らわしい）
GOODS = {"invoice": [("A4コピー用紙(500枚)", 450, 800), ("ボールペン(黒)10本", 600, 1200), ("名刺印刷(100枚)", 1500, 3500),
                     ("Webサイト保守(月額)", 20000, 50000), ("翻訳料(和文英訳)", 8000, 30000), ("チラシ印刷(1000部)", 9000, 18000),
                     ("ノートPCスタンド", 2500, 6000), ("会議用ホワイトボード", 12000, 25000), ("ソフトウェア利用料", 3000, 9000)],
         "receipt": [("食パン", 220, 480), ("クロワッサン", 160, 260), ("ブレンドコーヒー", 380, 520), ("りんご", 120, 250),
                     ("キャベツ", 150, 300), ("ノート B5", 150, 280), ("消しゴム", 80, 150), ("ホットサンド", 550, 780),
                     ("牛乳 1L", 200, 280), ("付箋", 180, 400)]}

def fmt_date(d, style):   # 日付の書き方を 3 通りにする
    if style == "漢字":
        return f"{d.year}年{d.month}月{d.day}日"
    if style == "スラッシュ":
        return f"{d.year}/{d.month:02d}/{d.day:02d}"
    return f"令和{d.year - 2018}年{d.month}月{d.day}日"          # 和暦: 令和元年 = 2019 年

def make_truth(rng, kind):
    d = date(2025, 1, 1) + timedelta(days=rng.randrange(365))
    items = []
    for name, lo, hi in rng.sample(GOODS[kind], rng.randint(2, 5)):
        qty, unit = rng.randint(1, 3 if kind == "receipt" else 10), rng.randrange(lo, hi, 10)
        items.append({"name": name, "quantity": qty, "unit_price": unit, "amount": qty * unit})
    subtotal = sum(it["amount"] for it in items)
    tax = subtotal // 10                                          # 消費税 10%（1 円未満は切り捨て）
    return {"vendor": rng.choice(VENDORS[kind]), "date": d.isoformat(), "items": items,
            "subtotal": subtotal, "tax": tax, "total": subtotal + tax}

def draw_invoice(t, rng, style):
    W = 1000
    img = Image.new("RGB", (W, 1300), "white"); d = ImageDraw.Draw(img)
    dd = date.fromisoformat(t["date"])
    d.text((W // 2, 60), "請 求 書", font=font(44, True), fill="black", anchor="mt")
    d.text((W - 60, 140), f"請求番号: INV-{rng.randint(1000, 9999)}", font=font(20), fill="black", anchor="ra")
    d.text((W - 60, 170), f"発行日: {fmt_date(dd, style)}", font=font(20), fill="black", anchor="ra")
    d.text((60, 220), f"{rng.choice(CUSTOMERS)} 御中", font=font(28), fill="black")
    d.line([(60, 260), (480, 260)], fill="black", width=2)
    d.text((60, 280), "下記のとおりご請求申し上げます。", font=font(20), fill="black")
    d.text((W - 60, 230), t["vendor"], font=font(24, True), fill="black", anchor="ra")
    d.text((W - 60, 265), f"〒000-{rng.randint(1000, 9999)} 架空県架空市{rng.randint(1, 9)}-{rng.randint(1, 30)}", font=font(18), fill="black", anchor="ra")
    d.text((W - 60, 290), f"TEL 000-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}", font=font(18), fill="black", anchor="ra")
    d.rectangle([60, 340, 560, 400], outline="black", width=2)
    d.text((80, 352), f"ご請求金額  ¥{t['total']:,}-", font=font(30, True), fill="black")
    cols, y = [60, 520, 640, 790, W - 60], 450                    # 明細の表（品名・数量・単価・金額）
    d.rectangle([cols[0], y, cols[-1], y + 40], fill=(220, 220, 220), outline="black")
    for x0, x1, h in zip(cols, cols[1:], ["品名", "数量", "単価", "金額"]):
        d.text(((x0 + x1) // 2, y + 8), h, font=font(20, True), fill="black", anchor="ma")
    for it in t["items"]:
        y += 40
        d.rectangle([cols[0], y, cols[-1], y + 40], outline="black")
        d.text((cols[0] + 10, y + 8), it["name"], font=font(20), fill="black")
        for x1, v in zip(cols[2:], [f"{it['quantity']}", f"{it['unit_price']:,}", f"{it['amount']:,}"]):
            d.text((x1 - 10, y + 8), v, font=font(20), fill="black", anchor="ra")
    for x in cols:
        d.line([(x, 450), (x, y + 40)], fill="black")
    y += 60
    for label, v in [("小計", t["subtotal"]), ("消費税(10%)", t["tax"]), ("合計", t["total"])]:
        d.text((cols[2], y), label, font=font(20, label == "合計"), fill="black")
        d.text((cols[-1] - 10, y), f"{v:,}", font=font(20, label == "合計"), fill="black", anchor="ra")
        y += 34
    d.text((60, y + 30), f"お支払期限: {fmt_date(dd + timedelta(days=30), style)}", font=font(20), fill="black")   # 発行日と紛らわしい日付
    d.text((60, y + 60), f"お振込先: 架空銀行 本店 普通 {rng.randint(1000000, 9999999)}", font=font(20), fill="black")
    return img.crop((0, 0, W, y + 120))

def draw_receipt(t, rng, style):
    W = 420
    img = Image.new("RGB", (W, 540 + len(t["items"]) * 60), "white"); d = ImageDraw.Draw(img)
    y = 30
    d.text((W // 2, y), t["vendor"], font=font(30, True), fill="black", anchor="ma"); y += 50
    d.text((W // 2, y), f"架空県架空市{rng.randint(1, 9)}-{rng.randint(1, 30)}", font=font(16), fill="black", anchor="ma"); y += 24
    d.text((W // 2, y), f"TEL 000-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}", font=font(16), fill="black", anchor="ma"); y += 36
    d.text((24, y), f"{fmt_date(date.fromisoformat(t['date']), style)} {rng.randint(9, 20)}:{rng.randint(0, 59):02d}", font=font(18), fill="black"); y += 30
    d.text((W // 2, y), "領 収 書", font=font(22, True), fill="black", anchor="ma"); y += 40
    for it in t["items"]:
        d.text((24, y), it["name"], font=font(18), fill="black")
        if it["quantity"] > 1:                                    # 2 点以上は「単価 × 数量」の行を足す
            y += 26
            d.text((48, y), f"¥{it['unit_price']:,} × {it['quantity']}点", font=font(16), fill="black")
        d.text((W - 24, y), f"¥{it['amount']:,}", font=font(18), fill="black", anchor="ra"); y += 30
    d.line([(24, y + 4), (W - 24, y + 4)], fill="black"); y += 14
    paid = (t["total"] // 1000 + 1) * 1000
    for label, v, big in [("小計", t["subtotal"], False), ("消費税等(10%)", t["tax"], False), ("合計", t["total"], True),
                          ("お預り", paid, False), ("お釣り", paid - t["total"], False)]:   # お預り・お釣りは合計と紛らわしい金額
        d.text((24, y), label, font=font(22 if big else 18, big), fill="black")
        d.text((W - 24, y), f"¥{v:,}", font=font(22 if big else 18, big), fill="black", anchor="ra"); y += 34 if big else 28
    return img.crop((0, 0, W, y + 30))

def degrade(img, rng):
    """スマートフォンで撮ったレシートを模す: 縮小、傾き、ぼけ、ノイズ"""
    img = img.convert("L").resize((int(img.width * 0.75), int(img.height * 0.75)), Image.BILINEAR)
    img = img.rotate(rng.uniform(-4, 4), expand=True, fillcolor=255, resample=Image.BICUBIC)
    a = np.asarray(img.filter(ImageFilter.GaussianBlur(0.8)), dtype=float)
    a += np.random.default_rng(rng.randrange(10**6)).normal(0, 20, a.shape)
    return Image.fromarray(a.clip(0, 255).astype("uint8")).convert("RGB")

rng = random.Random(0)
docs = []
for kind in ["invoice", "receipt"]:
    for i in range(12):
        t = make_truth(rng, kind)
        style = ["漢字", "スラッシュ", "令和"][i % 3]
        img = draw_invoice(t, rng, style) if kind == "invoice" else draw_receipt(t, rng, style)
        group = "請求書"
        if kind == "receipt":
            group = "レシート (劣化)" if i % 2 else "レシート"
            if i % 2:
                img = degrade(img, rng)
        docs.append({"kind": kind, "group": group, "style": style, "truth": t, "image": img})
print(len(docs), "枚。正解の例:", docs[12]["truth"])
display(docs[0]["image"].resize((500, docs[0]["image"].height // 2)), docs[12]["image"], docs[13]["image"])
```

```python
# VLM: Qwen3-VL-4B-Instruct (Apache-2.0)。画像なしでテキストだけを渡せば LLM としても使えるので、両方の方法で同じモデルを使う
# 初回はモデル（約 9GB）のダウンロードに数分かかる
import time
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

name = "Qwen/Qwen3-VL-4B-Instruct"
BF16 = torch.cuda.get_device_capability()[0] >= 8      # bfloat16 は Ampere (A100, L4) 以降。T4 では float16 を使う
processor = AutoProcessor.from_pretrained(name)
processor.tokenizer.padding_side = "left"              # 生成のバッチ処理では左パディング
model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16 if BF16 else torch.float16, device_map="cuda").eval()

# 出力の形式は例で示す。日付は「書かれたとおり」と「YYYY-MM-DD に直したもの」の両方を出させ、あとで比べる
PROMPT = """{source}から、次の項目を取り出して JSON で出力してください。
- vendor: 書類を発行した会社または店の名前 (宛先ではない)
- date_text: 発行日 (書類に書かれたとおり)
- date: 発行日を YYYY-MM-DD の形式に直したもの
- items: 明細の配列。各要素は name (品名), quantity (数量), unit_price (単価), amount (金額)
- subtotal: 小計, tax: 消費税, total: 合計
金額は円の整数で書いてください。次の形式の JSON だけを出力してください。
{{"vendor": "...", "date_text": "...", "date": "YYYY-MM-DD", "items": [{{"name": "...", "quantity": 1, "unit_price": 100, "amount": 100}}], "subtotal": 100, "tax": 10, "total": 110}}"""

def generate(conversations, max_new_tokens=600, batch_size=8):
    """会話（メッセージのリスト）の一覧をバッチでまとめて生成する"""
    outs = []
    for i in range(0, len(conversations), batch_size):
        inputs = processor.apply_chat_template(conversations[i:i + batch_size], add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt", processor_kwargs={"padding": True}).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        outs += [s.strip() for s in processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)]
    return outs

def ask_image(doc):    # (A) 画像から直接
    return [{"role": "user", "content": [{"type": "image", "image": doc["image"]}, {"type": "text", "text": PROMPT.format(source="この画像の書類")}]}]

# (A) VLM に画像から直接 JSON を出させる（T4 では 3 分前後かかる見込み）
torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
raw = {"VLM (画像→JSON)": generate([ask_image(d) for d in docs])}
print(f"{time.perf_counter() - t:.0f} 秒, 最大メモリ {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
print(raw["VLM (画像→JSON)"][12])
```

```python
# (B) OCR してから LLM で JSON にする。まず Tesseract で読む
import pytesseract

# レイアウトの指定 (--psm) で、表の読み方が変わる。psm 6 は全体を 1 つの文字のブロックとみなし、psm 4 は大きさの違う行が並ぶ 1 列とみなす
for psm in [6, 4]:
    print(f"--- psm {psm}\n" + pytesseract.image_to_string(docs[0]["image"], lang="jpn", config=f"--psm {psm}")[150:520])

t = time.perf_counter()
ocr_texts = [pytesseract.image_to_string(d["image"], lang="jpn", config="--psm 4") for d in docs]
print(f"OCR: 1 枚あたり {(time.perf_counter() - t) / len(docs):.2f} 秒 (CPU)")
print(ocr_texts[13])   # 劣化させたレシート
```

```python
# OCR のテキストを同じモデルに（画像なしで）渡して JSON にする（T4 では 1〜2 分かかる見込み）
def ask_text(text):    # (B) OCR のテキストから
    return [{"role": "user", "content": [{"type": "text", "text": PROMPT.format(source="次の OCR のテキスト (書類の画像から読み取ったもの) ") + "\n\nOCR のテキスト:\n" + text}]}]

t = time.perf_counter()
raw["OCR→LLM"] = generate([ask_text(s) for s in ocr_texts])
print(f"{time.perf_counter() - t:.0f} 秒")
print(raw["OCR→LLM"][12])
```

```python
# 出力を JSON として取り出し、スキーマで検証してから、項目ごとに正解と比べる
# extract_json は大規模言語モデルの章の「LLM の出力から JSON を取り出して検証する」と同じもの
import json, re, unicodedata
import pandas as pd
from pydantic import BaseModel, ValidationError

class Item(BaseModel):
    name: str
    quantity: int
    unit_price: int
    amount: int

class Receipt(BaseModel):
    vendor: str
    date_text: str
    date: str
    items: list[Item]
    subtotal: int
    tax: int
    total: int

def extract_json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find("{")
    if start < 0:
        raise ValueError("JSON が見つかりません")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj

def parse(out, schema=Receipt):
    try:
        return schema.model_validate(extract_json(out)).model_dump()
    except (ValueError, ValidationError):
        return None

def to_iso(s):
    """書かれたとおりの日付（2025年3月4日、2025/03/04、令和7年3月4日）を、和暦の変換も含めてコードで YYYY-MM-DD に直す。
    令和か 4 桁の西暦のときだけ受け付け、それ以外（OCR が「令和」を「信和」と読んだものなど）やありえない日付は None"""
    m = re.search(r"(?:令和\s*(\d+|元)|(\d{4}))\s*[年/.-]\s*(\d{1,2})\s*[月/.-]\s*(\d{1,2})", unicodedata.normalize("NFKC", s))
    if not m:
        return None
    year = int(m[2]) if m[2] else 2018 + (1 if m[1] == "元" else int(m[1]))   # 令和元年 = 2019 年
    try:
        return date(year, int(m[3]), int(m[4])).isoformat()
    except ValueError:
        return None

norm = lambda s: re.sub(r"\s", "", unicodedata.normalize("NFKC", s))   # 全角半角と空白の違いは許す
FIELDS = ["取引先", "日付 (モデルが変換)", "日付 (コードで変換)", "小計", "消費税", "合計", "明細 (すべての行)", "書類全体"]

def score(p, t):
    if p is None:                                  # JSON として読めない・スキーマに合わない
        return dict.fromkeys(FIELDS, False) | {"明細の行": 0.0}
    key = lambda it: (norm(it["name"]), it["quantity"], it["unit_price"], it["amount"])
    pred, gold = sorted(map(key, p["items"])), sorted(map(key, t["items"]))
    s = {"取引先": norm(p["vendor"]) == norm(t["vendor"]), "日付 (モデルが変換)": p["date"] == t["date"],
         "日付 (コードで変換)": to_iso(p["date_text"]) == t["date"],
         "小計": p["subtotal"] == t["subtotal"], "消費税": p["tax"] == t["tax"], "合計": p["total"] == t["total"],
         "明細 (すべての行)": pred == gold, "明細の行": sum(g in pred for g in gold) / len(gold)}   # 正解の行のうち、完全に一致した行の割合
    s["書類全体"] = all(s[k] for k in FIELDS[:-1] if k != "日付 (モデルが変換)")
    return s

parsed = {m: [parse(o) for o in outs] for m, outs in raw.items()}
rows = [{"方法": m, "書類": d["group"], "日付の書き方": d["style"], **score(p, d["truth"])}
        for m, ps in parsed.items() for p, d in zip(ps, docs)]
df = pd.DataFrame(rows)
print("JSON として読めなかった数:", {m: sum(p is None for p in ps) for m, ps in parsed.items()})
cols = FIELDS[:-1] + ["明細の行", "書類全体"]
print(df.groupby("方法", sort=False)[cols].mean().round(2).T.to_string())
print("\n書類の種類ごとの「書類全体」の正解率\n" + df.pivot_table(index="書類", columns="方法", values="書類全体", sort=False).round(2).to_string())
print("\n日付の書き方ごとの正解率 (モデルが変換 / コードで変換)\n" +
      df.groupby(["方法", "日付の書き方"], sort=False)[["日付 (モデルが変換)", "日付 (コードで変換)"]].mean().round(2).to_string())
print("\n日付がどちらかで誤った書類 (書かれたとおりの日付 → コードの変換 / モデルの変換 / 正解)")
for m, ps in parsed.items():
    for i, (p, d) in enumerate(zip(ps, docs)):
        if p is not None and (to_iso(p["date_text"]) != d["truth"]["date"] or p["date"] != d["truth"]["date"]):
            print(f"  {m} #{i} {d['style']}: 「{p['date_text']}」→ {to_iso(p['date_text'])} / {p['date']} / {d['truth']['date']}")
```

```python
# 比較: 書かれたとおりの日付 (date_text) を出させずに、いきなり YYYY-MM-DD を出させる（令和で書いた 8 枚。T4 では 1 分前後かかる見込み）
class ReceiptNoDateText(Receipt):
    date_text: str = ""                      # このプロンプトでは出させないので、なくてもよいことにする

PROMPT_NO_TEXT = PROMPT.replace("- date_text: 発行日 (書類に書かれたとおり)\n", "").replace(' "date_text": "...",', "")
assert "date_text" not in PROMPT_NO_TEXT
reiwa = [i for i, d in enumerate(docs) if d["style"] == "令和"]
outs = generate([[{"role": "user", "content": [{"type": "image", "image": docs[i]["image"]},
                                               {"type": "text", "text": PROMPT_NO_TEXT.format(source="この画像の書類")}]}] for i in reiwa])
get_date = lambda p: p["date"] if p else "(読めない)"
for i, o in zip(reiwa, outs):
    print(f"#{i} {docs[i]['group']}: 正解 {docs[i]['truth']['date']} → date_text なし {get_date(parse(o, ReceiptNoDateText))}"
          f" / あり {get_date(parsed['VLM (画像→JSON)'][i])}")
```

```python
# 検算: 数量×単価=金額、数量・金額が正、明細の合計=小計、小計の 10% = 消費税（端数の扱いの違いで ±1 円は許す）、小計+消費税=合計、
# 日付をコードで読めるか（令和か 4 桁の西暦か）、年が 2000 年から今年までか
def check(p):
    if p is None:
        return ["JSON として読めない"]
    errors = [f"「{it['name']}」の数量×単価 ({it['quantity']}×{it['unit_price']}) が金額 {it['amount']} と合わない"
              for it in p["items"] if it["quantity"] * it["unit_price"] != it["amount"]]
    if not p["items"] or any(it["quantity"] <= 0 or it["amount"] <= 0 for it in p["items"]):
        errors.append("明細がない、または数量・金額が 0 以下の行がある")
    if sum(it["amount"] for it in p["items"]) != p["subtotal"]:
        errors.append(f"明細の金額の合計 {sum(it['amount'] for it in p['items'])} が小計 {p['subtotal']} と合わない")
    if abs(p["subtotal"] * 0.1 - p["tax"]) > 1:
        errors.append(f"消費税 {p['tax']} が小計 {p['subtotal']} の 10% と合わない")
    if p["subtotal"] + p["tax"] != p["total"]:
        errors.append(f"小計+消費税 ({p['subtotal']}+{p['tax']}) が合計 {p['total']} と合わない")
    iso = to_iso(p["date_text"])
    if iso is None:
        errors.append(f"日付「{p['date_text']}」を読めない")
    elif not 2000 <= int(iso[:4]) <= date.today().year:
        errors.append(f"日付「{p['date_text']}」の年がありえない")
    return errors

df["検算"] = ["NG" if check(p) else "OK" for ps in parsed.values() for p in ps]
df["正誤"] = df["書類全体"].map({True: "全項目正しい", False: "誤りあり"})
print(pd.crosstab([df["方法"], df["検算"]], df["正誤"]).to_string())

# 検算を通ったのに誤りがあった書類: 何を間違えたか
for m, ps in parsed.items():
    for i, (p, d) in enumerate(zip(ps, docs)):
        s = score(p, d["truth"])
        if p is not None and not check(p) and not s["書類全体"]:
            wrong = [k for k in FIELDS[:-1] if k != "日付 (モデルが変換)" and not s[k]]
            print(f"{m} #{i} {d['group']}: {', '.join(wrong)} | 出力 {p['vendor']} {p['date_text']} "
                  f"{[(it['name'], it['quantity'], it['unit_price'], it['amount']) for it in p['items']]}")
```

```python
# 検算で不整合が見つかった書類だけ、どこが合わないかを伝えて読み直させる（T4 では 1〜2 分かかる見込み）
def show(q):   # 明細の（数量, 単価, 金額）と、小計・消費税・合計
    return [(it["quantity"], it["unit_price"], it["amount"]) for it in q["items"]] + [q["subtotal"], q["tax"], q["total"]]

retry = {}
for m, ps in parsed.items():
    idx = [i for i, p in enumerate(ps) if check(p)]
    first = [ask_image(docs[i]) if m.startswith("VLM") else ask_text(ocr_texts[i]) for i in idx]
    convs = [c + [{"role": "assistant", "content": raw[m][i]},
                  {"role": "user", "content": "検算で次の不整合が見つかりました。\n" + "\n".join(check(parsed[m][i])) +
                   "\n書類に書かれている数字をもう一度確かめて、JSON だけを出力し直してください。"}] for c, i in zip(first, idx)]
    retry[m] = dict(zip(idx, map(parse, generate(convs))))
    for i, p in retry[m].items():
        before, after = score(parsed[m][i], docs[i]["truth"]), score(p, docs[i]["truth"])
        print(f"{m} #{i} {docs[i]['group']}: 検算 {'OK' if not check(p) else 'NG'}, "
              f"書類全体 {before['書類全体']} → {after['書類全体']}, 明細の行 {before['明細の行']:.2f} → {after['明細の行']:.2f}")
        if m.startswith("VLM") and p is not None:   # 何が変わったか（正解、1 回目、読み直し後の明細）
            print(f"    正解 {show(docs[i]['truth'])}\n    1 回目 {show(parsed[m][i])}\n    読み直し {show(p)}")

# 読み直したあとの、検算の結果と正誤の対応
df["検算 (読み直し後)"] = ["NG" if check(retry[m].get(i, p)) else "OK" for m, ps in parsed.items() for i, p in enumerate(ps)]
df["正誤 (読み直し後)"] = [("全項目正しい" if score(retry[m].get(i, p), d["truth"])["書類全体"] else "誤りあり")
                         for m, ps in parsed.items() for i, (p, d) in enumerate(zip(ps, docs))]
print(pd.crosstab([df["方法"], df["検算 (読み直し後)"]], df["正誤 (読み直し後)"]).to_string())
```

- 数値はすべて float16（`BF16 = False`、T4 と同じ dtype）で実行したもの。書類は 24 枚（請求書 12、レシート 12 のうち 6 枚は縮小・傾き・ぼけ・ノイズで劣化させたもの）で、1 枚で正解率が 0.04（劣化させたレシート 6 枚だけの正解率では 0.17）動く。1 種類のフォントで描いた架空の書式の合成画像なので、手書き、かすれた感熱紙、多様なレイアウト、軽減税率 (8%) の混在や内税は試していない。対象と同じ種類の書類に正解を付けて、同じ方法で測ってから使う。
- 書類全体（日付はコードで変換したもの）がすべて正しかった割合は、VLM に画像から直接出させると 0.83、OCR→LLM では 0.54 だった（明細の行の一致率は 0.92 と 0.71、合計は 0.96 と 0.75）。同じモデルを使っているので、差は入力（画像か OCR のテキストか）の違いによる。Tesseract は `--psm 6` だと表の数値を「ie| oueel」のように崩し、`--psm 4` では行ごとに読めた。劣化させたレシートは、上の例のように「小計」「合計」のラベルだけが残って数字が読めないことがあり、OCR→LLM は 6 枚とも誤った。ほかに、請求書の宛先（株式会社テスト物産）を取引先と答えた。VLM も劣化させたレシートでは 6 枚中 2 枚しか全項目が正しくなく、「¥470 × 2点 ¥940」の単価に金額の 940 を入れる、ぼけた 190 を 170 と読む、「消しゴム」を「満しゴム」と読む誤りがあった。
- 日付は、書かれたとおりの文字列 (`date_text`) と YYYY-MM-DD の両方を出させた。`date_text` を出させずに YYYY-MM-DD だけを出させると、この例では、令和で書いたレシート 4 枚はすべて年を 2023 年とした（請求書の 4 枚は正しかった）。`date_text` も出させると 8 枚とも正しかった。8 枚での比較なので傾向の目安だが、書かれたとおりの値を残しておけば、和暦の変換のような決まった処理はコード (`to_iso`) で行い、モデルの変換と照合できる。`to_iso` は令和か 4 桁の西暦だけを受け付けるので、OCR→LLM で OCR が「令和」を「信和」と読んだ 2 枚（#2、#17）と、日付の区切りが消えて「202570723」となった 1 枚 (#13) では変換できなかった。モデルは #17 と #13 を推測で正しく直したので、OCR→LLM の日付の正解率はモデルの変換 0.83 に対してコードの変換 0.75 だった（#2 はモデルも 2024 年と誤った）。推測で直した値は確かめようがないので、変換できない書類は人が見るほうが安全である。
- 検算（数量×単価=金額、数量・金額が正、明細の合計=小計、消費税=小計の 10%、小計+消費税=合計、日付をコードで読めて年が 2000 年から今年まで）で NG になった書類は、VLM の 3 枚、OCR→LLM の 9 枚とも実際に誤りがあった（「信和」の 2 枚は日付の確認で NG になった）。一方、検算を通っても誤りのある書類が VLM で 1 枚（「満しゴム」）、OCR→LLM で 2 枚（宛先を取引先とした #3、品名を「ボールペン（上黒）」とした #6）あった。品名・取引先の誤りは検算では分からず、日付も形式と年の範囲しか見ないので、日の誤読（OCR→LLM の #23 は 3 月 24 日を 3 月 4 日とした）は通る。読み落とした行の数字どうしがたまたま合う場合も通る。検算は「NG なら確実に人が見る」ための仕組みで、「OK なら正しい」とは言えない。消費税の端数の扱い（切り捨て・四捨五入）は事業者で違うので ±1 円を許した。
- 検算で NG になった書類に、合わない箇所を伝えて読み直させると、VLM は 3 枚中 1 枚（単価に金額を入れていたもの）が直った。残りの 2 枚では、誤読した単価 170 をそのままにして正しかった消費税 209 を 203 に書き換えて検算を合わせたり、金額を数量倍して書類にない合計 9900 を作ったりした。OCR→LLM は 9 枚とも直らず、3 枚が検算に合う数字に書き換えて通ったので、検算を通るのに誤りがある書類が 2 枚から 5 枚に増えた（VLM でも 1 枚から 2 枚に増えた）。テキストにない数字は読み直しても取り戻せず、検算に合う数字を作ってしまう。読み直しは試してよいが、検算を通ったことを正しさの根拠にせず、NG の書類は人が確かめる流れにする。
- ここでは出力の形式を例で示し、`extract_json` と pydantic で検証した（大規模言語モデルの章の「LLM の出力から JSON を取り出して検証する」）。最初の 48 回の出力で JSON として読めなかったものはなかったが、形式を保証したいなら制約付きデコードや API の構造化出力を使う（LLM の API の章の「Gemini に決まった形式（JSON スキーマ）で出力させる」「Claude に決まった形式（JSON スキーマ）で出力させる」）。形式が保証されても、中の値が正しいとは限らないのは上のとおりである。実際の請求書やレシートには取引先や個人の情報が含まれるので、外部の API に送るかどうかは扱いの規程を確かめる（この手元のモデルなら外に出ない）。
- 時間とメモリ（検証の GPU で float16 で実行した値。T4 では測っていない）: VLM で 24 枚を 8 枚ずつのバッチで処理するのに 60 秒、OCR のテキストからの JSON 化に 82 秒（共有の GPU なので実行ごとに、前者は 53〜88 秒、後者は 38〜82 秒と変わった）。T4 ではこの 2〜3 倍かかると見込む。GPU メモリの最大使用量は 11.4GB（`torch.cuda.max_memory_allocated` の値で、CUDA の管理領域などは含まない）なので、T4 (16GB) に収まる見込みである。Tesseract は 1 枚 0.32 秒（検証の CPU での値。Colab の CPU ではこれより遅くなりうる）。Qwen3-VL-4B-Instruct は Apache-2.0、Tesseract と日本語のモデル (tesseract-ocr-jpn) は Apache-2.0、Noto CJK フォントは SIL Open Font License 1.1。画像はすべてこのコードで作った架空のもの。

## PDF のページを画像のまま検索して VLM に答えさせる（マルチモーダル RAG） {multimodal_rag gpu}

表やグラフの多い PDF について RAG をしたい。テキストを抽出して検索する方法と、ページの画像をそのまま埋め込んで検索する方法を比べ、検索したページの画像を視覚言語モデル (VLM) に読ませて出典付きで答えさせたい。

```python
# 所要時間の目安: T4 の Colab で 10〜15 分 (大半はモデルのダウンロード。ColQwen2 と Qwen3-VL で計約 13.5GB)
!pip install -q pymupdf rank_bm25 fugashi unidic-lite
```

```python
import re, urllib.request
import numpy as np
import pandas as pd
import pymupdf
from PIL import Image

# 総務省統計局「家計調査報告（家計収支編）2025年（令和７年）平均結果の概要」(24 ページ、表とグラフが多い)
# 公共データ利用規約 (PDL1.0、CC BY 4.0 互換)。出典: 総務省統計局ホームページ https://www.stat.go.jp/data/kakei/sokuhou/tsuki/index.html
urllib.request.urlretrieve("https://www.stat.go.jp/data/kakei/sokuhou/tsuki/pdf/fies_gaikyo2025.pdf", "kakei2025.pdf")
doc = pymupdf.open("kakei2025.pdf")
texts = [page.get_text() for page in doc]                                   # (A) 用: ページごとのテキスト (データの収集と入出力の章の「PDF からテキストを抽出する」)
pages = []                                                                  # (B) 用: ページごとの画像
for page in doc:
    pix = page.get_pixmap(dpi=200)
    pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
print(len(doc), "ページ", pages[0].size, "| 文字数:", [len(t) for t in texts])
print("9 ページ目 (棒グラフ) から抽出したテキスト:", " ".join(texts[8].split())[:160])
```

```python
# 評価セット: 質問と、答えが載っているページ番号 (PDF の 1 始まりのページ番号。印刷されたページ番号 + 3)
# 「本文」は文章に答えが書いてある質問、「図表」は表やグラフを見ないと答えられない質問
eval_set = [
    ("本文", "2025年のエンゲル係数は何％でしたか。", [7]),
    ("本文", "3月に消費支出が増加したのはなぜですか。", [5]),
    ("本文", "大阪・関西万博の来場者数は約何人でしたか。", [6]),
    ("本文", "勤労者世帯の平均消費性向は何％で、前年から何ポイント変わりましたか。", [14]),
    ("本文", "単身世帯の消費支出は1か月平均いくらでしたか。", [17, 18]),
    ("本文", "65歳以上の夫婦のみの無職世帯の平均消費性向は何％でしたか。", [20, 22]),
    ("本文", "勤労者世帯の可処分所得は前年に比べて実質何％変化しましたか。", [12]),
    ("本文", "2020年に消費支出が大きく減少したのはなぜですか。", [4]),
    ("図表", "消費支出の実質増減率への寄与度が最も大きかった費目はどれで、寄与度はいくつでしたか。", [9]),
    ("図表", "電気代の実質増減率は何％でしたか。", [8]),
    ("図表", "勤労者世帯で消費支出額が最も多かった世帯主の年齢階級と、その金額は？", [13]),
    ("図表", "世帯主が60歳以上の勤労者世帯の割合は、2006年に何％でしたか。", [16]),
    ("図表", "勤労者世帯の黒字額は2018年にいくらでしたか。", [15]),
    ("図表", "勤労者世帯の消費支出に占める教養娯楽の割合は何％でしたか。", [15]),
    ("図表", "65歳以上の単身無職世帯の消費支出に占める食料の割合は何％でしたか。", [21, 22]),
    ("図表", "2025年に消費支出の名目増減率が最も高かったのは何月でしたか。", [5]),
    ("図表", "二人以上の世帯に占める無職世帯の割合は2025年に何％でしたか。", [16]),
]
questions = [q for _, q, _ in eval_set]

def gold_rank(ranking, gold):
    """ranking: ページ番号 (0 始まり) を良い順に並べたもの。正解ページのうち最上位の順位を返す"""
    return min(list(ranking).index(g - 1) + 1 for g in gold)

def evaluate(rankings):
    rank = np.array([gold_rank(r, g) for r, (_, _, g) in zip(rankings, eval_set)])
    return {"Recall@1": (rank <= 1).mean(), "Recall@3": (rank <= 3).mean(), "Recall@5": (rank <= 5).mean(), "MRR": (1 / rank).mean()}
```

```python
# (A) テキストの検索: BM25 と、E5 の埋め込み (ページを 400 字のチャンクに分け、最もよく合うチャンクの類似度をページの得点にする)
from fugashi import Tagger
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

tagger = Tagger()
tok = lambda s: [w.surface.lower() for w in tagger(s) if w.feature.pos1 not in ("助詞", "助動詞", "補助記号", "空白")]
bm25 = BM25Okapi([tok(t) for t in texts])
rank_bm25 = [np.argsort(-bm25.get_scores(tok(q))) for q in questions]

chunks, owner = [], []
for i, t in enumerate(texts):
    t = " ".join(t.split())
    for s in range(0, len(t), 300):            # 100 字ずつ重ねる
        chunks.append(t[s:s + 400]); owner.append(i)
owner = np.array(owner)
e5 = SentenceTransformer("intfloat/multilingual-e5-small")
sims = e5.encode(["query: " + q for q in questions], normalize_embeddings=True) @ \
       e5.encode(["passage: " + c for c in chunks], normalize_embeddings=True).T
page_sims = np.stack([sims[:, owner == i].max(axis=1) for i in range(len(texts))], axis=1)
rank_e5 = [np.argsort(-s) for s in page_sims]
pd.DataFrame({"BM25": evaluate(rank_bm25), "E5 (チャンク)": evaluate(rank_e5)}).round(3).T
```

```python
# (B) ページ画像の検索: ColQwen2 (Qwen2-VL-2B に ColBERT 式の late interaction を組み合わせたもの)
import torch
from transformers import ColQwen2ForRetrieval, ColQwen2Processor

BF16 = torch.cuda.get_device_capability()[0] >= 8      # T4 では float16
dtype = torch.bfloat16 if BF16 else torch.float16
torch.cuda.reset_peak_memory_stats()
name = "vidore/colqwen2-v1.0-hf"
retriever = ColQwen2ForRetrieval.from_pretrained(name, dtype=dtype, device_map="cuda").eval()   # 重みは約 4.4GB (Apache-2.0)
processor = ColQwen2Processor.from_pretrained(name)

def embed(batch):
    """ページ画像 (PIL) か質問 (str) のリストを、それぞれ (トークン数, 128) の行列のリストにする"""
    x = processor(images=batch) if isinstance(batch[0], Image.Image) else processor(text=batch)
    with torch.no_grad():
        out = retriever(**x.to(retriever.device)).embeddings
    return [e[m.bool()] for e, m in zip(out, x["attention_mask"])]     # パディングを除く

def maxsim(q, d):
    """質問の各トークンについて、ページの全トークンとの内積の最大値をとり、それを足し合わせる"""
    return (q @ d.T).max(dim=1).values.sum().item()

query_emb = embed(questions)
rankings = {}
for max_tokens in [768, 1536]:                       # 1 ページあたりの画像トークン数の上限 (既定は 768)
    processor.image_processor.size = {"longest_edge": max_tokens * 28 * 28, "shortest_edge": 56 * 56}
    page_emb = [e for i in range(0, len(pages), 2) for e in embed(pages[i:i + 2])]
    scores = np.array([[maxsim(q, d) for d in page_emb] for q in query_emb])
    rankings[max_tokens] = [np.argsort(-s) for s in scores]
    n_vec = sum(len(e) for e in page_emb)
    print(f"上限 {max_tokens}: 索引 {len(page_emb)} ページ × 平均 {n_vec / len(page_emb):.0f} ベクトル × 128 次元 = {n_vec * 128 * 2 / 1e6:.1f} MB (16 ビット)")
print(f"(参考) E5 の索引: {len(chunks)} チャンク × 384 次元 = {len(chunks) * 384 * 2 / 1e6:.2f} MB")
print(f"VRAM の最大使用量: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
del retriever; torch.cuda.empty_cache()              # VLM を載せる前に検索モデルを解放する (質問はすべて検索済み)

def rrf(rankings, k=60):                              # RAG の章の「検索の評価セットを作って手法を比べる」と同じ順位の融合
    scores = np.zeros(rankings[0].shape)
    for r in rankings:
        scores[np.arange(len(r))[:, None], r] += 1 / (k + np.arange(1, r.shape[1] + 1))
    return np.argsort(-scores, axis=1)

rank_img = rankings[1536]
pd.DataFrame({"BM25": evaluate(rank_bm25), "E5 (チャンク)": evaluate(rank_e5),
              "ColQwen2 (768 トークン)": evaluate(rankings[768]), "ColQwen2 (1536 トークン)": evaluate(rank_img),
              "BM25 + ColQwen2 (RRF)": evaluate(rrf([np.array(rank_bm25), np.array(rank_img)]))}).round(3).T
```

```python
# 質問ごとの正解ページの順位 (1 が最良)。本文の質問と図表の質問で分けて見る
per_q = pd.DataFrame({"種類": [k for k, _, _ in eval_set], "質問": [q[:24] for q in questions],
                      **{n: [gold_rank(r, g) for r, (_, _, g) in zip(rs, eval_set)]
                         for n, rs in [("BM25", rank_bm25), ("E5", rank_e5), ("ColQwen2", rank_img)]}})
print(per_q.groupby("種類")[["BM25", "E5", "ColQwen2"]].agg(lambda r: (r == 1).mean()).round(2).rename_axis("Recall@1"))
per_q
```

```python
# 生成: 検索した上位 2 ページと質問を Qwen3-VL に渡し、資料の番号で出典を付けて答えさせる
# (B) はページの画像を、(A) は同じ VLM に BM25 で検索したページの抽出テキストを渡す
from transformers import AutoModelForImageTextToText, AutoProcessor

torch.cuda.reset_peak_memory_stats()
vlm_name = "Qwen/Qwen3-VL-4B-Instruct"
vlm = AutoModelForImageTextToText.from_pretrained(vlm_name, dtype=dtype, device_map="cuda").eval()   # 重みは約 8.9GB (Apache-2.0)
vlm_proc = AutoProcessor.from_pretrained(vlm_name)
vlm_proc.image_processor.size = {"longest_edge": 2048 * 32 * 32, "shortest_edge": 256 * 256}   # 1 ページを最大 2048 トークンに抑える

INSTRUCTION = ("上の資料だけに基づいて、質問に日本語で簡潔に答えてください。根拠にした資料の番号を文末に [1] のように付けてください。"
               "資料に答えがなければ「わかりません」と答えてください。\n質問: ")

def answer(qi, use_image=True, k=2):
    hits = [int(p) for p in (rank_img if use_image else rank_bm25)[qi][:k]]
    content = []
    for n, p in enumerate(hits, 1):
        if use_image:
            content += [{"type": "text", "text": f"資料 [{n}]:"}, {"type": "image", "image": pages[p]}]
        else:
            content += [{"type": "text", "text": f"資料 [{n}]: {' '.join(texts[p].split())}\n"}]
    content.append({"type": "text", "text": INSTRUCTION + questions[qi]})
    inputs = vlm_proc.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors="pt").to(vlm.device)
    with torch.no_grad():
        out = vlm.generate(**inputs, max_new_tokens=128, do_sample=False)
    text = vlm_proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    cited = [hits[int(n) - 1] + 1 for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= k]   # 資料の番号 → PDF のページ番号
    return " ".join(text.split()), [p + 1 for p in hits], cited

for qi in [0, 3, 8, 10, 12, 15]:
    kind, q, gold = eval_set[qi]
    print(f"[{kind}] Q: {q} (正解のページ: {gold})")
    for label, use_image in [("(A) テキスト", False), ("(B) 画像", True)]:
        text, hits, cited = answer(qi, use_image)
        mark = "出典○" if cited and set(cited) <= set(gold) else "出典×"     # 出典が正解のページかだけを見る (答えの正しさは見ていない)
        print(f"  {label}  検索: p.{hits}  出典: p.{cited} {mark} | {text[:150]}")
print(f"VRAM の最大使用量: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

- 画像のまま検索する方法（ColPali 系）は、ページの画像を VLM に通し、画像のパッチごとのベクトル（ここでは 1 ページ約 1500 個 × 128 次元）を索引にする。質問もトークンごとのベクトルにし、質問の各トークンについてページ内で最も近いベクトルとの内積をとって足し合わせる（MaxSim、late interaction）。文全体を 1 本のベクトルにする E5 より細かく照合できるが、索引はこの PDF でも 9.1MB と E5 (0.08MB) の 100 倍以上になる。1 ページの画像トークン数の上限を既定の 768 から 1536 にすると、Recall@1 は 0.588 から 0.765 に上がった（BM25 は 0.706）。日本語の小さな文字や表の数字を読むには解像度が要る（この章の「PDF のページを画像にして視覚言語モデルで読む（表とグラフ）」も参照）。注意書きの数値は、特に断らない限り T4 と同じ float16 で動かしたときのもの。
- この PDF はテキスト層があるので、テキストの検索も強い。BM25 は Recall@3 が 1.0 で、本文に答えがある質問の Recall@1 は 3 つの方法とも 0.88 で同じだった。差が出たのは図表の質問で、Recall@1 は BM25 が 0.56、E5 が 0.44、ColQwen2 が 0.67。表の項目名は「交 通 ・ 通 信」のように 1 字ずつ区切られて抽出され、グラフの数値は位置の情報なしに並ぶ。BM25 と ColQwen2 を RRF で融合しても ColQwen2 単独とほぼ同じで（MRR は float16 で 0.817、bfloat16 で 0.852。ColQwen2 単独は 0.831 と 0.841）、上回るとは言えない。float16 では融合後の Recall@1 が 0.706 に下がった。スキャンした PDF ならテキストは抽出できないので、画像の検索しか使えない。
- 生成でも差が出た。同じ Qwen3-VL に (A) 抽出したテキストを渡すと、6 問中はっきり正しく答えたのは平均消費性向の 1 問だけだった。エンゲル係数では答え (28.6%) のあるページを渡したのに「資料[1]に記載されていません」と答え、寄与度のグラフでは「教育、3.7%」と誤り、名目増減率では「12 月」と「5 月」を併記した。(B) ページの画像を渡すと、6 問中 4 問（エンゲル係数、平均消費性向、寄与度の「交通・通信 0.93」、名目増減率が最も高い「5 月 8.9%」）に正しく答えた。
- 画像を渡しても間違える。「2018 年の黒字額」は、正しいページを渡したのに、積み上げ棒グラフの同じ年の「その他」の値（-23.6 千円）を答えた（正しくは 139.8 千円。出典は正しいので「出典○」になる）。「消費支出額が最も多い年齢階級」は、検索で正解のページ（13 ページ）が 5 位になり、無職世帯のページ（20 ページ）からもっともらしい答えを作った。答えだけを見ても検索の失敗とは気づけないので、RAG の章の「RAG の失敗を検索と生成に切り分けて分析する」のように、検索と生成を分けて評価する。
- 出典は、ページの中に印刷されたページ番号（この PDF では PDF のページ番号と 3 ずれる）と紛れないように、資料の番号 ([1], [2]) で付けさせ、コードでページ番号に戻している。それでも、正しく答えたのに別の資料の番号を付けることがある（寄与度と名目増減率の質問）。RAG の章の「回答に出典を付けさせ、その出典を検証する」のように出典を検証する。
- 検索の単位はページ。1 ページに図が 2 つ以上ある（15、16 ページ）と、VLM は関係のない図も読むことになる。ページ画像は 1 枚で最大 2048 トークンを使うので、VLM に渡せるページは数枚まで。
- VRAM の最大使用量は ColQwen2 が 5.6GB、Qwen3-VL-4B が 10.6GB（2 ページ、1 ページ最大 2048 トークン）。ColQwen2 の重み (4.4GB) を残したまま VLM を動かすと約 15GB で、T4 で使える量にほぼ達する。そこで質問の検索を先にすべて済ませ、検索モデルを解放してから VLM を読み込んでいる。
- ColQwen2（vidore/colqwen2-v1.0-hf、Apache-2.0）は Qwen2-VL-2B-Instruct をもとにしたモデルで、モデルカードによると検索のための学習データは英語だけ（もとの Qwen2-VL の事前学習には多言語のデータが含まれる）。この `-hf` 版は transformers だけで動く。データは総務省統計局「家計調査報告（家計収支編）2025年（令和７年）平均結果の概要」（公共データ利用規約 第1.0版、CC BY 4.0 互換）。評価セットの 17 問と正解のページは筆者が PDF を読んで作ったもので、1 問で Recall が 0.06 動く。
