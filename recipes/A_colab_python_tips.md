# Colab と Python の小技

Colab で NLP の実験をするときに、毎回調べることになる定番の操作と、テキストのファイルを扱うときの Python の小技（JSON Lines、CSV / TSV、メモリに載らない大きなファイル）。NLP パイプラインのモジュール分割・テストや高速化・ボトルネック計測については、姉妹編『[自然言語処理の教科書](https://github.com/mamorlis/nlpbook)』の第 5 章（5-5 ツールキットの作成）も参照。

## Google Drive をマウントしてファイルを読み書きする {colab_drive norun}

Colab のランタイムは切断されるとファイルが消える。データやモデルを Google Drive に保存して、次回も使いたい。

```python
from google.colab import drive

drive.mount("/content/drive")   # 初回は認証ダイアログが出る
```

```python
from pathlib import Path

base = Path("/content/drive/MyDrive/Colab Notebooks/nlpcookbook/data")
base.mkdir(parents=True, exist_ok=True)

(base / "memo.txt").write_text("Drive に保存したメモ\n", encoding="utf-8")
print((base / "memo.txt").read_text(encoding="utf-8"))
print([p.name for p in base.iterdir()])
```

```python
# 大きなファイルは、いったんローカル (/content) にコピーしてから使うと速い
!cp "/content/drive/MyDrive/Colab Notebooks/nlpcookbook/data/memo.txt" /content/
!ls -la /content/memo.txt
```

- マイドライブは `/content/drive/MyDrive`、共有ドライブは `/content/drive/Shareddrives/ドライブ名`。
- Drive 上のファイルの読み書きは遅い。小さなファイルが大量にあるデータセットは、zip にまとめて置き、ローカルに展開して使う。
- 書き込みはすぐには反映されないことがある。ランタイムを切る前に `drive.flush_and_unmount()` を呼ぶと確実。

## ファイルをアップロード・ダウンロードする {colab_files norun}

手元の PC にあるデータを Colab に上げたい。実行結果のファイルを PC に落としたい。

```python
from google.colab import files

uploaded = files.upload()          # ファイル選択ダイアログが出る。複数選択可
for name, data in uploaded.items():
    print(name, len(data), "バイト")   # カレントディレクトリ (/content) にも保存されている
```

```python
import pandas as pd

df = pd.DataFrame({"text": ["こんにちは", "さようなら"], "label": [1, 0]})
df.to_csv("result.csv", index=False, encoding="utf-8-sig")
files.download("result.csv")       # ブラウザのダウンロードが始まる
```

```python
# フォルダごと落とすなら zip にまとめる
!zip -qr outputs.zip /content/sample_data
files.download("outputs.zip")

# URL が分かっているファイルは、アップロードせずに直接取得する
!wget -q https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip -O neko.zip
!unzip -o -q neko.zip && ls -la *.txt
```

- 左サイドバーのフォルダアイコンから、ドラッグ & ドロップでもアップロードできる。
- アップロードしたファイルはランタイムが切れると消える。繰り返し使うデータは Drive か、Hugging Face Hub / GitHub に置く。
- 数百 MB を超えるファイルはブラウザ経由だと遅く不安定。Drive に置いてマウントするほうが確実。

## API キーをシークレット機能で安全に扱う {colab_secrets norun}

API キーやアクセストークンをノートブックに直接書かずに使いたい。ノートブックを共有しても漏れないようにしたい。

```python
from google.colab import userdata

# 左サイドバーの 🔑 アイコンから名前と値を登録し、「ノートブックからのアクセス」をオンにしておく
api_key = userdata.get("ANTHROPIC_API_KEY")
print("取得できた:", bool(api_key), "/ 先頭:", api_key[:7] + "...")   # 値そのものは絶対に表示しない
```

```python
import os

# ライブラリが環境変数から読む場合は、環境変数に入れておく
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")     # Hugging Face (利用申請が必要なモデルのダウンロード、Hub へのアップロード)

from huggingface_hub import whoami
print(whoami()["name"])
```

```python
# 登録されていない場合のエラー処理
try:
    userdata.get("NO_SUCH_KEY")
except userdata.SecretNotFoundError:
    print("シークレットが登録されていません")
except userdata.NotebookAccessError:
    print("このノートブックからのアクセスが許可されていません")
```

- シークレットは Google アカウントに紐づいて保存され、ノートブックのファイルには含まれない。共有相手には見えない。
- キーを `print` した出力はノートブックに保存される。出力に残っていないか、共有や GitHub への保存の前に確認する。
- うっかり公開してしまったキーは、すぐに発行元で無効化して作り直す。

## GPU が使えるか確認し、メモリを解放する {colab_gpu gpu norun}

GPU ランタイムになっているか、メモリがどれくらい空いているかを確認したい。`CUDA out of memory` から復帰したい。

```python
import torch

print("PyTorch:", torch.__version__)
print("GPU が使える:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"空き {free / 1e9:.1f} GB / 全体 {total / 1e9:.1f} GB")
device = "cuda" if torch.cuda.is_available() else "cpu"   # GPU がなくても動くコードにしておく
```

```python
import gc

x = torch.zeros(256, 1024, 1024, device=device)   # 約 1GB を確保
print(f"確保後: {torch.cuda.memory_allocated() / 1e9:.2f} GB" if device == "cuda" else "CPU で実行中")

del x                      # 1. 参照を消す (モデルやテンソルを指す変数をすべて)
gc.collect()               # 2. Python のガベージコレクションを走らせる
torch.cuda.empty_cache()   # 3. PyTorch がキャッシュしている GPU メモリを返す
print(f"解放後: {torch.cuda.memory_allocated() / 1e9:.2f} GB" if device == "cuda" else "(GPU なし)")
```

```python
!nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv 2>/dev/null || echo "GPU なし (ランタイム → ランタイムのタイプを変更 → T4 GPU)"
```

- ノートブックでは、直前の出力 (`_`) や例外のトレースバックがテンソルへの参照を握っていて解放されないことがある。どうしても空かなければ、ランタイムを再起動するのが早い。
- メモリ不足のときの対処: バッチサイズを下げる → 系列長 (`max_length`) を下げる → 半精度 (`fp16` / `dtype=torch.float16`) → 勾配累積 → 量子化、の順に試す。
- 推論では `torch.inference_mode()` を忘れない。勾配用の中間結果を保持してメモリを浪費する。

## ライブラリのバージョンを確認・固定する {pip_versions}

Colab のプリインストールライブラリは予告なく更新される。動いていたコードが動かなくなったときに、バージョンを確認し、固定したい。

```python
import importlib.metadata as md
import sys

print("Python", sys.version.split()[0])
for pkg in ["torch", "transformers", "datasets", "tokenizers", "sentence-transformers", "scikit-learn", "numpy", "pandas", "spacy", "nltk"]:
    try:
        print(f"{pkg:24}{md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg:24}(未インストール)")
```

```python
# 実験したときの環境を丸ごと記録しておく（論文の再現や、後日の自分のため）
!pip freeze > requirements-lock.txt
!wc -l requirements-lock.txt && grep -i -E "^(torch|transformers|numpy)==" requirements-lock.txt
```

```python
# バージョンを指定して入れる例（実際に入れ替える場合はコメントを外す）
# !pip install -q "transformers==5.22.1" "tokenizers>=0.23,<0.24"
# !pip install -q -r requirements-lock.txt

# コードの中でバージョンによって分岐する
from packaging.version import Version
import transformers

print("v5 以降:", Version(transformers.__version__) >= Version("5.0.0"))
```

- 出力のバージョンは、検証したときの Colab のもの。Colab の更新で変わるので、自分の環境の出力と違っていてよい。
- すでに import 済みのライブラリを `pip install` で入れ替えた場合は、ランタイムの再起動（ランタイム → セッションを再起動）が必要。
- NumPy や PyTorch のような土台のライブラリを入れ替えると、他のプリインストールライブラリが壊れやすい。必要最小限にとどめる。
- 「前は動いたのに」というときは、まずエラーメッセージとライブラリのバージョンの変更履歴（リリースノート、移行ガイド）を確認する。

## 乱数シードを固定して実験を再現できるようにする {set_seed}

実行するたびに結果が変わってしまう。同じ結果を再現できるように、乱数を固定したい。

```python
import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

for trial in range(2):
    set_seed(0)
    print(random.random(), np.random.rand(), torch.rand(1).item())
```

```python
# Transformers を使っているなら、これ 1 つで Python・NumPy・PyTorch をまとめて固定できる
from transformers import set_seed as hf_set_seed

hf_set_seed(0)
layer = torch.nn.Linear(4, 2)
print(layer.weight[0].tolist())
hf_set_seed(0)
print(torch.nn.Linear(4, 2).weight[0].tolist())
```

```python
# GPU での完全な再現性が必要な場合 (遅くなることがある)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)   # 非決定的な演算があるとエラーにして知らせる

# グローバルな乱数に頼らず、専用の乱数生成器を使うのが安全
rng = np.random.default_rng(0)
g = torch.Generator().manual_seed(0)
print(rng.integers(0, 100, 3), torch.randperm(5, generator=g).tolist())
```

- シードを固定しても、GPU の機種、ライブラリのバージョン、並列処理の順序が違えば結果は変わりうる。完全な再現より、「複数シードで傾向が変わらないこと」を確認するほうが大事。
- モデルの初期化の前、データのシャッフルの前など、乱数を使う処理より前に呼ぶ。セルを実行し直すと乱数の状態が進むので、必要なら再度呼ぶ。
- `DataLoader` で `num_workers > 0` のときは、`worker_init_fn` と `generator` も指定する。

## 処理時間を計測する {timing}

どの処理が遅いのかを知りたい。2 つの実装のどちらが速いかを比べたい。

```python
%%time
# セル全体の実行時間を測る（%%time はセルの 1 行目に書く）
total = sum(i * i for i in range(3_000_000))
```

```python
import re

text = "自然言語処理 " * 10000
pattern = re.compile(r"\S+")

# %timeit: 何度も繰り返して平均と標準偏差を出す。短い処理の比較に向く
%timeit text.split()
%timeit pattern.findall(text)
```

```python
import random
import time
from collections import Counter
from contextlib import contextmanager

@contextmanager
def timer(name):
    """with ブロックの時間を測る。スクリプトに書き写しても使える"""
    start = time.perf_counter()
    yield
    print(f"[{name}] {time.perf_counter() - start:.3f} 秒")

# 計測の例として、200 万語のテキストを作る
random.seed(0)
words = ["自然", "言語", "処理", "を", "学ぶ", "の", "は", "データ", "モデル", "文章", "単語", "解析"]
big_text = " ".join(random.choices(words, k=2_000_000))

with timer("トークン化"):
    tokens = big_text.split()
with timer("頻度の集計"):
    freq = Counter(tokens)
```

```python
# どの関数に時間がかかっているかを調べる（プロファイラ）
import cProfile, pstats

def slow_pipeline():
    words = [w for w in text.split()]
    return sorted(set(words)), sum(len(w) for w in words), re.findall(r"処理", text)

profiler = cProfile.Profile(); profiler.runcall(slow_pipeline)
_ = pstats.Stats(profiler).sort_stats("cumulative").print_stats(5)
```

- GPU の処理は非同期なので、正確に測るには計測の前後で `torch.cuda.synchronize()` を呼ぶ。
- 初回の実行は、モデルの読み込みやキャッシュの作成で遅い。2 回目以降を測る。
- Colab の無料枠は実行のたびに CPU や GPU の割り当てが変わる。異なるセッションで測った時間は比較できない。
- `subprocess` の都度起動と常駐 `Popen` パイプのオーバーヘッド比較、NumPy/BLAS・疎行列演算による高速化、および `cProfile` を用いた I/O バウンド・計算バウンドの切り分けとアムダールの法則の実践例は、姉妹編『[自然言語処理の教科書](https://github.com/mamorlis/nlpbook)』第 5 章の [パイプラインの高速化・スケーラビリティとボトルネック計測](https://github.com/mamorlis/nlpbook/blob/main/notebooks/ch5_toolkit_benchmark.ipynb) も参照。

## 進捗バーを表示する (tqdm) {tqdm_progress}

時間のかかる処理が、いまどこまで進んでいて、あと何分かかるのかを知りたい。

```python
import time
from tqdm.auto import tqdm   # ノートブックでもスクリプトでも適切な表示になる

texts = [f"文 {i}" for i in range(200)]
results = []
pbar = tqdm(texts, desc="処理中")
for text in pbar:
    time.sleep(0.005)
    results.append(len(text))
print(pbar)   # 終わったときのバーの状態を文字で表示する
```

```python
# バッチ処理: 件数ベースで進める / 途中経過（損失など）を右側に表示する
batch_size = 32
with tqdm(total=len(texts), desc="推論", unit="文") as pbar:
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        time.sleep(0.02)
        pbar.update(len(batch))
        pbar.set_postfix(loss=f"{1 / (i + 1):.4f}")
print(pbar)
```

```python
# pandas の apply に進捗バーを付ける
import pandas as pd

tqdm.pandas(desc="apply")
df = pd.DataFrame({"text": texts})
df["length"] = df["text"].progress_apply(len)
print(df.tail(3))

# ジェネレータなど長さの分からないものは total を渡す。並列処理は tqdm.contrib.concurrent が便利
from tqdm.contrib.concurrent import thread_map

lengths = thread_map(lambda t: (time.sleep(0.005), len(t))[1], texts, max_workers=8, desc="並列")
print(sum(lengths))
```

- `tqdm.auto` を使うと、Colab ではウィジェットの進捗バー、端末ではテキストの進捗バーになる。
- `print(pbar)` で、バーの今の状態（件数・経過時間・速さ）を文字で表示できる。ログに残したいときに便利。
- ループの中で `print` するとバーが崩れる。代わりに `tqdm.write("...")` を使う。
- API 呼び出しのような待ち時間の長い処理は、`thread_map` でスレッド並列にすると大幅に速くなる（レート制限に注意）。

## 長時間の学習を途中から再開できるようにする {colab_checkpoint norun}

Colab は一定時間で切断される。学習の途中経過を Google Drive に保存して、切れても続きから再開したい。

```python
from google.colab import drive

drive.mount("/content/drive")
CKPT_DIR = "/content/drive/MyDrive/Colab Notebooks/nlpcookbook/checkpoints/my-run"
```

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir=CKPT_DIR,                 # チェックポイントを Drive に直接書く
    save_strategy="steps", save_steps=500,
    save_total_limit=2,                  # 古いものを消して Drive の容量を節約（最新の 2 つだけ残す）
    eval_strategy="steps", eval_steps=500,
    load_best_model_at_end=True,
    num_train_epochs=3, per_device_train_batch_size=32, report_to="none",
)
# trainer = Trainer(model=model, args=args, train_dataset=..., eval_dataset=...)
```

```python
import os
from transformers.trainer_utils import get_last_checkpoint

last = get_last_checkpoint(CKPT_DIR) if os.path.isdir(CKPT_DIR) else None
print("再開するチェックポイント:", last)
# trainer.train(resume_from_checkpoint=last)   # None なら最初から、あれば続きから（オプティマイザや学習率の状態も復元される）
```

```python
# 自前の学習ループの場合: モデルだけでなく、オプティマイザ・エポック数・乱数の状態も一緒に保存する
import torch

def save_checkpoint(path, model, optimizer, epoch, step):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "step": step,
                "torch_rng": torch.get_rng_state()}, tmp)
    os.replace(tmp, path)                # 書き込み途中で切断されても壊れたファイルが残らないようにする

def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"]); torch.set_rng_state(ckpt["torch_rng"])
    return ckpt["epoch"], ckpt["step"]
```

- モデルの重みだけを保存して再開すると、オプティマイザの状態や学習率スケジュールが初期化されて、学習が乱れる。
- 保存の間隔は「失っても許せる時間」で決める（15〜30 分に 1 回が目安）。保存自体にも時間と Drive の容量がかかる。
- 無料枠は最長 12 時間、操作がないとさらに早く切れる。数時間を超える学習は、最初から再開を前提に組む。

## フォームでパラメータを切り替えられるようにする {colab_forms}

コードを書き換えずに、モデル名やしきい値などの設定をフォームから選べるようにしたい。授業や共有用のノートブックに便利。

```python
#@title 設定 { display-mode: "form" }
model_name = "intfloat/multilingual-e5-small"  #@param ["intfloat/multilingual-e5-small", "sentence-transformers/LaBSE", "cl-nagoya/ruri-v3-30m"]
text = "自然言語処理は楽しい"  #@param {type:"string"}
top_k = 5  #@param {type:"slider", min:1, max:20, step:1}
threshold = 0.5  #@param {type:"number"}
normalize = True  #@param {type:"boolean"}
mode = "search"  #@param ["search", "cluster", "classify"] {allow-input: true}

print(model_name, text, top_k, threshold, normalize, mode)
```

```python
# Colab 以外（Jupyter、VS Code）でも動く対話的な部品は ipywidgets で作る
import ipywidgets as widgets
from IPython.display import display

slider = widgets.IntSlider(value=5, min=1, max=20, description="top_k")
box = widgets.Text(value="自然言語処理", description="クエリ")
out = widgets.Output()

def on_change(_):
    with out:
        out.clear_output()
        print(f"クエリ「{box.value}」で上位 {slider.value} 件を検索")

slider.observe(on_change, names="value"); box.observe(on_change, names="value")
display(box, slider, out)
on_change(None)  # 最初の表示（Output ウィジェットの中に出る）

# ウィジェットの値は、Python の変数のように .value で読める
print("クエリ:", box.value, "/ top_k:", slider.value)
```

- `#@param` は Colab 専用のコメント記法。他の環境ではただのコメントとして無視され、右辺の値がそのまま使われるので、コードとしては普通に動く。
- フォームの値を変えると、セルのコード（右辺の値）が書き換わる。セルを再実行して初めて反映される。
- `#@title` を付けて `display-mode: "form"` にすると、コードを隠してフォームだけを見せられる。
- ipywidgets の部品（テキスト欄・スライダー・`Output` の中の表示）は、セルを実行したときだけ現れる。保存されたノートブックを GitHub などで開いても表示されない。上の例では、最後の `print` の行だけが普通の出力として残る。

## pandas で長いテキストを省略せずに表示する {pandas_display}

DataFrame を表示すると、テキストの列が「…」で省略されて中身が読めない。全文を見たい。行や列も省略されたくない。

```python
import pandas as pd

df = pd.DataFrame({
    "id": range(1, 4),
    "text": ["吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。",
             "国境の長いトンネルを抜けると雪国であった。夜の底が白くなった。信号所に汽車が止まった。",
             "メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。"],
    "score": [0.912345, 0.5, 0.123456],
})
df   # 既定では 50 文字で切られる
```

```python
# 一時的に設定を変える (with を抜けると元に戻る)
with pd.option_context("display.max_colwidth", None, "display.max_rows", 100, "display.precision", 3):
    display(df)
```

```python
# 折り返し・左寄せ・色付けなど、見た目を整えて表示する
(df.style
   .set_properties(subset=["text"], **{"text-align": "left", "white-space": "pre-wrap", "width": "480px"})
   .format({"score": "{:.2f}"})
   .background_gradient(subset=["score"], cmap="Blues")
   .hide(axis="index"))
```

```python
# ノートブック全体で設定を変える場合
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.unicode.east_asian_width", True)   # print() での表示で、全角文字の桁を揃える
print(df[["id", "score"]].assign(title=["猫", "雪国", "メロス"]))
pd.reset_option("display.max_colwidth")
```

- Colab では `from google.colab import data_table; data_table.enable_dataframe_formatter()` を実行すると、並べ替えや絞り込みのできる対話的な表になる。
- エラー分析では、全文を読めることが何より大事。省略された表を眺めて分かった気にならない。

## JSON Lines 形式でデータを読み書きする {jsonl_io}

1 行に 1 件の JSON を書く JSONL 形式で、データセットや実験結果を保存・読み込みしたい。

```python
import json

records = [
    {"id": 1, "text": "今日は良い天気です。", "label": "positive", "tokens": ["今日", "は", "良い", "天気", "です", "。"]},
    {"id": 2, "text": "改行\nや\"引用符\"、絵文字😀を含む文", "label": "neutral", "tokens": []},
]

with open("data.jsonl", "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")   # ensure_ascii=False で日本語をそのまま書く

print(open("data.jsonl", encoding="utf-8").read())

with open("data.jsonl", encoding="utf-8") as f:
    loaded = [json.loads(line) for line in f if line.strip()]
print(loaded == records)
```

```python
import pandas as pd

df = pd.read_json("data.jsonl", lines=True)                    # pandas で読む
df.to_json("out.jsonl", orient="records", lines=True, force_ascii=False)   # pandas で書く
print(df)

# 途中で落ちても結果を失わないよう、1 件処理するごとに追記する (API 呼び出しや長時間の推論で重要)
import os

done = {json.loads(l)["id"] for l in open("results.jsonl", encoding="utf-8")} if os.path.exists("results.jsonl") else set()
with open("results.jsonl", "a", encoding="utf-8") as f:
    for r in records:
        if r["id"] in done:
            continue                                            # 再実行時は済んだものを飛ばす
        f.write(json.dumps({"id": r["id"], "length": len(r["text"])}, ensure_ascii=False) + "\n")
        f.flush()
print(open("results.jsonl", encoding="utf-8").read())
```

- CSV と違って、改行・カンマ・引用符を含むテキストやリスト・入れ子の構造を安全に扱える。NLP のデータ交換では事実上の標準。
- 1 行ずつ読めるので、メモリに載らない巨大なファイルも処理できる。`.jsonl.gz` のまま `gzip.open(path, "rt")` で読める。
- JSON Lines (`jsonl`) 通信による 4 段階 Unix フィルタパイプラインの分割、中間出力差分による原因ステージの局所化、スキーマ検証・ゴールデン回帰テストの実践例は、姉妹編『[自然言語処理の教科書](https://github.com/mamorlis/nlpbook)』第 5 章の [パイプライン処理のモジュール分割と前後処理・デバッグ](https://github.com/mamorlis/nlpbook/blob/main/notebooks/ch5_toolkit_pipeline.ipynb) も参照。

## CSV / TSV の文字化けと列ずれを避ける {csv_io}

日本語のテキストを CSV で保存して Excel で開くと文字化けする。テキスト中の改行やカンマで列がずれる。

```python
import csv
import pandas as pd

df = pd.DataFrame({"id": [1, 2, 3], "text": ["普通の文です。", "カンマ, と \"引用符\" を含む", "改行を\n含む文"], "label": ["A", "B", "A"]})

df.to_csv("data.csv", index=False)                                # 既定は UTF-8 (BOM なし)。pandas 同士なら問題ない
df.to_csv("for_excel.csv", index=False, encoding="utf-8-sig")     # Excel で開く人に渡すなら BOM 付き UTF-8
print(open("data.csv", encoding="utf-8").read())
print(pd.read_csv("data.csv").equals(df))
```

```python
# TSV: テキスト中のタブと改行に注意。引用符の処理を無効にするなら、事前に置換しておく
safe = df.assign(text=df["text"].str.replace(r"[\t\n\r]+", " ", regex=True))
safe.to_csv("data.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE)
back = pd.read_csv("data.tsv", sep="\t", quoting=csv.QUOTE_NONE)
print(back)

# 読み込み時の定番の落とし穴: 文字列の列が勝手に数値や NaN になる
tricky = pd.DataFrame({"code": ["001", "002", "NA", "null"], "word": ["なし", "None", "nan", ""]})
tricky.to_csv("tricky.csv", index=False)
print(pd.read_csv("tricky.csv").to_dict("list"))
print(pd.read_csv("tricky.csv", dtype=str, keep_default_na=False).to_dict("list"))
```

- Windows の Excel が出力した CSV は Shift_JIS (`encoding="cp932"`) のことが多い。
- `dtype=str, keep_default_na=False` を付けると、「001」が 1 になったり、「NA」「null」「None」という文字列が欠損値に化けたりするのを防げる。
- 対訳データなどで 1 行 1 文のプレーンテキストを使う場合も、文中の改行は必ず除いておく（行数がずれると対応が全部壊れる）。

## 巨大なテキストファイルをメモリに載せずに処理する {large_file_stream}

数 GB のコーパスを、全部メモリに読み込まずに、1 行ずつ、あるいは一定件数ずつ処理したい。

```python
import gzip
import random

# 動作確認用に 20 万行の gzip 圧縮ファイルを作る
random.seed(0)
words = ["自然", "言語", "処理", "機械", "学習", "翻訳", "モデル", "データ", "評価", "実験"]
with gzip.open("corpus.txt.gz", "wt", encoding="utf-8") as f:
    for i in range(200_000):
        f.write(" ".join(random.choices(words, k=random.randint(3, 12))) + "\n")
```

```python
from collections import Counter
from itertools import islice

def read_lines(path):
    """ジェネレータ: 1 行ずつ読んで返す。ファイル全体はメモリに載らない"""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")

def batched(iterable, n):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch

freq, n_lines = Counter(), 0
for batch in batched(read_lines("corpus.txt.gz"), 10_000):    # 1 万行ずつ処理 (モデルの推論ならここでバッチ推論する)
    n_lines += len(batch)
    for line in batch:
        freq.update(line.split())
print(n_lines, "行", freq.most_common(3))
```

```python
# ファイル全体を読まずに一様ランダムに k 行を選ぶ (リザーバサンプリング)
def reservoir_sample(iterable, k, seed=0):
    rng, sample = random.Random(seed), []
    for i, item in enumerate(iterable):
        if i < k:
            sample.append(item)
        elif (j := rng.randint(0, i)) < k:
            sample[j] = item
    return sample

print(reservoir_sample(read_lines("corpus.txt.gz"), 3))
```

- `f.readlines()` や `f.read().split("\n")` は全体をメモリに載せるので、大きなファイルでは使わない。
- 行数だけ知りたいなら `!zcat corpus.txt.gz | wc -l`、先頭を見たいなら `!zcat corpus.txt.gz | head` が速い。
- 表形式なら `pd.read_csv(path, chunksize=100_000)`、Hugging Face Datasets なら `streaming=True` が同じ役割をする。
- 常駐ワーカーへの `stdin`/`stdout` ストリーミング処理やバッファ付き I/O によるボトルネック解消、I/O vs 計算プロファイリングとアムダールの法則の実践例は、姉妹編『[自然言語処理の教科書](https://github.com/mamorlis/nlpbook)』第 5 章の [パイプラインの高速化・スケーラビリティとボトルネック計測](https://github.com/mamorlis/nlpbook/blob/main/notebooks/ch5_toolkit_benchmark.ipynb) も参照。
