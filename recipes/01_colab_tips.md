# Google Colab の小技

Colab で NLP の実験をするときに、毎回調べることになる定番の操作。

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

## GPU が使えるか確認し、メモリを解放する {colab_gpu}

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
- メモリ不足のときの対処: バッチサイズを下げる → 系列長 (`max_length`) を下げる → 半精度 (`fp16` / `torch_dtype=torch.float16`) → 勾配累積 → 量子化、の順に試す。
- 推論では `torch.inference_mode()` を忘れない。勾配用の中間結果を保持してメモリを浪費する。

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

## 処理時間を計測する {timing}

どの処理が遅いのかを知りたい。2 つの実装のどちらが速いかを比べたい。

```python
%%time
# セル全体の実行時間を測る (%%time はセルの 1 行目に書く)
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
import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    """with ブロックの時間を測る。スクリプトに書き写しても使える"""
    start = time.perf_counter()
    yield
    print(f"[{name}] {time.perf_counter() - start:.3f} 秒")

with timer("トークン化"):
    tokens = text.split()
with timer("頻度の集計"):
    from collections import Counter
    freq = Counter(tokens)
```

```python
# どの関数に時間がかかっているかを調べる (プロファイラ)
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
    save_total_limit=2,                  # 古いものを消して Drive の容量を節約 (最新の 2 つだけ残す)
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
# trainer.train(resume_from_checkpoint=last)   # None なら最初から、あれば続きから (オプティマイザや学習率の状態も復元される)
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
- 保存の間隔は「失っても許せる時間」で決める (15〜30 分に 1 回が目安)。保存自体にも時間と Drive の容量がかかる。
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
# Colab 以外 (Jupyter、VS Code) でも動く対話的な部品は ipywidgets で作る
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
on_change(None)
```

- `#@param` は Colab 専用のコメント記法。他の環境ではただのコメントとして無視され、右辺の値がそのまま使われるので、コードとしては普通に動く。
- フォームの値を変えると、セルのコード (右辺の値) が書き換わる。セルを再実行して初めて反映される。
- `#@title` を付けて `display-mode: "form"` にすると、コードを隠してフォームだけを見せられる。

## ライブラリのバージョンを確認・固定する {pip_versions}

Colab のプリインストールライブラリは予告なく更新される。動いていたコードが動かなくなったときに、バージョンを確認し、固定したい。

```python
import importlib.metadata as md
import platform
import sys

print("Python", sys.version.split()[0], "/", platform.platform())
for pkg in ["torch", "transformers", "datasets", "tokenizers", "sentence-transformers", "scikit-learn", "numpy", "pandas", "spacy", "nltk"]:
    try:
        print(f"{pkg:24}{md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg:24}(未インストール)")
```

```python
# 実験したときの環境を丸ごと記録しておく (論文の再現や、後日の自分のため)
!pip freeze > requirements-lock.txt
!wc -l requirements-lock.txt && grep -i -E "^(torch|transformers|numpy)==" requirements-lock.txt
```

```python
# バージョンを指定して入れる例 (実際に入れ替える場合はコメントを外す)
# !pip install -q "transformers==4.57.1" "tokenizers>=0.22,<0.23"
# !pip install -q -r requirements-lock.txt

# コードの中でバージョンによって分岐する
from packaging.version import Version
import transformers

print("v5 以降:", Version(transformers.__version__) >= Version("5.0.0"))
```

- すでに import 済みのライブラリを `pip install` で入れ替えた場合は、ランタイムの再起動 (ランタイム → セッションを再起動) が必要。
- NumPy や PyTorch のような土台のライブラリを入れ替えると、他のプリインストールライブラリが壊れやすい。必要最小限にとどめる。
- 「前は動いたのに」というときは、まずエラーメッセージとライブラリのバージョンの変更履歴 (リリースノート、移行ガイド) を確認する。

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

## 進捗バーを表示する (tqdm) {tqdm_progress}

時間のかかる処理が、いまどこまで進んでいて、あと何分かかるのかを知りたい。

```python
import time
from tqdm.auto import tqdm   # ノートブックでもスクリプトでも適切な表示になる

texts = [f"文 {i}" for i in range(200)]
results = []
for text in tqdm(texts, desc="処理中"):
    time.sleep(0.005)
    results.append(len(text))
```

```python
# バッチ処理: 件数ベースで進める / 途中経過 (損失など) を右側に表示する
batch_size = 32
with tqdm(total=len(texts), desc="推論", unit="文") as pbar:
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        time.sleep(0.02)
        pbar.update(len(batch))
        pbar.set_postfix(loss=f"{1 / (i + 1):.4f}")
```

```python
# pandas の apply に進捗バーを付ける
import pandas as pd

tqdm.pandas(desc="apply")
df = pd.DataFrame({"text": texts})
df["length"] = df["text"].progress_apply(len)

# ジェネレータなど長さの分からないものは total を渡す。並列処理は tqdm.contrib.concurrent が便利
from tqdm.contrib.concurrent import thread_map

lengths = thread_map(lambda t: (time.sleep(0.005), len(t))[1], texts, max_workers=8, desc="並列")
print(sum(lengths))
```

- `tqdm.auto` を使うと、Colab ではウィジェットの進捗バー、端末ではテキストの進捗バーになる。
- ループの中で `print` するとバーが崩れる。代わりに `tqdm.write("...")` を使う。
- API 呼び出しのような待ち時間の長い処理は、`thread_map` でスレッド並列にすると大幅に速くなる (レート制限に注意)。
