# レシピの書き方

ノートブック (`notebooks/**/*.ipynb`) と README の目次は、`recipes/*.md` から自動生成しています。
`.ipynb` を直接編集しても次のビルドで上書きされるので、Markdown のほうを編集してください。

## 手順

```bash
pip install nbformat nbclient ipykernel   # 初回のみ

# 1. recipes/NN_章名.md（付録は A_章名.md、B_章名.md）を編集する
# 2. ノートブックと目次を生成する（実行はしない）
python tools/build.py

# 3. 追加・変更したレシピを実行して、出力をノートブックに埋め込む
python tools/build.py --run -k スラッグ名

# 4. recipes/ と notebooks/ と README.md をまとめてコミットする
```

`--run` は、コードが前回の実行時から変わっていないレシピを飛ばします。全部やり直すときは `--force` を付けます。
`-k` はスラッグか章のファイル名の部分一致で対象を絞ります（`-k 05_` で 5 章全体、`-k A_` で付録 A 全体）。

## レシピの書式

1 ファイルが 1 章、`## ` で始まる見出しが 1 レシピです。

````markdown
# 章のタイトル

章の説明（目次に載る。省略可）

## レシピのタイトル（やりたいことを書く） {slug}

何をしたいのか、どんなときに使うのかを 1〜2 文で。

```python
!pip install -q some-library
```

```python
import some_library
print(some_library.do("例"))
```

- 補足や落とし穴を箇条書きで。
````

- `{slug}` はノートブックのファイル名になります (`notebooks/章のファイル名/slug.ipynb`)。英小文字・数字・アンダースコアのみで、全体で一意にしてください。
- ` ```python ` のブロックがコードセル、それ以外が Markdown セルになります。
- スラッグの後ろにタグを書けます: `{slug gpu}`、`{slug key norun}` など。

| タグ | 意味 |
|---|---|
| `gpu` | GPU ランタイムを既定にし、目次に ⚡ を付ける |
| `key` | API キーが必要。目次に 🔑 を付け、`--run` の対象から外す |
| `norun` | `--run` の対象から外す（Colab 専用の機能、GPU が必須、など）|

## 方針

- **1 レシピ 1 ネタ。** そのノートブックだけで完結させ、上から実行するだけで動くようにする（他のレシピで定義した関数に依存しない）。
- **Colab にプリインストールされていないライブラリは、必ず `!pip install -q` を書く。**
- データは、認証なしでダウンロードできる小さなものを使う（青空文庫、livedoor ニュースコーパス、NLTK や scikit-learn の付属データ、Hugging Face Hub の Parquet 形式のデータセットなど）。
- 出力は短く。長いリストは先頭だけ表示する。
- 補足には、動かしてみて実際に踏んだ落とし穴を書く。
- 実行できないレシピ（`norun`、`key`）は、API の仕様が変わっても気づけないので、最小限にとどめる。
- 和文の中のかっこは、中に日本語を含むものを全角（）にし、前後に空白を入れない。中が英数字だけのもの（`(LLM)`、`(18)` など）、英文の中のかっこ、数式や関数呼び出し（`log(x)` など）は半角 () のまま。コードセルの中も同じ規則で書く。

## Google Drive への配置

`notebooks/` 以下を Google Drive の `Colab Notebooks/nlpcookbook/` にコピーすると、Drive から直接開けます。
GitHub 上のノートブックは、README のバッジから Drive を経由せずに Colab で開けます。
