# NLP クックブック

自然言語処理の「やりたいこと」に対して「こうやればいい」をまとめた小ネタ集です。
1 レシピ = 1 ノートブックで、すべて Google Colab でそのまま実行できます。

- 日本語のテキスト処理を中心に、英語・多言語も扱います。
- 各ノートブックは上から順に実行するだけで動きます (必要なライブラリのインストールを含む)。
- 実行結果を埋め込んであるので、GitHub 上で眺めるだけでも使えます。

## 目次

<!-- INDEX:START -->
全 22 レシピ。タイトルをクリックすると GitHub 上で、バッジをクリックすると Google Colab で開きます。

- [1. 文字列処理と正規化](#1-文字列処理と正規化) (22)

### 1. 文字列処理と正規化

形態素解析やモデルに入れる前の、文字レベルの下ごしらえ。

| # | やりたいこと | Colab |
|---|---|---|
| 01-01 | [Unicode 正規化 (NFKC) で全角英数・半角カナを揃える](notebooks/01_string/nfkc.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_string/nfkc.ipynb) |
| 01-02 | [全角⇔半角を種類ごとに変換する](notebooks/01_string/zenhan.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_string/zenhan.ipynb) |
| 01-03 | [ひらがな⇔カタカナを変換する](notebooks/01_string/hira_kata.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_string/hira_kata.ipynb) |
| 01-04 | [文字種 (漢字・ひらがな・カタカナ・英数字) を判定する](notebooks/01_string/char_type.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_string/char_type.ipynb) |
| 01-05 | [neologdn で日本語の表記ゆれを正規化する](notebooks/01_string/neologdn.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_string/neologdn.ipynb) |
| 01-06 | [絵文字を検出・除去・テキスト化する](notebooks/01_string/emoji.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_string/emoji.ipynb) |
| 01-07 | [見た目どおりの文字数を数える (書記素クラスタ)](notebooks/01_string/grapheme.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_string/grapheme.ipynb) |
| 01-08 | [文字化けを修復する (ftfy)](notebooks/01_string/ftfy.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_string/ftfy.ipynb) |
| 01-09 | [文字コードを判定して UTF-8 に変換する](notebooks/01_string/encoding_detect.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_string/encoding_detect.ipynb) |
| 01-10 | [URL・メンション・ハッシュタグを除去する](notebooks/01_string/remove_url.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_string/remove_url.ipynb) |
| 01-11 | [HTML タグと実体参照を取り除く](notebooks/01_string/strip_html.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_string/strip_html.ipynb) |
| 01-12 | [漢数字⇔アラビア数字を変換する](notebooks/01_string/kansuji.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_string/kansuji.ipynb) |
| 01-13 | [和暦⇔西暦を変換する](notebooks/01_string/wareki.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_string/wareki.ipynb) |
| 01-14 | [全角を 2 として表示幅を数える・揃える](notebooks/01_string/display_width.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_string/display_width.ipynb) |
| 01-15 | [繰り返し文字を縮約する (「すごーーーい」「wwww」)](notebooks/01_string/squeeze_repeats.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_string/squeeze_repeats.ipynb) |
| 01-16 | [漢字かな交じり文をローマ字・ひらがなにする](notebooks/01_string/romaji.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_string/romaji.ipynb) |
| 01-17 | [日本語の文を句点で分割する (括弧内は分割しない)](notebooks/01_string/sentence_split_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_string/sentence_split_ja.ipynb) |
| 01-18 | [文字 n-gram を作る](notebooks/01_string/char_ngram.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_string/char_ngram.ipynb) |
| 01-19 | [日付・時刻の表現を datetime に変換する](notebooks/01_string/date_parse.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_string/date_parse.ipynb) |
| 01-20 | [電話番号・郵便番号・メールアドレスを正規表現で抜き出す](notebooks/01_string/regex_patterns.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_string/regex_patterns.ipynb) |
| 01-21 | [テキストの言語 (文字種) の割合を調べる](notebooks/01_string/script_ratio.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_string/script_ratio.ipynb) |
| 01-22 | [ゼロ幅文字・制御文字・異体字セレクタを取り除く](notebooks/01_string/invisible_chars.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_string/invisible_chars.ipynb) |

⚡ = GPU ランタイム推奨、🔑 = API キーが必要
<!-- INDEX:END -->

## 使い方

- **Colab で開く**: 目次のバッジをクリックします。自分用に書き換えるときは「ドライブにコピーを保存」してください。
- **Google Drive に置く**: `notebooks/` 以下を Drive の `Colab Notebooks/nlpcookbook/` にコピーすれば、Drive から直接開けます。

## レシピの追加・修正

ノートブックは `recipes/*.md` から自動生成しています。`.ipynb` を直接編集せず、Markdown を編集してから
`python tools/build.py` を実行してください。詳しくは [CONTRIBUTING.md](CONTRIBUTING.md) を参照。

## ライセンス

[Apache License 2.0](LICENSE)
