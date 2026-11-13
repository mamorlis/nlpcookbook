# NLP クックブック

自然言語処理の「やりたいこと」に対して「こうやればいい」をまとめた小ネタ集です。
1 レシピ = 1 ノートブックで、すべて Google Colab でそのまま実行できます。

- 日本語のテキスト処理を中心に、英語・多言語も扱います。
- 各ノートブックは上から順に実行するだけで動きます (必要なライブラリのインストールを含む)。
- 実行結果を埋め込んであるので、GitHub 上で眺めるだけでも使えます。

## 目次

<!-- INDEX:START -->
全 246 レシピ。タイトルをクリックすると GitHub 上で、バッジをクリックすると Google Colab で開きます。

- [1. Google Colab の小技](#1-google-colab-の小技) (11)
- [2. データの収集と入出力](#2-データの収集と入出力) (11)
- [3. 文字列処理と正規化](#3-文字列処理と正規化) (22)
- [4. 日本語の形態素解析](#4-日本語の形態素解析) (19)
- [5. 英語・多言語の前処理](#5-英語多言語の前処理) (20)
- [6. サブワードとトークナイザ](#6-サブワードとトークナイザ) (14)
- [7. コーパスの統計と可視化](#7-コーパスの統計と可視化) (15)
- [8. 構文解析と固有表現抽出](#8-構文解析と固有表現抽出) (17)
- [9. 単語ベクトル](#9-単語ベクトル) (12)
- [10. 文埋め込み・類似度・検索](#10-文埋め込み類似度検索) (16)
- [11. トピック・キーワード・要約](#11-トピックキーワード要約) (5)
- [12. テキスト分類](#12-テキスト分類) (16)
- [13. 系列ラベリング](#13-系列ラベリング) (8)
- [14. Transformers の基本操作](#14-transformers-の基本操作) (17)
- [15. 大規模言語モデル (LLM)](#15-大規模言語モデル-llm) (17)
- [16. 評価指標と統計的検定](#16-評価指標と統計的検定) (16)
- [17. 文法誤り訂正と言語教育](#17-文法誤り訂正と言語教育) (10)

### 1. Google Colab の小技

Colab で NLP の実験をするときに、毎回調べることになる定番の操作。

| # | やりたいこと | Colab |
|---|---|---|
| 01-01 | [Google Drive をマウントしてファイルを読み書きする](notebooks/01_colab_tips/colab_drive.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_colab_tips/colab_drive.ipynb) |
| 01-02 | [GPU が使えるか確認し、メモリを解放する](notebooks/01_colab_tips/colab_gpu.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_colab_tips/colab_gpu.ipynb) |
| 01-03 | [API キーをシークレット機能で安全に扱う](notebooks/01_colab_tips/colab_secrets.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_colab_tips/colab_secrets.ipynb) |
| 01-04 | [乱数シードを固定して実験を再現できるようにする](notebooks/01_colab_tips/set_seed.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_colab_tips/set_seed.ipynb) |
| 01-05 | [ファイルをアップロード・ダウンロードする](notebooks/01_colab_tips/colab_files.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_colab_tips/colab_files.ipynb) |
| 01-06 | [処理時間を計測する](notebooks/01_colab_tips/timing.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_colab_tips/timing.ipynb) |
| 01-07 | [長時間の学習を途中から再開できるようにする](notebooks/01_colab_tips/colab_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_colab_tips/colab_checkpoint.ipynb) |
| 01-08 | [フォームでパラメータを切り替えられるようにする](notebooks/01_colab_tips/colab_forms.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_colab_tips/colab_forms.ipynb) |
| 01-09 | [ライブラリのバージョンを確認・固定する](notebooks/01_colab_tips/pip_versions.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_colab_tips/pip_versions.ipynb) |
| 01-10 | [pandas で長いテキストを省略せずに表示する](notebooks/01_colab_tips/pandas_display.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_colab_tips/pandas_display.ipynb) |
| 01-11 | [進捗バーを表示する (tqdm)](notebooks/01_colab_tips/tqdm_progress.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/01_colab_tips/tqdm_progress.ipynb) |

### 2. データの収集と入出力

テキストを集める、ファイルから取り出す、きれいにして保存する。実験の前にやること。

| # | やりたいこと | Colab |
|---|---|---|
| 02-01 | [青空文庫から作品をダウンロードして本文を取り出す](notebooks/02_data_io/aozora_download.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_data_io/aozora_download.ipynb) |
| 02-02 | [Wikipedia の記事本文を API で取得する](notebooks/02_data_io/wikipedia_api.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_data_io/wikipedia_api.ipynb) |
| 02-03 | [Web ページから本文だけを抽出する (trafilatura)](notebooks/02_data_io/web_extract.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_data_io/web_extract.ipynb) |
| 02-04 | [PDF からテキストを抽出する (PyMuPDF)](notebooks/02_data_io/pdf_extract.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_data_io/pdf_extract.ipynb) |
| 02-05 | [Word ファイルからテキストと表を取り出す](notebooks/02_data_io/docx_extract.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_data_io/docx_extract.ipynb) |
| 02-06 | [JSON Lines 形式でデータを読み書きする](notebooks/02_data_io/jsonl_io.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_data_io/jsonl_io.ipynb) |
| 02-07 | [CSV / TSV の文字化けと列ずれを避ける](notebooks/02_data_io/csv_io.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_data_io/csv_io.ipynb) |
| 02-08 | [巨大なテキストファイルをメモリに載せずに処理する](notebooks/02_data_io/large_file_stream.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_data_io/large_file_stream.ipynb) |
| 02-09 | [テキストデータの重複と不良データを取り除く](notebooks/02_data_io/dedup_clean.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_data_io/dedup_clean.ipynb) |
| 02-10 | [データを学習・開発・評価に正しく分割する](notebooks/02_data_io/data_split.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_data_io/data_split.ipynb) |
| 02-11 | [対訳コーパスを取得して中身を確認する](notebooks/02_data_io/parallel_corpus.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_data_io/parallel_corpus.ipynb) |

### 3. 文字列処理と正規化

形態素解析やモデルに入れる前の、文字レベルの下ごしらえ。

| # | やりたいこと | Colab |
|---|---|---|
| 03-01 | [Unicode 正規化 (NFKC) で全角英数・半角カナを揃える](notebooks/03_string/nfkc.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/nfkc.ipynb) |
| 03-02 | [全角⇔半角を種類ごとに変換する](notebooks/03_string/zenhan.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/zenhan.ipynb) |
| 03-03 | [ひらがな⇔カタカナを変換する](notebooks/03_string/hira_kata.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/hira_kata.ipynb) |
| 03-04 | [文字種 (漢字・ひらがな・カタカナ・英数字) を判定する](notebooks/03_string/char_type.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/char_type.ipynb) |
| 03-05 | [neologdn で日本語の表記ゆれを正規化する](notebooks/03_string/neologdn.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/neologdn.ipynb) |
| 03-06 | [絵文字を検出・除去・テキスト化する](notebooks/03_string/emoji.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/emoji.ipynb) |
| 03-07 | [見た目どおりの文字数を数える (書記素クラスタ)](notebooks/03_string/grapheme.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/grapheme.ipynb) |
| 03-08 | [文字化けを修復する (ftfy)](notebooks/03_string/ftfy.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/ftfy.ipynb) |
| 03-09 | [文字コードを判定して UTF-8 に変換する](notebooks/03_string/encoding_detect.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/encoding_detect.ipynb) |
| 03-10 | [URL・メンション・ハッシュタグを除去する](notebooks/03_string/remove_url.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/remove_url.ipynb) |
| 03-11 | [HTML タグと実体参照を取り除く](notebooks/03_string/strip_html.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/strip_html.ipynb) |
| 03-12 | [漢数字⇔アラビア数字を変換する](notebooks/03_string/kansuji.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/kansuji.ipynb) |
| 03-13 | [和暦⇔西暦を変換する](notebooks/03_string/wareki.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/wareki.ipynb) |
| 03-14 | [全角を 2 として表示幅を数える・揃える](notebooks/03_string/display_width.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/display_width.ipynb) |
| 03-15 | [繰り返し文字を縮約する (「すごーーーい」「wwww」)](notebooks/03_string/squeeze_repeats.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/squeeze_repeats.ipynb) |
| 03-16 | [漢字かな交じり文をローマ字・ひらがなにする](notebooks/03_string/romaji.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/romaji.ipynb) |
| 03-17 | [日本語の文を句点で分割する (括弧内は分割しない)](notebooks/03_string/sentence_split_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/sentence_split_ja.ipynb) |
| 03-18 | [文字 n-gram を作る](notebooks/03_string/char_ngram.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/char_ngram.ipynb) |
| 03-19 | [日付・時刻の表現を datetime に変換する](notebooks/03_string/date_parse.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/date_parse.ipynb) |
| 03-20 | [電話番号・郵便番号・メールアドレスを正規表現で抜き出す](notebooks/03_string/regex_patterns.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/regex_patterns.ipynb) |
| 03-21 | [テキストの言語 (文字種) の割合を調べる](notebooks/03_string/script_ratio.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/script_ratio.ipynb) |
| 03-22 | [ゼロ幅文字・制御文字・異体字セレクタを取り除く](notebooks/03_string/invisible_chars.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/invisible_chars.ipynb) |

### 4. 日本語の形態素解析

MeCab (fugashi)・Sudachi・Janome で単語に区切り、品詞や原形、読みを取り出す。

| # | やりたいこと | Colab |
|---|---|---|
| 04-01 | [MeCab (fugashi + UniDic) で形態素解析する](notebooks/04_japanese_tokenization/mecab_fugashi.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/mecab_fugashi.ipynb) |
| 04-02 | [分かち書きする](notebooks/04_japanese_tokenization/wakati.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/wakati.ipynb) |
| 04-03 | [品詞で絞って内容語の原形だけ取り出す](notebooks/04_japanese_tokenization/content_words.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/content_words.ipynb) |
| 04-04 | [漢字の読みを取得してふりがな (ルビ) を付ける](notebooks/04_japanese_tokenization/furigana.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/furigana.ipynb) |
| 04-05 | [IPA 辞書で MeCab を使う](notebooks/04_japanese_tokenization/mecab_ipadic.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/mecab_ipadic.ipynb) |
| 04-06 | [Sudachi で分割の粒度 (A/B/C) を切り替える](notebooks/04_japanese_tokenization/sudachi_split.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/sudachi_split.ipynb) |
| 04-07 | [Sudachi の正規化形で表記ゆれを吸収する](notebooks/04_japanese_tokenization/sudachi_normalize.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/sudachi_normalize.ipynb) |
| 04-08 | [Janome で形態素解析する (Pure Python)](notebooks/04_japanese_tokenization/janome.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/janome.ipynb) |
| 04-09 | [Janome にユーザー辞書を追加する](notebooks/04_japanese_tokenization/janome_userdic.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/janome_userdic.ipynb) |
| 04-10 | [連続する名詞をつなげて複合名詞を抽出する](notebooks/04_japanese_tokenization/compound_noun.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/compound_noun.ipynb) |
| 04-11 | [日本語のストップワードを除去する](notebooks/04_japanese_tokenization/stopwords_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/stopwords_ja.ipynb) |
| 04-12 | [単語 n-gram を作る](notebooks/04_japanese_tokenization/word_ngram.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/word_ngram.ipynb) |
| 04-13 | [動詞・形容詞の活用型と活用形を調べる](notebooks/04_japanese_tokenization/conjugation.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/conjugation.ipynb) |
| 04-14 | [トークンの文字位置 (オフセット) を取得する](notebooks/04_japanese_tokenization/token_offsets.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/token_offsets.ipynb) |
| 04-15 | [DataFrame の大量テキストをまとめて分かち書きする](notebooks/04_japanese_tokenization/tokenize_pandas.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/tokenize_pandas.ipynb) |
| 04-16 | [解析器・辞書による分割の違いを比べる](notebooks/04_japanese_tokenization/tokenizer_compare.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/tokenizer_compare.ipynb) |
| 04-17 | [MeCab の N-best 解を得る](notebooks/04_japanese_tokenization/mecab_nbest.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/mecab_nbest.ipynb) |
| 04-18 | [辞書にない未知語を見つける](notebooks/04_japanese_tokenization/unknown_words.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/unknown_words.ipynb) |
| 04-19 | [日本語の単語を読みの五十音順に並べる](notebooks/04_japanese_tokenization/sort_by_reading.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/sort_by_reading.ipynb) |

### 5. 英語・多言語の前処理

英語を中心に、空白で区切られる言語や中国語などの基本的な前処理。

| # | やりたいこと | Colab |
|---|---|---|
| 05-01 | [英文を単語と文に分割する (NLTK)](notebooks/05_english_multilingual/nltk_tokenize.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_english_multilingual/nltk_tokenize.ipynb) |
| 05-02 | [語幹を取り出す (ステミング)](notebooks/05_english_multilingual/stemming.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_english_multilingual/stemming.ipynb) |
| 05-03 | [単語を辞書の見出し語に戻す (レンマ化)](notebooks/05_english_multilingual/lemmatize_spacy.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_english_multilingual/lemmatize_spacy.ipynb) |
| 05-04 | [英語のストップワードを除去する](notebooks/05_english_multilingual/stopwords_en.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_english_multilingual/stopwords_en.ipynb) |
| 05-05 | [テキストが何語で書かれているか判定する (langdetect)](notebooks/05_english_multilingual/langdetect.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_english_multilingual/langdetect.ipynb) |
| 05-06 | [高精度な言語判定と、混在テキストの言語区間検出 (lingua)](notebooks/05_english_multilingual/lingua.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_english_multilingual/lingua.ipynb) |
| 05-07 | [英語の短縮形を展開する (I'm → I am)](notebooks/05_english_multilingual/contractions.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_english_multilingual/contractions.ipynb) |
| 05-08 | [アクセント記号を外す・非 ASCII 文字を ASCII にする](notebooks/05_english_multilingual/unidecode.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_english_multilingual/unidecode.ipynb) |
| 05-09 | [英語のスペルミスを訂正する](notebooks/05_english_multilingual/spellcheck.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_english_multilingual/spellcheck.ipynb) |
| 05-10 | [略語に強いルールベースの文分割 (pySBD、多言語)](notebooks/05_english_multilingual/pysbd.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_english_multilingual/pysbd.ipynb) |
| 05-11 | [大文字・小文字を正しく無視して比較する (casefold)](notebooks/05_english_multilingual/casefold.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_english_multilingual/casefold.ipynb) |
| 05-12 | [Moses 方式でトークナイズ・デトークナイズする](notebooks/05_english_multilingual/sacremoses.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_english_multilingual/sacremoses.ipynb) |
| 05-13 | [英語の品詞タグ付けをする](notebooks/05_english_multilingual/pos_tag_en.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_english_multilingual/pos_tag_en.ipynb) |
| 05-14 | [WordNet で同義語・上位語・類似度を調べる (日本語も)](notebooks/05_english_multilingual/wordnet.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_english_multilingual/wordnet.ipynb) |
| 05-15 | [中国語を単語分割する (jieba)](notebooks/05_english_multilingual/jieba.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_english_multilingual/jieba.ipynb) |
| 05-16 | [中国語の簡体字と繁体字を変換する (OpenCC)](notebooks/05_english_multilingual/opencc.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_english_multilingual/opencc.ipynb) |
| 05-17 | [数値を単語で読み上げる形にする (num2words)](notebooks/05_english_multilingual/num2words.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_english_multilingual/num2words.ipynb) |
| 05-18 | [英単語の単数形・複数形・冠詞を変換する (inflect)](notebooks/05_english_multilingual/inflect.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_english_multilingual/inflect.ipynb) |
| 05-19 | [英文の読みやすさ (リーダビリティ) を測る](notebooks/05_english_multilingual/readability_en.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_english_multilingual/readability_en.ipynb) |
| 05-20 | [英語テキストを定番の手順でクリーニングする](notebooks/05_english_multilingual/clean_text_en.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_english_multilingual/clean_text_en.ipynb) |

### 6. サブワードとトークナイザ

SentencePiece や Hugging Face のトークナイザを学習・利用し、モデルに入れる ID 列を作る。

| # | やりたいこと | Colab |
|---|---|---|
| 06-01 | [SentencePiece でサブワードモデルを学習する](notebooks/06_subword/sentencepiece_train.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_subword/sentencepiece_train.ipynb) |
| 06-02 | [サブワード正則化: 分割をサンプリングしてデータ拡張する](notebooks/06_subword/subword_regularization.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_subword/subword_regularization.ipynb) |
| 06-03 | [BPE を 30 行で自作して仕組みを理解する](notebooks/06_subword/bpe_scratch.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_subword/bpe_scratch.ipynb) |
| 06-04 | [Hugging Face tokenizers で BPE / WordPiece を学習する](notebooks/06_subword/tokenizers_train.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_subword/tokenizers_train.ipynb) |
| 06-05 | [事前学習モデルのトークナイザを使う (日本語 BERT)](notebooks/06_subword/hf_tokenizer_basics.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_subword/hf_tokenizer_basics.ipynb) |
| 06-06 | [パディングと切り詰めをしてバッチのテンソルを作る](notebooks/06_subword/padding_truncation.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_subword/padding_truncation.ipynb) |
| 06-07 | [日本語 BERT のデコード結果から余分な空白を消す](notebooks/06_subword/decode_ja_spaces.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_subword/decode_ja_spaces.ipynb) |
| 06-08 | [トークンと元テキストの文字位置を対応付ける (offset mapping)](notebooks/06_subword/offset_mapping.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_subword/offset_mapping.ipynb) |
| 06-09 | [単語単位のラベルをサブワードに揃える (word_ids)](notebooks/06_subword/word_ids.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_subword/word_ids.ipynb) |
| 06-10 | [トークン数を数える (LLM のコンテキスト長・料金の見積もり)](notebooks/06_subword/count_tokens.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_subword/count_tokens.ipynb) |
| 06-11 | [語彙に新しいトークン・特殊トークンを追加する](notebooks/06_subword/add_tokens.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_subword/add_tokens.ipynb) |
| 06-12 | [文ペアを 1 つの入力にまとめる](notebooks/06_subword/sentence_pair.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_subword/sentence_pair.ipynb) |
| 06-13 | [トークン長の分布を調べて max_length を決める](notebooks/06_subword/token_length_dist.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_subword/token_length_dist.ipynb) |
| 06-14 | [未知語 ([UNK]) がどれくらい出るか調べる](notebooks/06_subword/unk_rate.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_subword/unk_rate.ipynb) |

### 7. コーパスの統計と可視化

単語を数える、特徴語を見つける、分布を図にする。コーパスを手に入れたらまずやること。

| # | やりたいこと | Colab |
|---|---|---|
| 07-01 | [単語の頻度を数える](notebooks/07_corpus_statistics/word_freq.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/word_freq.ipynb) |
| 07-02 | [Zipf の法則を確かめる (順位-頻度プロット)](notebooks/07_corpus_statistics/zipf.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/zipf.ipynb) |
| 07-03 | [matplotlib で日本語を文字化けさせずに表示する](notebooks/07_corpus_statistics/matplotlib_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/matplotlib_ja.ipynb) |
| 07-04 | [日本語のワードクラウドを作る](notebooks/07_corpus_statistics/wordcloud_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/wordcloud_ja.ipynb) |
| 07-05 | [TF-IDF で文書ごとの特徴語を取り出す](notebooks/07_corpus_statistics/tfidf_keywords.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/tfidf_keywords.ipynb) |
| 07-06 | [単語の共起を数えて PMI で結びつきの強さを測る](notebooks/07_corpus_statistics/pmi.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/pmi.ipynb) |
| 07-07 | [よく使われる言い回し (コロケーション) を抽出する](notebooks/07_corpus_statistics/collocations.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/collocations.ipynb) |
| 07-08 | [キーワードの前後の文脈を一覧する (KWIC)](notebooks/07_corpus_statistics/kwic.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/kwic.ipynb) |
| 07-09 | [語彙の豊富さを測る (TTR とその仲間)](notebooks/07_corpus_statistics/lexical_diversity.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/lexical_diversity.ipynb) |
| 07-10 | [n-gram 言語モデルを作ってパープレキシティを測り、文を生成する](notebooks/07_corpus_statistics/ngram_lm.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/ngram_lm.ipynb) |
| 07-11 | [共起ネットワークを描く](notebooks/07_corpus_statistics/cooccurrence_network.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/cooccurrence_network.ipynb) |
| 07-12 | [2 つのコーパスを比べて特徴語を見つける (対数尤度比)](notebooks/07_corpus_statistics/keyness.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/keyness.ipynb) |
| 07-13 | [文の長さや文字数の基本統計をとる](notebooks/07_corpus_statistics/corpus_stats.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/corpus_stats.ipynb) |
| 07-14 | [品詞の分布を調べる・比べる](notebooks/07_corpus_statistics/pos_distribution.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/pos_distribution.ipynb) |
| 07-15 | [文字や単語のエントロピーを計算する](notebooks/07_corpus_statistics/entropy.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/entropy.ipynb) |

### 8. 構文解析と固有表現抽出

係り受け解析で文の構造を取り出し、人名・組織名などの固有表現を見つける。

| # | やりたいこと | Colab |
|---|---|---|
| 08-01 | [GiNZA で日本語の依存構造解析 (係り受け解析) をする](notebooks/08_parsing_ner/ginza_dependency.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_parsing_ner/ginza_dependency.ipynb) |
| 08-02 | [依存構造木を図で表示する (displaCy)](notebooks/08_parsing_ner/displacy_dep.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_parsing_ner/displacy_dep.ipynb) |
| 08-03 | [文節に区切って文節単位の係り受けを得る](notebooks/08_parsing_ner/bunsetsu.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_parsing_ner/bunsetsu.ipynb) |
| 08-04 | [「誰が・何を・どうした」の三つ組を抜き出す](notebooks/08_parsing_ner/svo_extract.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_parsing_ner/svo_extract.ipynb) |
| 08-05 | [日本語の固有表現 (人名・地名・組織名など) を抽出する](notebooks/08_parsing_ner/ner_ginza.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_parsing_ner/ner_ginza.ipynb) |
| 08-06 | [英語の固有表現を抽出して色付きで表示する](notebooks/08_parsing_ner/ner_spacy_en.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_parsing_ner/ner_spacy_en.ipynb) |
| 08-07 | [Transformers の pipeline で日本語の固有表現抽出をする](notebooks/08_parsing_ner/ner_transformers_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_parsing_ner/ner_transformers_ja.ipynb) |
| 08-08 | [学習なしで好きなラベルの固有表現を抽出する (GLiNER)](notebooks/08_parsing_ner/gliner.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_parsing_ner/gliner.ipynb) |
| 08-09 | [品詞や原形のパターンで表現を抽出する (spaCy Matcher)](notebooks/08_parsing_ner/spacy_matcher.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_parsing_ner/spacy_matcher.ipynb) |
| 08-10 | [辞書やルールで固有表現を追加する (EntityRuler)](notebooks/08_parsing_ner/entity_ruler.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_parsing_ner/entity_ruler.ipynb) |
| 08-11 | [名詞句を抽出する](notebooks/08_parsing_ner/noun_chunks.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_parsing_ner/noun_chunks.ipynb) |
| 08-12 | [個人情報をマスクする (仮名化)](notebooks/08_parsing_ner/pii_mask.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_parsing_ner/pii_mask.ipynb) |
| 08-13 | [Stanza で多言語の形態素・依存構造解析をする](notebooks/08_parsing_ner/stanza.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_parsing_ner/stanza.ipynb) |
| 08-14 | [文脈自由文法を書いて構文解析する (NLTK)](notebooks/08_parsing_ner/cfg_parse.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_parsing_ner/cfg_parse.ipynb) |
| 08-15 | [解析結果を CoNLL-U 形式で保存・読み込みする](notebooks/08_parsing_ner/conllu.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_parsing_ner/conllu.ipynb) |
| 08-16 | [spaCy で大量の文書を高速に処理する](notebooks/08_parsing_ner/spacy_pipe.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_parsing_ner/spacy_pipe.ipynb) |
| 08-17 | [依存距離と構文木の深さで文の複雑さを測る](notebooks/08_parsing_ner/dep_distance.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_parsing_ner/dep_distance.ipynb) |

### 9. 単語ベクトル

単語を数百次元のベクトルで表し、類似度やアナロジーを計算する。

| # | やりたいこと | Colab |
|---|---|---|
| 09-01 | [word2vec を自分のコーパスで学習する](notebooks/09_word_vectors/word2vec_train.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_word_vectors/word2vec_train.ipynb) |
| 09-02 | [学習済みの英語単語ベクトルを読み込んで類似語を調べる](notebooks/09_word_vectors/pretrained_vectors_en.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_word_vectors/pretrained_vectors_en.ipynb) |
| 09-03 | [学習済みの日本語単語ベクトルを使う (chiVe)](notebooks/09_word_vectors/pretrained_vectors_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_word_vectors/pretrained_vectors_ja.ipynb) |
| 09-04 | [単語ベクトルでアナロジー (王 − 男 + 女 = 女王) を解く](notebooks/09_word_vectors/analogy.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_word_vectors/analogy.ipynb) |
| 09-05 | [fastText で未知語にもベクトルを与える](notebooks/09_word_vectors/fasttext_train.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_word_vectors/fasttext_train.ipynb) |
| 09-06 | [単語ベクトルを 2 次元に落として可視化する](notebooks/09_word_vectors/word_vectors_viz.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_word_vectors/word_vectors_viz.ipynb) |
| 09-07 | [単語ベクトルの平均で文ベクトルを作る](notebooks/09_word_vectors/avg_word_vectors.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_word_vectors/avg_word_vectors.ipynb) |
| 09-08 | [PPMI 行列の SVD で単語ベクトルを作る (カウントベース)](notebooks/09_word_vectors/ppmi_svd.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_word_vectors/ppmi_svd.ipynb) |
| 09-09 | [Doc2Vec で文書ベクトルを学習する](notebooks/09_word_vectors/doc2vec.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_word_vectors/doc2vec.ipynb) |
| 09-10 | [単語ベクトルに含まれるジェンダーバイアスを測る](notebooks/09_word_vectors/embedding_bias.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_word_vectors/embedding_bias.ipynb) |
| 09-11 | [BERT から文脈に応じた単語ベクトルを取り出す](notebooks/09_word_vectors/contextual_word_vectors.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_word_vectors/contextual_word_vectors.ipynb) |
| 09-12 | [単語類似度データセットで単語ベクトルを評価する](notebooks/09_word_vectors/wordsim_eval.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_word_vectors/wordsim_eval.ipynb) |

### 10. 文埋め込み・類似度・検索

文をベクトルにして類似度を測る、似た文書を探す、検索システムを組み立てる。

| # | やりたいこと | Colab |
|---|---|---|
| 10-01 | [文をベクトルにして意味の近さを測る (Sentence Transformers)](notebooks/10_similarity_search/sentence_embeddings.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_similarity_search/sentence_embeddings.ipynb) |
| 10-02 | [埋め込みで意味検索 (セマンティックサーチ) をする](notebooks/10_similarity_search/semantic_search.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_similarity_search/semantic_search.ipynb) |
| 10-03 | [FAISS で大量のベクトルから高速に近傍検索する](notebooks/10_similarity_search/faiss_index.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_similarity_search/faiss_index.ipynb) |
| 10-04 | [BM25 で日本語のキーワード検索をする](notebooks/10_similarity_search/bm25_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_similarity_search/bm25_ja.ipynb) |
| 10-05 | [キーワード検索と埋め込み検索を組み合わせる (ハイブリッド検索、RRF)](notebooks/10_similarity_search/hybrid_search.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_similarity_search/hybrid_search.ipynb) |
| 10-06 | [クロスエンコーダで検索結果を並べ替える (リランキング)](notebooks/10_similarity_search/rerank.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_similarity_search/rerank.ipynb) |
| 10-07 | [文をクラスタリングして話題ごとにまとめる](notebooks/10_similarity_search/sentence_clustering.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_similarity_search/sentence_clustering.ipynb) |
| 10-08 | [重複・言い換えの文ペアを見つける](notebooks/10_similarity_search/paraphrase_mining.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_similarity_search/paraphrase_mining.ipynb) |
| 10-09 | [編集距離とあいまい文字列マッチング (RapidFuzz)](notebooks/10_similarity_search/edit_distance.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_similarity_search/edit_distance.ipynb) |
| 10-10 | [TF-IDF とコサイン類似度で似た文書を探す](notebooks/10_similarity_search/tfidf_similarity.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_similarity_search/tfidf_similarity.ipynb) |
| 10-11 | [MinHash と LSH で大量文書からほぼ重複を見つける](notebooks/10_similarity_search/minhash_lsh.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_similarity_search/minhash_lsh.ipynb) |
| 10-12 | [言語をまたいで意味の近い文を対応付ける (対訳文の抽出)](notebooks/10_similarity_search/cross_lingual_similarity.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_similarity_search/cross_lingual_similarity.ipynb) |
| 10-13 | [文埋め込みを 2 次元に可視化する (UMAP)](notebooks/10_similarity_search/embedding_umap.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_similarity_search/embedding_umap.ipynb) |
| 10-14 | [文の意味的類似度 (STS) でモデルを評価する](notebooks/10_similarity_search/sts_eval.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_similarity_search/sts_eval.ipynb) |
| 10-15 | [埋め込みを量子化・次元削減して省メモリ化する](notebooks/10_similarity_search/embedding_quantization.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_similarity_search/embedding_quantization.ipynb) |
| 10-16 | [巨大な辞書の語をテキストから一括で探す (Aho-Corasick)](notebooks/10_similarity_search/aho_corasick.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_similarity_search/aho_corasick.ipynb) |

### 11. トピック・キーワード・要約

文書集合に潜む話題を見つける、文書を代表する語や文を選ぶ。教師なしで全体像をつかむための手法。

| # | やりたいこと | Colab |
|---|---|---|
| 11-01 | [LDA で文書集合のトピックを見つける](notebooks/11_topics_keywords/lda_topics.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/11_topics_keywords/lda_topics.ipynb) |
| 11-02 | [NMF でトピックを抽出する](notebooks/11_topics_keywords/nmf_topics.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/11_topics_keywords/nmf_topics.ipynb) |
| 11-03 | [文埋め込みとクラスタリングでトピックを作る (BERTopic 方式)](notebooks/11_topics_keywords/embedding_topics.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/11_topics_keywords/embedding_topics.ipynb) |
| 11-04 | [文書からキーワードを抽出する (埋め込みによる KeyBERT 方式)](notebooks/11_topics_keywords/keyword_extraction.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/11_topics_keywords/keyword_extraction.ipynb) |
| 11-05 | [重要な文を選んで抽出型要約を作る (LexRank)](notebooks/11_topics_keywords/extractive_summary.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/11_topics_keywords/extractive_summary.ipynb) |

### 12. テキスト分類

文書にラベルを付ける。古典的な TF-IDF から BERT のファインチューニング、ゼロショットまで。

| # | やりたいこと | Colab |
|---|---|---|
| 12-01 | [TF-IDF とロジスティック回帰で日本語のニュース記事を分類する](notebooks/12_classification/tfidf_logreg_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_classification/tfidf_logreg_ja.ipynb) |
| 12-02 | [分類結果を評価する (適合率・再現率・F1・混同行列)](notebooks/12_classification/classification_report.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_classification/classification_report.ipynb) |
| 12-03 | [分類に効いている単語を調べる (線形モデルの係数)](notebooks/12_classification/feature_importance.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_classification/feature_importance.ipynb) |
| 12-04 | [複数の分類器を交差検証で比較する](notebooks/12_classification/compare_classifiers.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_classification/compare_classifiers.ipynb) |
| 12-05 | [文埋め込みを特徴量にして少ない学習データで分類する](notebooks/12_classification/embedding_classifier.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_classification/embedding_classifier.ipynb) |
| 12-06 | [ラベル付きデータなしで分類する (ゼロショット分類、NLI)](notebooks/12_classification/zero_shot_nli.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_classification/zero_shot_nli.ipynb) |
| 12-07 | [ラベルの説明文との類似度でゼロショット分類する](notebooks/12_classification/zero_shot_embedding.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_classification/zero_shot_embedding.ipynb) |
| 12-08 | [日本語の感情分析 (ポジネガ判定) をする](notebooks/12_classification/sentiment_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_classification/sentiment_ja.ipynb) |
| 12-09 | [BERT をファインチューニングして文書分類する (Trainer)](notebooks/12_classification/bert_finetune.ipynb) ⚡ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_classification/bert_finetune.ipynb) |
| 12-10 | [クラスの偏ったデータを扱う (クラス重み・しきい値調整)](notebooks/12_classification/imbalanced.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_classification/imbalanced.ipynb) |
| 12-11 | [1 つの文書に複数のラベルを付ける (マルチラベル分類)](notebooks/12_classification/multilabel.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_classification/multilabel.ipynb) |
| 12-12 | [ハイパーパラメータをグリッドサーチで調整する](notebooks/12_classification/grid_search.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_classification/grid_search.ipynb) |
| 12-13 | [個々の予測の根拠を可視化する (LIME)](notebooks/12_classification/lime_explain.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_classification/lime_explain.ipynb) |
| 12-14 | [学習データを増やすと精度がどう変わるか調べる (学習曲線)](notebooks/12_classification/learning_curve.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_classification/learning_curve.ipynb) |
| 12-15 | [学習した分類器を保存・読み込みする](notebooks/12_classification/save_sklearn_model.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_classification/save_sklearn_model.ipynb) |
| 12-16 | [モデルが自信満々に間違えた事例を調べる (エラー分析)](notebooks/12_classification/error_analysis.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_classification/error_analysis.ipynb) |

### 13. 系列ラベリング

トークンごとにラベルを付ける。BIO タグの扱い、CRF、BERT による固有表現抽出の学習と評価。

| # | やりたいこと | Colab |
|---|---|---|
| 13-01 | [文字位置のアノテーションを BIO タグ列に変換する](notebooks/13_sequence_labeling/spans_to_bio.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/13_sequence_labeling/spans_to_bio.ipynb) |
| 13-02 | [BIO タグ列から固有表現のスパンを取り出す](notebooks/13_sequence_labeling/bio_to_spans.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/13_sequence_labeling/bio_to_spans.ipynb) |
| 13-03 | [固有表現抽出をエンティティ単位で評価する (seqeval)](notebooks/13_sequence_labeling/seqeval.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/13_sequence_labeling/seqeval.ipynb) |
| 13-04 | [CRF で固有表現抽出器を学習する](notebooks/13_sequence_labeling/crf_tagger.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/13_sequence_labeling/crf_tagger.ipynb) |
| 13-05 | [ビタビアルゴリズムを実装して最適なラベル列を求める](notebooks/13_sequence_labeling/viterbi.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/13_sequence_labeling/viterbi.ipynb) |
| 13-06 | [HMM で品詞タガーを作る](notebooks/13_sequence_labeling/hmm_tagger.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/13_sequence_labeling/hmm_tagger.ipynb) |
| 13-07 | [単語分割を文字単位の系列ラベリングとして解く](notebooks/13_sequence_labeling/char_segmentation.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/13_sequence_labeling/char_segmentation.ipynb) |
| 13-08 | [BERT をファインチューニングして日本語の固有表現抽出器を作る](notebooks/13_sequence_labeling/bert_ner_finetune.ipynb) ⚡ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/13_sequence_labeling/bert_ner_finetune.ipynb) |

### 14. Transformers の基本操作

Hugging Face Transformers と Datasets で、事前学習済みモデルを動かすときの定番の操作。

| # | やりたいこと | Colab |
|---|---|---|
| 14-01 | [マスクされた単語を BERT に予測させる (fill-mask)](notebooks/14_transformers/fill_mask.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_transformers/fill_mask.ipynb) |
| 14-02 | [BERT の出力から文ベクトルを自分で作る (mean pooling)](notebooks/14_transformers/mean_pooling.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_transformers/mean_pooling.ipynb) |
| 14-03 | [テキスト生成のデコード方法 (貪欲法・ビームサーチ・サンプリング) を比べる](notebooks/14_transformers/generation_decoding.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_transformers/generation_decoding.ipynb) |
| 14-04 | [言語モデルで文の自然さ (パープレキシティ) を測る](notebooks/14_transformers/lm_perplexity.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_transformers/lm_perplexity.ipynb) |
| 14-05 | [次に来るトークンの確率分布を見る](notebooks/14_transformers/next_token_probs.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_transformers/next_token_probs.ipynb) |
| 14-06 | [トークンごとのサプライザルを色で可視化する](notebooks/14_transformers/surprisal.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_transformers/surprisal.ipynb) |
| 14-07 | [多言語翻訳モデル NLLB で翻訳する](notebooks/14_transformers/translation_nllb.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_transformers/translation_nllb.ipynb) |
| 14-08 | [文書の中から質問の答えを抜き出す (抽出型 QA)](notebooks/14_transformers/extractive_qa.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_transformers/extractive_qa.ipynb) |
| 14-09 | [アテンションの重みをヒートマップで可視化する](notebooks/14_transformers/attention_viz.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_transformers/attention_viz.ipynb) |
| 14-10 | [バッチ処理と GPU で推論を高速化する](notebooks/14_transformers/batch_inference.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_transformers/batch_inference.ipynb) |
| 14-11 | [モデルのパラメータ数とメモリ使用量を調べる](notebooks/14_transformers/model_size.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_transformers/model_size.ipynb) |
| 14-12 | [Datasets でデータを読み込む・加工する](notebooks/14_transformers/datasets_basics.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_transformers/datasets_basics.ipynb) |
| 14-13 | [巨大なデータセットをダウンロードせずにストリーミングで読む](notebooks/14_transformers/datasets_streaming.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_transformers/datasets_streaming.ipynb) |
| 14-14 | [Hub のモデルのキャッシュ場所を確認・変更する](notebooks/14_transformers/hf_cache.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_transformers/hf_cache.ipynb) |
| 14-15 | [モデルを保存・読み込みする、Hub にアップロードする](notebooks/14_transformers/save_load_model.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_transformers/save_load_model.ipynb) |
| 14-16 | [マスク言語モデルで文の自然さを測る (疑似対数尤度)](notebooks/14_transformers/mlm_scoring.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_transformers/mlm_scoring.ipynb) |
| 14-17 | [2 つの文の含意関係を判定する (NLI)](notebooks/14_transformers/nli.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_transformers/nli.ipynb) |

### 15. 大規模言語モデル (LLM)

小さなオープンモデルを Colab 上で動かす方法と、Claude API を使う方法。プロンプト、構造化出力、RAG、LoRA。

| # | やりたいこと | Colab |
|---|---|---|
| 15-01 | [オープンな LLM とチャットする (チャットテンプレート)](notebooks/15_llm/llm_chat_local.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_llm/llm_chat_local.ipynb) |
| 15-02 | [生成結果を 1 文字ずつ表示する (ストリーミング)](notebooks/15_llm/llm_streaming_local.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_llm/llm_streaming_local.ipynb) |
| 15-03 | [Few-shot プロンプトでテキストを分類する](notebooks/15_llm/llm_few_shot.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_llm/llm_few_shot.ipynb) |
| 15-04 | [選択肢の尤度を比べて LLM に分類させる](notebooks/15_llm/llm_choice_scoring.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_llm/llm_choice_scoring.ipynb) |
| 15-05 | [複数のプロンプトをまとめてバッチ生成する](notebooks/15_llm/llm_batch_generation.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_llm/llm_batch_generation.ipynb) |
| 15-06 | [LLM の出力から JSON を取り出して検証する](notebooks/15_llm/llm_json_output.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_llm/llm_json_output.ipynb) |
| 15-07 | [プロンプトをテンプレートで管理し、似た例を動的に選ぶ](notebooks/15_llm/prompt_template.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_llm/prompt_template.ipynb) |
| 15-08 | [長い文書をチャンクに分割する](notebooks/15_llm/chunking.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_llm/chunking.ipynb) |
| 15-09 | [検索した文書を LLM に渡して答えさせる (最小の RAG)](notebooks/15_llm/rag_minimal.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_llm/rag_minimal.ipynb) |
| 15-10 | [LLM を 4bit 量子化して少ないメモリで動かす](notebooks/15_llm/llm_4bit.ipynb) ⚡ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_llm/llm_4bit.ipynb) |
| 15-11 | [LoRA で LLM を少ないメモリでファインチューニングする](notebooks/15_llm/lora_finetune.ipynb) ⚡ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_llm/lora_finetune.ipynb) |
| 15-12 | [Claude API を呼び出す](notebooks/15_llm/claude_basic.ipynb) 🔑 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_llm/claude_basic.ipynb) |
| 15-13 | [Claude の応答をストリーミングで受け取る](notebooks/15_llm/claude_streaming.ipynb) 🔑 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_llm/claude_streaming.ipynb) |
| 15-14 | [Claude に決まった形式 (JSON スキーマ) で出力させる](notebooks/15_llm/claude_structured.ipynb) 🔑 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_llm/claude_structured.ipynb) |
| 15-15 | [Claude で大量のテキストを安く処理する (Message Batches API)](notebooks/15_llm/claude_batch.ipynb) 🔑 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_llm/claude_batch.ipynb) |
| 15-16 | [Claude API のトークン数と料金を事前に見積もる](notebooks/15_llm/claude_count_tokens.ipynb) 🔑 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_llm/claude_count_tokens.ipynb) |
| 15-17 | [LLM に生成文の品質を採点させる (LLM-as-a-judge)](notebooks/15_llm/llm_judge.ipynb) 🔑 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_llm/llm_judge.ipynb) |

### 16. 評価指標と統計的検定

翻訳・要約・検索・分類の出力を数値で評価し、その差が偶然でないかを確かめる。

| # | やりたいこと | Colab |
|---|---|---|
| 16-01 | [機械翻訳を BLEU で評価する (sacreBLEU)](notebooks/16_evaluation/bleu.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_evaluation/bleu.ipynb) |
| 16-02 | [chrF と TER で翻訳を評価する](notebooks/16_evaluation/chrf_ter.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_evaluation/chrf_ter.ipynb) |
| 16-03 | [要約を ROUGE で評価する (日本語対応)](notebooks/16_evaluation/rouge_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_evaluation/rouge_ja.ipynb) |
| 16-04 | [BERTScore で意味的な近さを評価する](notebooks/16_evaluation/bertscore.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_evaluation/bertscore.ipynb) |
| 16-05 | [COMET で翻訳の品質を評価する](notebooks/16_evaluation/comet.ipynb) ⚡ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_evaluation/comet.ipynb) |
| 16-06 | [音声認識や OCR の誤り率 (WER / CER) を計算する](notebooks/16_evaluation/wer_cer.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_evaluation/wer_cer.ipynb) |
| 16-07 | [適合率・再現率・F1 を自分で計算する (micro と macro)](notebooks/16_evaluation/prf_manual.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_evaluation/prf_manual.ipynb) |
| 16-08 | [アノテーションの一致度を測る (κ 係数、Krippendorff の α)](notebooks/16_evaluation/agreement.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_evaluation/agreement.ipynb) |
| 16-09 | [ブートストラップ法で評価値の信頼区間を求める](notebooks/16_evaluation/bootstrap_ci.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_evaluation/bootstrap_ci.ipynb) |
| 16-10 | [2 つのシステムの差が有意かどうかを検定する](notebooks/16_evaluation/significance_test.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_evaluation/significance_test.ipynb) |
| 16-11 | [乱数シードによるばらつきを考慮して結果を報告する](notebooks/16_evaluation/seed_variance.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_evaluation/seed_variance.ipynb) |
| 16-12 | [自動評価指標と人手評価の相関を調べる](notebooks/16_evaluation/metric_correlation.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_evaluation/metric_correlation.ipynb) |
| 16-13 | [質問応答を完全一致と F1 で評価する](notebooks/16_evaluation/qa_em_f1.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_evaluation/qa_em_f1.ipynb) |
| 16-14 | [検索・ランキングを評価する (Recall@k、MRR、nDCG)](notebooks/16_evaluation/ranking_metrics.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_evaluation/ranking_metrics.ipynb) |
| 16-15 | [生成テキストの多様性と繰り返しを測る (distinct-n)](notebooks/16_evaluation/distinct_n.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_evaluation/distinct_n.ipynb) |
| 16-16 | [モデルの確信度が当てになるか調べる (較正、ECE)](notebooks/16_evaluation/calibration_ece.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_evaluation/calibration_ece.ipynb) |

### 17. 文法誤り訂正と言語教育

学習者の作文を訂正する、訂正箇所を取り出して分類・評価する、教材づくりに使う。

| # | やりたいこと | Colab |
|---|---|---|
| 17-01 | [2 つの文の違いを色付きで表示する](notebooks/17_gec_education/diff_highlight.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/17_gec_education/diff_highlight.ipynb) |
| 17-02 | [訂正前後の文から編集 (置換・挿入・削除) を取り出す](notebooks/17_gec_education/edit_extraction.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/17_gec_education/edit_extraction.ipynb) |
| 17-03 | [ERRANT で英語の訂正を誤りタイプ付きの編集に変換する](notebooks/17_gec_education/errant.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/17_gec_education/errant.ipynb) |
| 17-04 | [M2 形式のファイルを読み書きする](notebooks/17_gec_education/m2_format.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/17_gec_education/m2_format.ipynb) |
| 17-05 | [文法誤り訂正を編集単位の F0.5 で評価する](notebooks/17_gec_education/gec_f05.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/17_gec_education/gec_f05.ipynb) |
| 17-06 | [学習済みモデルで英語の文法誤りを訂正する](notebooks/17_gec_education/gec_model_en.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/17_gec_education/gec_model_en.ipynb) |
| 17-07 | [規則で誤りを注入して GEC の擬似学習データを作る](notebooks/17_gec_education/synthetic_errors.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/17_gec_education/synthetic_errors.ipynb) |
| 17-08 | [日本語の文章の読みやすさを測る](notebooks/17_gec_education/readability_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/17_gec_education/readability_ja.ipynb) |
| 17-09 | [単語の頻度から語彙の難しさを推定する (wordfreq)](notebooks/17_gec_education/word_frequency_level.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/17_gec_education/word_frequency_level.ipynb) |
| 17-10 | [穴埋め問題 (クローズテスト) を自動で作る](notebooks/17_gec_education/cloze_generation.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/17_gec_education/cloze_generation.ipynb) |

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
