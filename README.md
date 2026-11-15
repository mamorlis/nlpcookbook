# NLP クックブック

自然言語処理の「やりたいこと」に対して「こうやればいい」をまとめた小ネタ集です。
1 レシピ = 1 ノートブックで、すべて Google Colab でそのまま実行できます。

- 日本語のテキスト処理を中心に、英語・多言語も扱います。
- 各ノートブックは上から順に実行するだけで動きます (必要なライブラリのインストールを含む)。
- 実行結果を埋め込んであるので、GitHub 上で眺めるだけでも使えます。

## 目次

<!-- INDEX:START -->
全 325 レシピ。タイトルをクリックすると GitHub 上で、バッジをクリックすると Google Colab で開きます。

- [1. Google Colab の小技](#1-google-colab-の小技) (11)
- [2. データの収集と入出力](#2-データの収集と入出力) (10)
- [3. 文字列処理と正規化](#3-文字列処理と正規化) (24)
- [4. 日本語の形態素解析と文分割](#4-日本語の形態素解析と文分割) (24)
- [5. 英語・多言語の前処理](#5-英語多言語の前処理) (20)
- [6. サブワードとトークナイザ](#6-サブワードとトークナイザ) (14)
- [7. コーパスの統計と可視化](#7-コーパスの統計と可視化) (12)
- [8. 単語ベクトル](#8-単語ベクトル) (12)
- [9. 文埋め込みとクラスタリング](#9-文埋め込みとクラスタリング) (6)
- [10. トピックとキーワードの抽出](#10-トピックとキーワードの抽出) (7)
- [11. テキスト分類](#11-テキスト分類) (16)
- [12. Transformers の基本操作](#12-transformers-の基本操作) (15)
- [13. 大規模言語モデル (LLM)](#13-大規模言語モデル-llm) (14)
- [14. 構文解析](#14-構文解析) (20)
- [15. 固有表現抽出](#15-固有表現抽出) (12)
- [16. 意味解析](#16-意味解析) (11)
- [17. 評価指標と統計的検定](#17-評価指標と統計的検定) (12)
- [18. 検索](#18-検索) (16)
- [19. 質問応答](#19-質問応答) (8)
- [20. RAG (検索拡張生成)](#20-rag-検索拡張生成) (13)
- [21. 対話システム](#21-対話システム) (10)
- [22. 機械翻訳](#22-機械翻訳) (9)
- [23. 要約と平易化](#23-要約と平易化) (9)
- [24. 文法誤り訂正と言語教育](#24-文法誤り訂正と言語教育) (10)
- [25. 日本語入力 (かな漢字変換)](#25-日本語入力-かな漢字変換) (10)

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
| 03-17 | [文字 n-gram を作る](notebooks/03_string/char_ngram.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/char_ngram.ipynb) |
| 03-18 | [日付・時刻の表現を datetime に変換する](notebooks/03_string/date_parse.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/date_parse.ipynb) |
| 03-19 | [電話番号・郵便番号・メールアドレスを正規表現で抜き出す](notebooks/03_string/regex_patterns.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/regex_patterns.ipynb) |
| 03-20 | [テキストの言語 (文字種) の割合を調べる](notebooks/03_string/script_ratio.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/script_ratio.ipynb) |
| 03-21 | [ゼロ幅文字・制御文字・異体字セレクタを取り除く](notebooks/03_string/invisible_chars.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/invisible_chars.ipynb) |
| 03-22 | [編集距離とあいまい文字列マッチング (RapidFuzz)](notebooks/03_string/edit_distance.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/edit_distance.ipynb) |
| 03-23 | [MinHash と LSH で大量文書からほぼ重複を見つける](notebooks/03_string/minhash_lsh.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/minhash_lsh.ipynb) |
| 03-24 | [巨大な辞書の語をテキストから一括で探す (Aho-Corasick)](notebooks/03_string/aho_corasick.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_string/aho_corasick.ipynb) |

### 4. 日本語の形態素解析と文分割

MeCab (fugashi)・Sudachi・Janome で単語に区切り、品詞や原形、読みを取り出す。文への分割、単語分割や品詞タグ付けの仕組みも扱う。

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
| 04-20 | [日本語の文を句点で分割する (括弧内は分割しない)](notebooks/04_japanese_tokenization/sentence_split_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/sentence_split_ja.ipynb) |
| 04-21 | [日本語の文分割をライブラリで行って比べる](notebooks/04_japanese_tokenization/sentence_split_libs.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/sentence_split_libs.ipynb) |
| 04-22 | [単語分割を文字単位の系列ラベリングとして解く](notebooks/04_japanese_tokenization/char_segmentation.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/char_segmentation.ipynb) |
| 04-23 | [HMM で品詞タガーを作る](notebooks/04_japanese_tokenization/hmm_tagger.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/hmm_tagger.ipynb) |
| 04-24 | [ビタビアルゴリズムを実装して最適なラベル列を求める](notebooks/04_japanese_tokenization/viterbi.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_japanese_tokenization/viterbi.ipynb) |

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

単語を数える、共起や特徴語を調べる、分布を図にする。コーパスを手に入れたらまずやること。

| # | やりたいこと | Colab |
|---|---|---|
| 07-01 | [単語の頻度を数える](notebooks/07_corpus_statistics/word_freq.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/word_freq.ipynb) |
| 07-02 | [Zipf の法則を確かめる (順位-頻度プロット)](notebooks/07_corpus_statistics/zipf.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/zipf.ipynb) |
| 07-03 | [matplotlib で日本語を文字化けさせずに表示する](notebooks/07_corpus_statistics/matplotlib_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/matplotlib_ja.ipynb) |
| 07-04 | [単語の共起を数えて PMI で結びつきの強さを測る](notebooks/07_corpus_statistics/pmi.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/pmi.ipynb) |
| 07-05 | [よく使われる言い回し (コロケーション) を抽出する](notebooks/07_corpus_statistics/collocations.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/collocations.ipynb) |
| 07-06 | [キーワードの前後の文脈を一覧する (KWIC)](notebooks/07_corpus_statistics/kwic.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/kwic.ipynb) |
| 07-07 | [語彙の豊富さを測る (TTR とその仲間)](notebooks/07_corpus_statistics/lexical_diversity.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/lexical_diversity.ipynb) |
| 07-08 | [n-gram 言語モデルを作ってパープレキシティを測り、文を生成する](notebooks/07_corpus_statistics/ngram_lm.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/ngram_lm.ipynb) |
| 07-09 | [2 つのコーパスを比べて特徴語を見つける (対数尤度比)](notebooks/07_corpus_statistics/keyness.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/keyness.ipynb) |
| 07-10 | [文の長さや文字数の基本統計をとる](notebooks/07_corpus_statistics/corpus_stats.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/corpus_stats.ipynb) |
| 07-11 | [品詞の分布を調べる・比べる](notebooks/07_corpus_statistics/pos_distribution.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/pos_distribution.ipynb) |
| 07-12 | [文字や単語のエントロピーを計算する](notebooks/07_corpus_statistics/entropy.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_corpus_statistics/entropy.ipynb) |

### 8. 単語ベクトル

単語を数百次元のベクトルで表し、類似度やアナロジーを計算する。

| # | やりたいこと | Colab |
|---|---|---|
| 08-01 | [word2vec を自分のコーパスで学習する](notebooks/08_word_vectors/word2vec_train.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_word_vectors/word2vec_train.ipynb) |
| 08-02 | [学習済みの英語単語ベクトルを読み込んで類似語を調べる](notebooks/08_word_vectors/pretrained_vectors_en.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_word_vectors/pretrained_vectors_en.ipynb) |
| 08-03 | [学習済みの日本語単語ベクトルを使う (chiVe)](notebooks/08_word_vectors/pretrained_vectors_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_word_vectors/pretrained_vectors_ja.ipynb) |
| 08-04 | [単語ベクトルでアナロジー (王 − 男 + 女 = 女王) を解く](notebooks/08_word_vectors/analogy.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_word_vectors/analogy.ipynb) |
| 08-05 | [fastText で未知語にもベクトルを与える](notebooks/08_word_vectors/fasttext_train.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_word_vectors/fasttext_train.ipynb) |
| 08-06 | [単語ベクトルを 2 次元に落として可視化する](notebooks/08_word_vectors/word_vectors_viz.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_word_vectors/word_vectors_viz.ipynb) |
| 08-07 | [単語ベクトルの平均で文ベクトルを作る](notebooks/08_word_vectors/avg_word_vectors.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_word_vectors/avg_word_vectors.ipynb) |
| 08-08 | [PPMI 行列の SVD で単語ベクトルを作る (カウントベース)](notebooks/08_word_vectors/ppmi_svd.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_word_vectors/ppmi_svd.ipynb) |
| 08-09 | [Doc2Vec で文書ベクトルを学習する](notebooks/08_word_vectors/doc2vec.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_word_vectors/doc2vec.ipynb) |
| 08-10 | [単語ベクトルに含まれるジェンダーバイアスを測る](notebooks/08_word_vectors/embedding_bias.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_word_vectors/embedding_bias.ipynb) |
| 08-11 | [BERT から文脈に応じた単語ベクトルを取り出す](notebooks/08_word_vectors/contextual_word_vectors.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_word_vectors/contextual_word_vectors.ipynb) |
| 08-12 | [単語類似度データセットで単語ベクトルを評価する](notebooks/08_word_vectors/wordsim_eval.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_word_vectors/wordsim_eval.ipynb) |

### 9. 文埋め込みとクラスタリング

文をベクトルにして意味の近さを測る。言い換えの検出、クラスタリング、可視化。検索への応用は「検索」の章を参照。

| # | やりたいこと | Colab |
|---|---|---|
| 09-01 | [文をベクトルにして意味の近さを測る (Sentence Transformers)](notebooks/09_sentence_embeddings/sentence_embeddings.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_sentence_embeddings/sentence_embeddings.ipynb) |
| 09-02 | [文をクラスタリングして話題ごとにまとめる](notebooks/09_sentence_embeddings/sentence_clustering.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_sentence_embeddings/sentence_clustering.ipynb) |
| 09-03 | [重複・言い換えの文ペアを見つける](notebooks/09_sentence_embeddings/paraphrase_mining.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_sentence_embeddings/paraphrase_mining.ipynb) |
| 09-04 | [文埋め込みを PCA・t-SNE・UMAP で 2 次元に可視化して比べる](notebooks/09_sentence_embeddings/embedding_visualization.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_sentence_embeddings/embedding_visualization.ipynb) |
| 09-05 | [文の意味的類似度 (STS) でモデルを評価する](notebooks/09_sentence_embeddings/sts_eval.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_sentence_embeddings/sts_eval.ipynb) |
| 09-06 | [埋め込みを量子化・次元削減して省メモリ化する](notebooks/09_sentence_embeddings/embedding_quantization.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_sentence_embeddings/embedding_quantization.ipynb) |

### 10. トピックとキーワードの抽出

文書集合に潜む話題を見つける、文書を代表する語を選ぶ、ワードクラウドや共起ネットワークで眺める。

| # | やりたいこと | Colab |
|---|---|---|
| 10-01 | [日本語のワードクラウドを作る](notebooks/10_topics_keywords/wordcloud_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_topics_keywords/wordcloud_ja.ipynb) |
| 10-02 | [TF-IDF で文書ごとの特徴語を取り出す](notebooks/10_topics_keywords/tfidf_keywords.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_topics_keywords/tfidf_keywords.ipynb) |
| 10-03 | [文書からキーワードを抽出する (埋め込みによる KeyBERT 方式)](notebooks/10_topics_keywords/keyword_extraction.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_topics_keywords/keyword_extraction.ipynb) |
| 10-04 | [共起ネットワークを描く](notebooks/10_topics_keywords/cooccurrence_network.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_topics_keywords/cooccurrence_network.ipynb) |
| 10-05 | [LDA で文書集合のトピックを見つける](notebooks/10_topics_keywords/lda_topics.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_topics_keywords/lda_topics.ipynb) |
| 10-06 | [NMF でトピックを抽出する](notebooks/10_topics_keywords/nmf_topics.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_topics_keywords/nmf_topics.ipynb) |
| 10-07 | [文埋め込みとクラスタリングでトピックを作る (BERTopic 方式)](notebooks/10_topics_keywords/embedding_topics.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_topics_keywords/embedding_topics.ipynb) |

### 11. テキスト分類

文書にラベルを付ける。古典的な TF-IDF から BERT のファインチューニング、ゼロショットまで。

| # | やりたいこと | Colab |
|---|---|---|
| 11-01 | [TF-IDF とロジスティック回帰で日本語のニュース記事を分類する](notebooks/11_classification/tfidf_logreg_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/11_classification/tfidf_logreg_ja.ipynb) |
| 11-02 | [分類結果を評価する (適合率・再現率・F1・混同行列)](notebooks/11_classification/classification_report.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/11_classification/classification_report.ipynb) |
| 11-03 | [分類に効いている単語を調べる (線形モデルの係数)](notebooks/11_classification/feature_importance.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/11_classification/feature_importance.ipynb) |
| 11-04 | [複数の分類器を交差検証で比較する](notebooks/11_classification/compare_classifiers.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/11_classification/compare_classifiers.ipynb) |
| 11-05 | [文埋め込みを特徴量にして少ない学習データで分類する](notebooks/11_classification/embedding_classifier.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/11_classification/embedding_classifier.ipynb) |
| 11-06 | [ラベル付きデータなしで分類する (ゼロショット分類、NLI)](notebooks/11_classification/zero_shot_nli.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/11_classification/zero_shot_nli.ipynb) |
| 11-07 | [ラベルの説明文との類似度でゼロショット分類する](notebooks/11_classification/zero_shot_embedding.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/11_classification/zero_shot_embedding.ipynb) |
| 11-08 | [日本語の感情分析 (ポジネガ判定) をする](notebooks/11_classification/sentiment_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/11_classification/sentiment_ja.ipynb) |
| 11-09 | [BERT をファインチューニングして文書分類する (Trainer)](notebooks/11_classification/bert_finetune.ipynb) ⚡ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/11_classification/bert_finetune.ipynb) |
| 11-10 | [クラスの偏ったデータを扱う (クラス重み・しきい値調整)](notebooks/11_classification/imbalanced.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/11_classification/imbalanced.ipynb) |
| 11-11 | [1 つの文書に複数のラベルを付ける (マルチラベル分類)](notebooks/11_classification/multilabel.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/11_classification/multilabel.ipynb) |
| 11-12 | [ハイパーパラメータをグリッドサーチで調整する](notebooks/11_classification/grid_search.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/11_classification/grid_search.ipynb) |
| 11-13 | [個々の予測の根拠を可視化する (LIME)](notebooks/11_classification/lime_explain.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/11_classification/lime_explain.ipynb) |
| 11-14 | [学習データを増やすと精度がどう変わるか調べる (学習曲線)](notebooks/11_classification/learning_curve.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/11_classification/learning_curve.ipynb) |
| 11-15 | [学習した分類器を保存・読み込みする](notebooks/11_classification/save_sklearn_model.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/11_classification/save_sklearn_model.ipynb) |
| 11-16 | [モデルが自信満々に間違えた事例を調べる (エラー分析)](notebooks/11_classification/error_analysis.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/11_classification/error_analysis.ipynb) |

### 12. Transformers の基本操作

Hugging Face Transformers と Datasets で、事前学習済みモデルを動かすときの定番の操作。

| # | やりたいこと | Colab |
|---|---|---|
| 12-01 | [マスクされた単語を BERT に予測させる (fill-mask)](notebooks/12_transformers/fill_mask.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_transformers/fill_mask.ipynb) |
| 12-02 | [BERT の出力から文ベクトルを自分で作る (mean pooling)](notebooks/12_transformers/mean_pooling.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_transformers/mean_pooling.ipynb) |
| 12-03 | [テキスト生成のデコード方法 (貪欲法・ビームサーチ・サンプリング) を比べる](notebooks/12_transformers/generation_decoding.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_transformers/generation_decoding.ipynb) |
| 12-04 | [言語モデルで文の自然さ (パープレキシティ) を測る](notebooks/12_transformers/lm_perplexity.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_transformers/lm_perplexity.ipynb) |
| 12-05 | [次に来るトークンの確率分布を見る](notebooks/12_transformers/next_token_probs.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_transformers/next_token_probs.ipynb) |
| 12-06 | [トークンごとのサプライザルを色で可視化する](notebooks/12_transformers/surprisal.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_transformers/surprisal.ipynb) |
| 12-07 | [アテンションの重みをヒートマップで可視化する](notebooks/12_transformers/attention_viz.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_transformers/attention_viz.ipynb) |
| 12-08 | [バッチ処理と GPU で推論を高速化する](notebooks/12_transformers/batch_inference.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_transformers/batch_inference.ipynb) |
| 12-09 | [モデルのパラメータ数とメモリ使用量を調べる](notebooks/12_transformers/model_size.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_transformers/model_size.ipynb) |
| 12-10 | [Datasets でデータを読み込む・加工する](notebooks/12_transformers/datasets_basics.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_transformers/datasets_basics.ipynb) |
| 12-11 | [巨大なデータセットをダウンロードせずにストリーミングで読む](notebooks/12_transformers/datasets_streaming.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_transformers/datasets_streaming.ipynb) |
| 12-12 | [Hub のモデルのキャッシュ場所を確認・変更する](notebooks/12_transformers/hf_cache.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_transformers/hf_cache.ipynb) |
| 12-13 | [モデルを保存・読み込みする、Hub にアップロードする](notebooks/12_transformers/save_load_model.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_transformers/save_load_model.ipynb) |
| 12-14 | [マスク言語モデルで文の自然さを測る (疑似対数尤度)](notebooks/12_transformers/mlm_scoring.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_transformers/mlm_scoring.ipynb) |
| 12-15 | [2 つの文の含意関係を判定する (NLI)](notebooks/12_transformers/nli.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/12_transformers/nli.ipynb) |

### 13. 大規模言語モデル (LLM)

小さなオープンモデルを Colab 上で動かす方法と、Claude API を使う方法。プロンプト、構造化出力、LoRA、LLM による評価。検索と組み合わせる方法は「RAG」の章を参照。

| # | やりたいこと | Colab |
|---|---|---|
| 13-01 | [オープンな LLM とチャットする (チャットテンプレート)](notebooks/13_llm/llm_chat_local.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/13_llm/llm_chat_local.ipynb) |
| 13-02 | [生成結果を 1 文字ずつ表示する (ストリーミング)](notebooks/13_llm/llm_streaming_local.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/13_llm/llm_streaming_local.ipynb) |
| 13-03 | [Few-shot プロンプトでテキストを分類する](notebooks/13_llm/llm_few_shot.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/13_llm/llm_few_shot.ipynb) |
| 13-04 | [選択肢の尤度を比べて LLM に分類させる](notebooks/13_llm/llm_choice_scoring.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/13_llm/llm_choice_scoring.ipynb) |
| 13-05 | [複数のプロンプトをまとめてバッチ生成する](notebooks/13_llm/llm_batch_generation.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/13_llm/llm_batch_generation.ipynb) |
| 13-06 | [LLM の出力から JSON を取り出して検証する](notebooks/13_llm/llm_json_output.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/13_llm/llm_json_output.ipynb) |
| 13-07 | [プロンプトをテンプレートで管理し、似た例を動的に選ぶ](notebooks/13_llm/prompt_template.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/13_llm/prompt_template.ipynb) |
| 13-08 | [LLM を 4bit 量子化して少ないメモリで動かす](notebooks/13_llm/llm_4bit.ipynb) ⚡ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/13_llm/llm_4bit.ipynb) |
| 13-09 | [LoRA で LLM を少ないメモリでファインチューニングする](notebooks/13_llm/lora_finetune.ipynb) ⚡ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/13_llm/lora_finetune.ipynb) |
| 13-10 | [Claude API を呼び出す](notebooks/13_llm/claude_basic.ipynb) 🔑 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/13_llm/claude_basic.ipynb) |
| 13-11 | [Claude の応答をストリーミングで受け取る](notebooks/13_llm/claude_streaming.ipynb) 🔑 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/13_llm/claude_streaming.ipynb) |
| 13-12 | [Claude に決まった形式 (JSON スキーマ) で出力させる](notebooks/13_llm/claude_structured.ipynb) 🔑 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/13_llm/claude_structured.ipynb) |
| 13-13 | [Claude で大量のテキストを安く処理する (Message Batches API)](notebooks/13_llm/claude_batch.ipynb) 🔑 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/13_llm/claude_batch.ipynb) |
| 13-14 | [Claude API のトークン数と料金を事前に見積もる](notebooks/13_llm/claude_count_tokens.ipynb) 🔑 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/13_llm/claude_count_tokens.ipynb) |

### 14. 構文解析

係り受け (依存構造) と句構造の解析。文の骨組みを取り出し、評価し、解析の仕組みを理解する。

| # | やりたいこと | Colab |
|---|---|---|
| 14-01 | [GiNZA で日本語の依存構造解析 (係り受け解析) をする](notebooks/14_parsing/ginza_dependency.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_parsing/ginza_dependency.ipynb) |
| 14-02 | [依存構造木を図で表示する (displaCy)](notebooks/14_parsing/displacy_dep.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_parsing/displacy_dep.ipynb) |
| 14-03 | [文節に区切って文節単位の係り受けを得る](notebooks/14_parsing/bunsetsu.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_parsing/bunsetsu.ipynb) |
| 14-04 | [KWJA で文節・基本句の係り受けと並列構造を解析する](notebooks/14_parsing/kwja_dependency.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_parsing/kwja_dependency.ipynb) |
| 14-05 | [Stanza で多言語の形態素・依存構造解析をする](notebooks/14_parsing/stanza.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_parsing/stanza.ipynb) |
| 14-06 | [係り受けのパターンで特定の構文を探す (DependencyMatcher)](notebooks/14_parsing/dependency_matcher.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_parsing/dependency_matcher.ipynb) |
| 14-07 | [「誰が・何を・どうした」の三つ組を抜き出す](notebooks/14_parsing/svo_extract.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_parsing/svo_extract.ipynb) |
| 14-08 | [名詞句を抽出する](notebooks/14_parsing/noun_chunks.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_parsing/noun_chunks.ipynb) |
| 14-09 | [依存距離と構文木の深さで文の複雑さを測る](notebooks/14_parsing/dep_distance.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_parsing/dep_distance.ipynb) |
| 14-10 | [解析結果を CoNLL-U 形式で保存・読み込みする](notebooks/14_parsing/conllu.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_parsing/conllu.ipynb) |
| 14-11 | [依存構造解析の精度を UAS と LAS で評価する](notebooks/14_parsing/dep_eval.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_parsing/dep_eval.ipynb) |
| 14-12 | [遷移型の依存構造解析の動きを追う (arc-standard)](notebooks/14_parsing/transition_parsing.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_parsing/transition_parsing.ipynb) |
| 14-13 | [スコアが最大の依存構造木を求める (最大全域木)](notebooks/14_parsing/mst_decoding.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_parsing/mst_decoding.ipynb) |
| 14-14 | [spaCy で依存構造パーザを自分で学習する](notebooks/14_parsing/spacy_train_parser.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_parsing/spacy_train_parser.ipynb) |
| 14-15 | [文脈自由文法を書いて構文解析する (NLTK)](notebooks/14_parsing/cfg_parse.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_parsing/cfg_parse.ipynb) |
| 14-16 | [構文木を括弧表記で読み書きし、操作する (NLTK Tree)](notebooks/14_parsing/tree_bracket.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_parsing/tree_bracket.ipynb) |
| 14-17 | [学習済みモデルで句構造解析をする (Stanza)](notebooks/14_parsing/constituency_stanza.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_parsing/constituency_stanza.ipynb) |
| 14-18 | [Ckylark で日本語の句構造解析をする](notebooks/14_parsing/ckylark.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_parsing/ckylark.ipynb) |
| 14-19 | [組合せ範疇文法 (CCG) で文を導出する (NLTK)](notebooks/14_parsing/ccg_nltk.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_parsing/ccg_nltk.ipynb) |
| 14-20 | [spaCy で大量の文書を高速に処理する](notebooks/14_parsing/spacy_pipe.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/14_parsing/spacy_pipe.ipynb) |

### 15. 固有表現抽出

人名・組織名・地名などの固有表現を見つける。学習済みモデルの利用、辞書やパターンによる抽出、BIO タグの扱い、CRF や BERT による学習と評価。

| # | やりたいこと | Colab |
|---|---|---|
| 15-01 | [日本語の固有表現 (人名・地名・組織名など) を抽出する](notebooks/15_ner/ner_ginza.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_ner/ner_ginza.ipynb) |
| 15-02 | [英語の固有表現を抽出して色付きで表示する](notebooks/15_ner/ner_spacy_en.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_ner/ner_spacy_en.ipynb) |
| 15-03 | [Transformers の pipeline で日本語の固有表現抽出をする](notebooks/15_ner/ner_transformers_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_ner/ner_transformers_ja.ipynb) |
| 15-04 | [学習なしで好きなラベルの固有表現を抽出する (GLiNER)](notebooks/15_ner/gliner.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_ner/gliner.ipynb) |
| 15-05 | [辞書やルールで固有表現を追加する (EntityRuler)](notebooks/15_ner/entity_ruler.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_ner/entity_ruler.ipynb) |
| 15-06 | [品詞や原形のパターンで表現を抽出する (spaCy Matcher)](notebooks/15_ner/spacy_matcher.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_ner/spacy_matcher.ipynb) |
| 15-07 | [個人情報をマスクする (仮名化)](notebooks/15_ner/pii_mask.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_ner/pii_mask.ipynb) |
| 15-08 | [文字位置のアノテーションを BIO タグ列に変換する](notebooks/15_ner/spans_to_bio.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_ner/spans_to_bio.ipynb) |
| 15-09 | [BIO タグ列から固有表現のスパンを取り出す](notebooks/15_ner/bio_to_spans.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_ner/bio_to_spans.ipynb) |
| 15-10 | [固有表現抽出をエンティティ単位で評価する (seqeval)](notebooks/15_ner/seqeval.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_ner/seqeval.ipynb) |
| 15-11 | [CRF で固有表現抽出器を学習する](notebooks/15_ner/crf_tagger.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_ner/crf_tagger.ipynb) |
| 15-12 | [BERT をファインチューニングして日本語の固有表現抽出器を作る](notebooks/15_ner/bert_ner_finetune.ipynb) ⚡ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/15_ner/bert_ner_finetune.ipynb) |

### 16. 意味解析

「誰が何をどうした」「この語はどの意味か」「それは何を指すか」を取り出す。構文の一歩先の解析。

| # | やりたいこと | Colab |
|---|---|---|
| 16-01 | [日本語の述語項構造解析をする (KWJA)](notebooks/16_semantics/pas_kwja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_semantics/pas_kwja.ipynb) |
| 16-02 | [日本語の共参照解析をする (KWJA)](notebooks/16_semantics/coref_kwja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_semantics/coref_kwja.ipynb) |
| 16-03 | [英語の共参照解析をして代名詞を先行詞に置き換える (Stanza)](notebooks/16_semantics/coref_en.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_semantics/coref_en.ipynb) |
| 16-04 | [多義語の語義を文脈から判定する (語義曖昧性解消)](notebooks/16_semantics/wsd_en.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_semantics/wsd_en.ipynb) |
| 16-05 | [文脈化ベクトルをクラスタリングして語義を自動で分ける (語義推定)](notebooks/16_semantics/word_sense_induction.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_semantics/word_sense_induction.ipynb) |
| 16-06 | [FrameNet と PropBank で述語の意味フレームと意味役割を調べる](notebooks/16_semantics/framenet_propbank.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_semantics/framenet_propbank.ipynb) |
| 16-07 | [文から関係の三つ組を抽出する (REBEL)](notebooks/16_semantics/relation_extraction.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_semantics/relation_extraction.ipynb) |
| 16-08 | [固有表現を Wikidata のエンティティに結び付ける (エンティティリンキング)](notebooks/16_semantics/entity_linking.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_semantics/entity_linking.ipynb) |
| 16-09 | [日本語の時間表現を抽出して正規化する (ja-timex)](notebooks/16_semantics/timex_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_semantics/timex_ja.ipynb) |
| 16-10 | [否定・モダリティ・時制を取り出す](notebooks/16_semantics/negation_modality_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_semantics/negation_modality_ja.ipynb) |
| 16-11 | [文を一階述語論理の式に変換して真偽を確かめる (NLTK)](notebooks/16_semantics/logic_semantics.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/16_semantics/logic_semantics.ipynb) |

### 17. 評価指標と統計的検定

分類・抽出・生成の出力を数値で評価し、その差が偶然でないかを確かめる。LLM による評価も扱う。翻訳・要約・質問応答・検索に固有の指標は、それぞれの章を参照。

| # | やりたいこと | Colab |
|---|---|---|
| 17-01 | [BERTScore で意味的な近さを評価する](notebooks/17_evaluation/bertscore.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/17_evaluation/bertscore.ipynb) |
| 17-02 | [音声認識や OCR の誤り率 (WER / CER) を計算する](notebooks/17_evaluation/wer_cer.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/17_evaluation/wer_cer.ipynb) |
| 17-03 | [適合率・再現率・F1 を自分で計算する (micro と macro)](notebooks/17_evaluation/prf_manual.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/17_evaluation/prf_manual.ipynb) |
| 17-04 | [アノテーションの一致度を測る (κ 係数、Krippendorff の α)](notebooks/17_evaluation/agreement.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/17_evaluation/agreement.ipynb) |
| 17-05 | [ブートストラップ法で評価値の信頼区間を求める](notebooks/17_evaluation/bootstrap_ci.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/17_evaluation/bootstrap_ci.ipynb) |
| 17-06 | [2 つのシステムの差が有意かどうかを検定する](notebooks/17_evaluation/significance_test.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/17_evaluation/significance_test.ipynb) |
| 17-07 | [乱数シードによるばらつきを考慮して結果を報告する](notebooks/17_evaluation/seed_variance.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/17_evaluation/seed_variance.ipynb) |
| 17-08 | [自動評価指標と人手評価の相関を調べる](notebooks/17_evaluation/metric_correlation.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/17_evaluation/metric_correlation.ipynb) |
| 17-09 | [生成テキストの多様性と繰り返しを測る (distinct-n)](notebooks/17_evaluation/distinct_n.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/17_evaluation/distinct_n.ipynb) |
| 17-10 | [モデルの確信度が当てになるか調べる (較正、ECE)](notebooks/17_evaluation/calibration_ece.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/17_evaluation/calibration_ece.ipynb) |
| 17-11 | [CPU で動く LLM に生成文を採点させる (LLM-as-a-judge)](notebooks/17_evaluation/llm_judge_local.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/17_evaluation/llm_judge_local.ipynb) |
| 17-12 | [LLM に生成文の品質を採点させる (LLM-as-a-judge)](notebooks/17_evaluation/llm_judge.ipynb) 🔑 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/17_evaluation/llm_judge.ipynb) |

### 18. 検索

転置索引と BM25 によるキーワード検索、埋め込みによる意味検索、その組み合わせと並べ替え、評価、モデルの微調整。質問応答や RAG の土台になる。

| # | やりたいこと | Colab |
|---|---|---|
| 18-01 | [転置索引を自作してブール検索・フレーズ検索をする](notebooks/18_search/inverted_index.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/18_search/inverted_index.ipynb) |
| 18-02 | [BM25 を自分で実装してパラメータの働きを確かめる](notebooks/18_search/bm25_scratch.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/18_search/bm25_scratch.ipynb) |
| 18-03 | [BM25 で日本語のキーワード検索をする](notebooks/18_search/bm25_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/18_search/bm25_ja.ipynb) |
| 18-04 | [単語に区切らず文字 n-gram で検索する (表記ゆれに強い検索)](notebooks/18_search/char_ngram_search.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/18_search/char_ngram_search.ipynb) |
| 18-05 | [TF-IDF とコサイン類似度で似た文書を探す](notebooks/18_search/tfidf_similarity.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/18_search/tfidf_similarity.ipynb) |
| 18-06 | [埋め込みで意味検索 (セマンティックサーチ) をする](notebooks/18_search/semantic_search.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/18_search/semantic_search.ipynb) |
| 18-07 | [FAISS で大量のベクトルから高速に近傍検索する](notebooks/18_search/faiss_index.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/18_search/faiss_index.ipynb) |
| 18-08 | [キーワード検索と埋め込み検索を組み合わせる (ハイブリッド検索、RRF)](notebooks/18_search/hybrid_search.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/18_search/hybrid_search.ipynb) |
| 18-09 | [クロスエンコーダで検索結果を並べ替える (リランキング)](notebooks/18_search/rerank.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/18_search/rerank.ipynb) |
| 18-10 | [検索結果の上位からクエリを拡張する (擬似適合性フィードバック)](notebooks/18_search/query_expansion.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/18_search/query_expansion.ipynb) |
| 18-11 | [検索結果の重複を減らして多様にする (MMR)](notebooks/18_search/mmr_diversify.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/18_search/mmr_diversify.ipynb) |
| 18-12 | [長い文書を分割して検索し、文書単位にまとめる](notebooks/18_search/long_document_retrieval.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/18_search/long_document_retrieval.ipynb) |
| 18-13 | [言語をまたいで検索する (クロスリンガル検索)](notebooks/18_search/cross_lingual_search.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/18_search/cross_lingual_search.ipynb) |
| 18-14 | [入力の途中で候補を出す・打ち間違いを直す (サジェストと「もしかして」)](notebooks/18_search/search_suggest.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/18_search/search_suggest.ipynb) |
| 18-15 | [検索用の埋め込みモデルを自分のデータで微調整する](notebooks/18_search/embedding_finetune.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/18_search/embedding_finetune.ipynb) |
| 18-16 | [検索・ランキングを評価する (Recall@k、MRR、nDCG)](notebooks/18_search/ranking_metrics.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/18_search/ranking_metrics.ipynb) |

### 19. 質問応答

文書から答えを抜き出す、選択肢から選ぶ、表を読んで答える。データセットの扱いから学習・評価・質問の生成まで。検索と組み合わせた質問応答は「RAG」の章を参照。

| # | やりたいこと | Colab |
|---|---|---|
| 19-01 | [質問応答のデータセットを読み込んで中身を確認する (JSQuAD)](notebooks/19_qa/qa_dataset.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/19_qa/qa_dataset.ipynb) |
| 19-02 | [文書の中から質問の答えを抜き出す (抽出型 QA)](notebooks/19_qa/extractive_qa.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/19_qa/extractive_qa.ipynb) |
| 19-03 | [質問応答を完全一致と F1 で評価する](notebooks/19_qa/qa_em_f1.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/19_qa/qa_em_f1.ipynb) |
| 19-04 | [文書に答えがない質問を「答えなし」と判定する](notebooks/19_qa/qa_unanswerable.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/19_qa/qa_unanswerable.ipynb) |
| 19-05 | [抽出型 QA モデルをファインチューニングする](notebooks/19_qa/qa_finetune.ipynb) ⚡ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/19_qa/qa_finetune.ipynb) |
| 19-06 | [多肢選択式の問題を解く](notebooks/19_qa/multiple_choice_qa.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/19_qa/multiple_choice_qa.ipynb) |
| 19-07 | [表の内容について質問に答える](notebooks/19_qa/table_qa.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/19_qa/table_qa.ipynb) |
| 19-08 | [文書から質問と答えの組を自動で作る (質問生成)](notebooks/19_qa/question_generation.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/19_qa/question_generation.ipynb) |

### 20. RAG (検索拡張生成)

手元の文書を検索して LLM に答えさせる。分割・検索・生成・出典・評価まで、CPU で動く小型モデルで一通り組み立てる。検索の部品 (BM25、意味検索、ハイブリッド、リランキング) は「検索」の章を参照。

| # | やりたいこと | Colab |
|---|---|---|
| 20-01 | [検索した文書を LLM に渡して答えさせる (最小の RAG)](notebooks/20_rag/rag_minimal.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/20_rag/rag_minimal.ipynb) |
| 20-02 | [長い文書をチャンクに分割する](notebooks/20_rag/chunking.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/20_rag/chunking.ipynb) |
| 20-03 | [小さなチャンクで検索して、周りの文脈ごと LLM に渡す](notebooks/20_rag/rag_parent_child.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/20_rag/rag_parent_child.ipynb) |
| 20-04 | [ベクトル DB に保存して、メタデータで絞り込んで検索する (Qdrant)](notebooks/20_rag/rag_vector_db.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/20_rag/rag_vector_db.ipynb) |
| 20-05 | [回答に出典を付けさせ、その出典を検証する](notebooks/20_rag/rag_citation.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/20_rag/rag_citation.ipynb) |
| 20-06 | [答えが資料にないときに「わかりません」と答えさせる](notebooks/20_rag/rag_refusal.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/20_rag/rag_refusal.ipynb) |
| 20-07 | [検索しやすい形に質問を書き換える (文脈の補完と HyDE)](notebooks/20_rag/rag_query_rewrite.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/20_rag/rag_query_rewrite.ipynb) |
| 20-08 | [RAG の評価セットを用意する](notebooks/20_rag/rag_eval_dataset.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/20_rag/rag_eval_dataset.ipynb) |
| 20-09 | [検索の評価セットを作って手法を比べる](notebooks/20_rag/rag_eval_retrieval.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/20_rag/rag_eval_retrieval.ipynb) |
| 20-10 | [RAG 全体を評価して設定を比べる](notebooks/20_rag/rag_eval_pipeline.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/20_rag/rag_eval_pipeline.ipynb) |
| 20-11 | [RAG の失敗を検索と生成に切り分けて分析する](notebooks/20_rag/rag_error_analysis.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/20_rag/rag_error_analysis.ipynb) |
| 20-12 | [RAG の回答を LLM を使わずに評価する](notebooks/20_rag/rag_eval_answer.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/20_rag/rag_eval_answer.ipynb) |
| 20-13 | [RAG の回答の正しさを LLM に判定させる](notebooks/20_rag/rag_eval_llm_judge.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/20_rag/rag_eval_llm_judge.ipynb) |

### 21. 対話システム

規則ベース、FAQ 検索、タスク指向、LLM による雑談まで。チャットボットを組み立てる部品を、CPU で動く範囲で。

| # | やりたいこと | Colab |
|---|---|---|
| 21-01 | [規則だけでチャットボットを作る (ELIZA 方式)](notebooks/21_dialogue/eliza_rule.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/21_dialogue/eliza_rule.ipynb) |
| 21-02 | [よくある質問に自動で答える FAQ ボットを作る](notebooks/21_dialogue/faq_bot.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/21_dialogue/faq_bot.ipynb) |
| 21-03 | [発話の意図を分類する (少数の例から)](notebooks/21_dialogue/intent_classifier.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/21_dialogue/intent_classifier.ipynb) |
| 21-04 | [必要な情報を聞き出しながら予約を受け付ける (スロットフィリング)](notebooks/21_dialogue/slot_filling.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/21_dialogue/slot_filling.ipynb) |
| 21-05 | [対話の履歴を管理して、長くなったら古いターンを捨てる](notebooks/21_dialogue/chat_history.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/21_dialogue/chat_history.ipynb) |
| 21-06 | [古い対話を要約して長い対話を続ける](notebooks/21_dialogue/chat_summary_memory.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/21_dialogue/chat_summary_memory.ipynb) |
| 21-07 | [キャラクターや口調を設定する (ペルソナ)](notebooks/21_dialogue/persona_prompt.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/21_dialogue/persona_prompt.ipynb) |
| 21-08 | [資料を調べて答えるチャットボットを作る (複数ターンの RAG)](notebooks/21_dialogue/chat_rag.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/21_dialogue/chat_rag.ipynb) |
| 21-09 | [入力と出力に安全策を入れる (ガードレール)](notebooks/21_dialogue/dialogue_guardrail.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/21_dialogue/dialogue_guardrail.ipynb) |
| 21-10 | [ブラウザで使えるチャット画面を作る (Gradio)](notebooks/21_dialogue/gradio_chat.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/21_dialogue/gradio_chat.ipynb) |

### 22. 機械翻訳

翻訳モデルを動かす、対訳データを用意する、モデルを学習・微調整する、訳文を評価する。

| # | やりたいこと | Colab |
|---|---|---|
| 22-01 | [対訳コーパスを取得して中身を確認する](notebooks/22_machine_translation/parallel_corpus.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/22_machine_translation/parallel_corpus.ipynb) |
| 22-02 | [多言語翻訳モデル NLLB で翻訳する](notebooks/22_machine_translation/translation_nllb.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/22_machine_translation/translation_nllb.ipynb) |
| 22-03 | [Moses で統計的機械翻訳システムを作る](notebooks/22_machine_translation/moses_smt.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/22_machine_translation/moses_smt.ipynb) |
| 22-04 | [Transformer の翻訳モデルをゼロから学習する](notebooks/22_machine_translation/nmt_from_scratch.ipynb) ⚡ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/22_machine_translation/nmt_from_scratch.ipynb) |
| 22-05 | [学習済みの翻訳モデルを自分の分野のデータでファインチューニングする](notebooks/22_machine_translation/mt_finetune.ipynb) ⚡ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/22_machine_translation/mt_finetune.ipynb) |
| 22-06 | [言語をまたいで意味の近い文を対応付ける (対訳文の抽出)](notebooks/22_machine_translation/cross_lingual_similarity.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/22_machine_translation/cross_lingual_similarity.ipynb) |
| 22-07 | [機械翻訳を BLEU で評価する (sacreBLEU)](notebooks/22_machine_translation/bleu.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/22_machine_translation/bleu.ipynb) |
| 22-08 | [chrF と TER で翻訳を評価する](notebooks/22_machine_translation/chrf_ter.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/22_machine_translation/chrf_ter.ipynb) |
| 22-09 | [COMET で翻訳の品質を評価する](notebooks/22_machine_translation/comet.ipynb) ⚡ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/22_machine_translation/comet.ipynb) |

### 23. 要約と平易化

文章を短くまとめる、やさしく書き換える。抽出型と生成型の要約、3 行要約、見出し生成、テキスト平易化と、その評価。

| # | やりたいこと | Colab |
|---|---|---|
| 23-01 | [先頭の数文を要約とする (Lead-3) ベースラインと比べる](notebooks/23_summarization/lead3_baseline.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/23_summarization/lead3_baseline.ipynb) |
| 23-02 | [重要な文を選んで抽出型要約を作る (LexRank)](notebooks/23_summarization/extractive_summary.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/23_summarization/extractive_summary.ipynb) |
| 23-03 | [要約を ROUGE で評価する (日本語対応)](notebooks/23_summarization/rouge_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/23_summarization/rouge_ja.ipynb) |
| 23-04 | [LLM で文章を要約する (長さと観点を指定する)](notebooks/23_summarization/llm_summary.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/23_summarization/llm_summary.ipynb) |
| 23-05 | [ニュース記事を 3 行に要約する](notebooks/23_summarization/three_line_summary.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/23_summarization/three_line_summary.ipynb) |
| 23-06 | [長い文書を分割して要約する](notebooks/23_summarization/long_document_summary.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/23_summarization/long_document_summary.ipynb) |
| 23-07 | [要約が原文と食い違っていないか検査する](notebooks/23_summarization/summary_faithfulness.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/23_summarization/summary_faithfulness.ipynb) |
| 23-08 | [記事の見出しを生成して実際の見出しと比べる](notebooks/23_summarization/headline_generation.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/23_summarization/headline_generation.ipynb) |
| 23-09 | [文章をやさしい日本語に書き換える (テキスト平易化)](notebooks/23_summarization/text_simplification.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/23_summarization/text_simplification.ipynb) |

### 24. 文法誤り訂正と言語教育

学習者の作文を訂正する、訂正箇所を取り出して分類・評価する、教材づくりに使う。

| # | やりたいこと | Colab |
|---|---|---|
| 24-01 | [2 つの文の違いを色付きで表示する](notebooks/24_gec_education/diff_highlight.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/24_gec_education/diff_highlight.ipynb) |
| 24-02 | [訂正前後の文から編集 (置換・挿入・削除) を取り出す](notebooks/24_gec_education/edit_extraction.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/24_gec_education/edit_extraction.ipynb) |
| 24-03 | [ERRANT で英語の訂正を誤りタイプ付きの編集に変換する](notebooks/24_gec_education/errant.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/24_gec_education/errant.ipynb) |
| 24-04 | [M2 形式のファイルを読み書きする](notebooks/24_gec_education/m2_format.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/24_gec_education/m2_format.ipynb) |
| 24-05 | [文法誤り訂正を編集単位の F0.5 で評価する](notebooks/24_gec_education/gec_f05.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/24_gec_education/gec_f05.ipynb) |
| 24-06 | [学習済みモデルで英語の文法誤りを訂正する](notebooks/24_gec_education/gec_model_en.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/24_gec_education/gec_model_en.ipynb) |
| 24-07 | [規則で誤りを注入して GEC の擬似学習データを作る](notebooks/24_gec_education/synthetic_errors.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/24_gec_education/synthetic_errors.ipynb) |
| 24-08 | [日本語の文章の読みやすさを測る](notebooks/24_gec_education/readability_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/24_gec_education/readability_ja.ipynb) |
| 24-09 | [単語の頻度から語彙の難しさを推定する (wordfreq)](notebooks/24_gec_education/word_frequency_level.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/24_gec_education/word_frequency_level.ipynb) |
| 24-10 | [穴埋め問題 (クローズテスト) を自動で作る](notebooks/24_gec_education/cloze_generation.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/24_gec_education/cloze_generation.ipynb) |

### 25. 日本語入力 (かな漢字変換)

ローマ字からかなへの変換、辞書とラティス、ビタビアルゴリズムによるかな漢字変換、候補の提示、予測入力、履歴の学習、ニューラルかな漢字変換、評価まで、日本語入力システムの部品を作る。

| # | やりたいこと | Colab |
|---|---|---|
| 25-01 | [ローマ字をかなに変換する](notebooks/25_japanese_input/romaji_to_kana.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/25_japanese_input/romaji_to_kana.ipynb) |
| 25-02 | [読みから表記を引く辞書をコーパスから作る](notebooks/25_japanese_input/kkc_dictionary.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/25_japanese_input/kkc_dictionary.ipynb) |
| 25-03 | [単語の頻度だけでかな漢字変換をする (ビタビアルゴリズム)](notebooks/25_japanese_input/kkc_unigram.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/25_japanese_input/kkc_unigram.ipynb) |
| 25-04 | [前後の単語を考慮してかな漢字変換をする (bigram 言語モデル)](notebooks/25_japanese_input/kkc_bigram.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/25_japanese_input/kkc_bigram.ipynb) |
| 25-05 | [変換候補を複数出す (N-best)](notebooks/25_japanese_input/kkc_nbest.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/25_japanese_input/kkc_nbest.ipynb) |
| 25-06 | [入力の途中で続きを予測する (予測入力)](notebooks/25_japanese_input/predictive_input.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/25_japanese_input/predictive_input.ipynb) |
| 25-07 | [利用者の選択を学習して変換結果を変える](notebooks/25_japanese_input/kkc_user_history.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/25_japanese_input/kkc_user_history.ipynb) |
| 25-08 | [学習済みのニューラルモデルでかな漢字変換をする (zenz)](notebooks/25_japanese_input/kkc_neural.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/25_japanese_input/kkc_neural.ipynb) |
| 25-09 | [Web API でかな漢字変換をする (Google CGI API for Japanese Input)](notebooks/25_japanese_input/kkc_web_api.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/25_japanese_input/kkc_web_api.ipynb) |
| 25-10 | [かな漢字変換の精度を評価する](notebooks/25_japanese_input/kkc_eval.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/25_japanese_input/kkc_eval.ipynb) |

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
