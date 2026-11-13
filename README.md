# NLP クックブック

自然言語処理の「やりたいこと」に対して「こうやればいい」をまとめた小ネタ集です。
1 レシピ = 1 ノートブックで、すべて Google Colab でそのまま実行できます。

- 日本語のテキスト処理を中心に、英語・多言語も扱います。
- 各ノートブックは上から順に実行するだけで動きます (必要なライブラリのインストールを含む)。
- 実行結果を埋め込んであるので、GitHub 上で眺めるだけでも使えます。

## 目次

<!-- INDEX:START -->
全 159 レシピ。タイトルをクリックすると GitHub 上で、バッジをクリックすると Google Colab で開きます。

- [1. 文字列処理と正規化](#1-文字列処理と正規化) (22)
- [2. 日本語の形態素解析](#2-日本語の形態素解析) (19)
- [3. 英語・多言語の前処理](#3-英語多言語の前処理) (20)
- [4. サブワードとトークナイザ](#4-サブワードとトークナイザ) (14)
- [5. コーパスの統計と可視化](#5-コーパスの統計と可視化) (15)
- [6. 構文解析と固有表現抽出](#6-構文解析と固有表現抽出) (17)
- [7. 単語ベクトル](#7-単語ベクトル) (12)
- [8. 文埋め込み・類似度・検索](#8-文埋め込み類似度検索) (16)
- [9. テキスト分類](#9-テキスト分類) (16)
- [10. 系列ラベリング](#10-系列ラベリング) (8)

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

### 2. 日本語の形態素解析

MeCab (fugashi)・Sudachi・Janome で単語に区切り、品詞や原形、読みを取り出す。

| # | やりたいこと | Colab |
|---|---|---|
| 02-01 | [MeCab (fugashi + UniDic) で形態素解析する](notebooks/02_japanese_tokenization/mecab_fugashi.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_japanese_tokenization/mecab_fugashi.ipynb) |
| 02-02 | [分かち書きする](notebooks/02_japanese_tokenization/wakati.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_japanese_tokenization/wakati.ipynb) |
| 02-03 | [品詞で絞って内容語の原形だけ取り出す](notebooks/02_japanese_tokenization/content_words.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_japanese_tokenization/content_words.ipynb) |
| 02-04 | [漢字の読みを取得してふりがな (ルビ) を付ける](notebooks/02_japanese_tokenization/furigana.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_japanese_tokenization/furigana.ipynb) |
| 02-05 | [IPA 辞書で MeCab を使う](notebooks/02_japanese_tokenization/mecab_ipadic.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_japanese_tokenization/mecab_ipadic.ipynb) |
| 02-06 | [Sudachi で分割の粒度 (A/B/C) を切り替える](notebooks/02_japanese_tokenization/sudachi_split.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_japanese_tokenization/sudachi_split.ipynb) |
| 02-07 | [Sudachi の正規化形で表記ゆれを吸収する](notebooks/02_japanese_tokenization/sudachi_normalize.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_japanese_tokenization/sudachi_normalize.ipynb) |
| 02-08 | [Janome で形態素解析する (Pure Python)](notebooks/02_japanese_tokenization/janome.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_japanese_tokenization/janome.ipynb) |
| 02-09 | [Janome にユーザー辞書を追加する](notebooks/02_japanese_tokenization/janome_userdic.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_japanese_tokenization/janome_userdic.ipynb) |
| 02-10 | [連続する名詞をつなげて複合名詞を抽出する](notebooks/02_japanese_tokenization/compound_noun.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_japanese_tokenization/compound_noun.ipynb) |
| 02-11 | [日本語のストップワードを除去する](notebooks/02_japanese_tokenization/stopwords_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_japanese_tokenization/stopwords_ja.ipynb) |
| 02-12 | [単語 n-gram を作る](notebooks/02_japanese_tokenization/word_ngram.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_japanese_tokenization/word_ngram.ipynb) |
| 02-13 | [動詞・形容詞の活用型と活用形を調べる](notebooks/02_japanese_tokenization/conjugation.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_japanese_tokenization/conjugation.ipynb) |
| 02-14 | [トークンの文字位置 (オフセット) を取得する](notebooks/02_japanese_tokenization/token_offsets.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_japanese_tokenization/token_offsets.ipynb) |
| 02-15 | [DataFrame の大量テキストをまとめて分かち書きする](notebooks/02_japanese_tokenization/tokenize_pandas.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_japanese_tokenization/tokenize_pandas.ipynb) |
| 02-16 | [解析器・辞書による分割の違いを比べる](notebooks/02_japanese_tokenization/tokenizer_compare.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_japanese_tokenization/tokenizer_compare.ipynb) |
| 02-17 | [MeCab の N-best 解を得る](notebooks/02_japanese_tokenization/mecab_nbest.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_japanese_tokenization/mecab_nbest.ipynb) |
| 02-18 | [辞書にない未知語を見つける](notebooks/02_japanese_tokenization/unknown_words.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_japanese_tokenization/unknown_words.ipynb) |
| 02-19 | [日本語の単語を読みの五十音順に並べる](notebooks/02_japanese_tokenization/sort_by_reading.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/02_japanese_tokenization/sort_by_reading.ipynb) |

### 3. 英語・多言語の前処理

英語を中心に、空白で区切られる言語や中国語などの基本的な前処理。

| # | やりたいこと | Colab |
|---|---|---|
| 03-01 | [英文を単語と文に分割する (NLTK)](notebooks/03_english_multilingual/nltk_tokenize.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_english_multilingual/nltk_tokenize.ipynb) |
| 03-02 | [語幹を取り出す (ステミング)](notebooks/03_english_multilingual/stemming.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_english_multilingual/stemming.ipynb) |
| 03-03 | [単語を辞書の見出し語に戻す (レンマ化)](notebooks/03_english_multilingual/lemmatize_spacy.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_english_multilingual/lemmatize_spacy.ipynb) |
| 03-04 | [英語のストップワードを除去する](notebooks/03_english_multilingual/stopwords_en.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_english_multilingual/stopwords_en.ipynb) |
| 03-05 | [テキストが何語で書かれているか判定する (langdetect)](notebooks/03_english_multilingual/langdetect.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_english_multilingual/langdetect.ipynb) |
| 03-06 | [高精度な言語判定と、混在テキストの言語区間検出 (lingua)](notebooks/03_english_multilingual/lingua.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_english_multilingual/lingua.ipynb) |
| 03-07 | [英語の短縮形を展開する (I'm → I am)](notebooks/03_english_multilingual/contractions.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_english_multilingual/contractions.ipynb) |
| 03-08 | [アクセント記号を外す・非 ASCII 文字を ASCII にする](notebooks/03_english_multilingual/unidecode.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_english_multilingual/unidecode.ipynb) |
| 03-09 | [英語のスペルミスを訂正する](notebooks/03_english_multilingual/spellcheck.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_english_multilingual/spellcheck.ipynb) |
| 03-10 | [略語に強いルールベースの文分割 (pySBD、多言語)](notebooks/03_english_multilingual/pysbd.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_english_multilingual/pysbd.ipynb) |
| 03-11 | [大文字・小文字を正しく無視して比較する (casefold)](notebooks/03_english_multilingual/casefold.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_english_multilingual/casefold.ipynb) |
| 03-12 | [Moses 方式でトークナイズ・デトークナイズする](notebooks/03_english_multilingual/sacremoses.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_english_multilingual/sacremoses.ipynb) |
| 03-13 | [英語の品詞タグ付けをする](notebooks/03_english_multilingual/pos_tag_en.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_english_multilingual/pos_tag_en.ipynb) |
| 03-14 | [WordNet で同義語・上位語・類似度を調べる (日本語も)](notebooks/03_english_multilingual/wordnet.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_english_multilingual/wordnet.ipynb) |
| 03-15 | [中国語を単語分割する (jieba)](notebooks/03_english_multilingual/jieba.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_english_multilingual/jieba.ipynb) |
| 03-16 | [中国語の簡体字と繁体字を変換する (OpenCC)](notebooks/03_english_multilingual/opencc.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_english_multilingual/opencc.ipynb) |
| 03-17 | [数値を単語で読み上げる形にする (num2words)](notebooks/03_english_multilingual/num2words.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_english_multilingual/num2words.ipynb) |
| 03-18 | [英単語の単数形・複数形・冠詞を変換する (inflect)](notebooks/03_english_multilingual/inflect.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_english_multilingual/inflect.ipynb) |
| 03-19 | [英文の読みやすさ (リーダビリティ) を測る](notebooks/03_english_multilingual/readability_en.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_english_multilingual/readability_en.ipynb) |
| 03-20 | [英語テキストを定番の手順でクリーニングする](notebooks/03_english_multilingual/clean_text_en.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/03_english_multilingual/clean_text_en.ipynb) |

### 4. サブワードとトークナイザ

SentencePiece や Hugging Face のトークナイザを学習・利用し、モデルに入れる ID 列を作る。

| # | やりたいこと | Colab |
|---|---|---|
| 04-01 | [SentencePiece でサブワードモデルを学習する](notebooks/04_subword/sentencepiece_train.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_subword/sentencepiece_train.ipynb) |
| 04-02 | [サブワード正則化: 分割をサンプリングしてデータ拡張する](notebooks/04_subword/subword_regularization.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_subword/subword_regularization.ipynb) |
| 04-03 | [BPE を 30 行で自作して仕組みを理解する](notebooks/04_subword/bpe_scratch.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_subword/bpe_scratch.ipynb) |
| 04-04 | [Hugging Face tokenizers で BPE / WordPiece を学習する](notebooks/04_subword/tokenizers_train.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_subword/tokenizers_train.ipynb) |
| 04-05 | [事前学習モデルのトークナイザを使う (日本語 BERT)](notebooks/04_subword/hf_tokenizer_basics.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_subword/hf_tokenizer_basics.ipynb) |
| 04-06 | [パディングと切り詰めをしてバッチのテンソルを作る](notebooks/04_subword/padding_truncation.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_subword/padding_truncation.ipynb) |
| 04-07 | [日本語 BERT のデコード結果から余分な空白を消す](notebooks/04_subword/decode_ja_spaces.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_subword/decode_ja_spaces.ipynb) |
| 04-08 | [トークンと元テキストの文字位置を対応付ける (offset mapping)](notebooks/04_subword/offset_mapping.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_subword/offset_mapping.ipynb) |
| 04-09 | [単語単位のラベルをサブワードに揃える (word_ids)](notebooks/04_subword/word_ids.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_subword/word_ids.ipynb) |
| 04-10 | [トークン数を数える (LLM のコンテキスト長・料金の見積もり)](notebooks/04_subword/count_tokens.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_subword/count_tokens.ipynb) |
| 04-11 | [語彙に新しいトークン・特殊トークンを追加する](notebooks/04_subword/add_tokens.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_subword/add_tokens.ipynb) |
| 04-12 | [文ペアを 1 つの入力にまとめる](notebooks/04_subword/sentence_pair.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_subword/sentence_pair.ipynb) |
| 04-13 | [トークン長の分布を調べて max_length を決める](notebooks/04_subword/token_length_dist.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_subword/token_length_dist.ipynb) |
| 04-14 | [未知語 ([UNK]) がどれくらい出るか調べる](notebooks/04_subword/unk_rate.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/04_subword/unk_rate.ipynb) |

### 5. コーパスの統計と可視化

単語を数える、特徴語を見つける、分布を図にする。コーパスを手に入れたらまずやること。

| # | やりたいこと | Colab |
|---|---|---|
| 05-01 | [単語の頻度を数える](notebooks/05_corpus_statistics/word_freq.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_corpus_statistics/word_freq.ipynb) |
| 05-02 | [Zipf の法則を確かめる (順位-頻度プロット)](notebooks/05_corpus_statistics/zipf.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_corpus_statistics/zipf.ipynb) |
| 05-03 | [matplotlib で日本語を文字化けさせずに表示する](notebooks/05_corpus_statistics/matplotlib_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_corpus_statistics/matplotlib_ja.ipynb) |
| 05-04 | [日本語のワードクラウドを作る](notebooks/05_corpus_statistics/wordcloud_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_corpus_statistics/wordcloud_ja.ipynb) |
| 05-05 | [TF-IDF で文書ごとの特徴語を取り出す](notebooks/05_corpus_statistics/tfidf_keywords.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_corpus_statistics/tfidf_keywords.ipynb) |
| 05-06 | [単語の共起を数えて PMI で結びつきの強さを測る](notebooks/05_corpus_statistics/pmi.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_corpus_statistics/pmi.ipynb) |
| 05-07 | [よく使われる言い回し (コロケーション) を抽出する](notebooks/05_corpus_statistics/collocations.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_corpus_statistics/collocations.ipynb) |
| 05-08 | [キーワードの前後の文脈を一覧する (KWIC)](notebooks/05_corpus_statistics/kwic.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_corpus_statistics/kwic.ipynb) |
| 05-09 | [語彙の豊富さを測る (TTR とその仲間)](notebooks/05_corpus_statistics/lexical_diversity.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_corpus_statistics/lexical_diversity.ipynb) |
| 05-10 | [n-gram 言語モデルを作ってパープレキシティを測り、文を生成する](notebooks/05_corpus_statistics/ngram_lm.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_corpus_statistics/ngram_lm.ipynb) |
| 05-11 | [共起ネットワークを描く](notebooks/05_corpus_statistics/cooccurrence_network.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_corpus_statistics/cooccurrence_network.ipynb) |
| 05-12 | [2 つのコーパスを比べて特徴語を見つける (対数尤度比)](notebooks/05_corpus_statistics/keyness.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_corpus_statistics/keyness.ipynb) |
| 05-13 | [文の長さや文字数の基本統計をとる](notebooks/05_corpus_statistics/corpus_stats.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_corpus_statistics/corpus_stats.ipynb) |
| 05-14 | [品詞の分布を調べる・比べる](notebooks/05_corpus_statistics/pos_distribution.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_corpus_statistics/pos_distribution.ipynb) |
| 05-15 | [文字や単語のエントロピーを計算する](notebooks/05_corpus_statistics/entropy.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/05_corpus_statistics/entropy.ipynb) |

### 6. 構文解析と固有表現抽出

係り受け解析で文の構造を取り出し、人名・組織名などの固有表現を見つける。

| # | やりたいこと | Colab |
|---|---|---|
| 06-01 | [GiNZA で日本語の依存構造解析 (係り受け解析) をする](notebooks/06_parsing_ner/ginza_dependency.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_parsing_ner/ginza_dependency.ipynb) |
| 06-02 | [依存構造木を図で表示する (displaCy)](notebooks/06_parsing_ner/displacy_dep.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_parsing_ner/displacy_dep.ipynb) |
| 06-03 | [文節に区切って文節単位の係り受けを得る](notebooks/06_parsing_ner/bunsetsu.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_parsing_ner/bunsetsu.ipynb) |
| 06-04 | [「誰が・何を・どうした」の三つ組を抜き出す](notebooks/06_parsing_ner/svo_extract.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_parsing_ner/svo_extract.ipynb) |
| 06-05 | [日本語の固有表現 (人名・地名・組織名など) を抽出する](notebooks/06_parsing_ner/ner_ginza.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_parsing_ner/ner_ginza.ipynb) |
| 06-06 | [英語の固有表現を抽出して色付きで表示する](notebooks/06_parsing_ner/ner_spacy_en.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_parsing_ner/ner_spacy_en.ipynb) |
| 06-07 | [Transformers の pipeline で日本語の固有表現抽出をする](notebooks/06_parsing_ner/ner_transformers_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_parsing_ner/ner_transformers_ja.ipynb) |
| 06-08 | [学習なしで好きなラベルの固有表現を抽出する (GLiNER)](notebooks/06_parsing_ner/gliner.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_parsing_ner/gliner.ipynb) |
| 06-09 | [品詞や原形のパターンで表現を抽出する (spaCy Matcher)](notebooks/06_parsing_ner/spacy_matcher.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_parsing_ner/spacy_matcher.ipynb) |
| 06-10 | [辞書やルールで固有表現を追加する (EntityRuler)](notebooks/06_parsing_ner/entity_ruler.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_parsing_ner/entity_ruler.ipynb) |
| 06-11 | [名詞句を抽出する](notebooks/06_parsing_ner/noun_chunks.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_parsing_ner/noun_chunks.ipynb) |
| 06-12 | [個人情報をマスクする (仮名化)](notebooks/06_parsing_ner/pii_mask.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_parsing_ner/pii_mask.ipynb) |
| 06-13 | [Stanza で多言語の形態素・依存構造解析をする](notebooks/06_parsing_ner/stanza.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_parsing_ner/stanza.ipynb) |
| 06-14 | [文脈自由文法を書いて構文解析する (NLTK)](notebooks/06_parsing_ner/cfg_parse.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_parsing_ner/cfg_parse.ipynb) |
| 06-15 | [解析結果を CoNLL-U 形式で保存・読み込みする](notebooks/06_parsing_ner/conllu.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_parsing_ner/conllu.ipynb) |
| 06-16 | [spaCy で大量の文書を高速に処理する](notebooks/06_parsing_ner/spacy_pipe.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_parsing_ner/spacy_pipe.ipynb) |
| 06-17 | [依存距離と構文木の深さで文の複雑さを測る](notebooks/06_parsing_ner/dep_distance.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/06_parsing_ner/dep_distance.ipynb) |

### 7. 単語ベクトル

単語を数百次元のベクトルで表し、類似度やアナロジーを計算する。

| # | やりたいこと | Colab |
|---|---|---|
| 07-01 | [word2vec を自分のコーパスで学習する](notebooks/07_word_vectors/word2vec_train.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_word_vectors/word2vec_train.ipynb) |
| 07-02 | [学習済みの英語単語ベクトルを読み込んで類似語を調べる](notebooks/07_word_vectors/pretrained_vectors_en.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_word_vectors/pretrained_vectors_en.ipynb) |
| 07-03 | [学習済みの日本語単語ベクトルを使う (chiVe)](notebooks/07_word_vectors/pretrained_vectors_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_word_vectors/pretrained_vectors_ja.ipynb) |
| 07-04 | [単語ベクトルでアナロジー (王 − 男 + 女 = 女王) を解く](notebooks/07_word_vectors/analogy.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_word_vectors/analogy.ipynb) |
| 07-05 | [fastText で未知語にもベクトルを与える](notebooks/07_word_vectors/fasttext_train.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_word_vectors/fasttext_train.ipynb) |
| 07-06 | [単語ベクトルを 2 次元に落として可視化する](notebooks/07_word_vectors/word_vectors_viz.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_word_vectors/word_vectors_viz.ipynb) |
| 07-07 | [単語ベクトルの平均で文ベクトルを作る](notebooks/07_word_vectors/avg_word_vectors.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_word_vectors/avg_word_vectors.ipynb) |
| 07-08 | [PPMI 行列の SVD で単語ベクトルを作る (カウントベース)](notebooks/07_word_vectors/ppmi_svd.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_word_vectors/ppmi_svd.ipynb) |
| 07-09 | [Doc2Vec で文書ベクトルを学習する](notebooks/07_word_vectors/doc2vec.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_word_vectors/doc2vec.ipynb) |
| 07-10 | [単語ベクトルに含まれるジェンダーバイアスを測る](notebooks/07_word_vectors/embedding_bias.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_word_vectors/embedding_bias.ipynb) |
| 07-11 | [BERT から文脈に応じた単語ベクトルを取り出す](notebooks/07_word_vectors/contextual_word_vectors.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_word_vectors/contextual_word_vectors.ipynb) |
| 07-12 | [単語類似度データセットで単語ベクトルを評価する](notebooks/07_word_vectors/wordsim_eval.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/07_word_vectors/wordsim_eval.ipynb) |

### 8. 文埋め込み・類似度・検索

文をベクトルにして類似度を測る、似た文書を探す、検索システムを組み立てる。

| # | やりたいこと | Colab |
|---|---|---|
| 08-01 | [文をベクトルにして意味の近さを測る (Sentence Transformers)](notebooks/08_similarity_search/sentence_embeddings.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_similarity_search/sentence_embeddings.ipynb) |
| 08-02 | [埋め込みで意味検索 (セマンティックサーチ) をする](notebooks/08_similarity_search/semantic_search.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_similarity_search/semantic_search.ipynb) |
| 08-03 | [FAISS で大量のベクトルから高速に近傍検索する](notebooks/08_similarity_search/faiss_index.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_similarity_search/faiss_index.ipynb) |
| 08-04 | [BM25 で日本語のキーワード検索をする](notebooks/08_similarity_search/bm25_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_similarity_search/bm25_ja.ipynb) |
| 08-05 | [キーワード検索と埋め込み検索を組み合わせる (ハイブリッド検索、RRF)](notebooks/08_similarity_search/hybrid_search.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_similarity_search/hybrid_search.ipynb) |
| 08-06 | [クロスエンコーダで検索結果を並べ替える (リランキング)](notebooks/08_similarity_search/rerank.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_similarity_search/rerank.ipynb) |
| 08-07 | [文をクラスタリングして話題ごとにまとめる](notebooks/08_similarity_search/sentence_clustering.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_similarity_search/sentence_clustering.ipynb) |
| 08-08 | [重複・言い換えの文ペアを見つける](notebooks/08_similarity_search/paraphrase_mining.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_similarity_search/paraphrase_mining.ipynb) |
| 08-09 | [編集距離とあいまい文字列マッチング (RapidFuzz)](notebooks/08_similarity_search/edit_distance.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_similarity_search/edit_distance.ipynb) |
| 08-10 | [TF-IDF とコサイン類似度で似た文書を探す](notebooks/08_similarity_search/tfidf_similarity.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_similarity_search/tfidf_similarity.ipynb) |
| 08-11 | [MinHash と LSH で大量文書からほぼ重複を見つける](notebooks/08_similarity_search/minhash_lsh.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_similarity_search/minhash_lsh.ipynb) |
| 08-12 | [言語をまたいで意味の近い文を対応付ける (対訳文の抽出)](notebooks/08_similarity_search/cross_lingual_similarity.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_similarity_search/cross_lingual_similarity.ipynb) |
| 08-13 | [文埋め込みを 2 次元に可視化する (UMAP)](notebooks/08_similarity_search/embedding_umap.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_similarity_search/embedding_umap.ipynb) |
| 08-14 | [文の意味的類似度 (STS) でモデルを評価する](notebooks/08_similarity_search/sts_eval.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_similarity_search/sts_eval.ipynb) |
| 08-15 | [埋め込みを量子化・次元削減して省メモリ化する](notebooks/08_similarity_search/embedding_quantization.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_similarity_search/embedding_quantization.ipynb) |
| 08-16 | [巨大な辞書の語をテキストから一括で探す (Aho-Corasick)](notebooks/08_similarity_search/aho_corasick.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/08_similarity_search/aho_corasick.ipynb) |

### 9. テキスト分類

文書にラベルを付ける。古典的な TF-IDF から BERT のファインチューニング、ゼロショットまで。

| # | やりたいこと | Colab |
|---|---|---|
| 09-01 | [TF-IDF とロジスティック回帰で日本語のニュース記事を分類する](notebooks/09_classification/tfidf_logreg_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_classification/tfidf_logreg_ja.ipynb) |
| 09-02 | [分類結果を評価する (適合率・再現率・F1・混同行列)](notebooks/09_classification/classification_report.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_classification/classification_report.ipynb) |
| 09-03 | [分類に効いている単語を調べる (線形モデルの係数)](notebooks/09_classification/feature_importance.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_classification/feature_importance.ipynb) |
| 09-04 | [複数の分類器を交差検証で比較する](notebooks/09_classification/compare_classifiers.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_classification/compare_classifiers.ipynb) |
| 09-05 | [文埋め込みを特徴量にして少ない学習データで分類する](notebooks/09_classification/embedding_classifier.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_classification/embedding_classifier.ipynb) |
| 09-06 | [ラベル付きデータなしで分類する (ゼロショット分類、NLI)](notebooks/09_classification/zero_shot_nli.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_classification/zero_shot_nli.ipynb) |
| 09-07 | [ラベルの説明文との類似度でゼロショット分類する](notebooks/09_classification/zero_shot_embedding.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_classification/zero_shot_embedding.ipynb) |
| 09-08 | [日本語の感情分析 (ポジネガ判定) をする](notebooks/09_classification/sentiment_ja.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_classification/sentiment_ja.ipynb) |
| 09-09 | [BERT をファインチューニングして文書分類する (Trainer)](notebooks/09_classification/bert_finetune.ipynb) ⚡ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_classification/bert_finetune.ipynb) |
| 09-10 | [クラスの偏ったデータを扱う (クラス重み・しきい値調整)](notebooks/09_classification/imbalanced.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_classification/imbalanced.ipynb) |
| 09-11 | [1 つの文書に複数のラベルを付ける (マルチラベル分類)](notebooks/09_classification/multilabel.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_classification/multilabel.ipynb) |
| 09-12 | [ハイパーパラメータをグリッドサーチで調整する](notebooks/09_classification/grid_search.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_classification/grid_search.ipynb) |
| 09-13 | [個々の予測の根拠を可視化する (LIME)](notebooks/09_classification/lime_explain.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_classification/lime_explain.ipynb) |
| 09-14 | [学習データを増やすと精度がどう変わるか調べる (学習曲線)](notebooks/09_classification/learning_curve.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_classification/learning_curve.ipynb) |
| 09-15 | [学習した分類器を保存・読み込みする](notebooks/09_classification/save_sklearn_model.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_classification/save_sklearn_model.ipynb) |
| 09-16 | [モデルが自信満々に間違えた事例を調べる (エラー分析)](notebooks/09_classification/error_analysis.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/09_classification/error_analysis.ipynb) |

### 10. 系列ラベリング

トークンごとにラベルを付ける。BIO タグの扱い、CRF、BERT による固有表現抽出の学習と評価。

| # | やりたいこと | Colab |
|---|---|---|
| 10-01 | [文字位置のアノテーションを BIO タグ列に変換する](notebooks/10_sequence_labeling/spans_to_bio.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_sequence_labeling/spans_to_bio.ipynb) |
| 10-02 | [BIO タグ列から固有表現のスパンを取り出す](notebooks/10_sequence_labeling/bio_to_spans.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_sequence_labeling/bio_to_spans.ipynb) |
| 10-03 | [固有表現抽出をエンティティ単位で評価する (seqeval)](notebooks/10_sequence_labeling/seqeval.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_sequence_labeling/seqeval.ipynb) |
| 10-04 | [CRF で固有表現抽出器を学習する](notebooks/10_sequence_labeling/crf_tagger.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_sequence_labeling/crf_tagger.ipynb) |
| 10-05 | [ビタビアルゴリズムを実装して最適なラベル列を求める](notebooks/10_sequence_labeling/viterbi.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_sequence_labeling/viterbi.ipynb) |
| 10-06 | [HMM で品詞タガーを作る](notebooks/10_sequence_labeling/hmm_tagger.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_sequence_labeling/hmm_tagger.ipynb) |
| 10-07 | [単語分割を文字単位の系列ラベリングとして解く](notebooks/10_sequence_labeling/char_segmentation.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_sequence_labeling/char_segmentation.ipynb) |
| 10-08 | [BERT をファインチューニングして日本語の固有表現抽出器を作る](notebooks/10_sequence_labeling/bert_ner_finetune.ipynb) ⚡ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamorlis/nlpcookbook/blob/main/notebooks/10_sequence_labeling/bert_ner_finetune.ipynb) |

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
