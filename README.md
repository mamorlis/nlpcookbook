# NLP クックブック

自然言語処理の「やりたいこと」に対して「こうやればいい」をまとめた小ネタ集です。
1 レシピ = 1 ノートブックで、すべて Google Colab でそのまま実行できます。

- 日本語のテキスト処理を中心に、英語・多言語も扱います。
- 各ノートブックは上から順に実行するだけで動きます (必要なライブラリのインストールを含む)。
- 実行結果を埋め込んであるので、GitHub 上で眺めるだけでも使えます。

## 目次

<!-- INDEX:START -->
全 75 レシピ。タイトルをクリックすると GitHub 上で、バッジをクリックすると Google Colab で開きます。

- [1. 文字列処理と正規化](#1-文字列処理と正規化) (22)
- [2. 日本語の形態素解析](#2-日本語の形態素解析) (19)
- [3. 英語・多言語の前処理](#3-英語多言語の前処理) (20)
- [4. サブワードとトークナイザ](#4-サブワードとトークナイザ) (14)

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
