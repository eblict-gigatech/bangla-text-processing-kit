![Python 3.7 - 3.11](https://img.shields.io/badge/python-3.7--3.11-blue)
![PyTorch 2.6](https://img.shields.io/badge/pytorch-2.6-red)


![BKIT Logo](media/bkit_banner.png)

**Bangla Text Processing Kit (Bkit)** is a comprehensive Python toolkit for Bangla NLP. It provides robust tools for text normalization, cleaning, tokenization, lemmatization, stemming, parts-of-speech tagging, named entity recognition, dependency parsing, and coreference resolution. Bkit also offers intuitive visualizations to explore text structure and linguistic patterns, making it easy for researchers, developers, and educators to analyze and process Bangla text efficiently.


- [Bangla Text Processing Kit](#bangla-text-processingkit)
  - [How to use](#how-to-use)
    - [Installing](#installing)
    - [Checking text](#checking-text)
    - [Transforming text](#transforming-text)
      - [Normalizer](#normalizer)
      - [Character Normalization](#character-normalization)
      - [Punctuation space normalization](#punctuation-space-normalization)
      - [Zero width characters normalization](#zero-width-characters-normalization)
      - [Halant (হসন্ত) normalization](#halant-হসন্ত-normalization)
      - [Kar ambiguity](#kar-ambiguity)
      - [Clean text](#clean-text)
      - [Clean punctuations](#clean-punctuations)
      - [Clean digits](#clean-digits)
      - [Multiple spaces](#multiple-spaces)
      - [URLs](#urls)
      - [Emojis](#emojis)
      - [HTML tags](#html-tags)
      - [Multiple punctuations](#multiple-punctuations)
      - [Special characters](#special-characters)
      - [Non Bangla characters](#non-bangla-characters)
    - [Text Analysis](#text-analysis)
      - [Word count](#word-count)
      - [Sentence Count](#sentence-count)
    - [Lemmatization](#lemmatization)
      - [Lemmatize text](#lemmatize-text)
      - [Lemmatize word](#lemmatize-word)
    - [Stemmer](#stemmer)
      - [Stem word](#stem-word)
      - [Stem sentence](#stem-sentence)
    - [Tokenization](#tokenization)
      - [Word tokenization](#word-tokenization)
      - [Word and Punctuation tokenization](#word-and-punctuation-tokenization)
      - [Sentence tokenization](#sentence-tokenization)
    - [Text Similarity](#text-similarity)
      - [Word2Vec](#word2vec)
    - [Named Entity Recognition (NER)](#named-entity-recognition-ner)
      - [Named Entity Recognition (NER) Visualization](#named-entity-recognition-ner-visualization)
    - [Parts of Speech (PoS) tagging](#parts-of-speech-pos-tagging)
      - [Parts of Speech (PoS) Visualization](#parts-of-speech-pos-visualization)
    - [Shallow Parsing (Constituency Parsing)](#shallow-parsing-constituency-parsing)
        - [Shallow Parsing Visualization](#shallow-parsing-visualization)
    - [Dependency Parsing](#dependency-parsing)
        - [Dependency Parsing Visualization](#dependency-parsing-visualization)
    - [Coreference Resolution](#coref-resolution)
      - [Coreference Resolution Visualization](#coreference-resolution-visualization)
    

## How to use<a id="how-to-use"></a>

### Installing<a id="installing"></a>

There are three installation options of the bkit package. These are:

1. `bkit`: The most basic version of bkit with the normalization, cleaning and tokenization capabilities.

```bash
pip install bkit
```
2. `bkit[lemma]`: Everything in the basic version plus lemmatization capability.

```bash
pip install bkit[lemma]
```
3. `bkit[all]`: Everything that are available in bkit including normalization, cleaning, tokenization, lemmatization, NER, POS and shallow parsing. 

```bash
pip install bkit[all]
```

### Checking text<a id="checking-text"></a>

- `bkit.utils.is_bangla(text) -> bool`: Checks if text contains only Bangla characters, digits, spaces, punctuations and some symbols. Returns true if so, else return false.
- `bkit.utils.is_digit(text) -> bool`: Checks if text contains only **Bangla digit** characters. Returns true if so, else return false.
- `bkit.utils.contains_digit(text, check_english_digits) -> bool`: Checks if text contains **any digits**. By default checks only Bangla digits. Returns true if so, else return false.
- `bkit.utils.contains_bangla(text) -> bool`: Checks if text contains **any Bangla character**. Returns true if so, else return false.

### Transforming text<a id="transforming-text"></a>

Text transformation includes the normalization and cleaning procedures. To transform text, use the `bkit.transform` module. Supported functionalities are:

#### Normalizer<a id="normalizer"></a>

This module normalize Bangla text using the following steps:

<!-- no toc -->
1. [Character normalization](#character-normalization)
2. [Zero width character normalization](#zero-width-characters-normalization)
3. [Halant normalization](#halant-হসন্ত-normalization)
4. [Vowel-kar normalization](#kar-ambiguity)
5. [Punctuation space normalization](#punctuation-space-normalization)

```python
import bkit

text = 'অাামাব় । '
print(list(text))
# >>> ['অ', 'া', 'া', 'ম', 'া', 'ব', '়', ' ', '।', ' ']

normalizer = bkit.transform.Normalizer(
    normalize_characters=True,
    normalize_zw_characters=True,
    normalize_halant=True,
    normalize_vowel_kar=True,
    normalize_punctuation_spaces=True
)

clean_text = normalizer(text)
print(clean_text, list(clean_text))
# >>> আমার। ['আ', 'ম', 'া', 'র', '।']
```

#### Character Normalization<a id="character-normalization"></a>

This module performs character normalization in Bangla text. It performs nukta normalization, Assamese normalization, Kar normalization, legacy character normalization and Punctuation normalization sequentially.

```python
import bkit

text = 'আমাব়'
print(list(text))
# >>> ['আ', 'ম', 'া', 'ব', '়']

text = bkit.transform.normalize_characters(text)

print(list(text))
# >>> ['আ', 'ম', 'া', 'র']
```

#### Punctuation space normalization<a id="punctuation-space-normalization"></a>

Normalizes punctuation spaces i.e. adds necessary spaces before or after specific punctuations, also removes if necessary.

```python
import bkit

text = 'রহিম(২৩)এ কথা বলেন   ।তিনি (    রহিম ) আরও জানান, ১,২৪,৩৫,৬৫৪.৩২৩ কোটি টাকা ব্যায়ে...'

clean_text = bkit.transform.normalize_punctuation_spaces(text)
print(clean_text)
# >>> রহিম (২৩) এ কথা বলেন। তিনি (রহিম) আরও জানান, ১,২৪,৩৫,৬৫৪.৩২৩ কোটি টাকা ব্যায়ে...
```

#### Zero width characters normalization<a id="zero-width-characters-normalization"></a>

There are two zero-width characters. These are Zero Width Joiner (ZWJ) and Zero Width Non Joiner (ZWNJ) characters. Generally ZWNJ is not used with Bangla texts and ZWJ joiner is used with `র` only. So, these characters are normalized based on these intuitions.

```python
import bkit

text = 'র‍্য‌াকেট'
print(f"text: {text} \t Characters: {list(text)}")
# >>> text: র‍্য‌াকেট     Characters: ['র', '\u200d', '্', 'য', '\u200c', 'া', 'ক', 'ে', 'ট']

clean_text = bkit.transform.normalize_zero_width_chars(text)
print(f"text: {clean_text} \t Characters: {list(clean_text)}")
# >>> text: র‍্যাকেট     Characters: ['র', '\u200d', '্', 'য', 'া', 'ক', 'ে', 'ট']
```

#### Halant (হসন্ত) normalization<a id="halant-হসন্ত-normalization"></a>

This function normalizes halant (হসন্ত) [`0x09CD`] in Bangla text. While using this function, it is recommended to normalize the zero width characters at first, e.g. using the `bkit.transform.normalize_zero_width_chars()` function.

During the normalization it also handles the `ত্ -> ৎ` conversion. For a valid conjunct letter (যুক্তবর্ণ) where 'ত' is the former character, can take one of 'ত', 'থ', 'ন', 'ব', 'ম', 'য', and 'র' as the next character. The conversion is perform based on this intuition.

During the halant normalization, the following cases are handled.

- Remove any leading and tailing halant of a word and/or text.
- Replace two or more consecutive occurrences of halant by a single halant.
- Remove halant between any characters that do not follow or precede a halant character. Like a halant that follows or precedes a vowel, kar, য়, etc will be removed.
- Remove multiple fola (multiple ref, ro-fola and jo-fola)

```python
import bkit

text = 'আসন্্্ন আসফাকুল্লাহ্‌ আলবত্‍ আলবত্ র‍্যাব ই্সি'
print(list(text))
# >>> ['আ', 'স', 'ন', '্', '্', '্', 'ন', ' ', 'আ', 'স', 'ফ', 'া', 'ক', 'ু', 'ল', '্', 'ল', 'া', 'হ', '্', '\u200c', ' ', 'আ', 'ল', 'ব', 'ত', '্', '\u200d', ' ', 'আ', 'ল', 'ব', 'ত', '্', ' ', 'র', '\u200d', '্', 'য', 'া', 'ব', ' ', 'ই', '্', 'স', 'ি']

clean_text = bkit.transform.normalize_zero_width_chars(text)
clean_text = bkit.transform.normalize_halant(clean_text)
print(clean_text, list(clean_text))
# >>> আসন্ন আসফাকুল্লাহ আলবৎ আলবৎ র‍্যাব ইসি ['আ', 'স', 'ন', '্', 'ন', ' ', 'আ', 'স', 'ফ', 'া', 'ক', 'ু', 'ল', '্', 'ল', 'া', 'হ', ' ', 'আ', 'ল', 'ব', 'ৎ', ' ', 'আ', 'ল', 'ব', 'ৎ', ' ', 'র', '\u200d', '্', 'য', 'া', 'ব', ' ', 'ই', 'স', 'ি']
```

#### Kar ambiguity<a id="kar-ambiguity"></a>

Normalizes kar ambiguity with vowels, ঁ, ং, and ঃ. It removes any kar that is preceded by a vowel or consonant diacritics like: `আা` will be normalized to `আ`. In case of consecutive occurrence of kars like: `কাাাী`, only the first kar will be kept like: `কা`.

```python
import bkit

text = 'অংশইে অংশগ্রহণইে আাারো এখনওো আলবার্তোে সাধুু কাাাী'
print(list(text))
# >>> ['অ', 'ং', 'শ', 'ই', 'ে', ' ', 'অ', 'ং', 'শ', 'গ', '্', 'র', 'হ', 'ণ', 'ই', 'ে', ' ', 'আ', 'া', 'া', 'র', 'ো', ' ', 'এ', 'খ', 'ন', 'ও', 'ো', ' ', 'আ', 'ল', 'ব', 'া', 'র', '্', 'ত', 'ো', 'ে', ' ', 'স', 'া', 'ধ', 'ু', 'ু', ' ', 'ক', 'া', 'া', 'া', 'ী']

clean_text = bkit.transform.normalize_kar_ambiguity(text)
print(clean_text, list(clean_text))
# >>> অংশই অংশগ্রহণই আরো এখনও আলবার্তো সাধু কা ['অ', 'ং', 'শ', 'ই', ' ', 'অ', 'ং', 'শ', 'গ', '্', 'র', 'হ', 'ণ', 'ই', ' ', 'আ', 'র', 'ো', ' ', 'এ', 'খ', 'ন', 'ও', ' ', 'আ', 'ল', 'ব', 'া', 'র', '্', 'ত', 'ো', ' ', 'স', 'া', 'ধ', 'ু', ' ', 'ক', 'া']
```

#### Clean text<a id="clean-text"></a>

Clean text using the following steps sequentially:

<!-- no toc -->
1. [Removes all HTML tags](#html-tags)
2. [Removes all URLs](#urls)
3. [Removes all emojis (optional)](#emojis)
4. [Removes all digits (optional)](#clean-digits)
5. [Removes all punctuations (optional)](#clean-punctuations)
6. [Removes all extra spaces](#multiple-spaces)
7. [Removes all non bangla characters](#non-bangla-characters)

```python
import bkit

text = '<a href=some_URL>বাংলাদেশ</a>\nবাংলাদেশের   আয়তন ১.৪৭ লক্ষ কিলোমিটার!!!'

clean_text = bkit.transform.clean_text(text)
print(clean_text)
# >>> বাংলাদেশ বাংলাদেশের আয়তন লক্ষ কিলোমিটার
```

#### Clean punctuations<a id="clean-punctuations"></a>

Remove punctuations with the given `replace_with` character/string.

```python
import bkit

text = 'আমরা মাঠে ফুটবল খেলতে পছন্দ করি!'

clean_text = bkit.transform.clean_punctuations(text)
print(clean_text)
# >>> আমরা মাঠে ফুটবল খেলতে পছন্দ করি

clean_text = bkit.transform.clean_punctuations(text, replace_with=' PUNC ')
print(clean_text)
# >>> আমরা মাঠে ফুটবল খেলতে পছন্দ করি PUNC
```

#### Clean digits<a id="clean-digits"></a>

Remove any bangla digit from text by replacing with the given `replace_with` character/string.

```python
import bkit

text = 'তার বাসা ৭৯ নাম্বার রোডে।'

clean_text = bkit.transform.clean_digits(text)
print(clean_text)
# >>> তার বাসা    নাম্বার রোডে।

clean_text = bkit.transform.clean_digits(text, replace_with='#')
print(clean_text)
# >>> তার বাসা ## নাম্বার রোডে।
```

#### Multiple spaces<a id="multiple-spaces"></a>

Clean multiple consecutive whitespace characters including space, newlines, tabs, vertical tabs, etc. It also removes leading and trailing whitespace characters.

```python
import bkit

text = 'তার বাসা ৭৯   \t\t নাম্বার   রোডে।\nসে খুব \v ভালো ছেলে।'

clean_text = bkit.transform.clean_multiple_spaces(text)
print(clean_text)
# >>> তার বাসা ৭৯ নাম্বার রোডে। সে খুব ভালো ছেলে।

clean_text = bkit.transform.clean_multiple_spaces(text, keep_new_line=True)
print(clean_text)
# >>> তার বাসা ৭৯ নাম্বার রোডে।\nসে খুব \n ভালো ছেলে।
```

#### URLs<a id="urls"></a>

Clean URLs from text and replace the URLs with any given string.

```python
import bkit

text = 'আমি https://xyz.abc সাইটে ব্লগ লিখি। এই ftp://10.17.5.23/books সার্ভার থেকে আমার বইগুলো পাবে। এই https://bn.wikipedia.org/wiki/%E0%A6%A7%E0%A6%BE%E0%A6%A4%E0%A7%81_(%E0%A6%AC%E0%A6%BE%E0%A6%82%E0%A6%B2%E0%A6%BE_%E0%A6%AC%E0%A7%8D%E0%A6%AF%E0%A6%BE%E0%A6%95%E0%A6%B0%E0%A6%A3) লিঙ্কটিতে ভালো তথ্য আছে।'

clean_text = bkit.transform.clean_urls(text)
print(clean_text)
# >>> আমি   সাইটে ব্লগ লিখি। এই   সার্ভার থেকে আমার বইগুলো পাবে। এই   লিঙ্কটিতে ভালো তথ্য আছে।

clean_text = bkit.transform.clean_urls(text, replace_with='URL')
print(clean_text)
# >>> আমি URL সাইটে ব্লগ লিখি। এই URL সার্ভার থেকে আমার বইগুলো পাবে। এই URL লিঙ্কটিতে ভালো তথ্য আছে।
```

#### Emojis<a id="emojis"></a>

Clean emoji and emoticons from text and replace those with any given string.

```python
import bkit

text = 'কিছু ইমোজি হল: 😀🫅🏾🫅🏿🫃🏼🫃🏽🫃🏾🫃🏿🫄🫄🏻🫄🏼🫄🏽🫄🏾🫄🏿🧌🪸🪷🪹🪺🫘🫗🫙🛝🛞🛟🪬🪩🪫🩼🩻🫧🪪🟰'

clean_text = bkit.transform.clean_emojis(text, replace_with='<EMOJI>')
print(clean_text)
# >>> কিছু ইমোজি হল: <EMOJI>
```

#### HTML tags<a id="html-tags"></a>

Clean HTML tags from text and replace those with any given string.

```python
import bkit

text = '<a href=some_URL>বাংলাদেশ</a>'

clean_text = bkit.transform.clean_html(text)
print(clean_text)
# >>> বাংলাদেশ
```

#### Multiple punctuations<a id="multiple-punctuations"></a>

Remove multiple consecutive punctuations and keep the first punctuation only.

```python
import bkit

text = 'কি আনন্দ!!!!!'

clean_text = bkit.transform.clean_multiple_punctuations(text)
print(clean_text)
# >>> কি আনন্দ!
```

#### Special characters<a id="special-characters"></a>

Remove special characters like `$`, `#`, `@`, etc and replace them with the given string. If no character list is passed, `[$, #,  &, %, @]` are removed by default.

```python
import bkit

text = '#বাংলাদেশ$'

clean_text = bkit.transform.clean_special_characters(text, characters=['#', '$'], replace_with='')
print(clean_text)
# >>> বাংলাদেশ
```

#### Non Bangla characters<a id="non-bangla-characters"></a>

Non Bangla characters include characters and punctuation not used in Bangla like english or other language's alphabets  and replace them with the given string.

```python
import bkit

text = 'এই শূককীট হাতিশুঁড় Heliotropium indicum, অতসী, আকন্দ Calotropis gigantea গাছের পাতার রসালো অংশ আহার করে।'

clean_text = bkit.transform.clean_non_bangla(text, replace_with='')
print(clean_text)
# >>> এই শূককীট হাতিশুঁড়  , অতসী, আকন্দ  গাছের পাতার রসালো অংশ আহার করে
```

### Text Analysis<a id="text-analysis"></a>

#### Word count<a id="word-count"></a>

The `bkit.analysis.count_words` function can be used to get the word counts. It has the following paramerts:

```python
"""
Args:
  text (Tuple[str, List[str]]): The text to count words from. If a string is provided,
    it will be split into words. If a list of strings is provided, each string will
    be split into words and counted separately.
  clean_punctuation (bool, optional): Whether to clean punctuation from the words count. Defaults to False.
  punct_replacement (str, optional): The replacement for the punctuation. Only applicable if
    clean_punctuation is True. Defaults to "".
  return_dict (bool, optional): Whether to return the word count as a dictionary.
    Defaults to False.
  ordered (bool, optional): Whether to return the word count in descending order. Only
    applicable if return_dict is True. Defaults to False.

Returns:
  Tuple[int, Dict[str, int]]: If return_dict is True, returns a tuple containing the
    total word count and a dictionary where the keys are the words and the values
    are their respective counts. If return_dict is False, returns only the total
    word count as an integer.
"""

# examples

import bkit

text='অভিষেকের আগের দিন গতকাল রোববার ওয়াশিংটনে বিশাল এক সমাবেশে হাজির হন ট্রাম্প। তিনি উচ্ছ্বসিত ভক্ত-সমর্থকদের আমেরিকার পতনের যবনিকা ঘটানোর অঙ্গীকার করেন।'
total_words=bkit.analysis.count_words(text)
print(total_words)
# >>> 21

```

#### Sentence Count<a id="sentence-count"></a>

The bkit.analysis.count_sentences function can be used to get the word counts. It has the following paramerts:

```python
"""
Counts the number of sentences in the given text or list of texts.

Args:
  text (Tuple[str, List[str]]): The text or list of texts to count sentences from.
  return_dict (bool, optional): Whether to return the result as a dictionary. Defaults to False.
  ordered (bool, optional): Whether to order the result in descending order.
    Only applicable if return_dict is True. Defaults to False.

Returns:
  int or dict: The count of sentences. If return_dict is True, returns a dictionary with sentences as keys
    and their counts as values. If return_dict is False, returns the total count of sentences.

Raises:
  AssertionError: If ordered is True but return_dict is False.
"""

# examples
import bkit

text = 'তুমি কোথায় থাক? ঢাকা বাংলাদেশের\n রাজধানী। কি অবস্থা তার! ১২/০৩/২০২২ তারিখে সে ৪/ক ঠিকানায় গিয়ে ১২,৩৪৫.২৩ টাকা দিয়েছিল।'

count = bkit.analysis.count_sentences(text)
print(count)
# >>> 5

count = bkit.analysis.count_sentences(text, return_dict=True, ordered=True)
print(count)
# >>> {'তুমি কোথায় থাক?': 1, 'ঢাকা বাংলাদেশের\n': 1, 'রাজধানী।': 1, 'কি অবস্থা তার!': 1, '১২/০৩/২০২২ তারিখে সে ৪/ক ঠিকানায় গিয়ে ১২,৩৪৫.২৩ টাকা দিয়েছিল।': 1}
```

### Lemmatization<a id="lemmatization"></a>
Lemmatization is implemented based on our this paper **BanLemma: A Word Formation Dependent Rule and Dictionary Based Bangla Lemmatizer**  

[For more details](https://arxiv.org/pdf/2311.03078)
#### Lemmatize text<a id="lemmatize-text"></a>

Lemmatize a given text. Generally expects the text to be a sentence.

```python
import bkit

text = 'পৃথিবীর জনসংখ্যা ৮ বিলিয়নের কিছু কম'

lemmatized = bkit.lemmatizer.lemmatize(text)

print(lemmatized)
# >>> পৃথিবী জনসংখ্যা ৮ বিলিয়ন কিছু কম
```

#### Lemmatize word<a id="lemmatize-word"></a>

Lemmatize a word given the PoS information.

```python
import bkit

text = 'পৃথিবীর'

lemmatized = bkit.lemmatizer.lemmatize_word(text, 'noun')

print(lemmatized)
# >>> পৃথিবী
```

### Stemmer<a id="stemmer"></a>

Stemming is the process of reducing words to their base or root form. Our implementation achieves this by conditionally stripping away predefined prefixes and suffixes from each word.

#### Stem word<a id="stem-word"></a>
```python
import bkit

stemmer = bkit.stemmer.SimpleStemmer()
stemmer.word_stemmer('নগরবাসী')
# >>> নগর
```
#### Stem Sentence<a id="stem-sentence"></a>
```python
import bkit

stemmer = bkit.stemmer.SimpleStemmer()
stemmer.sentence_stemmer('বিকেলে রোদ কিছুটা কমেছে।')
# >>> বিকেল রোদ কিছু কম
```


### Tokenization<a id="tokenization"></a>

Tokenize a given text. The `bkit.tokenizer` module is used to tokenizer text into tokens. It supports three types of tokenization.

#### Word tokenization<a id="word-tokenization"></a>

Tokenize text into words. Also separates some punctuations including comma, danda (।), question mark, etc.

```python
import bkit

text = 'তুমি কোথায় থাক? ঢাকা বাংলাদেশের রাজধানী। কি অবস্থা তার! ১২/০৩/২০২২ তারিখে সে ৪/ক ঠিকানায় গিয়ে ১২,৩৪৫ টাকা দিয়েছিল।'

tokens = bkit.tokenizer.tokenize(text)

print(tokens)
# >>> ['তুমি', 'কোথায়', 'থাক', '?', 'ঢাকা', 'বাংলাদেশের', 'রাজধানী', '।', 'কি', 'অবস্থা', 'তার', '!', '১২/০৩/২০২২', 'তারিখে', 'সে', '৪/ক', 'ঠিকানায়', 'গিয়ে', '১২,৩৪৫', 'টাকা', 'দিয়েছিল', '।']
```

#### Word and Punctuation tokenization<a id="word-and-punctuation-tokenization"></a>

Tokenize text into words and any punctuation.

```python
import bkit

text = 'তুমি কোথায় থাক? ঢাকা বাংলাদেশের রাজধানী। কি অবস্থা তার! ১২/০৩/২০২২ তারিখে সে ৪/ক ঠিকানায় গিয়ে ১২,৩৪৫ টাকা দিয়েছিল।'

tokens = bkit.tokenizer.tokenize_word_punctuation(text)

print(tokens)
# >>> ['তুমি', 'কোথায়', 'থাক', '?', 'ঢাকা', 'বাংলাদেশের', 'রাজধানী', '।', 'কি', 'অবস্থা', 'তার', '!', '১২', '/', '০৩', '/', '২০২২', 'তারিখে', 'সে', '৪', '/', 'ক', 'ঠিকানায়', 'গিয়ে', '১২', ',', '৩৪৫', 'টাকা', 'দিয়েছিল', '।']
```

#### Sentence tokenization<a id="sentence-tokenization"></a>

Tokenize text into sentences.

```python
import bkit

text = 'তুমি কোথায় থাক? ঢাকা বাংলাদেশের রাজধানী। কি অবস্থা তার! ১২/০৩/২০২২ তারিখে সে ৪/ক ঠিকানায় গিয়ে ১২,৩৪৫ টাকা দিয়েছিল।'

tokens = bkit.tokenizer.tokenize_sentence(text)

print(tokens)
# >>> ['তুমি কোথায় থাক?', 'ঢাকা বাংলাদেশের রাজধানী।', 'কি অবস্থা তার!', '১২/০৩/২০২২ তারিখে সে ৪/ক ঠিকানায় গিয়ে ১২,৩৪৫ টাকা দিয়েছিল।']
```

### Text Similarity<a id="text-similarity"></a>

#### Word2Vec Similarity<a id="word2vec"></a>

Calculate the similarity between two texts using Word2Vec.

```python
from bkit.similarity.word2vec import BanglaWord2Vec

# Load the pretrained model
model = BanglaWord2Vec()

# Word analogy: "রাজা - পুত্র + মেয়ে"
result = model.most_similar(positive=["রাজা", "মেয়ে"], negative=["পুত্র"], topn=3)
print("\nWord analogy (রাজা - পুত্র + মেয়ে):")
for w, score in result:
    print(f"{w}: {score:.4f}")


# >>> Word analogy (রাজা - পুত্র + মেয়ে):
# >>> রাজকন্যা: 0.5810
# >>> রানি: 0.5664
# >>> রাজকুমারী: 0.5615
```


### Named Entity Recognition (NER)<a id="named-entity-recognition-ner"></a>

Predicts the tags of the Named Entities of a given text.

```python
import bkit

text = 'তুমি কোথায় থাক? ঢাকা বাংলাদেশের রাজধানী। কি অবস্থা তার! ১২/০৩/২০২২ তারিখে সে ৪/ক ঠিকানায় গিয়ে ১২,৩৪৫.২৩ টাকা দিয়েছিল।'

ner = bkit.ner.Infer('ner-noisy-label')
predictions = ner(text)

print(predictions)
# >>> [('তুমি', 'O', 0.9998692), ('কোথায়', 'O', 0.99988306), ('থাক?', 'O', 0.99983954), ('ঢাকা', 'B-GPE', 0.99891424), ('বাংলাদেশের', 'B-GPE', 0.99710876), ('রাজধানী।', 'O', 0.9995414), ('কি', 'O', 0.99989176), ('অবস্থা', 'O', 0.99980336), ('তার!', 'O', 0.99983263), ('১২/০৩/২০২২', 'B-D&T', 0.97921854), ('তারিখে', 'O', 0.9271435), ('সে', 'O', 0.99934834), ('৪/ক', 'B-NUM', 0.8297553), ('ঠিকানায়', 'O', 0.99728775), ('গিয়ে', 'O', 0.9994825), ('১২,৩৪৫.২৩', 'B-NUM', 0.99740463), ('টাকা', 'B-UNIT', 0.99914896), ('দিয়েছিল।', 'O', 0.9998908)]
```

#### Named Entity Recognition (NER) Visualization<a id="named-entity-recognition-ner-visualization"></a>
It takes the model's output and visualizes the NER tag for every word in the text.

```python
import bkit

text = 'তুমি কোথায় থাক? ঢাকা বাংলাদেশের রাজধানী। কি অবস্থা তার! ১২/০৩/২০২২ তারিখে সে ৪/ক ঠিকানায় গিয়ে ১২,৩৪৫.২৩ টাকা দিয়েছিল।'
ner = bkit.ner.Infer('ner-noisy-label')
# Or you can load Huggingface transformers model directly
ner = bkit.ner.Infer.from_huggingface("model_name_or_path")
predictions = ner(text)
bkit.ner.visualize(predictions)
```
[![NER.png](https://i.postimg.cc/Zq2RrF0z/NER.png)](https://postimg.cc/SXLk49Wg)


### Parts of Speech (PoS) tagging<a id="parts-of-speech-pos-tagging"></a>

Predicts the tags of the parts of speech of a given text.

```python
import bkit

text = 'গত কিছুদিন ধরেই জ্বালানিহীন অবস্থায় একটি ছোট মাছ ধরার নৌকায় ১৫০ জন রোহিঙ্গা আন্দামান সাগরে ভাসমান অবস্থায় রয়েছে ।'
pos = bkit.pos.Infer('pos-noisy-label')
# Or you can load Huggingface transformers model directly
pos = bkit.pos.Infer.from_huggingface("model_name_or_path")

predictions = pos(text)
print(predictions)
# >>> [('গত', 'ADJ', 0.98674506), ('কিছুদিন', 'NNC', 0.97954935), ('ধরেই', 'PP', 0.96124), ('জ্বালানিহীন', 'ADJ', 0.93195957), ('অবস্থায়', 'NNC', 0.9960413), ('একটি', 'QF', 0.9912915), ('ছোট', 'ADJ', 0.9810739), ('মাছ', 'NNC', 0.97365385), ('ধরার', 'NNC', 0.96641904), ('নৌকায়', 'NNC', 0.99680626), ('১৫০', 'QF', 0.996005), ('জন', 'NNC', 0.99434316), ('রোহিঙ্গা', 'NNP', 0.9141038), ('আন্দামান', 'NNP', 0.9856694), ('সাগরে', 'NNP', 0.7122378), ('ভাসমান', 'ADJ', 0.93841994), ('অবস্থায়', 'NNC', 0.9965629), ('রয়েছে', 'VF', 0.99680847), ('।', 'PUNCT', 0.9963098)]
```
#### Parts of Speech (PoS) Visualization<a id="parts-of-speech-pos-visualization"></a>
"It takes the model's output and visualizes the Part-of-Speech tag for every word in the text.

```python
import bkit

text = 'গত কিছুদিন ধরেই জ্বালানিহীন অবস্থায় একটি ছোট মাছ ধরার নৌকায় ১৫০ জন রোহিঙ্গা আন্দামান সাগরে ভাসমান অবস্থায় রয়েছে ।'
pos = bkit.pos.Infer('pos-noisy-label')
predictions = pos(text)
bkit.pos.visualize(predictions)
```
[![pos.png](https://i.postimg.cc/8CjcJnpb/pos.png)](https://postimg.cc/3yQYz1Dy)


### Shallow Parsing (Constituency Parsing)<a id="shallow-parsing-constituency-parsing"></a>

Predicts the shallow parsing tags of a given text.

```python
import bkit

text = 'তুমি কোথায় থাক? ঢাকা বাংলাদেশের রাজধানী। কি অবস্থা তার! ১২/০৩/২০২২ তারিখে সে ৪/ক ঠিকানায় গিয়ে ১২,৩৪৫.২৩ টাকা দিয়েছিল।'
shallow = bkit.shallow.Infer(pos_model='pos-noisy-label')
predictions = shallow(text)
print(predictions)
# >>> (S (VP (NP (PRO তুমি)) (VP (ADVP (ADV কোথায়)) (VF থাক))) (NP (NNP ?) (NNP ঢাকা) (NNC বাংলাদেশের)) (ADVP (ADV রাজধানী)) (NP (NP (NP (NNC ।)) (NP (PRO কি))) (NP (QF অবস্থা) (NNC তার)) (NP (PRO !))) (NP (NP (QF ১২/০৩/২০২২) (NNC তারিখে)) (VNF সে) (NP (QF ৪/ক) (NNC ঠিকানায়))) (VF গিয়ে))
```

#### Shallow Parsing Visualization<a id="shallow-parsing-visualization"></a>
It converts model predictions into an interactive shallow parsing Tree for clear and intuitive analysis

```python
from bkit.shallow import Infer, visualize
text = "কাতার বিশ্বকাপে আর্জেন্টিনার বিশ্বকাপ জয়ে মার্তিনেজের অবদান অনেক।"
model = Infer('pos-noisy-label')
predictions = model(text)
print(visualize(predictions))
```

[![shallow.png](https://i.postimg.cc/sgfvygbq/shallow.png)](https://postimg.cc/vcjQtbYt)


### Dependency Parsing<a id="dependency-parsing"></a>

Predicts the dependency parsing tags of a given text.

```python
from bkit import dependency

text = "কাতার বিশ্বকাপে আর্জেন্টিনার বিশ্বকাপ জয়ে মার্তিনেজের অবদান অনেক।"
dep =dependency.Infer('dependency-parsing')
predictions = dep(text)
print(predictions)
# >>>[{'text': 'কাতার বিশ্বকাপে আর্জেন্টিনার বিশ্বকাপ জয়ে মার্তিনেজের অবদান অনেক ।', 'predictions': [{'token_start': 1, 'token_end': 0, 'label': 'compound'}, {'token_start': 7, 'token_end': 1, 'label': 'obl'}, {'token_start': 4, 'token_end': 2, 'label': 'nmod'}, {'token_start': 4, 'token_end': 3, 'label': 'nmod'}, {'token_start': 7, 'token_end': 4, 'label': 'obl'}, {'token_start': 6, 'token_end': 5, 'label': 'nmod'}, {'token_start': 7, 'token_end': 6, 'label': 'nsubj'}, {'token_start': 7, 'token_end': 7, 'label': 'root'}, {'token_start': 7, 'token_end': 8, 'label': 'punct'}]}]
```

#### Dependency Parsing Visualization<a id="dependency-parsing-visualization"></a>
It converts model predictions into an interactive dependency graph for clear and intuitive analysis

```python
from bkit import dependency
text = "কাতার বিশ্বকাপে আর্জেন্টিনার বিশ্বকাপ জয়ে মার্তিনেজের অবদান অনেক।"
dep = dependency.Infer('dependency-parsing')
predictions = dep(text)
dependency.visualize(predictions)
```

[![dependency-visu.png](https://i.postimg.cc/MpsXG8ST/dependency-visu.png)](https://postimg.cc/K1Mm9w4S)

### Coreference Resolution<a id="coref-resolution"></a>

Predicts the coreferent clusters of a given text.

```python
import bkit

text = "তারাসুন্দরী ( ১৮৭৮ - ১৯৪৮ ) অভিনেত্রী । ১৮৮৪ সালে বিনোদিনীর সহায়তায় স্টার থিয়েটারে যোগদানের মাধ্যমে তিনি অভিনয় শুরু করেন । প্রথমে তিনি গিরিশচন্দ্র ঘোষের চৈতন্যলীলা নাটকে এক বালক ও সরলা নাটকে গোপাল চরিত্রে অভিনয় করেন ।"
coref = bkit.coref.Infer('coref')
predictions = coref(text)
print(predictions)
# >>> {'text': ['তারাসুন্দরী', '(', '১৮৭৮', '-', '১৯৪৮', ')', 'অভিনেত্রী', '।', '১৮৮৪', 'সালে', 'বিনোদিনীর', 'সহায়তায়', 'স্টার', 'থিয়েটারে', 'যোগদানের', 'মাধ্যমে', 'তিনি', 'অভিনয়', 'শুরু', 'করেন', '।', 'প্রথমে', 'তিনি', 'গিরিশচন্দ্র', 'ঘোষের', 'চৈতন্যলীলা', 'নাটকে', 'এক', 'বালক', 'ও', 'সরলা', 'নাটকে', 'গোপাল', 'চরিত্রে', 'অভিনয়', 'করেন', '।'], 'mention_indices': {0: [{'start_token': 0, 'end_token': 0}, {'start_token': 6, 'end_token': 6}, {'start_token': 10, 'end_token': 10}, {'start_token': 16, 'end_token': 16}, {'start_token': 22, 'end_token': 22}]}}
```

#### Coreference Resolution Visualization<a id="coreference-resolution-visualization"></a>
It takes the model's output and creates an interactive visualization to clearly depict coreference resolution, highlighting the relationships between entities in the text

```python
from bkit import coref

text = "তারাসুন্দরী ( ১৮৭৮ - ১৯৪৮ ) অভিনেত্রী । ১৮৮৪ সালে বিনোদিনীর সহায়তায় স্টার থিয়েটারে যোগদানের মাধ্যমে তিনি অভিনয় শুরু করেন । প্রথমে তিনি গিরিশচন্দ্র ঘোষের চৈতন্যলীলা নাটকে এক বালক ও সরলা নাটকে গোপাল চরিত্রে অভিনয় করেন ।"
coref = coref.Infer('coref')
predictions = coref(text)
coref.visualize(predictions)
```

[![coref.png](https://i.postimg.cc/26W5TbBb/coref.png)](https://postimg.cc/cgsZLJh0)




