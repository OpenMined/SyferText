import syft as sy
import torch
import syfertext
import numpy as np

hook = sy.TorchHook(torch)
me = hook.local_worker
lang = "en_core_web_lg"
nlp = syfertext.load(lang, owner=me)

# Get the vocab instance
vocab = nlp.vocab


def test_lex_rank():
    text1 = "Apple"
    text2 = "outofvocabulary"

    # Get the Lexeme object of text from Vocab
    lexeme1 = vocab[text1]
    lexeme2 = vocab[text2]

    # get the rank of lexeme
    rank1 = lexeme1.rank
    rank2 = lexeme2.rank

    # test rank is an integer for an
    # string which exist in vocabularu
    assert rank1 >= 0

    # Test for out of vocabulary strings
    #  rank doesn't exist
    assert rank2 is None


def test_lex_text():
    text = "Apple"

    # Get the Lexeme object of text from Vocab
    lexeme = vocab[text]

    # test the text attribute of lexme
    assert lexeme.text == text

    # test the orth_ attribute of lexme
    assert lexeme.orth_ == text


def test_lex_orth():
    text = "Apple"

    # Get the Lexeme object of text from Vocab
    lexeme = vocab[text]

    # get text token
    token = nlp(text)[0]

    # test orth of lexeme is same as
    # that of the original string
    assert lexeme.orth == token.orth


def test_lex_lower():
    text = "APple"

    # Get the Lexeme object of text from Vocab
    lexeme = vocab[text]

    # get the token of lowercase text
    token = nlp(text.lower())[0]

    # test the  lower attribute (lowercase string orth)
    assert lexeme.lower == token.orth

    # test if lower_ attribute (lowercase string)
    assert lexeme.lower_ == text.lower()


def test_lang_name():
    text = "apple"

    # Get the Lexeme object of text from Vocab
    lexeme = vocab[text]

    # test the language model name of lexeme
    assert lexeme.lang_ == lang


def test_lex_bool_attrs():

    # define strings for checking
    # corresponding attributes
    text = "apple"
    stop = "did"
    alpha = "Apple"
    not_alpha = "5Apple"
    ascii = ","
    not_ascii = "£"
    punct = "'"
    right_punct = "’"
    left_punct = "‛"
    oov = "outofvocabulary"
    digit = "104"
    lower = "apple"
    upper = "APPLE"
    space = "  "
    bracket = "("
    quote = "'"
    currency = "¥"
    title = "Openmined Syfertext"

    # Test is_oov (if out of vocabulary)
    assert vocab[oov].is_oov == True
    assert vocab[text].is_oov == False

    # test is_stop (if string is in SyferText stop words list defined)
    assert vocab[stop].is_stop == True
    assert vocab[text].is_stop == False

    # test is_alpha (if string contains alpha chars)
    assert vocab[alpha].is_alpha == True
    assert vocab[not_alpha].is_alpha == False

    # test is_ascii (if string is composed of ascii characters)
    assert vocab[ascii].is_ascii == True
    assert vocab[not_ascii].is_ascii == False

    # test is_digit (if string is a digit)
    assert vocab[digit].is_digit == True
    assert vocab[text].is_digit == False

    # test is_lower (if string is in lowercase)
    assert vocab[lower].is_lower == True
    assert vocab[upper].is_lower == False

    # test is_title (if string is in title case)
    assert vocab[title].is_title == True
    assert vocab[text].is_title == False

    # test is_punct (if string is a punctuation)
    assert vocab[punct].is_punct == True
    assert vocab[text].is_punct == False

    # test is_space (if string is composed of space character only )
    assert vocab[space].is_space == True
    assert vocab[text].is_space == False

    # test is_quote (if string is a quote char)
    assert vocab[quote].is_quote == True
    assert vocab[text].is_quote == False

    # test is_left_punct (if string is a left punctuation char)
    assert vocab[left_punct].is_left_punct == True
    assert vocab[text].is_left_punct == False

    # test is_right_punct (if string is a right punctuation char)
    assert vocab[right_punct].is_right_punct == True
    assert vocab[text].is_right_punct == False

    # test is_currency (if string is a currency char)
    assert vocab[currency].is_currency == True
    assert vocab[text].is_currency == False

    # test is_bracket (if string is a bracket char)
    assert vocab[bracket].is_bracket == True
    assert vocab[text].is_bracket == False


def test_lex_like_num():
    num = "10.8"
    text = "apple"

    # test if string is like number
    assert vocab[num].like_num == True
    assert vocab[text].like_num == False


def test_lex_like_email():
    text1 = "noobmaster69@endgame.com"
    text2 = "noobmaster@endgame"

    # test if the string is like an email
    assert vocab[text1].like_email == True
    assert vocab[text2].like_email == False


def test_lex_like_url():
    texts = {
        "http://ninjaflex_meta.com/": True,
        "google.com": True,
        "www.google.com": True,
        "https://amazondating.co/": True,
        "apple": False,
        "a.b": False,
    }

    for url, match in texts.items():
        # test for each string is a like url
        assert vocab[url].like_url == match


def test_lex_word_shape():
    words = {
        "Apple": "Xxxxx",
        "APPLE": "XXXX",
        "noobmaster69": "xxxxdd",
        "123456": "dddd",
        ",": ",",
    }

    for word, shape in words.items():
        # test shape of each word in the dict
        assert vocab[word].shape_ == shape
