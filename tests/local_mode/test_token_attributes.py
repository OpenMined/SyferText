import syft as sy
import torch
import syfertext
import numpy as np

hook = sy.TorchHook(torch)
me = hook.local_worker
lang = "en_core_web_lg"
nlp = syfertext.load(lang, owner=me)
vocab = nlp.vocab


def test_token_text_with_ws():
    text = "Green Apple "
    doc = nlp(text)
    tok1 = doc[0]
    tok2 = doc[1]

    assert tok1.text_with_ws + tok2.text_with_ws == text


def test_token_lex_id():
    text = "apple"

    # create a token object
    token = nlp(text)[0]

    # Get the Lexeme object from vocab
    lexeme = vocab[text]

    # test that lexeme rank and token lex_id are equal
    assert token.lex_id == lexeme.rank


def test_token_whitespace():
    doc = nlp("green apple")
    token1 = doc[0]
    token2 = doc[1]

    # test if there is a trailing whitespace after token
    assert token1.whitespace_ == " "
    assert token2.whitespace_ == ""


def test_token_rank():
    text = "Apple outofvocabulary"
    doc = nlp(text)

    # Get the token object of text from doc
    token1 = doc[0]
    token2 = doc[1]

    # get the rank of tokens
    rank1 = token1.rank
    rank2 = token2.rank

    # test rank is an integer for an
    # string which exist in vocabulary
    assert rank1 >= 0

    # Test for out of vocabulary strings
    # rank doesn't exist
    assert rank2 is None


def test_token_text():
    text = "Apple"
    token = nlp(text)[0]

    # test the text attribute of token
    assert token.text == text


def test_lang_name():
    text = "apple"
    token = nlp(text)[0]

    # test the parent language model name of yoken
    assert token.lang_ == lang


def test_token_bool_attrs():

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
    title = "Openmined"

    # Test is_oov (if out of vocabulary)
    assert nlp(oov)[0].is_oov == True
    assert nlp(text)[0].is_oov == False

    # test is_stop (if string is in SyferText stop words list defined)
    assert nlp(stop)[0].is_stop == True
    assert nlp(text)[0].is_stop == False

    # test is_alpha (if string contains alpha chars)
    assert nlp(alpha)[0].is_alpha == True
    assert nlp(not_alpha)[0].is_alpha == False

    # test is_ascii (if string is composed of ascii characters)
    assert nlp(ascii)[0].is_ascii == True
    assert nlp(not_ascii)[0].is_ascii == False

    # test is_digit (if string is a digit)
    assert nlp(digit)[0].is_digit == True
    assert nlp(text)[0].is_digit == False

    # test is_lower (if string is in lowercase)
    assert nlp(lower)[0].is_lower == True
    assert nlp(upper)[0].is_lower == False

    # test is_title (if string is in title case)
    assert nlp(title)[0].is_title == True
    assert nlp(text)[0].is_title == False

    # test is_punct (if string is a punctuation)
    assert nlp(punct)[0].is_punct == True
    assert nlp(text)[0].is_punct == False

    # test is_space (if string is composed of space character only )
    assert nlp(space)[0].is_space == True
    assert nlp(text)[0].is_space == False

    # test is_quote (if string is a quote char)
    assert nlp(quote)[0].is_quote == True
    assert nlp(text)[0].is_quote == False

    # test is_left_punct (if string is a left punctuation char)
    assert nlp(left_punct)[0].is_left_punct == True
    assert nlp(text)[0].is_left_punct == False

    # test is_right_punct (if string is a right punctuation char)
    assert nlp(right_punct)[0].is_right_punct == True
    assert nlp(text)[0].is_right_punct == False

    # test is_currency (if string is a currency char)
    assert nlp(currency)[0].is_currency == True
    assert nlp(text)[0].is_currency == False

    # test is_bracket (if string is a bracket char)
    assert nlp(bracket)[0].is_bracket == True
    assert nlp(text)[0].is_bracket == False


def test_token_like_num():
    num = "10.8"
    text = "apple"

    # test if string is like number
    assert nlp(num)[0].like_num == True
    assert nlp(text)[0].like_num == False


def test_token_like_email():
    text1 = "noobmaster69@endgame.com"
    text2 = "noobmaster@endgame"

    # test if the string is like an email
    assert nlp(text1)[0].like_email == True
    assert nlp(text2)[0].like_email == False


def test_token_like_url():
    texts = {
        "http://ninjaflex_meta.com/": True,
        "google.com": True,
        "www.google.com": True,
        "https://amazondating.co/": True,
        "apple": False,
        "a.b": False,
    }

    # test for each string is a like url
    for url, match in texts.items():
        assert nlp(url)[0].like_url == match


def test_token_word_shape():
    words = {
        "Apple": "Xxxxx",
        "APPLE": "XXXX",
        "noobmaster69": "xxxxdd",
        "123456": "dddd",
        ",": ",",
    }

    # test shape of each word in the dict
    for word, shape in words.items():
        assert nlp(word)[0].shape_ == shape
