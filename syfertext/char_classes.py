# Authors:   ines, svlandeg, adrianeboyd, azarezade,  mirfan899,  GreenRiverRUS,  tzano,  honnibal,
#            erip,  giannisdaras and   aliiae (github  usernames)

# Code taken  from Spacy a library for advanced Natural Language Processing in Python and Cython.
# https://github.com/explosion/spaCy/blob/master/spacy/lang/punctuation.py


from __future__ import unicode_literals
import re

split_chars = lambda char: list(char.strip().split(" "))
merge_chars = lambda char: char.strip().replace(" ", "|")
group_chars = lambda char: char.strip().replace(" ", "")


_units = (
    "km km² km³ m m² m³ dm dm² dm³ cm cm² cm³ mm mm² mm³ ha µm nm yd in ft "
    "kg g mg µg t lb oz m/s km/h kmh mph hPa Pa mbar mb MB kb KB gb GB tb "
    "TB T G M K % км км² км³ м м² м³ дм дм² дм³ см см² см³ мм мм² мм³ нм "
    "кг г мг м/с км/ч кПа Па мбар Кб КБ кб Мб МБ мб Гб ГБ гб Тб ТБ тб"
    "كم كم² كم³ م م² م³ سم سم² سم³ مم مم² مم³ كم غرام جرام جم كغ ملغ كوب اكواب"
)
_currency = r"\$ £ € ¥ ฿ US\$ C\$ A\$ ₽ ﷼ ₴"

# These expressions contain various unicode variations, including characters
# used in Chinese (see #1333, #1340, #1351) – unless there are cross-language
# conflicts, spaCy's base tokenizer should handle all of those by default
_punct = (
    r"… …… , : ; \! \? ¿ ؟ ¡ \( \) \[ \] \{ \} < > _ # \* & 。 ？ ！ ， 、 ； ： ～ · । ، ۔ ؛ ٪"
)
_quotes = r'\' " ” “ ` ‘ ´ ’ ‚ , „ » « 「 」 『 』 （ ） 〔 〕 【 】 《 》 〈 〉'
_hyphens = "- – — -- --- —— ~"

# Various symbols like dingbats, but also emoji
# Details: https://www.compart.com/en/unicode/category/So
_other_symbols = (
    r"\u00A6\u00A9\u00AE\u00B0\u0482\u058D\u058E\u060E\u060F\u06DE\u06E9\u06FD\u06FE\u07F6\u09FA\u0B70"
    r"\u0BF3-\u0BF8\u0BFA\u0C7F\u0D4F\u0D79\u0F01-\u0F03\u0F13\u0F15-\u0F17\u0F1A-\u0F1F\u0F34"
    r"\u0F36\u0F38\u0FBE-\u0FC5\u0FC7-\u0FCC\u0FCE\u0FCF\u0FD5-\u0FD8\u109E\u109F\u1390-\u1399"
    r"\u1940\u19DE-\u19FF\u1B61-\u1B6A\u1B74-\u1B7C\u2100\u2101\u2103-\u2106\u2108\u2109\u2114\u2116"
    r"\u2117\u211E-\u2123\u2125\u2127\u2129\u212E\u213A\u213B\u214A\u214C\u214D\u214F\u218A\u218B"
    r"\u2195-\u2199\u219C-\u219F\u21A1\u21A2\u21A4\u21A5\u21A7-\u21AD\u21AF-\u21CD\u21D0\u21D1\u21D3"
    r"\u21D5-\u21F3\u2300-\u2307\u230C-\u231F\u2322-\u2328\u232B-\u237B\u237D-\u239A\u23B4-\u23DB"
    r"\u23E2-\u2426\u2440-\u244A\u249C-\u24E9\u2500-\u25B6\u25B8-\u25C0\u25C2-\u25F7\u2600-\u266E"
    r"\u2670-\u2767\u2794-\u27BF\u2800-\u28FF\u2B00-\u2B2F\u2B45\u2B46\u2B4D-\u2B73\u2B76-\u2B95"
    r"\u2B98-\u2BC8\u2BCA-\u2BFE\u2CE5-\u2CEA\u2E80-\u2E99\u2E9B-\u2EF3\u2F00-\u2FD5\u2FF0-\u2FFB"
    r"\u3004\u3012\u3013\u3020\u3036\u3037\u303E\u303F\u3190\u3191\u3196-\u319F\u31C0-\u31E3"
    r"\u3200-\u321E\u322A-\u3247\u3250\u3260-\u327F\u328A-\u32B0\u32C0-\u32FE\u3300-\u33FF\u4DC0-\u4DFF"
    r"\uA490-\uA4C6\uA828-\uA82B\uA836\uA837\uA839\uAA77-\uAA79\uFDFD\uFFE4\uFFE8\uFFED\uFFEE\uFFFC"
    r"\uFFFD\U00010137-\U0001013F\U00010179-\U00010189\U0001018C-\U0001018E\U00010190-\U0001019B"
    r"\U000101A0\U000101D0-\U000101FC\U00010877\U00010878\U00010AC8\U0001173F\U00016B3C-\U00016B3F"
    r"\U00016B45\U0001BC9C\U0001D000-\U0001D0F5\U0001D100-\U0001D126\U0001D129-\U0001D164"
    r"\U0001D16A-\U0001D16C\U0001D183\U0001D184\U0001D18C-\U0001D1A9\U0001D1AE-\U0001D1E8"
    r"\U0001D200-\U0001D241\U0001D245\U0001D300-\U0001D356\U0001D800-\U0001D9FF\U0001DA37-\U0001DA3A"
    r"\U0001DA6D-\U0001DA74\U0001DA76-\U0001DA83\U0001DA85\U0001DA86\U0001ECAC\U0001F000-\U0001F02B"
    r"\U0001F030-\U0001F093\U0001F0A0-\U0001F0AE\U0001F0B1-\U0001F0BF\U0001F0C1-\U0001F0CF"
    r"\U0001F0D1-\U0001F0F5\U0001F110-\U0001F16B\U0001F170-\U0001F1AC\U0001F1E6-\U0001F202"
    r"\U0001F210-\U0001F23B\U0001F240-\U0001F248\U0001F250\U0001F251\U0001F260-\U0001F265"
    r"\U0001F300-\U0001F3FA\U0001F400-\U0001F6D4\U0001F6E0-\U0001F6EC\U0001F6F0-\U0001F6F9"
    r"\U0001F700-\U0001F773\U0001F780-\U0001F7D8\U0001F800-\U0001F80B\U0001F810-\U0001F847"
    r"\U0001F850-\U0001F859\U0001F860-\U0001F887\U0001F890-\U0001F8AD\U0001F900-\U0001F90B"
    r"\U0001F910-\U0001F93E\U0001F940-\U0001F970\U0001F973-\U0001F976\U0001F97A\U0001F97C-\U0001F9A2"
    r"\U0001F9B0-\U0001F9B9\U0001F9C0-\U0001F9C2\U0001F9D0-\U0001F9FF\U0001FA60-\U0001FA6D"
)

UNITS = merge_chars(_units)
CURRENCY = merge_chars(_currency)
PUNCT = merge_chars(_punct)
HYPHENS = merge_chars(_hyphens)
ICONS = _other_symbols

LIST_UNITS = split_chars(_units)
LIST_CURRENCY = split_chars(_currency)
LIST_QUOTES = split_chars(_quotes)
LIST_PUNCT = split_chars(_punct)
LIST_HYPHENS = split_chars(_hyphens)
LIST_ELLIPSES = [r"\.\.+", "…"]
LIST_ICONS = [r"[{i}]".format(i=_other_symbols)]

CONCAT_QUOTES = group_chars(_quotes)
CONCAT_ICONS = _other_symbols
