from .punctuations import TOKENIZER_PREFIXES


import re

def compile_prefix_regex(entries: str):
    """Compile a sequence of prefix rules into a regex object.
    entries (tuple): The prefix rules, e.g. spacy.lang.punctuation.TOKENIZER_PREFIXES.
    Returns (re.Pattern): The regex object. to be used for Tokenizer.prefix_search.
    """
    if "(" in entries:
        # Handle deprecated data
        expression = "|".join(
            ["^" + re.escape(piece) for piece in entries if piece.strip()]
        )
        return re.compile(expression)
    else:
        expression = "|".join(["^" + piece for piece in entries if piece.strip()])
        return re.compile(expression)

pre_re = compile_prefix_regex(TOKENIZER_PREFIXES)
s = "(sdgfsdg"
print(pre_re(s))