class DefaultTokenizer:
    def __init__(self, prefixes, suffixes, infixes, exceptions):

        self.prefixes = prefixes
        self.suffixes = suffixes
        self.infixes = infixes
        self.exceptions = exceptions

    def __call__(self, text: str):

        return text.split(" ")
