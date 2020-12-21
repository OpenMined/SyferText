class DefaultTokenizer:
    def __init__(self, prefixes, suffixes, infixes):

        self.prefixes = ["1", "2"]
        self.suffixes = ["3", "4"]
        self.infixes = ["5", "6"]

    def __call__(self, text: str):

        return text.split(" ")
