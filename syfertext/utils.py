import mmh3

def hash_string(string):

    key = mmh3.hash64(string, signed = False, seed = 1)[0]

    return key

