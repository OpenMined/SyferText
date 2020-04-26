import base64
from Crypto.Cipher import AES
from Crypto import Random

BS = 16
pad = lambda s: s + (BS - len(s) % BS) * chr(BS - len(s) % BS)
unpad = lambda s: s[: -ord(s[len(s) - 1 :])]

# TODO: Verify the security of ECB (Deterministic) mode.
# TODO: Encrypt integers ??


def encrypt(raw, key):
    """Encrypt string using key. The encryption algorithm followed
    is AES (Advanced Encryption Standard) in ECB mode.
    """
    raw = pad(raw)
    cipher = AES.new(key, AES.MODE_ECB)
    return base64.b64encode(cipher.encrypt(raw))


def decrypt(enc, key):
    """Decrypt AES (ECB mode) encrypted string using key.
    """
    enc = base64.b64decode(enc)
    cipher = AES.new(key, AES.MODE_ECB)
    return unpad(cipher.decrypt(enc))
