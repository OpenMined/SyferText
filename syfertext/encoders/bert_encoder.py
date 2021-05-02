from typing import Dict, List
from transformers import BertTokenizer

class BERTEncoder:
    
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def __call__(self, text:List) -> Dict:
        inputs = self.tokenizer(text)
        return {"token_ids": inputs["input_ids"]}