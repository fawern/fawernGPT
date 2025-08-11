import json 
import os 
from typing import List, Dict, Any
from config import Config


class Vocab:

    def __init__(
        self, 
        tokens: List[str],
        specials: Dict[str, str] = None
    ):
        specials: Dict[str, str] = specials or Config.SPECIAL_TOKENS
        self.specials : List[str] = list(specials.values())

        base: List[str] = []

        for s in self.specials:
            if s not in base:
                base.append(s)
        
        for t in tokens:
            if t not in base:
                base.append(t)
        
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(base)}
        self.id_to_token: Dict[int, str] = {i: t for i, t in enumerate(base)}

        self.pad_id: int = self.token_to_id[self.specials["PAD"]]
        self.bos_id: int = self.token_to_id[self.specials["BOS"]]
        self.eos_id: int = self.token_to_id[self.specials["EOS"]]
        self.unk_id: int = self.token_to_id[self.specials["UNK"]]

    
    def encode_tokens(
        self,
        tokens: List[str]
    ) -> List[int]:

        return [self.token_to_id.get(t, self.unk_id) for t in tokens]
    
    def decode_ids(
        self,
        ids: List[int]
    ) -> List[str]:
        return [self.id_to_token.get(i, self.specials["UNK"]) for i in ids]
    
    def save(
        self,
        path: str
    ):
        with open(os.path.join(path, 'vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(
                {
                    "tokens": self.id_to_token,
                    "specials": self.specials
                },
                f,
                ensure_ascii=False
            )

    
    @classmethod
    def load(
        cls,
        path: str
    ) -> "Vocab":
        
        with open(os.path.join(path, 'vocab.json'), 'r', encoding='utf-8') as f:
            blob = json.load(f)
        
        max_id = max(int(i) for i in blob["tokens"].keys()) if blob["tokens"] else -1
        tokens_ordered = [blob["tokens"][str(i)] for i in range(max_id+1)]

        return cls(
            tokens_ordered,
            specials={
                "PAD": tokens_ordered[0],
                "BOS": tokens_ordered[1],
                "EOS": tokens_ordered[2],
                "UNK": tokens_ordered[3]
            }
        )