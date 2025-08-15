import json 
import os 
from typing import List, Tuple, Generator

from .bpe import learn_bpe, apply_bpe_to_word
from .vocab import Vocab
from config import Config


class Tokenizer: 
    def __init__(
        self, 
        vocab: Vocab,
        merges: List[Tuple[str, str]],
    ):


    @classmethod
    def train(
        cls, 
        text_iter: Generator[str, None, None],
        vocab_size: int = Config.vocab_size,
        min_pair_freq: int = Config.min_pair_freq,
        lower_case: bool = Config.lower_case,
    ) -> "Tokenizer":

        merges, symbols = learn_bpe(
            text_iter, vocab_size=vocab_size, min_pair_freq=min_pair_freq
        )

        vocab_tokens = symbols
        vocab = Vocab(tokens=vocab_tokens, specials=Config.SPECIAL_TOKENS)
        return cls(vocab, merges)
        
    
    def encode(
        self, 
        text: str,
        add_bos: bool = True,
        lower_case: bool = Config.lower_case,
    ) -> List[int]:

        if lower_case:
            text = text.lower()

            ids: List[int] = []

            if add_bos:
                ids.append(self.vocab.bos_id)
            
            for word in text.strip().split():
                pieces = apply_bpe_to_word(word, self.merges)
                ids.extend(self.vocab.encode_tokens(pieces))
            
            if add_eos:
                ids.append(self.vocab.eos_id)
            
            return ids
    
    def decode(
        self, 
        ids: List[int],
    ) -> str:
    
        tokens = self.vocab.decode_ids(ids)

        words: List[str] = []
        cur: List[str] = []

        for token in tokens:
            if token in (
                Config.SPECIAL_TOKENS["BOS"], Config.SPECIAL_TOKENS["EOS"], Config.SPECIAL_TOKENS["PAD"], Config.SPECIAL_TOKENS["UNK"],
            ):
                continue

            if token.endswith(Config.ENCODING_SPECIALS["END_OF_WORD"]):
                cur.append(token.replace(Config.ENCODING_SPECIALS["END_OF_WORD"], ""))
                words.append("".join(cur))
                cur = []
            
            else:
                cur.append(token)
        
        if cur:
            words.append("".join(cur))
        
        return " ".join(words)
    

    def save(
        self,
        path: str,
    ):
        os.makedirs(path, exist_ok=True)
        self.vocab.save(path)

        with open(os.path.join(path, "merges.json"), "w", encoding="utf-8") as f:
            json.dump(self.merges, f, ensure_ascii=False
    
    @classmethod
    def load(
        cls,
        path: str,
    ) -> "Tokenizer":

        vocab = Vocab.load(path)
        with open(os.path.join(path, "merges.json"), "r", encoding="utf-8") as f:
            merges = json.load(f)
        return cls(vocab, merges)