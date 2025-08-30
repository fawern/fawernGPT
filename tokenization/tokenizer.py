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
        self.vocab = vocab
        self.merges = merges


    @classmethod
    def train(
        cls, 
        text_iter: Generator[str, None, None],
        vocab_size: int = 1000,
        min_pair_freq: int = 2,
        lower_case: bool = True,
    ) -> "Tokenizer":

        merges, symbols = learn_bpe(
            text_iter, vocab_size=vocab_size, min_pair_freq=min_pair_freq
        )

        config = Config()
        vocab_tokens = symbols
        vocab = Vocab(tokens=vocab_tokens, specials=config.SPECIAL_TOKENS)
        return cls(vocab, merges)
        
    
    def encode(
        self, 
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        lower_case: bool = True,
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

        config = Config()
        for token in tokens:
            if token in (
                config.SPECIAL_TOKENS["BOS"], config.SPECIAL_TOKENS["EOS"], config.SPECIAL_TOKENS["PAD"], config.SPECIAL_TOKENS["UNK"],
            ):
                continue

            if token.endswith(config.ENCODING_SPECIALS["END_OF_WORD"]):
                cur.append(token.replace(config.ENCODING_SPECIALS["END_OF_WORD"], ""))
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
            json.dump(self.merges, f, ensure_ascii=False)
    
    @classmethod
    def load(
        cls,
        path: str,
    ) -> "Tokenizer":

        vocab = Vocab.load(path)
        with open(os.path.join(path, "merges.json"), "r", encoding="utf-8") as f:
            merges = json.load(f)
        return cls(vocab, merges)