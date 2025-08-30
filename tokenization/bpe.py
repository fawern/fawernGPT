from collections import Counter, defaultdict
from config import Config
from typing import List, Dict, Tuple, Generator


def _word_to_symbols(word: str) -> List[str]:
    config = Config()
    return list(word) + [config.ENCODING_SPECIALS["END_OF_WORD"]]


def _get_stats(words: Dict[Tuple[str, ...], int]) -> Counter:

    pairs: Counter = Counter()

    for symbols, freq in words.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq

    return pairs


def _merge_pair(
    pair: Tuple[str, str], 
    words: Dict[Tuple[str, ...], int]
) -> Dict[Tuple[str, ...], int]:

    a, b = pair
    new_words: Dict[Tuple[str, ...], int] = {}
    bigram: Tuple[str, str] = (a, b)

    for symbols, freq in words.items():

        i = 0
        new_sym = []

        while i < len(symbols):

            if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == bigram:
                new_sym.append(a + b)
                i += 2

            else:
                new_sym.append(symbols[i])
                i += 1
                
        new_words[tuple(new_sym)] = freq
    return new_words



def learn_bpe(
    text_iter: Generator[str, None, None], 
    vocab_size: int = 1000, 
    min_pair_freq: int = 2, 
) -> Tuple[List[Tuple[str, str]], List[str]]:

    word_freq = Counter()
    for line in text_iter:
        for word in line.strip().split():
            word_freq[word] += 1


    
    words = {tuple(_word_to_symbols(word)): freq for word, freq in word_freq.items()}


    merges = []

    symbols = set()
    for w in words.keys():
        symbols.update(w)


    while len(symbols) < vocab_size:
        pairs = _get_stats(words)

        if not pairs:
            break

        (a, b), freq = pairs.most_common(1)[0]

        if freq < min_pair_freq:
            break

        merges.append((a, b))
        words = _merge_pair((a, b), words)
        symbols = set()
        for w in words.keys():
            symbols.update(w)

    symbols = list(symbols)
    return merges, symbols


def apply_bpe_to_word(
    word: str, 
    merges: List[Tuple[str, str]]
) -> List[str]:

    symbols = _word_to_symbols(word)
    merge_map = defaultdict(lambda: -1)
    for i, (a, b) in enumerate(merges):
        merge_map[(a, b)] = i

    def get_pair_priority(a: str, b: str) -> int:
        return merge_map[(a, b)]

    while True:

        candidates = []
        for i in range(len(symbols)-1):
            prio = get_pair_priority(symbols[i], symbols[i+1])

            if prio != -1:
                candidates.append((prio, i))

        if not candidates:
            break

        _, idx = min(candidates)
        merged = symbols[idx] + symbols[idx+1]
        symbols = symbols[:idx] + [merged] + symbols[idx+2:]

    return symbols