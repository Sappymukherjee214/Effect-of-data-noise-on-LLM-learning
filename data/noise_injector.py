import random
import numpy as np
from typing import List, Union

class NoiseInjector:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab = list(tokenizer.get_vocab().values())
        self.special_tokens = set(tokenizer.all_special_ids)

    def random_token_replacement(self, tokens: List[int], p: float = 0.15) -> List[int]:
        """
        Replaces p% of tokens with a random token from the vocabulary.
        """
        new_tokens = tokens.copy()
        for i in range(len(new_tokens)):
            if new_tokens[i] not in self.special_tokens and random.random() < p:
                new_tokens[i] = random.choice(self.vocab)
        return new_tokens

    def word_shuffling(self, tokens: List[int], window_size: int = 3) -> List[int]:
        """
        Shuffles tokens within a sliding window to disrupt local syntax.
        """
        new_tokens = tokens.copy()
        for i in range(0, len(new_tokens), window_size):
            end = min(i + window_size, len(new_tokens))
            chunk = new_tokens[i:end]
            random.shuffle(chunk)
            new_tokens[i:end] = chunk
        return new_tokens

    def grammar_corruption(self, text: str, mode: str = "remove_stops") -> str:
        """
        Applies rule-based grammatical corruption to raw text.
        """
        words = text.split()
        if mode == "remove_stops":
            # Simple simulation of removing high-frequency filler words
            # In practice, use a stopword list or POS tagging
            return " ".join([w for w in words if len(w) > 3])
        elif mode == "swap_case":
            return "".join([c.swapcase() if random.random() < 0.1 else c for c in text])
        return text

    def apply_noise(self, tokens: List[int], noise_type: str = "token_replacement", intensity: float = 0.1) -> List[int]:
        """
        Generalized interface to apply noise.
        """
        if noise_type == "token_replacement":
            return self.random_token_replacement(tokens, p=intensity)
        elif noise_type == "shuffling":
            return self.word_shuffling(tokens, window_size=int(intensity * 10) + 1)
        return tokens
