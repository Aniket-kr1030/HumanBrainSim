from collections import defaultdict
import random

class MarkovResponder:
    """Generate simple text responses using a bigram Markov chain."""

    def __init__(self):
        self.transitions = defaultdict(list)
        self.start_words = []
        # seed with a small corpus so responses are not empty
        self.add_sentence("hello there")
        self.add_sentence("how are you doing today")
        self.add_sentence("this is a demo")

    def add_sentence(self, sentence: str) -> None:
        words = sentence.lower().split()
        if not words:
            return
        self.start_words.append(words[0])
        for a, b in zip(words, words[1:]):
            self.transitions[a].append(b)
        self.transitions[words[-1]].append(None)

    def generate(self, max_words: int = 10) -> str:
        if not self.transitions:
            return "..."
        word = random.choice(self.start_words)
        out = [word]
        for _ in range(max_words - 1):
            next_candidates = self.transitions.get(word)
            if not next_candidates:
                break
            next_word = random.choice(next_candidates)
            if next_word is None:
                break
            out.append(next_word)
            word = next_word
        return " ".join(out)
