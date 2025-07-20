from collections import defaultdict
import random

class MarkovResponder:
    """Generate simple text responses using a bigram Markov chain."""

    def __init__(self):
        self.transitions = defaultdict(list)
        self.start_words = []
        # seed with a small corpus so responses are not empty and vary a bit
        default_corpus = [
            "hello there",
            "how are you doing today",
            "it's nice to meet you",
            "let's discuss your plans",
            "tell me about your projects",
            "what else is on your mind",
        ]
        for line in default_corpus:
            self.add_sentence(line)

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
