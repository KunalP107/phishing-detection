"""Attack simulation: obfuscated copies of phishing emails for the robustness test (owner: Anurag)."""
import random
import re

SUBS = {'a': '@', 'o': '0', 'i': '1', 'e': '3', 's': '$', 'l': '1'}


class Obfuscator:
    def __init__(self, benign_pool, seed=42):
        self.rng = random.Random(seed)
        self.benign_pool = list(benign_pool)

    def char_sub(self, text, rate=0.2):
        """Swap one letter for a look-alike symbol in about 20% of words, e.g. paypal -> paypa1."""
        out = []
        for w in text.split(' '):
            if len(w) > 3 and w.isalpha() and self.rng.random() < rate:
                pos = [i for i, ch in enumerate(w.lower()) if ch in SUBS]
                if pos:
                    i = self.rng.choice(pos)
                    w = w[:i] + SUBS[w[i].lower()] + w[i + 1:]
            out.append(w)
        return ' '.join(out)

    @staticmethod
    def url_disguise(text):
        """Defang links so they no longer look like URLs, e.g. https://bank.com -> hxxps://bank[.]com."""
        return re.sub(r'https?://(\S+)', lambda m: 'hxxps://' + m.group(1).replace('.', '[.]'), text, flags=re.IGNORECASE)

    def benign_padding(self, text, n_words=60):
        """Place about 60 words of legitimate email text before the phishing content."""
        words = ' '.join(self.rng.sample(self.benign_pool, 3)).split()[:n_words]
        return ' '.join(words) + ' ' + text

    def variants(self):
        return {'Original': lambda t: t,
                'Character substitution': self.char_sub,
                'Disguised URLs': self.url_disguise,
                'Benign padding': self.benign_padding,
                'All combined': lambda t: self.benign_padding(self.url_disguise(self.char_sub(t)))}
