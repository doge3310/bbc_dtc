"""Byte Pair Encoding tokenizer"""
import regex as re


class BPETokenizer:
    def __init__(
            self,
            text: str,
            max_tokens_count: int = 50000,
            **kwargs: int):
        self.max_tokens_count = max_tokens_count
        self.dct = None
        self.text = list(text)

    def _step(self):
        popular_map = {}

        for index, token in enumerate(self.text):
            if index + 1 == len(self.text):
                break

            token = token + self.text[index + 1]

            try:
                popular_map[token] += 1

            except KeyError:
                popular_map[token] = 1

        most_freq_token = max(popular_map, key=popular_map.get)
        is_skip = 0
        text = []

        for index, item in enumerate(self.text):
            if is_skip:
                is_skip = 0
                continue

            if index + 1 < len(self.text) and \
                most_freq_token == item + self.text[index + 1]:

                text.append(most_freq_token)
                is_skip = 1

                continue

            text.append(item)

        self.text = text

    def forward(self):
        while not self.dct or len(self.dct.values()) > self.max_tokens_count:
            self._step()
            unique = list(dict.fromkeys(self.text))
            self.dct = {token: index for index, token in enumerate(unique)}

    def tokenize(self, x: str):
        sorted_tokens = sorted(self.dct.keys(), key=len, reverse=True)
        sorted_tokens = [re.escape(item) for item in sorted_tokens]
        sorted_tokens = "|".join(sorted_tokens)

        sorted_tokens = re.findall(sorted_tokens, x)
        tokens = [self.dct.get(item) for item in sorted_tokens]

        return tokens


if __name__ == "__main__":
    tokenizer = BPETokenizer("adsj alksdf lkasjfd lka, asd, a.", 12)
    tokenizer.forward()

    print(tokenizer.tokenize("assj alksds"))
