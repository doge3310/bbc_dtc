class BPETokenizer:
    def __init__(
            self,
            max_tokens_count: int = 10000,
            unk_id: int = 1,
            pad_id: int = 0):
        self.max_tokens_count = max_tokens_count
        self.dct = {}
        self.unk_id = unk_id
        self.pad_id = pad_id
        self.reserved_tokens = {"PAD": pad_id, "UNK": unk_id}
        self.sorted_tokens = []

    def _most_popular(self, text: str):
        popular_map = {}

        for index, token in enumerate(text):
            if index + 1 == len(text):
                break

            token = token + text[index + 1]

            try:
                popular_map[token] += 1

            except KeyError:
                popular_map[token] = 1

        if not popular_map:
            return None

        return max(popular_map, key=popular_map.get)

    def _step(self, text: list[str]):
        most_freq_token = self._most_popular(text)
        is_skip = 0
        result = []

        for index, item in enumerate(text):
            if is_skip:
                is_skip = 0
                continue

            if index + 1 < len(text) and \
                most_freq_token == item + text[index + 1]:

                result.append(most_freq_token)
                is_skip = 1

                continue

            result.append(item)

        return result

    def forward(self, text: str):
        offset = len(self.reserved_tokens)
        self.dct = {token: index + offset
                    for index, token in enumerate(dict.fromkeys(text))}

        while len(self.dct) < self.max_tokens_count and len(text) > 1:
            text = self._step(text=text)

            unique = list(dict.fromkeys(text))
            self.dct = {token: index + offset for index, token in enumerate(unique)}

        self.sorted_tokens = sorted(
            [t for t in self.dct.keys() if t],
            key=len, reverse=True
        )

    def tokenize(self, x: str, tgt_length: int):
        tokens = []
        i = 0

        while len(tokens) < tgt_length:
            if i >= len(x):
                tokens.append(self.pad_id)
                continue

            for token in self.sorted_tokens:
                if x.startswith(token, i):
                    tokens.append(self.dct[token])
                    i += len(token)
                    break

            else:
                tokens.append(self.unk_id)
                i += 1

        return tokens
