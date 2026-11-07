import re
from collections import Counter

from src.interfaces.idata import ITokenizer


class WordTokenizer(ITokenizer):
	def __init__(self,
		pad_token: int = 0,
		unk_token: int = 1,
		dict_size: int = 2000):

		super().__init__(pad_token, unk_token, dict_size)

		self.dict_size = dict_size
		self.unk_token = unk_token
		self.dict = {"PAD": pad_token, "UNK": unk_token}

	def _split(self,
		text: str):

		return [w for w in re.split(pattern=r"[^\w]+", string=text.lower()) if w]

	def forward(self,
		text: str):

		super().forward(text)

		splited_text = self._split(text)
		counts = Counter(splited_text)
		top_words = counts.most_common(self.dict_size - 2)

		for token, (word, _) in enumerate(top_words, 2):
			self.dict[word] = token

		return self.dict

	def tokenize(self,
		text: str):

		super().tokenize(text)

		splited_text = self._split(text=text)
		tokenized_text = []

		for word in splited_text:
			if word:
				try:
					tokenized_text.append(self.dict[word.lower()])

				except KeyError:
					tokenized_text.append(self.unk_token)

		return tokenized_text
