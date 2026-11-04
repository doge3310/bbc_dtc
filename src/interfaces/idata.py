from abc import abstractmethod


class ITokenizer:
	def __init__(self, pad_token: int, unk_token: int, dict_size: int): pass

	@abstractmethod
	def forward(self, text: str) -> dict[str, int]: pass

	@abstractmethod
	def tokenize(self, text: str) -> list[int]: pass
