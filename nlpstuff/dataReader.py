import json
from typing import Dict, List, Iterator

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token


class CharLevelReader(DatasetReader):
    """
    DatasetReader for ner labels charachter level

    """

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            data = json.load(f)
            for example in data:
                content = example["content"]
                cursor = 0
                annotations = sorted(example["annotations"], key=lambda x: x["start"])
                slices: List[str] = []
                labels: List[str] = []
                for anno in annotations:
                    if cursor > anno["start"]:
                        continue
                    elif cursor < anno["start"]:
                        slices.append(content[cursor : anno["start"]])
                        labels.append("O")
                    slices.append(content[anno["start"] : anno["end"]])
                    labels.append(anno["tag"])
                    cursor = anno["end"]
                if cursor < len(content):
                    slices.append(content[cursor:])
                    labels.append("O")
                tokens: List[Token] = []
                tags: List[str] = []
                for s, t in zip(slices, labels):
                    res = self._slice_and_tag_to_tokens(s, t)
                    tokens += res[0]
                    tags += res[1]
                yield self.text_to_instance(tokens, tags)

    @staticmethod
    def _slice_and_tag_to_tokens(slice: str, tag: str):
        """

        :param slice: the input slice
        :param tag: the tag applied to the slice
        :return: tokens: charachters, tags, bilo encoded tags
        """
        tokens: List[Token] = []
        tags: List[str] = []
        for ix, tok in enumerate(list(slice)):
            if tag == "O":
                t = "O"
            elif ix == 0:
                t = "B-" + tag
            elif ix == len(slice) - 1:
                t = "L-" + tag
            else:
                t = "I-" + tag
            tokens.append(Token(tok))
            tags.append(t)
        return tokens, tags
