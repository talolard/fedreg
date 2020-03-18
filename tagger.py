from typing import Dict,List,Any

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model, CrfTagger
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy


class LstmTagger(CrfTagger):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs,label_namespace='labels')
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                labels: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:
        outputs = super().forward(tokens,labels,metadata,**kwargs)
        logits = outputs['logits']
        mask =outputs['mask']
        tags =outputs['tags'] #The viterbi predictions
        labels

        if labels is not None:
            outputs["loss"] = sequence_cross_entropy_with_logits(logits, labels, mask)

        return outputs
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {}
