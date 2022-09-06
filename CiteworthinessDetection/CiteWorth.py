"""
    The Implementation below is an adapted (but still copied) version from https://github.com/copenlu/cite-worth,
        which is the original implementation by Dustin Wright and Isabelle Augenstein from the Paper
        @inproceedings{wright2021citeworth,
                title={{CiteWorth: Cite-Worthiness Detection for Improved Scientific Document Understanding}},
                author={Dustin Wright and Isabelle Augenstein},
                booktitle = {Findings of ACL-IJCNLP},
                publisher = {Association for Computational Linguistics},
                year = 2021
            }
"""

from typing import List, Tuple, AnyStr

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import PreTrainedTokenizer

from CiteworthinessDetection.CiteworthinessDetection import CiteworthinessDetection


class CiteWorth(CiteworthinessDetection):

    def __init__(self, model_path: str, use_section_info: str, model_name: str = "allenai/longformer-base-4096"):
        """
        :param model_path: path to pretrained model *.pth file
        :param use_section_info: either 'always', 'first', 'extra' or None depending on the pretrained model variant
        :param model_name: name of the transformer model to be loaded from huggingface fitting the pretrained model variant
        """
        super().__init__()
        self.use_section_info = use_section_info

        # See if CUDA available
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")

        # initialize the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = _AutoTransformerForSentenceSequenceModeling(
            model_name,
            num_labels=2,
            sep_token_id=self.tokenizer.sep_token_id,
            is_section_info_extra=use_section_info == 'extra'
        ).to(self.device)

        # load the pretrained model
        model_dict = self.model.state_dict()
        load_model = self.model
        weights = torch.load(model_path, map_location=lambda storage, loc: storage)
        model_dict.update(weights)
        load_model.load_state_dict(model_dict)

        self.model.eval()

    def predict(self, sentences: List[Tuple[str, str]], section: str) -> List[Tuple[str, str]]:
        with torch.no_grad():
            # process input
            input_ids, masks = self._transform_predict_input_to_model_input(sentences, section)
            input_ids = torch.LongTensor(input_ids)
            masks = torch.LongTensor(masks)
            input_ids = input_ids.to(self.device)
            masks = masks.to(self.device)

            # perform the prediction
            raw_outputs = self.model(input_ids=input_ids, attention_mask=masks)['logits'].detach().cpu().numpy()
            preds = np.argmax(raw_outputs.reshape(-1, 2), axis=-1)

        # process the output
        citeworthy_sents = [s for s, label in zip(sentences, preds) if label == 1]
        return citeworthy_sents

    def _transform_predict_input_to_model_input(self, sentences: List[Tuple[str, str]], section: str):
        sents = [s[1] for s in sentences]

        if self.use_section_info == 'first':
            sents[0] = section + ':' + sents[0]
        elif self.use_section_info == 'always':
            sents = [section + ' ' + s for s in sents]
        elif self.use_section_info == 'extra':
            sents.insert(0, section)
        # Calls the text_to_batch function
        return self._text_to_sequence_batch_transformer(sents, self.tokenizer)

    @staticmethod
    def _text_to_sequence_batch_transformer(text: List, tokenizer: PreTrainedTokenizer) -> Tuple[List, List]:
        """Turn a list of text into a sequence of sentences separated by SEP token

        :param text: The text to tokenize and encode
        :param tokenizer: The tokenizer to use
        :return: A list of IDs and a mask
        """
        max_length = min(512, tokenizer.model_max_length)
        input_ids = [tokenizer.encode(t, add_special_tokens=True, max_length=max_length, truncation=True, verbose=False)
                     for
                     t in text]
        input_ids = [[id_ for i, sent in enumerate(input_ids) for j, id_ in enumerate(sent) if (i == 0 or j != 0)][
                     :tokenizer.model_max_length]]
        input_ids[0][-1] = tokenizer.sep_token_id

        masks = [[1] * len(i) for i in input_ids]

        return input_ids, masks


class _AutoTransformerForSentenceSequenceModeling(nn.Module):
    """
       Implements a transformer which performs sequence classification on a sequence of sentences
    """

    def __init__(self, transformer_model: AnyStr, num_labels: int = 2, sep_token_id: int = 2,
                 is_section_info_extra: bool = False):
        super(_AutoTransformerForSentenceSequenceModeling, self).__init__()

        config = AutoConfig.from_pretrained(transformer_model)
        self.config = config
        self.xformer = AutoModel.from_pretrained(transformer_model, config=config)

        # Pooling layers
        self.pooling = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.Tanh()

        # Create the classifier heads
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.num_labels = num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sep_token_id = sep_token_id
        self.is_section_info_extra = is_section_info_extra

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            lam=1.0
    ):
        outputs = self.xformer(
            input_ids,
            attention_mask=attention_mask,
        )

        # Gather all of the SEP hidden states
        hidden_states = outputs['last_hidden_state'].reshape(-1, self.config.hidden_size)
        locs = (input_ids == self.sep_token_id)
        if self.is_section_info_extra:
            # remove first SEP token for each batch in locs as it belongs to the section information
            # this is done by changing the first True to False for each batch
            for i, _ in enumerate(locs):
                locs[i][torch.nonzero(locs[i] == True, as_tuple=False)[0]] = False
        # (n * seq_len x d) -> (n * sep_len x d)
        locs = locs.view(-1)
        sequence_output = hidden_states[locs]
        assert sequence_output.shape[0] == sum(locs)
        assert sequence_output.shape[1] == self.config.hidden_size

        pooled_output = self.pooling(sequence_output)
        pooled_output = self.dropout(self.act(pooled_output))

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = lam * loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {'loss': loss, 'logits': logits}
