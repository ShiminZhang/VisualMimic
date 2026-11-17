"""CLIP architecture/tokenizer from official config; ALL weights loaded by TMED checkpoint."""
import torch
from torch import nn


class CheckpointClipTextEncoder(nn.Module):
    def __init__(self, modelpath, finetune=False, last_hidden_state=True, **kwargs):
        super().__init__()
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        if not last_hidden_state: raise ValueError('Only released TMED hidden-state conditioning is supported')
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath, local_files_only=True)
        self.text_model = AutoModel.from_config(AutoConfig.from_pretrained(modelpath, local_files_only=True))
        self.max_length = self.tokenizer.model_max_length
        self.text_encoded_dim = self.text_model.config.text_config.hidden_size
        if not finetune:
            self.text_model.requires_grad_(False)
        self.eval()

    def forward(self, texts):
        tokens = self.tokenizer(texts, padding='max_length', truncation=True,
                                max_length=self.max_length, return_tensors='pt')
        device = self.text_model.device
        ids = tokens.input_ids.to(device)
        mask = tokens.attention_mask.to(device)
        # Matches upstream ClipTextEncoder: no attention_mask passed into CLIP text_model.
        return self.text_model.text_model(ids).last_hidden_state, mask.bool()
