from bertmodel import MyBertModel

from transformers import BertModel
from config import BertConfig
from torch import nn

import torch



class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.config = BertConfig()
        self.bert = MyBertModel(self.config)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

    def initialize_pretrained_backbone(self):
        my_state_dict = self.bert.state_dict()

        hf_model = BertModel.from_pretrained('bert-base-uncased')
        # hf_model = torch.load('pytorch_model.bin')
        state_dict = hf_model.state_dict()
        mapped_state_dict = {}

        for my_key, hf_key in zip(my_state_dict.keys(), state_dict.keys()):
            mapped_state_dict[my_key] = state_dict[hf_key]

        self.bert.load_state_dict(mapped_state_dict)
        print("backbone successfully initialized!!")



