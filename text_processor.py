from transformers import BertModel, BertTokenizer
import time
import torch
from torch import nn

class TextProcessor(nn.Module):
    def __init__(self, bert_model_name=r'D:/ml/bert-base-uncased', device = 'cpu'):
        super(TextProcessor, self).__init__()
        self.device = device

        self.bert_model = BertModel.from_pretrained(bert_model_name).to(self.device)  # BERT model
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        for param in self.bert_model.parameters():
            param.requires_grad = False  # Freeze BERT
                
    def forward(self, text):
        start = time.time()
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            self.bert_output = self.bert_model(**inputs).last_hidden_state.mean(dim=1)  # BERT output
        end = time.time()
        print(f"Time taken for text encoding : {end - start} seconds")

        return self.bert_output