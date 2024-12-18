from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from Model.BERT.encoder import Encoder


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.use_gpu = self.opt.use_gpu

        # Initialize BERT
        self.init_bert()

        # Encoder
        self.encoder = Encoder(opt.enc_method, self.word_dim, opt.hidden_size, opt.out_size)

        # Classification layer
        self.cls = nn.Linear(opt.out_size, opt.num_labels)
        nn.init.uniform_(self.cls.weight, -0.1, 0.1)
        nn.init.uniform_(self.cls.bias, -0.1, 0.1)

        # Dropout layer
        self.dropout = nn.Dropout(self.opt.dropout)

    def forward(self, x):
        """
        Forward pass
        """
        # Get word embeddings from BERT
        word_embs = self.get_bert(x)

        # Pass through encoder and classifier
        x = self.encoder(word_embs)
        x = self.dropout(x)
        x = self.cls(x)  # batch_size * num_labels
        return x

    def init_bert(self):
        """
        Initialize the BERT model
        """
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.opt.bert_path)
        self.bert = AutoModel.from_pretrained(self.opt.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze BERT parameters
        self.word_dim = self.opt.bert_dim

    def get_bert(self, sentence_lists):
        """
        Get the BERT word embedding vectors for sentences
        """
        sentence_lists = [' '.join(x) for x in sentence_lists]
        ids = self.bert_tokenizer(sentence_lists, padding=True, return_tensors="pt")
        inputs = ids['input_ids']
        attention_mask = ids['attention_mask']

        if self.opt.use_gpu:
            inputs = inputs.to(self.opt.device)
            attention_mask = attention_mask.to(self.opt.device)

        # Pass inputs through BERT
        embeddings = self.bert(input_ids=inputs, attention_mask=attention_mask)
        return embeddings.last_hidden_state