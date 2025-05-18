import torch
import torch.nn as nn
from mamba_ssm import Mamba

class ResidualNet(nn.Module):
    def __init__(self, block, embed_size, hidden_size):
        super().__init__()
        self.block = block
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        out = self.block(x)
        out = self.norm(out + x)
        return out

class TranslateModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=4):
        super(TranslateModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        layers = []
        for _ in range(num_layers):
            layers.append(ResidualNet(Mamba(d_model=hidden_size, d_state=8,  d_conv=4,    expand=2,), embed_size, hidden_size))
        self.layers = nn.Sequential(*layers)
        self.final_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids) 
        x = self.layers(x)           
        x = self.final_norm(x)
        logits = self.fc(x)          
        return logits
