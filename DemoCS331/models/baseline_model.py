import torch.nn as nn
import torch

class FeedForwardLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.lnorm = nn.LayerNorm(hidden_dim)  # BatchNorm cho hidden_dim
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.lnorm(self.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class AddAndNorm(nn.Module):
    def __init__(self, input_dim):
        super(AddAndNorm, self).__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.feedForward = FeedForwardLayer(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x, addx):
        x = self.norm1(x + addx)
        x = self.norm2(x + self.feedForward(x))
        return x

class EncoderBlock(nn.Module):
    def __init__(self, dim_kv_input, dim_q_input):
        super(EncoderBlock, self).__init__()
        self.q_projection = nn.Linear(dim_q_input, dim_kv_input)
        self.attention = nn.MultiheadAttention(dim_kv_input, num_heads=1)
        self.aan = AddAndNorm(dim_kv_input)

    def forward(self, q, k, v):
        newq = self.q_projection(q)
        attention_output, _ = self.attention(newq, k, v)
        output = self.aan(attention_output, newq)
        return output

class BaselineModel(nn.Module):
    def __init__(self, dim_kv_input, dim_q_input, n_head=4):
        super(BaselineModel, self).__init__()
        self.n_head = n_head
        self.encoder_block = EncoderBlock(dim_kv_input, dim_q_input)
        self.W_o = nn.Sequential(
            nn.Linear(dim_kv_input * n_head, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            nn.Linear(128, 4)
        )

    def forward(self, qlist, k, v):
        output = []
        for i in range(4):
            output.append(self.encoder_block(qlist[:, i, :], k, v))
        concatenated_output = torch.cat(output, dim=1)
        output = self.W_o(concatenated_output)
        return output