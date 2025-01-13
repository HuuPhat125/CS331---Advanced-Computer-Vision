import torch.nn as nn
import torch

class FeedForwardLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim = 512):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.LazyLinear(hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.LazyLinear(input_dim)
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
class AddAndNorm(nn.Module):
    def __init__(self, input_dim):
        super(AddAndNorm, self).__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.feedForward = FeedForwardLayer(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
    
    def forward(self, x, addx):
        norm_x = self.norm1(x + addx)
        return self.norm2(norm_x + self.feedForward(norm_x))
    
class EncoderBlock(nn.Module):
    def __init__(self, dim_kv_input, dim_q_input, n_head = 4):
        super(EncoderBlock, self).__init__()
        self.n_head = n_head
        self.q_projection = nn.Linear(dim_q_input, dim_kv_input)
        self.multi_head_attention = nn.ModuleList([nn.MultiheadAttention(dim_kv_input, num_heads = 4) for _ in range(n_head)])
        self.flayer = nn.ModuleList([AddAndNorm(dim_kv_input) for _ in range(n_head)])
    
    def forward(self, qlist, k, v):
        qlist = self.q_projection(qlist)
        flayer = []
        for i in range(self.n_head):
            attention_output, _ = self.multi_head_attention[i](qlist[:, i, :], k, v)
            flayer.append(self.flayer[i](attention_output, qlist[:, i, :]))
        
        return flayer


class BaselineModel(nn.Module):
    def __init__(self, dim_kv_input, dim_q_input, n_head=4):
        super(BaselineModel, self).__init__()
        self.n_head = n_head
        self.encoder_block = EncoderBlock(dim_kv_input, dim_q_input)
        
        self.W_o = nn.Sequential(
                        nn.LazyLinear(512),
                        nn.ReLU(),
                        nn.LazyLinear(128),
                        nn.ReLU(),
                        nn.LazyLinear(4)
                    )
        
    def forward(self, qlist, k, v):
        output = self.encoder_block(qlist, k, v)
        
        concatenated_output = torch.cat(output, dim=1)
        output = self.W_o(concatenated_output)
        return output

