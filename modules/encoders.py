import torch
import torch.nn.functional as F
import time
import math
from torch import nn

class SubNet(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, n_class, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        # self.norm = nn.LayerNorm(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, n_class)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        
        dropped = self.drop(x)
        # normed = self.norm(dropped)
        y_1 = torch.relu(self.linear_1(dropped))
        fusion = self.linear_2(y_1)
        y_2 = torch.relu(fusion)
        y_3 = (self.linear_3(y_2))
        # return y_3,fusion
        return y_3

class RNNEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super().__init__()
        self.bidirectional = bidirectional

        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        # self.dropout = nn.Dropout(dropout)
        # self.linear_1 = nn.Linear((2 if bidirectional else 1)*hidden_size, out_size)

    def forward(self, x):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        output, (h,c) = self.rnn(x)
        # print('out_pack_data_shape:{}'.format(out_pack.data.shape))
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        if d_model % 2==1:
            dim = d_model+1
        else:
            dim = d_model

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # pe = self.pe[:x.size(0), :]
        # print(pe.shape)
        x = x + self.pe[:x.size(0), :,:self.d_model]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    "Transformer 编码器, 用于提取visual 和audio 特征"
    def __init__(self, ninp=300, nhead=4, nhid=128, nlayers=3, dropout=0.5):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout, 50)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        

    def generate_square_subsequent_mask(self, src, lenths):
        '''
        padding_mask
        src:max_lenth,batch_size,dim
        lenths:[lenth1,lenth2...]
        '''

        # mask num_of_sens x max_lenth
        mask = torch.ones(src.size(1), src.size(0)) == 1  # 全部初始化为 True
        # batch_size, seq_length
        for i in range(len(lenths)):# batch_size
            lenth = lenths[i]
            for j in range(lenth): 
                mask[i][j] = False  # 设置前面的部分为False

        return mask

    def forward(self, src, mask=None):
        '''
        src:num_of_all_sens,max_lenth,300
        '''
        if mask==None:
            src = src * math.sqrt(self.ninp)
            src = self.pos_encoder(src)
            output = self.transformer_encoder(src)
            output = output
        
        else:
            self.src_mask = mask

            src = src * math.sqrt(self.ninp)
            src = self.pos_encoder(src)
            output = self.transformer_encoder(src, src_key_padding_mask=self.src_mask)
            output = output
        return output
    
    def generate_padding_mask(self,seq):
        """
        seq: tensor of shape (max_length, batch_size, embedding_dim)
        pad_token: int, the padding token id
        """
        mask = torch.ones(seq.size(1), seq.size(0)) == 1
        seq = seq.permute(1,0,2)  # batch, len, dim
        mask[seq[:, :, 0] != 0.0]=False  # batch_size, len
        seq = seq.permute(1,0,2)
        return mask.to(torch.bool)

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m
