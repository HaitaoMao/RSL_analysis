import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self,input_size, hidden_size, dropout_embedding, dropout_encoder):
        super(Encoder, self).__init__()
        self.wordEmbeddingDim = input_size
        self.hidden_size = hidden_size
        self.encoder = nn.GRU(input_size,self.hidden_size, bidirectional=True)
        # 这里用的是GRU而不是LSTM
        self.dropout_embedding = nn.Dropout(p = dropout_embedding)
        self.dropout_encoder = nn.Dropout(p = dropout_encoder)
        # 哪里都给我来一个dropout

    def forward(self, seq, lens):
        batch_size = seq.shape[0]
        lens_sorted, lens_argsort = torch.sort(lens, 0, True)
        # 回传排序后的序列和排序后对应的原始标签
        _, lens_argsort_argsort = torch.sort(lens_argsort, 0)
        # 根据索引进行切割得到新的向量表达
        seq_ = torch.index_select(seq, 0, lens_argsort)
        # Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.
        seq_embd = self.dropout_embedding(seq_)
        packed = pack_padded_sequence(seq_embd, lens_sorted, batch_first=True)

        self.encoder.flatten_parameters()

        output, h = self.encoder(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output.contiguous()
        output = torch.index_select(output, 0, lens_argsort_argsort)  # B x m x 2l
        #last hidden state
        h = h.permute(1,0,2).contiguous().view(batch_size,1,-1)
        h = torch.index_select(h, 0, lens_argsort_argsort)
        # 把隐状态进行重新的选择，做法还是停车不哦的，以后要学会如何使用sort和index_sort
        output = self.dropout_encoder(output)
        h = self.dropout_encoder(h)
        return output, h
