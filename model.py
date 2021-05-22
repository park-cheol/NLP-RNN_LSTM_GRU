import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp) # ninp: input크기
        # nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None)
        # num_embeddings: embeddings dic의 크기 집합크기, embedding_dim: embedding vecotr의 크기(열 크기)
        # num_embeddings은 당연히 더 커야함
        # 문장 -> 단어에 부여된 고유 정수값 -> 임베딩층 통과 -> 밀집벡터
        # 행이 그 단어집합의 크기이므로 모든 단어들이 고유한 임베딩벡터를 가짐
        # token : 문장을 단어 또는 형태소 단위로 토큰화 / 토큰은 문장, 단어, 형태소가 될 수 있음
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        """LSTM"""
        # torch.nn.LSTM(input_size, hidden_size, num_layers
        # Inputs: input, (h_0, c_0)
        # input (seq_len, batch, input_size)
        # h_0 (num_layer * num_direction, batch, hidden_size)
        # c_0 (num_layers * num_directions, batch, hidden_size)
        # num_directions: 1, 2 (bool 인자임) 2경우 미래의 값이 현재의 값에 영향을 줄 수있음
        # seq_len = 단어개수(cell 개수), hidden_size = 출력층으로 보내는 값의 크기(output_dim)동일, 128,256식

        # https://sanghyu.tistory.com/52
        # input_size: input의 feature dimension넣어줌 (not time step)
        # hidden_size: 내부에서 어떤 feature dimension으로 바꿀지
        # num_layers: 얼마나 lstm 쌓을지
        # batch_first 인자도있음 보통 cnn에서 많이쓰니까 헷갈리지 않게 True하는 것도 나쁘지 않아보임
        # Input(batch, time_step, feature dimension)순
        # output(output, hidden, cell) tuple형식
        # output: (batch, time_step or seq_length, hidden size)
        # lstm과 gru rnn형식은 똑같음 cell만 없는것 빼고

        # 선택적으로 tie weights 사용
        # "Using the Output Embedding to Improve Language Models"
        # todo https://arxiv.org/abs/1608.05859
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # todo https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type =rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)
        # zero_ 하고 uniform 하는 이유

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input)) # todo input shape 확인
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output) # todo drop 2번?
        decoded = self.decoder(output) # nn.Linear(nhid, ntoken)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden
    # dim=1 : ntoken
    # softmax의 vanishing Gradients 문제 해결 또한 * / => + - 인 log-sum-exp trick

    def init_hidden(self, bsz): # bsz: batch_size
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

        # torch.new_zeros(size, dtype=None, devcie=None, requires_grad=False) - > Tensor
        # parameters c, h : (num of layers, batchs ize, number of hidden size)









