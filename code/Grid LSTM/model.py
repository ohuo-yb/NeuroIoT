import torch
import torch.nn as nn
import torch.nn.functional as F

# hidden_size = 64
# num_class = 6
# batch_Size = 3
# input_Size = 32

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Multi_Cross_Attention_Layer(nn.Module):

    # 用来实现mask-attention layer
    def __init__(self, hidden_dim, is_bi_rnn=False):
        super(Multi_Cross_Attention_Layer, self).__init__()

        self.hidden_dim = hidden_dim
        self.is_bi_rnn = is_bi_rnn

        # 下面使用nn的Linear层来定义Q，K，V矩阵
        if is_bi_rnn:
            # 是双向的RNN
            self.Q_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
            self.K_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
            self.V_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        else:
            # 单向的RNN
            self.t_Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.t_K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.t_V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

            self.d_Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.d_K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.d_V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

            self.out_linear = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, inputs_t, inputs_d=None):
        if inputs_d is None:
            inputs = inputs_t.permute(1, 0, 2)
            # size = inputs.size()
            # 计算生成QKV矩阵
            Q_t = self.t_Q_linear(inputs)
            K_t = self.t_K_linear(inputs).permute(0, 2, 1)  # 先进行一次转置
            V_t = self.t_V_linear(inputs)

            Q_d = self.d_Q_linear(inputs)
            K_d = self.d_K_linear(inputs).permute(0, 2, 1)  # 先进行一次转置
            V_d = self.d_V_linear(inputs)
        else:
            inputs_t = inputs_t.permute(1, 0, 2)
            inputs_d = inputs_d.permute(1, 0, 2)
            # 计算生成QKV矩阵
            Q_t = self.t_Q_linear(inputs_t)
            K_t = self.t_K_linear(inputs_t).permute(0, 2, 1)  # 先进行一次转置
            V_t = self.t_V_linear(inputs_t)

            Q_d = self.d_Q_linear(inputs_d)
            K_d = self.d_K_linear(inputs_d).permute(0, 2, 1)  # 先进行一次转置
            V_d = self.d_V_linear(inputs_d)

        # 还要计算生成mask矩阵
        # max_len = lens  # 最大的句子长度，生成mask矩阵
        # sentence_lengths = torch.Tensor(lens)  # 代表每个句子的长度
        # mask = torch.arange(sentence_lengths.max().item())[None, :] < sentence_lengths[:, None]
        # mask = mask.unsqueeze(dim=1)  # [batch_size, 1, max_len]
        # mask = mask.expand(size[0], size[1], max_len)  # [batch_size, max_len, max_len]

        # print('\nmask is :', mask.size())

        # 下面生成用来填充的矩阵
        # padding_num = torch.ones_like(mask)
        # padding_num = -2 ** 31 * padding_num.float()

        # print('\npadding num is :', padding_num.size())

        # 下面开始计算啦
        alpha_t = torch.matmul(Q_d, K_t)
        alpha_d = torch.matmul(Q_t, K_d)

        # 下面开始mask
        # alpha_t = torch.where(mask, alpha_t, padding_num)
        # alpha_d = torch.where(mask, alpha_d, padding_num)

        # 下面开始softmax
        alpha_t = F.softmax(alpha_t, dim=2)
        alpha_d = F.softmax(alpha_d, dim=2)

        # print('\nalpha is :', alpha)

        out_t = torch.matmul(alpha_t, V_t)
        out_d = torch.matmul(alpha_d, V_d)
        outputs = torch.cat((out_t, out_d), 2).permute(1, 0, 2)
        outputs = self.out_linear(outputs)
        return outputs


class Cross_Attention(nn.Module):

    # 用来实现mask-attention layer
    def __init__(self, hidden_dim, is_bi_rnn=False):
        super(Multi_Cross_Attention_Layer, self).__init__()

        self.hidden_dim = hidden_dim
        self.is_bi_rnn = is_bi_rnn

        # 下面使用nn的Linear层来定义Q，K，V矩阵
        if is_bi_rnn:
            # 是双向的RNN
            self.Q_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
            self.K_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
            self.V_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        else:
            # 单向的RNN
            self.t_Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.t_K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.t_V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

            self.d_Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.d_K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.d_V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

            self.out_linear = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, inputs_t, inputs_d, lens):
        inputs_t = inputs_t.permute(1, 0, 2)
        inputs_d = inputs_d.permute(1, 0, 2)

        size = inputs_t.size()
        # 计算生成QKV矩阵
        Q_t = self.t_Q_linear(inputs_t)
        K_t = self.t_K_linear(inputs_t).permute(0, 2, 1)  # 先进行一次转置
        V_t = self.t_V_linear(inputs_t)

        Q_d = self.d_Q_linear(inputs_d)
        K_d = self.d_K_linear(inputs_d).permute(0, 2, 1)  # 先进行一次转置
        V_d = self.d_V_linear(inputs_d)

        # 还要计算生成mask矩阵
        # max_len = lens  # 最大的句子长度，生成mask矩阵
        # sentence_lengths = torch.Tensor(lens)  # 代表每个句子的长度
        # mask = torch.arange(sentence_lengths.max().item())[None, :] < sentence_lengths[:, None]
        # mask = mask.unsqueeze(dim=1)  # [batch_size, 1, max_len]
        # mask = mask.expand(size[0], size[1], max_len)  # [batch_size, max_len, max_len]

        # print('\nmask is :', mask.size())

        # 下面生成用来填充的矩阵
        # padding_num = torch.ones_like(mask)
        # padding_num = -2 ** 31 * padding_num.float()

        # print('\npadding num is :', padding_num.size())

        # 下面开始计算啦
        alpha_t = torch.matmul(Q_d, K_t)
        alpha_d = torch.matmul(Q_t, K_d)

        # 下面开始mask
        # alpha_t = torch.where(mask, alpha_t, padding_num)
        # alpha_d = torch.where(mask, alpha_d, padding_num)

        # 下面开始softmax
        alpha_t = F.softmax(alpha_t, dim=2)
        alpha_d = F.softmax(alpha_d, dim=2)

        # print('\nalpha is :', alpha)

        out_t = torch.matmul(alpha_t, V_t)
        out_d = torch.matmul(alpha_d, V_d)
        outputs = torch.cat((out_t, out_d), 2).permute(1, 0, 2)
        outputs = self.out_linear(outputs)
        return outputs


class Cross_Attention_Layer(nn.Module):
    # 用来实现mask-attention layer
    def __init__(self, hidden_dim, is_bi_rnn=False):
        super(Cross_Attention_Layer, self).__init__()

        self.hidden_dim = hidden_dim
        self.is_bi_rnn = is_bi_rnn

        # 下面使用nn的Linear层来定义Q，K，V矩阵
        if is_bi_rnn:
            # 是双向的RNN
            self.Q_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
            self.K_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
            self.V_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        else:
            # 单向的RNN
            self.t_Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.t_K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.t_V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

            self.d_Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.d_K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.d_V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, inputs_t, inputs_d, lens):

        size = inputs_t.size()
        # 计算生成QKV矩阵
        Q_t = self.t_Q_linear(inputs_t)
        K_t = self.t_K_linear(inputs_t).permute(1, 0)  # 先进行一次转置
        V_t = self.t_V_linear(inputs_t)

        Q_d = self.d_Q_linear(inputs_d)
        K_d = self.d_K_linear(inputs_d).permute(1, 0)  # 先进行一次转置
        V_d = self.d_V_linear(inputs_d)

        # 还要计算生成mask矩阵
        max_len = lens  # 最大的句子长度，生成mask矩阵
        sentence_lengths = torch.Tensor(lens)  # 代表每个句子的长度
        mask = torch.arange(sentence_lengths.max().item())[None, :] < sentence_lengths[:, None]
        mask = mask.unsqueeze(dim=1)  # [batch_size, 1, max_len]
        mask = mask.expand(size[0], max_len, max_len)  # [batch_size, max_len, max_len]

        # print('\nmask is :', mask.size())

        # 下面生成用来填充的矩阵
        padding_num = torch.ones_like(mask)
        padding_num = -2 ** 31 * padding_num.float()

        # print('\npadding num is :', padding_num.size())

        # 下面开始计算啦
        alpha_t = torch.matmul(Q_d, K_t)
        alpha_d = torch.matmul(Q_t, K_d)

        # 下面开始mask
        alpha_t = torch.where(mask, alpha_t, padding_num)
        alpha_d = torch.where(mask, alpha_d, padding_num)

        # 下面开始softmax
        alpha_t = F.softmax(alpha_t, dim=2)
        alpha_d = F.softmax(alpha_d, dim=2)

        # print('\nalpha is :', alpha)

        out_t = torch.matmul(alpha_t, V_t)
        out_d = torch.matmul(alpha_d, V_d)

        return out_t, out_d


class GridLSTM(nn.Module):
    def __init__(self, hidden_dim=None, batch_size=None, input_size=None):
        super(GridLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.input_size = input_size
        # self.d_lstm = nn.LSTM(input_size=11, hidden_size=hidden_dim, num_layers=2, dropout=0.2, batch_first=True)
        # self.t_lstm = nn.LSTM(input_size=11, hidden_size=hidden_dim, num_layers=2, dropout=0.2, batch_first=True)
        # self.lin = nn.Linear(in_features=hidden_dim * 2, out_features=num_class)
        # self.out = nn.Softmax(dim=1)
        self.t_lstm_1 = nn.LSTMCell(input_size=input_size, hidden_size=hidden_dim)
        self.d_lstm_1 = nn.LSTMCell(input_size=input_size, hidden_size=hidden_dim)
        self.t_Linear = nn.Linear(self.hidden_dim, self.input_size)
        self.d_Linear = nn.Linear(self.hidden_dim, self.input_size)
        # self.Cross_attention_t = Cross_Attention_Layer(hidden_dim=input_size)
        # self.Cross_attention_d = Cross_Attention_Layer(hidden_dim=input_size)
        self.Cross_attention = Multi_Cross_Attention_Layer(hidden_dim=input_size)
        # self.Cross_attention = Cross_Attention(hidden_dim=input_size)

    def forward(self, x, x_d=None):
        if x_d is None:
            x = self.Cross_attention(x)
        else:
            x = self.Cross_attention(x, x_d)

        # h_t = torch.zeros(self.batch_size, self.hidden_dim).to(device)
        # C_t = torch.zeros(self.batch_size, self.hidden_dim).to(device)
        # h_d = torch.zeros(self.batch_size, self.hidden_dim).to(device)
        # C_d = torch.zeros(self.batch_size, self.hidden_dim).to(device)
        size = x.size()
        h_t = torch.zeros(size[1], self.hidden_dim).to(device)
        C_t = torch.zeros(size[1], self.hidden_dim).to(device)
        h_d = torch.zeros(size[1], self.hidden_dim).to(device)
        C_d = torch.zeros(size[1], self.hidden_dim).to(device)

        outputs_t, outputs_d = [], []
        for xt in x:
            # 时间
            t2h_t, d2h_t = self.t_Linear(h_t), self.d_Linear(h_d)
            # att_t2h_t, att_d2ht = self.Cross_attention_t(t2h_t, d2h_t, self.input_size)
            h_t, C_t = self.t_lstm_1(xt + d2h_t, (h_t, C_t))
            # h_t, C_t = self.t_lstm_1(xt+att_d2ht, (h_t, C_t))
            outputs_t.append(h_t)
            # 深度
            t2h_d, d2h_d = self.t_Linear(h_t), self.d_Linear(h_d)
            # att_t2h_d, att_d2h_d = self.Cross_attention_d(t2h_d, d2h_d, self.input_size)
            h_d, C_d = self.d_lstm_1(t2h_d + d2h_d, (h_d, C_d))
            # h_d, C_d = self.d_lstm_1(att_t2h_d+att_d2h_d, (h_d, C_d))
            outputs_d.append(h_d)

        outputs_t = torch.stack(outputs_t, dim=0)
        outputs_d = torch.stack(outputs_d, dim=0)

        return outputs_t, outputs_d  # [time_step, batch_size, hidden_dim]


class GridLSTM_Net(nn.Module):
    def __init__(self, hidden_dim=None, batch_size=None, input_size=None):
        super(GridLSTM_Net, self).__init__()

        self.GridLSTM_Layer = GridLSTM(hidden_dim=hidden_dim, batch_size=batch_size, input_size=input_size)
        self.t_linear = nn.Linear(hidden_dim, input_size)
        self.d_linear = nn.Linear(hidden_dim, input_size)

    def forward(self, x):

        outputs_t, outputs_d = self.GridLSTM_Layer(x)

        outputs_t, outputs_d = self.t_linear(outputs_t), self.d_linear(outputs_d)

        return outputs_t, outputs_d


class Mul_GridLSTM_Net(nn.Module):
    def __init__(self, hidden_dim=None, batch_size=None, input_size=None):
        super(Mul_GridLSTM_Net, self).__init__()

        self.GridLSTM_Layer = GridLSTM(hidden_dim=hidden_dim, batch_size=batch_size, input_size=input_size)
        self.t_linear = nn.Linear(hidden_dim, input_size)
        self.d_linear = nn.Linear(hidden_dim, input_size)
        self.decoder = GridLSTM(hidden_dim=hidden_dim, batch_size=batch_size, input_size=hidden_dim)

    def forward(self, x):
        outputs_t, outputs_d = self.GridLSTM_Layer(x)

        # outputs_t_1, outputs_d_1 = self.decoder(outputs_t, outputs_d)
        outputs_t_1, outputs_d_1 = self.decoder(outputs_t)

        outputs_t, outputs_d = self.t_linear(outputs_t_1), self.d_linear(outputs_d_1)

        return outputs_t, outputs_d

# if __name__ == '__main__':
#     seed = 2021
#     torch.manual_seed(seed)
#     torch.random.manual_seed(seed)
#     gridLSTM = GridLSTM(hidden_dim=hidden_size, batch_size=batch_Size, input_size=input_Size)
#     # state = torch.load('./save_model/gls.pth', map_location=torch.device('cpu'))
#     # gridLSTM.load_state_dict(state)
#     inputs = torch.zeros((5, 3, 32))
#     outputs = gridLSTM(inputs)
#     print(outputs)
#     print(outputs.shape)
#     print("Total number of parameters in {} is {}  ".format('Grid LSTM', sum(x.numel() for x in gridLSTM.parameters())))
