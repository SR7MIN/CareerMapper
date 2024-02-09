import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:  # 对数间距取样
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)  # 得到 [2^0, 2^1, ... ,2^(L-1)]
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)  # 得到 [2^0,...,2^(L-1)] 的等差数列

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:  # 这个数组里只有cos和sin
                embed_fns.append(lambda x, p_fn_=p_fn, freq_=freq: p_fn_(x * freq_))
                out_dim += d  # 每使用子编码公式一次就要把输出维度加上原始维度，因为每个待编码的位置维度是自定义input_dims

        self.embed_fns = embed_fns  # 相当于是一个编码公式列表[sin(2^0*x),cos(2^0*x),...]
        self.out_dim = out_dim

    def embed(self, inputs):
        # 对各个输入进行编码，给定一个输入，使用编码列表中的公式分别对他编码
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims):
    embed_kwargs = {
        'input_dims': input_dims,  # 输入给编码器的数据的维度
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = (Embedder(**embed_kwargs))
    # embed 现在相当于一个编码器
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Major_Rate_Model(nn.Module):
    def __init__(self, D, W, input_ch, output_ch):
        super(Major_Rate_Model, self).__init__()
        self.D = D  # D=8 netdepth网络的深度 ,也就是网络的层数 layers in network
        self.W = W  # W=256 netwidth网络宽度 , 也就每一层的神经元的个数 channels per layer
        self.input_ch = input_ch  # 输入维度
        self.output_ch = output_ch  # 输出维度

        Linears_list = [nn.Linear(input_ch, W)]
        for i in range(D - 1):
            Linears_list.append(nn.Linear(W, W))
        self.pts_linears = nn.ModuleList(Linears_list)

        # 在第9层添加24维度的方向数据和10维的距离数据，并且输出128维的信息
        self.out_linear = nn.Linear(W, output_ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        assert len(h.shape) == 1
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
        output = self.out_linear(h)
        output = self.sigmoid(output)
        return torch.softmax(output, 0)


class Model_block(nn.Module):
    def __init__(self, D, W, input_ch, add_chs):
        super(Model_block, self).__init__()
        self.D = D  # D=6 netdepth网络的深度 ,也就是网络的层数 layers in network
        self.W = W  # W=128 netwidth网络宽度 , 也就每一层的神经元的个数 channels per layer
        self.input_ch = input_ch  # 输入维度
        self.add_chs = add_chs  # 其他附加输入
        Linears_list = [nn.Linear(input_ch, W)]
        for i in range(D - 1):
            Linears_list.append(nn.Linear(W, W))
        Linears_list.append(nn.Linear(W, W // 2))
        self.base_linears = nn.ModuleList(Linears_list)

        Linears_list = []
        for ch in self.add_chs:
            Linears_list.append(nn.Linear(W // 2 + ch, W // 2))
        self.feature_linears = nn.ModuleList(Linears_list)

        self.out_linear = nn.Linear(W // 2, 1)


    def forward(self, gdb, adds):
        h = gdb
        for i, l in enumerate(self.base_linears):
            h = self.base_linears[i](h)
            h = F.relu(h)
        for i, l in enumerate(self.feature_linears):
            h = torch.cat([h, adds[i]])
            h = self.feature_linears[i](h)
            h = F.relu(h)
        output = self.out_linear(h)

        return output


class Subjob_predictive_model(nn.Module):
    def __init__(self, D, W, input_ch, add_chs, output_ch):
        super(Subjob_predictive_model, self).__init__()
        self.D = D  # D=6 netdepth网络的深度 ,也就是网络的层数 layers in network
        self.W = W  # W=128 netwidth网络宽度 , 也就每一层的神经元的个数 channels per layer
        self.input_ch = input_ch  # 输入维度
        self.add_chs = add_chs  # 其他附加输入
        self.output_ch = output_ch  # 输出维度
        model_list = [Model_block(D, W, input_ch, add_chs) for _ in range(output_ch)]
        self.model_list = nn.ModuleList(model_list)
        self.linear = nn.Linear(output_ch, output_ch)
        self.in_linear = nn.Linear(input_ch, input_ch * output_ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, gdb, adds):
        gdb = self.in_linear(gdb)
        output = torch.cat([model(gdb[i*self.input_ch:(i+1)*self.input_ch], adds) for i, model in enumerate(self.model_list)])
        output = F.relu(output)
        output = self.linear(output)
        # output = self.sigmoid(output)
        return output


class RegLSTM(nn.Module):
    def __init__(self, inp_dim, mid_dim, mid_layers):
        super(RegLSTM, self).__init__()

        self.in_linear = nn.Linear(inp_dim, inp_dim)
        self.model_list = nn.ModuleList([nn.LSTM(1, mid_dim, mid_layers) for _ in range(inp_dim)])
        self.model_linear_list = nn.ModuleList([nn.Linear(mid_dim, 1) for _ in range(inp_dim)])
        self.out_linear = nn.Linear(inp_dim, inp_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        seq_len, batch_size, hid_dim = x.shape
        h = self.in_linear(x.view(-1, hid_dim))
        h = F.relu(h).view(seq_len, batch_size, hid_dim)
        h = torch.stack([self.model_linear_list[i](self.model_list[i](h[..., i])[0]) for i in range(hid_dim)], -1)
        h = F.relu(h.view(-1, hid_dim))
        h = self.out_linear(h)
        h = h.view(seq_len, batch_size, -1)
        return self.sigmod(h)


class TextCnnModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=20, output_dim=47, out_channels=66, filter_sizes=(2, 3, 4)):
        """
        TextCNN 模型
        :param vocab_size: 语料大小
        :param embedding_dim: 每个词的词向量维度
        :param output_dim: 输出的维度，由于咱们是二分类问题，所以输出两个值，后面用交叉熵优化
        :param out_channels: 卷积核的数量
        :param filter_sizes: 卷积核尺寸
        :param dropout: 随机失活概率
        """
        super().__init__()
        conv_list = []
        for conv_size in filter_sizes:
            # 由于是文本的embedding，所以in_channels=1,常见的如图像的RGB，则in_channels=3
            conv_list.append(nn.Conv2d(1, out_channels, (conv_size, embedding_dim)))
            # nn.Conv2d 的使用：
            # (batch_size, 特征图个数, 特征图长, 特征图宽) -> 经过nn.conv2d(特征图个数,输出的特征图个数,卷积核的长,卷积核的宽) ->
            # (batch_size, 输出的特征图个数, a-b+1, ∂-ß+1)
        self.conv_model = nn.ModuleList(conv_list)
        # 最后的FC
        self.linear = nn.Linear(out_channels * len(filter_sizes), output_dim)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # 步骤2：使用多种shape的卷积核进行卷积操作
        conv_result_list = []
        for conv in self.conv_model:
            # 过一层卷积，然后把最后一个维度(值=1)剔除掉
            conv_out = F.relu(conv(x)).squeeze(3)  # shape=(batch_size, 66 out_channels, 19)
            # 步骤3：对每一层做最大池化，然后拼接起来
            max_conv_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # shape = (batch_size, 66 out_channels)
            conv_result_list.append(max_conv_out)
        # 步骤4：拼接起来
        concat_out = torch.cat(conv_result_list, dim=1)  # 这里要指定第二个维度（dim=0对应第一个维度）
        # 步骤5：
        model_out = self.linear(concat_out)
        return model_out

    def get_embedding(self, token_list: list):
        return self.embedding(torch.Tensor(token_list).long())




def MSE(x, y):
    return torch.mean((x - y) ** 2)
