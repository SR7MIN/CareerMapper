import torch
import torch.nn as nn
import torch.nn.functional as F

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

