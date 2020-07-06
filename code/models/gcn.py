import torch.nn as nn
import torch
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, mem_dim, layers, dropout, self_loop = False):
        super(GraphConvLayer, self).__init__()
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = dropout

        # linear transformation
        self.linear_output = nn.Linear(self.mem_dim, self.mem_dim)

        # dcgcn block
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.linear_output = self.linear_output.cuda()
        self.self_loop = self_loop


    def forward(self, adj, gcn_inputs):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs
        cache_list = [outputs]
        # 之前生成的output的内容
        output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            # 把输入转化到head dim
            if self.self_loop:
                AxW = AxW  + self.weight_list[l](outputs)  # self loop
            #     在GNN中增加自环和残差层的性质其实是类似的
            else:
                AxW = AxW

            AxW = AxW / denom
            # 在实现的过程当中其实是进行了归一化的，不是sun而是按照列进行平均
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs
        # 为了控制模型的表达能力 每一个head dim不大 最后才利用残差结合 中间增加了模型的深度
        # 注意resnet的做法不是在每层都要做，而是在开始和最后做，中间叠加的层次都不使用残差网络

        out = self.linear_output(gcn_outputs)

        return out

class MultiGraphConvLayer(nn.Module):
    """ A GCN module operated on multihead attention """

    def __init__(self,mem_dim, layers, heads, dropout):
        super(MultiGraphConvLayer, self).__init__()
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.heads = heads
        self.gcn_drop = dropout

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim * self.heads, self.mem_dim)
        self.weight_list = nn.ModuleList()

        for i in range(self.heads):
            for j in range(self.layers):
                self.weight_list.append(nn.Linear(self.mem_dim + self.head_dim * j, self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()

    def forward(self, adj_list, gcn_inputs):

        multi_head_list = []
        for i in range(self.heads):
            adj = adj_list[:,i]
            denom = adj.sum(2).unsqueeze(2) + 1
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []
            for l in range(self.layers):
                index = i * self.layers + l
                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)
                AxW = AxW + self.weight_list[index](outputs)  # self loop
                AxW = AxW / denom
                gAxW = F.relu(AxW)
                cache_list.append(gAxW)
                outputs = torch.cat(cache_list, dim=2)
                output_list.append(self.gcn_drop(gAxW))

            gcn_ouputs = torch.cat(output_list, dim=2)
            gcn_ouputs = gcn_ouputs + gcn_inputs

            multi_head_list.append(gcn_ouputs)

        final_output = torch.cat(multi_head_list, dim=2)
        out = self.Linear(final_output)

        return out
