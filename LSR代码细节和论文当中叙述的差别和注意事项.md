# LSR代码细节和论文当中叙述的差别和注意事项

1. 对于sen_ner处理的不同 

   ```
    for idx, vertex in enumerate(vertexSet, 1):
        for v in vertex:
           sen_pos[i][v['pos'][0]:v['pos'][1]] = idx
           ner_type_B = ner2id[v['type']]
           ner_type_I = ner_type_B+1
           sen_ner[i][v['pos'][0]] = ner_type_B
           sen_ner[i][v['pos'][0] + 1:v['pos'][1]] = ner_type_I
   ```

   和DocRed当中的处理不同，直接把entity_type转化过来就可以了   除了第一个词是对应的type 后面的词都是当前type的后一个type（这么做的原因是啥）

2. 送入LSTM当中的第一个步骤
   其实并没有把整个batch的内容送进去  **注意这个步骤完全没有并行化** 送入的不是一个batch的信息 利用index的方法 取出存储的信息 把一段话做成一个batch送入  **这样就不需要对句子的数量进行padding了**
   而且送入的不是LSTM而是GRU

3. LSTM出来后构造节点表达的时候没有先对mention当中的所有词汇取平均来构造mention的表达

   而是直接把每个word当作节点的表达 （从文中也没有看出来合并的意思） 只是在最后在entity把所有的mention word做平均

4. 在iterative refinement过后把所有的mention node的信息重新转化成为序列 和原始的序列相加（经过LSTM后的） 生成新的表达

   ```
   mention_node_position = mention_node_position.permute(0, 2, 1)
   output = torch.bmm(mention_node_position[:, :max_doc_len, :max_mention_num], output[:, :max_mention_num])
   context_output = torch.add(context_output, output) 
   # batch max_doc_len hidden_size
   ```

5. classifier 没有按照公式的做法 bilinear算子不能直接进行分类 而是映射到原始的隐空间当中   在最后的linear output才映射到最后的分类的n维空间当中

   ```
   self.bili = torch.nn.Bilinear(hidden_size+config.dis_size,  hidden_size+config.dis_size, hidden_size)
   self.self_att = SelfAttention(hidden_size)
   self.linear_output = nn.Linear(2 * hidden_size, config.relation_num)
   re_rep = self.dropout_rate(self.relu(self.bili(s_rep, t_rep)))
   # batch h_t_limit hidden_size
   re_rep = self.self_att(re_rep, re_rep, relation_mask)
   return self.linear_output(re_rep)
   ```

   attention计算方式  不是很清楚这个attention为什么要这么计算

   ```
   class SelfAttention(nn.Module):
       def __init__(self, input_size):
           super().__init__()
           self.input_linear = nn.Linear(input_size, 1, bias=False)
           self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))
   
       def forward(self, input, memory, mask):
           # att: WX/(sqrt(input size)) + XM
           # 利用输入不断地对memory信息进行attention更新
           input_dot = self.input_linear(input) #nan: cal the weight for the same word
           cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
           att = input_dot + cross_dot
           att = att - 1e30 * (1 - mask[:,None])
   
           weight_one = F.softmax(att, dim=-1)
           output_one = torch.bmm(weight_one, memory)
   
           return torch.cat([input, output_one], dim=-1)
   ```

6. 看的最要死要活的一段就是induction 

   估计这里他的代码是从哪里抄的，用到的内容其实并不多 其他的内容是否有意义呢

   这里算attention的时候，由于使用了bidirection 其实我们算的邻接矩阵的attention的时候 只用了一半的参量 

   ```
    if (self.bidirectional):
       input = input.view(batch_size, token_size, 2, dim_size // 2)
       sem_v = torch.cat((input[:, :, 0, :self.sem_dim_size // 2], input[:, :, 1, :self.sem_dim_size // 2]), 2)
       str_v = torch.cat((input[:, :, 0, self.sem_dim_size // 2:], input[:, :, 1, self.sem_dim_size // 2:]), 2)
   ```

   在做attention的时候 我们只用了sem_v 来计算  感觉好像并不是很合理  **其实我们只利用了一半的参量**

   计算output的环节我们这里就不多说了，其实也没有用到

   他最后输出的邻接矩阵 上面多拼接了第一个维度  但是其实我们用不上 注意输出时候的改进


   还有一个很迷的操作

   ```
   LLinv_diag = torch.diagonal(LLinv, dim1=-2, dim2=-1).unsqueeze(2)
   tmp1 = (A_ij.transpose(1, 2) * LLinv_diag).transpose(1, 2)
   ```

   上面的tmp1对应的是PijL^-1_ij （sorry这里公式不太会打 就是A矩阵构建的上半部分） 为什么转置两次 直接构建不香吗

7. desenGCN上的一些细节

   GCN的每一层的维度都比较小 感觉其实有点像multi head的机制  上来会根据维度的多少进行切割

   ```
   self.head_dim = self.mem_dim // self.layers
   ```

   **GCN不是一个sum操作 增加了自环和归一化（按照行归一化）**

   残差层是把最后的隐状态和输入相加  不是每一层都有这样的操作

   ```
   gcn_outputs = torch.cat(output_list, dim=2)
   gcn_outputs = gcn_outputs + gcn_inputs
   out = self.linear_output(gcn_outputs)
   ```

   **在输出之前还增加了一个线性变化**

8. 之前提到的N=2 指的是整体的n为2