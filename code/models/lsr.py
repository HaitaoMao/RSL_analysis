import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from models.encoder import Encoder
from models.attention import SelfAttention
from models.reasoner import DynamicReasoner
from models.reasoner import StructInduction

class LSR(nn.Module):
    def __init__(self, config):
        super(LSR, self).__init__()
        self.config = config

        self.finetune_emb = config.finetune_emb
        # 这里其实有提供finutune的接口

        self.word_emb = nn.Embedding(config.data_word_vec.shape[0], config.data_word_vec.shape[1])
        self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))
        if not self.finetune_emb:
            self.word_emb.weight.requires_grad = False

        self.ner_emb = nn.Embedding(13, config.entity_type_size, padding_idx=0)

        self.coref_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)

        hidden_size = config.rnn_hidden
        input_size = config.data_word_vec.shape[1] + config.coref_size + config.entity_type_size #+ char_hidden

        self.linear_re = nn.Linear(hidden_size * 2,  hidden_size)

        self.linear_sent = nn.Linear(hidden_size * 2,  hidden_size)

        self.bili = torch.nn.Bilinear(hidden_size, hidden_size, hidden_size)

        self.self_att = SelfAttention(hidden_size)
        # 这个东西在哪里用过呢

        self.bili = torch.nn.Bilinear(hidden_size+config.dis_size,  hidden_size+config.dis_size, hidden_size)
        self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)

        self.linear_output = nn.Linear(2 * hidden_size, config.relation_num)

        self.relu = nn.ReLU()

        self.dropout_rate = nn.Dropout(config.dropout_rate)

        self.rnn_sent = Encoder(input_size, hidden_size, config.dropout_emb, config.dropout_rate)
        self.hidden_size = hidden_size

        self.use_struct_att = config.use_struct_att
        if  self.use_struct_att == True:
            self.structInduction = StructInduction(hidden_size // 2, hidden_size, True)

        self.dropout_gcn = nn.Dropout(config.dropout_gcn)
        self.reasoner_layer_first = config.reasoner_layer_first
        self.reasoner_layer_second = config.reasoner_layer_second
        # 两层reasoner  这应该是后面的2表达的真正含义
        self.use_reasoning_block = config.use_reasoning_block
        if self.use_reasoning_block:
            self.reasoner = nn.ModuleList()
            self.reasoner.append(DynamicReasoner(hidden_size, self.reasoner_layer_first, self.dropout_gcn))
            self.reasoner.append(DynamicReasoner(hidden_size, self.reasoner_layer_second, self.dropout_gcn))

    def doc_encoder(self, input_sent, context_seg):
        # 这里对于i=0的特殊处理仍然不是很懂
        """
        :param sent: sent emb
        :param context_seg: segmentation mask for sentences in a document
        :return:
        """
        batch_size = context_seg.shape[0]
        docs_emb = [] # sentence embedding
        docs_len = []
        sents_emb = []

        for batch_no in range(batch_size):
            sent_list = []
            sent_lens = []
            sent_index = ((context_seg[batch_no] == 1).nonzero()).squeeze(-1).tolist() # array of start point for sentences in a document
            pre_index = 0
            for i, index in enumerate(sent_index):
                # 这里的疑惑是不是第一个位置出现了一个初始的标识符号 需要在这里进行特殊的处理
                if i != 0:
                    if i == 1:
                        sent_list.append(input_sent[batch_no][pre_index:index+1])
                        sent_lens.append(index - pre_index + 1)
                    else:
                        sent_list.append(input_sent[batch_no][pre_index+1:index+1])
                        sent_lens.append(index - pre_index)
                pre_index = index

            sents = pad_sequence(sent_list).permute(1,0,2)
            sent_lens_t = torch.LongTensor(sent_lens).cuda()
            docs_len.append(sent_lens)
            sents_output, sent_emb = self.rnn_sent(sents, sent_lens_t) # sentence embeddings for a document.

            doc_emb = None
            # 把所有的隐状态拼到一起 变成一个document级别的隐状态
            for i, (sen_len, emb) in enumerate(zip(sent_lens, sents_output)):
                if i == 0:
                    doc_emb = emb[:sen_len]
                else:
                    doc_emb = torch.cat([doc_emb, emb[:sen_len]], dim = 0)

            docs_emb.append(doc_emb)
            sents_emb.append(sent_emb.squeeze(1))

        docs_emb = pad_sequence(docs_emb).permute(1,0,2) # B * # sentence * Dimention
        sents_emb = pad_sequence(sents_emb).permute(1,0,2)
        # pad sequence的具体使用方法，参照https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e

        return docs_emb, sents_emb


    def forward(self, context_idxs, pos, context_ner, h_mapping, t_mapping,
                relation_mask, dis_h_2_t, dis_t_2_h, context_seg, mention_node_position, entity_position,
                mention_node_sent_num, all_node_num, entity_num_list, sdp_pos, sdp_num_list):
        """
        :param context_idxs: Token IDs
        :param pos: coref pos IDs
        :param context_ner: NER tag IDs
        :param h_mapping: Head
        :param t_mapping: Tail
        :param relation_mask: There are multiple relations for each instance so we need a mask in a batch
        :param dis_h_2_t: distance for head
        :param dis_t_2_h: distance for tail
        :param context_seg: mask for different sentences in a document
        :param mention_node_position: Mention node position
        :param entity_position: Entity node position
        :param mention_node_sent_num: number of mention nodes in each sentences of a document
        :param all_node_num: the number of nodes  (mention, entity, MDP) in a document
        :param entity_num_list: the number of entity nodes in each document
        :param sdp_pos: MDP node position
        :param sdp_num_list: the number of MDP node in each document
        :return:
        """

        '''===========STEP1: Encode the document============='''
        sent_emb = torch.cat([self.word_emb(context_idxs), self.coref_embed(pos), self.ner_emb(context_ner)], dim=-1)
        # 所有的节点也包括最基本的三个步骤
        docs_rep, sents_rep = self.doc_encoder(sent_emb, context_seg)
        # 得到两个等级的描述方式

        max_doc_len = docs_rep.shape[1]
        context_output = self.dropout_rate(torch.relu(self.linear_re(docs_rep)))

        '''===========STEP2: Extract all node reps of a document graph============='''
        '''extract Mention node representations'''
        mention_num_list = torch.sum(mention_node_sent_num, dim=1).long().tolist()
        max_mention_num = max(mention_num_list)
        mentions_rep = torch.bmm(mention_node_position[:, :max_mention_num, :max_doc_len], context_output) # mentions rep
        '''extract MDP(meta dependency paths) node representations'''
        sdp_num_list = sdp_num_list.long().tolist()
        max_sdp_num = max(sdp_num_list)
        sdp_rep = torch.bmm(sdp_pos[:,:max_sdp_num, :max_doc_len], context_output)
        '''extract Entity node representations'''
        entity_rep = torch.bmm(entity_position[:,:,:max_doc_len], context_output)
        # 这里并没有做平均！！！！！！！！！！！而是选择的项加的做法
        '''concatenate all nodes of an instance'''
        gcn_inputs = []
        all_node_num_batch = []
        for batch_no, (m_n, e_n, s_n) in enumerate(zip(mention_num_list, entity_num_list.long().tolist(), sdp_num_list)):
            m_rep = mentions_rep[batch_no][:m_n]
            e_rep = entity_rep[batch_no][:e_n]
            s_rep = sdp_rep[batch_no][:s_n]
            gcn_inputs.append(torch.cat((m_rep, e_rep, s_rep),dim=0))
            # 构造出对应的u矩阵
            node_num = m_n + e_n + s_n
            all_node_num_batch.append(node_num)

        gcn_inputs = pad_sequence(gcn_inputs).permute(1, 0, 2)
        output = gcn_inputs

        '''===========STEP3: Induce the Latent Structure============='''
        if self.use_reasoning_block:
            # 啊这 模型没有过先用reasoner的说法啊  额 整体全部叫做reasoner 我直接裂开
            for i in range(len(self.reasoner)):
                output = self.reasoner[i](output)

        elif self.use_struct_att:
            gcn_inputs, _ = self.structInduction(gcn_inputs)
            # 这里再做struct induction得到的结果有啥意义呢
            max_all_node_num = torch.max(all_node_num).item()
            assert (gcn_inputs.shape[1] == max_all_node_num)

        mention_node_position = mention_node_position.permute(0, 2, 1)
        output = torch.bmm(mention_node_position[:, :max_doc_len, :max_mention_num], output[:, :max_mention_num])
        context_output = torch.add(context_output, output)
        # 这里增加了一个残差上的表达  把原始的内容加在最后的表示上面 最后生成的还是一个序列

        start_re_output = torch.matmul(h_mapping[:, :, :max_doc_len], context_output) # aggregation
        end_re_output = torch.matmul(t_mapping[:, :, :max_doc_len], context_output) # aggregation
        # 先生成隐状态最后做映射的方法其实也是我没想到的  我以为最后生成的就是对应的节点表达方式

        s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
        t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)

        re_rep = self.dropout_rate(self.relu(self.bili(s_rep, t_rep)))

        re_rep = self.self_att(re_rep, re_rep, relation_mask)
        # re_rep为了增强type的表达 还增加了一层attention
        return self.linear_output(re_rep)

