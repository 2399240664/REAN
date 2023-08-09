import torch
import torch.nn as nn
import math
from transformers import AutoModel
# from utils.BiGRU import GRU, BiGRU
import numpy as np
import torch.nn.functional as F
import copy
# from semgcn import GCNAbsaModel
class REAN(nn.Module):
    def __init__(self, args, ent2id, rel2id):
        super().__init__()
        self.num_relations = len(rel2id)
        self.num_ents = len(ent2id)
        self.num_steps = 3
        # self.num_ways = args.num_ways

        self.bert_encoder = AutoModel.from_pretrained(args.bert_name, return_dict=True)
        self.bert1_encoder = AutoModel.from_pretrained(args.bert_name, return_dict=True)
        for param in self.bert1_encoder.parameters():
            param.requires_grad =False
        self.dim_hidden = self.bert_encoder.config.hidden_size

        self.Fatten = MultiFisonAttention(self.dim_hidden)
        # self.Conv = nn.Conv1d(1, 1, 3)
        # self.MLP = PoswiseFeedForwardNet(self.num_relations, 2 * self.num_relations)
        # self.fc = []
        self.step_encoders = []
        for i in range(self.num_steps):
            m = nn.Sequential(
                nn.Linear(self.dim_hidden, self.dim_hidden),
                nn.Tanh()
            )
            self.step_encoders.append(m)
            self.add_module('step_encoders_{}'.format(i), m)
        self.step_r_encoders = []
        for i in range(self.num_steps):
            m = nn.Sequential(
                nn.Linear(self.dim_hidden, self.dim_hidden),
                nn.Tanh()
            )
            self.step_r_encoders.append(m)
            self.add_module('step_r_encoders_{}'.format(i), m)
        self.fc = []
        for i in range(self.num_steps):
            m = nn.Linear(766, self.dim_hidden)
            self.fc.append(m)
            self.add_module('fc_{}'.format(i), m)
        # self.gru = []
        # for i in range(self.num_steps):
        #     m = GRU(self.dim_hidden,self.dim_hidden,2)
        #     self.gru.append(m)
        #     self.add_module('gru_{}'.format(i), m)
        self.rel_classifier = nn.Linear(self.dim_hidden, self.num_relations)
        self.in_feature = nn.Linear(self.dim_hidden, self.dim_hidden)
        # self.rel_classifier1 = nn.Linear(self.num_relations, self.num_relations)
        # self.LIN = nn.Linear(383, dim_hidden)
        self.Conv = nn.Conv1d(1, 1, 3)
        # self.pool = nn.MaxPool1d(2)
        # self.rel_classifier1 = nn.Linear(2*dim_hidden, 1)
        # self.rel_classifier1 = nn.Linear(num_relations, num_relations)
        self.hop_selector = nn.Linear(self.dim_hidden, self.num_steps)
        self.MLP = PoswiseFeedForwardNet(self.num_relations, 2 * self.num_relations)
        self.attn = MultiHeadAttention(3, self.dim_hidden)
        self.gcn_common = GCN(args, self.dim_hidden, 2)
        self.linearc = nn.Linear(766, self.dim_hidden)
        # self.gcn = GCNAbsaModel(self.dim_hidden,2)
        # self.MLP1 = PoswiseFeedForwardNet(self.num_relations, 2 * self.num_relations)
        # self.step_r_encoders = []
        # self.step_encoders = {}
        # self.hop_selectors = {}
        # self.rel_classifiers = {}
        #
        # for j in range(self.num_steps):
        #     m = nn.Sequential(
        #         nn.Linear(dim_hidden, dim_hidden),
        #         nn.Tanh()
        #     )
        #     name = 'way_{}_step_{}'.format(i, j)
        #     self.step_encoders[name] = m
        #     self.add_module(name, m)
        #
        # for j in range(self.num_steps):
        #     m = nn.Sequential(
        #         nn.Linear(dim_hidden, dim_hidden),
        #         nn.Tanh()
        #     )
        #     self.step_r_encoders.append(m)
        #     self.add_module('step_r_encoders_{}'.format(j), m)
        #
        # for j in range(self.num_steps):
        #     m = nn.Linear(766, dim_hidden)
        #     self.fc.append(m)
        #     self.add_module('fc_{}'.format(j), m)
        #
        #
        #     # self.rel_classifier = nn.Linear(dim_hidden, num_relations)
        #     m = nn.Linear(dim_hidden, self.num_steps)
        #     self.hop_selectors['way_{}'.format(i)] = m
        #     self.add_module('hop-way_{}'.format(i), m)
        #
        #     m = nn.Linear(dim_hidden, self.num_relations)
        #     self.rel_classifiers['way_{}'.format(i)] = m
        #     self.add_module('rel-way_{}'.format(i), m)
        
    def get_attn_pad_mask(self ,seq_q, seq_k):
        '''
        seq_q: [batch_size, seq_len]
        seq_k: [batch_size, seq_len]
        seq_len could be src_len or it could be tgt_len
        seq_len in seq_q and seq_len in seq_k maybe not equal
        '''
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # eq(zero) is PAD token
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
        return pad_attn_mask.expand(batch_size, len_q, len_k)

    def inputs_to_att_adj(self, input, score_mask):
        attn_tensor = self.attn(input, input, score_mask)  # [batch_size, head_num, seq_len, seq_len]
        attn_tensor = torch.sum(attn_tensor, dim=1)
        # attn_tensor = select(attn_tensor, 2) * attn_tensor
        return attn_tensor

    def follow(self, e, r,q):
        y = torch.sparse.mm(self.Mins, q.t())
        val = torch.sparse.mm(self.Mrel, r.t()) * y
        x = torch.sparse.mm(self.Msubj, e.t()) * val
        return torch.sparse.mm(self.Mobj.t(), x).t() # [bsz, Esize]

    def build_matrix(self,triples,idx):
        Tsize = len(triples)
        self.Msubj = torch.sparse.FloatTensor(
            torch.stack((idx, triples[:, 0])), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, self.num_ents])).cuda()
        self.Mobj = torch.sparse.FloatTensor(
            torch.stack((idx, triples[:, 2])), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, self.num_ents])).cuda()
        self.Mrel = torch.sparse.FloatTensor(
            torch.stack((idx, triples[:, 1])), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, self.num_relations])).cuda()
        self.Mins = torch.sparse.FloatTensor(
            torch.stack((idx, triples[:, 1])), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, self.num_relations])).cuda()

    def forward(self, heads, questions, answers=None, triples=None, entity_range=None,relation_embedd = None):
        q = self.bert_encoder(**questions)
        q_embeddings, q_word_h = q.pooler_output, q.last_hidden_state # (bsz, dim_h), (bsz, len, dim_h)
        bsz = len(heads)
        # device = heads.device
        r = self.bert1_encoder(**relation_embedd)
        r_em_h = r.last_hidden_state

        # e_score = []
        # last_h = torch.zeros_like(q_embeddings)

        last_e = heads
        word_attns = []
        rel_probs = []
        ent_probs = []
        for t in range(self.num_steps):
            cq_t = self.step_encoders[t](
                    q_embeddings) # consider history
                # [bsz, dim_h]
            q_logits = torch.sum(cq_t.unsqueeze(1) * q_word_h, dim=2) # [bsz, max_q]
            q_dist = torch.softmax(q_logits, 1) # [bsz, max_q]
            q_dist = q_dist * questions['attention_mask'].float()
            q_dist = q_dist / (torch.sum(q_dist, dim=1, keepdim=True) + 1e-6) # [bsz, max_q]
            word_attns.append(q_dist)
            ctx_h = (q_dist.unsqueeze(1) @ q_word_h).squeeze(1) # [bsz, dim_h]
                # ctx_h = ctx_h + cq_t
                # last_h = ctx_h
            # h_e = self.gcn(q_word_h,questions['input_ids'],questions['attention_mask'])
            score_mask = torch.matmul(q_word_h, q_word_h.transpose(-2, -1))
            score_mask = (score_mask == 0)
            score_mask = score_mask.unsqueeze(1).repeat(1, 3, 1, 1).cuda()
            att_adj = self.inputs_to_att_adj(q_word_h, score_mask)
            h_cse = self.gcn_common(att_adj, q_word_h, score_mask, 'semantic')
            n = q_word_h.size(1)
            Convolution = nn.Conv1d(n, n, 3).cuda()
            conve = F.relu(Convolution(h_cse))
            deep_q = self.linearc(conve)
            asp_wn = questions['attention_mask'].sum(dim=1).unsqueeze(-1)  # aspect words num
            mask = questions['attention_mask'].unsqueeze(-1).repeat(1, 1, 768)
            h_e = (deep_q * mask).sum(dim=1) / asp_wn
            insturt = h_e * ctx_h
            insturt = self.in_feature(insturt)

            cq_r = self.step_r_encoders[t](r_em_h)
            attn_mask = self.get_attn_pad_mask(relation_embedd['input_ids'], relation_embedd['input_ids'])
            r_out, attn = self.Fatten(cq_r, cq_r, cq_r, attn_mask, relation_embedd['attention_mask'].float())
            r_out = r_out.unsqueeze(1)
            r_out = F.relu(self.Conv(r_out)).squeeze(1)
            r_out = self.fc[t](r_out)

            gate = torch.matmul(insturt, r_out.t())

            rel2 = self.MLP(gate)
            # rel2 = torch.sigmoid(rel2)
            rel_logit = self.rel_classifier(ctx_h) # [bsz, num_relations]
                # rel_dist = torch.softmax(rel_logit, 1) # bad
            # rel_logit = torch.sigmoid(rel_logit)
            rel_logit = rel_logit + rel2
            # rel_logit = self.rel_classifier1(rel_logit)
            # rel_logit = self.MLP1(rel_logit)
            rel_dist = torch.sigmoid(rel_logit)
            rel_probs.append(rel_dist)
            # insturt = self.in_feature(insturt)
            # in_infor = rel_dist * insturt
                # new_e = []
                # T = {k: torch.cat([q[k] for q in triples], dim=0) for k in range(len(triples))}
                # batchids=[]
            # T = torch.cat(triples, dim=0)
            # batchfact =torch.LongTensor([i for i in range(len(T))])
            # self.build_matrix(T, batchfact)
            # last_e = self.follow(last_e, rel_dist, insturt)
            new_e = []
            for b in range(bsz):
                sub, rel, obj = triples[b][:, 0], triples[b][:, 1], triples[b][:, 2]
                sub_p = last_e[b:b + 1, sub]  # [1, #tri]
                rel_p = rel_dist[b:b + 1, rel]  # [1, #tri]
                # in_val = insturt[b:b + 1, rel]
                obj_p = sub_p * rel_p
                new_e.append(
                    torch.index_add(torch.zeros(1, self.num_ents).cuda(), 1, obj, obj_p))
            last_e = torch.cat(new_e, dim=0)


                # last_e = torch.cat(new_e, dim=0)

                # reshape >1 scores to 1 in a differentiable way
            m = last_e.gt(1).float()
            z = (m * last_e + (1-m)).detach()
            last_e = last_e / z

            ent_probs.append(last_e)

        hop_res = torch.stack(ent_probs, dim=1) # [bsz, num_hop, num_ent]
        hop_logit = self.hop_selector(q_embeddings)
        hop_attn = torch.softmax(hop_logit, dim=1).unsqueeze(2) # [bsz, num_hop, 1]
        last_e = torch.sum(hop_res * hop_attn, dim=1) # [bsz, num_ent]

        #     e_score.append(last_e)
        #
        # e_score = torch.prod(torch.stack(e_score), dim=0)

        if not self.training:
            return {
                'e_score': last_e,
                'word_attns': word_attns,
                'rel_probs': rel_probs,
                'ent_probs': ent_probs,
                # 'hop_attn': hop_attn.squeeze(2)
            }
        else:
            weight = answers * 9 + 1
            loss = torch.sum(entity_range * weight * torch.pow(last_e - answers, 2)) / torch.sum(entity_range * weight)

            return loss

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.fc = nn.Linear(18, 1, bias=False).cuda()
    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, len_q, d_k]
        K: [batch_size, len_k, d_k]
        V: [batch_size, en_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(768)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        attn = self.fc(attn.transpose(-1, -2))
        context = torch.matmul(attn.transpose(-1, -2), V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiFisonAttention(nn.Module):
    def __init__(self,d_model):
        super(MultiFisonAttention, self).__init__()
        self.d_model=d_model
        self.W_Q = nn.Linear(d_model, d_model , bias=False)
        self.W_K = nn.Linear(d_model, d_model , bias=False)
        self.W_V = nn.Linear(d_model, d_model , bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask,word_mask):
        '''
        input_Q: [batch_size, n_relation,len_q, d_model]
        input_K: [batch_size, n_relation,len_k, d_model]
        input_V: [batch_size, n_relation,len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        asp_wn=word_mask.sum(dim=1).unsqueeze(-1)
        mask=word_mask.unsqueeze(-1).repeat(1,1,768)
        residual =(residual*mask).sum(dim=1)/asp_wn
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q)  # Q: [batch_size, n_relation, len_q, d_k]
        K = self.W_K(input_K)  # K: [batch_size, n_relation, len_k, d_k]
        V = self.W_V(input_V)  # V: [batch_size, n_relation, len_v(=len_k), d_v]

        # attn_mask = attn_mask.unsqueeze(1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.squeeze(-2)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model,d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        # d_model=1912
        # d_ff=3824
        self.d_model=d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]

class GCN(nn.Module):
	def __init__(self, args, mem_dim, num_layers):
		super(GCN, self).__init__()
		self.args = args
		self.layers = num_layers
		self.mem_dim = mem_dim
		self.in_dim = 768
		self.linearc = nn.Linear(766, 768)

		self.fc = nn.Sequential(
			nn.Linear(768, 768, bias=False),
			nn.ReLU(),
			nn.Linear(768, 768, bias=False)
		)


		# drop out
		# self.in_drop = nn.Dropout(args.input_dropout)
		self.gcn_drop = nn.Dropout(0.4)

		# gcn layer
		self.W = nn.ModuleList()
		self.attn = nn.ModuleList()
		for layer in range(self.layers):
			input_dim = self.in_dim + layer * self.mem_dim
			self.W.append(nn.Linear(input_dim, self.mem_dim))

			# attention adj layer
			self.attn.append(MultiHeadAttention(3, input_dim)) if layer != 0 else None

	def GCN_layer(self, adj, gcn_inputs, denom, l):
		Ax = adj.bmm(gcn_inputs)
		AxW = self.W[l](Ax)
		AxW = AxW / denom
		gAxW = F.relu(AxW) + self.W[l](gcn_inputs)
		# if dataset is not laptops else gcn_inputs = self.gcn_drop(gAxW)
		gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
		return gcn_inputs

	def forward(self, adj, inputs, score_mask, type):
		# gcn
		denom = adj.sum(2).unsqueeze(2) + 1  # norm adj
		n = inputs.size(1)

		Convolution = nn.Conv1d(n, n, 3).cuda()

		out = self.GCN_layer(adj, inputs, denom, 0)

		conve = F.relu(Convolution(out))
		out = self.linearc(conve)
		# 第二层之后gcn输入的adj是根据前一层隐藏层输出求得的
		for i in range(1, self.layers):
			# concat the last layer's out with input_feature as the current input
			inputs = torch.cat((inputs, out), dim=-1)
			if type == 'semantic':
				# att_adj
				adj = self.attn[i - 1](inputs, inputs, score_mask)  # [batch_size, head_num, seq_len, dim]
				probability = F.softmax(adj.sum(dim=(-2, -1)), dim=0)
				max_idx = torch.argmax(probability, dim=1)
				adj = torch.stack([adj[i][max_idx[i]] for i in range(len(max_idx))], dim=0)
				adj = select(adj, 2) * adj
				denom = adj.sum(2).unsqueeze(2) + 1  # norm adj
			out = self.GCN_layer(adj, inputs, denom, i)
			out = self.fc(out)
		return out


def clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
	# d_model:hidden_dim，h:head_num
	def __init__(self, head_num, hidden_dim, dropout=0.1):
		super(MultiHeadAttention, self).__init__()
		# assert hidden_dim % head_num == 0

		self.d_k = int(hidden_dim // head_num)
		self.head_num = head_num
		self.linears = clones(nn.Linear(hidden_dim, hidden_dim), 2)
		self.dropout = nn.Dropout(p=dropout)

	def attention(self, query, key, score_mask, dropout=None):
		d_k = query.size(-1)
		scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
		if score_mask is not None:
			scores = scores.masked_fill(score_mask, -1e9)

		b = ~score_mask[:, :, :, 0:1]
		p_attn = F.softmax(scores, dim=-1) * b.float()
		if dropout is not None:
			p_attn = dropout(p_attn)
		return p_attn

	def forward(self, query, key, score_mask):
		nbatches = query.size(0)
		query, key = [l(x).view(nbatches, -1, self.head_num, self.d_k).transpose(1, 2)
					  for l, x in zip(self.linears, (query, key))]
		attn = self.attention(query, key, score_mask, dropout=self.dropout)

		return attn


def select(matrix, top_num):
	batch = matrix.size(0)
	len = matrix.size(1)
	matrix = matrix.reshape(batch, -1)
	maxk, _ = torch.topk(matrix, top_num, dim=1)

	for i in range(batch):
		matrix[i] = (matrix[i] >= maxk[i][-1])
	matrix = matrix.reshape(batch, len, len)
	matrix = matrix + matrix.transpose(-2, -1)

	# selfloop
	for i in range(batch):
		matrix[i].fill_diagonal_(1)

	return matrix
