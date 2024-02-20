import numpy as np
import torch
import torch.nn.functional as F
import pdb

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.gelu = torch.nn.GELU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.dropout1(self.gelu(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs

class MSSG(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(MSSG, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.sim_sum = torch.zeros(args.num_blocks)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.attn_dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        self.apply(self.init_weights)

        self.indices = torch.nn.Parameter(
            torch.arange(item_num+1, dtype=torch.long), requires_grad=False
        )

    def log2feats(self, log_seqs, user_ids, isEval):
        users = self.user_emb(torch.LongTensor(user_ids).to(self.dev))
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        if not isEval:
            Q = seqs
            users = users.unsqueeze(1).repeat(1,tl,1)
        else:
            Q = seqs[:,-1:,:]
            attention_mask = attention_mask[-1:,:]
            users = users.unsqueeze(1)
        seqs = torch.transpose(seqs, 0, 1)
        atts = []
        hiddens = []

        for i in range(len(self.attention_layers)):
            Q = torch.transpose(Q, 0, 1)
            QQ = self.attention_layernorms[i](Q)
            mha_outputs, att = self.attention_layers[i](QQ, seqs, seqs, 
                                            attn_mask=attention_mask)
            if isEval:
                atts.append(att)

            Q = Q + mha_outputs
            Q = torch.transpose(Q, 0, 1)

            QQ = self.forward_layernorms[i](Q)
            Q = self.forward_layers[i](QQ)

            if isEval:
                hiddens.append(Q)

        if isEval:
            atts = torch.stack(atts)
            hiddens = torch.stack(hiddens)

        log_feats = self.last_layernorm(Q) + users

        return log_feats, atts, self.last_layernorm(Q), users, hiddens

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats, _, _, _, _ = self.log2feats(log_seqs, user_ids, isEval=False)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs):
        log_feats, atts, st, lt, hiddens = self.log2feats(log_seqs, user_ids, isEval=True)

        final_feat = log_feats[:, -1, :]

        item_embs = self.item_emb(self.indices)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits, atts, st, lt, hiddens

    def init_weights(self, module):
        #we use the same initialization method for RAM and SASRec
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Parameter)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
