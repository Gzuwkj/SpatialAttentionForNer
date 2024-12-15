from torch import nn
from transformers import AutoModel
from fastNLP import seq_len_to_mask
from transformers.models.roberta.modeling_roberta import RobertaModel
from torch_scatter import scatter_max, scatter_sum, scatter
import torch
import torch.nn.functional as F

from .Transformer2D_VAN import T2d, T2dConfig
from .multi_head_biaffine import MultiHeadBiaffine
from .CA import CA, SA
from .QKV import KV
from .Transformer2D_VAN import T2dExternalAttention


class TableRnn(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.hidden_size = out_dim
        self.dropout = nn.Dropout(0.3)
        self.out_dim = out_dim
        self.rnn = nn.LSTM(
            input_size=in_dim,
            hidden_size=out_dim,
            batch_first=True,
            num_layers=1,
            bidirectional=False,
        )

    def forward(self, input_embeds, hx):
        batch_size, seq_len, hidden_size = input_embeds.size()

        # B x L x H
        input_embeds_expand = input_embeds.unsqueeze(1).repeat(1, 2, 1, 1)
        # B x 2 x L x H
        size = (batch_size, seq_len, seq_len, hidden_size)
        stride = list(input_embeds_expand.stride())
        stride[1] = stride[2]
        input_embeds_repeat = input_embeds_expand.as_strided(size, stride).contiguous().view(-1, seq_len, hidden_size)
        input_embeds_repeat = self.dropout(input_embeds_repeat)
        out_embeds_repeat, _ = self.rnn(input_embeds_repeat)
        out_embeds_repeat = out_embeds_repeat.view(batch_size, seq_len, seq_len, self.out_dim)

        triu = torch.triu(torch.ones((batch_size, seq_len, seq_len), dtype=torch.bool))
        ret = torch.zeros_like(out_embeds_repeat)

        ret[triu] = out_embeds_repeat[torch.flip(triu, [2])]
        # ret1 = ret + torch.triu(ret.permute(0, 2, 1, 3), -1)

        # eye = torch.eye(seq_len, dtype=torch.bool).expand(batch_size, -1, -1)
        # ret[eye] = input_embeds.view(-1, self.hidden_size)
        return ret


class B1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.rnn = TableRnn(in_dim, out_dim)

    def forward(self, input_embeds, hx):
        return self.rnn(input_embeds, hx)


class B2(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, size_embed_dim):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=mid_dim // 2,
            batch_first=True,
            num_layers=1,
            bidirectional=True,
        )

        self.head_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(mid_dim, mid_dim),
            nn.LeakyReLU(),
        )
        self.tail_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(mid_dim, mid_dim),
            nn.LeakyReLU(),
        )
        n_pos = 30
        self.size_embedding = torch.nn.Embedding(n_pos, size_embed_dim)
        _span_size_ids = torch.arange(512) - torch.arange(512).unsqueeze(-1)
        _span_size_ids.masked_fill_(_span_size_ids < -n_pos / 2, -n_pos / 2)
        _span_size_ids = _span_size_ids.masked_fill(_span_size_ids >= n_pos / 2, n_pos / 2 - 1) + n_pos / 2
        self.register_buffer('span_size_ids', _span_size_ids.long())

        self.W = torch.nn.Parameter(torch.empty(out_dim, mid_dim * 2 + size_embed_dim + 2))
        torch.nn.init.xavier_normal_(self.W.data)

        self.dropout = nn.Dropout(0.4)

    def forward(self, input_embeds):
        input_embeds = self.lstm(input_embeds)
        head_state = self.head_mlp(input_embeds)
        tail_state = self.tail_mlp(input_embeds)
        head_state = torch.cat([head_state, torch.ones_like(head_state[..., :1])], dim=-1)
        tail_state = torch.cat([tail_state, torch.ones_like(tail_state[..., :1])], dim=-1)
        affined_cat = torch.cat([self.dropout(head_state).unsqueeze(2).expand(-1, -1, tail_state.size(1), -1),
                                 self.dropout(tail_state).unsqueeze(1).expand(-1, head_state.size(1), -1, -1)], dim=-1)
        size_embedded = self.size_embedding(self.span_size_ids[:input_embeds.size(1), :input_embeds.size(1)])
        affined_cat = torch.cat([affined_cat,
                                 self.dropout(size_embedded).unsqueeze(0).expand(input_embeds.size(0), -1, -1, -1)], dim=-1)

        return torch.einsum('bmnh,kh->bkmn', affined_cat, self.W)



class CNNNer(nn.Module):
    def __init__(self, model_name, num_ner_tag, cnn_dim=200, biaffine_size=200,
                 size_embed_dim=0, logit_drop=0, kernel_size=3, n_head=4, cnn_depth=3):
        super(CNNNer, self).__init__()
        self.pretrain_model = RobertaModel.from_pretrained(model_name, return_dict=True)
        hidden_size = self.pretrain_model.config.hidden_size

        self.b1 = B1(hidden_size, cnn_dim)
        self.b2 = B2(hidden_size, biaffine_size, cnn_dim, size_embed_dim)

        config = T2dConfig('./T2dConfig.json')
        config.num_hidden_layers = 3
        config.kernel_size = 3
        config.hidden_size = cnn_dim
        self.t2d = T2d(config)

        self.down_fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(cnn_dim, cnn_dim),
            nn.GELU(),
            nn.Linear(cnn_dim, num_ner_tag)
        )
        self.logit_drop = logit_drop

    def forward(self, input_ids, bpe_len, indexes, matrix):
        attention_mask = seq_len_to_mask(bpe_len)  # bsz x length x length
        outputs = self.pretrain_model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs['last_hidden_state']
        hx = outputs['pooler_output']
        state = scatter_max(last_hidden_states, index=indexes, dim=1)[0][:, 1:]  # bsz x word_len x hidden_size

        scores1 = self.b1(state, hx)
        # scores2 = self.b2(state)
        scores = scores1.permute(0, 3, 1, 2) # bsz x dim x L x L

        lengths, _ = indexes.max(dim=-1)
        mask = seq_len_to_mask(lengths)  # bsz x length x length
        mask = mask[:, None] * mask.unsqueeze(-1)
        pad_mask = mask[:, None].eq(0)
        u_scores = scores.masked_fill(pad_mask, 0)
        u_scores = self.t2d(
            u_scores,
            mask,
            output_attentions=True
        )
        u_scores = u_scores[0]

        scores = scores + u_scores

        scores = self.down_fc(scores.permute(0, 2, 3, 1))

        assert scores.size(-1) == matrix.size(-1)

        if self.training:
            flat_scores = scores.reshape(-1)
            flat_matrix = torch.triu(matrix).reshape(-1)
            mask = flat_matrix.ne(-100).float().view(input_ids.size(0), -1)
            flat_loss = F.binary_cross_entropy_with_logits(flat_scores, flat_matrix.float(), reduction='none')
            loss = ((flat_loss.view(input_ids.size(0), -1)*mask).sum(dim=-1)).mean()
            return {'loss': loss}
        scores[torch.triu(torch.ones_like(matrix)) == 1] = scores.T[torch.triu(torch.ones_like(matrix)) == 1]
        return {'scores': scores}
