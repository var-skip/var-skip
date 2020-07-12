"""Autoregressive transformer.

Modeled after https://github.com/openai/gpt-2/blob/master/src/model.py.

Positional embeddings are not used, since in our context, we don't care about
sequence element order.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

#### Masking
#
# Assume we have (x0, x1, x2) and order=natural in all cases below.
#
# Scheme 0 (base) --> OK
#         Input  [SOS, x0, x1]
#         Hidden [p(x0), p(x1|x0), p(x2|x0,x1)]
#      (We use 1 block so hidden == output).
#
#      mask(3) is used:
#         mask (row=destination):
#         tensor([[1, 0, 0],
#                 [1, 1, 0],
#                 [1, 1, 1]], dtype=torch.uint8)
# This scheme only works for natural ordering.
#
# Scheme 1 (default) --> OK
#    # TODO: document
#
##### Some failed schemes
#
# Scheme F1 --> FAILS
# [column 0 depends on self; because Query(p(x0)) is produced from x0]
#    Input  [SOS, x0, x1, x2]
#    Hidden [SOS, p(x0), p(x1|x0), p(x2|x0,x1)]
#     mask (row=destination):
#     tensor([[1., 0., 0., 0.],
#             [1., 0., 0., 0.],
#             [1., 1., 0., 0.],
#             [1., 1., 1., 0.]], dtype=torch.float64)
#
# Scheme F2 --> FAILS
# [column 0 depends on self; because Query(p(x0)) is produced from x0]
#
#    Input  [x0, x1, x2, EOS]
#    Hidden [p(x0), p(x1|x0), p(x2|x0,x1), EOS]
#    mask (row=destination):
#    tensor([[1., 0., 0., 1.],
#           [1., 1., 0., 1.],
#           [1., 1., 1., 1.],
#           [0., 0., 0., 1.]], dtype=torch.float64)

DEFAULT_MASK_SCHEME = 0  # 0: original, only works for natural order
DEFAULT_MASK_SCHEME = 1
DEBUG_PRINT = True
DEBUG_PRINT = False

## Notes on unk_embeddings & pos_embeddings
#
#  pos_embeddings index: *position* of input sequence
#    - aganostic to what column index it is (or even if it is SOS)
#
#  unk_embeddings index: *natural_idx* of the column it's supposed to mask
#    - thus, SOS does not have an unk_embeddings
#
#  How they interact: potentially dropout first, then potentially add pos emb.


def pprint(*args):
    do_print = DEBUG_PRINT
    if do_print:
        print(*args)


def mask(n):
    # tensor([[1, 0, 0],
    #         [1, 1, 0],
    #         [1, 1, 1]], dtype=torch.uint8)
    # for ns = nd = 3.
    ns = n
    nd = n
    i = torch.arange(nd)[:, None]
    j = torch.arange(ns)
    m = i >= j - ns + nd
    m.requires_grad = False
    return m


# def order_respecting_mask(ncols, ordering, fill_diagonal=False):
#     """Construct appropriate mask for attention.

#     Assuming o=(2,0,1):
#      - set inputs = [ SOS=0, x0,         x1,         x2 ]
#      - so outputs = [ 0,   h(x0|x2), h(x1|x0,x2), h(x2) ]

#     There's an added column at the left end.

#     Desired mask (row=destination):
#         [[1, 0, 0, 0],
#          [1, 0, 0, 1],
#          [1, 1, 0, 1],
#          [1, 0, 0, 0]]

#     Mask after the first attention == above + "each col sees directly below":
#      - inputs =  [ 0, h(x0|x2), h(x1|x0,x2), h(x2) ]
#      - outputs = [ 0, h(x0|x2), h(x1|x0,x2), h(x2) ]

#         [[1, 0, 0, 0],
#          [1, 1, 0, 1],
#          [1, 1, 1, 1],
#          [1, 0, 0, 1]]
#       (just fill diagnoal with 1's)
#     """
#     # natural idx -> position
#     nat_to_pos = [None] * ncols
#     for natural_idx in range(ncols):
#         nat_to_pos[ordering[natural_idx]] = natural_idx

#     mask = np.zeros((ncols + 1, ncols + 1))
#     mask[:, 0] = 1  # First column is SOS -- everyone can see.
#     for pos_src in range(ncols):
#         src_nat_idx = ordering[pos_src]
#         for pos_dst in range(pos_src + 1, ncols):
#             # Variable at pos_dst should see pos_src.
#             dst_nat_idx = ordering[pos_dst]
#             mask[dst_nat_idx + 1, src_nat_idx + 1] = 1
#     if fill_diagonal:
#         np.fill_diagonal(mask, 1)

#     mask = torch.as_tensor(mask)
#     # mask = torch.as_tensor(mask, dtype=torch.uint8)
#     mask.requires_grad = False
#     return mask

# def order_respecting_mask2(ncols, ordering, fill_diagonal=False):
#     """Construct appropriate mask for attention.

#     Assuming o=(2,0,1):
#      - set inputs = [  x0,         x1,         x2 , EOS]
#      - so outputs = [ h(x0|x2), h(x1|x0,x2), h(x2), EOS]

#     There's an added column at the right end.

#     Desired mask (row=destination):
#         tensor([[0., 0., 1., 1.],
#                 [1., 0., 1., 1.],
#                 [0., 0., 0., 1.],
#                 [0., 0., 0., 1.]], dtype=torch.float64)

#     For natural order (0,1,2), the desired mask is:
#         tensor([[0., 0., 0., 1.],
#                 [1., 0., 0., 1.],
#                 [1., 1., 0., 1.],
#                 [0., 0., 0., 1.]], dtype=torch.float64)

#     Mask after the first attention == above + "each col sees directly below".
#     """
#     # natural idx -> position
#     nat_to_pos = [None] * ncols
#     for natural_idx in range(ncols):
#         nat_to_pos[ordering[natural_idx]] = natural_idx

#     mask = np.zeros((ncols + 1, ncols + 1))
#     mask[:, -1] = 1  # Last column is SOS -- everyone can see.
#     for pos_src in range(ncols):
#         src_nat_idx = ordering[pos_src]
#         for pos_dst in range(pos_src + 1, ncols):
#             # Variable at pos_dst should see pos_src.
#             dst_nat_idx = ordering[pos_dst]
#             mask[dst_nat_idx, src_nat_idx] = 1
#     if fill_diagonal:
#         np.fill_diagonal(mask, 1)

#     mask = torch.as_tensor(mask)
#     # mask = torch.as_tensor(mask, dtype=torch.uint8)
#     mask.requires_grad = False
#     return mask


def order_respecting_mask3(ncols, ordering, input_layer=True):
    """Construct appropriate mask for attention.

    Assuming o=(2,0,1):
     - set inputs = [ SOS=0,          x0,    x1,     x2 ]
     - so outputs = [ h(x0|x2), h(x1|x0,x2), h(x2), EOS ]

    No one connects to EOS.  SOS connects to everyone.

    Desired mask (row=destination):
        [[1, 0, 0, 1],
         [1, 1, 0, 1],
         [1, 0, 0, 0],
         [0, 0, 0, 0]]

    Mask after the first attention + see self (diagonal)
    Basically == shift above to the left 1 column, then fill diagonal
     - inputs  = [ h(x0|x2), h(x1|x0,x2), h(x2), EOS ]
     - outputs = [ h(x0|x2), h(x1|x0,x2), h(x2), EOS ]
        [[1, 0, 1, 0],
         [1, 1, 1, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 0]]
    """
    mask = np.zeros((ncols + 1, ncols + 1))

    if input_layer:
        mask[:, 0] = 1  # First column is SOS -- everyone can see.
        mask[-1, :] = 0  # No one connects to EOS
        for pos_src in range(ncols):
            src_nat_idx = ordering[pos_src]
            for pos_dst in range(pos_src + 1, ncols):
                # Variable at pos_dst should see pos_src.
                dst_nat_idx = ordering[pos_dst]
                mask[dst_nat_idx, src_nat_idx + 1] = 1
    else:
        for pos_src in range(ncols):
            src_nat_idx = ordering[pos_src]
            for pos_dst in range(pos_src, ncols):
                dst_nat_idx = ordering[pos_dst]
                mask[dst_nat_idx, src_nat_idx] = 1

    mask = torch.as_tensor(mask, dtype=torch.float32)
    mask.requires_grad = False
    return mask


class LayerNorm(nn.Module):
    """Norm to 0-mean 1-std , then do a learned diagonal affine transform."""

    def __init__(self, features, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.shift = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        s = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) * torch.rsqrt(s + self.eps)
        return self.scale * x + self.shift


class Conv1d(nn.Module):
    """Linear (with bias).  Weights ~ N(std), bias ~ 0."""

    def __init__(self, d_in, d_out, w_init_std=0.02):
        super(Conv1d, self).__init__()

        self.w = nn.Parameter(torch.zeros(d_in, d_out))
        self.b = nn.Parameter(torch.zeros(d_out))
        nn.init.normal_(self.w, std=w_init_std)
        nn.init.zeros_(self.b)
        self.d_in = d_in
        self.d_out = d_out

    def forward(self, x):
        *start, d_in = x.size()
        out = torch.matmul(x.view(-1, d_in), self.w) + self.b
        return out.view(start + [self.d_out])


class MultiHeadSelfAttention(nn.Module):
    """Params:

      d_model: last dim of input and output of this module.
      num_heads: number of parallel heads.

    Internally, queries, keys, and values are all produced from the input
    (hence "self"), and all of them are (d_model/num_heads)-dimensional.
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_state = d_model // num_heads

        self.qkv_linear = Conv1d(d_model, self.d_state * 3 * num_heads)
        self.linear = Conv1d(num_heads * self.d_state, d_model)

        self.attn_mask = None  # Will be set by caller.

    def _split_heads(self, x):
        # Each input has shape [bs, num cols, d_state * num_heads].
        *start, m = x.size()
        x = x.view(start + [self.num_heads, m // self.num_heads])
        return x.permute(0, 2, 1, 3)

    def _do_attention(self, query, key, value, mask):
        """Accepts Q,K,V each shaped [bs, num heads, num cols, d_state].

        Returns transformed [bs, num_heads, num cols, d_state].
        """
        d_k = query.size()[-1]
        scores = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(d_k)
        pprint('qk scores', scores)
        mask = mask.to(scores.dtype)
        scores = scores * mask - (1 - mask) * 1e10
        pprint('masked scores', scores)
        attn_weights = F.softmax(scores, dim=-1)
        pprint('attn weights', attn_weights)
        pprint('values', value)

        out = torch.matmul(attn_weights, value)
        pprint('read', out)
        return out

    def forward(self, x, query_input=None):
        """x: [bs, num cols, d_model].  Output has the same shape."""
        assert x.dim() == 3, x.size()
        bs, ncols, _ = x.size()
        pprint('In MultiHead', x.size())

        # [bs, num cols, d_state * 3 * num_heads]
        qkv = self.qkv_linear(x)
        # [bs, num heads, num cols, d_state] each
        qs, ks, vs = map(self._split_heads, torch.chunk(qkv, 3, dim=-1))

        if query_input is not None:
            # TODO: obviously can avoid redundant calc...
            qkv = self.qkv_linear(query_input)
            qs, _, _ = map(self._split_heads, torch.chunk(qkv, 3, dim=-1))

        # [bs, num heads, num cols, d_state]
        x = self._do_attention(qs, ks, vs, mask=self.attn_mask.to(x.device))

        # [bs, num cols, num heads, d_state]
        x = x.transpose(1, 2)
        # Concat all heads' outputs: [bs, num cols, num heads * d_state]
        x = x.contiguous().view(bs, ncols, -1)
        # Then do a transform: [bs, num cols, d_model].
        x = self.linear(x)
        return x


class GeLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Block(nn.Module):
    """Params:

      d_model: last dim of input and output of this module.
      d_ff: the hidden dim inside the 2-layer MLP.
      num_heads: number of parallel heads.
    """

    def __init__(self,
                 d_model,
                 d_ff,
                 num_heads,
                 activation='relu',
                 do_residual=False):
        super(Block, self).__init__()

        self.mlp = nn.Sequential(
            Conv1d(d_model, d_ff),
            nn.ReLU() if activation == 'relu' else GeLU(),
            Conv1d(d_ff, d_model),
        )
        self.norm1 = LayerNorm(features=d_model)
        self.norm2 = LayerNorm(features=d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.do_residual = do_residual

    def set_attn_mask(self, mask):
        self.attn.attn_mask = mask

    def forward(self, x, query_input=None):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, query_input=query_input)
        if self.do_residual:
            x += residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        if self.do_residual:
            x += residual

        return x


class Transformer(nn.Module):
    """An autoregressive Transformer (decoder only)."""

    def __init__(self,
                 num_blocks,
                 d_model,
                 d_ff,
                 num_heads,
                 nin,
                 input_bins,
                 use_positional_embs=False,
                 activation='relu',
                 dropout=False,
                 fixed_ordering=None,
                 draw_dropout_per_col=False,
                 seed=None,
                 first_query_shared=False,
                 prefix_dropout=False,
                 mask_scheme=DEFAULT_MASK_SCHEME):
        super().__init__()

        print('self.mask_scheme', mask_scheme)
        self.mask_scheme = mask_scheme

        # Common attributes below.
        self.nin = nin
        self.input_bins = input_bins
        encoded_bins = [d_model] * nin
        pprint('input_bins', self.input_bins)
        pprint('encoded', encoded_bins)
        self.logit_indices = np.cumsum(encoded_bins)
        self.nout = self.logit_indices[-1]
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.embed_size = d_model
        self.emb_dim = d_model
        self.use_positional_embs = use_positional_embs
        assert activation in ['relu', 'gelu']
        self.activation = activation
        self.fixed_ordering = fixed_ordering
        if fixed_ordering is None:
            natural = np.arange(nin)
            if seed is None:
                self.fixed_ordering = natural
            else:
                self.fixed_ordering = np.random.RandomState(seed).permutation(
                    natural)
        print('ordering', self.fixed_ordering)
        self.draw_dropout_per_col = draw_dropout_per_col
        self.first_query_shared = first_query_shared
        print('first_query_shared', first_query_shared)
        if first_query_shared:
            assert self.mask_scheme == 1, self.mask_scheme
        self.prefix_dropout = prefix_dropout

        # Build.
        self.blocks = nn.Sequential(*[
            Block(d_model,
                  d_ff,
                  num_heads,
                  activation,
                  do_residual=(self.mask_scheme == 0 or i > 0))
            for i in range(num_blocks)
        ])
        # Set masks.

        orig_mask = None
        if self.mask_scheme == 0:
            orig_mask = mask(nin)
        elif self.mask_scheme == 1:
            init_attn_mask = order_respecting_mask3(nin, self.fixed_ordering)
            attn_mask = order_respecting_mask3(nin,
                                               self.fixed_ordering,
                                               input_layer=False)
        else:
            assert False, self.mask_scheme

        if orig_mask is not None:
            print('using orig mask\n', orig_mask)
            for b in self.blocks:
                b.set_attn_mask(orig_mask)
        else:
            print('init_attn_mask\n', init_attn_mask)
            print('after 1st layer attn_mask\n', attn_mask)
            self.blocks[0].set_attn_mask(init_attn_mask)
            for b in self.blocks[1:]:
                b.set_attn_mask(attn_mask)

        self.norm = LayerNorm(d_model)

        # TODO: scale by sqrt(d_model)?
        self.embeddings = nn.ModuleList()
        if len(set(self.input_bins)) == 1:
            print("Detected tied columns")
            tied_embed = nn.Embedding(self.input_bins[0], d_model)
            for i in range(nin):
                self.embeddings.append(tied_embed)
        else:
            for i in range(nin):
                self.embeddings.append(nn.Embedding(self.input_bins[i], d_model))
        for e in self.embeddings:
            nn.init.normal_(e.weight, std=0.02)

        if use_positional_embs:
            if self.mask_scheme == 1:
                self.pos_embeddings = nn.Embedding(self.nin + 1, d_model)
            else:
                self.pos_embeddings = nn.Embedding(self.nin, d_model)
            nn.init.normal_(self.pos_embeddings.weight, std=0.01)

        print('Transformer ctor, dropout {}'.format(dropout))
        self.dropout = dropout
        if dropout or prefix_dropout:
            self.unk_embeddings = nn.ParameterList()
            for i, dist_size in enumerate(self.input_bins):
                self.unk_embeddings.append(nn.Parameter(torch.zeros(d_model)))

        # Interface required by ProgressiveSampling.
        self.input_bins_encoded_cumsum = np.cumsum(encoded_bins)
        self.orderings = [self.fixed_ordering]

    def name(self):
        n = 'transformer'
        n += '-blocks' + str(self.num_blocks)
        n += '-model' + str(self.d_model)
        n += '-ff' + str(self.d_ff)
        n += '-heads' + str(self.num_heads)
        if self.use_positional_embs:
            n += '-posEmb'
        n += '-' + self.activation
        if self.dropout:
            n += '-dropout'
        if self.mask_scheme == 1:
            n += '-scheme1'
        if self.first_query_shared:
            n += '-1stqshared'
        return n

    def EncodeInput(self, x, natural_col=None, out=None, return_pos_embs=False, skip_prefix=[]):
        """Right shift by one token.

        Suppose we want to model x=(x0,x1,x2).
        Set model inputs = [ SOS=0, x0, x1 ]
            (SOS = start of sequence)
        outputs =          [ p(x0); p(x1|x0); p(x2|x0,x1) ].
            (because output i depends on inputs <= i).

        If self.fixed_ordering is supplied and non-natural,
        we set inputs = [ SOS=0, x_o(0), x_o(1) ]
        so    outputs = [ p(x_o(0)), p(x_o(1) | x_o(0)), p(x_o(2) | x_o(0..1)) ]

        This (1) requires when calculating the loss, seq [x_o(0), ..., x_o(2)]
        is passed, (2) assumes we don't change the diagonal attention mask.

        Alternatively (assuming o=(2,0,1)):
          - change diagonal mask to respect ordering o
          - set inputs = [ SOS=0, x_o(0)=x2, x_o(1)=x0 ]
          - so outputs = [ p(x0|x2), p(x1|x0,x2), p(x2) ]
          - doesn't require feeding targets under order o
        """
        if natural_col is not None:
            assert not return_pos_embs
            return self.EncodeInputInference(x, natural_col, out)

        if x.dtype != torch.long:
            x = x.long()
        bs = x.size()[0]

        if self.mask_scheme == 0:
            # SOS = start of sequence symbol, just zeros.
            y_embed = [torch.zeros(bs, self.embed_size, device=x.device)]
            for nat_idx in range(self.nin - 1):
                y_embed.append(self.embeddings[nat_idx](x[:, nat_idx]))
        elif self.mask_scheme == 1:
            y_embed = [torch.zeros(bs, self.embed_size, device=x.device)]
            for nat_idx in range(self.nin):
                y_embed.append(self.embeddings[nat_idx](x[:, nat_idx]))
        else:
            assert False, self.mask_scheme

        # [batch size, num cols (+ 1), d_model].  +1 or not depends on scheme.
        inp = torch.stack(y_embed, 1)

        # pprint('after stacking embs', inp, 'shape', inp.shape)
        # pprint(self.embeddings[0].weight, self.embeddings[1].weight,
        #        self.embeddings[2].weight)

        inp_seq_len = inp.shape[1]

        if self.dropout:
            if self.draw_dropout_per_col:
                vecs = []
                for _ in range(inp_seq_len):
                    vecs.append(
                        torch.dropout(
                            torch.ones(bs, 1, 1, device=x.device),  # ncol=1
                            p=1.0 -
                            np.random.randint(1, self.nin + 1) * 1. / self.nin,
                            train=self.training))
                dropout_vec = torch.cat(vecs, dim=1)
            else:
                dropout_vec = torch.dropout(
                    torch.ones(bs, inp_seq_len, 1, device=x.device),
                    p=1.0 - np.random.randint(1, self.nin + 1) * 1. / self.nin,
                    train=self.training)
            # During training, non-dropped 1's are scaled by 1/(1-p), so we
            # clamp back to 1.  Shaped [bs, num cols, 1].
            batch_mask = torch.clamp(dropout_vec, 0, 1)
            # Shaped [1, num cols, d_model].
            dropped_repr = self.get_dropped_repr()
            inp = batch_mask * inp + (1. - batch_mask) * dropped_repr

        if self.prefix_dropout and self.training:
            prefix_mask = (
                torch.arange(1, inp_seq_len + 1).repeat(bs).reshape([bs, -1])
                > (torch.rand(bs) * inp_seq_len).unsqueeze(1)
            ).int().unsqueeze(2).to(x.device)
            dropped_repr = self.get_dropped_repr()
            inp = (prefix_mask * inp) + (1. - prefix_mask) * dropped_repr
        elif len(skip_prefix) > 0 and any(skip != 0 for skip in skip_prefix):
            dropped_repr = self.get_dropped_repr()
            pos_arr = torch.arange(inp_seq_len).repeat(bs).reshape([bs, -1])
            prefix_mask = torch.stack(
                [pos >= (off + 1) for pos, off in zip(pos_arr, skip_prefix)]
            ).int().unsqueeze(2).to(x.device)
            inp = (prefix_mask * inp) + (1. - prefix_mask) * dropped_repr

        if self.use_positional_embs:
            # [1, inp_seq_len, d_model]
            # NOTE: indexes into pos embs == positions \in [0, inp_seq_len).
            pos_embs = self.pos_embeddings(
                torch.arange(inp_seq_len, device=x.device)).unsqueeze(0)
            inp += pos_embs
            if return_pos_embs:
                return inp, pos_embs
            return inp

        assert not return_pos_embs
        return inp

    def get_dropped_repr(self):
        dropped_repr = torch.stack(tuple(self.unk_embeddings)).unsqueeze(0)
        if self.mask_scheme == 0:
            # Namely, [0, unk(0), unk(1)] for ncols=3.  This means:
            #   (1) SOS is never dropped.
            #   (2) indexing into unk_embeddings is based on natural_idx.
            dropped_repr = torch.cat((torch.zeros_like(
                dropped_repr[:, 0:1, :]), dropped_repr[:, :-1, :]),
                                     dim=1)
        else:
            dropped_repr = torch.cat(
                (torch.zeros_like(dropped_repr[:, 0:1, :]), dropped_repr),
                dim=1)
        return dropped_repr

    def EncodeInputInference(self, x, natural_col, out):
        """Special inference path.

        Args:
          x: [batch size, 1].  Just the data for column 'natural_col'.
          natural_col (int): [0, num cols).
          out: shaped [batch size, d_model].  To hold the encoded data.
        """
        if natural_col < 0:
            # Potentially handling SOS.
            if self.use_positional_embs:
                # Let's also add E_pos=0 to SOS (if enabled).
                out.copy_(
                    self.pos_embeddings(torch.as_tensor(
                        0,
                        device=x.device)).unsqueeze(0).expand(x.size()[0], -1))
            return

        skip = x[0, 0] < 0
        if skip:
            # [bs, d_model]
            embs = self.unk_embeddings[natural_col].unsqueeze(0).expand(
                x.shape[0], -1)
        else:
            # [bs, d_model]
            embs = self.embeddings[natural_col](x).squeeze(1)

        if self.use_positional_embs:
            # NOTE: this is tricky.  Under self.mask_scheme=0 or 1, E_pos=0 is added
            # to SOS, E_pos=1 is added to x0, etc.  So we need to take this into
            # account.
            pos = self.pos_embeddings(
                torch.as_tensor(natural_col + 1, device=x.device)).unsqueeze(0)
            embs = embs + pos

        out.copy_(embs)

    def forward(self, x, skip_prefix=[]):
        """Outputs logits for (x0, x1|x0, x2|x0,x1, ...)."""
        # [bs, ncols] -> [bs, ncols, d_model].  Right-shifted.
        if self.mask_scheme == 1:

            if self.first_query_shared:
                x = self.EncodeInput(x)
                # So that each position gets the same query vec, Q(0).
                pos_embs = torch.zeros_like(x, device=x.device)
            else:
                assert self.use_positional_embs, 'Need pos_embs for 1st layer query vecs'
                x, pos_embs = self.EncodeInput(x, return_pos_embs=True)

            x = self.blocks[0](x, query_input=pos_embs)
            for b in self.blocks[1:]:
                x = b(x)
        else:
            x = self.EncodeInput(x, skip_prefix=skip_prefix)
            pprint('going through {} blocks'.format(len(self.blocks)))
            x = self.blocks(x)

        # pprint('done blocks, out: ', x)
        x = self.norm(x)
        return x

    def forward_with_encoded_input(self, x):
        # [batch size, num cols * d_model] -> [bs, num cols, d_model]
        x = x.view(x.shape[0], -1, self.d_model)

        if self.mask_scheme == 1:
            inp_seq_len = x.shape[1]

            if self.first_query_shared:
                # So that each position gets the same query vec, Q(0).
                pos_embs = torch.zeros_like(x, device=x.device)
            else:
                assert self.use_positional_embs, 'Need pos_embs for 1st layer query vecs'
                pos_embs = self.pos_embeddings(
                    torch.arange(inp_seq_len, device=x.device)).unsqueeze(0)

            x = self.blocks[0](x, query_input=pos_embs)
            for b in self.blocks[1:]:
                x = b(x)
        else:
            x = self.blocks(x)

        x = self.norm(x)
        return x

    def nll(self, logits, data):
        """Calculates -log p(data), given logits (the conditionals).

        Args:
          logits: [batch size, ncols+1, d_model].
          data: [batch size, ncols].

        Returns:
          nll: [batch size].
        """
        if data.dtype != torch.long:
            data = data.long()
        nll = torch.zeros(logits.size()[0], device=logits.device)
        for i in range(self.nin):
            logits_i = self.logits_for_col(i, logits)
            ce = F.cross_entropy(logits_i, data[:, i], reduction='none')
            nll += ce
        return nll

    def logits_for_col(self, idx, logits):
        """Returns the logits (vector) corresponding to log p(x_i | x_(<i)).

        Args:
          idx: int, in natural (table) ordering.
          logits: [batch size, ncols+1, d_model] .

        Returns:
          logits_for_col: [batch size, domain size for column idx].
        """
        embed = self.embeddings[idx]
        return torch.matmul(logits[:, idx, :], embed.weight.t())


if __name__ == '__main__':
    # ncols = 4
    # d_model = 3
    # bs = 1
    # rng = np.random.RandomState(134)
    # attn = MultiHeadSelfAttention(d_model=d_model, num_heads=1)
    # attn.attn_mask = mask(ncols)

    # # Test i depends only on <i.
    # for i in range(ncols):
    #     inp = torch.tensor(rng.rand(bs, ncols, d_model).astype(np.float32),
    #                        requires_grad=True)

    #     out = attn(inp)
    #     l = out[:, i, :].view(-1, )
    #     pprint(l)

    #     out[:, i, :].view(-1, )[0].backward()

    #     pprint('inp.grad', inp.grad)
    #     depends = (inp.grad[0].numpy() != 0).astype(
    #         np.uint8)  # is there a gradient on the input for this k
    #     # pprint('inp.grad[0]', inp.grad[0])
    #     pprint('col', i, 'inp.grads', inp.grad, '\ndepends', depends)

    #     # Shape of depends: [ncols, d_model]
    #     assert np.all(
    #         depends[i + 1:, :].reshape(-1, ) == 0), 'i={} depends={}'.format(
    #             i, depends)

    # print('[MultiHeadSelfAttention] Passes autoregressive-ness check!')

    # block = Block(d_model=16, d_ff=64, num_heads=4)
    # block.set_attn_mask(mask(5))
    # out = block(torch.randn(1, 5, 16))

    ######

    num_cols = 3
    vocab = 1
    bs = 1
    num_cols = 11
    vocab = 5
    bs = 3
    orderings = [
        np.arange(num_cols),
        # [2, 0, 1],
        np.arange(num_cols)[::-1],
        np.random.permutation(np.arange(num_cols)),
    ]
    for ordering in orderings:
        print('Testing ordering', ordering)
        model = Transformer(
            num_blocks=2,  #2,
            # d_model=4,
            d_model=16,
            d_ff=64,
            num_heads=4,
            nin=num_cols,
            input_bins=[
                vocab,
            ] * num_cols,
            use_positional_embs=True,
            activation='gelu',
            fixed_ordering=ordering)
        print('attn_mask for blk 0', model.blocks[0].attn.attn_mask)

        for i in range(num_cols):
            nat_idx = ordering[i]
            print('\nchecking output column {} nat_idx {}...'.format(
                i, nat_idx))
            inp = torch.randint(vocab, (bs, num_cols))
            # [bs, num cols, d_model], the logits
            out = model(inp)
            pprint('inp:', inp)

            out[:, nat_idx, :].contiguous().view(-1,)[0].backward()
            pprint('grads:')
            ok = True
            for n, p in model.named_parameters():
                if 'embed' in n:
                    if p.grad is None:
                        print(n, p.grad)
                        continue
                    dep = (p.grad.reshape(-1) != 0).numpy().any()
                    pprint(
                        n,
                        'gradients?:',
                        dep,
                        'grads',  #p.grad,
                    )
                    # for j in range(i + 1, num_cols):
                    for j in range(i + 1, len(ordering)):
                        nat_idx_j = ordering[j]
                        # i.e., p corresponds to nat_idx j
                        if n == 'embeddings.{}.weight'.format(nat_idx_j):
                            ok &= (not dep)
            assert ok

        print('[Transformer] Passes autoregressive-ness check!')

        # inp = torch.randint(vocab, (bs, num_cols))
        # logits = model(inp)
        # print(model.nll(logits, inp))

        # from torchviz import make_dot, make_dot_from_trace
        # print(
        #     make_dot(logits, params=dict(model.named_parameters())))
