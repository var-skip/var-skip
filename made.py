import time

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# try:
#     from typing_extensions import Final
# except:
#     # If you don't have `typing_extensions` installed, you can use a
#     # polyfill from `torch.jit`.
#     from torch.jit import Final

# from typing import Optional


# This is a generic wrapper for any driver function you want to time
def time_this(f):

    def timed_wrapper(*args, **kw):
        start_time = time.time()
        result = f(*args, **kw)
        end_time = time.time()

        # Time taken = end_time - start_time
        print('| func:%r took: %2.4f seconds |' % \
              (f.__name__, end_time - start_time))
        # print('| func:%r args:[%r, %r] took: %2.4f seconds |' % \
        #       (f.__name__, args, kw, end_time - start_time))
        return result

    return timed_wrapper


class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    # masked_weight: Optional[torch.Tensor]

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 condition_on_ordering=False):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

        self.condition_ordering_linear = None
        if condition_on_ordering:
            self.condition_ordering_linear = nn.Linear(in_features,
                                                       out_features,
                                                       bias=False)

        self.masked_weight = None

    def set_mask(self, mask):
        """Accepts a mask of shape [in_features, out_features]."""
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def set_cached_mask(self, mask):
        self.mask.data.copy_(mask)

    def get_cached_mask(self):
        return self.mask.clone().detach()

    def forward(self, input):
        if self.masked_weight is None:
            mw = self.mask * self.weight
            out = F.linear(input, mw, self.bias)

            # NOTE: this tied-weight variant has much higher error.
            # if self.condition_ordering_linear is None:
            #     return out
            # return out + F.linear(torch.ones_like(input), mw)
        else:
            # ~17% speedup for Prog Sampling.
            out = F.linear(input, self.masked_weight, self.bias)

        if self.condition_ordering_linear is None:
            return out
        return out + F.linear(torch.ones_like(input),
                              self.mask * self.condition_ordering_linear.weight)


class MaskedResidualBlock(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 activation,
                 condition_on_ordering=False):
        assert in_features == out_features, [in_features, out_features]
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            MaskedLinear(in_features,
                         out_features,
                         bias=True,
                         condition_on_ordering=condition_on_ordering))
        self.layers.append(
            MaskedLinear(in_features,
                         out_features,
                         bias=True,
                         condition_on_ordering=condition_on_ordering))
        self.activation = activation

    def set_mask(self, mask):
        self.layers[0].set_mask(mask)
        self.layers[1].set_mask(mask)

    def set_cached_mask(self, mask):
        # They have the same mask.
        self.layers[0].mask.copy_(mask)
        self.layers[1].mask.copy_(mask)

    def get_cached_mask(self):
        return self.layers[0].mask.clone().detach()

    def forward(self, input):
        out = input
        out = self.activation(out)
        out = self.layers[0](out)
        out = self.activation(out)
        out = self.layers[1](out)
        return input + out


# class MADE(torch.jit.ScriptModule):
class MADE(nn.Module):

    def __init__(
            self,
            nin,
            hidden_sizes,
            nout,
            num_masks=1,
            natural_ordering=True,
            input_bins=None,
            activation=nn.ReLU,
            do_direct_io_connections=False,
            input_encoding=None,
            direct_io_bias=True,  # True for backward-compat of checkpoints
            output_encoding="one_hot",
            embed_size=32,
            input_no_emb_if_leq=True,
            embs_tied=False,
            residual_connections=False,
            dropout_p=0,
            fixed_dropout_p=False,
            factor_table=None,
            seed=11123,
            fixed_ordering=None,
            per_row_dropout_p=False,
            prefix_dropout=False,
            disable_learnable_unk=False,
    ):
        """MADE.

        Args:
          nin: integer; number of inputs
          hidden sizes: a list of integers; number of units in hidden layers
          nout: integer; number of outputs, which usually collectively
            parameterize some kind of 1D distribution. note: if nout is e.g. 2x
            larger than nin (perhaps the mean and std), then the first nin will
            be all the means and the second nin will be stds. i.e. output
            dimensions depend on the same input dimensions in "chunks" and
            should be carefully decoded downstream appropriately. the output of
            running the tests for this file makes this a bit more clear with
            examples.
          num_masks: can be used to train ensemble over orderings/connections
          natural_ordering: force natural ordering of dimensions, don't use
            random permutations
          input_bins: classes each input var can take on, e.g., [5, 2]
            means input x1 has values in {0, ..., 4} and x2 in {0, 1}.
        """
        super().__init__()
        self.nin = nin

        if num_masks > 1:
            # Double the weights, so need to reduce the size to be fair.
            hidden_sizes = [int(h // 2**0.5) for h in hidden_sizes]
            print("Auto reducing MO hidden sizes to", hidden_sizes, num_masks)
        # None: feed inputs as-is, no encoding applied.  Each column thus
        #     occupies 1 slot in the input layer.  For testing only.
        assert input_encoding in [
            None, "one_hot", "two_level", "binary", "binary_100p", "embed"
        ]
        self.input_encoding = input_encoding
        assert output_encoding in ["one_hot", "bits", "embed"]
        self.embed_size = self.emb_dim = embed_size
        self.output_encoding = output_encoding
        self.activation = activation
        self.nout = nout
        self.per_row_dropout_p = per_row_dropout_p
        self.prefix_dropout = prefix_dropout
        print("per row dropout", self.per_row_dropout_p)
        print("prefix dropout", self.prefix_dropout)
        self.hidden_sizes = hidden_sizes
        self.input_bins = input_bins
        self.input_no_emb_if_leq = input_no_emb_if_leq
        self.do_direct_io_connections = do_direct_io_connections
        self.embs_tied = embs_tied
        self.dropout_p = dropout_p
        if self.prefix_dropout or self.per_row_dropout_p:
            assert self.dropout_p
        self.fixed_dropout_p = fixed_dropout_p
        self.factor_table = factor_table
        self.residual_connections = residual_connections
        self.disable_learnable_unk = disable_learnable_unk
        self.num_masks = num_masks
        if nout > nin:
            # nout must be integer multiple of nin; or we're given more info.
            assert nout % nin == 0 or input_bins is not None

        self.fixed_ordering = fixed_ordering
        if fixed_ordering is not None:
            assert num_masks == 1
            print('** Fixed ordering {} supplied, ignoring natural_ordering'.
                  format(fixed_ordering))

        assert self.input_bins is not None
        encoded_bins = list(
            map(self._get_output_encoded_dist_size, self.input_bins))
        self.input_bins_encoded = list(
            map(self._get_input_encoded_dist_size, self.input_bins))
        self.input_bins_encoded_cumsum = np.cumsum(self.input_bins_encoded)

        hs = [nin] + hidden_sizes + [sum(encoded_bins)]
        # print('hs={}, nin={}, hiddens={}, encoded_bins={}'.format(
        #     hs, nin, hidden_sizes, encoded_bins))
        print('encoded_bins (output)', encoded_bins)
        print('encoded_bins (input)', self.input_bins_encoded)

        # define a simple MLP neural net
        self.net = []
        for h0, h1 in zip(hs, hs[1:]):
            if residual_connections:
                if h0 == h1:
                    self.net.extend([
                        MaskedResidualBlock(
                            h0,
                            h1,
                            activation=activation(inplace=False),
                            condition_on_ordering=self.num_masks > 1)
                    ])
                else:
                    self.net.extend([
                        MaskedLinear(h0,
                                     h1,
                                     condition_on_ordering=self.num_masks > 1),
                    ])
            else:
                self.net.extend([
                    MaskedLinear(h0,
                                 h1,
                                 condition_on_ordering=self.num_masks > 1),
                    activation(inplace=True),
                ])
        if not residual_connections:
            self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)

        if self.input_encoding is not None:
            # Input layer should be changed.
            assert self.input_bins is not None
            input_size = 0
            for i, dist_size in enumerate(self.input_bins):
                input_size += self._get_input_encoded_dist_size(dist_size)
            new_layer0 = MaskedLinear(input_size,
                                      self.net[0].out_features,
                                      condition_on_ordering=self.num_masks > 1)
            self.net[0] = new_layer0

        if self.input_encoding == "embed":
            self.embedding_networks = nn.ModuleList()
            if not self.embs_tied:
                self.embedding_networks_out = nn.ModuleList()
                for i, dist_size in enumerate(self.input_bins):
                    if dist_size <= self.embed_size and self.input_no_emb_if_leq:
                        embed = embed2 = None
                    else:
                        embed = nn.Embedding(dist_size, self.embed_size)
                        embed2 = nn.Embedding(dist_size, self.embed_size)

                    self.embedding_networks.append(embed)
                    self.embedding_networks_out.append(embed2)
            else:
                for i, dist_size in enumerate(self.input_bins):
                    if dist_size <= self.embed_size and self.input_no_emb_if_leq:
                        embed = None
                    else:
                        embed = nn.Embedding(dist_size, self.embed_size)
                    self.embedding_networks.append(embed)

        # Learnable [MASK] representation.
        if self.dropout_p:
            self.unk_embeddings = nn.ParameterList()
            print('Disable learnable?', disable_learnable_unk)
            for i, dist_size in enumerate(self.input_bins):
                self.unk_embeddings.append(
                    nn.Parameter(torch.zeros(1, self.input_bins_encoded[i]),
                                 requires_grad=not disable_learnable_unk))

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = seed if seed is not None else 11123  # for cycling through num_masks orderings
        print('self.seed', self.seed)

        self.direct_io_layer = None
        self.logit_indices = np.cumsum(encoded_bins)
        self.m = {}
        self.cached_masks = {}

        self.update_masks()  # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.
        # Logit indices for the columns.
        self.orderings = [self.m[-1]]

        # Optimization: cache some values needed in EncodeInput().
        self.bin_as_onehot_shifts = None

    def _build_or_update_direct_io(self):
        assert self.nout > self.nin and self.input_bins is not None
        direct_nin = self.net[0].in_features
        direct_nout = self.net[-1].out_features
        if self.direct_io_layer is None:
            self.direct_io_layer = MaskedLinear(
                direct_nin,
                direct_nout,
                condition_on_ordering=self.num_masks > 1)
        mask = np.zeros((direct_nout, direct_nin), dtype=np.uint8)

        print('in _build_or_update_direct_io(), self.m[-1]', self.m[-1])
        # Inverse: ord_idx -> natural idx.
        inv_ordering = [None] * self.nin
        for natural_idx in range(self.nin):
            inv_ordering[self.m[-1][natural_idx]] = natural_idx

        for ord_i in range(self.nin):
            nat_i = inv_ordering[ord_i]
            # x_(nat_i) in the input occupies range [inp_l, inp_r).
            inp_l = 0 if nat_i == 0 else self.input_bins_encoded_cumsum[nat_i -
                                                                        1]
            inp_r = self.input_bins_encoded_cumsum[nat_i]
            assert inp_l < inp_r

            for ord_j in range(ord_i + 1, self.nin):
                nat_j = inv_ordering[ord_j]
                # Output x_(nat_j) should connect to input x_(nat_i); it
                # occupies range [out_l, out_r) in the output.
                out_l = 0 if nat_j == 0 else self.logit_indices[nat_j - 1]
                out_r = self.logit_indices[nat_j]
                assert out_l < out_r
                # print('setting mask[{}:{}, {}:{}]'.format(
                #     out_l, out_r, inp_l, inp_r))
                mask[out_l:out_r, inp_l:inp_r] = 1
            # print('do_direct_io_connections mask', mask)
        # print('mask', mask)
        mask = mask.T
        self.direct_io_layer.set_mask(mask)

    def _get_input_encoded_dist_size(self, dist_size):
        if self.input_encoding == "two_level":
            dist_size += 1 + dist_size // 10
        elif self.input_encoding == "embed":
            if self.input_no_emb_if_leq:
                dist_size = min(dist_size, self.embed_size)
            else:
                dist_size = self.embed_size
        elif self.input_encoding == "one_hot":
            pass
            # if dist_size <= 2:
            #     dist_size = 1  # don't one-hot encode binary vals
        elif self.input_encoding == "binary":
            dist_size = max(1, int(np.ceil(np.log2(dist_size))))
        elif self.input_encoding == "binary_100p":
            if dist_size > 100:
                dist_size = max(1, int(np.ceil(np.log2(dist_size))))
        elif self.input_encoding is None:
            return 1
        else:
            assert False, self.input_encoding
        return dist_size

    def _get_output_encoded_dist_size(self, dist_size):
        if self.output_encoding == "two_level":
            dist_size += 1 + dist_size // 10
        elif self.output_encoding == "embed":
            if self.input_no_emb_if_leq:
                dist_size = min(dist_size, self.embed_size)
            else:
                dist_size = self.embed_size
        elif self.output_encoding == "one_hot":
            pass
            # if dist_size <= 2:
            #     dist_size = 1  # don't one-hot encode binary vals
        elif self.output_encoding == "binary":
            dist_size = max(1, int(np.ceil(np.log2(dist_size))))
        elif self.output_encoding == "binary_100p":
            if dist_size > 100:
                dist_size = max(1, int(np.ceil(np.log2(dist_size))))
        return dist_size

    def update_masks(self, invoke_order=None):
        """Update m() for all layers and change masks correspondingly.

        No-op if "self.num_masks" is 1.
        """
        if self.m and self.num_masks == 1:
            # FIXME
            # assert np.array_equal(invoke_order,
            #                       self.m[-1]), 'invoke={} curr={}'.format(
            #                           invoke_order, self.m[-1])
            return  # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)

        layers = [
            l for l in self.net if isinstance(l, MaskedLinear) or
            isinstance(l, MaskedResidualBlock)
        ]

        ### Precedence of several params determining ordering:
        #
        # invoke_order
        # orderings
        # fixed_ordering
        # natural_ordering
        #
        # from high precedence to low.

        # For multi-order models, we associate RNG seeds with orderings as
        # follows:
        #   orderings = [ o0, o1, o2, ... ]
        #   seeds = [ 0, 1, 2, ... ]
        # This must be consistent across training & inference.

        if invoke_order is not None:
            # Inference path.
            found = False
            for i in range(len(self.orderings)):
                if np.array_equal(self.orderings[i], invoke_order):
                    found = True
                    break
            if not found:
                print("WARNING: eval on order not trained on", invoke_order)
            assert found, 'specified={}, avail={}'.format(
                invoke_order, self.orderings)

            # print('found, order i=', i)

            if self.seed == (i + 1) % self.num_masks and np.array_equal(
                    self.m[-1], invoke_order):
                # During querying, after a multi-order model is configured to
                # take a specific ordering, it can be used to do multiple
                # forward passes per query.
                return

            self.seed = i
            rng = np.random.RandomState(self.seed)
            self.m[-1] = np.asarray(invoke_order)

            # print('looking up seed in cached masks:', self.seed)

            if self.seed in self.cached_masks:
                masks, direct_io_mask = self.cached_masks[self.seed]
                assert len(layers) == len(masks), (len(layers), len(masks))
                for l, m in zip(layers, masks):
                    l.set_cached_mask(m)

                if self.do_direct_io_connections:
                    assert direct_io_mask is not None
                    self.direct_io_layer.set_cached_mask(direct_io_mask)

                self.seed = (self.seed + 1) % self.num_masks
                # print('found, updated seed to', self.seed)
                return  # Early return

            curr_seed = self.seed
            self.seed = (self.seed + 1) % self.num_masks

        elif hasattr(self, 'orderings'):
            # Training path: cycle through the special orderings.
            rng = np.random.RandomState(self.seed)
            assert 0 <= self.seed and self.seed < len(self.orderings)
            self.m[-1] = self.orderings[self.seed]

            if self.seed in self.cached_masks:
                masks, direct_io_mask = self.cached_masks[self.seed]
                assert len(layers) == len(masks), (len(layers), len(masks))
                for l, m in zip(layers, masks):
                    l.set_cached_mask(m)

                if self.do_direct_io_connections:
                    assert direct_io_mask is not None
                    self.direct_io_layer.set_cached_mask(direct_io_mask)

                # print('using cached masks for seed', self.seed)
                self.seed = (self.seed + 1) % self.num_masks
                return  # Early return

            print('constructing masks with seed', self.seed, 'self.m[-1]',
                  self.m[-1])
            curr_seed = self.seed
            self.seed = (self.seed + 1) % self.num_masks

        else:
            # Train-time initial construction: either single-order, or
            # .orderings has not been assigned yet.
            rng = np.random.RandomState(self.seed)
            self.seed = (self.seed + 1) % self.num_masks
            self.m[-1] = np.arange(
                self.nin) if self.natural_ordering else rng.permutation(
                    self.nin)
            if self.fixed_ordering is not None:
                self.m[-1] = np.asarray(self.fixed_ordering)

        if self.nin > 1:
            for l in range(L):
                if self.residual_connections:
                    # sequential assignment for ResMade: https://arxiv.org/pdf/1904.05626.pdf
                    # FIXME: this seems incorrect since it's [1, ncols).
                    self.m[l] = np.array([
                        (k - 1) % (self.nin - 1)
                        # [(k - 1) % (self.nin - 1) + 1
                        for k in range(self.hidden_sizes[l])
                    ])
                else:
                    # Samples from [0, ncols - 1).
                    self.m[l] = rng.randint(self.m[l - 1].min(),
                                            self.nin - 1,
                                            size=self.hidden_sizes[l])
        else:
            # This should result in first layer's masks == 0.
            # So output units are disconnected to any inputs.
            for l in range(L):
                self.m[l] = np.asarray([-1] * self.hidden_sizes[l])

        # print('ordering', self.m[-1])
        # print('self.m', self.m)

        # construct the mask matrices
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]

        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        if self.nout > self.nin:
            # Last layer's mask needs to be changed.

            if self.input_bins is None:
                k = int(self.nout / self.nin)
                # replicate the mask across the other outputs
                # so [x1, x2, ..., xn], ..., [x1, x2, ..., xn].
                masks[-1] = np.concatenate([masks[-1]] * k, axis=1)
            else:
                # [x1, ..., x1], ..., [xn, ..., xn] where the i-th list has
                # input_bins[i - 1] many elements (multiplicity, # of classes).
                mask = np.asarray([])
                for k in range(masks[-1].shape[0]):
                    tmp_mask = []
                    for idx, x in enumerate(zip(masks[-1][k], self.input_bins)):
                        mval, nbins = x[0], self._get_output_encoded_dist_size(
                            x[1])
                        tmp_mask.extend([mval] * nbins)

                    tmp_mask = np.asarray(tmp_mask)
                    if k == 0:
                        mask = tmp_mask
                    else:
                        mask = np.vstack([mask, tmp_mask])
                masks[-1] = mask

        if self.input_encoding is not None:
            # Input layer's mask should be changed.

            assert self.input_bins is not None
            # [nin, hidden].
            mask0 = masks[0]
            new_mask0 = []
            for i, dist_size in enumerate(self.input_bins):
                dist_size = self._get_input_encoded_dist_size(dist_size)
                # [dist size, hidden]
                new_mask0.append(
                    np.concatenate([mask0[i].reshape(1, -1)] * dist_size,
                                   axis=0))
            # [sum(dist size), hidden]
            new_mask0 = np.vstack(new_mask0)
            masks[0] = new_mask0

        assert len(layers) == len(masks), (len(layers), len(masks))
        for l, m in zip(layers, masks):
            l.set_mask(m)

        dio_mask = None
        if self.do_direct_io_connections:
            self._build_or_update_direct_io()
            dio_mask = self.direct_io_layer.get_cached_mask()

        # Cache.
        if hasattr(self, 'orderings'):
            print('caching masks for seed', curr_seed)
            masks = [l.get_cached_mask() for l in layers]
            print('signatures:', [m.sum() for m in masks]
                + [dio_mask.sum() if dio_mask is not None else 0])
            assert curr_seed not in self.cached_masks
            self.cached_masks[curr_seed] = (masks, dio_mask)

    def name(self):
        n = 'made'
        if self.residual_connections:
            n += '-resmade'
        n += '-hidden' + '_'.join(str(h) for h in self.hidden_sizes)
        n += '-emb' + str(self.embed_size)
        if self.num_masks > 1:
            n += '-{}masks'.format(self.num_masks)
        if not self.natural_ordering:
            n += '-nonNatural'
        n += ('-no' if not self.do_direct_io_connections else '-') + 'directIo'
        n += '-{}In{}Out'.format(self.input_encoding, self.output_encoding)
        n += '-embsTied' if self.embs_tied else '-embsNotTied'
        if self.input_no_emb_if_leq:
            n += '-inputNoEmbIfLeq'
        if self.dropout_p:
            n += '-dropout'
            if self.disable_learnable_unk:
                n += '-nolearnableUnk'
            else:
                n += '-learnableUnk'
            if self.fixed_dropout_p:
                n += '-fixedDropout{:.2f}'.format(self.dropout_p)
        return n

    def get_unk(self, i):
        if self.disable_learnable_unk:
            return torch.zeros_like(self.unk_embeddings[i].detach())
        else:
            return self.unk_embeddings[i]

    # @torch.jit.script
    def Embed(self, data, natural_col=None, out=None):
        if data is None:
            if out is None:
                return self.get_unk(natural_col)
            out.copy_(self.get_unk(natural_col))
            return out

        bs = data.size()[0]
        y_embed = []
        data = data.long()

        if natural_col is not None:
            # Fast path only for inference.  One col.

            coli_dom_size = self.input_bins[natural_col]
            # Embed?
            if coli_dom_size > self.embed_size or not self.input_no_emb_if_leq:
                res = self.embedding_networks[natural_col](data.view(-1,))
                if out is not None:
                    out.copy_(res)
                    return out
                return res
            else:
                if out is None:
                    out = torch.zeros(bs, coli_dom_size, device=data.device)

                out.scatter_(1, data, 1)
                return out
        else:
            if self.per_row_dropout_p == 1 or self.per_row_dropout_p is True:
                row_dropout_probs = torch.rand(bs, device=data.device)
            elif self.per_row_dropout_p == 2:
                # Also per row masking, but makes more sense (draw num masked
                # tokens first).  In [0, 1).
                row_dropout_probs = torch.randint(
                    0, self.nin, (bs,), device=data.device).float() / self.nin

            row_dropout_lim = torch.rand(bs, device=data.device) * len(
                self.input_bins)
            for i, coli_dom_size in enumerate(self.input_bins):
                # Wildcard column? use -1 as special token.
                # Inference pass only (see estimators.py).
                not_skip = data[:, i] >= 0
                data_col = torch.clamp(data[:, i], 0)

                # Embed?
                if coli_dom_size > self.embed_size or not self.input_no_emb_if_leq:
                    # assert not self.dropout_p, "not implemented"
                    col_i_embs = self.embedding_networks[i](data_col)
                    if not self.dropout_p:
                        y_embed.append(col_i_embs)
                    else:
                        dropped_repr = self.get_unk(i)

                        # During training, non-dropped 1's are scaled by
                        # 1/(1-p), so we clamp back to 1.
                        def dropout_p():
                            if self.fixed_dropout_p:
                                return self.dropout_p
                            return 1. - np.random.randint(
                                1, self.nin + 1) * 1. / self.nin

                        batch_mask = torch.clamp(
                            torch.dropout(
                                torch.ones(bs, 1, device=data.device),
                                p=dropout_p(),
                                # np.random.randint(5, 12) * 1. / self.nin,
                                train=self.training),
                            0,
                            1)
                        if self.training and self.per_row_dropout_p:
                            # 1 means original repr, 0 means use masked repr.
                            batch_mask = (
                                torch.rand(bs, device=data.device) >=
                                row_dropout_probs).float().unsqueeze(1)
                        elif self.training and self.prefix_dropout:
                            batch_mask = (i * torch.ones(bs, device=data.device)
                                          >
                                          row_dropout_lim).float().unsqueeze(1)
                        elif not self.training:
                            batch_mask = not_skip.float().unsqueeze(1)
                        y_embed.append(batch_mask * col_i_embs +
                                       (1. - batch_mask) * dropped_repr)
                else:
                    y_onehot = torch.zeros(bs,
                                           coli_dom_size,
                                           device=data.device)
                    y_onehot.scatter_(1, data_col.view(-1, 1), 1)
                    if self.dropout_p:
                        dropped_repr = self.get_unk(i)
                        if self.factor_table and self.factor_table.columns[
                                i].factor_id:
                            pass  # use prev col's batch mask
                        else:
                            # During training, non-dropped 1's are scaled by
                            # 1/(1-p), so we clamp back to 1.
                            def dropout_p():
                                if self.fixed_dropout_p:
                                    return self.dropout_p
                                return 1. - np.random.randint(
                                    1, self.nin + 1) * 1. / self.nin

                            batch_mask = torch.clamp(
                                torch.dropout(
                                    torch.ones(bs, 1, device=data.device),
                                    # p=self.dropout_p,
                                    p=dropout_p(),
                                    # np.random.randint(5, 12) * 1. / self.nin,
                                    train=self.training),
                                0,
                                1)
                            if self.training and self.per_row_dropout_p:
                                # 1 means original repr, 0 means use masked repr.
                                batch_mask = (
                                    torch.rand(bs, device=data.device) >=
                                    row_dropout_probs).float().unsqueeze(1)
                            elif self.training and self.prefix_dropout:
                                batch_mask = (
                                    i * torch.ones(bs, device=data.device) >
                                    row_dropout_lim).float().unsqueeze(1)
                            elif not self.training:
                                batch_mask = not_skip.float().unsqueeze(1)
                        y_embed.append(batch_mask * y_onehot +
                                       (1. - batch_mask) * dropped_repr)
                    else:
                        y_embed.append(y_onehot)
            return torch.cat(y_embed, 1)

    def ToOneHot(self, data):
        assert not self.dropout_p, "not implemented"
        bs = data.size()[0]
        y_onehots = []
        data = data.long()
        for i, coli_dom_size in enumerate(self.input_bins):
            if coli_dom_size <= 2:
                y_onehots.append(data[:, i].view(-1, 1).float())
            else:
                y_onehot = torch.zeros(bs, coli_dom_size, device=data.device)
                y_onehot.scatter_(1, data[:, i].view(-1, 1), 1)
                y_onehots.append(y_onehot)

        # [bs, sum(dist size)]
        return torch.cat(y_onehots, 1)

    def ToBinaryAsOneHot(self, data, threshold=0, natural_col=None, out=None):
        if data is None:
            if out is None:
                return self.get_unk(natural_col)
            out.copy_(self.get_unk(natural_col))
            return out

        bs = data.size()[0]
        data = data.long()
        # print('data.device', data.device)

        if self.bin_as_onehot_shifts is None:
            # This caching gives very sizable gains.
            self.bin_as_onehot_shifts = [None] * self.nin
            const_one = torch.ones([], dtype=torch.long, device=data.device)
            for i, coli_dom_size in enumerate(self.input_bins):
                # Max with 1 to guard against cols with 1 distinct val.
                one_hot_dims = max(1, int(np.ceil(np.log2(coli_dom_size))))
                self.bin_as_onehot_shifts[i] = const_one << torch.arange(
                    one_hot_dims, device=data.device)
            # print('data.device', data.device, 'const_one', const_one.device,
            #       'bin_as_onehot_shifts', self.bin_as_onehot_shifts[0].device)

        if natural_col is None:
            # Train path.

            assert out is None
            y_onehots = [None] * self.nin
            if self.per_row_dropout_p == 1 or self.per_row_dropout_p is True:
                row_dropout_probs = torch.rand(bs, device=data.device)
            elif self.per_row_dropout_p == 2:
                # Also per row masking, but makes more sense (draw num masked
                # tokens first).  In [0, 1).
                row_dropout_probs = torch.randint(
                    0, self.nin, (bs,), device=data.device).float() / self.nin
            row_dropout_lim = torch.rand(bs, device=data.device) * len(
                self.input_bins)

            for i, coli_dom_size in enumerate(self.input_bins):
                if coli_dom_size > threshold:
                    # Bit shift in PyTorch + GPU is 27% faster than np.
                    # data_np = data[:, i].view(-1, 1)
                    data_np = data.narrow(1, i, 1)
                    # print(data_np.device, self.bin_as_onehot_shifts[i].device)
                    binaries = (data_np & self.bin_as_onehot_shifts[i]) > 0
                    y_onehots[i] = binaries

                    if self.dropout_p:
                        dropped_repr = self.get_unk(i)

                        # During training, non-dropped 1's are scaled by
                        # 1/(1-p), so we clamp back to 1.
                        def dropout_p():
                            if self.fixed_dropout_p:
                                return self.dropout_p
                            return 1. - np.random.randint(
                                1, self.nin + 1) * 1. / self.nin

                        batch_mask = torch.clamp(
                            torch.dropout(
                                torch.ones(bs, 1, device=data.device),
                                # p=self.dropout_p,
                                p=dropout_p(),
                                # np.random.randint(5, 12) * 1. / self.nin,
                                train=self.training),
                            0,
                            1)  #.to(torch.int8, non_blocking=True, copy=False)
                        if self.training and self.per_row_dropout_p:
                            batch_mask = (
                                torch.rand(bs, device=data.device) >=
                                row_dropout_probs).float().unsqueeze(1)
                        elif self.training and self.prefix_dropout:
                            batch_mask = (i * torch.ones(bs, device=data.device)
                                          >
                                          row_dropout_lim).float().unsqueeze(1)
                        binaries = binaries.to(torch.float32,
                                               non_blocking=True,
                                               copy=False)
                        # print(batch_mask.dtype, binaries.dtype, dropped_repr.dtype)
                        # assert False
                        y_onehots[i] = batch_mask * binaries + (
                            1. - batch_mask) * dropped_repr

                else:
                    # encode as plain one-hot
                    y_onehot = torch.zeros(bs,
                                           coli_dom_size,
                                           device=data.device)
                    y_onehot.scatter_(1, data[:, i].view(-1, 1), 1)
                    y_onehots[i] = y_onehot

            # [bs, sum(log2(dist size))]
            res = torch.cat(y_onehots, 1)
            return res.to(torch.float32, non_blocking=True, copy=False)

        else:
            # Inference path.
            natural_idx = natural_col
            coli_dom_size = self.input_bins[natural_idx]

            # skip = data is None #data[0, 0] < 0
            # if skip:
            #     if out is None:
            #         return self.unk_embeddings[natural_idx]
            #     out.copy_(self.unk_embeddings[natural_idx])
            #     return out

            if coli_dom_size > threshold:
                # Bit shift in PyTorch + GPU is 27% faster than np.
                # data_np = data[:, i].view(-1, 1)
                data_np = data  #.narrow(1, 0, 1)
                # print(data_np.device, self.bin_as_onehot_shifts[i].device)
                if out is None:
                    res = (data_np & self.bin_as_onehot_shifts[natural_idx]) > 0
                    return res.to(torch.float32, non_blocking=True, copy=False)
                else:
                    out.copy_(
                        (data_np & self.bin_as_onehot_shifts[natural_idx]) > 0)
                    return out
            else:
                assert False, 'inference'
                # encode as plain one-hot
                if out is None:
                    y_onehot = torch.zeros(bs,
                                           coli_dom_size,
                                           device=data.device)
                    y_onehot.scatter_(
                        1,
                        data,  #data[:, i].view(-1, 1),
                        1)
                    res = y_onehot
                    return res.to(torch.float32, non_blocking=True, copy=False)

                out.scatter_(1, data, 1)
                return out

    def ToTwoLevel(self, data):
        bs = data.size()[0]
        y_onehots = []
        data = data.long()
        for i, coli_dom_size in enumerate(self.input_bins):

            y_onehot = torch.zeros(bs, coli_dom_size, device=data.device)
            y_onehot.scatter_(1, data[:, i].view(-1, 1), 1)
            y_onehot = torch.dropout(y_onehot, p=0.3, train=self.training)

            # add on one-hot encoding at coarser second-level
            # e.g., for domain of 35, the 2nd level will have domain size of 4
            second_level_dom_size = 1 + coli_dom_size // 10
            y2_onehot = torch.zeros(bs,
                                    second_level_dom_size,
                                    device=data.device)
            y2_onehot.scatter_(1, data[:, i].view(-1, 1) // 10, 1)

            y_onehots.append(y_onehot)
            y_onehots.append(y2_onehot)

        # [bs, sum(dist size) + sum(2nd_level)]
        return torch.cat(y_onehots, 1)

    # @time_this
    def EncodeInput(self, data, natural_col=None, out=None):
        """"Warning: this could take up a significant portion of a forward pass.

        Args:
          natural_col: if specified, 'data' has shape [N, 1] corresponding to
              col-'natural-col'.  Otherwise 'data' corresponds to all cols.
          out: if specified, assign results into this Tensor storage.
        """
        if self.input_encoding == "binary":
            # TODO: try out=out see if it helps dmv11
            return self.ToBinaryAsOneHot(data, natural_col=natural_col, out=out)
        elif self.input_encoding == "embed":
            return self.Embed(data, natural_col=natural_col, out=out)
        elif self.input_encoding is None:
            return data
        elif self.input_encoding == "one_hot":
            return self.ToOneHot(data)
        elif self.input_encoding == "two_level":
            return self.ToTwoLevel(data)
        elif self.input_encoding == "binary_100p":
            return self.ToBinaryAsOneHot(data, threshold=100)
        else:
            assert False, self.input_encoding

    # @torch.jit.script_method
    # @torch.jit.ignore
    def forward(self, x, skip_prefix=[]):
        """Calculates unnormalized logits.

        If self.input_bins is not specified, the output units are ordered as:
            [x1, x2, ..., xn], ..., [x1, x2, ..., xn].
        So they can be reshaped as thus and passed to a cross entropy loss:
            out.view(-1, model.nout // model.nin, model.nin)

        Otherwise, they are ordered as:
            [x1, ..., x1], ..., [xn, ..., xn]
        And they can't be reshaped directly.

        Args:
          x: [bs, ncols].
        """
        if skip_prefix:
            assert len(skip_prefix) == x.shape[0], (len(skip_prefix), x.shape)
            for i, n in enumerate(skip_prefix):
                x[i][:n] = -1
        x = self.EncodeInput(x)

        if self.direct_io_layer is not None:
            residual = self.direct_io_layer(x)
            return self.net(x) + residual

        return self.net(x)

    # @time_this
    # @torch.jit.export
    def forward_with_encoded_input(self, x):

        if self.direct_io_layer is not None:
            residual = self.direct_io_layer(x)
            return self.net(x) + residual

        return self.net(x)

    # FIXME: be careful about this.... if we add this inference on old MADE ckpts seems broken
    def do_forward(self, x, ordering):
        """Performs forward pass, invoking a specified ordering."""
        self.update_masks(invoke_order=ordering)

        if self.direct_io_layer is not None:
            residual = self.direct_io_layer(x)
            return self.net(x) + residual

        return self.net(x)

    def logits_for_col(self, idx, logits):
        """Returns the logits (vector) corresponding to log p(x_i | x_(<i)).

        Args:
          idx: int, in natural (table) ordering.
          logits: [batch size, hidden] where hidden can either be sum(dom
            sizes), or emb_dims.

        Returns:
          logits_for_col: [batch size, domain size for column idx].
        """
        assert self.input_bins is not None

        if idx == 0:
            # print('slicing out', self.logit_indices[0])
            logits_for_var = logits[:, :self.logit_indices[0]]
        else:
            # print('slicing out', self.logit_indices[idx - 1], 'and', self.logit_indices[idx])
            logits_for_var = logits[:, self.logit_indices[idx - 1]:self.
                                    logit_indices[idx]]
        if self.output_encoding != 'embed':
            return logits_for_var

        if self.embs_tied:
            embed = self.embedding_networks[idx]
        else:
            # some ckpts do not tie weights....
            embed = self.embedding_networks_out[idx]

        if embed is None:
            # Can be None for small domain size columns.
            return logits_for_var

        # Otherwise, dot with embedding matrix to get the true logits.
        # [bs, emb] * [emb, dom size for idx]
        return torch.matmul(
            logits_for_var,
            # embed.weight.t().to(torch.float32)
            embed.weight.t())
        # * torch.rsqrt(torch.tensor(self.embed_size, dtype=torch.float))

    def HasMaterializedOutput(self, natural_idx):
        return self.input_bins[natural_idx] < 1e6

    def nll(self, logits, data):
        """Calculates -log p(data), given logits (the conditionals).

        Args:
          logits: [batch size, hidden] where hidden can either be sum(dom
            sizes), or emb_dims.
          data: [batch size, nin].

        Returns:
          nll: [batch size].
        """
        if data.dtype != torch.long:
            data = data.long()
        nll = torch.zeros(logits.size()[0], device=logits.device)
        for i in range(self.nin):
            if self.HasMaterializedOutput(i):
                logits_i = self.logits_for_col(i, logits)
                nll += F.cross_entropy(logits_i, data[:, i], reduction='none')
            else:
                # assert False
                # Discretized MoL.
                mixture_params_i = self.logits_for_col

        return nll

    def sample(self, num=1, device=None):
        assert self.natural_ordering
        with torch.no_grad():
            sampled = torch.zeros((num, self.nin), device=device)

            if self.nout > self.nin:
                if self.input_bins is None:
                    assert num == 1, 'implement me'
                    # Softmax on discrete classes.
                    for i in range(self.nin):
                        logits = self.forward(sampled)
                        l = logits[0].view(self.nout // self.nin, self.nin)
                        l = torch.softmax(l[:, i], 0)
                        sampled[0, i] = torch.multinomial(l, 1)
                else:
                    indices = np.cumsum(self.input_bins)
                    for i in range(self.nin):
                        logits = self.forward(sampled)
                        if i > 0:
                            scores_for_i = logits[:, indices[i - 1]:indices[i]]
                        else:
                            scores_for_i = logits[:, :indices[0]]
                        s = torch.multinomial(torch.softmax(scores_for_i, -1),
                                              1)
                        sampled[:, i] = s.view(-1,)
            else:
                assert num == 1, 'implement me'
                # Binary variables.
                for i in range(self.nin):
                    logits = self.forward(sampled)
                    p = torch.sigmoid(logits[0, i])
                    # Turn on the pixel with probability p.
                    sampled[0, i] = 1 if np.random.rand() < p else 0

            return sampled


if __name__ == '__main__':
    # Checks for the autoregressive property.
    rng = np.random.RandomState(14)
    # (nin, hiddens, nout, input_bins, direct_io)
    configs_with_input_bins = [
        # (4, [32, 512], 122 * 4, [122] * 4, False),
        (2, [10], 2 + 5, [2, 5], False),
        (2, [10, 30], 2 + 5, [2, 5], False),
        (3, [6], 2 + 2 + 2, [2, 2, 2], False),
        (3, [4, 4], 2 + 1 + 2, [2, 1, 2], False),
        (4, [16, 8, 16], 2 + 3 + 1 + 2, [2, 3, 1, 2], False),
        (2, [10], 2 + 5, [2, 5], True),
        (2, [10, 30], 2 + 5, [2, 5], True),
        (3, [6], 2 + 2 + 2, [2, 2, 2], True),
        (3, [4, 4], 2 + 1 + 2, [2, 1, 2], True),
        (4, [16, 8, 16], 2 + 3 + 1 + 2, [2, 3, 1, 2], True),
    ]
    for nin, hiddens, nout, input_bins, direct_io in configs_with_input_bins:
        print(nin, hiddens, nout, input_bins, direct_io, '...', end='')
        model = MADE(nin,
                     hiddens,
                     nout,
                     input_bins=input_bins,
                     natural_ordering=True,
                     do_direct_io_connections=direct_io)
        model.eval()
        print(model)
        for k in range(nout):
            inp = torch.tensor(rng.rand(1, nin).astype(np.float32),
                               requires_grad=True)
            loss = model(inp)
            l = loss[0, k]
            l.backward()
            depends = (inp.grad[0].numpy() != 0).astype(
                np.uint8)  # is there a gradient on the input for this k

            depends_ix = np.where(depends)[0].astype(np.int32)  #indexes where
            var_idx = np.argmax(k < np.cumsum(input_bins))
            prev_idxs = np.arange(var_idx).astype(np.int32)

            # Asserts that k depends only on < var_idx.
            print('depends', depends_ix, 'prev_idxs', prev_idxs)
            assert len(torch.nonzero(inp.grad[0, var_idx:])) == 0
        print('ok')
