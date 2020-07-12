import collections
import json
import os
import hashlib
import multiprocessing.dummy as mp
import random
import time

import numpy as np
import pandas as pd

import made
import torch
from transformer import Transformer

OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal
}


def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the 'true' ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering


class CardEst(object):
    """Base class for a cardinality estimator."""

    def __init__(self):
        self.query_starts = []
        self.query_dur_ms = []
        self.errs = []
        self.est_cards = []
        self.true_cards = []

        self.name = 'CardEst'

    def Query(self, columns, operators, vals):
        """Estimates cardinality with the specified conditions.

        Args:
            columns: list of Column objects to filter on.
            operators: list of string representing what operation to perform on
              respective columns; e.g., ['<', '>='].
            vals: list of raw values to filter columns on; e.g., [50, 100000].
              These are not bin IDs.
        Returns:
            Predicted cardinality.
        """
        raise NotImplementedError

    def OnStart(self):
        self.query_starts.append(time.time())

    def OnEnd(self):
        self.query_dur_ms.append((time.time() - self.query_starts[-1]) * 1e3)

    def AddError(self, err):
        self.errs.append(err)

    def AddError(self, err, est_card, true_card):
        self.errs.append(err)
        self.est_cards.append(est_card)
        self.true_cards.append(true_card)

    def __str__(self):
        return self.name

    def get_stats(self):
        return [
            self.query_starts,
            self.query_dur_ms,
            self.errs,
            self.est_cards,
            self.true_cards  #, self.under_count, self.over_count
        ]

    def merge_stats(self, state):
        self.query_starts.extend(state[0])
        self.query_dur_ms.extend(state[1])
        self.errs.extend(state[2])
        self.est_cards.extend(state[3])
        self.true_cards.extend(state[4])
        # self.under_count += state[5]
        # self.over_count += state[6]

    def report(self):
        est = self
        if not est.errs:
            print("no errors to report")
            return
        print(str(est), "max", np.max(est.errs), "99th",
              np.quantile(est.errs, 0.99), "95th", np.quantile(est.errs, 0.95),
              "median", np.quantile(est.errs, 0.5), "time_ms",
              np.mean(est.query_dur_ms))
        # print(
        #     "Estimator error",
        #     est,
        #     "max",
        #     np.max(est.errs),
        #     "mean",
        #     np.mean(est.errs),
        #     "median",
        #     np.median(est.errs),
        #     # "over-estimates",
        #     # est.over_count,
        #     # "under-estimates", est.under_count,
        #     "time_ms",
        #     np.mean(est.query_dur_ms))


def QueryToPredicate(columns, operators, vals, wrap_as_string_cols=None):
    """Converts from (c,o,v) to sql string (for Postgres/SQLServer)."""
    v_s = [
        str(v).replace('T', ' ') if type(v) is np.datetime64 else v
        for v in vals
    ]
    v_s = ["\'" + v + "\'" if type(v) is str else str(v) for v in v_s]

    if wrap_as_string_cols is not None:
        for i in range(len(columns)):
            if columns[i].name in wrap_as_string_cols:
                v_s[i] = "'" + str(v_s[i]) + "'"

    preds = [
        c.pg_name + ' ' + o + ' ' + v
        for c, o, v in zip(columns, operators, v_s)
    ]
    s = ' and '.join(preds)
    return ' where ' + s


def FillInUnqueriedColumns(table, columns, operators, vals):
    """Allows for some columns to be unqueried (i.e., wildcard).

    Returns cols, ops, vals, where all 3 lists of all size len(table.columns),
    in the table's natural column order.

    A None in ops/vals means that column slot is unqueried.
    """
    ncols = len(table.columns)
    cs = table.columns
    os, vs = [None] * ncols, [None] * ncols

    for c, o, v in zip(columns, operators, vals):
        idx = table.ColumnIndex(c.name)
        os[idx] = o
        vs[idx] = v

    return cs, os, vs


class ProgressiveSamplingMade(CardEst):
    """Progressive sampling."""

    def __init__(
            self,
            model,
            table,
            r,
            device=None,
            seed=False,
            half=False,
            cardinality=None,
            shortcircuit=False  # Skip sampling on wildcards?
    ):
        super(ProgressiveSamplingMade, self).__init__()
        torch.set_grad_enabled(False)
        self.model = model
        self.table = table
        self.half = half
        self.shortcircuit = shortcircuit

        if r <= 1.0:
            self.r = r  # Reduction ratio.
            self.num_samples = None
        else:
            self.num_samples = r

        self.seed = seed
        self.device = device

        self.cardinality = cardinality
        if cardinality is None:
            self.cardinality = table.cardinality

        # For data shift experiment.  In natural order.
        self.domain_masks = None

        with torch.no_grad():
            # print('psample device', device)
            if self.half:
                self.init_logits = self.model(
                    torch.zeros(1,
                                self.model.nin,
                                device=device,
                                dtype=torch.float16))
            else:
                self.init_logits = self.model(
                    torch.zeros(1, self.model.nin, device=device))

        self.dom_sizes = [c.DistributionSize() for c in self.table.columns]
        self.dom_sizes = np.cumsum(self.dom_sizes)

        # Inference optimizations below.

        self.traced_fwd = None
        # We can't seem to trace this because it depends on a scalar input.
        self.traced_encode_input = model.EncodeInput

        # Important: enable this inference-time optimization only when model is
        # single-order.  When multiple orders, the masks would be updated so
        # don't cache a single version of "mask * weight".
        if 'MADE' in str(model) and self.model.num_masks == 1:
            for layer in model.net:
                if type(layer) == made.MaskedLinear:
                    if layer.masked_weight is None:
                        layer.masked_weight = layer.mask * layer.weight
                        print('Setting masked_weight in MADE, do not retrain!',
                              layer)
        # for p in model.parameters():
        #     p.detach_()
        #     p.requires_grad = False
        # self.init_logits.detach_()

        self.workers = 1
        # mp.set_start_method('fork')
        # mp.set_start_method('spawn', force=True)
        if self.workers > 1:
            print('Using {} workers'.format(self.workers))
            self.pool = mp.Pool(processes=self.workers)

        with torch.no_grad():
            if self.half:
                print(
                    '  (self.num_samples, self.model.nin), device=self.device',
                    self.num_samples, self.model.nin, self.device)
                self.kZeros = torch.zeros((self.num_samples, self.model.nin),
                                          device=self.device,
                                          dtype=torch.float16)  #.half()
                # self.num_samples, self.model.nin,
                # device=self.device, dtype=torch.float16)#.half()
                self.inp = self.traced_encode_input(self.kZeros).half()
            else:
                self.kZeros = torch.zeros(self.num_samples,
                                          self.model.nin,
                                          device=self.device)
                self.inp = self.traced_encode_input(self.kZeros)
                # assert False, 'after traced_encode_input {}'.format(self.inp)

            # For transformer, need to flatten [num cols, d_model].
            self.inp = self.inp.view(self.num_samples, -1)
            # self.inp[:,:] = 0

    def __str__(self):
        if self.num_samples:
            n = self.num_samples
        else:
            n = int(self.r * self.table.columns[0].DistributionSize())
        if self.shortcircuit:
            return 'psample_shortcircuit_{}'.format(n)
        else:
            return 'psample_{}'.format(n)

    def _sample_n(self,
                  num_samples,
                  ordering,
                  columns,
                  operators,
                  vals,
                  worker_id=None,
                  inp=None):
        # operators = torch.jit.annotate(List[str], [])
        # for op in operators0:
        #     operators.append(op)

        # torch.set_grad_enabled(False)
        ncols = len(columns)
        logits = self.init_logits
        if inp is None:
            l = worker_id * num_samples
            r = l + num_samples
            inp = self.inp[l:r]
        masked_logits = []

        # Use the query to filter each column's domain.
        valid_i_list = [None] * ncols  # None means all valid.
        for i in range(ncols):
            natural_idx = ordering[i]
            # natural_idx = i if ordering is None else ordering[i]

            # Column i.
            op = operators[natural_idx]
            if op is not None:
                # There exists a filter.
                # if op == '>':
                #     # dvs = torch.jit.annotate(List[str], [])
                #     dvs = columns[natural_idx]
                #     dvs = dvs.all_distinct_values

                #     valid_i = ( dvs> vals[natural_idx]).astype(np.float32,
                #                                          copy=False)

                valid_i = OPS[op](columns[natural_idx].all_distinct_values,
                                  vals[natural_idx]).astype(np.float32,
                                                            copy=False)

                if self.domain_masks is not None:
                    valid_i *= self.domain_masks[natural_idx]

            elif self.domain_masks is not None:
                # If enabled, even wildcard domains should get masked.
                valid_i = self.domain_masks[natural_idx]
            else:
                continue

            # This line triggers a host -> gpu copy, showing up as a
            # hotspot in cprofile.
            if self.half:
                valid_i_list[i] = torch.as_tensor(valid_i,
                                                  device=self.device,
                                                  dtype=torch.float16)
            else:
                valid_i_list[i] = torch.as_tensor(valid_i, device=self.device)
                # valid_i_list[i] = torch.from_numpy(valid_i).to(device=self.device, non_blocking=True, copy=False)

        # Fill in wildcards, if enabled.
        if self.shortcircuit:
            for i in range(ncols):
                natural_idx = i if ordering is None else ordering[i]
                if operators[natural_idx] is None and natural_idx != ncols - 1:
                    if natural_idx == 0:
                        self.model.EncodeInput(
                            None,
                            natural_col=0,
                            out=inp[:, :self.model.
                                    input_bins_encoded_cumsum[0]])
                    else:
                        l = self.model.input_bins_encoded_cumsum[natural_idx -
                                                                 1]
                        r = self.model.input_bins_encoded_cumsum[natural_idx]
                        self.model.EncodeInput(None,
                                               natural_col=natural_idx,
                                               out=inp[:, l:r])

        # Actual progressive sampling.  Repeat:
        #   Sample next var from curr logits -> fill in next var
        #   Forward pass -> curr logits
        for i in range(ncols):
            natural_idx = i if ordering is None else ordering[i]

            # If wildcard enabled, 'logits' wasn't assigned last iter.
            if not self.shortcircuit or operators[natural_idx] is not None:
                probs_i = torch.softmax(
                    self.model.logits_for_col(natural_idx, logits), 1)

                valid_i = valid_i_list[i]
                logits_i = probs_i
                if valid_i is not None:
                    logits_i *= valid_i

                logits_i_summed = logits_i.sum(1)

                masked_logits.append(logits_i_summed)

                # If some paths have vanished (~0 prob), assign some nonzero
                # mass to the whole row so that multinomial() doesn't complain.
                paths_vanished = (logits_i_summed <= 0).view(-1, 1)
                logits_i = logits_i.masked_fill_(paths_vanished, 1.0)

            if i < ncols - 1:
                # Num samples to drawfor column i.
                if i != 0:
                    num_i = 1
                else:
                    num_i = num_samples if num_samples else int(
                        self.r * self.dom_sizes[natural_idx])

                if self.shortcircuit and operators[natural_idx] is None:
                    data_to_encode = None
                else:
                    samples_i = torch.multinomial(
                        logits_i, num_samples=num_i,
                        replacement=True)  # [bs, num_i]
                    data_to_encode = samples_i.view(-1, 1)

                # Encode input: i.e., put sampled vars into input buffer.
                if data_to_encode is not None:  # Wildcards are encoded already.
                    if not isinstance(self.model, Transformer):
                        if natural_idx == 0:
                            self.model.EncodeInput(
                                data_to_encode,
                                natural_col=0,
                                out=inp[:, :self.model.
                                        input_bins_encoded_cumsum[0]])
                        else:
                            l = self.model.input_bins_encoded_cumsum[natural_idx
                                                                     - 1]
                            r = self.model.input_bins_encoded_cumsum[
                                natural_idx]
                            self.model.EncodeInput(data_to_encode,
                                                   natural_col=natural_idx,
                                                   out=inp[:, l:r])
                    else:
                        # Transformer.  Need special treatment due to right-shift.
                        l = (natural_idx + 1) * self.model.d_model
                        r = l + self.model.d_model
                        if i == 0:
                            # Let's also add E_pos=0 to SOS (if enabled).
                            # This is a no-op if disabled pos embs.
                            self.model.EncodeInput(
                                data_to_encode,  # Will ignore.
                                natural_col=-1,  # Signals SOS.
                                out=inp[:, :self.model.d_model])

                        if self.model.mask_scheme == 1:
                            # Should encode natural_col \in [0, ncols).
                            self.model.EncodeInput(data_to_encode,
                                                   natural_col=natural_idx,
                                                   out=inp[:, l:r])
                        elif natural_idx < self.model.nin - 1:
                            # If scheme is 0, should not encode the last variable.
                            self.model.EncodeInput(data_to_encode,
                                                   natural_col=natural_idx,
                                                   out=inp[:, l:r])

                # Actual forward pass.
                next_natural_idx = i + 1 if ordering is None else ordering[i +
                                                                           1]
                if self.shortcircuit and operators[next_natural_idx] is None:
                    # If next variable in line is wildcard, then don't do
                    # this forward pass.  Var 'logits' won't be accessed.
                    continue

                if hasattr(self.model, 'do_forward'):
                    # With a specific ordering.
                    if isinstance(self.model, made.MADE):
                        # print('do_forward gets ordering:', InvertOrder(ordering))
                        logits = self.model.do_forward(inp,
                                                       InvertOrder(ordering))
                    else:
                        logits = self.model.do_forward(inp, ordering)
                else:
                    assert False, "missing do forward method"
                    if self.traced_fwd is not None:
                        logits = self.traced_fwd(inp)
                    else:
                        # logits = self.model(inp)
                        # logits = self.model.forward_with_encoded_input(
                        #     inp.half())
                        logits = self.model.forward_with_encoded_input(inp)

        # Doing this convoluted scheme because m_l[0] is a scalar, and
        # we want the corret shape to broadcast.
        p = masked_logits[1]
        for ls in masked_logits[2:]:
            p *= ls
        p *= masked_logits[0]

        return p.mean().item()

    # @torch.jit.script_method
    # @torch.jit.script
    def Query(self, columns, operators, vals):
        # Massages queries into natural order.
        columns, operators, vals = FillInUnqueriedColumns(
            self.table, columns, operators, vals)

        # TODO: we can move these attributes to ctor.
        ordering = None
        if hasattr(self.model, 'orderings'):
            # RelNet.
            # assert len(self.model.orderings) == 1
            ordering = self.model.orderings[0]
            orderings = self.model.orderings
        elif hasattr(self.model, 'm'):
            # MADE.
            # assert self.model.num_masks == 1
            ordering = self.model.m[-1]
            orderings = [self.model.m[-1]]
        else:
            print('****Warning: defaulting to natural order')
            ordering = np.arange(len(columns))
            orderings = [ordering]
        num_orderings = len(orderings)

        with torch.no_grad():
            inp_buf = self.inp.zero_()
            # Fast (?) path.
            if num_orderings == 1:
                ordering = orderings[0]
                if self.workers > 1:
                    args = [(self.num_samples // self.workers, ordering,
                             columns, operators, vals, i)
                            for i in range(self.workers)]
                    pool = self.pool
                    # print(args)
                    self.OnStart()
                    # Blocks.
                    # mp.spawn(self._sample_n, args=args, nprocs=2)
                    ps = pool.starmap(self._sample_n, args, chunksize=1)
                    # pool.join()
                    # print(ps)
                    self.OnEnd()
                    # pool.close()
                    # pool.terminate()
                    return np.ceil(np.mean(ps) * self.cardinality).astype(
                        dtype=np.int32, copy=False)

                else:
                    # print('inp_buf', inp_buf)
                    self.OnStart()
                    p = self._sample_n(
                        self.num_samples,
                        # MADE's 'orderings', 'm[-1]' are in an inverted space
                        # --- opposite semantics of what _sample_n() and what
                        # Transformer's ordering expect.
                        ordering if isinstance(self.model, Transformer) else
                        InvertOrder(ordering),
                        columns,
                        operators,
                        vals,
                        inp=inp_buf)
                    self.OnEnd()
                    # print('density={}'.format(p))
                    # print('scaling with', self.cardinality)
                    return np.ceil(p * self.cardinality).astype(dtype=np.int32,
                                                                copy=False)

            if hasattr(self.model, 'use_query_order'):
                remaining = list(orderings[0])
                order = []

                for i, op in enumerate(operators):
                    if op is not None:
                        order.append(i)
                        remaining.remove(i)
                order.extend(list(remaining))
                print("Query order", order)
                orderings = [np.array(order)]
            elif hasattr(self.model, 'use_best_order'):
                rank = []
                for order in orderings:
                    order = list(order)
                    distance_sum = 0
                    for i, op in enumerate(operators):
                        if op is not None:
                            # order = [3, 1, 7, ...]
                            # means that variable 0 is @ pos 3, etc.
                            distance_sum += order[i]
                    rank.append((distance_sum, order))
                rank.sort()
                best = rank[0][-1]
                print("Best order", best)
                orderings = [np.array(best)]
            elif hasattr(self.model, 'use_worst_order'):
                rank = []
                for order in orderings:
                    order = list(order)
                    distance_sum = 0
                    for i, op in enumerate(operators):
                        if op is not None:
                            distance_sum += order.index(i)
                    rank.append((-distance_sum, order))
                rank.sort()
                best = rank[0][-1]
                print("Worst order", best)
                orderings = [np.array(best)]
            num_orderings = len(orderings)

            # Num orderings > 1.
            ps = []
            self.OnStart()
            for ordering in orderings:
                ordering = ordering if isinstance(
                    self.model, Transformer) else InvertOrder(ordering)
                ns = self.num_samples // num_orderings
                if ns < 1:
                    print("WARNING: rounding up to 1", self.num_samples, num_orderings)
                    ns = 1
                # print('ordering for _sample_n()', ordering)
                p_scalar = self._sample_n(ns,
                                          ordering,
                                          columns,
                                          operators,
                                          vals,
                                          worker_id=0)
                ps.append(p_scalar)
            self.OnEnd()
            return np.ceil(np.mean(ps) * self.cardinality).astype(
                dtype=np.int32, copy=False)


class Heuristic(CardEst):
    """Uses independence assumption."""

    def __init__(self, table):
        super(Heuristic, self).__init__()
        self.table = table
        self.size = self.table.cardinality

    def __str__(self):
        return 'heuristic'

    def LessThanQuery(self, age_lt, salary_lt):
        """Special case for less than query with just 2 columns"""
        self.OnStart()
        age = self.table.Columns()[0].data
        salary = self.table.Columns()[1].data

        age_sel = (age < age_lt).sum() / self.size
        sal_sel = (salary < salary_lt).sum() / self.size
        sel = age_sel * sal_sel

        self.OnEnd()
        return np.ceil(sel * self.size).astype(np.int32)

    def Query(self, columns, operators, vals):
        self.OnStart()

        sels = [
            OPS[o](c.data if isinstance(c.data, np.ndarray) else c.data.values,
                   v).sum() / self.size
            for c, o, v in zip(columns, operators, vals)
        ]
        sel = np.prod(sels)

        self.OnEnd()
        return np.ceil(sel * self.size).astype(np.int32)


class Oracle(CardEst):
    """Returns true cardinalities."""

    def __init__(self, table, limit_first_n=None, cache_dir=None):
        super(Oracle, self).__init__()
        self.table = table
        self.limit_first_n = limit_first_n
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

    def __str__(self):
        return 'oracle'

    def Query(self, columns, operators, vals, return_masks=False):
        assert len(columns) == len(operators) == len(vals)
        self.OnStart()

        key = ""
        if self.cache_dir:
            for c, o, v in zip(columns, operators, vals):
                key += "{}{}{},".format(c.name, o, str(v))
            key = hashlib.sha256(key.encode("utf-8")).hexdigest()
            key = os.path.join(self.cache_dir, key)
            if os.path.exists(key):
                try:
                    return int(open(key).read())
                except:
                    print("FAILED TO READ CACHED VALUE", key)

        bools = None
        for c, o, v in zip(columns, operators, vals):
            if self.limit_first_n is None:
                inds = OPS[o](c.data, v)
            else:
                # For data shifts experiment.
                inds = OPS[o](c.data[:self.limit_first_n], v)

            if bools is None:
                bools = inds
            else:
                bools &= inds
        c = bools.sum()
        self.OnEnd()
        if return_masks:
            return bools

        if key:
            with open(key, "w") as f:
                f.write(str(c))
        return c


class Sampling(CardEst):
    """Keep p% of samples in memory."""

    def __init__(self, table, p):
        super(Sampling, self).__init__()
        self.table = table

        self.p = p
        self.num_samples = int(p * table.cardinality)
        self.size = table.cardinality

        # TODO: add seed for repro.
        self.tuples = table.data.sample(n=self.num_samples)

        self.sample_durs_ms = []

    def __str__(self):
        if self.p * 100 != int(self.p * 100):
            return 'sample_{:.1f}%'.format(self.p * 100)
        return 'sample_{}%'.format(int(self.p * 100))

    def Query(self, columns, operators, vals):
        assert len(columns) == len(operators) == len(vals)
        self.OnStart()

        qualifying_tuples = []
        for col, op, val in zip(columns, operators, vals):
            qualifying_tuples.append(OPS[op](self.tuples[col.name], val))
        s = np.all(qualifying_tuples, axis=0).sum()
        sel = s * 1.0 / self.num_samples

        self.OnEnd()
        # print('\nsample: {} matches {:.4f}%'.format(int(s), float(sel) * 100))
        return np.ceil(sel * self.table.cardinality).astype(dtype=np.int32)
