import argparse
import collections
import glob
import os
import pickle
import re

import datasets
import torch
from common import *
from estimators import *
from made import MADE

# For inference speed.
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

parser = argparse.ArgumentParser()

parser.add_argument("--inference-opts", action='store_true', help="Trace?")
parser.add_argument(
    "--use-query-order",
    action='store_true',
    help="Whether to use the variable order from the query in eval.")
parser.add_argument(
    "--use-best-order",
    action='store_true',
    help="Whether to use best available multi-order for the query in eval.")
parser.add_argument(
    "--use-worst-order",
    action='store_true',
    help="Whether to use worst available multi-order for the query in eval.")
parser.add_argument("--early", action='store_true', help="Early stop?")
parser.add_argument("--print-unk",
                    action='store_true',
                    help="Print UNK embeddings?")
parser.add_argument("--special-orders",
                    type=int,
                    default=-1,
                    help="Number of special orderings to use.  Disabled if -1.")

parser.add_argument("--num-queries", type=int, default=2000, help="# queries.")
parser.add_argument("--dataset", type=str, default='dmv', help="Dataset.")
parser.add_argument("--err-csv",
                    type=str,
                    default="out.csv",
                    help="Save to what path?")
parser.add_argument("--glob",
                    type=str,
                    default="models/*",
                    help="Ckpts to glob under models/.")
parser.add_argument("--psample", type=int, default=4000, help="#psamples.")
parser.add_argument("--order",
                    nargs="+",
                    type=int,
                    required=False,
                    help="Use a specific order?")

parser.add_argument("--blacklist", type=str, help="Remove some glob'd files.")

# MADE.
parser.add_argument("--fc-hiddens",
                    type=int,
                    default=128,
                    help="# Units in FC.")
parser.add_argument("--layers", type=int, default=0, help="# layers in FC.")
parser.add_argument("--residual", action='store_true', help="ResMade?")
parser.add_argument("--direct-io", action='store_true', help="Do direct IO?")

parser.add_argument("--dropout", action='store_true', help="Dropout?")
parser.add_argument("--no-emb-opt",
                    action='store_true',
                    help="No embedding optimization?")
parser.add_argument("--special-dmv-arch",
                    action='store_true',
                    help="For testing...")
parser.add_argument("--inv-order",
                    action='store_true',
                    help="MADE needs inverse order...")
parser.add_argument("--embs-tied",
                    action='store_true',
                    help="tie in/out embeddings?")

# Transformer.
parser.add_argument("--blocks",
                    type=int,
                    default=0,
                    help="Transformer: num blocks.")
parser.add_argument("--dmodel",
                    type=int,
                    default=0,
                    help="Transformer: d_model.")
parser.add_argument("--dff", type=int, default=0, help="Transformer: d_ff.")
parser.add_argument("--heads",
                    type=int,
                    default=0,
                    help="Transformer: num heads.")
parser.add_argument("--transformer-act",
                    type=str,
                    default='gelu',
                    help="Transformer activation.")
parser.add_argument("--pos-emb",
                    action='store_true',
                    help="Use positional embs?")
parser.add_argument("--first-query-shared",
                    action='store_true',
                    help="First query vec shared?")


def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the "true" ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering


def ErrorMetric(est_card, card):
    if card == 0 and est_card != 0:
        return est_card
    if card != 0 and est_card == 0:
        return card
    if card == 0 and est_card == 0:
        return 1.0
    return max(est_card / card, card / est_card)


def SampleTupleThenRandom(dataset,
                          all_cols,
                          num_filters,
                          rng,
                          table,
                          return_col_idx=False,
                          force_query_cols=None):
    s = table.data.iloc[rng.randint(0, len(table.data))]
    vals = s.values

    if dataset == 'dmv':
        # Giant hack for DMV.
        vals[6] = vals[6].to_datetime64()
    elif dataset == 'dmv-full':
        vals[13] = vals[13].to_datetime64()
        vals[14] = vals[14].to_datetime64()

    if force_query_cols:
        idxs = force_query_cols
        num_filters = len(idxs)
        print("Forcing columns to query", idxs)
    else:
        idxs = rng.choice(len(all_cols), replace=False, size=num_filters)
    cols = np.take(all_cols, idxs)

    # If dom size >= 10, okay to place a range filter.
    # Otherwise, low domain size columns should be queried with equality.
    ops = rng.choice(['<=', '>=', '='], size=num_filters)
    #     ops = rng.choice(['<=', '>=',], size=num_filters)
    ops_all_eqs = ['='] * num_filters
    sensible_to_do_range = [c.DistributionSize() >= 10 for c in cols]
    #     sensible_to_do_range = [c.DistributionSize() >= 1000000000 for c in cols]
    ops = np.where(sensible_to_do_range, ops, ops_all_eqs)

    if num_filters == len(all_cols):
        if return_col_idx:
            return np.arange(len(all_cols)), ops, vals
        return all_cols, ops, vals

    vals = vals[idxs]
    if return_col_idx:
        return idxs, ops, vals

    return cols, ops, vals


def SampleFromDomains(all_cols, num_filters, rng, table, return_col_idx=False):
    idxs = rng.choice(len(all_cols), replace=False, size=num_filters)
    cols = np.take(all_cols, idxs)

    # If dom size >= 10, okay to place a range filter.
    # Otherwise, low domain size columns should be queried with equality.
    ops = rng.choice(['<=', '>=', '='], size=num_filters)
    #     ops = rng.choice(['<=', '>=',], size=num_filters)
    ops_all_eqs = ['='] * num_filters
    sensible_to_do_range = [c.DistributionSize() >= 10 for c in cols]
    #     sensible_to_do_range = [c.DistributionSize() >= 1000000000 for c in cols]
    ops = np.where(sensible_to_do_range, ops, ops_all_eqs)

    vals = []
    for c in cols:
        l = 0
        if pd.isnull(c.all_distinct_values[0]):
            l = 1
        vals.append(c.all_distinct_values[rng.randint(l, c.distribution_size)])

    if return_col_idx:
        return idxs, ops, vals

    return cols, ops, vals


def GenerateQuery(dataset, all_cols, rng, table, return_col_idx=False, query_filters=None, force_query_cols=None):

    ### Generate a random query.
    if query_filters:
        num_filters = min(len(all_cols),
                          rng.randint(query_filters[0], query_filters[1]))
    elif dataset in 'dmv':
        num_filters = rng.randint(len(all_cols) // 2, len(all_cols) + 1)
    elif dataset == 'synthetic':
        num_filters = rng.randint(2, 3)
    else:
        num_filters = min(len(all_cols), rng.randint(5, 12))

    if hasattr(table, 'data'):
        cols, ops, vals = SampleTupleThenRandom(dataset,
                                                all_cols,
                                                num_filters,
                                                rng,
                                                table,
                                                return_col_idx=return_col_idx,
                                                force_query_cols=force_query_cols)
    else:
        assert not force_query_cols, "not implemented"
        cols, ops, vals = SampleFromDomains(all_cols,
                                            num_filters,
                                            rng,
                                            table,
                                            return_col_idx=return_col_idx)

    # print('generated query', cols, ops, vals)
    # assert False, type(cols)
    return cols, ops, vals


def Query(estimators,
          do_print=True,
          oracle_card=None,
          query=None,
          table=None,
          oracle_est=None):
    assert query is not None
    cols, ops, vals = query

    ### Actually estimate the query.

    def pprint(*args, **kwargs):
        if do_print:
            print(*args, **kwargs)

    # Actual.
    s = time.time()
    card = oracle_est.Query(cols, ops,
                            vals) if oracle_card is None else oracle_card
    d = time.time() - s
    pprint('{:.3}s oracle_est.Query()'.format(d))
    if card == 0:
        return

    pprint('Q(', end='')
    for c, o, v in zip(cols, ops, vals):
        pprint('{} {} {}, '.format(c.name, o, str(v)), end='')
    pprint('): ', end='')

    pprint('\n  actual {} ({:.3f}%) '.format(card,
                                             card / table.cardinality * 100),
           end='')

    for est in estimators:
        # print(cols, ops, vals)
        est_card = est.Query(cols, ops, vals)
        err = ErrorMetric(est_card, card)
        est.AddError(err, est_card, card)
        pprint('{} {} (err={:.3f}) '.format(str(est), est_card, err), end='')
    pprint()


def ReportEsts(estimators):
    v = -1
    for est in estimators:
        # print("Estimator error", est, "mean", np.mean(est.errs), "max",
        #       np.max(est.errs), "95th", np.quantile(est.errs, 0.95),
        #       "90th", np.quantile(est.errs, 0.9), "median",
        #       np.quantile(est.errs, 0.5))
        print(str(est), "max", np.max(est.errs), "99th",
              np.quantile(est.errs, 0.99), "95th", np.quantile(est.errs, 0.95),
              "median", np.quantile(est.errs, 0.5))
        v = max(v, np.max(est.errs))
    return v


def RunN(table,
         cols,
         estimators,
         rng=None,
         num=20,
         log_every=50,
         num_filters=11,
         oracle_cards=None,
         oracle_est=None):
    if rng is None:
        rng = np.random.RandomState(1234)

    last_time = None
    for i in range(num):
        do_print = False
        if i % log_every == 0:
            if last_time is not None:
                print('{:.1f} queries/sec'.format(log_every /
                                                  (time.time() - last_time)))
            do_print = True
            print('Query {}:'.format(i), end=' ')
            last_time = time.time()
        query = GenerateQuery(args.dataset, cols, rng, table)
        Query(estimators,
              do_print,
              oracle_card=oracle_cards[i] if oracle_cards is not None else None,
              query=query,
              table=table,
              oracle_est=oracle_est)

        max_err = ReportEsts(estimators)
        if args.early and max_err > 370:
            return True

    return False


# TODO: unify MakeModel from eval/train files into one.
def MakeModel(scale, cols_to_train, seed, fixed_ordering=None):

    if args.inv_order:
        print('Inverting order!!!!!!!!!!')
        fixed_ordering = InvertOrder(fixed_ordering)

    return MADE(
        nin=len(cols_to_train),
        hidden_sizes=[
            scale,
        ] * 4,
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        input_encoding="embed",
        output_encoding="embed",
        embed_size=64,
        # input_no_emb_if_leq=False,
        embs_tied=args.embs_tied,
        input_no_emb_if_leq=True,
        seed=seed,
        natural_ordering=False if seed is not None else True,
        residual_connections=args.residual,
        fixed_ordering=fixed_ordering,
        do_direct_io_connections=args.direct_io,
        dropout_p=args.dropout,
    ).to(DEVICE)


def MakeMadeDmv(cols_to_train, seed, fixed_ordering=None):

    if args.inv_order:
        print('Inverting order!!!!!!!!!!')
        fixed_ordering = InvertOrder(fixed_ordering)

    if args.special_dmv_arch:
        return MADE(
            nin=len(cols_to_train),
            hidden_sizes=[256] * 5,
            nout=sum([c.DistributionSize() for c in cols_to_train]),
            input_bins=[c.DistributionSize() for c in cols_to_train],
            input_encoding="embed",
            output_encoding="embed",
            embed_size=128,
            input_no_emb_if_leq=True,
            embs_tied=True,
            seed=seed,
            do_direct_io_connections=True,  #args.direct_io,
            natural_ordering=False if seed is not None else True,
            residual_connections=args.residual,
            fixed_ordering=fixed_ordering,
            dropout_p=args.dropout,
        ).to(DEVICE)

    hiddens = [args.fc_hiddens] * args.layers
    natural_ordering = False

    if args.layers == 0:
        # Default ckpt.
        hiddens = [512, 256, 512, 128, 1024]
        natural_ordering = True

    model = MADE(
        nin=len(cols_to_train),
        hidden_sizes=hiddens,
        residual_connections=args.residual,
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        input_encoding="embed"
        if args.dataset in ["dmv-full", "kdd", "synthetic"] else "binary",
        output_encoding="embed"
        if args.dataset in ["dmv-full", "kdd", "synthetic"] else "one_hot",
        seed=seed,
        do_direct_io_connections=args.direct_io,
        natural_ordering=False if seed is not None else True,
        fixed_ordering=fixed_ordering,
        dropout_p=args.dropout,
        num_masks=max(1, args.special_orders),
    ).to(DEVICE)

    # XXX this is copied from train_many_orderings
    if args.special_orders > 0:
        special_orders = [
            # # MutInfo Max Marg
            # np.array([6, 1, 4, 0, 7, 3, 5, 2, 10, 9, 8]),
            # # CL Max Marg/Dom
            # np.array([6, 1, 4, 0, 5, 7, 3, 2, 10, 9, 8]),
            # # Random
            # np.random.RandomState(0).permutation(np.arange(11)),
        ][:args.special_orders]
        k = len(special_orders)
        for i in range(k, args.special_orders):
            special_orders.append(
                np.random.RandomState(i - k + 1).permutation(
                    np.arange(len(cols_to_train))))
        print('Special orders', np.array(special_orders))

        if args.inv_order:
            for i, order in enumerate(special_orders):
                special_orders[i] = np.asarray(InvertOrder(order))
            print('Inverted special orders:', special_orders)

        model.orderings = special_orders

    if args.use_query_order:
        model.use_query_order = True

    if args.use_best_order:
        model.use_best_order = True

    if args.use_worst_order:
        model.use_worst_order = True

    return model


def MakeTransformer(cols_to_train, fixed_ordering, seed=None):
    return Transformer(
        num_blocks=args.blocks,
        d_model=args.dmodel,
        d_ff=args.dff,
        num_heads=args.heads,
        nin=len(cols_to_train),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        use_positional_embs=args.pos_emb,
        activation=args.transformer_act,
        fixed_ordering=fixed_ordering,
        dropout=args.dropout,
        seed=seed,
        first_query_shared=args.first_query_shared,
    ).to(DEVICE)


def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        #         print (p)
        #         assert 'embedding' not in name, name
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print("number of model parameters: {} (~= {:.1f}MB)".format(num_params, mb))
    #     for name, param in model.named_parameters():
    #         print(name, ':', np.prod(param.size()))
    print(model)
    return mb


def SaveEstimators(path, estimators, return_df=False):
    # name, query_dur_ms, errs, est_cards, true_cards
    results = pd.DataFrame()
    for est in estimators:
        data = {
            'est': [est.name] * len(est.errs),
            # 'est': [str(est)] * len(est.errs),
            'err': est.errs,
            'est_card': est.est_cards,
            'true_card': est.true_cards,
            'query_dur_ms': est.query_dur_ms,
        }
        results = results.append(pd.DataFrame(data))
    if return_df:
        return results
    results.to_csv(path, index=False)
