#!/usr/bin/env python

import argparse
import copy
import math
import pickle
import time
from collections import namedtuple

import os
import numpy as np
import pandas as pd
import random

import ray
from ray import tune
from eval_model import Query, GenerateQuery, ReportEsts
import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_infer import TrainedModel, infer_naive, infer_skip, q_error
from common import Column, CsvTable, Table, TableDataset
from estimators import *
from made import MADE, MaskedLinear
from torch.utils.data import DataLoader, Dataset
from transformer import Transformer

# Pass SILENT=1 to make query evaluation less verbose.
SILENT = "SILENT" in os.environ

parser = argparse.ArgumentParser()

parser.add_argument(
    "--run",
    nargs="+",
    default=["test_simple", "test_url"],
    type=str,
    required=False,
    help="List of experiments to run")
args = parser.parse_args()


def gen_dryad_query_set():
    print("Generating query set")
    rng = np.random.RandomState(0)
    lines = open("datasets/article-urls.trim").readlines()
    data = "".join(lines)
    queries = []
    likelihoods = []
    for i in range(100):
        pos = rng.randint(0, len(data) - 10)
        k = rng.choice([2, 3, 4, 5])
        token = data[pos:pos + k]
        queries.append(token)
#        likelihood = data.count(token)
#        print(i, token, likelihood)
    print(queries)
    return queries


# Common config. Each key is auto set as an attribute (i.e. NaruTrainer.<attr>)
# so try to avoid any name conflicts with members of that class.
BASE_CONFIG = {
    "cwd": os.getcwd(),
    "epochs_per_iteration": 1,
    "num_eval_queries_per_iteration": 100,
    "num_eval_queries_at_end": 1000,
    "epochs": 10,
    "seed": None,
    "order_seed": None,
    "bs": 2048,
    "order": None,
    "layers": 2,
    "fc_hiddens": 128,
    "warmups": 1000,
    "residual": True,
    "direct_io": True,
    "query_filters": [5, 12],
    "force_query_cols": None,
    "embs_tied": False,
    "embed_size": 32,
    "input_no_emb_if_leq": True,

    # If set, load this checkpoint and run eval immediately. No training.
    "checkpoint_to_load": None,

    # Dropout for wildcard skipping.
    "disable_learnable_unk": False,
    "per_row_dropout": True,
    "dropout": 0,
    "fixed_dropout_ratio": False,
    "asserts": None,
    "special_orders": 0,
    "special_order_seed": 0,
    "shuffle_at_data_level": False,

    # Eval.
    "eval_heuristic": True,
    "eval_psamples": [100, 1000, 10000],

    # Text modeling options.
    "use_transformer": False,
    "prefix_dropout": False,
    "transformer_args": {},
    "compute_test_loss": False,
    "text_eval_corpus": [],
    "text_eval_fraction": 1,

    # TODO do the below options actually work?
    "entropy_order": False,
    "reverse_entropy": False,
    "num_orderings": 1,
}

EXPERIMENT_CONFIGS = {
    ### TEST CONFIGS ###
    # These are run by default if you don't specify --run.
    "test_simple": dict(
        BASE_CONFIG, **{
            "dataset": "census",
            "order_seed": None,
            "epochs": 50,
            "epochs_per_iteration": 10,
            "num_eval_queries_per_iteration": 2,
            "num_eval_queries_at_end": 20,
            "special_orders": 10,  # <-- comment out to disable MO
            "fc_hiddens": 256,   # <-- 256 vs 180
            "layers": 4,
            "bs": 128,
        }),
    "test_url": dict(
        BASE_CONFIG, **{
            "dataset": "url-tiny",
            "order_seed": None,
            "use_transformer": True,
            "prefix_dropout": True,
            "per_row_dropout": False,
            "compute_test_loss": True,
            "layers": 4,
            "fc_hiddens": 256,
            "epochs": 1000,
            "epochs_per_iteration": 100,
            "num_eval_queries_per_iteration": 0,
            "num_eval_queries_at_end": 0,
            "bs": 128,
            "text_eval_fraction": 0.1,
            "eval_psamples": [100, 1000],
            "transformer_args": {
                "num_blocks": 4,
                "d_model": 16,
                "d_ff": 64,
                "num_heads": 4,
            },
            "text_eval_corpus": [
                "hoo",
            ],
        }),

    # dataset from https://datadryad.org/stash/dataset/doi:10.5061/dryad.p8s0j
    # postprocessed via awk '{print $2}' to strip the line numbers
    "dryad": dict(
        BASE_CONFIG,
        **{
            "dataset": "dryad-urls",
            "order_seed": None,
            "use_transformer": True,
            "prefix_dropout": True,
            "compute_test_loss": True,
            "bs": 512,
            "epochs": 20,
            "epochs_per_iterations": 20,
            "layers": 4,
            "eval_psamples": [100, 1000],
            "fc_hiddens": 256,
            "transformer_args": {
                "num_blocks": 8,
                "d_model": 32,
                "d_ff": 256,
                "num_heads": 4,
            },
            "embed_size": 4,
            "num_eval_queries_per_iteration": 0,
            "num_eval_queries_at_end": 0,
            "text_eval_corpus": gen_dryad_query_set,
            "text_eval_fraction": 1,
        }),

    ### EXPERIMENT CONFIGS ###
    # Run multiple experiments concurrently by using the --run flag, ex:
    # $ ./train.py --run kdd census
    "kdd": dict(
        BASE_CONFIG, **{
            "dataset": tune.grid_search(["kdd"]),
            "order_seed": tune.grid_search([None]),
            "epochs": 200,
            "epochs_per_iteration": 50,
            "warmups": 1000,
            "layers": 4,
            "fc_hiddens": 256,
            "per_row_dropout": True,
            "input_no_emb_if_leq": False,
        }),
    "census": dict(
        BASE_CONFIG, **{
            "dataset": tune.grid_search(["census"]),
            "order_seed": tune.grid_search([None]),
            "epochs": 20,
            "epochs_per_iteration": 5,
            "warmups": 2000,
            "layers": 4,
            "fc_hiddens": 256,
            "per_row_dropout": True,
            "input_no_emb_if_leq": False,
        }),
    "dmv-full": dict(
        BASE_CONFIG, **{
            "dataset": tune.grid_search(["dmv-full"]),
            "order_seed": tune.grid_search([None]),
            "warmups": 6000,
            "epochs": 20,
            "epochs_per_iteration": 5,
            "layers": 4,
            "fc_hiddens": 256,
            "per_row_dropout": True,
            "input_no_emb_if_leq": False,
        }),
}

EXPERIMENT_CONFIGS["dryad-small"] = dict(
    EXPERIMENT_CONFIGS["dryad"],
    **{
        "dataset": "dryad-urls-small",
        "prefix_dropout": True,
        "embed_size": 8,
        "bs": 512,
        "warmups": 100,
        "epochs": 1000,
        "epochs_per_iteration": 5,
        "text_eval_corpus": [
            ".com",  # 1.8m
            #        "x",  # 591742
            #        "rea",  # 150133
            "bbc",  # 21000
            #        "zz",  # 9241
            "query",  # 58
        ],
        "eval_psamples": [100, 1000],
    })

for key in ["kdd", "dmv-full", "census"]:
    config = EXPERIMENT_CONFIGS[key]
    # Ablation study for different architectures.
    EXPERIMENT_CONFIGS[key + "-arch"] = dict(
        config, **{
            "order_seed": None,
            "layers": tune.grid_search([2, 4, 6]),
            "fc_hiddens": tune.grid_search([64, 128, 512]),
        })
    # See if disabling embed learning matters
    EXPERIMENT_CONFIGS[key + "-nolearnunk"] = dict(
        config, **{
            "disable_learnable_unk": True,
        })
    # See if disabling non embed
    EXPERIMENT_CONFIGS[key + "-forceembed"] = dict(
        config, **{
            "input_no_emb_if_leq": False,
        })
    # FINAL icml
    EXPERIMENT_CONFIGS[key + "-final"] = dict(
        config, **{
            "per_row_dropout": tune.grid_search([False, 2]),
            "num_eval_queries_per_iteration": 0,
            "num_eval_queries_at_end": 1000,
            "order_seed": tune.grid_search([0, 1, 2, 3, 4, 5, 6, 7]),
        })
    # FINAL icml mo
    EXPERIMENT_CONFIGS[key + "-final-mo"] = dict(
        config, **{
            "per_row_dropout": tune.grid_search([False, 2]),
            "num_eval_queries_per_iteration": 0,
            "num_eval_queries_at_end": 1000,
            "special_orders": 10,
            "special_order_seed": tune.grid_search([0, 1, 2, 3, 4, 5, 6, 7]),
            "order_seed": None,
        })


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# Training.

# For multi-order experiments, we want to have all randomly sampled orders.
_SPECIAL_ORDERS = {
    'dmv': [],
    'dmv-full': [],
    'census': [],
    'kdd': [],
}


def Entropy(name, data, bases=None):
    import scipy.stats
    s = 'Entropy of {}:'.format(name)
    ret = []
    for base in bases:
        assert base == 2 or base == 'e' or base is None
        e = scipy.stats.entropy(data, base=base if base != 'e' else None)
        ret.append(e)
        unit = 'nats' if (base == 'e' or base is None) else 'bits'
        s += ' {:.4f} {}'.format(e, unit)
    print(s)
    return ret


def run_epoch(split,
              model,
              opt,
              train_data,
              val_data=None,
              batch_size=100,
              upto=None,
              epoch_num=None,
              verbose=False,
              log_every=10,
              return_losses=False,
              child=None,
              table_bits=None,
              warmups=1000):
    torch.set_grad_enabled(split == 'train')
    model.train() if split == 'train' else model.eval()
    if child:
        child.train() if split == 'train' else child.eval()
    dataset = train_data if split == 'train' else val_data
    losses = []

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == 'train'))

    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, 'orderings'):
        nsamples = len(model.orderings)
        if not SILENT:
            print('setting nsamples to', nsamples)

    for step, xb in enumerate(loader):
        if split == 'train':
            base_lr = 8e-4
            for param_group in opt.param_groups:
                t = warmups
                d_model = model.embed_size
                global_steps = len(loader) * epoch_num + step + 1
                lr = (d_model**-0.5) * min(
                    (global_steps**-.5), global_steps * (t**-1.5))

                # lr = 5e-4
                param_group['lr'] = lr

        if upto and step >= upto:
            break

        xb = xb.to(get_device()).to(torch.float32)

        # Forward pass, potentially through several orderings.
        xbhat = None
        model_logits = []
        num_orders_to_forward = 1
        if split == 'test' and nsamples > 1:
            # At test, we want to test the "true" nll under all orderings.
            num_orders_to_forward = nsamples

        for i in range(num_orders_to_forward):
            if hasattr(model, 'update_masks'):
                # We want to update_masks even for first ever batch.
                model.update_masks()

            model_out = model(xb)
            model_logits.append(model_out)
            if xbhat is None:
                xbhat = torch.zeros_like(model_out)
            xbhat += model_out

        if xbhat.shape == xb.shape:
            if mean:
                xb = (xb * std) + mean
            loss = F.binary_cross_entropy_with_logits(
                xbhat, xb, size_average=False) / xbhat.size()[0]
        else:
            if model.input_bins is None:
                # NOTE: we have to view() it in this order due to the mask
                # construction within MADE.  The masks there on the output unit
                # determine which unit sees what input vars.
                xbhat = xbhat.view(-1, model.nout // model.nin, model.nin)
                # Equivalent to:
                loss = F.cross_entropy(xbhat, xb.long(), reduction='none') \
                        .sum(-1).mean()
                # NOTE: do NOT use reduction='mean' (default behavior)!
                # loss = F.cross_entropy(xbhat, xb.long(), reduction='sum') / xbhat.size()[0]
            else:
                if num_orders_to_forward == 1:
                    loss = model.nll(xbhat, xb).mean()
                    if child:
                        # Distillation loss
                        child_loss = model.kl_div(model_out.detach(), child,
                                                  child_out)
                        child_loss = child_loss.mean()
                        child_ref_loss = child.nll(child_out, xb).mean()
                else:
                    # Average across orderings & then across minibatch.
                    #
                    #   p(x) = 1/N sum_i p_i(x)
                    #   log(p(x)) = log(1/N) + log(sum_i p_i(x))
                    #             = log(1/N) + logsumexp ( log p_i(x) )
                    #             = log(1/N) + logsumexp ( - nll_i (x) )
                    #
                    # Used only at test time.
                    logps = []  # [batch size, num orders]
                    assert len(model_logits) == num_orders_to_forward, len(
                        model_logits)
                    for logits in model_logits:
                        # Note the minus.
                        logps.append(-model.nll(logits, xb))
                    logps = torch.stack(logps, dim=1)
                    logps = logps.logsumexp(dim=1) + torch.log(
                        torch.tensor(1.0 / nsamples, device=logps.device))
                    loss = (-logps).mean()

        losses.append(loss.item())

        if step % log_every == 0 and not SILENT:
            if split == 'train':
                print(
                    'Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr'
                    .format(epoch_num, step, split,
                            loss.item() / np.log(2) - table_bits,
                            loss.item() / np.log(2), table_bits, lr))
                if child:
                    print(
                        'Epoch {} Iter {}, {} child entropy gap {:.4f} bits {:.5f} lr'
                        .format(epoch_num, step, split,
                                child_ref_loss.item() / np.log(2) - table_bits,
                                lr))
                    print('Distillation loss {}'.format(child_loss.item()))
            else:
                print('Epoch {} Iter {}, {} loss {:.4f} nats / {:.4f} bits'.
                      format(epoch_num, step, split, loss.item(),
                             loss.item() / np.log(2)))

        if split == 'train':
            opt.zero_grad()
            loss.backward()
            if child:
                child_loss.backward()
            opt.step()

        if verbose:
            print("%s epoch average loss: %f" % (split, np.mean(losses)))
    if return_losses:
        return losses
    return np.mean(losses)


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


def MakeMade(scale,
             cols_to_train,
             seed,
             dataset,
             fixed_ordering=None,
             special_orders=[],
             layers=4,
             residual=False,
             dropout=False,
             per_row_dropout=False,
             prefix_dropout=False,
             fixed_dropout_ratio=False,
             disable_learnable_unk=False,
             input_no_emb_if_leq=True,
             embs_tied=False,
             embed_size=32):
    # TODO: if passed in a single heuristic order, be sure to InvertOrder().
    num_masks = 1
    if len(special_orders):
        num_masks = len(special_orders)
    model = MADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] * layers
        if layers > 0 else [512, 256, 512, 128, 1024],
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        input_encoding="embed",
        output_encoding="embed",
        seed=seed,
        do_direct_io_connections=False,
        natural_ordering=False if seed is not None else True,
        residual_connections=residual,
        embed_size=embed_size,
        fixed_ordering=fixed_ordering,
        dropout_p=dropout or per_row_dropout or prefix_dropout,
        fixed_dropout_p=fixed_dropout_ratio,
        num_masks=num_masks,
        per_row_dropout_p=per_row_dropout,
        prefix_dropout=prefix_dropout,
        disable_learnable_unk=disable_learnable_unk,
        input_no_emb_if_leq=input_no_emb_if_leq,
        embs_tied=embs_tied,
    ).to(get_device())

    if len(special_orders):
        print('assigning to model.orderings:')
        print(special_orders)
        model.orderings = special_orders

    return model


def weight_init(m):
    if type(m) == MaskedLinear or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if type(m) == nn.Embedding:
        nn.init.normal_(m.weight, std=0.02)


class NaruTrainer(tune.Trainable):

    def _setup(self, config):
        print('NaruTrainer config:', config)
        os.chdir(config["cwd"])
        for k, v in config.items():
            setattr(self, k, v)
        self.epoch = 0

        if callable(self.text_eval_corpus):
            self.text_eval_corpus = self.text_eval_corpus()

        # Try to make all the runs the same, except for input orderings.
        torch.manual_seed(0)
        np.random.seed(0)

        assert self.dataset in [
            'dmv', 'dmv-full', 'census',
            'synthetic', 'kdd', 'kdd-full', 'url', 'url-tiny', 'dryad-urls',
            'dryad-urls-small'
        ]
        if self.shuffle_at_data_level:
            data_order_seed = self.order_seed
        else:
            data_order_seed = None
        if self.dataset == 'dmv-full':
            table = datasets.LoadDmv(full=True, order_seed=data_order_seed)
        elif self.dataset == 'dmv':
            table = datasets.LoadDmv(order_seed=data_order_seed)
        elif self.dataset == 'synthetic':
            table = datasets.LoadSynthetic(order_seed=data_order_seed)
        elif self.dataset == 'census':
            table = datasets.LoadCensus(order_seed=data_order_seed)
        elif self.dataset == 'kdd':
            table = datasets.LoadKDD(order_seed=data_order_seed)
        elif self.dataset == 'kdd-full':
            table = datasets.LoadKDD(full=True, order_seed=data_order_seed)
        elif self.dataset == 'url-tiny':
            table = datasets.LoadURLTiny()
        elif self.dataset == 'dryad-urls':
            table = datasets.LoadDryadURLs()
        elif self.dataset == 'dryad-urls-small':
            table = datasets.LoadDryadURLs(small=True)
        self.table = table
        self.oracle = Oracle(
            table, cache_dir=os.path.expanduser("~/oracle_cache"))
        try:
            self.table_bits = Entropy(
                self.table,
                self.table.data.fillna(value=0).groupby(
                    [c.name for c in table.columns]).size(), [2])[0]
        except Exception as e:
            print("Error computing table bits", e)
            self.table_bits = 0  # TODO(ekl) why does dmv-full crash on ec2

        fixed_ordering = None
        if self.special_orders <= 1:
            fixed_ordering = list(range(len(table.columns)))

        if self.entropy_order:
            assert self.num_orderings == 1
            res = []
            for i, c in enumerate(table.columns):
                bits = Entropy(c.name, table.data.groupby(c.name).size(), [2])
                res.append((bits[0], i))
            s = sorted(res, key=lambda b: b[0], reverse=self.reverse_entropy)
            fixed_ordering = [t[1] for t in s]
            print('Using fixed ordering:', '_'.join(map(str, fixed_ordering)))
            print(s)

        if self.order is not None:
            print('Using passed-in order:', self.order)
            fixed_ordering = self.order

        if self.order_seed is not None and not self.shuffle_at_data_level:
            if self.order_seed == "reverse":
                fixed_ordering = fixed_ordering[::-1]
            else:
                rng = np.random.RandomState(self.order_seed)
                rng.shuffle(fixed_ordering)
            print('Using generated order:', fixed_ordering)

        print(table.data.info())
        self.fixed_ordering = fixed_ordering

        table_train = table

        if self.special_orders > 0:
            special_orders = _SPECIAL_ORDERS[self.dataset][:self.special_orders]
            k = len(special_orders)
            seed = self.special_order_seed * 10000
            for i in range(k, self.special_orders):
                special_orders.append(
                    np.random.RandomState(seed + i - k + 1).permutation(
                        np.arange(len(table.columns))))
            print('Special orders', np.array(special_orders))
        else:
            special_orders = []

        if self.use_transformer:
            args = {
                "num_blocks": 4,
                "d_model": 64,
                "d_ff": 256,
                "num_heads": 4,
                "nin": len(table.columns),
                "input_bins": [c.DistributionSize() for c in table.columns],
                "use_positional_embs": True,
                "activation": "gelu",
                "fixed_ordering": fixed_ordering,
                "dropout": False,
                "seed": self.seed,
                "first_query_shared": False,
                "prefix_dropout": self.prefix_dropout,
                "mask_scheme": 0,  # XXX only works for default order?
            }
            args.update(self.transformer_args)
            model = Transformer(**args).to(get_device())
        else:
            model = MakeMade(
                scale=self.fc_hiddens,
                cols_to_train=table.columns,
                seed=self.seed,
                dataset=self.dataset,
                fixed_ordering=fixed_ordering,
                special_orders=special_orders,
                layers=self.layers,
                residual=self.residual,
                embed_size=self.embed_size,
                dropout=self.dropout,
                per_row_dropout=self.per_row_dropout,
                prefix_dropout=self.prefix_dropout,
                fixed_dropout_ratio=self.fixed_dropout_ratio,
                input_no_emb_if_leq=self.input_no_emb_if_leq,
                disable_learnable_unk=self.disable_learnable_unk,
                embs_tied=self.embs_tied)

        child = None

        print(model.nin, model.nout, model.input_bins)
        blacklist = None
        mb = ReportModel(model, blacklist=blacklist)
        self.mb = mb

        if not isinstance(model, Transformer):
            print('applying weight_init()')
            model.apply(weight_init)

        if isinstance(model, Transformer):
            opt = torch.optim.Adam(
                list(model.parameters()) + (list(child.parameters())
                                            if child else []),
                2e-4,
                betas=(0.9, 0.98),
                eps=1e-9,
            )
        else:
            opt = torch.optim.Adam(
                list(model.parameters()) + (list(child.parameters())
                                            if child else []), 2e-4)

        self.train_data = TableDataset(table_train)

        self.model = model
        self.opt = opt

        if self.checkpoint_to_load:
            self.model.load_state_dict(torch.load(self.checkpoint_to_load))

    def _train(self):
        if self.checkpoint_to_load:
            self.model.model_bits = 0
            return {
                "epoch": 0,
                "done": True,
                "results": self.evaluate(self.num_eval_queries_at_end, True),
            }

        for _ in range(self.epochs_per_iteration):
            mean_epoch_train_loss = run_epoch(
                'train',
                self.model,
                self.opt,
                train_data=self.train_data,
                val_data=self.train_data,
                batch_size=self.bs,
                epoch_num=self.epoch,
                log_every=200,
                child=None,
                table_bits=self.table_bits,
                warmups=self.warmups)
            self.epoch += 1
        self.model.model_bits = mean_epoch_train_loss / np.log(2)

        done = self.epoch >= self.epochs
        results = self.evaluate(
            self.num_eval_queries_at_end
            if done else self.num_eval_queries_per_iteration, done)
        returns = {
            "epochs": self.epoch,
            "done": done,
            "mean_loss": self.model.model_bits - self.table_bits,
            "train_bits": self.model.model_bits,
            "train_bit_gap": self.model.model_bits - self.table_bits,
            "results": results,
        }

        if self.compute_test_loss:
            returns["test_loss"] = run_epoch(
                'test',
                self.model,
                self.opt,
                train_data=self.train_data,
                val_data=self.train_data,
                batch_size=self.bs,
                epoch_num=self.epoch,
                log_every=200,
                child=None,
                table_bits=self.table_bits,
                warmups=self.warmups) / np.log(2)

        if done and self.asserts:
            for key, max_val in self.asserts.items():
                assert results[key] < max_val, (key, results[key], max_val)

        return returns

    def _save(self, tmp_checkpoint_dir):
        if self.checkpoint_to_load:
            return {}

        if self.fixed_ordering is None:
            if self.seed is not None:
                PATH = "models/{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}.pt".format(
                    self.dataset, self.mb, self.model.model_bits,
                    self.table_bits, self.model.name(), self.epoch, self.seed)
            else:
                PATH = "models/{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}-{}.pt".format(
                    self.dataset, self.mb, self.model.model_bits,
                    self.table_bits, self.model.name(), self.epoch, self.seed,
                    time.time())
        else:
            annot = ""
            PATH = "models/{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}-order{}{}.pt".format(
                self.dataset, self.mb, self.model.model_bits, self.table_bits,
                self.model.name(), self.epoch, self.seed,
                str(self.order_seed)
                if self.order_seed is not None else '_'.join(
                    map(str, self.fixed_ordering))[:60], annot)
        os.makedirs('models/', exist_ok=True)
        torch.save(self.model.state_dict(), PATH)
        print("Saved to:", PATH)
        return {"path": PATH}

    def evaluate(self, num_queries, done):

        def bootstrap_variance(estimator, data):
            estimates = []
            for _ in range(100):
                estimates.append(
                    estimator(
                        np.random.choice(data, size=len(data), replace=True)))
            return np.std(estimates)

        self.model.eval()
        results = {}
        if num_queries:
            oracle_est = None
            estimators = []
            dropout = self.dropout or self.per_row_dropout or self.prefix_dropout
            for n in self.eval_psamples:
                estimators.append(
                    ProgressiveSamplingMade(self.model,
                                            self.table,
                                            n,
                                            device=get_device(),
                                            shortcircuit=dropout))
                if dropout:
                    estimators.append(
                        ProgressiveSamplingMade(self.model,
                                                self.table,
                                                n,
                                                device=get_device(),
                                                shortcircuit=False))
            if self.eval_heuristic:
                estimators.append(Heuristic(self.table))

            rng = np.random.RandomState(1234)
            last_time = None
            for i in range(num_queries):
                if last_time is not None:
                    print('{:.1f} queries/sec'.format(time.time() - last_time))
                print('Query {}:'.format(i), end=' ')
                last_time = time.time()

                query = GenerateQuery(
                    self.dataset,
                    self.table.columns,
                    rng,
                    self.table,
                    query_filters=self.query_filters,
                    force_query_cols=self.force_query_cols)
                Query(
                    estimators,
                    do_print=not SILENT,
                    oracle_card=None,
                    query=query,
                    table=self.table,
                    oracle_est=self.oracle)
                if i % 100 == 0:
                    for est in estimators:
                        est.report()

            for est in estimators:
                results[str(est) + "_max"] = np.max(est.errs)
                results[str(est) + "_max_std"] = bootstrap_variance(
                    np.max, est.errs)
                results[str(est) + "_p99"] = np.quantile(est.errs, 0.99)
                results[str(est) + "_p99_std"] = bootstrap_variance(
                    lambda x: np.quantile(x, 0.99), est.errs)
                results[str(est) + "_median"] = np.median(est.errs)
                results[str(est) + "_median_std"] = bootstrap_variance(
                    np.median, est.errs)
                est.report()

        if self.text_eval_corpus:
            text_eval = {}
            m = TrainedModel(self.model, self.table, get_device())
            num_queries = len(self.text_eval_corpus)
            if not done:
                num_queries = max(1, int(self.text_eval_fraction * num_queries))
            for i in self.eval_psamples:
                naive_errs = []
                prog_errs = []
                skip_errs = []
                for query in self.text_eval_corpus[:num_queries]:
                    ground_truth = m.true_prob(query) * m.count()
                    print("query:", query)
                    naive_est = infer_naive(m, query, i)
                    err = q_error(naive_est, ground_truth)
                    print("naive inference err w/", i, "samples:", err,
                          naive_est, ground_truth)
                    naive_errs.append(err)
                    print("query:", query)
                    prog_est = infer_naive(m, query, i, progressive=True)
                    err = q_error(prog_est, ground_truth)
                    print("prog inference err w/", i, "samples:", err, prog_est,
                          ground_truth)
                    prog_errs.append(err)
                    if self.prefix_dropout:
                        skip_est = infer_skip(m, query, i)
                        err = q_error(skip_est, ground_truth)
                        print("skip inference err w/", i, "samples:", err,
                              skip_est, ground_truth)
                        skip_errs.append(err)
                    print("ground truth prob:", ground_truth)
                results.update({
                    "psample_{}_max".format(i): np.max(naive_errs),
                    "psample_{}_p99".format(i): np.quantile(naive_errs, 0.99),
                    "psample_{}_p95".format(i): np.quantile(naive_errs, 0.95),
                    "psample_{}_median".format(i): np.median(naive_errs),
                    "psample_{}_max_std".format(i): bootstrap_variance(
                        np.max, naive_errs),
                    "psample_{}_p99_std".format(i): bootstrap_variance(
                        lambda x: np.quantile(x, 0.99), naive_errs),
                    "psample_{}_p95_std".format(i): bootstrap_variance(
                        lambda x: np.quantile(x, 0.95), naive_errs),
                    "psample_{}_median_std".format(i): bootstrap_variance(
                        np.median, naive_errs),
                })
                results.update({
                    "psample_prog_{}_max".format(i): np.max(prog_errs),
                    "psample_prog_{}_p99".format(i): np.quantile(
                        prog_errs, 0.99),
                    "psample_prog_{}_p95".format(i): np.quantile(
                        prog_errs, 0.95),
                    "psample_prog_{}_median".format(i): np.median(prog_errs),
                    "psample_prog_{}_max_std".format(i): bootstrap_variance(
                        np.max, prog_errs),
                    "psample_prog_{}_p99_std".format(i): bootstrap_variance(
                        lambda x: np.quantile(x, 0.99), prog_errs),
                    "psample_prog_{}_p95_std".format(i): bootstrap_variance(
                        lambda x: np.quantile(x, 0.95), prog_errs),
                    "psample_prog_{}_median_std".format(i): bootstrap_variance(
                        np.median, prog_errs),
                })
                if skip_errs:
                    results.update({
                        "psample_shortcircuit_{}_max".format(i): np.max(
                            skip_errs),
                        "psample_shortcircuit_{}_p99".format(i): np.quantile(
                            skip_errs, 0.99),
                        "psample_shortcircuit_{}_p95".format(i): np.quantile(
                            skip_errs, 0.95),
                        "psample_shortcircuit_{}_median".format(i): np.median(
                            skip_errs),
                        "psample_shortcircuit_{}_max_std".format(i): bootstrap_variance(
                            np.max, skip_errs),
                        "psample_shortcircuit_{}_p99_std".format(i): bootstrap_variance(
                            lambda x: np.quantile(x, 0.99), skip_errs),
                        "psample_shortcircuit_{}_p95_std".format(i): bootstrap_variance(
                            lambda x: np.quantile(x, 0.95), skip_errs),
                        "psample_shortcircuit_{}_median_std".format(i): bootstrap_variance(
                            np.median, skip_errs),
                    })

        return results


if __name__ == "__main__":
    ray.init()

    tune.run_experiments(
        {
            k: {
                "run": NaruTrainer,
                "checkpoint_at_end": True,
                #                "checkpoint_freq": 1,
                "resources_per_trial": {
                    "gpu": 1 if torch.cuda.is_available() else 0,
                    "cpu": 1,
                },
                "max_failures": 0,
                "config": EXPERIMENT_CONFIGS[k],
            } for k in args.run
        },
        concurrent=True)
