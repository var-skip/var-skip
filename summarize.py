#!/usr/bin/env python

import numpy as np
import json
import collections
import glob
import os
import sys

REPORT_LOSS = "REPORT_LOSS" in os.environ

if len(sys.argv) < 2:
    print(
        "Usage: ./summarize.py <RESULTS_SUBDIR> [REQUIRE_KEY=V, [K=V, [...]]]")
    exit()

data = {}
pattern = os.path.expanduser("results/{}/ray_results/*/*/result.json".format(
    sys.argv[1]))
print("SEARCHING FOR", pattern)
constraints = {}
for arg in sys.argv[2:]:
    k, v = arg.split("=")
    constraints[k] = v
print("CONSTRAINTS", constraints)
for result in glob.glob(pattern):
    key = None
    ok = True
    for line in open(result).readlines():
        line = json.loads(line)
        for k, v in constraints.items():
            conf = line["config"]
            if str(conf[k]) != v:
                print("Ignoring", k, conf[k], "not matching", v)
                ok = False
        if not ok:
            break
        iter = line["training_iteration"]
        key = (line["config"]["dataset"], line["config"]["per_row_dropout"] or
               line["config"]["dropout"], line["config"]["order_seed"],
               result)
        l = line.get("test_loss")
        line = line["results"]
        line["test_loss"] = l
#        break
#        if iter > 10:
#            break
    if key and line and ok:
        data[key] = line

table = []
for key in data:
    dataset, dropout, seed, filename = key
    for quantile in ["max", "p99", "median"]:
        if dropout:
            table.append(
                ("{}_{}-mask-ctrl-{}".format(seed, dataset,
                                             quantile), key, False, quantile))
            table.append(
                ("{}_{}-mask-skip-{}".format(seed, dataset,
                                             quantile), key, True, quantile))
        else:
            table.append(
                ("{}_{}-control-{}".format(seed, dataset,
                                           quantile), key, False, quantile))

samples = [10, 40, 100, 400, 1000, 4000, 10000]
table = sorted(table)

by_stat = collections.defaultdict(list)
by_stat_loss = collections.defaultdict(list)
by_stat_std = collections.defaultdict(float)
for row, key, skip, quantile in table:
    dataset, dropout, seed, filename = key
    cols = [row]
    if skip:
        suffix = "shortcircuit_"
    else:
        suffix = ""
    for n in samples:
        try:
            loss = data[key]["test_loss"]
            value = data[key]["psample_{}{}_{}".format(suffix, n, quantile)]
            std = data[key]["psample_{}{}_{}_std".format(suffix, n, quantile)]
        except:
            continue
        cols.append(str(value))
        if not dropout:
            tpe = "control"
        elif skip:
            tpe = "dropout-skip"
        else:
            tpe = "dropout-ctrl"
        by_stat[(dataset, tpe, n, quantile)].append(value)
        by_stat_std[(dataset, tpe, n, quantile)] = std
        by_stat_loss[(dataset, tpe, n, quantile)].append(loss)
    print(",".join(cols))

print()
for n in [10, 40, 100, 400, 1000, 4000, 10000]:
    print("====", n, "=====")
    for key, values in by_stat.items():
        (dataset, skip, ni, quantile) = key
        loss = by_stat_loss[key]
        std = by_stat_std[key]
        if n == ni:
            if REPORT_LOSS:
                print(dataset, skip, n, quantile, "loss", np.mean(loss), "std", np.std(loss))
            else:
                print(dataset, skip, n, quantile, sorted(values), "bstd", std)
    print()
