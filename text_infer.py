import random
import time
import torch


class TrainedModel:

    def __init__(self, model, table, device):
        self.model = model
        self.table = table
        self.device = device
        self.oracle = OracleModel(
            ["".join([c for c in x]) for x in table.data.values])

    def char_width(self):
        return self.oracle.char_width()

    def count(self):
        return self.oracle.count()

    def true_prob(self, substr):
        return self.oracle.true_prob(substr)

    def p(self, substr, i):
        return self.prob_occurs_starting_at(substr, [i])

    def prob_occurs_starting_at(self, substrs, indices, prog_query=None):
        if type(substrs) is str:
            substrs = [substrs] * len(indices)
        model, table = self.model, self.table
        query_bins = []
        nonzero_indices = []
        for i, (substr, index) in enumerate(zip(substrs, indices)):
            if len(substr) + index > len(table.columns):
                continue
            has_error = False
            query_bin = [0] * index
            for char, col in zip(substr, table.columns[index:]):
                try:
                    query_bin.append(col.ValToBin(char))
                except IndexError:
                    #print("WARN: Out of distribution character", char, col)
                    has_error = True
            if has_error:
                continue
            nonzero_indices.append(i)
            query_bins.append(query_bin)
        assert nonzero_indices
        assert query_bins

        model.eval()
        with torch.no_grad():
            batch = torch.tensor([
                query_bin + [0] * (len(table.columns) - len(query_bin))
                for query_bin in query_bins
            ]).long().to(self.device)
            ix = [indices[i] for i in nonzero_indices]
            logits = model.forward(batch, skip_prefix=ix)
            needed_i = set()
            for j in range(len(nonzero_indices)):
                query_bin = query_bins[j]
                for i in range(indices[nonzero_indices[j]], len(query_bin)):
                    needed_i.add(i)
            dists = [
                torch.softmax(model.logits_for_col(i, logits), -1)
                if i in needed_i else None for i in range(len(table.columns))
            ]
            probs = [1.0] * len(query_bins)
            for j in range(len(nonzero_indices)):  # for each batch elem
                query_bin = query_bins[j]
                start = indices[nonzero_indices[j]]
                if prog_query is not None:
                    # only count cond probs for the query, not prefix chars
                    start = len(query_bin) - len(prog_query)
                for i in range(start, len(query_bin)):  # sample next char
                    dist = dists[i]
                    probs[j] *= dist[j][query_bin[i]]

        return nonzero_indices, [float(p) for p in probs]

    def sample_next(self, substrs, ignore_prefix_sizes, last_batch=None):
        model, table = self.model, self.table

        results = [None] * len(substrs)

        if last_batch is not None:
            # Continue building the last batch. Just need to fill out the last char.
            for i, (substr, ignore_prefix_size) in enumerate(
                    zip(substrs, ignore_prefix_sizes)):
                has_error = False
                j = len(substr) - 1
                char = substr[j]
                col = table.columns[j]
                try:
                    bin_val = col.ValToBin(char)
                except IndexError:
                    #print("WARN: Out of distribution character", char, col)
                    has_error = True
                if has_error:
                    results[i] = substr  # no sampling
                else:
                    last_batch[i][j] = int(bin_val)
        else:
            query_bins = [None] * len(substrs)
            for i, (substr, ignore_prefix_size) in enumerate(
                    zip(substrs, ignore_prefix_sizes)):
                has_error = False
                query_bin = [0] * ignore_prefix_size
                for char, col in zip(substr[ignore_prefix_size:],
                                     table.columns[ignore_prefix_size:]):
                    try:
                        query_bin.append(col.ValToBin(char))
                    except IndexError:
                        #print("WARN: Out of distribution character", char, col)
                        has_error = True
                        break
                if has_error:
                    results[i] = substr  # no sampling
                query_bins[i] = query_bin
            assert all([q is not None for q in query_bins])

        model.eval()
        has_next = False
        with torch.no_grad():
            if last_batch is not None:
                batch = last_batch
            else:
                batch = torch.tensor([
                    query_bin + [0] * (len(table.columns) - len(query_bin))
                    for query_bin in query_bins
                ]).long().to(self.device)
            logits = model.forward(batch, skip_prefix=ignore_prefix_sizes)

            needed_i = set()
            for batch_idx, substr in enumerate(substrs):
                needed_i.add(len(substr))

            samples = [
                torch.multinomial(
                    torch.softmax(model.logits_for_col(i, logits), -1), 1)
                if i in needed_i else None for i in range(len(table.columns))
            ]

            for batch_idx, substr in enumerate(substrs):
                if results[batch_idx] is not None:
                    continue  # no extension possible
                col_i = len(substr)
                if col_i >= self.char_width():
                    results[batch_idx] = substr  # no extension possible
                    continue
                elif col_i + 1 < self.char_width():
                    has_next = True
                bin_val = int(samples[col_i][batch_idx][0])
                results[
                    batch_idx] = substr + table.columns[col_i].BinToVal(bin_val)

        assert all([r is not None for r in results])
        return results, has_next, batch

    def sample(self, prefixes=[""], ignore_prefix_sizes=[0]):
        next_batch = None
        while True:
            prefixes, has_next, next_batch = self.sample_next(
                prefixes, ignore_prefix_sizes, next_batch)
            if not has_next:
                return prefixes


class OracleModel(object):

    def __init__(self, strings):
        self.strings = strings
        lens = set(len(x) for x in strings)
        assert len(lens) == 1, lens

    def char_width(self):
        return len(self.strings[0])

    def count(self):
        return len(self.strings)

    def true_prob(self, substr):
        num_matched = 0
        for string in self.strings:
            if substr in string:
                num_matched += 1
        return float(num_matched) / len(self.strings)

    def prob_occurs_starting_at(self, substr, index):
        num_matched = 0
        for string in self.strings:
            if string[index:].startswith(substr):
                num_matched += 1
        return float(num_matched) / len(self.strings)

    def sample_next(self, prefix, ignore_prefix_size=0):
        candidates = []
        for string in self.strings:
            if string[ignore_prefix_size:].startswith(
                    prefix[ignore_prefix_size:]):
                candidates.append(string)
        if not candidates:
            return None
        candidate = random.choice(candidates)
        ret = candidate[:len(prefix) + 1]
        ret = "?" * ignore_prefix_size + ret[ignore_prefix_size:]
        if len(ret) == len(prefix):
            return False  # no more
        return ret

    def sample(self, prefix="", ignore_prefix_size=0):
        while True:
            next = self.sample_next(prefix, ignore_prefix_size)
            if next:
                prefix = next
            else:
                if next is None:
                    return None
                return prefix


def infer_skip(model, query, num_samples=100):
    start = time.time()
    width = model.char_width()
    batch_size = max(1, num_samples // width)
    print("batch size", batch_size)

    all_p = []
    indices = list(range(width))
    nonzero_indices, probs = model.prob_occurs_starting_at(query, indices)
    print("probs", probs)

    #    print("full 0 dist", {
    #        "aaaaaaaaaa": model.p("aaaaaaaaaa", 0),
    #        "https://fi": model.p("https://fi", 0),
    #        "https://go": model.p("https://go", 0),
    #        "https://ya": model.p("https://ya", 0),
    #    })
    #    print("index 8 dist", {
    #        "a": model.p("a", 8),
    #        "f": model.p("f", 8),
    #        "g": model.p("g", 8),
    #        "y": model.p("y", 8),
    #    })
    #    print("index 9 dist", {
    #        "i": model.p("i", 9),
    #        "o": model.p("o", 9),
    #        "a": model.p("a", 9),
    #    })

    nonzero_probs = [p for p in probs if p > 0]
    print("nonzero_probs", nonzero_probs)
    nonzero_indices = [
        nonzero_indices[i] for i in range(len(probs)) if probs[i] > 0
    ]
    print("nonzero indices", nonzero_indices)
    substrs = []
    zero_in = []
    for _ in range(batch_size):
        substrs.extend(["?" * i + query for i in nonzero_indices])
        zero_in.extend(nonzero_indices)
    print("substrs", substrs[:10], "count", len(substrs))
    assert len(substrs) == len(zero_in)
    completions = model.sample(substrs, zero_in)
    assert len(completions) == len(substrs)
    print("completions", completions[:10])
    remainings = [
        completions[i][zero_in[i] + len(query):]
        for i in range(len(completions))
    ]
    print(remainings[:10])
    num_rej = 0.0
    acc_count, rej_count = 0, 0
    for i in range(len(remainings)):
        if query not in remainings[
                i]:  # remove duplicate, skip for rejection sampling
            all_p.append(nonzero_probs[i % len(nonzero_probs)])
            acc_count += 1
        else:
            num_rej += nonzero_probs[i % len(nonzero_probs)]
            rej_count += 1
    print("density rejected", num_rej, rej_count)
    print("density accepted", sum(all_p), acc_count)
    print("infer skip time", time.time() - start, "for", num_samples)

    return model.count() * sum(all_p) / batch_size


def progressive_sampling(model, query, samples):
    width = model.char_width()
    num_samples = len(samples)
    batch_size = max(1, num_samples // width)
    i = 0
    query_in = []
    all_p = []
    for s in samples:
        rep = s[:i] + query
        rep = rep[:width]
        if i + len(query) <= width:
            query_in.append(rep)
        i += 1
        i = i % width
    nonzero_indices, probs = model.prob_occurs_starting_at(
        query_in, [0] * len(query_in), prog_query=query)
    #    print("prog probs", probs)

    nonzero_probs = [p for p in probs if p > 0]
    #    print("prog nonzero_probs", nonzero_probs)
    nonzero_indices = [
        nonzero_indices[i] for i in range(len(probs)) if probs[i] > 0
    ]
    #    print("prog nonzero indices", nonzero_indices)
    substrs = [query_in[i] for i in nonzero_indices]
    print("prog substrs", substrs[:10], "count", len(substrs))
    completions = model.sample(substrs, [0] * len(substrs))
    assert len(completions) == len(substrs)
    print("prog completions", completions[:10])
    remainings = [
        completions[i][len(substrs[i]):] for i in range(len(completions))
    ]
    print("prog remainings", remainings[:10])
    num_rej = 0.0
    acc_count, rej_count = 0, 0
    for i in range(len(remainings)):
        if query not in remainings[
                i]:  # remove duplicate, skip for rejection sampling
            all_p.append(nonzero_probs[i % len(nonzero_probs)])
            acc_count += 1
        else:
            num_rej += nonzero_probs[i % len(nonzero_probs)]
            rej_count += 1
    return model.count() * sum(all_p) / batch_size


def infer_naive(model, query, num_samples=100, progressive=False):
    start = time.time()
    num_match = 0
    samples = model.sample([""] * num_samples, [0] * num_samples)
    print("naive samples", samples[:10])
    #print(samples)
    for sample in samples:
        if query in sample:
            num_match += 1
    print("infer naive time", time.time() - start, "for", num_samples)

    if progressive:
        return progressive_sampling(model, query, samples)

    return model.count() * float(num_match) / num_samples


def q_error(est, real):
    real = max(1, real)
    return round(max(est, real) / max(1, min(est, real)), 2)


# query: google
# naive inference err w/ 10 samples: 1.09
# naive inference err w/ 100 samples: 1.02
# skip inference err w/ 10 samples: 1.0
# skip inference err w/ 100 samples: 1.0
# ground truth prob: 55.0
# -----
# query: https
# naive inference err w/ 10 samples: 600.0
# naive inference err w/ 100 samples: 1.4
# skip inference err w/ 10 samples: 1.0
# skip inference err w/ 100 samples: 1.0
# ground truth prob: 6.0
# -----
# query: rare
# naive inference err w/ 10 samples: 5.6
# naive inference err w/ 100 samples: 2.8
# skip inference err w/ 10 samples: 1.0
# skip inference err w/ 100 samples: 1.0
# ground truth prob: 1.0
# -----
if __name__ == "__main__":
    text = [
        "https://google.com?q=somethingrandom",
        "https://google.com?q=otherrandomstri",
        "https://firefox.com?query=google.com",
        "https://firefox.com?query=google.com",
        "https://firefox.com?query=google.com",
        "https://yahoo.com/some/rare/match---",
    ]
    for _ in range(50):
        text.append("http://google/is/some/other/google/c")

    model = OracleModel(text)

    for query in ["google", "https", "rare"]:
        ground_truth = model.true_prob(query) * model.count()
        print("query:", query)
        for i in [10, 100]:
            naive_est = infer_naive(model, query, i)
            print("naive inference err w/", i, "samples:",
                  q_error(naive_est, ground_truth))
        for i in [10, 100]:
            skip_est = infer_skip(model, query, i)
            print("skip inference err w/", i, "samples:",
                  q_error(skip_est, ground_truth))
        print("ground truth prob:", ground_truth)
        print("-----")
