import copy
import time

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, IterableDataset


class Column(object):
    """A column.  Data is write-once, immutable-after.

    Typical usage:
      col = Column('myCol').Fill(data).SetDistribution(domain_vals)

    "data" and "domain_vals" are NOT copied.
    """

    def __init__(self, name, distribution_size=None, pg_name=None):
        self.name = name

        # Data related fields.
        self.data = None
        self.all_distinct_values = None
        self.distribution_size = distribution_size

        # pg_name is the name of the corresponding column in Postgres.  This is
        # put here since, e.g., PG disallows whitespaces in names.
        self.pg_name = pg_name if pg_name else name

        self._val_to_bin_cache = {}

    def Name(self):
        """Name of this column."""
        return self.name

    def DistributionSize(self):
        """This column will take on discrete values in [0, N).

        Used to dictionary-encode values to this discretized range.
        """
        return self.distribution_size

    def BinToVal(self, bin_id):
        assert bin_id >= 0 and bin_id < self.distribution_size, bin_id
        return self.all_distinct_values[bin_id]

    def ValToBin(self, val):
        if val in self._val_to_bin_cache:
            return self._val_to_bin_cache[val]

        if isinstance(self.all_distinct_values, list):
            return self.all_distinct_values.index(val)
        inds = np.where(self.all_distinct_values == val)
        if len(inds[0]) <= 0:
            raise IndexError("Value not found")

        res = inds[0][0]
        self._val_to_bin_cache[val] = res
        return res

    def SetDistribution(self, distinct_values):
        """This is all the values this column will ever see."""
        assert self.all_distinct_values is None
        # pd.isnull returns true for both np.nan and np.datetime64('NaT').
        is_nan = pd.isnull(distinct_values)
        contains_nan = np.any(is_nan)
        dv_no_nan = distinct_values[~is_nan]
        # IMPORTANT: np.sort puts NaT values at beginning, and NaN values at
        # end for our purposes we always add any null value to the beginning.
        vs = np.sort(np.unique(dv_no_nan))
        if contains_nan and np.issubdtype(distinct_values.dtype, np.datetime64):
            vs = np.insert(vs, 0, np.datetime64('NaT'))
        elif contains_nan:
            vs = np.insert(vs, 0, np.nan)
        if self.distribution_size is not None:
            assert len(vs) == self.distribution_size
        self.all_distinct_values = vs
        self.distribution_size = len(vs)
        return self

    def Fill(self, data_instance, infer_dist=False):
        assert self.data is None
        self.data = data_instance
        # If no distribution is currently specified, then infer distinct values
        # from data.
        if infer_dist:
            self.SetDistribution(self.data)
        return self

    def __repr__(self):
        return 'Column({}, distribution_size={})'.format(
            self.name, self.distribution_size)


class Table(object):
    """A collection of Columns."""

    def __init__(self, name, columns, pg_name=None, validate_cardinality=True):
        """Creates a Table.

        Args:
            name: Name of this table object.
            columns: List of Column instances to populate this table.
            pg_name: name of the corresponding table in Postgres.
        """
        self.name = name
        if validate_cardinality:
            self.cardinality = self._validate_cardinality(columns)
        else:
            # Used as a wrapper, not a real table.
            self.cardinality = None
        self.columns = columns

        # Bin to val funcs useful for sampling.  Takes
        #   (col 1's bin id, ..., col N's bin id)
        # and converts it to
        #   (col 1's val, ..., col N's val).
        self.column_bin_to_val_funcs = [c.BinToVal for c in columns]
        self.val_to_bin_funcs = [c.ValToBin for c in columns]
        self.name_to_index = {c.Name(): i for i, c in enumerate(self.columns)}

        if pg_name:
            self.pg_name = pg_name
        else:
            self.pg_name = name

    def __repr__(self):
        return '{}({})'.format(self.name, self.columns)

    def _validate_cardinality(self, columns):
        """Checks that all the columns have same the number of rows."""
        cards = [len(c.data) for c in columns]
        c = np.unique(cards)
        assert len(c) == 1, c
        return c[0]

    def to_df(self):
        return pd.DataFrame({c.name: c.data for c in self.columns})

    def Name(self):
        """Name of this table."""
        return self.name

    def Columns(self):
        """Return the list of Columns under this table."""
        return self.columns

    def ColumnIndex(self, name):
        """Returns index of column with the specified name."""
        assert name in self.name_to_index
        return self.name_to_index[name]


class CsvTable(Table):

    def __init__(self,
                 name,
                 filename_or_df,
                 cols,
                 type_casts,
                 pg_name=None,
                 pg_cols=None,
                 dropna=False,
                 is_str_col=False,
                 order_seed=None,
                 char_limit=200,
                 tie_cols=False,
                 **kwargs):
        """Accepts same arguments as pd.read_csv().

        Args:
            filename_or_df: pass in str to reload; otherwise accepts a loaded
              pd.Dataframe.
        """
        self.name = name
        self.pg_name = pg_name
        self.tie_cols = tie_cols

        if isinstance(filename_or_df, str):
            self.data = self._load(filename_or_df, cols, **kwargs)
        else:
            assert (isinstance(filename_or_df, pd.DataFrame))
            self.data = filename_or_df

        if is_str_col:
            self.data = self._separate_characters(self.data, cols, char_limit)

        if order_seed is not None:
            ordering = self.data.columns.tolist()
            rng = np.random.RandomState(order_seed)
            rng.shuffle(ordering)
            print(
                "Rearranging columns from", self.data.columns.tolist(),
                "to", ordering, "seed", order_seed)
            self.data = self.data[ordering]

        self.dropna = dropna
        if dropna:
            # NOTE: this might make the resulting dataframe much smaller.
            self.data = self.data.dropna()

        cols = self.data.columns
        self.columns = self._build_columns(self.data, cols, type_casts, pg_cols)

        super(CsvTable, self).__init__(name, self.columns, pg_name)

    def _load(self, filename, cols, **kwargs):
        print('Loading csv...', end=' ')
        s = time.time()
        # Use [cols] here anyway to reorder columns by 'cols'.
        df = pd.read_csv(filename, usecols=cols, **kwargs)[cols]
        print('done, took {:.1f}s'.format(time.time() - s))
        return df

    def _build_columns(self, data, cols, type_casts, pg_cols):
        """Example args:

            cols = ['Model Year', 'Reg Valid Date', 'Reg Expiration Date']
            type_casts = {'Model Year': int}

        Returns: a list of Columns.
        """
        print('Parsing...', end=' ')
        s = time.time()
        for col, typ in type_casts.items():
            if col not in data:
                continue
            if typ != np.datetime64:
                data[col] = data[col].astype(typ, copy=False)
            else:
                # Both infer_datetime_format and cache are critical for perf.
                data[col] = pd.to_datetime(data[col],
                                           infer_datetime_format=True,
                                           cache=True)

        # Discretize & create Columns.
        columns = []
        if pg_cols is None:
            pg_cols = [None] * len(cols)

        if self.tie_cols:
            vocab = np.concatenate([
                data[c].value_counts(dropna=False).index.values for c in cols
            ])
            vocab = np.sort(np.unique(vocab))
        else:
            vocab = None

        for c, p in zip(cols, pg_cols):
            col = Column(c, pg_name=p)
            col.Fill(data[c])

            # dropna=False so that if NA/NaN is present in data,
            # all_distinct_values will capture it.
            #
            # For numeric: np.nan
            # For datetime: np.datetime64('NaT')
            # For strings: ?? (haven't encountered yet)
            #
            # To test for former, use np.isnan(...).any()
            # To test for latter, use np.isnat(...).any()

            if vocab is not None:
                col.SetDistribution(vocab)
            else:
                col.SetDistribution(data[c].value_counts(dropna=False).index.values)
            columns.append(col)
        print('done, took {:.1f}s'.format(time.time() - s))
        return columns

    def _separate_characters(self, data, cols, limit):
        assert len(cols) == 1, "should only have 1 str col"
        str_col = data[cols[0]]
        return pd.DataFrame(str_col.apply(lambda x: list(x[:limit])).tolist()).fillna(value="$")


class TableDataset(Dataset):
    """Wraps a Table and yields each row as a Dataset element."""

    def __init__(self, table, input_encoding=None):
        """Wraps a Table.

        Args:
          table: the Table.
        """
        super(TableDataset, self).__init__()
        self.table = copy.deepcopy(table)

        print('Discretizing table...', end=' ')
        s = time.time()
        # [cardianlity, num cols].
        self.tuples_np = np.stack(
            [self.Discretize(c) for c in self.table.Columns()], axis=1)
        self.tuples = torch.as_tensor(
            self.tuples_np.astype(np.float32, copy=False))
        print('done, took {:.1f}s'.format(time.time() - s))

    def Discretize(self, col):
        """Discretize values into its Column's bins.

        Args:
          col: the Column.
        Returns:
          col_data: discretized version; an np.ndarray of type np.int32.
        """
        return Discretize(col)

    def size(self):
        return len(self.tuples)

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]


def Discretize(col, data=None):
    """Transforms data values into integers using a Column's vocab.

    Args:
        col: the Column.
        data: list-like data to be discretized.  If None, defaults to col.data.
    Returns:
        col_data: discretized version; an np.ndarray of type np.int32.
    """
    # pd.Categorical() does not allow categories be passed in an array
    # containing np.nan.  It makes it a special case to return code -1
    # for NaN values.

    if data is None:
        data = col.data

    # pd.isnull returns true for both np.nan and np.datetime64('NaT').
    isnan = pd.isnull(col.all_distinct_values)
    if isnan.any():
        # We always add nan or nat to the beginning.
        assert isnan.sum() == 1, isnan
        assert isnan[0], isnan

        dvs = col.all_distinct_values[1:]
        bin_ids = pd.Categorical(data, categories=dvs).codes
        assert len(bin_ids) == len(data)

        # Since nan/nat bin_id is supposed to be 0 but pandas returns -1, just
        # add 1 to everybody
        bin_ids = bin_ids + 1
    else:
        # This column has no nan or nat values.
        dvs = col.all_distinct_values
        bin_ids = pd.Categorical(data, categories=dvs).codes
        assert len(bin_ids) == len(data), (len(bin_ids), len(data))

    bin_ids = bin_ids.astype(np.int32, copy=False)
    assert (bin_ids >= 0).all(), (col, data, bin_ids)
    return bin_ids
