"""
Local PrivateSMOTE extension for the outlier pipeline.

`FocusedPrivateSMOTE` is identical to `newPrivateSMOTE` (same interpolation + Laplace-DP
mechanism, same categorical handling) EXCEPT that the synthetic seeds are supplied
explicitly (with multiplicity) instead of drawn uniformly at random from the single-out
rows. This lets the caller implement a coverage-first / focused allocation of the fixed
oversampling budget toward the outlier single-outs, without modifying code/main.

Reuses everything from code/main/privatesmote_old.py; only the seed-selection line of
over_sampling changes.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from main.privatesmote_old import (
    newPrivateSMOTE,
    keep_numbers,
    check_and_adjust_data_types,
)


class FocusedPrivateSMOTE(newPrivateSMOTE):
    """PrivateSMOTE whose seeds are an explicit list of positional row indices."""

    def over_sampling_focused(self, seed_indices):
        seed_indices = list(seed_indices)
        desired = len(seed_indices)
        # find highest-risk cases (sets up 'highest_risk' col used for shapes)
        self.samples = self.khighest_risk()
        self.X_train_shape = self.samples.loc[
            self.samples['highest_risk'] == 1, self.samples.columns[:-2]].shape
        self.synthetic = np.empty(
            shape=(desired, self.X_train_shape[1] + 1), dtype='float32')
        self.x = self.enc_data()
        self.min_values = [self.x[:, i].min() if not self.is_object_type[i] else np.nan
                           for i in range(self.x.shape[1])]
        self.max_values = [self.x[:, i].max() if not self.is_object_type[i] else np.nan
                           for i in range(self.x.shape[1])]
        self.std_values = [np.std(self.x[:, i]) if not self.is_object_type[i] else np.nan
                           for i in range(self.x.shape[1])]

        # ONE synthetic row per supplied seed (multiplicity = focus)
        for i in seed_indices:
            if self.precomputed_neighbor_indices is not None:
                nnarray = self.precomputed_neighbor_indices[i]
            else:
                nnarray = self.neighbors.kneighbors(
                    self.standardized_data[i].reshape(1, -1), return_distance=False)[0]
            self._populate(1, i, nnarray)

        highest_risk_col = np.ones((self.synthetic.shape[0], 1), dtype=self.synthetic.dtype)
        new = np.concatenate((self.synthetic, highest_risk_col), axis=1)
        new = pd.DataFrame(new, index=range(new.shape[0]), columns=self.samples.columns)
        new = new.astype(dtype=self.samples.dtypes)
        if np.any(self.is_object_type):
            new = self.decode_categorical_columns(new)
        return new


def apply_private_smote_focused(data, epsilon, seed_indices, knn, k, key_vars,
                                single_out_column, precomputed_neighbor_indices=None):
    """Mirror of apply_private_smote_new, but with explicit seeds (len == budget)."""
    data = keep_numbers(data)
    # cast 0/1 numeric features to object so they are not interpolated (same as stock)
    for col in data.columns[:-1]:
        uv = set(data[col].dropna().unique())
        if uv in ({0.0, 1.0}, {0, 1}, {0.0, 1}, {0, 1.0}):
            data[col] = data[col].astype('object')

    tgt_obj = data[data.columns[-1]].dtypes == 'object'
    if tgt_obj:
        target_encoder = LabelEncoder()
        data[data.columns[-1]] = target_encoder.fit_transform(data[data.columns[-1]])

    new_samples = FocusedPrivateSMOTE(
        data, knn, epsilon, k, key_vars, single_out_column,
        precomputed_neighbor_indices=precomputed_neighbor_indices,
    ).over_sampling_focused(seed_indices)

    if tgt_obj:
        new_samples[new_samples.columns[-1]] = target_encoder.inverse_transform(
            new_samples[new_samples.columns[-1]])
    new_samples = check_and_adjust_data_types(data, new_samples)
    return new_samples
