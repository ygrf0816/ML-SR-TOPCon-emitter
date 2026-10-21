"""Train/test splits for SR evaluation (no PySR import)."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from topcon_experiments.config import RANDOM_STATE, TEST_SIZE


def split_train_test(n: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def split_group_train_test(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split by group id so all curve points of one sample stay together."""
    uniq = np.unique(groups)
    train_g, test_g = train_test_split(
        uniq, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_mask = np.isin(groups, train_g)
    test_mask = np.isin(groups, test_g)
    return np.where(train_mask)[0], np.where(test_mask)[0]
