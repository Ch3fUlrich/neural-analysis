from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

# Placeholder for future more advanced CV strategies like stratified grouped K-folds
# Currently sklearn KFold handles basics, but this module allows extension without bloated imports in decoders.py


def create_folds(
    labels: npt.NDArray[np.floating[Any]],
    n_folds: int = 5,
    stratify: bool = False,
    random_state: int = 42,
) -> Iterator[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]:
    """
    Creates cross-validation folds.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    if stratify and not np.issubdtype(labels.dtype, np.floating):
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        return kf.split(np.zeros(len(labels)), labels)  # type: ignore
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        return kf.split(np.zeros(len(labels)))  # type: ignore
