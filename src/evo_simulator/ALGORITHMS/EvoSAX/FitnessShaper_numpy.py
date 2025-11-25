from typing import Union, Optional
import numpy as np


class FitnessShaper(object):
    def __init__(self, centered_rank: Union[bool, int] = False, z_score: Union[bool, int] = False, norm_range: Union[bool, int] = False, w_decay: float = 0.0, maximize: Union[bool, int] = False, fitness_trafo: Optional[str] = None):
        """NumPy-compatible fitness shaping tool."""
        self.w_decay = w_decay
        self.maximize = bool(maximize)

        if fitness_trafo in ["centered_rank", "z_score", "norm_range", "raw"]:
            self.centered_rank = fitness_trafo == "centered_rank"
            self.z_score = fitness_trafo == "z_score"
            self.norm_range = fitness_trafo == "norm_range"
        else:
            self.centered_rank = bool(centered_rank)
            self.z_score = bool(z_score)
            self.norm_range = bool(norm_range)
        # Check that only a single fitness shaping transformation is used
        num_options_on = self.centered_rank + self.z_score + self.norm_range
        assert (
            num_options_on < 2
        ), "Only use one fitness shaping transformation."

    def apply(self, x: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Max objective transformation, rank shaping, z-scoring & weight decay."""
        if self.maximize:
            fitness = -1 * fitness

        # Apply weight decay before normalization - makes it easier to tune
        # "Reduce" fitness based on L2 norm of parameters
        if self.w_decay > 0.0:
            l2_fit_red = self.w_decay * compute_l2_norm(x)
            fitness += l2_fit_red

        if self.centered_rank:
            fitness = centered_rank_trafo(fitness)

        if self.z_score:
            fitness = z_score_trafo(fitness)

        if self.norm_range:
            fitness = range_norm_trafo(fitness, -1.0, 1.0)

        return fitness


def z_score_trafo(arr: np.ndarray) -> np.ndarray:
    """Make fitness 'Gaussian' by subtracting mean and dividing by std."""
    return (arr - np.nanmean(arr)) / (np.nanstd(arr) + 1e-10)


def compute_ranks(fitness: np.ndarray) -> np.ndarray:
    """Return fitness ranks in [0, len(fitness))."""
    ranks = np.zeros(len(fitness))
    ranks[fitness.argsort()] = np.arange(len(fitness))
    return ranks


def centered_rank_trafo(fitness: np.ndarray) -> np.ndarray:
    """Return ~ -0.5 to 0.5 centered ranks (best to worst - min!)."""
    y = compute_ranks(fitness)
    y /= fitness.size - 1
    return y - 0.5


def compute_l2_norm(x: np.ndarray) -> np.ndarray:
    """Compute L2-norm of x_i. Assumes x has shape (popsize, num_dims)."""
    return np.nanmean(x * x, axis=1)


def range_norm_trafo(arr: np.ndarray, min_val: float = -1.0, max_val: float = 1.0) -> np.ndarray:
    """Map scores into a specified min/max range."""
    arr = np.clip(arr, -1e10, 1e10)
    normalized_arr = (max_val - min_val) * (arr - np.nanmin(arr)) / (
        np.nanmax(arr) - np.nanmin(arr) + 1e-10
    ) + min_val
    return normalized_arr
