from typing import Tuple, Optional
import numpy as np
from dataclasses import dataclass
from .FitnessShaper_numpy import FitnessShaper
# from .des import get_des_weights


@dataclass
class EvoState:
    mean: np.ndarray
    sigma: np.ndarray
    weights: np.ndarray
    noise: np.ndarray
    pop_params: np.ndarray
    best_member: np.ndarray
    best_fitness: float = np.finfo(np.float32).max
    gen_counter: int = 0


@dataclass
class EvoParams:
    lrate_mean: float = 1.0
    lrate_sigma: float = 1.0
    sigma_init: float = 1.0
    temperature: float = 0.0
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = np.finfo(np.float32).min
    clip_max: float = np.finfo(np.float32).max


def get_snes_weights(popsize: int, use_baseline: bool = True) -> np.ndarray:
    """Get recombination weights for different ranks."""

    def get_weight(i):
        return np.maximum(0, np.log(popsize / 2 + 1) - np.log(i))

    indices = np.arange(1, popsize + 1)
    weights = np.vectorize(get_weight)(indices)
    weights_norm = weights / np.sum(weights)
    # print("(weights_norm - use_baseline * (1 / popsize))[:, None]", (weights_norm - use_baseline * (1 / popsize))[:, None])
    # print("(\n\nweights_norm - use_baseline * (1 / popsize))", (weights_norm - use_baseline * (1 / popsize)))
    # exit()
    return (weights_norm - use_baseline * (1 / popsize))[:, None]


class SNES_numpy:
    def __init__(
        self,
        popsize: int,
        num_dims: int,
        # mean_init: float = 0.0,
        sigma_init: float = 1.0,
        mean_decay: float = 0.0,
        seed: Optional[int] = None,
        fitness_shaper:FitnessShaper = None,

        # temperature: float = 0.0,  # good values tend to be between 12 and 20
        # pholder_params: Optional[Union[dict, np.ndarray]] = None,
        # n_devices: Optional[int] = None,
        # **fitness_kwargs: Union[bool, int, float]
    ):
        """Separable Exponential Natural ES (Wierstra et al., 2014)
        Reference: https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
        """
        # super().__init__(
        #     popsize, num_dims, pholder_params, mean_decay, n_devices, **fitness_kwargs
        # )

        self.strategy_name = "SNES"

        self.popsize:int = popsize
        self.num_dims:int = num_dims

        # self.mean_init = mean_init
        self.sigma_init = sigma_init

        self.seed = seed if seed is not None else np.random.default_rng().integers(0, 2**32)
        self.rng:np.random.Generator = np.random.default_rng(seed=self.seed)

        self.mean_decay = mean_decay
        self.is_mean_decay = mean_decay > 0.0

        self.fitness_shaper:FitnessShaper = fitness_shaper if fitness_shaper is not None else FitnessShaper()
        self.is_fitness_shaper = fitness_shaper is not None

    @property
    def default_params(self) -> EvoParams: # params_strategy equivalent
        """Return default parameters of evolutionary strategy."""
        lrate_sigma = (3 + np.log(self.num_dims)) / (5 * np.sqrt(self.num_dims))

        params = EvoParams(
            lrate_sigma=lrate_sigma,
            sigma_init=self.sigma_init,
            # temperature=self.temperature,
        )
        return params

    def initialize(self, rng: np.random.Generator, params: EvoParams) -> EvoState: # initialize_strategy equivalent
        """`initialize` the evolutionary strategy."""
        initialization:np.ndarray = self.rng.uniform(
            params.init_min,
            params.init_max,
            size=(self.num_dims,),
        )
        # use_des_weights = params.temperature > 0.0
        # if use_des_weights:
        #     weights = get_des_weights(self.popsize, params.temperature)
        # else:
        weights:np.ndarray = get_snes_weights(self.popsize)
        state = EvoState(
            mean=initialization,
            sigma=params.sigma_init * np.ones(self.num_dims),
            weights=weights,
            best_member=initialization,
            noise=None,
            pop_params=None,
        )

        return state


    def update_mean_and_sigma_with_new_pop_params(self, new_pop_params:np.ndarray, evo_state:EvoState, alpha:float = 0.2, beta:float = 0.1, min_sigma:float = 1e-3) -> None:

        # 1 - Reconstruct the implicit mean with the new population parameters
        mean_candidates = new_pop_params - evo_state.noise * evo_state.sigma.reshape(1, -1)

        # 2 - Compute the new mean
        new_mean = mean_candidates.mean(axis=0)

        # 3 - Update the mean with the new mean based on the alpha parameter
        evo_state.mean = (1 - alpha) * evo_state.mean + alpha * new_mean

        # 4 - Update the sigma with the new mean based on the beta parameter
        centered        = new_pop_params - new_pop_params.mean(axis=0)
        target_sigma    = np.sqrt((centered ** 2).mean(axis=0))
        evo_state.sigma = np.maximum((1 - beta) * evo_state.sigma + beta * target_sigma, min_sigma)


    def ask(self, rng:np.random.Generator, evo_state:EvoState, evo_params:EvoParams) -> Tuple[np.ndarray, EvoState]: # ask_strategy equivalent
        """`ask` for new parameter candidates to evaluate next."""

        # 1 - Noise sampling
        evo_state.noise = self.rng.normal(size=(self.popsize, self.num_dims))

        # 2 - Generate population parameters with noise
        pop_params:np.ndarray = evo_state.mean + evo_state.noise * evo_state.sigma.reshape(1, -1)
        # pop_params_2:np.ndarray = evo_state.mean + noise * evo_state.sigma.reshape(1, self.num_dims)

        # 3 - Clip parameters into allowed range
        pop_params:np.ndarray = np.clip(pop_params, evo_params.clip_min, evo_params.clip_max)

        return pop_params, evo_state

    def tell(self, x: np.ndarray, fitness: np.ndarray, evo_state: EvoState, params: EvoParams) -> EvoState:
        """`tell` performance data for strategy state update."""
    
        # 0 - Apply fitness shaping if necessary
        if self.is_fitness_shaper: fitness = self.fitness_shaper.apply(x, fitness)
        
        # 1. Sort fitness and noise
        s = (x - evo_state.mean) / evo_state.sigma
        ranks = np.argsort(fitness)
        sorted_noise = s[ranks]

        # 2. Compute gradients
        grad_mean = np.sum(evo_state.weights * sorted_noise, axis=0)
        grad_sigma = np.sum(evo_state.weights * (sorted_noise**2 - 1), axis=0)

        # 3. Update mean and sigma based on gradients
        mean = evo_state.mean + params.lrate_mean * evo_state.sigma * grad_mean
        sigma = evo_state.sigma * np.exp(params.lrate_sigma / 2 * grad_sigma)

        # 4. Update evo_state
        evo_state.mean = mean if self.is_mean_decay == False else (1 - self.mean_decay) * mean
        evo_state.sigma = sigma
        return evo_state