"""Model-specific coordinates; training and evaluation stay model independent."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    path: str
    conditions: tuple[str, ...]
    observable: str | tuple[str, ...]
    native_id: int
    method: str
    nonnegative: bool = True

    @property
    def observables(self):
        return (self.observable,) if isinstance(self.observable, str) else self.observable

    @property
    def fixed(self):
        return {'spot': 1.0} if self.observable == 'spot' and 'spot' not in self.conditions else {}


MODELS = {
    'heston': Model('equity/markovian/heston',
        ('kappa', 'gamma', 'theta', 'rho', 'initial_variance', 'T'), 'spot', 0,
        'Heston QE-M, dt=1/504'),
    'bates': Model('equity/markovian/bates',
        ('kappa', 'gamma', 'theta', 'rho', 'initial_variance', 'jump_intensity',
         'jump_log_mean', 'jump_log_volatility', 'T'), 'spot', 1,
        'Heston QE-M with terminal aggregate jumps, dt=1/504'),
    'rough_heston': Model('equity/rough/rough_heston',
        ('initial_variance', 'mean_reversion', 'variance_drift',
         'volatility_of_variance', 'hurst_exponent', 'rho', 'T'), 'spot', 2,
        '7-factor lift, exact device preparation at horizon 2 years, dt=1/504'),
    'cir': Model('fixed_income/cir',
        ('mean_reversion', 'long_term_mean', 'volatility', 'initial_state', 'T'),
        'state', 3, 'Exact terminal noncentral chi-square transition'),
    'black_scholes': Model('equity/markovian/black_scholes', ('volatility', 'T'), 'spot', -1, 'Exact terminal transition', True),
    'cev': Model('equity/markovian/cev', ('spot', 'sigma', 'beta', 'T'), 'spot', -1, 'Source sample dynamics, dt=1/504', True),
    'heston_3_2': Model('equity/markovian/heston_3_2', ('initial_variance', 'mean_reversion', 'long_run_variance', 'volatility_of_variance', 'rho', 'T'), 'spot', -1, 'Source sample dynamics, dt=1/504', True),
    'kou': Model('equity/markovian/kou', ('volatility', 'jump_intensity', 'up_probability', 'positive_jump_rate', 'negative_jump_rate', 'T'), 'spot', -1, 'Exact terminal transition', True),
    'merton': Model('equity/markovian/merton', ('volatility', 'jump_intensity', 'jump_log_mean', 'jump_log_volatility', 'T'), 'spot', -1, 'Exact terminal transition', True),
    'normal_inverse_gaussian': Model('equity/markovian/normal_inverse_gaussian', ('alpha', 'beta', 'delta', 'T'), 'spot', -1, 'Exact terminal transition', True),
    'sabr': Model('equity/markovian/sabr', ('spot', 'initial_volatility', 'volatility_of_volatility', 'rho', 'beta', 'T'), 'spot', -1, 'Source sample dynamics, dt=1/504', True),
    'schobel_zhu': Model('equity/markovian/schobel_zhu', ('initial_volatility', 'mean_reversion', 'long_run_volatility', 'volatility_of_volatility', 'correlation', 'T'), 'spot', -1, 'Source sample dynamics, dt=1/504', True),
    'variance_gamma': Model('equity/markovian/variance_gamma', ('sigma', 'nu', 'theta', 'T'), 'spot', -1, 'Exact terminal transition', True),
    'log_modulated_rough_bergomi': Model('equity/rough/log_modulated_rough_bergomi', ('xi_0', 'eta', 'hurst_exponent', 'rho', 'log_modulation_scale', 'log_modulation_power', 'T'), 'spot', -1, 'Source Volterra FFT terminal sample pipeline, dt=1/504', True),
    'quadratic_rough_heston': Model('equity/rough/quadratic_rough_heston', ('initial_feedback', 'quadratic_scale', 'quadratic_shift', 'variance_floor', 'feedback_rate', 'feedback_volatility', 'hurst_exponent', 'T'), 'spot', -1, '7-factor lift, exact device preparation at horizon 2 years, dt=1/504', True),
    'rough_bergomi': Model('equity/rough/rough_bergomi', ('xi_0', 'eta', 'hurst_exponent', 'rho', 'T'), 'spot', -1, 'Source Volterra FFT terminal sample pipeline, dt=1/504', True),
    'rough_sabr': Model('equity/rough/rough_sabr', ('spot', 'xi_0', 'eta', 'hurst_exponent', 'rho', 'beta', 'T'), 'spot', -1, 'Source Volterra FFT terminal sample pipeline, dt=1/504', True),
    'rough_stein_stein': Model('equity/rough/rough_stein_stein', ('initial_volatility', 'volatility_level', 'mean_reversion', 'volatility_of_volatility', 'hurst_exponent', 'rho', 'T'), 'spot', -1, 'Linear resolvent hybrid FFT with independent initial volatility and long-run level, dt=1/504', True),
    'g2': Model('fixed_income/g2', ('mean_reversion_x', 'volatility_x', 'mean_reversion_y', 'volatility_y', 'correlation', 'initial_state_x', 'initial_state_y', 'T'), ('state_x', 'state_y'), -1, 'Exact terminal transition', False),
    'ornstein_uhlenbeck': Model('fixed_income/ornstein_uhlenbeck', ('mean_reversion', 'volatility', 'initial_state', 'T'), 'state', -1, 'Exact terminal transition', False),
    'vasicek': Model('fixed_income/vasicek', ('mean_reversion', 'long_term_mean', 'volatility', 'initial_state', 'T'), 'state', -1, 'Exact terminal transition', False),
}


def model_for_path(path):
    text = str(path).replace('\\', '/')
    for name, model in MODELS.items():
        if f'/samples/{model.path}/' in '/' + text:
            return name, model
    raise ValueError('No qualified native-reference adapter for this dataset; supported: '
                     + ', '.join(MODELS))
