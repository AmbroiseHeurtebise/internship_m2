# Authors: Hugo Richard, Pierre Ablin
# License: BSD 3 clause

from ._multiviewica_delay import multiviewica_delay, _noisy_ica_step
from ._multiviewica import multiviewica
from ._groupica import groupica
from ._sameica import sameica
from ._permica import permica, _hungarian
from .optimization_tau import (
    _optimization_tau,
    _apply_delay,
    _apply_delay_one_sub,
    _optimization_tau_approach1,
    _optimization_tau_approach2,
    _delay_estimation,
    _optimization_tau_with_f,
    _apply_delay_by_source,
    _apply_delay_one_source_or_sub,
    _optimization_tau_by_source,
    _optimization_tau_continuous_delays,
    _apply_continuous_delays,
)
from .sources_generation import (
    _create_sources,
    create_sources_pierre,
    create_model,
    generate_data,
    data_generation,
    data_generation_pierre,
    plot_sources,
    _plot_delayed_sources,
)
from ._univiewica import univiewica

__version__ = "0.1"
