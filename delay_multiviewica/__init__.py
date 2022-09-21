# Authors: Hugo Richard, Pierre Ablin
# License: BSD 3 clause

from ._delay_multiviewica import delay_multiviewica
from ._multiviewica import multiviewica
from ._groupica import groupica
from ._permica import permica, _hungarian
from .optimization_tau import _optimization_tau, _loss_delay, _apply_delay, _apply_delay_one_sub, _loss_delay_ref, _optimization_tau_approach1, _optimization_tau_approach2, _delay_estimation
from .sources_generation import _create_sources, create_sources_pierre, create_model, generate_data, plot_sources, _plot_delayed_sources
from ._univiewica import univiewica

__version__ = '0.1'
