from unitraj.models.autobot.autobot import AutoBotEgo
from unitraj.models.baseline_linear.baseline_linear import BaselineLinear
from unitraj.models.mtr.MTR import MotionTransformer
from unitraj.models.traisformer.traisformer import TrAISformer
from unitraj.models.wayformer.wayformer import Wayformer

__all__ = {
    'autobot': AutoBotEgo,
    'wayformer': Wayformer,
    'wayformer_ais': Wayformer,  # Use same Wayformer model for AIS data
    'MTR': MotionTransformer,
    'baseline_linear': BaselineLinear,
    'traisformer': TrAISformer,
}


def build_model(config):
    model = __all__[config.method.model_name](
        config=config
    )

    return model
