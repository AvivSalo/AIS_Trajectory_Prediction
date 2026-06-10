from .MTR_dataset import MTRDataset
from .autobot_dataset import AutoBotDataset
from .wayformer_dataset import WayformerDataset
from .ais_dataset import AISDataset

__all__ = {
    'autobot': AutoBotDataset,
    'wayformer': WayformerDataset,
    'wayformer_ais': AISDataset,  # AIS-specific wayformer dataset
    'MTR': MTRDataset,
    'traisformer': AISDataset,    # TrAISformer baseline shares the AIS dataset
    'ais_acnet': AISDataset,      # AIS-ACNet baseline shares the AIS dataset
    'gat_lstm': AISDataset,       # GAT-LSTM baseline shares the AIS dataset
}


def build_dataset(config, val=False):
    dataset = __all__[config.method.model_name](
        config=config, is_validation=val
    )
    return dataset
