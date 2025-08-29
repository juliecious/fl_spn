from dataclasses import dataclass
from enum import Enum
from simple_einet.layers.distributions.normal import Normal


@dataclass
class SupervisedFLConfig:
    """Supervised Federated SPN Configuration"""

    # FL settings
    num_features: int
    num_classes: int
    num_clients: int = 3

    # EiNet settings
    depth: int = 2
    num_sums: int = 2
    num_leaves: int = 4
    num_repetitions: int = 2

    # Training settings
    epochs: int = 10
    learning_rate: float = 0.01
    dropout: float = 0.0
    leaf_type = Normal
    leaf_kwargs = {}

    # Data partition settings
    label_skew: float = 0.0  # 0 = even; 1 = skewed
    feature_overlap_ratio: float = 0.0
    sample_overlap_ratio: float = 0.0

    random_seed: int = 42
