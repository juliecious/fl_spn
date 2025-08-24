from dataclasses import dataclass
from enum import Enum
from simple_einet.einet import EinetConfig
from simple_einet.layers.distributions.normal import Normal


class FLMode(Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    HYBRID = "hybrid"


class TaskType(Enum):
    CLASSIFICATION = "classification"
    DENSITY_ESTIMATION = "density_estimation"


@dataclass
class SupervisedFLConfig:
    """Supervised Federated EiNet Configuration"""

    # FL settings
    fl_mode: FLMode
    task_type: TaskType
    num_clients: int
    num_features: int
    num_classes: int

    # EiNet settings
    depth: int = 2
    num_sums: int = 2
    num_leaves: int = 4
    num_repetitions: int = 2

    # Training settings
    local_epochs: int = 10
    learning_rate: float = 0.01
    dropout: float = 0.0
    leaf_type = Normal
    leaf_kwargs = {}

    # Data partition settings
    label_skew: float = 0.0  # 0 = even; 1 = skewed
    feature_overlap_ratio: float = 0.0
    sample_overlap_ratio: float = 0.0

    random_seed: int = 42

    def create_einet_config(self, num_features: int) -> EinetConfig:
        """Create EiNet config for local models"""
        return EinetConfig(
            num_features=num_features,
            depth=self.depth,
            num_sums=self.num_sums,
            num_leaves=self.num_leaves,
            num_repetitions=self.num_repetitions,
            leaf_type=self.leaf_type,
            leaf_kwargs=self.leaf_kwargs,
            dropout=self.dropout,
        )


if __name__ == "__main__":
    from utils import load_adult_income_dataset

    adult_data = load_adult_income_dataset(test_size=0.2)
    adult_horizontal_uniform_config = SupervisedFLConfig(
        fl_mode=FLMode.HORIZONTAL,
        task_type=TaskType.CLASSIFICATION,
        num_clients=4,
        num_features=adult_data["n_features"],
        num_classes=adult_data["n_classes"],
        depth=3,
        num_leaves=4,
        num_repetitions=3,
        local_epochs=15,
        learning_rate=0.015,
        label_skew=0.0,  # normal dist
        random_seed=42,
    ).create_einet_config(num_features=adult_data["n_features"])
    print(adult_horizontal_uniform_config)
