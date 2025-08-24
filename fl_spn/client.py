import numpy as np

from config import SupervisedFLConfig, TaskType
from typing import Optional, Dict
from simple_einet.einet import Einet
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedClient:
    """Federated client"""

    def __init__(self, client_id: int, config: SupervisedFLConfig):
        self.client_id = client_id
        self.config = config
        self.model: Optional[Einet] = None
        self.training_history = []
        self.client_data = None

    def receive_data(self, partition_data: Dict):
        """Receive partitioned data"""
        self.client_data = partition_data

        # create einet
        local_features = partition_data["X"].shape[1]
        einet_config = SupervisedFLConfig(
            num_features=local_features,
            num_classes=self.config.num_classes,
            depth=self.config.depth,
            num_leaves=self.config.num_leaves,
            num_repetitions=self.config.num_repetitions,
            task_type=self.config.task_type,
        ).create_einet_config(num_features=local_features)

        self.model = Einet(einet_config)
        logger.info(
            f"Client {self.client_id}: Initialized EiNet with {local_features} features, {self.config.num_classes} classes"
        )

    def local_training(self) -> Dict:
        if self.model is None or self.client_data is None:
            raise ValueError(f"Client {self.client_id}: No model or data available")

        X, y = self.client_data["X"], self.client_data["y"]

        logger.info(f"Client {self.client_id}: Starting training on {X.shape} data")

        initial_ll = (
            self.model.log_likelihood(X, y) if self.model.trained else float("-inf")
        )

        self.model.fit(
            X, y, epochs=self.config.local_epochs, lr=self.config.learning_rate
        )

        final_ll = self.model.log_likelihood(X, y)

        accuracy = 0.0
        if self.config.task_type == TaskType.CLASSIFICATION:
            predictions = self.model.predict(X)
            accuracy = np.mean(predictions == y)

        training_result = {
            "client_id": self.client_id,
            "model_parameters": self.model.get_parameters(),
            "data_shape": X.shape,
            "label_distribution": np.bincount(
                y, minlength=self.config.num_classes
            ).tolist(),
            "feature_indices": self.client_data["feature_indices"],
            "sample_indices": self.client_data["sample_indices"],
            "partition_type": self.client_data["partition_type"],
            "initial_log_likelihood": initial_ll,
            "final_log_likelihood": final_ll,
            "training_improvement": final_ll - initial_ll
            if initial_ll != float("-inf")
            else final_ll,
            "accuracy": accuracy,
            "num_samples": len(self.client_data["sample_indices"]),
            "num_features": len(self.client_data["feature_indices"]),
        }

        self.training_history.append(training_result)
        logger.info(
            f"Client {self.client_id}: Training completed. LL: {final_ll:.4f}, Acc: {accuracy:.4f}"
        )

        return training_result
