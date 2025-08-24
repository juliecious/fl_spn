from typing import Dict, List, Tuple, Union, Optional
from fl_spn.config import SupervisedFLConfig, FLMode
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedDataPartitioner:
    """Handles data partitioning for different FL modes"""

    def __init__(self, config: SupervisedFLConfig):
        self.config = config
        np.random.seed(config.random_seed)

    def partition_data(self, X: np.ndarray, y: np.ndarray) -> Dict[int, Dict]:
        """Partition data according to FL mode"""
        if self.config.fl_mode == FLMode.HORIZONTAL:
            return self._horizontal_partition(X, y)
        elif self.config.fl_mode == FLMode.VERTICAL:
            return self._vertical_partition(X, y)
        else:  # HYBRID
            return self._hybrid_partition(X, y)

    def _horizontal_partition(self, X: np.ndarray, y: np.ndarray) -> Dict[int, Dict]:
        """Horizontal FL: same features, different samples"""
        n_samples = X.shape[0]
        partitions = {}

        if self.config.label_skew == 0.0:
            # 均匀分布
            indices = np.random.permutation(n_samples)
            samples_per_client = n_samples // self.config.num_clients

            for i in range(self.config.num_clients):
                start_idx = i * samples_per_client
                end_idx = (
                    (i + 1) * samples_per_client
                    if i < self.config.num_clients - 1
                    else n_samples
                )

                client_indices = indices[start_idx:end_idx]
                partitions[i] = {
                    "X": X[client_indices],
                    "y": y[client_indices],
                    "client_id": i,
                    "sample_indices": client_indices.tolist(),
                    "feature_indices": list(range(X.shape[1])),
                    "partition_type": "horizontal_uniform",
                }
        else:
            # 非均匀分布：标签偏斜
            classes = np.unique(y)

            for i in range(self.config.num_clients):
                # 主要类别
                primary_class = classes[i % len(classes)]

                client_indices = []

                # 分配主要类别样本
                cls_indices = np.where(y == primary_class)[0]
                n_primary = int(
                    len(cls_indices)
                    * (0.6 + 0.3 * self.config.label_skew)
                    / self.config.num_clients
                )
                if len(cls_indices) > 0:
                    selected = np.random.choice(
                        cls_indices, min(n_primary, len(cls_indices)), replace=False
                    )
                    client_indices.extend(selected)

                # 分配次要类别样本
                for cls in classes:
                    if cls != primary_class:
                        cls_indices = np.where(y == cls)[0]
                        n_secondary = max(
                            1,
                            int(
                                len(cls_indices)
                                * (0.4 - 0.3 * self.config.label_skew)
                                / self.config.num_clients
                            ),
                        )

                        available_indices = [
                            idx for idx in cls_indices if idx not in client_indices
                        ]
                        if available_indices:
                            selected = np.random.choice(
                                available_indices,
                                min(n_secondary, len(available_indices)),
                                replace=False,
                            )
                            client_indices.extend(selected)

                client_indices = np.array(client_indices)

                partitions[i] = {
                    "X": X[client_indices],
                    "y": y[client_indices],
                    "client_id": i,
                    "sample_indices": client_indices.tolist(),
                    "feature_indices": list(range(X.shape[1])),
                    "partition_type": f"horizontal_skewed_{self.config.label_skew}",
                }

        logger.info(f"Horizontal partitioning: {self.config.num_clients} clients")
        for i, partition in partitions.items():
            y_dist = np.bincount(partition["y"], minlength=len(np.unique(y)))
            logger.info(
                f"  Client {i}: {len(partition['y'])} samples, label dist: {y_dist}"
            )

        return partitions

    def _vertical_partition(self, X: np.ndarray, y: np.ndarray) -> Dict[int, Dict]:
        """Vertical FL: same samples, different features"""
        n_features = X.shape[1]
        features_per_client = n_features // self.config.num_clients

        partitions = {}
        for i in range(self.config.num_clients):
            start_idx = i * features_per_client
            end_idx = (
                (i + 1) * features_per_client
                if i < self.config.num_clients - 1
                else n_features
            )

            feature_indices = list(range(start_idx, end_idx))

            partitions[i] = {
                "X": X[:, feature_indices],
                "y": y.copy(),
                "client_id": i,
                "sample_indices": list(range(X.shape[0])),
                "feature_indices": feature_indices,
                "partition_type": "vertical",
            }

        logger.info(f"Vertical partitioning: {self.config.num_clients} clients")
        return partitions

    def _hybrid_partition(self, X: np.ndarray, y: np.ndarray) -> Dict[int, Dict]:
        """Hybrid FL: both samples and features partitioned"""
        n_samples, n_features = X.shape

        partitions = {}
        for i in range(self.config.num_clients):
            # 样本和特征范围
            samples_per_client = n_samples // self.config.num_clients
            features_per_client = n_features // self.config.num_clients

            sample_start = i * samples_per_client
            sample_end = (
                (i + 1) * samples_per_client
                if i < self.config.num_clients - 1
                else n_samples
            )

            feature_start = i * features_per_client
            feature_end = (
                (i + 1) * features_per_client
                if i < self.config.num_clients - 1
                else n_features
            )

            sample_indices = list(range(sample_start, sample_end))
            feature_indices = list(range(feature_start, feature_end))

            partitions[i] = {
                "X": X[np.ix_(sample_indices, feature_indices)],
                "y": y[sample_indices],
                "client_id": i,
                "sample_indices": sample_indices,
                "feature_indices": feature_indices,
                "partition_type": "hybrid",
            }

        logger.info(f"Hybrid partitioning: {self.config.num_clients} clients")
        return partitions


if __name__ == "__main__":
    pass
