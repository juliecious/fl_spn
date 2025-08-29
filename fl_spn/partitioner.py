import logging
import random
from typing import Dict

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedDataPartitioner:
    """Partition tabular data for federated learning in horizontal, vertical, or hybrid settings.

    Supports federated splits consistent with the semantics of probabilistic circuits and federated circuits:
    - Horizontal: Same feature space, different (disjoint) samples per client.
    - Vertical: Same samples, non-overlapping feature shards per client.
    - Hybrid: Partial overlaps on both samples and features.

    Args:
        X: 2D array-like of shape (n_samples, n_features). Full feature matrix.
        y: 1D array-like of shape (n_samples,). Corresponding labels/target vector.
        feature_names: List of string feature names matching columns of X.
        numeric_features: List of feature names designated numeric.
        categorical_features: List of feature names designated categorical.
    """

    def __init__(self, X, y, feature_names, numeric_features, categorical_features):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def horizontal_partition(
        self, num_clients: int = 3, random_state: int = 42
    ) -> Dict:
        """Partition samples horizontally to emulate federated learning: same features, different samples.

        Randomly shuffles and splits dataset samples into approximately equal, non-overlapping blocks per client.
        Every client holds the full feature set. This partition type matches horizontal federated learning assumptions.

        Args:
            num_clients: Number of clients.
            random_state: Seed for reproducible shuffling.

        Returns:
            dict: A summary of partitioning, with information and metadata per client.
        """
        logger.info(
            f"🔄 Performing HORIZONTAL partitioning into {num_clients} clients..."
        )

        np.random.seed(random_state)
        n_samples = len(self.X)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        clients = {}
        samples_per_client = n_samples // num_clients

        for i in range(num_clients):
            start_idx = i * samples_per_client
            if i == num_clients - 1:
                end_idx = n_samples
            else:
                end_idx = (i + 1) * samples_per_client

            client_indices = indices[start_idx:end_idx]

            clients[f"client_{i}"] = {
                "X": self.X[client_indices],
                "y": self.y[client_indices],
                "features": self.feature_names,
                "numeric_features": self.numeric_features,
                "categorical_features": self.categorical_features,
                "n_samples": len(client_indices),
                "n_features": len(self.feature_names),
                "sample_indices": client_indices,
                "feature_indices": list(range(len(self.feature_names))),
                "feature_overlap": self.feature_names,  # All clients have the same feature set
            }

            logger.info(
                f"  Client {i}: {len(client_indices)} samples, {len(self.feature_names)} features"
            )

        return {
            "type": "horizontal",
            "clients": clients,
            "total_samples": n_samples,
            "total_features": len(self.feature_names),
        }

    def vertical_partition(self, num_clients: int = 3, random_state: int = 42) -> Dict:
        """Partition features vertically: same samples, mutually exclusive feature subsets per client.

        Randomly shuffles the feature set and assigns unique feature blocks to each client.
        All clients retain access to all samples. This matches vertical federated learning settings.

        Args:
            num_clients: Number of clients.
            random_state: Seed for reproducible shuffling.

        Returns:
            dict: Summary of partitioning, with information and metadata per client.
        """
        logger.info(f"🔄 Performing VERTICAL partitioning into {num_clients} clients...")

        random.seed(random_state)
        all_features = self.feature_names.copy()
        random.shuffle(all_features)

        features_per_client = len(all_features) // num_clients
        clients = {}

        for i in range(num_clients):
            start_idx = i * features_per_client
            if i == num_clients - 1:
                end_idx = len(all_features)
            else:
                end_idx = (i + 1) * features_per_client

            client_features = all_features[start_idx:end_idx]
            client_numeric = [f for f in client_features if f in self.numeric_features]
            client_categorical = [
                f for f in client_features if f in self.categorical_features
            ]

            feature_indices = [self.feature_names.index(f) for f in client_features]

            clients[f"client_{i}"] = {
                "X": self.X[:, feature_indices],
                "y": self.y,
                "features": client_features,
                "numeric_features": client_numeric,
                "categorical_features": client_categorical,
                "n_samples": len(self.X),
                "n_features": len(client_features),
                "feature_indices": feature_indices,
                "sample_indices": list(range(len(self.X))),
                "feature_overlap": [],  # No feature overlap between clients
            }

            logger.info(
                f"  Client {i}: {len(self.X)} samples, {len(client_features)} features"
            )

        return {
            "type": "vertical",
            "clients": clients,
            "total_samples": len(self.X),
            "total_features": len(self.feature_names),
        }

    def hybrid_partition(
        self,
        num_clients: int = 4,
        sample_overlap_ratio: float = 0.3,
        feature_overlap_ratio: float = 0.2,
        random_state: int = 42,
    ) -> Dict:
        """Create a robust hybrid partition: controlled sample and feature overlap across clients.

        For each client, samples consist of a unique subset plus a fraction drawn from an overlapping global pool.
        Similarly, features consist of a unique subset plus a fraction of features overlapped across clients.

        Args:
            num_clients: Number of clients.
            sample_overlap_ratio: Fraction of all samples added to the global overlap set.
            feature_overlap_ratio: Fraction of all features added to the global overlap set.
            random_state: Seed for reproducibility.

        Returns:
            dict: Partitioning info and metadata, including per-client details.

        Notes:
            - Guarantees every client has at least one feature, even in degenerate cases.
            - Overlaps are sampled without replacement when possible.
            - Suitable for studying mixed horizontal/vertical FL.
        """
        logger.info(f"🔄 Performing HYBRID partitioning into {num_clients} clients...")
        logger.info(f"  Sample overlap ratio: {sample_overlap_ratio * 100:.1f}%")
        logger.info(f"  Feature overlap ratio: {feature_overlap_ratio * 100:.1f}%")

        np.random.seed(random_state)
        random.seed(random_state)

        n_samples = len(self.X)
        n_features = len(self.feature_names)

        # FIXME: refactor into util function
        # --- Sample allocation: base samples for each client and a global overlap pool.
        base_samples_per_client = max(1, int(n_samples * 0.5 / num_clients))
        overlap_sample_count = int(n_samples * sample_overlap_ratio)
        all_sample_indices = np.arange(n_samples, dtype=int)
        np.random.shuffle(all_sample_indices)

        base_samples_end = min(base_samples_per_client * num_clients, n_samples)
        base_sample_indices = all_sample_indices[:base_samples_end]

        if overlap_sample_count > 0 and base_samples_end < n_samples:
            remaining_samples = all_sample_indices[base_samples_end:]
            overlap_sample_indices = remaining_samples[
                : min(overlap_sample_count, len(remaining_samples))
            ]
        else:
            overlap_sample_indices = np.array([], dtype=int)

        # --- Feature allocation: base features for each client and a global overlap pool.
        base_features_per_client = max(1, int(n_features * 0.6 / num_clients))
        overlap_feature_count = int(n_features * feature_overlap_ratio)
        all_feature_indices = np.arange(n_features, dtype=int)
        np.random.shuffle(all_feature_indices)

        base_features_end = min(base_features_per_client * num_clients, n_features)
        base_feature_indices = all_feature_indices[:base_features_end]

        if overlap_feature_count > 0 and base_features_end < n_features:
            remaining_features = all_feature_indices[base_features_end:]
            overlap_feature_indices = remaining_features[
                : min(overlap_feature_count, len(remaining_features))
            ]
        else:
            overlap_feature_indices = np.array([], dtype=int)

        logger.info(
            f"  Base samples: {len(base_sample_indices)}; Overlap sample pool: {len(overlap_sample_indices)}"
        )
        logger.info(
            f"  Base features: {len(base_feature_indices)}; Overlap feature pool: {len(overlap_feature_indices)}"
        )

        # Distribute data to each clients
        clients = {}

        for i in range(num_clients):
            # --- Assign sample indices: each client gets unique base samples and some overlap samples.
            client_base_start = i * base_samples_per_client
            client_base_end = min(
                (i + 1) * base_samples_per_client, len(base_sample_indices)
            )
            client_base_samples = base_sample_indices[client_base_start:client_base_end]

            if len(overlap_sample_indices) > 0:
                overlap_sample_size = min(
                    len(overlap_sample_indices), len(client_base_samples) // 2
                )
                if overlap_sample_size > 0:
                    client_overlap_samples = np.random.choice(
                        overlap_sample_indices, size=overlap_sample_size, replace=False
                    )
                else:
                    client_overlap_samples = np.array([], dtype=int)
            else:
                client_overlap_samples = np.array([], dtype=int)

            if len(client_overlap_samples) > 0:
                client_sample_indices = np.concatenate(
                    [client_base_samples, client_overlap_samples]
                )
            else:
                client_sample_indices = client_base_samples.copy()
            client_sample_indices = np.unique(client_sample_indices)

            # --- Assign feature indices: each client gets unique base features + overlap features.
            client_base_feat_start = i * base_features_per_client
            client_base_feat_end = min(
                (i + 1) * base_features_per_client, len(base_feature_indices)
            )
            client_base_features = base_feature_indices[
                client_base_feat_start:client_base_feat_end
            ]

            if len(overlap_feature_indices) > 0:
                guaranteed_overlap_size = max(1, len(overlap_feature_indices) // 2)
                guaranteed_overlap_features = overlap_feature_indices[
                    :guaranteed_overlap_size
                ]
                client_overlap_features = guaranteed_overlap_features
            else:
                client_overlap_features = np.array([], dtype=int)

            if len(client_overlap_features) > 0:
                client_feature_indices = np.concatenate(
                    [client_base_features, client_overlap_features]
                )
            else:
                client_feature_indices = client_base_features.copy()
            client_feature_indices = np.unique(client_feature_indices.astype(int))

            # Every client gets at least 1 feature
            if len(client_feature_indices) == 0:
                client_feature_indices = np.array([0], dtype=int)

            # Get feature name
            client_features = [
                self.feature_names[idx] for idx in client_feature_indices
            ]
            client_numeric = [f for f in client_features if f in self.numeric_features]
            client_categorical = [
                f for f in client_features if f in self.categorical_features
            ]

            # Calculate overlapped features
            overlap_features_names = []
            if len(client_overlap_features) > 0:
                overlap_features_names = [
                    self.feature_names[idx] for idx in client_overlap_features
                ]

            clients[f"client_{i}"] = {
                "X": self.X[np.ix_(client_sample_indices, client_feature_indices)],
                "y": self.y[client_sample_indices],
                "features": client_features,
                "numeric_features": client_numeric,
                "categorical_features": client_categorical,
                "n_samples": len(client_sample_indices),
                "n_features": len(client_features),
                "feature_indices": client_feature_indices,
                "sample_indices": client_sample_indices,
                "base_sample_count": len(client_base_samples),
                "overlap_sample_count": len(client_overlap_samples),
                "base_feature_count": len(client_base_features),
                "overlap_feature_count": len(client_overlap_features),
                "feature_overlap": overlap_features_names,
            }

            logger.info(
                f"  Client {i}: {len(client_sample_indices)} samples × {len(client_features)} features"
            )

        return {
            "type": "hybrid_robust",
            "clients": clients,
            "total_samples": n_samples,
            "total_features": n_features,
            "sample_overlap_ratio": sample_overlap_ratio,
            "feature_overlap_ratio": feature_overlap_ratio,
        }
