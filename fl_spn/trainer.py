import logging
from typing import Dict, List
import time
import torch
import numpy as np
import pandas as pd
from simple_einet.einet import EinetConfig, Einet
from sklearn.metrics import accuracy_score, f1_score
from simple_einet.layers.distributions.piecewise_linear import PiecewiseLinear
from simple_einet.dist import DataType, Domain

from config import SupervisedFLConfig
from utils import _accuracy, _f1_score

logger = logging.getLogger(__name__)


class FederatedEiNetTrainer:
    """
    Trains and manages probabilistic circuit (EiNet) models for each client in a federated learning setting.

    This trainer coordinates the full federated workflow: partitioning data across clients, creating feature domains,
    training an independent probabilistic circuit model for each client, aggregating results, and evaluating both
    individual and ensemble model performance.

    Intended for experiments on different data splits such as horizontal, vertical, and hybrid partitioning.

    Attributes:
        partition_info (Dict): Partitioning metadata and data for all clients.
        client_models (Dict): Trained EiNet models for each client.
        client_domains (Dict): Domains for features on each client.
        training_history (Dict): Optional storage for training history per client.
    """

    def __init__(self, partition_info: Dict):
        self.partition_info = partition_info
        self.client_models = {}
        self.client_domains = {}
        self.training_history = {}

    def create_domains(
        self,
        features: List[str],
        numeric_features: List[str],
        X_processed: pd.DataFrame,
    ) -> List:
        """
        Create domains for a given list of features.

        Numeric features use their min and max value from the provided pandas DataFrame if available,
        otherwise default to [-3.0, 3.0]. Categorical features use sorted unique values, or [0, 1] if missing.

        Args:
            features: List of feature names.
            numeric_features: List of features considered numeric.
            X_processed: DataFrame containing feature values.

        Returns:
            List of Domain instances corresponding to each feature.
        """
        domains = []

        for feature in features:
            if feature in numeric_features:
                if feature in X_processed.columns:
                    min_val = float(X_processed[feature].min())
                    max_val = float(X_processed[feature].max())
                    domains.append(Domain.continuous_range(min_val, max_val))
                else:
                    # Use default range for missing numeric features
                    domains.append(Domain.continuous_range(-3.0, 3.0))
            else:
                if feature in X_processed.columns:
                    values = sorted(X_processed[feature].unique().tolist())
                    domains.append(Domain.discrete_bins(values))
                else:
                    # Use default bin values for missing categorical features
                    domains.append(Domain.discrete_bins([0, 1]))

        return domains

    def train_client(
        self,
        client_id: str,
        client_data: Dict,
        X_processed: pd.DataFrame,
        epochs: int = SupervisedFLConfig.epochs,
        verbose: bool = False,
    ) -> Dict:
        """
        Train an EiNet model on a specific client's data.

        The model depth and other hyperparameters are set dynamically based on feature count.
        Domains are constructed for this client's features. Training and performance metrics are returned.

        Args:
            client_id: The identifier of the client.
            client_data: Dictionary containing client-specific features, X, y, etc.
            X_processed: The processed DataFrame used to get domains.
            epochs: Number of training epochs.
            verbose: If True, log training progress.

        Returns:
            dict: Results containing model, domains, accuracy, F1, elapsed time, and config.
        """

        X_client = client_data["X"]
        y_client = client_data["y"]

        X_client_reshaped = X_client.unsqueeze(1)

        # FIXME: not always piecewise leaf type; should adjust later on
        # Create the domains for this client's features needed for Piecewise leaf type
        domains = self.create_domains(
            client_data["features"],
            client_data["numeric_features"],
            X_processed,
        )

        num_features = client_data["n_features"]

        # Dynamically adjust model complexity based on the number of features
        if num_features < 3:
            depth, num_sums, num_leaves = 1, 4, 4
        elif num_features < 6:
            depth, num_sums, num_leaves = 1, 8, 8
        else:
            depth, num_sums, num_leaves = 2, 12, 12

        config = EinetConfig(
            num_features=num_features,
            depth=depth,
            num_sums=num_sums,
            num_leaves=num_leaves,
            num_repetitions=3,
            num_classes=2,
            leaf_type=PiecewiseLinear,  # FIXME: adjust later
            leaf_kwargs={"alpha": 0.1},
            dropout=0.0,
        )

        model = Einet(config)
        model.leaf.base_leaf.initialize(
            X_client_reshaped, domains
        )  # FIXME: adjust later

        criterion = torch.nn.CrossEntropyLoss()  # FIXME: adjust later
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # FIXME: adjust later

        if verbose:
            logger.info(
                f"    📊 Model config: depth={depth}, sums={num_sums}, leaves={num_leaves}"
            )
            logger.info(f"    🔧 Feature domains: {len(domains)} domain created")

        start_time = time.time()

        for epoch in range(epochs):
            optimizer.zero_grad()
            ll = model(X_client_reshaped)
            loss = criterion(ll, y_client)
            loss.backward()
            optimizer.step()

            if verbose and (epoch + 1) % 5 == 0:
                acc_train = _accuracy(model, X_client_reshaped, y_client)
                f1_train = _f1_score(model, X_client_reshaped, y_client)
                logger.info(
                    f"Epoch: {epoch + 1:2d}, Loss: {loss.item():.4f}, "
                    f"Train Acc: {acc_train:.2f}%, F1: {f1_train:.2f}%"
                )

        training_time = time.time() - start_time
        train_accuracy = _accuracy(model, X_client_reshaped, y_client)
        train_f1 = _f1_score(model, X_client_reshaped, y_client)

        if verbose:
            logger.info(f"    ✅ Training accuracy: {train_accuracy:.3f}")
            logger.info(f"    📈 Training F1 score: {train_f1:.3f}")
            logger.info(f"    ⏱️ Training time: {training_time:.3f} seconds")

        return {
            "client_id": client_id,
            "model": model,
            "domains": domains,
            "train_accuracy": train_accuracy,
            "train_f1": train_f1,
            "training_time": training_time,
            "config": config,
        }

    def train_federated_learning(
        self,
        X_processed: pd.DataFrame,
        epochs: int = SupervisedFLConfig.epochs,
        verbose: bool = True,
    ) -> Dict:
        """
        Run federated training across all clients specified in the partition.

        Trains individual EiNet models on each client's data and tracks progress using weighted accuracy
        and F1 across the federation. Training statistics, accuracy, and F1 are logged.

        Args:
            X_processed: DataFrame for extracting feature value ranges (domains).
            epochs: Training epochs for each client.
            verbose: If True, log information about training and results.

        Returns:
            dict: Aggregated statistics and per-client results for the entire federation.
        """

        logger.info(f"\n🚀 Starting {self.partition_info['type']} federated learning...")
        logger.info(f"Training parameters: epochs={epochs}")

        start_time = time.time()
        results = {}

        # Iterate through all clients and train their local models
        for client_id, client_data in self.partition_info["clients"].items():
            if verbose:
                logger.info(f"\n📍 Training client {client_id}...")
                logger.info(
                    f"   Data shape: {client_data['n_samples']} samples × {client_data['n_features']} features"
                )
                if client_data.get("feature_overlap"):
                    logger.info(
                        f"   🔗 Overlapping features: {len(client_data['feature_overlap'])} feature(s)- {client_data['feature_overlap']}"
                    )

            # Train local model
            client_result = self.train_client(
                client_id, client_data, X_processed, epochs, verbose=verbose
            )

            # Store results and models
            self.client_models[client_id] = client_result["model"]
            self.client_domains[client_id] = client_result["domains"]

            results[client_id] = {
                "train_accuracy": client_result["train_accuracy"],
                "train_f1": client_result["train_f1"],
                "training_time": client_result["training_time"],
                "n_samples": client_data["n_samples"],
                "n_features": client_data["n_features"],
                "feature_overlap": client_data.get("feature_overlap", []),
                "config": client_result["config"],
                "domains_count": len(client_result["domains"]),
            }

            if verbose:
                logger.info(f"   🎯 Client {client_id} training complete")

        # Compute federation-level statistics
        total_samples = sum(r["n_samples"] for r in results.values())
        weighted_accuracy = (
            sum(r["train_accuracy"] * r["n_samples"] for r in results.values())
            / total_samples
        )

        weighted_f1 = (
            sum(r["train_f1"] * r["n_samples"] for r in results.values())
            / total_samples
        )

        total_time = time.time() - start_time

        if verbose:
            logger.info(
                f"\n📊 {self.partition_info['type']} federated learning complete!"
            )
            logger.info(f"   ⏱️ Total training time: {total_time:.2f} seconds")
            logger.info(
                f"   🎯 Weighted average training accuracy: {weighted_accuracy:.3f}"
            )
            logger.info(f"   📈 Weighted average training F1 score: {weighted_f1:.3f}")
            logger.info(f"   🏢 Number of participating clients:  {len(results)}")
            logger.info(f"   📊 Total samples: {total_samples}")

        return {
            "type": self.partition_info["type"],
            "client_results": results,
            "weighted_accuracy": weighted_accuracy,
            "weighted_f1": weighted_f1,
            "total_training_time": total_time,
            "total_samples": total_samples,
            "num_clients": len(results),
        }

    def evaluate_on_test(self, X_test, y_test, test_feature_names) -> Dict:
        """
        Evaluate the federated model ensemble on a test dataset.

        Each client's model predicts using only the features it trained on. Individual predictions are collected;
        ensemble voting and mean probability ensemble are computed. All results and ensemble statistics are logged.

        Args:
            X_test: The test feature matrix (numpy array).
            y_test: The test target labels.
            test_feature_names: List of test feature names.

        Returns:
            dict: Client-level evaluations and ensemble results, including best ensemble accuracy and F1.
        """

        logger.info("\n📋 Evaluating federated EiNet models on test set...")

        client_evaluations = {}
        predictions_ensemble = []
        probabilities_ensemble = []

        for client_id, model in self.client_models.items():
            client_data = self.partition_info["clients"][client_id]

            # Find feature indices in the test set for this client
            client_feature_indices = []
            for feature in client_data["features"]:
                if feature in test_feature_names:
                    client_feature_indices.append(test_feature_names.index(feature))

            if len(client_feature_indices) == 0:
                logger.warning(
                    f"   ⚠️  Client {client_id}: No matching features in test set"
                )
                continue

            X_test_client = X_test[:, client_feature_indices]
            X_test_client_reshaped = X_test_client.unsqueeze(1)

            try:
                acc = _accuracy(model, X_test_client_reshaped, y_test)
                fscore = _f1_score(model, X_test_client_reshaped, y_test)

                probs = torch.exp(model(X_test_client_reshaped))
                predictions = probs.argmax(dim=-1)

                client_evaluations[client_id] = {
                    "accuracy": acc,
                    "f1_score": fscore,
                    "n_test_features": len(client_feature_indices),
                    "predictions": predictions,
                }

                predictions_ensemble.append(predictions.detach().numpy())
                probabilities_ensemble.append(probs.detach().numpy())

                logger.info(
                    f"   Client {client_id}: Accuracy {acc:.3f}, F1 {fscore:.3f} ({len(client_feature_indices)} features)"
                )

            except Exception as e:
                logger.error(f"   ❌ Client{client_id}: Evaluation failed - {str(e)}")

        total_test_samples = len(y_test)
        total_clients = len(self.partition_info["clients"].items())

        weighted_accuracy = np.mean(
            [r["accuracy"] for r in client_evaluations.values()]
        )

        weighted_f1 = np.mean([r["f1_score"] for r in client_evaluations.values()])

        logger.info(f"\n📊 {self.partition_info['type']} federated learning complete!")
        logger.info(f"   🎯 Weighted average testing accuracy: {weighted_accuracy:.3f}")
        logger.info(f"   📈 Weighted average testing F1 score: {weighted_f1:.3f}")
        logger.info(f"   🏢 Number of participating clients:  {total_clients}")
        logger.info(f"   📊 Total test samples: {total_test_samples}")

        # Ensemble predictions (majority voting and average probability)
        if predictions_ensemble and probabilities_ensemble:

            # Majority voting
            predictions_array = np.array(predictions_ensemble)
            ensemble_predictions_vote = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=0, arr=predictions_array
            )

            # Average probability
            ensemble_probabilities = np.mean(probabilities_ensemble, axis=0)
            ensemble_predictions_prob = np.argmax(ensemble_probabilities, axis=1)

            vote_accuracy = accuracy_score(y_test, ensemble_predictions_vote)
            vote_f1 = f1_score(y_test, ensemble_predictions_vote, average="weighted")
            #
            prob_accuracy = accuracy_score(y_test, ensemble_predictions_prob)
            prob_f1 = f1_score(y_test, ensemble_predictions_prob, average="weighted")

            if prob_accuracy >= vote_accuracy:
                ensemble_accuracy = prob_accuracy
                ensemble_f1 = prob_f1
                ensemble_predictions = ensemble_predictions_prob
                ensemble_method = "probability averaging"
            else:
                ensemble_accuracy = vote_accuracy
                ensemble_f1 = vote_f1
                ensemble_predictions = ensemble_predictions_vote
                ensemble_method = "majority voting"

        else:
            ensemble_accuracy = 0.0
            ensemble_f1 = 0.0
            ensemble_predictions = None
            ensemble_method = "none"

        logger.info(f"\n🎯 Ensemble result ({ensemble_method}):")
        logger.info(f"   Accuracy: {ensemble_accuracy:.3f}")
        logger.info(f"   F1 score: {ensemble_f1:.3f}")

        return {
            "client_evaluations": client_evaluations,
            "test_accuracy": weighted_accuracy,
            "test_f1": weighted_f1,
            "ensemble_accuracy": ensemble_accuracy,
            "ensemble_f1": ensemble_f1,
            "ensemble_predictions": ensemble_predictions,
            "ensemble_method": ensemble_method,
        }
