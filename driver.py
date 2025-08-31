from fl_spn.partitioner import FederatedDataPartitioner
from trainer import FederatedEiNetTrainer
from utils import load_dataset
from fl_spn.config import SupervisedFLConfig
import logging

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    data = load_dataset(name="adult")

    partitioner = FederatedDataPartitioner(
        X=data["X_train"],
        y=data["y_train"],
        feature_names=data["X_processed"].columns.tolist(),
        numeric_features=data["numeric_features"],
        categorical_features=data["categorical_features"],
    )

    print("\n" + "=" * 60)
    print("🔵 Test 1: Horizontal Partitioning")
    print("=" * 60)

    horizontal_partition = partitioner.horizontal_partition(
        num_clients=SupervisedFLConfig.num_clients,
        random_state=SupervisedFLConfig.random_seed,
    )

    horizontal_trainer = FederatedEiNetTrainer(horizontal_partition)

    horizontal_results = horizontal_trainer.train_federated_learning(
        data["X_processed"], epochs=SupervisedFLConfig.epochs, verbose=True
    )

    horizontal_eval = horizontal_trainer.evaluate_on_test(
        data["X_test"], data["y_test"], data["X_processed"].columns.tolist()
    )

    # print("\n" + "=" * 60)
    # print("🟡 Test 3: Hybrid Partitioning")
    # print("=" * 60)
    #
    # hybrid_partition = partitioner.hybrid_partition(
    #     num_clients=SupervisedFLConfig.num_clients,
    #     sample_overlap_ratio=SupervisedFLConfig.sample_overlap_ratio,
    #     feature_overlap_ratio=SupervisedFLConfig.feature_overlap_ratio,
    #     random_state=SupervisedFLConfig.random_seed,
    # )
    # hybrid_trainer = FederatedEiNetTrainer(hybrid_partition)
    # hybrid_results = hybrid_trainer.train_federated_learning(
    #     data["X_processed"], epochs=SupervisedFLConfig.epochs, verbose=True
    # )
    #
    # hybrid_eval = hybrid_trainer.evaluate_on_test(
    #     data["X_test"], data["y_test"], data["X_processed"].columns.tolist()
    # )
