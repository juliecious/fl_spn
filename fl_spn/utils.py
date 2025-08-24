import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_adult_income_dataset(test_size=0.2, random_state=42):
    """加载并预处理Adult Income数据集"""

    np.random.seed(random_state)
    n_samples = 32561  # 原数据集大小

    data = {}

    # continuous feature
    data["age"] = np.random.normal(39, 13, n_samples).clip(17, 90)
    data["fnlwgt"] = np.random.normal(189778, 105549, n_samples).clip(12285, 1484705)
    data["education_num"] = np.random.randint(1, 17, n_samples)
    data["capital_gain"] = np.random.exponential(1077, n_samples).clip(0, 99999)
    data["capital_loss"] = np.random.exponential(87, n_samples).clip(0, 4356)
    data["hours_per_week"] = np.random.normal(40, 12, n_samples).clip(1, 99)

    # categorical feature - fixed prob distribution
    workclass_options = [
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
    ]
    workclass_probs = np.array([0.70, 0.08, 0.05, 0.03, 0.06, 0.04, 0.01, 0.03])
    workclass_probs = workclass_probs / workclass_probs.sum()  # 标准化
    data["workclass"] = np.random.choice(
        workclass_options, n_samples, p=workclass_probs
    )

    education_options = [
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool",
    ]
    education_probs = np.array(
        [
            0.16,
            0.19,
            0.04,
            0.32,
            0.02,
            0.03,
            0.04,
            0.01,
            0.02,
            0.01,
            0.04,
            0.005,
            0.03,
            0.01,
            0.01,
            0.005,
        ]
    )
    education_probs = education_probs / education_probs.sum()
    data["education"] = np.random.choice(
        education_options, n_samples, p=education_probs
    )

    marital_options = [
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    ]
    marital_probs = np.array([0.46, 0.13, 0.33, 0.03, 0.03, 0.01, 0.01])
    marital_probs = marital_probs / marital_probs.sum()
    data["marital_status"] = np.random.choice(
        marital_options, n_samples, p=marital_probs
    )

    occupation_options = [
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces",
    ]
    occupation_probs = np.array(
        [
            0.03,
            0.13,
            0.10,
            0.11,
            0.13,
            0.14,
            0.04,
            0.06,
            0.11,
            0.03,
            0.04,
            0.005,
            0.02,
            0.005,
        ]
    )
    occupation_probs = occupation_probs / occupation_probs.sum()
    data["occupation"] = np.random.choice(
        occupation_options, n_samples, p=occupation_probs
    )

    relationship_options = [
        "Wife",
        "Own-child",
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Unmarried",
    ]
    relationship_probs = np.array([0.15, 0.15, 0.40, 0.25, 0.03, 0.02])
    relationship_probs = relationship_probs / relationship_probs.sum()
    data["relationship"] = np.random.choice(
        relationship_options, n_samples, p=relationship_probs
    )

    race_options = [
        "White",
        "Asian-Pac-Islander",
        "Amer-Indian-Eskimo",
        "Other",
        "Black",
    ]
    race_probs = np.array([0.85, 0.03, 0.01, 0.01, 0.10])
    race_probs = race_probs / race_probs.sum()
    data["race"] = np.random.choice(race_options, n_samples, p=race_probs)

    data["sex"] = np.random.choice(["Male", "Female"], n_samples, p=[0.67, 0.33])

    country_probs = np.array([0.90, 0.02, 0.08])
    country_probs = country_probs / country_probs.sum()
    data["native_country"] = np.random.choice(
        ["United-States", "Mexico", "Other"], n_samples, p=country_probs
    )

    # make target variable more realistic
    income_prob = (
        0.25 * (data["education_num"] > 10)
        + 0.20 * (data["age"] > 35)
        + 0.15 * (data["hours_per_week"] > 40)
        + 0.15 * (data["sex"] == "Male")
        + 0.15 * (data["capital_gain"] > 0)
        + 0.10 * np.random.random(n_samples)
    )
    data["income"] = (income_prob > 0.6).astype(int)  # 调整阈值使分布更合理

    df = pd.DataFrame(data)

    X = df.drop("income", axis=1)
    y = df["income"].values

    numerical_features = [
        "age",
        "fnlwgt",
        "education_num",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
    ]
    categorical_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    # get feature names
    num_feature_names = numerical_features
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(
        categorical_features
    )
    feature_names = list(num_feature_names) + list(cat_feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state, stratify=y
    )

    dataset_info = {
        "X_train": X_train.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "class_names": ["<=50K", ">50K"],
        "n_samples": n_samples,
        "n_features": X_processed.shape[1],
        "n_classes": 2,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "preprocessor": preprocessor,
    }

    logger.info(f"Adult Income dataset created:")
    logger.info(f"  Training samples: {X_train.shape[0]}")
    logger.info(f"  Test samples: {X_test.shape[0]}")
    logger.info(
        f"  Features: {X_processed.shape[1]} (6 numerical + {len(cat_feature_names)} categorical)"
    )
    logger.info(f"  Classes: 2 (Income <=50K vs >50K)")
    logger.info(f"  Training label distribution: {np.bincount(y_train)}")
    logger.info(
        f"  Class balance: {np.bincount(y_train)[1] / len(y_train):.3f} positive rate"
    )

    return dataset_info


def generate_demo_data(
    n_samples: int = 800, n_features: int = 12, random_seed: int = 42
) -> np.ndarray:
    """Generate correlated synthetic data for EiNet demonstration"""
    np.random.seed(random_seed)

    # Create multiple clusters with different characteristics
    cluster_size = n_samples // 3

    # Cluster 1: Low variance, centered around origin
    cluster1 = np.random.multivariate_normal(
        mean=np.zeros(n_features), cov=np.eye(n_features) * 0.5, size=cluster_size
    )

    # Cluster 2: Medium variance, shifted mean
    cluster2 = np.random.multivariate_normal(
        mean=np.ones(n_features) * 2.0, cov=np.eye(n_features) * 1.0, size=cluster_size
    )

    # Cluster 3: High variance, different mean
    cluster3 = np.random.multivariate_normal(
        mean=np.ones(n_features) * -1.5,
        cov=np.eye(n_features) * 1.5,
        size=n_samples - 2 * cluster_size,
    )

    # Combine clusters
    data = np.vstack([cluster1, cluster2, cluster3])

    # Shuffle
    indices = np.random.permutation(len(data))
    data = data[indices]

    return data


if __name__ == "__main__":
    print(load_adult_income_dataset(test_size=0.2, random_state=42))
