from setuptools import setup, find_packages

setup(
    name="fl_spn",
    version="0.1.0",
    description="Federated Learning with Sum-Product Networks",
    author="juliecious",
    author_email="",
    url="https://github.com/juliecious/fl_spn",
    packages=find_packages(
        exclude=["*.ipynb", "*.ipynb_checkpoints", "tests*"]
    ),  # 自動尋找子模組
    install_requires=[
        "numpy==1.26.4",
        "pandas==2.3.2",
        "pre-commit",
        "simple-einet @ git+https://github.com/juliecious/simple-einet.git@20a4cb4f905d9f100d41184e4c107107b5d6899f",
        "torch==2.8.0",
        "tqdm",
        "plotly",
        "matplotlib",
        "seaborn",
        "scikit-learn==1.7.1",
        "scipy==1.14.1",
    ],
    python_requires=">=3.10",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
