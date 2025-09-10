from setuptools import setup, find_packages

setup(
    name="fl_spn",
    version="0.1.0",
    description="Federated Learning with Sum-Product Networks",
    author="juliecious",
    author_email="",
    url="https://github.com/juliecious/fl_spn",
    packages=find_packages(
        include=["fl_spn", "fl_spn.*"],
        exclude=["*.ipynb", "*.ipynb_checkpoints", "tests*"],
    ),  # 自動尋找子模組
    install_requires=[
        "simple-einet @ git+https://github.com/juliecious/simple-einet.git@20a4cb4f905d9f100d41184e4c107107b5d6899f",
    ],
    python_requires=">=3.10",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
