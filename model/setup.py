from setuptools import setup

setup(
    name = "Siamese Neural Network",
    version = "0.0",
    description = "Siamese Neural Networks for bioactivity prediction",
    author = "Daniel Fernández Llaneza",
    author_email = "danielfllaneza@gmail.com",
    packages = ["model"],
    install_requires = [
        "numpy          == 1.16.4",
        "pandas         == 1.0.1",
        "pytorch        == 1.3.1",
        "rdkit          == 2018.09.1",
        "scipy          == 1.3.0",
        "scikit-learn   == 0.21.2",
        "tqdm           == 4.32.1",
    ]
)