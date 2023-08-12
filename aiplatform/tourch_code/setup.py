from setuptools import setup

REQUIRED_PACKAGES = ['torch', 'transformers']

setup(
    name="email_classification",
    version="0.1",
    scripts=["model_prediction.py"],
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES
)
