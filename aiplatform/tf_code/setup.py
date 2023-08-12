from setuptools import setup

REQUIRED_PACKAGES = ['torch', 'transformers']

setup(
    name="my_custom_code",
    version="0.1",
    include_package_data=True,
    scripts=["predictor.py"],
    install_requires=REQUIRED_PACKAGES
)
