from setuptools import setup, find_packages

with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

setup(
    name='model_zoo',
    packages=find_packages(include=('model_zoo',)),
    install_requires=requirements,
)
