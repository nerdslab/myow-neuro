from setuptools import setup, find_packages


version = '0.0.1'

install_requires = [
    'absl-py',
    'tensorboard',
    'scipy',
    'numpy',
    'tqdm',
    'scikit-learn',
    'sklearn',
    'pandas',
    'h5py'
]

setup(
    name='myow-neuro',
    version=version,
    packages=find_packages(),
    install_requires=install_requires,
    author="Mehdi Azabou",
)
