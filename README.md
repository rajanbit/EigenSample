[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# EigenSample: Python package for generating synthetic samples in eigenspace to minimize distortion. 

This repository contains a Python implementation of the EigenSample algorithm by [Jayadeva et al., 2017](https://doi.org/10.1016/j.asoc.2017.08.017), designed to generate synthetic samples in the eigenspace while minimizing distortion. Please note that this implementation is intended solely for learning purposes and does not claim any original work or contributions. Feel free to explore, learn from, and contribute to this repository!

### Requirements
- Python >=3.10
- Git

### Installation
**pip**
```
git clone https://github.com/rajanbit/EigenSample.git
cd EigenSample/
python -m pip install --upgrade build
python -m build
pip install dist/EigenSample-0.1.0-py3-none-any.whl
```

**conda**
```
conda create -n eigensample python=3.10 git
conda activate eigensample
pip install git+https://github.com/rajanbit/EigenSample.git#egg=EigenSample
```

### Reference
Jayadeva, Soman, S., & Saxena, S. (2018). EigenSample: A non-iterative technique for adding samples to small datasets. In Applied Soft Computing (Vol. 70, pp. 1064–1077). Elsevier BV. [https://doi.org/10.1016/j.asoc.2017.08.017 ](https://doi.org/10.1016/j.asoc.2017.08.017)
