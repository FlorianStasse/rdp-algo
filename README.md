# Ramer-Douglas-Peucker

Ramer-Douglas-Peucker python implementation.

## Installation

```commandline
pip install rdp-algo
```

## Usage

The rdp function supports both lists and numpy arrays of arbitrary dimensions.

```python
>>> from rdp_algo import rdp
>>> rdp([[0,0],[1,1],[2,0]], epsilon=1)
[[0,0],[2,0]]
```

```python
>>> import numpy as np
>>> from rdp_algo import rdp
>>> rdp(np.array([[0,0],[1,1],[2,0]]), epsilon=1)
array([[0,0],[2,0]])
```
