# Ramer-Douglas-Peucker

Ramer-Douglas-Peucker python implementation.

## Installation

```commandline
pip install pyrdp
```

## Usage

The rdp function supports both lists and numpy arrays of arbitrary dimensions.

```python
>>> from pyrdp import rdp
>>> rdp([[0,0],[1,1],[2,0]], epsilon=1)
[[0,0],[2,0]]
```

```python
>>> import numpy as np
>>> from pyrdp import rdp
>>> rdp(np.array([[0,0],[1,1],[2,0]]), epsilon=1)
array([[0,0],[2,0]])
```
