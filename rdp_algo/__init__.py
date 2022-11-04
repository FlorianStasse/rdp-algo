"""Simple package to apply the Ramer-Douglas-Peucker algorithm"""
from __future__ import annotations

from typing import TypeVar

import numpy as np
import numpy.typing as npt

Array = TypeVar("Array", npt.NDArray, list)


def _compute_distances(
    points: npt.NDArray[float], start: npt.NDArray[float], end: npt.NDArray[float]
) -> npt.NDArray[float]:
    """Compute the distances between all points and the line defined by start and end.

    :param points: Points to compute distance for.
    :param start: Starting point of the line
    :param end: End point of the line

    :return: Points distance to the line.
    """
    line = end - start
    if (line_length := np.linalg.norm(line)) == 0:
        return np.linalg.norm(points - start, axis=-1)
    if line.size == 2:
        return abs(np.cross(line, start - points)) / line_length  # 2D case
    return (
        abs(np.linalg.norm(np.cross(line, start - points), axis=-1)) / line_length
    )  # 3D case


def _rdp(points: npt.NDArray[float], epsilon: float) -> npt.NDArray[float]:
    stack = [[0, len(points) - 1]]
    indices = np.ones(len(points), dtype=bool)

    while stack:
        start_index, last_index = stack.pop()

        local_points = points[indices][start_index + 1 : last_index]
        if len(local_points) == 0:
            continue
        distances = _compute_distances(
            local_points, points[start_index], points[last_index]
        )
        dist_max = max(distances)
        index_max = start_index + 1 + np.argmax(distances)

        if dist_max > epsilon:
            stack.append([start_index, index_max])
            stack.append([index_max, last_index])
        else:
            indices[start_index + 1 : last_index] = False
    return points[indices]


def rdp(points: Array, epsilon: float) -> Array:
    """Simplifies a list or an array of points using the Ramer-Douglas-Peucker
    algorithm.

    :param points: Array of points (Nx2)
    :param epsilon: epsilon in the rdp algorithm

    :return: Simplified list of points.
    """
    if isinstance(points, list):
        result = _rdp(np.array(points), epsilon).tolist()
        assert isinstance(result, list)
        return result
    return _rdp(points, epsilon)
