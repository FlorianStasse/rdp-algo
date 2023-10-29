"""Simple package to apply the Ramer-Douglas-Peucker algorithm"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


def _compute_distances(
    points: npt.NDArray[np.float_],
    start: npt.NDArray[np.float_],
    end: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
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


def _mask(points: npt.NDArray[np.float_], epsilon: float) -> npt.NDArray[np.float_]:
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
    return indices


def _rdp(
    points: npt.NDArray[np.float_], epsilon: float, *, return_mask=False
) -> npt.NDArray[np.float_]:
    mask = _mask(points, epsilon)
    if return_mask:
        return mask
    return points[mask]


def rdp(
    points: npt.ArrayLike, epsilon: float, *, return_mask=False
) -> npt.NDArray[np.float_]:
    """Simplifies a list or an array of points using the Ramer-Douglas-Peucker
    algorithm.

    :param points: Array of points (Nx2)
    :param epsilon: epsilon in the rdp algorithm
    :param return_mask: If True, returns the mask used to filter the points.

    :return: Simplified array of points, or mask if return_mask is True.
    """
    if not isinstance(points, np.ndarray):
        return _rdp(np.array(points), epsilon, return_mask=return_mask)
    return _rdp(points, epsilon, return_mask=return_mask)
