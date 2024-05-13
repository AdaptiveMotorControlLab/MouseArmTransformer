import numpy as np


def _f32(arr):
    return arr.astype("float32")


def _f64(arr):
    return arr.astype("float64")


def to_homographic(point, axis="auto"):
    if point.ndim != 2:
        raise ValueError("Point needs to have two dimensions.")

    if axis == "auto":
        if point.shape[0] in {2, 3}:
            axis = 0
        if point.shape[1] in {2, 3}:
            if axis == 0:
                raise ValueError(
                    f"auto can only be used for non-ambigous shapes, got "
                    f"{point.shape}. Specify an axis instead."
                )
            axis = 1
        if axis == "auto":
            raise ValueError(f"Invalid shape: {point.shape}")
    if point.shape[axis] not in {2, 3}:
        raise ValueError(f"Invalid shape: {point.shape}")

    concat_shape = [point.shape[1]]
    concat_shape.insert(axis, 1)

    return np.concatenate([point, np.ones(concat_shape)], axis=axis)


def pack_array(array):
    shape = array.shape
    return array.reshape(-1, shape[-1]), shape[:-1]


def unpack_array(array, shape):
    shape = tuple(shape)
    shape = shape + (-1,)
    return array.reshape(shape)
