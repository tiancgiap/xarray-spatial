import numpy as np
import xarray as xa

from xrspatial.hydro import flow_direction


def create_test_arr(arr):
    n, m = arr.shape
    raster = xa.DataArray(arr, dims=['y', 'x'])
    raster['y'] = np.linspace(0, n, n)
    raster['x'] = np.linspace(0, m, m)
    return raster


def test_flow_direction():

    elevation = np.array([[78, 72, 69, 71, 58, 49],
                          [74, 67, 58, 49, 46, 50],
                          [69, 53, 44, 37, 38, 48],
                          [64, 58, 55, 22, 31, 24],
                          [68, 61, 47, 21, 16, 19],
                          [74, 53, 34, 12, 11, 12]], dtype=np.float64)

    elevation = create_test_arr(elevation)

    direction = np.array([[2, 2, 2, 4, 4, 8],
                          [2, 2, 2, 4, 4, 8],
                          [1, 1, 2, 4, 8, 4],
                          [128, 128, 1, 2, 4, 8],
                          [2, 2, 1, 4, 4, 4],
                          [1, 1, 1, 1, 4, 16]], dtype=np.float64)

    direction = create_test_arr(direction)

    result = flow_direction(elevation)

    assert isinstance(result, xa.DataArray)
    assert result.name == 'flow_direction'
    assert (result.data == direction.data).all()
