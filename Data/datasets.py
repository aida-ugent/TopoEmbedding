import os
import numpy as np
import math
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from scipy import stats


def str_to_data(dataname):
    if dataname == 'circle':
        return get_circle_dataset

    if dataname == 'random':
        return get_random_dataset

    if dataname == 'cell_cycle':
        return get_cell_cycle_dataset

    if dataname == 'cell_bifurcating':
        return get_bifurcating_dataset
    else:
        return ValueError(f"Dataset name {dataname} unknown!")


def get_cell_cycle_dataset(**kwargs):
    file_name = os.path.join("Data", "CellCycle.rds")
    cell_info = ro.r["readRDS"](file_name)
    cell_info = dict(zip(cell_info.names, list(cell_info)))

    pandas2ri.activate()
    data = ro.conversion.rpy2py(cell_info["expression"])
    t = list(ro.conversion.rpy2py(cell_info["cell_info"])
             .rename(columns={"milestone_id": "group_id"}).loc[:, "group_id"])
    pandas2ri.deactivate()
    return data, t


def get_bifurcating_dataset(**kwargs):
    file_name = os.path.join("Data", "CellBifurcating.rds")
    cell_info = ro.r["readRDS"](file_name)
    cell_info = dict(zip(cell_info.names, list(cell_info)))

    pandas2ri.activate()
    data = ro.conversion.rpy2py(cell_info["expression"])
    t = list(ro.conversion.rpy2py(cell_info["cell_info"])
             .rename(columns={"milestone_id": "group_id"}).loc[:, "group_id"])
    pandas2ri.deactivate()
    return data, t


def get_circle_dataset(n, ndim=500, variance=0.0675,
                       seed=420, noise="uniform", zscore=False):
    """Generator of synthetic dataset with points sampled from 
    a 2D circle and noise in the remaining dimensions.

    Args:
        n (int): Number of points.
        ndim (int, optional): Dimensions including noise. Defaults to 500.
        variance (float, optional): Variance of uniform noise per dimension. 
                                    Defaults to 0.0675 (with range [-0.45,0.45] for uniform noise).
        seed (int, optional): Random seed for reproducibility. Defaults to 420.
        noise (string, optional): Distribution used to sample noise {'uniform', 'normal'}
        zscore (bool, optional): Whether to normalize data using zscores.

    Returns:
        np array [n, ndim]: Synthetic data array.
        np array: Labels of datapoints
    """

    np.random.seed(seed)
    t = np.random.uniform(low=0, high=2 * np.pi, size=n)
    X = np.concatenate(
        [np.transpose(np.array([np.cos(t), np.sin(t)])), np.zeros([n, ndim - 2])], axis=1)
    if noise == "uniform":
        sigma = get_uniform_limits(variance)
        N = np.random.uniform(low=-sigma, high=sigma, size=[n, ndim])
    elif noise == "normal":
        N = np.random.normal(
            loc=0.0, scale=math.sqrt(variance), size=[n, ndim])
    else:
        raise ValueError(f"Noise distribution {noise} unknown.")
    data = X + N

    if zscore:
        data = stats.zscore(data, axis=0, ddof=1)
    return data, t


def get_uniform_limits(var):
    """Compute the limits [-a, a] for a uniform distribution
       with the specified variance.

    Args:
        var (float): The desired variance of the distribution.

    Returns:
        float: upper and lower limit for uniform distribution
    """
    limit = 0.5 * math.sqrt(var * 12.0)
    return limit


def get_random_dataset(n, ndim=500, variance=0.0675,
                       seed=420, noise="uniform", zscore=False):
    """Generator of synthetic dataset with points sampled from 
    a 2D circle and noise in the remaining dimensions.

    Args:
        n (int): Number of points.
        ndim (int, optional): Noise dimensions. Defaults to 500.
        variance (float, optional): Variance of uniform noise per dimension. 
                                    Defaults to 0.0675 (with range [-0.45,0.45] for uniform noise).
        seed (int, optional): Random seed for reproducibility. Defaults to 420.
        noise (string, optional): Distribution used to sample noise {'uniform', 'normal'}
        zscore (bool, optional): Whether to normalize data using zscores.

    Returns:
        np array [n, ndim]: Synthetic data array.
        np array: Labels of datapoints
    """

    np.random.seed(seed)
    if noise == "uniform":
        sigma = get_uniform_limits(variance)
        data = np.random.uniform(low=-sigma, high=sigma, size=[n, ndim])
    elif noise == "normal":
        data = np.random.normal(
            loc=0.0, scale=math.sqrt(variance), size=[n, ndim])
    else:
        raise ValueError(f"Noise distribution {noise} unknown.")

    if zscore:
        data = stats.zscore(data, axis=1, ddof=1)
    return data, None
