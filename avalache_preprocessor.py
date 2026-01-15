"""Avalanche preprocessing and binning functions.

Basic avalanche detection and binning functions:

- avalanche_preprocessor: detect avalanche events in data
- bin_avalanches: bin avalanche events into contiguous time bins
- _optimal_bin_size: determine optimal bin size for avalanche detection"""

import numpy as np
import scipy.ndimage as ndi
from scipy.stats import median_abs_deviation

def avalanche_preprocessor(data, k=3.0, thresholding='std'):
    """
    Detect avalanche events in the data.

    Params
    ------
    data : np.ndarray
        Input data array of size (n_channels, n_samples).
    k : float
        Threshold multiplier for standard deviation.
    thresholding : str
        Method for thresholding ('std' or 'mad').
    
    Returns
    -------
    res : np.ndarray
        Binary array (uint8) of the same shape as data, 
        with 1s at the absolute peak of each detected event.

    """
    # compute absolute normalized data
    if thresholding == 'std':
        row_means = data.mean(axis=1, keepdims=True)
        row_stds = data.std(axis=1, keepdims=True)
        row_stds[row_stds == 0] = 1e-10
        abs_data = np.abs((data - row_means) / row_stds)

    elif thresholding == 'mad':
        row_medians = np.median(data, axis=1, keepdims=True)
        robust_sd = median_abs_deviation(data, axis=1, scale='normal', keepdims=True)
        robust_sd[robust_sd == 0] = 1e-10
        abs_data = np.abs((data - row_medians) / robust_sd)
    
    else:
        raise ValueError(f"Unsupported thresholding method: {thresholding}")
    
    # create mask and structure for ndimage
    mask = (abs_data > k).astype(np.uint8)
    structure = np.array([[0, 0, 0],
                          [1, 1, 1],
                          [0, 0, 0]])

    # label theshold-crossing events
    labels, num_features = ndi.label(mask, structure=structure)

    binary_data = np.zeros_like(data, dtype=np.uint8)

    if num_features != 0:
        max_positions = ndi.maximum_position(abs_data, labels, index=np.arange(1, num_features + 1))
        rows, cols = zip(*max_positions)
        binary_data[rows, cols] = 1

    return binary_data

def _optimal_bin_size(binary_data):
    """
    Determine the optimal bin size for avalanche detection.

    Params
    ------
    binary_data : np.ndarray
        Binary array of size (n_channels, n_samples) indicating avalanche events.

    Returns
    -------
    optimal_bin_size : int
        Optimal bin size for binning avalanches.

    Notes
    -----
    The optimal bin size is calculated as the average inter-event interval (IEI),
    based on Beggs, John M., and Dietmar Plenz. "Neuronal avalanches in neocortical circuits" (2003).
    """
    network_activity = np.sum(binary_data, axis=0)
    active_indices = np.where(network_activity > 0)[0]
    if len(active_indices) < 2:
        return 1
    ieis = np.diff(active_indices)
    return int(np.round(np.mean(ieis)))

def bin_avalanches(binary_data, bin_size=None):
    """
    Bin avalanche events into contiguous time bins.

    Params
    ------
    binary_data : np.ndarray
        Binary array of size (n_channels, n_samples) indicating avalanche events.

    Returns
    -------
    binned_array : np.ndarray
        1-D ndarray of binned avalanche events.
    """
    if bin_size is None:
        bin_size = _optimal_bin_size(binary_data)

    network_activity = np.sum(binary_data, axis=0)
    n_samples = network_activity.shape[0]
    n_bins = n_samples // bin_size
    trimmed_activity = network_activity[:n_bins * bin_size]
    binned_array = trimmed_activity.reshape(n_bins, bin_size).sum(axis=1)
    return binned_array