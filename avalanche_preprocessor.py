r"""Avalanche preprocessing and binning functions.

Basic avalanche detection and binning functions:

- avalanche_preprocessor: detect avalanche events in data
- _optimal_bin_size: determine optimal bin size for avalanche detection
- bin_avalanches: bin avalanche events into contiguous time bins
- detect_avalanches: detect avalanche start and end indices in binned array
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.stats import median_abs_deviation

def avalanche_preprocessor(data, fs, k=3.0, thresholding='std'):
    r"""
    Detect avalanche events in the data.

    Params
    ------
    data : np.ndarray
        Input data array of size (n_channels, n_samples).
    fs : float
        Sampling frequency of the data in Hz.
    k : float
        Threshold multiplier for standard deviation.
    thresholding : str
        Method for thresholding ('std' or 'mad').
    
    Returns
    -------
    res : np.ndarray
        Binary array (uint8) of the same shape as data, 
        with 1s at the absolute peak of each detected event.
    fs : float
        Sampling frequency (passed through).
    """
    # compute absolute normalized data
    if thresholding == 'std':
        row_means = data.mean(axis=1, keepdims=True)
        row_stds = data.std(axis=1, keepdims=True)
        row_stds[row_stds == 0] = 1e-10
        abs_data = data - row_means 
        abs_data /= row_stds         
        np.abs(abs_data, out=abs_data)

    elif thresholding == 'mad':
        row_medians = np.median(data, axis=1, keepdims=True)
        robust_sd = median_abs_deviation(data, axis=1, scale='normal', keepdims=True)
        robust_sd[robust_sd == 0] = 1e-10
        abs_data = data - row_medians 
        abs_data /= robust_sd         
        np.abs(abs_data, out=abs_data)
    
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

    return binary_data, fs

def _optimal_bin_size(binary_data):
    r"""
    Determine the optimal bin size for avalanche detection.

    Params
    ------
    binary_data : np.ndarray
        Binary array of size (n_channels, n_samples) indicating avalanche events.

    Returns
    -------
    int
        Optimal bin size (in samples) for binning avalanches.

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

def bin_avalanches(binary_data, fs, bin_size_sec=None):
    r"""
    Bin avalanche events into contiguous time bins.

    Params
    ------
    binary_data : np.ndarray
        Binary array of size (n_channels, n_samples) indicating avalanche events.
    fs : float
        Sampling frequency of the data in Hz.
    bin_size_sec : float, optional
        Size of each bin in seconds. If None, the optimal bin size is computed.

    Returns
    -------
    binned_array : np.ndarray
        1-D ndarray of binned avalanche events.
    bin_size_sec : float
        The bin size used in seconds.
    """
    if bin_size_sec is None:
        bin_size_samples = _optimal_bin_size(binary_data) # >=1
        bin_size_sec = bin_size_samples / fs
    else:
        bin_size_samples = int(np.round(bin_size_sec * fs))
        if bin_size_samples < 1:
            bin_size_samples = 1
            bin_size_sec = 1.0 / fs

    network_activity = np.sum(binary_data, axis=0)
    n_samples = network_activity.shape[0]
    n_bins = n_samples // bin_size_samples
    trimmed_activity = network_activity[:n_bins * bin_size_samples] # trim to fit bins

    binned_array = trimmed_activity.reshape(n_bins, bin_size_samples).sum(axis=1)

    return binned_array, bin_size_sec

def detect_avalanches(binned_array, bin_size_sec):
    r"""
    Detect avalanche start and end indices in the binned array.

    Params
    ------
    binned_array : np.ndarray
        1-D ndarray of binned avalanche events.
    bin_size_sec : float
        The bin size used in seconds.

    Returns
    ------- 
    avalanche_dict : dict
        Dictionary with keys:
        - 'data': original binned_array
        - 'indices': np.ndarray of shape (n_avalanches, 2) with start and end indices of each avalanche.
            the end index is inclusive.
        - 'bin_size_sec': float, the bin size used (in seconds).

    """
    is_active = (binned_array > 0).astype(int)
    n = len(is_active)

    if n == 0 or not np.any(is_active):
        return {
            'data': binned_array,
            'indices': np.empty((0, 2), dtype=int),
            'bin_size_sec': bin_size_sec
        }   
    
    diffs = np.diff(is_active, prepend=0, append=0) # pad to detect edges

    start_indices = np.where(diffs == 1)[0]
    end_indices = np.where(diffs == -1)[0] - 1 # end is inclusive

    # Filter edge cases
    if len(start_indices) > 0:
        if start_indices[0] == 0:
            start_indices = start_indices[1:]
            end_indices = end_indices[1:]
        if len(end_indices) > 0:
            last_bin_idx = len(binned_array) - 1
            if end_indices[-1] == last_bin_idx:
                start_indices = start_indices[:-1]
                end_indices = end_indices[:-1]

    return {
        'data': binned_array,
        'indices': np.column_stack((start_indices, end_indices)),
        'bin_size_sec': bin_size_sec
    }
