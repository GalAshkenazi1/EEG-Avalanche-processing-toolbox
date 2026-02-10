r"""1st-Generation Avalanche analysis functions

- branching_parameter: calculate branching parameter of avalanches
- alpha_exponent: calculate alpha exponent of avalanche size distribution
- tau_exponent: calculate tau exponent of avalanche duration distribution
"""
import numpy as np
from utils import _fit_truncated_power_law

def branching_parameter(avalanche_dict: dict, 
                        method: str = 'naive', 
                        n_channels: int = None) -> float:
    r"""
    Calculate the branching parameter of avalanche events.
    
    Params
    ------
    avalanche_dict : dict
        Dictionary with keys:
        - 'data': 1-D ndarray of binned avalanche events
        - 'indices': np.ndarray of shape (n_avalanches, 2) with start and end indices of each avalanche.
    method : str, optional
        Method to calculate branching parameter:
        * 'naive': Mean of ratios.
        * 'weighted': Ratio of sums.
        * 'corrected': Weighted method with refractoriness correction.
          Default is 'naive'.
    n_channels : int, optional
        Total number of channels in the recording. 
        Required only for 'corrected' method.

    Returns
    -------
    sigma : float
        Branching parameter of the avalanches.
    
    Raises
    ------
    ValueError
        If an unsupported method is provided or if n_elctrodes is not provided for 'corrected' method.  
    
    Notes
    -----
    Based on Beggs, John M., and Dietmar Plenz. "Neuronal avalanches in neocortical circuits" (2003).
    
    * Naive: Treats every avalanche as an equal statistical event.
    * Weighted: Treats every active electrode (ancestor) as a statistical event.
    * Corrected: Accounts for the 'ceiling effect' where active electrodes cannot produce
      new descendants in the immediate next bin due to refractoriness or saturation.
    """

    binned_array = avalanche_dict['data']
    indices = avalanche_dict['indices']

    if indices.shape[0] == 0:
        return 0.0
    
    starts = indices[:, 0]
    n_a = binned_array[starts].astype(np.float64)
    n_d = binned_array[starts + 1].astype(np.float64)

    if method == 'naive':
        sigma = np.mean(n_d / n_a)

    elif method == 'weighted':
        # TODO: check if this is correct
        sigma = np.sum(n_d) / np.sum(n_a)
        Warning.warn("Weighted method wasn't varified.")

    elif method == 'corrected':
        if n_channels is None:
            raise ValueError("n_channels must be provided for corrected method.")

        denom_correction = n_channels - n_a
        valid = denom_correction > 0

        valid_n_a = n_a[valid]
        if len(valid_n_a) == 0:
            return 0.0

        correction_factor = (n_channels - 1) / denom_correction[valid]
        corrected_descendants = n_d[valid] * correction_factor
        sigma = np.sum(corrected_descendants) / np.sum(valid_n_a)
        Warning.warn("Corrected method wasn't varified.")

    else:
        raise ValueError(f"Unsupported method: {method}")
    
    return sigma

def alpha_exponent(avalanche_dict: dict, 
                   n_channels: int = None) -> float:
    r"""
    Calculate the Alpha exponent of avalanche size distribution.

    Params
    ------
    avalanche_dict : dict
        Dictionary with keys:
        - 'data': 1-D ndarray of binned avalanche events
        - 'indices': np.ndarray of shape (n_avalanches, 2) with start and end indices of each avalanche.
    n_channels : int, optional
        The physical limit of the recording system (e.g., the total number of channels).

    Returns
    -------
    alpha : float
        Alpha exponent of the avalanche size distribution.
        Returns np.nan if not enough avalanches are detected.
    """
    binned_array = avalanche_dict['data']
    indices = avalanche_dict['indices']

    if indices.shape[0] == 0:
        return np.nan

    sizes = np.add.reduceat(binned_array, indices[:, 0]) # C-level array operation

    if n_channels is None:
        Warning.warn("n_channels not provided. Alpha exponent fit may be unreliable.")
        system_size = np.max(sizes) if sizes.size > 0 else None
        fit_results = _fit_truncated_power_law(sizes, system_size=system_size)
    else:
        fit_results = _fit_truncated_power_law(sizes, system_size=n_channels)

    return fit_results['exponent']

def tau_exponent(avalanche_dict: dict, 
                 t_max_method: str = 'max') -> float:
    r"""
    Calculate the tau exponent of avalanche duration distribution.

    Note: The fitting is performed on the discrete bin counts (integers) to satisfy 
    the Discrete MLE assumptions. The resulting exponent 'tau' is scale-invariant 
    and valid for physical time units as well.

    Params
    ------
    avalanche_dict : dict
            Dictionary with keys:
            - 'indices': np.ndarray of shape (n_avalanches, 2) with start and end indices of each avalanche.
            - 'data': 1-D ndarray of binned avalanche events.
            - 'n_bins': the number of bins in the binned array.
    t_max_method : str, optional
        Method to determine the maximum duration (t_max) for fitting:
        * 'max': Use the maximum observed duration in the data.
        * 'lab': Use the theoretical t_max based on the maximum avalanche size and average activity.

    Returns
    -------
    tau : float
        Tau exponent of the avalanche duration distribution.
    """
    indices = avalanche_dict['indices']

    if indices.shape[0] == 0:
        return np.nan

    durations_bins = indices[:, 1] - indices[:, 0] + 1
    
    if t_max_method == 'max':
        t_max = np.max(durations_bins)
    elif t_max_method == 'lab':
        # Compute theoretical t_max based on maximun avalanche size
        binned_array = avalanche_dict['data']
        n_bins = avalanche_dict['n_bins']
        max_size = np.max(np.add.reduceat(binned_array, indices[:, 0]))
        coeff = np.sum(binned_array) / n_bins
        t_max = np.sqrt(max_size / coeff)
    else:
        raise ValueError(f"Unsupported t_max_method: {t_max_method}")
    
    fit_results = _fit_truncated_power_law(durations_bins, system_size=t_max)

    return fit_results['exponent']