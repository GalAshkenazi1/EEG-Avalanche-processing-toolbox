import numpy as np
import scipy.ndimage as ndi

def branching_parameter(binned_array, method='naive', n_electrodes=None):
    """
    Calculate the branching parameter of avalanche events.
    
    Params
    ------
    binned_array : np.ndarray
        1-D ndarray of binned avalanche events.
    method : str, optional
        Method to calculate branching parameter:
        * 'naive': Mean of ratios.
        * 'weighted': Ratio of sums.
        * 'corrected': Weighted method with refractoriness correction.
          Default is 'naive'.
    n_electrodes : int, optional
        Total number of electrodes/channels in the recording. 
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
    is_active = binned_array > 0
    starts_mask = is_active.copy()
    starts_mask[1:] &= ~is_active[:-1]

    if not np.any(starts_mask):
        return 0.0
    
    start_indices = np.where(starts_mask)[0]
    start_indices = start_indices[start_indices < len(binned_array) - 1]

    n_a = binned_array[start_indices].astype(np.float64)
    n_d = binned_array[start_indices + 1].astype(np.float64)

    if method == 'naive':
        sigma = np.mean(n_d / n_a)

    elif method == 'weighted':
        return np.sum(n_d) / np.sum(n_a)

    elif method == 'corrected':
        if n_electrodes is None:
            raise ValueError("n_electrodes must be provided for corrected method.")

        denom_correction = n_electrodes - n_a
        valid = denom_correction > 0

        valid_n_a = n_a[valid]
        if len(valid_n_a) == 0:
            return 0.0

        correction_factor = (n_electrodes - 1) / denom_correction[valid]
        corrected_descendants = n_d[valid] * correction_factor
        sigma = np.sum(corrected_descendants) / np.sum(valid_n_a)

    else:
        raise ValueError(f"Unsupported method: {method}")
    
    return sigma