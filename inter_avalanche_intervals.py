import numpy as np

def iai_series_preprocessor(avalanche_dict: dict) -> dict:
    r"""
    Preprocesses avalanche data for Inter-Avalanche Interval (IAI) analysis.
    
    Extracts time series required for IAI analysis:
    1. Quiet Times ($Q$): The silence interval between the end of one avalanche 
       and the start of the next.
    2. Waiting Times ($W$): The interval between the onset of consecutive avalanches.

    Parameters
    ----------
    avalanche_dict : dict
        A dictionary containing avalanche data, specifically:
        - 'indices': np.ndarray of shape (N, 2) where each row is [start, end] 
          indices of an avalanche.
        - 'bin_size_sec': float, the bin size used in seconds (passed through).
        - 'n_bins': int, the total number of bins in the recording.
        - 'fs': float, the sampling frequency in Hz.

    Returns
    -------
    iai_data : dict
        A dictionary containing the extracted series:
        - 'q_bins' (np.ndarray): Integer array of Quiet Times.
        - 'w_bins' (np.ndarray): Integer array of Waiting Times.
        - 'bin_size_sec' (float): The input bin size, passed through.
        - 'n_bins' (int): The input number of bins, passed through.
        - 'fs' (float): The input sampling frequency, passed through.

    Raises
    ------
    ValueError
        If any calculated interval is non-positive ($Q \le 0$ or $W \le 0$).
        This indicates an issue with the input data, such as:
        - Unsorted avalanche indices.
        - Overlapping avalanches.

    """
    indices = avalanche_dict['indices']
    bin_size_sec = avalanche_dict['bin_size_sec']
    n_bins = avalanche_dict['n_bins']
    fs = avalanche_dict['fs']
    
    if indices.shape[0] < 2:
        return {
            'q_bins': np.array([]), 
            'w_bins': np.array([]), 
            'bin_size_sec': bin_size_sec,
            'n_bins': n_bins,
            'fs': fs
        }

    starts = indices[:, 0]
    ends = indices[:, 1]

    w_bins = starts[1:] - starts[:-1]
    q_bins = starts[1:] - ends[:-1]

    if any(w_bins <= 0) or any(q_bins <= 0):
        raise ValueError("Detected non-positive waiting or quiet times. " \
                        "Please check avalanche indices.")

    return {
        'q_bins': q_bins.astype(int),
        'w_bins': w_bins.astype(int),
        'bin_size_sec': bin_size_sec,
        'n_bins': n_bins,
        'fs': fs
    }

