r"""
1st-Generation Avalanche analysis functions
- _fit_truncated_power_law: fit truncated power-law to avalanche data
- branching_parameter: calculate branching parameter of avalanches
- alpha_exponent: calculate alpha exponent of avalanche size distribution
- tau_exponent: calculate tau exponent of avalanche duration distribution"""

import numpy as np
import scipy.ndimage as ndi

import numpy as np
from scipy.optimize import minimize_scalar

def _fit_truncated_power_law(data, system_size=None, xmin_range=range(1, 11), n_min=20, 
                            xmax_search_step=5):
    r"""
    Estimates the power-law exponent using a Truncated Maximum Likelihood Estimation (MLE) 
    combined with Kolmogorov-Smirnov (KS) distance minimization.

    This method follows the "Fekete-style" Algorithm, which adapts the 
    Clauset et al. (2009) framework to handle finite-size effects in biological systems. 
    It performs a grid search over potential window boundaries [xmin, xmax] to identify 
    the optimal range where the data most closely follows a power-law distribution.

    Parameters
    ----------
    data : np.ndarray
        A 1-D array of avalanche metrics (e.g., sizes or durations). 
    system_size : int, optional
        The physical limit of the recording system (e.g., the total number of electrodes). 
        If provided, it defines the search upper bound.
        If None, the search range is capped at the maximum observed value in the data.
    xmin_range : range, default=range(1, 11)
        A sequence of candidate values for the lower bound cutoff (xmin).
    n_min : int, default=20
        The minimum number of data points required within a [xmin, xmax] window 
        to consider a fit statistically reliable.
    xmax_search_step : int, default=5
        The step size for searching candidate upper bound cutoffs (xmax) when system_size is not provided.

    Returns
    -------
    results : dict
        A dictionary containing the parameters of the best-fitting truncated power law:
        
        * 'exponent' (float): The estimated power-law index. It represents the probability 
          density function slope $P(x) \propto x^{-exponent}$.
        * 'xmin' (float): The lower bound of the optimal fitting window.
        * 'xmax' (float): The upper bound of the optimal fitting window.
        * 'ks' (float): The Kolmogorov-Smirnov distance. Lower values indicate a better fit.
        * 'n_tail' (int): The number of individual avalanche events contained 
          within the selected [xmin, xmax] range.

    Notes
    -----
    The grid search minimizes the KS distance across all valid [xmin, xmax] pairs. 
    For each window, an MLE fit is performed by minimizing the negative log-likelihood 
    of a discrete power law specifically normalized over that truncated interval.
    """

    # Data Preparation
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    data = data[data >= 1]
    
    if data.size < n_min:
        return {'exponent': np.nan, 'xmin': np.nan, 'xmax': np.nan, 'ks': np.nan, 'n_tail': 0}

    data = np.sort(data)

    # Precompute cumulative sum of logs for O(1) MLE calculation
    # Insert 0 at the beginning to handle the index logic easily
    log_data_cumsum = np.insert(np.cumsum(np.log(data)), 0, 0.0)

    s_max_data = data[-1]
    
    if system_size is not None:
        # typical for sizes
        xmax_candidates = range(xmin_range[-1] + 1, int(1.5 * system_size) + 1)
    else:
        # typical for durations
        xmax_candidates = range(xmin_range[-1] + 1, int(s_max_data) + 1, xmax_search_step)

    # Filter candidates
    xmaxs_valid = [x for x in xmax_candidates if x <= s_max_data]
    
    if not xmaxs_valid:
        xmaxs_valid = [int(s_max_data)]

    # Grid Search (Minimize KS)
    best_ks = np.inf
    best_params = {'exponent': np.nan, 'xmin': np.nan, 'xmax': np.nan, 'ks': np.nan, 'n_tail': 0}

    for xmin in xmin_range:
        if xmin >= s_max_data:
            continue
        
        start_idx = np.searchsorted(data, xmin, side='left')
        
        for xmax in xmaxs_valid:
            if xmax <= xmin:
                continue
            
            end_idx = np.searchsorted(data, xmax, side='right')

            n = end_idx - start_idx            
            if n < n_min:
                continue

            # --- MLE Fit ---
            sum_log_data = log_data_cumsum[end_idx] - log_data_cumsum[start_idx]
            x_vals = np.arange(xmin, xmax + 1)

            # Negative Log-Likelihood Function
            def _nll(exponent):
                r"""
                Calculate the Negative Log-Likelihood (NLL) for a discrete truncated power-law distribution.

                The NLL is derived from the probability mass function (PMF):
                $$P(x) = \frac{x^{-\gamma}}{Z(\gamma, x_{min}, x_{max})}$$
                where $Z$ is the transcendental Hurwitz zeta-like normalization constant:
                $$Z(\gamma, x_{min}, x_{max}) = \sum_{k=x_{min}}^{x_{max}} k^{-\gamma}$$

                The objective function to minimize is:
                $$\mathcal{L}(\gamma) = \gamma \sum_{i=1}^{n} \ln(x_i) + n \ln \left( \sum_{k=x_{min}}^{x_{max}} k^{-\gamma} \right)$$

                Parameters
                ----------
                exponent : float
                    The power-law exponent ($\gamma$) being optimized.

                Returns
                -------
                float
                    The negative log-likelihood value. Returns infinity if the exponent $\le 1$ 
                    or if the normalization factor $Z$ is non-finite.
            """
                if exponent <= 1: 
                    return np.inf
                Z = np.sum(x_vals ** (-exponent))
                if Z <= 0 or not np.isfinite(Z): 
                    return np.inf
                return exponent * sum_log_data + n * np.log(Z)

            # Optimization
            res = minimize_scalar(_nll, bounds=(1.0001, 10), method="bounded")
            exponent = float(res.x)

            # --- KS Distance Calculation ---

            # Theoretical CDF
            x_vals = np.arange(xmin, xmax + 1)
            pdf_theory = x_vals ** (-exponent)
            pdf_theory /= pdf_theory.sum() # Normalize
            cdf_theory = np.cumsum(pdf_theory)
            
            # Empirical CDF
            window_slice = data[start_idx:end_idx]
            cdf_emp = np.searchsorted(window_slice, x_vals, side="right") / n

            # Compute KS
            ks = np.max(np.abs(cdf_emp - cdf_theory))

            # --- Update Best Fit ---
            if ks < best_ks:
                best_ks = ks
                best_params = {
                    'exponent': exponent,
                    'xmin': xmin,
                    'xmax': xmax,
                    'ks': ks,
                    'n_tail': n
                }
    
    return best_params

def branching_parameter(avalanche_dict, method='naive', n_electrodes=None):
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
        sigma = np.sum(n_d) / np.sum(n_a)

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

def alpha_exponent(avalanche_dict, system_size=None):
    r"""
    Calculate the alpha exponent of avalanche size distribution.

    Params
    ------
    avalanche_dict : dict
        Dictionary with keys:
        - 'data': 1-D ndarray of binned avalanche events
        - 'indices': np.ndarray of shape (n_avalanches, 2) with start and end indices of each avalanche.
    system_size : int, optional
        The physical limit of the recording system (e.g., the total number of electrodes).

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

    if system_size is None:
        system_size = np.max(sizes) if sizes.size > 0 else None

    fit_results = _fit_truncated_power_law(sizes, system_size=system_size)

    return fit_results['exponent']

def tau_exponent(avalanche_dict):
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

    Returns
    -------
    tau : float
        Tau exponent of the avalanche duration distribution.
    """
    indices = avalanche_dict['indices']

    if indices.shape[0] == 0:
        return np.nan

    durations_bins = indices[:, 1] - indices[:, 0] + 1

    fit_results = _fit_truncated_power_law(durations_bins, system_size=None)

    return fit_results['exponent']