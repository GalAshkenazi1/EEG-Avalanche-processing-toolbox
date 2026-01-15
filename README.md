# Neuronal Avalanche Analysis

This repository contains tools for detecting and analyzing neuronal avalanches from electrophysiological data.

## Files and Functions

### `avalache_preprocessor.py`

Handles signal preprocessing and event detection.

* **`avalanche_preprocessor`**: Detects avalanche events in the input data. Supports thresholding based on standard deviation (`'std'`) or Median Absolute Deviation (`'mad'`).
* **`bin_avalanches`**: Bins binary avalanche events into contiguous time bins.
* **`_optimal_bin_size`**: internal helper function that determines the optimal bin size based on the average inter-event interval (IEI).

### `avalache_1st_gen.py`

Handles statistical analysis of the binned data.
* **`branching_parameter`**: Calculates the branching parameter ($\sigma$) using one of three methods:
    * `'naive'`: Mean of ratios.
    * `'weighted'`: Ratio of sums (treats every active electrode as a statistical event).
    * `'corrected'`: Weighted method with refractoriness correction (requires `n_electrodes`).

More metrics are to come!

## References

[1] Beggs, John M., and Dietmar Plenz. "Neuronal avalanches in neocortical circuits." Journal of neuroscience 23.35 (2003): 11167-11177.

## Dependencies

The code requires the following Python libraries:
* `numpy`
* `scipy`

