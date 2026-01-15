import numpy as np
from avalanche_preprocessor import avalanche_preprocessor, bin_avalanches
from avalanche_1st_gen import branching_parameter

# 1. Generate dummy data (n_channels, n_samples)
data = np.random.randn(60, 10000)

# 2. Detect avalanches (convert to binary raster)
# Using standard deviation thresholding with k=3.0
binary_data = avalanche_preprocessor(data, k=3.0, thresholding='std')

# 3. Bin the data
# Ideally, bin_size is calculated automatically based on IEI
binned_activity = bin_avalanches(binary_data)

# 4. Calculate Branching Parameter
sigma = branching_parameter(binned_activity, method='weighted')
print(f"Branching Parameter (sigma): {sigma}")

# Using the corrected method (requires total number of electrodes)
sigma_corrected = branching_parameter(
    binned_activity, 
    method='corrected', 
    n_electrodes=60
)
print(f"Corrected Sigma: {sigma_corrected}")