import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
# import datetime

x_50 = np.linspace(0, 10, 50)
y4_50 = np.cos(x_50) * 30 + np.random.uniform(-20, 30, size=50)  # Cosine-like trend with variation

# Adjusting the data points for smoothing
x_smooth_50 = np.linspace(0, 10, 500)  # Increasing the number of x points for smoothness
spl_50 = make_interp_spline(x_50, y4_50, k=3)  # Cubic spline interpolation
y4_smooth_50 = spl_50(x_smooth_50)

# Plot the smoothed curve with 50 points
plt.figure(figsize=(10, 6))
plt.plot(x_smooth_50, y4_smooth_50, label='Curve 4: Smoothed Cosine-like Trend (50 points)', color='purple')

# Set the y-axis limits as requested
plt.ylim(200, -200)


y4_upper = y4_50 + 30  # Adding a buffer for the upper bound
y4_lower = y4_50 - 30  # Subtracting a buffer for the lower bound

# Smoothing the upper and lower bound curves
spl_upper = make_interp_spline(x_50, y4_upper, k=3)  # Cubic spline for upper bound
y4_upper_smooth = spl_upper(x_smooth_50)

spl_lower = make_interp_spline(x_50, y4_lower, k=3)  # Cubic spline for lower bound
y4_lower_smooth = spl_lower(x_smooth_50)

# Plot the smoothed curve with bounds and shaded region
plt.figure(figsize=(10, 6))

# Plot the main curve
plt.plot(x_smooth_50, y4_smooth_50, label='Main Curve: Cosine-like Trend', color='purple')

# Plot the upper and lower bounds
plt.plot(x_smooth_50, y4_upper_smooth, label='Upper Bound', linestyle='--', color='blue')
plt.plot(x_smooth_50, y4_lower_smooth, label='Lower Bound', linestyle='--', color='red')

# Shade the region between the upper and lower bounds
plt.fill_between(x_smooth_50, y4_lower_smooth, y4_upper_smooth, color='gray', alpha=0.3)

# Set the y-axis limits as requested
plt.ylim(200, -200)

# Show the plot
# plt.grid(True)
plt.show()