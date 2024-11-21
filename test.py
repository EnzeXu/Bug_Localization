# from dataset import make_dataset
#
#
# if __name__ == "__main__":
#     res = 0
#     while not res:
#         res = make_dataset()


import matplotlib.pyplot as plt
import numpy as np

def best_arange(data):
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val

    # Determine an approximate interval as one-tenth of the range
    raw_interval = range_val / 10

    # Find the nearest "nice" interval (1, 2, 5, 10, 20, 50, ...) or their decimal counterparts
    magnitude = 10 ** int(np.floor(np.log10(raw_interval)))  # Get the magnitude (e.g., 10, 1, 0.1)
    if raw_interval / magnitude <= 1.5:
        nice_interval = 1 * magnitude
    elif raw_interval / magnitude <= 3:
        nice_interval = 2 * magnitude
    elif raw_interval / magnitude <= 7:
        nice_interval = 5 * magnitude
    else:
        nice_interval = 10 * magnitude

    # Calculate start, stop, and interval for np.arange
    start = np.floor(min_val / nice_interval) * nice_interval
    stop = np.ceil(max_val / nice_interval) * nice_interval
    interval = nice_interval
    data_range = np.arange(start, stop + interval, interval)
    print(f"data_range: {data_range}")

    return data_range


def analyze_metrics_distribution(data, save_path, title):
    """
    Generates and saves a styled boxplot and bar chart for the distribution of input metrics data.
    Marks the Q1, median (Q2), Q3, mean, and any outliers on the boxplot.

    Parameters:
    - data (list of float): List of float values representing the metrics.
    - save_path (str): Path where the plot image will be saved.
    """
    # Calculate histogram for bar chart on the right
    bins = best_arange(data) #np.arange(0, 11, 1)  # Define bins (0-100 in intervals of 10)
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_labels = [f"{bin_edges[i]:.1f} to <{bin_edges[i + 1]:.1f}" for i in range(len(bin_edges) - 1)]

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1.5]})

    # Left subplot: Boxplot with specified colors
    box = ax1.boxplot(data, patch_artist=True,
                      boxprops=dict(facecolor="#8dc2ff", color="#499cff"),
                      medianprops=dict(color="#499cff"),
                      whiskerprops=dict(color="#499cff"),
                      capprops=dict(color="#499cff"),
                      flierprops=dict(marker='o', markerfacecolor="#8dc2ff", markeredgecolor="#499cff", markersize=8))

    # Customize Q1-Q3 border color separately
    for whisker in box['whiskers']:
        whisker.set_color("#499cff")
    for cap in box['caps']:
        cap.set_color("#499cff")
    for median in box['medians']:
        median.set_color("#499cff")
    for box_patch in box['boxes']:
        box_patch.set_facecolor("#8dc2ff")
        box_patch.set_edgecolor("#499cff")

    # Calculate Q1, Q2 (median), Q3, and Mean
    q1 = np.percentile(data, 25)
    median_value = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    mean_value = np.mean(data)

    # Plot and label Q1, Q2 (Median), Q3, and Mean on the boxplot
    ax1.plot(1, q1, 'o', color="#499cff", label='Q1')
    ax1.plot(1, median_value, 'o', color="#499cff", label='Median (Q2)')
    ax1.plot(1, q3, 'o', color="#499cff", label='Q3')
    ax1.plot(1, mean_value, 'D', color="#8dc2ff", markeredgecolor="#499cff", label='Mean')

    # Annotate the points on the plot
    ax1.annotate(f'Q1: {q1:.2f}', xy=(1.1, q1), xytext=(1.15, q1),
                 arrowprops=dict(arrowstyle='->', color="#499cff"))
    ax1.annotate(f'Q2 (Median): {median_value:.2f}', xy=(1.1, median_value), xytext=(1.15, median_value),
                 arrowprops=dict(arrowstyle='->', color="#499cff"))
    ax1.annotate(f'Q3: {q3:.2f}', xy=(1.1, q3), xytext=(1.15, q3),
                 arrowprops=dict(arrowstyle='->', color="#499cff"))
    ax1.annotate(f'Mean: {mean_value:.2f}', xy=(1.1, mean_value), xytext=(1.15, mean_value),
                 arrowprops=dict(arrowstyle='->', color="#8dc2ff"))
    # ax1.annotate(f'Q1: {q1:.2f}', xy=(1.1, q1-0.01), xytext=(1.15, q1-0.03),
    #                  arrowprops=dict(arrowstyle='->', color="#499cff"))
    #     ax1.annotate(f'Q2 (Median): {median_value:.2f}', xy=(1.1, median_value+0.01), xytext=(1.15, median_value+0.03),
    #                  arrowprops=dict(arrowstyle='->', color="#499cff"))

    # # Mark outliers, if any
    # for flier in box['fliers']:
    #     outliers = flier.get_ydata()
    #     for outlier in outliers:
    #         ax1.plot(1, outlier, 'o', color="#8dc2ff", markeredgecolor="#499cff")
    #         ax1.annotate(f'Outlier: {outlier:.2f}', xy=(1.1, outlier), xytext=(1.15, outlier),
    #                      arrowprops=dict(arrowstyle='->', color="#499cff"))

    # Set title and labels
    ax1.set_title(title, fontsize=16)
    # ax1.set_ylabel("Value")
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks([])

    # Right subplot: Vertical bar chart
    ax2.barh(bin_labels, hist, color="#8dc2ff", edgecolor="#499cff", alpha=0.7)
    ax2.set_title("Distribution", fontsize=16)
    ax2.set_xlabel("Frequency", fontsize=16)
    ax2.invert_yaxis()  # To match the example layout
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Save the plot to the specified path
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=400)
    plt.close()
#
# def draw_distribution(data_list, save_path):
#     """
#     Draws a bar chart to display the distribution of float values in `data_list`.
#
#     Parameters:
#         data_list (list of float): List of float values to display the distribution.
#         save_path (str): Path to save the bar chart image.
#     """
#     # Define the number of bins
#     num_bins = 10  # You can adjust the number of bins based on your data
#
#     # Create the histogram bins and counts
#     counts, bins = np.histogram(data_list, bins=num_bins)
#
#     # Calculate the bin centers for plotting
#     bin_centers = (bins[:-1] + bins[1:]) / 2
#
#     # Plot the bar chart
#     plt.figure(figsize=(10, 6))
#     plt.bar(bin_centers, counts, width=(bins[1] - bins[0]), edgecolor='black', alpha=0.7)
#     plt.xlabel('Value Range')
#     plt.ylabel('Frequency')
#     plt.title('Distribution of Values')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#
#     # Save the chart to the specified path
#     plt.savefig(save_path, format='png')
#     plt.close()

if __name__ == "__main__":
    file_path = "method_length.npy"
    data = list(np.load(file_path))

    data.sort()
    data = data[1000:-1000]
    analyze_metrics_distribution(data, file_path.replace(".npy", ".png"), title="Method Length Distribution")
