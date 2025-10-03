# IGRF-MODEL
our first project
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ppigrf

def study_igrf_coefficients():
    """
    Loads IGRF-13 coefficients, processes them, and plots the long-term 
    variation of the primary Gauss coefficients (dipole and quadrupole terms) 
    from 1900 to 2025.
    """
    print("Fetching and loading IGRF-13 coefficient data...")
    
    # 1. Load the IGRF-13 spherical harmonic coefficient file using ppigrf
    # This function automatically fetches the latest IGRF file (igrf13.shc)
    coeffs = ppigrf.load_shc_file()

    # The 'coeffs' object contains the years and the g and h coefficients.
    # coeffs.time contains the years as decimals (e.g., 1900.0, 1905.0, ...)
    # coeffs.gh contains the coefficients in a NumPy array.
    
    years = coeffs.time
    
    # 2. Organize the data into a Pandas DataFrame for easy handling
    
    # Create column headers from the years
    # We will only consider the main field models, which end at 2025.0
    main_field_indices = np.where(years <= 2025.0)[0]
    main_field_years = years[main_field_indices].astype(int)
    
    # The ppigrf library stores coefficients in a flat array. We need to know the mapping.
    # The order is g_1^0, g_1^1, h_1^1, g_2^0, g_2^1, h_2^1, g_2^2, h_2^2, ...
    # Let's create index labels for our DataFrame
    max_degree = coeffs.max_degree
    index_labels = []
    for n in range(1, max_degree + 1):
        for m in range(n + 1):
            if m == 0:
                index_labels.append(f'g_{n}^{m}')
            else:
                index_labels.append(f'g_{n}^{m}')
                index_labels.append(f'h_{n}^{m}')

    # Create the DataFrame
    df_coeffs = pd.DataFrame(
        coeffs.gh[:, main_field_indices], 
        index=index_labels, 
        columns=main_field_years
    )

    print("Successfully loaded and structured coefficients.")
    print("DataFrame Head:")
    print(df_coeffs.head())

    # 3. Select the most significant coefficients for plotting
    # These are the low-degree terms that describe the dominant field structure.
    # g_1^0: Axial Dipole (main north-south component)
    # g_1^1, h_1^1: Equatorial Dipole (tilt of the magnetic axis)
    # g_2^0: Axial Quadrupole (measures deviation from a perfect dipole)
    
    coeffs_to_plot = {
        'g_1^0': '$g_1^0$ (Axial Dipole)',
        'g_1^1': '$g_1^1$ (Equatorial Dipole)',
        'h_1^1': '$h_1^1$ (Equatorial Dipole)',
        'g_2^0': '$g_2^0$ (Axial Quadrupole)'
    }

    # 4. Plotting the coefficients vs. time
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    for code, label in coeffs_to_plot.items():
        # Extract the time series for each coefficient from the DataFrame
        time_series = df_coeffs.loc[code]
        ax.plot(time_series.index, time_series.values, marker='o', linestyle='-', label=label)

    # Formatting the plot
    ax.set_title('Long-Term Variation of IGRF Gauss Coefficients (1900-2025)', fontsize=16)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Coefficient Value (nT)', fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xticks(np.arange(1900, 2030, 10)) # Set x-axis ticks every 10 years
    plt.xticks(rotation=45)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping

    print("\nPlot generated. Displaying now...")
    plt.show()


if __name__ == '__main__':
    study_igrf_coefficients()
