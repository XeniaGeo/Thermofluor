## This workbook was adapted to Deep eutectic solvent screening, meaning that every sample group has its own blanks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, peak_widths
import os

# --- Step 1: Load and process the measurement data ---
try:
    df = pd.read_excel(r"C:/Users/maritimum/Nextcloud/2025_LAB/Thermofluor_results/20260923_Transaminase_in_DES80_LS_XG.xls", sheet_name="Raw Data", skiprows=8)
except FileNotFoundError:
    print("Error: '20260923_Transaminase_in_DES80_LS_XG.xls' not found. Please make sure the file is in the same directory as the script.")
    exit() # Exit if the file is not found

df.columns = ["Well", "Reading", "Fluorescence"]
df = df.dropna(subset=["Well", "Reading", "Fluorescence"])
df["Reading"] = df["Reading"].str.extract(r"(\d+)").astype(int)
pivot_df = df.pivot(index="Reading", columns="Well", values="Fluorescence")

# --- Step 2: Define constants ---
initial_temp = 25
max_temp = 95
n_reads = pivot_df.shape[0]

# Ensure n_reads is not zero to prevent division by zero or errors
if n_reads == 0:
    print("Error: No data readings found in the Excel file. Please check 'Raw Data' sheet.")
    exit()

# More robust way to generate temperatures using numpy.linspace
# This ensures the first temp is initial_temp and the last is max_temp, with n_reads points
temperatures = np.linspace(initial_temp, max_temp, n_reads)

# Tune these in degrees Celsius rather than reading counts. Tm is the maximum
# slope of a sustained fluorescence rise (the positive first derivative).
SMOOTHING_WIDTH_C = 5.0
MIN_PEAK_WIDTH_C = 2.0
MIN_PROMINENCE_FRACTION = 0.10
MIN_PROMINENCE_NOISE_MULTIPLIER = 3.0
EDGE_EXCLUSION_C = 2.0


def transition_peak(temp, signal):
    """Return smoothed fluorescence, its derivative and the broadest-area peak."""
    if len(temp) < 7 or not np.all(np.diff(temp) > 0):
        return None
    spacing = np.median(np.diff(temp))
    window = max(5, int(round(SMOOTHING_WIDTH_C / spacing)))
    window += 1 - window % 2
    window = min(window, len(temp) if len(temp) % 2 else len(temp) - 1)
    if window < 5:
        return None
    smooth = savgol_filter(signal, window, 3)
    derivative = np.gradient(smooth, temp)
    # Estimate noise from the local high-frequency derivative residual.
    residual = derivative - savgol_filter(derivative, window, 2)
    noise = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    minimum_prominence = max(MIN_PROMINENCE_FRACTION * np.ptp(derivative),
                             MIN_PROMINENCE_NOISE_MULTIPLIER * noise)
    peaks, properties = find_peaks(derivative, prominence=minimum_prominence)
    widths, _, left, right = peak_widths(derivative, peaks, rel_height=0.5)
    candidates = []
    for peak, width, lo, hi in zip(peaks, widths, left, right):
        if (derivative[peak] <= 0 or width * spacing < MIN_PEAK_WIDTH_C
                or temp[peak] - temp[0] < EDGE_EXCLUSION_C
                or temp[-1] - temp[peak] < EDGE_EXCLUSION_C):
            continue
        # Area above the half-prominence base measures the size of the
        # sustained transition; narrow, high noise spikes score poorly.
        lo_idx, hi_idx = max(0, int(np.floor(lo))), min(len(temp) - 1, int(np.ceil(hi)))
        baseline = min(derivative[lo_idx], derivative[hi_idx])
        area = np.trapezoid(np.maximum(derivative[lo_idx:hi_idx + 1] - baseline, 0),
                        temp[lo_idx:hi_idx + 1])
        candidates.append((area, peak))
    chosen = max(candidates)[1] if candidates else None
    return smooth, derivative, chosen, peaks

# --- Step 3: Define well groups ---
# Put each group's own blank wells under "blanks".
sample_groups = {
    "BetGly_14_80": {
        "blanks": ["B11"],
        "5uM": ["B2", "C2", "D2"],
        "10uM": ["B3", "C3", "D3"],
        "15uM": ["B4", "C4", "D4"],
    },
    "BetGly_16_80": {
        "blanks": ["C11"],
        "5uM": ["B5", "C5", "D5"],
        "10uM": ["B6", "C6", "D6"],
        "15uM": ["B7", "C7", "D7"],
    },
    "BetGly_18_80": {
        "blanks": ["D11"],
        "5uM": ["B8", "C8", "D8"],
        "10uM": ["B9", "C9", "D9"],
        "15uM": ["B10", "C10", "D10"],
    },
    "SarGly_14_80": {
        "blanks": ["E11"],
        "5uM": ["E2", "F2", "G2"],
        "10uM": ["E3", "F3", "G3"],
        "15uM": ["E4", "F4", "G4"],
    },
    "SarGly_16_80": {
        "blanks": ["F11"],
        "5uM": ["E5", "F5", "G5"],
        "10uM": ["E6", "F6", "G6"],
        "15uM": ["E7", "F7", "G7"],
    },
    "SarGly_18_80": {
        "blanks": ["G11"],
        "5uM": ["E8", "F8", "G8"],
        "10uM": ["E9", "F9", "G9"],
        "15uM": ["E10", "F10", "G10"],
    },
    "Buffer": {
        "blanks": ["H1"],
        "5uM": ["H2", "H5", "J8"],
        "10uM": ["H3", "H6", "H9"],
        "15uM": ["H4", "H7", "H10"],
    },
}


def blank_mean_for(blank_wells, sample_name):
    """Average fluorescence of this group's blank wells (per reading)."""
    valid_blank_wells = [well for well in blank_wells if well in pivot_df.columns]
    missing = [well for well in blank_wells if well not in pivot_df.columns]
    for well in missing:
        print(
            f"Warning: Blank well '{well}' for {sample_name} not found in data. "
            "It will be ignored for blank subtraction."
        )
    if not valid_blank_wells:
        print(
            f"Warning: No valid blank wells for {sample_name}. "
            "Setting blank_mean to 0."
        )
        return 0
    return pivot_df[valid_blank_wells].mean(axis=1)


# --- Step 5 & 6 & 7: Plot each sample group, calculate Tm, and save ---
output_folder = "plots_first_derivative"
os.makedirs(output_folder, exist_ok=True)

all_tms_summary = {} # Stores Tms for CSV output (individual concentrations)
average_tms_summary = {} # Stores average Tms for CSV output (per sample group)

for sample_name, conc_dict in sample_groups.items():
    fig, (ax, derivative_ax) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    tm_values = {} # Stores Tms for current sample's concentrations only
    blank_mean = blank_mean_for(conc_dict.get("blanks", []), sample_name)

    for conc, wells in conc_dict.items():
        if conc == "blanks":
            continue
        valid_sample_wells = [well for well in wells if well in pivot_df.columns]
        if not valid_sample_wells:
            print(f"Warning: No valid wells found for {sample_name} - {conc}. Skipping this group.")
            tm_values[conc] = np.nan
            all_tms_summary[f"{sample_name}_{conc}"] = np.nan
            continue

        sample_mean = pivot_df[valid_sample_wells].mean(axis=1)

        if isinstance(blank_mean, pd.Series):
            corrected = sample_mean - blank_mean
        else:
            corrected = sample_mean

        valid_indices = np.isfinite(corrected)
        if not np.any(valid_indices):
            print(f"Warning: No valid (non-NaN/inf) data points for {sample_name} - {conc}. Tm set to NaN.")
            tm_values[conc] = np.nan
            all_tms_summary[f"{sample_name}_{conc}"] = np.nan
            ax.plot(temperatures, corrected, 'o', markersize=3, alpha=0.6, label=f"{conc} (No valid data)")
            continue

        temp_for_filter = temperatures[valid_indices]
        signal_for_filter = corrected[valid_indices]

        result = transition_peak(temp_for_filter, signal_for_filter.to_numpy())
        if result is None:
            print(f"Warning: Insufficient usable data for {sample_name} - {conc}.")
            tm_values[conc] = np.nan
            all_tms_summary[f"{sample_name}_{conc}"] = np.nan
            ax.plot(temp_for_filter, signal_for_filter, '.', label=f"{conc} (insufficient data)")
            continue

        smooth, derivative, chosen, candidates = result
        line, = ax.plot(temp_for_filter, smooth, label=conc)
        color = line.get_color()
        derivative_ax.plot(temp_for_filter, derivative, color=color, label=conc)
        if chosen is None:
            print(f"Warning: No broad, prominent positive transition for {sample_name} - {conc}; Tm set to NaN.")
            tm_values[conc] = np.nan
            all_tms_summary[f"{sample_name}_{conc}"] = np.nan
        else:
            tm_value = float(temp_for_filter[chosen])
            tm_values[conc] = tm_value
            all_tms_summary[f"{sample_name}_{conc}"] = tm_value
            line.set_label(f"{conc} (Tm={tm_value:.2f}°C)")
            ax.plot(tm_value, smooth[chosen], 'o', color=color, markeredgecolor='black')
            derivative_ax.plot(tm_value, derivative[chosen], 'o', color=color,
                               markeredgecolor='black')
            derivative_ax.annotate(f"{conc}: {tm_value:.1f}°C",
                                   (tm_value, derivative[chosen]), xytext=(5, 6),
                                   textcoords='offset points', fontsize=8, color=color)
        other_peaks = [peak for peak in candidates if peak != chosen]
        if other_peaks:
            derivative_ax.plot(temp_for_filter[other_peaks], derivative[other_peaks],
                               'x', color=color, alpha=0.6)

    # --- Calculate Average Tm and Std Dev for the current sample_name ---
    current_sample_tms = [tm for tm in tm_values.values() if not np.isnan(tm)]

    avg_tm_text = "" # Initialize empty string for text
    if current_sample_tms: # Check if there are any valid Tms to average
        avg_tm = np.mean(current_sample_tms)
        std_tm = np.std(current_sample_tms)
        avg_tm_text = f"Avg Tm: {avg_tm:.2f}°C ± {std_tm:.2f} (SD)"
        # Store numerical average and std dev in the new summary dict
        average_tms_summary[sample_name] = {'Average Tm (°C)': avg_tm, 'Std Dev Tm (°C)': std_tm}
    else:
        avg_tm_text = "Avg Tm: Not available"
        # Store NaN if average cannot be calculated
        average_tms_summary[sample_name] = {'Average Tm (°C)': np.nan, 'Std Dev Tm (°C)': np.nan}


    # --- Add Average Tm and Std Dev to the Legend ---
    # Create a dummy plot entry for the legend
    ax.plot([], [], ' ', label=avg_tm_text)
    ax.set_ylabel("Corrected fluorescence")
    ax.set_title(f"Melting curves – {sample_name}")
    ax.legend(title="Concentration", loc='best')
    derivative_ax.axhline(0, color='gray', linewidth=0.8)
    derivative_ax.set_xlabel("Temperature (°C)")
    derivative_ax.set_ylabel("First derivative (fluorescence/°C)")
    derivative_ax.set_title("Derivative used for Tm (circles: selected; crosses: other peaks)")
    derivative_ax.legend(loc='best')
    fig.tight_layout()
    filename = f"{output_folder}/{sample_name}_tm_curve_derivative.png"
    fig.savefig(filename, dpi=180)
    plt.close(fig)

print("\n--- All Calculated Tm Values (Individual Concentrations) ---")
for key, value in all_tms_summary.items():
    if not np.isnan(value):
        print(f"{key}: {value:.2f} °C")
    else:
        print(f"{key}: Tm Not Determined (NaN)")

tms_df = pd.DataFrame.from_dict(all_tms_summary, orient='index', columns=['Tm (°C)'])
tms_df.index.name = 'Sample_Concentration'
tms_df.to_csv(f"{output_folder}/all_tms_individual_summary.csv") # Renamed for clarity
print(f"\nAll individual Tm values saved to {output_folder}/all_tms_individual_summary.csv")


# Print and save the average Tms summary
print("\n--- Average Tm Values (Per Sample Group) ---")
# Convert the average_tms_summary dictionary to a DataFrame
avg_tms_df = pd.DataFrame.from_dict(average_tms_summary, orient='index')
avg_tms_df.index.name = 'Sample_Group'

# Print to console
print(avg_tms_df.to_string()) # Use to_string() for better console formatting

# Save to CSV
avg_tms_filename = f"{output_folder}/average_tms_summary.csv"
avg_tms_df.to_csv(avg_tms_filename)
print(f"\nAverage Tm values saved to {avg_tms_filename}")


# --- NEW PLOT: Bar Chart of Average Tm Values ---
if not avg_tms_df.empty and not avg_tms_df['Average Tm (°C)'].dropna().empty:
    plt.figure(figsize=(12, 7))

    plot_df = avg_tms_df.dropna(subset=['Average Tm (°C)']).copy() # Use .copy() to avoid SettingWithCopyWarning

    if not plot_df.empty:
        plt.bar(plot_df.index,
                plot_df['Average Tm (°C)'],
                yerr=plot_df['Std Dev Tm (°C)'],
                capsize=5,
                color='skyblue',
                edgecolor='black',
                alpha=0.8)

        plt.xlabel("Sample Group", fontsize=12)
        plt.ylabel("Average Tm (°C)", fontsize=12)
        plt.title("Average Tm Values per Sample Group with Standard Deviation", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)

        for index, row in plot_df.iterrows():
            plt.text(index, row['Average Tm (°C)'],
                     f"{row['Average Tm (°C)']:.2f}",
                     ha='center', va='bottom', fontsize=9, color='black')

        plt.tight_layout()

        bar_chart_filename = f"{output_folder}/average_tms_bar_chart.png"
        try:
            plt.savefig(bar_chart_filename)
            print(f"\nBar chart of average Tm values with standard deviation saved to {bar_chart_filename}")
        except Exception as e:
            print(f"Error saving bar chart to {bar_chart_filename}: {e}")
        finally:
            plt.close()
    else:
        print("\nNo valid average Tm values available to create the overall bar chart after filtering NaN entries.")
else:
    print("\nNo average Tm data generated, skipping overall bar chart creation.")


# --- NEW PLOT: Bar Chart of Tm Values Relative to PsBDH ---
# Check if 'PsBDH' data is available and valid
if 'PsBDH' in avg_tms_df.index and not pd.isna(avg_tms_df.loc['PsBDH', 'Average Tm (°C)']):
    psBDH_avg_tm = avg_tms_df.loc['PsBDH', 'Average Tm (°C)']
    psBDH_std_tm = avg_tms_df.loc['PsBDH', 'Std Dev Tm (°C)']

    # Create a new DataFrame for relative Tms
    relative_tms_df = pd.DataFrame(index=avg_tms_df.index, columns=['Relative Tm (°C)', 'Combined Std Dev (°C)'])

    for sample_group in avg_tms_df.index:
        current_avg_tm = avg_tms_df.loc[sample_group, 'Average Tm (°C)']
        current_std_tm = avg_tms_df.loc[sample_group, 'Std Dev Tm (°C)']

        if not pd.isna(current_avg_tm) and not pd.isna(current_std_tm):
            relative_tm = current_avg_tm - psBDH_avg_tm
            
            # Combined standard deviation: sqrt(sigma1^2 + sigma2^2)
            combined_std_dev = np.sqrt(current_std_tm**2 + psBDH_std_tm**2)
            
            relative_tms_df.loc[sample_group, 'Relative Tm (°C)'] = relative_tm
            relative_tms_df.loc[sample_group, 'Combined Std Dev (°C)'] = combined_std_dev
        else:
            relative_tms_df.loc[sample_group, 'Relative Tm (°C)'] = np.nan
            relative_tms_df.loc[sample_group, 'Combined Std Dev (°C)'] = np.nan

    # Filter out any rows with NaN values for plotting
    plot_relative_df = relative_tms_df.dropna(subset=['Relative Tm (°C)'])

    if not plot_relative_df.empty:
        plt.figure(figsize=(12, 7))

        # Determine colors: highlight PsBDH or a different color for negative/positive relative Tms
        colors = ['lightcoral' if val < 0 else 'lightgreen' for val in plot_relative_df['Relative Tm (°C)']]
        # Or a single color: colors = 'lightcoral'
        
        # If PsBDH exists, make its bar a distinct color (e.g., 'grey') and its relative Tm will be 0
        if 'PsBDH' in plot_relative_df.index:
             # Find the index of 'PsBDH' and change its color to grey
            psbdh_idx = plot_relative_df.index.get_loc('PsBDH')
            colors[psbdh_idx] = 'lightgrey'


        plt.bar(plot_relative_df.index,
                plot_relative_df['Relative Tm (°C)'],
                yerr=plot_relative_df['Combined Std Dev (°C)'],
                capsize=5,
                color=colors, # Apply the dynamic colors
                edgecolor='black',
                alpha=0.8)

        # Add a horizontal line at y=0 for reference
        plt.axhline(0, color='grey', linestyle='--', linewidth=1)

        plt.xlabel("Sample Group", fontsize=12)
        plt.ylabel("Relative Tm vs PsBDH (°C)", fontsize=12)
        plt.title(f"Relative Tm Values vs PsBDH (Avg Tm: {psBDH_avg_tm:.2f}°C ± {psBDH_std_tm:.2f}°C)", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)

        # Add value labels on top of the bars
        for index, row in plot_relative_df.iterrows():
            plt.text(index, row['Relative Tm (°C)'],
                     f"{row['Relative Tm (°C)']:.2f}",
                     ha='center', va='bottom' if row['Relative Tm (°C)'] >= 0 else 'top',
                     fontsize=9, color='black')

        plt.tight_layout()

        relative_bar_chart_filename = f"{output_folder}/relative_tms_bar_chart_vs_PsBDH.png"
        try:
            plt.savefig(relative_bar_chart_filename)
            print(f"\nBar chart of relative Tm values vs PsBDH saved to {relative_bar_chart_filename}")
        except Exception as e:
            print(f"Error saving relative bar chart to {relative_bar_chart_filename}: {e}")
        finally:
            plt.close()
    else:
        print("\nNo valid relative Tm values available to create the relative bar chart after filtering NaN entries.")
else:
    print("\n'PsBDH' data not found or is invalid, skipping relative bar chart creation.")
