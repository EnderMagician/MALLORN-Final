import random
import pandas as pd
import matplotlib.pyplot as plt

# SEED = 15
# random.seed(SEED)

FILTER_COLORS = {
    'u': 'blue',      # Ultraviolet
    'g': 'green',     # Green
    'r': 'red',       # Red
    'i': 'orange',    # Infrared (Near)
    'z': 'brown',     # Infrared (Mid)
    'y': 'black'      # Infrared (Far)
}

# Lightcurve Plotting Function
def plot_lightcurve(object_id, df):
    obj_data = df[df['object_id'] == object_id]
    
    if obj_data.empty:
        print(f"Error: Object ID '{object_id}' not found in the dataframe.")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Loop through each filter color and plot it
    for filt in ['u', 'g', 'r', 'i', 'z', 'y']:
        # Get data for just this filter
        filt_data = obj_data[obj_data['Filter'] == filt]
        
        if not filt_data.empty:
            plt.errorbar(
                x=filt_data['Time (MJD)'], 
                y=filt_data['Flux'], 
                yerr=filt_data['Flux_err'], # Vertical error bars
                fmt='o',                    # 'o' = circle markers
                color=FILTER_COLORS.get(filt, 'gray'),
                label=f'Band {filt}',
                alpha=0.7,                  # Transparency
                markersize=5,
                capsize=2                   # Little caps on the error bars
            )

    # 4. Formatting
    plt.title(f"Lightcurve for Object: {object_id}", fontsize=14, fontweight='bold')
    plt.xlabel("Time (MJD)", fontsize=12)
    plt.ylabel("Flux (microjansky)", fontsize=12)
    plt.legend(title="Filter")
    plt.grid(True, alpha=0.3)
    
    plt.show()

# Random Lightcurve Plotter
def plot_random(log_df=pd.DataFrame, lc_df=pd.DataFrame, TDE=bool):
    if log_df.empty or lc_df.empty:
        print("Error: log_df and lc_df must be provided and non-empty.")
        return
    
    if TDE:
        candidate_ids = log_df[log_df['target'] == 1]['object_id'].unique()
    else:
        candidate_ids = log_df[log_df['target'] == 0]['object_id'].unique()
        
    if len(candidate_ids) == 0:
        print("Error: No candidate IDs found for the specified target type.")
        return
    
    random_id = random.choice(candidate_ids)
    plot_lightcurve(random_id, lc_df)