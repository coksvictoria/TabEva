import os
import pickle
from typing import List

from tabeva.metrics.sample import plot_nnd
import pandas as pd


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def main(distances_path: str = os.path.join("output", "adult", "distances.pickle"),
         results_path: str = os.path.join("output", "adult", "results.pickle"),
         output_dir: str = os.path.join("output", "adult")) -> None:
    if not os.path.exists(distances_path):
        raise FileNotFoundError(f"Distances pickle not found: {distances_path}")

    distances = load_pickle(distances_path)

    # try to load names from results.pickle to match order
    names: List[str] = None
    if os.path.exists(results_path):
        try:
            results = load_pickle(results_path)
            names = list(results.keys())
        except Exception:
            names = None

    # Normalize distances into a list
    if isinstance(distances, dict):
        names = list(distances.keys())
        dfs = [distances[k] for k in names]
    elif isinstance(distances, list):
        dfs = distances
        if names is None:
            names = [f"synth_{i}" for i in range(len(dfs))]
    else:
        raise ValueError("Unsupported distances object in pickle. Expected list or dict.")

    sample_out = os.path.join(output_dir, "sample")
    os.makedirs(sample_out, exist_ok=True)

    # Build combined d1nn DataFrame: one column per synthesizer
    d1nn_plot = {}
    dc_plot = {}
    for idx, dis in enumerate(dfs):
        if isinstance(dis, dict):
            # sometimes stored as dict; try to retrieve DataFrame
            continue
        if "distance_1nn" in dis.columns:
            d1nn_plot[idx] = dis["distance_1nn"].reset_index(drop=True)
        
        if "distance_to_centroid" in dis.columns:
            dc_plot[idx] = dis["distance_to_centroid"].reset_index(drop=True)

    if not d1nn_plot:
        print("No d1nn distances found in pickle.")
        return
    
    if not dc_plot:
        print("No distance_to_centroid distances found in pickle.")
        return

    d_1nns_df = pd.DataFrame(d1nn_plot)
    dc_df = pd.DataFrame(dc_plot)

    d_1nns_df.columns=['SMOTE','TVAE','CTGAN','CTABGAN','DP-CTGAN','TabDDPM','GReaT','Tabula','TabSyn']
    dc_df.columns=['SMOTE','TVAE','CTGAN','CTABGAN','DP-CTGAN','TabDDPM','GReaT','Tabula','TabSyn']

    plot_nnd(d_1nns_df, filename=os.path.join(sample_out, "d1nn_all.pdf"))
    plot_nnd(dc_df, filename=os.path.join(sample_out, "dc_all.pdf"))



if __name__ == "__main__":
    main()
