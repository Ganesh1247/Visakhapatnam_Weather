import pandas as pd
import os

def generate_metrics():
    """
    Read the real metrics from the CSV written by train_hybrid_model.py.
    The CSV is at <project_root>/data/metrics_scientific.csv.
    Previously this function returned hardcoded fake values — now it returns
    the live post-training ground truth.
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # training script writes here
    TRAINING_CSV = os.path.join(BASE_DIR, "data", "metrics_scientific.csv")
    # legacy path kept for backward compat (used by /stats)
    OUTPUT_PATH  = os.path.join(BASE_DIR, "metrics_scientific.csv")

    if not os.path.exists(TRAINING_CSV):
        print(f"[generate_metrics] WARNING: {TRAINING_CSV} not found – using empty metrics.")
        return

    df = pd.read_csv(TRAINING_CSV)

    # Normalise column names – training script uses 'Target','RMSE','MAE','R2'
    col_map = {c.lower(): c for c in df.columns}
    rename = {}
    for src in list(df.columns):
        lc = src.lower()
        if lc == 'target':   rename[src] = 'target'
        elif lc == 'rmse':   rename[src] = 'rmse'
        elif lc == 'mae':    rename[src] = 'mae'
        elif lc in ('r2', 'r²', 'r2_score'): rename[src] = 'r2'
    df = df.rename(columns=rename)

    # Write to the legacy path so the /stats route picks it up without change
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[generate_metrics] Synced real training metrics → {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_metrics()
