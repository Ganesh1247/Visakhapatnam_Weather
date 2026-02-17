import pickle
import pandas as pd
import os

def check_pm10():
    path = "models/xgb_chain_pm10.pkl"
    with open(path, "rb") as f:
        model = pickle.load(f)
    
    print(f"Model keys: {model.__dict__.keys()}")
    if hasattr(model, "feature_names_in_"):
        print(f"feature_names_in_: {list(model.feature_names_in_)}")
        print(f"Count: {len(model.feature_names_in_)}")
    else:
        # Native XGB
        booster = model.get_booster()
        print(f"Booster feature names: {booster.feature_names}")
        print(f"Count: {len(booster.feature_names)}")

if __name__ == "__main__":
    check_pm10()
