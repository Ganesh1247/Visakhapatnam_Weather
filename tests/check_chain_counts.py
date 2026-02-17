import pickle
import os

models_dir = 'models'
for f in os.listdir(models_dir):
    if f.startswith('xgb_chain_') and f.endswith('.pkl'):
        path = os.path.join(models_dir, f)
        with open(path, 'rb') as file:
            model = pickle.load(file)
            count = getattr(model, 'n_features_in_', 'Unknown')
            print(f"{f}: {count}")
