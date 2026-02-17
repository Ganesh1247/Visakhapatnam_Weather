import pickle
import os

models_dir = 'models'
for f in os.listdir(models_dir):
    if f.endswith('.pkl'):
        path = os.path.join(models_dir, f)
        with open(path, 'rb') as file:
            model = pickle.load(file)
            if hasattr(model, 'n_features_in_'):
                print(f"{f}: {model.n_features_in_}")
            else:
                print(f"{f}: (No n_features_in_)")
