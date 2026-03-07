from hmmlearn.hmm import CategoricalHMM
import pandas as pd
import numpy as np
import os

prefix_to_symbol = {"41": 0, "42": 1, "43": 2, "44": 3}
symbol_to_prefix = {0: "41", 1: "42", 2: "43", 3: "44"}

def train_model_from_in_progress(csv_path="in_progress.csv"):
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if df.empty or len(df) < 2:
        return None

    symbols = []
    for v in df["Formation"]:
        prefix = str(v)[:2]
        if prefix in prefix_to_symbol:
            symbols.append(prefix_to_symbol[prefix])

    if len(symbols) < 2:
        return None

    X = np.array(symbols, dtype=int).reshape(-1, 1)

    transmat_prior = np.array([
        [10.0, 10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0, 10.0],
    ])

    try:
        model = CategoricalHMM(n_components=4, n_iter=200, random_state=0, n_features=4)
    except TypeError:
        model = CategoricalHMM(n_components=4, n_iter=200, random_state=0)
        model.n_features = 4

    model.startprob_prior = np.array([1.0, 1.0, 1.0, 1.0])
    model.transmat_prior = transmat_prior
    model.emissionprob_prior = np.ones((4, 4))

    model.fit(X)
    return model

def predict_next_from_prefix(model, last_prefix):
    last_state = prefix_to_symbol[last_prefix]
    probs = model.transmat_[last_state]
    next_state = int(np.argmax(probs))
    return symbol_to_prefix[next_state], probs
