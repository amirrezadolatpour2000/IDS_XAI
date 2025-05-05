import os, lime
import numpy as np
import pandas as pd
import lime.lime_tabular
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


DATA_PATH         = "cicids2017_cleaned.csv"
OUT_DIR           = "plots"
RANDOM_STATE      = 42
TEST_SIZE         = 0.3
SAMPLE_RATE       = 0.1     
RF_N_ESTIMATORS   = 30     
LIME_NUM_SAMPLES  = 200     
N_LIME_FEATURES   = 10      
N_SAMPLES         = 3       

os.makedirs(OUT_DIR, exist_ok=True)


def load_data(data_path, sample_rate, random_state, label_col = 'Attack Type'):    
    df = pd.read_csv(data_path)
    df = df.sample(frac=sample_rate, random_state=random_state)
    X = df.drop(columns=[label_col])
    y = df[label_col]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, label_encoder, y_encoded

def split_data(X, encoded_y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, encoded_y,
        test_size=test_size,
        random_state=random_state,
        stratify=encoded_y
    )
    scaler  = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test  = pd.DataFrame(scaler.transform(X_test),    columns=X.columns)
    
    return X_train, X_test, y_train, y_test
    
def train_model( X_train, X_test, y_train, y_test, label_encoder, rf_n_estimators, random_state):
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=rf_n_estimators,
            n_jobs=-1,
            random_state=random_state
        ),
        "MLPClassifier": MLPClassifier(
            hidden_layer_sizes=(50, 50), 
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
        ),
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name} Report")
        preds = model.predict(X_test)
        print(classification_report(y_test, preds, target_names=label_encoder.classes_))

    return models

def explainer_lime(X, random_state, label_encoder, n_lime_features, lime_num_samples):
    contributions = defaultdict(dict)
    explainer = lime.lime_tabular.LimeTabularExplainer( X_train.values, feature_names = X.columns.tolist(), class_names = label_encoder.classes_.tolist(), discretize_continuous = True, random_state = random_state)
    
    for model_name, model in models.items():
        preds, proba = model.predict(X_test), model.predict_proba
        for idx, cls in enumerate(label_encoder.classes_):
            tp_idx = np.where((y_test == idx) & (preds == idx))[0]
            if tp_idx.size == 0:
                print(f"{model_name}: no TPs for '{cls}'")
                continue
            weight_acc = defaultdict(list)
            for i in tp_idx[:N_SAMPLES]:
                inst = X_test.iloc[i].values
                exp  = explainer.explain_instance(
                    inst, proba,
                    num_features=n_lime_features,
                    top_labels=1,
                    num_samples=lime_num_samples,
                )
                for feat, w in exp.as_list(label=idx):
                    weight_acc[feat].append(w)

            avg_w = {f: np.mean(ws) for f, ws in weight_acc.items()}
            contributions[cls][model_name] = avg_w

    return contributions



X, label_encoder, encoded_y = load_data(DATA_PATH, SAMPLE_RATE, RANDOM_STATE)
X_train, X_test, y_train, y_test = split_data(X, encoded_y, TEST_SIZE, RANDOM_STATE)
models = train_model(X_train, X_test, y_train, y_test, label_encoder, RF_N_ESTIMATORS, RANDOM_STATE)
contributions = explainer_lime(X, RANDOM_STATE, label_encoder, N_LIME_FEATURES, LIME_NUM_SAMPLES)

models_list, TOP_K, bar_width = list(models.keys()), N_LIME_FEATURES, 0.4
for cls, mdl_dict in contributions.items():
    feat_sets = []
    for w in mdl_dict.values():
        top_feats = sorted(w.items(), key=lambda kv: abs(kv[1]), reverse=True)[:TOP_K]
        feat_sets.append({f for f,_ in top_feats})
    feats = list(set().union(*feat_sets))

    y_pos = np.arange(len(feats))
    plt.figure(figsize=(8, 0.5*len(feats)+1))

    for i, model_name in enumerate(models_list):
        weights = mdl_dict.get(model_name, {})
        vals = [weights.get(f, 0.0) for f in feats]
        offset = (i - (len(models_list)-1)/2) * bar_width
        plt.barh(y_pos + offset, vals, height=bar_width, label=model_name)

    plt.yticks(y_pos, feats)
    plt.xlabel("Avg. LIME weight")
    plt.title(f"Feature contributions for '{cls}'")
    plt.legend(loc="best")
    plt.tight_layout()

    fname = os.path.join(OUT_DIR, f"combined__{cls}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")
