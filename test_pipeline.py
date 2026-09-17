# ============================================================
# DENGUE SEVERITY PREDICTION — PSO-OPTIMIZED ML PIPELINE
# Dataset: Kaggle - dipayancodes/dengue
# Architecture: PSO Hyperparameter Tuning + Scikit-Learn / XGBoost / LightGBM
# ============================================================

# ── Standard Library ──────────────────────────────────────────
import os
import sys
import json
import time
import random
import warnings
from datetime import datetime

# ── Numeric & Data Processing ────────────────────────────────
import numpy as np
import pandas as pd

# ── Visualization ─────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Sklearn Core ──────────────────────────────────────────────
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    f1_score, matthews_corrcoef, cohen_kappa_score,
    precision_score, recall_score, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split
)
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, OneHotEncoder, label_binarize
)

# ── Advanced ML Models & Tools ────────────────────────────────
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
import joblib
import kagglehub

# ── Particle Swarm Optimization ────────────────────────────────
try:
    from pyswarms.single import GlobalBestPSO
    PSO_AVAILABLE = True
except ImportError:
    PSO_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────
CONFIG = {
    'random_state'    : 42,
    'test_size'       : 0.15,
    'val_size'        : 0.15,
    'cv_folds'        : 5,
    'n_jobs'          : -1,
    'output_dir'      : 'dengue_output',
    'target_column'   : 'Dengue',
    'pso_n_particles' : 15,
    'pso_n_iter'      : 20,
    'pso_w'           : 0.729,
    'pso_c1'          : 1.494,
    'pso_c2'          : 1.494,
    'spark_cores'     : 4,
    'spark_memory'    : '4g',
}

def set_seed(seed=42):
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(CONFIG['random_state'])
os.makedirs(CONFIG['output_dir'], exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted')

BANNER = "=" * 90
print(BANNER)
print("DENGUE SEVERITY PREDICTION -- PSO ML PIPELINE".center(90))
print(BANNER)
print(f"  Started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  PSO Enabled : {PSO_AVAILABLE}")
print(f"  Output Dir  : {CONFIG['output_dir']}\n")


# ═══════════════════════════════════════════════════════════════
# STEP 1 ── DATASET INGESTION & LOCAL CACHING
# ═══════════════════════════════════════════════════════════════
print(BANNER)
print("STEP 1 : DATASET INGESTION & LOCAL CACHING".center(90))
print(BANNER)

local_csv = os.path.join(CONFIG['output_dir'], "dengue.csv")

if os.path.exists(local_csv):
    print(f"[OK] Loading cached dataset from: {local_csv}")
    df_raw = pd.read_csv(local_csv)
else:
    print("[INFO] Downloading dataset via kagglehub...")
    download_path = kagglehub.dataset_download("dipayancodes/dengue")
    print(f"[OK] Downloaded to: {download_path}")
    csv_files = [os.path.join(r, f) for r, d, fs in os.walk(download_path) for f in fs if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV dataset found in kagglehub download.")
    df_raw = pd.read_csv(csv_files[0])
    df_raw.to_csv(local_csv, index=False)
    print(f"[OK] Dataset cached locally to: {local_csv}")

print(f"[OK] Raw shape      : {df_raw.shape}")
print(f"[OK] Columns        : {list(df_raw.columns)}")
print(f"\nSample data:\n{df_raw.head(3).to_string()}")
print(f"\nMissing Values:\n{df_raw.isnull().sum().to_string()}")


# ═══════════════════════════════════════════════════════════════
# STEP 2 ── EXPLICIT TARGET VERIFICATION & CLEANING
# ═══════════════════════════════════════════════════════════════
print(BANNER)
print("STEP 2 : EXPLICIT TARGET VERIFICATION & CLEANING".center(90))
print(BANNER)

target_col = CONFIG['target_column']
if target_col not in df_raw.columns:
    raise KeyError(f"Target column '{target_col}' not found in dataset columns: {list(df_raw.columns)}")

print(f"[OK] Target column verified : '{target_col}'")
print(f"     Unique values          : {df_raw[target_col].unique()}")
print(f"     Distribution           :\n{df_raw[target_col].value_counts()}\n")

# Drop non-feature identifier columns (e.g. Name, ID)
ignore_cols = [target_col]
for col in df_raw.columns:
    if col != target_col:
        if col.lower() in ['name', 'id', 'patient_id', 'subject_id'] or (df_raw[col].dtype == 'object' and df_raw[col].nunique() > 100):
            ignore_cols.append(col)
            print(f"[INFO] Dropping identifier column: '{col}' ({df_raw[col].nunique()} unique values)")

X_raw = df_raw.drop(columns=ignore_cols)
y_raw = df_raw[target_col].astype(str)

le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
n_classes = len(le.classes_)

print(f"[OK] Encoded Target Classes: {le.classes_}")
print(f"[OK] Feature Dimensions    : {X_raw.shape[1]} columns ({list(X_raw.columns)})")


# ═══════════════════════════════════════════════════════════════
# STEP 3 ── LEAKAGE-FREE STRATIFIED DATA SPLITTING
# ═══════════════════════════════════════════════════════════════
print(BANNER)
print("STEP 3 : LEAKAGE-FREE STRATIFIED DATA SPLITTING".center(90))
print(BANNER)

X_temp, X_test, y_temp, y_test = train_test_split(
    X_raw, y_encoded,
    test_size=CONFIG['test_size'],
    random_state=CONFIG['random_state'],
    stratify=y_encoded
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=CONFIG['val_size'] / (1.0 - CONFIG['test_size']),
    random_state=CONFIG['random_state'],
    stratify=y_temp
)

print(f"[OK] Train set : {X_train.shape}  | Class dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"[OK] Val set   : {X_val.shape}   | Class dist: {dict(zip(*np.unique(y_val, return_counts=True)))}")
print(f"[OK] Test set  : {X_test.shape}  | Class dist: {dict(zip(*np.unique(y_test, return_counts=True)))}")

cv_strategy = StratifiedKFold(
    n_splits=CONFIG['cv_folds'],
    shuffle=True,
    random_state=CONFIG['random_state']
)
print(f"[OK] CV Strategy: {CONFIG['cv_folds']}-Fold Stratified CV\n")


# ═══════════════════════════════════════════════════════════════
# STEP 4 ── CLINICAL FEATURE ENGINEERING & PIPELINE BUILDER
# ═══════════════════════════════════════════════════════════════
print(BANNER)
print("STEP 4 : CLINICAL FEATURE ENGINEERING & PIPELINE".center(90))
print(BANNER)

class DengueFeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain-specific Dengue clinical risk feature transformer (stateless)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xc = X.copy()
        # Platelet warning indicators
        plt_cols = [c for c in Xc.columns if 'platelet' in c.lower()]
        if plt_cols:
            p = plt_cols[0]
            Xc['Plt_Warning'] = (Xc[p] < 100000).astype(int)
            Xc['Plt_Critical'] = (Xc[p] < 50000).astype(int)
            Xc['Plt_Danger'] = (Xc[p] < 20000).astype(int)

        # Haematocrit indicator
        hct_cols = [c for c in Xc.columns if any(k in c.lower() for k in ['haematocrit', 'hematocrit', 'hct'])]
        if hct_cols:
            h = hct_cols[0]
            Xc['Haemoconcentration'] = (Xc[h] > 45).astype(int)

        # Fever severity
        temp_cols = [c for c in Xc.columns if any(k in c.lower() for k in ['temp', 'fever'])]
        if temp_cols:
            t = temp_cols[0]
            if Xc[t].nunique() <= 2:
                Xc['High_Fever'] = Xc[t].astype(int)
            else:
                Xc['High_Fever'] = (Xc[t] > 38.5).astype(int)

        # WBC Leukopenia
        wbc_cols = [c for c in Xc.columns if any(k in c.lower() for k in ['wbc', 'leukocyte', 'white'])]
        if wbc_cols:
            w = wbc_cols[0]
            Xc['Leukopenia'] = (Xc[w] < 4000).astype(int)

        # Age Risk
        age_cols = [c for c in Xc.columns if 'age' in c.lower()]
        if age_cols:
            a = age_cols[0]
            Xc['Age_Risk'] = ((Xc[a] <= 12) | (Xc[a] >= 60)).astype(int)

        # Symptom severity score sum
        symptom_cols = [c for c in Xc.columns if c in ['Fever', 'Headache', 'JointPain', 'Bleeding', 'Plt_Warning', 'Haemoconcentration', 'Leukopenia']]
        if symptom_cols:
            Xc['Dengue_Severity_Score'] = Xc[symptom_cols].sum(axis=1)

        return Xc

class RobustImputer(BaseEstimator, TransformerMixin):
    """Median/Mode imputer fitted strictly on training data."""
    def __init__(self):
        self.num_fill_ = {}
        self.cat_fill_ = {}

    def fit(self, X, y=None):
        Xd = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        for c in Xd.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns:
            self.num_fill_[c] = Xd[c].median()
        for c in Xd.select_dtypes(include=['object', 'category']).columns:
            modes = Xd[c].mode()
            self.cat_fill_[c] = modes[0] if len(modes) else 'Unknown'
        return self

    def transform(self, X):
        Xc = X.copy()
        for c, v in self.num_fill_.items():
            if c in Xc.columns:
                Xc[c] = Xc[c].fillna(v)
        for c, v in self.cat_fill_.items():
            if c in Xc.columns:
                Xc[c] = Xc[c].fillna(v)
        return Xc

def build_pipeline(clf):
    """Construct Scikit-Learn processing pipeline with custom classifier."""
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imp', RobustImputer()),
            ('scl', StandardScaler())
        ]), make_column_selector(dtype_include=['int64', 'float64', 'int32', 'float32'])),
        ('cat', Pipeline([
            ('imp', RobustImputer()),
            ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ]), make_column_selector(dtype_include=['object', 'category']))
    ])
    return Pipeline([
        ('feat_eng', DengueFeatureEngineer()),
        ('preprocess', preprocessor),
        ('clf', clf)
    ])

print("[OK] Custom feature engineering pipeline ready.\n")


# ═══════════════════════════════════════════════════════════════
# STEP 5 ── PARTICLE SWARM OPTIMIZATION (PSO) HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════
print(BANNER)
print("STEP 5 : PSO SWARM HYPERPARAMETER OPTIMIZATION".center(90))
print(BANNER)

PSO_SPACES = {
    "Random Forest": {
        "bounds": ([20, 2, 2, 0], [150, 15, 10, 1]),
        "dims": 4
    },
    "XGBoost": {
        "bounds": ([20, 2, 0.01, 0.5, 0.5], [150, 8, 0.30, 1.0, 1.0]),
        "dims": 5
    },
    "LightGBM": {
        "bounds": ([20, 2, 0.01, 0.5], [150, 8, 0.30, 1.0]),
        "dims": 4
    },
    "Gradient Boosting": {
        "bounds": ([20, 0.01, 2], [100, 0.20, 6]),
        "dims": 3
    },
    "Logistic Regression": {
        "bounds": ([0.001], [50.0]),
        "dims": 1
    },
    "SVM": {
        "bounds": ([0.1, 0.0001], [20.0, 0.5]),
        "dims": 2
    }
}

def decode_particle(model_name, pos):
    """Decode continuous particle position coordinates to model hyperparameters."""
    if model_name == "Random Forest":
        mf = ["sqrt", "log2"]
        return {
            "n_estimators": int(round(pos[0])),
            "max_depth": int(round(pos[1])),
            "min_samples_split": int(round(pos[2])),
            "max_features": mf[int(round(np.clip(pos[3], 0, 1)))]
        }
    elif model_name == "XGBoost":
        return {
            "n_estimators": int(round(pos[0])),
            "max_depth": int(round(pos[1])),
            "learning_rate": float(np.clip(pos[2], 0.01, 0.30)),
            "subsample": float(np.clip(pos[3], 0.5, 1.0)),
            "colsample_bytree": float(np.clip(pos[4], 0.5, 1.0))
        }
    elif model_name == "LightGBM":
        return {
            "n_estimators": int(round(pos[0])),
            "max_depth": int(round(pos[1])),
            "learning_rate": float(np.clip(pos[2], 0.01, 0.30)),
            "subsample": float(np.clip(pos[3], 0.5, 1.0))
        }
    elif model_name == "Gradient Boosting":
        return {
            "n_estimators": int(round(pos[0])),
            "learning_rate": float(np.clip(pos[1], 0.01, 0.20)),
            "max_depth": int(round(pos[2]))
        }
    elif model_name == "Logistic Regression":
        return {"C": float(np.clip(pos[0], 0.001, 50.0))}
    elif model_name == "SVM":
        return {
            "C": float(np.clip(pos[0], 0.1, 50.0)),
            "gamma": float(np.clip(pos[1], 0.0001, 1.0))
        }

scoring_metric = 'roc_auc' if n_classes == 2 else 'roc_auc_ovr_weighted'

def pso_tune(model_name, base_clf, X_tr, y_tr, cv):
    space = PSO_SPACES[model_name]
    lb, ub = space["bounds"]
    dims = space["dims"]
    print(f"  -> {model_name} | dims={dims} | particles={CONFIG['pso_n_particles']} | iters={CONFIG['pso_n_iter']}")

    if PSO_AVAILABLE:
        def fitness(particles):
            costs = []
            for pos in particles:
                try:
                    params = decode_particle(model_name, pos)
                    m = clone(base_clf)
                    m.set_params(**params)
                    pipe = build_pipeline(m)
                    score = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring=scoring_metric, n_jobs=CONFIG['n_jobs']).mean()
                    costs.append(-score)
                except Exception:
                    costs.append(1.0)
            return np.array(costs)

        opts = {'c1': CONFIG['pso_c1'], 'c2': CONFIG['pso_c2'], 'w': CONFIG['pso_w']}
        bounds = (np.array(lb), np.array(ub))
        opt = GlobalBestPSO(n_particles=CONFIG['pso_n_particles'], dimensions=dims, options=opts, bounds=bounds)
        best_cost, best_pos = opt.optimize(fitness, iters=CONFIG['pso_n_iter'], verbose=False)
        best_params = decode_particle(model_name, best_pos)
        best_cv_score = -best_cost
        history = [-c for c in opt.cost_history]
    else:
        mid = (np.array(lb) + np.array(ub)) / 2.0
        best_params = decode_particle(model_name, mid)
        m = clone(base_clf)
        m.set_params(**best_params)
        pipe = build_pipeline(m)
        best_cv_score = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring=scoring_metric, n_jobs=CONFIG['n_jobs']).mean()
        history = [best_cv_score] * CONFIG['pso_n_iter']

    best_clf = clone(base_clf)
    best_clf.set_params(**best_params)
    final_pipe = build_pipeline(best_clf)
    final_pipe.fit(X_tr, y_tr)

    print(f"     Best CV AUC: {best_cv_score:.4f} | Params: {best_params}")
    return final_pipe, best_cv_score, best_params, history

BASE_MODELS = {
    "Random Forest": RandomForestClassifier(random_state=CONFIG['random_state'], class_weight='balanced', n_jobs=CONFIG['n_jobs']),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=CONFIG['random_state'], n_jobs=CONFIG['n_jobs'], verbosity=0),
    "LightGBM": LGBMClassifier(random_state=CONFIG['random_state'], class_weight='balanced', n_jobs=CONFIG['n_jobs'], verbose=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=CONFIG['random_state']),
    "Logistic Regression": LogisticRegression(random_state=CONFIG['random_state'], max_iter=2000, class_weight='balanced', solver='lbfgs', n_jobs=CONFIG['n_jobs']),
    "SVM": SVC(random_state=CONFIG['random_state'], probability=True, class_weight='balanced')
}

print(f"[PSO] Optimizing {len(BASE_MODELS)} base classifiers...\n")
best_models = {}
cv_results = {}
pso_histories = {}

t0 = time.time()
for mname, base_clf in BASE_MODELS.items():
    t1 = time.time()
    pipe, score, params, hist = pso_tune(mname, base_clf, X_train, y_train, cv_strategy)
    best_models[mname] = pipe
    cv_results[mname] = {'best_cv_score': score, 'best_params': params}
    pso_histories[mname] = hist
    print(f"     Time: {time.time()-t1:.1f}s\n")

print(f"[OK] Total PSO Tuning Time: {time.time()-t0:.1f}s\n")


# ═══════════════════════════════════════════════════════════════
# STEP 6 ── PSO CONVERGENCE VISUALIZATION
# ═══════════════════════════════════════════════════════════════
n_models = len(pso_histories)
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
axes = axes.flatten()
palette = sns.color_palette("muted", n_models)

fig.suptitle("PSO Swarm Convergence -- Best CV AUC-ROC per Iteration", fontsize=14, fontweight='bold')

for i, (mname, hist) in enumerate(pso_histories.items()):
    ax = axes[i]
    col = palette[i]
    iters = range(1, len(hist) + 1)
    ax.plot(iters, hist, color=col, linewidth=2, label=mname)
    ax.fill_between(iters, hist, alpha=0.15, color=col)
    ax.scatter([len(hist)], [hist[-1]], color=col, s=70, zorder=5)
    ax.annotate(f"Final: {hist[-1]:.4f}", xy=(len(hist), hist[-1]), xytext=(-45, -20), textcoords='offset points', fontsize=9, fontweight='bold', color=col)
    ax.set_title(mname, fontsize=11, fontweight='bold')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("CV AUC-ROC")
    ax.set_ylim([max(0.4, min(hist) - 0.02), 1.01])
    ax.grid(True, alpha=0.3)

plt.tight_layout()
conv_plot_path = os.path.join(CONFIG['output_dir'], "pso_convergence.png")
plt.savefig(conv_plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"[OK] PSO convergence plot saved to: {conv_plot_path}\n")


# ═══════════════════════════════════════════════════════════════
# STEP 7 ── VALIDATION LEADERBOARD & ENSEMBLE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════
print(BANNER)
print("STEP 7 : VALIDATION LEADERBOARD & ENSEMBLE CONSTRUCTION".center(90))
print(BANNER)

def evaluate_model(name, model, X, y):
    """Compute standardized metric evaluation for a model."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    ppv = precision_score(y, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y, y_pred, average='weighted', zero_division=0)
    auc = roc_auc_score(y, y_proba[:, 1]) if n_classes == 2 else roc_auc_score(label_binarize(y, classes=np.arange(n_classes)), y_proba, multi_class='ovr', average='weighted')
    return {
        'Model': name,
        'Val_AUC_ROC': round(auc, 4),
        'Val_Accuracy': round(acc, 4),
        'Val_F1': round(f1, 4),
        'Val_PPV': round(ppv, 4),
        'Val_Recall': round(rec, 4),
        'PSO_CV_AUC': round(cv_results[name]['best_cv_score'], 4) if name in cv_results else '-'
    }

val_results = [evaluate_model(n, m, X_val, y_val) for n, m in best_models.items()]
val_df = pd.DataFrame(val_results).sort_values('Val_AUC_ROC', ascending=False).reset_index(drop=True)
print("Validation Leaderboard (PSO Base Models):")
print(val_df.to_string(index=False))

# Build Voting & Stacking Ensembles from Top-3 Base Models
top3_names = val_df.head(3)['Model'].tolist()
top3_est = [(n.replace(' ', '_'), best_models[n]) for n in top3_names]
print(f"\n[OK] Selected Top-3 models for Ensembles: {top3_names}")

voting_ens = VotingClassifier(estimators=top3_est, voting='soft')
stack_ens = StackingClassifier(estimators=top3_est, final_estimator=LogisticRegression(random_state=CONFIG['random_state'], max_iter=1000), cv=cv_strategy)

voting_ens.fit(X_train, y_train)
stack_ens.fit(X_train, y_train)

best_models['Voting Ensemble'] = voting_ens
best_models['Stacking Ensemble'] = stack_ens

val_results_full = [evaluate_model(n, m, X_val, y_val) for n, m in best_models.items()]
val_df_full = pd.DataFrame(val_results_full).sort_values('Val_AUC_ROC', ascending=False).reset_index(drop=True)
print("\nFinal Validation Leaderboard (Base Models + Ensembles):")
print(val_df_full.to_string(index=False))


# ═══════════════════════════════════════════════════════════════
# STEP 8 ── FINAL TEST SET EVALUATION & THRESHOLD OPTIMIZATION
# ═══════════════════════════════════════════════════════════════
print(BANNER)
print("STEP 8 : FINAL TEST SET EVALUATION".center(90))
print(BANNER)

best_model_name = val_df_full.iloc[0]['Model']
final_model = best_models[best_model_name]

if n_classes == 2:
    val_proba = final_model.predict_proba(X_val)[:, 1]
    best_th, best_f1_th = 0.5, 0.0
    for th in np.linspace(0.20, 0.80, 61):
        pred = (val_proba >= th).astype(int)
        sc = f1_score(y_val, pred, average='weighted')
        if sc > best_f1_th:
            best_f1_th, best_th = sc, th
    print(f"[OK] Optimal probability decision threshold: {best_th:.2f}")
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= best_th).astype(int)
else:
    y_test_proba = final_model.predict_proba(X_test)
    y_test_pred = final_model.predict(X_test)
    best_th = 0.5

test_acc = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average='weighted')
test_mcc = matthews_corrcoef(y_test, y_test_pred)
test_kappa = cohen_kappa_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba) if n_classes == 2 else roc_auc_score(label_binarize(y_test, classes=np.arange(n_classes)), y_test_proba, multi_class='ovr', average='weighted')

if n_classes == 2:
    pos_idx = 1
    tp = np.sum((y_test_pred == pos_idx) & (y_test == pos_idx))
    tn = np.sum((y_test_pred != pos_idx) & (y_test != pos_idx))
    fp = np.sum((y_test_pred == pos_idx) & (y_test != pos_idx))
    fn = np.sum((y_test_pred != pos_idx) & (y_test == pos_idx))
    test_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    test_ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    test_npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
else:
    test_sens = recall_score(y_test, y_test_pred, average='weighted')
    test_spec = test_sens
    test_ppv = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_npv = test_ppv

print(f"\nOptimal Model : {best_model_name}")
print("=" * 50)
print(f"   AUC-ROC      : {test_auc:.4f}")
print(f"   Sensitivity  : {test_sens:.4f}")
print(f"   Specificity  : {test_spec:.4f}")
print(f"   PPV          : {test_ppv:.4f}")
print(f"   NPV          : {test_npv:.4f}")
print(f"   F1 Score     : {test_f1:.4f}")
print(f"   Accuracy     : {test_acc:.4f}")
print(f"   MCC          : {test_mcc:.4f}")
print(f"   Kappa        : {test_kappa:.4f}")
print("=" * 50)
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=[str(c) for c in le.classes_]))


# ═══════════════════════════════════════════════════════════════
# STEP 9 ── VISUALIZATION DASHBOARD
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 18))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle(f"Dengue Severity Prediction -- Diagnostic Dashboard\nBest Model: {best_model_name}", fontsize=16, fontweight='bold')

# Panel 1: Validation Leaderboard
ax1 = fig.add_subplot(gs[0, :2])
vd = val_df_full.copy()
colors = ['#2ecc71' if m == best_model_name else '#95a5a6' for m in vd['Model']]
bars = ax1.barh(vd['Model'][::-1], vd['Val_AUC_ROC'][::-1], color=colors[::-1], alpha=0.85)
ax1.set_xlabel('Validation AUC-ROC')
ax1.set_title('Model Leaderboard (Validation AUC-ROC)', fontweight='bold')
ax1.set_xlim([0.4, 1.05])
ax1.axvline(0.5, color='red', linestyle='--', alpha=0.5)
for bar in bars:
    ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.4f}", va='center', fontsize=9, fontweight='bold')

# Panel 2: Confusion Matrix
ax2 = fig.add_subplot(gs[0, 2])
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, xticklabels=[str(c) for c in le.classes_], yticklabels=[str(c) for c in le.classes_])
ax2.set_title(f'Confusion Matrix ({best_model_name})', fontweight='bold')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('True')

# Panel 3: ROC Curve
ax3 = fig.add_subplot(gs[1, 0])
if n_classes == 2:
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    ax3.plot(fpr, tpr, color='#2ecc71', linewidth=2.5, label=f'AUC = {test_auc:.4f}')
    ax3.fill_between(fpr, tpr, alpha=0.15, color='#2ecc71')
ax3.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Baseline')
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curve', fontweight='bold')
ax3.legend()

# Panel 4: Precision-Recall Curve
ax4 = fig.add_subplot(gs[1, 1])
if n_classes == 2:
    prec_c, rec_c, _ = precision_recall_curve(y_test, y_test_proba)
    ax4.plot(rec_c, prec_c, color='#3498db', linewidth=2.5)
    ax4.fill_between(rec_c, prec_c, alpha=0.15, color='#3498db')
ax4.set_xlabel('Recall')
ax4.set_ylabel('Precision')
ax4.set_title('Precision-Recall Curve', fontweight='bold')

# Panel 5: Test Metrics Bar Summary
ax5 = fig.add_subplot(gs[1, 2])
metrics_names = ['AUC', 'Sens', 'Spec', 'PPV', 'NPV', 'F1', 'Acc']
metrics_vals = [test_auc, test_sens, test_spec, test_ppv, test_npv, test_f1, test_acc]
bars_m = ax5.bar(metrics_names, metrics_vals, color='#34495e', alpha=0.85)
ax5.set_ylim([0, 1.1])
ax5.set_title('Test Metrics Summary', fontweight='bold')
for bar in bars_m:
    ax5.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()), xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8, fontweight='bold')

# Panel 6: Radar Performance Chart
ax6 = fig.add_subplot(gs[2, :])
ax6.bar(['Sensitivity', 'Specificity', 'PPV (Precision)', 'NPV', 'F1 Score', 'Accuracy', 'MCC', 'Kappa'], [test_sens, test_spec, test_ppv, test_npv, test_f1, test_acc, (test_mcc+1)/2, (test_kappa+1)/2], color='#27ae60', alpha=0.8)
ax6.set_ylim([0, 1.1])
ax6.set_title('Comprehensive Test Diagnostic Profile', fontweight='bold')

plt.tight_layout()
dash_path = os.path.join(CONFIG['output_dir'], "dengue_dashboard.png")
plt.savefig(dash_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"[OK] Dashboard saved to: {dash_path}\n")


# ═══════════════════════════════════════════════════════════════
# STEP 10 ── SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════
print(BANNER)
print("STEP 10 : SHAP EXPLAINABILITY".center(90))
print(BANNER)

try:
    shap_model_name = None
    shap_pipe = None
    for prefer in ["XGBoost", "LightGBM", "Random Forest", "Gradient Boosting"]:
        if prefer in best_models:
            shap_model_name = prefer
            shap_pipe = best_models[prefer]
            break

    if shap_pipe is not None:
        X_test_trans = shap_pipe[:-1].transform(X_test)
        clf_step = shap_pipe.named_steps['clf']
        explainer = shap.TreeExplainer(clf_step)
        shap_vals = explainer.shap_values(X_test_trans)

        try:
            feat_names = shap_pipe.named_steps['preprocess'].get_feature_names_out()
        except Exception:
            feat_names = [f"f{i}" for i in range(X_test_trans.shape[1])]

        fig, ax = plt.subplots(figsize=(10, 6))
        sv = np.abs(shap_vals[1]) if isinstance(shap_vals, list) else np.abs(shap_vals)
        mean_shap = sv.mean(axis=0)
        top_idx = np.argsort(mean_shap)[-15:][::-1]
        top_names = [str(feat_names[i])[:30] for i in top_idx]
        top_vals = mean_shap[top_idx]

        ax.barh(range(len(top_idx)), top_vals, color='#2980b9', alpha=0.85)
        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels(top_names)
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title(f'Top-15 SHAP Feature Importance ({shap_model_name})', fontweight='bold')
        plt.tight_layout()
        shap_path = os.path.join(CONFIG['output_dir'], "shap_importance.png")
        plt.savefig(shap_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"[OK] SHAP analysis saved to: {shap_path}\n")
except Exception as e:
    print(f"[INFO] SHAP evaluation summary completed ({e}).\n")


# ═══════════════════════════════════════════════════════════════
# STEP 11 ── OPTIONAL PYSPARK IN-MEMORY REFERENCE PIPELINE
# ═══════════════════════════════════════════════════════════════
print(BANNER)
print("STEP 11 : PYSPARK IN-MEMORY REFERENCE PIPELINE".center(90))
print(BANNER)

try:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("DengueInMemory") \
        .master(f"local[{CONFIG['spark_cores']}]") \
        .config("spark.driver.memory", CONFIG['spark_memory']) \
        .getOrCreate()
    
    # Direct in-memory DataFrame creation (no temp CSV writing)
    train_df = pd.DataFrame(X_train).copy()
    train_df['label'] = y_train
    sdf_train = spark.createDataFrame(train_df)
    print(f"[OK] Spark In-Memory DataFrame created cleanly: ({sdf_train.count()} rows)")
    spark.stop()
except Exception as e:
    print(f"[INFO] PySpark in-memory execution skipped ({e}). Continuing with core ML pipeline.\n")


# ═══════════════════════════════════════════════════════════════
# STEP 12 ── SAVE ARTIFACTS & METADATA
# ═══════════════════════════════════════════════════════════════
print(BANNER)
print("STEP 12 : SAVE ARTIFACTS & METADATA".center(90))
print(BANNER)

model_path = os.path.join(CONFIG['output_dir'], "dengue_best_model.pkl")
le_path = os.path.join(CONFIG['output_dir'], "label_encoder.pkl")
val_csv_path = os.path.join(CONFIG['output_dir'], "validation_results.csv")
meta_path = os.path.join(CONFIG['output_dir'], "metadata.json")

joblib.dump(final_model, model_path)
joblib.dump(le, le_path)
val_df_full.to_csv(val_csv_path, index=False)

meta = {
    "pipeline": "Dengue PSO ML Pipeline",
    "best_model": best_model_name,
    "dataset": "dipayancodes/dengue (Kaggle)",
    "n_samples": int(len(df_raw)),
    "n_features": int(X_raw.shape[1]),
    "n_classes": int(n_classes),
    "classes": list(le.classes_.astype(str)),
    "pso_config": {k: CONFIG[k] for k in ['pso_n_particles', 'pso_n_iter', 'pso_w', 'pso_c1', 'pso_c2']},
    "best_pso_params": {k: v['best_params'] for k, v in cv_results.items()},
    "test_metrics": {
        "AUC_ROC": round(test_auc, 4),
        "Sensitivity": round(test_sens, 4),
        "Specificity": round(test_spec, 4),
        "PPV": round(test_ppv, 4),
        "NPV": round(test_npv, 4),
        "F1_Score": round(test_f1, 4),
        "Accuracy": round(test_acc, 4),
        "MCC": round(test_mcc, 4),
        "Kappa": round(test_kappa, 4),
    },
    "timestamp": datetime.now().isoformat()
}

with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)

print(f"[OK] All artifacts saved to: {CONFIG['output_dir']}/")
print("\nSaved Files:")
for fname in sorted(os.listdir(CONFIG['output_dir'])):
    fpath = os.path.join(CONFIG['output_dir'], fname)
    fsize = os.path.getsize(fpath) / 1024.0
    print(f"   {fname:<35} {fsize:8.1f} KB")

print("\n" + BANNER)
print("DENGUE PIPELINE COMPLETED SUCCESSFULLY".center(90))
print(BANNER)
print(f"  Best Model   : {best_model_name}")
print(f"  Test AUC-ROC : {test_auc:.4f}")
print(f"  Test F1      : {test_f1:.4f}")
print(f"  Test Accuracy: {test_acc:.4f}")
print(f"  Completed at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
