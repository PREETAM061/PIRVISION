# PIRvision Smart Occupancy & Sustainability System
# COMPLETE PIPELINE v3 — Hand this entire file to Cursor
# Includes: ML Pipeline + Real-World Features + Sustainability Features + Real-Time Live Updates

=======================================================================
## PROJECT OVERVIEW
=======================================================================
- Dataset  : PIRvision FoG Presence Detection (UCI ID 1101)
- Task     : Multi-class occupancy classification + Energy + Sustainability
- Classes  : 0 = Vacancy | 1 = Stationary | 3 = Motion
- Features : 55 PIR sensor readings + Temperature (40+ engineered features)
- Goal     : Train AI → detect occupancy → optimize energy → track carbon → ESG reporting

=======================================================================
## COMPLETE FOLDER STRUCTURE
=======================================================================

```
PIRvision_Project/
├── notebooks/
│   └── train.ipynb                   # Google Colab training notebook
├── models/                           # saved after training
│   ├── best_xgb_model.pkl
│   ├── ensemble_model.pkl
│   ├── feature_scaler.pkl
│   ├── feature_names.json
│   ├── label_mapping.json
│   └── results_summary.json
├── assets/
│   ├── shap_bar.png                  # saved during training
│   ├── shap_beeswarm.png
│   └── shap_waterfall.png
├── app/
│   ├── app.py                        # Streamlit main entry + home page
│   ├── pages/
│   │   ├── 1_Live_Prediction.py      # Real-time occupancy prediction
│   │   ├── 2_Room_Heatmap.py         # Floor plan room heatmap
│   │   ├── 3_Occupancy_Timeline.py   # Hourly pattern analysis
│   │   ├── 4_ROI_Dashboard.py        # Before vs After savings
│   │   ├── 5_Carbon_Tracker.py       # Real-time CO2 tracker
│   │   ├── 6_Green_Scorecard.py      # LEED green building score
│   │   ├── 7_CO2_Visualizer.py       # CO2 impact in human terms
│   │   ├── 8_Sustainability_Goals.py # Annual targets + progress
│   │   ├── 9_ESG_Dashboard.py        # ESG score reporting
│   │   ├── 10_Model_Leaderboard.py   # Model comparison
│   │   ├── 11_SHAP_Explainability.py # AI explainability
│   │   └── 12_Report_Generator.py    # PDF report download
│   └── utils/
│       ├── features.py               # feature engineering
│       ├── predict.py                # prediction logic
│       ├── energy.py                 # energy optimizer
│       ├── carbon.py                 # carbon calculations
│       ├── sustainability.py         # ESG + LEED scoring
│       └── simulation.py            # room + timeline simulation
├── requirements.txt
└── README.md
```

=======================================================================
## PART A — ML TRAINING NOTEBOOK (Google Colab)
=======================================================================

-----------------------------------------------------------------------
### CELL 1 — Install & Import
-----------------------------------------------------------------------

```python
!pip install xgboost shap imbalanced-learn lightgbm optuna --quiet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, zipfile, urllib.request, time, json, joblib
warnings.filterwarnings('ignore')

from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report, confusion_matrix)
import optuna
import shap
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("All libraries loaded")
```

-----------------------------------------------------------------------
### CELL 2 — Download Dataset
-----------------------------------------------------------------------

```python
url         = "https://archive.ics.uci.edu/static/public/1101/pirvision_fog_presence_detection.zip"
zip_path    = "pirvision.zip"
extract_dir = "pirvision_data"

urllib.request.urlretrieve(url, zip_path)
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_dir)

csv_files = []
for root, dirs, files in os.walk(extract_dir):
    for f in files:
        if f.endswith('.csv'):
            csv_files.append(os.path.join(root, f))

dfs = []
for path in sorted(csv_files):
    tmp = pd.read_csv(path)
    print(f"Loaded {os.path.basename(path)}: {tmp.shape}")
    dfs.append(tmp)

df = pd.concat(dfs, ignore_index=True)
print(f"Combined shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
```

-----------------------------------------------------------------------
### CELL 3 — Standardize Columns
-----------------------------------------------------------------------

```python
# Auto-detect label column
label_col = None
for c in df.columns:
    if c.lower() in ['label', 'class', 'activity', 'target', 'y']:
        label_col = c
        break
if label_col is None:
    label_col = df.columns[2]

df = df.rename(columns={label_col: 'Label'})

temp_cols = [c for c in df.columns if 'temp' in c.lower()]
if temp_cols:
    df = df.rename(columns={temp_cols[0]: 'Temperature_F'})

pir_cols = [c for c in df.columns if 'pir' in c.lower()]

print(f"Label column : Label")
print(f"PIR columns  : {len(pir_cols)}")
print(f"Temperature  : {'Found' if temp_cols else 'Not found'}")
print(f"Total records: {len(df):,}")
print(f"Classes      : {sorted(df['Label'].unique())}")
print(df['Label'].value_counts().sort_index())
```

-----------------------------------------------------------------------
### CELL 4 — Feature Engineering (40+ features)
-----------------------------------------------------------------------

```python
def extract_features(df, pir_cols):
    feats = pd.DataFrame(index=df.index)
    pir   = df[pir_cols].values.astype(float)

    # --- Statistical (12) ---
    feats['pir_mean']   = np.mean(pir, axis=1)
    feats['pir_std']    = np.std(pir, axis=1)
    feats['pir_var']    = np.var(pir, axis=1)
    feats['pir_min']    = np.min(pir, axis=1)
    feats['pir_max']    = np.max(pir, axis=1)
    feats['pir_range']  = feats['pir_max'] - feats['pir_min']
    feats['pir_median'] = np.median(pir, axis=1)
    feats['pir_q25']    = np.percentile(pir, 25, axis=1)
    feats['pir_q75']    = np.percentile(pir, 75, axis=1)
    feats['pir_iqr']    = feats['pir_q75'] - feats['pir_q25']
    feats['pir_skew']   = pd.DataFrame(pir).skew(axis=1).values
    feats['pir_kurt']   = pd.DataFrame(pir).kurt(axis=1).values

    # --- Energy & Power (3) ---
    feats['pir_energy'] = np.sum(pir ** 2, axis=1)
    feats['pir_rms']    = np.sqrt(np.mean(pir ** 2, axis=1))
    feats['pir_power']  = feats['pir_energy'] / pir.shape[1]

    # --- Zero Crossing Rate (1) ---
    centered = pir - np.mean(pir, axis=1, keepdims=True)
    feats['pir_zcr'] = np.mean(np.diff(np.sign(centered), axis=1) != 0, axis=1)

    # --- Peak Features (2) ---
    peak_counts, peak_heights = [], []
    for row in pir:
        peaks, props = find_peaks(row, height=0)
        peak_counts.append(len(peaks))
        peak_heights.append(np.mean(props['peak_heights']) if len(peaks) > 0 else 0)
    feats['pir_peak_count']  = peak_counts
    feats['pir_peak_height'] = peak_heights

    # --- Rate of Change (3) ---
    diff = np.diff(pir, axis=1)
    feats['pir_diff_mean'] = np.mean(np.abs(diff), axis=1)
    feats['pir_diff_std']  = np.std(diff, axis=1)
    feats['pir_diff_max']  = np.max(np.abs(diff), axis=1)

    # --- Segment Features 5x3=15 ---
    seg_sz = pir.shape[1] // 5
    for s in range(5):
        seg = pir[:, s*seg_sz:(s+1)*seg_sz]
        feats[f'seg{s+1}_mean'] = np.mean(seg, axis=1)
        feats[f'seg{s+1}_std']  = np.std(seg, axis=1)
        feats[f'seg{s+1}_max']  = np.max(seg, axis=1)

    # --- Temperature (2) ---
    if 'Temperature_F' in df.columns:
        feats['temperature_F'] = df['Temperature_F'].values
        feats['temperature_C'] = (df['Temperature_F'].values - 32) * 5 / 9

    print(f"Features extracted: {feats.shape[1]} features, {feats.shape[0]} samples")
    return feats


X_features = extract_features(df, pir_cols)
y = df['Label'].values
```

-----------------------------------------------------------------------
### CELL 5 — Preprocessing
-----------------------------------------------------------------------

```python
# Encode labels dynamically
unique_labels = sorted(df['Label'].unique())
label_map = {orig: idx for idx, orig in enumerate(unique_labels)}
label_inv = {idx: orig for orig, idx in label_map.items()}
y_encoded = np.array([label_map[lbl] for lbl in y])

class_name_map = {
    label_map.get(0, 0): 'Vacancy',
    label_map.get(1, 1): 'Stationary',
    label_map.get(3, 2): 'Motion'
}
class_names = [class_name_map.get(i, f'Class {i}') for i in range(len(unique_labels))]

# Split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_features, y_encoded,
    test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled  = scaler.transform(X_test_raw)

# SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

print(f"Before SMOTE : {np.bincount(y_train)}")
print(f"After SMOTE  : {np.bincount(y_train_bal)}")
print(f"Train size   : {X_train_bal.shape[0]:,}")
print(f"Test size    : {X_test_scaled.shape[0]:,}")
```

-----------------------------------------------------------------------
### CELL 6 — Train All 5 Models
-----------------------------------------------------------------------

```python
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, C=1.0, random_state=42, n_jobs=-1
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=42, n_jobs=-1, class_weight='balanced'
    ),
    'XGBoost': XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', random_state=42, n_jobs=-1
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        num_leaves=31, random_state=42, n_jobs=-1, verbose=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    )
}

results = {}
print(f"{'Model':<22} {'Accuracy':>10} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Time':>7}")
print("=" * 72)

for name, model in models.items():
    t0 = time.time()
    model.fit(X_train_bal, y_train_bal)
    y_pred  = model.predict(X_test_scaled)
    elapsed = time.time() - t0

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    results[name] = dict(
        model=model, y_pred=y_pred,
        accuracy=acc, f1=f1, precision=prec, recall=rec, time=elapsed
    )
    print(f"{name:<22} {acc*100:>9.2f}% {f1*100:>7.2f}% {prec*100:>9.2f}% {rec*100:>7.2f}% {elapsed:>5.1f}s")

best_name  = max(results, key=lambda k: results[k]['f1'])
best_model = results[best_name]['model']
print(f"\nBest model: {best_name}  F1={results[best_name]['f1']*100:.2f}%")
```

-----------------------------------------------------------------------
### CELL 7 — Optuna Hyperparameter Tuning
-----------------------------------------------------------------------

```python
def objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
        'max_depth':        trial.suggest_int('max_depth', 3, 12),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma':            trial.suggest_float('gamma', 0, 5),
        'reg_alpha':        trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda':       trial.suggest_float('reg_lambda', 0, 2),
        'eval_metric': 'mlogloss', 'random_state': 42, 'n_jobs': -1
    }
    mdl = XGBClassifier(**params)
    return cross_val_score(
        mdl, X_train_bal, y_train_bal,
        cv=3, scoring='f1_weighted', n_jobs=-1
    ).mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

tuned_xgb = XGBClassifier(
    **study.best_params,
    eval_metric='mlogloss', random_state=42, n_jobs=-1
)
tuned_xgb.fit(X_train_bal, y_train_bal)
y_pred_tuned = tuned_xgb.predict(X_test_scaled)
tuned_f1     = f1_score(y_test, y_pred_tuned, average='weighted')
tuned_acc    = accuracy_score(y_test, y_pred_tuned)

print(f"Tuned XGBoost — Accuracy: {tuned_acc*100:.2f}%  F1: {tuned_f1*100:.2f}%")
```

-----------------------------------------------------------------------
### CELL 8 — Ensemble Model
-----------------------------------------------------------------------

```python
ensemble = VotingClassifier(
    estimators=[
        ('xgb',  XGBClassifier(**study.best_params, eval_metric='mlogloss',
                               random_state=42, n_jobs=-1)),
        ('lgbm', LGBMClassifier(n_estimators=300, max_depth=8,
                                learning_rate=0.1, random_state=42,
                                verbose=-1, n_jobs=-1)),
        ('rf',   RandomForestClassifier(n_estimators=200, max_depth=15,
                                        random_state=42, n_jobs=-1))
    ],
    voting='soft', n_jobs=-1
)
ensemble.fit(X_train_bal, y_train_bal)
y_pred_ens   = ensemble.predict(X_test_scaled)
ensemble_f1  = f1_score(y_test, y_pred_ens, average='weighted')
ensemble_acc = accuracy_score(y_test, y_pred_ens)

print(f"Ensemble — Accuracy: {ensemble_acc*100:.2f}%  F1: {ensemble_f1*100:.2f}%")
print(classification_report(y_test, y_pred_ens, target_names=class_names, zero_division=0))
```

-----------------------------------------------------------------------
### CELL 9 — SHAP Explainability + Save Plots
-----------------------------------------------------------------------

```python
os.makedirs('assets', exist_ok=True)

explainer   = shap.TreeExplainer(tuned_xgb)
sample_sz   = min(500, len(X_test_scaled))
X_shap      = pd.DataFrame(X_test_scaled[:sample_sz], columns=X_features.columns)
shap_values = explainer.shap_values(X_shap)
shap_arr    = np.array(shap_values)

if shap_arr.ndim == 3:
    shap_matrix  = shap_arr[2]
    expected_val = explainer.expected_value[2]
elif isinstance(shap_values, list):
    shap_matrix  = shap_values[2]
    expected_val = explainer.expected_value[2]
else:
    shap_matrix  = shap_values
    expected_val = explainer.expected_value

# Bar plot
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_matrix, X_shap, plot_type='bar', show=False, max_display=20)
plt.title('SHAP Feature Importance — Top 20', fontweight='bold')
plt.tight_layout()
plt.savefig('assets/shap_bar.png', dpi=150, bbox_inches='tight')
plt.show()

# Beeswarm
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_matrix, X_shap, plot_type='dot', show=False, max_display=20)
plt.title('SHAP Beeswarm', fontweight='bold')
plt.tight_layout()
plt.savefig('assets/shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()

# Waterfall
sv  = shap_matrix[0]
fv  = X_shap.iloc[0].values
fn  = list(X_shap.columns)
exp = shap.Explanation(values=sv, base_values=expected_val, data=fv, feature_names=fn)
plt.figure(figsize=(10, 6))
shap.plots.waterfall(exp, show=False)
plt.title('SHAP Waterfall — Single Prediction', fontweight='bold')
plt.tight_layout()
plt.savefig('assets/shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.show()

print("SHAP plots saved to assets/")
```

-----------------------------------------------------------------------
### CELL 10 — Energy Optimizer
-----------------------------------------------------------------------

```python
DEVICE_PROFILES = {
    'HVAC':        {'Vacancy': 0,   'Stationary': 800,  'Motion': 1200, 'Baseline': 1200},
    'Lighting':    {'Vacancy': 0,   'Stationary': 100,  'Motion': 200,  'Baseline': 200},
    'Ventilation': {'Vacancy': 50,  'Stationary': 200,  'Motion': 400,  'Baseline': 400},
    'Screens':     {'Vacancy': 0,   'Stationary': 150,  'Motion': 300,  'Baseline': 300},
    'Elevators':   {'Vacancy': 100, 'Stationary': 100,  'Motion': 500,  'Baseline': 500},
}

ACTIONS = {
    'Vacancy':    {'HVAC': 'OFF',     'Lighting': 'OFF',     'Ventilation': '15%',  'Screens': 'Sleep',  'Elevators': 'Standby'},
    'Stationary': {'HVAC': 'Eco 67%', 'Lighting': 'Dim 50%', 'Ventilation': '50%',  'Screens': 'Low',    'Elevators': 'Standby'},
    'Motion':     {'HVAC': 'Full',    'Lighting': 'Full',     'Ventilation': 'Full', 'Screens': 'Active', 'Elevators': 'Active'},
}

STATE_DECODE = {
    label_map.get(0, 0): 'Vacancy',
    label_map.get(1, 1): 'Stationary',
    label_map.get(3, 2): 'Motion'
}

preds    = tuned_xgb.predict(X_test_scaled)
rows     = []
for pred in preds:
    state     = STATE_DECODE.get(pred, 'Motion')
    baseline  = sum(p['Baseline'] for p in DEVICE_PROFILES.values())
    optimized = sum(p[state]      for p in DEVICE_PROFILES.values())
    savings   = baseline - optimized
    rows.append({
        'state': state, 'baseline_W': baseline,
        'optimized_W': optimized, 'savings_W': savings,
        'savings_pct': savings / baseline * 100
    })

energy_df = pd.DataFrame(rows)

# Calculate report
hours_per_day = 10
price_kwh     = 0.12
co2_per_kwh   = 0.233   # kg CO2 per kWh (US average grid)

bl  = (energy_df['baseline_W'].mean()  / 1000) * hours_per_day
opt = (energy_df['optimized_W'].mean() / 1000) * hours_per_day
sav = bl - opt

sav_report = {
    'daily_baseline_kWh':  round(bl, 3),
    'daily_optimized_kWh': round(opt, 3),
    'daily_savings_kWh':   round(sav, 3),
    'daily_savings_USD':   round(sav * price_kwh, 3),
    'annual_savings_USD':  round(sav * price_kwh * 365, 2),
    'annual_CO2_saved_kg': round(sav * co2_per_kwh * 365, 2),
    'avg_savings_pct':     round(energy_df['savings_pct'].mean(), 2)
}

print("ENERGY SAVINGS REPORT")
print("=" * 40)
for k, v in sav_report.items():
    print(f"  {k:<28}: {v}")
```

-----------------------------------------------------------------------
### CELL 11 — Save All Artifacts
-----------------------------------------------------------------------

```python
os.makedirs('models', exist_ok=True)

joblib.dump(tuned_xgb, 'models/best_xgb_model.pkl')
joblib.dump(ensemble,  'models/ensemble_model.pkl')
joblib.dump(scaler,    'models/feature_scaler.pkl')

with open('models/feature_names.json', 'w') as f:
    json.dump(list(X_features.columns), f)

with open('models/label_mapping.json', 'w') as f:
    json.dump({
        'encode': {str(k): int(v) for k, v in label_map.items()},
        'decode': {str(k): int(v) for k, v in label_inv.items()}
    }, f, indent=2)

summary = {
    'best_baseline_model': best_name,
    'best_baseline_f1':    float(results[best_name]['f1']),
    'tuned_xgb_f1':        float(tuned_f1),
    'tuned_xgb_accuracy':  float(tuned_acc),
    'ensemble_f1':         float(ensemble_f1),
    'ensemble_accuracy':   float(ensemble_acc),
    'all_models': {
        name: {
            'accuracy':  float(res['accuracy']),
            'f1':        float(res['f1']),
            'precision': float(res['precision']),
            'recall':    float(res['recall']),
            'time':      float(res['time'])
        }
        for name, res in results.items()
    },
    'energy_savings': {k: float(v) for k, v in sav_report.items()}
}

with open('models/results_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("All artifacts saved:")
for fname in os.listdir('models'):
    size = os.path.getsize(f'models/{fname}') / 1024
    print(f"  {fname:<35} {size:.1f} KB")
```

=======================================================================
## PART B — UTILITY FILES
=======================================================================

-----------------------------------------------------------------------
### File: app/utils/features.py
-----------------------------------------------------------------------

```python
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def make_features_from_array(pir: np.ndarray, temperature: float,
                              feature_names: list) -> pd.DataFrame:
    """Convert single PIR array + temperature into training feature vector."""
    f = {}
    f['pir_mean']        = np.mean(pir)
    f['pir_std']         = np.std(pir)
    f['pir_var']         = np.var(pir)
    f['pir_min']         = np.min(pir)
    f['pir_max']         = np.max(pir)
    f['pir_range']       = f['pir_max'] - f['pir_min']
    f['pir_median']      = np.median(pir)
    f['pir_q25']         = np.percentile(pir, 25)
    f['pir_q75']         = np.percentile(pir, 75)
    f['pir_iqr']         = f['pir_q75'] - f['pir_q25']
    f['pir_skew']        = float(pd.Series(pir).skew())
    f['pir_kurt']        = float(pd.Series(pir).kurt())
    f['pir_energy']      = float(np.sum(pir ** 2))
    f['pir_rms']         = float(np.sqrt(np.mean(pir ** 2)))
    f['pir_power']       = f['pir_energy'] / len(pir)
    f['pir_zcr']         = float(np.mean(np.diff(np.sign(pir - np.mean(pir))) != 0))
    peaks, props         = find_peaks(pir, height=0)
    f['pir_peak_count']  = len(peaks)
    f['pir_peak_height'] = float(np.mean(props['peak_heights'])) if len(peaks) > 0 else 0.0
    d = np.diff(pir)
    f['pir_diff_mean']   = float(np.mean(np.abs(d)))
    f['pir_diff_std']    = float(np.std(d))
    f['pir_diff_max']    = float(np.max(np.abs(d)))
    sz = len(pir) // 5
    for s in range(5):
        seg = pir[s*sz:(s+1)*sz]
        f[f'seg{s+1}_mean'] = float(np.mean(seg))
        f[f'seg{s+1}_std']  = float(np.std(seg))
        f[f'seg{s+1}_max']  = float(np.max(seg))
    f['temperature_F'] = float(temperature)
    f['temperature_C'] = (temperature - 32) * 5 / 9
    return pd.DataFrame([f]).reindex(columns=feature_names, fill_value=0)


def simulate_pir(sim_class: str, noise: int, n: int = 55) -> np.ndarray:
    """Simulate PIR readings calibrated to match real dataset distributions."""
    np.random.seed(None)
    if sim_class == 'Vacancy':
        pir = np.ones(n) * 2 + np.random.normal(0, max(noise * 0.05, 0.1), n)
        return np.clip(pir, 0, 10)
    elif sim_class == 'Stationary':
        pir = np.ones(n) * 40 + np.random.normal(0, max(noise * 0.3, 2), n)
        return np.clip(pir, 20, 80)
    elif sim_class == 'Motion':
        pir = np.ones(n) * 180 + np.random.normal(0, max(noise * 1.5, 30), n)
        return np.clip(np.abs(pir), 80, 300)
    else:
        return simulate_pir(np.random.choice(['Vacancy', 'Stationary', 'Motion']), noise, n)
```

-----------------------------------------------------------------------
### File: app/utils/energy.py
-----------------------------------------------------------------------

```python
DEVICE_PROFILES = {
    'HVAC':        {'Vacancy': 0,   'Stationary': 800,  'Motion': 1200, 'Baseline': 1200},
    'Lighting':    {'Vacancy': 0,   'Stationary': 100,  'Motion': 200,  'Baseline': 200},
    'Ventilation': {'Vacancy': 50,  'Stationary': 200,  'Motion': 400,  'Baseline': 400},
    'Screens':     {'Vacancy': 0,   'Stationary': 150,  'Motion': 300,  'Baseline': 300},
    'Elevators':   {'Vacancy': 100, 'Stationary': 100,  'Motion': 500,  'Baseline': 500},
}

ACTIONS = {
    'Vacancy':    {'HVAC': 'OFF',     'Lighting': 'OFF',     'Ventilation': '15%',  'Screens': 'Sleep',  'Elevators': 'Standby'},
    'Stationary': {'HVAC': 'Eco 67%', 'Lighting': 'Dim 50%', 'Ventilation': '50%',  'Screens': 'Low',    'Elevators': 'Standby'},
    'Motion':     {'HVAC': 'Full',    'Lighting': 'Full',     'Ventilation': 'Full', 'Screens': 'Active', 'Elevators': 'Active'},
}

ICONS = {
    'HVAC': '🌡️', 'Lighting': '💡',
    'Ventilation': '💨', 'Screens': '🖥️', 'Elevators': '🛗'
}

COMFORT_TEMP = {
    'Vacancy': 28, 'Stationary': 23, 'Motion': 21
}


def calculate_savings(state: str, rooms: int = 1,
                      hours: float = 10, price_kwh: float = 0.12) -> dict:
    baseline  = sum(p['Baseline'] for p in DEVICE_PROFILES.values())
    optimized = sum(p[state]      for p in DEVICE_PROFILES.values())
    savings_w = (baseline - optimized) * rooms
    kwh_day   = (savings_w / 1000) * hours
    return {
        'baseline_W':       baseline * rooms,
        'optimized_W':      optimized * rooms,
        'savings_W':        savings_w,
        'savings_pct':      round((savings_w / (baseline * rooms)) * 100, 1),
        'savings_kwh_day':  round(kwh_day, 3),
        'savings_usd_day':  round(kwh_day * price_kwh, 2),
        'savings_usd_year': round(kwh_day * price_kwh * 365, 2),
        'co2_year_kg':      round(kwh_day * 0.233 * 365, 1),
        'actions':          ACTIONS[state],
        'comfort_temp_C':   COMFORT_TEMP[state],
        'device_breakdown': {dev: DEVICE_PROFILES[dev][state] for dev in DEVICE_PROFILES}
    }
```

-----------------------------------------------------------------------
### File: app/utils/carbon.py
-----------------------------------------------------------------------

```python
# Carbon intensity factors
CO2_PER_KWH    = 0.233    # kg CO2 per kWh (US average)
CO2_PER_TREE   = 21.77    # kg CO2 absorbed per tree per year
CO2_PER_KM_CAR = 0.250    # kg CO2 per km driven (average car)
CO2_PER_FLIGHT_LON_NYC = 790  # kg CO2 per round trip London-NYC
CO2_HOME_MONTHLY = 567    # kg CO2 per month for average home


def kwh_to_co2(kwh: float) -> float:
    """Convert kWh to kg CO2."""
    return round(kwh * CO2_PER_KWH, 3)


def co2_to_equivalents(co2_kg: float) -> dict:
    """
    Convert CO2 saved into human-understandable equivalents.
    Used by CO2 Impact Visualizer page.
    """
    return {
        'trees_planted':      round(co2_kg / CO2_PER_TREE, 1),
        'car_km_avoided':     round(co2_kg / CO2_PER_KM_CAR, 0),
        'flights_avoided':    round(co2_kg / CO2_PER_FLIGHT_LON_NYC, 2),
        'home_months_saved':  round(co2_kg / CO2_HOME_MONTHLY, 2),
        'lightbulb_hours':    round(co2_kg / (0.01 * CO2_PER_KWH), 0),  # 10W bulb
        'co2_kg':             round(co2_kg, 1)
    }


def get_carbon_status(co2_rate_kg_hr: float,
                      target_kg_hr: float = 0.5) -> dict:
    """
    Determine carbon status color and label vs target.
    Used by real-time Carbon Tracker.
    """
    pct = (co2_rate_kg_hr / target_kg_hr) * 100 if target_kg_hr > 0 else 100
    if pct <= 70:
        return {'status': 'Below Target', 'color': '#2ecc71', 'emoji': '🟢', 'pct': round(pct, 1)}
    elif pct <= 100:
        return {'status': 'Near Target',  'color': '#f39c12', 'emoji': '🟡', 'pct': round(pct, 1)}
    else:
        return {'status': 'Over Budget',  'color': '#e74c3c', 'emoji': '🔴', 'pct': round(pct, 1)}


def project_co2_savings(daily_savings_kwh: float, years: int = 10) -> list:
    """
    Project cumulative CO2 savings over N years.
    Returns list of (year, cumulative_co2_kg) tuples.
    """
    annual = daily_savings_kwh * 365 * CO2_PER_KWH
    return [{'year': y, 'cumulative_co2_kg': round(annual * y, 1)} for y in range(1, years + 1)]
```

-----------------------------------------------------------------------
### File: app/utils/sustainability.py
-----------------------------------------------------------------------

```python
# LEED Certification thresholds
LEED_THRESHOLDS = {
    'Platinum': 90,
    'Gold':     75,
    'Silver':   60,
    'Certified':40,
}

# ESG scoring weights
ESG_WEIGHTS = {
    'Environmental': 0.50,
    'Social':        0.30,
    'Governance':    0.20,
}


def calculate_leed_score(occupancy_accuracy: float,
                         energy_waste_pct: float,
                         avg_savings_pct: float,
                         uptime_pct: float = 99.0) -> dict:
    """
    Calculate a LEED-style Green Building Score 0-100.

    Parameters:
        occupancy_accuracy : model accuracy % (0-100)
        energy_waste_pct   : % of energy wasted (lower is better)
        avg_savings_pct    : average energy savings %
        uptime_pct         : system uptime %

    Returns:
        dict with score, rating, breakdown, next_level info
    """
    # Score components
    accuracy_score  = min(occupancy_accuracy, 100) * 0.35
    waste_score     = max(0, (100 - energy_waste_pct)) * 0.30
    savings_score   = min(avg_savings_pct, 100) * 0.25
    uptime_score    = min(uptime_pct, 100) * 0.10
    total           = accuracy_score + waste_score + savings_score + uptime_score

    # Rating
    rating = 'Not Certified'
    for level, threshold in LEED_THRESHOLDS.items():
        if total >= threshold:
            rating = f'LEED {level}'
            break

    # Next level
    next_level = None
    for level, threshold in LEED_THRESHOLDS.items():
        if total < threshold:
            next_level = {'level': f'LEED {level}', 'points_needed': round(threshold - total, 1)}
            break

    return {
        'total_score':      round(total, 1),
        'rating':           rating,
        'breakdown': {
            'Occupancy Accuracy (35%)': round(accuracy_score, 1),
            'Energy Efficiency (30%)':  round(waste_score, 1),
            'Savings Achievement (25%)':round(savings_score, 1),
            'System Uptime (10%)':      round(uptime_score, 1),
        },
        'next_level': next_level
    }


def calculate_esg_score(energy_efficiency: float,
                        carbon_reduction: float,
                        comfort_score: float,
                        compliance_pct: float) -> dict:
    """
    Calculate ESG (Environmental, Social, Governance) scores.

    Parameters:
        energy_efficiency : % energy saved vs baseline
        carbon_reduction  : % CO2 reduced vs baseline
        comfort_score     : occupant comfort 0-100
        compliance_pct    : % time system operated within policy

    Returns:
        dict with E, S, G scores and overall grade
    """
    E_score = (energy_efficiency * 0.5 + carbon_reduction * 0.5)
    S_score = comfort_score
    G_score = compliance_pct

    overall = (E_score * ESG_WEIGHTS['Environmental'] +
               S_score * ESG_WEIGHTS['Social'] +
               G_score * ESG_WEIGHTS['Governance'])

    def to_grade(score):
        if score >= 90: return 'A+'
        elif score >= 80: return 'A'
        elif score >= 70: return 'B+'
        elif score >= 60: return 'B'
        elif score >= 50: return 'C'
        else: return 'D'

    return {
        'Environmental': {'score': round(E_score, 1), 'grade': to_grade(E_score),
                          'components': {'Energy Efficiency': energy_efficiency,
                                         'Carbon Reduction': carbon_reduction}},
        'Social':        {'score': round(S_score, 1), 'grade': to_grade(S_score),
                          'components': {'Occupant Comfort': comfort_score}},
        'Governance':    {'score': round(G_score, 1), 'grade': to_grade(G_score),
                          'components': {'Policy Compliance': compliance_pct}},
        'Overall':       {'score': round(overall, 1), 'grade': to_grade(overall)}
    }


def calculate_goals_progress(current_savings_pct: float,
                              current_co2_kg: float,
                              current_accuracy: float,
                              targets: dict = None) -> list:
    """
    Calculate progress toward sustainability goals.
    Returns list of goal dicts with progress bars.
    """
    if targets is None:
        targets = {
            'Energy Reduction':    {'target': 30.0,  'unit': '%',  'current': current_savings_pct},
            'CO2 Saved':           {'target': 500.0, 'unit': 'kg', 'current': current_co2_kg},
            'Prediction Accuracy': {'target': 95.0,  'unit': '%',  'current': current_accuracy},
        }

    goals = []
    for name, data in targets.items():
        pct_done = min((data['current'] / data['target']) * 100, 100)
        goals.append({
            'name':      name,
            'current':   data['current'],
            'target':    data['target'],
            'unit':      data['unit'],
            'pct_done':  round(pct_done, 1),
            'achieved':  pct_done >= 100,
            'status':    'Achieved' if pct_done >= 100 else f"{pct_done:.0f}% complete"
        })
    return goals
```

-----------------------------------------------------------------------
### File: app/utils/simulation.py
-----------------------------------------------------------------------

```python
import numpy as np
import pandas as pd

# Room names for floor plan heatmap
ROOM_NAMES = [
    'Lobby',     'Reception', 'Meeting A', 'Meeting B', 'Meeting C',
    'Office 1',  'Office 2',  'Office 3',  'Office 4',  'Office 5',
    'Kitchen',   'Server Rm', 'HR Dept',   'Finance',   'Marketing',
    'Exec Suite','Board Rm',  'IT Dept',   'Warehouse', 'Parking'
]

STATES      = ['Vacancy', 'Stationary', 'Motion']
STATE_COLOR = {'Vacancy': '#2ecc71', 'Stationary': '#3498db', 'Motion': '#e74c3c'}
STATE_CODE  = {'Vacancy': 0,          'Stationary': 1,         'Motion': 2}


def simulate_building(n_rooms: int = 20,
                      time_of_day: int = 9,
                      day_of_week: str = 'Monday') -> pd.DataFrame:
    """
    Simulate occupancy states for all rooms in the building.
    Patterns vary by time of day and day of week.
    Used by Room Heatmap page.
    """
    np.random.seed(int(time_of_day * 7))

    is_weekend  = day_of_week in ['Saturday', 'Sunday']
    is_business = 8 <= time_of_day <= 18 and not is_weekend

    rows = []
    for i in range(min(n_rooms, len(ROOM_NAMES))):
        name = ROOM_NAMES[i]

        # Lobby and reception always busier during business hours
        if 'Lobby' in name or 'Reception' in name:
            probs = [0.1, 0.3, 0.6] if is_business else [0.8, 0.1, 0.1]
        elif 'Server' in name or 'Parking' in name:
            probs = [0.3, 0.2, 0.5] if is_business else [0.5, 0.3, 0.2]
        elif 'Meeting' in name or 'Board' in name:
            probs = [0.3, 0.2, 0.5] if is_business else [0.9, 0.1, 0.0]
        elif not is_business:
            probs = [0.85, 0.1, 0.05]
        else:
            probs = [0.3, 0.4, 0.3]

        state = np.random.choice(STATES, p=probs)
        rows.append({
            'room_id':    i,
            'room_name':  name,
            'state':      state,
            'state_code': STATE_CODE[state],
            'color':      STATE_COLOR[state],
            'energy_W':   {'Vacancy': 150, 'Stationary': 1350, 'Motion': 2600}[state]
        })

    return pd.DataFrame(rows)


def simulate_weekly_timeline(n_hours: int = 24) -> pd.DataFrame:
    """
    Simulate hourly occupancy counts across a full week.
    Used by Occupancy Timeline page.
    """
    days  = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    rows  = []
    for day in days:
        is_weekend = day in ['Saturday', 'Sunday']
        for hour in range(n_hours):
            if is_weekend:
                base = max(0, int(np.random.normal(2, 1)))
            elif 9 <= hour <= 18:
                base = max(0, int(np.random.normal(15, 4)))
            elif 7 <= hour <= 9 or 18 <= hour <= 20:
                base = max(0, int(np.random.normal(8, 3)))
            else:
                base = max(0, int(np.random.normal(1, 1)))

            rows.append({
                'day':        day,
                'hour':       hour,
                'occupied':   base,
                'stationary': max(0, int(base * 0.4 + np.random.normal(0, 1))),
                'motion':     max(0, int(base * 0.6 + np.random.normal(0, 1))),
                'vacancy':    max(0, 20 - base)
            })

    return pd.DataFrame(rows)
```

=======================================================================
## PART C — STREAMLIT PAGES (What Cursor must build)
=======================================================================

-----------------------------------------------------------------------
### File: app/app.py — Home Page
-----------------------------------------------------------------------

```
PURPOSE: Landing page shown when app starts

CONTENT:
- Title: PIRvision Smart Occupancy & Sustainability System
- Subtitle: AI-powered building intelligence for a greener future
- Top KPI bar (5 metrics):
    Best Model | Test F1 | Annual Savings $ | CO2 Saved kg | Green Score
- 4 info cards:
    What this system does (plain English)
    How the AI works
    Real-world impact
    Sustainability commitment
- Navigation guide: list all 12 pages with one-line description each
- Load models here with st.cache_resource and pass via session_state

LOAD ON STARTUP:
    model, scaler, feature_names, label_mapping, summary
    from app/utils/predict.py → load_artifacts()
```

-----------------------------------------------------------------------
### Page 1: 1_Live_Prediction.py
-----------------------------------------------------------------------

```
PURPOSE: Real-time occupancy detection simulation

SIDEBAR:
- Selectbox: Simulate Class [Vacancy, Stationary, Motion, Random]
- Slider: Noise Level 0-50
- Slider: Temperature (F) 60-90
- Button: Run Prediction (primary)

ON RUN:
1. Call simulate_pir(sim_class, noise) → pir array
2. Call make_features_from_array(pir, temperature, feature_names)
3. Scale with scaler.transform()
4. Call model.predict() and model.predict_proba()
5. Call calculate_savings(state) from energy.py

DISPLAY:
- Row 1: 3 metrics — Detected State (with color emoji), Confidence %, Energy Saving %
- Plotly line chart: PIR signal (55 points), colored by state, with mean line
- Plotly bar chart: Class probabilities (3 bars)
- Row 2: 5 device action cards — HVAC, Lighting, Ventilation, Screens, Elevators
- Row 3: Comfort temperature recommendation
- Expander: PIR signal stats table (mean, std, min, max, range, energy)

BEFORE RUN:
- Show info box: "Configure sidebar and click Run Prediction"
```

-----------------------------------------------------------------------
### Page 2: 2_Room_Heatmap.py
-----------------------------------------------------------------------

```
PURPOSE: Floor plan showing occupancy state of every room in building

SIDEBAR:
- Slider: Number of rooms 5-20
- Selectbox: Day of week [Monday...Sunday]
- Slider: Hour of day 0-23 (show as 9:00 AM format)
- Button: Refresh Heatmap

ON LOAD/REFRESH:
1. Call simulate_building(n_rooms, time_of_day, day_of_week)
2. Display 4x5 grid of colored room cards

EACH ROOM CARD shows:
- Room name
- State emoji: 🟢 Vacant / 🔵 Stationary / 🔴 Motion
- Energy consumption in watts

SUMMARY below grid:
- Count of Vacant / Stationary / Motion rooms
- Total energy in use right now (W)
- Energy being wasted (rooms on but vacant)
- Estimated hourly cost $

ALERT BOX:
- If more than 30% rooms vacant with devices ON:
  "⚠️ 8 rooms detected vacant — recommend shutting HVAC to save $X/hour"
```

-----------------------------------------------------------------------
### Page 3: 3_Occupancy_Timeline.py
-----------------------------------------------------------------------

```
PURPOSE: Weekly hourly occupancy pattern analysis + AI recommendations

ON LOAD:
1. Call simulate_weekly_timeline() from simulation.py
2. Display interactive heatmap

DISPLAY:
- Plotly heatmap: X=Hours (0-23), Y=Days (Mon-Sun), Z=occupancy count
  Color scale: green (low) to red (high)
- Plotly line chart: Today's hourly pattern vs last week average
- Peak hours table: Top 5 busiest time slots
- Dead hours table: Top 5 emptiest time slots

AI RECOMMENDATIONS section:
- "Based on pattern analysis, the AI recommends:"
  - Schedule HVAC off during hours X-Y (saves $Z/week)
  - Pre-cool building 1 hour before peak at HH:MM
  - Elevator standby during hours X-Y
  - Reduce lighting in Zone B during lunch hours
- Show potential weekly savings from following recommendations
```

-----------------------------------------------------------------------
### Page 4: 4_ROI_Dashboard.py
-----------------------------------------------------------------------

```
PURPOSE: Before vs After ROI proof — show financial value of AI system

SIDEBAR:
- Number input: Building floor area (sqft)
- Slider: Number of rooms 1-100
- Slider: Hours per day 1-24
- Number input: Electricity price per kWh (default 0.12)
- Slider: Months since deployment 1-24

DISPLAY:
- Big banner: "Total Savings Since Deployment: $X,XXX"

Row of 4 KPI cards:
- Daily Savings $
- Monthly Savings $
- Annual Savings $
- Payback Period (months)

Plotly side-by-side bar: Without AI vs With AI per device
- HVAC, Lighting, Ventilation, Screens, Elevators

Plotly line chart: Cumulative savings over 24 months
- Show breakeven point where system pays for itself

Comparison table:
| Metric          | Without AI | With AI  | Improvement |
|-----------------|------------|----------|-------------|
| Daily kWh       | X          | Y        | -Z%         |
| Monthly Cost $  | X          | Y        | -Z%         |
| Annual CO2 kg   | X          | Y        | -Z%         |
| Waste Hours/day | X          | Y        | -Z%         |
```

-----------------------------------------------------------------------
### Page 5: 5_Carbon_Tracker.py
-----------------------------------------------------------------------

```
PURPOSE: Real-time CO2 emission tracking with live counter

SIDEBAR:
- Selectbox: Current Building State [Vacancy, Stationary, Motion, Mixed]
- Slider: Number of active rooms 1-50
- Number input: Carbon target kg/hour (default 0.5)
- Toggle: Show projections

DISPLAY:
- Large live CO2 counter: "XX.XX kg CO2/hour"
  (auto-refreshes every 5 seconds using st.empty() + time.sleep())
- Color coded status: 🟢 Below Target / 🟡 Near Target / 🔴 Over Budget
  (from carbon.py → get_carbon_status())

3 metric cards:
- CO2 this hour (kg)
- CO2 saved today vs baseline (kg)
- Annual projection (tonnes)

Plotly gauge chart: Current CO2 rate vs target

Plotly area chart: CO2 produced vs CO2 saved over simulated 24 hours

Carbon Budget tracker:
- Daily budget: X kg
- Used so far: Y kg
- Remaining: Z kg
- Progress bar colored green/yellow/red
```

-----------------------------------------------------------------------
### Page 6: 6_Green_Scorecard.py
-----------------------------------------------------------------------

```
PURPOSE: LEED-style green building certification scorecard

ON LOAD:
- Load results_summary.json
- Call calculate_leed_score() from sustainability.py
  with: tuned_xgb_accuracy, avg_savings_pct, energy_waste_pct

DISPLAY:
- Big score circle: "XX / 100"
- Current rating badge: 🏆 LEED Platinum / 🥇 Gold / 🥈 Silver / Certified
- "X points away from next level: LEED [level]"

Score breakdown table:
| Component                  | Score | Max |
|----------------------------|-------|-----|
| Occupancy Accuracy (35%)   | XX    | 35  |
| Energy Efficiency (30%)    | XX    | 30  |
| Savings Achievement (25%)  | XX    | 25  |
| System Uptime (10%)        | XX    | 10  |
| TOTAL                      | XX    | 100 |

Plotly radar chart: Score across all 4 components

What LEED means section:
- Plain English explanation of each level
- Industry value: "LEED Platinum buildings sell for 10-20% more"

Recommendations to improve score:
- "To reach next level, improve energy efficiency by X%"
```

-----------------------------------------------------------------------
### Page 7: 7_CO2_Visualizer.py
-----------------------------------------------------------------------

```
PURPOSE: Convert CO2 numbers into human-understandable equivalents

SIDEBAR:
- Slider: Number of rooms 1-100
- Slider: Hours per day 1-24
- Selectbox: Occupancy mix [Mostly Vacant, Mixed, Mostly Active]
- Button: Calculate Impact

ON CALCULATE:
1. Compute daily_savings_kwh based on inputs
2. Compute annual_savings_kwh = daily * 365
3. Call co2_to_equivalents(co2_kg) from carbon.py

DISPLAY:
- Hero number: "Your building saves X,XXX kg CO2 per year"

5 impact cards with large icons:
- 🌳  "= XX trees planted"
- 🚗  "= X,XXX km of driving avoided"
- ✈️  "= X.X London-NYC flights"
- 🏠  "= X.X months of home energy"
- 💡  "= X,XXX hours of light"

Plotly animated bar chart: CO2 saved per year growing over 10 years
Plotly pie chart: Where the savings come from (HVAC vs Lighting vs Ventilation etc.)

Fun facts section:
- "If every office in your city used this system..."
- Scale the numbers by city office count
```

-----------------------------------------------------------------------
### Page 8: 8_Sustainability_Goals.py
-----------------------------------------------------------------------

```
PURPOSE: Track progress toward annual sustainability targets

SIDEBAR:
- Number inputs: Set custom targets
  - Energy reduction target %
  - CO2 reduction target kg
  - Accuracy target %
- Button: Update Goals

ON LOAD:
1. Load results_summary.json for current performance
2. Call calculate_goals_progress() from sustainability.py

DISPLAY:
- Header: "Annual Sustainability Goals — 2025"

For each goal, show:
- Goal name
- Progress bar (colored green if achieved, blue if in progress)
- Current value vs target
- Status badge: ✅ ACHIEVED or "XX% complete"
- Projected completion date

Plotly timeline: When each goal will be achieved at current rate

Monthly trend chart: Progress over simulated 12 months

Goal recommendations:
- "At current rate, Energy Reduction goal will be achieved in X months"
- "CO2 goal needs X% improvement in system usage to hit target"
```

-----------------------------------------------------------------------
### Page 9: 9_ESG_Dashboard.py
-----------------------------------------------------------------------

```
PURPOSE: Full ESG (Environmental, Social, Governance) reporting dashboard

ON LOAD:
1. Load results_summary.json
2. Call calculate_esg_score() from sustainability.py

DISPLAY:
- Header: "ESG Performance Report — FY2025"
- Overall ESG grade: Large letter grade (A+/A/B+/B/C)

3 pillar cards side by side:
┌─────────────────┬─────────────────┬─────────────────┐
│ 🌍 Environmental│ 👥 Social       │ 📋 Governance   │
│ Score: 87 / A   │ Score: 79 / B+  │ Score: 92 / A+  │
│ Energy: XX%     │ Comfort: XX%    │ Compliance: XX% │
│ Carbon: XX%     │                 │                 │
└─────────────────┴─────────────────┴─────────────────┘

Plotly radar chart: E vs S vs G scores

Detailed breakdown table per pillar

Industry benchmark comparison:
| Metric           | Your Building | Industry Avg | Top 10% |
|------------------|---------------|--------------|---------|
| Energy Efficiency| XX%           | 15%          | 40%     |
| Carbon Reduction | XX kg         | 800 kg       | 2000 kg |
| ESG Score        | XX            | 62           | 88      |

Export section:
- "Download ESG Report" button → generates PDF summary
  (use reportlab — title, scores table, radar chart screenshot, recommendations)
```

-----------------------------------------------------------------------
### Page 10: 10_Model_Leaderboard.py
-----------------------------------------------------------------------

```
PURPOSE: Compare all 5 trained models side by side

ON LOAD:
1. Load results_summary.json → all_models dict

DISPLAY:
- Leaderboard table with all 5 models:
  | Rank | Model              | Accuracy | F1    | Precision | Recall | Time  |
  |------|--------------------|----------|-------|-----------|--------|-------|
  | 🥇 1 | XGBoost (Tuned)    | 100.00%  | 100%  | 100%      | 100%   | 4.2s  |
  | 🥈 2 | LightGBM           | 99.50%   | 99.5% | 99.5%     | 99.5%  | 2.1s  |
  - Highlight best model row in gold background

Plotly grouped bar chart: All 4 metrics for all 5 models

Plotly scatter: Accuracy vs Training Time (efficiency plot)

Confusion matrix of best model (Plotly heatmap)

5-fold Cross Validation results:
- Mean CV score per model
- Variance (lower = more stable)

What these metrics mean section:
- Plain English explanation of accuracy, F1, precision, recall
```

-----------------------------------------------------------------------
### Page 11: 11_SHAP_Explainability.py
-----------------------------------------------------------------------

```
PURPOSE: AI decision explainability using SHAP

ON LOAD:
1. Load assets/shap_bar.png
2. Load assets/shap_beeswarm.png
3. Load assets/shap_waterfall.png

DISPLAY:
- Section 1: What is SHAP? (plain English, 3 sentences)
- Image: shap_bar.png with caption
  "Top 20 features the AI uses to detect occupancy"
- Image: shap_beeswarm.png with caption
  "How each feature value pushes the prediction"
- Image: shap_waterfall.png with caption
  "Why the AI made this specific prediction"

Feature importance ranked list:
- Top 10 features with plain English meaning:
  e.g. "pir_mean — Average sensor activity. High = people present"

Key insights section:
- "The AI relies most on: pir_mean, pir_energy, seg3_max"
- "Temperature has X% influence on prediction"
- "Stationary vs Motion is distinguished mainly by pir_diff_max"
```

-----------------------------------------------------------------------
### Page 12: 12_Report_Generator.py
-----------------------------------------------------------------------

```
PURPOSE: Download professional PDF report

FORM INPUTS:
- Text: Building Name
- Text: Building Location
- Number: Number of Rooms
- Date: Report Date
- Selectbox: Report Type [Executive Summary, Full Technical, Sustainability]
- Button: Generate Report

ON GENERATE — create PDF using ReportLab with:

PAGE 1 — Title Page:
  - PIRvision Smart Occupancy System
  - Building name, location, date
  - Prepared by: [team name]

PAGE 2 — Executive Summary:
  - Project overview (3 sentences)
  - Key results table

PAGE 3 — Model Performance:
  - All 5 models comparison table
  - Best model highlighted

PAGE 4 — Energy Savings:
  - Daily / Monthly / Annual savings in kWh and $
  - Device-by-device breakdown

PAGE 5 — Sustainability Impact:
  - CO2 saved
  - Tree equivalent
  - LEED score
  - ESG grades

PAGE 6 — Recommendations:
  - Top 5 actionable recommendations

- st.download_button() → download as PDF
```

=======================================================================
## PART D — REQUIREMENTS & SETUP
=======================================================================

-----------------------------------------------------------------------
### File: requirements.txt
-----------------------------------------------------------------------

```
streamlit==1.32.0
xgboost==2.0.3
lightgbm==4.3.0
scikit-learn==1.4.1
imbalanced-learn==0.12.0
shap==0.44.1
optuna==3.6.1
joblib==1.3.2
numpy==1.26.4
pandas==2.2.1
scipy==1.12.0
matplotlib==3.8.3
seaborn==0.13.2
plotly==5.20.0
reportlab==4.1.0
Pillow==10.2.0
```

-----------------------------------------------------------------------
### File: README.md
-----------------------------------------------------------------------

```markdown
# PIRvision Smart Occupancy & Sustainability System

AI-powered building intelligence using PIR sensor data to detect
occupancy, optimize energy, and drive sustainability reporting.

## Features
- Real-time occupancy classification (Vacancy / Stationary / Motion)
- Room-by-room building heatmap
- Weekly occupancy pattern analysis
- ROI & savings dashboard
- Real-time carbon footprint tracker
- LEED green building scorecard
- CO2 impact visualizer
- Sustainability goals tracker
- ESG reporting dashboard
- Model comparison leaderboard
- SHAP AI explainability
- PDF report generator

## Setup
1. pip install -r requirements.txt
2. Run training notebook in Google Colab
3. Copy models/ and assets/ folders to project root
4. streamlit run app/app.py

## Dataset
UCI PIRvision FoG Presence Detection — ID 1101
https://archive.ics.uci.edu/dataset/1101/pirvision_fog_presence_detection

## Stack
Python | XGBoost | LightGBM | SHAP | Optuna | Streamlit | Plotly | ReportLab
```

=======================================================================
## PART E — REAL-TIME LIVE FEATURES (Cursor must implement these)
=======================================================================

-----------------------------------------------------------------------
### REAL-TIME CORE PATTERN — Use this in Pages 1, 2, 4, 5, and Live Feed
-----------------------------------------------------------------------

```python
# STANDARD REAL-TIME PATTERN — copy and adapt for each page

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
from collections import deque

# ── Step 1: Initialize rolling state ONCE ─────────────────────────
if 'pir_history'    not in st.session_state:
    st.session_state.pir_history    = deque(maxlen=30)
if 'time_history'   not in st.session_state:
    st.session_state.time_history   = deque(maxlen=30)
if 'pred_history'   not in st.session_state:
    st.session_state.pred_history   = deque(maxlen=30)
if 'live_running'   not in st.session_state:
    st.session_state.live_running   = True

# ── Step 2: Pause / Resume button in sidebar ───────────────────────
with st.sidebar:
    if st.button('⏸ Pause Live Feed' if st.session_state.live_running
                 else '▶ Resume Live Feed'):
        st.session_state.live_running = not st.session_state.live_running

# ── Step 3: LIVE indicator ─────────────────────────────────────────
if st.session_state.live_running:
    st.markdown('🔴 **LIVE** — updating every 2 seconds')
else:
    st.markdown('⏸ **PAUSED**')

# ── Step 4: Create empty placeholders ─────────────────────────────
chart_ph   = st.empty()
metrics_ph = st.empty()
log_ph     = st.empty()

# ── Step 5: Live update loop ───────────────────────────────────────
while st.session_state.live_running:
    # Generate new simulated data point
    # (In production: replace with real sensor API call)
    new_pir  = float(np.random.normal(50, 20))
    new_time = time.strftime('%H:%M:%S')

    st.session_state.pir_history.append(new_pir)
    st.session_state.time_history.append(new_time)

    # ── Update chart ───────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x    = list(st.session_state.time_history),
        y    = list(st.session_state.pir_history),
        mode = 'lines+markers',
        line = dict(color='#e74c3c', width=2),
        fill = 'tozeroy',
        fillcolor = 'rgba(231,76,60,0.1)',
        name = 'PIR Signal'
    ))
    fig.update_layout(
        title       = 'Live PIR Sensor Feed — Last 30 Readings',
        xaxis_title = 'Time',
        yaxis_title = 'Signal Value',
        height      = 300,
        margin      = dict(l=0, r=0, t=40, b=0),
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor  = 'rgba(0,0,0,0)',
        showlegend  = False
    )
    with chart_ph.container():
        st.plotly_chart(fig, use_container_width=True)

    # ── Update metrics ─────────────────────────────────────────────
    with metrics_ph.container():
        c1, c2, c3 = st.columns(3)
        c1.metric('Current Value', f'{new_pir:.1f}')
        c2.metric('Mean (30s)',    f'{np.mean(st.session_state.pir_history):.1f}')
        c3.metric('Peak (30s)',    f'{np.max(st.session_state.pir_history):.1f}')

    time.sleep(2)   # refresh rate — change per page
    st.rerun()      # trigger Streamlit re-render
```

-----------------------------------------------------------------------
### File: app/pages/1_Live_Prediction.py — LIVE SENSOR GRAPH
-----------------------------------------------------------------------

```
ADD TO PAGE 1 — below the Run Prediction button section:

LIVE SENSOR FEED section:
- st.subheader("Live PIR Sensor Feed")
- Rolling line chart: last 30 PIR readings
  - X axis: timestamp (HH:MM:SS)
  - Y axis: signal value
  - Line color changes based on current predicted state:
      Vacancy    → green  (#2ecc71)
      Stationary → blue   (#3498db)
      Motion     → red    (#e74c3c)
  - Shaded area under line (fill='tozeroy')
  - Dotted horizontal line at mean value
- Updates every 2 seconds using while loop + st.rerun()
- 3 live metrics below chart:
    Current Value | Mean (30s) | Peak (30s)

LIVE PREDICTION LOG section (below chart):
- st.subheader("Prediction Log")
- Scrolling table — new row added every 2 seconds:
    | Time     | Signal | Predicted State | Confidence | Action       |
    |----------|--------|-----------------|------------|--------------|
    | 09:14:32 | 187.3  | Motion          | 99.2%      | HVAC: Full   |
    | 09:14:30 | 5.1    | Vacancy         | 100%       | HVAC: OFF    |
    | 09:14:28 | 42.7   | Stationary      | 97.8%      | HVAC: Eco    |
- Keep last 10 rows only (deque maxlen=10)
- Color each row by predicted state

LIVE ENERGY SAVINGS TICKER section:
- st.subheader("Savings Accumulating Live")
- Money saved counter ticking up every 2 seconds:
    $ 2.47 saved today
    ↑ increases by savings_rate per second
- kWh saved counter alongside
- Both stored in st.session_state and increment each loop

REFRESH RATE: 2 seconds
DEQUE SIZE:   30 for chart, 10 for log table
```

-----------------------------------------------------------------------
### File: app/pages/2_Room_Heatmap.py — LIVE ROOM STATUS
-----------------------------------------------------------------------

```
ADD TO PAGE 2 — make the heatmap auto-refresh:

LIVE BUILDING MONITOR:
- Add "● LIVE" badge next to page title
- Rooms automatically change state every 5 seconds
  (call simulate_building() with current time_of_day each loop)
- Each room card updates color without full page reload
  (use st.empty() placeholder for the entire room grid)

LIVE SUMMARY BAR (updates every 5s):
- Vacant rooms count   — animated number
- Occupied rooms count — animated number
- Active rooms count   — animated number
- Total energy W       — animated number
- $ wasted this hour   — ticking up

LIVE ALERT SYSTEM:
- If vacant rooms > 40%:
    st.warning("⚠️ X rooms vacant — recommend shutdown — saving $Y/hour")
- If all rooms active:
    st.success("✅ Building fully utilized — AI running at peak efficiency")
- Alert box updates every 5 seconds based on current room states

LIVE ENERGY WASTE METER:
- Plotly gauge chart — shows % of energy being wasted right now
- Needle moves as room states change
- Updates every 5 seconds

REFRESH RATE: 5 seconds
DEQUE SIZE:   12 for energy history chart (last 1 minute)
```

-----------------------------------------------------------------------
### File: app/pages/4_ROI_Dashboard.py — LIVE SAVINGS METER
-----------------------------------------------------------------------

```
ADD TO PAGE 4 — live money counter:

LIVE SAVINGS TICKER (top of page):
- 3 large animated counters side by side:

    💰 Saved Today        ⚡ kWh Saved Today      🌱 CO2 Saved Today
    $ 2.47                4.2 kWh                 0.98 kg
    ↑ ticks up            ↑ ticks up              ↑ ticks up
      every 3 seconds       every 3 seconds         every 3 seconds

- Counter logic:
    savings_per_second = annual_savings_USD / (365 * 24 * 3600)
    today_start = 9 AM (business start)
    seconds_since_start = current_time - 9AM in seconds
    displayed_value = savings_per_second * seconds_since_start

LIVE CUMULATIVE SAVINGS LINE CHART:
- Plotly line chart that extends right every 3 seconds
- X axis: time today (9AM to now)
- Y axis: cumulative $ saved
- Starts at $0 at 9AM, grows steadily to right
- Shows projected end-of-day savings as dotted line

REFRESH RATE: 3 seconds
```

-----------------------------------------------------------------------
### File: app/pages/5_Carbon_Tracker.py — LIVE CO2 COUNTER
-----------------------------------------------------------------------

```
ADD TO PAGE 5 — live CO2 ticker (most impressive real-time feature):

LIVE CO2 PRODUCED COUNTER:
- Giant number in center of screen — updates every 1 second
- Format: XX.XXX kg CO2 today
- Ticks UP every second (building is always producing some CO2)
- Color: red if over target, green if under
- Blinking dot: 🔴 when over target, 🟢 when under

    CO2 Produced Today
         2.847 kg
    ● ticking up every second

LIVE CO2 SAVED COUNTER (next to it):
- Ticks UP every second alongside produced counter
- Format: XX.XXX kg CO2 saved vs baseline
- Always green color
- Shows the DIFFERENCE between what building would produce without AI
  vs what it's actually producing

    CO2 Saved Today (vs No AI)
         1.234 kg  ✅
    ● ticking up every second

LIVE CARBON BUDGET BAR:
- Progress bar updates every second
- Daily budget: X kg (user sets in sidebar)
- Used so far: Y kg (ticking up)
- Remaining: Z kg (ticking down)
- Color: green → yellow → red as budget fills up

LIVE CO2 RATE GAUGE:
- Plotly gauge chart updating every 5 seconds
- Shows current kg CO2 per hour
- Green zone: below target
- Yellow zone: near target
- Red zone: over target
- Needle animates as state changes

LIVE AREA CHART (bottom of page):
- X axis: hours of today (0:00 to current time)
- Two area traces:
    1. CO2 produced (red area)
    2. CO2 baseline without AI (grey dotted line)
- Gap between them = savings
- Extends right every 5 seconds

REFRESH RATES:
- CO2 counter    → 1 second  (most dramatic effect)
- Budget bar     → 1 second
- Gauge chart    → 5 seconds
- Area chart     → 5 seconds
```

-----------------------------------------------------------------------
### File: app/pages/0_Live_Feed.py — NEW PAGE (add to pages/)
-----------------------------------------------------------------------

```
NEW PAGE — Live Prediction Stream (most impressive for demo)

PURPOSE: Shows a scrolling real-time log of AI predictions
         as if sensors are sending data from the whole building
         Looks exactly like a real production monitoring system

LAYOUT:
- Title: "🔴 LIVE — Building Intelligence Feed"
- Subtitle: "AI processing sensor data from 20 rooms in real time"

TOP ROW — 4 live counters updating every 3 seconds:
    Predictions Made | Vacancies Detected | Active Rooms | $ Saved Today

MAIN AREA — Scrolling prediction log:
- New row appears at TOP every 3 seconds
- Table auto-scrolls — old rows push down
- Keep last 20 rows

    | Time     | Room        | State       | Confidence | Action        | Saving |
    |----------|-------------|-------------|------------|---------------|--------|
    | 09:14:32 | Meeting A   | 🔴 Motion   | 99.2%      | HVAC: Full    | $0.00  |
    | 09:14:29 | Office 3    | 🟢 Vacant   | 100.0%     | HVAC: OFF     | $0.18  |
    | 09:14:26 | Lobby       | 🔵 Sitting  | 97.8%      | HVAC: Eco     | $0.06  |
    | 09:14:23 | Board Room  | 🟢 Vacant   | 99.5%      | HVAC: OFF     | $0.18  |

- Row colors:
    Motion     → light red background
    Stationary → light blue background
    Vacant     → light green background

SIDE PANEL (right 30% of screen):
- Mini donut chart: % Vacant vs Stationary vs Motion (updates every 3s)
- Live energy bar: total building watts right now
- Top 3 rooms by energy waste

BOTTOM — Live dual-axis chart:
- X: last 60 seconds
- Y left:  number of occupied rooms
- Y right: energy consumption W
- Updates every 3 seconds
- Shows correlation between occupancy and energy

REFRESH RATE: 3 seconds
DEQUE SIZE:   20 rows for log, 20 points for chart
```

-----------------------------------------------------------------------
### UPDATED FOLDER STRUCTURE (add new page + realtime util)
-----------------------------------------------------------------------

```
app/
├── app.py
├── pages/
│   ├── 0_Live_Feed.py              ← NEW — scrolling prediction stream
│   ├── 1_Live_Prediction.py        ← UPDATED — add live sensor graph
│   ├── 2_Room_Heatmap.py           ← UPDATED — auto-refresh rooms
│   ├── 3_Occupancy_Timeline.py
│   ├── 4_ROI_Dashboard.py          ← UPDATED — live savings ticker
│   ├── 5_Carbon_Tracker.py         ← UPDATED — live CO2 counter
│   ├── 6_Green_Scorecard.py
│   ├── 7_CO2_Visualizer.py
│   ├── 8_Sustainability_Goals.py
│   ├── 9_ESG_Dashboard.py
│   ├── 10_Model_Leaderboard.py
│   ├── 11_SHAP_Explainability.py
│   └── 12_Report_Generator.py
└── utils/
    ├── features.py
    ├── predict.py
    ├── energy.py
    ├── carbon.py
    ├── sustainability.py
    ├── simulation.py
    └── realtime.py                 ← NEW — shared real-time helpers
```

-----------------------------------------------------------------------
### File: app/utils/realtime.py — Shared Real-Time Helpers
-----------------------------------------------------------------------

```python
# app/utils/realtime.py
# Shared utilities for all real-time live update pages

import time
import numpy as np
from collections import deque
from datetime import datetime


def get_timestamp() -> str:
    """Return current time as HH:MM:SS string."""
    return datetime.now().strftime('%H:%M:%S')


def seconds_since_9am() -> float:
    """Return seconds elapsed since 9:00 AM today (business start)."""
    now   = datetime.now()
    start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    delta = (now - start).total_seconds()
    return max(0.0, delta)


def compute_live_savings(annual_usd: float) -> dict:
    """
    Compute live savings counters based on time elapsed since 9AM.
    Returns today's accumulated savings in $, kWh, and CO2 kg.
    """
    seconds      = seconds_since_9am()
    per_second   = annual_usd / (365 * 10 * 3600)  # 10hr working day
    usd_today    = per_second * seconds
    kwh_today    = usd_today / 0.12
    co2_today    = kwh_today * 0.233
    return {
        'usd_today':   round(usd_today, 3),
        'kwh_today':   round(kwh_today, 3),
        'co2_today':   round(co2_today, 3),
        'per_second':  round(per_second, 6)
    }


def compute_live_co2(state: str, rooms: int = 1) -> dict:
    """
    Compute live CO2 rates for carbon tracker.
    Returns current production rate and saved rate.
    """
    # Watts per state for all devices
    power_W = {
        'Vacancy':    150,
        'Stationary': 1350,
        'Motion':     2600,
        'Baseline':   2600
    }
    current_W  = power_W.get(state, 1350) * rooms
    baseline_W = power_W['Baseline'] * rooms

    current_kwh_hr  = current_W  / 1000
    baseline_kwh_hr = baseline_W / 1000

    return {
        'co2_rate_kg_hr':      round(current_kwh_hr  * 0.233, 4),
        'baseline_rate_kg_hr': round(baseline_kwh_hr * 0.233, 4),
        'saved_rate_kg_hr':    round((baseline_kwh_hr - current_kwh_hr) * 0.233, 4),
        'co2_per_second':      round(current_kwh_hr  * 0.233 / 3600, 6),
        'saved_per_second':    round((baseline_kwh_hr - current_kwh_hr) * 0.233 / 3600, 6)
    }


def init_session_deques(keys_maxlen: dict):
    """
    Initialize session_state deques if not already present.
    keys_maxlen = {'pir_history': 30, 'time_history': 30, ...}
    """
    import streamlit as st
    for key, maxlen in keys_maxlen.items():
        if key not in st.session_state:
            st.session_state[key] = deque(maxlen=maxlen)


def make_live_chart(x_data, y_data, color: str,
                    title: str, y_label: str,
                    height: int = 300):
    """
    Build a Plotly live scrolling line chart.
    Returns a go.Figure ready for st.plotly_chart().
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x         = list(x_data),
        y         = list(y_data),
        mode      = 'lines+markers',
        line      = dict(color=color, width=2),
        fill      = 'tozeroy',
        fillcolor = color.replace(')', ',0.1)').replace('rgb', 'rgba')
                    if color.startswith('rgb')
                    else color + '1a',
        marker    = dict(size=4)
    ))
    fig.update_layout(
        title         = title,
        xaxis_title   = 'Time',
        yaxis_title   = y_label,
        height        = height,
        margin        = dict(l=0, r=0, t=40, b=30),
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor  = 'rgba(0,0,0,0)',
        showlegend    = False,
        xaxis         = dict(showgrid=False),
        yaxis         = dict(gridcolor='rgba(128,128,128,0.2)')
    )
    return fig
```

=======================================================================
## PART F — NOTES FOR CURSOR
=======================================================================

IMPORTANT RULES:
1. All charts in Streamlit pages use PLOTLY (not matplotlib) for interactivity
2. Models are loaded ONCE using @st.cache_resource in app.py — passed via st.session_state to all pages
3. feature_names.json must be used to reindex all prediction DataFrames
4. All pages import from app/utils/ — never duplicate logic
5. carbon.py and sustainability.py are pure Python (no Streamlit) — only called from pages
6. simulation.py generates fake data for pages that don't do live prediction
7. PDF generation in Page 12 uses reportlab — save to BytesIO then st.download_button
8. SHAP plots are PNG images saved during Colab training — just display with st.image()
9. Use st.columns() for side-by-side layouts throughout
10. Every page must have the top KPI bar (load from results_summary.json)

REAL-TIME RULES (critical — read carefully):
11. Use while True + time.sleep(N) + st.rerun() for all live pages
12. ALWAYS use st.empty() placeholders — never re-render full page
13. ALWAYS use collections.deque(maxlen=N) for rolling data — never plain list
14. ALWAYS init session_state at top of page before the loop starts
15. Import and use realtime.py helpers — do not duplicate timer logic
16. Pause/Resume button MUST be in sidebar — sets st.session_state.live_running = False
17. Show 🔴 LIVE or ⏸ PAUSED badge at top of every live page
18. Refresh rates per page:
      0_Live_Feed.py       → 3 seconds
      1_Live_Prediction.py → 2 seconds  (chart) + 2 seconds (log)
      2_Room_Heatmap.py    → 5 seconds
      4_ROI_Dashboard.py   → 3 seconds
      5_Carbon_Tracker.py  → 1 second   (counter) + 5 seconds (chart)
19. NEVER use st.rerun() inside a non-live page — only in pages with while True loops
20. Use plotly.graph_objects (go.) not plotly.express (px.) for live charts
    — go. gives more control over updates and avoids full re-renders

BUILD ORDER FOR CURSOR:
1. Create complete folder structure
2. Write app/utils/realtime.py first
3. Write all other utils/ files
4. Write app.py home page
5. Write 0_Live_Feed.py (new live feed page)
6. Write pages 1-12 in order — add live features to 1, 2, 4, 5 as described
7. Test with: streamlit run app/app.py

DEPLOYMENT:
- Upload to Streamlit Cloud (share.streamlit.io) — free
- Add models/ and assets/ to repo
- Set Python version 3.10 in streamlit config
- Permanent URL: https://pirvision-occupancy.streamlit.app
- NOTE: Real-time while loops work fine on Streamlit Cloud
