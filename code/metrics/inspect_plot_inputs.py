import os
import pandas as pd
import json

FOLDER = 'results_metrics/linkability_results/to_plot'
FEATURE = 'linkability_value'
OUT = 'results_metrics/linkability_results/aggregated/plot_inputs_preview.json'
MAX_SHOW = 50

feature_aliases = {
    "value": ["value", "linkability_value"],
    "linkability_value": ["linkability_value", "value"],
}

candidate_features = feature_aliases.get(FEATURE, [FEATURE])

data = {}

for root, _, files in os.walk(FOLDER):
    for file in files:
        if not file.endswith('.csv'):
            continue
        path = os.path.join(root, file)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print('ERR read', path, e)
            continue
        resolved_feature = next((col for col in candidate_features if col in df.columns), None)
        if resolved_feature is None:
            # try lowercase match
            lc = {c.lower(): c for c in df.columns}
            for c in candidate_features:
                if c.lower() in lc:
                    resolved_feature = lc[c.lower()]
                    break
        if resolved_feature is None:
            continue
        values = df[resolved_feature].dropna()
        # coerce numeric
        values = pd.to_numeric(values, errors='coerce').dropna()
        if values.empty:
            continue
        rel = os.path.relpath(path, FOLDER)
        parts = os.path.normpath(rel).split(os.sep)
        top = parts[0] if parts else 'root'
        data.setdefault(top, []).append({'file': os.path.basename(path), 'values': values.tolist()})

# Build preview
preview = {}
for k, items in data.items():
    all_vals = []
    files = []
    for it in items:
        files.append(it['file'])
        all_vals.extend(it['values'])
    s = pd.Series(all_vals)
    preview[k] = {
        'n_files': len(items),
        'n_values': int(s.size),
        'n_zeros': int((s==0).sum()),
        'n_nonzero': int((s!=0).sum()),
        'min': float(s.min()) if not s.empty else None,
        'max': float(s.max()) if not s.empty else None,
        'mean': float(s.mean()) if not s.empty else None,
        'median': float(s.median()) if not s.empty else None,
        'sample_first_values': s.head(MAX_SHOW).tolist(),
        'sample_last_values': s.tail(MAX_SHOW).tolist(),
        'files_sample': files[:10]
    }

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as fh:
    json.dump(preview, fh, indent=2)

print('Wrote preview to', OUT)
for k,v in preview.items():
    print('---', k)
    print('n_files', v['n_files'], 'n_values', v['n_values'], 'n_zeros', v['n_zeros'])
    print('min/max/median/mean', v['min'], v['max'], v['median'], v['mean'])
    print('first values (up to 50):', v['sample_first_values'][:10], '...')
    print('files sample:', v['files_sample'])
