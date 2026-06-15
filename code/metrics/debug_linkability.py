import os
import json
import pandas as pd

FOLDER = 'results_metrics/linkability_results/to_plot'
FEATURE_CANDIDATES = ['linkability_value', 'value', 'linkability']
OUT_JSON = 'results_metrics/linkability_results/aggregated/linkability_diagnostics.json'

def collect(folder_path, feature_candidates):
    data = {}
    for root, _, files in os.walk(folder_path):
        for f in files:
            if not f.endswith('.csv'):
                continue
            path = os.path.join(root, f)
            try:
                df = pd.read_csv(path)
            except Exception as e:
                print('ERR reading', path, e)
                continue
            col = None
            for c in feature_candidates:
                if c in df.columns:
                    col = c
                    break
            if col is None:
                # try lowercase match
                lc = {c.lower(): c for c in df.columns}
                for c in feature_candidates:
                    if c.lower() in lc:
                        col = lc[c.lower()]
                        break
            if col is None:
                print('No feature column in', path)
                continue
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            if vals.empty:
                continue
            # key by top-level folder under provided folder_path
            rel = os.path.relpath(path, folder_path)
            parts = os.path.normpath(rel).split(os.sep)
            top = parts[0] if parts else 'root'
            data.setdefault(top, []).append({'file': path, 'values': vals.tolist(), 'col': col})
    return data


def summarize(data):
    out = {}
    for top, items in data.items():
        all_vals = []
        files = []
        cols = set()
        for it in items:
            files.append(os.path.basename(it['file']))
            cols.add(it['col'])
            all_vals.extend(it['values'])
        s = pd.Series(all_vals)
        stats = {
            'n_files': len(items),
            'n_values': int(s.size),
            'n_zeros': int((s==0).sum()),
            'n_nonzero': int((s!=0).sum()),
            'min': float(s.min()) if not s.empty else None,
            'max': float(s.max()) if not s.empty else None,
            'median': float(s.median()) if not s.empty else None,
            'mean': float(s.mean()) if not s.empty else None,
            '25%': float(s.quantile(0.25)) if not s.empty else None,
            '75%': float(s.quantile(0.75)) if not s.empty else None,
            'unique_rounded_counts': s.round(6).value_counts().head(10).to_dict(),
            'sample_values_nonzero': s[s!=0].sample(min(10, (s!=0).sum())).tolist() if (s!=0).sum()>0 else [],
            'cols_seen': list(cols),
            'files_sample': files[:10]
        }
        out[top] = stats
    return out


def main():
    data = collect(FOLDER, FEATURE_CANDIDATES)
    if not data:
        print('No data found under', FOLDER)
        return
    summary = summarize(data)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as fh:
        json.dump({'summary': summary}, fh, indent=2)
    print('Wrote diagnostics to', OUT_JSON)
    for k,v in summary.items():
        print('---', k)
        for kk,vv in v.items():
            print(f'{kk}: {vv}')

if __name__ == '__main__':
    main()
