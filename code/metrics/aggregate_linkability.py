import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

IN_ROOT = 'results_metrics/linkability_results/to_plot'
OUT_ROOT = 'results_metrics/linkability_results/aggregated'
TOP_FOLDERS = []
# discover top folders inside IN_ROOT
if os.path.isdir(IN_ROOT):
    TOP_FOLDERS = [d for d in os.listdir(IN_ROOT) if os.path.isdir(os.path.join(IN_ROOT, d))]
else:
    print(f'Input folder {IN_ROOT} not found')

os.makedirs(OUT_ROOT, exist_ok=True)

created = []
for top in TOP_FOLDERS:
    collected = []
    src_dir = os.path.join(IN_ROOT, top)
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(root, f))
                except Exception:
                    continue
                # try common column names
                col = None
                for c in ['linkability_value', 'value', 'linkability']:
                    if c in df.columns:
                        col = c
                        break
                if col is None:
                    continue
                vals = pd.to_numeric(df[col], errors='coerce').dropna()
                if vals.empty:
                    continue
                tmp = pd.DataFrame({'file': f, 'linkability_value': vals.values, 'source': os.path.relpath(root, IN_ROOT)})
                collected.append(tmp)
    if not collected:
        continue
    agg_df = pd.concat(collected, ignore_index=True)
    out_dir = os.path.join(OUT_ROOT, top)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'linkability_aggregated.csv')
    agg_df.to_csv(out_csv, index=False)
    created.append((top, out_csv))
    print(f'Wrote {out_csv} with {len(agg_df)} rows')

# If we created aggregates for at least 2 tops, plot combined boxplot
if created:
    data = {}
    for top, csv in created:
        df = pd.read_csv(csv)
        data[top] = df['linkability_value'].dropna().values

    if data:
        plt.figure(figsize=(6, 4))
        ax = sns.boxplot(data=pd.DataFrame.from_dict(data, orient='index').T, showfliers=False)
        # zoom to 1st-99th percentiles across data
        combined_list = [pd.Series(v) for v in data.values()]
        if combined_list:
            combined = pd.concat(combined_list, ignore_index=True)
            try:
                lo, hi = combined.quantile([0.01, 0.99]).values
                ax.set_ylim(lo, hi)
            except Exception:
                pass
        out_png = os.path.join(OUT_ROOT, 'linkability_combined.png')
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        print('Saved combined plot to', out_png)
else:
    print('No aggregates created; nothing to plot')
