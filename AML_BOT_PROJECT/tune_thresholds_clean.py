"""
Clean KS-threshold search for the localized drift-aware policy (BotDriftBench v0.8).

Difference from the search inside run_v08_comparison.py:
  * retraining uses TRAINING accounts only (the original search also put the
    validation accounts into the retraining pool, which made the validation
    score optimistic);
  * validation accounts are used only to score each (KS, fraction) pair;
  * test accounts are never touched.

Criterion: mean post-drift validation Macro-F1 (windows >= earliest onset), seed 2025.
Run from the AML_BOT_PROJECT folder:   python tune_thresholds_clean.py
Result obtained for the released data: best = (KS 0.06, frac 0.20), 0.9119, 5 retrains.
"""
import importlib.util, json, numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location("rc", "run_v08_comparison.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)

df = pd.read_csv("dataset_v08/botdriftbench_v08.csv")
meta = pd.read_csv("dataset_v08/account_metadata_v08.csv")
feat = [c for c in df.columns if c not in rc.META_COLS]
tr = set(df[df.split == "train"].account_id)
va = set(df[df.split == "validation"].account_id)
train_df, val_df = df[df.account_id.isin(tr)], df[df.account_id.isin(va)]
onsets = meta[meta.drift_onset != -1].drift_onset.dropna()
gmo = int(onsets.min())

res = {}
for ks in [0.06, 0.09, 0.12, 0.15, 0.20]:
    for fr in [0.20, 0.25, 0.30, 0.35, 0.40]:
        r = rc.run_da_localized(train_df.copy(), val_df, feat, 2025, tr, gmo, ks, fr)
        post = [x["f1_macro"] for x in r["per_window"] if x["window"] >= gmo]
        res[f"{ks}_{fr}"] = (float(np.mean(post)), r["retrains"])
        print(ks, fr, round(res[f"{ks}_{fr}"][0], 4), r["retrains"], flush=True)

best = max(res, key=lambda k: res[k][0])
print("BEST", best, res[best])
json.dump(res, open("tune_thresholds_clean_results.json", "w"), indent=2)
