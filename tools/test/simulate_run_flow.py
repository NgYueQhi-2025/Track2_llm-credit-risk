import sys, os
_THIS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import pandas as pd
from backend import integrations

# use demo data from app
from backend.app import load_demo_data

df = load_demo_data('Demo A')
print('Loaded df:')
print(df)

features_list = []
for i, (_idx, row) in enumerate(df.iterrows(), start=1):
    print('\nProcessing row', i)
    res = integrations.run_feature_extraction(row.to_dict(), mock=True)
    features = res.get('features', {})
    if 'applicant_id' not in features:
        features['applicant_id'] = row.get('id', i)
    parsed = res.get('parsed', {})
    features['_parsed'] = parsed
    norm = integrations.expand_parsed_to_fields(parsed)
    for k, v in norm.items():
        if k == 'risky_phrases':
            features['risky_phrases_list'] = v
        else:
            if k not in features or features.get(k) is None:
                features[k] = v
    features_list.append(features)

preds_rows = []
for feat in features_list:
    pred = integrations.predict(feat)
    merged = {**feat, **pred}
    preds_rows.append(merged)

preds_df = pd.DataFrame(preds_rows)
print('\nPreds df columns:', preds_df.columns.tolist())
if 'applicant_id' in preds_df.columns:
    preds_df = preds_df.rename(columns={'applicant_id': 'id'})
    try:
        preds_df['id'] = preds_df['id'].astype(df['id'].dtype)
    except Exception:
        pass
    out_df = df.merge(preds_df, on='id', how='left')
else:
    out_df = pd.concat([df.reset_index(drop=True), preds_df.reset_index(drop=True)], axis=1)

print('\nMerged output:')
print(out_df.head())
