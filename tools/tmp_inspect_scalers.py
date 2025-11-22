import json
p = r'results_noquick_debug\generated_figures_tables_archive\transfer_k_80\eps_inf\delta_1eneg05\objective_1\lca_artifacts\debug_weights\debug_weight_fedavg_client_4_1763460619909.json'
with open(p,'r',encoding='utf-8') as fh:
    j = json.load(fh)
sc = j.get('scaler', {})
idxs = [14,27,34,44,75,93,95,101,105,115]
for i in idxs:
    m = sc.get('mean')
    s = sc.get('scale')
    if m:
        print(f"{i}: mean={m[i]} scale={s[i] if s else None}")
    else:
        print(f"{i}: mean=None scale=None")
