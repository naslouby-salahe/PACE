"""Read-only sweep support analysis; no fitted models or scientific artifacts are changed."""
from pathlib import Path
import json
import numpy as np
from verify_frozen_reporting import load, targets, rates, interval

def analyze():
    records, _, _ = load()
    output = {}
    for alpha in (.01, .05):
        for n in (100, 400):
            groups = targets(records, alpha, n)
            complete_ids, complete_values, cells = [], [], []
            covered = exact = ambiguous = 0
            for identity, ts in groups.items():
                target_values = []
                for t in sorted(ts, key=lambda t: t['seed']):
                    sweep = sorted(t['budget_sweep'], key=lambda s: s['false_alert_rate'])
                    ctpr, cfpr = rates(t, 'MEAN_ENSEMBLE_LOCAL_BRANCH')
                    xs = np.array([s['false_alert_rate'] for s in sweep])
                    ys = np.array([s['detection_rate'] for s in sweep])
                    support = bool(xs[0] <= cfpr <= xs[-1])
                    matches = ys[xs == cfpr]
                    exact += int(bool(len(matches)))
                    ambiguous += int(len(matches)>1 and np.ptp(matches)>0)
                    if cfpr <= xs[0]: adjusted = ys[0]
                    elif cfpr >= xs[-1]: adjusted = ys[-1]
                    else: adjusted = np.interp(cfpr, xs, ys)
                    value = float(adjusted-ctpr)
                    if support:
                        covered += 1
                        target_values.append(value)
                    cells.append(dict(corpus=identity[0], target=identity[1], seed=t['seed'], covered=support,
                                      comparator_fpr=float(cfpr), minimum_fpr=float(xs[0]),maximum_fpr=float(xs[-1]),
                                      exact_stored_fpr=bool(len(matches)), adjusted_gain=value if support else None))
                if len(target_values)==3:
                    complete_ids.append(identity); complete_values.append(float(np.mean(target_values)))
            output[f'{alpha}/{n}'] = dict(total_cells=len(cells),covered_cells=covered,
                exact_stored_fpr_cells=exact,tied_fpr_with_different_tpr_cells=ambiguous,
                all_seed_supported_targets=len(complete_ids),complete_target_gain=float(np.mean(complete_values)),
                complete_target_ci=interval(complete_ids,complete_values), complete_target_ids=complete_ids,cells=cells)
    return output

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=analyze(); args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for k,v in result.items(): print(k,{kk:vv for kk,vv in v.items() if kk not in ('cells','complete_target_ids')})
