from pathlib import Path
import json, hashlib, csv
import numpy as np
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2]
ORDER=['IOT_23','TON_IOT','GOTHAM_2025','CTU_13','N_BAIOT','UNSW_IOT_ATTACK_FLOWS']
def load():
    unique={}; manifest=[]; repeats=0
    for p in sorted((ROOT/'outputs/campaigns').rglob('results.json')):
        o=json.loads(p.read_text())
        payload={k:v for k,v in o.items() if k!='content_digest'}
        assert hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==o['content_digest'],p
        manifest.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'content_digest':o['content_digest'],'implementation_digest':o['implementation_digest']})
        a=.01 if o['scientific_configuration']['alpha']=='ALPHA_001' else .05
        n=o['scientific_configuration']['local_detectors']['local_training_rows']
        for t in o['outcome']['targets']:
            key=(a,n,t['study_stratum'],t['target']['name'],t['seed'])
            if key in unique: assert unique[key]==t; repeats+=1
            unique[key]=t
    return unique,manifest,repeats
def interval(ids,values):
    rng=np.random.default_rng(314159); estimates=np.zeros(10000)
    for dataset in ORDER:
        group=sorted((name,float(v)) for (ds,name),v in zip(ids,values) if ds==dataset)
        if not group: continue
        v=np.array([x[1] for x in group])
        choices=rng.integers(0,len(v),size=(10000,len(v)))
        estimates+=v[choices].mean(axis=1)*len(v)/len(values)
    return np.quantile(estimates,[.025,.975],method='linear').tolist()
def rates(t,method):
    if method=='PACE':
        return [np.mean([x['detection_rate'] for x in t['attack_types']]),t['realized_false_alert_rate']]
    if method in ['RECORDED_MAX','RECORDED_BONFERRONI']:
        k='max_fusion' if method=='RECORDED_MAX' else 'bonferroni'
        return [np.mean([x[k+'_detection_rate'] for x in t['attack_types']]),t[k+'_false_alert_rate']]
    arm=next(x for x in t['arms'] if x['arm']==method)
    return [np.mean(arm['detection_rates']),arm['false_alert_rate']]
def targets(records,a,n,allowed=None):
    grouped=defaultdict(list)
    for (aa,nn,st,name,seed),t in records.items():
        if (aa,nn)==(a,n) and (allowed is None or (st,name) in allowed):
            grouped[(t['target']['dataset'],name)].append(t)
    assert all(sorted(t['seed'] for t in ts)==[6,7,8] for ts in grouped.values())
    return dict(sorted(grouped.items()))
def means(grouped,method):
    return np.array([np.mean([rates(t,method) for t in ts],axis=0) for ts in grouped.values()])
def run(out):
    out.mkdir(parents=True,exist_ok=True)
    records,manifest,repeats=load()
    methods=['PACE']+[x['arm'] for x in next(iter(records.values()))['arms']]+['RECORDED_MAX','RECORDED_BONFERRONI']
    with (out/'target-seed-rates.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['alpha','N','stratum','target','seed','method','TPR','FPR'])
        for (a,n,st,name,seed),t in sorted(records.items()):
            for method in methods:
                tpr,fpr=rates(t,method)
                w.writerow([a,n,st,name,seed,method,tpr,fpr])
    rows=[]; diagnostics={}
    for a in [.01,.05]:
      for n in [100,400]:
        g=targets(records,a,n); assert len(g)==58
        ids=list(g); p=means(g,'PACE'); c=means(g,'MEAN_ENSEMBLE_LOCAL_BRANCH'); d=p-c
        key=f'{a}/{n}'
        dd={'primary':{'tpr':p[:,0].mean(),'comparator_tpr':c[:,0].mean(),'gain':d[:,0].mean(),'ci':interval(ids,d[:,0]),'fpr':p[:,1].mean(),'comparator_fpr':c[:,1].mean(),'fpr_difference':d[:,1].mean(),'fpr_ci':interval(ids,d[:,1])}}
        ex=np.array([ds!='TON_IOT' for ds,name in ids])
        dd['ex_ton']={'gain':d[ex,0].mean(),'ci':interval([i for i,e in zip(ids,ex) if e],d[ex,0])}
        dd['top_three_share']=np.sort(d[:,0])[-3:].sum()/d[:,0].sum()
        keep=np.argsort(d[:,0])[:-5]; dd['remove_top_five']={'gain':d[keep,0].mean(),'ci':interval([ids[i] for i in keep],d[keep,0])}
        dd['win_tie_loss']=[int((d[:,0]>.001).sum()),int((abs(d[:,0])<=.001).sum()),int((d[:,0]<-.001).sum())]
        dd['lower_higher_equal_fpr']=[int((d[:,1]<0).sum()),int((d[:,1]>0).sum()),int((d[:,1]==0).sum())]
        sweepdiff=[]; below=above=0
        for ts in g.values():
            vals=[]
            for t in ts:
                sweep=sorted(t['budget_sweep'],key=lambda x:x['false_alert_rate'])
                ar=rates(t,'MEAN_ENSEMBLE_LOCAL_BRANCH')
                below+=ar[1]<sweep[0]['false_alert_rate']; above+=ar[1]>sweep[-1]['false_alert_rate']
                if ar[1]<=sweep[0]['false_alert_rate']: adjusted=sweep[0]['detection_rate']
                elif ar[1]>=sweep[-1]['false_alert_rate']: adjusted=sweep[-1]['detection_rate']
                else: adjusted=np.interp(ar[1],[x['false_alert_rate'] for x in sweep],[x['detection_rate'] for x in sweep])
                vals.append(adjusted-ar[0])
            sweepdiff.append(np.mean(vals))
        dd['sweep']={'gain':float(np.mean(sweepdiff)),'ci':interval(ids,sweepdiff),'below':below,'above':above}
        if a==.05 and n==400:
            keep=np.array([name not in ['d073d5018308','74c63b29d71d'] for ds,name in ids])
            dd['fpr_without_two']={'difference':d[keep,1].mean(),'ci':interval([i for i,e in zip(ids,keep) if e],d[keep,1]),'contribution':d[~keep,1].sum()/58}
        diagnostics[key]=dd
        for method in methods:
            m=means(g,method); delta=p[:,0]-m[:,0]; ci=interval(ids,delta)
            rows.append(dict(alpha=a,N=n,method=method,tpr=m[:,0].mean(),fpr=m[:,1].mean(),difference=delta.mean(),lower=ci[0],upper=ci[1]))
    # High-resource paired budget change on the same 15 targets.
    high=targets(records,.05,20000)
    low=targets(records,.05,100,set((ts[0]['study_stratum'],name) for (ds,name),ts in high.items()))
    assert list(low)==list(high) and len(high)==15
    changes=means(high,'PACE')[:,0]-means(low,'PACE')[:,0]
    keep=np.array([name!='192.168.1.190' for ds,name in high])
    diagnostics['high_resource']={'host_change':changes[~keep].item(),'other_mean':changes[keep].mean(),'other_ci':interval([i for i,e in zip(high,keep) if e],changes[keep])}
    diagnostics['ctu_seed']={}
    for n in [100,400]:
        ts=targets(records,.01,n)[('CTU_13','CTU13-scenario-12')]
        diagnostics['ctu_seed'][str(n)]=[{'seed':t['seed'],'pace':rates(t,'PACE')[0],'comparator':rates(t,'MEAN_ENSEMBLE_LOCAL_BRANCH')[0],'peer':np.mean([x['peer_union_detection_rate'] for x in t['attack_types']])} for t in sorted(ts,key=lambda t:t['seed'])]
    # Validate pre-existing printed primary summaries without changing them.
    with (ROOT/'results/tables/scaling.csv').open() as f:
        for row in csv.DictReader(f):
            if row['analysis']!='universal_primary': continue
            d=diagnostics[f"{float(row['alpha'])}/{int(row['local_rows'])}"]['primary']
            for col,key in [('pace_detection','tpr'),('mean_ensemble_detection','comparator_tpr'),('gain','gain')]:
                assert f'{d[key]:.6f}'==row[col]
            assert [f'{v:.6f}' for v in d['ci']]==[row['lower'],row['upper']]
    with (out/'comparison-inventory.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'reporting-verification.json').write_text(json.dumps({'artifact_count':len(manifest),'records':len(records)+repeats,'unique_records':len(records),'identical_repeats':repeats,'diagnostics':diagnostics},indent=2))
    (out/'campaign-manifest.json').write_text(json.dumps(manifest,indent=2))
    return rows,diagnostics
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description='Recompute reporting only; never fit or score a model.')
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args(); rows,d=run(args.output_dir)
    print(json.dumps(d,indent=2))
