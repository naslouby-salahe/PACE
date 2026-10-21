from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(__file__).resolve().parent))
from verify_frozen_reporting import load,targets,means
root=Path(__file__).resolve().parents[2]; records,_,_=load()
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True)
out=ap.parse_args().output_dir;out.mkdir(parents=True,exist_ok=True)
cache={}; sources=[]
for p in (root/'outputs/cache/clients').glob('*/manifest.json'):
    for i,c in enumerate(json.loads(p.read_text())['clients']):
        key=(c['study_stratum'],c['client']); cache[key]=(p.parent,i,c)
        sources.append({'stratum':key[0],'client':key[1],'manifest':str(p.relative_to(root)),'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'schema':c['feature_schema'],'sources':sorted(set(x['source_file'] for x in c['captures']))})
absent=[]; total=0; sizes=[]; fprs=[]
for (a,n,st,name,seed),t in records.items():
    if (a,n,seed)!=(.01,100,6):continue
    minrows=400 if st in ['IOT_23','TON_IOT','CTU_13'] else 30 if st.startswith('UNSW') else 600
    peerlabels=set()
    for peer in t['peer_model_seeds']:
        ds=peer['owner']['name']; directory,i,c=cache[(st,ds)]
        for j,cap in enumerate(c['captures']):
            arr=np.load(directory/f'{i}.{j}.order.npy',mmap_mode='r')
            if len(arr)>=minrows: peerlabels.add(cap['attack_type'])
    for attack in t['attack_types']:
        label=attack['attack_type']['identifier']; total+=1
        if label not in peerlabels: absent.append([st,name,label])
    directory,i,c=cache[(st,name)]
    order=np.load(directory/f'{i}.order.npy',mmap_mode='r')
    cal=int(len(order)*(.4+.2+.2))-int(len(order)*(.4+.2))
    sizes.append(cal)
    ts=targets(records,.01,100)[(t['target']['dataset'],name)]
    fprs.append(np.mean([x['realized_false_alert_rate'] for x in ts]))
directory,i,c=cache[('IOT_23','7-1')]
arr=np.load(directory/f'{i}.benign.npy',mmap_mode='r')
order=np.argsort(np.load(directory/f'{i}.order.npy'),kind='stable')
col=c['feature_names'].index('orig_ip_bytes'); refend=int(len(order)*(.4+.2)); calend=int(len(order)*(.4+.2+.2))
cal=arr[order[refend:calend],col]; test=arr[order[calend:],col]
result={'absent_labels':absent,'absent_count':len(absent),'pairs':total,'calibration_size_spearman':spearmanr(sizes,fprs).statistic,'iot23_7_1_outside_calibration_range':float(np.mean((test<cal.min())|(test>cal.max())))}
(out/'cache-diagnostics.json').write_text(json.dumps(result,indent=2))
(out/'dataset-input-manifest.json').write_text(json.dumps(sources,indent=2))
print(json.dumps(result,indent=2))
for st in sorted(set(x['stratum'] for x in sources)):
    paths=sorted(set(p for x in sources if x['stratum']==st for p in x['sources']))
    print(st,len(paths),paths[:2])

