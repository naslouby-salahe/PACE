"""Reproduce manuscript figures from immutable reporting tables; no model fitting."""
from pathlib import Path
import argparse, csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2]
COLORS=['#19658a','#b54b11','#7157ad','#ab2368','#22754c','#856b14','#555555']
STRATA=['IOT_23','TON_IOT','GOTHAM_2025','CTU_13','N_BAIOT','UNSW_IOT_ATTACK_FLOWS_G1','UNSW_IOT_ATTACK_FLOWS_G2']
LABELS=['IoT-23','TON_IoT','Gotham','CTU-13','N-BaIoT','UNSW G1','UNSW G2']
MARKERS=['o','^','s','D','v','P','X']
SETTINGS=[(.01,100),(.01,400),(.05,100),(.05,400)]
def table(name):
    with (ROOT/'results/tables'/name).open() as f:return list(csv.DictReader(f))
def save(fig,out,name):
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
def base(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(True,color='#eeeeee',linewidth=.8);ax.set_axisbelow(True)
def architecture(out):
    fig,ax=plt.subplots(figsize=(10,10));ax.set(xlim=(0,10),ylim=(0,10));ax.axis('off')
    edge='#465d6b';fill='#edf3f5'
    def box(x,y,w,h,text,fs=10,fc=fill):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.035',linewidth=1.2,edgecolor=edge,facecolor=fc))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,linespacing=1.35)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color=edge,lw=1.3))
    ax.text(.05,9.82,'SUPERVISED PEER ROUTE',weight='bold',fontsize=12)
    ax.text(3.65,9.82,'TARGET-LOCAL ROUTES',weight='bold',fontsize=12)
    box(.03,8.45,3.05,1.08,'Labeled peer data\n\u2192 fixed supervised classifiers')
    box(3.6,8.45,6.33,1.08,'Target benign training: first N rows\nFit local detectors, rarity models, and response IF')
    box(.03,6.92,3.05,1.13,'Attack-minus-benign margins\n\u2192 target-relative evidence')
    arrow((1.55,8.4),(1.55,8.08))
    box(3.6,6.1,2.94,1.65,'Six-detector anchor\nPCA \u00b7 IF \u00b7 LOF \u00b7 kNN\nHistogram rarity\nGaussian distance')
    box(6.89,6.1,3.04,1.65,'Six-expert bank\nReuse IF, histogram, Gaussian\nMarginal + pair rarity\nPeer-response IF',9.6)
    arrow((4.02,8.4),(4.02,7.8));arrow((9.53,8.4),(9.53,7.8))
    ax.plot([3.1,3.3,3.3,8.42],[7.5,7.5,8.03,8.03],color=edge,lw=1.2)
    ax.text(5.95,8.12,'Peer evidence vectors \u2192 response IF',ha='center',fontsize=9,color=edge)
    arrow((8.42,8.04),(8.42,7.8))
    box(3.6,4.58,2.94,1.1,'Reference-transform scores\n\u2192 mean of six evidences')
    box(6.89,4.58,3.04,1.1,'Reference-transform\nsix expert scores')
    arrow((5.07,6.07),(5.07,5.72));arrow((8.42,6.07),(8.42,5.72))
    box(.03,3.08,3.05,1.1,'Separate peer thresholds\n\u2192 calibrated peer union')
    box(3.6,3.08,2.94,1.1,'Calibrate singleton anchor\n\u2192 anchor decision')
    box(6.89,3.08,3.04,1.1,'Separate bank thresholds\n\u2192 calibrated bank union')
    arrow((1.55,6.88),(1.55,4.23));arrow((5.07,4.55),(5.07,4.23));arrow((8.42,4.55),(8.42,4.23))
    box(1.4,1.74,7.15,.9,'FINAL ALERT = peer union OR anchor OR bank union',11,'#dce9ef')
    for x in [1.55,5.07,8.42]:arrow((x,3.05),(x,2.7))
    box(.03,.02,9.9,1.42,'TARGET BENIGN ROLES: chronological 40 / 20 / 20 / 20\nTraining \u2192 local fitting     Reference \u2192 all score transforms\nCalibration \u2192 all three decisions     Test \u2192 held-out FPR only\nHeld-out target attacks \u2192 detection evaluation only: no fitting or calibration',10,'#f5f5f3')
    save(fig,out,'architecture')
def generate(out):
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':12,'pdf.fonttype':42})
    targets=table('targets.csv');scaling=table('scaling.csv')
    inputs={'sources':{name:hashlib.sha256((ROOT/'results/tables'/name).read_bytes()).hexdigest() for name in ['targets.csv','scaling.csv']},'target_gains':[],'false_alerts':[],'budget_scaling':[]}
    architecture(out)
    fig,axs=plt.subplots(2,2,figsize=(10,10),sharey=True)
    for i,((a,n),ax) in enumerate(zip(SETTINGS,axs.flat)):
        rows=[r for r in targets if (float(r['alpha']),int(r['local_rows']))==(a,n)]
        rows=sorted(rows,key=lambda r:(float(r['gain']),r['stratum'],r['target']))
        assert len(rows)==58
        for st,color,marker in zip(STRATA,COLORS,MARKERS):
            xy=[(j+1,float(r['gain'])) for j,r in enumerate(rows) if r['stratum']==st]
            ax.scatter(*zip(*xy),c=color,marker=marker,s=42,edgecolors='white',linewidths=.5,zorder=3)
        ax.axhline(0,color='#444444',lw=.8);base(ax)
        ax.set(title=rf'({chr(97+i)}) $\alpha={a},\,N={n}$',xticks=[1,15,30,45,58],xlim=(0,59),ylim=(-.055,.49))
        if i%2==0:ax.set_ylabel('Detection-rate difference')
        if i>=2:ax.set_xlabel('Target rank (sorted within panel)')
        inputs['target_gains'].append({'alpha':a,'N':n,'points':[{'rank':j+1,'stratum':r['stratum'],'target':r['target'],'gain':float(r['gain'])} for j,r in enumerate(rows)]})
    handles=[Line2D([],[],color=c,marker=m,linestyle='',label=l) for c,m,l in zip(COLORS,MARKERS,LABELS)]
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False)
    fig.subplots_adjust(bottom=.15,hspace=.27,wspace=.13)
    save(fig,out,'target-gains')
    # Standard right-continuous ECDF: each target adds 1/58, including the final target.
    fig,axs=plt.subplots(2,2,figsize=(10,10),sharey=True)
    for i,((a,n),ax) in enumerate(zip(SETTINGS,axs.flat)):
        rows=[r for r in targets if (float(r['alpha']),int(r['local_rows']))==(a,n)]
        for method,col,color,style in [('PACE','pace_false_alert',COLORS[0],'-'),('Mean local comparator','mean_ensemble_false_alert',COLORS[1],'--')]:
            x=np.sort([float(r[col]) for r in rows]);y=np.arange(1,len(x)+1)/len(x)
            ax.step(x,y,where='post',label=method,color=color,ls=style,lw=1.8)
            inputs['false_alerts'].append({'alpha':a,'N':n,'method':method,'x':x.tolist(),'y':y.tolist()})
        # Recorded max-fusion FPR is stored in the immutable JSON, not targets.csv.
        from verify_frozen_reporting import load,targets as group_targets,means
        records,_,_=load();g=group_targets(records,a,n)
        x=np.sort(means(g,'RECORDED_MAX')[:,1]);y=np.arange(1,len(x)+1)/len(x)
        ax.step(x,y,where='post',label='Max-fusion baseline',color=COLORS[-1],ls='-.',lw=1.8)
        inputs['false_alerts'].append({'alpha':a,'N':n,'method':'Max-fusion baseline','x':x.tolist(),'y':y.tolist()})
        for guide in [a,2*a]:ax.axvline(guide,color='#777777',ls=':',lw=1)
        base(ax);ax.set(title=rf'({chr(97+i)}) $\alpha={a},\,N={n}$',xlim=(-.01,.7),ylim=(0,1.02))
        if i%2==0:ax.set_ylabel('Cumulative fraction of targets')
        if i>=2:ax.set_xlabel('Realized target FPR')
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False)
    fig.subplots_adjust(bottom=.12,hspace=.3,wspace=.13);save(fig,out,'false-alert-distribution')
    fig,axs=plt.subplots(1,2,figsize=(10,4.6),sharey=True)
    for a,ax in zip([.01,.05],axs):
        for kind,label,color,marker,ls in [('universal_robustness','58 targets',COLORS[0],'o','-'),('broad_scaling','47 targets',COLORS[1],'s','--'),('high_resource','15 targets',COLORS[-1],'^','-.')]:
            rows=sorted([r for r in scaling if r['analysis']==kind and float(r['alpha'])==a],key=lambda r:int(r['local_rows']))
            assert len(rows)==(3 if kind!='high_resource' else 4),(kind,len(rows))
            x=np.array([int(r['local_rows']) for r in rows]);y=np.array([float(r['gain']) for r in rows]);lo=np.array([float(r['lower']) for r in rows]);hi=np.array([float(r['upper']) for r in rows])
            ax.errorbar(x,y,yerr=np.array([y-lo,hi-y]),color=color,marker=marker,ls=ls,label=label,capsize=3)
            ax.fill_between(x,lo,hi,color=color,alpha=.09)
            inputs['budget_scaling'].append({'alpha':a,'cohort':kind,'N':x.tolist(),'gain':y.tolist(),'lower':lo.tolist(),'upper':hi.tolist()})
        base(ax);ax.axhline(0,color='#444444',lw=.8)
        ax.set_xscale('log');ax.set(title=rf'$\alpha={a}$',xlabel='Local fitting rows',xticks=[100,400,2000,20000],xticklabels=['100','400','2,000','20,000'])
    axs[0].set_ylabel('Detection-rate difference')
    h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False)
    fig.subplots_adjust(bottom=.24,wspace=.14)
    save(fig,out,'budget-scaling')
    (out/'figure-inputs.json').write_text(json.dumps(inputs,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True)
    generate(ap.parse_args().output_dir)
