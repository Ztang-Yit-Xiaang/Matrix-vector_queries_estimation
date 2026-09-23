"""Build three report figures from reviewed artifacts only; zero oracle queries.

Writes only into a new explicitly supplied directory. No historical figure,
result, estimator, threshold, or bootstrap is modified or recomputed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'urop-figure-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUTS = [
    'coordinate_gate_failure_audit_20260916/paths.csv',
    'coordinate_gate_failure_audit_20260916/exact_nullspace_witnesses.json',
    'direct_rademacher_certification_phase1a_bootstrap.csv',
    'direct_rademacher_certification_phase1a_operating_rates.csv',
    'direct_rademacher_certification_phase1a_catastrophic.csv',
    'direct_rademacher_certification_phase1b_budget_bootstrap.csv',
    'direct_rademacher_certification_phase1b_budget_paths.parquet',
]
BLUE, ORANGE, GREY, INK = '#31688E', '#C7752C', '#7A8085', '#22282D'
ESTIMATORS = ['sample_variance', 'mom_w1', 'mom_w2']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def primary(df):
    mask = ((df.eta == 1e-6) & (df.budget == 160) & (df.s == 16)
            & (df.pair == 'primary') & np.isclose(df.epsilon, 1/3, rtol=0, atol=1e-12))
    return df.loc[mask].copy()


def load_reviewed_data():
    result = ROOT/'results'
    coords = pd.read_csv(result/INPUTS[0]).query('orientation == 0').copy()
    assert len(coords) == 20 and not coords.duplicated(['trial','method']).any()
    assert (coords.q+coords.r+coords.ell == 60).all()
    assert (coords.r == coords.q).all() and (coords.risk_rademacher > 0).all()
    gated = coords.query("method == 'gated'")
    bad = gated.loc[gated.risk_rademacher.idxmax()]
    other = gated[gated.trial != bad.trial].sort_values('risk_rademacher')
    typical = other.iloc[len(other)//2]
    assert int(bad.trial) == 7 and bad.signal_rank == 4
    witnesses = json.loads((result/INPUTS[1]).read_text())
    witness = next(w for w in witnesses if w['trial'] == int(bad.trial) and w['method'] == 'gated')
    S1 = np.asarray(witness['signal_sketch'], dtype=int)
    assert S1.shape == (5,8) and np.isin(S1,[-1,1]).all()
    np.testing.assert_array_equal(S1[2], -S1[3])
    np.testing.assert_array_equal(S1.T @ np.array([0,0,1,1,0]), np.zeros(8))

    cert = pd.read_csv(result/INPUTS[2]).query('s == 16').copy()
    assert len(cert) == 15 and cert.evaluable.all()
    assert not cert.duplicated(['estimator','metric']).any()
    assert (cert.ci_low <= cert.point).all() and (cert.point <= cert.ci_high).all()
    rates = primary(pd.read_csv(result/INPUTS[3]))
    rates = rates[rates.estimator.isin(ESTIMATORS)]
    cat = primary(pd.read_csv(result/INPUTS[4]))
    cat = cat[(cat.fraction == .05) & cat.estimator.isin(ESTIMATORS)]
    assert len(rates) == len(cat) == 3
    for _, row in rates.iterrows():
        p = cert[(cert.estimator == row.estimator) & (cert.metric == 'false_safe')].point.item()
        np.testing.assert_allclose(p,row.false_safe,atol=1e-12)
    for _, row in cat.iterrows():
        p = cert[(cert.estimator == row.estimator) & (cert.metric == 'catastrophic_detection')].point.item()
        np.testing.assert_allclose(p,row.detection,atol=1e-12)

    budget = primary(pd.read_csv(result/INPUTS[5]))
    budget = budget[budget.estimator == 'sample_variance'].copy()
    assert len(budget) == 6 and not budget.metric.duplicated().any()
    paths = pd.read_parquet(result/INPUTS[6], filters=[('eta','=',1e-6),('budget','=',160),
        ('s','=',16),('pair','=','primary'),('estimator','=','sample_variance')])
    paths = primary(paths)
    assert len(paths) == 600 and not paths.duplicated(['step_rank','basis_trial']).any()
    assert paths.groupby('step_rank').size().to_dict() == {5:200,15:200,30:200}
    assert paths.accounting_feasible.all() and (paths.pathwise_net_ratio > 0).all()
    assert (paths.committed_cost+paths.s+paths.ell_paid == paths.budget).all()
    assert np.isfinite(paths.pathwise_net_ratio).all()
    np.testing.assert_allclose(paths.pathwise_net_ratio,
        paths.risk_paid_selected_mean/paths.risk_original_baseline,rtol=1e-12)
    for metric, numerator in [('selected_mean_ratio','risk_paid_selected_mean'),
                              ('paid_oracle_mean_ratio','risk_paid_oracle'),
                              ('paid_baseline_mean_ratio','risk_paid_baseline')]:
        by_rank = paths.groupby('step_rank')[[numerator,'risk_original_baseline']].mean()
        expected = (by_rank[numerator]/by_rank.risk_original_baseline).mean()
        np.testing.assert_allclose(budget.loc[budget.metric == metric,'point'].item(),expected,rtol=1e-12)
    harmed = float((paths.pathwise_net_ratio > 1).mean())
    np.testing.assert_allclose(harmed,budget.loc[budget.metric == 'fraction_paths_harmed','point'].item())
    return dict(coords=coords,bad=bad,typical=typical,S1=S1,cert=cert,rates=rates,cat=cat,
                budget=budget,paths=paths,harmed=harmed)


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
        'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'axes.edgecolor':GREY,'axes.labelcolor':INK,'text.color':INK,
        'xtick.color':INK,'ytick.color':INK,'svg.fonttype':'none','svg.hashsalt':'urop-20260916',
        'figure.facecolor':'white','axes.facecolor':'white'})


def header(fig,title,subtitle):
    fig.text(.035,.963,title,fontsize=11,weight='bold',ha='left',va='top')
    fig.text(.035,.897,subtitle,fontsize=7.5,ha='left',va='top',color=GREY)


def save(fig,out,name):
    fig.canvas.draw()
    # Check outer-canvas clipping before export; overlaps also receive visual QA.
    renderer = fig.canvas.get_renderer()
    labels = list(fig.texts)
    for axis in fig.axes:
        labels.extend([axis.title, axis._left_title, axis.xaxis.label, axis.yaxis.label])
    for label in labels:
        if label.get_visible() and label.get_text():
            box = label.get_window_extent(renderer)
            assert (box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width
                    and box.y1 <= fig.bbox.height), f'Clipped label: {label.get_text()}'
    fig.savefig(out/f'{name}.png',dpi=300,facecolor='white')
    fig.savefig(out/f'{name}.svg',metadata={'Date':None},facecolor='white')
    plt.close(fig)


def figure1(data,out):
    fig = plt.figure(figsize=(7.2,3.6))
    grid = fig.add_gridspec(1,3,left=.08,right=.98,bottom=.24,top=.74,
                            wspace=.55,width_ratios=[1.25,1,.9])
    ax,bx,cx = [fig.add_subplot(grid[0,i]) for i in range(3)]
    header(fig,'Coordinate pilot capture and residual risk',
           'Rank-five step matrix • d = 100, m = 60, tail = 0.001 • original 10 coordinate paths')
    for method,color,marker,label in [('hpp',BLUE,'o','Standard'),('gated',ORANGE,'D','Gated')]:
        group=data['coords'].query('method == @method')
        ax.scatter(group.trial,group.risk_rademacher,s=20,edgecolors=color,marker=marker,
                   facecolors='none' if method=='hpp' else color,label=label,zorder=3)
    ax.set(yscale='log',ylim=(3e-8,.2),xticks=range(10),xlabel='Original trial index',
           ylabel='Conditional Rademacher risk (log)')
    ax.set_title('a  All coordinate paths',loc='left',weight='bold')
    ax.legend(loc='upper left',handletextpad=.3,borderpad=0)
    ax.annotate('Missed mode',xy=(7,data['bad'].risk_rademacher),xytext=(4,4e-4),
                arrowprops={'arrowstyle':'->','color':INK,'lw':.7},fontsize=7,ha='center')
    ax.grid(axis='y',color='#ECEEEF',lw=.5)
    ritz_rows=[]
    for row,color,marker,linestyle,label in [
        (data['typical'],GREY,'o','--',f'Reference trial {int(data["typical"].trial)}'),
        (data['bad'],ORANGE,'D','-',f'Failure trial {int(data["bad"].trial)}')]:
        vals=np.asarray(json.loads(row.ritz_values))
        bx.plot(np.arange(1,9),vals,color=color,marker=marker,markersize=3,lw=1,
                linestyle=linestyle,label=label)
        ritz_rows.extend(dict(trial=int(row.trial),ritz_index=i+1,ritz_value=float(v),
                              selection=label,q=int(row.q),r=int(row.r)) for i,v in enumerate(vals))
    bx.set(yscale='log',ylim=(.0005,2),xticks=range(1,9),xlabel='Ordered Ritz index',ylabel='Ritz value (log)')
    bx.set_title('b  Pilot Ritz spectra',loc='left',weight='bold')
    bx.legend(loc='center left',fontsize=6.5,handlelength=1.6,borderpad=0)
    cx.imshow(data['S1'],cmap=ListedColormap(['#E3E6E8','#FFFFFF']),vmin=-1,vmax=1,aspect='auto')
    for i in range(5):
        for j in range(8):
            cx.text(j,i,'+' if data['S1'][i,j]>0 else '−',ha='center',va='center',fontsize=8)
    cx.add_patch(Rectangle((-.5,1.5),8,2,fill=False,ec=ORANGE,lw=1.4))
    cx.set(xticks=range(8),xticklabels=range(1,9),yticks=range(5),yticklabels=range(1,6),
           xlabel='Sketch column',ylabel='Signal row')
    cx.set_title('c  Signal sketch',loc='left',weight='bold')
    cx.tick_params(length=0)
    cx.text(.5,-.29,'Rows 3 and 4 are opposites',transform=cx.transAxes,ha='center',fontsize=7)
    fig.text(.035,.085,'Full accepted rank: 8 on every gated path. Failure signal rank: 4, not 5.',fontsize=7.5)
    fig.text(.035,.035,'Gated: q = r = 8, ℓ = 44. Standard: q = r = 20, ℓ = 20. Points are exact conditional risks.',fontsize=7,color=GREY)
    pd.DataFrame(ritz_rows).to_csv(out/'figure1_ritz_source.csv',index=False)
    pd.DataFrame(data['S1'],columns=[f'column_{i}' for i in range(1,9)]).rename_axis('signal_row_zero_based').to_csv(out/'figure1_signal_source.csv')
    data['coords'].to_csv(out/'figure1_path_source.csv',index=False)
    save(fig,out,'figure1_capture_failure')


def figure2(data,out):
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.2,3.4),sharey=True)
    fig.subplots_adjust(left=.19,right=.97,bottom=.23,top=.74,wspace=.27)
    header(fig,'Empirical selective-risk metrics',
           'Phase 1A • s = 16, m = 160, tail = 10⁻⁶ • candidate q = r* + 1 versus baseline q = r*')
    names=['Sample variance','MoM: 1 pair/block','MoM: 2 pairs/block']
    for i,est in enumerate(ESTIMATORS):
        color=BLUE if i==0 else ORANGE
        marker=['o','s','D'][i]
        for axis,metric in [(ax,'false_safe'),(bx,'catastrophic_detection')]:
            row=data['cert'].query('estimator == @est and metric == @metric').iloc[0]
            axis.errorbar(100*row.point,2-i,xerr=[[100*(row.point-row.ci_low)],
                [100*(row.ci_high-row.point)]],fmt=marker,color=color,ms=5,capsize=2,lw=1,
                mfc='white' if i==1 else color)
    ax.set(xscale='log',xlim=(.001,12),xticks=[.001,.01,.1,1,10],
           xticklabels=['0.001','0.01','0.1','1','10'],yticks=[2,1,0],yticklabels=names,
           ylim=(-.5,2.8),xlabel='Accepted among truly worse paths (%)')
    bx.set(xlim=(0,100),xticks=[0,25,50,75,100],xlabel='Eligible catastrophic paths accepted (%)')
    ax.set_title('a  False-safe acceptance',loc='left',weight='bold')
    bx.set_title('b  Catastrophic detection',loc='left',weight='bold')
    for axis,threshold,label in [(ax,.5,'0.5% ceiling'),(bx,75,'75% target')]:
        axis.axvline(threshold,color=GREY,ls='--',lw=.8,zorder=0)
        axis.text(threshold,2.58,label,fontsize=7,color=GREY,ha='center')
        axis.grid(axis='y',color='#ECEEEF',lw=.5)
    fig.text(.035,.09,'Points: equal-rank path-averaged rates. Bars: original conditional 95% percentile intervals.',fontsize=7.2)
    fig.text(.035,.04,'Two selected gate criteria, not the full gate. The empirical guard is not a confidence certificate.',fontsize=7,color=GREY)
    data['cert'].to_csv(out/'figure2_bootstrap_source.csv',index=False)
    data['rates'].to_csv(out/'figure2_eligibility_source.csv',index=False)
    data['cat'].to_csv(out/'figure2_catastrophic_source.csv',index=False)
    save(fig,out,'figure2_empirical_signal')


def figure3(data,out):
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.2,3.5))
    fig.subplots_adjust(left=.19,right=.97,bottom=.24,top=.74,wspace=.36)
    header(fig,'Certification cost and pathwise net effects',
           'Phase 1B • sample variance, s = 16, m = 160, tail = 10⁻⁶ • 600 frozen paths')
    for y,metric,color,marker in [(2,'paid_baseline_mean_ratio',GREY,'s'),
        (1,'paid_oracle_mean_ratio',BLUE,'o'),(0,'selected_mean_ratio',ORANGE,'D')]:
        row=data['budget'].query('metric == @metric').iloc[0]
        ax.errorbar(row.point,y,xerr=[[row.point-row.ci_low],[row.ci_high-row.point]],
                     fmt=marker,color=color,ms=5,capsize=2,lw=1)
        ax.text(row.point,y+.20,f'{row.point:.4f}',ha='center',fontsize=7)
    ax.set(xscale='log',xlim=(.008,2.8),xticks=[.01,.1,1],xticklabels=['0.01','0.1','1'],
           yticks=[2,1,0],yticklabels=['Paid fallback','Paid oracle','Empirical selection'],
           ylim=(-.5,2.7),xlabel='Equal-rank mean of mean-risk ratios')
    ax.set_title('a  Aggregate risk comparison',loc='left',weight='bold')
    ax.axvline(1,color=GREY,lw=.8,ls='--')
    ax.grid(axis='y',color='#ECEEEF',lw=.5)
    ratio=np.sort(data['paths'].pathwise_net_ratio.to_numpy())
    bx.step(np.r_[ratio[0],ratio],np.r_[0,100*np.arange(1,len(ratio)+1)/len(ratio)],
            where='post',color=ORANGE,lw=1.2)
    bx.axvline(1,color=GREY,lw=.8,ls='--')
    bx.set(xscale='log',xlim=(5e-8,4),ylim=(0,103),xticks=[1e-7,1e-5,1e-3,.1,1],
           xlabel='Pathwise selected/original risk ratio (log)',ylabel='Cumulative fraction of paths (%)')
    bx.set_title('b  All pathwise net effects',loc='left',weight='bold')
    bx.text(.04,.78,f'{100*data["harmed"]:.2f}% of paths harmed',transform=bx.transAxes,fontsize=8)
    bx.grid(axis='y',color='#ECEEEF',lw=.5)
    fig.text(.035,.095,'Ratio 1: the unstarted adjacent baseline. The two panels show different estimands.',fontsize=7.5)
    fig.text(.035,.042,'a: original shared-bootstrap 95% intervals. b: empirical distribution; no confidence band. Costs are charged.',fontsize=7,color=GREY)
    data['budget'].to_csv(out/'figure3_bootstrap_source.csv',index=False)
    data['paths'].to_csv(out/'figure3_path_source.csv',index=False)
    save(fig,out,'figure3_certification_cost')


def build(output_dir):
    out=Path(output_dir)
    if out.exists():
        raise FileExistsError('Choose a new output directory; no historical overwrite.')
    protected={p:sha(ROOT/'results'/p) for p in INPUTS}
    data=load_reviewed_data()
    out.mkdir(parents=True)
    style()
    figure1(data,out); figure2(data,out); figure3(data,out)
    assert protected == {p:sha(ROOT/'results'/p) for p in INPUTS}
    for path in out.glob('*.svg'):
        assert '<text ' in path.read_text(), 'SVG labels must remain editable.'
    manifest=dict(matrix_vector_queries=0,new_bootstrap_replicates=0,
        backend='Python/matplotlib',matplotlib=matplotlib.__version__,numpy=np.__version__,
        pandas=pd.__version__,source_sha256=protected,script_sha256=sha(__file__),
        figure1_paths=20,figure1_failure_trial=int(data['bad'].trial),
        figure1_reference_trial=int(data['typical'].trial),
        figure2_estimators=ESTIMATORS,figure2_intervals='Frozen 10000-replicate percentile intervals',
        figure3_paths=600,fraction_paths_harmed=data['harmed'],
        output_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('source_sha256','output_sha256')},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    build(parser.parse_args().output_dir)
