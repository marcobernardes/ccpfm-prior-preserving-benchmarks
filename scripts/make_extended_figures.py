from pathlib import Path
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/mnt/data/final_revision_results/extended')
out=Path('/mnt/data/final_revision_figures'); out.mkdir(exist_ok=True)
df=pd.read_csv(root/'aggregate_results.csv')
labels={'wls_strong':'WLS strong form','cls_gfd':'CLS-GFD-type','ccpfm_flux':'CCPFM flux'}
# Smooth variable coefficient: 2x2 panels L2
fig,axes=plt.subplots(2,2,figsize=(8.4,6.8),sharex=True,sharey=True)
for ax,dom in zip(axes.flat,['circle','ellipse','star','annulus']):
    sub=df[(df.problem=='smooth_variable')&(df.domain==dom)]
    for m in ['wls_strong','cls_gfd','ccpfm_flux']:
        d=sub[sub.method==m].sort_values('h',ascending=False)
        ax.loglog(d.h,d.median_l2,marker='o',label=labels[m])
    ax.set_title(dom.title()); ax.grid(True,which='both',alpha=.25)
    ax.set_xlabel('Target spacing $h$'); ax.set_ylabel('Median $L_2$ error')
handles, labs=axes.flat[0].get_legend_handles_labels()
fig.legend(handles,labs,loc='upper center',ncol=3,fontsize=8)
fig.suptitle('Smooth variable-coefficient diffusion on identical cloud families',y=.98)
fig.tight_layout(rect=[0,0,1,.93]); fig.savefig(out/'variable_diffusion_convergence.pdf',bbox_inches='tight'); plt.close(fig)
# Finest h grouped L2/Linf
fin=df[(df.problem=='smooth_variable') & np.isclose(df.h,0.10)]
fig,axes=plt.subplots(1,2,figsize=(8.4,3.5))
x=np.arange(4); width=.24
for j,m in enumerate(['wls_strong','cls_gfd','ccpfm_flux']):
    vals=[float(fin[(fin.domain==d)&(fin.method==m)].median_l2.iloc[0]) for d in ['circle','ellipse','star','annulus']]
    axes[0].bar(x+(j-1)*width,vals,width,label=labels[m])
    vals2=[float(fin[(fin.domain==d)&(fin.method==m)].median_linf.iloc[0]) for d in ['circle','ellipse','star','annulus']]
    axes[1].bar(x+(j-1)*width,vals2,width,label=labels[m])
for ax,title,ylabel in zip(axes,['$L_2$ error','$L_\\infty$ error'],['Median $L_2$','Median $L_\\infty$']):
    ax.set_xticks(x,['Circle','Ellipse','Star','Annulus'],rotation=20); ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(True,axis='y',alpha=.25)
axes[0].legend(fontsize=7)
fig.suptitle('Smooth variable-coefficient diffusion at $h=0.10$')
fig.tight_layout(); fig.savefig(out/'variable_diffusion_finest.pdf',bbox_inches='tight'); plt.close(fig)
# Interface convergence
fig,ax=plt.subplots(figsize=(5.6,4.0))
sub=df[df.problem=='circular_interface']
for m in ['wls_strong','ccpfm_flux']:
    d=sub[sub.method==m].sort_values('h',ascending=False)
    ax.loglog(d.h,d.median_l2,marker='o',label=labels[m])
ax.set_xlabel('Target spacing $h$'); ax.set_ylabel('Median $L_2$ error'); ax.grid(True,which='both',alpha=.25); ax.legend(fontsize=8)
ax.set_title('Discontinuous circular-interface benchmark')
fig.tight_layout(); fig.savefig(out/'interface_diffusion_convergence.pdf',bbox_inches='tight'); plt.close(fig)
print('wrote figures')
