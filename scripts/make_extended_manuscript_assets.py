#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def readcsv(path):
 with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def fv(r,k):
 try:return float(r[k])
 except:return float('nan')
def esc(s):return str(s).replace('_','\\_')
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',default='extended_results/aggregate_results.csv');p.add_argument('--outdir',default='extended_results/manuscript_assets');a=p.parse_args()
 root=Path(__file__).resolve().parent;rows=readcsv((root/a.input).resolve());out=(root/a.outdir).resolve();out.mkdir(parents=True,exist_ok=True)
 # One convergence figure per problem/domain.
 groups=sorted({(r['problem'],r['domain']) for r in rows})
 for prob,dom in groups:
  fig,ax=plt.subplots(figsize=(5.2,4.0))
  for method in ('wls_strong','ccpfm_flux'):
   rr=sorted([r for r in rows if r['problem']==prob and r['domain']==dom and r['method']==method],key=lambda z:fv(z,'h'))
   if rr:ax.loglog([fv(z,'h') for z in rr],[fv(z,'median_l2') for z in rr],marker='o',label=method.replace('_',' '))
  ax.set_xlabel('Target spacing h');ax.set_ylabel('Median L2 error');ax.grid(True,which='both',alpha=.3);ax.legend();fig.tight_layout();fig.savefig(out/f'{prob}_{dom}_l2_convergence.pdf');plt.close(fig)
 # Runtime/conditioning summary.
 for metric,ylabel,name in [('median_assembly_seconds','Median assembly time (s)','extended_runtime.pdf'),('median_condition','Median condition estimate','extended_conditioning.pdf')]:
  finest={}
  for r in rows:
   key=(r['problem'],r['domain'],r['method']);
   if key not in finest or fv(r,'h')<fv(finest[key],'h'):finest[key]=r
  labels=sorted({f"{k[0]}\n{k[1]}" for k in finest});methods=['wls_strong','ccpfm_flux'];x=np.arange(len(labels));w=.36
  fig,ax=plt.subplots(figsize=(max(6,.9*len(labels)),4.2))
  for j,m in enumerate(methods):
   vals=[]
   for lab in labels:
    prob,dom=lab.split('\n');vals.append(fv(finest.get((prob,dom,m),{}),metric))
   ax.bar(x+(j-.5)*w,vals,w,label=m.replace('_',' '))
  ax.set_xticks(x);ax.set_xticklabels(labels,rotation=30,ha='right');ax.set_ylabel(ylabel)
  if metric=='median_condition':ax.set_yscale('log')
  ax.legend();fig.tight_layout();fig.savefig(out/name);plt.close(fig)
 # LaTeX tables from measured data.
 finest=[]
 for key in sorted({(r['problem'],r['domain'],r['method']) for r in rows}):
  rr=[r for r in rows if (r['problem'],r['domain'],r['method'])==key];finest.append(min(rr,key=lambda z:fv(z,'h')))
 lines=['\\begin{table*}[t]','\\centering','\\caption{Extended diffusion benchmarks at the finest tested spacing. All entries are generated from the accompanying scripts.}','\\label{tab:extended_diffusion}','\\begin{tabular}{lllrrrrrr}','\\toprule','Problem & Domain & Method & $h$ & Median $L_2$ & Median $L_\\infty$ & Order & Unstable \\\\','\\midrule']
 for r in finest:lines.append(f"{esc(r['problem'])} & {esc(r['domain'])} & {esc(r['method'])} & {fv(r,'h'):.3f} & {fv(r,'median_l2'):.3e} & {fv(r,'median_linf'):.3e} & {fv(r,'local_order_l2'):.2f} & {int(fv(r,'unstable_count'))} \\\\")
 lines += ['\\bottomrule','\\end{tabular}','\\end{table*}']
 (out/'extended_accuracy_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 lines=['\\begin{table*}[t]','\\centering','\\caption{Conditioning and cost for the extended diffusion benchmarks at the finest tested spacing.}','\\label{tab:extended_cost}','\\begin{tabular}{lllrrrr}','\\toprule','Problem & Domain & Method & $N$ & $\\widehat\\kappa(A)$ & Assembly (s) & Solve (s) \\\\','\\midrule']
 for r in finest:lines.append(f"{esc(r['problem'])} & {esc(r['domain'])} & {esc(r['method'])} & {fv(r,'median_n_points'):.0f} & {fv(r,'median_condition'):.3e} & {fv(r,'median_assembly_seconds'):.3f} & {fv(r,'median_solve_seconds'):.3f} \\\\")
 lines += ['\\bottomrule','\\end{tabular}','\\end{table*}']
 (out/'extended_cost_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(out)
if __name__=='__main__':main()
