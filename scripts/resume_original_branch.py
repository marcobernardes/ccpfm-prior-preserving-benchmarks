#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json, sys, inspect
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

def loadmod(path):
    spec=importlib.util.spec_from_file_location('branchmod',path)
    m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m); return m

def main():
    p=argparse.ArgumentParser(); p.add_argument('script'); p.add_argument('outdir'); p.add_argument('--workers',type=int,default=16)
    a=p.parse_args(); m=loadmod(a.script); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    args=m.parse_args(['--domains','circle','ellipse','star','annulus','--operators','wls','ccpfm','--hs','0.20','0.16','0.12','0.10','0.08','--trials','5','--workers',str(a.workers),'--outdir',str(out)])
    ops=m.normalize_operators(args.operators)
    tasks=[]
    for domain in args.domains:
      for trial in range(args.trials):
       seed=args.family_seed_base+1000*list(m.DOMAIN_FACTORIES.keys()).index(domain)+trial
       for h in args.hs:
        for op in ops:
         d=dict(domain_name=domain,operator=op,h=float(h),family_seed=int(seed),trial=int(trial),outdir=str(out),alpha=float(args.alpha),min_neighbors=int(args.min_neighbors),max_neighbors=int(args.max_neighbors),boundary_neighbor_boost=int(args.boundary_neighbor_boost),boundary_layer_factor=float(args.boundary_layer_factor),boundary_weight_boost=float(args.boundary_weight_boost),boundary_split_factor=float(args.boundary_split_factor),selective_enrichment_neighbors=int(args.selective_enrichment_neighbors),selective_core_factor=float(args.selective_core_factor),selective_cond_trigger=float(args.selective_cond_trigger),selective_sigma_trigger=float(args.selective_sigma_trigger),selective_distance_factor=float(args.selective_distance_factor),added_edge_penalty=float(args.added_edge_penalty),force_min_flagged_core_nodes=int(getattr(args,'force_min_flagged_core_nodes',0)),force_relaxed_distance_factor=float(getattr(args,'force_relaxed_distance_factor',5.0)),save_points=False,plot_diagnostics=False)
         tasks.append(d)
    rows=[]; keys=set()
    for f in out.glob('*.json'):
      if f.name.startswith('summary_'): continue
      try:
       r=json.loads(f.read_text()); k=(r['domain'],r['operator'],round(float(r['h_target']),12),int(r['family_seed']),int(r['trial']))
       if k not in keys: rows.append(r); keys.add(k)
      except Exception: pass
    pending=[]
    for t in tasks:
      k=(t['domain_name'],t['operator'],round(t['h'],12),t['family_seed'],t['trial'])
      if k not in keys: pending.append(t)
    print(f'existing={len(rows)} pending={len(pending)} total={len(tasks)}')
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
      allowed=set(inspect.signature(m.solve_poisson_case).parameters); futs={ex.submit(m.solve_poisson_case,**{k:v for k,v in t.items() if k in allowed}):t for t in pending}
      for i,f in enumerate(as_completed(futs),1):
       r=f.result(); rows.append(r); print(f'[{i}/{len(pending)}] {r["domain"]} {r["operator"]} h={r["h_target"]}',flush=True)
    rows.sort(key=lambda r:(r['domain'],r['operator'],float(r['h_target']),int(r['family_seed']),int(r['trial'])))
    m.save_csv(rows,out/'summary_cases.csv'); m.save_json(rows,out/'summary_cases.json')
    agg,unst=m.aggregate_rows(rows); m.save_csv(agg,out/'summary_aggregate.csv'); m.save_json(agg,out/'summary_aggregate.json'); m.save_csv(unst,out/'summary_unstable_cases.csv'); m.save_json(unst,out/'summary_unstable_cases.json')
    print('done',len(rows),len(agg))
if __name__=='__main__': main()
