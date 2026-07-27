#!/usr/bin/env python3
"""Extended CCPFM diffusion benchmarks.

Adds two problems to the original constant-coefficient Poisson harness:
1. Smooth variable-coefficient diffusion: -div(k grad u)=f.
2. Circular discontinuous-coefficient interface diffusion with continuity of
   solution and normal flux.

The module reuses the original cloud, graph, prior, correction, and matrix
routines. It does not modify the recovered benchmark scripts.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh, spsolve
from scipy.spatial import cKDTree

Array = np.ndarray


def load_base_module(path: str | Path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("ccpfm_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import base benchmark: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Problem:
    name: str
    exact_u: Callable[[Array, Array], Array]
    coefficient: Callable[[Array, Array], Array]
    rhs: Callable[[Array, Array], Array]
    grad_coefficient: Callable[[Array, Array], Tuple[Array, Array]] | None = None
    interface_radius: float | None = None


def smooth_exact_u(x: Array, y: Array) -> Array:
    return np.sin(np.pi*x)*np.sin(np.pi*y) + 0.15*np.cos(2*np.pi*x)*np.sin(3*np.pi*y)


def smooth_grad_u(x: Array, y: Array) -> Tuple[Array, Array]:
    ux = np.pi*np.cos(np.pi*x)*np.sin(np.pi*y) - 0.30*np.pi*np.sin(2*np.pi*x)*np.sin(3*np.pi*y)
    uy = np.pi*np.sin(np.pi*x)*np.cos(np.pi*y) + 0.45*np.pi*np.cos(2*np.pi*x)*np.cos(3*np.pi*y)
    return ux, uy


def smooth_minus_laplacian_u(x: Array, y: Array) -> Array:
    return 2*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y) + 1.95*np.pi**2*np.cos(2*np.pi*x)*np.sin(3*np.pi*y)


def smooth_k(x: Array, y: Array) -> Array:
    return 1.0 + 0.30*np.sin(np.pi*x)*np.cos(np.pi*y)


def smooth_grad_k(x: Array, y: Array) -> Tuple[Array, Array]:
    return (0.30*np.pi*np.cos(np.pi*x)*np.cos(np.pi*y),
            -0.30*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y))


def smooth_rhs(x: Array, y: Array) -> Array:
    ux, uy = smooth_grad_u(x, y)
    kx, ky = smooth_grad_k(x, y)
    return smooth_k(x, y)*smooth_minus_laplacian_u(x, y) - kx*ux - ky*uy


def make_smooth_problem() -> Problem:
    return Problem("smooth_variable", smooth_exact_u, smooth_k, smooth_rhs, smooth_grad_k)


def make_interface_problem(r_interface: float = 0.52, k_inside: float = 10.0,
                           k_outside: float = 1.0) -> Problem:
    # u_in=a r^2+c, u_out=b r^2, with k_in*a=k_out*b and continuity at R.
    a = 1.0
    b = k_inside / k_outside
    c = (b-a)*r_interface**2

    def coeff(x: Array, y: Array) -> Array:
        return np.where(np.hypot(x, y) <= r_interface, k_inside, k_outside)

    def exact(x: Array, y: Array) -> Array:
        r2 = x*x+y*y
        return np.where(np.sqrt(r2) <= r_interface, a*r2+c, b*r2)

    def rhs(x: Array, y: Array) -> Array:
        # -div(k grad u)=-4*k*a in each region; flux matching makes it identical.
        return np.full_like(x, -4.0*k_inside*a, dtype=float)

    return Problem("circular_interface", exact, coeff, rhs, None, r_interface)




def make_problem(problem_key: str) -> Problem:
    """Construct a benchmark problem from a pickle-safe string identifier."""
    aliases = {
        "smooth": "smooth_variable",
        "smooth_variable": "smooth_variable",
        "interface": "circular_interface",
        "circular_interface": "circular_interface",
    }
    try:
        canonical = aliases[problem_key]
    except KeyError as exc:
        raise ValueError(f"Unknown problem identifier: {problem_key!r}") from exc
    if canonical == "smooth_variable":
        return make_smooth_problem()
    return make_interface_problem()

def harmonic_mean(a: Array, b: Array) -> Array:
    return 2.0*a*b/np.maximum(a+b, 1.0e-15)


def polynomial_matrix(dx: Array, dy: Array, scale: float) -> Array:
    xs, ys = dx/scale, dy/scale
    return np.column_stack([np.ones_like(xs), xs, ys, xs*xs, xs*ys, ys*ys])


def local_strong_form_stencil(center: Array, neighbors: Array, k: float,
                              grad_k: Tuple[float, float], rcond: float=1e-12):
    pts = np.vstack([center[None,:], neighbors])
    dx, dy = pts[:,0]-center[0], pts[:,1]-center[1]
    r = np.hypot(dx,dy)
    scale = max(float(np.max(r[1:])), 1e-12)
    q = r/scale
    w = np.exp(-4*q*q)+1e-14
    P = polynomial_matrix(dx,dy,scale)
    sqrtw=np.sqrt(w)
    Pw=P*sqrtw[:,None]
    s=np.linalg.svd(Pw,compute_uv=False)
    cond=float(s[0]/max(s[-1],1e-15))
    pinv=np.linalg.pinv(Pw,rcond=rcond)
    kx,ky=grad_k
    # -div(k grad u) = -k(u_xx+u_yy)-kx u_x-ky u_y.
    functional=np.array([0.0,-kx/scale,-ky/scale,-2*k/scale**2,0.0,-2*k/scale**2])
    coeff=(functional@pinv)*sqrtw
    return np.asarray(coeff),cond,float(s[-1])


def build_wls_strong(base, points: Array, is_boundary: Array, problem: Problem,
                     alpha=2.2,min_neighbors=18,max_neighbors=44):
    n=len(points); tree=cKDTree(points); radii=base.estimate_radii(points,tree)
    rows=[]; cols=[]; data=[]; b=np.zeros(n); conds=[]
    x,y=points[:,0],points[:,1]
    kval=problem.coefficient(x,y); rhs=problem.rhs(x,y); ub=problem.exact_u(x,y)
    if problem.grad_coefficient is None:
        kx=np.zeros(n); ky=np.zeros(n)
    else:
        kx,ky=problem.grad_coefficient(x,y)
    for i in range(n):
        if is_boundary[i]:
            rows.append(i);cols.append(i);data.append(1.0);b[i]=ub[i];continue
        best=None
        for extra in (0,4,8):
            nbrs=base.local_neighbor_indices(i,points,tree,radii,alpha,
                min(min_neighbors+extra,n-1),min(max_neighbors+2*extra,n-1))
            c,cond,smin=local_strong_form_stencil(points[i],points[nbrs],float(kval[i]),(float(kx[i]),float(ky[i])))
            if best is None or cond<best[2]: best=(nbrs,c,cond,smin)
        nbrs,c,cond,smin=best; conds.append(cond)
        for j,v in zip(np.r_[i,nbrs],c): rows.append(i);cols.append(int(j));data.append(float(v))
        b[i]=rhs[i]
    A=coo_matrix((data,(rows,cols)),shape=(n,n)).tocsr()
    return A,b,{"median_local_cond":float(np.median(conds)),"max_local_cond":float(np.max(conds))}


def assemble_variable_constraints(points,is_boundary,adjacency,edge_index,volumes,
                                  problem: Problem, protected_band: Array|None=None):
    rows=[];cols=[];data=[];rhs=[];row_node=[];row_label=[];row=0
    x,y=points[:,0],points[:,1]
    k=problem.coefficient(x,y)
    if problem.grad_coefficient is None:
        kx=np.zeros(len(points));ky=np.zeros(len(points))
    else:
        kx,ky=problem.grad_coefficient(x,y)
    labels=("x","y","x2h","xy","y2h")
    for i in range(len(points)):
        if is_boundary[i] or (protected_band is not None and protected_band[i]): continue
        nbrs=adjacency[i]
        if len(nbrs)<5: continue
        dx=points[nbrs,0]-x[i];dy=points[nbrs,1]-y[i]
        s=max(float(np.median(np.hypot(dx,dy))),1e-12)
        # Constraint convention: sum lambda*(p_j-p_i)=V*div(k grad p).
        targets=[volumes[i]*kx[i]/s, volumes[i]*ky[i]/s,
                 volumes[i]*k[i]/s**2, 0.0, volumes[i]*k[i]/s**2]
        rhs.extend(targets); row_node.extend([i]*5); row_label.extend(labels)
        for j in nbrs:
            e=(i,int(j)) if i<int(j) else (int(j),i); ek=edge_index[e]
            xi=(points[j,0]-x[i])/s; yi=(points[j,1]-y[i])/s
            vals=[xi,yi,0.5*xi*xi,xi*yi,0.5*yi*yi]
            for rr,v in enumerate(vals): rows.append(row+rr);cols.append(ek);data.append(float(v))
        row+=5
    A=coo_matrix((data,(rows,cols)),shape=(row,len(edge_index))).tocsr()
    return A,np.asarray(rhs),{"row_node":np.asarray(row_node),"row_label":np.asarray(row_label,dtype=object)}


def build_ccpfm_flux(base,points,is_boundary,problem:Problem,interface_band_factor=1.5,
                     alpha=2.35,min_neighbors=24,max_neighbors=64,
                     regularization=1e-12):
    n=len(points);tree=cKDTree(points);radii=base.estimate_radii(points,tree)
    volumes=base.compute_node_volumes(radii);distb=base.compute_boundary_distances(points,is_boundary)
    edges,edge_index,adjacency,local_conds,local_sigmins,edge_added,node_flag,node_force,node_added=base.build_edge_list(
        points,is_boundary,radii,alpha,min_neighbors,max_neighbors,4,2.0,
        selective_enrichment_neighbors=0,force_min_flagged_core_nodes=0)
    lam_geom,weights,distances=base.compute_packing_prior(points,radii,edges)
    k=problem.coefficient(points[:,0],points[:,1])
    ei=np.asarray([e[0] for e in edges]);ej=np.asarray([e[1] for e in edges])
    ke=harmonic_mean(k[ei],k[ej])
    lambda0=lam_geom*ke
    protected=np.zeros(n,dtype=bool)
    if problem.interface_radius is not None:
        h=float(np.median(radii[~is_boundary]));r=np.hypot(points[:,0],points[:,1])
        protected=np.abs(r-problem.interface_radius)<=interface_band_factor*h
    Acons,bcons,meta=assemble_variable_constraints(points,is_boundary,adjacency,edge_index,volumes,problem,protected)
    edge_classes=base.classify_ccpfm_edges(edges,is_boundary,distb,float(np.median(radii[~is_boundary])),edge_added,2.0)
    lambdas,cstats=base.solve_ccpfm_correction(lambda0,weights,Acons,bcons,regularization,
        edge_classes=edge_classes,original_prior_anchor=1.0,core_prior_anchor=1.0,
        boundary_band_prior_anchor=1.0,added_prior_anchor=1.0)
    A=base.build_ccpfm_matrix(points,is_boundary,adjacency,edge_index,lambdas,volumes)
    b=np.where(is_boundary,problem.exact_u(points[:,0],points[:,1]),problem.rhs(points[:,0],points[:,1]))
    stats={"median_local_cond":float(np.nanmedian(local_conds[~is_boundary])),
           "max_local_cond":float(np.nanmax(local_conds[~is_boundary])),
           "n_edges":len(edges),"protected_interface_nodes":int(np.sum(protected)),**cstats}
    return A,b,stats,{"volumes":volumes,"radii":radii,"protected":protected,"lambdas":lambdas}


def estimate_condition(A:csr_matrix,is_boundary:Array)->float:
    idx=np.where(~is_boundary)[0]
    if len(idx)<3:return float("nan")
    B=A[idx][:,idx]
    try:
        # Singular-value estimate via B^T B.
        C=(B.T@B).tocsr()
        lmax=float(eigsh(C,k=1,which="LM",return_eigenvectors=False)[0])
        lmin=float(eigsh(C,k=1,which="SM",return_eigenvectors=False,tol=1e-4,maxiter=20000)[0])
        return math.sqrt(max(lmax,0)/max(lmin,1e-30))
    except Exception:return float("nan")


def solve_case(base_path:str|Path,problem_key:str,domain_name:str,method:str,h:float,
               seed:int,trial:int,outdir:str|Path,save_npz=False)->Dict[str,object]:
    # Build the Problem inside the worker.  Passing Problem instances through a
    # ProcessPoolExecutor is unsafe because the interface problem contains local
    # callables, which are not pickleable under Python's multiprocessing protocol.
    problem=make_problem(problem_key)
    base=load_base_module(base_path);domain=base.DOMAIN_FACTORIES[domain_name]()
    t0=time.perf_counter();points,is_boundary=base.generate_point_cloud(domain,h,seed)
    ta=time.perf_counter()
    if method=="wls_strong": A,b,stats=build_wls_strong(base,points,is_boundary,problem);extra={}
    elif method=="ccpfm_flux": A,b,stats,extra=build_ccpfm_flux(base,points,is_boundary,problem)
    else: raise ValueError(method)
    tb=time.perf_counter();u=spsolve(A,b);tc=time.perf_counter()
    ue=problem.exact_u(points[:,0],points[:,1]);err=u-ue
    vols=extra.get("volumes",base.compute_node_volumes(base.estimate_radii(points,cKDTree(points))))
    interior=~is_boundary
    l2=float(np.sqrt(np.sum(vols[interior]*err[interior]**2)/np.sum(vols[interior])))
    linf=float(np.max(np.abs(err[interior])))
    cond=estimate_condition(A,is_boundary)
    row={"problem":problem.name,"domain":domain_name,"method":method,"h":h,"seed":seed,"trial":trial,
         "n_points":len(points),"l2":l2,"linf":linf,"condition_estimate":cond,
         "cloud_seconds":ta-t0,"assembly_seconds":tb-ta,"solve_seconds":tc-tb,"total_seconds":tc-t0,
         "unstable":bool((not np.isfinite(l2)) or linf>100),**stats}
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    tag=f"{problem.name}_{domain_name}_{method}_h{h:.4f}_s{seed}_t{trial}"
    (out/f"{tag}.json").write_text(json.dumps(row,indent=2,default=float),encoding="utf-8")
    if save_npz:np.savez_compressed(out/f"{tag}.npz",points=points,is_boundary=is_boundary,u=u,u_exact=ue,error=err,**extra)
    return row


def write_csv(rows:Sequence[Dict[str,object]],path:str|Path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=sorted({k for r in rows for k in r})
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def aggregate(rows:Sequence[Dict[str,object]])->List[Dict[str,object]]:
    groups={}
    for r in rows:groups.setdefault((r["problem"],r["domain"],r["method"],r["h"]),[]).append(r)
    out=[]
    for key,rs in sorted(groups.items()):
        a=lambda name:np.asarray([float(r[name]) for r in rs])
        out.append({"problem":key[0],"domain":key[1],"method":key[2],"h":key[3],"trials":len(rs),
                    "median_l2":float(np.nanmedian(a("l2"))),"median_linf":float(np.nanmedian(a("linf"))),
                    "median_condition":float(np.nanmedian(a("condition_estimate"))),
                    "median_assembly_seconds":float(np.nanmedian(a("assembly_seconds"))),
                    "median_solve_seconds":float(np.nanmedian(a("solve_seconds"))),
                    "unstable_count":int(sum(bool(r["unstable"]) for r in rs)),
                    "median_n_points":float(np.median(a("n_points")))})
    # empirical orders per problem/domain/method
    for r in out:
        same=sorted([q for q in out if q["problem"]==r["problem"] and q["domain"]==r["domain"] and q["method"]==r["method"]],key=lambda q:q["h"],reverse=True)
        idx=same.index(r)
        if idx==0:r["local_order_l2"]=float("nan")
        else:
            coarse=same[idx-1];r["local_order_l2"]=math.log(coarse["median_l2"]/r["median_l2"])/math.log(coarse["h"]/r["h"])
    return out
