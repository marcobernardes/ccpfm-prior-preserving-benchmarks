from pathlib import Path
import pandas as pd, numpy as np, re, math
src=Path('/mnt/data/CAMWA-D-26-00757_Reviewed.tex').read_text()
root=Path('/mnt/data/final_revision_results')
# Load original families
specs=[
 ('WLS',root/'original/frozen/summary_cases.csv','wls'),
 ('NAMLS-reconstructed-v1',root/'namls/summary_cases.csv','namls'),
 ('CCPFM-frozen',root/'original/frozen/summary_cases.csv','ccpfm'),
 ('CCPFM-livegate',root/'original/livegate/summary_cases.csv','ccpfm'),
 ('CCPFM-sparsecore',root/'original/sparsecore_forced/summary_cases.csv','ccpfm'),
 ('CCPFM-selective',root/'original/sparsecore_selective/summary_cases.csv','ccpfm'),
]
frames={}
for name,path,op in specs:
 d=pd.read_csv(path)
 col='operator'
 d=d[d[col].astype(str).str.lower()==op].copy()
 frames[name]=d

domains=['circle','ellipse','star','annulus']
methods=[x[0] for x in specs]
short={'NAMLS-reconstructed-v1':'NAMLS-r1'}
def median(name,dom,key,h=.08):
 d=frames[name]; q=d[(d.domain==dom)&np.isclose(d.h_target,h)]
 return float(np.nanmedian(q[key]))
def unstable(name,dom,h=.08):
 q=frames[name][(frames[name].domain==dom)&np.isclose(frames[name].h_target,h)]
 vals=q.unstable.astype(str).str.lower().isin(['true','1','yes'])
 return int(vals.sum()),len(q)
def slope(name,dom,key):
 d=frames[name]; g=d[d.domain==dom].groupby('h_target')[key].median().dropna()
 g=g[(g>0)&np.isfinite(g)]
 return float(np.polyfit(np.log(g.index.to_numpy(float)),np.log(g.to_numpy(float)),1)[0]) if len(g)>=2 else float('nan')
# Full finest table
rows=[]
for dom in domains:
 for m in methods:
  u,n=unstable(m,dom)
  rows.append(f"{dom.title()} & {short.get(m,m)} & {median(m,dom,'l2'):.3e} & {median(m,dom,'linf'):.3e} & {median(m,dom,'l2_near_boundary_interior'):.3e} & {median(m,dom,'l2_core_interior'):.3e} & {u}/{n} & {slope(m,dom,'l2'):.2f} \\\\")
finest='''\\begin{table*}[t]
\\centering
\\footnotesize
\\caption{Finest-grid benchmark summary ($h=0.08$) across domains and methods. All NAMLS entries are outputs of \\texttt{NAMLS-reconstructed-v1}.}
\\label{tab:finest_metrics}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{llcccccc}
\\toprule
Domain & Method & Median $L_2$ & Median $L_\\infty$ & Near-boundary $L_2$ & Core $L_2$ & Unstable/Trials & Slope($L_2$) \\\\
\\midrule
'''+"\n".join(rows)+'''\n\\bottomrule
\\end{tabular}%
}
\\vspace{0.35em}
\\begin{minipage}{0.96\\linewidth}
\\footnotesize\\emph{Note.} The four CCPFM/WLS branches use five matched cloud trials; the reconstructed NAMLS comparator uses ten trials, including the same first five seeds. Slopes are empirical finite-range log--log fits, not formal asymptotic orders.
\\end{minipage}
\\end{table*}'''
# Orders table
rows=[]
for dom in domains:
 for m in methods:
  rows.append(f"{dom.title()} & {short.get(m,m)} & {slope(m,dom,'l2'):.2f} & {slope(m,dom,'linf'):.2f} & {slope(m,dom,'l2_near_boundary_interior'):.2f} & {slope(m,dom,'l2_core_interior'):.2f} \\\\")
orders='''\\begin{table*}[t]
\\centering
\\footnotesize
\\caption{Empirical finite-range fitted slopes by domain and method.}
\\label{tab:orders}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{llcccc}
\\toprule
Domain & Method & Slope($L_2$) & Slope($L_\\infty$) & Near-boundary slope & Core slope \\\\
\\midrule
'''+"\n".join(rows)+'''\n\\bottomrule
\\end{tabular}%
}
\\vspace{0.35em}
\\begin{minipage}{0.96\\linewidth}
\\footnotesize\\emph{Note.} Slopes are fitted to aggregate median errors across all tested refinement levels. A missing value indicates that a region metric was unavailable or nonpositive at too many levels for a reliable fit.
\\end{minipage}
\\end{table*}'''
# runtime table
rows=[]
for dom in domains:
 frozen=median('CCPFM-frozen',dom,'total_seconds')
 for m in methods:
  t=median(m,dom,'total_seconds'); ov=100*(t-frozen)/frozen
  rows.append(f"{dom.title()} & {short.get(m,m)} & {t:.3f} & {ov:+.1f} \\\\")
runtime='''\\begin{table*}[t]
\\centering
\\footnotesize
\\caption{Median finest-grid total runtime and overhead relative to the frozen CCPFM baseline.}
\\label{tab:runtime}
\\begin{tabular}{llcc}
\\toprule
Domain & Method & Median total time (s) & Overhead vs. frozen (\\%) \\\\
\\midrule
'''+"\n".join(rows)+'''\n\\bottomrule
\\end{tabular}
\\vspace{0.35em}
\\begin{minipage}{0.96\\linewidth}
\\footnotesize\\emph{Note.} Timings are wall-clock measurements from the corresponding reconstructed benchmark runs and should be interpreted as implementation-specific rather than hardware-independent complexity measures.
\\end{minipage}
\\end{table*}'''
# winner table based raw metrics
cats=[('l2',False),('linf',False),('l2_near_boundary_interior',False),('l2_core_interior',False)]
wrows=[]
for dom in domains:
 winners=[]
 # robustness: lowest rate, tie by order listed
 rates={m:unstable(m,dom)[0]/max(unstable(m,dom)[1],1) for m in methods}
 winners.append(min(rates,key=rates.get))
 for key,_ in cats:
  vals={m:median(m,dom,key) for m in methods}
  winners.append(min(vals,key=vals.get))
 # overall geometric mean normalized error metrics
 scores={}
 mins={k:min(median(m,dom,k) for m in methods) for k,_ in cats}
 for m in methods:
  scores[m]=float(np.exp(np.mean([np.log(median(m,dom,k)/mins[k]) for k,_ in cats]))) * (1+.05*unstable(m,dom)[0])
 overall=min(scores,key=scores.get); winners.append(overall)
 wrows.append(f"{dom.title()} & "+" & ".join(short.get(x,x) for x in winners)+" & no \\\\")
winner='''\\begin{table*}[t]
\\centering
\\footnotesize
\\caption{Raw winner summary at the finest tested spacing.}
\\label{tab:winners}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{lccccccc}
\\toprule
Domain & Robustness & $L_2$ & $L_\\infty$ & Near-boundary & Core & Overall practical & Margin held \\\\
\\midrule
'''+"\n".join(wrows)+'''\n\\bottomrule
\\end{tabular}%
}
\\vspace{0.35em}
\\begin{minipage}{0.96\\linewidth}
\\footnotesize\\emph{Note.} The overall score is the geometric mean of normalized $L_2$, $L_\\infty$, near-boundary, and core errors with a small instability penalty. The table distinguishes the strongest overall comparator from the strongest member of the CCPFM family.
\\end{minipage}
\\end{table*}'''

def replace_table(text,label,new):
    anchor='\\label{'+label+'}'
    pos=text.find(anchor)
    if pos<0: raise RuntimeError(('missing label',label))
    begin=text.rfind('\\begin{table*}[t]',0,pos)
    end=text.find('\\end{table*}',pos)
    if begin<0 or end<0: raise RuntimeError(('table bounds',label,begin,end))
    end += len('\\end{table*}')
    return text[:begin]+new+text[end:]

src=replace_table(src,'tab:finest_metrics',finest)
src=replace_table(src,'tab:orders',orders)
src=replace_table(src,'tab:winners',winner)
src=replace_table(src,'tab:runtime',runtime)
# Title, abstract, highlights, keywords
src=src.replace('Conservative Benchmark Assessment of Prior-Preserving CCPFM Variants for 2D Poisson Problems on Irregular Point Clouds','Prior-Preserving CCPFM on Irregular Point Clouds: Analysis, Poisson Benchmarks, and Extended Diffusion Tests')
abstract='''This paper develops and assesses a prior-preserving constrained correction point-flux method (CCPFM) for elliptic problems on irregular point clouds. The corrected edge coefficients are characterized as the unique weighted projection of a geometric/WLS prior onto an affine polynomial-consistency manifold. The analysis establishes projection well-posedness, prior-distance minimality, a correction bound, local consistency under a weighted higher-moment condition, and conditional global convergence under mesh-independent discrete coercivity. The constant-coefficient Poisson benchmark uses circle, rotated-ellipse, star, and annular domains and compares WLS, the fully rerun \\texttt{NAMLS-reconstructed-v1} comparator, a frozen CCPFM baseline, a live-boundary gate, and sparse deep-core enrichment ablations. The reconstructed NAMLS comparator gives the smallest finest-grid error in the tested Poisson cases, whereas the frozen CCPFM baseline remains the strongest member of the CCPFM family. Additional same-cloud tests consider smooth variable-coefficient diffusion and a discontinuous circular-interface problem. A constrained least-squares generalized finite-difference-type (CLS-GFD-type) comparator is included for the smooth variable-coefficient case. The variable-coefficient results show convergent behavior for all three constructions, with CLS-GFD-type giving the lowest finest-grid error on three domains and CCPFM flux narrowly leading on the star. The interface experiment is a negative but informative result: neither the naive strong-form reference nor the current coefficient-aware CCPFM flux treatment provides reliable monotone convergence, demonstrating that explicit interface transmission treatment is still required. The contribution is therefore a mathematically explicit projection framework together with reproducible positive and negative evidence that separates consistency, prior fidelity, graph stability, coefficient variation, and interface limitations.'''
src=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda m:'\\begin{abstract}\n'+abstract+'\n\\end{abstract}',src,count=1,flags=re.S)
src=src.replace('\\item Boundary gating and sparse enrichment remain geometry-sensitive secondary variants.','\\item Same-cloud variable-coefficient, interface, and CLS-GFD-type comparisons are reported.')
src=src.replace('meshfree method \\sep point cloud \\sep Poisson equation \\sep constrained correction \\sep moving least squares \\sep boundary treatment \\sep sparse enrichment','meshfree method \\sep point cloud \\sep constrained correction \\sep variable-coefficient diffusion \\sep interface problem \\sep generalized finite difference')
# Replace introduction scope paragraphs
old='''The benchmark uses one smooth manufactured Dirichlet Poisson problem on four domains---circle, rotated ellipse, five-point star, and annulus---with fixed irregular cloud families across refinement. WLS and neighbor-adaptive normalized moving least squares (NAMLS) serve as comparators. The diagnostics include global and region-split errors, empirical convergence, patch defects, correction magnitudes, multiplier-system conditioning, runtime, instability incidence, and enrichment activation. The numerical scope remains deliberately limited to a constant-coefficient scalar elliptic problem. Consequently, the conclusions concern the constrained projection mechanism and this benchmark class; they do not establish performance for variable coefficients, interfaces, elasticity, transient equations, or three-dimensional clouds.'''
new='''The verification now has three levels. First, the original smooth Dirichlet Poisson problem is rerun on circle, rotated ellipse, five-point star, and annular domains with fixed irregular cloud families. Second, the same cloud generator is used for smooth variable-coefficient diffusion, comparing a strong-form WLS reference, a CLS-GFD-type constrained least-squares comparator, and a coefficient-aware CCPFM flux construction. Third, a discontinuous circular-interface problem tests whether harmonic edge coefficients and protected interface bands are sufficient without explicit transmission constraints. The diagnostics include global and region-split errors, empirical convergence, conditioning, runtime, instability incidence, and enrichment activation.'''
src=src.replace(old,new)
src=src.replace('The results support a conservative conclusion. The frozen CCPFM baseline remains the strongest overall CCPFM recommendation.','The results support a conservative conclusion. The reconstructed NAMLS comparator is the most accurate method in the constant-coefficient Poisson benchmark, while the frozen CCPFM baseline remains the strongest member of the CCPFM family.')
# Add model problem subsections after Poisson description
needle='This choice avoids the misleading simplicity of a single-mode test while keeping the benchmark fully verifiable.\n'
modeladd=r'''
\subsection{Smooth variable-coefficient diffusion}

The first extension solves
\begin{equation}
-\nabla\cdot\left(k(x,y)\nabla u\right)=f_k,
\qquad
k(x,y)=1+0.30\sin(\pi x)\cos(\pi y),
\label{eq:variable_diffusion}
\end{equation}
with the same exact field in Eq.~\eqref{eq:uexact}. The forcing is evaluated analytically as
\begin{equation}
f_k=-k\Delta u_{\mathrm{exact}}-\nabla k\cdot\nabla u_{\mathrm{exact}}.
\end{equation}
Because $0.7\le k\le1.3$, this case remains uniformly elliptic while changing both first- and second-derivative contributions.

\subsection{Discontinuous circular-interface diffusion}

The interface test uses a circle with interface radius $R=0.52$ and piecewise coefficient $k=10$ inside and $k=1$ outside. The exact radial field is
\begin{equation}
u(r)=\begin{cases}r^2+9R^2,&r\le R,\\10r^2,&r>R,\end{cases}
\label{eq:interface_exact}
\end{equation}
which is continuous at $R$ and satisfies continuity of normal flux because $k_{\mathrm{in}}u'_{\mathrm{in}}=k_{\mathrm{out}}u'_{\mathrm{out}}$. The forcing is $-40$ in both subdomains. This benchmark is intentionally used to test whether coefficient averaging alone can recover an interface solution; it is not assumed that the answer is positive.
'''
src=src.replace(needle,needle+modeladd)
# Add CLS-GFD method before frozen CCPFM
marker='\\subsection{Frozen CCPFM baseline}'
cls=r'''
\subsection{CLS-GFD-type comparator for variable diffusion}

For the smooth variable-coefficient extension, a constrained least-squares generalized finite-difference-type comparator is assembled on the identical clouds. At each interior node, the stencil minimizes a compactly weighted coefficient norm subject to exact action on the normalized quadratic basis. Candidate neighborhoods of 14--34 points are tested and selected by a conditioning, singular-value, and reproduction-defect score. A trace-scaled regularization of $10^{-12}$ is used. This implementation is labeled \texttt{CLS-GFD-type-v1}; it is a transparent implementation of the established constrained least-squares GFD principle rather than a claim of reproducing third-party software \cite{cls_gsp2023}.

\subsection{Coefficient-aware CCPFM flux extension}

For variable diffusion, the geometric edge prior is multiplied by the harmonic mean of the nodal coefficients, and the moment targets are modified to reproduce $-\nabla\cdot(k\nabla p)$ on the local quadratic basis. For the interface test, nodes within $1.5h$ of the interface are protected from smooth-coefficient moment correction. This is a deliberately minimal extension: it tests coefficient-aware flux scaling but does not impose interface jump conditions as explicit constraints.

'''
src=src.replace(marker,cls+marker)
# Insert extended results before scope subsection
ext=pd.read_csv(root/'extended/aggregate_results.csv')
# generate table rows finest smooth h=.10
srows=[]
for dom in domains:
 for m,label in [('wls_strong','WLS strong'),('cls_gfd','CLS-GFD-type'),('ccpfm_flux','CCPFM flux')]:
  q=ext[(ext.problem=='smooth_variable')&(ext.domain==dom)&(ext.method==m)&np.isclose(ext.h,.10)].iloc[0]
  # global fitted slope
  gg=ext[(ext.problem=='smooth_variable')&(ext.domain==dom)&(ext.method==m)].dropna(subset=['median_l2'])
  sl=float(np.polyfit(np.log(gg.h),np.log(gg.median_l2),1)[0])
  srows.append(f"{dom.title()} & {label} & {q.median_l2:.3e} & {q.median_linf:.3e} & {sl:.2f} & {int(q.unstable_count)}/{int(q.trials)} \\\\")
# interface rows all h
irows=[]
for h in [.20,.16,.12,.10]:
 for m,label in [('wls_strong','WLS strong'),('ccpfm_flux','CCPFM flux')]:
  q=ext[(ext.problem=='circular_interface')&(ext.method==m)&np.isclose(ext.h,h)].iloc[0]
  irows.append(f"{h:.2f} & {label} & {q.median_l2:.3e} & {q.median_linf:.3e} & {int(q.unstable_count)}/{int(q.trials)} \\\\")
extsec=r'''
\subsection{Smooth variable-coefficient diffusion and controlled CLS-GFD-type comparison}
\label{sec:variable_results}

Figure~\ref{fig:variableconv} reports same-cloud convergence for the smooth variable-coefficient problem. All three methods reduce the median $L_2$ error under refinement. At $h=0.10$, the CLS-GFD-type comparator gives the smallest median $L_2$ error on the circle, ellipse, and annulus, while CCPFM flux is narrowly lower on the star. The WLS strong-form reference is more variable and includes one coarse star instability. These results show that the constrained projection extends to smooth coefficient variation, but they do not establish universal superiority over a well-designed local constrained least-squares method.

\begin{figure}[t]
\centering
\includegraphics[width=0.95\textwidth]{variable_diffusion_convergence.pdf}
\caption{Median $L_2$ convergence for smooth variable-coefficient diffusion on identical cloud families. The CLS-GFD-type and CCPFM flux methods both converge across all domains; their relative ranking is geometry dependent.}
\label{fig:variableconv}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.95\textwidth]{variable_diffusion_finest.pdf}
\caption{Finest tested smooth variable-coefficient errors at $h=0.10$. The controlled same-cloud comparison includes WLS strong form, \texttt{CLS-GFD-type-v1}, and the coefficient-aware CCPFM flux extension.}
\label{fig:variablefinest}
\end{figure}

\begin{table*}[t]
\centering
\footnotesize
\caption{Smooth variable-coefficient diffusion at $h=0.10$ with fitted $L_2$ slopes over $h\in\{0.20,0.16,0.12,0.10\}$.}
\label{tab:variable_results}
\begin{tabular}{llcccc}
\toprule
Domain & Method & Median $L_2$ & Median $L_\infty$ & Fitted slope & Unstable/Trials \\
\midrule
'''+"\n".join(srows)+r'''
\bottomrule
\end{tabular}
\end{table*}

\subsection{Discontinuous-interface stress test}
\label{sec:interface_results}

The discontinuous circular-interface benchmark gives a materially different outcome. Figure~\ref{fig:interfaceconv} and Table~\ref{tab:interface_results} show that the naive strong-form reference decreases only slowly, while the current harmonic-prior CCPFM flux construction is nonmonotone and develops one unstable finest-grid trial. Thus, harmonic coefficient averaging and a protected interface band are not substitutes for explicit solution- and flux-transmission constraints. This negative result directly limits the scope of the present CCPFM extension and identifies the required next method development.

\begin{figure}[t]
\centering
\includegraphics[width=0.72\textwidth]{interface_diffusion_convergence.pdf}
\caption{Median $L_2$ behavior for the discontinuous circular-interface problem. Neither tested construction provides reliable interface convergence; explicit transmission treatment is required.}
\label{fig:interfaceconv}
\end{figure}

\begin{table}[t]
\centering
\footnotesize
\caption{Discontinuous circular-interface results on the circle.}
\label{tab:interface_results}
\begin{tabular}{llccc}
\toprule
$h$ & Method & Median $L_2$ & Median $L_\infty$ & Unstable/Trials \\
\midrule
'''+"\n".join(irows)+r'''
\bottomrule
\end{tabular}
\end{table}

'''
src=src.replace('\\subsection{Scope of the numerical evidence}',extsec+'\\subsection{Scope of the numerical evidence}')
# Replace scope and conclusion limitations paragraphs
src=src.replace('The reviewer correctly notes that changing the geometry does not change the differential operator. The present computations therefore verify the method only for a smooth, constant-coefficient, scalar Dirichlet Poisson problem in two dimensions. The star and annulus create demanding boundary and topology configurations, but they do not establish robustness for variable diffusion, discontinuous interfaces, vector-valued systems, transient PDEs, or three-dimensional clouds.','The revised computations now include smooth variable-coefficient diffusion and a discontinuous circular-interface stress test in addition to constant-coefficient Poisson. The smooth-coefficient case broadens the verified operator class. The interface case does not validate the current extension; instead, it demonstrates that explicit transmission constraints are necessary. Vector-valued systems, transient PDEs, and three-dimensional clouds remain outside the evidence.')
# remove obsolete next-stage paragraph if exact
src=re.sub(r'A meaningful next verification stage should use the same cloud.*?prevent extrapolation beyond the evidence\.','The next interface-development stage must incorporate explicit solution and normal-flux transmission constraints and compare them with an independently validated interface-capable meshfree method. No such positive interface claim is made here.',src,flags=re.S)
src=src.replace('The numerical evidence supports three conclusions. First, the frozen CCPFM baseline remains the strongest overall CCPFM variant in the tested problem class.','The numerical evidence supports four conclusions. First, the reconstructed NAMLS comparator is the most accurate method in the constant-coefficient Poisson study, while the frozen baseline remains the strongest CCPFM variant.')
src=src.replace('Third, sparse enrichment is mechanically feasible and can activate selectively, but its present effect on practical error is negligible. These results do not support the claim that more adaptivity is uniformly better.','Third, sparse enrichment is mechanically feasible and can activate selectively, but its present effect on practical error is negligible. Fourth, smooth variable coefficients can be handled by both CLS-GFD-type and CCPFM flux constructions, whereas the discontinuous-interface test remains unresolved and exposes the need for explicit transmission treatment. These results do not support the claim that more adaptivity is uniformly better.')
src=src.replace('The current numerical scope remains limited to a smooth constant-coefficient scalar Poisson equation in two dimensions. Extending the evidence to smooth variable-coefficient diffusion and discontinuous interface problems is the immediate priority. Those cases should retain identical clouds and reporting definitions and should add coefficient-aware flux construction, interface transmission diagnostics, and comparison with established meshfree diffusion operators. Vector PDEs and three-dimensional clouds require further analysis and are not implied by the present results.','The numerical scope now covers constant-coefficient Poisson, smooth variable-coefficient diffusion, and a discontinuous-interface stress test in two dimensions. Only the first two classes show reliable refinement behavior. The interface result is explicitly negative. Vector PDEs, transient problems, and three-dimensional clouds require further analysis and are not implied by the present results.')
src=src.replace('The present revision does not claim variable-coefficient or interface results beyond separately identified extension experiments.','The archive additionally contains the completed variable-coefficient and interface case outputs and the transparent \\texttt{CLS-GFD-type-v1} comparator. The interface outputs are retained as negative evidence rather than filtered or tuned away.')
Path('/mnt/data/CAMWA-D-26-00757_Final_Reviewed.tex').write_text(src)
print('written',len(src))
