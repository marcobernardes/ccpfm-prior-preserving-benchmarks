#!/usr/bin/env python3
from __future__ import annotations
import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
cmd1=[sys.executable,str(root/'run_extended_diffusion_benchmarks.py'),*sys.argv[1:]]
subprocess.run(cmd1,check=True,cwd=root)
subprocess.run([sys.executable,str(root/'make_extended_manuscript_assets.py')],check=True,cwd=root)
