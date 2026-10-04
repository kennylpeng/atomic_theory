"""Find shared corpus row ranges for paired-model sampling."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import math

from pathlib import Path

from scripts.paired_examples import GEMINI_CACHE, NEMOTRON_CACHE, read_json

ROOT=_paper_path(__file__).resolve().parents[1]

OUT=ROOT/'full_experiments/plots/platonic_features_131k'

SEED=20260906

BANDS=((0,.5),(.5,.65),(.65,.75),(.75,.85),(.85,math.inf))

def shared_ranges():
 gp=read_json(GEMINI_CACHE/'plan.json'); np_=read_json(NEMOTRON_CACHE/'plan.json')
 common={x['relative_shard'] for x in gp['source_shards']} & {x['relative_shard'] for x in np_['source_shards']}
 def get(plan): return [(int(x['global_row_start']),int(x['global_row_stop'])) for x in plan['source_shards'] if x['relative_shard'] in common]
 return get(gp),get(np_)
