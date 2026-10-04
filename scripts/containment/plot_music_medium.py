"""Simple medium-sized illustration of selected music feature hierarchies."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
from pathlib import Path
import json,textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=(_paper_path(__file__).resolve().parents[2] / "full_experiments/results/gemini_131k_hierarchical_families")
R=_paper_path('/resources/parent_child_dir/gemini_m131072_k128_gt_0p05_rlower_0p7')
groups=[[117052, 'Jazz', [[130651, 'Charlie Parker / bebop'], [86096, 'John Coltrane'], [88107, 'Miles Davis'], [24353, 'Mahavishnu / John McLaughlin'], [129158, 'Marsalis family']]], [25024, 'Hip-hop / rap', [[94737, 'N.W.A'], [45908, 'Eminem'], [1290, 'Snoop Dogg'], [123105, 'Tupac Shakur'], [105791, 'Lil Wayne']]], [22005, 'Country music', [[122748, 'George Jones'], [20266, 'Rascal Flatts'], [421, 'Patty Loveless / Vince Gill'], [22250, 'Hank Williams']]], [10590, 'Rock / metal', [[61236, 'Hard-rock musicians / bands'], [16925, 'Nu / alternative metal'], [123141, 'Power / symphonic metal'], [75431, 'Doom / stoner metal']]], [130808, 'Electronic / dance', [[5560, 'Detroit techno cluster'], [111755, 'Trance mixes / compilations'], [17166, 'Basement Jaxx'], [35810, 'deadmau5']]], [18522, 'The Beatles', [[19564, 'Please Please Me / early Beatles'], [43221, 'Hey Jude'], [94589, 'Let It Be / Get Back'], [67785, 'Abbey Road / late Beatles']]]]
labels={33809:'Music'};edges=[]
for p,name,cs in groups:
 labels[p]=name;labels.update(dict(cs));edges.append((33809,p));edges.extend((p,c) for c,_ in cs)
d=pd.read_csv(R/'immediate_parent_child_pairs.csv');idx=d.set_index(['parent_feature','child_feature'])
s=json.load(open(_paper_location(R/'summary.json')));blocks=[(b,np.load(_paper_location(b['path']),mmap_mode='r')) for b in s['blocks']]
def co(p,c):
 for b,m in blocks:
  if b['row_start']<=p<b['row_stop']:return int(m[p-b['row_start'],c])
 raise ValueError(p)
rows=[]
for p,c in edges:
 r=idx.loc[(p,c)].to_dict();assert co(p,c)==r['cooccurrence_count'] and co(p,p)==r['parent_support'] and co(c,c)==r['child_support']
 rows.append(dict(parent_feature=p,child_feature=c,parent_label=labels[p],child_label=labels[c],**r))
pd.DataFrame(rows).to_csv(O/'music_medium_edges.csv',index=False)
extra=d[d.parent_feature.isin(labels)&d.child_feature.isin(labels)]
assert set(zip(extra.parent_feature,extra.child_feature))==set(edges),'Selected nodes contain an extra retained edge'
ex={}
for name in ['all_music_child_examples','music_indirect_child_examples','music_expanded_examples']:
 ex.update(json.load(open(_paper_location(O/(name+'.json')))))
(O/'music_medium_selection.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2))
(O/'music_medium_examples.json').write_text(json.dumps({str(f):dict(label=label,examples=ex[str(f)]) for f,label in labels.items()},ensure_ascii=False,indent=2))
from render_simple_hierarchy import render
render(groups, 'Music', 33809, O/'music_hierarchy_medium')
print(f'Saved music_hierarchy_medium PNG/SVG/PDF; verified {len(labels)} features and {len(edges)} edges.')
