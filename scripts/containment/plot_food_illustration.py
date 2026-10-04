"""Simple medium-sized illustration of selected food feature hierarchies."""
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
groups=[
 (59195,'Cheese',[(21483,'Cheddar'),(29251,'Mozzarella / related cheeses'),(101286,'Blue cheeses'),(20034,'Gruyère / Emmental'),(125850,'Brie / Camembert')]),
 (36255,'Potatoes',[(98628,'Potato cultivars'),(94830,'Mashed potatoes'),(13508,'Baked potatoes'),(26485,'Hash browns / rösti'),(38138,'Tater tots')]),
 (54683,'Fast food',[(44349,'Burger King'),(49792,"Wendy’s"),(80927,'Jack in the Box'),(119916,"Hardee’s"),(13640,"McDonald’s burgers")]),
 (112985,'Rice dishes',[(28414,'Biryani'),(11153,'Nasi goreng'),(122816,'Fried rice')]),
 (3953,'Sausages',[(22069,'Chorizo'),(48955,'Salami'),(110894,'German sausages'),(109777,'Hot dogs')]),
 (50195,'Nutritional information',[(37858,'Fast-food calorie questions'),(38187,'Calorie-breakdown template'),(23538,'Percent Daily Value text'),(120312,'Vitamins / minerals')]),
]
labels={10077:'Food'};edges=[]
for p,name,cs in groups:
 labels[p]=name;labels.update(dict(cs));edges.append((10077,p));edges.extend((p,c) for c,_ in cs)
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
pd.DataFrame(rows).to_csv(O/'food_illustration_edges.csv',index=False)
extra=d[d.parent_feature.isin(labels)&d.child_feature.isin(labels)]
assert set(zip(extra.parent_feature,extra.child_feature))==set(edges),'Selected nodes contain an extra retained edge'
ex={}
for name in ['branching_parent_examples','candidate_child_examples','grandchild_examples','food_illustration_extra_examples','food_new_branch_parent_examples','food_pizza_icecream_child_examples','food_sausage_child_examples','food_all_branching_parent_examples','food_nonwestern_child_examples']:
 ex.update(json.load(open(_paper_location(O/(name+'.json')))))
(O/'food_illustration_selection.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2))
(O/'food_illustration_examples.json').write_text(json.dumps({str(f):dict(label=label,examples=ex[str(f)]) for f,label in labels.items()},ensure_ascii=False,indent=2))
from render_simple_hierarchy import render
render(groups, 'Food', 10077, O/'food_hierarchy_illustration')
(O/'FOOD_ILLUSTRATION.md').write_text("""# Food hierarchy illustration

A selected tree with 33 features and 32 retained edges. Root: Food #10077. Branches: Cheese #59195, Potatoes #36255, Fast food #54683, Rice dishes #112985, Sausages #3953, Nutritional information #50195.

Rice dishes has three selected children: Biryani #28414, Nasi goreng #11153, and Fried rice #122816. The parent also covers other rice dishes, including pilaf, locrio, and arroz con pollo, so it is not labeled as exclusively Asian. Their supports are 279, 290, and 511 corpus rows respectively. All displayed edges and supports were checked against the exact co-activation matrix. No additional retained edges among these selected nodes were omitted. Labels summarize cached examples and are provisional; some clusters mix related concepts. This is not an exhaustive food hierarchy or uniformly a taxonomic tree.

Sampled activating examples: food_illustration_examples.json. Edge statistics: food_illustration_edges.csv. From the source directory, run `python paper.py script --config /path/to/resources.json -- scripts/containment/plot_food_illustration.py` after preparing the required resources.
""")
print(f'Saved food_hierarchy_illustration PNG/SVG/PDF; verified {len(labels)} features and {len(edges)} edges.')
