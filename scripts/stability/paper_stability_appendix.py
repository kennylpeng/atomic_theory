"""Figure 6: exclusive signed-cosine persistence across all larger widths."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import csv
import json
from pathlib import Path
import numpy as np
from stability_plotting import plot_stability_panels

ROOT=_paper_path(__file__).resolve().parents[2]
STRICT=ROOT/'full_experiments/results/matches/t1_0.7_t2_0.7'
OUT=ROOT/'full_experiments/results/stability_appendix_0.7'
WIDTHS=[512,1024,2048,4096,8192,16384,32768,65536,131072]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'witnesses').mkdir(exist_ok=True)
    rows=[]
    previous=list(csv.DictReader((STRICT/'all_larger_dictionary_match_counts.csv').open()))
    for model in ['gemini','nemotron']:
        for i,width in enumerate(WIDTHS[:-1]):
            active=None;destinations={}
            for larger in WIDTHS[i+1:]:
                d=json.loads((STRICT/'pairs'/f'{model}_sae_m{width}_to_m{larger}.json').read_text())
                assert d['model']==model and d['representation']=='sae'
                assert d['left_width']==width and d['right_width']==larger
                assert d['t1']==d['t2']==.7
                u,v=d['matched_u'],d['matched_v']
                assert len(u)==len(set(u))==len(v)==len(set(v))==d['matches']
                assert all(0<=x<width for x in u) and all(0<=x<larger for x in v)
                current=set(u);active=current if active is None else active & current
                destinations[larger]=dict(zip(u,v))
            selected=np.array(sorted(active),dtype=int)
            np.savez_compressed(_paper_location(OUT/'witnesses'/f'{model}_strict_{width}.npz'),source=selected,
                **{f'target_{m}':np.array([lookup[int(u)] for u in selected],dtype=int) for m,lookup in destinations.items()})
            expected=[r for r in previous if r['model']==model and r['representation']=='sae' and int(r['left_width'])==width]
            assert len(expected)==1 and len(selected)==int(expected[0]['matches'])
            strict_count=len(selected)
            rows.append(dict(model=model,width=width,criterion='strict_all_larger',
                             count=strict_count,proportion=strict_count/width))
    with (OUT/'counts.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plot_stability_panels(rows,OUT/'strict_all_larger',persistent=True)
    (OUT/'metadata.json').write_text(json.dumps(dict(
        strict_all_larger=dict(similarity='signed cosine > 0.7',exclusivity='no other partner with signed cosine > 0.7 at either endpoint',aggregation='intersection of exclusive source feature IDs across every larger trained width',source=str(STRICT)),
        checks='All 16 exclusive intersections reproduce reference summaries'),indent=2)+'\n')
    for r in rows:
        if r['width'] in [4096,65536]:print(r)
    print('Validated 16 exclusive intersections; wrote the Figure 6 panels.')

if __name__=='__main__':main()
