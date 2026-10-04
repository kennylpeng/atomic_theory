"""Verify fixed-rule persistence intersections and every pairwise maximum."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import json
import numpy as np
from scipy.sparse import load_npz
from plot_persistence import OUT, atomic_json
from scripts.stability.persistent_matching import independent_assignments


def verify():
    path=OUT/'validation.json';audit=json.loads(path.read_text());checked=0
    for row in audit['results']:
        directory=OUT/row['group'];width=row['width']
        summary=json.loads((directory/'summary.json').read_text())
        result=next(r for r in summary['rows'] if r['metric']==row['metric'] and r['width']==width)
        prefix='cosine' if row['metric']=='decoder_cosine' else 'pearson'
        graphs=[load_npz(_paper_location(directory/f'{prefix}_{width}_to_{t}.npz')) for t in result['target_widths']]
        selected,assignments=independent_assignments(graphs)
        with np.load(_paper_location(directory/f"{row['metric']}_witness_{width}.npz")) as data:
            np.testing.assert_array_equal(selected,data['source'])
            for target,a in zip(result['target_widths'],assignments):
                np.testing.assert_array_equal(data[f'target_{target}'],a[selected])
        assert len(selected)==row['count']
        checked+=len(graphs)
    audit['independent_verification']=dict(pairwise_maxima_certified=checked,intersections_verified=len(audit['results']))
    atomic_json(path,audit)
    print(f"Verified {len(audit['results'])} independent intersections and {checked} pairwise maximum matchings.")

if __name__=='__main__':verify()
