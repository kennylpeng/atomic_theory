"""Build hierarchical, full-text tables from all available binned samples."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
from pathlib import Path
import json, re, unicodedata, csv, shutil, html
O=(_paper_path(__file__).resolve().parents[2] / "full_experiments/results/gemini_131k_hierarchical_families" / "latex_examples")
S=O.parent
samples=json.loads((S/'food_music_all_binned_examples.json').read_text())
font_source=_paper_path('/resources/workspace/experiments/colors/results/gemini_color_feature_examples_latex/fonts')
(O/'fonts').mkdir(exist_ok=True)
for f in font_source.iterdir():
    shutil.copy2(f,O/'fonts'/f.name)
MAX_DISPLAY_CHARS=120
def excerpt(text):
    text=' '.join(text.split())
    if len(text)<=MAX_DISPLAY_CHARS:
        return text
    prefix=text[:MAX_DISPLAY_CHARS-1]
    if ' ' in prefix:
        prefix=prefix.rsplit(' ',1)[0]
    return prefix.rstrip()+'…'
def normalized(t):
    return ' '.join(unicodedata.normalize('NFKC',t).split()).casefold()
def escape(t):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(mapping.get(c,c) for c in t)
def tex(t):
    # Native multilingual text for XeLaTeX.
    parts=re.split(r'([\u0400-\u052f]+|[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]+(?:[ \t]+[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]+)*|[\u0e00-\u0e7f\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]+)',t)
    out=[]
    for part in parts:
        e=escape(part)
        if re.search(r'[\u0400-\u052f]',part):
            e=r'{\HierarchyCyrillic '+e+'}'
        elif re.search(r'[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]',part):
            e=r'\RL{\HierarchyArabic '+e+'}'
        elif re.search(r'[\u0e00-\u0e7f\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]',part):
            e=r'{\HierarchyFallback '+e+'}'
        out.append(e)
    return ''.join(out)
records=[]; all_nodes={}; pages=[]
for domain,root,selection in [('Food',10077,'food_illustration_selection.json'),('Music',33809,'music_medium_selection.json')]:
    nodes=[(root,domain,0,None)]
    for p,label,children in json.loads((S/selection).read_text()):
        nodes.append((p,label,1,root))
        nodes.extend((c,l,2,p) for c,l in children)
    assert len(nodes)==len(set(n[0] for n in nodes))==33
    lines=[r'\par',r'\begingroup',r'\setlength{\parindent}{0pt}',r'\tiny',r'\setlength{\tabcolsep}{3pt}',r'\renewcommand{\arraystretch}{1.04}',r'\noindent\begin{minipage}{\linewidth}',r'{\large\bfseries '+domain+r'}\par\vspace{3pt}',r'\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.19\linewidth}>{\raggedright\arraybackslash}p{\dimexpr0.81\linewidth-6pt\relax}@{}}',r'\toprule',r'Feature / hierarchy & Five unique activating texts \\',r'\midrule']
    h=['<h1>'+domain+'</h1><table><thead><tr><th>Feature / hierarchy</th><th>Activating text</th><th>Activation</th></tr></thead><tbody>']
    selected=[]
    for fid,label,level,parent in nodes:
        seen=set(); chosen=[]
        for x in sorted(samples[str(fid)],key=lambda x:(-x['activation'],x['global_row'])):
            key=normalized(x['full_text'])
            if not key or key in seen:continue
            seen.add(key);chosen.append(x)
            if len(chosen)==5:break
        assert len(chosen)==5,fid
        selected.append(dict(feature=fid,label=label,level=level,parent=parent,examples=chosen))
        if level<2:lines.append(r'\midrule')
        example_cells=[]
        for i,x in enumerate(chosen):
            # Repeat path metadata in source records; visibly indent features by depth.
            cell=(r'\hspace*{'+str(level*.75)+r'em}'+(r'\textbf{'+tex(label)+'}' if level<2 else tex(label))+r'\newline\hspace*{'+str(level*.75)+r'em}{\scriptsize f'+str(fid)+'}') if i==0 else ''
            text=excerpt(x['full_text'])
            lines.append('% feature '+str(fid)+'; level '+str(level)+'; row '+str(x['global_row'])+'; dataset '+x['dataset'])
            example_cells.append(tex(text))
            records.append(dict(domain=domain,level=level,parent_feature=parent,feature=fid,label=label,rank=i+1,activation=x['activation'],dataset=x['dataset'],global_row=x['global_row'],full_text=x['full_text']))
            hc=('<div style="padding-left:'+str(level*16)+'px"><strong>'+html.escape(label)+'</strong><br><small>f'+str(fid)+'</small></div>') if i==0 else ''
            h.append('<tr><td>'+hc+'</td><td dir="auto">'+html.escape(text)+'</td><td>'+f"{x['activation']:.4f}"+'</td></tr>')
        feature_cell=r'\hspace*{'+str(level*.6)+r'em}'+(r'\textbf{'+tex(label)+'}' if level<2 else tex(label))+r' {\tiny f'+str(fid)+'}'
        lines.append(feature_cell+' & '+r' \enspace\textbullet\enspace '.join(example_cells)+r' \\[1.3pt]')
        assert all(chosen[i]['activation']>=chosen[i+1]['activation'] for i in range(4))
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{minipage}',r'\endgroup'])
    lines=[r'\begin{table}[p]',r'\centering',r'\caption{'+domain+r' feature hierarchy in the Gemini 131K SAE. Each feature is shown with its five highest-activation unique cached text samples; excerpts are truncated to 120 characters. Indentation indicates hierarchy depth.}',r'\label{tab:'+domain.lower()+r'-hierarchy}',*lines,r'\end{table}']
    (O/(domain.lower()+'_table.tex')).write_text('\n'.join(lines)+'\n')
    (O/(domain.lower()+'_examples.json')).write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
    all_nodes[domain]=selected;pages.append(''.join(h)+'</tbody></table>')
with (O/'selected_examples.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
assert len(records)==330
(O/'setup.tex').write_text(r'''% XeLaTeX required. Load after other packages because bidi is loaded last.
\usepackage{fontspec}
\usepackage{graphicx,booktabs,array,adjustbox,xcolor}
\providecommand{\HierarchyAssetPath}{.}
\newfontfamily\HierarchyCyrillic[Path=\HierarchyAssetPath/fonts/,BoldFont=DejaVuSans-Bold.ttf]{DejaVuSans.ttf}
\newfontfamily\HierarchyFallback[Path=\HierarchyAssetPath/fonts/]{DroidSansFallbackFull.ttf}
\newfontfamily\HierarchyArabic[Path=\HierarchyAssetPath/fonts/,Script=Arabic]{DejaVuSans.ttf}
\usepackage{bidi}
''')
(O/'main.tex').write_text(r'''\documentclass[10pt]{article}
\usepackage[a4paper,margin=8mm]{geometry}
\input{setup.tex}
\setlength{\parindent}{0pt}
\setlength{\emergencystretch}{2em}
\begin{document}
\pagestyle{empty}
\input{food_table.tex}
\clearpage
\input{music_table.tex}
\end{document}
''')
compact_pages=[]
for domain,ns in all_nodes.items():
    rows=[]
    for node in ns:
        examples=' <span class="separator">•</span> '.join('<span dir="auto">'+html.escape(excerpt(x['full_text']))+'</span>' for x in node['examples'])
        rows.append('<tr class="level'+str(node['level'])+'"><td><div style="padding-left:'+str(node['level']*6)+'px">'+html.escape(node['label'])+' <small>f'+str(node['feature'])+'</small></div></td><td>'+examples+'</td></tr>')
    compact_pages.append('<section><h1>'+domain+'</h1><table><thead><tr><th>Feature / hierarchy</th><th>Five unique activating texts</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table><p class="note">Highest-ranked available cached samples, not guaranteed corpus-wide maxima. Texts shortened to 120 characters; ellipses mark omissions. Indentation follows root, branch, leaf.</p></section>')
(O/'preview.html').write_text('<!doctype html><meta charset="utf-8"><title>Food and Music: one page each</title><style>@page{size:A4;margin:8mm}body{font:7pt/1.15 Arial,sans-serif;background:#eee;margin:0}section{box-sizing:border-box;width:210mm;min-height:297mm;padding:8mm;margin:15px auto;background:white;break-after:page}h1{font-size:12pt;margin:0 0 3pt}table{border-collapse:collapse;width:100%;table-layout:fixed}th{text-align:left;border-bottom:1pt solid}td{padding:2pt 2pt;vertical-align:top}th:first-child,td:first-child{width:19%}.level0 td,.level1 td{border-top:.5pt solid}.level0 td:first-child,.level1 td:first-child{font-weight:bold}small{font-size:5.8pt;font-weight:normal}.score,.note{color:#555}.note{font-size:6pt}@media print{body{background:white}section{margin:0;padding:0;width:auto;min-height:0}}</style>'+''.join(compact_pages))
print('Generated two 33-feature tables, 330 full-text examples, JSON/CSV provenance, and HTML preview.')
