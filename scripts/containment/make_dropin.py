"""Make standalone XeLaTeX table snippets using installed fonts."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
from pathlib import Path
import re,zipfile
S=(_paper_path(__file__).resolve().parents[2] / "full_experiments/results/gemini_131k_hierarchical_families" / "latex_examples")
O=S.parent/'latex_dropin';O.mkdir(exist_ok=True)
def family(command,names,options=''):
    failure=r'\PackageError{hierarchy-tables}{Missing font: '+names[0]+r'}{Install '+names[0]+r' or change the font list at the start of this table.}'
    body=failure
    for name in reversed(names):
        declaration=chr(92)+'newfontfamily'+chr(92)+command+('['+options+']' if options else '')+'{'+name+'}'
        body=r'\IfFontExistsTF{'+name+'}{'+declaration+'}{'+body+'}'
    return body
fontcode='\n'.join([
 family('FMCyrillic',['DejaVu Sans','FreeSerif','CMU Serif']),
 family('FMCJK',['Noto Sans CJK SC','Noto Serif CJK SC','Droid Sans Fallback']),
 family('FMArabic',['Amiri','Noto Naskh Arabic','DejaVu Sans'],'Script=Arabic'),
 family('FMThai',['Noto Sans Thai','Noto Serif Thai','Garuda','Droid Sans Fallback'],'Script=Thai'),
])
for name in ['food','music']:
    s=(S/(name+'_table.tex')).read_text()
    s=s.replace(r'\begingroup',r'\begingroup'+'\n% Fonts are local to this table; the document font is unchanged.\n'+fontcode,1)
    s=s.replace(r'\HierarchyCyrillic',r'\FMCyrillic').replace(r'\HierarchyArabic',r'\FMArabic')
    s=re.sub(r'\{\\HierarchyFallback ([^{}]*)\}',lambda m:'{'+(r'\FMThai ' if re.search(r'[\u0e00-\u0e7f]',m[1]) else r'\FMCJK ')+m[1]+'}',s)
    s='% Compile with XeLaTeX. Required packages: fontspec, graphicx, booktabs, array, xcolor, bidi.\n'+s
    assert 'HierarchyAssetPath' not in s and 'HierarchyFallback' not in s and 'includegraphics' not in s
    assert s.count('% feature ')==165 and s.count(r'\\[1.3pt]')==33
    (O/(name+'_table.tex')).write_text(s)
(O/'preamble.tex').write_text(r'''% Compile with XeLaTeX. Place after your other package imports.
\usepackage{fontspec,graphicx,booktabs,array,xcolor}
\usepackage{bidi} % Load last.
''')
(O/'main.tex').write_text(r'''\documentclass[10pt]{article}
\usepackage[a4paper,margin=8mm]{geometry}
\usepackage{fontspec,graphicx,booktabs,array,xcolor}
\usepackage{bidi}
\begin{document}
\pagestyle{empty}
\input{food_table.tex}
\clearpage
\input{music_table.tex}
\end{document}
''')
(O/'README.md').write_text('''# Drop-in Food and Music tables (XeLaTeX)

Copy food_table.tex and music_table.tex anywhere in your LaTeX project.
No setup.tex, asset directory, custom path macro, or bundled font files are needed.
Each table defines its own font commands locally and fills the available line
width. The table does not change document margins or global fonts.

Add these lines after your other package imports:

```latex
\\usepackage{fontspec,graphicx,booktabs,array,xcolor}
\\usepackage{bidi} % Load last.
```

Insert the tables, or paste their entire contents into your document:

```latex
\\input{food_table.tex}
\\clearpage
\\input{music_table.tex}
```

main.tex is a complete two-page example. Compile with XeLaTeX. If your document
has two columns, put these on full-width pages; the snippets are ordinary boxed
tabulars, not longtables or floating table environments. Narrower margins in the
example improve readability, but are not a dependency of either table.

Fonts are selected from installed families, with fallback choices listed at the
start of each table. These include DejaVu Sans/FreeSerif/CMU Serif, Noto CJK/Droid
Sans Fallback, Amiri/Noto Naskh Arabic/DejaVu Sans, and Noto Thai/Garuda/Droid Sans
Fallback. A clear error is raised if a script has no available font; in that case,
change the local font list or install one of the named fonts. No font paths are used.

Each table retains all 33 features in hierarchy order, five examples per feature,
and feature IDs. Long examples are capped at 120 characters
with ellipses. These are highest-ranked cached samples, not corpus-wide maxima.
Full texts remain in the original latex_examples JSON/CSV files.

Validation: both tables contain 33 rows and 165 examples, brace balance and asset
independence were checked. Compilation is unverified: no TeX engine is installed.
''')
for name in ['food','music']:
    depth=0
    for m in re.finditer(r'(?<!\\)[{}]',(O/(name+'_table.tex')).read_text()):
        depth+=1 if m.group()=='{' else -1
        assert depth>=0
    assert depth==0
with zipfile.ZipFile(S.parent/'food_music_hierarchy_dropin.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(O.iterdir()):z.write(f,f.name)
print('Built two independent drop-in snippets and minimal preamble; no custom paths or assets.')
