"""Fit hierarchy tables by font size and row spacing, never by geometric scaling."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
from pathlib import Path
import re
ROOT = _paper_path(__file__).resolve().parents[2] / "full_experiments/results/gemini_131k_hierarchical_families"
START=r'''% Measure at full width; shrink the font only if necessary.
\ifdefined\FMTableBox\else\newsavebox{\FMTableBox}\fi
\ifdefined\FMFontSize\else\newlength{\FMFontSize}\fi
\ifdefined\FMLeading\else\newlength{\FMLeading}\fi
\ifdefined\FMTargetHeight\else\newlength{\FMTargetHeight}\fi
\ifdefined\FMContentHeight\else\newlength{\FMContentHeight}\fi
\ifdefined\FMRowGap\else\newlength{\FMRowGap}\fi
\ifdefined\FMExtraSpace\else\newlength{\FMExtraSpace}\fi
\setlength{\FMFontSize}{7pt}
\setlength{\FMLeading}{8.1pt}
\setlength{\FMTargetHeight}{0.9\textheight}
\setlength{\FMRowGap}{1.3pt}
\def\FMTableBody{%
'''
END=r'''}
\def\FMMeasure{%
  \sbox{\FMTableBox}{\fontsize{\FMFontSize}{\FMLeading}\selectfont\FMTableBody}%
  \setlength{\FMContentHeight}{\dimexpr\ht\FMTableBox+\dp\FMTableBox\relax}%
}
\FMMeasure
\loop\ifdim\FMContentHeight>\FMTargetHeight
  \addtolength{\FMFontSize}{-0.1pt}%
  \setlength{\FMLeading}{1.157\FMFontSize}%
  \FMMeasure
  \ifdim\FMFontSize<3pt
    \PackageError{hierarchy-tables}{Text area too short for this table}{Use a full-width page with more available height.}%
    \setlength{\FMTargetHeight}{\FMContentHeight}%
  \fi
\repeat
% Fill the remaining height with equal extra space after each of 33 rows.
\setlength{\FMExtraSpace}{\dimexpr\FMTargetHeight-\FMContentHeight\relax}
\divide\FMExtraSpace by 33
\addtolength{\FMRowGap}{\FMExtraSpace}
\FMMeasure
\noindent\usebox{\FMTableBox}
'''
def transform(s):
    if '\\def\\FMMeasure' in s:return s
    s=re.sub(r'\\begin\{adjustbox\}\{[^\n]*\}\n',START, s, count=1) if False else s
    a=s.index('\\begin{adjustbox}')
    b=s.index('\n',a)+1
    s=s[:a]+START+s[b:]
    s=s.replace(r'\\[1.3pt]',r'\\[\FMRowGap]')
    s=s.replace(r'\fontsize{5.8}{6.5}',r'\fontsize{0.83\FMFontSize}{0.96\FMFontSize}')
    s=s.replace(r'\end{adjustbox}',END.rstrip())
    s=s.replace('array, adjustbox, xcolor','array, xcolor')
    return s
if __name__=='__main__':
    for folder in ['latex_examples','latex_dropin']:
        for domain in ['food','music']:
            p=ROOT/folder/(domain+'_table.tex')
            s=transform(p.read_text());p.write_text(s)
            assert 'begin{adjustbox}' not in s and s.count(r'\\[\FMRowGap]')==33
            assert s.count('% feature ')==165
            depth=0
            for m in re.finditer(r'(?<!\\)[{}]',s):
                depth+=1 if m.group()=='{' else -1
                assert depth>=0
            assert depth==0
    print('Fitted tables using font size and row spacing; no text scaling.')
