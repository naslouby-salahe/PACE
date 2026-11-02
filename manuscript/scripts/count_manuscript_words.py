"""LaTeX-aware prose counts, with floats and appendix tables separated.
Requires the existing TeXcount executable; writes only the requested JSON.
"""
import argparse, json, re, subprocess, tempfile
from pathlib import Path

SECTIONS = ['introduction','related-work','problem-formulation','methodology',
            'experimental-methodology','results','discussion','limitations','conclusion']
ENV = re.compile(r"\\begin\{(table|figure|longtable)\}.*?\\end\{\1\}", re.S)

def count(source, weights='1,0,0'):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'count.tex'
        path.write_text(source)
        result = subprocess.run(['texcount','-1','-sum='+weights,str(path)],
                                check=True,capture_output=True,text=True)
        return int(result.stdout.strip().splitlines()[-1])

def expand(text, root):
    return re.sub(r"\\input\{([^}]+)\}",
                  lambda m: expand((root/(m[1]+'.tex')).read_text(),root),text)

def analyze(root):
    main=(root/'main.tex').read_text()
    sections={name:expand((root/'sections'/(name+'.tex')).read_text(),root) for name in SECTIONS}
    prose={name:count(ENV.sub('',text)) for name,text in sections.items()}
    appendices='\n'.join(expand((root/'sections'/(n+'.tex')).read_text(),root)
                         for n in ['comparison-inventory','dataset-provenance'])
    abstract=re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",main,re.S)[1]
    full='\n'.join(sections.values())+'\n'+appendices
    return {'method':'TeXcount text words; headings, inline/display math, citations, float/table bodies and bibliography excluded from main prose. Captions counted separately by TeXcount. Table content and bibliography reported as inventories, not included in prose totals.',
            'main_prose_by_section':prose,'main_prose_total':sum(prose.values()),
            'abstract_prose':count(abstract),'appendix_prose':count(ENV.sub('',appendices)),
            'captions':count(full,'0,0,1'),
            'bibliography_entries':len(re.findall(r'^@', (root/'references.bib').read_text(),re.M)),
            'float_and_longtable_environments':len(ENV.findall(full))}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=analyze(args.root)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
