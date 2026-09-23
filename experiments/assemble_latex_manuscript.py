"""Deterministic editorial assembly; no experiments or PDF generation."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import mistune

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'reports/urop_theorem_proof_report_20260917.md'
OUT = ROOT/'reports/latex/adaptive_hutchpp_20260917'
TEXT = SOURCE.read_text()
AST = mistune.create_markdown(renderer='ast', plugins=['table'])
KINDS = {1:'proposition',2:'theorem',3:'theorem',4:'proposition',5:'lemma',
         6:'proposition',7:'lemma',8:'proposition',9:'proposition',10:'proposition',
         11:'proposition',12:'proposition',13:'theorem',14:'lemma',15:'proposition'}
PARTS = {'I':'preliminaries','II':'allocation','III':'capture','IV':'certification',
         'V':'confidence','VI':'experiments'}
FIGURES = {
 'dependencies.png': ('fig:dependencies','Proof structure and the missing confidence assumption'),
 'workflow.png': ('fig:workflow','Frozen estimation and candidate-first certification workflows'),
 'models.png': ('fig:models','Four risk models and their different assumptions'),
 'figure1_capture_failure.png': ('fig:capture','Full accepted rank with incomplete dominant-subspace capture'),
 'figure2_empirical_signal.png': ('fig:signal','Empirical direct-risk discrimination'),
 'figure3_certification_cost.png': ('fig:cost','Certification cost and rare-path protection'),
}
TABLE_COUNT = 0


def escape(s):
    s = s.replace('—', '---').replace('–', '--').replace('’', "'")
    s = s.replace('“', chr(96)*2).replace('”', "''").replace('·', r'\textperiodcentered{}')
    return ''.join({'&':r'\&','%':r'\%','#':r'\#','_':r'\_',
                    '{':r'\{','}':r'\}','~':r'\textasciitilde{}',
                    '^':r'\textasciicircum{}','\\':r'\textbackslash{}'}.get(c,c) for c in s)


def refs(s):
    s = re.sub(r'\bT4\.1\b', lambda _:r'Corollary~\ref{cor:ideal}', s)
    s = re.sub(r'\bT(\d+)\b', lambda m:KINDS[int(m[1])].title()+r'~\ref{thm:T'+m[1]+'}', s)
    s = re.sub(r'\bE([1-6])\b', lambda m:r'Section~\ref{exp:E'+m[1]+'}', s)
    return re.sub(r'\bPart (VI|IV|III|II|V|I)\b',
                  lambda m:r'Section~\ref{sec:'+PARTS[m[1]]+'}', s)


def section_body(prefix):
    match = re.search(r'^### '+re.escape(prefix)+r'.*\n', TEXT, re.M)
    assert match, prefix
    remaining = TEXT[match.end():]
    end = re.search(r'^(?:### |## |---\s*$)', remaining, re.M)
    return remaining[:end.start() if end else None].strip()


def strip_figures(body):
    return re.sub(r'!\[[^\]]*\]\([^)]+\)\n+\*\*(?:Structure [ABC]|Data figure \d)\.\*\*.*?(?=\n\n|$)',
                  '', body, flags=re.S)


def render(body, table_caption='Recorded quantities'):
    math = {}
    def hold(m, block=False):
        token = f'LATEXMATH{len(math):04d}TOKEN'
        math[token] = (block, m[1].replace('^{,2}', r'^{\,2}'))
        return '\n\n'+token+'\n\n' if block else token
    body = re.sub(r'\$\$(.*?)\$\$', lambda m:hold(m, True), body, flags=re.S)
    body = re.sub(r'(?<!\\)\$(.*?)(?<!\\)\$', hold, body, flags=re.S)
    def walk(n):
        global TABLE_COUNT
        t = n['type']
        children = lambda: ''.join(walk(c) for c in n.get('children', []))
        if t == 'text':
            return refs(escape(n['raw']))
        if t in ('softbreak','linebreak'):
            return '\n'
        if t == 'blank_line':
            return '\n'
        if t in ('strong','emphasis'):
            return ('\\textbf{' if t=='strong' else '\\emph{')+children()+'}'
        if t == 'codespan':
            return r'\texttt{'+escape(n['raw'])+'}'
        if t == 'link':
            url = n['attrs']['url']
            key = ('odonnell2007' if 'odonnell' in url else
                   'cortinovis2022' if 'springer' in url else 'project2026')
            return children()+r'~\cite{'+key+'}'
        if t in ('paragraph','block_text'):
            value = children()
            if value.strip() in math and math[value.strip()][0]:
                return '\\begin{equation}\n'+math[value.strip()][1]+'\n\\end{equation}\n\n'
            return value+'\n\n'
        if t == 'block_quote':
            return '\\begin{quote}\n'+children()+'\\end{quote}\n'
        if t == 'thematic_break':
            return '\n'
        if t == 'heading':
            return '\\paragraph{'+children()+'}\n'
        if t == 'list':
            env = 'enumerate' if n.get('attrs',{}).get('ordered') else 'itemize'
            return '\\begin{'+env+'}\n'+children()+'\\end{'+env+'}\n'
        if t == 'list_item':
            return '\\item '+children()
        if t == 'image':
            raise ValueError('Figures require an explicit caption.')
        if t == 'table':
            TABLE_COUNT += 1
            head = n['children'][0]['children']
            rows = [head]+[r['children'] for r in n['children'][1]['children']]
            count = len(head)
            weights = {2:[.21,.79],3:[.48,.22,.30],5:[.16,.21,.21,.21,.21]}.get(count, [1/count]*count)
            if 'histor' in table_caption.lower():
                weights = [.23,.32,.45]
            spec = '@{}'+''.join(r'>{\raggedright\arraybackslash}p{'+f'{.88*w:.4f}'+r'\linewidth}' for w in weights)+'@{}'
            output = '{\\small\n\\begin{longtable}{'+spec+'}\n'
            output += '\\caption{'+escape(table_caption)+'}\\label{tab:auto'+str(TABLE_COUNT)+'}\\\\\n\\toprule\n'
            for i, row in enumerate(rows):
                line=' & '.join(''.join(walk(c) for c in cell.get('children',[])) for cell in row)+r' \\'+'\n'
                output += line
                if i == 0:
                    output += '\\midrule\n\\endfirsthead\n\\toprule\n'+line+'\\midrule\n\\endhead\n'
            return output+'\\bottomrule\n\\end{longtable}\n}\n'
        raise ValueError(t)
    output = ''.join(walk(n) for n in AST(body))
    for key, (block, val) in math.items():
        if not block:
            output = output.replace(key, '$'+val+'$')
    assert 'LATEXMATH' not in output
    return output


def figure(name):
    pattern = r'!\[[^\]]*\]\(([^)]*/'+re.escape(name)+r')\)\n+\*\*(?:Structure [ABC]|Data figure \d)\.\*\* (.*?)(?=\n\n|$)'
    m = re.search(pattern,TEXT,re.S)
    assert m, name
    source = (SOURCE.parent/m[1]).resolve()
    destination = OUT/'figures'/name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source,destination)
    label, short = FIGURES[name]
    return ('\\begin{figure}[tbp]\n\\centering\n'
            '\\includegraphics[width=\\linewidth]{figures/'+name+'}\n'
            '\\caption['+short+']{'+render(m[2]).strip()+'}\n\\label{'+label+'}\n\\end{figure}\n')


def environment(kind,title,label,statement,proof):
    statement = statement.replace('**Assumptions.**','').replace('**Statement.**','')
    return ('\\begin{'+kind+'}['+escape(title)+']\\label{'+label+'}\n'+
            render(statement)+'\\end{'+kind+'}\n\\begin{proof}\n'+
            render(proof)+'\\end{proof}\n\n')


def theorem(n):
    body = strip_figures(section_body(f'T{n}.'))
    title = re.search(r'^### T'+str(n)+r'\. (.*?) \(', TEXT,re.M)[1]
    if n == 5:
        assumptions, rest = body.split('**Statement A: residual energy.**')
        statement_a, rest = rest.split('**Proof.**',1)
        proof_a, rest = rest.split(r'$\square$',1)
        remark_a, rest = rest.split('**Statement B: square signal block.**')
        statement_b, rest = rest.split('**Proof.**',1)
        proof_b, remark_b = rest.split(r'$\square$',1)
        return (environment('lemma','Structured residual energy','thm:T5',assumptions+statement_a,proof_a)+
                render(remark_a)+
                environment('lemma','Square-sketch graph and principal angles','thm:T5b',
                            assumptions+statement_b,proof_b)+render(remark_b))
    if n == 12:
        prelude, rest = body.split('**Statement.**')
        statement, rest = rest.split('**Proof.**')
        proof, tail = rest.split(r'$\square$',1)
        return render(prelude)+environment('proposition','Truncation--Bernstein lower-tail vacuity',
                                          'thm:T12',statement,proof)+render(tail)
    marker = '**Proof of the moments.**' if n == 7 else '**Proof.**'
    statement, rest = body.split(marker,1)
    proof, tail = rest.split(r'$\square$',1)
    proof = proof.replace('**Proof of the decomposition.**','**Hoeffding decomposition.**')
    return environment(KINDS[n],title,'thm:T'+str(n),statement,proof)+render(tail)


def write(name,body):
    (OUT/'sections'/f'{name}.tex').write_text('% Editorially assembled from the preserved September 17 report.\n'+body)


def main():
    (OUT/'sections').mkdir(parents=True,exist_ok=True)
    intro = render(section_body('1. Assumption'), 'Notation and conditioning')
    write('preliminaries','\\section{Preliminaries and the estimator}\\label{sec:preliminaries}\n'+
          intro+theorem(1)+figure('workflow.png')+r"""
\begin{algorithm}
\caption{Frozen-action trace estimation: accounting template}\label{alg:frozen}
\textbf{Input:} symmetric matrix--vector oracle, budget $m$, a range policy using $q$ products.\\
\textbf{Output:} an unbiased trace estimate, provided $\ell=m-q-r>0$.
\begin{enumerate}
\item Construct $AS$ with $q$ oracle products, retaining any reusable pilot prefix.
\item Compute an orthonormal accepted basis $Q$ of rank $r$ and query/cache $AQ$ with $r$ products.
\item Freeze the action and set $\ell=m-q-r$.
\item Draw $\ell$ fresh independent isotropic probes $g_j$.
\item Form $h_j=g_j-Q(Q^\top g_j)$, query $Ah_j$, and return
$\tr(Q^\top AQ)+\ell^{-1}\sum_j h_j^\top Ah_j$.
\end{enumerate}
\end{algorithm}
"""+theorem(2)+theorem(3))
    ideal = section_body('Corollary T4.1.')
    assumptions, proof = ideal.split('**Statement and proof.**')
    proof, tail = proof.split(r'$\square$',1)
    statement = assumptions+r"""
The ideal successful extension is beneficial if and only if
$$ (m-q-r)\lambda_{r+1}^2>2T(r). $$
In the full-rank ideal model the marginal numerator is nonincreasing on the
feasible adjacent grid. For the step spectrum with $0<\eta<1$, the condition
$2k+2(d-k)\eta^2<m<2d$ makes the feasible knee $k$ the unique minimizer.
"""
    write('allocation','\\section{Rank-aware allocation analysis}\\label{sec:allocation}\n'+
          render(strip_figures(section_body('2. Four objectives')))+figure('models.png')+
          theorem(4)+environment('corollary','Ideal spectral marginal and step knee','cor:ideal',
                                  statement,proof)+render(tail))
    write('capture','\\section{Subspace capture beyond numerical rank}\\label{sec:capture}\n'+
          theorem(5)+theorem(6))
    write('certification','\\section{Realized-risk estimation and conditional safety}\\label{sec:certification}\n'+
          render(section_body('3. The new target'))+r"""
\begin{algorithm}
\caption{Common-probe risk diagnostic for preconstructed actions}\label{alg:diagnostic}
\textbf{Input:} cached $Q_x,AQ_x$ for every action $x$, fixed positive $\ell_x$, and even $s\ge2$.\\
\textbf{Output:} sample-variance risk estimates; not a confidence certificate.
\begin{enumerate}
\item Draw $s$ fresh coordinate-Rademacher probes $g_j$, independent of the fixed actions.
\item Query $Ag_j$ once per probe; share each result across all actions.
\item For each $x,j$, compute $h_{x,j}=g_j-Q_xQ_x^\top g_j$ and
$X_{x,j}=h_{x,j}^\top[Ag_j-AQ_x(Q_x^\top g_j)]$.
\item Return $\widehat\sigma_{x,s}^{\,2}/\ell_x$, where
$\widehat\sigma_{x,s}^{\,2}=(s-1)^{-1}\sum_j(X_{x,j}-\bar X_x)^2$.
\end{enumerate}
\textbf{Cost:} exactly $s$ new oracle queries; cached products were paid during construction.
Certification probes are not reused as final residual probes.
\end{algorithm}
"""+''.join(theorem(n) for n in (7,8,9,10)))
    pair = section_body('Empirical finding E4.')
    split = pair.index('For disjoint pairs')
    pair_empirical, pair_theory = pair[:split],pair[split:]
    write('confidence','\\section{Confidence bounds and their budget limits}\\label{sec:confidence}\n'+
          render(section_body('4. Imported moment'))+theorem(11)+
          '\\subsection{Truncation of the nondegenerate projection}\n'+theorem(12)+
          '\\subsection{Paired differences and structural prior information}\n'+render(pair_theory)+
          theorem(13)+
          '\\paragraph{Supporting moment calculations.} Appendix~\\ref{sec:technical} gives the corrected quadratic-chaos fourth moment and the exact scale-estimation boundary. These refinements do not turn a degree-two bound into a degree-four one.\n')
    experiments='\\section{Experimental validation}\\label{sec:experiments}\n'
    experiments+='The following findings concern the recorded frozen experiments, not universal optimality or safety claims. All detailed protocols and archived outputs are mapped in the accompanying source package~\\cite{project2026}.\n'
    captions = {1:'Mean conditional risks at the knee (budget 160)',2:'Primary sample-variance operating rates (16 certification probes)',
                5:'Recovered rank-five step experiment',6:'Exploratory synthetic-data Hessian results'}
    for n in range(1,7):
        title = re.search(r'^### Empirical finding E'+str(n)+r'\. (.*)',TEXT,re.M)[1]
        title=title.replace('Phase 1A found','Fresh probes contain')
        body = section_body(f'Empirical finding E{n}.') if n!=4 else pair_empirical
        experiments+='\\subsection{'+escape(title)+'}\\label{exp:E'+str(n)+'}\n'
        experiments+=render(strip_figures(body), captions.get(n,'Empirical comparison'))
        if n in (2,3,5):
            experiments+=figure({2:'figure2_empirical_signal.png',3:'figure3_certification_cost.png',5:'figure1_capture_failure.png'}[n])
    write('experiments',experiments)
    write('technical','\\section{Supporting moment and scale calculations}\\label{sec:technical}\n'+theorem(14)+theorem(15))
    record='\\section{Proof map, experimental provenance, and historical scope}\\label{sec:record}\n'
    record+=figure('dependencies.png')
    record+='\\subsection{Development history}\n'+render(section_body('5. Historical'), 'Historical progression and surviving conclusions')
    record+='\\subsection{Reproducibility and figure provenance}\n'+render(section_body('6. Reproducibility'))
    record+=r"""
\subsection{Manuscript preparation and attribution}
This draft uses the organization of Meyer et al.~\cite{meyer2021} as a
writing reference: early contribution statements, numbered mathematical
results, theory before experiments, and supporting appendices.
It does not copy their prose or imply that their optimality theorem applies
to the adaptive policies studied here.

The source package preserves the detailed September 17 theorem--proof
report and an evidence mapping. It contains all fifteen report-local result
blocks, the spectral corollary, and all six empirical findings, reorganized
into publication-style sections. The capture block is split into two lemmas.
The byline is deliberately neutral pending author confirmation; the
document is a research draft, not a submitted or accepted paper.

Three conceptual figures were generated with an AI image tool and reviewed
for logical labels; they are explanatory illustrations, not data. The three
quantitative figures were rendered from validated frozen numerical records.
Their originals are copied without alteration. AI assistance was used for
editorial restructuring and source preparation; mathematical and scientific
claims require author review before submission.
"""
    write('record',record)
    provenance=OUT/'provenance'
    provenance.mkdir(exist_ok=True)
    shutil.copy2(SOURCE,provenance/'source_report.md')
    linked = {}
    for url in re.findall(r'\]\(([^)]+)\)',TEXT):
        if not url.startswith('http'):
            p=(SOURCE.parent/url.split('#')[0]).resolve()
            if p.is_file() and p.suffix=='.md' and p.name not in ('memory.md','UROP_TRACKER.md'):
                linked[p.name]=str(p.relative_to(ROOT))
                shutil.copy2(p,provenance/p.name)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={'source_report':str(SOURCE.relative_to(ROOT)), 'source_sha256':digest(SOURCE),
              'preserved_result_blocks':list(range(1,16)), 'spectral_corollary':True,
              'preserved_empirical_blocks':list(range(1,7)),
              'notation_correction':r'^{,2} -> ^{\,2} (typesetting only)',
              'provenance_mapping':linked,
              'figures':{n:digest(OUT/'figures'/n) for n in FIGURES}}
    (OUT/'assembly_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Assembled 8 section files; retained 15 result blocks, spectral corollary, 6 empirical blocks, 6 figures.')


if __name__=='__main__':
    main()
