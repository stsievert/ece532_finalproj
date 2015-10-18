*For our [ECE 532] 2015 final project*

### Project proposal
The project proposal is written in latex (because it includes citations, which
don't work smoothly in the Jupyter notebook). I can compile with

```shell
latexmk Project\ Proposal.tex
```

which does all the commands necessary to build a LaTeX document. Normally, it's
something like

```shell
pdflatex Project\ Proposal.tex
bibtex Project\ Proposal.tex
pdflatex Project\ Proposal.tex
pdflatex Project\ Proposal.tex
pdflatex Project\ Proposal.tex
```

[ECE 532]:http://nowak.ece.wisc.edu/ece532/
