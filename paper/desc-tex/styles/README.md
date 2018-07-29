## The DESC Note class

The Note class is based on the `aastex61` class, which will be familiar to many authors. A minimal example follows.

Makefile:
```make
export TEXINPUTS:=./desc-tex/styles/:./desc-tex/logos/:

all:
	[ -e .logos ] || { ln -s ./desc-tex/logos .logos >/dev/null 2>&1; }
	latexmk -pdf -g example.tex

clean:
	rm -f *.aux *.fls *.log *Notes.bib *.fdb_latexmk
```

example.tex
```tex
\documentclass[modern]{lsstdescnote}
\usepackage{lsstdesc_macros}
\begin{document}

\title{Minimal Example}
\author{A.\ N.\ Author}
\date{\today}

\begin{abstract}
  Abstraction
\end{abstract}

\maketitle

\section{Introduction}
Hello World.

\end{document}
```
