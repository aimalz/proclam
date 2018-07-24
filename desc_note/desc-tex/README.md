# desc-tex

This repository contains supporting files to aid writing LSST DESC papers in LaTeX. Requests and suggestions should be raised in the issues.

Below:
* What's here
* Using DESC macros (`lsstdesc_macros.sty`)
* Using the DESC standard acknowledgements
* Using the DESC Note class
* Using the DESC bibliography
* How to incorporate the `desc-tex` repo into other projects
* Acknowledgements

## What's here

* [`ack/`](ack/) contains the standard portion of the Acknowledgements sections of DESC Key and Standard papers.
* [`bib/lsstdesc.bib`](bib/) will provide a bibliography of DESC papers, to facilitate citing them
* [`bst/`](bst/) contains bibliography styles for common journals
* [`logos/`](logos/) contains graphics used in the DESC Note class
* [`styles/`](styles/) contains class and style files for common journals, the DESC Note class, and useful macros in `lsstdesc_macros.sty`

## Using DESC macros (`lsstdesc_macros.sty`)

`\usepackage{desc-tex/styles/lsstdesc_macros}`

**TODO**: more documentation

## Using the DESC standard acknowledgements

E.g. `\input{desc-tex/ack/standard}`. Note that these files contain only the "standard wording" part of the acknowledgements; specific acknowledements for funding agencies, software, etc. associated with a given paper should still be included (see [`ack/README.md`](ack/)).

## Using the DESC Note class

See the example in the [styles README](styles/).

## Using the DESC bibliography

`\bibliography{desc-tex/bib/lsstdesc,<other .bib file(s)>}`

Note that, currently, we plan on including only DESC papers in `lsstdesc.bib`.

## How to incorporate the `desc-tex` repo into other projects

### If your paper is not in a Git repo

Clone `desc-tex` as normal: `git clone git@github.com:LSSTDESC/desc-tex.git`

If you later `git init` to turn your project into a repository, it shouldn't be necessary to remove `desc-tex` first. After creating the repo, just run the `submodule add` command as in the next case, and `git` will figure out that `desc-tex` is already present, and simply register it.

### To add `desc-tex` to a paper in a Git repo

Add `desc-tex` as a submodule: `git submodule add git@github.com:LSSTDESC/desc-tex.git`. The `desc-tex` folder now operates as its own independent repository; the parent repository is aware of it and tracks what state `desc-tex` is in, but does not actually version its files. You can interact with `desc-tex` in the usual way when it is the working directory, e.g. to update it: `cd desc-tex; git pull`.

### If you have cloned a repo that includes `desc-tex` as a submodule

If you used the `--recursive` flag when cloning, everything will be set up. Otherwise, you will see an empty `desc-tex/` folder. Run `git submodule update --init`. Thereafter, everything behaves as in the case above.

### Getting `desc-tex` without using Git

Standalone deployment of `desc-tex` is possible by clicking the "Clone or download" button at the top right of this page and selecting "Download ZIP". This can also be automated deployment using the [`deploy_from_github_zip.bash`](LSSTDESC/start_paper/blob/master/deploy_from_github_zip.bash) script, as in
```
bash ./deploy_from_github_zip.bash desc-tex LSSTDESC/desc-tex master
```

This method is useful, for example, for writing papers in Overleaf, since Overleaf does not support git submodules.

## Acknowledgements

Much of the material in `desc-tex` was cruelly appropriated from [`start_paper`](https://github.com/LSSTDESC/start_paper), which was developed by 
* Phil Marshall
* Alex Drlica-Wagner
* Heather Kelly
* Jonathan Sick

The DESC Note class is maintained by Alex Drlica-Wagner and Phil Marshall.

Otherwise, this project is currently the responsibility of the DESC Publications Board:
* Seth Digel (Publication Manager)
* Pierre Astier
* David Kirkby
* Rachel Mandelbaum
* Adam Mantz
* Phil Marshall
* Hiranya Peiris
* Michael Wood-Vasey
