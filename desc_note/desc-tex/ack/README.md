# Standard portion of acknowledgements for DESC papers

These files contain the standard wording part of the acknowledgements that should appear in DESC papers, in LaTeX format.
These can be included with the `\input` command in LaTeX. There are slightly different wordings for Key and Standard papers; a shorter version for Standard papers should be used *only* for publications with strict word-count limits such as letters.

Note that the standard wording is only one part of the acknowledgements. A rough template might be:

```
This paper has undergone internal review in the LSST Dark Energy Science Collaboration. % REQUIRED
% The internal reviewers were \ldots. % Optional but recommended
% Standard papers only: author contribution statements. For examples, see http://blogs.nature.com/nautilus/2007/11/post_12.html
% This work used TBD kindly provided by Not-A-DESC Member and benefitted from comments by Another Non-DESC person.
% Standard papers only: A.B.C. acknowledges support from grant 1234 from ...
\input{desc-tex/ack/standard}
% This work used some telescope which is operated/funded by some agency or consortium or foundation ...
% We acknowledge the use of An-External-Tool-like-NED-or-ADS.
```

When in doubt, refer to the [DESC Author Guide](https://github.com/LSSTDESC/Author_Guide/raw/compiled/Author_Guide.pdf).
