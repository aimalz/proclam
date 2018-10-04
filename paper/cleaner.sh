sed -e "s/{\\textbackslash}'\\\\{i\\\\}/\\'{i}/g" main.bib | grep -E -v '^\W*(url|date|abstract|file|keywords|isbn|note)' > main.bib.tmp
mv main.bib.tmp main.bib
