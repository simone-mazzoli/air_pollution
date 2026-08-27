# Air Pollution Report

`main_merged.tex` is the final report source. The current compiled PDF is
`build/main_merged.pdf`.

- `references.bib` contains bibliography entries.
- `natbiger.bst` is the bibliography style used by the report.
- `Figures/generated/` contains report figures built from saved outputs.
- `Figures/UniKonstanz_Logo_Optimum_sRGB.jpg` is used on the title page.
- `Figures/FIGURE_SOURCES.md` explains where the report figures come from.

Build locally from this directory with:

```bash
latexmk -pdf -outdir=build main_merged.tex
```

Routine LaTeX build files in `build/` are not part of the report content. The
PDF `build/main_merged.pdf` is kept so the reviewed report can be opened
directly.
