# Air Pollution Report

`main_merged.tex` is the active working report. The current compiled PDF is
`build/main_merged.pdf`.

- `references.bib` contains bibliography entries.
- `Figures/` contains report figures and image assets used by the report.
- `legacy/` preserves older `.tex` versions.
- `build/` contains generated LaTeX files and is not tracked.
- `main_merged_text_pass.tex` and `build/main_merged_text_pass.pdf` are older
  text-pass files. Keep them unless the team decides they are no longer needed.

Compile locally with:

```bash
cd Air_pollution_report
latexmk -pdf main_merged.tex
```

In VS Code, open `main_merged.tex` and use LaTeX Workshop. The repository settings build on save with `latexmk`, keep SyncTeX enabled, and open the PDF preview in a VS Code tab.
