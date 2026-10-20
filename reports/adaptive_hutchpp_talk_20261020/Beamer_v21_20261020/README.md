# Adaptive Hutch++: Beamer seminar

Ten 16:9 slides for a roughly 12–14 minute research talk, followed by discussion. The presentation connects the earlier investigation of reliable allocation decisions with the recent empirical allocation result. The detailed v10 dictionary and all earlier PowerPoint/PDF versions are preserved separately.

Revision v21 replaces only the right-hand example on slide 8 with the user-approved lower-median non-power held-out case. Title, bullets, takeaway and footnote are unchanged. The figure subtitle and speaker notes explain the mixed comparison. Other slides remain unchanged.

## Edit and compile

Open `main.tex`. With a standard TeX Live or MiKTeX installation containing Beamer, TikZ and Latin Modern, run:

```sh
latexmk -pdf main.tex
```

For Overleaf, upload the complete source ZIP, select `main.tex` as the main document, and use pdfLaTeX. All figure paths are relative. The included vector PDF figures let the deck compile without Python; the matching CSV files preserve the plotted values for replotting.

## Notes and sources

Every frame contains an English `\note{...}` with spoken explanation, assumptions and source guidance. `SPEAKER_NOTES.md` provides the same text in an easy-to-read form. The presentation PDF hides the notes and contains exactly ten pages. `SOURCES.md` records numerical provenance and the important scientific qualifications.

## Presentation constraints

- Slides 1–7 and 9–10 retain their existing element counts. Slide 8 follows the latest specifically requested title, three bullets, subtitle, two panels, takeaway and footnote.
- No overlays, animations, section-divider pages, or extra bibliography slide.
- Equations are native LaTeX. Diagrams use TikZ; all data figures are vector PDFs.
- The recent quarter-budget result is offline, finite-sample evidence. It does not provide a universal guarantee or a completed online selection rule.
- The recorded paid-policy experiment evaluates a sample-variance heuristic. It does not test the conservative confidence bound on the preceding page.

## Replot slide 8

`plot_slide8.py --root /path/to/research-workspace` reads the saved V1 per-q curves, per-case policies, held-out summary, manifest and Phase4 curve CSVs and writes only this project's figure/data copies. It runs no experiments. Compilation uses the included vector PDF and needs no Python. The prior base plotting helper is `.build/revision14/figures.py`; the v17 panel update used inline code in the revision history.
