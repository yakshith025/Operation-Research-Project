# Presentation — CP 320 course-project presentation (LaTeX / Beamer)

Slides for the **EV Charging Network Optimization** project
(Yakshith Maddineni, SR 26080 · Kartikeya Gaur).

## Files
- `presentation.tex` — the deck. Title slide is complete; the slides after it
  are a starter scaffold to replace with your content.
- `logos/` — holds `iisc_logo.png` (official IISc seal, shown centred on the title
  slide). A second logo can be added back later — see `logos/README.txt`.

## Build
From this `presentation/` folder, run **twice** (second pass fixes the slide numbers):

```
pdflatex presentation.tex
pdflatex presentation.tex
```

or, in one go:

```
latexmk -pdf presentation.tex
```

Output: `presentation.pdf` (16:9 widescreen).

## Style
- 16:9 aspect ratio, `pdflatex`-safe (no special fonts needed).
- Colour `headblue` RGB(38,70,120) and accent RGB(199,120,20) match the
  written reports, so figures from `../formulation_4/report/figures/`
  drop in cleanly.

## Adding slides
- New topic: `\section{Name}` (it appears in the **Outline** frame).
- New slide: `\begin{frame}{Title} ... \end{frame}`.
- A figure from the reports:
  `\includegraphics[width=\textwidth]{../formulation_4/report/figures/exp_tornado.pdf}`
