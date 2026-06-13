# ============================================================
#  Makefile — LightRet paper
#  Usage:
#    make          — full build (pdflatex x2 + bibtex)
#    make clean    — remove aux files
#    make view     — open PDF (macOS: open, Linux: xdg-open)
# ============================================================

MAIN  = main
LATEX = pdflatex
BIBTEX= bibtex

.PHONY: all clean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/*.tex references.bib
	$(LATEX) -interaction=nonstopmode $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(LATEX) -interaction=nonstopmode $(MAIN).tex
	$(LATEX) -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot \
	       *.fls *.fdb_latexmk *.synctex.gz

view: $(MAIN).pdf
	@if command -v open >/dev/null 2>&1; then open $(MAIN).pdf; \
	elif command -v xdg-open >/dev/null 2>&1; then xdg-open $(MAIN).pdf; \
	else echo "Open $(MAIN).pdf manually."; fi
