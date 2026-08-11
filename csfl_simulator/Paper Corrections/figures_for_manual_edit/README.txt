Standalone copies of the two figures that were edited by hand, so they can be
adjusted further without digging them out of the manuscript folder.

fig3_dirichlet_severity.pdf   (Fig. 3 in the paper)
    The upper subplot kept its own x tick labels even though it shares an x
    axis with the panel below. Those labels sat at y 117.8..129.1 while the
    lower subplot's top spine is at y 125.4, so they ran about four points into
    the plot underneath. The eight labels have been removed, along with the two
    minus-sign strokes belonging to the 10^-1 labels. Nothing else was touched:
    both spines, all tick marks and the "+0.1 .. +0.3" annotations are intact.

fig6_scale_fairness.pdf       (Fig. 6 in the paper)
    The legend was anchored flush to the top of the axes, so the top spine and
    its tick marks were drawn straight through the first row ("SCOPE-FD" and
    "cycle-aligned"). The legend has been redrawn 5 pt lower, fully clear of
    the spine, and the spine and ticks it was covering were restored.

Both are the exact files now used by main.tex. The .png next to each is a
400 dpi render for quick reference; the manuscript uses the PDFs.

If you regenerate these from data rather than editing them, the two defects are
now fixed at source in
Scope_FD_Revision_Package/make_revision_figures.py:
  * _sweep()    tick_params(..., which="both", labelbottom=False) so the log
                axis minor labels are suppressed too, which is what let them
                through on the upper panel
  * fig_scale() legend bbox_to_anchor lowered from 0.995 to 0.965
