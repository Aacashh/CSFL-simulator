"""Drop the x tick labels from the upper subplot of the paired sweep figures.

The generator draws both panels with a shared x axis but only suppresses the
label on one of them, so the upper subplot keeps its own tick labels. Those
labels sit at y 117.8..129.1 while the lower subplot's top spine is at y 125.4,
so they run about four points into the plot below. This is the overlap visible
in Fig. 3 and Fig. 4.

Removing only the text is what matplotlib's tick_params(labelbottom=False)
would have done. Line art is explicitly preserved so the lower subplot's top
spine survives; the two minus-sign glyphs are drawn as paths rather than text,
so they are painted over separately inside the inter-axes gap.
"""
import fitz
import os
import sys

SRC = sys.argv[1]
DST = sys.argv[2]

doc = fitz.open(SRC)
page = doc[0]

# The upper axes bottom spine and the lower axes top spine, read off the file.
UPPER_SPINE = 117.4
PANEL_X = [(38.7, 249.2), (299.8, 510.4)]   # the two axes' x extents
ANNOT_TOP = 127.4                           # topmost lower-subplot annotation
LOWER_SPINE = 125.4

targets = []
for b in page.get_text("dict")["blocks"]:
    for l in b.get("lines", []):
        for s in l["spans"]:
            if not s["text"].strip():
                continue
            x0, y0, x1, y1 = s["bbox"]
            # a tick label belonging to the upper axes starts below its spine
            # and above the lower axes' own tick labels (which sit near y=172)
            if not (UPPER_SPINE <= y0 <= LOWER_SPINE + 2 and y0 < 140):
                continue
            # Must sit inside a panel's plotting width. Without this the
            # lower-right subplot's topmost y-tick, which happens to fall in
            # the same horizontal band, gets swept up too.
            xc = 0.5 * (x0 + x1)
            if not any(ax0 < xc < ax1 for ax0, ax1 in PANEL_X):
                continue
            # apply_redactions clears everything intersecting the box, and the
            # lower subplot's value annotations begin at y=127.4 directly under
            # these labels, so the box has to stop short of them.
            targets.append((fitz.Rect(x0 - 0.5, y0 - 0.5, x1 + 0.5,
                                      min(y1 + 0.5, ANNOT_TOP - 0.05)),
                            s["text"]))

print("removing %d upper-axis tick labels:" % len(targets))
for r, t in targets:
    print("   %-6s at [%6.1f %6.1f %6.1f %6.1f]" % (repr(t), r.x0, r.y0, r.x1, r.y1))
    page.add_redact_annot(r)

# Text only. Keep every vector stroke, so the lower subplot's top spine and the
# tick marks are untouched.
page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                      graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                      text=fitz.PDF_REDACT_TEXT_REMOVE)

# The minus signs of the 10^-1 labels are strokes, not glyphs. They live in the
# gap between the two axes, so painting them out cannot touch either spine.
painted = 0
for dr in page.get_drawings():
    r = dr["rect"]
    if (UPPER_SPINE + 1 < r.y0 and r.y1 < LOWER_SPINE - 1
            and r.width < 6 and r.height < 1.5):
        page.draw_rect(fitz.Rect(r.x0 - 1, r.y0 - 1.2, r.x1 + 1, r.y1 + 1.2),
                       color=None, fill=(1, 1, 1), overlay=True)
        painted += 1
print("painted out %d minus-sign strokes in the inter-axes gap" % painted)

doc.save(DST, garbage=4, deflate=True)
print("wrote %s (%.1f KB)" % (DST, os.path.getsize(DST) / 1024.0))
