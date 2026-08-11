"""Move Fig. 6's legend clear of the axes top spine.

The legend is drawn at the very top of the axes, so the top spine at y=3.29 and
its tick marks run straight through the first row ("SCOPE-FD", "cycle-aligned").
matplotlib would fix this with a lower bbox_to_anchor; with no run data on hand
the legend is instead erased and redrawn 5pt lower, and the spine and ticks it
was covering are restored.
"""
import fitz
import os
import sys

SRC, DST = sys.argv[1], sys.argv[2]
DY = 5.0                      # how far down the legend moves
SPINE_Y = 3.294               # axes top spine
DARK = (0.06668192893266678,) * 3
BLUE = (0.0, 0.4470588266849518, 0.6980392336845398)
ORANGE = (0.8352941274642944, 0.3686274588108063, 0.0)
GREY = (0.41957733035087585, 0.41963836550712585, 0.41957733035087585)
WHITE = (1.0, 1.0, 1.0)

doc = fitz.open(SRC)
page = doc[0]

# 1. remove the legend labels (text only, so the spine survives)
LABELS = ('SCOPE-FD', 'Random', 'cycle-aligned', 'rolling window')
origins = {}
for b in page.get_text('dict')['blocks']:
    for l in b.get('lines', []):
        for s in l['spans']:
            t = s['text'].strip()
            if t in LABELS and s['bbox'][1] < 20:
                origins[t] = s['origin']
                r = fitz.Rect(s['bbox'])
                page.add_redact_annot(fitz.Rect(r.x0 - 1, r.y0 - 1, r.x1 + 1, r.y1 + 1))
page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                      graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                      text=fitz.PDF_REDACT_TEXT_REMOVE)
print("removed %d legend labels" % len(origins))

# 2. paint out the swatches, which are line art and so survived redaction
PAINT = fitz.Rect(136.5, 1.4, 195.6, 16.6)
page.draw_rect(PAINT, color=None, fill=WHITE, overlay=True)

# 3. put back the spine segment and the ticks that the paint covered
sh = page.new_shape()
sh.draw_line(fitz.Point(PAINT.x0, SPINE_Y), fitz.Point(PAINT.x1, SPINE_Y))
sh.finish(color=DARK, width=0.5)
sh.commit()
for x, ylen in ((150.3567, 2.600), (181.1126, 2.600), (165.7346, 1.400)):
    sh = page.new_shape()
    sh.draw_line(fitz.Point(x, SPINE_Y), fitz.Point(x, SPINE_Y + ylen))
    sh.finish(color=DARK, width=0.5)
    sh.commit()

# 4. redraw the legend DY lower
def line(x0, x1, y, colour):
    s = page.new_shape()
    s.draw_line(fitz.Point(x0, y + DY), fitz.Point(x1, y + DY))
    s.finish(color=colour, width=1.5, lineCap=1)
    s.commit()

def circle(cx, cy, r, fill, edge, w):
    s = page.new_shape()
    s.draw_circle(fitz.Point(cx, cy + DY), r)
    s.finish(color=edge, fill=fill, width=w)
    s.commit()

def diamond(cx, cy, h, edge, fill, w):
    cy += DY
    s = page.new_shape()
    s.draw_polyline([fitz.Point(cx, cy - h), fitz.Point(cx + h, cy),
                     fitz.Point(cx, cy + h), fitz.Point(cx - h, cy),
                     fitz.Point(cx, cy - h)])
    s.finish(color=edge, fill=fill, width=w)
    s.commit()

line(137.331, 147.531, 4.550, BLUE)
circle(142.431, 4.550, 1.700, BLUE, WHITE, 0.70)
line(137.331, 147.531, 13.289, ORANGE)
circle(142.431, 13.289, 1.700, ORANGE, WHITE, 0.70)
circle(192.326, 4.550, 1.700, GREY, WHITE, 0.70)
diamond(192.326, 13.305, 2.192, GREY, WHITE, 0.95)

for label, (ox, oy) in origins.items():
    page.insert_text(fitz.Point(ox, oy + DY), label,
                     fontname="tiro", fontsize=6.8, color=(0, 0, 0))
print("redrew legend %.1f pt lower" % DY)

doc.save(DST, garbage=4, deflate=True)
print("wrote %s (%.1f KB)" % (DST, os.path.getsize(DST) / 1024.0))
