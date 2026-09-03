"""Build one editable slide on the generative-AI numerical workflow."""

from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "slides" / "generative_ai_numerical_examples.pptx"
FIGURE = ROOT / "figures" / "Figure3.png"

NAVY = "17324D"
BLUE = "2878A8"
PALE_BLUE = "EAF4F8"
PALE_RED = "FBEDEE"
RED = "D62728"
GRAY = "59636E"
LIGHT_GRAY = "D9E1E6"
WHITE = "FFFFFF"


def rgb(hex_color):
    return RGBColor.from_string(hex_color)


def add_text(slide, x, y, w, h, text, size, color=NAVY, bold=False,
             align=PP_ALIGN.LEFT, font="Arial", margin=0.0):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.margin_left = frame.margin_right = Inches(margin)
    frame.margin_top = frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    return box


def add_box(slide, x, y, w, h, fill, line=LIGHT_GRAY, radius=True):
    kind = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(line)
    shape.line.width = Pt(0.8)
    return shape


def add_arrow(slide, x1, y1, x2, y2):
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    line.line.color.rgb = rgb(BLUE)
    line.line.width = Pt(1.6)
    line.line.end_arrowhead = True
    return line


def crop_figure():
    """Keep the evolution and overturned-limb panels of manuscript Figure 3."""
    image = Image.open(FIGURE)
    cropped = image.crop((0, 0, int(0.685 * image.width), image.height))
    temporary = NamedTemporaryFile(suffix=".png", delete=False)
    cropped.save(temporary.name, dpi=(220, 220))
    return Path(temporary.name)


def build_slide():
    presentation = Presentation()
    presentation.slide_width = Inches(13.333)
    presentation.slide_height = Inches(7.5)
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    background = slide.background.fill
    background.solid()
    background.fore_color.rgb = rgb(WHITE)

    add_text(
        slide, 0.48, 0.24, 12.35, 0.48,
        "Generative AI sped the search; physical judgment chose the model",
        26, bold=True,
    )
    add_text(
        slide, 0.50, 0.78, 12.2, 0.42,
        "Theory fixed the target: a steady Full-Stokes flow carrying surface-deposited isochrones to  J = ∂x/∂x₀|τ < 0",
        14.5, color=GRAY,
    )

    add_box(slide, 0.48, 1.38, 5.05, 1.03, PALE_BLUE, line="B7D6E5")
    add_text(slide, 0.72, 1.49, 4.55, 0.20, "STARTING PROMPT", 10, color=BLUE, bold=True)
    add_text(
        slide, 0.72, 1.70, 4.55, 0.63,
        "“Take the theory, run ice-flow models in Firedrake,\nand adjust the boundary conditions\nto produce overturned isochrones.”",
        12.5, color=NAVY,
    )

    add_text(slide, 0.50, 2.62, 5.0, 0.22, "AGENTIC ITERATION", 11, color=BLUE, bold=True)
    nodes = [
        (0.55, 2.96, "Write / modify\nFiredrake model"),
        (3.35, 2.96, "Solve steady\nFull Stokes"),
        (3.35, 3.85, "Trace isochrones\nand compute J"),
        (0.55, 3.85, "Revise geometry\nand boundaries"),
    ]
    for x, y, label in nodes:
        add_box(slide, x, y, 1.80, 0.62, WHITE, line="AFC3CF")
        add_text(slide, x + 0.07, y + 0.04, 1.66, 0.54, label, 11.2,
                 align=PP_ALIGN.CENTER)
    add_arrow(slide, 2.35, 3.27, 3.35, 3.27)
    add_arrow(slide, 4.25, 3.58, 4.25, 3.85)
    add_arrow(slide, 3.35, 4.16, 2.35, 4.16)
    add_arrow(slide, 1.45, 3.85, 1.45, 3.58)

    metrics = [("9", "committed iterations"), ("≥46", "documented solves"), ("5", "retained workflows")]
    for index, (number, label) in enumerate(metrics):
        x = 0.52 + 1.68 * index
        add_text(slide, x, 4.58, 1.55, 0.36, number, 22, color=RED, bold=True,
                 align=PP_ALIGN.CENTER)
        add_text(slide, x, 4.91, 1.55, 0.25, label, 9.5, color=GRAY,
                 align=PP_ALIGN.CENTER)

    add_box(slide, 0.48, 5.34, 5.05, 1.47, PALE_RED, line="EABFC2")
    add_text(slide, 0.72, 5.47, 4.55, 0.21, "HUMAN PHYSICAL FILTER", 10, color=RED, bold=True)
    add_text(
        slide, 0.72, 5.68, 4.55, 1.00,
        "Physical judgment rejected inherited finite-amplitude folds and a prescribed-speed basal “conveyor belt”; tested whether recirculation was required; and demanded gravity-driven power, a dissipative linear sliding law, open trajectories, and positive boundary thickness.",
        11.3, color=NAVY,
    )

    add_text(slide, 5.85, 1.39, 6.95, 0.24, "FINAL RETAINED EXAMPLE", 11, color=BLUE, bold=True)
    cropped = crop_figure()
    slide.shapes.add_picture(str(cropped), Inches(5.82), Inches(1.73), width=Inches(7.02))
    add_box(slide, 5.91, 5.05, 6.84, 0.53, PALE_RED, line="EABFC2")
    add_text(
        slide, 6.08, 5.10, 6.50, 0.42,
        "Overturned limb: red circles mark  J < 0",
        14.5, color=RED, bold=True, align=PP_ALIGN.CENTER,
    )
    add_text(
        slide, 5.92, 5.72, 6.80, 0.50,
        "“Would it be possible … using a more conventional sliding law?”",
        13.2, color=NAVY, align=PP_ALIGN.CENTER,
    )
    add_text(
        slide, 5.92, 6.18, 6.80, 0.50,
        "“Is it possible … if the ice thickness does NOT taper to zero?”",
        13.2, color=NAVY, align=PP_ALIGN.CENTER,
    )

    add_text(
        slide, 0.50, 7.12, 12.3, 0.18,
        "Repository audit: nine numerical commits, 6–22 August 2026; ≥46 Firedrake solves in five retained scripts. Exploratory reruns and particle integrations are not counted.",
        8.5, color=GRAY, align=PP_ALIGN.CENTER,
    )

    presentation.save(OUTPUT)


if __name__ == "__main__":
    build_slide()
    print(OUTPUT)
