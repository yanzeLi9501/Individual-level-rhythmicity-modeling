#!/usr/bin/env python3
"""Create a merged SCH supplementary time-series figure from existing panels."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
NC_DIR = Path(__file__).resolve().parent
SOURCE = ROOT / "FigureS2_sch_rdi_workflow.png"
OUTPUT_STEM = "FigureS1_S2_sch_time_series_merged"

SOURCE_LABEL_ERASE_BOXES = {
    # The source figure has baked-in letters positioned inside legends or plot
    # bodies. Keep the legends/axes intact and only mask the old letter glyphs.
    "A": [((190, 35, 255, 95), "white")],
    "B": [((340, 30, 414, 120), ("sample", 420, 80))],
    "C": [((215, 150, 275, 230), "neutral_text")],
    "D": [((365, 175, 445, 255), "white")],
}


def erase_source_labels(panel: Image.Image, source_label: str) -> None:
    draw = ImageDraw.Draw(panel)
    pixels = panel.load()
    for erase_box, fill in SOURCE_LABEL_ERASE_BOXES[source_label]:
        if fill == "neutral_text":
            left, top, right, bottom = erase_box
            for y in range(top, bottom):
                for x in range(left, right):
                    red, green, blue = pixels[x, y]
                    is_dark_neutral = max(red, green, blue) < 180 and max(red, green, blue) - min(red, green, blue) < 40
                    is_faint_neutral = 210 < min(red, green, blue) < 250 and max(red, green, blue) - min(red, green, blue) < 20
                    if is_dark_neutral or is_faint_neutral:
                        pixels[x, y] = (255, 255, 255)
        elif isinstance(fill, tuple) and fill[0] == "sample":
            _, sample_x, sample_y = fill
            draw.rectangle(erase_box, fill=panel.getpixel((sample_x, sample_y)))
        else:
            draw.rectangle(erase_box, fill=fill)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf") if bold else Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf") if bold else Path(r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def crop_panel(image: Image.Image, box: tuple[int, int, int, int], label: str, source_label: str) -> Image.Image:
    panel = image.crop(box).convert("RGB")
    draw = ImageDraw.Draw(panel)
    erase_source_labels(panel, source_label)
    draw.text((24, 20), label, fill="black", font=load_font(58, bold=True))
    return panel


def make_merged_png() -> Path:
    image = Image.open(SOURCE).convert("RGB")
    width, height = image.size
    half_w = width // 2
    half_h = height // 2
    panels = [
        crop_panel(image, (0, half_h, half_w, height), "A", "C"),
        crop_panel(image, (0, 0, half_w, half_h), "B", "A"),
        crop_panel(image, (half_w, half_h, width, height), "C", "D"),
        crop_panel(image, (half_w, 0, width, half_h), "D", "B"),
    ]
    gap = 80
    canvas = Image.new("RGB", (width + gap, height + gap), "white")
    positions = [(0, 0), (half_w + gap, 0), (0, half_h + gap), (half_w + gap, half_h + gap)]
    for panel, position in zip(panels, positions):
        canvas.paste(panel, position)
    out_png = NC_DIR / f"{OUTPUT_STEM}.png"
    canvas.save(out_png, dpi=(300, 300))
    return out_png


def save_pdf_tif(png_path: Path) -> None:
    image = plt.imread(png_path)
    fig = plt.figure(figsize=(15.2, 11.1), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.axis("off")
    for ext in ("pdf", "tif"):
        kwargs = {"dpi": 300, "bbox_inches": "tight", "pad_inches": 0.02, "facecolor": "white"}
        if ext == "tif":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(NC_DIR / f"{OUTPUT_STEM}.{ext}", **kwargs)
    plt.close(fig)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    png = make_merged_png()
    save_pdf_tif(png)
    print(f"Merged SCH supplementary figure written to {NC_DIR / OUTPUT_STEM}.[png/pdf/tif]")


if __name__ == "__main__":
    main()