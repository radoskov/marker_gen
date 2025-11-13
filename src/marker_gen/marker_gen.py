#!/usr/bin/env python3
"""
aruco_maker.py

Generate ArUco markers (single or grid), ChArUco diamonds, and ChArUco boards
to exact physical sizes on A5/A4/A3 or custom paper. Exports PNG (with DPI) or
PDF (page size in mm) so printouts are dimensionally accurate when printed
at 100% / "Actual size" with scaling disabled.

Examples
--------
# 1) Single marker (50 mm) centered on A4, PDF:
python aruco_maker.py marker \
  --id 23 --dictionary DICT_6X6_250 --marker-size-mm 50 \
  --paper A4 --format pdf --output marker_23_A4.pdf

# 2) Grid of markers on A4 (4x3), 40 mm each, 8 mm spacing, PNG 300 DPI:
python aruco_maker.py sheet \
  --ids 1 2 3 4 5 6 7 8 9 10 11 12 \
  --dictionary DICT_5X5_100 --marker-size-mm 40 --rows 3 --cols 4 \
  --spacing-mm 8 --paper A4 --format png --dpi 300 --output sheet.png

# 3) ChArUco diamond (IDs 10,11,12,13), square=40 mm, marker=28 mm, A5 PDF:
python aruco_maker.py diamond \
  --ids 10 11 12 13 --square-mm 40 --marker-mm 28 \
  --paper A5 --format pdf --output diamond.pdf

# 4) ChArUco board 10x7, square=25 mm, marker=18 mm, custom paper 300x220 mm:
python aruco_maker.py charuco \
  --squares-x 10 --squares-y 7 --square-mm 25 --marker-mm 18 \
  --dictionary DICT_4X4_50 \
  --paper custom --paper-mm 300 220 --format png --dpi 300 --output board.png

# 5) List available dictionaries with details:
python aruco_maker.py --list-dicts
python aruco_maker.py --explain-dicts
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import sys
import re
import math
import argparse

import numpy as np

try:
    import cv2
    from cv2 import aruco
except Exception as e:
    sys.stderr.write(
        "ERROR: OpenCV with aruco module is required. Install:\n"
        "  pip install opencv-contrib-python\n"
    )
    raise

try:
    from PIL import Image
except Exception:
    sys.stderr.write("ERROR: Pillow is required for PNG export. Install: pip install pillow\n")
    raise

try:
    from reportlab.pdfgen import canvas as pdfcanvas
    from reportlab.lib.units import mm as RL_MM
    from reportlab.lib.utils import ImageReader
except Exception:
    # Only error at runtime if user requests PDF output.
    pdfcanvas = None
    RL_MM = None
    ImageReader = None


# ---------------------------
# Unit / paper / helpers
# ---------------------------

class UnitHelper:
    """Utility static methods for unit conversions between mm, inches, pixels, and points."""
    @staticmethod
    def mm_to_inches(mm: float) -> float:
        return mm / 25.4

    @staticmethod
    def mm_to_px(mm: float, dpi: int) -> int:
        return int(round(UnitHelper.mm_to_inches(mm) * dpi))

    @staticmethod
    def px_to_mm(px: int, dpi: int) -> float:
        return (px / dpi) * 25.4

    @staticmethod
    def mm_to_points(mm: float) -> float:
        # PDF points: 1 in = 72 pt; 1 mm = 72/25.4 pt
        return (mm / 25.4) * 72.0


@dataclass
class PaperSize:
    """Represents a physical paper size in millimeters."""
    width_mm: float
    height_mm: float

    @staticmethod
    def from_name(name: str, landscape: bool = False) -> 'PaperSize':
        name = name.upper()
        # ISO 216 sizes
        sizes = {
            "A3": (297.0, 420.0),
            "A4": (210.0, 297.0),
            "A5": (148.0, 210.0),
        }
        if name not in sizes:
            raise ValueError(f"Unsupported paper '{name}'. Use A3/A4/A5 or 'custom'.")
        w, h = sizes[name]
        if landscape:
            w, h = h, w
        return PaperSize(w, h)


# ---------------------------
# Dictionary utilities
# ---------------------------

class DictionaryInfo:
    """
    Provides introspection, resolution, and explanation of available cv2.aruco dictionaries.
    """

    @staticmethod
    def list_available() -> Dict[str, int]:
        """
        Return a mapping of dictionary attribute names to numeric constants
        available in cv2.aruco for this OpenCV build.
        """
        result = {}
        for attr in dir(aruco):
            if attr.startswith("DICT_"):
                result[attr] = getattr(aruco, attr)
        return dict(sorted(result.items()))

    @staticmethod
    def resolve(name: str):
        """
        Resolve a dictionary by (case-insensitive) attribute name.
        Allows inputs like 'DICT_6X6_250', '6x6_250', '4X4_50', 'ARUCO_ORIGINAL', 'APRILTAG_36h11'.
        """
        name = name.strip().upper()
        # Normalize common shortcuts
        if not name.startswith("DICT_"):
            if re.match(r'^\d+X\d+_\d+$', name):
                name = "DICT_" + name
            elif name in ("ARUCO_ORIGINAL", "ARUCO-ORIGINAL"):
                name = "DICT_ARUCO_ORIGINAL"
            elif name.startswith("APRILTAG"):
                name = "DICT_" + name  # e.g., APRILTAG_36H11
            else:
                name = "DICT_" + name

        if not hasattr(aruco, name):
            available = ", ".join(DictionaryInfo.list_available().keys())
            raise ValueError(f"Dictionary '{name}' not found. Available: {available}")
        return aruco.getPredefinedDictionary(getattr(aruco, name)), name

    @staticmethod
    def infer_bits_from_name(dict_name: str) -> Optional[int]:
        """Try to infer the bit width (e.g., 4,5,6,7) from a dictionary name."""
        m = re.search(r'(\d+)X\1', dict_name.upper())
        if m:
            return int(m.group(1))
        # For ARUCO_ORIGINAL / some APRILTAG families, bits aren't of the NxN ArUco family.
        return None

    @staticmethod
    def explain() -> str:
        """
        Human-readable explanation of dictionary families and how to choose.
        """
        lines = []
        lines.append("Dictionary families you may see (depending on your OpenCV build):")
        lines.append("")
        lines.append("• DICT_4X4_xxx, DICT_5X5_xxx, DICT_6X6_xxx, DICT_7X7_xxx")
        lines.append("  - ArUco marker families with an NxN binary payload and an outer black border.")
        lines.append("  - Higher N stores more bits and is more robust to false positives.")
        lines.append("  - The suffix (e.g., '_50', '_100', '_250', '_1000') is the dictionary size (how many unique IDs).")
        lines.append("")
        lines.append("• DICT_ARUCO_ORIGINAL")
        lines.append("  - The original ArUco dictionary (roughly 5x5 payload).")
        lines.append("")
        lines.append("• DICT_APRILTAG_* (e.g., APRILTAG_16h5, 25h9, 36h11, 41h12)")
        lines.append("  - AprilTag families available through OpenCV's aruco module.")
        lines.append("  - Different error correction properties and payload sizes.")
        lines.append("")
        lines.append("Choosing a dictionary:")
        lines.append("  - For general use: DICT_5X5_100 or DICT_6X6_250 are common, with IDs that cover most needs.")
        lines.append("  - For small physical markers, prefer 4x4 (fewer bits, larger squares).")
        lines.append("  - For larger markers or when many unique IDs are needed, 6x6 or 7x7.")
        lines.append("  - For interop with existing systems, match their dictionary and ID range.")
        return "\n".join(lines)


# ---------------------------
# Rendering / export helpers
# ---------------------------

class PageCanvas:
    """
    A white page canvas in pixel space that represents the physical page.
    You can place content (numpy images) onto it by mm coordinates.
    """
    def __init__(self, paper: PaperSize, dpi: int):
        self.paper = paper
        self.dpi = dpi
        self.width_px = UnitHelper.mm_to_px(paper.width_mm, dpi)
        self.height_px = UnitHelper.mm_to_px(paper.height_mm, dpi)
        # BGR canvas for OpenCV drawing
        self.img = np.full((self.height_px, self.width_px, 3), 255, dtype=np.uint8)

    def place(self, tile: np.ndarray, x_mm: float, y_mm: float, width_mm: float) -> None:
        """
        Paste 'tile' (H x W x 3 or H x W grayscale) onto the page so that its top-left corner
        is at (x_mm, y_mm). If width_mm is given, the tile is resized to that width preserving
        aspect ratio before being placed.
        """
        if tile.ndim == 2:
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)

        target_w_px = max(1, UnitHelper.mm_to_px(width_mm, self.dpi))
        aspect = tile.shape[0] / tile.shape[1]
        target_h_px = max(1, int(round(target_w_px * aspect)))
        resized = cv2.resize(tile, (target_w_px, target_h_px), interpolation=cv2.INTER_NEAREST)

        x_px = UnitHelper.mm_to_px(x_mm, self.dpi)
        y_px = UnitHelper.mm_to_px(y_mm, self.dpi)

        # Clip to page if needed
        x2 = min(self.width_px, x_px + resized.shape[1])
        y2 = min(self.height_px, y_px + resized.shape[0])
        x1 = max(0, x_px)
        y1 = max(0, y_px)

        roi = self.img[y1:y2, x1:x2]
        src = resized[(y1 - y_px):(y2 - y_px), (x1 - x_px):(x2 - x_px)]
        if roi.size and src.size:
            roi[:] = src

    def draw_mm_ruler(self, start_mm: float = 10.0, y_mm: float = None, length_mm: float = 100.0) -> None:
        """
        Draw a simple mm ruler on the page to verify scale.
        """
        if y_mm is None:
            y_mm = self.paper.height_mm - 10.0  # 10 mm from bottom
        x0_px = UnitHelper.mm_to_px(start_mm, self.dpi)
        y_px = UnitHelper.mm_to_px(y_mm, self.dpi)
        length_px = UnitHelper.mm_to_px(length_mm, self.dpi)
        cv2.line(self.img, (x0_px, y_px), (x0_px + length_px, y_px), (0, 0, 0), 1)
        for mm_tick in range(0, int(length_mm) + 1):
            x = x0_px + UnitHelper.mm_to_px(mm_tick, self.dpi)
            tick_len = 10 if (mm_tick % 10 == 0) else (6 if (mm_tick % 5 == 0) else 3)
            cv2.line(self.img, (x, y_px - tick_len), (x, y_px + tick_len), (0, 0, 0), 1)
            if mm_tick % 10 == 0:
                label = f"{mm_tick} mm"
                cv2.putText(self.img, label, (x + 2, y_px - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)


class Exporter:
    """
    Handles saving a PageCanvas to PNG (with embedded DPI) or PDF (exact page size).
    """
    def __init__(self, dpi: int = 300):
        self.dpi = dpi

    def save_png(self, page: PageCanvas, path: str) -> None:
        """
        Save the page canvas to a PNG with DPI metadata. This helps print software
        respect physical size (still choose 'Actual size' in the print dialog).
        """
        rgb = cv2.cvtColor(page.img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.save(path, format="PNG", dpi=(self.dpi, self.dpi))

    def save_pdf(self, page: PageCanvas, path: str, paper: PaperSize) -> None:
        """
        Save a PDF page with exact physical dimensions. The whole composed page image is
        placed at 1:1 on a PDF page sized in mm.
        """
        if pdfcanvas is None:
            raise RuntimeError("reportlab is required for PDF export. Install: pip install reportlab")
        w_pt = UnitHelper.mm_to_points(paper.width_mm)
        h_pt = UnitHelper.mm_to_points(paper.height_mm)

        rgb = cv2.cvtColor(page.img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        img_reader = ImageReader(pil)

        c = pdfcanvas.Canvas(path, pagesize=(w_pt, h_pt))
        # Place image to fill the page exactly
        c.drawImage(img_reader, 0, 0, width=w_pt, height=h_pt, preserveAspectRatio=False, mask='auto')
        c.showPage()
        c.save()


# ---------------------------
# ArUco content generators
# ---------------------------

class MarkerGenerator:
    """
    Creates single ArUco markers as numpy images and annotates legend text
    onto the black border area (small white text).
    """

    def __init__(self, dictionary, dict_name: str, dpi: int = 300):
        self.dictionary = dictionary
        self.dict_name = dict_name
        self.dpi = dpi

    @staticmethod
    def _fit_side_to_modules(target_side_px: int, bits: int, border_bits: int) -> int:
        """
        Ensure the final side length is an integer multiple of the module size,
        so edges are crisp. Module count = bits + 2*border_bits.
        """
        modules = bits + 2 * border_bits
        module_px = max(1, round(target_side_px / modules))
        return module_px * modules

    @staticmethod
    def _auto_font_scale(text: str, max_w: int, max_h: int) -> Tuple[float, int]:
        """
        Find a cv2 font scale (and thickness) to fit 'text' within the given bounding box.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Start small and grow; keep it robust and fast.
        scale = 0.1
        thickness = 1
        while True:
            (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
            if w > max_w or (h + baseline) > max_h:
                # step back
                scale *= 0.92
                break
            scale *= 1.08
            if scale > 5.0:
                break
        # Clamp a bit
        scale = max(0.1, min(scale, 5.0))
        return scale, thickness

    def draw_single(self,
                    marker_id: int,
                    size_mm: float,
                    border_bits: int = 2,
                    legend: Optional[str] = None,
                    bits_hint: Optional[int] = None) -> np.ndarray:
        """
        Generate a single ArUco marker image of exact physical size (when placed with DPI).
        The legend is written in white into the bottom black border area.

        Returns BGR image (OpenCV).
        """
        # Infer bits (4,5,6,7) from dict name, or use hint if provided.
        bits = DictionaryInfo.infer_bits_from_name(self.dict_name) or bits_hint or 5

        # Compute target pixel side and snap to module multiple.
        target_side_px = UnitHelper.mm_to_px(size_mm, self.dpi)
        side_px = self._fit_side_to_modules(target_side_px, bits, border_bits)

        # Draw marker in grayscale then convert to BGR
        marker = aruco.drawMarker(self.dictionary, marker_id, side_px, borderBits=border_bits)
        marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

        # Build default legend if not provided
        if legend is None:
            legend = f"id={marker_id} | dict={self.dict_name} | bits={bits} | size={size_mm:.2f}mm | border={border_bits}"

        # Compute border height in px
        modules = bits + 2 * border_bits
        module_px = side_px // modules
        border_px = border_bits * module_px

        # Place legend inside bottom border
        padding = max(1, module_px // 3)
        max_w = side_px - 2 * padding
        max_h = max(1, int(border_px * 0.8))
        scale, thick = self._auto_font_scale(legend, max_w, max_h)
        (tw, th), baseline = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)

        x = max(padding, (side_px - tw) // 2)
        y = side_px - int((border_px - th) / 2) - baseline - padding
        y = min(y, side_px - padding)

        cv2.putText(marker, legend, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)
        return marker


class MultiMarkerSheet:
    """
    Lays out multiple annotated markers into a grid on a PageCanvas.
    """

    def __init__(self, marker_gen: MarkerGenerator, page: PageCanvas):
        self.marker_gen = marker_gen
        self.page = page

    def render(self,
               ids: List[int],
               dictionary_name: str,
               marker_size_mm: float,
               rows: int,
               cols: int,
               spacing_mm: float,
               margin_mm: float,
               border_bits: int = 2) -> PageCanvas:
        """
        Create a grid of markers placed on the page with given spacing and margins.
        """
        x0 = margin_mm
        y0 = margin_mm

        # Pre-generate one marker to compute its pixel/module behavior (for crispness)
        # but we'll place by mm width on the page.
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if idx >= len(ids):
                    continue
                marker_id = ids[idx]
                legend = f"id={marker_id} | dict={dictionary_name} | size={marker_size_mm:.2f}mm"
                tile = self.marker_gen.draw_single(marker_id, marker_size_mm,
                                                   border_bits=border_bits,
                                                   legend=legend)
                x_mm = x0 + c * (marker_size_mm + spacing_mm)
                y_mm = y0 + r * (marker_size_mm + spacing_mm)
                self.page.place(tile, x_mm, y_mm, marker_size_mm)
        return self.page


class DiamondGenerator:
    """
    Generates a ChArUco diamond and writes small white legend text into black areas.
    """

    def __init__(self, dictionary, dict_name: str, dpi: int = 300):
        self.dictionary = dictionary
        self.dict_name = dict_name
        self.dpi = dpi

    def draw(self,
             ids4: List[int],
             square_mm: float,
             marker_mm: float,
             legend_lines: Optional[List[str]] = None) -> np.ndarray:
        """
        Create a ChArUco diamond image (BGR) sized so that one square equals 'square_mm'
        when placed at the corresponding mm width.
        """
        if len(ids4) != 4:
            raise ValueError("A ChArUco diamond needs exactly 4 IDs.")

        # Choose pixel side for square and marker; keep integers to preserve edges.
        square_px = max(40, UnitHelper.mm_to_px(square_mm, self.dpi))
        marker_px = max(10, UnitHelper.mm_to_px(marker_mm, self.dpi))

        img = aruco.drawCharucoDiamond(self.dictionary, np.array(ids4, dtype=np.int32),
                                       square_px, marker_px)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Build default legend if not given
        if legend_lines is None:
            legend_lines = [
                f"ids={ids4}",
                f"dict={self.dict_name}",
                f"square={square_mm:.2f}mm, marker={marker_mm:.2f}mm"
            ]

        # Try to place legend into black triangular regions between markers.
        # We'll find 4 anchor points relative to the center and nudge until pixel is dark.
        H, W = img.shape[:2]
        cx, cy = W // 2, H // 2
        anchors = [
            (int(0.2 * W), cy),        # left side
            (int(0.8 * W), cy),        # right side
            (cx, int(0.2 * H)),        # top
            (cx, int(0.8 * H))         # bottom
        ]
        # Helper to nudge to a dark pixel
        def move_to_dark(x, y, dx, dy, steps=200):
            for _ in range(steps):
                x = max(0, min(W - 1, x + dx))
                y = max(0, min(H - 1, y + dy))
                if img[y, x, 0] < 50 and img[y, x, 1] < 50 and img[y, x, 2] < 50:
                    return x, y
            return x, y

        targets = [
            move_to_dark(anchors[0][0], anchors[0][1], -2, 0),
            move_to_dark(anchors[1][0], anchors[1][1], +2, 0),
            move_to_dark(anchors[2][0], anchors[2][1], 0, -2),
            move_to_dark(anchors[3][0], anchors[3][1], 0, +2),
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, line in enumerate(legend_lines[:4]):
            x, y = targets[i]
            # Fit tiny font into local area
            scale, thick = 0.4, 1
            (tw, th), bl = cv2.getTextSize(line, font, scale, thick)
            # Recenter roughly
            x = int(x - tw / 2)
            y = int(y + th / 2)
            x = max(2, min(W - tw - 2, x))
            y = max(th + 2, min(H - 2, y))
            cv2.putText(img, line, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        return img


class CharucoBoardGenerator:
    """
    Generates ChArUco chessboards and writes small white legend text into black squares.
    """

    def __init__(self, dictionary, dict_name: str, dpi: int = 300):
        self.dictionary = dictionary
        self.dict_name = dict_name
        self.dpi = dpi

    def draw(self,
             squares_x: int,
             squares_y: int,
             square_mm: float,
             marker_mm: float,
             margin_mm: float = 6.0,
             border_bits: int = 1,
             legend_lines: Optional[List[str]] = None) -> np.ndarray:
        """
        Create a ChArUco board image (BGR). The board image area (without page margins)
        will be squares_x*square_mm by squares_y*square_mm approximately.
        """
        # Convert to pixel units for the rendered board image
        square_px = max(20, UnitHelper.mm_to_px(square_mm, self.dpi))
        marker_px = max(10, UnitHelper.mm_to_px(marker_mm, self.dpi))
        margin_px = max(0, UnitHelper.mm_to_px(margin_mm, self.dpi))

        board = aruco.CharucoBoard_create(
            squaresX=squares_x, squaresY=squares_y,
            squareLength=1.0,  # logical units; actual size determined by draw()
            markerLength=marker_mm / square_mm,  # relative (ignored for draw), but keep ratio
            dictionary=self.dictionary
        )

        # OpenCV's draw uses 'outSize' and 'marginSize'. We'll build a board area and draw into it.
        img_w = squares_x * square_px
        img_h = squares_y * square_px
        img = board.draw(outSize=(img_w, img_h), marginSize=margin_px, borderBits=border_bits)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Default legend
        if legend_lines is None:
            w_mm = squares_x * square_mm
            h_mm = squares_y * square_mm
            legend_lines = [
                f"{squares_x}x{squares_y} squares | dict={self.dict_name}",
                f"square={square_mm:.2f}mm | marker={marker_mm:.2f}mm",
                f"board={w_mm:.1f} x {h_mm:.1f} mm"
            ]

        # Write legend pieces into several black squares near corners/edges.
        # For a chessboard with top-left white, black squares are those where (x+y)%2==1.
        # We'll use (1,0), (0,1), (squares_x-2,squares_y-1), (squares_x-1,squares_y-2) if valid.
        candidates = []
        if squares_x >= 2 and squares_y >= 1:
            candidates.append((1, 0))
        if squares_x >= 1 and squares_y >= 2:
            candidates.append((0, 1))
        if squares_x >= 2 and squares_y >= 1:
            candidates.append((squares_x - 2, squares_y - 1))
        if squares_x >= 1 and squares_y >= 2:
            candidates.append((squares_x - 1, squares_y - 2))

        font = cv2.FONT_HERSHEY_SIMPLEX
        placed = 0
        for (sx, sy), text in zip(candidates, legend_lines):
            # If this square is not black, nudge to nearest black neighbor
            if (sx + sy) % 2 == 0:
                sx = min(squares_x - 1, sx + 1)  # nudge
            # Square rectangle in pixels
            x0 = sx * square_px + margin_px
            y0 = sy * square_px + margin_px
            x1 = x0 + square_px
            y1 = y0 + square_px

            # Fit text inside padding box
            pad = max(2, square_px // 12)
            avail_w = square_px - 2 * pad
            avail_h = square_px - 2 * pad

            # Find scale to fit
            scale, thick = 0.5, 1
            (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
            while (tw > avail_w or (th + bl) > avail_h) and scale > 0.15:
                scale *= 0.92
                (tw, th), bl = cv2.getTextSize(text, font, scale, thick)

            tx = int(x0 + (square_px - tw) / 2)
            ty = int(y0 + (square_px + th) / 2)
            cv2.putText(img, text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
            placed += 1

        # If any legend lines remain, write them faintly across central black squares (if available).
        if placed < len(legend_lines):
            remaining = legend_lines[placed:]
            cx_sq = squares_x // 2
            cy_sq = squares_y // 2
            for i, text in enumerate(remaining):
                sx = min(squares_x - 1, max(0, cx_sq + (i % 3) - 1))
                sy = min(squares_y - 1, max(0, cy_sq + (i // 3) - 1))
                if (sx + sy) % 2 == 0:
                    sx = (sx + 1) % squares_x
                x0 = sx * square_px + margin_px
                y0 = sy * square_px + margin_px
                pad = max(2, square_px // 12)
                avail_w = square_px - 2 * pad
                scale, thick = 0.45, 1
                (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
                if tw > avail_w:
                    scale = max(0.15, scale * (avail_w / (tw + 1)))
                    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
                tx = int(x0 + pad)
                ty = int(y0 + th + pad)
                cv2.putText(img, text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

        return img


# ---------------------------
# CLI Orchestration
# ---------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate ArUco markers, ChArUco diamonds and boards with exact physical sizing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--list-dicts", action="store_true",
                   help="List available cv2.aruco dictionaries and exit.")
    p.add_argument("--explain-dicts", action="store_true",
                   help="Explain dictionary families and usage, then exit.")

    sub = p.add_subparsers(dest="mode", required=False)

    # Common paper/output args
    def add_paper_args(sp):
        sp.add_argument("--paper", choices=["A5", "A4", "A3", "custom"], default="A4",
                        help="Paper size.")
        sp.add_argument("--landscape", action="store_true", help="Use landscape orientation.")
        sp.add_argument("--paper-mm", nargs=2, type=float, metavar=("WIDTH_MM", "HEIGHT_MM"),
                        help="Custom paper size in mm (requires --paper custom).")
        sp.add_argument("--format", choices=["png", "pdf"], default="pdf",
                        help="Output format.")
        sp.add_argument("--dpi", type=int, default=300,
                        help="Raster DPI (affects PNG pixel size and internal raster for PDF).")
        sp.add_argument("--output", required=True, help="Output file path.")
        sp.add_argument("--draw-ruler", action="store_true",
                        help="Draw a 100 mm ruler at bottom to verify scale.")

    # --- marker ---
    sp_marker = sub.add_parser("marker", help="Single ArUco marker.")
    add_paper_args(sp_marker)
    sp_marker.add_argument("--id", type=int, required=True, help="Marker ID.")
    sp_marker.add_argument("--dictionary", type=str, required=True,
                           help="Dictionary name (e.g. DICT_6X6_250, 5X5_100, ARUCO_ORIGINAL).")
    sp_marker.add_argument("--marker-size-mm", type=float, required=True, help="Marker side length in mm.")
    sp_marker.add_argument("--border-bits", type=int, default=2, help="Black border thickness (in modules).")
    sp_marker.add_argument("--margin-mm", type=float, default=10.0, help="Margin from page edges.")

    # --- sheet ---
    sp_sheet = sub.add_parser("sheet", help="Grid of multiple markers.")
    add_paper_args(sp_sheet)
    sp_sheet.add_argument("--ids", type=int, nargs="+", required=True, help="Marker IDs.")
    sp_sheet.add_argument("--dictionary", type=str, required=True, help="Dictionary name.")
    sp_sheet.add_argument("--marker-size-mm", type=float, required=True, help="Marker side length in mm.")
    sp_sheet.add_argument("--rows", type=int, required=True)
    sp_sheet.add_argument("--cols", type=int, required=True)
    sp_sheet.add_argument("--spacing-mm", type=float, default=6.0, help="Spacing between markers.")
    sp_sheet.add_argument("--border-bits", type=int, default=2)
    sp_sheet.add_argument("--margin-mm", type=float, default=10.0)

    # --- diamond ---
    sp_diamond = sub.add_parser("diamond", help="ChArUco diamond.")
    add_paper_args(sp_diamond)
    sp_diamond.add_argument("--ids", type=int, nargs=4, required=True,
                            help="Four marker IDs for the diamond.")
    sp_diamond.add_argument("--dictionary", type=str, default="DICT_4X4_50",
                            help="Dictionary name.")
    sp_diamond.add_argument("--square-mm", type=float, required=True,
                            help="Square size (diamond base square) in mm.")
    sp_diamond.add_argument("--marker-mm", type=float, required=True,
                            help="Marker size inside the square in mm.")
    sp_diamond.add_argument("--margin-mm", type=float, default=10.0)

    # --- charuco ---
    sp_charuco = sub.add_parser("charuco", help="ChArUco (ArUco chessboard) board.")
    add_paper_args(sp_charuco)
    sp_charuco.add_argument("--squares-x", type=int, required=True, help="Number of squares along X.")
    sp_charuco.add_argument("--squares-y", type=int, required=True, help="Number of squares along Y.")
    sp_charuco.add_argument("--square-mm", type=float, required=True, help="Square size in mm.")
    sp_charuco.add_argument("--marker-mm", type=float, required=True, help="Marker size in mm.")
    sp_charuco.add_argument("--dictionary", type=str, required=True, help="Dictionary name.")
    sp_charuco.add_argument("--margin-mm", type=float, default=6.0, help="Outer white margin in mm.")
    sp_charuco.add_argument("--border-bits", type=int, default=1, help="Marker border bits.")

    return p


def resolve_paper(args) -> PaperSize:
    if args.paper != "custom":
        return PaperSize.from_name(args.paper, landscape=args.landscape)
    if not args.paper_mm:
        raise ValueError("Specify --paper-mm WIDTH HEIGHT when --paper custom.")
    w, h = map(float, args.paper_mm)
    if args.landscape:
        w, h = h, w
    return PaperSize(w, h)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_dicts:
        avail = DictionaryInfo.list_available()
        print("Available dictionaries in this OpenCV build:")
        for k in avail:
            print("  " + k)
        return

    if args.explain_dicts:
        print(DictionaryInfo.explain())
        return

    if not args.mode:
        parser.print_help()
        sys.exit(0)

    # Resolve paper and build page canvas
    paper = resolve_paper(args)
    exporter = Exporter(dpi=args.dpi)
    page = PageCanvas(paper, dpi=args.dpi)

    # Mode-specific generation
    if args.mode == "marker":
        dictionary, dict_name = DictionaryInfo.resolve(args.dictionary)
        gen = MarkerGenerator(dictionary, dict_name, dpi=args.dpi)

        # Create the marker tile (annotated)
        tile = gen.draw_single(
            marker_id=args.id,
            size_mm=args.marker_size_mm,
            border_bits=args.border_bits
        )

        # Center on page (with margin guard)
        x_mm = max(args.margin_mm, (paper.width_mm - args.marker_size_mm) / 2.0)
        y_mm = max(args.margin_mm, (paper.height_mm - args.marker_size_mm) / 2.0)
        page.place(tile, x_mm, y_mm, args.marker_size_mm)

        if args.draw_ruler:
            page.draw_mm_ruler()

        # Export
        if args.format == "png":
            exporter.save_png(page, args.output)
        else:
            exporter.save_pdf(page, args.output, paper)

    elif args.mode == "sheet":
        dictionary, dict_name = DictionaryInfo.resolve(args.dictionary)
        gen = MarkerGenerator(dictionary, dict_name, dpi=args.dpi)
        sheet = MultiMarkerSheet(gen, page)

        # Simple fit check
        total_w = args.margin_mm * 2 + args.cols * args.marker_size_mm + (args.cols - 1) * args.spacing_mm
        total_h = args.margin_mm * 2 + args.rows * args.marker_size_mm + (args.rows - 1) * args.spacing_mm
        if total_w > paper.width_mm + 1e-6 or total_h > paper.height_mm + 1e-6:
            raise ValueError(f"Grid does not fit page: needs {total_w:.1f}x{total_h:.1f} mm, "
                             f"page is {paper.width_mm:.1f}x{paper.height_mm:.1f} mm.")

        sheet.render(ids=args.ids,
                     dictionary_name=dict_name,
                     marker_size_mm=args.marker_size_mm,
                     rows=args.rows, cols=args.cols,
                     spacing_mm=args.spacing_mm,
                     margin_mm=args.margin_mm,
                     border_bits=args.border_bits)

        if args.draw_ruler:
            page.draw_mm_ruler()

        if args.format == "png":
            exporter.save_png(page, args.output)
        else:
            exporter.save_pdf(page, args.output, paper)

    elif args.mode == "diamond":
        dictionary, dict_name = DictionaryInfo.resolve(args.dictionary)
        gen = DiamondGenerator(dictionary, dict_name, dpi=args.dpi)

        tile = gen.draw(ids4=args.ids, square_mm=args.square_mm, marker_mm=args.marker_mm)

        # Place centered, respecting margin
        avail_w = paper.width_mm - 2 * args.margin_mm
        avail_h = paper.height_mm - 2 * args.margin_mm
        # Diamond image is roughly 3 squares wide; place by making its width 3*square_mm
        est_w_mm = 3.0 * args.square_mm
        width_mm = min(avail_w, est_w_mm)
        x_mm = (paper.width_mm - width_mm) / 2.0
        # Maintain aspect by deriving height from image ratio
        page.place(tile, x_mm, (paper.height_mm - width_mm * (tile.shape[0]/tile.shape[1])) / 2.0, width_mm)

        if args.draw_ruler:
            page.draw_mm_ruler()

        if args.format == "png":
            exporter.save_png(page, args.output)
        else:
            exporter.save_pdf(page, args.output, paper)

    elif args.mode == "charuco":
        dictionary, dict_name = DictionaryInfo.resolve(args.dictionary)
        gen = CharucoBoardGenerator(dictionary, dict_name, dpi=args.dpi)
        tile = gen.draw(squares_x=args.squares_x, squares_y=args.squares_y,
                        square_mm=args.square_mm, marker_mm=args.marker_mm,
                        margin_mm=args.margin_mm, border_bits=args.border_bits)

        # Compute board physical width to place on page
        board_w_mm = args.squares_x * args.square_mm + 2 * args.margin_mm
        board_h_mm = args.squares_y * args.square_mm + 2 * args.margin_mm
        if board_w_mm > paper.width_mm + 1e-6 or board_h_mm > paper.height_mm + 1e-6:
            raise ValueError(f"Board image ({board_w_mm:.1f}x{board_h_mm:.1f} mm with margins) exceeds page "
                             f"({paper.width_mm:.1f}x{paper.height_mm:.1f} mm). Reduce sizes or pick bigger paper.")

        x_mm = (paper.width_mm - board_w_mm) / 2.0
        y_mm = (paper.height_mm - board_h_mm) / 2.0
        page.place(tile, x_mm, y_mm, board_w_mm)

        if args.draw_ruler:
            page.draw_mm_ruler()

        if args.format == "png":
            exporter.save_png(page, args.output)
        else:
            exporter.save_pdf(page, args.output, paper)

    else:
        parser.error(f"Unknown mode {args.mode!r}")


if __name__ == "__main__":
    main()
