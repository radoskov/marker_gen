#!/usr/bin/env python3
"""
OpenCV 4.12+ compatible (w/ legacy fallback)

Generates:
  • Single ArUco marker or a grid (sheet)
  • ChArUco diamond (via CharucoBoard(3x3, ids=...))
  • ChArUco chessboard (CharucoBoard)

All outputs are sized to exact physical dimensions on A5/A4/A3 or custom paper.
Exports PNG (with embedded DPI) or PDF (page sized in mm). Legends are drawn in
white within black areas (marker border / black squares).

API notes (OpenCV ≥ 4.12):
  - Single marker: aruco.generateImageMarker(…)
    (doc: canonical & ready to print)
  - ChArUco board: aruco.CharucoBoard(...).generateImage(...)
    (doc: ready to be printed)
  - ChArUco diamond: construct CharucoBoard(Size(3,3), ... , ids=Vec4i) then generateImage(...)
    (per new tutorial)

Examples
--------
# Single marker (50 mm) centered on A4, PDF:
python aruco_maker_v2.py marker \
  --id 23 --dictionary DICT_6X6_250 --marker-size-mm 50 \
  --paper A4 --format pdf --output marker_23_A4.pdf

# Sheet of 4x3 markers, 40 mm each, spacing 8 mm, PNG 300 DPI:
python aruco_maker_v2.py sheet \
  --ids 1 2 3 4 5 6 7 8 9 10 11 12 \
  --dictionary 5X5_100 --marker-size-mm 40 --rows 3 --cols 4 \
  --spacing-mm 8 --paper A4 --format png --dpi 300 --output sheet.png

# ChArUco diamond (ids 10,11,12,13), square=40 mm, marker=28 mm, A5 PDF:
python aruco_maker_v2.py diamond \
  --ids 10 11 12 13 --square-mm 40 --marker-mm 28 \
  --paper A5 --format pdf --output diamond.pdf

# ChArUco board 10x7, square=25 mm, marker=18 mm, custom paper 300x220 mm:
python aruco_maker_v2.py charuco \
  --squares-x 10 --squares-y 7 --square-mm 25 --marker-mm 18 \
  --dictionary DICT_4X4_50 \
  --paper custom --paper-mm 300 220 --format png --dpi 300 --output board.png

# List & explain dictionaries:
python marker_gen.py --list-dicts
python marker_gen.py --explain-dicts
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import sys
import re
import argparse
import numpy as np

try:
    import cv2
    from cv2 import aruco
except ImportError:
    sys.stderr.write("ERROR: OpenCV with aruco module is required.\n"
                     "Install: pip install opencv-contrib-python\n")
    raise

try:
    from PIL import Image
except Exception:
    sys.stderr.write("ERROR: Pillow is required for PNG export. Install: pip install pillow\n")
    raise

try:
    from reportlab.pdfgen import canvas as pdfcanvas
    from reportlab.lib.utils import ImageReader
except Exception:
    pdfcanvas = None
    ImageReader = None


# ---------------------------
# Units, paper, helpers
# ---------------------------

class UnitHelper:
    """Unit conversions between mm, inches, pixels and PDF points (1/72 inch)."""
    @staticmethod
    def mm_to_inches(mm: float) -> float:
        return mm / 25.4

    @staticmethod
    def mm_to_px(mm: float, dpi: int) -> int:
        return int(round(UnitHelper.mm_to_inches(mm) * dpi))

    @staticmethod
    def mm_to_pdf_points(mm: float) -> float:
        # PDF geometry is in points: 1 inch = 72 pt (fixed, not printer DPI).
        return UnitHelper.mm_to_inches(mm) * 72.0


@dataclass
class PaperSize:
    """Physical paper size in millimeters."""
    width_mm: float
    height_mm: float

    @staticmethod
    def from_name(name: str, landscape: bool = False) -> 'PaperSize':
        name = name.upper()
        # ISO 216 sizes
        sizes = {"A3": (297.0, 420.0), "A4": (210.0, 297.0), "A5": (148.0, 210.0)}
        if name not in sizes:
            raise ValueError(f"Unsupported paper '{name}'. Use A3/A4/A5 or 'custom'.")
        w, h = sizes[name]
        return PaperSize((h, w)[landscape], (w, h)[landscape])


# ---------------------------
# ArUco dictionary utilities
# ---------------------------

class DictionaryInfo:
    """Resolve & explain cv2.aruco predefined dictionaries."""
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
        raw = name.strip().upper()
        # Normalize common shortcuts
        if not raw.startswith("DICT_"):
            if re.match(r'^\d+X\d+_\d+$', raw):
                raw = "DICT_" + raw
            elif raw in ("ARUCO_ORIGINAL", "ARUCO-ORIGINAL"):
                raw = "DICT_ARUCO_ORIGINAL"
            elif raw.startswith("APRILTAG"):
                raw = "DICT_" + raw
            else:
                raw = "DICT_" + raw
        if not hasattr(aruco, raw):
            avail = ", ".join(sorted(DictionaryInfo.list_available().keys()))
            raise ValueError(f"Dictionary '{raw}' not found. Available: {avail}")
        return aruco.getPredefinedDictionary(getattr(aruco, raw)), raw

    @staticmethod
    def infer_bits_from_name(dict_name: str) -> Optional[int]:
        """Try to infer the bit width (e.g., 4,5,6,7) from a dictionary name."""
        m = re.search(r'(\d+)X\1', dict_name.upper())
        return int(m.group(1)) if m else None

    @staticmethod
    def explain() -> str:
        """
        Human-readable explanation of dictionary families and how to choose.
        """
        lines = [
            "• DICT_4X4_xxx / 5X5_xxx / 6X6_xxx / 7X7_xxx:",
            "  NxN ArUco families; suffix is dictionary size (IDs).",
            "• DICT_ARUCO_ORIGINAL:",
            "  The original ArUco family.",
            "• DICT_APRILTAG_*:",
            "  AprilTag families exposed via cv2.aruco.",
            "",
            "Choosing:",
            "  - Small physical markers → 4x4.",
            "  - Most general use → 5x5_100 or 6x6_250.",
            "  - Many unique IDs or large markers → 6x6 or 7x7.",
            "  - Interop → match the external system’s family and ID range."
        ]
        return "\n".join(lines)


# ---------------------------
# Version-adaptive ArUco gen
# ---------------------------

class ArucoGen:
    """Tiny layer that picks the right OpenCV API (new vs legacy)."""

    @staticmethod
    def generate_marker_img(dictionary, marker_id: int, side_px: int, border_bits: int) -> np.ndarray:
        """
        Generate a single marker (grayscale) using the newest available API.
        Prefers aruco.generateImageMarker; falls back to aruco.drawMarker.
        """
        try:
            # OpenCV ≥ 4.7 exposed generateImageMarker; in 4.12+ it's the preferred call.
            return aruco.generateImageMarker(dictionary, marker_id, side_px, borderBits=border_bits)
        except TypeError:
            # Some builds may not accept borderBits kw; try without it.
            try:
                return aruco.generateImageMarker(dictionary, marker_id, side_px)
            except Exception:
                pass
        # Legacy (OpenCV 4.5.x and earlier)
        return aruco.drawMarker(dictionary, marker_id, side_px, borderBits=border_bits)

    @staticmethod
    def make_charuco_board(squares_x: int, squares_y: int,
                           square_len_rel: float, marker_len_rel: float,
                           dictionary,
                           ids: Optional[List[int]] = None):
        """
        Create a CharucoBoard using the new class API when available.
        - New: aruco.CharucoBoard((sx,sy), squareLength, markerLength, dictionary[, ids])
        - Legacy: aruco.CharucoBoard_create(...)
        ids: optional list of marker IDs. For a diamond, pass 4 IDs with size=(3,3).
        """
        try:
            if ids is not None:
                return aruco.CharucoBoard((squares_x, squares_y),
                                          square_len_rel, marker_len_rel,
                                          dictionary, np.array(ids, dtype=np.int32))
            return aruco.CharucoBoard((squares_x, squares_y),
                                      square_len_rel, marker_len_rel, dictionary)
        except Exception:
            # Legacy factory:
            board = aruco.CharucoBoard_create(
                squaresX=squares_x, squaresY=squares_y,
                squareLength=square_len_rel, markerLength=marker_len_rel,
                dictionary=dictionary
            )
            # Legacy doesn't accept ids at construction; try to set them if provided
            if ids is not None:
                try:
                    # Many builds expose .ids as a numpy int32 array
                    board.ids = np.array(ids, dtype=np.int32)
                except Exception:
                    pass
            return board

    @staticmethod
    def board_to_image(board, out_w_px: int, out_h_px: int,
                       margin_px: int, border_bits: int) -> np.ndarray:
        """Render a Board/CharucoBoard to grayscale using available API."""
        try:
            img = board.generateImage((out_w_px, out_h_px), margin_px, border_bits)
        except Exception:
            # Legacy API: draw(outSize=, marginSize=, borderBits=)
            img = board.draw(outSize=(out_w_px, out_h_px),
                             marginSize=margin_px, borderBits=border_bits)
        return img


# ---------------------------
# Page composition & export
# ---------------------------

class PageCanvas:
    """A white page (BGR) in pixel space, sized from paper mm and DPI.
    A white page canvas in pixel space that represents the physical page.
    You can place content (numpy images) onto it by mm coordinates.
    """
    def __init__(self, paper: PaperSize, dpi: int):
        self.paper = paper
        self.dpi = dpi
        self.width_px = UnitHelper.mm_to_px(paper.width_mm, dpi)
        self.height_px = UnitHelper.mm_to_px(paper.height_mm, dpi)
        # BGR canvas for OpenCV drawing
        self.img = np.full((self.height_px, self.width_px, 3), 255, np.uint8)

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
        x1, y1 = max(0, x_px), max(0, y_px)
        x2, y2 = min(self.width_px, x_px + resized.shape[1]), min(self.height_px, y_px + resized.shape[0])
        if x2 > x1 and y2 > y1:
            self.img[y1:y2, x1:x2] = resized[(y1 - y_px):(y2 - y_px), (x1 - x_px):(x2 - x_px)]

    def draw_mm_ruler(self, start_mm: float = 10.0, y_mm: Optional[float] = None, length_mm: float = 100.0) -> None:
        """Draw a 100 mm test ruler for quick scale verification."""
        if y_mm is None:
            y_mm = self.paper.height_mm - 10.0
        x0 = UnitHelper.mm_to_px(start_mm, self.dpi)
        y = UnitHelper.mm_to_px(y_mm, self.dpi)
        L = UnitHelper.mm_to_px(length_mm, self.dpi)
        cv2.line(self.img, (x0, y), (x0 + L, y), (0, 0, 0), 1)
        for mm in range(0, int(length_mm) + 1):
            x = x0 + UnitHelper.mm_to_px(mm, self.dpi)
            tick = 10 if mm % 10 == 0 else (6 if mm % 5 == 0 else 3)
            cv2.line(self.img, (x, y - tick), (x, y + tick), (0, 0, 0), 1)
            if mm % 10 == 0:
                cv2.putText(self.img, f"{mm} mm", (x + 2, y - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)


class Exporter:
    """Saves a PageCanvas as PNG (with DPI) or PDF (exact page size)."""
    def __init__(self, dpi: int = 300):
        self.dpi = dpi

    def save_png(self, page: PageCanvas, path: str) -> None:
        """
        Save the page canvas to a PNG with DPI metadata. This helps print software
        respect physical size (still choose 'Actual size' in the print dialog).
        """

        rgb = cv2.cvtColor(page.img, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(path, format="PNG", dpi=(self.dpi, self.dpi))

    def save_pdf(self, page: PageCanvas, path: str, paper: PaperSize) -> None:
        """
        Save a PDF page with exact physical dimensions. The whole composed page image is
        placed at 1:1 on a PDF page sized in mm.
        """
        if pdfcanvas is None:
            raise RuntimeError("reportlab is required for PDF export. Install: pip install reportlab")
        if ImageReader is None:
            raise RuntimeError("pillow is required for PDF export. Install: pip install pillow")

        w_pt = UnitHelper.mm_to_pdf_points(paper.width_mm)
        h_pt = UnitHelper.mm_to_pdf_points(paper.height_mm)
        pil = Image.fromarray(cv2.cvtColor(page.img, cv2.COLOR_BGR2RGB))
        c = pdfcanvas.Canvas(path, pagesize=(w_pt, h_pt))
        # Place image to fill the page exactly
        c.drawImage(ImageReader(pil), 0, 0, width=w_pt, height=h_pt,
                    preserveAspectRatio=False, mask='auto')
        c.showPage()
        c.save()


# ---------------------------
# Content generators
# ---------------------------

class MarkerGenerator:
    """
    Creates single ArUco markers as numpy images and annotates legend text
    onto the black border area (small white text).
    """
    def __init__(self, dictionary, dict_name: str, dpi: int):
        self.dictionary = dictionary
        self.dict_name = dict_name
        self.dpi = dpi

    @staticmethod
    def _snap_side_to_modules(target_side_px: int, bits: Optional[int], border_bits: int) -> int:
        """
        Ensure side_px is a multiple of the module count for crisp edges.
        If bits is unknown (e.g., AprilTag), just return target_side_px.
        Ensure the final side length is an integer multiple of the module size,
        so edges are crisp. Module count = bits + 2*border_bits.
        """
        if not bits:
            return target_side_px
        modules = bits + 2 * border_bits
        module_px = max(1, round(target_side_px / modules))
        return module_px * modules

    def draw_single(self, marker_id: int, size_mm: float,
                    border_bits: int = 2,
                    legend: Optional[str] = None) -> np.ndarray:
        """
        Generate a single ArUco marker image of exact physical size (when placed with DPI).
        The legend is written in white into the bottom black border area.

        Returns BGR image (OpenCV).
        """
        # Infer bits (4,5,6,7) from dict name, or use hint if provided.
        bits = DictionaryInfo.infer_bits_from_name(self.dict_name) or None

        # Compute target pixel side and snap to module multiple.
        side_px = UnitHelper.mm_to_px(size_mm, self.dpi)
        side_px = self._snap_side_to_modules(side_px, bits, border_bits)

        # Draw marker in grayscale then convert to BGR
        img = ArucoGen.generate_marker_img(self.dictionary, marker_id, side_px, border_bits)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Build default legend if not provided
        if legend is None:
            legend = f"id={marker_id} | dict={self.dict_name} | bits={bits or '?'} | size={size_mm:.2f}mm | border={border_bits}"

        # Height of the black border in px:
        if bits:
            modules = bits + 2 * border_bits
            module_px = side_px // modules
            border_px = border_bits * module_px
        else:
            # Conservative fallback
            border_px = max(6, side_px // 16)

        # Fit small white text into bottom border
        pad = max(2, border_px // 6)
        avail_w = side_px - 2 * pad
        avail_h = max(1, int(border_px * 0.8))

        scale, thick = 0.5, 1
        (tw, th), bl = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        while (tw > avail_w or th + bl > avail_h) and scale > 0.12:
            scale *= 0.92
            (tw, th), bl = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)

        x = max(pad, (side_px - tw) // 2)
        y = side_px - max(pad, (border_px - th) // 2) - bl
        cv2.putText(img, legend, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
        return img


class MultiMarkerSheet:
    """Lay out multiple annotated markers on a page in a grid."""
    def __init__(self, marker_gen: MarkerGenerator, page: PageCanvas):
        self.marker_gen = marker_gen
        self.page = page

    def render(self, ids: List[int], dictionary_name: str,
               marker_size_mm: float, rows: int, cols: int,
               spacing_mm: float, margin_mm: float,
               border_bits: int = 2) -> PageCanvas:
        """
        Create a grid of markers placed on the page with given spacing and margins.
        """
        x0, y0 = margin_mm, margin_mm

        # Pre-generate one marker to compute its pixel/module behavior (for crispness)
        # but we'll place by mm width on the page.
        for r in range(rows):
            for c in range(cols):
                i = r * cols + c
                if i >= len(ids):
                    continue
                m_id = ids[i]
                legend = f"id={m_id} | dict={dictionary_name} | size={marker_size_mm:.2f}mm"
                tile = self.marker_gen.draw_single(m_id, marker_size_mm, border_bits, legend=legend)
                x_mm = x0 + c * (marker_size_mm + spacing_mm)
                y_mm = y0 + r * (marker_size_mm + spacing_mm)
                self.page.place(tile, x_mm, y_mm, marker_size_mm)
        return self.page


class DiamondGenerator:
    """
    Generate a ChArUco diamond using the new recommended approach:
    CharucoBoard(Size(3,3), squareLength, markerLength, dictionary, ids=Vec4i),
    then board.generateImage(...). This mirrors current OpenCV tutorial guidance.
    """
    def __init__(self, dictionary, dict_name: str, dpi: int):
        self.dictionary = dictionary
        self.dict_name = dict_name
        self.dpi = dpi

    def draw(self, ids4: List[int], square_mm: float, marker_mm: float,
             margin_mm: float = 10.0,
             border_bits: int = 1,
             legend_lines: Optional[List[str]] = None) -> np.ndarray:
        """
        Create a ChArUco diamond image (BGR) sized so that one square equals 'square_mm'
        when placed at the corresponding mm width.
        """
        if len(ids4) != 4:
            raise ValueError("A ChArUco diamond needs exactly 4 marker IDs.")

        square_px = max(20, UnitHelper.mm_to_px(square_mm, self.dpi))
        marker_px = max(10, UnitHelper.mm_to_px(marker_mm, self.dpi))
        margin_px = max(0, UnitHelper.mm_to_px(margin_mm, self.dpi))

        # Board uses relative sizes; we pass 1.0 and the ratio.
        board = ArucoGen.make_charuco_board(3, 3,
                                            square_len_rel=1.0,
                                            marker_len_rel=float(marker_mm) / float(square_mm),
                                            dictionary=self.dictionary,
                                            ids=ids4)

        out_side = 3 * square_px + 2 * margin_px
        img = ArucoGen.board_to_image(board, out_side, out_side, margin_px, border_bits)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Legend defaults
        if legend_lines is None:
            legend_lines = [
                f"diamond ids={ids4}",
                f"dict={self.dict_name}",
                f"square={square_mm:.2f}mm | marker={marker_mm:.2f}mm"
            ]

        # Write small white text into black squares (3x3). Black squares: (x+y)%2==1 with TL white.
        H, W = img.shape[:2]
        cell = square_px
        font = cv2.FONT_HERSHEY_SIMPLEX
        targets = [(1, 0), (0, 1), (2, 1), (1, 2)]  # four distinct black squares
        for (sx, sy), text in zip(targets, legend_lines):
            if (sx + sy) % 2 == 0:
                sx = (sx + 1) % 3
            x0 = margin_px + sx * cell
            y0 = margin_px + sy * cell
            pad = max(2, cell // 10)
            avail_w = cell - 2 * pad
            avail_h = cell - 2 * pad
            # Fit tiny font into local area
            scale, thick = 0.5, 1
            (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
            while (tw > avail_w or th + bl > avail_h) and scale > 0.15:
                scale *= 0.92
                (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
            # Recenter roughly
            tx = int(x0 + (cell - tw) / 2)
            ty = int(y0 + (cell + th) / 2)
            cv2.putText(img, text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        return img


class CharucoBoardGenerator:
    """Generate standard ChArUco boards and write legends into black squares."""
    def __init__(self, dictionary, dict_name: str, dpi: int):
        self.dictionary = dictionary
        self.dict_name = dict_name
        self.dpi = dpi

    def draw(self, squares_x: int, squares_y: int,
             square_mm: float, marker_mm: float,
             margin_mm: float = 6.0, border_bits: int = 1,
             legend_lines: Optional[List[str]] = None) -> np.ndarray:
        """
        Create a ChArUco board image (BGR). The board image area (without page margins)
        will be squares_x*square_mm by squares_y*square_mm approximately.
        """
        # Convert to pixel units for the rendered board image
        square_px = max(20, UnitHelper.mm_to_px(square_mm, self.dpi))
        marker_px = max(10, UnitHelper.mm_to_px(marker_mm, self.dpi))
        margin_px = max(0, UnitHelper.mm_to_px(margin_mm, self.dpi))

        board = ArucoGen.make_charuco_board(
            squares_x, squares_y,
            square_len_rel=1.0,
            marker_len_rel=float(marker_mm) / float(square_mm),
            dictionary=self.dictionary
        )

        # OpenCV's draw uses 'outSize' and 'marginSize'. We'll build a board area and draw into it.
        out_w = squares_x * square_px + 2 * margin_px
        out_h = squares_y * square_px + 2 * margin_px
        img = ArucoGen.board_to_image(board, out_w, out_h, margin_px, border_bits)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Default legend
        if legend_lines is None:
            w_mm = squares_x * square_mm
            h_mm = squares_y * square_mm
            legend_lines = [
                f"{squares_x}x{squares_y} squares | dict={self.dict_name}",
                f"square={square_mm:.2f}mm | marker={marker_mm:.2f}mm",
                f"board={w_mm:.1f}×{h_mm:.1f} mm"
            ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        cell = square_px
        placed = 0
        # Write legend pieces into several black squares near corners/edges.
        # For a chessboard with top-left white, black squares are those where (x+y)%2==1.
        # We'll use (1,0), (0,1), (squares_x-2,squares_y-1), (squares_x-1,squares_y-2) if valid.
        candidates = [(1, 0), (0, 1), (squares_x - 2, squares_y - 1), (squares_x - 1, squares_y - 2)]
        for (sx, sy), text in zip(candidates, legend_lines):
            # If this square is not black, nudge to nearest black neighbor
            if sx < 0 or sy < 0 or sx >= squares_x or sy >= squares_y:
                continue
            if (sx + sy) % 2 == 0:
                sx = (sx + 1) % squares_x
            # Square rectangle in pixels
            x0 = margin_px + sx * cell
            y0 = margin_px + sy * cell

            # Fit text inside padding box
            pad = max(2, cell // 10)
            avail_w = cell - 2 * pad
            avail_h = cell - 2 * pad

            # Find scale to fit
            scale, thick = 0.5, 1
            (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
            while (tw > avail_w or th + bl > avail_h) and scale > 0.15:
                scale *= 0.92
                (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
            tx = int(x0 + (cell - tw) / 2)
            ty = int(y0 + (cell + th) / 2)
            cv2.putText(img, text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
            placed += 1

        return img


# ---------------------------
# CLI
# ---------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate ArUco markers, ChArUco diamonds and boards with exact physical sizing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--list-dicts", action="store_true", help="List available cv2.aruco dictionaries.")
    p.add_argument("--explain-dicts", action="store_true", help="Explain dictionary families and usage.")

    sub = p.add_subparsers(dest="mode", required=False)

    # Common paper/output args
    def add_paper_args(sp):
        sp.add_argument("--paper", choices=["A5", "A4", "A3", "custom"], default="A4", help="Paper size.")
        sp.add_argument("--landscape", action="store_true", help="Use landscape orientation.")
        sp.add_argument("--paper-mm", nargs=2, type=float, metavar=("WIDTH_MM", "HEIGHT_MM"), help="Custom paper size in mm (requires --paper custom).")
        sp.add_argument("--format", choices=["png", "pdf"], default="pdf", help="Output format.")
        sp.add_argument("--dpi", type=int, default=300, help="Raster DPI (affects PNG pixel size and internal raster for PDF).")
        sp.add_argument("--output", "-o", required=True, help="Output file path.")
        sp.add_argument("--draw-ruler", "--ruler", action="store_true", help="Draw a 100 mm ruler at bottom to verify scale.")

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
    sp_sheet.add_argument("--dictionary", "--dict", "-d", type=str, required=True, help="Dictionary name.")
    sp_sheet.add_argument("--marker-size-mm", "-m", type=float, required=True, help="Marker side length in mm.")
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
    return PaperSize((h, w)[args.landscape], (w, h)[args.landscape])


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_dicts:
        print("Available dictionaries in this OpenCV build:")
        for k in sorted(DictionaryInfo.list_available().keys()):
            print(k)
        return

    if args.explain_dicts:
        print(DictionaryInfo.explain())
        return

    if not args.mode:
        parser.print_help()
        return

    # Resolve paper and build page canvas
    paper = resolve_paper(args)
    page = PageCanvas(paper, dpi=args.dpi)
    exporter = Exporter(dpi=args.dpi)

    # Mode-specific generation
    if args.mode == "marker":
        dictionary, dict_name = DictionaryInfo.resolve(args.dictionary)
        gen = MarkerGenerator(dictionary, dict_name, dpi=args.dpi)

        # Create the marker tile (annotated)
        tile = gen.draw_single(marker_id=args.id,
                               size_mm=args.marker_size_mm,
                               border_bits=args.border_bits)

        # Center on page (with margin guard)
        x_mm = max(args.margin_mm, (paper.width_mm - args.marker_size_mm) / 2.0)
        y_mm = max(args.margin_mm, (paper.height_mm - args.marker_size_mm) / 2.0)
        page.place(tile, x_mm, y_mm, args.marker_size_mm)
        if args.draw_ruler:
            page.draw_mm_ruler()

        # Export
        (exporter.save_png if args.format == "png" else exporter.save_pdf)(page, args.output, paper)

    elif args.mode == "sheet":
        dictionary, dict_name = DictionaryInfo.resolve(args.dictionary)
        gen = MarkerGenerator(dictionary, dict_name, dpi=args.dpi)
        sheet = MultiMarkerSheet(gen, page)

        # Simple fit check
        total_w = args.margin_mm * 2 + args.cols * args.marker_size_mm + (args.cols - 1) * args.spacing_mm
        total_h = args.margin_mm * 2 + args.rows * args.marker_size_mm + (args.rows - 1) * args.spacing_mm
        if total_w > paper.width_mm + 1e-6 or total_h > paper.height_mm + 1e-6:
            raise ValueError(f"Grid does not fit: needs {total_w:.1f}×{total_h:.1f} mm, "
                             f"page is {paper.width_mm:.1f}×{paper.height_mm:.1f} mm.")

        sheet.render(ids=args.ids, dictionary_name=dict_name,
                     marker_size_mm=args.marker_size_mm,
                     rows=args.rows, cols=args.cols,
                     spacing_mm=args.spacing_mm, margin_mm=args.margin_mm,
                     border_bits=args.border_bits)
        if args.draw_ruler:
            page.draw_mm_ruler()
        (exporter.save_png if args.format == "png" else exporter.save_pdf)(page, args.output, paper)

    elif args.mode == "diamond":
        dictionary, dict_name = DictionaryInfo.resolve(args.dictionary)
        gen = DiamondGenerator(dictionary, dict_name, dpi=args.dpi)
        tile = gen.draw(ids4=args.ids, square_mm=args.square_mm, marker_mm=args.marker_mm,
                        margin_mm=args.margin_mm, border_bits=args.border_bits)

        # Place centered, respecting margin
        width_mm = min(paper.width_mm - 2 * args.margin_mm, 3.0 * args.square_mm + 2 * args.margin_mm)
        x_mm = (paper.width_mm - width_mm) / 2.0
        # Height == width for 3x3
        page.place(tile, x_mm, (paper.height_mm - width_mm) / 2.0, width_mm)
        if args.draw_ruler:
            page.draw_mm_ruler()
        (exporter.save_png if args.format == "png" else exporter.save_pdf)(page, args.output, paper)

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
            raise ValueError(f"Board ({board_w_mm:.1f}×{board_h_mm:.1f} mm) exceeds page "
                             f"({paper.width_mm:.1f}×{paper.height_mm:.1f} mm).")
        x_mm = (paper.width_mm - board_w_mm) / 2.0
        y_mm = (paper.height_mm - board_h_mm) / 2.0
        page.place(tile, x_mm, y_mm, board_w_mm)
        if args.draw_ruler:
            page.draw_mm_ruler()
        (exporter.save_png if args.format == "png" else exporter.save_pdf)(page, args.output, paper)

    else:
        parser.error(f"Unknown mode {args.mode!r}")


if __name__ == "__main__":
    main()
