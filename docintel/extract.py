"""PDF -> text per page. Scanned pages are detected and OCR'd with ocrmypdf (Tesseract)."""

import logging
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pymupdf

log = logging.getLogger(__name__)

# A page with less text than this but with images on it is treated as a scan.
MIN_TEXT_CHARS = 40
# Hindi text layers in Indian PDFs are often unusable: the font maps conjunct glyphs to
# stray Latin-Extended / private-use code points ("निविदा" comes out as "न ȡ व ȡ दा").
# Such pages are re-read with OCR instead of trusting the text layer.
GARBLE_THRESHOLD = 0.05


def garbled_devanagari(text: str) -> bool:
    deva = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    if deva < 20:
        return False
    junk = sum(
        1
        for c in text
        if "\u0180" <= c <= "\u024f" or "\ue000" <= c <= "\uf8ff" or "\u0300" <= c <= "\u036f"
    )
    return junk / deva > GARBLE_THRESHOLD


@dataclass
class Page:
    number: int  # 1-based, as people cite pages
    text: str
    ocr: bool = False


@dataclass
class Extraction:
    pages: list[Page]
    ocr_pages: int


class OcrUnavailable(RuntimeError):
    pass


_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def clean_text(text: str) -> str:
    """NFKC turns ligatures back into letters ("ﬁnancial" -> "financial", "oﬃce" ->
    "office"); without it BM25 cannot match those words at all. Words hyphenated across
    a line break are rejoined ("e-\nprocurement" -> "e-procurement")."""
    return _HYPHEN_BREAK.sub(r"\1-\2", unicodedata.normalize("NFKC", text))


def _read(path: Path) -> list[tuple[str, bool]]:
    """(text, looks_scanned) per page."""
    out = []
    with pymupdf.open(path) as doc:
        for page in doc:
            text = page.get_text("text", sort=True)
            garbled = garbled_devanagari(text)  # check before NFKC rewrites the junk
            text = clean_text(text)
            scanned = len(text.strip()) < MIN_TEXT_CHARS and bool(page.get_images(full=False))
            out.append((text, scanned or garbled))
    return out


def ocr_pdf(src: Path, dest: Path, languages: str = "eng+hin", *, force: bool = False) -> None:
    """Run ocrmypdf. --skip-text leaves pages that already have text alone; --force-ocr
    re-reads pages whose text layer exists but is garbage."""
    if not shutil.which("ocrmypdf"):
        raise OcrUnavailable("ocrmypdf is not installed (it is in the DocIntel Docker image)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ocrmypdf",
            "--force-ocr" if force else "--skip-text",
            "--language",
            languages,
            "--output-type",
            "pdf",
            "--quiet",
            str(src),
            str(dest),
        ],
        check=True,
        timeout=600,
    )


def extract(path: Path, *, ocr_dir: Path | None = None) -> Extraction:
    pages = _read(path)
    scanned = [i for i, (_, s) in enumerate(pages) if s]
    ocr_done: set[int] = set()
    if scanned and ocr_dir is not None:
        target = ocr_dir / f"{path.stem}.ocr.pdf"
        try:
            # Garbled text layers need --force-ocr; plain scans only need --skip-text.
            ocr_pdf(path, target, force=any(garbled_devanagari(pages[i][0]) for i in scanned))
            redone = _read(target)
            for i in scanned:
                if len(redone[i][0].strip()) >= MIN_TEXT_CHARS and not garbled_devanagari(
                    redone[i][0]
                ):
                    pages[i] = (redone[i][0], True)
                    ocr_done.add(i)
        except OcrUnavailable as exc:
            log.warning("%s: %d scanned pages, but %s", path.name, len(scanned), exc)
    elif scanned:
        log.warning("%s: %d scanned pages and OCR disabled", path.name, len(scanned))
    return Extraction(
        pages=[Page(i + 1, text, ocr=i in ocr_done) for i, (text, _) in enumerate(pages)],
        ocr_pages=len(ocr_done),
    )
