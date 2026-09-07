import io
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None


# =========================
# Configuration Windows
# =========================
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"

if pytesseract is not None:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# =========================
# Formats métier
# =========================
AMOUNT_PATTERN = (
    r"[-+]?\d{1,3}(?:\.\d{3})*,\d{2}"
    r"|[-+]?\d+,\d{2}"
)

DATE_PATTERN = r"\d{1,2}/\d{1,2}/\d{2,4}"
PERCENT_PATTERN = r"\d{1,2}(?:[.,]\d{1,2})?"


# =========================
# Data model OCR
# =========================
@dataclass
class OCRBlock:
    text: str
    x: int
    y: int
    w: int
    h: int
    confidence: float


# =========================
# Outils généraux
# =========================
def normalize_text(text: str) -> str:
    """
    Normalise les accents, espaces et caractères spéciaux
    pour faciliter les comparaisons.
    """
    value = text or ""

    replacements = [
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
        ("ü", "u"),
        ("ñ", "n"),
        ("Á", "A"),
        ("É", "E"),
        ("Í", "I"),
        ("Ó", "O"),
        ("Ú", "U"),
        ("Ü", "U"),
        ("Ñ", "N"),
        ("º", "o"),
        ("°", "o"),
        ("ª", "a"),
    ]

    for source, target in replacements:
        value = value.replace(source, target)

    value = re.sub(
        r"[\u2010\u2011\u2012\u2013\u2014\u2015\ufe58\ufe63\uff0d]",
        "-",
        value,
    )

    return re.sub(r"\s+", " ", value).strip().lower()


def clean_lines(text: str) -> List[str]:
    """
    Nettoie les lignes sans détruire leur ordre.
    """
    return [
        re.sub(r"\s+", " ", line).strip()
        for line in (text or "").splitlines()
        if line.strip()
    ]


def clean_amount(value: Optional[str]) -> Optional[str]:
    """
    Retire les symboles monétaires et les espaces.
    """
    if value is None:
        return None

    return re.sub(
        r"[€$£\s\u00a0]",
        "",
        value,
    ).strip()


def amount_to_float(value: Optional[str]) -> Optional[float]:
    """
    Convertit un montant européen en float.

    Exemple :
    1.004,67 -> 1004.67
    """
    cleaned = clean_amount(value)

    if not cleaned:
        return None

    normalized = cleaned.replace(".", "").replace(",", ".")

    try:
        return float(normalized)
    except ValueError:
        return None


def format_european_amount(value: Optional[float]) -> Optional[str]:
    """
    Convertit un float en montant européen.

    Exemple :
    1004.67 -> 1.004,67
    """
    if value is None:
        return None

    formatted = f"{value:,.2f}"

    return (
        formatted
        .replace(",", "#")
        .replace(".", ",")
        .replace("#", ".")
    )


def normalize_percentage(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    return (
        value
        .replace(".", ",")
        .replace("%", "")
        .strip()
    )


def first_regex(
    pattern: str,
    text: str,
    flags: int = re.IGNORECASE | re.DOTALL,
    group: int = 1,
) -> Optional[str]:
    """
    Retourne le premier résultat correspondant au pattern.
    """
    match = re.search(pattern, text, flags)

    if not match:
        return None

    return match.group(group).strip()


def last_regex(
    pattern: str,
    text: str,
    flags: int = re.IGNORECASE | re.DOTALL,
    group: int = 1,
) -> Optional[str]:
    """
    Retourne le dernier résultat correspondant au pattern.
    """
    matches = list(re.finditer(pattern, text, flags))

    if not matches:
        return None

    return matches[-1].group(group).strip()


def field(
    value: Optional[str],
    confidence: float,
    source: str,
) -> Dict[str, object]:
    """
    Crée la structure JSON d'un champ extrait.
    """
    return {
        "value": value,
        "confidence": round(
            confidence if value is not None else 0.0,
            3,
        ),
        "source": (
            source
            if value is not None
            else "not_detected"
        ),
    }


# =========================
# Images et prétraitement
# =========================
def pil_to_cv2(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Prétraitement utilisé uniquement pour le fallback OCR :
    gris + réduction du bruit + binarisation Otsu.
    """
    image_cv = pil_to_cv2(image)

    gray = cv2.cvtColor(
        image_cv,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]

    return Image.fromarray(binary)


def load_pages_from_bytes(
    file_bytes: bytes,
    filename: str,
) -> List[Image.Image]:
    """
    Convertit le PDF en images pour la visualisation et l'OCR fallback.
    """
    suffix = filename.lower()

    if suffix.endswith(".pdf"):
        if convert_from_bytes is None:
            raise RuntimeError(
                "Le paquet 'pdf2image' n'est pas installé."
            )

        if not os.path.exists(POPPLER_PATH):
            raise RuntimeError(
                f"Poppler est introuvable : {POPPLER_PATH}"
            )

        return convert_from_bytes(
            file_bytes,
            dpi=300,
            poppler_path=POPPLER_PATH,
        )

    return [
        Image.open(
            io.BytesIO(file_bytes)
        ).convert("RGB")
    ]


# =========================
# Lecture directe du PDF
# =========================
def extract_embedded_pdf_text(file_bytes: bytes) -> str:
    """
    Lit le texte natif intégré au PDF avec PyMuPDF.

    Cette méthode est prioritaire car elle évite les erreurs OCR
    comme 1 lu comme 4 ou les colonnes mélangées.
    """
    if fitz is None:
        return ""

    try:
        document = fitz.open(
            stream=file_bytes,
            filetype="pdf",
        )

        # sort=True ynadhem les blocs de texte par position
        # verticale puis horizontale (ordre de lecture ).
        # Menghir hetha, PyMuPDF yguid le texte f l'ordre du flux
        # interne du PDF, qui peut mélanger deux encadrés côte à
        # côte (FACTURA / Protocolo) selon le générateur du PDF.
        return "\n".join(
            page.get_text("text", sort=True)
            for page in document
        )

    except Exception:
        return ""


# =========================
# OCR fallback
# =========================
def run_tesseract_ocr_raw(
    image: Image.Image,
    psm: int = 11,
) -> List[OCRBlock]:
    """
    Retourne les mots et leurs positions.
    Cette partie sert surtout à la visualisation.
    """
    if pytesseract is None:
        raise RuntimeError(
            "Le paquet 'pytesseract' n'est pas installé."
        )

    if not os.path.exists(TESSERACT_CMD):
        raise RuntimeError(
            f"Tesseract est introuvable : {TESSERACT_CMD}"
        )

    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config=f"--oem 3 --psm {psm} -l spa+eng",
    )

    blocks: List[OCRBlock] = []

    for index, raw_text in enumerate(data["text"]):
        text = str(raw_text).strip()

        if not text:
            continue

        try:
            confidence = max(
                0.0,
                min(
                    1.0,
                    float(data["conf"][index]) / 100.0,
                ),
            )
        except Exception:
            confidence = 0.0

        blocks.append(
            OCRBlock(
                text=text,
                x=int(data["left"][index]),
                y=int(data["top"][index]),
                w=int(data["width"][index]),
                h=int(data["height"][index]),
                confidence=confidence,
            )
        )

    return blocks


def merge_nearby_words(
    blocks: List[OCRBlock],
    y_tolerance: int = 12,
    x_gap: int = 55,
) -> List[OCRBlock]:
    """
    Fusion prudente des mots proches sur une même ligne.

    x_gap=55 évite de fusionner plusieurs colonnes éloignées.
    """
    if not blocks:
        return []

    ordered = sorted(
        blocks,
        key=lambda block: (block.y, block.x),
    )

    merged: List[OCRBlock] = []
    current = ordered[0]

    for candidate in ordered[1:]:
        same_line = (
            abs(candidate.y - current.y)
            <= y_tolerance
        )

        close_horizontally = (
            0
            <= candidate.x - (current.x + current.w)
            <= x_gap
        )

        if same_line and close_horizontally:
            x1 = min(current.x, candidate.x)
            y1 = min(current.y, candidate.y)
            x2 = max(
                current.x + current.w,
                candidate.x + candidate.w,
            )
            y2 = max(
                current.y + current.h,
                candidate.y + candidate.h,
            )

            current = OCRBlock(
                text=(
                    f"{current.text} {candidate.text}"
                    .strip()
                ),
                x=x1,
                y=y1,
                w=x2 - x1,
                h=y2 - y1,
                confidence=(
                    current.confidence
                    + candidate.confidence
                ) / 2.0,
            )

        else:
            merged.append(current)
            current = candidate

    merged.append(current)

    return merged


def draw_boxes(
    image: Image.Image,
    blocks: List[OCRBlock],
    minimum_confidence: float,
) -> Image.Image:
    visual = image.convert("RGB").copy()
    draw = ImageDraw.Draw(visual)

    for block in blocks:
        if block.confidence < minimum_confidence:
            continue

        draw.rectangle(
            [
                block.x,
                block.y,
                block.x + block.w,
                block.y + block.h,
            ],
            outline=(255, 0, 0),
            width=2,
        )

    return visual


def ocr_pages_to_text(
    pages: List[Image.Image],
    apply_preprocessing: bool,
) -> str:
    """
    OCR de secours pour les scans ou images.

    Toutes les pages sont traitées, pas seulement la page affichée.
    """
    if pytesseract is None:
        return ""

    page_texts: List[str] = []

    for page in pages:
        source = (
            preprocess_image(page)
            if apply_preprocessing
            else page
        )

        text = pytesseract.image_to_string(
            source,
            config="--oem 3 --psm 11 -l spa+eng",
        )

        page_texts.append(text)

    return "\n".join(page_texts)


# =========================
# Détection de la famille
# =========================
def detect_document_family(document_text: str) -> str:
    """
    Distingue les factures de registre et les factures notariales.
    """
    normalized = normalize_text(document_text)

    registry_markers = [
        "registro de la propiedad",
        "el registrador",
        "registrador titular",
    ]

    if any(
        marker in normalized
        for marker in registry_markers
    ):
        return "registry"

    return "notary"


# =========================
# Extraction facture registre
# =========================
def extract_registry_fields(
    document_text: str,
) -> Dict[str, object]:
    lines = clean_lines(document_text)
    text = "\n".join(lines)

    issuer_name: Optional[str] = None

    for index, line in enumerate(lines):
        if re.fullmatch(
            r"El Registrador(?: Titular)?",
            line,
            re.IGNORECASE,
        ):
            if index + 1 < len(lines):
                issuer_name = lines[index + 1]
                break

    nif_cif = first_regex(
        r"N\.?\s*I\.?\s*F\.?\s*:"
        r"\s*([A-Z]?\d{7,8}[A-Z0-9]?)",
        text,
    )

    if nif_cif:
        nif_cif = nif_cif.upper()

    invoice_header = re.search(
        rf"SERIE\s*\n"
        rf"NUM FACTURA\s*\n"
        rf"FECHA\s*\n"
        rf"([A-Z])\s*\n"
        rf"(\d+)\s*\n"
        rf"({DATE_PATTERN})",
        text,
        re.IGNORECASE,
    )

    invoice_number = None
    invoice_date = None

    if invoice_header:
        invoice_number = (
            f"{invoice_header.group(1).upper()} "
            f"{invoice_header.group(2)}"
        )

        invoice_date = invoice_header.group(3)

    else:
        flattened = re.sub(
            r"\s+",
            " ",
            text,
        )

        invoice_header = re.search(
            rf"SERIE\s+"
            rf"NUM FACTURA\s+"
            rf"FECHA\s+"
            rf"([A-Z])\s+"
            rf"(\d+)\s+"
            rf"({DATE_PATTERN})",
            flattened,
            re.IGNORECASE,
        )

        if invoice_header:
            invoice_number = (
                f"{invoice_header.group(1).upper()} "
                f"{invoice_header.group(2)}"
            )

            invoice_date = invoice_header.group(3)

    protocol_number = first_regex(
        r"N[ºo°]\s*Protocolo\s*:"
        r"\s*(\d+\s*/\s*\d{4})",
        text,
    )

    if protocol_number:
        protocol_number = re.sub(
            r"\s+",
            " ",
            protocol_number,
        )

    base_amount = first_regex(
        rf"BASE IMPONIBLE"
        rf"\s*\n?\s*"
        rf"({AMOUNT_PATTERN})"
        rf"\s*€?",
        text,
    )

    iva_match = re.search(
        rf"IMPORTE\s+"
        rf"I\.?\s*V\.?\s*A\.?"
        rf"\s*\(\s*"
        rf"({PERCENT_PATTERN})"
        rf"\s*%\s*\)"
        rf"\s*\n?\s*"
        rf"({AMOUNT_PATTERN})"
        rf"\s*€?",
        text,
        re.IGNORECASE,
    )

    iva_percentage = (
        normalize_percentage(
            iva_match.group(1)
        )
        if iva_match
        else None
    )

    iva_amount = (
        iva_match.group(2)
        if iva_match
        else None
    )

    retention_match = re.search(
        rf"IMPORTE\s+"
        rf"(?:IRPF|RETENCI[ÓO]N)"
        rf"\s*\(\s*"
        rf"({PERCENT_PATTERN})"
        rf"\s*%\s*\)"
        rf"\s*\n?\s*"
        rf"({AMOUNT_PATTERN})"
        rf"\s*€?",
        text,
        re.IGNORECASE,
    )

    if retention_match:
        retention_percentage = normalize_percentage(
            retention_match.group(1)
        )

        retention_float = amount_to_float(
            retention_match.group(2)
        )

        retention_amount = format_european_amount(
            abs(retention_float)
            if retention_float is not None
            else None
        )

        retention_source = "direct_pdf_text"
        retention_confidence = 0.99

    else:
        retention_percentage = "0,00"
        retention_amount = "0,00"
        retention_source = "default_absent_label"
        retention_confidence = 0.85

    # Règle métier :
    # sans base de rétention spécifique,
    # la base de rétention = base imposable.
    retention_base_amount = base_amount

    non_taxable_amount = first_regex(
        rf"(?:"
        rf"IMPORTE\s+NO\s+SUJETO"
        rf"|BASE\s+EXENTA\s+IVA"
        rf")"
        rf"\s*\n?\s*"
        rf"({AMOUNT_PATTERN})"
        rf"\s*€?",
        text,
    )

    if non_taxable_amount is None:
        non_taxable_amount = "0,00"
        non_taxable_source = "default_absent_label"
        non_taxable_confidence = 0.85

    else:
        non_taxable_source = "direct_pdf_text"
        non_taxable_confidence = 0.99

    net_amount = first_regex(
        rf"(?m)^TOTAL"
        rf"\s*\n?\s*"
        rf"({AMOUNT_PATTERN})"
        rf"\s*€?",
        text,
    )

    return {
        "document_family": "registry",

        "invoice_reference": {
            "issuer_name": field(
                issuer_name,
                0.99,
                "header_after_registrador",
            ),

            "nif_cif": field(
                nif_cif,
                0.99,
                "first_issuer_nif",
            ),

            "protocol_number": field(
                protocol_number,
                0.99,
                "direct_pdf_text",
            ),

            "invoice_number": field(
                invoice_number,
                0.99,
                "series_number_header",
            ),

            "invoice_date": field(
                invoice_date,
                0.99,
                "series_number_header",
            ),
        },

        "financial": {
            "base_amount": field(
                base_amount,
                0.99,
                "summary_label",
            ),

            "retention_base_amount": field(
                retention_base_amount,
                0.90,
                "business_default_base_amount",
            ),

            "iva_percentage": field(
                iva_percentage,
                0.99,
                "summary_label",
            ),

            "retention_percentage": field(
                retention_percentage,
                retention_confidence,
                retention_source,
            ),

            "iva_amount": field(
                iva_amount,
                0.99,
                "summary_label",
            ),

            "retention_amount": field(
                retention_amount,
                retention_confidence,
                retention_source,
            ),

            "non_taxable_amount": field(
                non_taxable_amount,
                non_taxable_confidence,
                non_taxable_source,
            ),

            "net_amount": field(
                net_amount,
                0.99,
                "total_summary_label",
            ),
        },
    }

def extract_reference_numbers(
    text: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Identifie le numéro de protocole et le numéro de facture par leur
    FORME plutôt que par leur position relative à un label — car le
    texte natif du PDF peut mélanger l'ordre visuel de deux encadrés
    côte à côte (FACTURA / Protocolo) selon le générateur du PDF.
    """
    invoice_number = first_regex(
        r"(\d{4,10}-[A-Za-z]{1,4})",
        text,
    )

    invoice_prefix = None
    if invoice_number:
        prefix_match = re.match(
            r"(\d{4,10})",
            invoice_number,
        )
        if prefix_match:
            invoice_prefix = prefix_match.group(1)

    # Le protocole est cherché dans une fenêtre proche du mot
    # "Protocolo" plutôt que n'importe où dans le texte — sinon un
    # numéro NIF suivi d'une lettre (ex: "74913735K") est aspiré par
    # erreur, puisque la lettre n'est pas un chiffre et passe le test
    # de "nombre autonome" avant même d'atteindre le vrai protocole.
    protocol_number = None

    for protocol_label in re.finditer(r"Protocolo", text, re.IGNORECASE):
        window = text[protocol_label.end():protocol_label.end() + 40]
        candidate = first_regex(r"(?<!\d)(\d{5,10})(?!\d)", window)

        if candidate and candidate != invoice_prefix:
            protocol_number = candidate
            break

    if protocol_number is None:
        # Repli : nombre autonome dans tout le texte, en excluant
        # explicitement le NIF/CIF et les numéros de téléphone/fax,
        # qui ont la même forme numérique qu'un protocole.
        excluded_numbers = set()

        for excluded_match in re.finditer(
            r"(?:NIF|CIF|Tlf|Fax)\s*:?\s*([A-Za-z]?\d{7,9})",
            text,
            re.IGNORECASE,
        ):
            digits_only = re.sub(r"\D", "", excluded_match.group(1))
            excluded_numbers.add(digits_only)

        plain_numbers = [
            n for n in re.findall(r"(?<!\d)(\d{5,10})(?!\d)", text)
            if n != invoice_prefix and n not in excluded_numbers
        ]

        protocol_number = (
            plain_numbers[0]
            if plain_numbers
            else invoice_prefix
        )

    return protocol_number, invoice_number


def extract_latest_date(text: str, window: int = 300) -> Optional[str]:
    """
    nrecuperiw ekher date fi l'en-tête du document.
    La facture est toujours émise le jour de la signature du
    protocole ou après donc f cas d'ambiguïté d'ordre, la date
    la plus tardive heya la Fecha Factura.
    """
    header = text[:window]
    dates = re.findall(DATE_PATTERN, header)

    if not dates:
        return None

    def sort_key(date_str: str):
        day, month, year = re.split(r"/", date_str)
        year = int(year)
        year += 2000 if year < 100 else 0
        return (year, int(month), int(day))

    return max(dates, key=sort_key)


def extract_issuer_name(text: str) -> Optional[str]:
    """
    Extrait le nom du notaire ligne par ligne.
    Un nom répété dans l'en-tête et près de la signature
    est prioritaire sur une ville comme FUENGIROLA MÁLAGA.
    """
    lines = clean_lines(text)

    prepared_lines = []

    for line in lines:
        candidate = re.sub(
            r"^(?:FACTURA|EL\s+NOTARIO|NOTARIO)"
            r"\s*[:.\-]?\s*",
            "",
            line,
            flags=re.IGNORECASE,
        ).strip()

        prepared_lines.append(candidate)

    name_pattern = re.compile(
        r"[A-ZÁÉÍÓÚÑÜ]{2,}"
        r"(?:[-'’][A-ZÁÉÍÓÚÑÜ]{2,})?"
        r"(?:\s+[A-ZÁÉÍÓÚÑÜ]{2,}"
        r"(?:[-'’][A-ZÁÉÍÓÚÑÜ]{2,})?){1,5}"
    )

    forbidden_words = {
        "factura", "notario", "notaria", "protocolo",
        "nif", "cif", "tlf", "fax", "avda", "avenida",
        "calle", "europa", "base", "total", "retencion",
        "desglose", "suplidos",
    }

    notario_indexes = [
        index
        for index, line in enumerate(lines)
        if re.fullmatch(
            r"(?:EL\s+)?NOTARIO\.?",
            line,
            re.IGNORECASE,
        )
    ]

    candidates = []

    for index, candidate in enumerate(prepared_lines):
        if not name_pattern.fullmatch(candidate):
            continue

        normalized = normalize_text(candidate)
        words = normalized.split()

        if any(word in forbidden_words for word in words):
            continue

        occurrences = sum(
            normalize_text(line) == normalized
            for line in prepared_lines
        )

        distance_to_notario = min(
            (
                abs(index - notario_index)
                for notario_index in notario_indexes
            ),
            default=99,
        )

        near_identifier = any(
            re.search(
                r"\b(?:NIF|CIF|Tlf|Fax)\b",
                lines[position],
                re.IGNORECASE,
            )
            for position in range(
                max(0, index - 3),
                min(len(lines), index + 4),
            )
        )

        if (
            occurrences < 2
            and distance_to_notario > 4
            and not near_identifier
        ):
            continue

        score = (
            occurrences * 100
            + (40 if near_identifier else 0)
            + max(0, 30 - distance_to_notario * 5)
            + len(words)
        )

        candidates.append((score, candidate))

    if not candidates:
        return None

    return max(candidates, key=lambda item: item[0])[1]

def extract_suplidos_total(
    text: str,
    max_block_lines: int = 12,
) -> Optional[str]:
    """
    Total des suplidos (débours) : correspond au montant non sujeto
    quand la facture n'expose pas ce libellé explicitement.

    Le libellé doit occuper toute une ligne ("Suplidos" ou
    "Desglose Suplidos"), pour ne pas confondre avec l'en-tête
    d'arancel "SUPLIDOS Y DERECHOS (REAL DECRETO ...)".

    Formats gérés :
      - libellé + montant sur la même ligne   -> montant direct
      - bloc avec "Total" sans deux-points    -> ce total, non ressommé
      - bloc multi-lignes sans total          -> somme des montants

    Le bloc s'arrête au premier libellé de synthèse globale
    (TOTAL:, BASE, RETENCION, IMPORTE, IVA, LIQUIDO, ARANCELES)
    ou après max_block_lines lignes.
    """
    lines = clean_lines(text)

    label_pattern = re.compile(
        rf"^(?:Desglose\s+)?Suplidos\s*:?\s*({AMOUNT_PATTERN})?\s*€?$",
        re.IGNORECASE,
    )

    # Testé AVANT block_total_pattern : "TOTAL:" (avec deux-points)
    # est le total global de la facture, pas celui du bloc suplidos.
    section_stop_pattern = re.compile(
        r"^(?:TOTAL\s*:|BASE\b|RETENCI[ÓO]N|IMPORTE|IVA\b|L[ÍI]QUIDO|ARANCELES)",
        re.IGNORECASE,
    )

    block_total_pattern = re.compile(
        r"^Total\b(?!\s*:)",
        re.IGNORECASE,
    )

    label_index = None

    for index, line in enumerate(lines):
        label_match = label_pattern.match(line)

        if label_match:
            if label_match.group(1):
                return label_match.group(1)

            label_index = index
            break

    if label_index is None:
        return None

    block_lines: List[str] = []
    explicit_total_line: Optional[str] = None

    for line in lines[label_index + 1 : label_index + 1 + max_block_lines]:
        if section_stop_pattern.match(line):
            break

        if block_total_pattern.match(line):
            explicit_total_line = line
            break

        block_lines.append(line)

    if explicit_total_line:
        return first_regex(
            rf"({AMOUNT_PATTERN})",
            explicit_total_line,
        )

    amounts = re.findall(
        AMOUNT_PATTERN,
        "\n".join(block_lines),
    )

    if not amounts:
        return None

    total = sum(
        amount_to_float(amount) or 0.0
        for amount in amounts
    )

    return format_european_amount(total)

# =========================
# Extraction facture notaire
# =========================
def extract_notary_fields(
    document_text: str,
) -> Dict[str, object]:
    lines = clean_lines(document_text)
    text = "\n".join(lines)

    # Le nom du notaire est cherché près du mot NOTARIO plutôt que
    # pris comme première ligne — voir extract_issuer_name.
    issuer_name = extract_issuer_name(text)

    if issuer_name is None:
        issuer_name = (
            lines[0]
            if lines
            else None
        )

    nif_cif = first_regex(
        r"(?:"
        r"C\.?\s*I\.?\s*F\.?"
        r"|N\.?\s*I\.?\s*F\.?"
        r")"
        r"\s*[:.]?\s*"
        r"([A-Z]?\d{7,8}[A-Z0-9]?)",
        text,
    )

    if nif_cif:
        nif_cif = nif_cif.upper()

    protocol_number, invoice_number = extract_reference_numbers(text)

    if invoice_number is None:
        invoice_number = first_regex(
            r"(?m)^FACTURA"
            r"\s*\n?\s*"
            r"(\d{4,10}-[A-Z]{1,4})",
            text,
        )

    invoice_date = first_regex(
        rf"Fecha\s+Factura"
        rf"\s*:\s*"
        rf"({DATE_PATTERN})",
        text,
    )

    if invoice_date is None:
        invoice_date = extract_latest_date(text)


        invoice_date = first_regex(
            rf"{re.escape(invoice_number)}"
            rf"\s+({DATE_PATTERN})",
            re.sub(
                r"\s+",
                " ",
                text,
            ),
        )

    # Layout notarial avec les titres puis les valeurs en dessous.
    summary_match = re.search(
        rf"Base Exenta IVA"
        rf"\s*\n"
        rf"Base imponible"
        rf"\s*\n"
        rf"Impuestos"
        rf"\s*\n"
        rf"({AMOUNT_PATTERN})"
        rf"\s*€?"
        rf"\s*\n"
        rf"({AMOUNT_PATTERN})"
        rf"\s*€?"
        rf"\s*\n"
        rf"IVA\s*\(\s*"
        rf"({PERCENT_PATTERN})"
        rf"\s*%\s*\)"
        rf"\s*\n"
        rf"({AMOUNT_PATTERN})"
        rf"\s*€?",
        text,
        re.IGNORECASE,
    )

    if summary_match:
        non_taxable_amount = summary_match.group(1)
        base_amount = summary_match.group(2)

        iva_percentage = normalize_percentage(
            summary_match.group(3)
        )

        iva_amount = summary_match.group(4)

    else:
        # Fallback pour les factures du type :
        # BASE 820,51
        # IVA[21%](B.Imponible: 820,51) 172,31

        iva_line_match = re.search(
            rf"IVA\s*[\[(]?\s*({PERCENT_PATTERN})\s*%\s*[\])]?"
            rf"\s*\(?\s*B\.?\s*Imponible\s*:?\s*({AMOUNT_PATTERN})\s*\)?"
            rf"[^\n]*?({AMOUNT_PATTERN})\s*€?\s*$",
            text,
            re.IGNORECASE | re.MULTILINE,
        )

        if iva_line_match:
            iva_percentage = normalize_percentage(iva_line_match.group(1))
            base_amount = iva_line_match.group(2)
            iva_amount = iva_line_match.group(3)
        else:
            iva_percentage = None
            base_amount = None
            iva_amount = None

        non_taxable_amount = first_regex(
            rf"(?:"
            rf"IMPORTE\s+NO\s+SUJETO"
            rf"|BASE\s+EXENTA\s+IVA"
            rf")"
            rf"\s*[:\-]?\s*"
            rf"({AMOUNT_PATTERN})",
            text,
        )

        if non_taxable_amount is None:
            non_taxable_amount = extract_suplidos_total(text)

        if non_taxable_amount is None:
            non_taxable_amount = "0,00"

    retention_match = re.search(
        rf"({AMOUNT_PATTERN})"
        rf"\s*€?"
        rf"\s*\n"
        rf"RETENCI[ÓO]N"
        rf"\s*\(\s*"
        rf"({PERCENT_PATTERN})"
        rf"\s*%\s*\)"
        rf"\s*\n"
        rf"({AMOUNT_PATTERN})"
        rf"\s*€?",
        text,
        re.IGNORECASE,
    )

    if retention_match:
        retention_base_amount = retention_match.group(1)

        retention_percentage = normalize_percentage(
            retention_match.group(2)
        )

        retention_float = amount_to_float(
            retention_match.group(3)
        )

        retention_amount = format_european_amount(
            abs(retention_float)
            if retention_float is not None
            else None
        )

    else:
        # Fallback pour :
        # RETENCION:15% (B.Imponible 820,51) -123,08
        retention_inline = re.search(
            rf"RETENCI[ÓO]N"
            rf"\s*[:\-]?\s*"
            rf"({PERCENT_PATTERN})"
            rf"\s*%"
            rf"[^\n]*?"
            rf"({AMOUNT_PATTERN})"
            rf"[^\n]*?"
            rf"({AMOUNT_PATTERN})"
            rf"\s*$",
            text,
            re.IGNORECASE | re.MULTILINE,
        )

        if retention_inline:
            retention_percentage = normalize_percentage(
                retention_inline.group(1)
            )

            retention_base_amount = (
                retention_inline.group(2)
            )

            retention_float = amount_to_float(
                retention_inline.group(3)
            )

            retention_amount = format_european_amount(
                abs(retention_float)
                if retention_float is not None
                else None
            )

        else:
            retention_base_amount = base_amount
            retention_percentage = "0,00"
            retention_amount = "0,00"

    # Selon la définition métier du boss,
    # le net final correspond au montant à payer.
    net_amount = first_regex(
        rf"IMPORTE TOTAL"
        rf"\s*€?"
        rf"\s*\n?\s*"
        rf"({AMOUNT_PATTERN})"
        rf"\s*€?",
        text,
    )

    if net_amount is None:
        net_amount = first_regex(
            rf"L[ÍI]QUIDO"
            rf"\s*[:\-]?\s*"
            rf"({AMOUNT_PATTERN})"
            rf"\s*€?",
            text,
        )

    return {
        "document_family": "notary",

        "invoice_reference": {
            "issuer_name": field(
                issuer_name,
                0.98,
                "document_header",
            ),

            "nif_cif": field(
                nif_cif,
                0.98,
                "first_issuer_nif_cif",
            ),

            "protocol_number": field(
                protocol_number,
                0.98,
                "reference_label",
            ),

            "invoice_number": field(
                invoice_number,
                0.98,
                "reference_label",
            ),

            "invoice_date": field(
                invoice_date,
                0.98,
                "reference_label",
            ),
        },

        "financial": {
            "base_amount": field(
                base_amount,
                0.98,
                "financial_summary",
            ),

            "retention_base_amount": field(
                retention_base_amount,
                0.92,
                "financial_summary_or_business_default",
            ),

            "iva_percentage": field(
                iva_percentage,
                0.98,
                "financial_summary",
            ),

            "retention_percentage": field(
                retention_percentage,
                0.98,
                "financial_summary",
            ),

            "iva_amount": field(
                iva_amount,
                0.98,
                "financial_summary",
            ),

            "retention_amount": field(
                retention_amount,
                0.98,
                "financial_summary",
            ),

            "non_taxable_amount": field(
                non_taxable_amount,
                0.90,
                "financial_summary_or_zero_default",
            ),

            "net_amount": field(
                net_amount,
                0.98,
                "final_payable_label",
            ),
        },
    }


# =========================
# Validation métier
# =========================
def validate_and_reconcile(
    extraction: Dict[str, object],
) -> Tuple[Dict[str, object], List[str]]:
    """
    Vérifie les règles demandées :

    IVA = Base imposable × % IVA

    Rétention = Base rétention × % rétention

    Neto = Base imposable
           + IVA
           + montant non soumis
           - rétention
    """
    financial = extraction["financial"]

    base = amount_to_float(
        financial["base_amount"]["value"]
    )

    retention_base = amount_to_float(
        financial["retention_base_amount"]["value"]
    )

    iva_percentage = amount_to_float(
        financial["iva_percentage"]["value"]
    )

    retention_percentage = amount_to_float(
        financial["retention_percentage"]["value"]
    )

    iva_amount = amount_to_float(
        financial["iva_amount"]["value"]
    )

    retention_amount = amount_to_float(
        financial["retention_amount"]["value"]
    )

    non_taxable = amount_to_float(
        financial["non_taxable_amount"]["value"]
    )

    net_amount = amount_to_float(
        financial["net_amount"]["value"]
    )

    warnings: List[str] = []

    non_taxable_for_calculation = (
        non_taxable or 0.0
    )

    retention_for_calculation = (
        retention_amount or 0.0
    )

    if (
        base is not None
        and iva_percentage is not None
        and iva_amount is not None
    ):
        expected_iva = round(
            base * iva_percentage / 100.0,
            2,
        )

        if abs(expected_iva - iva_amount) > 0.06:
            warnings.append(
                "IVA incohérente : "
                f"extraite={iva_amount:.2f}, "
                f"calculée={expected_iva:.2f}."
            )

    if (
        retention_base is not None
        and retention_percentage is not None
        and retention_amount is not None
    ):
        expected_retention = round(
            retention_base
            * retention_percentage
            / 100.0,
            2,
        )

        if (
            abs(
                expected_retention
                - retention_amount
            )
            > 0.06
        ):
            warnings.append(
                "Rétention incohérente : "
                f"extraite={retention_amount:.2f}, "
                f"calculée={expected_retention:.2f}."
            )

    if (
        base is not None
        and iva_amount is not None
    ):
        expected_net = round(
            base
            + iva_amount
            + non_taxable_for_calculation
            - retention_for_calculation,
            2,
        )

        if net_amount is None:
            financial["net_amount"] = field(
                format_european_amount(
                    expected_net
                ),
                0.88,
                "derived_business_formula",
            )

        elif abs(expected_net - net_amount) > 0.08:
            warnings.append(
                "Net incohérent : "
                f"extrait={net_amount:.2f}, "
                f"calculé={expected_net:.2f}."
            )

    return extraction, warnings


def calculate_global_confidence(
    extraction: Dict[str, object],
) -> float:
    confidences: List[float] = []

    for section_name in (
        "invoice_reference",
        "financial",
    ):
        section = extraction.get(
            section_name,
            {},
        )

        for result in section.values():
            if (
                isinstance(result, dict)
                and result.get("value") is not None
            ):
                confidences.append(
                    float(
                        result.get(
                            "confidence",
                            0.0,
                        )
                    )
                )

    if not confidences:
        return 0.0

    return round(
        sum(confidences) / len(confidences),
        3,
    )


def extract_document(
    document_text: str,
    text_source: str,
) -> Tuple[Dict[str, object], List[str]]:
    """
    Route le document vers le bon extracteur.
    """
    family = detect_document_family(
        document_text
    )

    if family == "registry":
        extraction = extract_registry_fields(
            document_text
        )

    else:
        extraction = extract_notary_fields(
            document_text
        )

    extraction["text_source"] = text_source

    extraction, warnings = validate_and_reconcile(
        extraction
    )

    extraction["global_confidence"] = (
        calculate_global_confidence(
            extraction
        )
    )

    return extraction, warnings


# =========================
# Export PDF
# =========================
def generate_extraction_pdf(
    extraction: Dict[str, object],
) -> bytes:
    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    story = [
        Paragraph(
            "IDP Invoice OCR - Extraction Report",
            styles["Title"],
        ),
        Spacer(1, 0.7 * cm),
    ]

    reference = extraction.get(
        "invoice_reference",
        {},
    )

    financial = extraction.get(
        "financial",
        {},
    )

    rows = [
        ["Field", "Extracted value"],

        [
            "Document family",
            extraction.get(
                "document_family",
                "Not detected",
            ),
        ],

        [
            "Notary / Registrar name",
            reference
            .get("issuer_name", {})
            .get("value")
            or "Not detected",
        ],

        [
            "NIF / CIF",
            reference
            .get("nif_cif", {})
            .get("value")
            or "Not detected",
        ],

        [
            "Protocol number",
            reference
            .get("protocol_number", {})
            .get("value")
            or "Not detected",
        ],

        [
            "Invoice number",
            reference
            .get("invoice_number", {})
            .get("value")
            or "Not detected",
        ],

        [
            "Invoice date",
            reference
            .get("invoice_date", {})
            .get("value")
            or "Not detected",
        ],

        [
            "Base imponible",
            financial
            .get("base_amount", {})
            .get("value")
            or "Not detected",
        ],

        [
            "Base retencion",
            financial
            .get("retention_base_amount", {})
            .get("value")
            or "Not detected",
        ],

        [
            "% IVA",
            financial
            .get("iva_percentage", {})
            .get("value")
            or "Not detected",
        ],

        [
            "% Retencion",
            financial
            .get("retention_percentage", {})
            .get("value")
            or "Not detected",
        ],

        [
            "Importe IVA",
            financial
            .get("iva_amount", {})
            .get("value")
            or "Not detected",
        ],

        [
            "Importe Retencion",
            financial
            .get("retention_amount", {})
            .get("value")
            or "Not detected",
        ],

        [
            "Importe no sujeto",
            financial
            .get("non_taxable_amount", {})
            .get("value")
            or "Not detected",
        ],

        [
            "Importe Neto",
            financial
            .get("net_amount", {})
            .get("value")
            or "Not detected",
        ],
    ]

    table = Table(
        rows,
        colWidths=[
            6.5 * cm,
            8.5 * cm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(table)

    document.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


# =========================
# Application Streamlit
# =========================
st.set_page_config(
    page_title="IDP Invoice Extraction - Robust V2",
    layout="wide",
)

st.title(
    "IDP — Extraction robuste de factures · ROBUST V2"
)

st.success(
    "Moteur ROBUST V2 actif : "
    "texte PDF natif + OCR fallback + "
    "détection notaire / registre"
)

st.caption(
    "Lecture PDF native d'abord · "
    "OCR fallback · "
    "validation financière métier"
)

with st.sidebar:
    st.header("Paramètres")

    apply_preprocessing = st.checkbox(
        "Prétraitement OCR",
        value=True,
    )

    minimum_confidence = st.slider(
        "Confiance minimale des blocs affichés",
        0.0,
        1.0,
        0.20,
        0.05,
    )

    show_raw_blocks = st.checkbox(
        "Afficher les blocs OCR bruts",
        value=False,
    )


uploaded_file = st.file_uploader(
    "Charge une facture (PDF, PNG, JPG, JPEG)",
    type=[
        "pdf",
        "png",
        "jpg",
        "jpeg",
    ],
)

if uploaded_file is None:
    st.info(
        "Charge un document pour lancer l'extraction."
    )
    st.stop()


# Le fichier est lu une seule fois.
file_bytes = uploaded_file.getvalue()


try:
    pages = load_pages_from_bytes(
        file_bytes,
        uploaded_file.name,
    )

except Exception as error:
    st.error(
        f"Erreur de chargement : {error}"
    )
    st.stop()


# =========================
# Sélection de la source texte
# =========================
embedded_text = ""

if uploaded_file.name.lower().endswith(".pdf"):
    embedded_text = extract_embedded_pdf_text(
        file_bytes
    )


if len(embedded_text.strip()) >= 100:
    document_text = embedded_text
    text_source = "embedded_pdf_text"

else:
    document_text = ocr_pages_to_text(
        pages,
        apply_preprocessing=apply_preprocessing,
    )

    text_source = "tesseract_ocr_fallback"


if not document_text.strip():
    st.error(
        "Aucun texte n'a pu être extrait du document."
    )
    st.stop()


# Extraction sur le document complet.
extraction, financial_warnings = extract_document(
    document_text,
    text_source,
)


# =========================
# Page affichée
# =========================
page_index = st.selectbox(
    "Page affichée",
    options=list(range(len(pages))),
    format_func=lambda index: (
        f"Page {index + 1}"
    ),
)

original_page = pages[page_index]

ocr_page = (
    preprocess_image(original_page)
    if apply_preprocessing
    else original_page
)


# Les blocs servent à la visualisation,
# pas à l'extraction des PDF numériques.
try:
    raw_blocks = run_tesseract_ocr_raw(
        ocr_page,
        psm=11,
    )

    merged_blocks = merge_nearby_words(
        raw_blocks
    )

except Exception as error:
    raw_blocks = []
    merged_blocks = []

    st.warning(
        "La visualisation OCR n'est pas disponible, "
        "mais l'extraction peut continuer : "
        f"{error}"
    )


visualized_page = draw_boxes(
    original_page,
    merged_blocks,
    minimum_confidence,
)


# =========================
# Onglets
# =========================
tabs = st.tabs(
    [
        "Résultats",
        "Visualisation",
        "Texte document",
        "Blocs OCR",
        "Export",
    ]
)


with tabs[0]:
    col_a, col_b, col_c = st.columns(3)

    col_a.metric(
        "Famille",
        extraction.get(
            "document_family",
            "unknown",
        ),
    )

    col_b.metric(
        "Source du texte",
        extraction.get(
            "text_source",
            "unknown",
        ),
    )

    col_c.metric(
        "Confiance globale",
        extraction.get(
            "global_confidence",
            0.0,
        ),
    )

    st.json(extraction)

    st.subheader("Lecture rapide")

    reference = extraction[
        "invoice_reference"
    ]

    financial = extraction[
        "financial"
    ]

    readable_rows = [
        (
            "Nom notaire / registrateur",
            "issuer_name",
            reference,
        ),
        (
            "NIF / CIF",
            "nif_cif",
            reference,
        ),
        (
            "Nº protocole",
            "protocol_number",
            reference,
        ),
        (
            "Nº facture",
            "invoice_number",
            reference,
        ),
        (
            "Date facture",
            "invoice_date",
            reference,
        ),
        (
            "Base imponible",
            "base_amount",
            financial,
        ),
        (
            "Base retención",
            "retention_base_amount",
            financial,
        ),
        (
            "% IVA",
            "iva_percentage",
            financial,
        ),
        (
            "% Retención",
            "retention_percentage",
            financial,
        ),
        (
            "Importe IVA",
            "iva_amount",
            financial,
        ),
        (
            "Importe Retención",
            "retention_amount",
            financial,
        ),
        (
            "Importe no sujeto",
            "non_taxable_amount",
            financial,
        ),
        (
            "Importe Neto",
            "net_amount",
            financial,
        ),
    ]

    table_data = []

    for label, key, section in readable_rows:
        result = section.get(
            key,
            {},
        )

        table_data.append(
            {
                "Champ": label,
                "Valeur": result.get("value"),
                "Confiance": result.get(
                    "confidence"
                ),
                "Source": result.get("source"),
            }
        )

    st.dataframe(
        pd.DataFrame(table_data),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Contrôle de cohérence financière"
    )

    if financial_warnings:
        for warning in financial_warnings:
            st.warning(warning)

    else:
        st.success(
            "IVA, rétention et montant net "
            "sont cohérents avec les règles métier."
        )


with tabs[1]:
    left, right = st.columns(2)

    with left:
        st.subheader("Document")

        st.image(
            original_page,
            use_container_width=True,
        )

    with right:
        st.subheader("Blocs OCR")

        st.image(
            visualized_page,
            use_container_width=True,
        )


with tabs[2]:
    st.caption(
        "Le moteur traite toutes les pages "
        "du document."
    )

    st.text_area(
        "Texte utilisé par le moteur",
        value=document_text,
        height=650,
    )


with tabs[3]:
    blocks_to_show = (
        raw_blocks
        if show_raw_blocks
        else merged_blocks
    )

    block_rows = [
        asdict(block)
        for block in blocks_to_show
        if (
            block.confidence
            >= minimum_confidence
        )
    ]

    st.caption(
        f"{'Blocs bruts' if show_raw_blocks else 'Blocs fusionnés'} "
        f"— {len(block_rows)} blocs"
    )

    st.dataframe(
        pd.DataFrame(block_rows),
        use_container_width=True,
    )


with tabs[4]:
    pdf_report = generate_extraction_pdf(
        extraction
    )

    st.download_button(
        label="Télécharger le rapport PDF",
        data=pdf_report,
        file_name="invoice_extraction_report.pdf",
        mime="application/pdf",
    )
