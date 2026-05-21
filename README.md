# IDP Invoice OCR MVP

This project is a first MVP for Intelligent Document Processing of invoice documents.

## Features

- PDF and image upload
- OCR using Tesseract
- PDF conversion using Poppler
- OCR block visualization
- Raw and merged OCR blocks
- Multi-strategy field extraction
- Confidence scoring
- Debug mode

## Extracted fields

- Protocol number
- Invoice number
- Invoice date
- Net amount

## Run

```bash
pip install -r requirements.txt
streamlit run idp_mvp_streamlit.py