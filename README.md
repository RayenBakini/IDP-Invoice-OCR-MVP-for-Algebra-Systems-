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
- Clean PDF export report without confidence scores

## Extracted fields

- Protocol number
- Invoice number
- Invoice date
- Net amount

 ## PDF Export

The application can generate a clean PDF extraction report containing only the extracted business fields:

- Protocol number
- Invoice number
- Invoice date
- Net amount

Confidence scores and debug information remain visible inside the Streamlit interface, but they are not included in the exported PDF report.

## Run

```bash
pip install -r requirements.txt
streamlit run idp_mvp_streamlit.py
