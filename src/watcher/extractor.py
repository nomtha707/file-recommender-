import os
from docx import Document
import io
try:
    import pdfplumber
except ImportError:  # Use ImportError for missing modules
    print("Watcher Info: pdfplumber not installed. PDF extraction disabled.")
    pdfplumber = None
except Exception as e:  # Catch other potential errors during import
    print(f"Watcher Error: Failed to import pdfplumber: {e}")
    pdfplumber = None

# lightweight text extraction for common types.


def extract_text(filepath: str) -> str:
    """Extracts text content from various file types."""
    # Ensure filepath is a string
    filepath_str = str(filepath)
    ext = os.path.splitext(filepath_str)[1].lower()

    try:
        if ext in ['.txt', '.md', '.py', '.csv', '.json', '.html', '.xml', '.log']:  # Added more text types
            # Try reading with utf-8 first, fallback to latin-1 for broader compatibility
            try:
                with open(filepath_str, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                print(
                    f"Watcher Warning: UTF-8 decode failed for {filepath_str}. Trying latin-1.")
                with open(filepath_str, 'r', encoding='latin-1') as f:
                    return f.read()
        elif ext == '.docx':
            try:
                doc = Document(filepath_str)
                return "\n".join(p.text for p in doc.paragraphs if p.text)
            except Exception as docx_err:
                print(
                    f"Watcher Error: Failed to process DOCX {filepath_str}: {docx_err}")
                return ""  # Return empty on specific docx error
        elif ext == '.pdf' and pdfplumber is not None:
            text = []
            try:
                with pdfplumber.open(filepath_str) as pdf:
                    for i, page in enumerate(pdf.pages):
                        # Add a basic page separator
                        # text.append(f"\n--- Page {i+1} ---")
                        # Adjust tolerance if needed
                        page_text = page.extract_text(
                            x_tolerance=1, y_tolerance=1)
                        if page_text:
                            text.append(page_text)
                return "\n".join(text)
            except Exception as pdf_err:
                print(
                    f"Watcher Error: Failed to process PDF {filepath_str}: {pdf_err}")
                return ""  # Return empty on specific pdf error
        else:
            # You could add more extractors here (e.g., for .pptx, .xlsx)
            # print(f"Watcher Info: Unsupported file type '{ext}' for text extraction: {filepath_str}") # Optional Info
            return ''  # Return empty for unsupported types

    except FileNotFoundError:
        print(
            f"Watcher Error: File not found during extraction: {filepath_str}")
        return ''
    except Exception as e:
        # Catch-all for other unexpected errors during extraction
        print(f"Watcher Error: Generic error extracting {filepath_str}: {e}")
        return ''