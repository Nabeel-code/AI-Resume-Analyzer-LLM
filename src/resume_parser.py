# utils/resume_parser.py
import docx2txt

def extract_text_from_resume(file) -> str:
    """
    Accepts Streamlit uploaded file-like object (docx).
    Returns cleaned plain text.
    """
    try:
        # docx2txt accepts file path or file-like; streamlit uploader gives a SpooledTemporaryFile
        # docx2txt.process can accept a file object; if not, read bytes and write to temp file.
        text = docx2txt.process(file)
        if not text:
            return ""
        return text.replace("\r", " ").replace("\n\n", "\n").strip()
    except Exception:
        try:
            # fallback: try reading as bytes and decode, best-effort
            file.seek(0)
            raw = file.read()
            if isinstance(raw, bytes):
                raw = raw.decode(errors="ignore")
            return str(raw)[:20000]
        except Exception:
            return ""
