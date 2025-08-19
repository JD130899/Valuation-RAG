# gdrive_utils.py
import os
import io
import json
import re
from typing import List, Dict, Optional

# Streamlit is optional but we use it if available for st.secrets
try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# ----------------------- Auth helpers ----------------------- #
def _creds_from_st_secrets() -> Optional[Credentials]:
    """
    Try to load service account credentials from st.secrets["service_account"].
    Returns Credentials or None if not available.
    """
    if st is None:
        return None
    try:
        info = st.secrets.get("service_account", None)
        if info:
            return Credentials.from_service_account_info(info, scopes=SCOPES)
    except Exception:
        pass
    return None


def _creds_from_env_json() -> Optional[Credentials]:
    """
    Try GOOGLE_SERVICE_ACCOUNT_JSON (inline JSON) or GOOGLE_APPLICATION_CREDENTIALS (path).
    Returns Credentials or None.
    """
    # Inline JSON env var
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if raw:
        try:
            info = json.loads(raw)
            return Credentials.from_service_account_info(info, scopes=SCOPES)
        except Exception:
            # continue to try the file path
            pass

    # File path env var
    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if path and os.path.exists(path):
        try:
            return Credentials.from_service_account_file(path, scopes=SCOPES)
        except Exception:
            pass

    return None


def get_drive_service():
    """
    Build and return a Drive v3 service using the first available source:
    1) st.secrets["service_account"]
    2) GOOGLE_SERVICE_ACCOUNT_JSON (inline JSON)
    3) GOOGLE_APPLICATION_CREDENTIALS (file path)
    """
    creds = _creds_from_st_secrets() or _creds_from_env_json()
    if creds is None:
        raise RuntimeError(
            "Google Drive credentials not found. Provide either st.secrets['service_account'], "
            "GOOGLE_SERVICE_ACCOUNT_JSON, or GOOGLE_APPLICATION_CREDENTIALS."
        )
    return build("drive", "v3", credentials=creds)


# ----------------------- ID / URL utils ----------------------- #
_FOLDER_ID_RE = re.compile(r"/folders/([A-Za-z0-9_\-]{10,})")

def _extract_folder_id(folder_id_or_url: str) -> str:
    """
    Accept either a raw folder ID or a Google Drive folder URL.
    Returns the folder ID.
    """
    s = (folder_id_or_url or "").strip().strip("/")
    if not s:
        raise ValueError("Empty folder id/url.")
    # If it's a URL, capture the ID after /folders/
    m = _FOLDER_ID_RE.search(s)
    if m:
        return m.group(1)
    # Otherwise assume it's already an ID
    return s


# ----------------------- Drive ops ----------------------- #
def get_all_pdfs(service, folder_id_or_url: str, max_pages: int = 50) -> List[Dict]:
    """
    List PDFs in the given Drive folder (by ID or URL).
    Returns a list of dicts: {id, name, createdTime, size}
    Ordered by createdTime desc.
    """
    folder_id = _extract_folder_id(folder_id_or_url)

    q = (
        f"'{folder_id}' in parents and trashed = false "
        f"and mimeType = 'application/pdf'"
    )

    files: List[Dict] = []
    page_token = None
    pages_fetched = 0

    while True:
        params = {
            "q": q,
            "orderBy": "createdTime desc",
            "fields": "nextPageToken, files(id, name, createdTime, size, mimeType)",
            "pageSize": 100,
        }
        if page_token:
            params["pageToken"] = page_token

        resp = service.files().list(**params).execute()
        batch = resp.get("files", []) or []
        files.extend(batch)

        page_token = resp.get("nextPageToken")
        pages_fetched += 1
        if not page_token or pages_fetched >= max_pages:
            break

    # Normalize types (optional)
    result = []
    for f in files:
        result.append({
            "id": f.get("id"),
            "name": f.get("name"),
            "createdTime": f.get("createdTime"),
            "size": int(f.get("size")) if f.get("size") not in (None, "",) else None,
        })
    return result


def download_pdf(service, file_id: str, file_name: str, out_dir: str = "downloaded") -> str:
    """
    Download the Drive file (by id) to out_dir/file_name.
    Returns the absolute path of the downloaded file.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Sanitize filename (basic)
    safe_name = re.sub(r"[\\/:\*\?\"<>\|]+", "_", file_name or "file.pdf")
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"

    out_path = os.path.abspath(os.path.join(out_dir, safe_name))

    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(out_path, "wb")
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    try:
        while not done:
            status, done = downloader.next_chunk()
            # You could log status.progress() if desired
    finally:
        fh.close()

    return out_path
