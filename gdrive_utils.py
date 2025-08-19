# gdrive_utils.py (only the auth parts changed)
import os, json, re, io
from typing import Optional, List, Dict
try:
    import streamlit as st
except Exception:
    st = None

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def _creds_from_st_secrets() -> Optional[Credentials]:
    if st is None:
        return None
    try:
        info = st.secrets.get("service_account")
        if info:
            return Credentials.from_service_account_info(info, scopes=SCOPES)
    except Exception:
        pass
    return None

def _creds_from_env() -> Optional[Credentials]:
    """
    Cloud Run / generic env:
    - GOOGLE_SERVICE_ACCOUNT  (JSON string from Secret Manager)
    - GOOGLE_SERVICE_ACCOUNT_JSON  (JSON string)
    - GOOGLE_APPLICATION_CREDENTIALS (path to JSON file)
    """
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if raw:
        try:
            info = json.loads(raw)
            return Credentials.from_service_account_info(info, scopes=SCOPES)
        except Exception:
            # fall through to file path
            pass

    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if path and os.path.exists(path):
        try:
            return Credentials.from_service_account_file(path, scopes=SCOPES)
        except Exception:
            pass

    return None

# gdrive_utils.py
import os, json, io, re
from typing import Optional, List, Dict

try:
    import streamlit as st
except Exception:
    st = None

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def get_drive_service():
    """Return a Google Drive service, using either st.secrets (Streamlit) or env vars (Cloud Run)."""
    creds = None

    # --- Streamlit Cloud path ---
    if st is not None:
        try:
            info = st.secrets.get("service_account")
            if info:
                creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        except Exception:
            pass

    # --- Cloud Run path (env var contains JSON) ---
    if creds is None:
        raw = os.getenv("GOOGLE_SERVICE_ACCOUNT")
        if raw:
            info = json.loads(raw)
            creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)

    # --- Local dev path (file path) ---
    if creds is None:
        path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if path and os.path.exists(path):
            creds = service_account.Credentials.from_service_account_file(path, scopes=SCOPES)

    if creds is None:
        raise RuntimeError("No Google service account credentials found")

    return build("drive", "v3", credentials=creds)

# rest of your helpers (get_all_pdfs, download_pdf, etc) stay the same




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
