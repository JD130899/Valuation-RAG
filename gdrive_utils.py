# gdrive_utils.py
import os, io, json
try:
    import streamlit as st
except Exception:
    st = None

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth import default as google_auth_default   # <-- ADC

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FOLDER_ID = "1XGyBBFhhQFiG43jpYJhNzZYi7C-_l5me"  # fallback; ignored if a link/id is provided

from functools import lru_cache

@lru_cache(maxsize=32)
def _list_files_cached(folder_id: str):
    # The inner function requires a fresh service because creds aren’t serializable.
    service = get_drive_service()
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        orderBy="createdTime desc",
        pageSize=100,
        fields="files(id, name, mimeType)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    return tuple((f["id"], f["name"], f.get("mimeType","")) for f in results.get("files", []))
    

def _emit(msg, level="info"):
    if st:
        fn = getattr(st, level, st.info)
        fn(msg)
    else:
        print(msg)


def get_drive_service():
    """
    Prefer SERVICE_ACCOUNT_JSON (local/dev). Otherwise use ADC
    (Cloud Run’s service account). Both are scoped for Drive Readonly.
    """
    # 1) Explicit JSON (local/dev or if you kept the secret mapped)
    saj = os.environ.get("SERVICE_ACCOUNT_JSON")
    if saj:
        info = json.loads(saj)
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
        return build('drive', 'v3', credentials=creds, cache_discovery=False)

    # 2) Application Default Credentials (Cloud Run)
    creds, _ = google_auth_default(scopes=SCOPES)
    if not creds:
        raise RuntimeError("ADC not available (no credentials found)")
    return build('drive', 'v3', credentials=creds, cache_discovery=False)


def _extract_folder_id(folder_id_or_url: str) -> str:
    s = (folder_id_or_url or "").strip()
    if not s:
        return FOLDER_ID
    if s.startswith("http"):
        s = s.rstrip("/").split("/")[-1]
    return s

def get_all_pdfs(service, folder_id_or_url: str = None):
    folder_id = _extract_folder_id(folder_id_or_url)
    try:
        files = _list_files_cached(folder_id)
        return [{"id": i, "name": n} for (i, n, m) in files if n.lower().endswith(".pdf")]
    except Exception as e:
        _emit(f"❌ Error accessing Drive folder: {e}", "error")
        return []


def download_pdf(service, file_id, file_name):
    try:
        request = service.files().get_media(fileId=file_id)
        os.makedirs("uploaded", exist_ok=True)
        file_path = os.path.join("uploaded", file_name)
        with io.FileIO(file_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        return file_path
    except Exception as e:
        _emit(f"❌ Failed to download PDF: {e}", "error")
        return None
