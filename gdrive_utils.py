# gdrive_utils.py  (writes to /tmp, works on Cloud Run)
import os, io, json
try:
    import streamlit as st
except Exception:
    st = None

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth import default as google_auth_default

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FOLDER_ID = "1XGyBBFhhQFiG43jpYJhNzZYi7C-_l5me"

# writable dirs
DATA_DIR = os.getenv("DATA_DIR", "/tmp/uw_agent")
UPLOADED_DIR = os.path.join(DATA_DIR, "uploaded")
os.makedirs(UPLOADED_DIR, exist_ok=True)

def _emit(msg, level="info"):
    if st:
        fn = getattr(st, level, st.info)
        fn(msg)
    else:
        print(msg)

def get_drive_service():
    saj = os.environ.get("SERVICE_ACCOUNT_JSON")
    if saj:
        info = json.loads(saj)
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
        return build('drive', 'v3', credentials=creds, cache_discovery=False)
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
    query = f"'{folder_id}' in parents and trashed = false"
    try:
        results = service.files().list(
            q=query,
            orderBy="createdTime desc",
            pageSize=100,
            fields="files(id, name, mimeType)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        files = results.get("files", [])
        return [f for f in files if f["name"].lower().endswith(".pdf")]
    except Exception as e:
        _emit(f"❌ Error accessing Drive folder: {e}", "error")
        return []

def download_pdf(service, file_id, file_name):
    try:
        request = service.files().get_media(fileId=file_id)
        file_path = os.path.join(UPLOADED_DIR, file_name)
        with io.FileIO(file_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        return file_path
    except Exception as e:
        _emit(f"❌ Failed to download PDF: {e}", "error")
        return None
