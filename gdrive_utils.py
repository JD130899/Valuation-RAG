# gdrive_utils.py
import os, io, json
try:
    import streamlit as st   # may not exist in some contexts
except Exception:
    st = None

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FOLDER_ID = "1VglZDFbufOxHTZ4qZ_feUw_XHaxacPxr"  # fallback; not used if a link/id is passed


def _emit(msg, level="info"):
    if st:
        fn = getattr(st, level, st.info)
        fn(msg)
    else:
        print(msg)


def get_drive_service():
    # If you kept SERVICE_ACCOUNT_JSON in Cloud Run, we’ll use it.
    # Otherwise ADC (default service account) is used by your app code.
    service_account_json = os.environ.get("SERVICE_ACCOUNT_JSON")
    if service_account_json:
        info = json.loads(service_account_json)
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
        return build('drive', 'v3', credentials=creds)
    # Fall back to default credentials (provided by Cloud Run)
    return build('drive', 'v3')


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
