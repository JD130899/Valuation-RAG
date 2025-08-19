# gdrive_utils.py
import os
import io
import json
import streamlit as st

from google.auth import default as google_auth_default
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Drive scopes we need (read-only is enough)
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Optional fallback folder (will be ignored if you pass a link/id)
FOLDER_ID = "1XGyBBFhhQFiG43jpYJhNzZYi7C-_l5me"



def get_drive_service():
    """
    Try explicit SERVICE_ACCOUNT_JSON first (if set),
    otherwise use Application Default Credentials (Cloud Run runtime SA).
    """
    sa_json = os.environ.get("SERVICE_ACCOUNT_JSON", "").strip()

    if sa_json:
        # Explicit JSON path (Secret) flow
        service_account_info = json.loads(sa_json)
        creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    else:
        # Default credentials (Cloud Run runtime account)
        creds, _ = google_auth_default(scopes=SCOPES)

    return build("drive", "v3", credentials=creds)


def _extract_folder_id(folder_id_or_url: str) -> str:
    s = (folder_id_or_url or "").strip()
    if not s:
        return FOLDER_ID
    if s.startswith("http"):
        # Handles: https://drive.google.com/drive/folders/<ID>
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
        st.error(f"❌ Error accessing Drive folder: {e}")
        return []


def download_pdf(service, file_id, file_name):
    try:
        request = service.files().get_media(fileId=file_id)
        file_path = os.path.join("uploaded", file_name)
        os.makedirs("uploaded", exist_ok=True)
        with io.FileIO(file_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        return file_path
    except Exception as e:
        st.error(f"❌ Failed to download PDF: {e}")
        return None
