import os
import io
import json
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# === CONFIG ===
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']



# === Auth ===
def get_drive_service():
    service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

# === Fetch Latest PDF ===
# gdrive_utils.py
# ...
#FOLDER_ID = "1VglZDFbufOxHTZ4qZ_feUw_XHaxacPxr"  # default

def _extract_folder_id(folder_id_or_url: str) -> str:
    """
    Resolve folder ID. Preference order:
      1) explicit argument (ID or full URL)
      2) env var GOOGLE_DRIVE_FOLDER (from Secret)
    """
    # 1) explicit arg wins
    s = (folder_id_or_url or "").strip()
    if s:
        if s.startswith("http"):
            return s.rstrip("/").split("/")[-1]
        return s

    # 2) env var fallback
    env_val = os.getenv("GOOGLE_DRIVE_FOLDER", "").strip()
    if env_val:
        if env_val.startswith("http"):
            return env_val.rstrip("/").split("/")[-1]
        return env_val

    raise ValueError("No Google Drive folder ID provided (argument or GOOGLE_DRIVE_FOLDER).")

def get_all_pdfs(service, folder_id_or_url: str = None):
    folder_id = _extract_folder_id(folder_id_or_url)
    query = (
        f"'{folder_id}' in parents and "
        "trashed = false and "
        "mimeType = 'application/pdf'"
    )
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
        if not files:
            st.sidebar.warning("No PDFs returned by Drive for that folder ID.")
        return files
    except Exception as e:
        st.sidebar.error(f"❌ Drive list error: {e}")
        raise




# === Download PDF ===
def download_pdf(service, file_id, file_name):
    try:
        request = service.files().get_media(fileId=file_id)
        file_path = os.path.join("uploaded", file_name)
        os.makedirs("uploaded", exist_ok=True)
        with io.FileIO(file_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
        #st.success(f"📥 Downloaded {file_name} to {file_path}")
        return file_path
    except Exception as e:
        st.error(f"❌ Failed to download PDF: {e}")
        return None
