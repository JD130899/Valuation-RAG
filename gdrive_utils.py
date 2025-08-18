# gdrive_utils.py
import os
import io
import re
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# === CONFIG ===
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# You can paste either:
#  - a folder ID: "1VglZDFbufOxHTZ4qZ_feUw_XHaxacPxr"
#  - or a full folder URL: "https://drive.google.com/drive/folders/1VglZDFbufOxHTZ4qZ_feUw_XHaxacPxr?usp=..."
FOLDER_LINK_OR_ID = st.secrets.get("GOOGLE_DRIVE_FOLDER", "1VglZDFbufOxHTZ4qZ_feUw_XHaxacPxr")


def _extract_folder_id(link_or_id: str) -> str:
    """Accepts a folder ID or a full Google Drive folder URL and returns the folder ID."""
    if not link_or_id:
        return ""
    # If it already looks like an ID (no slashes and ~30+ chars), just return it
    if "/" not in link_or_id and len(link_or_id) >= 10:
        return link_or_id.strip()
    m = re.search(r"/folders/([A-Za-z0-9_\-]{10,})", link_or_id)
    return m.group(1) if m else link_or_id.strip()


# === Auth ===
def get_drive_service():
    service_account_info = st.secrets["service_account"]   # share the folder with this service account email!
    creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)


# === List PDFs in the folder (works for My Drive, Shared with me, Shared Drives) ===
def get_all_pdfs(service, folder_link_or_id: str = FOLDER_LINK_OR_ID):
    folder_id = _extract_folder_id(folder_link_or_id)
    if not folder_id:
        st.error("No Google Drive folder configured.")
        return []

    query = (
        f"'{folder_id}' in parents and trashed = false and "
        "mimeType = 'application/pdf'"
    )

    try:
        results = service.files().list(
            q=query,
            # Use these so it works for shared drives / 'Shared with me'
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            orderBy="modifiedTime desc",
            pageSize=100,
            fields="files(id, name, mimeType, modifiedTime, size, driveId)",
        ).execute()
        files = results.get("files", [])

        if not files:
            st.warning("📭 No PDF files found in the specified Drive folder.")
        return files

    except Exception as e:
        st.error(f"❌ Error accessing Drive folder: {e}")
        return []


# === Download a PDF by id ===
def download_pdf(service, file_id: str, file_name: str):
    try:
        request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
        os.makedirs("uploaded", exist_ok=True)
        file_path = os.path.join("uploaded", file_name)
        with io.FileIO(file_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
        return file_path
    except Exception as e:
        st.error(f"❌ Failed to download PDF: {e}")
        return None
