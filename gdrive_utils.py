# gdrive_utils.py
import os
import io
import re
from typing import List, Dict, Optional

import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def _extract_folder_id(s: str) -> str:
    """
    Accepts either a Drive folder URL or a raw folder ID and returns the ID.
    """
    if not s:
        return ""
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", s)
    return m.group(1) if m else s.strip()

def get_drive_service():
    # expects your service account JSON under st.secrets["service_account"]
    sa = st.secrets["service_account"]
    creds = Credentials.from_service_account_info(sa, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)

def _get_folder_meta(service, folder_id: str) -> Dict:
    """
    Return {id,name,driveId} for a folder, or raise the underlying API error.
    """
    return (
        service.files()
        .get(fileId=folder_id, fields="id,name,driveId", supportsAllDrives=True)
        .execute()
    )

def get_all_pdfs(service, folder_id_or_url: str) -> List[Dict]:
    """
    Lists PDFs in the given folder. Works for:
    - My Drive folders
    - Shared with me folders
    - Shared Drives (aka Team Drives)
    """
    folder_id = _extract_folder_id(folder_id_or_url)
    try:
        meta = _get_folder_meta(service, folder_id)  # validate access + learn driveId
    except Exception as e:
        st.error(
            f"❌ Cannot access folder {folder_id}. Share it with the service account. "
            f"Details: {e}"
        )
        return []

    drive_id = meta.get("driveId")
    params = {
        "q": f"'{folder_id}' in parents and trashed = false and mimeType = 'application/pdf'",
        "orderBy": "createdTime desc",
        "pageSize": 100,
        "fields": "files(id,name,mimeType)",
        "supportsAllDrives": True,
        "includeItemsFromAllDrives": True,
    }
    # If it's a Shared Drive, use the drive corpus for faster/cleaner listing
    if drive_id:
        params["corpora"] = "drive"
        params["driveId"] = drive_id
    else:
        params["corpora"] = "allDrives"

    try:
        resp = service.files().list(**params).execute()
        files = resp.get("files", [])
        if not files:
            st.warning("📭 No PDF files found in this folder.")
        return files
    except Exception as e:
        st.error(f"❌ Error listing PDFs: {e}")
        return []

def download_pdf(service, file_id: str, file_name: str) -> Optional[str]:
    """
    Downloads a Drive file by ID to ./uploaded/<file_name> and returns the path.
    """
    try:
        req = service.files().get_media(fileId=file_id)
        os.makedirs("uploaded", exist_ok=True)
        path = os.path.join("uploaded", file_name)
        with io.FileIO(path, "wb") as f:
            dl = MediaIoBaseDownload(f, req)
            done = False
            while not done:
                _, done = dl.next_chunk()
        return path
    except Exception as e:
        st.error(f"❌ Failed to download PDF: {e}")
        return None
