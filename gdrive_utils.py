# gdrive_utils.py
import os
import io
import re
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_ID = "IMZHc_WawXZkPAiQcEWR213TVSrxilnC"  # or parse from URL

def get_drive_service():
    svc_info = st.secrets["service_account"]
    creds = Credentials.from_service_account_info(svc_info, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)

def _get_folder_meta(service, folder_id: str):
    """Return {'id','name','mimeType','driveId'(optional)} or None if not visible."""
    try:
        meta = service.files().get(
            fileId=folder_id,
            fields="id,name,mimeType,driveId",
            supportsAllDrives=True,
        ).execute()
        return meta
    except Exception as e:
        st.error(f"❌ Cannot access folder {folder_id}. "
                 f"Share it with the service account. Details: {e}")
        return None

def get_all_pdfs(service):
    # 1) Verify we can see the folder and detect whether it's in a Shared Drive
    meta = _get_folder_meta(service, FOLDER_ID)
    if not meta:
        return []

    if meta.get("mimeType") != "application/vnd.google-apps.folder":
        st.error("❌ The provided ID is not a folder.")
        return []

    query = (
        f"'{FOLDER_ID}' in parents and trashed = false and "
        "mimeType = 'application/pdf'"
    )

    list_kwargs = dict(
        q=query,
        orderBy="modifiedTime desc",     # use modifiedTime; createdTime is fine too
        pageSize=100,
        fields="files(id,name,mimeType,modifiedTime,size)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
    )

    # If it's a Shared Drive folder, restrict the corpus to that drive
    if meta.get("driveId"):
        list_kwargs.update(
            corpora="drive",
            driveId=meta["driveId"],
        )

    try:
        res = service.files().list(**list_kwargs).execute()
        files = res.get("files", [])
        if not files:
            st.warning("📭 No PDF files found in the Drive folder.")
        return files
    except Exception as e:
        st.error(f"❌ Error accessing Drive folder: {e}")
        return []

def download_pdf(service, file_id, file_name):
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
