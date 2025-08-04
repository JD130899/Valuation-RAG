import os
import io
import json
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# === CONFIG ===
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FOLDER_ID = "1VglZDFbufOxHTZ4qZ_feUw_XHaxacPxr"  # Folder to watch
HARDCODED_FILE_ID = "1S1ZkbQBQVYBUIKVuXAIsL-AZttzmX2So"  # ✅ Your known PDF ID

# === Auth ===
def get_drive_service():
    service_account_info = st.secrets["service_account"]
    creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

# === Fetch Latest PDF ===
def get_all_pdfs(service):
    query = f"'{FOLDER_ID}' in parents and trashed = false"
    try:
        results = service.files().list(
            q=query,
            orderBy="createdTime desc",
            pageSize=50,
            fields="files(id, name, mimeType)"
        ).execute()
        files = results.get("files", [])
        return [f for f in files if f["name"].lower().endswith(".pdf")]
    except Exception as e:
        st.error(f"❌ Error accessing Drive folder: {e}")
        return []


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
        st.success(f"📥 Downloaded {file_name} to {file_path}")
        return file_path
    except Exception as e:
        st.error(f"❌ Failed to download PDF: {e}")
        return None
