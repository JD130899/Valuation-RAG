import os
import io
import json
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
# === Auth ===
import streamlit as st
from google.oauth2.service_account import Credentials
# === CONFIG ===
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FOLDER_ID = "1VglZDFbufOxHTZ4qZ_feUw_XHaxacPxr"  # Folder to watch
st.write("✅ Keys in secrets:", list(st.secrets.keys()))




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
            pageSize=20,  # You can increase this if needed
            fields="files(id, name, mimeType)"
        ).execute()
        files = results.get("files", [])

        pdfs = [file for file in files if file["name"].lower().endswith(".pdf")]

        if not pdfs:
            st.warning("📭 No PDF files found in Google Drive folder.")
        return pdfs

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
        #st.success(f"📥 Downloaded {file_name} to {file_path}")
        return file_path
    except Exception as e:
        st.error(f"❌ Failed to download PDF: {e}")
        return None
