import os
import io
import json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FOLDER_ID = "1VglZDFbufOxHTZ4qZ_feUw_XHaxacPxr"  # 🔁 Update with your folder ID

import streamlit as st




def get_drive_service():
    service_account_info = st.secrets["service_account"]
    creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def get_latest_pdf(service):
    query = f"'{FOLDER_ID}' in parents and mimeType='application/pdf'"
    results = service.files().list(q=query, orderBy="createdTime desc", pageSize=1,
                                   fields="files(id, name)").execute()
    files = results.get("files", [])
    return files[0] if files else None

def download_pdf(service, file_id, file_name):
    request = service.files().get_media(fileId=file_id)
    file_path = os.path.join("uploaded", file_name)
    os.makedirs("uploaded", exist_ok=True)
    with io.FileIO(file_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
    return file_path
