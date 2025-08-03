import os
import pickle
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import io

# === Google Drive Setup ===
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = 'service_account.json'

# === Cache directory for parsed PDFs ===
CACHE_DIR = "processed_data"
os.makedirs(CACHE_DIR, exist_ok=True)

# === Load Drive Service ===
def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build('drive', 'v3', credentials=creds)

# === List all PDFs from Drive folder ===
def get_all_pdfs(service, folder_id="your_folder_id_here"):
    query = f"'{folder_id}' in parents and mimeType='application/pdf'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    return results.get('files', [])

# === Download and save PDF if not already processed ===
def download_pdf(service, file):
    pdf_id = file['id']
    pdf_name = file['name']
    local_path = os.path.join("pdfs", pdf_name)

    os.makedirs("pdfs", exist_ok=True)
    cached_path = os.path.join(CACHE_DIR, f"{pdf_name}.pkl")

    if os.path.exists(cached_path):
        print(f"✅ Already processed: {pdf_name}")
        return cached_path  # Skip reprocessing

    request = service.files().get_media(fileId=pdf_id)
    fh = io.FileIO(local_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Downloading {pdf_name}: {int(status.progress() * 100)}%")

    print(f"✅ Downloaded: {pdf_name}")
    return local_path  # Now use this for first-time parsing
