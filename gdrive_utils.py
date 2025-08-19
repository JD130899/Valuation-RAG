# gdrive_utils.py
import os, json
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from google.auth import default as google_auth_default

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def get_drive_service():
    """
    Prefer Application Default Credentials (ADC) on Cloud Run.
    Falls back to SERVICE_ACCOUNT_JSON env secret if ADC isn't available.
    """
    # Try ADC first (this is the Cloud Run default compute service account)
    try:
        creds, _ = google_auth_default(scopes=SCOPES)
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception:
        pass  # fall back to explicit JSON below

    # Fallback: use the old secret if present
    sa_json = os.getenv("SERVICE_ACCOUNT_JSON")
    if not sa_json:
        raise RuntimeError(
            "No Google credentials available. ADC failed and SERVICE_ACCOUNT_JSON is not set."
        )
    info = json.loads(sa_json)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)



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
