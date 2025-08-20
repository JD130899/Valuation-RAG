# gdrive_utils.py
import os
import io
import json
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# === CONFIG ===
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FALLBACK_FOLDER_ID = "1VglZDFbufOxHTZ4qZ_feUw_XHaxacPxr"  # default if nothing provided


def _extract_folder_id(folder_id_or_url: str | None) -> str:
    s = (folder_id_or_url or "").strip()
    if not s:
        return FALLBACK_FOLDER_ID
    if s.startswith("http"):
        # works for https://drive.google.com/drive/folders/<ID>
        s = s.rstrip("/").split("/")[-1]
    return s


def _load_service_account_info():
    """
    Tries several locations for Drive creds, in this order:
    1) st.secrets["service_account"] (Streamlit Cloud style)
    2) SERVICE_ACCOUNT_JSON   (env var containing the *JSON string*)
    3) SERVICE_ACCOUNT_FILE   (env var containing a *path* to the JSON file)
    Returns: dict or None
    """
    # 1) Streamlit secrets
    try:
        if "service_account" in st.secrets:
            info = st.secrets["service_account"]
            # st.secrets returns a SecretDict; cast to plain dict
            return json.loads(json.dumps(info))
    except Exception:
        pass

    # 2) Env var with JSON string
    sa_json = os.getenv("SERVICE_ACCOUNT_JSON")
    if sa_json:
        try:
            return json.loads(sa_json)
        except Exception as e:
            st.error(f"Invalid SERVICE_ACCOUNT_JSON: {e}")

    # 3) Env var with file path
    sa_file = os.getenv("SERVICE_ACCOUNT_FILE")
    if sa_file and os.path.exists(sa_file):
        try:
            with open(sa_file, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Failed to read SERVICE_ACCOUNT_FILE: {e}")

    return None


def get_drive_service():
    """
    Returns a Drive API client or None.

    Prefers explicit service-account credentials; if not found,
    tries Google Application Default Credentials (ADC) — suitable for Cloud Run
    when the service account attached to the revision has Drive access.
    """
    info = _load_service_account_info()
    creds = None

    if info:
        try:
            creds = Credentials.from_service_account_info(info, scopes=SCOPES)
        except Exception as e:
            st.error(f"❌ Could not create credentials from service account info: {e}")
            return None
    else:
        # Try ADC (Cloud Run / GCE / local with `gcloud auth application-default login`)
        try:
            import google.auth
            creds, _ = google.auth.default(scopes=SCOPES)
        except Exception as e:
            st.warning(
                "Google Drive credentials not found. "
                "Set Streamlit secrets [service_account], or env "
                "SERVICE_ACCOUNT_JSON / SERVICE_ACCOUNT_FILE, or use ADC."
            )
            st.info(f"(Details: {e})")
            return None

    try:
        # cache_discovery=False avoids write-permission issues in containerized envs
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception as e:
        st.error(f"❌ Failed to initialize Drive client: {e}")
        return None


def get_all_pdfs(service, folder_id_or_url: str | None = None):
    if service is None:
        return []
    folder_id = _extract_folder_id(folder_id_or_url)
    query = f"'{folder_id}' in parents and trashed = false"
    try:
        results = (
            service.files()
            .list(
                q=query,
                orderBy="createdTime desc",
                pageSize=100,
                fields="files(id, name, mimeType)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        files = results.get("files", [])
        return [f for f in files if f["name"].lower().endswith(".pdf")]
    except Exception as e:
        st.error(f"❌ Error accessing Drive folder: {e}")
        return []


def download_pdf(service, file_id, file_name):
    if service is None:
        return None
    try:
        request = service.files().get_media(fileId=file_id)
        file_path = os.path.join("uploaded", file_name)
        os.makedirs("uploaded", exist_ok=True)
        with io.FileIO(file_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
        return file_path
    except Exception as e:
        st.error(f"❌ Failed to download PDF: {e}")
        return None
