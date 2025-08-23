# gdrive_utils.py
import os
import io
import json
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FALLBACK_FOLDER_ID = "1VglZDFbufOxHTZ4qZ_feUw_XHaxacPxr"

def _on_cloud() -> bool:
    """
    Detect Cloud Run / GCP hosted env.
    K_SERVICE is set on Cloud Run. (GAE_ENV for App Engine, etc.)
    """
    return bool(os.getenv("K_SERVICE") or os.getenv("GAE_ENV") or os.getenv("GOOGLE_CLOUD_PROJECT"))

def _extract_folder_id(folder_id_or_url: str | None) -> str:
    s = (folder_id_or_url or "").strip()
    if not s:
        return FALLBACK_FOLDER_ID
    if s.startswith("http"):
        s = s.rstrip("/").split("/")[-1]
    return s

def _load_service_account_info():
    """
    Try, in order:
    1) st.secrets["service_account"] (works on Streamlit Cloud + local if .streamlit/secrets.toml exists)
    2) SERVICE_ACCOUNT_JSON (env var containing raw JSON)
    3) GOOGLE_APPLICATION_CREDENTIALS / SERVICE_ACCOUNT_FILE (env var path to JSON file)
    """
    # 1) Streamlit secrets
    try:
        if "service_account" in st.secrets:
            info = st.secrets["service_account"]
            return json.loads(json.dumps(info))  # cast SecretDict -> plain dict
    except Exception:
        pass

    # 2) Raw JSON in env
    sa_json = os.getenv("SERVICE_ACCOUNT_JSON")
    if sa_json:
        try:
            return json.loads(sa_json)
        except Exception as e:
            st.error(f"Invalid SERVICE_ACCOUNT_JSON: {e}")

    # 3) Path in env
    for key in ("GOOGLE_APPLICATION_CREDENTIALS", "SERVICE_ACCOUNT_FILE"):
        path = os.getenv(key)
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Failed to read {key}: {e}")

    return None

def get_drive_service():
    """
    Returns a Drive API client or None.
    * Local: require explicit service-account (secrets or env). Do NOT try ADC to avoid metadata timeouts.
    * Cloud: allow ADC fallback (Cloud Run SA must have Drive access).
    """
    info = _load_service_account_info()
    creds = None

    if info:
        # Use service-account if provided (works both local & cloud)
        try:
            creds = Credentials.from_service_account_info(info, scopes=SCOPES)
        except Exception as e:
            st.error(f"❌ Could not create credentials from service account info: {e}")
            return None
    else:
        # No explicit credentials. Only try ADC when actually on cloud.
        if _on_cloud():
            try:
                import google.auth
                creds, _ = google.auth.default(scopes=SCOPES)
            except Exception as e:
                st.error(
                    "❌ Google default credentials not available. "
                    "Attach a service account to the Cloud Run revision or provide secrets/env."
                )
                #st.info(f"(Details: {e})")
                return None
        else:
            # Local and no creds -> clear guidance instead of metadata timeout
            st.error(
                "❌ Google Drive credentials not found locally.\n"
                "Add .streamlit/secrets.toml with [service_account] or set "
                "SERVICE_ACCOUNT_JSON / GOOGLE_APPLICATION_CREDENTIALS."
            )
            return None

    try:
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
