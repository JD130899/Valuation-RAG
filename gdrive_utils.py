# === gdrive_utils.py ===
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


def get_drive_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("drive", "v3", credentials=creds)


def list_pdfs(service, folder_id):
    query = f"'{folder_id}' in parents and mimeType='application/pdf'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    return results.get("files", [])


def download_pdf(service, file_id, destination):
    request = service.files().get_media(fileId=file_id)
    with open(destination, "wb") as f:
        f.write(request.execute())


# === streamlit sidebar integration ===
import streamlit as st
from gdrive_utils import get_drive_service, list_pdfs, download_pdf

FOLDER_ID = "<YOUR_FOLDER_ID_HERE>"  # Replace with your Google Drive folder ID

st.sidebar.markdown("### Google Drive PDFs")
if "gdrive_files" not in st.session_state:
    try:
        service = get_drive_service()
        st.session_state.gdrive_files = list_pdfs(service, FOLDER_ID)
        st.session_state.gdrive_service = service
    except Exception as e:
        st.sidebar.error(f"\u274c Google Drive auth failed: {e}")

if "gdrive_files" in st.session_state and st.session_state.gdrive_files:
    selected_name = st.sidebar.selectbox("Pick a PDF", [f["name"] for f in st.session_state.gdrive_files])
    selected_file = next(f for f in st.session_state.gdrive_files if f["name"] == selected_name)
    PDF_PATH = os.path.join("uploaded", selected_name)

    if not os.path.exists(PDF_PATH):
        with st.spinner(f"\ud83d\udcc5 Downloading {selected_name} from Google Drive..."):
            os.makedirs("uploaded", exist_ok=True)
            download_pdf(st.session_state.gdrive_service, selected_file["id"], PDF_PATH)
        st.success(f"\u2705 Downloaded {selected_name}!")

    # Simulate upload for downstream processing
    uploaded_file = open(PDF_PATH, "rb")
    file_name = selected_name
else:
    st.sidebar.info("No PDFs found in Drive folder.")
