# gdrive_utils.py
from typing import List, Dict, Optional
import re
import io
import os
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def _extract_folder_id(s: str) -> str:
    if not s:
        return ""
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", s)
    return m.group(1) if m else s.strip()

def get_drive_service():
    sa = st.secrets["service_account"]
    creds = Credentials.from_service_account_info(sa, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)

def _get_folder_meta(service, folder_id: str) -> Dict:
    return (
        service.files()
        .get(fileId=folder_id, fields="id,name,driveId", supportsAllDrives=True)
        .execute()
    )

from googleapiclient.errors import HttpError
import json


def debug_folder_meta(service, folder_id: str):
    try:
        meta = service.files().get(
            fileId=folder_id,
            fields="id,name,mimeType,driveId,parents",
            supportsAllDrives=True
        ).execute()
        return {"ok": True, "meta": meta}
    except HttpError as e:
        # Try to decode the API's JSON error body; fall back to status/message
        try:
            body = json.loads(e.content.decode("utf-8"))
        except Exception:
            body = {"status": getattr(e.resp, "status", None), "message": str(e)}
        return {"ok": False, "error": body}


def _resolve_folder_id(service, folder_id: str) -> Dict:
    """
    Resolves folder_id; if it's a shortcut, returns the target folder id.
    Returns a dict: {"id", "name", "driveId"} for the *actual* folder.
    """
    meta = service.files().get(
        fileId=folder_id,
        fields="id,name,driveId,mimeType,shortcutDetails",
        supportsAllDrives=True,
    ).execute()

    # If it's a shortcut, follow it to the real folder
    if meta.get("mimeType") == "application/vnd.google-apps.shortcut":
        target_id = meta.get("shortcutDetails", {}).get("targetId")
        if not target_id:
            raise ValueError("Shortcut has no targetId")
        meta = service.files().get(
            fileId=target_id,
            fields="id,name,driveId,mimeType",
            supportsAllDrives=True,
        ).execute()

    # Safety: must be a folder
    if meta.get("mimeType") != "application/vnd.google-apps.folder":
        raise ValueError(f"ID is not a folder: {meta.get('mimeType')}")
    return {"id": meta["id"], "name": meta.get("name"), "driveId": meta.get("driveId")}

def get_all_pdfs(service, folder_id_or_url: str):
    folder_id = _extract_folder_id(folder_id_or_url)
    try:
        meta = _resolve_folder_id(service, folder_id)
    except Exception as e:
        st.error(f"❌ Cannot access folder {folder_id}. Share it with the service account. Details: {e}")
        return []

    drive_id = meta.get("driveId")
    params = {
        "q": f"'{meta['id']}' in parents and trashed = false and mimeType = 'application/pdf'",
        "orderBy": "createdTime desc",
        "pageSize": 100,
        "fields": "files(id,name,mimeType)",
        "supportsAllDrives": True,
        "includeItemsFromAllDrives": True,
    }
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
