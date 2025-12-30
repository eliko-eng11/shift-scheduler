import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

st.title("בדיקת JWT / Secrets")

def extract_sheet_id(url: str) -> str:
    if "/spreadsheets/d/" in url:
        return url.split("/spreadsheets/d/")[1].split("/")[0]
    return url.strip()

def client():
    info = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)

sheet_link = st.text_input("קישור לשיטס")
if st.button("בדוק"):
    gc = client()
    sh = gc.open_by_key(extract_sheet_id(sheet_link))
    st.success(f"נפתח: {sh.title}")
