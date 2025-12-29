import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd

st.title("🔍 בודק חיבור לגוגל שיטס")

# 1. בדיקת קיום ה-Secrets
st.subheader("1. בדיקת 'הכספת' (Secrets)")
if "gcp_service_account" not in st.secrets:
    st.error("❌ המפתח 'gcp_service_account' לא נמצא ב-Secrets!")
    st.info("וודא שהגדרת אותו בתוך .streamlit/secrets.toml או ב-Dashboard של סטרימליט")
    st.stop()
else:
    st.success("✅ המפתח נמצא בכספת")

# 2. ניסיון התחברות ל-Google Auth
st.subheader("2. ניסיון אימות (Authentication)")
try:
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds_info = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(creds_info, scopes=scope)
    client = gspread.authorize(creds)
    st.success("✅ התחברות לשירותי גוגל הצליחה!")
except Exception as e:
    st.error(f"❌ נכשלה ההתחברות לגוגל. שגיאה: {e}")
    st.stop()

# 3. בדיקת גישה לקובץ ספציפי
st.subheader("3. בדיקת גישה לקובץ וטאבים")
sheet_url = st.text_input("הדבק כאן את קישור ה-Google Sheet לבדיקה:")

if sheet_url:
    try:
        sh = client.open_by_url(sheet_url)
        st.success(f"✅ הצלחתי לפתוח את הקובץ: {sh.title}")
        
        # בדיקת קיום הטאבים הדרושים
        required_sheets = ["workers", "requirements", "preferences"]
        existing_sheets = [s.title for s in sh.worksheets()]
        
        for name in required_sheets:
            if name in existing_sheets:
                st.write(f"✔️ גיליון '{name}' נמצא.")
            else:
                st.warning(f"⚠️ גיליון '{name}' חסר בקובץ!")
                
    except gspread.exceptions.SpreadsheetNotFound:
        st.error("❌ הקובץ לא נמצא. וודא שהקישור תקין.")
    except gspread.exceptions.APIError as e:
        if "403" in str(e):
            st.error("❌ שגיאת הרשאה (403).")
            st.info(f"**הפתרון:** עליך לשתף (Share) את הקובץ עם המייל: `{creds_info['client_email']}`")
        else:
            st.error(f"שגיאת API: {e}")
    except Exception as e:
        st.error(f"שגיאה לא צפויה: {e}")
