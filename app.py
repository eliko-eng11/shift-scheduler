import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
# ... (שאר הייבואים שלך: sqlite3, hashlib וכו')

# =============================
# 1) התחברות ל-Google Sheets
# =============================
def get_gspread_client():
    # במידה ואתה מריץ מקומית, וודא שיש לך קובץ secrets.toml או שנה לנתיב לקובץ ה-JSON
    # ב-Streamlit Cloud משתמשים ב-st.secrets
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    
    # טעינת קרדנציאלים מ-Streamlit Secrets
    creds_info = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(creds_info, scopes=scope)
    client = gspread.authorize(creds)
    return client

def get_df_from_sheet(spreadsheet, sheet_name):
    try:
        sheet = spreadsheet.worksheet(sheet_name)
        data = sheet.get_all_records()
        return pd.DataFrame(data)
    except gspread.exceptions.WorksheetNotFound:
        st.error(f"הגיליון '{sheet_name}' לא נמצא בקובץ!")
        return pd.DataFrame()

# =============================
# ... (כאן נשארות פונקציות ה-AUTH וה-simple_assignment שלך ללא שינוי)
# =============================

# (כאן פונקציית build_schedule שלך - נשארת כמעט אותו דבר)
# וודא שהיא מחזירה את ה-schedule_df בסוף

# =============================
# 4) UI של האפליקציה
# =============================
st.title("🛠️ מערכת שיבוץ משמרות - Google Sheets Edition")

# קלט מהמשתמש: קישור לקובץ
sheet_url = st.text_input("הדבק כאן את הקישור (URL) של ה-Google Sheets שלך:")
week_number = st.number_input("מספר שבוע לשיבוץ", min_value=1, step=1, value=1)

if not sheet_url:
    st.info("אנא הכנס קישור לקובץ גוגל שיטס כדי להתחיל. וודא שהקובץ משותף עם ה-Client Email.")
    st.stop()

if st.button("🚀 בצע שיבוץ ועדכן בגוגל שיטס"):
    try:
        client = get_gspread_client()
        # פתיחת הקובץ לפי URL
        sh = client.open_by_url(sheet_url)
        
        with st.spinner("מושך נתונים מהגיליונות..."):
            workers_df = get_df_from_sheet(sh, "workers")
            req_df = get_df_from_sheet(sh, "requirements")
            pref_df = get_df_from_sheet(sh, "preferences")

        if workers_df.empty or req_df.empty or pref_df.empty:
            st.error("אחד או יותר מהגיליונות (workers, requirements, preferences) ריקים או חסרים.")
            st.stop()

        # הרצת האלגוריתם שלך
        schedule_df, unassigned_pairs = build_schedule(workers_df, req_df, pref_df, week_number)

        st.success("✅ השיבוץ הוכן בהצלחה!")
        st.dataframe(schedule_df, use_container_width=True)

        if unassigned_pairs:
            for d, s in unassigned_pairs:
                st.warning(f"⚠️ לא שובץ אף אחד ל־{d} - {s}")

        # עדכון בחזרה לגוגל שיטס
        new_sheet_name = f"שבוע {int(week_number)}"
        
        # בדיקה אם הגיליון כבר קיים - אם כן מוחקים/מנקים אותו, אם לא יוצרים
        try:
            worksheet = sh.worksheet(new_sheet_name)
            worksheet.clear() # מנקה תוכן קיים
        except gspread.exceptions.WorksheetNotFound:
            worksheet = sh.add_worksheet(title=new_sheet_name, rows="100", cols="20")

        # כתיבת הנתונים (כולל כותרות)
        worksheet.update([schedule_df.columns.values.tolist()] + schedule_df.values.tolist())
        
        st.balloons()
        st.success(f"השיבוץ נשמר בהצלחה בגיליון: {new_sheet_name}")

    except Exception as e:
        st.error(f"שגיאה במהלך השיבוץ או הגישה ל-Sheets: {e}")
