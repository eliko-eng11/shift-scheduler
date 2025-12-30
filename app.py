import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import sqlite3
import hashlib
import os
import hmac

# =============================
# 1) הגדרות דף וחיבור לגוגל
# =============================
st.set_page_config(page_title="מערכת שיבוץ חכמה בדיקה - Google Sheets", layout="wide")

def get_gspread_client():
    """חיבור מאובטח לגוגל שיטס באמצעות Secrets"""
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    try:
        creds_info = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_info, scopes=scope)
        return gspread.authorize(creds)
    except Exception as e:
        st.error("❌ שגיאה בחיבור ל-Google Cloud. בדוק את ה-Secrets שלך.")
        st.exception(e)
        st.stop()

def get_df_from_sheet(sh, sheet_name):
    """משיכת נתונים מגיליון ספציפי ל-DataFrame"""
    try:
        worksheet = sh.worksheet(sheet_name)
        data = worksheet.get_all_records()
        return pd.DataFrame(data)
    except gspread.exceptions.WorksheetNotFound:
        st.error(f"❌ הגיליון '{sheet_name}' לא נמצא בקובץ!")
        st.stop()

# =============================
# 2) מערכת אימות (SQLite)
# =============================
DB_PATH = "users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return dk.hex()

def verify_user(username: str, password: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT password_hash, salt FROM users WHERE username = ?", (username.strip(),))
    row = cur.fetchone()
    conn.close()
    if not row: return False
    stored_hash, salt = row
    return hmac.compare_digest(stored_hash, hash_password(password, salt))

def auth_gate():
    init_db()
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    
    if st.session_state.logged_in:
        st.sidebar.success(f"מחובר: {st.session_state.username}")
        if st.sidebar.button("התנתקות"):
            st.session_state.logged_in = False
            st.rerun()
        return True

    st.title("🔐 התחברות למערכת")
    u = st.text_input("שם משתמש")
    p = st.text_input("סיסמה", type="password")
    if st.button("התחבר"):
        if verify_user(u, p):
            st.session_state.logged_in = True
            st.session_state.username = u
            st.rerun()
        else:
            st.error("פרטים שגויים")
    st.stop()

# =============================
# 3) אלגוריתם השיבוץ (הלוגיקה שלך)
# =============================
def simple_assignment(cost_matrix):
    used_rows, used_cols, assignments = set(), set(), []
    rows, cols = cost_matrix.shape
    for _ in range(min(rows, cols)):
        best, best_cost = None, 10 ** 12
        for i in range(rows):
            if i in used_rows: continue
            for j in range(cols):
                if j in used_cols: continue
                if cost_matrix[i][j] < best_cost:
                    best_cost, best = cost_matrix[i][j], (i, j)
        if best is None: break
        r, c = best
        assignments.append((r, c))
        used_rows.add(r); used_cols.add(c)
    if not assignments: return [], []
    rr, cc = zip(*assignments)
    return list(rr), list(cc)

def build_schedule(workers_df, req_df, pref_df, week_number):
    # ניקוי שמות עמודות
    for df in [workers_df, req_df, pref_df]:
        df.columns = df.columns.str.strip()
    
    # מיפוי עמודות לפי הלוגיקה שלך
    workers_df = workers_df.rename(columns={"שם עובד": "worker"})
    req_df = req_df.rename(columns={"יום": "day", "משמרת": "shift", "כמות נדרשת": "required"})
    pref_df = pref_df.rename(columns={"עדיפות": "preference", "עובד": "worker", "יום": "day", "משמרת": "shift"})

    workers = workers_df["worker"].dropna().astype(str).str.strip().tolist()
    shift_slots = []
    day_shift_pairs = []

    for _, row in req_df.iterrows():
        req = int(row["required"])
        if req <= 0: continue
        pair = (str(row["day"]).strip(), str(row["shift"]).strip())
        if pair not in day_shift_pairs: day_shift_pairs.append(pair)
        for i in range(req): shift_slots.append((pair[0], pair[1], i))

    ordered_days = list(dict.fromkeys([d for d, _, _ in shift_slots]))
    full_shifts = list(dict.fromkeys([s for _, s, _ in shift_slots]))

    pref_dict = {(str(r["worker"]).strip(), str(r["day"]).strip(), str(r["shift"]).strip()): int(r["preference"]) 
                 for _, r in pref_df.iterrows()}

    worker_copies = [(w, d, s) for w in workers for (d, s) in day_shift_pairs if pref_dict.get((w, d, s), -1) >= 0]
    
    if not worker_copies: raise ValueError("אין מספיק עדיפויות חוקיות בגיליון preferences")

    cost_matrix = []
    for w, d, s in worker_copies:
        row_costs = []
        for sd, ss, _ in shift_slots:
            if (d, s) == (sd, ss):
                p = pref_dict.get((w, d, s), 0)
                row_costs.append(100 if p == 0 else 4 - p)
            else: row_costs.append(1e6)
        cost_matrix.append(row_costs)

    row_ind, col_ind = simple_assignment(np.array(cost_matrix))
    
    # עיבוד תוצאות (לפי המגבלות שלך)
    assignments, used_slots = [], set()
    worker_shift_count = {w: 0 for w in workers}
    worker_daily_shifts = {w: {d: [] for d in ordered_days} for w in workers}
    max_shifts = len(shift_slots) // len(workers) + 1

    for r, c in zip(row_ind, col_ind):
        worker, d, s = worker_copies[r]
        sd, ss, si = shift_slots[c]
        if worker_shift_count[worker] < max_shifts and (sd, ss, si) not in used_slots:
            assignments.append({"שבוע": week_number, "יום": sd, "משמרת": ss, "עובד": worker})
            used_slots.add((sd, ss, si))
            worker_shift_count[worker] += 1
            worker_daily_shifts[worker][sd].append(ss)

    df_res = pd.DataFrame(assignments)
    df_res["יום_מספר"] = df_res["יום"].apply(lambda x: ordered_days.index(x))
    return df_res.sort_values(["יום_מספר", "משמרת"]), []

# =============================
# 4) ממשק משתמש סופי
# =============================
if auth_gate():
    st.title("🛠️ שיבוץ משמרות מבוסס Google Sheets")
    
    sheet_url = st.text_input("הדבק קישור לקובץ גוגל שיטס:")
    week_num = st.number_input("מספר שבוע", min_value=1, value=1)

    if st.button("🚀 בצע שיבוץ ועדכן ענן"):
        if not sheet_url:
            st.warning("אנא הכנס קישור לקובץ.")
        else:
            try:
                client = get_gspread_client()
                sh = client.open_by_url(sheet_url)
                
                with st.spinner("מושך נתונים..."):
                    w_df = get_df_from_sheet(sh, "workers")
                    r_df = get_df_from_sheet(sh, "requirements")
                    p_df = get_df_from_sheet(sh, "preferences")

                res_df, _ = build_schedule(w_df, r_df, p_df, week_num)
                
                st.success("✅ השיבוץ הושלם!")
                st.dataframe(res_df)

                # עדכון הגיליון
                new_sheet_name = f"שבוע {int(week_num)}"
                try:
                    ws = sh.worksheet(new_sheet_name)
                    ws.clear()
                except gspread.exceptions.WorksheetNotFound:
                    ws = sh.add_worksheet(title=new_sheet_name, rows="100", cols="20")

                # כתיבה לענן
                ws.update([res_df.columns.values.tolist()] + res_df.values.tolist())
                st.balloons()
                st.info(f"הנתונים נשמרו בטאב: {new_sheet_name}")

            except Exception as e:
                st.error("שגיאה בתהליך:")
                st.exception(e)

