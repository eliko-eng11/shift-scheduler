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
st.set_page_config(page_title="מערכת שיבוץ חכמה - רישום והתחברות", layout="wide")

def get_gspread_client():
    """חיבור מאובטח לגוגל שיטס באמצעות Secrets"""
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    try:
        creds_info = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_info, scopes=scope)
        return gspread.authorize(creds)
    except Exception as e:
        st.error("❌ שגיאה ב-Secrets: וודא שהגדרת את gcp_service_account ב-Settings.")
        st.stop()

def get_df_from_sheet(sh, sheet_name):
    """משיכת נתונים מגיליון ספציפי"""
    try:
        worksheet = sh.worksheet(sheet_name)
        data = worksheet.get_all_records()
        return pd.DataFrame(data)
    except gspread.exceptions.WorksheetNotFound:
        st.error(f"❌ הגיליון '{sheet_name}' חסר בקובץ הגוגל שיטס!")
        st.stop()

# =============================
# 2) מערכת ניהול משתמשים (Auth)
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

def create_user(username: str, password: str) -> bool:
    username = username.strip()
    if not username or not password: return False
    salt = os.urandom(16).hex()
    p_hash = hash_password(password, salt)
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)", (username, p_hash, salt))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError: return False

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
        st.session_state.username = ""

    if st.session_state.logged_in:
        st.sidebar.success(f"מחובר כ: {st.session_state.username}")
        if st.sidebar.button("התנתקות"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
        return True

    st.title("🔐 ברוכים הבאים למערכת השיבוץ")
    tab_login, tab_reg = st.tabs(["התחברות", "רישום משתמש חדש"])

    with tab_login:
        u = st.text_input("שם משתמש", key="login_u")
        p = st.text_input("סיסמה", type="password", key="login_p")
        if st.button("כניסה"):
            if verify_user(u, p):
                st.session_state.logged_in = True
                st.session_state.username = u
                st.rerun()
            else:
                st.error("שם משתמש או סיסמה לא נכונים")

    with tab_reg:
        new_u = st.text_input("בחר שם משתמש", key="reg_u")
        new_p = st.text_input("בחר סיסמה", type="password", key="reg_p")
        new_p_confirm = st.text_input("אימות סיסמה", type="password", key="reg_pc")
        if st.button("הירשם עכשיו"):
            if new_p != new_p_confirm: st.error("הסיסמאות לא תואמות")
            elif len(new_p) < 4: st.error("סיסמה קצרה מדי")
            else:
                if create_user(new_u, new_p):
                    st.success("נרשמת בהצלחה! עכשיו אפשר להתחבר בלשונית 'התחברות'")
                else: st.error("שם המשתמש כבר תפוס")
    st.stop()

# =============================
# 3) אלגוריתם השיבוץ
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
    # ניקוי ומיפוי נתונים
    workers = workers_df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    req_df.columns = ["day", "shift", "required"]
    pref_df.columns = ["worker", "day", "shift", "preference"]

    shift_slots = []
    day_shift_pairs = []
    for _, row in req_df.iterrows():
        req = int(row["required"])
        pair = (str(row["day"]).strip(), str(row["shift"]).strip())
        if pair not in day_shift_pairs: day_shift_pairs.append(pair)
        for i in range(req): shift_slots.append((pair[0], pair[1], i))

    ordered_days = list(dict.fromkeys([d for d, _, _ in shift_slots]))
    pref_dict = {(str(r["worker"]).strip(), str(r["day"]).strip(), str(r["shift"]).strip()): int(r["preference"]) 
                 for _, r in pref_df.iterrows()}

    worker_copies = [(w, d, s) for w in workers for (d, s) in day_shift_pairs if pref_dict.get((w, d, s), -1) >= 0]
    
    cost_matrix = []
    for w, d, s in worker_copies:
        row_costs = [ (100 if pref_dict.get((w,d,s),0)==0 else 4-pref_dict.get((w,d,s),0)) if (d,s)==(sd,ss) else 1e6 
                     for sd, ss, _ in shift_slots ]
        cost_matrix.append(row_costs)

    row_ind, col_ind = simple_assignment(np.array(cost_matrix))
    
    res = []
    used_slots = set()
    for r, c in zip(row_ind, col_ind):
        worker, d, s = worker_copies[r]
        sd, ss, si = shift_slots[c]
        if (sd, ss, si) not in used_slots:
            res.append({"שבוע": week_number, "יום": sd, "משמרת": ss, "עובד": worker})
            used_slots.add((sd, ss, si))

    df_res = pd.DataFrame(res)
    df_res["יום_מספר"] = df_res["יום"].apply(lambda x: ordered_days.index(x))
    return df_res.sort_values(["יום_מספר", "משמרת"]), []

# =============================
# 4) הרצה ראשית
# =============================
if auth_gate():
    st.title("🛠️ לוח בקרה - שיבוץ משמרות")
    
    url = st.text_input("הדבק קישור לגוגל שיטס שלך:")
    week = st.number_input("מספר שבוע", min_value=1, value=1)

    if st.button("🚀 בצע שיבוץ ועדכן גוגל שיטס"):
        if not url: st.warning("אנא הכנס קישור.")
        else:
            try:
                client = get_gspread_client()
                sh = client.open_by_url(url)
                
                with st.spinner("טוען נתונים מהענן..."):
                    w_df = get_df_from_sheet(sh, "workers")
                    r_df = get_df_from_sheet(sh, "requirements")
                    p_df = get_df_from_sheet(sh, "preferences")

                schedule, _ = build_schedule(w_df, r_df, p_df, week)
                
                st.success("✅ השיבוץ הושלם בהצלחה!")
                st.dataframe(schedule, use_container_width=True)

                # כתיבה חזרה
                new_tab = f"שבוע {int(week)}"
                try:
                    ws = sh.worksheet(new_tab); ws.clear()
                except:
                    ws = sh.add_worksheet(title=new_tab, rows="100", cols="20")
                
                ws.update([schedule.columns.values.tolist()] + schedule.values.tolist())
                st.balloons()
                st.info(f"השיבוץ נשמר בטאב: {new_tab}")

            except gspread.exceptions.APIError as e:
                if "403" in str(e):
                    st.error("❌ שגיאת הרשאה! עליך לשתף את הקובץ (Share) עם המייל של הבוט ב-Secrets.")
                else: st.exception(e)
            except Exception as e:
                st.error("ארעה שגיאה:")
                st.exception(e)
