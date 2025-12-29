import streamlit as st
import pandas as pd
import numpy as np
import sqlite3, hashlib, os, hmac
import gspread
from google.oauth2.service_account import Credentials

# =============================
# 0) חובה: set_page_config ראשון
# =============================
st.set_page_config(page_title="מערכת שיבוץ (Google Sheets)", layout="wide")

# =============================
# 1) AUTH (SQLite) - Login/Register
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
    if not username or not password:
        return False
    salt = os.urandom(16).hex()
    p_hash = hash_password(password, salt)
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO users(username, password_hash, salt) VALUES (?, ?, ?)", (username, p_hash, salt))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(username: str, password: str) -> bool:
    username = username.strip()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT password_hash, salt FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return False
    stored_hash, salt = row
    check_hash = hash_password(password, salt)
    return hmac.compare_digest(stored_hash, check_hash)

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
        return

    st.title("🔐 התחברות למערכת השיבוץ")
    tab_login, tab_register = st.tabs(["התחברות", "רישום"])

    with tab_login:
        u = st.text_input("שם משתמש", key="login_user")
        p = st.text_input("סיסמה", type="password", key="login_pass")
        if st.button("התחבר"):
            if verify_user(u, p):
                st.session_state.logged_in = True
                st.session_state.username = u.strip()
                st.rerun()
            else:
                st.error("שם משתמש או סיסמה לא נכונים")

    with tab_register:
        new_u = st.text_input("שם משתמש חדש", key="reg_user")
        new_p = st.text_input("סיסמה חדשה", type="password", key="reg_pass")
        new_p2 = st.text_input("אימות סיסמה", type="password", key="reg_pass2")
        if st.button("צור משתמש"):
            if new_p != new_p2:
                st.error("הסיסמאות לא תואמות")
            elif len(new_p) < 4:
                st.error("סיסמה קצרה מדי (מינימום 4 תווים)")
            else:
                ok = create_user(new_u, new_p)
                if ok:
                    st.success("נרשמת בהצלחה! עכשיו תתחבר בלשונית התחברות.")
                else:
                    st.error("שם המשתמש תפוס או נתונים לא תקינים")

    st.stop()

auth_gate()

# =============================
# 2) Google Sheets
# =============================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def extract_sheet_id(url_or_id: str) -> str:
    s = (url_or_id or "").strip()
    if "/spreadsheets/d/" in s:
        return s.split("/spreadsheets/d/")[1].split("/")[0]
    return s

def get_gspread_client():
    if "gcp_service_account" not in st.secrets:
        raise ValueError("חסר Secrets בשם gcp_service_account (JSON של ה-service account).")
    info = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)

def read_sheet_as_df(sh, worksheet_name: str, expected_cols=None) -> pd.DataFrame:
    ws = sh.worksheet(worksheet_name)
    values = ws.get_all_values()
    if not values or len(values) < 2:
        raise ValueError(f"הטאב '{worksheet_name}' ריק או חסר נתונים.")

    df = pd.DataFrame(values[1:], columns=[c.strip() for c in values[0]])
    df.columns = df.columns.str.strip()

    if expected_cols:
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(f"חסרות כותרות בטאב '{worksheet_name}': {missing}")

    return df

def write_df_to_worksheet(sh, worksheet_name: str, df: pd.DataFrame):
    # יוצר/מנקה טאב ואז כותב
    try:
        ws = sh.worksheet(worksheet_name)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=1000, cols=20)

    data = [df.columns.tolist()] + df.astype(str).values.tolist()
    ws.update(data, value_input_option="RAW")

# =============================
# 3) אלגוריתם שיבוץ
# =============================
def simple_assignment(cost_matrix):
    used_rows, used_cols = set(), set()
    assignments = []
    rows = len(cost_matrix)
    cols = len(cost_matrix[0]) if rows > 0 else 0

    for _ in range(min(rows, cols)):
        best, best_cost = None, 10**12
        for i in range(rows):
            if i in used_rows:
                continue
            for j in range(cols):
                if j in used_cols:
                    continue
                c = cost_matrix[i][j]
                if c < best_cost:
                    best_cost = c
                    best = (i, j)
        if best is None:
            break
        r, c = best
        assignments.append((r, c))
        used_rows.add(r)
        used_cols.add(c)

    if not assignments:
        return [], []
    rr, cc = zip(*assignments)
    return list(rr), list(cc)

def build_schedule(workers_df, req_df, pref_df, week_number):
    # ניקוי עמודות
    for df in (workers_df, req_df, pref_df):
        df.columns = df.columns.str.strip()

    # כותרות (אצלך כבר באנגלית, אבל נשאיר תאימות)
    workers_df = workers_df.rename(columns={"שם עובד": "worker"})
    req_df     = req_df.rename(columns={"יום": "day", "משמרת": "shift", "כמות נדרשת": "required"})
    pref_df    = pref_df.rename(columns={"עדיפות": "preference", "עובד": "worker", "יום": "day", "משמרת": "shift"})

    # ניקוי טקסט
    workers_df["worker"] = workers_df["worker"].astype(str).str.strip()
    for df in (req_df, pref_df):
        df["day"] = df["day"].astype(str).str.strip()
        df["shift"] = df["shift"].astype(str).str.strip()
        if "worker" in df.columns:
            df["worker"] = df["worker"].astype(str).str.strip()

    # המרות מספריות (קריטי!)
    req_df["required"] = pd.to_numeric(req_df["required"], errors="coerce").fillna(0).astype(int)
    pref_df["preference"] = pd.to_numeric(pref_df["preference"], errors="coerce").fillna(-1).astype(int)

    workers = workers_df["worker"].dropna().tolist()
    if not workers:
        raise ValueError("לא נמצאו עובדים בטאב workers")

    # בניית סלוטים
    shift_slots = []
    day_shift_pairs = []
    for _, row in req_df.iterrows():
        day, shift, req = row["day"], row["shift"], int(row["required"])
        if req <= 0:
            continue
        pair = (day, shift)
        if pair not in day_shift_pairs:
            day_shift_pairs.append(pair)
        for i in range(req):
            shift_slots.append((day, shift, i))

    if not shift_slots:
        raise ValueError("אין דרישות משמרות בטאב requirements")

    ordered_days = list(dict.fromkeys([d for d, _, _ in shift_slots]))
    full_shifts  = list(dict.fromkeys([s for _, s, _ in shift_slots]))

    # העדפות למילון
    pref_dict = {(r["worker"], r["day"], r["shift"]): int(r["preference"]) for _, r in pref_df.iterrows()}

    # צירופי עובדים אפשריים (pref>=0)
    worker_copies = []
    for w in workers:
        for d, s in day_shift_pairs:
            if pref_dict.get((w, d, s), -1) >= 0:
                worker_copies.append((w, d, s))
    if not worker_copies:
        raise ValueError("אין העדפות חוקיות (>=0) בטאב preferences")

    # עלויות
    cost_matrix = []
    for w, d, s in worker_copies:
        row_costs = []
        for sd, ss, _ in shift_slots:
            if (d, s) == (sd, ss):
                pref = pref_dict.get((w, d, s), 0)
                row_costs.append(100 if pref == 0 else 4 - pref)
            else:
                row_costs.append(1e6)
        cost_matrix.append(row_costs)

    cost_matrix = np.array(cost_matrix, dtype=float)
    row_ind, col_ind = simple_assignment(cost_matrix)

    assignments = []
    used_slots = set()
    worker_shift_count = {w: 0 for w in workers}
    worker_daily_shifts = {w: {d: [] for d in ordered_days} for w in workers}
    worker_day_shift_assigned = set()
    max_shifts_per_worker = len(shift_slots) // len(workers) + 1

    pairs = sorted(zip(row_ind, col_ind), key=lambda x: cost_matrix[x[0], x[1]])

    for r, c in pairs:
        worker, _, _ = worker_copies[r]
        slot_day, slot_shift, slot_i = shift_slots[c]
        slot = (slot_day, slot_shift, slot_i)
        wds_key = (worker, slot_day, slot_shift)

        if cost_matrix[r][c] >= 1e6:
            continue
        if wds_key in worker_day_shift_assigned:
            continue
        if slot in used_slots:
            continue
        if worker_shift_count[worker] >= max_shifts_per_worker:
            continue

        # מניעת משמרות צמודות
        try:
            current_shift_index = full_shifts.index(slot_shift)
        except ValueError:
            current_shift_index = 0

        if any(abs(full_shifts.index(x) - current_shift_index) == 1 for x in worker_daily_shifts[worker][slot_day]):
            continue

        used_slots.add(slot)
        worker_day_shift_assigned.add(wds_key)
        assignments.append({"שבוע": int(week_number), "יום": slot_day, "משמרת": slot_shift, "עובד": worker})
        worker_shift_count[worker] += 1
        worker_daily_shifts[worker][slot_day].append(slot_shift)

    # השלמות
    remaining_slots = [slot for slot in shift_slots if slot not in used_slots]
    unassigned_pairs = set()

    for slot_day, slot_shift, slot_i in remaining_slots:
        assigned = False
        for w in workers:
            if pref_dict.get((w, slot_day, slot_shift), -1) < 0:
                continue

            try:
                current_shift_index = full_shifts.index(slot_shift)
            except ValueError:
                current_shift_index = 0

            if any(abs(full_shifts.index(x) - current_shift_index) == 1 for x in worker_daily_shifts[w][slot_day]):
                continue

            wds_key = (w, slot_day, slot_shift)
            if wds_key in worker_day_shift_assigned:
                continue

            used_slots.add((slot_day, slot_shift, slot_i))
            worker_day_shift_assigned.add(wds_key)
            assignments.append({"שבוע": int(week_number), "יום": slot_day, "משמרת": slot_shift, "עובד": w})
            worker_shift_count[w] += 1
            worker_daily_shifts[w][slot_day].append(slot_shift)
            assigned = True
            break

        if not assigned:
            unassigned_pairs.add((slot_day, slot_shift))

    df = pd.DataFrame(assignments)
    if df.empty:
        raise ValueError("לא נוצר שיבוץ. בדוק נתונים/העדפות.")
    df["יום_מספר"] = df["יום"].apply(lambda x: ordered_days.index(x))
    df = df.sort_values(by=["שבוע", "יום_מספר", "משמרת", "עובד"])
    df = df[["שבוע", "יום", "משמרת", "עובד"]]
    return df, unassigned_pairs

# =============================
# 4) UI
# =============================
st.title("🛠️ מערכת שיבוץ משמרות (Google Sheets)")

sheet_link = st.text_input("הדבק קישור של Google Sheet (עם טאבים workers/requirements/preferences)")
week_number = st.number_input("מספר שבוע לשיבוץ", min_value=1, step=1, value=1)

if st.button("🚀 בצע שיבוץ וכתוב חזרה ל-Google Sheet"):
    try:
        sheet_id = extract_sheet_id(sheet_link)
        if not sheet_id:
            st.error("לא זיהיתי Sheet ID. הדבק קישור מלא של Google Sheets.")
            st.stop()

        gc = get_gspread_client()
        sh = gc.open_by_key(sheet_id)

        # debug מידע ברור
        st.info(f"✅ נפתח הקובץ: {sh.title}")
        st.info(f"✅ טאבים קיימים: {[w.title for w in sh.worksheets()]}")

        workers_df = read_sheet_as_df(sh, "workers", expected_cols=["worker"])
        req_df     = read_sheet_as_df(sh, "requirements", expected_cols=["day","shift","required"])
        pref_df    = read_sheet_as_df(sh, "preferences", expected_cols=["worker","day","shift","preference"])

        st.info(f"📌 workers={len(workers_df)} | requirements={len(req_df)} | preferences={len(pref_df)}")

        schedule_df, unassigned_pairs = build_schedule(workers_df, req_df, pref_df, int(week_number))

        out_name = f"שבוע {int(week_number)}"
        write_df_to_worksheet(sh, out_name, schedule_df)

        st.success(f"✅ נכתב בהצלחה לטאב: {out_name}")
        st.dataframe(schedule_df, use_container_width=True)

        if unassigned_pairs:
            for d, s in sorted(list(unassigned_pairs)):
                st.warning(f"⚠️ לא שובץ אף אחד ל־{d} - {s}")

    except Exception as e:
        st.exception(e)
        st.info("אם זה עדיין נופל: 99% שזה שיתוף/הרשאה. ודא שיתפת את ה-Sheet למייל של ה-service account כ-Editor.")
