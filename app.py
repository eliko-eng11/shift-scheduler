import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import sqlite3
import hashlib
import os
import hmac
import streamlit as st

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
    # PBKDF2-HMAC-SHA256
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
        # username already exists
        return False

def verify_user(username: str, password: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT password_hash, salt FROM users WHERE username = ?", (username.strip(),))
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

    # אם מחובר - מציג התנתקות וממשיך לאפליקציה
    if st.session_state.logged_in:
        st.sidebar.success(f"מחובר כ: {st.session_state.username}")
        if st.sidebar.button("התנתקות"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
        return

    # אם לא מחובר - מציג Login/Register ועוצר פה
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



# -----------------------------
# פונקציית הקצאה חמדנית (במקום scipy)
# -----------------------------
def simple_assignment(cost_matrix):
    """
    מקבל מטריצת עלויות ומחזיר התאמות (rows, cols) בצורה חמדנית.
    זה לא האלגוריתם ההונגרי המלא, אבל עובד טוב לדמו וליישום שלך.
    """
    used_rows = set()
    used_cols = set()
    assignments = []

    rows = len(cost_matrix)
    cols = len(cost_matrix[0]) if rows > 0 else 0

    for _ in range(min(rows, cols)):
        best = None
        best_cost = 10 ** 12
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


# -----------------------------
# בניית שיבוץ מתוך שלושת הגיליונות
# -----------------------------
def build_schedule(workers_df, req_df, pref_df, week_number):
    # ניקוי שמות עמודות
    workers_df.columns = workers_df.columns.str.strip()
    req_df.columns = req_df.columns.str.strip()
    pref_df.columns = pref_df.columns.str.strip()

    # התאמת שמות עמודות בעברית לאנגלית פנימית
    workers_df = workers_df.rename(columns={"שם עובד": "worker"})
    req_df = req_df.rename(columns={"יום": "day", "משמרת": "shift", "כמות נדרשת": "required"})
    pref_df = pref_df.rename(columns={"עדיפות": "preference", "עובד": "worker", "יום": "day", "משמרת": "shift"})

    # ניקוי רווחים מיותרים בשדות הטקסט
    if "worker" in workers_df.columns:
        workers_df["worker"] = workers_df["worker"].astype(str).str.strip()

    for df in (req_df, pref_df):
        if "day" in df.columns:
            df["day"] = df["day"].astype(str).str.strip()
        if "shift" in df.columns:
            df["shift"] = df["shift"].astype(str).str.strip()
        if "worker" in df.columns:
            df["worker"] = df["worker"].astype(str).str.strip()

    # רשימת עובדים
    workers = (
        workers_df["worker"]
        .dropna()
        .astype(str)
        .tolist()
    )

    if not workers:
        raise ValueError("לא נמצאו עובדים בגיליון 'workers'")

    # סלוטים של משמרות לפי הדרישות
    req_df["required"] = req_df["required"].fillna(0).astype(int)
    shift_slots = []
    day_shift_pairs = []

    for _, row in req_df.iterrows():
        day = str(row["day"])
        shift = str(row["shift"])
        req = int(row["required"])

        if req <= 0:
            continue

        pair = (day, shift)
        if pair not in day_shift_pairs:
            day_shift_pairs.append(pair)

        for i in range(req):
            shift_slots.append((day, shift, i))

    if not shift_slots:
        raise ValueError("לא נמצאו דרישות משמרות בגיליון 'requirements'")

    # רשימת ימים ומשמרות לסידור
    ordered_days = list(dict.fromkeys([d for d, _, _ in shift_slots]))
    full_shifts = list(dict.fromkeys([s for _, s, _ in shift_slots]))

    # העדפות למילון
    pref_dict = {}
    for _, row in pref_df.iterrows():
        w = str(row["worker"])
        d = str(row["day"])
        s = str(row["shift"])
        try:
            p = int(row["preference"])
        except Exception:
            continue
        pref_dict[(w, d, s)] = p

    # worker_copies – רק צירופים שהעדפה שלהם >= 0
    worker_copies = []
    for w in workers:
        for (d, s) in day_shift_pairs:
            p = pref_dict.get((w, d, s), -1)
            if p >= 0:
                worker_copies.append((w, d, s))

    if not worker_copies:
        raise ValueError("לא נמצאו העדיפויות החוקיות (>=0) בגיליון 'preferences'")

    # מטריצת עלויות
    cost_matrix = []
    for w, d, s in worker_copies:
        row_costs = []
        for sd, ss, _ in shift_slots:
            if (d, s) == (sd, ss):
                pref = pref_dict.get((w, d, s), 0)
                if pref == 0:
                    # אפשרי אך לא מומלץ
                    row_costs.append(100)
                else:
                    # עדיפות גבוהה = עלות נמוכה
                    row_costs.append(4 - pref)
            else:
                row_costs.append(1e6)
        cost_matrix.append(row_costs)

    cost_matrix = np.array(cost_matrix, dtype=float)

    # הקצאה חמדנית
    row_ind, col_ind = simple_assignment(cost_matrix)

    assignments = []
    used_workers_in_shift = set()          # (worker, day, shift)
    used_slots = set()                     # מלאו סלוט מסוים (day, shift, i)
    worker_shift_count = {w: 0 for w in workers}
    worker_daily_shifts = {w: {d: [] for d in ordered_days} for w in workers}
    worker_day_shift_assigned = set()      # מניעת כפילויות עובד-יום-משמרת

    max_shifts_per_worker = len(shift_slots) // len(workers) + 1

    # סידור לפי עלות
    pairs = list(zip(row_ind, col_ind))
    pairs.sort(key=lambda x: cost_matrix[x[0], x[1]])

    # סיבוב ראשון – הקצאה לפי עלויות (עדיין שומרים על הוגנות)
    for r, c in pairs:
        worker, day, shift = worker_copies[r]
        slot = shift_slots[c]  # (day, shift, i)
        slot_day, slot_shift, _ = slot

        # מפתח ייחודי למניעת כפילות עובד-יום-משמרת
        wds_key = (worker, slot_day, slot_shift)

        if cost_matrix[r][c] >= 1e6:
            continue
        if wds_key in worker_day_shift_assigned:
            continue
        if slot in used_slots:
            continue
        if worker_shift_count[worker] >= max_shifts_per_worker:
            continue

        # בדיקת משמרות צמודות באותו יום
        try:
            current_shift_index = full_shifts.index(shift)
        except ValueError:
            current_shift_index = 0

        if any(
            abs(full_shifts.index(x) - current_shift_index) == 1
            for x in worker_daily_shifts[worker][day]
        ):
            continue

        used_slots.add(slot)
        used_workers_in_shift.add(wds_key)
        worker_day_shift_assigned.add(wds_key)

        assignments.append(
            {"שבוע": week_number, "יום": slot_day, "משמרת": slot_shift, "עובד": worker}
        )
        worker_shift_count[worker] += 1
        worker_daily_shifts[worker][day].append(shift)

    # סיבוב שני – השלמת משמרות שלא שובצו
    # כאן אנו פחות מחמירים עם מגבלת מספר המשמרות לעובד,
    # כדי לוודא שלא נשארות משמרות ריקות.
    remaining_slots = [slot for slot in shift_slots if slot not in used_slots]
    unassigned_pairs = set()

    for slot in remaining_slots:
        d, s, _ = slot
        assigned = False
        for w in workers:
            # לא בודקים כאן את worker_shift_count[w] מול max_shifts_per_worker
            # כי המטרה היא קודם למלא חורים.
            pref = pref_dict.get((w, d, s), -1)
            if pref < 0:
                continue

            try:
                current_shift_index = full_shifts.index(s)
            except ValueError:
                current_shift_index = 0

            if any(
                abs(full_shifts.index(x) - current_shift_index) == 1
                for x in worker_daily_shifts[w][d]
            ):
                continue

            wds_key = (w, d, s)
            if wds_key in worker_day_shift_assigned:
                continue

            used_slots.add(slot)
            used_workers_in_shift.add(wds_key)
            worker_day_shift_assigned.add(wds_key)

            assignments.append(
                {"שבוע": week_number, "יום": d, "משמרת": s, "עובד": w}
            )
            worker_shift_count[w] += 1
            worker_daily_shifts[w][d].append(s)
            assigned = True
            break

        if not assigned:
            unassigned_pairs.add((d, s))

    df = pd.DataFrame(assignments)

    if df.empty:
        raise ValueError("לא נוצר אף שיבוץ. בדוק את הנתונים בגיליונות.")

    # סידור לפי ימים, משמרת, עובד
    df["יום_מספר"] = df["יום"].apply(lambda x: ordered_days.index(x))
    df = df.sort_values(by=["שבוע", "יום_מספר", "משמרת", "עובד"])
    df = df[["שבוע", "יום", "משמרת", "עובד"]]

    return df, unassigned_pairs


# -----------------------------
# אפליקציית Streamlit
# -----------------------------
st.set_page_config(page_title="מערכת שיבוץ חכמה לעובדים", layout="wide")
st.set_page_config(page_title="מערכת שיבוץ חכמה לעובדים", layout="wide")
auth_gate()  # 👈 זה נועל את האפליקציה עד התחברות

st.title("🛠️ מערכת שיבוץ משמרות מעולה")

uploaded_file = st.file_uploader("העלה קובץ אקסל קיים", type=["xlsx"])
week_number = st.number_input("מספר שבוע לשיבוץ", min_value=1, step=1, value=1)

if uploaded_file is None:
    st.info("העלה קובץ אקסל עם הגיליונות workers, requirements, preferences כדי להתחיל.")
    st.stop()

if st.button("🚀 בצע שיבוץ והוסף גיליון חדש לקובץ"):
    try:
        xls = pd.ExcelFile(uploaded_file)

        workers_df = pd.read_excel(xls, sheet_name="workers")
        req_df = pd.read_excel(xls, sheet_name="requirements")
        pref_df = pd.read_excel(xls, sheet_name="preferences")

        schedule_df, unassigned_pairs = build_schedule(
            workers_df, req_df, pref_df, week_number
        )

        # איפוס אינדקס כדי שהעמודה הראשונה לא תהיה 9,0,1...
        schedule_df = schedule_df.reset_index(drop=True)
        schedule_df.index += 1

        st.success("✅ השיבוץ הוכן בהצלחה!")
        st.dataframe(schedule_df, use_container_width=True)

        if unassigned_pairs:
            for d, s in unassigned_pairs:
                st.warning(f"⚠️ לא שובץ אף אחד ל־{d} - {s}")

        # כתיבת הקובץ המעודכן לבאפר
        new_sheet_name = f"שבוע {int(week_number)}"
        original_sheet_names = xls.sheet_names

        if new_sheet_name in original_sheet_names:
            st.warning(
                f"קיים כבר גיליון בשם '{new_sheet_name}'. הגיליון החדש ייקרא '{new_sheet_name} (2)'."
            )
            new_sheet_name = f"{new_sheet_name} (2)"

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for sheet in original_sheet_names:
                df_old = pd.read_excel(xls, sheet_name=sheet)
                df_old.to_excel(writer, sheet_name=sheet, index=False)

            schedule_df.to_excel(writer, sheet_name=new_sheet_name, index=False)

        output.seek(0)

        st.download_button(
            label="⬇️ הורד את הקובץ המעודכן (עם היסטוריית השבועות)",
            data=output,
            file_name=uploaded_file.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"שגיאה במהלך השיבוץ: {e}")

