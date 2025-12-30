import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import sqlite3, hashlib, os, hmac

# =============================
# 1) חובה: page_config ראשון
# =============================
st.set_page_config(page_title="שיבוץ משמרות (Excel) + התחברות", layout="wide")

# =============================
# 2) AUTH (SQLite) - Login/Register
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
    # normalize headers
    workers_df.columns = workers_df.columns.str.strip()
    req_df.columns = req_df.columns.str.strip()
    pref_df.columns = pref_df.columns.str.strip()

    # accept both hebrew/english headers
    workers_df = workers_df.rename(columns={"שם עובד": "worker", "עובד": "worker"})
    req_df = req_df.rename(columns={"יום": "day", "משמרת": "shift", "כמות נדרשת": "required"})
    pref_df = pref_df.rename(columns={"עדיפות": "preference", "עובד": "worker", "יום": "day", "משמרת": "shift"})

    # required columns check
    if "worker" not in workers_df.columns:
        raise ValueError("בגליון workers חייבת להיות עמודה בשם worker (או 'שם עובד').")
    if not all(c in req_df.columns for c in ["day", "shift", "required"]):
        raise ValueError("בגליון requirements חייבות להיות העמודות: day, shift, required (או בעברית).")
    if not all(c in pref_df.columns for c in ["worker", "day", "shift", "preference"]):
        raise ValueError("בגליון preferences חייבות להיות העמודות: worker, day, shift, preference (או בעברית).")

    # clean strings
    workers_df["worker"] = workers_df["worker"].astype(str).str.strip()
    req_df["day"] = req_df["day"].astype(str).str.strip()
    req_df["shift"] = req_df["shift"].astype(str).str.strip()
    pref_df["worker"] = pref_df["worker"].astype(str).str.strip()
    pref_df["day"] = pref_df["day"].astype(str).str.strip()
    pref_df["shift"] = pref_df["shift"].astype(str).str.strip()

    workers = workers_df["worker"].dropna().tolist()
    if not workers:
        raise ValueError("לא נמצאו עובדים בגיליון workers.")

    req_df["required"] = pd.to_numeric(req_df["required"], errors="coerce").fillna(0).astype(int)

    # build slots
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
        raise ValueError("לא נמצאו דרישות משמרות בגיליון requirements (required צריך להיות > 0).")

    ordered_days = list(dict.fromkeys([d for d, _, _ in shift_slots]))
    full_shifts = list(dict.fromkeys([s for _, s, _ in shift_slots]))

    # preferences dict
    pref_dict = {}
    for _, row in pref_df.iterrows():
        try:
            p = int(row["preference"])
        except Exception:
            continue
        pref_dict[(str(row["worker"]), str(row["day"]), str(row["shift"]))] = p

    # only legal prefs >= 0
    worker_copies = []
    for w in workers:
        for (d, s) in day_shift_pairs:
            p = pref_dict.get((w, d, s), -1)
            if p >= 0:
                worker_copies.append((w, d, s))

    if not worker_copies:
        raise ValueError("לא נמצאו העדפות חוקיות (preference >= 0) בגיליון preferences.")

    # cost matrix
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

    pairs = list(zip(row_ind, col_ind))
    pairs.sort(key=lambda x: cost_matrix[x[0], x[1]])

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

        try:
            current_shift_index = full_shifts.index(slot_shift)
        except ValueError:
            current_shift_index = 0

        # avoid adjacent shifts same day
        if any(abs(full_shifts.index(x) - current_shift_index) == 1 for x in worker_daily_shifts[worker][slot_day]):
            continue

        used_slots.add(slot)
        worker_day_shift_assigned.add(wds_key)
        assignments.append({"שבוע": int(week_number), "יום": slot_day, "משמרת": slot_shift, "עובד": worker})
        worker_shift_count[worker] += 1
        worker_daily_shifts[worker][slot_day].append(slot_shift)

    # fill remaining slots greedily
    remaining_slots = [slot for slot in shift_slots if slot not in used_slots]
    unassigned_pairs = set()

    for slot_day, slot_shift, slot_i in remaining_slots:
        assigned = False
        for w in workers:
            pref = pref_dict.get((w, slot_day, slot_shift), -1)
            if pref < 0:
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
        raise ValueError("לא נוצר אף שיבוץ. בדוק נתונים ב־requirements/preferences.")
    df["יום_מספר"] = df["יום"].apply(lambda x: ordered_days.index(x))
    df = df.sort_values(by=["שבוע", "יום_מספר", "משמרת", "עובד"])
    df = df[["שבוע", "יום", "משמרת", "עובד"]]
    return df, unassigned_pairs

# =============================
# 4) Excel helpers - שימור היסטוריה
# =============================
def safe_new_sheet_name(existing_names, base_name: str) -> str:
    """אם 'שבוע 1' כבר קיים, ייצור 'שבוע 1 (2)' וכו' כדי לא לדרוס היסטוריה."""
    if base_name not in existing_names:
        return base_name
    i = 2
    while True:
        candidate = f"{base_name} ({i})"
        if candidate not in existing_names:
            return candidate
        i += 1

# =============================
# 5) UI - Excel
# =============================
st.title("🧠 מערכת שיבוץ משמרות (Excel)")
uploaded = st.file_uploader("העלה קובץ Excel (xlsx) עם טאבים: workers / requirements / preferences", type=["xlsx"])
week_number = st.number_input("מספר שבוע לשיבוץ", min_value=1, step=1, value=1)

if uploaded and st.button("🚀 בצע שיבוץ והורד קובץ חדש"):
    try:
        # לקרוא את כל הגליונות מהקובץ המקורי
        xls = pd.ExcelFile(uploaded)  # requires openpyxl installed in environment
        sheet_names = xls.sheet_names
        lower_map = {s.lower(): s for s in sheet_names}

        needed = {"workers", "requirements", "preferences"}
        if not needed.issubset(set(lower_map.keys())):
            st.error(f"חסרים טאבים. צריך: {sorted(list(needed))}. יש לך: {sheet_names}")
            st.stop()

        # לקרוא את שלושת הגליונות הנדרשים
        workers_df = pd.read_excel(uploaded, sheet_name=lower_map["workers"])
        req_df     = pd.read_excel(uploaded, sheet_name=lower_map["requirements"])
        pref_df    = pd.read_excel(uploaded, sheet_name=lower_map["preferences"])

        schedule_df, unassigned = build_schedule(workers_df, req_df, pref_df, int(week_number))

        # להכין קובץ פלט: כל הגליונות המקוריים + גליון חדש לשבוע
        out = BytesIO()
        base_new_name = f"שבוע {int(week_number)}"
        new_sheet_name = safe_new_sheet_name(sheet_names, base_new_name)

        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            # כותבים את *כל* הגליונות המקוריים כמו שהם (כדי לשמור היסטוריה/שאר טאבים)
            for s in sheet_names:
                df_s = pd.read_excel(uploaded, sheet_name=s)
                df_s.to_excel(writer, sheet_name=s, index=False)

            # מוסיפים את גליון השיבוץ החדש
            schedule_df.to_excel(writer, sheet_name=new_sheet_name, index=False)

        out.seek(0)

        st.success(f"✅ מוכן! נוסף גליון חדש: {new_sheet_name} (בלי למחוק שום גליון).")
        st.dataframe(schedule_df, use_container_width=True)

        if unassigned:
            st.warning("⚠️ משמרות שלא שובצו:")
            for d, s in sorted(list(unassigned)):
                st.write(f"- {d} / {s}")

        st.download_button(
            "⬇️ הורד קובץ אקסל חדש",
            data=out.getvalue(),
            file_name=f"shift_schedule_week_{int(week_number)}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.exception(e)
