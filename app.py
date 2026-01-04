import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import sqlite3, hashlib, os, hmac
from datetime import datetime

# DB (Neon/Postgres)
from sqlalchemy import create_engine, text

# Charts
import plotly.express as px

# =============================
# 1) חובה: page_config ראשון
# =============================
st.set_page_config(page_title="שיבוץ משמרות (Excel) + DB", layout="wide")

# =============================
# 2) סטייל: RTL + הגדלת כתב
# =============================
st.markdown(
    """
    <style>
      html, body, [class*="css"]  { direction: rtl; text-align: right; }
      .stApp { direction: rtl; }
      /* הגדלת כתב כללית */
      p, li, label, span, div { font-size: 1.05rem !important; }
      /* כותרות קצת יותר גדולות */
      h1 { font-size: 2.1rem !important; }
      h2 { font-size: 1.7rem !important; }
      h3 { font-size: 1.35rem !important; }
      /* כפתורים קצת יותר גדולים */
      .stButton button { font-size: 1.05rem !important; padding: 0.6rem 1rem; }
      /* טבלאות */
      .stDataFrame { font-size: 1.0rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# 3) AUTH (SQLite) - Login/Register
# =============================
DB_PATH = "users.db"

def init_auth_db():
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
    init_auth_db()

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

    st.title("התחברות למערכת השיבוץ")
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
# 4) DB (Neon/Postgres) חיבור + טבלה למערכת מידע
# =============================
def get_engine():
    # חייב להיות ב-Secrets:
    # [db]
    # url = "postgresql://..."
    db_url = st.secrets["db"]["url"]
    return create_engine(db_url, pool_pre_ping=True)

engine = get_engine()

def init_info_db():
    # טבלת "מערכת מידע" לשיבוצים
    ddl = """
    CREATE TABLE IF NOT EXISTS schedules (
        id BIGSERIAL PRIMARY KEY,
        username TEXT NOT NULL,
        customer_name TEXT NOT NULL,
        week INT NOT NULL,
        day TEXT NOT NULL,
        shift TEXT NOT NULL,
        worker TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_schedules_user_week ON schedules(username, week);
    """
    with engine.begin() as conn:
        # אפשר להריץ כמה statements בבת אחת ב-Postgres
        for stmt in ddl.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))

init_info_db()

def overwrite_week_schedule(username: str, customer_name: str, week: int, schedule_df: pd.DataFrame):
    """
    דריסה לפי שבוע (עבור המשתמש).
    אם כבר יש נתונים על אותו שבוע -> מוחקים ואז מכניסים מחדש.
    """
    if schedule_df.empty:
        return

    rows = []
    now = datetime.utcnow().isoformat()
    for _, r in schedule_df.iterrows():
        rows.append({
            "username": username,
            "customer_name": customer_name,
            "week": int(week),
            "day": str(r["יום"]),
            "shift": str(r["משמרת"]),
            "worker": str(r["עובד"]),
            "created_at": now
        })

    with engine.begin() as conn:
        # 1) מוחקים הכל לשבוע הזה (למשתמש הזה) -> זה ה"דריסה"
        conn.execute(
            text("DELETE FROM schedules WHERE username = :u AND week = :w"),
            {"u": username, "w": int(week)}
        )
        # 2) מכניסים מחדש
        conn.execute(
            text("""
                INSERT INTO schedules (username, customer_name, week, day, shift, worker, created_at)
                VALUES (:username, :customer_name, :week, :day, :shift, :worker, :created_at)
            """),
            rows
        )

def load_week_schedule(username: str, week: int) -> pd.DataFrame:
    with engine.connect() as conn:
        res = conn.execute(
            text("""
                SELECT customer_name AS "לקוח",
                       week AS "שבוע",
                       day AS "יום",
                       shift AS "משמרת",
                       worker AS "עובד",
                       created_at AS "נוצר בתאריך"
                FROM schedules
                WHERE username = :u AND week = :w
                ORDER BY day, shift, worker
            """),
            {"u": username, "w": int(week)}
        )
        rows = res.mappings().all()
    return pd.DataFrame(rows)

def list_weeks(username: str):
    with engine.connect() as conn:
        res = conn.execute(
            text("SELECT DISTINCT week FROM schedules WHERE username = :u ORDER BY week DESC"),
            {"u": username}
        )
        return [int(r[0]) for r in res.fetchall()]

# =============================
# 5) אלגוריתם שיבוץ (כמו שלך)
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
    workers_df.columns = workers_df.columns.str.strip()
    req_df.columns = req_df.columns.str.strip()
    pref_df.columns = pref_df.columns.str.strip()

    workers_df = workers_df.rename(columns={"שם עובד": "worker", "עובד": "worker"})
    req_df = req_df.rename(columns={"יום": "day", "משמרת": "shift", "כמות נדרשת": "required"})
    pref_df = pref_df.rename(columns={"עדיפות": "preference", "עובד": "worker", "יום": "day", "משמרת": "shift"})

    if "worker" not in workers_df.columns:
        raise ValueError("בגליון workers חייבת להיות עמודה בשם worker (או 'שם עובד').")
    if not all(c in req_df.columns for c in ["day", "shift", "required"]):
        raise ValueError("בגליון requirements חייבות להיות העמודות: day, shift, required (או בעברית).")
    if not all(c in pref_df.columns for c in ["worker", "day", "shift", "preference"]):
        raise ValueError("בגליון preferences חייבות להיות העמודות: worker, day, shift, preference (או בעברית).")

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

    pref_dict = {}
    for _, row in pref_df.iterrows():
        try:
            p = int(row["preference"])
        except Exception:
            continue
        pref_dict[(str(row["worker"]), str(row["day"]), str(row["shift"]))] = p

    worker_copies = []
    for w in workers:
        for (d, s) in day_shift_pairs:
            p = pref_dict.get((w, d, s), -1)
            if p >= 0:
                worker_copies.append((w, d, s))

    if not worker_copies:
        raise ValueError("לא נמצאו העדפות חוקיות (preference >= 0) בגיליון preferences.")

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

        if any(abs(full_shifts.index(x) - current_shift_index) == 1 for x in worker_daily_shifts[worker][slot_day]):
            continue

        used_slots.add(slot)
        worker_day_shift_assigned.add(wds_key)
        assignments.append({"שבוע": int(week_number), "יום": slot_day, "משמרת": slot_shift, "עובד": worker})
        worker_shift_count[worker] += 1
        worker_daily_shifts[worker][slot_day].append(slot_shift)

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
# 6) Excel helpers - שימור היסטוריה
# =============================
def safe_new_sheet_name(existing_names, base_name: str) -> str:
    if base_name not in existing_names:
        return base_name
    i = 2
    while True:
        candidate = f"{base_name} ({i})"
        if candidate not in existing_names:
            return candidate
        i += 1

# =============================
# 7) UI: ניווט + דפים
# =============================
username = st.session_state.username

st.sidebar.title("תפריט")
page = st.sidebar.radio("ניווט", ["שיבוץ", "דשבורד"], index=0)

# -----------------------------
# PAGE: שיבוץ
# -----------------------------
if page == "שיבוץ":
    st.title("מערכת שיבוץ משמרות (Excel)")

    customer_name = st.text_input("שם הלקוח", placeholder="לדוגמה: לקוח א'")
    uploaded = st.file_uploader("העלה קובץ Excel (xlsx) עם טאבים: workers / requirements / preferences", type=["xlsx"])
    week_number = st.number_input("מספר שבוע לשיבוץ", min_value=1, step=1, value=1)

    colA, colB = st.columns([1, 2], vertical_alignment="center")
    with colA:
        run_btn = st.button("בצע שיבוץ ושמור למערכת מידע")
    with colB:
        st.caption("הערה: שמירה למערכת מידע *תדרוס* נתונים קיימים לאותו שבוע.")

    if uploaded and run_btn:
        if not customer_name.strip():
            st.error("חובה למלא שם לקוח לפני שמירה.")
            st.stop()

        try:
            xls = pd.ExcelFile(uploaded)
            sheet_names = xls.sheet_names
            lower_map = {s.lower(): s for s in sheet_names}

            needed = {"workers", "requirements", "preferences"}
            if not needed.issubset(set(lower_map.keys())):
                st.error(f"חסרים טאבים. צריך: {sorted(list(needed))}. יש לך: {sheet_names}")
                st.stop()

            workers_df = pd.read_excel(uploaded, sheet_name=lower_map["workers"])
            req_df     = pd.read_excel(uploaded, sheet_name=lower_map["requirements"])
            pref_df    = pd.read_excel(uploaded, sheet_name=lower_map["preferences"])

            schedule_df, unassigned = build_schedule(workers_df, req_df, pref_df, int(week_number))

            # להוסיף עמודת לקוח לתצוגה/אקסל (לא חובה אבל ביקשת "להכניס לרשומות")
            schedule_df_display = schedule_df.copy()
            schedule_df_display.insert(0, "לקוח", customer_name.strip())

            # 1) שמירה ל-DB (דריסה לפי שבוע)
            overwrite_week_schedule(
                username=username,
                customer_name=customer_name.strip(),
                week=int(week_number),
                schedule_df=schedule_df
            )

            st.success(f"✅ השיבוץ נשמר למערכת מידע (שבוע {int(week_number)}) — אם היה קיים, הוא נדרס.")
            st.dataframe(schedule_df_display, use_container_width=True)

            if unassigned:
                st.warning("⚠️ משמרות שלא שובצו:")
                for d, s in sorted(list(unassigned)):
                    st.write(f"- {d} / {s}")

            # 2) יצוא לקובץ אקסל: כל הגליונות המקוריים + גליון שבוע חדש
            out = BytesIO()
            base_new_name = f"שבוע {int(week_number)}"
            new_sheet_name = safe_new_sheet_name(sheet_names, base_new_name)

            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                # שומרים את כל הגליונות המקוריים
                for s in sheet_names:
                    df_s = pd.read_excel(uploaded, sheet_name=s)
                    df_s.to_excel(writer, sheet_name=s, index=False)

                # מוסיפים גליון שיבוץ חדש עם "לקוח"
                schedule_df_display.to_excel(writer, sheet_name=new_sheet_name, index=False)

            out.seek(0)

            st.download_button(
                "הורד קובץ אקסל חדש",
                data=out.getvalue(),
                file_name=f"shift_schedule_week_{int(week_number)}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        except Exception as e:
            st.error("❌ שגיאה בתהליך")
            st.exception(e)

# -----------------------------
# PAGE: דשבורד
# -----------------------------
else:
    st.title("דשבורד")

    weeks = list_weeks(username)
    if not weeks:
        st.info("אין עדיין נתונים במערכת המידע. בצע שיבוץ ושמור למערכת.")
        st.stop()

    week_selected = st.selectbox("בחר שבוע להצגה", options=weeks, index=0)

    df_week = load_week_schedule(username, week_selected)
    if df_week.empty:
        st.warning("לא נמצאו נתונים לשבוע הזה.")
        st.stop()

    st.subheader("טבלת שיבוצים מהמערכת")
    st.dataframe(df_week, use_container_width=True)

    # תרשים: כמה עבד כל עובד בחלוקה לימים (stacked)
    st.subheader("כמה עבד כל עובד בחלוקה לימים")

    # נניח שכל שורה = משמרת אחת
    chart_df = (
        df_week.groupby(["עובד", "יום"])
        .size()
        .reset_index(name="כמות משמרות")
    )

    fig = px.bar(
        chart_df,
        x="עובד",
        y="כמות משמרות",
        color="יום",
        barmode="stack",
        title=f"שבוע {week_selected} — כמות משמרות לכל עובד (לפי ימים)"
    )
    st.plotly_chart(fig, use_container_width=True)
