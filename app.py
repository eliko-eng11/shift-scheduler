import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import hashlib, os, hmac
import psycopg2
from psycopg2.extras import RealDictCursor
import uuid
from datetime import datetime, timezone

# =============================
# 1) Page config (MUST be first)
# =============================
st.set_page_config(page_title="מערכת שיבוץ משמרות", layout="wide")

# =============================
# 2) RTL + Professional UI CSS
# =============================
st.markdown(
    """
    <style>
      html, body, [class*="css"]  {
        direction: rtl;
        text-align: right;
      }
      .stApp { direction: rtl; }
      h1, h2, h3, h4, h5, h6, p, div, label, span { direction: rtl; text-align: right; }
      .block-container { padding-top: 2rem; }
      .stDataFrame, .stTable { direction: rtl; }
      /* nicer cards */
      .card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 14px 16px;
      }
      .muted { opacity: 0.8; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# 3) DB (Neon/Postgres)
# =============================
def get_db_url() -> str:
    try:
        return st.secrets["database"]["url"]
    except Exception:
        st.error("❌ חסר Secrets: database.url (Streamlit Settings → Secrets)")
        st.stop()

def db_connect():
    return psycopg2.connect(get_db_url(), cursor_factory=RealDictCursor)

def db_init():
    """Create tables if not exist."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schedule_runs (
                    run_id UUID PRIMARY KEY,
                    created_by TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
                    week INTEGER NOT NULL,
                    source_filename TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schedule_rows (
                    id BIGSERIAL PRIMARY KEY,
                    run_id UUID NOT NULL REFERENCES schedule_runs(run_id) ON DELETE CASCADE,
                    week INTEGER NOT NULL,
                    day TEXT NOT NULL,
                    shift TEXT NOT NULL,
                    worker TEXT NOT NULL
                );
            """)
        conn.commit()

db_init()

def db_fetch_all(query: str, params=None):
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
            return cur.fetchall()

def db_execute(query: str, params=None):
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
        conn.commit()

# =============================
# 4) Auth helpers (Postgres)
# =============================
def hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return dk.hex()

def create_user(username: str, password: str) -> bool:
    username = (username or "").strip()
    if not username or not password:
        return False

    salt = os.urandom(16).hex()
    p_hash = hash_password(password, salt)

    try:
        db_execute(
            "INSERT INTO users(username, password_hash, salt) VALUES (%s, %s, %s)",
            (username, p_hash, salt),
        )
        return True
    except Exception:
        return False

def verify_user(username: str, password: str) -> bool:
    username = (username or "").strip()
    rows = db_fetch_all(
        "SELECT password_hash, salt FROM users WHERE username = %s",
        (username,),
    )
    if not rows:
        return False
    stored_hash = rows[0]["password_hash"]
    salt = rows[0]["salt"]
    check_hash = hash_password(password, salt)
    return hmac.compare_digest(stored_hash, check_hash)

def auth_gate():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ""

    if st.session_state.logged_in:
        st.sidebar.markdown(f"<div class='card'>✅ מחובר כ: <b>{st.session_state.username}</b></div>", unsafe_allow_html=True)
        if st.sidebar.button("התנתקות"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
        return

    st.title("🔐 התחברות למערכת השיבוץ")
    st.markdown("<div class='muted'>המערכת שומרת משתמשים והיסטוריית שיבוצים במסד נתונים חיצוני (Postgres).</div>", unsafe_allow_html=True)

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
# 5) Scheduling algorithm (your logic)
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
# 6) Excel helper
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
# 7) Save schedule to DB
# =============================
def save_schedule_to_db(schedule_df: pd.DataFrame, week_number: int, source_filename: str, created_by: str) -> uuid.UUID:
    run_id = uuid.uuid4()
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO schedule_runs(run_id, created_by, week, source_filename) VALUES (%s, %s, %s, %s)",
                (str(run_id), created_by, int(week_number), source_filename),
            )

            rows = []
            for _, r in schedule_df.iterrows():
                rows.append((str(run_id), int(r["שבוע"]), str(r["יום"]), str(r["משמרת"]), str(r["עובד"])))

            cur.executemany(
                "INSERT INTO schedule_rows(run_id, week, day, shift, worker) VALUES (%s, %s, %s, %s, %s)",
                rows,
            )
        conn.commit()
    return run_id

def load_run_rows(run_id: str) -> pd.DataFrame:
    rows = db_fetch_all(
        "SELECT week AS שבוע, day AS יום, shift AS משמרת, worker AS עובד FROM schedule_rows WHERE run_id = %s ORDER BY id",
        (run_id,),
    )
    return pd.DataFrame(rows)

# =============================
# 8) Layout / Navigation
# =============================
st.sidebar.markdown("<div class='card'><b>ניווט</b><div class='muted'>בחר מה לעשות</div></div>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "תפריט",
    ["שיבוץ מאקסל", "היסטוריה", "דאשבורד"],
    label_visibility="collapsed",
)

# =============================
# 9) Page: Excel → Schedule
# =============================
if page == "שיבוץ מאקסל":
    st.title("🧠 מערכת שיבוץ משמרות (Excel)")

    st.markdown(
        "<div class='card'>"
        "<b>איך זה עובד?</b><br>"
        "<span class='muted'>מעלים קובץ Excel עם טאבים: workers / requirements / preferences → המערכת מייצרת שיבוץ → שומרת ל-DB → ומחזירה קובץ חדש עם גליון נוסף.</span>"
        "</div>",
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader("העלה קובץ Excel (xlsx)", type=["xlsx"])
    week_number = st.number_input("מספר שבוע לשיבוץ", min_value=1, step=1, value=1)

    if uploaded and st.button("🚀 בצע שיבוץ"):
        try:
            xls = pd.ExcelFile(uploaded)
            sheet_names = xls.sheet_names
            lower_map = {s.lower(): s for s in sheet_names}

            needed = {"workers", "requirements", "preferences"}
            if not needed.issubset(set(lower_map.keys())):
                st.error(f"חסרים טאבים. צריך: {sorted(list(needed))}. יש לך: {sheet_names}")
                st.stop()

            workers_df = pd.read_excel(uploaded, sheet_name=lower_map["workers"])
            req_df = pd.read_excel(uploaded, sheet_name=lower_map["requirements"])
            pref_df = pd.read_excel(uploaded, sheet_name=lower_map["preferences"])

            schedule_df, unassigned = build_schedule(workers_df, req_df, pref_df, int(week_number))

            # Save to DB
            run_id = save_schedule_to_db(
                schedule_df=schedule_df,
                week_number=int(week_number),
                source_filename=getattr(uploaded, "name", None),
                created_by=st.session_state.username
            )

            # Build output Excel with original sheets + new schedule sheet
            out = BytesIO()
            base_new_name = f"שבוע {int(week_number)}"
            new_sheet_name = safe_new_sheet_name(sheet_names, base_new_name)

            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                for s in sheet_names:
                    df_s = pd.read_excel(uploaded, sheet_name=s)
                    df_s.to_excel(writer, sheet_name=s, index=False)
                schedule_df.to_excel(writer, sheet_name=new_sheet_name, index=False)

            out.seek(0)

            st.success(f"✅ שיבוץ הוכן ונשמר למסד נתונים! מזהה ריצה: {run_id}")
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
            st.error("❌ שגיאה בתהליך השיבוץ")
            st.exception(e)

# =============================
# 10) Page: History
# =============================
elif page == "היסטוריה":
    st.title("🗂️ היסטוריית שיבוצים")

    runs = db_fetch_all(
        """
        SELECT run_id, week, source_filename, created_at
        FROM schedule_runs
        WHERE created_by = %s
        ORDER BY created_at DESC
        LIMIT 50
        """,
        (st.session_state.username,),
    )

    if not runs:
        st.info("אין עדיין ריצות שיבוץ בחשבון הזה. עבור לעמוד 'שיבוץ מאקסל' והריץ פעם ראשונה.")
        st.stop()

    runs_df = pd.DataFrame(runs)
    runs_df["created_at"] = runs_df["created_at"].astype(str)

    st.markdown("<div class='card'><b>הריצות האחרונות שלך</b></div>", unsafe_allow_html=True)
    st.dataframe(runs_df.rename(columns={
        "run_id": "מזהה ריצה",
        "week": "שבוע",
        "source_filename": "קובץ מקור",
        "created_at": "נוצר בתאריך",
    }), use_container_width=True)

    run_ids = [str(r["run_id"]) for r in runs]
    selected = st.selectbox("בחר ריצה להצגה", run_ids)

    if selected:
        df_rows = load_run_rows(selected)
        st.markdown("<div class='card'><b>תוצאות שיבוץ</b></div>", unsafe_allow_html=True)
        st.dataframe(df_rows, use_container_width=True)

        # download this run as excel
        out = BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df_rows.to_excel(writer, sheet_name="Schedule", index=False)
        out.seek(0)

        st.download_button(
            "⬇️ הורד את הריצה הזו כ-Excel",
            data=out.getvalue(),
            file_name=f"schedule_run_{selected}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# =============================
# 11) Page: Dashboard
# =============================
elif page == "דאשבורד":
    st.title("📊 דאשבורד")

    # Pull recent rows for analysis
    rows = db_fetch_all(
        """
        SELECT sr.week, sr.created_at, s.day, s.shift, s.worker
        FROM schedule_rows s
        JOIN schedule_runs sr ON sr.run_id = s.run_id
        WHERE sr.created_by = %s
        ORDER BY sr.created_at DESC
        LIMIT 5000
        """,
        (st.session_state.username,),
    )

    if not rows:
        st.info("אין מספיק נתונים לדאשבורד עדיין. תבצע שיבוץ מאקסל קודם.")
        st.stop()

    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"])

    col1, col2, col3 = st.columns(3)
    col1.metric("כמות שיבוצים (שורות)", int(len(df)))
    col2.metric("מספר שבועות", int(df["week"].nunique()))
    col3.metric("כמות עובדים שונים", int(df["worker"].nunique()))

    st.markdown("<div class='card'><b>התפלגות משמרות לפי עובד</b></div>", unsafe_allow_html=True)
    by_worker = df.groupby("worker").size().sort_values(ascending=False).head(20)
    st.bar_chart(by_worker)

    st.markdown("<div class='card'><b>התפלגות משמרות לפי יום</b></div>", unsafe_allow_html=True)
    by_day = df.groupby("day").size().sort_values(ascending=False)
    st.bar_chart(by_day)

    st.markdown("<div class='card'><b>10 הריצות האחרונות</b></div>", unsafe_allow_html=True)
    runs = db_fetch_all(
        """
        SELECT run_id, week, source_filename, created_at
        FROM schedule_runs
        WHERE created_by = %s
        ORDER BY created_at DESC
        LIMIT 10
        """,
        (st.session_state.username,),
    )
    st.dataframe(pd.DataFrame(runs).assign(created_at=lambda x: x["created_at"].astype(str)), use_container_width=True)
