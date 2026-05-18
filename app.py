import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import sqlite3, hashlib, os, hmac
import psycopg2
import psycopg2.extras
import plotly.express as px

# =============================
# 1) חובה: page_config ראשון
# =============================
st.set_page_config(page_title="מערכת שיבוץ משמרות", layout="wide")

# =============================
# RTL + UI (כתב מוגדל ומסודר)
# =============================
st.markdown(
    """
    <style>
      html, body, [class*="css"]  { direction: rtl; text-align: right; font-size: 18px; }
      h1, h2, h3, h4, h5, h6 { direction: rtl; text-align: right; }
      .stDataFrame, .stTable { direction: rtl; }
      label, p, div { direction: rtl; }
      section[data-testid="stSidebar"] { direction: rtl; }
      .block-container { padding-top: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# DB (Neon / Postgres)
# =============================
def get_pg_conn():
    if "db" not in st.secrets or "url" not in st.secrets["db"]:
        st.error("חסר Secrets: db.url (Streamlit Settings → Secrets)")
        st.stop()
    return psycopg2.connect(st.secrets["db"]["url"])

def init_pg():
    conn = get_pg_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS schedules (
            id BIGSERIAL PRIMARY KEY,
            username TEXT NOT NULL,
            customer_name TEXT NOT NULL,
            week INT NOT NULL,
            day TEXT NOT NULL,
            shift TEXT NOT NULL,
            worker TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_schedules_user_week
        ON schedules(username, customer_name, week);
    """)
    conn.commit()
    cur.close()
    conn.close()

def upsert_week_schedule(username: str, customer_name: str, week: int, df: pd.DataFrame):
    """
    דריסה לפי: username + customer_name + week
    מוחק מה שהיה לשבוע הזה ומכניס חדש.
    """
    conn = get_pg_conn()
    cur = conn.cursor()

    cur.execute(
        "DELETE FROM schedules WHERE username=%s AND customer_name=%s AND week=%s",
        (username, customer_name, week)
    )

    rows = []
    for _, r in df.iterrows():
        rows.append((
            username,
            customer_name,
            int(week),
            str(r["יום"]),
            str(r["משמרת"]),
            str(r["עובד"])
        ))

    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO schedules (username, customer_name, week, day, shift, worker)
        VALUES %s
        """,
        rows
    )
    conn.commit()
    cur.close()
    conn.close()

def list_weeks(username: str, customer_name: str | None = None):
    conn = get_pg_conn()
    cur = conn.cursor()
    if customer_name:
        cur.execute(
            """
            SELECT DISTINCT week
            FROM schedules
            WHERE username=%s AND customer_name=%s
            ORDER BY week DESC
            """,
            (username, customer_name)
        )
    else:
        cur.execute(
            """
            SELECT DISTINCT week
            FROM schedules
            WHERE username=%s
            ORDER BY week DESC
            """,
            (username,)
        )
    weeks = [int(x[0]) for x in cur.fetchall()]
    cur.close()
    conn.close()
    return weeks

def list_customers(username: str):
    conn = get_pg_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT customer_name
        FROM schedules
        WHERE username=%s
        ORDER BY customer_name ASC
        """,
        (username,)
    )
    customers = [x[0] for x in cur.fetchall()]
    cur.close()
    conn.close()
    return customers

def load_week_schedule(username: str, customer_name: str, week: int) -> pd.DataFrame:
    conn = get_pg_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT customer_name AS "לקוח",
               week AS "שבוע",
               day AS "יום",
               shift AS "משמרת",
               worker AS "עובד",
               created_at AS "נוצר בתאריך"
        FROM schedules
        WHERE username=%s AND customer_name=%s AND week=%s
        ORDER BY day, shift, worker
        """,
        (username, customer_name, week)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return pd.DataFrame(rows)

def load_all_schedules(username: str) -> pd.DataFrame:
    conn = get_pg_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT customer_name AS "לקוח",
               week AS "שבוע",
               day AS "יום",
               shift AS "משמרת",
               worker AS "עובד",
               created_at AS "נוצר בתאריך"
        FROM schedules
        WHERE username=%s
        ORDER BY week DESC, day, shift, worker
        """,
        (username,)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return pd.DataFrame(rows)

# =============================
# AUTH (SQLite) - Login/Register
# =============================
DB_PATH = "users.db"

def init_sqlite():
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
    init_sqlite()
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

    st.title("🔐 התחברות למערכת")
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

# =============================
# אלגוריתם שיבוץ (שלך)
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
# =============================
# MAX FLOW (בדיקת כיסוי בלבד)
# =============================
from collections import defaultdict, deque

class MaxFlow:
    def __init__(self):
        self.graph = defaultdict(dict)

    def add_edge(self, u, v, capacity):
        self.graph[u][v] = capacity
        self.graph[v][u] = 0

    def bfs(self, s, t, parent):
        visited = set()
        queue = deque([s])
        visited.add(s)

        while queue:
            u = queue.popleft()
            for v in self.graph[u]:
                if v not in visited and self.graph[u][v] > 0:
                    parent[v] = u
                    visited.add(v)
                    queue.append(v)
                    if v == t:
                        return True
        return False

    def max_flow(self, source, sink):
        parent = {}
        flow = 0

        while self.bfs(source, sink, parent):
            path_flow = float('inf')
            s = sink

            while s != source:
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            flow += path_flow

            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

        return flow


def run_max_flow(workers_df, req_df, pref_df):
    mf = MaxFlow()

    source = "S"
    sink = "T"

    workers = workers_df["worker"].tolist()

    shift_slots = []
    for _, row in req_df.iterrows():
        for i in range(int(row["required"])):
            shift_slots.append((row["day"], row["shift"], i))

    pref_dict = {}
    for _, row in pref_df.iterrows():
        pref_dict[(row["worker"], row["day"], row["shift"])] = row["preference"]

    max_shifts_per_worker = len(shift_slots) // len(workers) + 1

    for w in workers:
        mf.add_edge(source, w, max_shifts_per_worker)

    for w in workers:
        for (d, s, i) in shift_slots:
            if pref_dict.get((w, d, s), -1) >= 0:
                mf.add_edge(w, f"{d}_{s}_{i}", 1)

    for (d, s, i) in shift_slots:
        mf.add_edge(f"{d}_{s}_{i}", sink, 1)
    st.write(req_df)
    st.write("סכום אמיתי:", req_df["required"].sum())
    return mf.max_flow(source, sink), len(shift_slots)
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
# Excel helpers
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
# START APP
# =============================
auth_gate()
init_pg()

username = st.session_state.username

st.sidebar.title("תפריט")
page = st.sidebar.radio("ניווט", ["שיבוץ", "שיבוץ מקסימלי", "דשבורד", "מערכת מידע"], index=0)

# -----------------------------
# PAGE: שיבוץ
# -----------------------------
if page == "שיבוץ":
    st.title("🧠 שיבוץ משמרות (Excel)")

    customer_name = st.text_input("שם הלקוח", placeholder="לדוגמה: מסעדת הבוקר / לקוח A")
    uploaded = st.file_uploader("העלה קובץ Excel (xlsx) עם טאבים: workers / requirements / preferences", type=["xlsx"])
    week_number = st.number_input("מספר שבוע לשיבוץ", min_value=1, step=1, value=1)

    st.markdown("💡 השמירה למערכת המידע תדרוס נתונים קיימים אם תעלה שוב אותו שבוע לאותו לקוח.")

    if uploaded and st.button("🚀 בצע שיבוץ"):
        if not customer_name.strip():
            st.error("חייב למלא שם לקוח לפני שיבוץ.")
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

            st.success("✅ השיבוץ מוכן!")
            st.dataframe(schedule_df, use_container_width=True)

            if unassigned:
                st.warning("⚠️ משמרות שלא שובצו:")
                for d, s in sorted(list(unassigned)):
                    st.write(f"- {d} / {s}")

            # כתיבה לאקסל חדש (מוריד)
            out = BytesIO()
            base_new_name = f"שבוע {int(week_number)}"
            new_sheet_name = safe_new_sheet_name(sheet_names, base_new_name)

            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                for s in sheet_names:
                    df_s = pd.read_excel(uploaded, sheet_name=s)
                    df_s.to_excel(writer, sheet_name=s, index=False)
                schedule_df.to_excel(writer, sheet_name=new_sheet_name, index=False)

            out.seek(0)

            st.download_button(
                "⬇️ הורד קובץ אקסל עם גליון חדש",
                data=out.getvalue(),
                file_name=f"shift_schedule_week_{int(week_number)}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # שמירה ל-DB (דריסה לפי שבוע + לקוח + משתמש)
            upsert_week_schedule(username, customer_name.strip(), int(week_number), schedule_df)
            st.success(f"✅ נשמר למערכת המידע! (לקוח: {customer_name.strip()} | שבוע: {int(week_number)})")

        except Exception as e:
            st.exception(e)

# -----------------------------
# PAGE: שיבוץ מקסימלי
# -----------------------------
elif page == "שיבוץ מקסימלי":
    st.title("🔍 בדיקת כיסוי משמרות (Max Flow)")

    uploaded = st.file_uploader("העלה Excel לבדיקה", type=["xlsx"], key="maxflow")

    if uploaded:
        try:
            xls = pd.ExcelFile(uploaded)
            lower_map = {s.lower(): s for s in xls.sheet_names}

            workers_df = pd.read_excel(uploaded, sheet_name=lower_map["workers"])
            req_df     = pd.read_excel(uploaded, sheet_name=lower_map["requirements"])
            pref_df    = pd.read_excel(uploaded, sheet_name=lower_map["preferences"])

            workers_df = workers_df.rename(columns={"שם עובד": "worker"})
            req_df = req_df.rename(columns={"יום": "day", "משמרת": "shift", "כמות נדרשת": "required"})
            pref_df = pref_df.rename(columns={"עובד": "worker", "יום": "day", "משמרת": "shift", "עדיפות": "preference"})

            if st.button("בדוק כיסוי משמרות"):
                  max_assign, total = run_max_flow(workers_df, req_df, pref_df)

                  num_workers = workers_df["worker"].nunique()

                  st.write("מספר עובדים:", num_workers)
                  st.write("סה״כ דרישות:", total)

                  max_shifts_per_worker = total // num_workers + 1
                  st.write("קיבולת מקסימלית לעובד:", max_shifts_per_worker)
                  # 👇 התוצאות הרגילות שלך
                  st.info(f"מקסימום שיבוצים אפשרי: {max_assign}")
                  st.info(f"סה״כ דרישות משמרות: {total}")

                  if max_assign == total:
                        st.success("✅ ניתן לאייש את כל המשמרות")
                  else:
                        st.warning(f"⚠️ חסרים {total - max_assign} שיבוצים")


        except Exception as e:
            st.exception(e)

# -----------------------------
# PAGE: דשבורד
# -----------------------------
elif page == "דשבורד":
    st.title("דשבורד")

    customers = list_customers(username)
    if not customers:
        st.info("אין עדיין נתונים במערכת המידע. בצע שיבוץ ושמור למערכת.")
        st.stop()

    customer_pick = st.selectbox("בחר לקוח", customers, index=0)

    tab_week, tab_all = st.tabs(["שבוע ספציפי", "כל השבועות"])

    # ===== TAB 1: שבוע ספציפי =====
    with tab_week:
        weeks = list_weeks(username, customer_pick)
        if not weeks:
            st.info("אין שבועות ללקוח הזה עדיין.")
            st.stop()

        week_selected = st.selectbox("בחר שבוע להצגה", options=weeks, index=0)

        df_week = load_week_schedule(username, customer_pick, week_selected)
        if df_week.empty:
            st.warning("לא נמצאו נתונים לשבוע הזה.")
            st.stop()

        st.subheader("מערכת מידע - שיבוצים לשבוע הנבחר")
        st.dataframe(df_week, use_container_width=True)

        st.subheader("כמה עבד כל עובד בחלוקה לימים (שבוע נבחר)")
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
            title=f"לקוח: {customer_pick} | שבוע {week_selected} — כמות משמרות לכל עובד (לפי ימים)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ===== TAB 2: כל השבועות =====
    with tab_all:
        df_all = load_all_schedules(username)
        df_all = df_all[df_all["לקוח"] == customer_pick]

        if df_all.empty:
            st.info("אין נתונים ללקוח הזה עדיין.")
            st.stop()

        st.subheader("פילטר טווח שבועות (אופציונלי)")
        weeks_all = sorted(df_all["שבוע"].dropna().unique().tolist())
        min_w, max_w = int(min(weeks_all)), int(max(weeks_all))
        week_range = st.slider("טווח שבועות", min_value=min_w, max_value=max_w, value=(min_w, max_w))

        df_f = df_all[(df_all["שבוע"] >= week_range[0]) & (df_all["שבוע"] <= week_range[1])]

        st.subheader("כמה משמרות עבד כל עובד לפי יום — לאורך כל השבועות")
        agg = (
            df_f.groupby(["עובד", "יום"])
            .size()
            .reset_index(name="כמות משמרות")
        )

        fig2 = px.bar(
            agg,
            x="עובד",
            y="כמות משמרות",
            color="יום",
            barmode="stack",
            title=f"לקוח: {customer_pick} — סה״כ כמות משמרות לכל עובד לפי ימים (כל השבועות / טווח מסונן)"
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("טבלת סיכום (Pivot) — עובד מול ימים")
        pivot = agg.pivot_table(index="עובד", columns="יום", values="כמות משמרות", fill_value=0, aggfunc="sum")
        st.dataframe(pivot, use_container_width=True)

# -----------------------------
# PAGE: מערכת מידע
# -----------------------------
elif page == "מערכת מידע":
    st.title("מערכת מידע")

    df_all = load_all_schedules(username)
    if df_all.empty:
        st.info("אין נתונים במערכת המידע עדיין.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        customers = ["הכול"] + sorted(df_all["לקוח"].dropna().unique().tolist())
        customer_pick = st.selectbox("לקוח", customers, index=0)
    with col2:
        weeks_all = sorted(df_all["שבוע"].dropna().unique().tolist())
        week_pick = st.selectbox("שבוע", ["הכול"] + [str(w) for w in weeks_all], index=0)
    with col3:
        worker_pick = st.text_input("חיפוש עובד", placeholder="הקלד שם עובד...")

    df_f = df_all.copy()

    if customer_pick != "הכול":
        df_f = df_f[df_f["לקוח"] == customer_pick]

    if week_pick != "הכול":
        df_f = df_f[df_f["שבוע"] == int(week_pick)]

    if worker_pick.strip():
        df_f = df_f[df_f["עובד"].astype(str).str.contains(worker_pick.strip(), case=False, na=False)]

    st.subheader("טבלת שיבוצים מהמערכת")
    st.dataframe(df_f, use_container_width=True)

    st.download_button(
        "הורד CSV",
        data=df_f.to_csv(index=False).encode("utf-8-sig"),
        file_name="system_schedules.csv",
        mime="text/csv"
    )
