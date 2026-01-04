# app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import hashlib, os, hmac
from datetime import datetime, timezone

from sqlalchemy import create_engine, text

# =============================
# 1) חובה: page_config ראשון
# =============================
st.set_page_config(page_title="מערכת שיבוץ משמרות", layout="wide")

# =============================
# 2) RTL + הגדלת כתב + טבלאות מיושרות
# =============================
st.markdown(
    """
    <style>
      html, body, [class*="css"]  { direction: rtl; text-align: right; }
      .block-container { padding-top: 1.2rem; }
      h1, h2, h3, h4, h5, h6, p, div, span, label { direction: rtl; }
      /* הגדלת פונט כללית */
      html { font-size: 18px; }
      /* dataframe */
      .stDataFrame { direction: rtl; }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# 3) DB (Neon/Postgres) + AUTH (Users בטבלה)
# =============================
# שים ב-Secrets:
# [db]
# url = "postgresql://USER:PASSWORD@HOST/DB?sslmode=require"
#
# או לחלופין:
# database_url = "..."
#
def get_db_url() -> str:
    if "db" in st.secrets and "url" in st.secrets["db"]:
        return st.secrets["db"]["url"]
    if "database_url" in st.secrets:
        return st.secrets["database_url"]
    raise RuntimeError("חסר Secrets: db.url (Streamlit Settings -> Secrets)")

@st.cache_resource
def get_engine():
    url = get_db_url()
    return create_engine(url, pool_pre_ping=True)

def init_db():
    eng = get_engine()
    with eng.begin() as conn:
        # users
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """))
        # schedules
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS schedules (
                id BIGSERIAL PRIMARY KEY,
                customer TEXT NOT NULL,
                week INT NOT NULL,
                day TEXT NOT NULL,
                shift TEXT NOT NULL,
                worker TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """))
        # index for faster filtering
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_schedules_customer_week
            ON schedules(customer, week);
        """))

def hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return dk.hex()

def create_user(username: str, password: str) -> bool:
    username = username.strip()
    if not username or not password:
        return False
    salt = os.urandom(16).hex()
    p_hash = hash_password(password, salt)

    eng = get_engine()
    try:
        with eng.begin() as conn:
            conn.execute(
                text("INSERT INTO users(username, password_hash, salt) VALUES (:u, :ph, :s)"),
                {"u": username, "ph": p_hash, "s": salt},
            )
        return True
    except Exception:
        return False

def verify_user(username: str, password: str) -> bool:
    username = username.strip()
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(
            text("SELECT password_hash, salt FROM users WHERE username = :u"),
            {"u": username},
        ).fetchone()

    if not row:
        return False
    stored_hash, salt = row[0], row[1]
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
                    st.error("שם המשתמש תפוס או נתונים לא תקינים / בעיית DB")

    st.stop()

auth_gate()

# =============================
# 4) אלגוריתם שיבוץ
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

        # לא משבצים משמרות צמודות באותו יום (לפי הרשימה)
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

    df["יום_מספר"] = df["יום"].apply(lambda x: ordered_days.index(x) if x in ordered_days else 999)
    df = df.sort_values(by=["שבוע", "יום_מספר", "משמרת", "עובד"])
    df = df[["שבוע", "יום", "משמרת", "עובד"]]
    return df, unassigned_pairs

# =============================
# 5) Excel helpers
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
# 6) UI helpers (יישור מרכז לחותמת זמן בכל הטבלאות)
# =============================
def center_timestamp(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    # מרכז את "נוצר בתאריך" אם קיימת
    sty = df.style
    if "נוצר בתאריך" in df.columns:
        sty = sty.set_properties(subset=["נוצר בתאריך"], **{"text-align": "center"})
    # כותרות מרכז
    sty = sty.set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
    return sty

# =============================
# 7) פעולות DB לשיבוצים
# =============================
def upsert_week_customer(customer: str, week: int, schedule_df: pd.DataFrame):
    """דריסה לפי (customer, week): מוחק ואז מכניס מחדש"""
    eng = get_engine()
    created_at = datetime.now(timezone.utc)

    with eng.begin() as conn:
        conn.execute(
            text("DELETE FROM schedules WHERE customer = :c AND week = :w"),
            {"c": customer, "w": int(week)},
        )

        # insert bulk
        rows = []
        for _, r in schedule_df.iterrows():
            rows.append({
                "customer": customer,
                "week": int(r["שבוע"]),
                "day": str(r["יום"]),
                "shift": str(r["משמרת"]),
                "worker": str(r["עובד"]),
                "created_at": created_at,
            })

        conn.execute(
            text("""
                INSERT INTO schedules(customer, week, day, shift, worker, created_at)
                VALUES (:customer, :week, :day, :shift, :worker, :created_at)
            """),
            rows
        )

def load_schedules(customer: str | None = None, week: int | None = None) -> pd.DataFrame:
    eng = get_engine()
    q = "SELECT customer AS לקוח, week AS שבוע, day AS יום, shift AS משמרת, worker AS עובד, created_at AS \"נוצר בתאריך\" FROM schedules"
    conds = []
    params = {}
    if customer:
        conds.append("customer = :c")
        params["c"] = customer
    if week is not None:
        conds.append("week = :w")
        params["w"] = int(week)
    if conds:
        q += " WHERE " + " AND ".join(conds)
    q += " ORDER BY week DESC, day ASC, shift ASC, worker ASC"

    with eng.begin() as conn:
        df = pd.read_sql(text(q), conn, params=params)
    return df

def list_customers() -> list[str]:
    eng = get_engine()
    with eng.begin() as conn:
        rows = conn.execute(text("SELECT DISTINCT customer FROM schedules ORDER BY customer")).fetchall()
    return [r[0] for r in rows]

def list_weeks(customer: str | None = None) -> list[int]:
    eng = get_engine()
    if customer:
        with eng.begin() as conn:
            rows = conn.execute(
                text("SELECT DISTINCT week FROM schedules WHERE customer = :c ORDER BY week DESC"),
                {"c": customer},
            ).fetchall()
    else:
        with eng.begin() as conn:
            rows = conn.execute(text("SELECT DISTINCT week FROM schedules ORDER BY week DESC")).fetchall()
    return [int(r[0]) for r in rows]

# =============================
# 8) ניווט
# =============================
st.sidebar.title("תפריט")
page = st.sidebar.radio("ניווט", ["שיבוץ", "מערכת מידע", "דשבורד"], index=0)

# =============================
# 9) שיבוץ (Excel -> Excel + שמירה ל-DB)
# =============================
if page == "שיבוץ":
    st.title("🧠 שיבוץ משמרות (Excel)")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        customer = st.text_input("שם הלקוח (יישמר ברשומות)", placeholder="לדוגמה: מסעדת דניאלה")
    with c2:
        week_number = st.number_input("מספר שבוע לשיבוץ", min_value=1, step=1, value=1)
    with c3:
        save_to_db = st.toggle("שמור למערכת מידע (DB)", value=True)

    uploaded = st.file_uploader("העלה קובץ Excel (xlsx) עם טאבים: workers / requirements / preferences", type=["xlsx"])

    if uploaded and st.button("🚀 בצע שיבוץ"):
        if not customer.strip():
            st.error("חובה למלא שם לקוח לפני שמבצעים שיבוץ.")
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

            # --- כתיבה לקובץ חדש (כל הגליונות + גליון שבוע)
            out = BytesIO()
            base_new_name = f"שבוע {int(week_number)}"
            new_sheet_name = safe_new_sheet_name(sheet_names, base_new_name)

            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                for s in sheet_names:
                    df_s = pd.read_excel(uploaded, sheet_name=s)
                    df_s.to_excel(writer, sheet_name=s, index=False)
                schedule_df.to_excel(writer, sheet_name=new_sheet_name, index=False)

            out.seek(0)

            # --- שמירה ל-DB (דריסה לפי שבוע+לקוח)
            if save_to_db:
                upsert_week_customer(customer.strip(), int(week_number), schedule_df)

            st.success(f"✅ מוכן! נוסף גליון חדש: {new_sheet_name}")
            st.dataframe(schedule_df, use_container_width=True)

            if unassigned:
                st.warning("⚠️ משמרות שלא שובצו:")
                for d, s in sorted(list(unassigned)):
                    st.write(f"- {d} / {s}")

            st.download_button(
                "⬇️ הורד קובץ אקסל חדש",
                data=out.getvalue(),
                file_name=f"{customer.strip()}_week_{int(week_number)}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        except Exception as e:
            st.error("שגיאה בתהליך:")
            st.exception(e)

# =============================
# 10) מערכת מידע (טבלה מלאה + פילטרים)
# =============================
elif page == "מערכת מידע":
    st.title("📚 מערכת מידע (רשומות שיבוץ)")

    customers = list_customers()
    weeks_all = list_weeks()

    c1, c2 = st.columns([2, 1])
    with c1:
        customer_filter = st.selectbox("בחר לקוח (אופציונלי)", ["הכול"] + customers, index=0)
    with c2:
        week_filter = st.selectbox("בחר שבוע (אופציונלי)", ["הכול"] + [str(w) for w in weeks_all], index=0)

    customer_val = None if customer_filter == "הכול" else customer_filter
    week_val = None if week_filter == "הכול" else int(week_filter)

    df = load_schedules(customer=customer_val, week=week_val)

    if df.empty:
        st.info("אין נתונים להצגה לפי הפילטרים שבחרת.")
    else:
        st.subheader("טבלת שיבוצים מהמערכת")
        st.dataframe(center_timestamp(df), use_container_width=True)

# =============================
# 11) דשבורד (שבוע נבחר + כל השבועות)
# =============================
elif page == "דשבורד":
    st.title("דשבורד")

    customers = list_customers()
    if not customers:
        st.info("אין נתונים במערכת עדיין. תעלה שיבוץ ותשמור ל-DB.")
        st.stop()

    c1, c2 = st.columns([2, 1])
    with c1:
        customer = st.selectbox("בחר לקוח", customers, index=0)
    with c2:
        weeks = list_weeks(customer)
        if not weeks:
            st.info("אין שבועות ללקוח הזה עדיין.")
            st.stop()
        week = st.selectbox("בחר שבוע להצגה", weeks, index=0)

    # --- נתוני שבוע נבחר
    df_week = load_schedules(customer=customer, week=int(week))
    st.subheader("טבלת שיבוצים מהמערכת")
    st.dataframe(center_timestamp(df_week), use_container_width=True)

    # תרשים: כמה עבד כל עובד בחלוקה לימים (שבוע נבחר)
    st.subheader("כמה עבד כל עובד בחלוקה לימים (שבוע נבחר)")
    pivot_week = (
        df_week
        .groupby(["עובד", "יום"])
        .size()
        .reset_index(name="כמות משמרות")
        .pivot(index="עובד", columns="יום", values="כמות משמרות")
        .fillna(0)
    )
    st.bar_chart(pivot_week)

    st.divider()

    # --- כל השבועות (ללקוח): פילוח עובד X יום לאורך זמן
    st.subheader("פילוח לאורך כלל השבועות: כמה משמרות עובד עושה בכל יום")
    df_all = load_schedules(customer=customer, week=None)

    if df_all.empty:
        st.info("אין נתונים להצגה.")
        st.stop()

    pivot_all = (
        df_all
        .groupby(["עובד", "יום"])
        .size()
        .reset_index(name="כמות משמרות")
        .pivot(index="עובד", columns="יום", values="כמות משמרות")
        .fillna(0)
        .sort_index()
    )
    st.bar_chart(pivot_all)

    st.subheader("טבלת סיכום (כל השבועות)")
    pivot_all_tbl = pivot_all.reset_index()
    st.dataframe(center_timestamp(pivot_all_tbl), use_container_width=True)
