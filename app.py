import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, timezone
import hashlib, os, hmac

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# =============================
# 1) חובה: page_config ראשון
# =============================
st.set_page_config(page_title="מערכת שיבוץ משמרות", layout="wide")

# =============================
# 2) RTL + עיצוב + פונט גדול
# =============================
st.markdown(
    """
    <style>
      html, body, [class*="css"]  { direction: rtl; text-align: right; }
      h1, h2, h3, h4, h5, h6, p, div, span, label { direction: rtl; text-align: right; }

      /* פונט גדול יותר לכל האפליקציה */
      html, body, [class*="css"]  { font-size: 18px !important; }
      .stDataFrame { direction: rtl; }

      /* כפתורים/קלטים */
      .stButton button { font-size: 18px !important; padding: 0.6rem 1rem; }
      input, textarea { font-size: 18px !important; }

      /* כותרת צד */
      section[data-testid="stSidebar"] * { font-size: 17px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# 3) DB (Neon/Postgres) helpers
# =============================
@st.cache_resource
def get_engine():
    if "db" not in st.secrets or "url" not in st.secrets["db"]:
        st.error("חסר Secrets: db.url (Streamlit Settings → Secrets)")
        st.stop()

    url = st.secrets["db"]["url"].strip()

    # הגנה נפוצה: משתמשים מדביקים snippet של Neon שמתחיל ב-psql '...'
    if url.lower().startswith("psql"):
        st.error("נראה שהדבקת ל-Secrets snippet שמתחיל ב- 'psql'. ב-Secrets צריך להיות רק ה-URL שמתחיל ב-postgresql://")
        st.stop()

    # הגנה: אם הדביקו URL עם גרשיים
    url = url.strip("'").strip('"')

    try:
        eng = create_engine(url, pool_pre_ping=True)
        return eng
    except Exception as e:
        st.error("שגיאה בבניית חיבור DB. בדוק את db.url ב-Secrets.")
        st.exception(e)
        st.stop()

def init_db():
    eng = get_engine()
    ddl_users = """
    CREATE TABLE IF NOT EXISTS users (
      username TEXT PRIMARY KEY,
      password_hash TEXT NOT NULL,
      salt TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """
    ddl_schedules = """
    CREATE TABLE IF NOT EXISTS schedules (
      id BIGSERIAL PRIMARY KEY,
      client_name TEXT NOT NULL,
      week INTEGER NOT NULL,
      day TEXT NOT NULL,
      shift TEXT NOT NULL,
      worker TEXT NOT NULL,
      created_by TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_schedules_week ON schedules(week);
    CREATE INDEX IF NOT EXISTS idx_schedules_worker ON schedules(worker);
    CREATE INDEX IF NOT EXISTS idx_schedules_client ON schedules(client_name);
    """
    try:
        with eng.begin() as conn:
            conn.execute(text(ddl_users))
            conn.execute(text(ddl_schedules))
    except SQLAlchemyError as e:
        st.error("שגיאה ביצירת טבלאות DB. בדוק שה-DB פעיל ושיש הרשאות.")
        st.exception(e)
        st.stop()

# =============================
# 4) AUTH (Postgres) - Login/Register
# =============================
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
    except SQLAlchemyError:
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
                    st.error("שם המשתמש תפוס / שגיאה ביצירה")

    st.stop()

auth_gate()

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
# 6) DB פעולות: שמירה/שליפה
# =============================
def save_schedule_to_db(df: pd.DataFrame, client_name: str, week: int, created_by: str):
    """שומר שיבוץ. דורש: אם כבר יש שבוע+לקוח -> מוחק ודורס."""
    eng = get_engine()
    client_name = client_name.strip()

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "client_name": client_name,
            "week": int(week),
            "day": str(r["יום"]),
            "shift": str(r["משמרת"]),
            "worker": str(r["עובד"]),
            "created_by": created_by
        })

    with eng.begin() as conn:
        conn.execute(
            text("DELETE FROM schedules WHERE client_name = :c AND week = :w"),
            {"c": client_name, "w": int(week)}
        )
        if rows:
            conn.execute(
                text("""
                    INSERT INTO schedules (client_name, week, day, shift, worker, created_by)
                    VALUES (:client_name, :week, :day, :shift, :worker, :created_by)
                """),
                rows
            )

def load_schedules(client_name: str | None, week: int | None):
    eng = get_engine()
    q = "SELECT client_name, week, day, shift, worker, created_by, created_at FROM schedules WHERE 1=1"
    params = {}

    if client_name and client_name.strip():
        q += " AND client_name = :c"
        params["c"] = client_name.strip()
    if week is not None:
        q += " AND week = :w"
        params["w"] = int(week)

    q += " ORDER BY week DESC, created_at DESC, day, shift, worker"

    with eng.begin() as conn:
        rows = conn.execute(text(q), params).fetchall()

    if not rows:
        return pd.DataFrame(columns=["לקוח", "שבוע", "עובד", "יום", "משמרת", "נוצר בתאריך", "נוצר על ידי"])

    df = pd.DataFrame(rows, columns=["client_name", "week", "day", "shift", "worker", "created_by", "created_at"])
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.rename(columns={
        "client_name": "לקוח",
        "week": "שבוע",
        "worker": "עובד",
        "day": "יום",
        "shift": "משמרת",
        "created_at": "נוצר בתאריך",
        "created_by": "נוצר על ידי",
    })
    return df[["שבוע", "עובד", "יום", "משמרת", "נוצר בתאריך", "נוצר על ידי", "לקוח"]]

def chart_worker_by_day(df_view: pd.DataFrame, title: str):
    """
    df_view כולל עמודות: עובד, יום (ועוד)
    מציג stacked bar: X=עובד, צבע=יום, ערך=כמות משמרות
    """
    if df_view.empty:
        st.info("אין נתונים להצגה.")
        return

    ctab = (
        df_view.groupby(["עובד", "יום"])
        .size()
        .reset_index(name="כמות משמרות")
    )
    pivot = ctab.pivot(index="עובד", columns="יום", values="כמות משמרות").fillna(0).astype(int)

    st.subheader(title)
    st.bar_chart(pivot)

# =============================
# 7) Excel helpers (ייצוא)
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
# 8) UI ראשי
# =============================
st.title("🧠 מערכת שיבוץ משמרות")

menu = st.sidebar.radio("ניווט", ["שיבוץ", "דשבורד", "מערכת מידע"], index=0)

# ----------- שיבוץ -----------
if menu == "שיבוץ":
    st.header("שיבוץ משמרות")

    col1, col2, col3 = st.columns([1.2, 0.8, 1.2])
    with col1:
        client_name = st.text_input("שם לקוח", placeholder="לדוגמה: מסעדת דניאל", key="client_name")
    with col2:
        week_number = st.number_input("מספר שבוע", min_value=1, step=1, value=1, key="week_number")
    with col3:
        uploaded = st.file_uploader(
            "העלה Excel (xlsx) עם טאבים: workers / requirements / preferences",
            type=["xlsx"],
            key="excel_upload",
        )

    st.caption("הערה: שמירה ל-DB תדרוס נתונים קיימים עבור אותו 'לקוח + שבוע'.")

    if uploaded and st.button("🚀 בצע שיבוץ, שמור ל-DB והפק אקסל"):
        if not client_name.strip():
            st.error("חייב למלא שם לקוח לפני שמירה למערכת.")
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

            # שמירה ל-DB (דריסה לפי לקוח+שבוע)
            save_schedule_to_db(
                schedule_df.rename(columns={"שבוע":"שבוע","יום":"יום","משמרת":"משמרת","עובד":"עובד"}),
                client_name=client_name,
                week=int(week_number),
                created_by=st.session_state.username
            )

            # הפקת אקסל חדש (שומר את כל הגליונות המקוריים + גליון חדש)
            out = BytesIO()
            base_new_name = f"שבוע {int(week_number)}"
            new_sheet_name = safe_new_sheet_name(sheet_names, base_new_name)

            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                for s in sheet_names:
                    df_s = pd.read_excel(uploaded, sheet_name=s)
                    df_s.to_excel(writer, sheet_name=s, index=False)
                schedule_df.to_excel(writer, sheet_name=new_sheet_name, index=False)

            out.seek(0)

            st.success(f"✅ השיבוץ נשמר ל-DB עבור לקוח '{client_name}' שבוע {int(week_number)} (דריסה אם היה קיים).")
            st.subheader("תוצאת שיבוץ")
            st.dataframe(
                schedule_df.style.set_properties(**{"text-align": "center"}),
                use_container_width=True
            )

            if unassigned:
                st.warning("⚠️ משמרות שלא שובצו:")
                for d, s in sorted(list(unassigned)):
                    st.write(f"- {d} / {s}")

            st.download_button(
                "⬇️ הורד קובץ אקסל עם גליון נוסף",
                data=out.getvalue(),
                file_name=f"shift_schedule_week_{int(week_number)}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        except Exception as e:
            st.error("שגיאה בתהליך:")
            st.exception(e)

# ----------- דשבורד -----------
elif menu == "דשבורד":
    st.header("דשבורד")

    colA, colB = st.columns([1.2, 0.8])
    with colA:
        client_filter = st.text_input("סינון לפי לקוח (אופציונלי)", placeholder="השאר ריק לכל הלקוחות", key="dash_client")
    with colB:
        week_filter = st.number_input("בחר שבוע להצגה", min_value=1, step=1, value=1, key="dash_week")

    df_week = load_schedules(client_filter if client_filter.strip() else None, int(week_filter))

    st.subheader("טבלת שיבוצים מהמערכת")
    st.dataframe(
        df_week.style.set_properties(**{"text-align": "center"}),
        use_container_width=True
    )

    # תרשים שבועי: כמה עבד כל עובד בחלוקה לימים
    if not df_week.empty:
        df_for_chart_week = df_week.rename(columns={"עובד":"עובד","יום":"יום"})
        chart_worker_by_day(df_for_chart_week, f"שבוע {int(week_filter)} — כמה עבד כל עובד בחלוקה לימים")

    st.divider()

    # תרשים לכל השבועות: פילוח ימים לאורך כל התקופות
    st.subheader("ניתוח לכל השבועות — פילוח ימים לאורך זמן")
    df_all = load_schedules(client_filter if client_filter.strip() else None, None)

    if df_all.empty:
        st.info("אין נתונים כלליים להצגה.")
    else:
        df_all_chart = df_all.rename(columns={"עובד":"עובד","יום":"יום"})
        chart_worker_by_day(df_all_chart, "כל השבועות — כמה משמרות לכל עובד לפי ימים")

# ----------- מערכת מידע -----------
else:
    st.header("מערכת מידע")

    colA, colB = st.columns([1.2, 0.8])
    with colA:
        client_filter = st.text_input("סינון לפי לקוח (אופציונלי)", placeholder="השאר ריק לכל הלקוחות", key="info_client")
    with colB:
        week_opt = st.selectbox("סינון לפי שבוע", options=["הכול"] + [str(i) for i in range(1, 54)], index=0)

    week_val = None if week_opt == "הכול" else int(week_opt)

    df_info = load_schedules(client_filter if client_filter.strip() else None, week_val)

    st.subheader("טבלת השיבוצים במערכת")
    st.dataframe(
        df_info.style.set_properties(**{"text-align": "center"}),
        use_container_width=True
    )

    if df_info.empty:
        st.info("אין רשומות לפי הסינון הנוכחי.")
