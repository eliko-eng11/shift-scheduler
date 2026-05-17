import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, timezone
import hashlib, os, hmac

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import sqlite3, hashlib, os, hmac
import psycopg2
import psycopg2.extras
import plotly.express as px

# =============================
# 1) חובה: page_config ראשון
# =============================
st.set_page_config(page_title="מערכת שיבוץ משמרות", layout="wide")

# =============================
# 2) RTL + עיצוב + פונט גדול
# RTL + UI (כתב מוגדל ומסודר)
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
      html, body, [class*="css"]  { direction: rtl; text-align: right; font-size: 18px; }
      h1, h2, h3, h4, h5, h6 { direction: rtl; text-align: right; }
      .stDataFrame, .stTable { direction: rtl; }
      label, p, div { direction: rtl; }
      section[data-testid="stSidebar"] { direction: rtl; }
      .block-container { padding-top: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
    unsafe_allow_html=True
)

# =============================
# 3) DB (Neon/Postgres) helpers
# DB (Neon / Postgres)
# =============================
@st.cache_resource
def get_engine():
def get_pg_conn():
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
    דריסה לפי: username + customer_name + week
    מוחק מה שהיה לשבוע הזה ומכניס חדש.
    """
    try:
        with eng.begin() as conn:
            conn.execute(text(ddl_users))
            conn.execute(text(ddl_schedules))
    except SQLAlchemyError as e:
        st.error("שגיאה ביצירת טבלאות DB. בדוק שה-DB פעיל ושיש הרשאות.")
        st.exception(e)
        st.stop()
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
# 4) AUTH (Postgres) - Login/Register
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
@@ -111,36 +215,31 @@ def create_user(username: str, password: str) -> bool:
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
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO users(username, password_hash, salt) VALUES (?, ?, ?)", (username, p_hash, salt))
        conn.commit()
        conn.close()
        return True
    except SQLAlchemyError:
    except sqlite3.IntegrityError:
        return False

def verify_user(username: str, password: str) -> bool:
    username = username.strip()
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(
            text("SELECT password_hash, salt FROM users WHERE username = :u"),
            {"u": username},
        ).fetchone()

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

    init_sqlite()
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ""
@@ -181,14 +280,12 @@ def auth_gate():
                if ok:
                    st.success("נרשמת בהצלחה! עכשיו תתחבר בלשונית התחברות.")
                else:
                    st.error("שם המשתמש תפוס / שגיאה ביצירה")
                    st.error("שם המשתמש תפוס או נתונים לא תקינים")

    st.stop()

auth_gate()

# =============================
# 5) אלגוריתם שיבוץ (כמו שלך)
# אלגוריתם שיבוץ (שלך)
# =============================
def simple_assignment(cost_matrix):
    used_rows, used_cols = set(), set()
@@ -382,92 +479,7 @@ def build_schedule(workers_df, req_df, pref_df, week_number):
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
# Excel helpers
# =============================
def safe_new_sheet_name(existing_names, base_name: str) -> str:
    if base_name not in existing_names:
@@ -480,33 +492,31 @@ def safe_new_sheet_name(existing_names, base_name: str) -> str:
        i += 1

# =============================
# 8) UI ראשי
# START APP
# =============================
st.title("🧠 מערכת שיבוץ משמרות")
auth_gate()
init_pg()

menu = st.sidebar.radio("ניווט", ["שיבוץ", "דשבורד", "מערכת מידע"], index=0)
username = st.session_state.username

# ----------- שיבוץ -----------
if menu == "שיבוץ":
    st.header("שיבוץ משמרות")
st.sidebar.title("תפריט")
page = st.sidebar.radio("ניווט", ["שיבוץ", "דשבורד", "מערכת מידע"], index=0)

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
# -----------------------------
# PAGE: שיבוץ
# -----------------------------
if page == "שיבוץ":
    st.title("🧠 שיבוץ משמרות (Excel)")

    st.caption("הערה: שמירה ל-DB תדרוס נתונים קיימים עבור אותו 'לקוח + שבוע'.")
    customer_name = st.text_input("שם הלקוח", placeholder="לדוגמה: מסעדת הבוקר / לקוח A")
    uploaded = st.file_uploader("העלה קובץ Excel (xlsx) עם טאבים: workers / requirements / preferences", type=["xlsx"])
    week_number = st.number_input("מספר שבוע לשיבוץ", min_value=1, step=1, value=1)

    if uploaded and st.button("🚀 בצע שיבוץ, שמור ל-DB והפק אקסל"):
        if not client_name.strip():
            st.error("חייב למלא שם לקוח לפני שמירה למערכת.")
    st.markdown("💡 השמירה למערכת המידע תדרוס נתונים קיימים אם תעלה שוב אותו שבוע לאותו לקוח.")

    if uploaded and st.button("🚀 בצע שיבוץ"):
        if not customer_name.strip():
            st.error("חייב למלא שם לקוח לפני שיבוץ.")
            st.stop()

        try:
@@ -525,15 +535,15 @@ def safe_new_sheet_name(existing_names, base_name: str) -> str:

            schedule_df, unassigned = build_schedule(workers_df, req_df, pref_df, int(week_number))

            # שמירה ל-DB (דריסה לפי לקוח+שבוע)
            save_schedule_to_db(
                schedule_df.rename(columns={"שבוע":"שבוע","יום":"יום","משמרת":"משמרת","עובד":"עובד"}),
                client_name=client_name,
                week=int(week_number),
                created_by=st.session_state.username
            )
            st.success("✅ השיבוץ מוכן!")
            st.dataframe(schedule_df, use_container_width=True)

            # הפקת אקסל חדש (שומר את כל הגליונות המקוריים + גליון חדש)
            if unassigned:
                st.warning("⚠️ משמרות שלא שובצו:")
                for d, s in sorted(list(unassigned)):
                    st.write(f"- {d} / {s}")

            # כתיבה לאקסל חדש (מוריד)
            out = BytesIO()
            base_new_name = f"שבוע {int(week_number)}"
            new_sheet_name = safe_new_sheet_name(sheet_names, base_new_name)
@@ -546,83 +556,144 @@ def safe_new_sheet_name(existing_names, base_name: str) -> str:

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
                "⬇️ הורד קובץ אקסל עם גליון חדש",
                data=out.getvalue(),
                file_name=f"shift_schedule_week_{int(week_number)}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # שמירה ל-DB (דריסה לפי שבוע + לקוח + משתמש)
            upsert_week_schedule(username, customer_name.strip(), int(week_number), schedule_df)
            st.success(f"✅ נשמר למערכת המידע! (לקוח: {customer_name.strip()} | שבוע: {int(week_number)})")

        except Exception as e:
            st.error("שגיאה בתהליך:")
            st.exception(e)

# ----------- דשבורד -----------
elif menu == "דשבורד":
    st.header("דשבורד")
# -----------------------------
# PAGE: דשבורד
# -----------------------------
elif page == "דשבורד":
    st.title("דשבורד")

    colA, colB = st.columns([1.2, 0.8])
    with colA:
        client_filter = st.text_input("סינון לפי לקוח (אופציונלי)", placeholder="השאר ריק לכל הלקוחות", key="dash_client")
    with colB:
        week_filter = st.number_input("בחר שבוע להצגה", min_value=1, step=1, value=1, key="dash_week")
    customers = list_customers(username)
    if not customers:
        st.info("אין עדיין נתונים במערכת המידע. בצע שיבוץ ושמור למערכת.")
        st.stop()

    df_week = load_schedules(client_filter if client_filter.strip() else None, int(week_filter))
    customer_pick = st.selectbox("בחר לקוח", customers, index=0)

    st.subheader("טבלת שיבוצים מהמערכת")
    st.dataframe(
        df_week.style.set_properties(**{"text-align": "center"}),
        use_container_width=True
    )
    tab_week, tab_all = st.tabs(["שבוע ספציפי", "כל השבועות"])

    # תרשים שבועי: כמה עבד כל עובד בחלוקה לימים
    if not df_week.empty:
        df_for_chart_week = df_week.rename(columns={"עובד":"עובד","יום":"יום"})
        chart_worker_by_day(df_for_chart_week, f"שבוע {int(week_filter)} — כמה עבד כל עובד בחלוקה לימים")
    # ===== TAB 1: שבוע ספציפי =====
    with tab_week:
        weeks = list_weeks(username, customer_pick)
        if not weeks:
            st.info("אין שבועות ללקוח הזה עדיין.")
            st.stop()

        week_selected = st.selectbox("בחר שבוע להצגה", options=weeks, index=0)

    st.divider()
        df_week = load_week_schedule(username, customer_pick, week_selected)
        if df_week.empty:
            st.warning("לא נמצאו נתונים לשבוע הזה.")
            st.stop()

    # תרשים לכל השבועות: פילוח ימים לאורך כל התקופות
    st.subheader("ניתוח לכל השבועות — פילוח ימים לאורך זמן")
    df_all = load_schedules(client_filter if client_filter.strip() else None, None)
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
        st.info("אין נתונים כלליים להצגה.")
    else:
        df_all_chart = df_all.rename(columns={"עובד":"עובד","יום":"יום"})
        chart_worker_by_day(df_all_chart, "כל השבועות — כמה משמרות לכל עובד לפי ימים")
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

# ----------- מערכת מידע -----------
else:
    st.header("מערכת מידע")
    df_f = df_all.copy()

    colA, colB = st.columns([1.2, 0.8])
    with colA:
        client_filter = st.text_input("סינון לפי לקוח (אופציונלי)", placeholder="השאר ריק לכל הלקוחות", key="info_client")
    with colB:
        week_opt = st.selectbox("סינון לפי שבוע", options=["הכול"] + [str(i) for i in range(1, 54)], index=0)
    if customer_pick != "הכול":
        df_f = df_f[df_f["לקוח"] == customer_pick]

    week_val = None if week_opt == "הכול" else int(week_opt)
    if week_pick != "הכול":
        df_f = df_f[df_f["שבוע"] == int(week_pick)]

    df_info = load_schedules(client_filter if client_filter.strip() else None, week_val)
    if worker_pick.strip():
        df_f = df_f[df_f["עובד"].astype(str).str.contains(worker_pick.strip(), case=False, na=False)]

    st.subheader("טבלת השיבוצים במערכת")
    st.dataframe(
        df_info.style.set_properties(**{"text-align": "center"}),
        use_container_width=True
    )
    st.subheader("טבלת שיבוצים מהמערכת")
    st.dataframe(df_f, use_container_width=True)

    if df_info.empty:
        st.info("אין רשומות לפי הסינון הנוכחי.")
    st.download_button(
        "הורד CSV",
        data=df_f.to_csv(index=False).encode("utf-8-sig"),
        file_name="system_schedules.csv",
        mime="text/csv"
    )
