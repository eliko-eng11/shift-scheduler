import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import os
import hmac

# =============================
# page config
# =============================
st.set_page_config(page_title="מערכת שיבוץ משמרות", layout="wide")

# =============================
# RTL
# =============================
st.markdown("""
<style>
html, body, [class*="css"] {
    direction: rtl;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

# =============================
# AUTH (SQLite)
# =============================
DB_PATH = "users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password_hash TEXT,
        salt TEXT
    )
    """)
    conn.commit()
    conn.close()

def hash_password(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000).hex()

def create_user(username, password):
    salt = os.urandom(16).hex()
    p_hash = hash_password(password, salt)

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO users VALUES (?, ?, ?)", (username, p_hash, salt))
        conn.commit()
        conn.close()
        return True
    except:
        return False

def verify_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT password_hash, salt FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return False

    stored_hash, salt = row
    return hmac.compare_digest(stored_hash, hash_password(password, salt))

def auth():
    init_db()

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        return

    st.title("🔐 התחברות")

    user = st.text_input("שם משתמש")
    pwd = st.text_input("סיסמה", type="password")

    if st.button("התחבר"):
        if verify_user(user, pwd):
            st.session_state.logged_in = True
            st.session_state.username = user
            st.rerun()
        else:
            st.error("שגיאה")

    st.subheader("הרשמה")
    new_user = st.text_input("שם משתמש חדש")
    new_pwd = st.text_input("סיסמה חדשה", type="password")

    if st.button("הרשם"):
        if create_user(new_user, new_pwd):
            st.success("נרשמת!")
        else:
            st.error("משתמש קיים")

    st.stop()

auth()

# =============================
# ALGORITHM (Greedy)
# =============================
def simple_assignment(cost_matrix):
    used_rows = set()
    used_cols = set()
    assignments = []

    rows = len(cost_matrix)
    cols = len(cost_matrix[0])

    for _ in range(min(rows, cols)):
        best = None
        best_cost = 1e9

        for i in range(rows):
            if i in used_rows:
                continue
            for j in range(cols):
                if j in used_cols:
                    continue

                if cost_matrix[i][j] < best_cost:
                    best_cost = cost_matrix[i][j]
                    best = (i, j)

        if best is None:
            break

        r, c = best
        assignments.append((r, c))
        used_rows.add(r)
        used_cols.add(c)

    return assignments

# =============================
# BUILD SCHEDULE
# =============================
def build_schedule(workers_df, req_df, pref_df):
    workers = workers_df["worker"].tolist()

    shift_slots = []
    for _, row in req_df.iterrows():
        for i in range(int(row["required"])):
            shift_slots.append((row["day"], row["shift"], i))

    pref_dict = {}
    for _, row in pref_df.iterrows():
        pref_dict[(row["worker"], row["day"], row["shift"])] = row["preference"]

    cost_matrix = []

    for w in workers:
        row_costs = []
        for (d, s, _) in shift_slots:
            pref = pref_dict.get((w, d, s), -1)

            if pref == -1:
                row_costs.append(1e6)
            elif pref == 0:
                row_costs.append(100)
            else:
                row_costs.append(4 - pref)

        cost_matrix.append(row_costs)

    assignments = simple_assignment(cost_matrix)

    result = []
    for r, c in assignments:
        w = workers[r]
        d, s, _ = shift_slots[c]
        result.append({"עובד": w, "יום": d, "משמרת": s})

    return pd.DataFrame(result)

# =============================
# UI
# =============================
st.title("📊 שיבוץ עובדים")

uploaded = st.file_uploader("העלה Excel", type=["xlsx"])

if uploaded:
    workers_df = pd.read_excel(uploaded, sheet_name="workers")
    req_df = pd.read_excel(uploaded, sheet_name="requirements")
    pref_df = pd.read_excel(uploaded, sheet_name="preferences")

    if st.button("🚀 בצע שיבוץ"):
        result = build_schedule(workers_df, req_df, pref_df)
        st.success("שיבוץ מוכן")
        st.dataframe(result)
