import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import sqlite3, hashlib, os, hmac
import psycopg2
import psycopg2.extras
import plotly.express as px
from collections import defaultdict, deque

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="מערכת שיבוץ משמרות", layout="wide")

st.markdown("""
<style>
html, body {direction: rtl; text-align: right; font-size:18px;}
</style>
""", unsafe_allow_html=True)

# =============================
# DB
# =============================
def get_pg_conn():
    if "db" not in st.secrets:
        st.error("חסר DB")
        st.stop()
    return psycopg2.connect(st.secrets["db"]["url"])

def init_pg():
    conn = get_pg_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS schedules (
        id BIGSERIAL PRIMARY KEY,
        username TEXT,
        customer_name TEXT,
        week INT,
        day TEXT,
        shift TEXT,
        worker TEXT
    )
    """)
    conn.commit()
    conn.close()

def upsert_week_schedule(username, customer_name, week, df):
    conn = get_pg_conn()
    cur = conn.cursor()

    cur.execute("DELETE FROM schedules WHERE username=%s AND customer_name=%s AND week=%s",
                (username, customer_name, week))

    rows = [(username, customer_name, week, r["יום"], r["משמרת"], r["עובד"]) for _, r in df.iterrows()]

    psycopg2.extras.execute_values(cur,
        "INSERT INTO schedules (username, customer_name, week, day, shift, worker) VALUES %s",
        rows)

    conn.commit()
    conn.close()

# =============================
# AUTH
# =============================
DB_PATH = "users.db"

def init_sqlite():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS users(username TEXT, password TEXT)")
    conn.commit()
    conn.close()

def auth():
    init_sqlite()
    if "logged" not in st.session_state:
        st.session_state.logged = False

    if st.session_state.logged:
        return

    u = st.text_input("משתמש")
    p = st.text_input("סיסמה", type="password")

    if st.button("התחבר"):
        st.session_state.logged = True
        st.session_state.user = u
        st.rerun()

    st.stop()

auth()
init_pg()

# =============================
# GREEDY
# =============================
def simple_assignment(cost):
    used_r, used_c = set(), set()
    result = []

    for _ in range(min(len(cost), len(cost[0]))):
        best = None
        best_val = 1e9

        for i in range(len(cost)):
            if i in used_r: continue
            for j in range(len(cost[0])):
                if j in used_c: continue
                if cost[i][j] < best_val:
                    best_val = cost[i][j]
                    best = (i,j)

        if not best:
            break

        r,c = best
        result.append((r,c))
        used_r.add(r)
        used_c.add(c)

    return result

def build_schedule(workers_df, req_df, pref_df):
    workers = workers_df.iloc[:,0].tolist()

    slots = []
    for _,r in req_df.iterrows():
        for i in range(int(r[2])):
            slots.append((r[0], r[1]))

    pref = {(r[0],r[1],r[2]):r[3] for _,r in pref_df.iterrows()}

    cost = []
    for w in workers:
        row=[]
        for d,s in slots:
            p = pref.get((w,d,s), -1)
            if p==-1: row.append(1e6)
            elif p==0: row.append(100)
            else: row.append(4-p)
        cost.append(row)

    pairs = simple_assignment(cost)

    out=[]
    for r,c in pairs:
        out.append({"עובד":workers[r],"יום":slots[c][0],"משמרת":slots[c][1]})

    return pd.DataFrame(out)

# =============================
# MAX FLOW
# =============================
class MaxFlow:
    def __init__(self):
        self.g = defaultdict(dict)

    def add(self,u,v,c):
        self.g[u][v]=c
        self.g[v][u]=0

    def bfs(self,s,t,p):
        q=[s]; seen={s}
        while q:
            u=q.pop(0)
            for v in self.g[u]:
                if v not in seen and self.g[u][v]>0:
                    p[v]=u
                    seen.add(v)
                    q.append(v)
                    if v==t:return True
        return False

    def maxflow(self,s,t):
        p={}
        f=0
        while self.bfs(s,t,p):
            path=1e9
            v=t
            while v!=s:
                path=min(path,self.g[p[v]][v])
                v=p[v]
            f+=path
            v=t
            while v!=s:
                u=p[v]
                self.g[u][v]-=path
                self.g[v][u]+=path
                v=u
        return f

def run_max_flow(workers_df, req_df, pref_df):
    mf=MaxFlow()
    S="S";T="T"

    workers = workers_df.iloc[:,0].tolist()

    slots=[]
    for _,r in req_df.iterrows():
        for i in range(int(r[2])):
            slots.append((r[0],r[1],i))

    pref={(r[0],r[1],r[2]):r[3] for _,r in pref_df.iterrows()}

    for w in workers:
        mf.add(S,w,3)

    for w in workers:
        for d,s,i in slots:
            if pref.get((w,d,s),-1)>=0:
                mf.add(w,f"{d}_{s}_{i}",1)

    for d,s,i in slots:
        mf.add(f"{d}_{s}_{i}",T,1)

    return mf.maxflow(S,T), len(slots)

# =============================
# UI
# =============================
page = st.sidebar.radio("ניווט", ["שיבוץ","שיבוץ מקסימלי"])

# -----------------------------
# שיבוץ רגיל
# -----------------------------
if page=="שיבוץ":
    file=st.file_uploader("Excel")

    if file:
        w=pd.read_excel(file,"workers")
        r=pd.read_excel(file,"requirements")
        p=pd.read_excel(file,"preferences")

        if st.button("שבץ"):
            df=build_schedule(w,r,p)
            st.dataframe(df)

# -----------------------------
# MAX FLOW PAGE
# -----------------------------
elif page=="שיבוץ מקסימלי":
    file=st.file_uploader("Excel לבדיקה")

    if file:
        w=pd.read_excel(file,"workers")
        r=pd.read_excel(file,"requirements")
        p=pd.read_excel(file,"preferences")

        if st.button("חשב"):
            m,total=run_max_flow(w,r,p)
            st.metric("מקסימום שיבוצים",m)
            st.metric("דרישות",total)

            if m==total:
                st.success("אפשר לאייש הכול")
            else:
                st.warning(f"חסרים {total-m}")
