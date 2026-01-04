import streamlit as st
import psycopg2

st.set_page_config(page_title="DB Test")

conn = psycopg2.connect(st.secrets["db"]["url"])
cur = conn.cursor()
cur.execute("SELECT 1;")
cur.fetchone()
cur.close()
conn.close()

st.success("🔥 החיבור ל-Neon עובד!")
