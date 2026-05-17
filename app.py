import pandas as pd
import streamlit as st
from collections import defaultdict, deque

# =====================================
# עזר: המרת עדיפות לעלות
# =====================================
def pref_to_cost(p):
    if p == -1:
        return None  # לא אפשרי (אין קשת)
    if p == 0:
        return 100   # קנס גבוה
    return 4 - p     # 3->1, 2->2, 1->3

# =====================================
# בניית משמרות (slots)
# =====================================
def build_shift_slots(req_df):
    slots = []
    for _, row in req_df.iterrows():
        d, s, k = row["day"], row["shift"], int(row["required"])
        for i in range(k):
            slots.append((d, s, i))
    return slots

# =====================================
# GREEDY – שיבוץ לפי עלות מינימלית
# =====================================
def greedy_schedule(workers_df, req_df, pref_df):
    workers = workers_df["worker"].tolist()
    shift_slots = build_shift_slots(req_df)

    # קיבולת עובד (כמו בקוד שלך)
    max_shifts_per_worker = len(shift_slots) // max(1, len(workers)) + 1

    # pref dict
    pref = {}
    for _, r in pref_df.iterrows():
        pref[(r["worker"], r["day"], r["shift"])] = r["preference"]

    # בונים רשימת "קשתות" עם עלויות
    edges = []
    for w in workers:
        for (d, s, i) in shift_slots:
            p = pref.get((w, d, s), -1)
            c = pref_to_cost(p)
            if c is not None:
                edges.append((c, w, d, s, i))

    # מיון לפי עלות (חמדני)
    edges.sort(key=lambda x: x[0])

    used_worker = defaultdict(int)
    used_slot = set()
    assignments = []

    for cost, w, d, s, i in edges:
        if used_worker[w] >= max_shifts_per_worker:
            continue
        if (d, s, i) in used_slot:
            continue
        assignments.append((w, d, s))
        used_worker[w] += 1
        used_slot.add((d, s, i))

    return assignments, max_shifts_per_worker

# =====================================
# MAX FLOW – Ford-Fulkerson (BFS/Edmonds-Karp)
# =====================================
class MaxFlow:
    def __init__(self):
        self.g = defaultdict(dict)

    def add_edge(self, u, v, cap):
        if v not in self.g[u]:
            self.g[u][v] = 0
        if u not in self.g[v]:
            self.g[v][u] = 0
        self.g[u][v] += cap  # allow multi-edges
        # reverse edge already 0 or existing

    def bfs(self, s, t, parent):
        parent.clear()
        parent[s] = None
        q = deque([s])
        while q:
            u = q.popleft()
            for v, cap in self.g[u].items():
                if v not in parent and cap > 0:
                    parent[v] = u
                    if v == t:
                        return True
                    q.append(v)
        return False

    def max_flow(self, s, t):
        parent = {}
        flow = 0
        while self.bfs(s, t, parent):
            # find bottleneck
            v = t
            f = float('inf')
            while parent[v] is not None:
                u = parent[v]
                f = min(f, self.g[u][v])
                v = u
            # augment
            v = t
            while parent[v] is not None:
                u = parent[v]
                self.g[u][v] -= f
                self.g[v][u] += f
                v = u
            flow += f
        return flow

def run_max_flow(workers_df, req_df, pref_df):
    mf = MaxFlow()
    S, T = "S", "T"

    workers = workers_df["worker"].tolist()
    slots = build_shift_slots(req_df)

    max_shifts_per_worker = len(slots) // max(1, len(workers)) + 1

    # pref dict
    pref = {}
    for _, r in pref_df.iterrows():
        pref[(r["worker"], r["day"], r["shift"])] = r["preference"]

    # S -> workers
    for w in workers:
        mf.add_edge(S, f"W::{w}", max_shifts_per_worker)

    # workers -> slots
    for w in workers:
        for (d, s, i) in slots:
            p = pref.get((w, d, s), -1)
            if p >= 0:  # זמין
                mf.add_edge(f"W::{w}", f"S::{d}_{s}_{i}", 1)

    # slots -> T
    for (d, s, i) in slots:
        mf.add_edge(f"S::{d}_{s}_{i}", T, 1)

    max_assign = mf.max_flow(S, T)
    return max_assign

# =====================================
# UI – Streamlit
# =====================================
st.set_page_config(page_title="Shift Scheduling", layout="wide")
st.title("📊 מערכת שיבוץ עובדים למשמרות")

uploaded = st.file_uploader("העלה קובץ Excel עם גיליונות: workers, requirements, preferences", type=["xlsx"])

def normalize_cols(workers_df, req_df, pref_df):
    workers_df = workers_df.rename(columns={"שם עובד": "worker", "עובד": "worker"})
    req_df = req_df.rename(columns={"יום": "day", "משמרת": "shift", "כמות נדרשת": "required"})
    pref_df = pref_df.rename(columns={"עובד": "worker", "יום": "day", "משמרת": "shift", "עדיפות": "preference"})
    return workers_df, req_df, pref_df

if uploaded:
    try:
        workers_df = pd.read_excel(uploaded, sheet_name="workers")
        req_df     = pd.read_excel(uploaded, sheet_name="requirements")
        pref_df    = pd.read_excel(uploaded, sheet_name="preferences")

        workers_df, req_df, pref_df = normalize_cols(workers_df, req_df, pref_df)

        st.subheader("תצוגת נתונים")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Workers")
            st.dataframe(workers_df)
        with col2:
            st.write("Requirements")
            st.dataframe(req_df)
        with col3:
            st.write("Preferences")
            st.dataframe(pref_df)

        # כפתור GREEDY
        if st.button("🚀 הפעלת שיבוץ (Greedy)"):
            assigns, cap = greedy_schedule(workers_df, req_df, pref_df)
            st.success(f"מספר שיבוצים: {len(assigns)} | קיבולת לעובד: {cap}")

            if assigns:
                df_out = pd.DataFrame(assigns, columns=["worker", "day", "shift"])
                st.dataframe(df_out)
            else:
                st.warning("לא נמצא שיבוץ")

        # כפתור MAX FLOW
        if st.button("🔍 בדיקת כיסוי מקסימלי (Max Flow)"):
            max_assign = run_max_flow(workers_df, req_df, pref_df)
            total_required = int(req_df["required"].sum())
            st.info(f"מקסימום שיבוצים אפשרי: {max_assign} מתוך דרישה כוללת: {total_required}")
            if max_assign < total_required:
                st.warning("לא ניתן לכסות את כל המשמרות עם הנתונים הקיימים")
            else:
                st.success("ניתן לכסות את כל המשמרות")

    except Exception as e:
        st.error("שגיאה בעיבוד הקובץ")
        st.exception(e)
else:
    st.write("העלה קובץ כדי להתחיל")
