import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="🗕️ שיבוץ עובדים", layout="wide")
st.markdown("<h1 style='text-align:center; color:#2C3E50;'>🛠️ מערכת שיבוץ חכמה לעובדים</h1>", unsafe_allow_html=True)

# ------------------------------------------------
# פונקציית הקצאה חמדנית במקום scipy
# ------------------------------------------------
def simple_assignment(cost_matrix):
    used_rows = set()
    used_cols = set()
    assignments = []

    rows = len(cost_matrix)
    cols = len(cost_matrix[0]) if rows > 0 else 0

    for _ in range(min(rows, cols)):
        best = None
        best_cost = 10**12
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

# ------------------------------------------------
# שמירת שיבוץ לקובץ היסטוריה באקסל
# ------------------------------------------------
def save_schedule_to_excel(df, week_number, month_str, file_path="shifts_history.xlsx"):
    sheet_name = f"שבוע_{week_number}_{month_str}".replace(" ", "_")
    file = Path(file_path)

    if file.exists():
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

# ------------------------------------------------
# קלט מהמשתמש: שבוע + חודש + קובץ אקסל
# ------------------------------------------------
week_number = st.number_input("מספר שבוע (לשמירת השיבוץ בהיסטוריה)", min_value=1, step=1)
month_str = st.text_input("חודש (למשל: נובמבר_2025)", value="")

uploaded_file = st.file_uploader("📂 העלה קובץ אקסל של שיבוצים (shifts_template.xlsx)", type=["xlsx"])

ordered_days = ['ראשון', 'שני', 'שלישי', 'רביעי', 'חמישי', 'שישי', 'שבת']
full_shifts = ['משמרת בוקר', 'משמרת אחצ', 'משמרת לילה']

if uploaded_file is None:
    st.info("העלה קובץ אקסל כדי להתחיל (workers, requirements, preferences).")
    st.stop()

# ------------------------------------------------
# קריאת הנתונים מהקובץ שהעלית
# ------------------------------------------------
try:
    xls = pd.ExcelFile(uploaded_file)

    workers_df = pd.read_excel(xls, sheet_name="workers")
    req_df = pd.read_excel(xls, sheet_name="requirements")
    pref_df = pd.read_excel(xls, sheet_name="preferences")
except Exception as e:
    st.error(f"שגיאה בקריאת הקובץ: {e}")
    st.stop()

# מצפים לעמודות:
# workers: worker
# requirements: day, shift, required
# preferences: worker, day, shift, preference

workers = workers_df["worker"].dropna().astype(str).tolist()

# ימים ומשמרות מתוך הדרישות
req_df["day"] = req_df["day"].astype(str)
req_df["shift"] = req_df["shift"].astype(str)
req_df["required"] = req_df["required"].astype(int)

active_days = sorted(req_df["day"].unique(), key=lambda d: ordered_days.index(d) if d in ordered_days else 99)

# בניית רשימת "משבצות משמרת"
shift_slots = []
for _, row in req_df.iterrows():
    d = row["day"]
    s = row["shift"]
    r = row["required"]
    for i in range(r):
        shift_slots.append((d, s, i))

# העדפות למילון
preferences = {}
pref_df["worker"] = pref_df["worker"].astype(str)
pref_df["day"] = pref_df["day"].astype(str)
pref_df["shift"] = pref_df["shift"].astype(str)

for _, row in pref_df.iterrows():
    key = (row["worker"], row["day"], row["shift"])
    preferences[key] = int(row["preference"])

# ------------------------------------------------
# כפתור שיבוץ
# ------------------------------------------------
if st.button("🚀 בצע שיבוץ מהקובץ"):
    if not workers:
        st.error("לא נמצאו עובדים בגיליון workers.")
        st.stop()
    if not shift_slots:
        st.error("לא נמצאו דרישות בגיליון requirements.")
        st.stop()

    # מועמדים חוקיים: מי שלא סימן העדפה שלילית
    worker_copies = []
    for w in workers:
        for _, row in req_df.iterrows():
            d = row["day"]
            s = row["shift"]
            pref = preferences.get((w, d, s), 0)  # אם אין – נניח 0
            if pref >= 0:
                worker_copies.append((w, d, s))

    # מטריצת עלות
    cost_matrix = []
    for w, d, s in worker_copies:
        row = []
        for sd, ss, _ in shift_slots:
            if (d, s) == (sd, ss):
                pref = preferences.get((w, d, s), 0)
                if pref == 0:
                    row.append(100)
                else:
                    row.append(4 - pref)
            else:
                row.append(1e6)
        cost_matrix.append(row)

    cost_matrix = np.array(cost_matrix, dtype=float)
    row_ind, col_ind = simple_assignment(cost_matrix)

    assignments = []
    used_workers_in_shift = set()
    used_slots = set()
    worker_shift_count = {w: 0 for w in workers}
    worker_daily_shifts = {w: {d: [] for d in active_days} for w in workers}
    max_shifts_per_worker = len(shift_slots) // len(workers) + 1 if workers else 0

    pairs = list(zip(row_ind, col_ind))
    pairs.sort(key=lambda x: cost_matrix[x[0], x[1]])

    for r, c in pairs:
        worker, day, shift = worker_copies[r]
        slot = shift_slots[c]
        shift_key = (worker, slot[0], slot[1])

        if cost_matrix[r][c] >= 1e6:
            continue
        if shift_key in used_workers_in_shift or slot in used_slots:
            continue
        if worker_shift_count[worker] >= max_shifts_per_worker:
            continue

        current_shift_index = full_shifts.index(shift) if shift in full_shifts else 0
        if any(abs(full_shifts.index(x) - current_shift_index) == 1 for x in worker_daily_shifts[worker][day] if x in full_shifts):
            continue

        used_workers_in_shift.add(shift_key)
        used_slots.add(slot)
        assignments.append({"יום": slot[0], "משמרת": slot[1], "עובד": worker})
        worker_shift_count[worker] += 1
        worker_daily_shifts[worker][day].append(shift)

    # השלמות
    remaining_slots = [slot for slot in shift_slots if slot not in used_slots]
    unassigned_pairs = set()
    for slot in remaining_slots:
        d, s, _ = slot
        assigned = False
        for w in workers:
            if worker_shift_count[w] >= max_shifts_per_worker:
                continue
            pref = preferences.get((w, d, s), -1)
            if pref < 0:
                continue
            current_shift_index = full_shifts.index(s) if s in full_shifts else 0
            if any(abs(full_shifts.index(x) - current_shift_index) == 1 for x in worker_daily_shifts[w][d] if x in full_shifts):
                continue
            shift_key = (w, d, s)
            if shift_key in used_workers_in_shift:
                continue

            used_workers_in_shift.add(shift_key)
            used_slots.add(slot)
            assignments.append({"יום": d, "משמרת": s, "עובד": w})
            worker_shift_count[w] += 1
            worker_daily_shifts[w][d].append(s)
            assigned = True
            break
        if not assigned:
            unassigned_pairs.add((d, s))

    for d, s in unassigned_pairs:
        st.warning(f"⚠️ לא שובץ אף אחד ל־{d} - {s}")

    df = pd.DataFrame(assignments)
    if df.empty:
        st.info("לא בוצע אף שיבוץ.")
    else:
        df["יום_מספר"] = df["יום"].apply(lambda x: ordered_days.index(x) if x in ordered_days else 99)
        df = df.sort_values(by=["יום_מספר", "משמרת", "עובד"])
        df = df[["יום", "משמרת", "עובד"]]

        st.success("✅ השיבוץ הושלם!")
        st.dataframe(df, use_container_width=True)

        # שמירת היסטוריה
        try:
            save_schedule_to_excel(df, week_number, month_str or "ללא_חודש")
            st.success(f"השיבוץ נשמר בקובץ 'shifts_history.xlsx' בגיליון 'שבוע_{week_number}_{month_str}'.")
        except Exception as e:
            st.error(f"שגיאה בשמירה לאקסל: {e}")

        # סיכום לפי עובד
        st.subheader("📊 סיכום משמרות לכל עובד")
        worker_counts = df["עובד"].value_counts().reset_index()
        worker_counts.columns = ["עובד", "מספר משמרות"]
        st.dataframe(worker_counts, use_container_width=True)
        st.bar_chart(worker_counts.set_index("עובד"))

        # הורדת CSV
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ הורד את השיבוץ כ-CSV",
            data=csv,
            file_name=f"shibutz_week_{week_number}_{month_str}.csv",
            mime="text/csv",
        )
