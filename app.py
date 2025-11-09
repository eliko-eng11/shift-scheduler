import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="🗕️ שיבוץ עובדים", layout="wide")
st.markdown("<h1 style='text-align:center; color:#2C3E50;'>🛠️ מערכת שיבוץ חכמה לעובדים</h1>", unsafe_allow_html=True)

# -----------------------------
# פונקציית הקצאה פשוטה (חמדנית)
# -----------------------------
def simple_assignment(cost_matrix):
    used_rows, used_cols, assignments = set(), set(), []
    rows, cols = len(cost_matrix), len(cost_matrix[0]) if len(cost_matrix) > 0 else 0

    for _ in range(min(rows, cols)):
        best, best_cost = None, 10**9
        for i in range(rows):
            if i in used_rows: continue
            for j in range(cols):
                if j in used_cols: continue
                if cost_matrix[i][j] < best_cost:
                    best_cost, best = cost_matrix[i][j], (i, j)
        if best is None: break
        r, c = best
        used_rows.add(r)
        used_cols.add(c)
        assignments.append((r, c))
    return list(zip(*assignments)) if assignments else ([], [])

# -----------------------------
# קבועים
# -----------------------------
ordered_days = ['ראשון', 'שני', 'שלישי', 'רביעי', 'חמישי', 'שישי', 'שבת']
full_shifts = ['משמרת בוקר', 'משמרת אחצ', 'משמרת לילה']

# -----------------------------
# טעינת נתונים מאקסל
# -----------------------------
st.subheader("📤 העלאת נתונים מאקסל")
uploaded_file = st.file_uploader(
    "העלה קובץ Excel עם גיליונות: workers, requirements, preferences",
    type=["xlsx", "xls"],
    help="אם לא תעלה קובץ – תוכל להזין נתונים ידנית (לא מומלץ)."
)

use_excel, df_workers, df_req, df_pref = False, None, None, None

if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        df_workers = pd.read_excel(xls, "workers")
        df_req = pd.read_excel(xls, "requirements")
        df_pref = pd.read_excel(xls, "preferences")
        use_excel = True
        st.success("✅ הנתונים נטענו בהצלחה מהקובץ!")
    except Exception as e:
        st.error(f"שגיאה בקריאת הקובץ: {e}")

# -----------------------------
# אם הועלה קובץ, מייצרים ממנו את כל הנתונים
# -----------------------------
if use_excel:
    workers = df_workers["worker"].astype(str).tolist()
    active_days = df_req["day"].astype(str).unique().tolist()
    shift_slots = []
    for _, row in df_req.iterrows():
        for i in range(int(row["required"])):
            shift_slots.append((row["day"], row["shift"], i))

    preferences = {}
    for _, row in df_pref.iterrows():
        preferences[(row["worker"], row["day"], row["shift"])] = int(row["pref"])

    st.dataframe(df_workers, use_container_width=True)
    st.dataframe(df_req, use_container_width=True)
    st.dataframe(df_pref, use_container_width=True)

    if st.button("🚀 בצע שיבוץ"):
        # בונים מטריצת עלות
        worker_copies = [(w, d, s) for (w, d, s), p in preferences.items() if p >= 0]
        cost_matrix = []
        for w, d, s in worker_copies:
            row = []
            for sd, ss, _ in shift_slots:
                if (d, s) == (sd, ss):
                    pref = preferences.get((w, d, s), 0)
                    row.append(4 - pref if pref > 0 else 100)
                else:
                    row.append(1e6)
            cost_matrix.append(row)
        cost_matrix = np.array(cost_matrix, dtype=float)

        row_ind, col_ind = simple_assignment(cost_matrix)
        assignments, used_workers, used_slots = [], set(), set()
        worker_shift_count = {w: 0 for w in workers}

        for r, c in zip(row_ind, col_ind):
            w, d, s = worker_copies[r]
            slot = shift_slots[c]
            if cost_matrix[r][c] >= 1e6 or slot in used_slots:
                continue
            assignments.append({"יום": slot[0], "משמרת": slot[1], "עובד": w})
            used_slots.add(slot)
            used_workers.add(w)
            worker_shift_count[w] += 1

        df = pd.DataFrame(assignments)
        if not df.empty:
            st.success("✅ השיבוץ הושלם!")
            st.dataframe(df, use_container_width=True)

            # שמירת תוצאה לגיליון schedule
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_workers.to_excel(writer, sheet_name="workers", index=False)
                df_req.to_excel(writer, sheet_name="requirements", index=False)
                df_pref.to_excel(writer, sheet_name="preferences", index=False)
                df.to_excel(writer, sheet_name="schedule", index=False)
            output.seek(0)

            st.download_button(
                label="⬇️ הורד קובץ אקסל עם השיבוץ (כולל גיליון schedule)",
                data=output,
                file_name="shifts_with_schedule.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # מדד איכות
            high_pref_count = sum(preferences.get((a["עובד"], a["יום"], a["משמרת"]), 0) == 3 for a in assignments)
            st.markdown(f"📊 **{high_pref_count} שיבוצים עם עדיפות 3 מתוך {len(assignments)}**")

else:
    st.info("📎 העלה קובץ אקסל כדי לבצע שיבוץ אוטומטי (ריק יוצר בעיות 😉)")
