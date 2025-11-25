import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO


# -----------------------------
# פונקציית הקצאה חמדנית (כבר לא בשימוש כרגע, אבל נשאיר למקרה שתרצה)
# -----------------------------
def simple_assignment(cost_matrix):
    """
    מקבל מטריצת עלויות ומחזיר התאמות (rows, cols) בצורה חמדנית.
    זה לא האלגוריתם ההונגרי המלא, אבל עובד טוב לדמו וליישום שלך.
    """
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


# -----------------------------
# בניית שיבוץ מתוך שלושת הגיליונות
# -----------------------------
def build_schedule(workers_df, req_df, pref_df, week_number):
    # ניקוי שמות עמודות ומיפוי במקרה של שמות בעברית
    workers_df.columns = workers_df.columns.str.strip()
    req_df.columns = req_df.columns.str.strip()
    pref_df.columns = pref_df.columns.str.strip()

    # אם אצלך שמות אחרים – תעדכן פה:
    workers_df = workers_df.rename(columns={
        "שם עובד": "worker",
    })
    req_df = req_df.rename(columns={
        "יום": "day",
        "משמרת": "shift",
        "כמות נדרשת": "required"
    })
    pref_df = pref_df.rename(columns={
        "עדיפות": "preference",
        "עובד": "worker",
        "יום": "day",
        "משמרת": "shift"
    })

    # רשימת עובדים
    workers = (
        workers_df["worker"]
        .dropna()
        .astype(str)
        .tolist()
    )

    if not workers:
        raise ValueError("לא נמצאו עובדים בגיליון 'workers'")

    # סלוטים של משמרות לפי הדרישות
    req_df["required"] = req_df["required"].fillna(0).astype(int)
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
        raise ValueError("לא נמצאו דרישות משמרות בגיליון 'requirements'")

    # רשימת ימים וסוגי משמרות (לסידור)
    ordered_days = list(dict.fromkeys([d for d, _, _ in shift_slots]))
    full_shifts = list(dict.fromkeys([s for _, s, _ in shift_slots]))

    # העדפות למילון
    pref_dict = {}
    for _, row in pref_df.iterrows():
        w = str(row["worker"])
        d = str(row["day"])
        s = str(row["shift"])
        try:
            p = int(row["preference"])
        except Exception:
            continue
        pref_dict[(w, d, s)] = p

    # מבני נתונים לניהול השיבוץ
    assignments = []
    worker_shift_count = {w: 0 for w in workers}
    worker_daily_shifts = {w: {d: [] for d in ordered_days} for w in workers}
    unassigned_pairs = set()

    # כמה שיבוצים מקסימום לעובד (אותו רעיון כמו קודם – חלוקה הוגנת)
    max_shifts_per_worker = len(shift_slots) // len(workers) + 1 if workers else 0

    # ⬇️ הכנה: לכל סלוט – מי בכלל יכול לעבוד שם (בלי קשר למגבלות הוגנות)
    base_slot_candidates = {}
    for slot in shift_slots:
        d, s, _ = slot
        cands = []
        for w in workers:
            pref = pref_dict.get((w, d, s), -1)
            if pref >= 0:  # שלילי = לא זמין בכלל
                cands.append(w)
        base_slot_candidates[slot] = cands

    # נסדר את הסלוטים לפי:
    # 1. כמה מועמדים יש להם (כמה שפחות -> קודם)
    # 2. יום בשבוע (כדי שיהיה יציב)
    # 3. סוג משמרת (סדר המשמרות)
    def slot_sort_key(slot):
        d, s, _ = slot
        return (
            len(base_slot_candidates.get(slot, [])),
            ordered_days.index(d),
            full_shifts.index(s) if s in full_shifts else 0,
        )

    ordered_slots = sorted(shift_slots, key=slot_sort_key)

    # עכשיו נעבור סלוט סלוט, כדי למלא קודם את המשמרות ה"בעייתיות"
    for slot in ordered_slots:
        d, s, _ = slot
        possible_workers = base_slot_candidates.get(slot, [])

        if not possible_workers:
            # אף אחד לא זמין למשמרת הזו
            unassigned_pairs.add((d, s))
            continue

        chosen_worker = None

        # ננסה בשלוש רמות הקשחה:
        # רמה 1: לכבד הכל – לא לעבור מקסימום, לא צמודות, עדיפות גבוהה
        # רמה 2: מרפים את כלל הצמודות (עדיין מכבדים מקסימום)
        # רמה 3: מרפים גם את המקסימום כדי לא להשאיר חורים
        for relax_level in [1, 2, 3]:
            best_w = None
            best_pref = -999
            best_shifts_so_far = 10**9

            for w in possible_workers:
                pref = pref_dict.get((w, d, s), -1)
                if pref < 0:
                    continue

                # רמת הקשחה 1: לכבד מקסימום + לא צמודות
                if relax_level <= 2:
                    if worker_shift_count[w] >= max_shifts_per_worker:
                        continue

                # בדיקת צמודות רק ברמות 1
                if relax_level == 1:
                    try:
                        current_shift_index = full_shifts.index(s)
                    except ValueError:
                        current_shift_index = 0

                    if any(
                        abs(full_shifts.index(x) - current_shift_index) == 1
                        for x in worker_daily_shifts[w][d]
                    ):
                        continue

                # ברמה 3 – לא בודקים כלום חוץ מזמינות
                # בחירה מבוססת עדיפות, ואם יש תיקו – מי שעבד פחות
                shifts_so_far = worker_shift_count[w]
                if pref > best_pref or (pref == best_pref and shifts_so_far < best_shifts_so_far):
                    best_pref = pref
                    best_w = w
                    best_shifts_so_far = shifts_so_far

            if best_w is not None:
                chosen_worker = best_w
                break  # יציאה מה-relax_level loop

        if chosen_worker is None:
            # לא הצלחנו לשבץ אף אחד גם אחרי ריכוך
            unassigned_pairs.add((d, s))
            continue

        # מוסיפים את השיבוץ
        assignments.append(
            {
                "שבוע": week_number,
                "יום": d,
                "משמרת": s,
                "עובד": chosen_worker,
            }
        )
        worker_shift_count[chosen_worker] += 1
        worker_daily_shifts[chosen_worker][d].append(s)

    # יצירת DataFrame
    df = pd.DataFrame(assignments)

    if df.empty:
        raise ValueError("לא נוצר אף שיבוץ. בדוק את הנתונים בגיליונות.")

    # סידור לפי ימים, משמרת, עובד
    df["יום_מספר"] = df["יום"].apply(lambda x: ordered_days.index(x))
    df = df.sort_values(by=["שבוע", "יום_מספר", "משמרת", "עובד"])
    df = df[["שבוע", "יום", "משמרת", "עובד"]]

    return df, unassigned_pairs


# -----------------------------
# אפליקציית Streamlit
# -----------------------------
st.set_page_config(page_title="מערכת שיבוץ חכמה לעובדים", layout="wide")
st.title("🛠️ מערכת שיבוץ משמרות מעולה")

uploaded_file = st.file_uploader("העלה קובץ אקסל קיים", type=["xlsx"])

week_number = st.number_input("מספר שבוע לשיבוץ", min_value=1, step=1, value=1)

if uploaded_file is None:
    st.info("העלה קובץ אקסל עם הגיליונות workers, requirements, preferences כדי להתחיל.")
    st.stop()

# כפתור הפעלת שיבוץ
if st.button("🚀 בצע שיבוץ והוסף גיליון חדש לקובץ"):
    try:
        xls = pd.ExcelFile(uploaded_file)

        workers_df = pd.read_excel(xls, sheet_name="workers")
        req_df = pd.read_excel(xls, sheet_name="requirements")
        pref_df = pd.read_excel(xls, sheet_name="preferences")

        schedule_df, unassigned_pairs = build_schedule(
            workers_df, req_df, pref_df, week_number
        )

        st.success("✅ השיבוץ הוכן בהצלחה!")
        st.dataframe(schedule_df, use_container_width=True)

        # הצגת אזהרות על משמרות שלא שובצו
        if unassigned_pairs:
            for d, s in sorted(unassigned_pairs):
                st.warning(f"⚠️ לא שובץ אף אחד ל־{d} - {s}")

        # -----------------------------
        # כתיבת הקובץ המעודכן לבאפר
        # -----------------------------
        new_sheet_name = f"שבוע {int(week_number)}"
        original_sheet_names = xls.sheet_names

        # אם יש כבר גיליון בשם הזה – נוסיף (2)
        if new_sheet_name in original_sheet_names:
            st.warning(
                f"קיים כבר גיליון בשם '{new_sheet_name}'. הגיליון החדש ייקרא '{new_sheet_name} (2)'."
            )
            new_sheet_name = f"{new_sheet_name} (2)"

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            # העתקת כל הגיליונות הקיימים
            for sheet in original_sheet_names:
                df_old = pd.read_excel(xls, sheet_name=sheet)
                df_old.to_excel(writer, sheet_name=sheet, index=False)

            # הוספת גיליון השבוע החדש
            schedule_df.to_excel(writer, sheet_name=new_sheet_name, index=False)

        output.seek(0)

        st.download_button(
            label="⬇️ הורד את הקובץ המעודכן (עם היסטוריית השבועות)",
            data=output,
            file_name=uploaded_file.name,  # שומר על אותו שם קובץ
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"שגיאה במהלך השיבוץ: {e}")
