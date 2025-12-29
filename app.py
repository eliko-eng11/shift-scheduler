st.title("🛠️ מערכת שיבוץ משמרות (Google Sheets)")

sheet_link = st.text_input("הדבק קישור Google Sheet (עם tabs: workers / requirements / preferences)")
week_number = st.number_input("מספר שבוע לשיבוץ", min_value=1, step=1, value=1)

if st.button("🚀 בצע שיבוץ וכתוב חזרה ל-Google Sheet"):
    try:
        st.write("שלב 1: חילוץ Sheet ID...")
        sheet_id = extract_sheet_id(sheet_link)
        st.write("Sheet ID:", sheet_id)
        if not sheet_id:
            st.error("לא זיהיתי Sheet ID. הדבק קישור מלא של Google Sheets.")
            st.stop()

        st.write("שלב 2: יצירת חיבור ל-Google (service account)...")
        gc = get_gspread_client()
        st.success("✅ התחברות ל-Google הצליחה")

        st.write("שלב 3: פתיחת הקובץ לפי ID...")
        sh = gc.open_by_key(sheet_id)
        st.success(f"✅ נפתח הקובץ: {sh.title}")

        st.write("שלב 4: רשימת טאבים בקובץ:")
        tab_names = [w.title for w in sh.worksheets()]
        st.write(tab_names)

        # בדיקת טאבים
        required_tabs = {"workers", "requirements", "preferences"}
        if not required_tabs.issubset(set(tab_names)):
            st.error(f"חסרים טאבים. חייב להיות: {sorted(list(required_tabs))}")
            st.stop()

        st.write("שלב 5: קריאת הנתונים...")
        workers_df = read_sheet_as_df(sh, "workers")
        req_df     = read_sheet_as_df(sh, "requirements")
        pref_df    = read_sheet_as_df(sh, "preferences")

        st.write("שורות שנקראו:",
                 {"workers": len(workers_df), "requirements": len(req_df), "preferences": len(pref_df)})

        st.write("שלב 6: הרצת שיבוץ...")
        schedule_df, unassigned_pairs = build_schedule(workers_df, req_df, pref_df, int(week_number))

        st.write("שלב 7: כתיבה חזרה לטאב חדש...")
        new_ws_name = f"שבוע {int(week_number)}"
        write_df_to_worksheet(sh, new_ws_name, schedule_df)

        st.success(f"✅ השיבוץ נכתב בהצלחה! טאב חדש: {new_ws_name}")
        st.dataframe(schedule_df, use_container_width=True)

        if unassigned_pairs:
            st.warning(f"⚠️ לא שובצו: {sorted(list(unassigned_pairs))}")

    except Exception as e:
        st.exception(e)
