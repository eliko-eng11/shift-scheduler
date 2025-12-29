st.title("🛠️ מערכת שיבוץ משמרות (Google Sheets)")

sheet_link = st.text_input("הדבק קישור Google Sheet (עם workers/requirements/preferences)")
week_number = st.number_input("מספר שבוע לשיבוץ", min_value=1, step=1, value=1)

if st.button("🚀 בצע שיבוץ וכתוב חזרה ל-Google Sheet"):
    try:
        sheet_link = (sheet_link or "").strip()

        # 1) חילוץ ID מהקישור
        sheet_id = extract_sheet_id(sheet_link)
        st.write("DEBUG sheet_id:", sheet_id)

        if not sheet_id:
            st.error("לא זיהיתי Sheet ID. הדבק קישור מלא של Google Sheets.")
            st.stop()

        # 2) התחברות לגוגל
        gc = get_gspread_client()
        st.success("✅ התחברתי ל-Google API")

        # 3) פתיחת ה-Sheet
        sh = gc.open_by_key(sheet_id)
        st.success(f"✅ נפתח הקובץ: {sh.title}")

        # DEBUG: רשימת טאבים
        tabs = [ws.title for ws in sh.worksheets()]
        st.write("DEBUG tabs:", tabs)

        # 4) קריאת טאבים
        workers_df = read_sheet_as_df(sh, "workers")
        req_df     = read_sheet_as_df(sh, "requirements")
        pref_df    = read_sheet_as_df(sh, "preferences")

        st.write("DEBUG sizes:",
                 "workers", len(workers_df),
                 "requirements", len(req_df),
                 "preferences", len(pref_df))

        # 5) הרצת שיבוץ
        schedule_df, unassigned_pairs = build_schedule(
            workers_df, req_df, pref_df, int(week_number)
        )

        # 6) כתיבה חזרה
        new_ws_name = f"שבוע {int(week_number)}"
        write_df_to_worksheet(sh, new_ws_name, schedule_df)

        st.success(f"✅ השיבוץ נכתב בהצלחה! (טאב חדש: {new_ws_name})")
        st.dataframe(schedule_df, use_container_width=True)

        if unassigned_pairs:
            for d, s in sorted(list(unassigned_pairs)):
                st.warning(f"⚠️ לא שובץ אף אחד ל־{d} - {s}")

    except Exception as e:
        st.exception(e)
