import os
import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

# =======================
# DB CONFIG (sesuai docker)
# =======================
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5777"))
DB_NAME = os.getenv("DB_NAME", "mock_interview")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mock999")

st.set_page_config(page_title="Mock Interview – HRD", layout="wide")

@st.cache_resource
def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        cursor_factory=RealDictCursor,
    )


# =======================
# AMBIL DATA KANDIDAT
# =======================
def load_candidates():
    conn = get_db_connection()
    cur = conn.cursor()

    # Pastikan kolom id diambil eksplisit
    cur.execute("""
        SELECT
            id,
            name,
            email,
            address,
            role,
            linkedin,
            cv_path,
            hard_score,
            soft_score,
            system_score,
            hrd_score,
            final_score,
            status
        FROM candidates
        ORDER BY id DESC
    """)

    rows = cur.fetchall()
    cur.close()

    # Kalau tabel kosong → kembalikan DF kosong dengan kolom yang benar
    if not rows:
        return pd.DataFrame(columns=[
            "id","name","email","address","role","linkedin","cv_path",
            "hard_score","soft_score","system_score","hrd_score",
            "final_score","status"
        ])

    return pd.DataFrame(rows)

DISPLAY_COLUMNS = {
    "id": "ID",
    "name": "Nama",
    "email": "Email / WhatsApp",
    "address": "Alamat",
    "role": "Bidang",
    "linkedin": "LinkedIn",
    "cv_path": "Dokumen CV",
    "hard_score": "Score Hard Skill",
    "soft_score": "Score Soft Skill",
    "system_score": "Total Score System",
    "hrd_score": "HRD Score",
    "final_score": "Final Score",
    "status": "Status"
}

# =======================
# UPDATE NILAI HRD
# =======================
def update_review(candidate_id, hrd, final, status):
    conn = get_db_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE candidates
                SET hrd_score=%s,
                    final_score=%s,
                    status=%s
                WHERE id=%s
            """, (hrd, final, status, candidate_id))


# =======================
# LOGIN HRD
# =======================
USERS = {"hrd": "12345", "admin": "admin"}

def login():
    st.title("Login HRD")

    userLogin = (st.text_input("Username")).lower()
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if USERS.get(userLogin) == password:
            st.session_state["login"] = True
            st.session_state["username"] = userLogin  
            st.rerun()
        else:
            st.error("Login gagal!")


# =======================
# MAIN
# =======================
def main():
    if "login" not in st.session_state:
        return login()

    username = st.session_state.get("username", "Unknown")
    st.sidebar.success(f"Login as {(username).upper()}") 
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

    df = load_candidates()

    st.title("📋 Daftar Kandidat")

    # --- HANDLE JIKA DATABASE KOSONG ---
    if df.empty:
        st.warning("Belum ada data kandidat di database.")
        st.stop()

    # --- VALIDASI KOLom ID ---
    if "id" not in df.columns:
        st.error("Kolom 'id' tidak ditemukan di tabel candidates.")
        st.write("Kolom yang tersedia:", list(df.columns))
        st.stop()

    df_display = df.rename(columns=DISPLAY_COLUMNS)
    st.dataframe(df_display, use_container_width=True)

    # --- PILIH ID ---
    selected_id = st.selectbox("Pilih Kandidat (ID)", df["id"].tolist())

    row = df[df["id"] == selected_id].iloc[0]

    # =======================
    # DETAIL KANDIDAT
    # =======================
    st.subheader("Detail Kandidat")

    c1, c2 = st.columns(2)
    with c1:
        st.write("Nama:", row["name"])
        st.write("Email:", row["email"])
        st.write("Alamat:", row["address"])
        st.write("Role:", row["role"])
        st.write("LinkedIn:", row["linkedin"])
        st.write("Status:", row["status"])

        # Download CV
        if row.get("cv_path") and os.path.exists(row["cv_path"]):
            with open(row["cv_path"], "rb") as f:
                st.download_button("📄 Download CV", data=f, file_name=os.path.basename(row["cv_path"]))
        else:
            st.info("CV tidak tersedia.")

    with c2:
        st.metric("Hard Skill", row["hard_score"])
        st.metric("Soft Skill", row["soft_score"])
        st.metric("System Score", row["system_score"])

    # =======================
    # PENILAIAN HRD
    # =======================
    st.subheader("Penilaian HRD")

    current_hrd = 0.0 if pd.isna(row.get("hrd_score")) else float(row.get("hrd_score"))
    hrd = st.slider("HRD Score", 0.0, 100.0, current_hrd)

    status = st.selectbox("Status", ["Invalidated", "Validated"])

    final = round(0.8 * float(row["system_score"]) + 0.2 * hrd, 2)
    st.metric("Final Score", final)

    if st.button("💾 Simpan Penilaian"):
        update_review(selected_id, hrd, final, status)
        st.success("✅ Penilaian tersimpan")
        st.rerun()


if __name__ == "__main__":
    main()
