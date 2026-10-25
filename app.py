import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ------------------------------------------------------------------
# Konfigurasi halaman
# ------------------------------------------------------------------
st.set_page_config(page_title="Klasifikasi Hujan Harian Kolkata",
                   page_icon="🌦️", layout="wide")
sns.set_style("whitegrid")

BASE_DIR = Path(__file__).resolve().parent
LABEL_NAMES = ["Tidak Hujan", "Hujan"]
PALETTE = {"Tidak Hujan": "#E8A33D", "Hujan": "#2F6DB5"}
NAMA_BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
              "Agustus", "September", "Oktober", "November", "Desember"]
FILE_PKL = ["model.pkl", "feature_columns.pkl", "season_categories.pkl",
            "eval_data.pkl", "model_info.pkl"]


def get_musim(bulan):
    if bulan in (12, 1, 2):
        return "Dingin"
    if bulan in (3, 4, 5):
        return "Panas"
    if bulan in (6, 7, 8, 9):
        return "Monsun"
    return "Pasca-Monsun"


# ------------------------------------------------------------------
# Memuat model dan artefak hasil notebook
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model dan data evaluasi...")
def load_artifacts():
    hasil = {}
    for nama in FILE_PKL:
        with open(BASE_DIR / nama, "rb") as f:
            hasil[nama.replace(".pkl", "")] = pickle.load(f)
    return hasil


hilang = [n for n in FILE_PKL if not (BASE_DIR / n).exists()]
if hilang:
    st.error("File berikut tidak ditemukan di folder yang sama dengan app.py: "
             + ", ".join(hilang)
             + ". Jalankan Cell 17 pada notebook untuk membuatnya, lalu letakkan di folder ini.")
    st.stop()

art = load_artifacts()
model = art["model"]
feature_columns = art["feature_columns"]
season_categories = art["season_categories"]
ev = art["eval_data"]
info = art["model_info"]

data = ev["data"]
numeric_features = ev.get("numeric_features", info["numeric_features"])
cm = np.asarray(ev["cm"])
fpr, tpr, auc_value = ev["fpr"], ev["tpr"], float(ev["auc"])

metrics_df = ev["metrics_df"]
metrics = dict(zip(metrics_df["Metrik"], metrics_df["Nilai"]))

imp = ev["importances"]
if isinstance(imp, pd.Series):
    imp = imp.reset_index()
imp = imp.copy()
imp.columns = ["Fitur", "Importance"]          # kolom hasil reset_index() bernama 'index' dan 0
imp = imp.sort_values("Importance")

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.header("⚙️ Informasi Model")
params = model.get_params()
st.sidebar.code("\n".join(f"{k} = {params[k]}" for k in
                          ["n_estimators", "learning_rate", "max_depth", "random_state"]
                          if k in params))
st.sidebar.markdown("**Split data:** 80% latih / 20% uji")
st.sidebar.markdown(f"**Jumlah fitur:** {len(feature_columns)}")

# ------------------------------------------------------------------
# Judul
# ------------------------------------------------------------------
st.title("🌦️ Klasifikasi Hujan Harian Kota Kolkata dengan XGBoost")
st.markdown(
    "Dashboard ini menampilkan hasil klasifikasi apakah pada suatu hari **terjadi hujan** "
    "(curah hujan `rain_sum` > 0 mm) atau **tidak hujan**, berdasarkan suhu, angin, "
    "durasi sinar matahari, panjang siang, dan musim. Model yang digunakan adalah "
    "**XGBoost Classifier**. Kolom `weather_code`, `precipitation_sum`, `rain_sum`, dan "
    "`precipitation_hours` tidak dipakai sebagai fitur karena akan menyebabkan *data leakage*."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Ringkasan Dataset", "📊 EDA", "🤖 Evaluasi Model", "🔮 Prediksi Manual"])

# ------------------------------------------------------------------
# Tab 1: Ringkasan dataset
# ------------------------------------------------------------------
with tab1:
    n_data = len(data)
    n_hujan = int(data["hujan"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah data", f"{n_data:,}")
    c2.metric("Jumlah fitur (setelah encoding)", len(feature_columns))
    c3.metric("Hari hujan", f"{n_hujan:,} ({n_hujan / n_data:.1%})")
    c4.metric("Hari tidak hujan", f"{n_data - n_hujan:,} ({1 - n_hujan / n_data:.1%})")

    st.caption(f"Rentang tanggal: {pd.to_datetime(data['time']).min().date()} "
               f"s.d. {pd.to_datetime(data['time']).max().date()}")
    st.subheader("Cuplikan data")
    st.dataframe(data.head(10))
    st.subheader("Statistik deskriptif fitur")
    st.dataframe(data[numeric_features].describe().T.round(2))

# ------------------------------------------------------------------
# Tab 2: EDA
# ------------------------------------------------------------------
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Proporsi kelas target")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=data, x="label", hue="label", order=LABEL_NAMES,
                      legend=False, palette=PALETTE, ax=ax)
        for p in ax.patches:
            n = int(p.get_height())
            ax.annotate(f"{n}\n({n / len(data):.1%})",
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom")
        ax.set_xlabel("")
        ax.set_ylabel("Jumlah hari")
        ax.set_ylim(0, data["label"].value_counts().max() * 1.18)
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.subheader("Distribusi fitur")
        fitur = st.selectbox("Pilih fitur", numeric_features,
                             index=numeric_features.index("temp_range")
                             if "temp_range" in numeric_features else 0)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(data=data, x=fitur, hue="label", hue_order=LABEL_NAMES,
                     kde=True, bins=30, palette=PALETTE, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Korelasi antar fitur")
    corr = data[numeric_features + ["hujan"]].corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

# ------------------------------------------------------------------
# Tab 3: Evaluasi model
# ------------------------------------------------------------------
with tab3:
    cols = st.columns(len(metrics))
    for col, (nama, nilai) in zip(cols, metrics.items()):
        col.metric(nama, f"{nilai:.4f}")

    st.subheader("Tabel ringkasan metrik")
    st.dataframe(metrics_df, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Confusion Matrix")
        cm_persen = cm / cm.sum(axis=1, keepdims=True)
        anotasi = np.array([[f"{cm[i, j]}\n({cm_persen[i, j]:.1%})" for j in range(2)]
                            for i in range(2)])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=anotasi, fmt="", cmap="Blues", cbar=False,
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)
        plt.close(fig)
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"TN = {tn} | FP = {fp} | FN = {fn} | TP = {tp}")

    with c2:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color="#2F6DB5", lw=2, label=f"XGBoost (AUC = {auc_value:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Tebakan acak (AUC = 0.5)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp["Fitur"], imp["Importance"], color="#2F6DB5")
    for i, v in enumerate(imp["Importance"]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Importance (gain)")
    st.pyplot(fig)
    plt.close(fig)

# ------------------------------------------------------------------
# Tab 4: Prediksi manual
# ------------------------------------------------------------------
with tab4:
    st.subheader("Coba prediksi sendiri")
    st.caption("Masukkan kondisi cuaca harian, lalu klik tombol prediksi.")

    med = data[numeric_features].median()
    bulan_data = data["bulan"] if "bulan" in data.columns else pd.to_datetime(data["time"]).dt.month
    daylight_by_month = data.groupby(bulan_data)["daylight_hours"].median()

    with st.form("form_prediksi"):
        c1, c2, c3 = st.columns(3)
        with c1:
            bulan_nama = st.selectbox("Bulan", NAMA_BULAN, index=6)
            t_mean = st.number_input("Suhu rata-rata (°C)", 0.0, 50.0,
                                     float(round(med["temperature_2m_mean"], 1)), 0.1)
            t_max = st.number_input("Suhu maksimum (°C)", 0.0, 55.0,
                                    float(round(med["temperature_2m_max"], 1)), 0.1)
        with c2:
            t_min = st.number_input("Suhu minimum (°C)", 0.0, 50.0,
                                    float(round(med["temperature_2m_min"], 1)), 0.1)
            w_max = st.number_input("Kecepatan angin maks (km/jam)", 0.0, 100.0,
                                    float(round(med["wind_speed_10m_max"], 1)), 0.1)
            w_gust = st.number_input("Hembusan angin maks (km/jam)", 0.0, 150.0,
                                     float(round(med["wind_gusts_10m_max"], 1)), 0.1)
        with c3:
            w_dir = st.slider("Arah angin dominan (derajat)", 0, 360,
                              int(med["wind_direction_10m_dominant"]))
            sun = st.slider("Durasi sinar matahari (jam)", 0.0, 14.0,
                            float(round(med["sunshine_hours"], 1)), 0.1)
        submit = st.form_submit_button("Prediksi")

    if submit:
        bulan = NAMA_BULAN.index(bulan_nama) + 1
        musim = get_musim(bulan)
        if not (t_min <= t_mean <= t_max):
            st.error("Pastikan suhu minimum ≤ suhu rata-rata ≤ suhu maksimum.")
        else:
            baris = {
                "temperature_2m_mean": t_mean, "temperature_2m_max": t_max,
                "temperature_2m_min": t_min, "temp_range": t_max - t_min,
                "wind_speed_10m_max": w_max, "wind_gusts_10m_max": w_gust,
                "wind_direction_10m_dominant": w_dir, "sunshine_hours": sun,
                "daylight_hours": float(daylight_by_month.loc[bulan]),
            }
            for s in season_categories:
                baris[f"musim_{s}"] = int(s == musim)
            x_input = pd.DataFrame([baris])[feature_columns]   # urutan kolom sama dengan saat pelatihan

            proba = float(model.predict_proba(x_input)[0, 1])
            if proba >= 0.5:
                st.success(f"Prediksi: **HUJAN** (probabilitas {proba:.1%})")
            else:
                st.info(f"Prediksi: **TIDAK HUJAN** (probabilitas hujan {proba:.1%})")
            st.progress(proba)
            st.caption(f"Musim terdeteksi: {musim}. "
                       "Panjang siang diisi otomatis dari median bulan tersebut.")