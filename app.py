"""Aplikasi web TA-03: Simulasi Monte Carlo Profit Ritel (Supermarket Sales)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Monte Carlo Profit Ritel", layout="wide")
st.title("Simulasi Monte Carlo: Prediksi Profit Ritel")
st.caption("TA-03 Pemodelan dan Simulasi | Dataset: Supermarket Sales (Kaggle)")

# ---------------- Sidebar: input ----------------
st.sidebar.header("Pengaturan")
berkas = st.sidebar.file_uploader("Unggah CSV Supermarket Sales", type="csv")
nim = st.sidebar.number_input("Seed (NIM)", value=301230043, step=1)
n_scen = st.sidebar.slider("Jumlah skenario", 5000, 50000, 10000, step=1000)
margin = st.sidebar.slider("Asumsi margin profit (% dari penjualan)", 1, 30, 10) / 100
faktor = st.sidebar.slider("Target profit (x rata-rata historis)", 0.80, 1.30, 1.05, step=0.01)

if berkas is None:
    st.info("Unggah file CSV dataset Supermarket Sales di sidebar untuk memulai.")
    st.stop()

# ---------------- Load dan cleaning ----------------
df = pd.read_csv(berkas)
df.columns = df.columns.str.strip()
df = df.rename(columns={c: {"unit price": "Unit price", "quantity": "Quantity",
                            "date": "Date"}[c.lower()]
                        for c in df.columns if c.lower() in ("unit price", "quantity", "date")})
perlu = {"Unit price", "Quantity", "Date"}
if not perlu.issubset(df.columns):
    st.error(f"Kolom wajib tidak lengkap. Dibutuhkan: {sorted(perlu)}")
    st.stop()

n_awal = len(df)
df = df.drop_duplicates()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Unit price", "Quantity", "Date"])

def tanpa_outlier(s, k=3.0):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return s.between(q1 - k * iqr, q3 + k * iqr)

df = df[tanpa_outlier(df["Unit price"]) & tanpa_outlier(df["Quantity"])].copy()
st.write(f"Baris awal: **{n_awal}**, setelah cleaning: **{len(df)}**")

harga, qty = df["Unit price"], df["Quantity"]

# ---------------- Parameter dan distribusi ----------------
def pilih_distribusi(x):
    mu, sd = x.mean(), x.std(ddof=1)
    ks_n = stats.kstest(x, "norm", args=(mu, sd)).statistic
    ks_u = stats.kstest(x, "uniform", args=(x.min(), x.max() - x.min())).statistic
    return "uniform" if ks_u < ks_n else "normal"

dist_h, dist_q = pilih_distribusi(harga), pilih_distribusi(qty)

st.subheader("1. Parameter dari data asli")
st.dataframe(pd.DataFrame({
    "Mean": [harga.mean(), qty.mean()], "Std Dev": [harga.std(ddof=1), qty.std(ddof=1)],
    "Min": [harga.min(), qty.min()], "Max": [harga.max(), qty.max()],
    "Distribusi terpilih": [dist_h, dist_q]}, index=["Unit price", "Quantity"]).round(4))

c1, c2 = st.columns(2)
f1, a1 = plt.subplots(); a1.hist(harga, bins=30, color="skyblue", edgecolor="black")
a1.set_title("Histogram Unit Price"); a1.set_xlabel("Unit price"); a1.set_ylabel("Frekuensi")
c1.pyplot(f1)
f2, a2 = plt.subplots()
a2.hist(qty, bins=np.arange(qty.min() - 0.5, qty.max() + 1.5, 1), color="lightgreen", edgecolor="black")
a2.set_title("Histogram Quantity"); a2.set_xlabel("Quantity"); a2.set_ylabel("Frekuensi")
c2.pyplot(f2)

# ---------------- Pola bulanan ----------------
df["Bulan"] = df["Date"].dt.to_period("M")
df["Penjualan"] = df["Unit price"] * df["Quantity"]
bulanan = df.groupby("Bulan").agg(transaksi=("Penjualan", "size"), penjualan=("Penjualan", "sum"))
bulanan["profit"] = bulanan["penjualan"] * margin
n_tx = int(round(bulanan["transaksi"].mean()))
profit_hist = bulanan["profit"].mean()
target = faktor * profit_hist

st.subheader("2. Pola bulanan historis")
tab = bulanan.copy(); tab.index = tab.index.astype(str)
st.dataframe(tab.round(2))
m1, m2, m3 = st.columns(3)
m1.metric("Transaksi per bulan", n_tx)
m2.metric("Rata-rata profit historis", f"{profit_hist:,.0f}")
m3.metric("Target profit", f"{target:,.0f}")

# ---------------- Simulasi ----------------
rng = np.random.default_rng(int(nim))

def bangkitkan(x, dist, ukuran, bulat=False):
    mu, sd, lo, hi = x.mean(), x.std(ddof=1), x.min(), x.max()
    if dist == "normal":
        v = np.clip(rng.normal(mu, sd, ukuran), lo, hi)
        return np.rint(v) if bulat else v
    if bulat:
        return rng.integers(int(lo), int(hi) + 1, ukuran).astype(float)
    return rng.uniform(lo, hi, ukuran)

sim_h = bangkitkan(harga, dist_h, (n_scen, n_tx))
sim_q = bangkitkan(qty, dist_q, (n_scen, n_tx), bulat=True)
profit_sim = (sim_h * sim_q).sum(axis=1) * margin

# ---------------- Hasil ----------------
p_capai = float(np.sum(profit_sim >= target) / n_scen)
st.subheader("3. Hasil simulasi dan analisis risiko")
r1, r2, r3 = st.columns(3)
r1.metric("P(profit >= target)", f"{p_capai * 100:.2f}%")
r2.metric("Risiko (profit < target)", f"{(1 - p_capai) * 100:.2f}%")
r3.metric("Rata-rata profit simulasi", f"{profit_sim.mean():,.0f}")

fig, ax = plt.subplots(figsize=(10, 5))
_, bins, patches = ax.hist(profit_sim, bins=50, color="skyblue", edgecolor="black")
for b, p in zip(bins[:-1], patches):
    if b < target:
        p.set_facecolor("salmon")
ax.axvline(target, color="red", linestyle="--", linewidth=2, label=f"Target = {target:,.0f}")
ax.axvline(profit_sim.mean(), color="green", linewidth=2, label=f"Rata-rata = {profit_sim.mean():,.0f}")
ax.set_title(f"Distribusi Profit Bulanan ({n_scen:,} skenario)")
ax.set_xlabel("Profit bulanan"); ax.set_ylabel("Frekuensi"); ax.legend()
st.pyplot(fig)

st.write("**Sensitivitas target**")
st.dataframe(pd.DataFrame([{"Faktor target": f, "Target profit": round(f * profit_hist, 2),
    "P(profit >= target)": round(float(np.mean(profit_sim >= f * profit_hist)), 4)}
    for f in [0.90, 0.95, 1.00, 1.05, 1.10, 1.15]]))

st.caption("Asumsi: margin profit ditentukan pengguna; transaksi independen; "
           "data hanya mencakup tiga bulan.")