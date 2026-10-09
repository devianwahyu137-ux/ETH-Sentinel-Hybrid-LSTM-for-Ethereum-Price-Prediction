# 📦 Legacy Prototype Archive (Arsip Artefak Lama)

Folder ini berisi berkas-berkas kode dan notebook prototipe awal proyek penelitian **ETH-Sentinel** sebelum dilakukannya perombakan metodologis berstandar jurnal ilmiah terakreditasi (*Journal of Accounting and Investment* - SINTA 2 / Scopus Indexed).

---

### ⚠️ Catatan Penting & Integritas Ilmiah
Seluruh berkas dalam folder ini **TELAH DIARSIPKAN SECARA PERMANEN** dan **TIDAK DIGUNAKAN** dalam pipeline komputasi resmi, studi ablasi, maupun penarikan kesimpulan pada naskah publikasi final, dengan pertimbangan metodologis berikut:

1. **`01_fetch_news.py`**:
   - Skrip penarik berita lama via CryptoCompare publik (terbatas 20 artikel per panggilan dan saat ini telah dikunci di balik skema komersial berbayar oleh penyedia data).
   - Telah digantikan oleh [`Code/01_fetch_fng_sentiment.py`](file:///Users/devianwahyu/Documents/GitHub/ETH-Sentinel-Hybrid-LSTM-for-Ethereum-Price-Prediction./Code/01_fetch_fng_sentiment.py) (*Crypto Fear & Greed Index* - 100% terbuka, bebas biaya, historis lengkap sejak 2018).

2. **`02_create_dataset.py`**:
   - Pembuatan dataset awal dengan ukuran sampel mikro (~2.450 baris) tanpa isolasi train/val/test dan tanpa uji stasioneritas formal.
   - Telah digantikan oleh [`Code/02_create_clean_dataset.py`](file:///Users/devianwahyu/Documents/GitHub/ETH-Sentinel-Hybrid-LSTM-for-Ethereum-Price-Prediction./Code/02_create_clean_dataset.py) (4.370 bar 4H, uji ADF $p < 0.0001$, penyelarasan waktu UTC, dan fitur volatilitas Garman-Klass).

3. **`03_EDA_Visualisasi.ipynb` & `04_4H_data.ipynb`**:
   - Notebook eksplorasi awal yang menggunakan penskalaan global (*data leakage risk*).
   - Telah digantikan oleh modul independen [`Code/03_prepare_sequences.py`](file:///Users/devianwahyu/Documents/GitHub/ETH-Sentinel-Hybrid-LSTM-for-Ethereum-Price-Prediction./Code/03_prepare_sequences.py) dengan `RobustScaler` terisolasi pada set pelatihan saja.

4. **`05_train_daily_model.ipynb`**:
   - Notebook pelatihan lama yang memuat anomali metodologis shift target ke belakang (`shift(-1)` terbalik) dan evaluasi akurasi semu (~87%).
   - Telah digantikan oleh pipeline benchmarking ekonometrika resmi [`Code/04_train_technical_baselines.py`](file:///Users/devianwahyu/Documents/GitHub/ETH-Sentinel-Hybrid-LSTM-for-Ethereum-Price-Prediction./Code/04_train_technical_baselines.py), [`Code/05_train_hybrid_sentinel.py`](file:///Users/devianwahyu/Documents/GitHub/ETH-Sentinel-Hybrid-LSTM-for-Ethereum-Price-Prediction./Code/05_train_hybrid_sentinel.py), dan [`Code/06_sensitivity_analysis_threshold.py`](file:///Users/devianwahyu/Documents/GitHub/ETH-Sentinel-Hybrid-LSTM-for-Ethereum-Price-Prediction./Code/06_sensitivity_analysis_threshold.py).

---

### 🚀 Pipeline Resmi yang Bersih & Tervalidasi
Untuk mereplikasi penelitian, gunakan kode produksi terstandarisasi di direktori `Code/`:
```bash
python Code/01_fetch_fng_sentiment.py
python Code/02_create_clean_dataset.py
python Code/03_prepare_sequences.py
python Code/04_train_technical_baselines.py
python Code/05_train_hybrid_sentinel.py
python Code/06_sensitivity_analysis_threshold.py
```
