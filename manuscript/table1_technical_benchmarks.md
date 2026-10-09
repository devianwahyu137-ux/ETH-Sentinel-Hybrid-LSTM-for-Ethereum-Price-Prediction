# Tabel 1 & Analisis Hasil: Benchmark Model Berbasis Fitur Teknikal Murni (Ablation Baseline)

Dokumen ini memuat format tabel Markdown, tabel LaTeX siap pakai untuk draf manuskrip jurnal ilmiah (*Journal of Accounting and Investment* - SINTA 2 / Scopus indexed), serta catatan analisis ekonometrika dan finansial.

---

## 1. Tabel Hasil Komparasi (Format Markdown)

**Tabel 1. Performa Komparatif Model Prediksi Arah Harga Ethereum (Out-of-Sample Test Set, $N = 638$ Bar 4-Jam, Juni–September 2026)**

| Model | Directional Accuracy (DA %) | Macro F1-Score | ROC-AUC | MCC | Brier Score | Cumulative Return (%) | Annualized Sharpe Ratio | Maximum Drawdown (%) | Turnover Trades | Total Friction Cost (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Naive (Buy & Hold)** | 50.63% | 0.3361 | 0.5000 | 0.0000 | 0.2500 | **+33.57%** | **2.14** | **-17.73%** | 1 | 0.10% |
| **Logistic Regression (Ridge)** | 50.31% | 0.5031 | 0.5011 | 0.0063 | 0.2554 | -24.87% | -2.74 | -26.13% | 272 | 27.20% |
| **Random Forest ($N=100, D=5$)** | **53.13%** | **0.5308** | **0.5492** | **0.0621** | **0.2485** | +1.88% | 0.36 | -21.41% | 192 | 19.20% |
| **Proposed 2-Layer LSTM** | 51.57% | 0.4743 | 0.5339 | 0.0293 | 0.2494 | **+13.73%** | **1.20** | -24.34% | **39** | **3.90%** |

*Catatan Metodologis:*
1. Partisi data: Train (70%, $N=3.049$ bar), Validation (15%, $N=653$ bar), Test (15%, $N=655$ bar).
2. Lookback sekuens $W = 18$ bar (72 jam / 3 hari kalender), menghasilkan $N = 638$ sampel uji terisolasi.
3. Fitur teknikal masukan: `Log_Return`, `GK_Volatility` (Garman-Klass), `CloseToClose_Vol`, dan `RSI_14`. Normalisasi menggunakan `RobustScaler` yang di-fit secara eksklusif pada data training (*zero information leakage*).
4. Asumsi friksi perdagangan: Biaya transaksi riil sebesar 10 basis poin (0.10%) dikenakan pada setiap pergantian posisi (*turnover*). Rasio Sharpe disetahunkan menggunakan faktor akar kuadrat frekuensi tahunan pasar kripto 24/7 ($\sqrt{2.190} \approx 46.797$).

---

## 2. Format Tabel LaTeX (Siap Pakai untuk Naskah Jurnal)

```latex
\begin{table*}[t]
\centering
\small
\caption{Empirical Comparison of Out-of-Sample Predictive and Economic Performance across Technical-Only Models (Test Period: June 3, 2026 -- September 17, 2026; $N = 638$ 4-Hour Bars)}
\label{tab:technical_benchmark}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccccc@{}}
\toprule
\textbf{Model Architecture} & \textbf{DA (\%)} & \textbf{Macro F1} & \textbf{ROC-AUC} & \textbf{MCC} & \textbf{Brier} & \textbf{Cum. Ret (\%)} & \textbf{Sharpe} & \textbf{MDD (\%)} & \textbf{Trades} & \textbf{Friction (\%)} \\
\midrule
Naive (Buy \& Hold)          & 50.63 & 0.3361 & 0.5000 & 0.0000 & 0.2500 & \textbf{+33.57} & \textbf{2.14} & \textbf{-17.73} & 1   & 0.10 \\
Logistic Regression (Ridge)  & 50.31 & 0.5031 & 0.5011 & 0.0063 & 0.2554 & -24.87          & -2.74          & -26.13          & 272 & 27.20 \\
Random Forest ($N=100, D=5$) & \textbf{53.13} & \textbf{0.5308} & \textbf{0.5492} & \textbf{0.0621} & \textbf{0.2485} & +1.88  & 0.36  & -21.41          & 192 & 19.20 \\
Proposed 2-Layer LSTM        & 51.57 & 0.4743 & 0.5339 & 0.0293 & 0.2494 & \textbf{+13.73} & \textbf{1.20} & -24.34          & \textbf{39}  & \textbf{3.90} \\
\bottomrule
\end{tabular*}
\vspace{1ex}
\raggedright
\footnotesize{\textit{Notes:} DA denotes Directional Accuracy; MCC denotes Matthews Correlation Coefficient; Brier denotes Brier Probability Loss Score; Cum. Ret represents net compounded portfolio return after deducting 10 bps transaction fee per turnover trade; Sharpe represents the annualized Sharpe Ratio computed using $\sqrt{2,190}$ for 4-hour cryptocurrency intervals; MDD represents the Maximum Drawdown.}
\end{table*}
```

---

## 3. Pembahasan Hasil Empiris (Draft Naskah Bagian "Results & Discussion")

### A. Jebakan Frekuensi Transaksi Tinggi (*The Turnover / Friction Trap*)
Temuan empiris membuktikan anomali klasik dalam literatur komputasi finansial: **keunggulan akurasi statistik tidak serta merta bertransformasi menjadi keuntungan ekonomi riil**. Model Random Forest berhasil memenangkan seluruh metrik klasifikasi statistik pada data uji (DA = 53.13%, ROC-AUC = 0.5492, MCC = 0.0621). Namun, karena sensitivitasnya yang tinggi terhadap fluktuasi acak bar 4-jam (*microstructure noise*), Random Forest menghasilkan 192 kali pergantian sinyal posisi. Beban komisi transaksi kumulatif mencapai 19.20%, sehingga memangkas profit kotor dan menyisakan *Net Return* tipis sebesar +1.88% (Sharpe Ratio 0.36).

Kondisi yang jauh lebih fatal dialami oleh Logistic Regression (Ridge). Model linear ini terombang-ambing (*whipsawed*) oleh noise harga dengan frekuensi transaksi mencapai 272 kali, menyebabkan akumulasi biaya komisi sebesar 27.20% yang menyeret performa portofolio ke jurang kerugian drastis sebesar -24.87% dengan Sharpe Ratio negatif (-2.74).

### B. Keunggulan Mekanisme Memori Temporal (*Temporal Noise Filtering*)
Berbeda dengan model tabular non-sekuensial, arsitektur Proposed 2-Layer LSTM (dengan integrasi *Dropout* dan *Recurrent Dropout*) membuktikan efektivitasnya sebagai *temporal filter*. Struktur *gating mechanism* (forget, input, and output gates) pada LSTM mampu membedakan anomali volatilitas jangka pendek dari persistensi tren multi-periode ($W = 18$ bar / 72 jam). 

Hasilnya, LSTM hanya melakukan 39 kali pergantian posisi (beban friksi hanya 3.90%), secara disiplin menghindari *whipsaw*, dan sukses membukukan *Cumulative Return* sebesar **+13.73%** dengan **Annualized Sharpe Ratio 1.20**.

### C. Keterbatasan Fitur Teknikal Murni & Justifikasi Teoretis Model Hibrida (*The Need for Sentiment*)
Meskipun LSTM mengungguli seluruh model *machine learning* lainnya, performanya masih tertinggal dibandingkan strategi Naive Buy-and-Hold (+33.57%, Sharpe 2.14). Analisis kurva ekuitas menunjukkan bahwa *lag* indikator teknikal (seperti RSI yang mengalami saturasi pada zona *overbought*) memaksa model LSTM mengambil sikap defensif (beralih ke posisi *Cash*) tepat ketika aset Ethereum memasuki fase reli momentum kuat pada akhir Agustus 2026 (*cash drag phenomenon*).

Temuan ini memberikan fondasi teoretis dan justifikasi ilmiah yang sangat kokoh untuk penelitian ini: **indikator teknikal bersifat reaktif (*lagging*) dan tidak memiliki visibilitas terhadap katalis naratif fundamental pasar**. Oleh karena itu, integrasi variabel sentimen berita keuangan terdiskon secara eksponensial (*EMA Sentiment Decay*) diperlukan untuk menyuplai sinyal *forward-looking* yang dapat menahan posisi *Long* saat reli fundamental berlangsung.
