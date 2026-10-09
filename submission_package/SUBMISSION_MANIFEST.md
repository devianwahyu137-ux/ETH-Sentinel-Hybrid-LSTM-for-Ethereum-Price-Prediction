# SUBMISSION MANIFEST & OJS METADATA CHECKLIST
**Target Journal:** *Journal of Accounting and Investment* (JAI), Universitas Muhammadiyah Yogyakarta (UMY)  
**Accreditation:** SINTA 2 Accredited / Indexed in Dimensions, DOAJ, Google Scholar, SINTA  
**Article Type:** Empirical Research Article (Computational Finance & Risk Management)  
**Submission Package Assembly Date:** September 19, 2026  

---

## A. METADATA ARTIKEL (ARTICLE METADATA)

### 1. Judul Lengkap (Full Manuscript Title)
> **Downside Risk Mitigation in Cryptocurrency Markets: A Dual-Stream Gated Hybrid LSTM Framework under Market Frictions**

### 2. Running Title / Judul Singkat
> **Downside Risk Mitigation in Cryptocurrency via Dual-Stream Hybrid LSTM**

### 3. Abstrak Terstruktur (Structured Abstract — 214 Words)
**Objective:** This study investigates downside tail-risk mitigation and transaction friction erosion in algorithmic cryptocurrency portfolio management under market microstructure frictions. High-frequency models often succumb to the "Turnover Trap," where excessive trading commissions erode net predictive returns.  
**Method:** We propose the *ETH-Sentinel Hybrid LSTM*, a dual-stream architecture decoupling high-frequency price dynamics from behavioral sentiment (Crypto Fear & Greed Index) via an adaptive Gated Cross-Modality Fusion Unit (GFU). To eliminate test-set snooping, a confidence dead-band hysteresis policy ($\delta \in [0.00, 0.10]$) is calibrated exclusively on validation data ($\delta^*_{\text{val}} = 0.01$) before single-pass evaluation on out-of-sample Ethereum test bars (fee: 10 bps).  
**Results:** In an unhedged execution regime ($\delta = 0.00$), sentiment fusion functions as an asymmetric crash cushion, compressing Maximum Drawdown from -24.42% (Technical LSTM) to -18.82% (a 5.60 percentage points tail-risk reduction). Under active validation-selected hysteresis ($\delta^* = 0.01$), decision inertia shifts sentiment's marginal role toward turnover suppression: slashing trading frequency by 60.0% (40 to 16 trades), preserving net alpha with a Compounded Return of +15.99%, an Annualized Sharpe Ratio of 1.42, and an MDD of -18.19%, outperforming the technical baseline (+15.18%, Sharpe 1.29).  
**Implications/Originality:** This study provides a verifiable selective-abstention framework for digital asset corporate treasury management, demonstrating that behavioral sentiment mitigates catastrophic crashes under unhedged regimes and curbs commission drag under hysteresis execution.

### 4. Kata Kunci (Keywords)
`Cryptocurrency; Dual-Stream LSTM; Sentiment Analysis; Downside Risk; Hysteresis Policy; Market Frictions.`

### 5. Klasifikasi JEL (JEL Classification Codes)
`C45; C58; G11; G14.`  
*(C45: Neural Networks; C58: Financial Econometrics; G11: Portfolio Choice / Investment Decisions; G14: Information & Market Efficiency)*

---

## B. DATA PENULIS & AFILIASI (AUTHOR INFORMATION — UNBLINDED)
*(Sesuai berkas 03_Title_Page_Unblinded.docx / .pdf untuk Dewan Editor)*

- **First / Corresponding Author:** [FILL IN: Full Name]
  - **Affiliation:** [FILL IN: Institutional Affiliation / Department / Faculty / University]
  - **City & Country:** [FILL IN: City, Country]
  - **Email:** [FILL IN: Official Institutional Email]
  - **ORCID ID:** [FILL IN: https://orcid.org/XXXX-XXXX-XXXX-XXXX]
- **Co-Author(s):** [FILL IN: Additional Co-Authors as appropriate]
- **Author Contribution (CRediT Taxonomy):**
  - Conceptualization, Methodology, Software, Formal Analysis, Data Curation, Writing – Original Draft, Writing – Review & Editing, Supervision.

---

## C. PERNYATAAN RESMI PENULIS (AUTHOR DECLARATIONS)

1. **Orisinalitas & Bebas Plagiarisme:**
   Naskah ini adalah karya orisinal, belum pernah dipublikasikan di jurnal ilmiah manapun, dan tidak sedang dalam proses pertimbangan/tinjauan di jurnal lain secara simultan.
2. **Ketiadaan Konflik Kepentingan (Conflict of Interest):**
   Para penulis menyatakan tidak memiliki afiliasi keuangan, komersial, atau hubungan personal dengan bursa kripto, penerbit token, atau platform analitik pihak ketiga yang dapat memengaruhi objektivitas penelitian ini.
3. **Pernyataan Pendanaan (Funding Statement):**
   Riset ini bersifat mandiri (*independent and self-funded*) tanpa menerima hibah atau sponsor komersial pihak ketiga.
4. **Kepatuhan Double-Blind Peer Review:**
   Seluruh identitas penulis, nama kampus, dan kontak email telah disanitasi secara ketat dari naskah utama (`01_Main_Manuscript_Blind.docx` dan `.pdf`).
5. **Ketersediaan Data & Kode (Data Availability Statement):**
   Seluruh dataset 4H Ethereum, rekam historis Crypto Fear & Greed Index, tensor sekuens preprocessed, model weights `.keras`, dan skrip eksekusi didepositkan secara publik pada repositori resmi:  
   👉 **https://github.com/devianwahyu/ETH-Sentinel-Hybrid-LSTM-for-Ethereum-Price-Prediction**

---

## D. PANDUAN PEMETAAN UNGGAH OJS JAI UMY (STEP-BY-STEP OJS UPLOAD GUIDE)

Saat mengunggah ke portal Open Journal Systems (OJS) JAI UMY (*journal.umy.ac.id/index.php/ai*):

| Langkah OJS | Slot Unggah Sistem | Nama Berkas yang Dipilih | Deskripsi untuk Editor |
|:---|:---|:---|:---|
| **Step 2 (Upload Submission)** | **Submission File** | `01_Main_Manuscript_Blind.docx` *(atau .pdf)* | Article Text (Double-Blind Anonymized Master Paper) |
| **Step 2 (Upload Library/Supplementary)** | Supplementary File #1 | `02_Supplementary_Appendix.pdf` *(atau .docx)* | Supplementary File: Technical Appendices A–E (Proofs, Pseudocode, ADF, Bootstrap) |
| **Step 2 (Upload Library/Supplementary)** | Supplementary File #2 | `03_Title_Page_Unblinded.pdf` *(atau .docx)* | Title Page with Complete Author Details, Affiliations, and Official GitHub Repository Link |
| **Step 2 (Upload Library/Supplementary)** | Supplementary File #3 | `04_Cover_Letter.pdf` *(atau .docx)* | Official Submission Cover Letter to Editor-in-Chief JAI UMY |
| **Step 2 (Upload Library/Supplementary)** | Supplementary File #4 | `05_Peer_Review_Audit_Trail.pdf` *(atau .docx)* | Internal Peer-Review Audit Trail & Methodological Remediation Log (Rounds 1–4) |
| **Step 3 (Enter Metadata)** | Form Metadata OJS | Salin dari Bagian A & B di atas | Judul, Running Title, Abstrak Terstruktur, Keywords, JEL Codes, Data Penulis |
| **Step 4 (Confirmation)** | Finish Submission | Review Final Check | Klik Submit untuk mengirim paket naskah resmi ke JAI UMY |

---
*Verified and Certified Submission Package — Ready for Editorial Processing.*
