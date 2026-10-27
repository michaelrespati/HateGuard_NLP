# Laporan Audit Metodologis, Reproducibility, dan Verifikasi Sistem — HateGuard AI

Dokumen ini mencatat hasil audit teknis dan metodologis terhadap pipeline pelatihan (`HateGuard_Training_Final.ipynb`) serta aplikasi inferensi (`HateGuard_App_Final.ipynb`) untuk menjamin integritas bukti eksperimen pada naskah publikasi.

---

## 1. Ringkasan Verifikasi Integritas & Paritas Lintas-Perangkat

| Komponen Audit | Metode Verifikasi | Hasil Aktual pada Eksekusi Final | Status |
| :--- | :--- | :--- | :---: |
| **Integritas Bobot Model** | Pemeriksaan streaming SHA-256 pada `model.safetensors` terhadap `trusted_model_sha256` di manifest. | `3ec22197cb5248be993782d490cc080efb9f2d0f341ba7884f1b7d99ff31e0df` (Cocok 100%). | **LULUS** |
| **Integritas Modul Inti (`hateguard_core.py`)** | Pemeriksaan SHA-256 modul bersama sebelum diimpor oleh Training, App, dan Smoke Test. | `e07ad4c7524b1ed3e30fa0fa5a91a9163c40ab4a0547663afc15ef7f861a91b6` (v1.0.0, Cocok 100%). | **LULUS** |
| **Paritas Preprocessing (`clean_text`)** | Pengujian hash SHA-256 pada vektor sampel uji 4 bahasa (`PREPROCESSING_PARITY_SAMPLES`). | `0ebe0f02665f2f7b49c5965ceb4ce35f17ba932da93ff528b4e691d6cc5152f8` (Identik di Training & App). | **LULUS** |
| **Paritas Inferensi GPU vs CPU (200 Sampel Berstrata)** | Membandingkan probabilitas FP32 pada aplikasi (CPU) terhadap referensi Training (GPU) pada `inference_parity_reference.csv`. | Selisih absolut maksimum = `1.27e-05` (`0.0000127`), rata-rata = `1.26e-06` (batas toleransi `1e-3`), `tier_flips_total = 0`. | **LULUS** |
| **Standalone Smoke Test (Proses Python Terpisah)** | Eksekusi `smoke_test_standalone.py` di luar memori Jupyter Notebook menggunakan CPU. | `status: PASS`, selisih maksimum = `1.55e-05`, `tier_flips_within_tolerance_zone = 0`. | **LULUS** |
| **Audit Truncation (`max_length = 128`)** | Pemeriksaan distribusi panjang token pada seluruh 40.934 baris Internal Test. | `frac_truncated = 0.0` (0% terpotong; median = 24 token, p99 = 75 token, maksimum = 118 token). | **LULUS** |
| **Self-Test Mesin Status Moderator & Database** | Pengujian headless pada database SQLite terisolasi untuk memvalidasi aturan transaksi atomik HITL. | Menolak item non-STRONG, menolak duplikasi keputusan final, dan memvalidasi kelengkapan kolom audit. | **LULUS** |

---

## 2. Perbaikan Metodologis Utama yang Diterapkan pada Versi Final

1. **Penyatuan Jalur Inferensi FP32 (`hateguard_core.logits_fp32`):**
   Evaluasi pada notebook pelatihan sebelumnya menggunakan `Trainer.predict` dengan mixed-precision (`fp16=True`), yang dapat menghasilkan pergeseran numerik tipis dibanding inferensi CPU (`float32`) di dekat ambang batas (tercatat selisih maksimum 0,0188 yang menggeser 1 label biner dan 5 tier dari 40.934 baris). Pada versi final ini, seluruh evaluasi dan aplikasi menggunakan satu fungsi inferensi FP32 tunggal tanpa autocast.
2. **Pelaporan Ganda (*Dual-Reporting*) & Resampling Berbasis Grup (`group_id`):**
   Untuk mencegah bias optimisme akibat tingginya proporsi teks augmentasi (90,5% baris test), metrik dilaporkan secara berdampingan untuk subset teks asli (`ORIGINAL_ONLY`, N = 3.907), seluruh baris (`ALL_TEST row-weighted`), dan bobot setara per grup (`ALL_TEST group-weighted`), dengan 95% Confidence Interval dari 1.000 grouped bootstrap resamples.
3. **Pembanding Klasik & Evaluasi LLM Berbasis Logit (N = 400):**
   Menambahkan baseline `TF-IDF + Logistic Regression` (C = 2,0) pada split identik serta memperluas evaluasi berpasangan melawan `Qwen/Qwen2.5-1.5B-Instruct` menjadi 400 grup independen menggunakan skor selisih logit token pertama dan kalibrasi ambang pada validation set terpisah (N = 200).
4. **Analisis Kalibrasi Probabilitas & Ablasi Kebijakan Guardrail:**
   Menyertakan evaluasi Expected Calibration Error (ECE), Brier Score, Reliability Diagram, pemasangan Temperature Scaling (T = 2,8091 pada validation set), perbandingan 6 kebijakan moderasi (Kebijakan A s.d. F), serta proyeksi beban moderator pada skenario prevalensi rendah (1%, 5%, 10%).
5. **Keamanan Deserialisasi & Akuntabilitas Log Aplikasi:**
   Aplikasi memverifikasi hash SHA-256 dari `preprocessing_bundle.joblib` sebelum memanggil `joblib.load()`, mendeteksi komentar panjang yang melebihi 128 token (`was_truncated`), dan mewajibkan pencatatan identitas moderator (`decided_by`) pada setiap keputusan final.

---

## 3. Catatan Provenance & Batasan Bukti yang Didokumentasikan secara Transparan

- Evaluasi dijalankan dalam mode `verify_frozen` pada checkpoint hasil pelatihan yang telah dibekukan (`TRAINED_IN_THIS_RUN = False`). Himpunan teks dan label pada Internal Test (40.934 baris) dan Official Held-Out Test (1.995 baris) terverifikasi 100% identik dengan CSV prediksi dari proses pelatihan asli, dan parameter pelatihan terverifikasi cocok melalui `config.json`.
- Manifest dari sesi pelatihan awal tidak menyimpan objek `training_config` secara terpisah dan memiliki perbedaan skema pengurutan string fingerprint untuk partisi train/validation; fakta ini dicatat secara eksplisit pada `experiment_manifest_addendum.json` dan bagian *Limitations* artikel ilmiah.
- Ablasi normalisasi (V0–V3) dilakukan pada waktu inferensi menggunakan bobot model beku, dan pelatihan hanya menggunakan satu random seed (`seed = 42`).