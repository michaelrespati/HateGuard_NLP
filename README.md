# HateGuard AI: Multi-Tiered Guardrail & Human-in-the-Loop Moderation for Indonesian Code-Mixed Social Media

Repositori resmi untuk artefak penelitian dan sistem perangkat lunak **HateGuard AI** oleh **Kelompok 2 — Program Studi Teknik Informatika, Fakultas Teknologi Informasi, Universitas Tarumanagara**:
- **Michael Respati Sanjaya Ho** (535250008)
- **Felix Lin** (535250044)
- **Charlie Jhodya Soe** (535250180)

Dokumentasi ini menjelaskan spesifikasi environment, akses dataset, protokol training dan evaluation, mekanisme inference, serta pelaksanaan smoke test dari environment bersih. Seluruh angka evaluasi yang dilaporkan pada naskah artikel ilmiah mengacu pada satu sumber tunggal di `results/RESULTS_SINGLE_SOURCE.json`.

---

## 1. Struktur Repositori

```text
HateGuard-Repository/
├── source/          # Notebook pelatihan, notebook aplikasi Gradio, modul inti (hateguard_core.py), & skrip smoke test
├── config/          # Konfigurasi eksperimen (config.json), manifest utama, manifest addendum, & requirements_frozen.txt
├── results/         # Hasil evaluasi utama (RESULTS_SINGLE_SOURCE.json), tabel CSV, & subfolder figures/ (300 DPI)
├── paper/           # Naskah artikel ilmiah dalam format Jurnal RESTI
├── DATA_CARD.md     # Dokumentasi provenance, komposisi partisi, dan batasan etika dataset
├── Audit_Report.md  # Laporan audit metodologis, verifikasi paritas lintas-perangkat, & integritas artefak
├── LICENSE          # Lisensi perangkat lunak (MIT License)
└── .gitignore       # Aturan pengecualian file bobot berukuran besar dan log privasi moderator
```

---

## 2. Spesifikasi Environment & Instalasi

Eksperimen dieksekusi pada lingkungan Google Colab GPU (NVIDIA Tesla T4, CUDA 13.0) dengan Python 3.13.15, PyTorch 2.11.0, Transformers 5.20.1, Datasets 4.8.5, dan scikit-learn 1.6.1. Seluruh generator angka acak dikunci pada `seed = 42` (`PYTHONHASHSEED`, `random`, `numpy`, `torch`, cuDNN deterministik, dan `CUBLAS_WORKSPACE_CONFIG=:4096:8`).

Untuk menyiapkan environment bersih di mesin lokal atau server baru, jalankan perintah berikut di terminal:

```bash
pip install -r config/requirements_frozen.txt
```

---

## 3. Akses Dataset & Provenance

Sistem ini menggunakan gabungan dua sumber korpus publik:
1. **Indonesian Twitter Hate Speech Dataset** (Ibrohim & Budi, 2019) beserta kamus normalisasi bahasa alay (`new_kamusalay.csv`, 15.195 entri). Nilai SHA-256 file mentah tercatat pada `config/experiment_manifest_addendum.json`.
2. **Code-Mixed Indonesian Hate Speech Dataset** (`aiatums/codemixed-id-hate-speech`, konfigurasi *Indonesian-Javanese* dan *Indonesian-Sundanese*, lisensi CC BY 4.0, commit `19a03eba294171be64a5fcb5dcf7f36b6e86d72f`).

Rincian pembersihan label kontradiktif, deduplikasi, dan pembagian partisi bebas bocor (`GroupShuffleSplit` berbasis `group_id`) didokumentasikan secara lengkap pada `DATA_CARD.md`.

---

## 4. Akses Artefak Checkpoint Beku & Verifikasi Checksum SHA-256

Sesuai praktik terbaik pengelolaan repositori Git, file bobot model berukuran besar (`model.safetensors`, ~440 MB) tidak disimpan langsung di dalam Git, melainkan dihosting pada penyimpanan awan dan diverifikasi menggunakan checksum SHA-256:

- **Tautan Google Drive Artefak (`hateguard_artifacts`)**: `https://drive.google.com/drive/folders/1oKxbmulcBvUbaBUtoGvRC2hwZqyku0vQ?usp=drive_link`
- **Model Dasar**: `indolem/indobertweet-base-uncased` (Apache-2.0)
- **SHA-256 Bobot Model (`trusted_model_sha256`)**: `3ec22197cb5248be993782d490cc080efb9f2d0f341ba7884f1b7d99ff31e0df`
- **SHA-256 Modul Inti (`hateguard_core.py`)**: `e07ad4c7524b1ed3e30fa0fa5a91a9163c40ab4a0547663afc15ef7f861a91b6`
- **SHA-256 Paritas Preprocessing (`preprocessing_parity_hash`)**: `0ebe0f02665f2f7b49c5965ceb4ce35f17ba932da93ff528b4e691d6cc5152f8`

---

## 5. Prosedur Training, Verifikasi, dan Evaluasi

Buka `source/HateGuard_Training_Final.ipynb` dan pilih mode eksekusi melalui variabel `RUN_MODE` pada Sel 1:
- `RUN_MODE = "verify_frozen"` *(Default)*: Memverifikasi keaslian checksum bobot model beku, lalu menjalankan seluruh evaluasi statistik (1.000 grouped bootstrap resamples dalam presisi FP32), pembanding klasik (TF-IDF + Logistic Regression), komparasi Zero-Shot LLM (`Qwen/Qwen2.5-1.5B-Instruct` pada N = 400 grup independen), studi ablasi kebijakan guardrail & normalisasi, kalibrasi suhu (Temperature Scaling), serta pembuatan grafik publikasi tanpa menimpa bobot model asli.
- `RUN_MODE = "retrain_new_dir"`: Melakukan fine-tuning ulang dari awal dan menyimpan hasilnya ke direktori terpisah (`./hateguard_artifacts_retrain/`).

---

## 6. Menjalankan Aplikasi Inferensi (Gradio HITL) & Smoke Test

### A. Smoke Test dari Environment Bersih (Terminal Mandiri)
Unduh folder `hateguard_artifacts` dari tautan Google Drive di atas, lalu jalankan perintah berikut dari terminal di luar Jupyter Notebook:

```bash
python source/smoke_test_standalone.py ./hateguard_artifacts cpu
```

Skrip akan memverifikasi seluruh checksum SHA-256, menguji determinisme prediksi, dan memeriksa paritas probabilitas FP32 pada 200 sampel acuan berstrata (batas toleransi <= 0.001). Kode keluar `0` (`SMOKE TEST LULUS`) menandakan artefak valid sepenuhnya.

### B. Aplikasi Web Interaktif (Gradio & SQLite Audit Dashboard)
Jalankan seluruh sel pada `source/HateGuard_App_Final.ipynb`. Aplikasi menerapkan prinsip *Hard Fail* saat startup apabila terdapat ketidaksesuaian hash artefak, memuat asisten `Qwen/Qwen2.5-1.5B-Instruct` secara *lazy loading* khusus untuk antrean `STRONG`, dan mencatat setiap keputusan moderator ke dalam database SQLite (`moderator_audit.db`).

---

## 7. Keamanan Data Log & Etika

File log operasional aplikasi (`moderator_audit.db` dan `moderator_log.json`) menyimpan teks komentar yang diproses dan dikecualikan dari repositori publik melalui `.gitignore`. Sistem ini dirancang sebagai pendukung keputusan (*decision-support*) berbasis *Human-in-the-Loop* dan bukan pengganti penilaian moderator manusia secara otonom.