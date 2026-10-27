# Data Card — Korpus Moderasi Ujaran Kebencian Code-Mixed HateGuard

Dokumen ini merangkum asal-usul (*provenance*), komposisi partisi, protokol pembersihan, serta batasan dan risiko etis dari korpus yang digunakan dalam penelitian **HateGuard AI**.

---

## 1. Ringkasan Spesifikasi Korpus

| Komponen | Deskripsi & Protokol |
| :--- | :--- |
| **Tugas Utama** | Klasifikasi biner ujaran kebencian (*Hate Speech* = `1` vs *Non-Hate Speech* = `0`) pada teks media sosial Indonesia bercampur bahasa (*code-mixed*). |
| **Sumber Data 1** | Korpus Twitter Indonesia Ibrohim & Budi (2019) (`re_dataset.csv`, SHA-256: `44c04e31ad4b7ee4a95f1884e7af4da2c44b69762143eb2de0ede7f90502735e`) dan kamus bahasa alay (`new_kamusalay.csv`, SHA-256: `fe00f0c728ee8ecc40dca3c3001ff2701dba7d4417a4178fe5b5745243564f9d`). Setiap tweet unik merepresentasikan 1 grup independen (`group_id`). |
| **Sumber Data 2** | Dataset Hugging Face `aiatums/codemixed-id-hate-speech` (konfigurasi *Indonesian-Javanese* dan *Indonesian-Sundanese*, lisensi CC BY 4.0, commit `19a03eba294171be64a5fcb5dcf7f36b6e86d72f`). Setiap rekaman memuat 1 teks asli (`original`) dan 2 variasi augmentasi (`augmented_1`, `augmented_2`) yang diikat ke dalam 1 `group_id` yang sama. |
| **Prosedur Pembersihan** | Dari 283.887 baris gabungan mentah: (1) penghapusan teks kosong, (2) penghapusan 954 teks unik yang memiliki label kontradiktif antar-baris (11.503 baris dihapus), dan (3) deduplikasi berdasarkan `text_clean` (72.648 duplikat dihapus), menghasilkan **199.360 baris bersih** (111.761 Non-HS [56,06%] dan 87.599 HS [43,94%]). |
| **Protokol Partisi Internal** | Pembagian bebas kebocoran menggunakan `GroupShuffleSplit` (`seed = 42`) berdasarkan `group_id`: **Train = 142.647 baris**, **Validation = 15.779 baris**, dan **Internal Test = 40.934 baris (4.100 grup unik; terdiri dari 3.907 teks original dan 37.027 teks augmentasi)**. |
| **Official Held-Out Test** | Diambil dari split `test` resmi dataset Jawa dan Sunda (hanya teks `original`), disaring dari teks kosong, duplikat, dan overlap terhadap seluruh partisi internal, menghasilkan **1.995 teks original** (996 Jawa dan 999 Sunda). |
| **Normalisasi & Masking** | Normalisasi input model menggabungkan kamus alay (15.195 entri) dan leksikon manual 4 bahasa (Indonesia, Inggris gaul, Jawa, Sunda). Penyensoran tampilan publik (*lexical masking*) menggunakan daftar terpisah berisi 47 kata sensitif. |

---

## 2. Rincian Komposisi Partisi dan Prevalensi Kelas

| Partisi | Domain Sumber | Jumlah Baris | Jumlah Grup Unik (`group_id`) | Rasio Baris/Grup | Prevalensi HS (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Train** | `codemixed_javanese` (Aug) | 64.007 | 2.678 | 23,90 | 62,39% |
| **Train** | `codemixed_javanese_orig` | 2.677 | 2.677 | 1,00 | 62,38% |
| **Train** | `codemixed_sundanese` (Aug) | 64.565 | 2.698 | 23,93 | 26,30% |
| **Train** | `codemixed_sundanese_orig` | 2.016 | 2.016 | 1,00 | 18,15% |
| **Train** | `indonesian_twitter` (Orig) | 9.382 | 9.382 | 1,00 | 42,02% |
| **Validation** | `codemixed_javanese` (Aug) | 6.833 | 285 | 23,98 | 55,55% |
| **Validation** | `codemixed_javanese_orig` | 285 | 285 | 1,00 | 55,44% |
| **Validation** | `codemixed_sundanese` (Aug) | 7.384 | 309 | 23,90 | 22,85% |
| **Validation** | `codemixed_sundanese_orig` | 231 | 231 | 1,00 | 17,32% |
| **Validation** | `indonesian_twitter` (Orig) | 1.046 | 1.046 | 1,00 | 42,93% |
| **Internal Test** | `codemixed_javanese` (Aug) | 18.965 | 793 | 23,92 | 61,71% |
| **Internal Test** | `codemixed_javanese_orig` | 792 | 792 | 1,00 | 61,74% |
| **Internal Test** | `codemixed_sundanese` (Aug) | 18.062 | 754 | 23,95 | 28,61% |
| **Internal Test** | `codemixed_sundanese_orig` | 561 | 561 | 1,00 | 20,50% |
| **Internal Test** | `indonesian_twitter` (Orig) | 2.554 | 2.554 | 1,00 | 42,99% |
| **Official Held-Out** | `official_test_jv_orig` | 996 | 996 | 1,00 | 60,54% |
| **Official Held-Out** | `official_test_su_orig` | 999 | 999 | 1,00 | 34,53% |

---

## 3. Keterbatasan & Risiko Etis (*Limitations*)

1. **Rasio Teks Augmentasi terhadap Unit Independen:** Pada partisi *Internal Test* (40.934 baris), sebanyak 90,5% baris (37.027 baris) merupakan variasi augmentasi yang berasal dari 1.546 grup *code-mixed* (~23,9 baris per grup). Oleh sebab itu, evaluasi wajib menggunakan *grouped bootstrap* pada level `group_id` dan melaporkan kinerja pada subset teks asli (`ORIGINAL_ONLY`, N = 3.907) sebagai acuan utama.
2. **Asumsi Label Augmentasi:** Variasi augmentasi diasumsikan mempertahankan label dari teks aslinya (*label-preserving*) sebagaimana disediakan oleh dataset sumber dan belum diverifikasi ulang secara manual oleh anotator manusia.
3. **Perbedaan Prevalensi Kelas Antar-Domain:** Prevalensi kelas *Hate Speech* pada korpus penelitian ini (~43,9%–45,4%, dengan Jawa ~61,7% dan Sunda original ~20,5%) jauh lebih tinggi dibandingkan trafik media sosial dunia nyata pada umumnya.
4. **Cakupan Bahasa & Anotasi:** Korpus hanya mencakup variasi Bahasa Indonesia, Inggris gaul, Jawa, dan Sunda dari sumber yang digunakan, sehingga tidak secara otomatis mewakili dialek daerah lain atau platform media sosial yang berbeda.
5. **Penanganan Konten Sensitif:** Karena dataset mengandung contoh ujaran kebencian dan kata kasar nyata, contoh kualitatif yang ditampilkan pada publikasi dan antarmuka telah melalui penyensoran leksikal (*masked*).