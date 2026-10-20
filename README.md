# UTS Machine Learning — Linear Regression & Logistic Regression

## Identitas Mahasiswa

- **Nama:** Muhamad Reyvan Lukni Maolana
- **NIM:** 241011400610
- **Kelas:** Reg B
- **Random state:** `10`

## Tujuan

Tugas ini menerapkan Linear Regression untuk memperkirakan penjualan berdasarkan biaya iklan dan Logistic Regression untuk mengklasifikasikan email sebagai spam atau bukan spam. Model utama, pembagian data, serta metrik evaluasi diimplementasikan sendiri menggunakan NumPy sesuai instruksi tugas. scikit-learn hanya digunakan pada bagian bonus sebagai pembanding.

## Dataset

- **`Ads_Data.csv`** — 40 baris, kolom `BiayaIklan` (juta rupiah) dan `Penjualan` (juta rupiah).
- **`Email_Data.csv`** — 60 baris, kolom `JumlahLink`, `KataPromo`, dan `Spam` (1 = spam, 0 = bukan spam).

Pembagian data menggunakan `random_state=10`. Untuk regresi, 32 baris digunakan untuk training dan 8 untuk testing. Untuk klasifikasi, 45 baris digunakan untuk training dan 15 untuk testing.

## Bagian A — Linear Regression

Model dilatih menggunakan **Normal Equation**. Persamaan yang diperoleh:

\[
\hat{y} = 29.3864 + 9.0711 \times \text{BiayaIklan}
\]

Artinya, dalam rentang dan pola data yang diamati, kenaikan biaya iklan sebesar Rp1 juta berkaitan dengan kenaikan prediksi penjualan sekitar Rp9,07 juta. Ini adalah hubungan model pada dataset, bukan bukti bahwa biaya iklan merupakan satu-satunya penyebab perubahan penjualan.

### Evaluasi model

| Data | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Training | 3.7920 | 24.4404 | 4.9437 | 0.9618 |
| Testing | 7.2023 | 87.1316 | 9.3344 | 0.8464 |

RMSE testing sekitar **9,33 juta rupiah**, sehingga besar kesalahan prediksi tipikal masih perlu diperhatikan. Nilai R² testing sekitar **0,8464** menunjukkan model menjelaskan sekitar 84,64% variasi penjualan pada bagian testing ini. Hasil testing lebih buruk daripada training, sehingga performa pada data baru tidak dapat diasumsikan sama baiknya.

### Eksperimen Gradient Descent

Gradient Descent dijalankan selama 5.000 iterasi dengan tiga learning rate:

| Learning rate | Intercept akhir | Koefisien akhir | MSE akhir | Status |
|---:|---:|---:|---:|---|
| 0.001 | 20.150382 | 10.484470 | 43.398146 | Selesai, belum sedekat hasil Normal Equation |
| 0.02 | 29.386389 | 9.071079 | 24.440353 | Konvergen dekat Normal Equation |
| 0.1 | Tidak finite | Tidak finite | Tidak finite | Divergen |

Learning rate `0.001` bergerak lebih lambat dan belum mencapai solusi sebaik `0.02` dalam 5.000 iterasi. Learning rate `0.02` mendekati hasil Normal Equation. Learning rate `0.1` terlalu besar untuk skala fitur ini dan mengalami divergensi; hasilnya tidak diperlakukan sebagai model terlatih yang valid.

Prediksi untuk biaya iklan 12 juta rupiah merupakan ekstrapolasi karena nilai tersebut di luar rentang biaya iklan dataset (maksimum 9,9 juta), sehingga lebih tidak pasti dibanding prediksi di dalam rentang pengamatan.

## Bagian B — Logistic Regression

Model klasifikasi menggunakan fitur `JumlahLink` dan `KataPromo`, dengan sigmoid dan Gradient Descent (`learning_rate=0.1`, `n_iterations=10000`). Log loss turun dari sekitar **0,690241** menjadi **0,263808**.

### Evaluasi data testing

Confusion matrix (baris = kelas aktual; kolom = prediksi):

| | Prediksi bukan spam | Prediksi spam |
|---|---:|---:|
| Aktual bukan spam | 5 (TN) | 0 (FP) |
| Aktual spam | 1 (FN) | 9 (TP) |

| Metrik | Hasil |
|---|---:|
| Accuracy | 0.9333 (93,33%) |
| Precision | 1.0000 (100%) |
| Recall | 0.9000 (90%) |
| F1-score | 0.9474 (94,74%) |

Pada sampel testing ini, tidak ada email bukan spam yang keliru ditandai spam (FP=0), sedangkan satu email spam lolos sebagai bukan spam (FN=1). Karena jumlah data testing hanya 15, hasil tersebut perlu diuji lagi pada data yang lebih banyak sebelum digunakan sebagai filter sungguhan.

### Eksperimen threshold

Threshold 0,3; 0,5; dan 0,7 menghasilkan confusion matrix yang sama pada sampel testing: TN=5, FP=0, FN=1, TP=9. Precision tetap 1,0 dan recall 0,9. Threshold 0,3 dipilih untuk prediksi email baru dengan pertimbangan memprioritaskan penangkapan spam, tetapi pada data testing ini belum tampak perbedaan hasil antarthreshold.

## Bonus — Perbandingan dengan scikit-learn

- **Linear Regression:** koefisien, intercept, RMSE testing (9,334432), dan R² testing (0,846418) cocok dengan implementasi Normal Equation manual pada presisi yang ditampilkan.
- **Logistic Regression:** kedua implementasi menghasilkan confusion matrix testing yang sama, `[[5, 0], [1, 9]]`, dan metrik klasifikasi yang sama pada threshold 0,5 untuk dataset ini. Koefisiennya berbeda karena scikit-learn menggunakan regularisasi L2 secara default (`C=1.0`), sedangkan model manual tidak menggunakan regularisasi.

## Kesimpulan

Pada dataset ini, Linear Regression menunjukkan kecocokan yang cukup kuat tetapi kesalahan testing masih sekitar 9,33 juta rupiah dan prediksi di luar rentang data lebih tidak pasti. Logistic Regression mendapatkan akurasi testing 93,33% dengan satu false negative. Hasil ini menjanjikan untuk eksperimen pembelajaran, tetapi ukuran dataset dan sampel testing yang terbatas tidak cukup untuk menjamin performa pada data nyata.

## Struktur Repository

```text
uts-machine-learning-241011400610/
├── README.md
├── tugas_UTS.ipynb
├── Ads_Data.csv
└── Email_Data.csv
```

## Menjalankan Notebook

Lingkungan yang digunakan untuk verifikasi memiliki Python 3.13 serta paket `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `nbformat`, `nbclient`, dan `ipykernel`. Buka `tugas_UTS.ipynb` di Jupyter Notebook atau VS Code, pilih kernel Python dengan dependensi tersebut, lalu jalankan **Run All**. Dataset CSV harus berada di folder yang sama dengan notebook.
