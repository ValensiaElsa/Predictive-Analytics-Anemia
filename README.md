# Laporan Proyek Machine Learning - Valensia Elsa Kurnia

## Domain Proyek
Anemia adalah gangguan kesehatan yang umum di seluruh dunia, yang ditandai dengan penurunan jumlah sel darah merah atau kadar hemoglobin dalam darah. Kondisi ini dapat menyebabkan berkurangnya kemampuan darah untuk mengangkut oksigen ke seluruh tubuh, yang berpotensi mengurangi kualitas hidup dan meningkatkan risiko komplikasi medis serius, seperti penyakit jantung.  Kondisi ini sering kali disebabkan oleh kekurangan zat besi, defisiensi vitamin B12, atau masalah kesehatan lainnya. Berdasarkan data terbaru dari Organisasi Kesehatan Dunia (WHO), pada tahun 2023, sekitar 30,7% wanita usia 15–49 tahun mengalami anemia, dengan 35,5% di antaranya adalah wanita hamil. Selain itu, pada tahun 2019, prevalensi anemia pada anak-anak usia 6–59 bulan mencapai 39,8% secara global[[1]](https://www.who.int/data/gho/data/themes/topics/anaemia_in_women_and_children). Penyakit anemia biasanya terdeteksi melalui pemeriksaan medis yang memerlukan tes laboratorium untuk mengukur berbagai parameter darah, seperti kadar hemoglobin dan jumlah sel darah merah. Namun, prosedur ini dapat memakan waktu dan biaya. Selain itu, diagnosis sering kali terlambat, yang mengarah pada keterlambatan pengobatan dan peningkatan risiko komplikasi. 

Masalah ini dapat diselesaikan dengan menerapkan machine learning dalam bentuk model prediktif untuk mendiagnosis anemia lebih cepat dan lebih akurat. Dengan memanfaatkan data medis seperti jumlah sel darah merah, kadar hemoglobin, dan ukuran rata-rata sel darah merah (MCV), model machine learning dapat memberikan diagnosis dini yang lebih efisien [[2]](https://www.researchgate.net/publication/368845592_PREDICTION_OF_ANEMIA_USING_MACHINE_LEARNING_ALGORITHMS). Proyek ini menggunakan algoritma klasifikasi untuk menganalisis pola dalam data pasien dan memprediksi kemungkinan seseorang menderita anemia. Hal ini tidak hanya mempercepat proses diagnosis tetapi juga memungkinkan intervensi lebih cepat, yang dapat mencegah komplikasi lebih lanjut. 

Dengan penggunaan machine learning dalam analisis prediktif, kita dapat memanfaatkan data yang sudah ada untuk memberikan prediksi yang lebih presisi, lebih cepat, dan lebih terjangkau, yang pada gilirannya dapat memperbaiki manajemen kesehatan masyarakat. Ini juga memungkinkan tenaga medis untuk mengidentifikasi pasien yang membutuhkan perhatian segera, tanpa harus menunggu hasil tes laboratorium yang memakan waktu. Dengan demikian, proyek ini bertujuan untuk mengatasi masalah keterlambatan diagnosis dan akses terbatas ke fasilitas medis, yang dapat diselesaikan dengan solusi berbasis teknologi yang efisien dan lebih mudah diakses.

Referensi:

[[1] World Health Organization, "Anaemia in women and children," Global Health Observatory Data Repository, 2023. [Online]. Available: https://www.who.int/data/gho/data/themes/topics/anaemia_in_women_and_children. [Accessed: May 6, 2025].](https://www.who.int/data/gho/data/themes/topics/anaemia_in_women_and_children)

[[2] K. L. Seerangan, A. R. K. Saravanan, and P. K. R. S. Anandan, "Machine learning for prediction of anemia using laboratory data," Journal of Medical Systems, vol. 42, no. 5, 2018.](https://www.researchgate.net/publication/368845592_PREDICTION_OF_ANEMIA_USING_MACHINE_LEARNING_ALGORITHMS)

## Business Understanding

### Problem Statements
Berdasarkan uraian yang telah dipaparkan pada latar belakang diatas, maka dapat diambil sebuah rumusan masalah yang dirumuskan sebagai berikut:
- Bagaimana cara memprediksi apakah seseorang menderita anemia hanya dengan menggunakan data medis dasar seperti kadar hemoglobin, MCV (Mean Corpuscular Volume), MCH (Mean Corpuscular Hemoglobin), dan MCHC (Mean Corpuscular Hemoglobin Concentration), tanpa memerlukan tes laboratorium yang mahal dan memakan waktu?
- Bagaimana meningkatkan akurasi model prediktif untuk mendeteksi anemia dengan menggunakan berbagai algoritma machine learning, serta melakukan optimasi model untuk mendapatkan hasil yang lebih baik?
- Bagaimana cara membandingkan performa beberapa algoritma machine learning yang berbeda dalam mendeteksi anemia, dan memilih model terbaik berdasarkan metrik evaluasi yang relevan?

### Goals
Berdasarkan rumusan masalah yang telah dipaparkan di atas, maka proyek penelitian ini memiliki tujuan, yaitu:
- Membangun model prediksi berbasis machine learning yang dapat memberikan diagnosis dini anemia dengan data medis dasar, seperti jumlah sel darah merah, kadar hemoglobin, MCV, MCH, dan MCHC.
- Meningkatkan kinerja model prediktif dengan menggunakan beberapa algoritma machine learning, seperti Logistic Regression, Decision Trees, dan Random Forest, serta melakukan hyperparameter tuning untuk memperoleh hasil yang lebih optimal.
- Menerapkan metrik evaluasi yang tepat, seperti akurasi, precision, recall, dan F1-score, untuk memilih model terbaik yang memberikan keseimbangan optimal antara prediksi yang akurat dan kemampuan untuk mendeteksi kasus anemia secara efektif.

### Solution statements
Berdasarkan tujuan yang telah dipaparkan diatas, maka proyek penelitian ini memiliki solusi atau tahapan sebagai berikut:
- Menggunakan beberapa algoritma machine learning, seperti Logistic Regression, Decision Trees, dan Random Forest, untuk membangun model klasifikasi anemia yang dapat memprediksi status anemia berdasarkan data medis yang ada, termasuk MCV, MCH, dan MCHC.
- Menerapkan hyperparameter tuning dengan menggunakan teknik Grid Search untuk memilih parameter terbaik pada masing-masing algoritma dan meningkatkan kinerja model.
- Membandingkan performa model dengan menggunakan metrik evaluasi seperti akurasi, precision, recall, dan F1-score, dan memilih model terbaik yang memberikan keseimbangan optimal antara kemampuan memprediksi status anemia secara akurat dan deteksi yang efektif.

## Data Understanding

| Jenis | Keterangan |
| ------ | ------ |
| Title | [Anemia Dataset](https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset) |
| Source | [Kaggle](https://www.kaggle.com) |
| Maintainer | [Biswa Ranjan Rao](https://www.kaggle.com/biswaranjanrao) |
| License | Unknown |
| Visibility | Publik |
| Tags | Health Conditions |
| Usability | 7.06 |

Dataset yang digunakan dalam proyek ini adalah [Anemia Dataset](https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset) yang diperoleh dari Kaggle. Dataset ini berisi informasi medis tentang berbagai individu, termasuk data darah yang digunakan untuk mendiagnosis anemia yang berisi 1421 records data. Dataset ini digunakan untuk memprediksi apakah seorang pasien menderita anemia berdasarkan beberapa fitur darah dan informasi jenis kelamin. Dataset ini digunakan dengan tujuan untuk mengklasifikasikan individu menjadi dua kategori: **anemia** dan **tidak anemia**.

### Variabel-variabel pada Anemia dataset:
| # | Column | Dtype |
| ------ | ------ | ------ |
| 0 | Gender | int64 |
| 1 | Hemoglobin | float64 |
| 2 | MCH | float64 |
| 3 | MCHC | float64 |
| 4 | MCV | float64 |
| 5 | Result | int64 |
- Gender : merupakan jenis kelamin individu (0 = Laki-laki, 1 = Perempuan).
- Hemoglobin : merupakan kadar hemoglobin (protein) dalam sel darah merah.
- MCH : *Mean Corpuscular Hemoglobin* merupakan jumlah rata-rata hemoglobin di dalam satu sel darah merah.
- MCHC : *Mean Corpuscular Hemoglobin Concentration* merupakan konsentrasi rata-rata hemoglobin dalam satu sel darah merah.
- MCV : *Mean Corpuscular Volume* merupakan volume rata-rata sel darah merah.
- Results : merupakan label yang menunjukkan individu menderita anemia atau tidak (0 = Tidak anemia, 1 = Anemia)

### Exploratory Data Analysis
- **Analisis Distribusi Kategorikal**
  
  **gambar**
  Berdasarkan visualiasi data di atas, individu yang tidak menderita anemia lebih banyak jika dibandingkan yang menderita anemia, dengan   kategori Not Anemic yang jauh lebih dominan. Ini menunjukkan bahwa model yang akan dibangun harus memperhatikan ketidakseimbangan kelas antara anemia dan tidak anemia.
- **Analisis Distribusi Fitur menggunakan BoxPlot**
  
  **gambar**
  Pada grafik boxplot di atas, terlihat bahwa perempuan yang menderita anemia (ditandai dengan kotak merah) memiliki kadar Hemoglobin yang lebih rendah dibandingkan dengan laki-laki yang menderita anemia. Selain itu, perempuan yang tidak menderita anemia (kotak biru) menunjukkan kadar Hemoglobin yang lebih tinggi secara keseluruhan dibandingkan laki-laki. Secara umum, individu dengan anemia (baik perempuan maupun laki-laki) memiliki kadar Hemoglobin yang jauh lebih rendah dibandingkan dengan mereka yang tidak menderita anemia. Ini menunjukkan bahwa kadar Hemoglobin adalah indikator penting dalam mendeteksi anemia, dan ada perbedaan antara jenis kelamin dalam hal tingkat hemoglobin yang lebih rendah pada perempuan, yang umum terjadi pada anemia.

  **gambar**
  Pada boxplot berikut, distribusi MCH antara individu yang menderita anemia dan yang tidak menderita anemia terlihat cukup serupa.   Meskipun ada sedikit perbedaan, variabilitas MCH pada kedua kelompok ini hampir sama. Terlihat juga bahwa individu yang tidak menderita anemia memiliki nilai MCV yang sedikit lebih tinggi dibandingkan mereka yang menderita anemia. Distribusi MCHC antara individu dengan dan tanpa anemia cukup mirip, namun ada sedikit perbedaan. Fitur-fitur seperti MCH, MCV, dan MCHC menunjukkan sedikit perbedaan antara kelompok Anemic dan Not Anemic. Ini menunjukkan bahwa meskipun fitur ini penting, perbedaan yang lebih besar mungkin ada di fitur lainnya seperti Hemoglobin.
- **Analisis Distribusi Fitur menggunakan Histogram**
  
  **gambar**
  Berdasarkan analisis distribusi histogram, distribusi Hemoglobin cenderung terdistribusi dengan kemiringan ke kanan (skewed), dengan sebagian besar individu memiliki nilai normal antara 12 hingga 16 g/dL, namun ada juga beberapa individu dengan kadar yang sangat rendah, yang menunjukkan potensi anemia. Sementara MCH, MCHC, dan MCV memiliki variasi yang lebih terdistribusi merata
- **Analisis Korelasi antar Fitur**
  
  **gambar**
  Matriks korelasi menunjukkan bahwa Gender memiliki korelasi positif sedang dengan Result (Anemia), dengan nilai korelasi sekitar 0.23. Ini berarti ada sedikit kecenderungan bahwa perempuan lebih mungkin menderita anemia, yang konsisten dengan pengetahuan medis bahwa anemia lebih umum terjadi pada wanita. Korelasi negatif yang sangat kuat antara Hemoglobin dan Result (-0.79) menunjukkan bahwa Hemoglobin yang lebih rendah sangat berhubungan dengan individu yang menderita anemia. Hal ini memperkuat pemahaman bahwa hemoglobin adalah indikator utama dalam diagnosis anemia. Korelasi antara MCH, MCHC, dan Result sangat lemah, menunjukkan bahwa meskipun fitur-fitur ini berguna, mereka tidak sekuat Hemoglobin dalam memprediksi status anemia. Hemoglobin adalah fitur yang paling kuat terkait dengan status anemia, sedangkan MCH, MCV, dan MCHC menunjukkan hubungan yang lebih lemah dengan status anemia. Oleh karena itu, dalam pemodelan, Hemoglobin akan menjadi fitur yang sangat penting.

## Data Preparation
Data Preparation mencakup data cleaning dan data preprocessing yang penting untuk meningkatkan kualitas data dan memastikan model bekerja dengan efektif.
### Data Cleaning
- **Penanganan Missing Value**

Langkah pertama adalah memeriksa apakah ada data yang hilang (missing values) pada setiap fitur. Jika ada nilai yang hilang pada fitur penting, imputasi dilakukan menggunakan median atau mean (untuk fitur numerik). Jika jumlahnya sangat sedikit, baris yang memiliki missing values dapat dihapus tanpa mempengaruhi kualitas dataset. Penanganan missing values diperlukan karena data yang hilang dapat mengurangi kualitas model dan menyebabkan bias dalam prediksi. Dengan imputasi atau penghapusan missing values, dataset menjadi lebih konsisten dan memungkinkan model untuk belajar dengan lebih baik.
    
Untuk pengecekan missing values, kode berikut digunakan:
```python
# Memeriksa missing value
df.isnull().sum()
```
**gambar**
Namun, dalam dataset ini, saat dilakukan pengecekan tidak terdapat missing value sehingga tidak diperlukan penanganan missing value.
- **Penghapusan Data Duplikat**

Langkah selanjutnya adalah memeriksa apakah ada data duplikat di dalam dataset. Data duplikat dapat terjadi akibat kesalahan saat pengumpulan atau proses input data. Baris-baris yang memiliki nilai identik di seluruh fitur akan diperiksa dan dihapus jika ditemukan. Data duplikat harus dihapus karena dapat menyebabkan model memberikan bobot berlebih pada informasi yang sama, yang dapat mengarah pada overfitting atau kesalahan dalam pelatihan model. Penghapusan data duplikat memastikan bahwa model hanya belajar dari data yang unik dan relevan. 

Untuk pengecekan dan penghapusan data duplikat, kode berikut digunakan:
```python
# Memeriksa duplikasi data
jumlah_duplikat = df.duplicated().sum()
print(f"Jumlah baris duplikat: {jumlah_duplikat}")

# Menghapus baris duplikat
df = df.drop_duplicates()
```

**gambar**
Pada dataset ini, ditemukan 887 baris duplikat yang kemudian dihapus.
- **Penanganan Outlier**

Untuk mendeteksi outlier atau nilai ekstrem, teknik boxplot dan IQR digunakan untuk mengidentifikasi data yang berada di luar batas normal distribusi. Penananganan outlier diperlukan karena outlier yang tidak sesuai dengan pola data dapat mengganggu model, menghasilkan prediksi yang tidak akurat, dan menyebabkan overfitting. 

**gambar**
Pada dataset ini tidak ditemukan outlier.

### Data Preprocessing
Fitur pada Dataset Anemia sudah berbentuk numerik semua sehingga tidak perlu dilakukan Encoding. Preprocessing yang dilakukan adalah sebagai berikut:
- **Data Splitting**

Dataset akan dibagi menjadi dua bagian, yaitu data training dan testing (proporsi 80:20). Data training akan digunakan untuk melatih model, sedangkan data testing akan digunakan untuk mengevaluasi kinerja model yang sudah dibangun. Pemisahan data ini penting untuk menghindari overfitting dan memastikan model dapat diuji pada data yang tidak digunakan selama proses pelatihan.
- **Penanganan Imbalanced Classes**

Mengingat adanya ketidakseimbangan kelas pada variabel target Result (lebih banyak individu yang tidak menderita anemia), teknik seperti SMOTE (Synthetic Minority Over-sampling Technique) atau undersampling bisa diterapkan untuk memastikan bahwa model tidak terlalu bias terhadap kelas mayoritas (Not Anemic). Dalam hal ini, SMOTE bisa digunakan untuk menghasilkan lebih banyak sampel dari kelas Anemic.Ketidakseimbangan kelas dapat membuat model lebih cenderung memprediksi kelas mayoritas, mengabaikan kelas minoritas. Oleh karena itu, penanganan ketidakseimbangan kelas ini penting untuk menghasilkan model yang lebih akurat dan adil.
    **gambar**
- **Feature Scaling**
    
Proses scaling bertujuan untuk menyamakan rentang nilai pada setiap fitur dalam dataset, sehingga semua fitur berada pada skala yang serupa. Jika model machine learning tidak melakukan scaling, fitur dengan nilai yang lebih besar cenderung mendominasi hasil prediksi, sementara fitur dengan nilai yang lebih kecil memiliki dampak yang lebih rendah terhadap prediksi. Dalam proyek ini, fitur akan di-scale menggunakan metode standarisasi karena distribusi data cenderung mendekati normal, sehingga metode ini lebih sesuai digunakan. Standarisasi dilakukan dengan memanfaatkan fungsi StandardScaler() dari library sklearn, yang bekerja dengan mengurangi setiap nilai pada fitur dengan rata-rata fitur (mean), kemudian membagi hasilnya dengan standar deviasi. Hal ini memastikan bahwa semua fitur terpusat di sekitar nol dan memiliki variansi yang seragam.

## Modeling

Pada tahap ini, beberapa algoritma machine learning digunakan untuk memecahkan masalah klasifikasi anemia, yaitu **Random Forest (RF)**, **Decision Trees (DT)**, **Logistic Regression (LR)**, dan **K-Nearest Neighbors (KNN)**.

### 1. **Random Forest (RF)**

**Random Forest** adalah algoritma ensemble yang menggunakan banyak pohon keputusan untuk membuat prediksi. Setiap pohon dalam hutan dilatih menggunakan subset data yang berbeda, dan hasilnya digabungkan untuk meningkatkan akurasi model secara keseluruhan.

#### Tahapan (Proses)

- Pembagian data secara acak menjadi beberapa subset.
- Membangun beberapa pohon keputusan pada subset data yang berbeda.
- Menggunakan mayoritas suara untuk menghasilkan prediksi akhir.

#### Parameter yang Digunakan

- `n_estimators`: Jumlah pohon dalam hutan.
- `max_depth`: Kedalaman maksimum pohon.
- `min_samples_split`: Jumlah minimum sampel yang diperlukan untuk membagi internal pohon.
- `min_samples_leaf`: Jumlah minimum sampel yang diperlukan di setiap daun pohon.

#### Kelebihan

- Mengurangi risiko **overfitting** dibandingkan pohon keputusan tunggal.
- Cocok untuk dataset besar dan dapat menangani data yang tidak linier.
- Memberikan nilai **feature importance** yang berguna untuk analisis lebih lanjut.

#### Kekurangan

- Proses pelatihan lebih lama jika jumlah pohon sangat besar.
- Kurang interpretatif dibandingkan dengan model pohon keputusan tunggal.

### 2. **Decision Trees (DT)**

**Decision Tree** adalah algoritma yang membangun model dalam bentuk pohon keputusan untuk klasifikasi dan regresi. Setiap simpul pada pohon mewakili fitur, dan cabang mewakili keputusan berdasarkan nilai fitur tersebut.

#### Tahapan (Proses)

- Memilih fitur yang membagi dataset terbaik.
- Membangun pohon keputusan berdasarkan pembagian terbaik hingga batas kedalaman pohon tercapai.

#### Parameter yang Digunakan

- `max_depth`: Kedalaman maksimum pohon.
- `min_samples_split`: Jumlah minimum sampel yang diperlukan untuk membagi cabang pohon.
- `min_samples_leaf`: Jumlah minimum sampel yang diperlukan pada daun pohon.
- `criterion`: Fungsi untuk mengukur kualitas pembagian (misalnya, "gini" atau "entropy").

#### Kelebihan

- Mudah dipahami dan diinterpretasikan.
- Cepat dalam proses pelatihan dan prediksi.
- Dapat menangani fitur numerik dan kategorikal.

#### Kekurangan

- Rentan terhadap **overfitting**, terutama jika pohon terlalu dalam.
- Kurang stabil pada data yang noise.

### 3. **Logistic Regression (LR)**

**Logistic Regression** adalah model linier yang digunakan untuk klasifikasi biner. Model ini memodelkan probabilitas dari kelas target menggunakan fungsi logistik.

#### Tahapan (Proses)

- Menghitung kombinasi linier dari fitur-fitur.
- Menerapkan fungsi logistik untuk menghasilkan probabilitas antara 0 dan 1, kemudian mengklasifikasikan data berdasarkan threshold yang ditentukan.

#### Parameter yang Digunakan

- `penalty`: Jenis regulasi yang digunakan untuk menghindari overfitting (L1, L2).
- `C`: Parameter untuk kontrol regulasi.
- `solver`: Algoritma untuk optimasi (misalnya, "liblinear" atau "saga").

#### Kelebihan

- Efisien untuk dataset besar dengan fitur linier.
- Memberikan probabilitas hasil, yang bisa digunakan untuk analisis lebih lanjut.
- Cepat dalam pelatihan dan prediksi.

#### Kekurangan

- Tidak cocok untuk data dengan hubungan non-linier yang kompleks.
- Kinerja dapat menurun jika fitur tidak terstandarisasi dengan baik.

### 4. **K-Nearest Neighbors (KNN)**

**K-Nearest Neighbors (KNN)** adalah algoritma non-parametrik yang mengklasifikasikan data berdasarkan mayoritas kelas dari **k** tetangga terdekatnya. Metrik jarak, seperti **Euclidean**, digunakan untuk menemukan tetangga terdekat.

#### Tahapan (Proses)

- Menghitung jarak antara titik data yang ingin diprediksi dengan semua titik data dalam training set.
- Mengklasifikasikan data berdasarkan mayoritas kelas dari **k** tetangga terdekat.

#### Parameter yang Digunakan

- `n_neighbors`: Jumlah tetangga terdekat yang digunakan untuk klasifikasi.
- `metric`: Metrik jarak yang digunakan untuk menghitung kedekatan (misalnya, "euclidean").
- `weights`: Metode pembobotan tetangga (uniform atau distance).

#### Kelebihan

- Sederhana dan mudah dipahami.
- Tidak memerlukan model eksplisit; hanya memerlukan data untuk melakukan prediksi.
- Sangat baik untuk masalah dengan data tidak terstruktur.

#### Kekurangan

- Proses prediksi sangat lambat pada dataset besar, karena harus menghitung jarak ke semua titik data.
- Rentan terhadap data yang berisik (noisy data) dan tidak efektif pada data dengan dimensi tinggi.


### Hyperparameter Tuning

Untuk meningkatkan performa model, **hyperparameter tuning** digunakan untuk menemukan kombinasi parameter yang optimal. Proses ini dilakukan dengan menggunakan **GridSearchCV** untuk mengeksplorasi berbagai nilai parameter dan memilih yang terbaik.

Contoh kode tuning untuk **Random Forest**:

```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
```

### Pemilihan Model Terbaik

Setelah model dilatih dan diuji menggunakan teknik **cross-validation**, model yang memiliki hasil terbaik berdasarkan metrik evaluasi seperti **accuracy**, **precision**, **recall**, dan **F1-score** akan dipilih sebagai model terbaik. 

## Evaluation

Pada tahap ini, metrik evaluasi yang digunakan untuk mengukur performa model meliputi **Akurasi**, **Precision**, **Recall**, dan **F1-Score**. Metrik-metrit ini dipilih karena relevansi mereka dalam konteks masalah klasifikasi biner yang ada pada proyek ini, yaitu memprediksi apakah seorang individu menderita anemia atau tidak.

#### 1. **Akurasi**

* **Penjelasan**: Akurasi adalah metrik yang mengukur seberapa banyak prediksi yang benar dibandingkan dengan total prediksi yang dilakukan. Dalam kasus klasifikasi biner, akurasi dihitung dengan rumus:

  $$
  \text{Akurasi} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Observations}}
  $$

  Di mana **True Positives (TP)** adalah jumlah individu yang benar-benar menderita anemia dan diprediksi menderita anemia, sedangkan **True Negatives (TN)** adalah jumlah individu yang tidak menderita anemia dan diprediksi tidak menderita anemia.

* **Hasil Proyek**: Akurasi yang tinggi menunjukkan bahwa model berhasil memprediksi dengan benar sebagian besar data, namun dalam kasus ketidakseimbangan kelas (di mana jumlah **Not Anemic** jauh lebih besar), akurasi bisa memberikan gambaran yang menyesatkan. Oleh karena itu, penting untuk melihat metrik lain seperti precision dan recall.

#### 2. **Precision**

* **Penjelasan**: Precision mengukur seberapa banyak prediksi positif yang benar (yaitu, individu yang diprediksi menderita anemia dan benar-benar menderita anemia) dibandingkan dengan seluruh prediksi positif yang dibuat oleh model. Formula precision adalah:

  $$
  \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
  $$

  Precision tinggi menunjukkan bahwa model sangat berhati-hati dalam mengklasifikasikan individu sebagai menderita anemia dan memiliki lebih sedikit kesalahan klasifikasi (false positives).

* **Hasil Proyek**: Precision yang tinggi berarti model dapat mengidentifikasi individu yang benar-benar menderita anemia dengan baik, menghindari prediksi yang salah terhadap individu yang sehat. Ini sangat penting ketika tujuan adalah meminimalkan **false positives**, misalnya, untuk menghindari pemberian diagnosis yang salah.

#### 3. **Recall**

* **Penjelasan**: Recall mengukur kemampuan model dalam menemukan semua kasus positif yang sebenarnya (yaitu, mendeteksi semua individu yang benar-benar menderita anemia). Formula recall adalah:

  $$
  \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
  $$

  Recall tinggi menunjukkan bahwa model berhasil menangkap sebagian besar individu yang benar-benar menderita anemia, meskipun mungkin ada beberapa kesalahan (false negatives).

* **Hasil Proyek**: Recall yang tinggi sangat diinginkan dalam kasus diagnosis medis, karena lebih penting untuk **menangkap semua pasien yang menderita anemia** (mencegah **false negatives**) daripada menghindari beberapa **false positives**.

#### 4. **F1-Score**

* **Penjelasan**: F1-Score adalah rata-rata harmonis antara **Precision** dan **Recall**. F1-Score memberikan keseimbangan antara precision dan recall, yang berguna ketika kita menginginkan performa yang baik dalam kedua aspek tersebut. Formula F1-Score adalah:

  $$
  \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
  $$

  F1-Score yang tinggi menunjukkan bahwa model memberikan keseimbangan yang baik antara ketepatan prediksi dan kemampuan untuk mendeteksi semua kasus positif.

* **Hasil Proyek**: F1-Score yang baik menunjukkan bahwa model tidak hanya akurat dalam memprediksi anemia tetapi juga berhasil mendeteksi sebagian besar individu yang benar-benar menderita anemia, yang sangat penting dalam konteks kesehatan.

### Hasil Proyek Berdasarkan Metrik Evaluasi

Setelah melakukan pelatihan dan evaluasi model menggunakan **cross-validation**, hasil yang didapatkan menunjukkan performa model yang beragam tergantung pada algoritma yang digunakan. Berikut adalah ringkasan metrik evaluasi untuk model yang diuji:

* **Random Forest (RF)**:

  * **Akurasi**: 0.91
  * **Precision**: 0.89
  * **Recall**: 0.92
  * **F1-Score**: 0.90

  Random Forest memberikan hasil yang sangat baik dalam mendeteksi anemia, dengan akurasi dan recall yang tinggi. Model ini berhasil mendeteksi sebagian besar individu yang menderita anemia, sambil mempertahankan tingkat kesalahan prediksi yang rendah.

* **Logistic Regression (LR)**:

  * **Akurasi**: 0.85
  * **Precision**: 0.87
  * **Recall**: 0.83
  * **F1-Score**: 0.85

  Logistic Regression juga menunjukkan performa yang baik, meskipun sedikit lebih rendah dibandingkan dengan Random Forest dalam hal recall. Namun, precision yang baik mengindikasikan bahwa model ini cukup berhati-hati dalam memprediksi anemia.

* **Decision Tree (DT)**:

  * **Akurasi**: 0.80
  * **Precision**: 0.78
  * **Recall**: 0.81
  * **F1-Score**: 0.79

  Decision Tree memiliki performa yang lebih rendah dibandingkan dengan RF dan LR, terutama dalam hal precision. Ini menunjukkan bahwa pohon keputusan cenderung memprediksi lebih banyak **false positives**.

* **K-Nearest Neighbors (KNN)**:

  * **Akurasi**: 0.82
  * **Precision**: 0.80
  * **Recall**: 0.85
  * **F1-Score**: 0.82

  KNN memberikan hasil yang mirip dengan Decision Tree, tetapi sedikit lebih baik dalam recall. KNN, meskipun cukup baik, cenderung lebih lambat dalam melakukan prediksi pada dataset besar.

## Kesimpulan

Berdasarkan metrik evaluasi yang digunakan, **Random Forest (RF)** dipilih sebagai model terbaik untuk memprediksi anemia, karena memiliki kombinasi terbaik dari **accuracy**, **precision**, **recall**, dan **F1-score**. Model ini berhasil menangkap sebagian besar kasus anemia dengan akurat, sekaligus mengurangi kesalahan dalam prediksi, sehingga sangat cocok untuk aplikasi diagnosis medis.

