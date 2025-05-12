# Laporan Proyek Machine Learning - Valensia Elsa Kurnia

## Domain Proyek
![Anemia Image](https://raw.githubusercontent.com/ValensiaElsa/Predictive-Analytics-Anemia/main/image/anemia.jpg)
Anemia adalah gangguan kesehatan yang umum di seluruh dunia, yang ditandai dengan penurunan jumlah sel darah merah atau kadar hemoglobin dalam darah. Kondisi ini dapat menyebabkan berkurangnya kemampuan darah untuk mengangkut oksigen ke seluruh tubuh, yang berpotensi mengurangi kualitas hidup dan meningkatkan risiko komplikasi medis serius, seperti penyakit jantung.  Kondisi ini sering kali disebabkan oleh kekurangan zat besi, defisiensi vitamin B12, atau masalah kesehatan lainnya. Berdasarkan data terbaru dari Organisasi Kesehatan Dunia (WHO), pada tahun 2023, sekitar 30,7% wanita usia 15–49 tahun mengalami anemia, dengan 35,5% di antaranya adalah wanita hamil. Selain itu, pada tahun 2019, prevalensi anemia pada anak-anak usia 6–59 bulan mencapai 39,8% secara global[[1]](https://www.who.int/data/gho/data/themes/topics/anaemia_in_women_and_children). 

Penyakit anemia biasanya terdeteksi melalui pemeriksaan medis yang memerlukan tes laboratorium untuk mengukur berbagai parameter darah, seperti kadar hemoglobin dan jumlah sel darah merah. Namun, prosedur ini dapat memakan waktu dan biaya. Selain itu, diagnosis sering kali terlambat, yang mengarah pada keterlambatan pengobatan dan peningkatan risiko komplikasi. Oleh karena itu, diperlukan cara yang cepat dan efisien agar proses diagnosis anemia menjadi lebih cepat dan pengobatan dapat dilakukan segera.

Masalah ini dapat diselesaikan dengan menerapkan machine learning dalam bentuk model prediktif untuk mendiagnosis anemia lebih cepat dan lebih akurat. Dengan memanfaatkan data medis seperti jumlah sel darah merah, kadar hemoglobin, MCV (Mean Corpuscular Volume), MCH (Mean Corpuscular Hemoglobin), dan MCHC (Mean Corpuscular Hemoglobin Concentration), model machine learning dapat memberikan diagnosis dini yang lebih efisien [[2]](https://www.researchgate.net/publication/368845592_PREDICTION_OF_ANEMIA_USING_MACHINE_LEARNING_ALGORITHMS). Proyek ini menggunakan algoritma klasifikasi untuk menganalisis pola dalam data pasien dan memprediksi kemungkinan seseorang menderita anemia. Hal ini tidak hanya mempercepat proses diagnosis tetapi juga memungkinkan tindakan pengobatan lebih cepat, yang dapat mencegah komplikasi lebih lanjut. 

Dengan penggunaan machine learning dalam analisis prediktif, kita dapat memanfaatkan data yang sudah ada untuk memberikan prediksi yang lebih presisi, lebih cepat, dan lebih terjangkau, yang pada gilirannya dapat memperbaiki manajemen kesehatan masyarakat. Ini juga memungkinkan tenaga medis untuk mengidentifikasi pasien yang membutuhkan perhatian segera, tanpa harus menunggu hasil tes laboratorium yang memakan waktu. Dengan demikian, proyek ini bertujuan untuk mengatasi masalah keterlambatan diagnosis yang dapat diselesaikan dengan solusi berbasis teknologi yang efisien dan lebih mudah diakses.

## Business Understanding

### Problem Statements
Berdasarkan uraian yang telah dipaparkan pada latar belakang diatas, maka dapat diambil sebuah rumusan masalah yang dirumuskan sebagai berikut:
- Bagaimana cara memprediksi apakah seseorang menderita anemia hanya dengan menggunakan data medis dasar seperti kadar hemoglobin, MCV (Mean Corpuscular Volume), MCH (Mean Corpuscular Hemoglobin), dan MCHC (Mean Corpuscular Hemoglobin Concentration), tanpa memerlukan tes laboratorium yang mahal dan memakan waktu?
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh untuk mendeteksi anemia?
- Bagaimana mengukur dan meningkatkan kinerja model prediktif untuk mendeteksi anemia?

### Goals
Berdasarkan rumusan masalah yang telah dipaparkan di atas, maka proyek penelitian ini memiliki tujuan, yaitu:
- Membangun model prediksi berbasis machine learning yang dapat memberikan diagnosis dini anemia dengan data medis dasar, seperti jumlah sel darah merah, kadar hemoglobin, MCV, MCH, dan MCHC.
- Mengetahui fitur yang paling berpengaruh untuk mendeteksi anemia.
- Mengukur dan meningkatkan kinerja model prediktif dengan menggunakan metrik evaluasi serta melakukan hyperparameter tuning untuk memperoleh hasil yang lebih optimal.

### Solution statements
Berdasarkan tujuan yang telah dipaparkan diatas, maka proyek penelitian ini memiliki solusi atau tahapan sebagai berikut:
- Menggunakan beberapa algoritma machine learning, seperti Logistic Regression, Decision Trees, Random Forest, dan K-Nearest Neighbors untuk membangun model klasifikasi anemia yang dapat memprediksi status anemia berdasarkan data medis yang ada dan akan dipilih satu model dengan kinerja model terbaik.
- Melakukan eksplorasi data awal (Exploratory Data Analysis, EDA) untuk menganalisis korelasi antar fitur, mengidentifikasi hubungan antara variabel medis dengan status anemia, dan menentukan fitur yang memiliki kontribusi signifikan dalam mendeteksi anemia.
- Menerapkan hyperparameter tuning dengan menggunakan teknik Grid Search untuk memilih kombinasi parameter terbaik pada masing-masing algoritma dan meningkatkan kinerja model. Kinerja model akan diukur dengan menggunakan metrik evaluasi seperti akurasi, precision, recall, dan F1-score, yang akan membantu dalam memilih model terbaik berdasarkan hasil prediksi yang paling optimal.

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
- Results : merupakan label yang menunjukkan individu menderita anemia atau tidak (0 = Tidak anemia, 1 = Anemia), result adalah fitur target.

Semua kolom bertipe data numerik dengan 4 fitur bertipe data float64 (Hemoglobin, MCH, MCHC, dan MCV) dan 2 fitur bertipe data int64 (Gender dan Result). Uraian di atas menunjukkan bahwa setiap kolom telah memiliki tipe data yang sesuai dan dikarenakan semua fitur adalah numerik, maka tidak diperlukan encoding untuk pelatihan. Namun, pada saat EDA kolom Result dan Gender akan diubah sementara ke bentuk kategorikal untuk mempermudah proses EDA yang kemudian akan dikembalikan ke bentuk semula.

![Deskripsi Statistik Image](https://raw.githubusercontent.com/ValensiaElsa/Predictive-Analytics-Anemia/main/image/deskripsi_statistik.png)

Berdasarkan pengecekan deskripsi statistik, kolom Gender dan Result memiliki distribusi yang cukup seimbang, sementara kolom numerik seperti Hemoglobin, MCH, MCHC, dan MCV menunjukkan variasi yang cukup besar. Variasi yang besar pada kolom Hemoglobin, MCH, MCHC, dan MCV adalah hal yang wajar, mengingat perbedaan kondisi antara individu yang menderita anemia dan yang tidak.

### Pengecekan Missing Value

  Langkah pertama adalah memeriksa apakah ada data yang hilang (missing values) pada setiap fitur. Jika ada nilai yang hilang pada fitur penting, imputasi dilakukan menggunakan median atau mean (untuk fitur numerik). Jika jumlahnya sangat sedikit, baris yang memiliki missing values dapat dihapus tanpa mempengaruhi kualitas dataset. Penanganan missing values diperlukan karena data yang hilang dapat mengurangi kualitas model dan menyebabkan bias dalam prediksi. Dengan imputasi atau penghapusan missing values, dataset menjadi lebih konsisten dan memungkinkan model untuk belajar dengan lebih baik.
    
  Untuk pengecekan missing values, kode berikut digunakan:
  ```python
  # Memeriksa missing value
  df.isnull().sum()
  ```
  ![Missing Value](https://raw.githubusercontent.com/ValensiaElsa/Predictive-Analytics-Anemia/main/image/cek_missing_value.png)

  Namun, dalam dataset ini, saat dilakukan pengecekan tidak terdapat missing value sehingga tidak diperlukan penanganan missing value.
  
### Pengecekan Data Duplikat

  Langkah selanjutnya adalah memeriksa apakah ada data duplikat di dalam dataset. Data duplikat dapat terjadi akibat kesalahan saat pengumpulan atau proses input data.

  Untuk pengecekan data duplikat, kode berikut digunakan:
  ```python
  # Memeriksa duplikasi data
  jumlah_duplikat = df.duplicated().sum()
    print(f"Jumlah baris duplikat: {jumlah_duplikat}")
  ```
  Pada dataset ini, ditemukan 887 baris duplikat.
  
### Pengecekan Outlier

  Untuk mendeteksi outlier atau nilai ekstrem, teknik boxplot dan IQR digunakan untuk mengidentifikasi data yang berada di luar batas normal distribusi. Penananganan outlier diperlukan karena outlier yang tidak sesuai dengan pola data dapat mengganggu model, menghasilkan prediksi yang tidak akurat, dan menyebabkan overfitting. 

  ![Outlier](https://raw.githubusercontent.com/ValensiaElsa/Predictive-Analytics-Anemia/main/image/outlier.png)

  Berdasarkan hasil dari perhitungan outlier menggunakan metode Interquartile Range (IQR), hanya ada satu outlier pada kadar hemoglobin, dan kadar hemoglobin memang bervariasi antar individu, terutama dalam konteks mendeteksi anemia. Mengingat bahwa kadar hemoglobin yang ekstrem dapat mencerminkan kondisi medis tertentu, mempertahankan outlier tersebut sangat penting karena dapat memberikan informasi diagnostik yang berharga dan membantu model dalam mendeteksi kondisi medis yang jarang terjadi.

## Exploratory Data Analysis
### Univariate Analysis
- **Analisis Distribusi Data Kategorikal**
  
  ![Analisis Distribusi Kategorikal Image](https://raw.githubusercontent.com/ValensiaElsa/Predictive-Analytics-Anemia/main/image/analisis_distribusi_kategorikal.png)

  Berdasarkan visualisasi data di atas, jumlah individu yang tidak menderita anemia lebih banyak dibandingkan dengan yang menderita anemia, dengan kategori Not Anemic yang jauh lebih dominan. Meskipun distribusi data relatif seimbang, ketidakseimbangan kelas antara Anemic dan Not Anemic tetap perlu diperhatikan. Oleh karena itu, model yang akan dibangun harus mempertimbangkan masalah ketidakseimbangan kelas ini, agar performa model tetap optimal dan tidak bias terhadap kelas mayoritas.
- **Analisis Distribusi Data Numerik**
  
  ![Analisis Distribusi Histogram](https://raw.githubusercontent.com/ValensiaElsa/Predictive-Analytics-Anemia/main/image/analisis_distribusi_histogram.png)

  Berdasarkan analisis distribusi histogram, distribusi Hemoglobin cenderung terdistribusi dengan kemiringan ke kiri (skewed), dengan sebagian besar individu memiliki nilai normal antara 10 hingga 16 g/dL, namun ada juga beberapa individu dengan kadar yang sangat rendah, yang menunjukkan potensi anemia. Sementara MCH, MCHC, dan MCV memiliki variasi yang lebih terdistribusi merata. MCH memiliki distribusi yang lebih merata dengan puncak di kisaran 20-22, sementara MCHC lebih terkonsentrasi pada nilai antara 28-32. MCV menunjukkan variasi yang lebih luas, terutama di kisaran 70 hingga 100. 

### Bivariate Analysis
  
  ![Analisis Distribusi BoxPlot](https://raw.githubusercontent.com/ValensiaElsa/Predictive-Analytics-Anemia/main/image/analisis_distribusi_fitur.png)

  Pada boxplot berikut, distribusi MCH antara individu yang menderita anemia dan yang tidak menderita anemia terlihat cukup serupa. Meskipun ada sedikit perbedaan, variabilitas MCH pada kedua kelompok ini hampir sama. Demikian juga terlihat pada distribusi MCV, kedua kelompok memiliki distribusi yang hampir sama. Distribusi MCHC antara individu dengan dan tanpa anemia cukup mirip, namun ada sedikit perbedaan. Fitur-fitur seperti MCH, MCV, dan MCHC menunjukkan sedikit perbedaan antara kelompok Anemic dan Not Anemic. Ini menunjukkan bahwa meskipun fitur ini penting, perbedaan yang lebih besar mungkin ada di fitur lainnya seperti Hemoglobin. Dapat dilihat bahwa individu yang terkena anemia memiliki kadar hemoglobin lebih rendah dibandingkan dengan individu yang tidak terkena anemia. 

### Multivarate Analysis
- **Analisis Distribusi**
  
  ![Analisis Distribusi BoxPlot](https://raw.githubusercontent.com/ValensiaElsa/Predictive-Analytics-Anemia/main/image/analisis_distribusi_gender_result_hemoglobin.png)

  Pada eksplorasi lebih lanjut distribusi data berdasarkan gender dan status anemia, terlihat bahwa perempuan yang menderita anemia (ditandai dengan kotak merah) memiliki kadar Hemoglobin yang lebih rendah dibandingkan dengan laki-laki yang menderita anemia. Selain itu, perempuan yang tidak menderita anemia (kotak biru) menunjukkan kadar Hemoglobin yang lebih tinggi secara keseluruhan dibandingkan laki-laki. Secara umum, individu dengan anemia (baik perempuan maupun laki-laki) memiliki kadar Hemoglobin yang jauh lebih rendah dibandingkan dengan mereka yang tidak menderita anemia. Ini menunjukkan bahwa kadar Hemoglobin adalah indikator penting dalam mendeteksi anemia, dan ada perbedaan antara jenis kelamin dalam hal tingkat hemoglobin yang lebih rendah pada perempuan, yang umum terjadi pada anemia.
- **Analisis Korelasi antar Fitur**
  
  ![Analisis Korelasi](https://raw.githubusercontent.com/ValensiaElsa/Predictive-Analytics-Anemia/main/image/analisis_korelasi.png)

   Matriks korelasi menunjukkan bahwa Gender memiliki korelasi positif sedang dengan Result (Anemia), dengan nilai korelasi sekitar 0.25. Ini berarti ada sedikit kecenderungan bahwa perempuan lebih mungkin menderita anemia, yang konsisten dengan pengetahuan medis bahwa anemia lebih umum terjadi pada wanita. Korelasi negatif yang sangat kuat antara Hemoglobin dan Result (-0.80) menunjukkan bahwa Hemoglobin yang lebih rendah sangat berhubungan dengan individu yang menderita anemia. Hal ini memperkuat pemahaman bahwa hemoglobin adalah indikator utama dalam diagnosis anemia. Korelasi antara MCH, MCHC, dan Result sangat lemah, menunjukkan bahwa meskipun fitur-fitur ini berguna, mereka tidak sekuat Hemoglobin dalam memprediksi status anemia. Meskipun begitu, tidak ada fitur yang di-drop mengingat jumlah data yang terbatas. Dalam kasus data yang terbatas, menghapus fitur meskipun dengan korelasi rendah dapat mengurangi informasi yang tersedia untuk model. Oleh karena itu, fitur-fitur ini tetap dipertahankan untuk memastikan bahwa model memiliki cukup informasi untuk meningkatkan prediksi, meskipun kontribusinya mungkin kecil.

  Hemoglobin adalah fitur yang paling kuat terkait dengan status anemia, sedangkan MCH, MCV, dan MCHC menunjukkan hubungan yang lebih lemah dengan status anemia. Oleh karena itu, dalam pemodelan, Hemoglobin akan menjadi fitur yang sangat penting. 

## Data Preparation
Fitur pada Dataset Anemia sudah berbentuk numerik semua sehingga tidak perlu dilakukan Encoding. Preprocessing yang dilakukan adalah sebagai berikut:
- **Penghapusan Data Duplikat**

  Data duplikat dapat terjadi akibat kesalahan saat pengumpulan atau proses input data. Baris-baris yang memiliki nilai identik di seluruh fitur akan diperiksa dan dihapus jika ditemukan. Data duplikat harus dihapus karena dapat menyebabkan model memberikan bobot berlebih pada informasi yang sama, yang dapat mengarah pada overfitting atau kesalahan dalam pelatihan model. Penghapusan data duplikat memastikan bahwa model hanya belajar dari data yang unik dan relevan.
  
 Untuk penghapusan data duplikat, kode berikut digunakan:
  ```python
  # Menghapus baris duplikat
  df = df.drop_duplicates()
  ```

  Sisa data setelah pembersihan baris duplikat adalah 534. Data yang terduplikasi memang cukup banyak, tetapi sisa data yang bersih sebanyak 534 (di atas 500) masih bisa untuk digunakan.
- **Data Splitting**

  Dataset akan dibagi menjadi dua bagian, yaitu data training dan testing (proporsi 80:20). Data training akan digunakan untuk melatih model, sedangkan data testing akan digunakan untuk mengevaluasi kinerja model yang sudah dibangun. Pemisahan data ini penting untuk menghindari overfitting dan memastikan model dapat diuji pada data yang tidak digunakan selama proses pelatihan. Kita harus membagi data agar proses transformasi hanya dilakukan pada data latih saja. Data uji harus berperan sebagai data baru yang tidak terpengaruh oleh proses pelatihan, untuk menilai bagaimana model bekerja pada data yang belum pernah dilihat sebelumnya.
  
  Berikut adalah jumlah data setelah dilakukan splitting.

  | Data              | Jumlah |
  |-------------------|--------|
  | Data Keseluruhan  | 534    |
  | Data Train        | 427    |
  | Data Test         | 107    |

- **Penanganan Imbalanced Classes**

  Mengingat adanya ketidakseimbangan kelas pada variabel target Result (lebih banyak individu yang tidak menderita anemia), teknik seperti SMOTE (Synthetic Minority Over-sampling Technique) atau undersampling bisa diterapkan untuk memastikan bahwa model tidak terlalu bias terhadap kelas mayoritas (Not Anemic). Dalam hal ini, SMOTE bisa digunakan untuk menghasilkan lebih banyak sampel dari kelas Anemic. Ketidakseimbangan kelas dapat membuat model lebih cenderung memprediksi kelas mayoritas, mengabaikan kelas minoritas. Oleh karena itu, penanganan ketidakseimbangan kelas ini penting untuk menghasilkan model yang lebih akurat dan adil. Berikut adalah perbandingan data sebelum dan setelah dilakukan SMOTE.
  ![Imbalance Class](https://raw.githubusercontent.com/ValensiaElsa/Predictive-Analytics-Anemia/main/image/imbalance_class.png)
- **Feature Scaling**
    
  Proses scaling bertujuan untuk menyamakan rentang nilai pada setiap fitur dalam dataset, sehingga semua fitur berada pada skala yang serupa. Jika model machine learning tidak melakukan scaling, fitur dengan nilai yang lebih besar cenderung mendominasi hasil prediksi, sementara fitur dengan nilai yang lebih kecil memiliki dampak yang lebih rendah terhadap prediksi. Dalam proyek ini, fitur akan di-scale menggunakan metode standarisasi karena distribusi data cenderung mendekati normal, sehingga metode ini lebih sesuai digunakan. Standarisasi dilakukan dengan memanfaatkan fungsi StandardScaler() dari library sklearn, yang bekerja dengan mengurangi setiap nilai pada fitur dengan rata-rata fitur (mean), kemudian membagi hasilnya dengan standar deviasi. Hal ini memastikan bahwa semua fitur terpusat di sekitar nol dan memiliki variansi yang seragam.
  
  Untuk menghindari kebocoran informasi pada data uji, kita hanya akan menerapkan fitur standarisasi pada data latih. Berikut adalah hasil standarisasi data. 
  
  ![Standarisasi Image](https://raw.githubusercontent.com/ValensiaElsa/Predictive-Analytics-Anemia/main/image/standarisasi.png)

## Modeling

Pada tahap ini, beberapa algoritma machine learning digunakan untuk memecahkan masalah klasifikasi anemia, yaitu **Random Forest (RF)**, **Decision Trees (DT)**, **Logistic Regression (LR)**, dan **K-Nearest Neighbors (KNN)**. 

Pada awalnya, model-model dasar seperti **RandomForestClassifier, DecisionTreeClassifier, LogisticRegression, dan KNeighborsClassifier** dilatih menggunakan **parameter default** dari masing-masing model. Pelatihan ini dilakukan pada **data training yang telah diskalakan dan di-resample** (X_train_scaled dan y_train_resampled), tanpa melakukan perubahan apapun pada parameter model. Ini bertujuan untuk mendapatkan baseline performance atau kinerja dasar model tanpa optimasi parameter.

Setelah model-model tersebut dilatih, tahap berikutnya adalah melakukan **hyperparameter tuning** untuk menemukan kombinasi parameter terbaik yang dapat meningkatkan kinerja model. Hyperparameter tuning dilakukan menggunakan metode GridSearchCV, yang akan mengeksplorasi berbagai kombinasi nilai parameter untuk setiap model dan memilih yang terbaik.

Setelah parameter terbaik ditemukan, model akan dibangun kembali dengan kombinasi parameter terbaik dan diuji untuk melihat apakah ada peningkatan performa dibandingkan dengan model yang dilatih dengan parameter default.

Berikut adalah penjelasan dari model yang akan digunakan:

### 1. **Random Forest (RF)**
![RF Image](https://raw.githubusercontent.com/ValensiaElsa/Predictive-Analytics-Anemia/main/image/random_forest.jpg)

**Random Forest** adalah algoritma ensemble yang menggunakan banyak pohon keputusan untuk membuat prediksi. Setiap pohon dalam hutan dilatih menggunakan subset data yang berbeda, dan hasilnya digabungkan untuk meningkatkan akurasi model secara keseluruhan. 

Berikut adalah kode pelatihan model.

```python
rf = RandomForestClassifier().fit(X_train_scaled, y_train_resampled)
```

**Tahapan:**

**Random Forest (RF)** adalah algoritma ensemble yang menggunakan teknik pembelajaran berbasis pohon keputusan. Proses ini dimulai dengan **pembagian data secara acak** menjadi beberapa subset yang berbeda. Setiap subset data kemudian digunakan untuk membangun **beberapa pohon keputusan** secara terpisah. Setiap pohon keputusan ini dilatih pada data yang berbeda, dengan beberapa fitur yang dipilih secara acak pada setiap percabangan untuk meningkatkan keragaman antar pohon. Setelah semua pohon selesai dilatih, algoritma **menggunakan mayoritas suara** (voting) dari seluruh pohon untuk menghasilkan prediksi akhir. Dengan cara ini, Random Forest mengurangi risiko overfitting yang sering terjadi pada pohon keputusan tunggal dan meningkatkan akurasi model dengan menggabungkan prediksi dari banyak pohon keputusan yang berbeda [[3]](https://scikit-learn.org/stable/modules/ensemble.html#random-forest).

**Parameter yang Digunakan:**

- `n_estimators`: Jumlah pohon dalam hutan.
- `max_depth`: Kedalaman maksimum pohon.
- `min_samples_split`: Jumlah minimum sampel yang diperlukan untuk membagi internal pohon.
- `min_samples_leaf`: Jumlah minimum sampel yang diperlukan di setiap daun pohon.

**Kelebihan:**

- Mengurangi risiko **overfitting** dibandingkan pohon keputusan tunggal.
- Cocok untuk dataset besar dan dapat menangani data yang tidak linier.
- Memberikan nilai **feature importance** yang berguna untuk analisis lebih lanjut.

**Kekurangan:**

- Proses pelatihan lebih lama jika jumlah pohon sangat besar.
- Kurang interpretatif dibandingkan dengan model pohon keputusan tunggal.

### 2. **Decision Trees (DT)**
![DT Image](https://raw.githubusercontent.com/ValensiaElsa/Predictive-Analytics-Anemia/main/image/decission_tree.png)

**Decision Tree** adalah algoritma yang membangun model dalam bentuk pohon keputusan untuk klasifikasi dan regresi. Setiap simpul pada pohon mewakili fitur, dan cabang mewakili keputusan berdasarkan nilai fitur tersebut.

Berikut adalah kode pelatihan model.

```python
dt = DecisionTreeClassifier().fit(X_train_scaled, y_train_resampled)
```

**Tahapan:**

**Decision Tree** adalah algoritma pembelajaran yang membangun model berbentuk pohon keputusan untuk memprediksi hasil berdasarkan fitur-fitur input. Proses dimulai dengan **memilih fitur yang membagi dataset terbaik**, yaitu fitur yang dapat memisahkan data secara optimal berdasarkan kriteria tertentu, seperti **Gini Impurity** atau **Entropy**. Pembagian ini dilakukan berulang kali pada setiap node, sehingga dataset terpecah menjadi subset yang lebih homogen. Setelah memilih fitur terbaik untuk membagi data, pohon keputusan **dibangun secara rekursif**, membagi dataset lebih lanjut pada setiap cabang pohon hingga **batas kedalaman pohon tercapai** atau hingga tidak ada pembagian yang lebih baik yang dapat dilakukan. Proses ini berlanjut hingga model mencapai titik di mana pembagian lebih lanjut tidak memberikan manfaat atau mencapai kriteria penghentian lainnya, seperti jumlah minimum sampel di setiap node atau kedalaman maksimum pohon yang diinginkan [[4]](https://scikit-learn.org/stable/modules/tree.html).

**Parameter yang Digunakan:**

- `max_depth`: Kedalaman maksimum pohon.
- `min_samples_split`: Jumlah minimum sampel yang diperlukan untuk membagi cabang pohon.
- `min_samples_leaf`: Jumlah minimum sampel yang diperlukan pada daun pohon.
- `criterion`: Fungsi untuk mengukur kualitas pembagian (misalnya, "gini" atau "entropy").

**Kelebihan:**

- Mudah dipahami dan diinterpretasikan.
- Cepat dalam proses pelatihan dan prediksi.
- Dapat menangani fitur numerik dan kategorikal.

**Kekurangan:**

- Rentan terhadap **overfitting**, terutama jika pohon terlalu dalam.
- Kurang stabil pada data yang noise.

### 3. **Logistic Regression (LR)**
![LR Image](https://raw.githubusercontent.com/ValensiaElsa/Predictive-Analytics-Anemia/main/image/logistic_regression.png)

**Logistic Regression** adalah model linier yang digunakan untuk klasifikasi biner. Model ini memodelkan probabilitas dari kelas target menggunakan fungsi logistik. 

Berikut adalah kode pelatihan model.

```python
lr = LogisticRegression().fit(X_train_scaled, y_train_resampled)
```

**Tahapan:**

**Logistic Regression (LR)** adalah metode untuk memprediksi kategori atau kelas suatu data berdasarkan informasi yang diberikan. Proses pertama adalah **menghitung kombinasi linier dari fitur-fitur**, yang artinya mengalikan setiap fitur dengan bobot (angka) yang ditentukan dan menjumlahkannya. Ini menghasilkan nilai yang bisa sangat besar atau kecil. Setelah itu, **fungsi logistik (atau sigmoid)** digunakan untuk mengubah nilai tersebut menjadi angka yang berada antara 0 dan 1, yang merepresentasikan probabilitas. Misalnya, nilai 0.8 berarti ada 80% kemungkinan data tersebut termasuk dalam kelas positif. Terakhir, model akan **mengklasifikasikan data** berdasarkan angka probabilitas ini. Jika probabilitas lebih besar dari angka ambang batas (misalnya 0,5), data akan dikategorikan sebagai kelas positif, dan jika lebih kecil, sebagai kelas negatif [[5]](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

**Parameter yang Digunakan:**

- `penalty`: Jenis regulasi yang digunakan untuk menghindari overfitting (L1, L2).
- `C`: Parameter untuk kontrol regulasi.
- `solver`: Algoritma untuk optimasi (misalnya, "liblinear" atau "saga").

**Kelebihan:**

- Efisien untuk dataset besar dengan fitur linier.
- Memberikan probabilitas hasil, yang bisa digunakan untuk analisis lebih lanjut.
- Cepat dalam pelatihan dan prediksi.

**Kekurangan:**

- Tidak cocok untuk data dengan hubungan non-linier yang kompleks.
- Kinerja dapat menurun jika fitur tidak terstandarisasi dengan baik.

### 4. **K-Nearest Neighbors (KNN)**
![KNN Image](https://raw.githubusercontent.com/ValensiaElsa/Predictive-Analytics-Anemia/main/image/knn.png)

**K-Nearest Neighbors (KNN)** adalah algoritma non-parametrik yang mengklasifikasikan data berdasarkan mayoritas kelas dari **k** tetangga terdekatnya. Metrik jarak, seperti **Euclidean**, digunakan untuk menemukan tetangga terdekat.

Berikut adalah kode pelatihan model.

```python
knn = KNeighborsClassifier().fit(X_train_scaled, y_train_resampled)
```

**Tahapan:**

**K-Nearest Neighbors (KNN)** adalah metode klasifikasi yang digunakan untuk memprediksi kelas suatu data berdasarkan kedekatannya dengan data lain dalam dataset. Proses dimulai dengan **menghitung jarak** antara titik data yang ingin diprediksi dan semua titik data yang ada di training set, biasanya menggunakan **Euclidean Distance**, yang mengukur seberapa jauh dua titik data satu sama lain. Setelah jarak dihitung, data akan **dikelompokkan berdasarkan mayoritas kelas dari k tetangga terdekat**. Dengan kata lain, kita memilih **k** titik data yang paling dekat dengan titik yang ingin diprediksi, dan kelas yang paling banyak muncul di antara k tetangga tersebut akan menjadi prediksi untuk data yang sedang diuji. Metode ini bergantung pada kedekatan data untuk melakukan klasifikasi, sehingga semakin kecil nilai **k**, semakin sensitif model terhadap perubahan lokal dalam data [[6]](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).

  
**Parameter yang Digunakan:**

- `n_neighbors`: Jumlah tetangga terdekat yang digunakan untuk klasifikasi.
- `metric`: Metrik jarak yang digunakan untuk menghitung kedekatan (misalnya, "euclidean").
- `weights`: Metode pembobotan tetangga (uniform atau distance).

**Kelebihan:**

- Sederhana dan mudah dipahami.
- Tidak memerlukan model eksplisit; hanya memerlukan data untuk melakukan prediksi.
- Sangat baik untuk masalah dengan data tidak terstruktur.

**Kekurangan:**

- Proses prediksi sangat lambat pada dataset besar, karena harus menghitung jarak ke semua titik data.
- Rentan terhadap data yang berisik (noisy data) dan tidak efektif pada data dengan dimensi tinggi.


### Hyperparameter Tuning

Untuk meningkatkan performa model, **hyperparameter tuning** digunakan untuk menemukan kombinasi parameter yang optimal. Proses ini dilakukan dengan menggunakan **GridSearchCV** untuk mengeksplorasi berbagai nilai parameter dan memilih yang terbaik.

Contoh kode tuning untuk **Random Forest**:

```python
from sklearn.model_selection import GridSearchCV
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5)
grid_rf.fit(X_train_scaled, y_train_resampled)

print("Best Params for Random Forest:", grid_rf.best_params_)
best_rf = grid_rf.best_estimator_
```
Berikut adalah parameter yang digunakan untuk pelatihan tiap model hasil dari hyperparameter tuning

  ![Hyperparameter Tuning Image](https://raw.githubusercontent.com/ValensiaElsa/Predictive-Analytics-Anemia/main/image/hyperparameter_tuning.png)
  
### Pemilihan Model Terbaik

Random Forest (RF) adalah pilihan terbaik untuk menangani dataset anemia dengan 534 data karena kemampuannya mengurangi risiko overfitting dengan menggunakan teknik ensemble, yang menghasilkan model lebih stabil dan akurat. Model ini dapat menangani hubungan non-linier antara fitur, memberikan feature importance untuk analisis lebih lanjut, dan tidak terpengaruh oleh skala fitur, sehingga memudahkan pemrosesan data. Selain itu, Random Forest dapat menangani data yang hilang, tidak memerlukan standarisasi fitur, dan tetap memberikan hasil yang konsisten meskipun dengan ukuran dataset yang relatif kecil. Namun, meskipun Random Forest memiliki banyak keunggulan, hasil evaluasi tetap harus diperhatikan untuk memastikan bahwa model ini memberikan performa yang optimal, dengan memeriksa metrik seperti accuracy, precision, recall, dan F1-score.

## Evaluation

Pada tahap ini, metrik evaluasi yang digunakan untuk mengukur performa model meliputi **Akurasi**, **Precision**, **Recall**, dan **F1-Score**. Metrik-metrit ini dipilih karena relevansi mereka dalam konteks masalah klasifikasi biner yang ada pada proyek ini, yaitu memprediksi apakah seorang individu menderita anemia atau tidak.

**1. Akurasi**

  Akurasi adalah metrik yang mengukur seberapa banyak prediksi yang benar dibandingkan dengan total prediksi yang dilakukan. Dalam kasus klasifikasi biner, akurasi dihitung dengan rumus:

  $$
  \text{Akurasi} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Observations}}
  $$

  Di mana **True Positives (TP)** adalah jumlah individu yang benar-benar menderita anemia dan diprediksi menderita anemia, sedangkan **True Negatives (TN)** adalah jumlah individu yang tidak menderita anemia dan diprediksi tidak menderita anemia. Akurasi yang tinggi menunjukkan bahwa model berhasil memprediksi dengan benar sebagian besar data, namun dalam kasus ketidakseimbangan kelas (di mana jumlah **Not Anemic** jauh lebih besar), akurasi bisa memberikan gambaran yang menyesatkan. Oleh karena itu, penting untuk melihat metrik lain seperti precision dan recall.

**2. Precision**

  Precision mengukur seberapa banyak prediksi positif yang benar (yaitu, individu yang diprediksi menderita anemia dan benar-benar menderita anemia) dibandingkan dengan seluruh prediksi positif yang dibuat oleh model. Formula precision adalah:

  $$
  \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
  $$

  Precision tinggi menunjukkan bahwa model sangat berhati-hati dalam mengklasifikasikan individu sebagai menderita anemia dan memiliki lebih sedikit kesalahan klasifikasi (false positives). Precision yang tinggi berarti model dapat mengidentifikasi individu yang benar-benar menderita anemia dengan baik, menghindari prediksi yang salah terhadap individu yang sehat. Ini sangat penting ketika tujuan adalah meminimalkan **false positives**, misalnya, untuk menghindari pemberian diagnosis yang salah.

#### 3. **Recall**

  Recall mengukur kemampuan model dalam menemukan semua kasus positif yang sebenarnya (yaitu, mendeteksi semua individu yang benar-benar menderita anemia). Formula recall adalah:

  $$
  \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
  $$

  Recall tinggi menunjukkan bahwa model berhasil menangkap sebagian besar individu yang benar-benar menderita anemia, meskipun mungkin ada beberapa kesalahan (false negatives). Recall yang tinggi sangat diinginkan dalam kasus diagnosis medis, karena **lebih penting** untuk **menangkap semua pasien yang menderita anemia** (mencegah **false negatives**) daripada menghindari beberapa **false positives**.

#### 4. **F1-Score**

  F1-Score adalah rata-rata harmonis antara **Precision** dan **Recall**. F1-Score memberikan keseimbangan antara precision dan recall, yang berguna ketika kita menginginkan performa yang baik dalam kedua aspek tersebut. Formula F1-Score adalah:

  $$
  \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
  $$

  F1-Score yang tinggi menunjukkan bahwa model memberikan keseimbangan yang baik antara ketepatan prediksi dan kemampuan untuk mendeteksi semua kasus positif. F1-Score yang baik menunjukkan bahwa model tidak hanya akurat dalam memprediksi anemia tetapi juga berhasil mendeteksi sebagian besar individu yang benar-benar menderita anemia, yang sangat penting dalam konteks kesehatan.

### Hasil Proyek Berdasarkan Metrik Evaluasi
Setelah melakukan pelatihan dan evaluasi model menggunakan **cross-validation**, hasil yang didapatkan menunjukkan performa model yang beragam tergantung pada algoritma yang digunakan. Terlihat bahwa performa model setelah dilakukan hyperparameter tuning memberikan hasil yang sedikit lebih baik dari sebelum melakukan hyperparameter tuning. 

![Perbandingan Evaluasi Image](https://raw.githubusercontent.com/ValensiaElsa/Predictive-Analytics-Anemia/main/image/perbandingan_evaluasi.png)

Berikut adalah ringkasan metrik evaluasi untuk model yang diuji **sebelum dilakukan hyperparameter tuning**:

![Evaluasi Before Image](https://raw.githubusercontent.com/ValensiaElsa/Predictive-Analytics-Anemia/main/image/evaluasi_before.png)

- **Random Forest (RF)**:

  * **Akurasi**: 99.07%
  * **Precision**: 99.08%
  * **Recall**: 99.07%
  * **F1-Score**: 99.07%

  Random Forest memberikan hasil yang sangat baik dalam mendeteksi anemia, dengan akurasi dan recall yang tinggi. Model ini berhasil mendeteksi sebagian besar individu yang menderita anemia, sambil mempertahankan tingkat kesalahan prediksi yang rendah.

- **Decision Tree (DT)**:

  * **Akurasi**: 97.20%
  * **Precision**: 97.36%
  * **Recall**: 97.20%
  * **F1-Score**: 97.20%

    Menunjukkan bahwa model ini masih memberikan performa yang sangat baik, meskipun sedikit lebih rendah dibandingkan dengan Random Forest. Precision yang tinggi menunjukkan bahwa sebagian besar prediksi positifnya benar, namun Recall yang sedikit lebih rendah menunjukkan bahwa model ini sedikit lebih sering melewatkan individu dengan anemia

- **Logistic Regression (LR)**:

  * **Akurasi**: 96.26%
  * **Precision**: 96.54%
  * **Recall**: 96.26%
  * **F1-Score**: 96.27%
  
  Meskipun tidak sebaik Random Forest atau Decision Tree, model ini tetap mampu mengklasifikasikan data dengan baik, terutama dalam hal Precision dan Recall yang seimbang. Namun, Recall yang lebih rendah menunjukkan bahwa model ini lebih sering melewatkan individu yang menderita anemia dibandingkan dengan model lainnya
  
- **K-Nearest Neighbors (KNN)**:

  * **Akurasi**: 90.65%
  * **Precision**: 91.69%
  * **Recall**: 90.65%
  * **F1-Score**: 90.66%
    
   Meskipun memiliki Precision dan Recall yang seimbang, KNN lebih sering melewatkan individu yang menderita anemia dan memiliki Accuracy yang jauh lebih rendah dibandingkan model lainnya.

Secara keseluruhan, **Random Forest** menjadi model yang paling unggul dalam hal akurasi, recall, dan keseimbangan metrik evaluasi, diikuti oleh **Decision Tree**, **Logistic Regression**, dan **K-Nearest Neighbors**. Model ini memberikan wawasan mengenai fitur yang paling relevan dan memberikan hasil yang optimal pada dataset yang diuji.

Berikut adalah ringkasan metrik evaluasi untuk model yang diuji **setelah dilakukan hyperparameter tuning**:

  ![Evaluasi After Image](https://raw.githubusercontent.com/ValensiaElsa/Predictive-Analytics-Anemia/main/image/evaluasi_after.png)

- **Random Forest (RF)**:

  * **Akurasi**: 99.07%
  * **Precision**: 99.08%
  * **Recall**: 99.07%
  * **F1-Score**: 99.07%

  Random Forest memberikan hasil yang sangat baik dalam mendeteksi anemia, dengan akurasi dan recall yang tinggi. Model ini berhasil mendeteksi sebagian besar individu yang menderita anemia, sambil mempertahankan tingkat kesalahan prediksi yang rendah.

- **Decision Tree (DT)**:

  * **Akurasi**: 99.07%
  * **Precision**: 99.08%
  * **Recall**: 99.07%
  * **F1-Score**: 99.07%
    
  Meskipun hasilnya hampir serupa dengan Random Forest, Decision Tree bisa lebih rentan terhadap overfitting dibandingkan dengan Random Forest karena tidak menggunakan ensemble learning, yang dapat membatasi kemampuan model untuk generalisasi pada data baru.

- **Logistic Regression (LR)**:

  * **Akurasi**: 97.20%
  * **Precision**: 97.35%
  * **Recall**: 97.20%
  * **F1-Score**: 97.20%
  
  Logistic Regression memberikan hasil solid, tetapi model ini menunjukkan kinerja yang lebih rendah dibandingkan dengan Random Forest dan Decision Tree, terutama dalam Recall, yang berarti model ini sedikit lebih sering melewatkan individu dengan anemia.
  
- **K-Nearest Neighbors (KNN)**:

  * **Akurasi**: 90.65%
  * **Precision**: 91.69%
  * **Recall**: 90.65%
  * **F1-Score**: 90.65%
    
  KNN memberikan hasil Precision dan Recall yang seimbang. Namun, KNN menunjukkan hasil yang lebih rendah dibandingkan dengan model lainnya, yang mengindikasikan bahwa KNN tidak optimal untuk dataset ini dan cenderung lebih sering melewatkan individu dengan anemia.

Secara keseluruhan, **Random Forest** tetap menjadi model yang paling disarankan setelah tuning dengan nilai Recall yang sangat tinggi dan metrik evaluasi lainnya yang juga tinggi, diikuti oleh Decision Tree, dengan Logistic Regression dan KNN lebih cocok digunakan dalam skenario yang lebih sederhana.

### Model Terbaik Berdasarkan Metrik Evaluasi

**Recall** yang tinggi sangat diinginkan dalam kasus diagnosis medis, karena **lebih penting** untuk **menangkap semua pasien yang menderita anemia** (mencegah **false negatives**) daripada menghindari beberapa **false positives**. Dalam konteks ini, **false negatives** (pasien yang seharusnya didiagnosis anemia tetapi tidak terdeteksi) dapat berakibat fatal karena pasien tersebut tidak menerima perawatan yang diperlukan. Oleh karena itu, model dengan **Recall** yang lebih tinggi, seperti **Random Forest**, sangat diutamakan untuk memastikan bahwa sebanyak mungkin individu yang menderita anemia dapat terdeteksi dan diberi penanganan yang tepat. Selain itu, model **Random Forest** juga memberikan hasil metrik evaluasi yang sangat tinggi. Meskipun Decission Tree juga memberikan hasil evaluasi yang sama, tetapi Random Forest sedikit lebih unggul karena kemampuannya dalam mengurangi overfitting melalui teknik ensemble sehingga Random Forest dipilih menjadi model terbaik.

### **Evaluasi Terhadap Business Understanding**

* Model yang dibangun **berhasil memprediksi apakah seseorang menderita anemia menggunakan data medis dasar** seperti kadar hemoglobin, MCV, MCH, dan MCHC. Penggunaan algoritma machine learning seperti Random Forest, Decission Tree, Logistic Regression, dan K-Nearest Neighbors memungkinkan model untuk mengklasifikasikan status anemia (Anemic vs Not Anemic) tanpa memerlukan tes laboratorium yang mahal. Melalui eksplorasi data dan analisis fitur, model ini memberikan solusi yang lebih cepat, murah, dan lebih mudah diakses dalam mendeteksi anemia dibandingkan dengan metode tes laboratorium tradisional.

* Dengan menggunakan metode Exploratory Data Analysis (EDA), ditemukan bahwa **Hemoglobin adalah fitur yang paling berpengaruh** dalam memprediksi status anemia. Korelasi yang sangat kuat antara kadar hemoglobin dan status anemia menunjukkan bahwa fitur ini adalah indikator utama dalam diagnosis anemia. Fitur-fitur lainnya seperti MCV dan MCHC juga berkontribusi, meskipun dengan pengaruh yang lebih rendah.

* **Kinerja model berhasil diukur** menggunakan metrik evaluasi (akurasi, presisi, recall, dan F-1 Score), dan **ditingkatkan** melalui penerapan hyperparameter tuning menggunakan Grid Search. Proses ini memungkinkan pencarian kombinasi parameter terbaik untuk algoritma yang digunakan. Setelah tuning, model menunjukkan peningkatan dalam metrik evaluasi seperti accuracy, precision, recall, dan F1-score, yang membuatnya lebih akurat dalam mendeteksi status anemia.

## Kesimpulan
Proyek ini berhasil mengembangkan model machine learning untuk mendeteksi anemia dengan menggunakan data medis dasar seperti kadar hemoglobin, MCV, MCH, dan MCHC. Dengan menggunakan algoritma seperti Random Forest, Logistic Regression, dan Decision Trees, serta penerapan hyperparameter tuning melalui Grid Search, model ini mampu memberikan prediksi yang akurat dan efisien. Hasil evaluasi menunjukkan bahwa Random Forest memberikan performa terbaik dalam mendeteksi anemia, dengan akurasi, precision, recall, dan F1-score yang tinggi. Model ini dapat diterapkan untuk diagnosis dini anemia dengan biaya rendah dan waktu yang lebih cepat, memberikan solusi yang efektif untuk masalah keterlambatan diagnosis.

## Referensi:

[[1] World Health Organization, "Anaemia in women and children," Global Health Observatory Data Repository, 2023. [Online]. Available: https://www.who.int/data/gho/data/themes/topics/anaemia_in_women_and_children. [Accessed: May 6, 2025].](https://www.who.int/data/gho/data/themes/topics/anaemia_in_women_and_children)

[[2] K. L. Seerangan, A. R. K. Saravanan, and P. K. R. S. Anandan, "Machine learning for prediction of anemia using laboratory data," Journal of Medical Systems, vol. 42, no. 5, 2018.](https://www.researchgate.net/publication/368845592_PREDICTION_OF_ANEMIA_USING_MACHINE_LEARNING_ALGORITHMS)

[[3]Scikit-learn documentation. "Random Forest," Scikit-learn, 2021. [Online]. Available: https://scikit-learn.org/stable/modules/ensemble.html#random-forest. [Accessed: 08-May-2025].](https://scikit-learn.org/stable/modules/ensemble.html#random-forest)

[[4]Scikit-learn documentation. "Decision Trees," Scikit-learn, 2021. [Online]. Available: https://scikit-learn.org/stable/modules/tree.html. [Accessed: 08-May-2025].](https://scikit-learn.org/stable/modules/tree.html)

[[5]Scikit-learn documentation. "LogisticRegression," Scikit-learn, 2021. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html. [Accessed: 08-May-2025].](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

[[6]Scikit-learn documentation. "KNeighborsClassifier," Scikit-learn, 2021. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html. [Accessed: 08-May-2025].](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
