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
1. **Data Cleaning**
  - Penanganan Missing Value
    Langkah pertama adalah memeriksa apakah ada data yang hilang (missing values) pada setiap fitur. Jika ada nilai yang hilang pada fitur penting, imputasi dilakukan menggunakan median atau mean (untuk fitur numerik). Jika jumlahnya sangat sedikit, baris yang memiliki missing values dapat dihapus tanpa mempengaruhi kualitas dataset. Penanganan missing values diperlukan karena data yang hilang dapat mengurangi kualitas model dan menyebabkan bias dalam prediksi. Dengan imputasi atau penghapusan missing values, dataset menjadi lebih konsisten dan memungkinkan model untuk belajar dengan lebih baik.
    
    Untuk pengecekan missing values, kode berikut digunakan:
    ```python
    # Memeriksa missing value
    df.isnull().sum()
    ```

    **gambar**
    Namun, dalam dataset ini, saat dilakukan pengecekan tidak terdapat missing value sehingga tidak diperlukan penanganan missing value.
  - Penghapusan Data Duplikat 
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
  - Penanganan Outlier
    Untuk mendeteksi outlier atau nilai ekstrem, teknik boxplot dan IQR digunakan untuk mengidentifikasi data yang berada di luar batas normal distribusi. Penananganan outlier diperlukan karena outlier yang tidak sesuai dengan pola data dapat mengganggu model, menghasilkan prediksi yang tidak akurat, dan menyebabkan overfitting. 

    **gambar**
    Pada dataset ini tidak ditemukan outlier.

2. **Data Preprocessing**
  Fitur pada Dataset Anemia sudah berbentuk numerik semua sehingga tidak perlu dilakukan Encoding. Preprocessing yang dilakukan adalah sebagai berikut:
  - Data Splitting
    Dataset akan dibagi menjadi dua bagian, yaitu data training dan testing (proporsi 80:20). Data training akan digunakan untuk melatih model, sedangkan data testing akan digunakan untuk mengevaluasi kinerja model yang sudah dibangun. Pemisahan data ini penting untuk menghindari overfitting dan memastikan model dapat diuji pada data yang tidak digunakan selama proses pelatihan.
  - Penanganan Imbalanced Classes
    Mengingat adanya ketidakseimbangan kelas pada variabel target Result (lebih banyak individu yang tidak menderita anemia), teknik seperti SMOTE (Synthetic Minority Over-sampling Technique) atau undersampling bisa diterapkan untuk memastikan bahwa model tidak terlalu bias terhadap kelas mayoritas (Not Anemic). Dalam hal ini, SMOTE bisa digunakan untuk menghasilkan lebih banyak sampel dari kelas Anemic.Ketidakseimbangan kelas dapat membuat model lebih cenderung memprediksi kelas mayoritas, mengabaikan kelas minoritas. Oleh karena itu, penanganan ketidakseimbangan kelas ini penting untuk menghasilkan model yang lebih akurat dan adil.
    **gambar**
  - Feature Scaling
    Proses scaling bertujuan untuk menyamakan rentang nilai pada setiap fitur dalam dataset, sehingga semua fitur berada pada skala yang serupa. Jika model machine learning tidak melakukan scaling, fitur dengan nilai yang lebih besar cenderung mendominasi hasil prediksi, sementara fitur dengan nilai yang lebih kecil memiliki dampak yang lebih rendah terhadap prediksi. Dalam proyek ini, fitur akan di-scale menggunakan metode standarisasi karena distribusi data cenderung mendekati normal, sehingga metode ini lebih sesuai digunakan. Standarisasi dilakukan dengan memanfaatkan fungsi StandardScaler() dari library sklearn, yang bekerja dengan mengurangi setiap nilai pada fitur dengan rata-rata fitur (mean), kemudian membagi hasilnya dengan standar deviasi. Hal ini memastikan bahwa semua fitur terpusat di sekitar nol dan memiliki variansi yang seragam.
