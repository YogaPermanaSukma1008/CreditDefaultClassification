# Penggunaan Hyperparameter Tuning dan Feature Selection Untuk Meningkatkan Akurasi Algoritma Machine Learning : Studi Kasus Pada Syntetic Dataset Loan Default
## Domain Proyek (Latar Belakang)
Krisis finansial global tahun 2008 menjadi salah satu contoh nyata kegagalan sistem keuangan, khususnya lembaga perbankan, dalam menjalankan fungsi dasarnya sebagai intermediasi keuangan. Dalam teori ekonomi, bank memiliki peran vital sebagai penyalur dana dari pihak yang surplus (penabung) ke pihak yang defisit (peminjam) melalui proses seleksi dan evaluasi risiko yang ketat atau yang selanjutnya proses intermediasi keuangan (Diamond & Dybvig, 1983). Namun, pada masa menjelang krisis, banyak bank di Amerika Serikat dan negara maju lainnya gagal dalam menerapkan prinsip kehati-hatian dalam penyaluran kredit. Mereka memberikan subprime mortgage loans, yaitu kredit perumahan kepada nasabah dengan profil risiko tinggi yang seharusnya tidak layak mendapatkan pinjaman berdasarkan standar konvensional. Dorongan untuk mengejar keuntungan jangka pendek dan inovasi produk keuangan yang kompleks (seperti mortgage-backed securities) membuat proses evaluasi kelayakan kredit diabaikan. Akibatnya terjadi gelembung aset dan credit boom (Laeven,Claessens, & Dell'Ariccia, 2010). Ketika tingkat gagal bayar meningkat drastis, pasar keuangan global terguncang dan memicu krisis sistemik. Hal ini menegaskan bahwa tanpa pengelolaan risiko yang baik, fungsi intermediasi bank tidak hanya gagal menjalankan perannya secara efektif, tetapi juga dapat menjadi sumber instabilitas keuangan global. Oleh karena itu sejak tahun 2008 dikembangkan sistem manajemen risiko untuk memberi pedoman pada lembaga keuangan khususnya dalam hal penyaluran kredit agar NPL perbankan dapat terjaga (Amaral & Lemos, 2015).

Seiring dengan perkembangan teknologi seleksi kredit tidak lagi dilakukan secara manual namun secara otomatis menggunakan algoritma machine learning dengan menggunakan data - data nasabah (Sayjadah et al, 2018). Dari sudut pandang mikroekonomi, proses yang demikian dapat meningkatkan efektivitas dan efisiensi sehingga mampu meringankan beban operasional perbankan. Sementara dari sudut pandang makroekonomi, ketika sistem seleksi kredit berbasis machine learning mencapai akurasi yang baik maka potensi instabilitas sistem keuangan sangat kecil. Namun penggunaan Machine Learning dalam seleksi kredit tidak secara langsung dapat menghasilkan prediksi yang akurat. Bisa saja orang yang layak mendapatkan kredit justru dideteksi sebagai orang yang berpotensi gagal bayar atau sebaliknya. 


## Problem Statement
Berdasarkan uraian pendahuluan diatas, dirumuskan beberapa pernyataan masalah sebagai berikut:
1.	Tingkat akurasi model sering kali dipengaruhi oleh fitur – fitur yang digunakan. Pemilihan fitur yang tidak relevan dapat berpengaruh  pada rendahnya akurasi model.  Selain itu penggunaan model tanpa pengaturan parameter yang optimal berpotensi akan mengalami underfitting atau overfitting. 
2.	Model membutuhkan evaluasi yang objektif dan menyeluruh untuk memastikan model mampu memprediksi loan default secara tepat. Oleh karena itu sangat penting memilih metrik evaluasi yang tepat untuk menilai keandalan model yang telah dibangun.  

## Goals
1.	Mencapai akurasi model dengan akurasi tinggi sebesar lebih dari 85 persen. Model mampu memprediksi loan default dan non default dengan baik. 
2.	Menentukan metrik evaluasi yang sesuai dengan algoritma klasifikasi yang telah dipilih sehingga mampu memberikan penilaian secara objektif terhadap performa model. 

## Solusi
Untuk mencapai tujuan di atas, proyek ini mengusulkan beberapa solusi teknis dan pendekatan:
1.	Menggunakan tiga model machine learning yang berbeda sebagai baseline:
- 	Logistic Regression 
- 	Random Forest
- 	Support Vector Machine (SVM)
2.	Melakukan pre-processing dan optimasi dataset melalui:
- 	Feature selection dengan berbasis korelasi. Mengingat data yang digunakan mempunyai fitur target binary (0 dan 1) maka teknik feature selection yang dapat diaplikasikan dan cukup sederhana adalah feature selection berbasis korelasi. Fitur - fitur independen yang saling berkaitan (multicolinearity) akan dihapus salah satunya sehingga tidak redundant atau menghapus fitur independen yang memiliki korelasi lemah terhadap fitur target. Untuk menerapkan feature selection jenis ini, proyek ini akan menggunakan teknik visualisasi data berupa heatmap (peta panas) sehingga dapat terpetakan dan mudah dibaca korelasi antar fitur.
- 	Penyeimbangan data (jika tidak seimbang) menggunakan SMOTE (Synthetic Minority Oversampling Technique). Proses ini dilakukan mengingat data non default lebih banyak dibandingkan data default. 
3.	Melakukan hyperparameter tuning dengan teknik seperti GridSearchCV atau RandomizedSearchCV untuk masing-masing model agar diperoleh konfigurasi yang optimal. Penggunaan Hyperparameter Tuning sangat bermanfaat karena berpotensi dapat meningkatkan akurasi sekaligus menghindari terjadinya overfitting. Dalam proyek ini akurasi yang ingin dicapai adalah 85 persen dengan menggunakan 3 algoritma machine learning yaitu logistic regression, random forest, dan support vector machine (SVM). Untuk logistic regression hyperparameter yang digunakan umumnya adalah regularisasi, random forest adalah n_estimator dan max depth sementara itu untuk SVM menggunakan kernel, c, dan gamma.
4.	Mengukur performa setiap model secara terstandar menggunakan metrik evaluasi: Accuracy, Precision, Recall, dan F1-Score.
5.	Model dengan performa terbaik akan dipilih sebagai rekomendasi akhir untuk diimplementasikan dalam sistem penilaian kredit berbasis machine learning.

## Data Understanding
### Dataset
Dataset yang digunakan dalam proyek ini berasal dari kaggle dengan link sebagai berikut [Dataset](https://www.kaggle.com/datasets/kmldas/loan-default-prediction/data). Data ini adalah kumpulan data sintetis yang dibuat menggunakan data aktual dari lembaga keuangan. Data telah dimodifikasi untuk menghilangkan fitur yang dapat diidentifikasi dan angka diubah untuk memastikan data tidak terkait dengan sumber asli (lembaga keuangan).

Dataset ini berisi 10000 baris dengan 5 kolom (4 fitur independen dan 1 fitur target). Berikut adalah fitur - fitur yang terkandung pada dataset:
1. Index : nomor seri atau pengenal unik dari penerima pinjaman
2. Employed : Menyatakan tentang status pekerjaan (apakah bekerja atau tidak). Data pada fitur ini dinyatakan sebagai binary, dimana 1 menyatakan seseorang bekerja (employed) dan 0 menyarakan seseorang tidak bekerja (not employed).
3. Bank Balance : Merupakan data numerik yang menyatakan jumlah saldo pada akun bank nasabah.
4. Annual Salary : Merupakan data numerik yang menyatakan gaji tahunan nasabah.
5. Default : Merupakan data binary (boolean) yang menyatakan status default nasabah. Angka 1 menyatakan default (gagal bayar) dan 0 adalah not default (tidak gagal bayar).
Sebelum dilakukan proses EDA, perlu memahami data structure dengan melakukan loading data. Berdasarkan tahap loading data dan data understanding, terdapat 10000 baris dengan 5 kolom (index, Employed, Bank Balance, Annual Salary dan Default). Seluruh fitur data merupakan data numerik dengan 3 fitur integer dan 2 fitur float. Tidak terdapat data kosong atau data duplikat sehingga tidak perlu membersihkan data karena data ini telah bersih.

Notes: Fitur target adalah kolom default sementara independen adalah Employed, Bank Balance, dan Annual Salary. Kolom index tidak diperlukan karena tidak memberikan informasi yang dapat menjelaskan kondisi default peminjam sehingga dilakukan proses dropping atau penghapusan kolom index. 

### EDA (Exploratory Data Analysis)
EDA (Exploratory Data Analysis) adalah proses eksplorasi awal terhadap data sebelum modeling, dengan tujuan memahami struktur, pola, anomali, dan hubungan antar variabel dalam dataset. Dalam hal ini, EDA yang digunakan yaitu:
1. Melihat distribusi data menggunakan histogram.
2. Melihat persentase kelas 0 dan 1 menggunakan diagram lingkaran.
3. Melihat adanya outlier menggunakan boxplot dan IQR.
4. Melihat korelasi antar fitur menggunakan heatmap.

Pada proses EDA, ditemukan outlier hanya pada fitur bank balance. Namun outlier ini tidak dimungkinkan untuk dihapus karena berisi informasi yang merepresentasikan kondisi peminjam yang sebenarnya. Tingginya bank balance merepresentasikan kemampuan bayar dari peminjam sangat tinggi. Sementara itu distribusi Bank Balance terdistribusi positif (right skewed) atau memiliki kecenderungan data terpusat pada nilai dibawah mean. Pada Annual Salary terdistribusi bimodal atau memiliki dua puncak. Analisis korelasi secara sekaligus digunakan untuk feature selecetion berbasis korelasi. Hasil analisis korelasi menggunakan heatmap menemukan bahwa Bank Balance mempunyai korelasi positif terhadap Default sebesar 0.35 sementara itu terjadi korelasi tinggi antara Employed dan Annual Salary sebesar 0.75.  

## Data Preparation
### Penghapusan Kolom yang tidak relevan (Drop)
Kolom index tidak diperlukan dalam analisis klasifikasi ini karena tidak mempunyai informasi yang jelas untuk dijadikan sebagai fitur independent sehingga perlu dilakukan dropping atau penghapusan kolom. 

### Feature Selection
Merupakan pemilihan fitur yang relevan dan tidak redundant. Dalam proyek ini feature scaling yang digunakan adalah berbasis korelasi. Fitur yang memiliki korelasi besar terhadap target akan dipilih. Sementara jika antar fitur independen saling berkorelasi akan dipilih salah satunya. Hasil analisis korelasi menunjukkan bahwa Bank Balance mempunyai korelasi yang lebih tinggi dibandingkan 2 fitur lainnya sebesar 0.35. Terjadi multicolinearity antara fitur independen yaitu Employed dan Annual Salary sehingga dapat menghapus salah satunya sebesar 0.75. Korelasi keduanya bisa sangat tinggi karena orang yang bekerja sudah tentu mempunyai penghasilan. Namun dalam hal ini, annual salary lah yang terpilih dibandingkan employee status. Hal ini dikarenakan seseorang dapat mempunyai penghasilan meskipun tidak bekerja misalnya pemberian orang tua, pendapatan pasif seperti hasil investasi, dana sumbangan atau hal lainnnya sehingga dapat dijadikan salah satu faktor penting dalam penilaian kredit. 

### Data Splitting
Terdapat dua proses data splitting. Pertama memisahkan variabel independen dengan variabel target. Kedua memisahkan data untuk data train dan data test, sehingga pada masing-masing variabel terdapat data train dan testnya.

Tahapan yang dilakukan dalam proses data splitting yaitu:
1. Menentukan variabel independen sebagai X dan target sebagai Y
2. Membagi X dan Y menjadi data train (X_train, y_train) dan data test (X_test, y_test).

### Data Scaling
Teknik data preparation yang digunakan dalam proyek ini adalah data scaling. Terdapat beberapa metode scaling diantaranya adalah normalisasi dan standarisasi. Namun dalam proyek ini digunakan teknik normalisasi atau mengubah skala data menjadi rentang 0 - 1. Tujuan dari data scaling ialah:
- Menyamakan skala fitur agar model ML (misalnya KNN, SVM, ANN) tidak berat sebelah terhadap fitur dengan nilai besar.
- Mempercepat proses konvergensi dalam model berbasis gradient descent (misalnya neural networks, logistic regression).
- Menghindari dominasi variabel besar terhadap yang kecil.

Tahapan yang dilakukan dalam proses data scaling yaitu:
1. Melakukan data splitting terlebih dahulu karena scaling dilakukan hanya pada data train. 
2. Mendefinisikan variabel atau fitur mana yang akan di scaling.
3. Inisialisasi fungsi minmax scaler.
4. Menggunakan fungsi minmax scaler pada fitur yang telah didefinisikan sebelumnya.

### SMOTE
Mengingat kelas 0 dan 1 tidak seimbang (imbalance) maka digunakan teknik SMOTE. SMOTE adalah metode oversampling yang digunakan untuk menangani masalah class imbalance dalam data klasifikasi, yaitu saat jumlah data pada satu kelas (biasanya kelas "1" atau positif) jauh lebih sedikit dibanding kelas lainnya. SMOTE digunakan hanya pada data train.

Tahapan yang dilakukan dalam proses SMOTE yaitu:
1. Lakukan splitting pada data train dan test. Penerapan SMOTE hanya dilakukan pada data training saja.
2. Identifikasi sebaran (distribusi) kelas pada data train. Jika kelas 1 lebih sedikit dibandingkan kelas 0 maka kelas 1 diperbanyak sehingga jumlahnya sama dengan kelas 0.
3. Cek distribusi hasil SMOTE.

## Modelling
Dalam proyek ini dipilih 3 algoritma Machine Learning yaitu Logistic Regression, Random Forest, dan Support Vector Machine.
1. Logistic Regression digunakan sebagai model baseline karena algoritma ini sederhana, mudah diinterpretasikan, dan efektif untuk memodelkan hubungan linier antara fitur input dengan probabilitas kejadian kelas tertentu. Penggunaan LR cocok pada data target yang mempunyai 2 kelas saja. Namun model ini memiliki kekurangan yaitu sensitif terhadap outlier dan multikolinearitas.

2. Random Forest dipilih karena merupakan metode ensambel yang kuat, mampu menangani hubungan non-linier antar fitur, serta relatif tahan terhadap outlier dan overfitting. Penggunaan RF dipilih mengingat terdapat data fitur yang memiliki outlier. Kekurangan model ini adalah waktu pelatihan yang lebih lama bergantung pada jumlah pohon.

3. Support Vector Machine digunakan karena kemampuannya dalam mencari batas pemisah (hyperplane) yang optimal antara kelas-kelas dengan margin maksimum, serta kemampuannya menangani data non-linier melalui penggunaan kernel. Sama seperti RF, proses training model ini juga lama serta memerlukan tuning parameter (seperti C, gamma, dan jenis kernel) yang sensitif dan dapat memengaruhi hasil akhir secara signifikan.

Dalam penggunaan ketiga model tersebut, sebelumnya perlu menemukan Hyperparameter yang sesuai menggunakan GridSearch. Hyperparameter tuning adalah proses mencari kombinasi parameter terbaik di luar model (hyperparameter) yang tidak dipelajari dari data, agar performa model Machine Learning menjadi optimal. Berikut adalah Hyperparameter yang digunakan pada masing-masing model:
1.	Logistic Regression:
- C  : merupakan inverse dari kekuatan regularisasi (regularization strength) dalam model Logistic Regression. Regularisasi adalah teknik yang digunakan untuk mencegah overfitting (model terlalu baik pada data latih tetapi buruk pada data baru) dengan menambahkan penalti pada koefisien model. Nilai C dalam param_grid_lr: [0.01, 0.1, 1, 10], semakin kecil menunjukkan regulrisasi yang kuat sementara semakin besar menunjukkan regularisasi yang lemah. 

- Penalty : menentukan jenis norma (norm) yang digunakan untuk regularisasi. Dalam Logistic Regression yang diimplementasikan di scikit-learn, pilihan umum adalah 'l1' (Lasso) dan 'l2' (Ridge).  Nilai l dalam param_grid_lr: ['l1, l2'], L1 memiliki kemampuan untuk melakukan seleksi fitur dengan memaksa beberapa koefisien menjadi nol, secara efektif menghilangkan fitur-fitur yang tidak relevan. Penalti L2 menambahkan kuadrat dari besarnya koefisien ke fungsi kerugian.
- solver : menentukan algoritma iteratif yang digunakan untuk mengoptimalkan fungsi kerugian dalam model Logistic Regression. Pilihan solver yang berbeda cocok untuk jenis data dan ukuran dataset yang berbeda. Nilai dalam param_grid_lr: ['lbfgs']. 

- max-iter : menentukan jumlah maksimum iterasi yang dilakukan oleh solver untuk mencapai konvergensi (yaitu, menemukan nilai koefisien optimal yang meminimalkan fungsi kerugian). Nilai dalam param_grid_lr: [100, 200], semakin kecil nilainya dapat mempercepat pelatihan. 

2. Random Forest:
-	n_estimators: jumlah pohon, Semakin banyak pohon, model cenderung lebih stabil, tapi akan membutuhkan lebih banyak waktu komputasi. Nilai yang digunakan yaitu  : 100, 200, 300
-	max_depth: kedalaman maksimum tiap pohon, mengontrol kompleksitas model. kedalaman yang terlalu besar bisa menyebabkan overfitting. Nilai yang diuji: 5, 10, 15
-	min_samples_split: minimal data untuk split, Jika nilainya lebih besar, pohon menjadi lebih sederhana dan bisa mengurangi overfitting. Nilai yang diuji: 2, 5, 10
-	min_samples_leaf : Jumlah minimum sampel yang harus dimiliki oleh node daun (leaf node), memastikan bahwa setiap leaf node memiliki cukup data. Nilai lebih tinggi bisa mencegah model mempelajari noise. Nilai yang diuji: 1, 2, 4

3.	SVM:
-	C: trade-off antara margin besar dan error kecil, semakin kecil nilainya margin lebih lebar, lebih toleran terhadap kesalahan (mencegah overfitting. Nilai yang digunakan : 0.1, 1, 10. 
-	kernel: jenis kernel (linear, rbf, poly, dll). Mengubah input data ke dalam ruang berdimensi lebih tinggi agar data yang tidak linier dapat dipisahkan. Nilai yang digunakan adalah linier dan rbf. 
-	gamma: Menentukan sejauh mana pengaruh satu data terhadap lainnya. Nilai yang digunakan adalah scale dan auto. 


## Evaluation
Metrik Evaluasi yang digunakan untuk mengevaluasi prediksi model yaitu:
1. Accouraccy : Persentase prediksi yang benar dari semua data.
2. Precision : Proporsi prediksi true positive terhadap total true positive dan false positive.
3. Recall : Proporsi data positif yang berhasil dikenali model.
4. F1 Score : Harmonic mean antara precision dan recall.
5. Confusion Matrix : Matrik evaluasi untuk melihat sebaran data prediksi yang sesuai dengan data aktualnya.

- Hasil Evaluasi model tanpa SMOTE: 
Berdasarkan hasil pembangunan model dengan beberapa teknik, didapatkan hasil bahwa seluruh model mencapai target akurasi 85 persen sesuai dengan goals yang ditetapkan. Namun seluruh model kurang mengenali kelas 1 (default) mengingat kelas 0 (non default) lebih banyak atau disebut sebagai imbalance. Berikut adalah hasil ringkasan evaluasinya:

Hasil Evaluasi Model Tanpa SMOTE : 
1. Logistic Regression: 
  - Akurasi: 0.8705
  - Presisi: 0.1935483870967742
  - Recall: 0.8695652173913043
  - F1-score: 0.316622691292876

2. Random Forest:
  - Akurasi: 0.96
  - Presisi: 0.4
  - Recall: 0.3188405797101449
  - F1-score: 0.3548387096774194

3. SVM:
  - Akurasi: 0.8675
  - Presisi: 0.189873417721519
  - Recall: 0.8695652173913043
  - F1-score: 0.3116883116883117
    
Notes: 
- LR & SVM: Kedua model ini sangat agresif dalam memprediksi kelas positif, sehingga recall tinggi (positif yang benar banyak dikenali), tapi precision sangat rendah (banyak false positives). F1-Score juga rendah, karena precision buruk.
- RF : Akurasi tinggi kemungkinan besar disebabkan oleh ketidakseimbangan kelas (class imbalance), sehingga model lebih sering memprediksi kelas mayoritas. Namun, precision, recall, dan F1-score rendah, menunjukkan buruk dalam mengenali kelas minoritas (misalnya default). 


- Hasil Evaluasi Model dengan SMOTE: 
Meskipun telah menggunakan SMOTE, namun hasil membuktikan bahwa penggunaan SMOTE pada kasus ini justru menurunkan akurasi model dan model tidak signifikan mengenali dengan baik pada kelas 1. Berikut adalah hasil evaluasi model dengan SMOTE: 
Berikut adalah hasil prediksi model dengan SMOTE:

1. Random Forest:
  - Accuracy: 0.9020
  - Precision: 0.2019
  - Recall: 0.6232
  - F1-score: 0.3050

2. Logistic Regression:
  - Accuracy: 0.8500
  - Precision: 0.1709
  - Recall: 0.8696
  - F1-score: 0.2857

3. SVM:
  - Accuracy: 0.8605
  - Precision: 0.1799
  - Recall: 0.8551
  - F1-score: 0.2972
Notes: Secara keseluruhan setelah penggunaan SMOTE, akurasi seluruh model menurun. Meskipun demikian, seluruh model masih berada pada target yang ditentukan (>85%). Random Forest menjadi model terbaik dikarenakan tingkat akurasi yang lebih tinggi dibandingkan yang lainnya. Sementara itu, Model terbaik untuk recall tinggi (menghindari false negatives) yaitu Logistic Regression.


## Kesimpulan
1. Data yang digunakan untuk pembangunan model berasal dari pemilihan fitur berbasis korelasi (feature selection). Teknik ini cukup sederhana yaitu hanya dengan melihat korelasi antar fitur. Jika korelasi fitur independen dengan fitur target besar maka akan dipilih. Sementara jika terjadi korelasi tinggi antar fitur independen maka akan dipilih salah satunya. Dengan pendekatan ini model akan bebas dari multikolinearitas terutama pada model yang sensitif pada multikolinearitas.

2. Berdasarkan hasil proyek, penggunaan hyperparameter tuning meningkatkan akurasi prediksi gagal bayar pinjaman untuk ketiga model machine learning yang dipertimbangkan: logistic regression, random forest, dan support vector machine (SVM) hingga mencapai lebih dari 85 persen. Dengan menyesuaikan hyperparameter model, kita dapat meningkatkan kemampuan model untuk mempelajari pola dalam data, membuat prediksi yang lebih akurat, dan menghindari overfitting data training.

3. Metrik evaluasi yang digunakan dalam proyek ini terdapat 4 metrik yaitu accouracy, pressicion, F1-Score, recall, dan confusion matrix. Secara umum seluruh model telah mencapai akurasi yang diharapkan dalam proyek ini yakni 85 persen. Namun model kurang mengenali kelas 1 dengan baik. Hal ini kemudian dilakukan teknik SMOTE untuk memperbesar sample pada kelas 1 sehingga diharapkan dapat meningkatkan model mengenali kelas 1.   


## Referensi
Diamond, Douglas W., and Dybvig, Philip H. 1983. Bank runs, deposit insurance, and liquidity. Journal of Political Economy 91 (June): 401–19. Reprinted in this issue of the Federal Reserve Bank of Minneapolis Quarterly Review. [Link](https://www.journals.uchicago.edu/doi/abs/10.1086/261155)

Laeven, M. L., Igan, M. D., Claessens, M. S., & Dell'Ariccia, M. G. (2010). Lessons and policy implications from the global financial crisis. International Monetary Fund. [Link](https://www.imf.org/external/pubs/ft/wp/2010/wp1044.pdf)

Amaral, M., & Lemos, K. (2015). Banking Risks: Lessons from the First Financial Crisis of the 21st Century. American International Journal of Business Management (AIJBM). [Link](https://www.aijbm.com/wp-content/uploads/2019/07/E275060.pdf)

Sayjadah, Y., Hashem, I. A. T., Alotaibi, F., & Kasmiran, K. A. (2018, October). Credit card default prediction using machine learning techniques. In 2018 Fourth International Conference on Advances in Computing, Communication & Automation (ICACCA) (pp. 1-4). IEEE. [Link](https://ieeexplore.ieee.org/abstract/document/9275986/)

