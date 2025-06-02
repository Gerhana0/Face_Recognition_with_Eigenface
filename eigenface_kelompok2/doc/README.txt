## Kelompok 2 ##

- Faadhilah Hana Gustie Fatimah (L0124012)
- Haliza Hana Maulina (L0124017)
- Jelita Kustyara Nanda Safitri (L0124020)

# Face Recognition using EigenFace

Program ini merupakan implementasi face recognition menggunakan metode **EigenFace** dengan Python.  
Aplikasi ini mampu mengenali wajah berdasarkan dataset gambar, menggunakan teknik Eigen decomposition dan Euclidean distance sebagai pengukuran kemiripan.

---

## Fitur

- Face recognition berbasis **EigenFace**
- Upload dataset gambar wajah
- Upload test image
- Adjustable **threshold Euclidean distance** via slider di Streamlit
- Menampilkan **Euclidean distance hasil pengukuran**
- Menampilkan waktu eksekusi recognition
- Menampilkan test image & gambar dataset paling mirip
- Dilengkapi dummy image jika wajah tidak ditemukan

---

## Struktur Program
-eigenface_kelompok2
    - doc
        - README.txt
    - src
        - dummy
            - makasih.jpg
        - eigenface.py 
        - interface.py
        - main.py   
    - test

---

## Cara menjalankan Program

1. Download library numpy, Pillow, opencv-Python, dan streamlit di terminal
2. Ketik streamlit run src/interface.py di terminal
3. Masukkan path folder dataset gambar wajah (berisi gambar JPG/JPEG/PNG)
4. Upload test image
5. Atur threshold Euclidean distance (default: 500000)
6. Klik Start Recognition
7. Hasil pengenalan wajah dan jarak Euclidean ditampilkan di web
8. Apabila wajah tidak ditemukan, maka kemungkinan yang terjadi antara lain:
    - Wajahnya berbeda
    - Foto yang diberi bukan foto muka, jadi foto harus diperbesar mukanya
    - Dataset tidak terlalu banyak datanya

# Jika ingin foto bisa direcognition dengan detail maka lakukan:
1. img.size dari dataset dan input diperbesar jadi 256x256 atau terserah (Tetapi perlu dipertimbangkan, bisa saja foto yang dibandingkan memiliki hasil yang berbeda)
2. Perbesar threshold (semakin besar jarak euclidean maka semakin besar perbedaannya)
