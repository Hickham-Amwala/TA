import numpy as np
import cv2
import glob

# ============================================================================
# 1. PERSIAPAN PARAMETER PAPAN CATUR
# ============================================================================

# Tentukan jumlah sudut internal pada papan catur
# Contoh: Papan catur 20x13 kotak memiliki 19x12 sudut internal
CHECKERBOARD = (12, 17) 

# Kriteria untuk penyempurnaan sudut (sub-pixel accuracy)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Menyiapkan "object points" (titik objek) dalam koordinat 3D
# Ini adalah koordinat nyata dari sudut-sudut papan catur (0,0,0), (1,0,0), ..., (7,4,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Array untuk menyimpan object points dan image points dari semua gambar
objpoints = [] # Titik 3D di dunia nyata
imgpoints = [] # Titik 2D di bidang gambar (pixel)


# ============================================================================
# 2. MEMBACA GAMBAR DAN MENCARI SUDUT
# ============================================================================

# Ganti 'gambar_kalibrasi/*.jpg' dengan path ke folder gambar Anda
images = glob.glob(r'D:\Hickham\Kuliah\TA\kalibrasi\foto2\*.jpg')

print(f"Menemukan {len(images)} gambar untuk kalibrasi...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mencari sudut-sudut papan catur
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # Jika sudut ditemukan, tambahkan object points dan image points
    if ret == True:
        objpoints.append(objp)

        # Menyempurnakan lokasi sudut untuk akurasi yang lebih tinggi
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # (Opsional) Menggambar sudut yang terdeteksi untuk verifikasi
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Gambar Terdeteksi', img)
        cv2.waitKey(500) # Tampilkan selama 0.5 detik

cv2.destroyAllWindows()
print("Pencarian sudut selesai.")

# ============================================================================
# 3. MELAKUKAN KALIBRASI DAN MENYIMPAN HASIL
# ============================================================================

if len(objpoints) > 0 and len(imgpoints) > 0:
    print("Memulai proses kalibrasi...")
    # Fungsi calibrateCamera akan mengembalikan matriks kamera, koefisien distorsi,
    # vektor rotasi, dan vektor translasi.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Menampilkan hasil
    print("\n✅ Kalibrasi Berhasil!")
    print("-----------------------------------------------------")
    print("Matriks Kamera (mtx):\n", mtx)
    # Ekstraksi parameter dari matriks kamera
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]

    print("\n✅ Parameter Kamera Ditemukan:")
    print(f"  Focal Length (fx): {fx}")
    print(f"  Focal Length (fy): {fy}")
    print(f"  Principal Point (cx): {cx}")
    print(f"  Principal Point (cy): {cy}")
    print("\nKoefisien Distorsi (dist):\n", dist)
    print("-----------------------------------------------------")

    # Menyimpan hasil kalibrasi agar bisa digunakan lagi nanti
    np.savez('hasil_kalibrasi.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print("Hasil kalibrasi telah disimpan ke 'hasil_kalibrasi.npz'")

else:
    print("❌ Kalibrasi gagal. Tidak cukup sudut yang terdeteksi.")