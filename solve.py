import numpy as np
import cv2

# ============================================================================
# LANGKAH 1: Muat hasil kalibrasi (Fondasi Anda)
# ============================================================================
with np.load('hasil_kalibrasi.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

# ============================================================================
# LANGKAH 2: Siapkan object points dan sumbu 3D yang akan digambar
# ============================================================================
# Object points (koordinat 3D papan catur), sama seperti di skrip kalibrasi
CHECKERBOARD = (12, 19)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Sumbu 3D untuk digambar (panjang 3 satuan di arah X, Y, Z)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

# ============================================================================
# LANGKAH 3: Proses satu gambar baru menggunakan solvePnP
# ============================================================================
# Misal, Anda memuat satu gambar baru
frame = cv2.imread(r'D:\Hickham\Kuliah\TA\kalibrasi\foto\20250818231637.jpg')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Temukan sudut papan catur di gambar ini
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

if ret == True:
    # Sempurnakan sudutnya
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    # ========================================================================
    # INTI-NYA DI SINI: Menggunakan solvePnP dengan hasil kalibrasi
    # ========================================================================
    # Temukan rotasi (rvec) dan translasi (tvec) dari papan catur
    _, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)

    # Proyeksikan titik sumbu 3D ke bidang gambar 2D
    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)

    # Gambar garis sumbu pada frame
    origin = tuple(corners2[0].ravel().astype(int))
    frame = cv2.line(frame, origin, tuple(imgpts[0].ravel().astype(int)), (255,0,0), 5) # Sumbu X (Biru)
    frame = cv2.line(frame, origin, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 5) # Sumbu Y (Hijau)
    frame = cv2.line(frame, origin, tuple(imgpts[2].ravel().astype(int)), (0,0,255), 5) # Sumbu Z (Merah)

    cv2.imshow('Pose Estimation dengan solvePnP', frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()