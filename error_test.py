import numpy as np
import cv2
import pandas as pd

# Gegebene Bild- und Objektpunkte
img_points = np.array([
    [1588.0, 1942.0],
    [1588.0, 1460.0],
    [1685.0,  530.4],
    [2067.0, 1616.0],
    [2782.0, 1954.0],
    [2160.0,  950.0],
    [2765.0,  737.3],
    [1169.0, 1989.0],
    [1133.0,  147.4],
    [ 480.0, 1275.0]
], dtype=np.float32)

obj_points = np.array([
    [   0.0,    0.0,    0.0],
    [   0.0,  300.0,    0.0],
    [  70.0,  870.0,    0.0],
    [ 300.0,  200.0,    0.0],
    [ 720.0,    0.0,    0.0],
    [ 350.0,  615.0,    0.0],
    [ 710.0,  730.0,    0.0],
    [-230.0,    0.0, -125.0],
    [-250.0, 1070.0, -125.0],
    [-230.0,  475.0, -950.0]
], dtype=np.float32)

# Schritt 1: Grobe Startwerte für Intrinsics
camera_matrix_init = np.array([
    [1333.0,    0.0, 1920/2],
    [   0.0, 1333.0, 1080/2],
    [   0.0,    0.0,    1.0]
], dtype=np.float64)

dist_coeffs_init = np.zeros(5, dtype=np.float64)

# Schritt 2: Kalibrierung (nur ein Satz von Punkten)
obj_pts_list = [obj_points]
img_pts_list = [img_points]

ret, camera_matrix_refined, dist_coeffs_refined, rvecs_refined, tvecs_refined = cv2.calibrateCamera(
    obj_pts_list,
    img_pts_list,
    (1920, 1080),
    camera_matrix_init,
    dist_coeffs_init,
    flags=cv2.CALIB_USE_INTRINSIC_GUESS
)

# Ausgabe der verfeinerten Parameter
print("Verfeinerte Kameramatrix:\n", camera_matrix_refined)
print("Verfeinerte Verzerrungskoeffizienten:", dist_coeffs_refined.ravel())

rvec_refined = rvecs_refined[0]
tvec_refined = tvecs_refined[0]
print("Verfeinerter rvec:", rvec_refined.ravel())
print("Verfeinerter tvec:", tvec_refined.ravel(), "\n")

# Schritt 3: 3D → 2D Reprojektion mit den verfeinerten Parametern
projected_refined, _ = cv2.projectPoints(
    obj_points,
    rvec_refined,
    tvec_refined,
    camera_matrix_refined,
    dist_coeffs_refined
)
projected_refined = projected_refined.reshape(-1, 2)

# Schritt 4: Reprojektion-Fehler berechnen und in DataFrame packen
errors_refined = np.linalg.norm(img_points - projected_refined, axis=1)

df_errors = pd.DataFrame({
    'Punkt': np.arange(len(img_points)),
    'Gemessen_u': img_points[:, 0],
    'Gemessen_v': img_points[:, 1],
    'Projiziert_u': projected_refined[:, 0],
    'Projiziert_v': projected_refined[:, 1],
    'Fehler_px': errors_refined
})

print("Reprojektionsfehler nach Verfeinerung:")
print(df_errors.to_string(index=False))
print(f"\nMittlerer Reproj.-Error: {errors_refined.mean():.2f} px")
