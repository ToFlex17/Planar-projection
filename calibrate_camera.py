import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_reprojection(image, obj_points, img_points, rvec, tvec, K, dist_coeffs):
    # 1. Reprojektionspunkte berechnen
    img_points_proj, _ = cv2.projectPoints(obj_points, rvec, tvec, K, dist_coeffs)
    img_points_proj = img_points_proj.reshape(-1, 2)

    # 2. Bild anzeigen
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Reprojektionsfehler-Visualisierung')

    # 3. Bildpunkte einzeichnen
    plt.scatter(img_points[:, 0], img_points[:, 1], c='r', label='Gemessene Bildpunkte')
    plt.scatter(img_points_proj[:, 0], img_points_proj[:, 1], c='g', marker='+', label='Reprojektierte Punkte')

    # 4. Fehlerlinien zeichnen
    for p_meas, p_proj in zip(img_points, img_points_proj):
        plt.plot([p_meas[0], p_proj[0]], [p_meas[1], p_proj[1]], 'y-', linewidth=0.8)

    # 5. Koordinatensystem-Projektion
    axis_length = 100  # in gleichen Einheiten wie obj_points, z. B. mm
    axis_3d = np.float32([
        [0, 0, 0],                    # Ursprung
        [axis_length, 0, 0],         # X
        [0, axis_length, 0],         # Y
        [0, 0, axis_length]          # Z
    ])
    axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, K, dist_coeffs)
    axis_2d = axis_2d.reshape(-1, 2)

    o = axis_2d[0]  # Ursprung
    x, y, z = axis_2d[1], axis_2d[2], axis_2d[3]

    plt.plot([o[0], x[0]], [o[1], x[1]], 'r-', linewidth=2, label='X-Achse')
    plt.plot([o[0], y[0]], [o[1], y[1]], 'g-', linewidth=2, label='Y-Achse')
    plt.plot([o[0], z[0]], [o[1], z[1]], 'b-', linewidth=2, label='Z-Achse')

    # 6. Darstellung
    plt.legend()
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.grid(True)
    plt.tight_layout()
    plt.show()




obj_points = np.array([
    [0   , 0   , 0   ],
    [7   , 57  , 0   ],
    [0   , 30  , 0   ],
    [72  , 0   , 0   ],
    [30  , 20  , 0   ],
    [71  , 73  , 0   ],
    [61.5, 35  , 0   ],
    [23  , 0   , -12.5],
    [25  , 107 , -12.5],
    [23  , 47.5, -95  ]
], dtype=np.float32)

obj_points = (obj_points + np.array([-30, -55.5, 0], dtype=np.float32)) *10

img_points = np.array([
    [1.588E3, 1.942E3],
    [1.588E3, 1.460E3],
    [1.685E3, 530.4  ],
    [2.067E3, 1.616E3],
    [2.782E3, 1.954E3],
    [2.160E3, 950.0  ],
    [2.765E3, 737.3  ],
    [1.169E3, 1.989E3],
    [1.133E3, 147.4  ],
    [480.0  , 1.275E3]
], dtype=np.float32)




# Bildgröße 4K iPhone 12
image_size = (3840, 2160)

# Initialschätzung der intrinsischen Kamera-Matrix
focal_length = 2758  # grobe Schätzung, etwas höher wegen 4K
center = (image_size[0] / 2, image_size[1] / 2)

K_init = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float32)

dist_init = np.zeros((5,), dtype=np.float32)  # Start: keine Verzerrung

flags = cv2.CALIB_USE_INTRINSIC_GUESS



ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    [obj_points], [img_points],
    image_size,
    K_init,
    dist_init,
    flags=flags
)

print(obj_points)

print("Kalibrierungsfehler:", ret)
print("Intrinsische Matrix K:\n", K)
print("Verzerrungskoeffizienten:\n", dist_coeffs.ravel())

# Extrinsische Parameter der Kamera
rvec = rvecs[0]
tvec = tvecs[0]


R, _ = cv2.Rodrigues(rvec)
camera_position = -R.T @ tvec

print("Rotation (R):\n", R)
print("Translation (t):\n", tvec)
print("Kameraposition im Weltkoordinatensystem:\n", camera_position)

# Optional: Reprojektionsfehler prüfen
proj_img_points, _ = cv2.projectPoints(obj_points, rvec, tvec, K, dist_coeffs)
for i, (orig, proj) in enumerate(zip(img_points, proj_img_points.squeeze())):
    error = np.linalg.norm(orig - proj)
    print(f"Punkt {i}: Fehler {error:.2f} Pixel")


image_path = '../Planar-projection/vlcsnap-2025-05-29-17h37m21s555.png'
image = cv2.imread(image_path)


visualize_reprojection(image ,obj_points, img_points, rvec, tvec, K, dist_coeffs )