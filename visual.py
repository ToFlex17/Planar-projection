import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_axes_overlay(image, K, dist_coeffs, rvec, tvec, axis_length=100):
    img_axes = image.copy()

    # Achsen im Weltkoordinatensystem
    axis_3d = np.float32([
        [0, 0, 0],  # Ursprung
        [axis_length, 0, 0],  # X
        [0, axis_length, 0],  # Y
        [0, 0, axis_length]   # Z
    ])

    axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, K, dist_coeffs)
    axis_2d = axis_2d.reshape(-1, 2).astype(int)

    origin = tuple(axis_2d[0])
    cv2.line(img_axes, origin, tuple(axis_2d[1]), (0, 0, 255), 3)  # X = rot
    cv2.line(img_axes, origin, tuple(axis_2d[2]), (0, 255, 0), 3)  # Y = grün
    cv2.line(img_axes, origin, tuple(axis_2d[3]), (255, 0, 0), 3)  # Z = blau

    return img_axes

def show_3d_scene(obj_points, rvec, tvec, K=None, dist_coeffs=None):
    R, _ = cv2.Rodrigues(rvec)
    cam_position = -R.T @ tvec
    obj_points = np.array(obj_points).reshape(-1, 3)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Weltpunkte
    ax.scatter(obj_points[:, 0], obj_points[:, 1], obj_points[:, 2], c='r', label='Weltpunkte')

    # Kamera (schwarzes Kreuz)
    ax.scatter(cam_position[0], cam_position[1], cam_position[2], c='k', marker='x', s=100, label='Kamera')

    # Kameraachsen
    origin = cam_position.flatten()
    axis_length = 200
    axes = np.array([
        origin + R.T @ np.array([axis_length, 0, 0]),  # X
        origin + R.T @ np.array([0, axis_length, 0]),  # Y
        origin + R.T @ np.array([0, 0, axis_length])   # Z
    ])
    ax.plot([origin[0], axes[0][0]], [origin[1], axes[0][1]], [origin[2], axes[0][2]], 'r', label='X-Achse')
    ax.plot([origin[0], axes[1][0]], [origin[1], axes[1][1]], [origin[2], axes[1][2]], 'g', label='Y-Achse')
    ax.plot([origin[0], axes[2][0]], [origin[1], axes[2][1]], [origin[2], axes[2][2]], 'b', label='Z-Achse')

    # Rückprojektion der Bildpunkte (optional)
    if K is not None and dist_coeffs is not None:
        # Reprojektion der Weltpunkte auf Bild
        img_points, _ = cv2.projectPoints(obj_points, rvec, tvec, K, dist_coeffs)
        img_points = img_points.reshape(-1, 2)

        # Richtung im Kamera-KS (normierte Bildpunkte)
        img_points_normalized = cv2.undistortPoints(img_points.reshape(-1, 1, 2), K, dist_coeffs)
        rays_camera = np.concatenate([img_points_normalized, np.ones((len(img_points), 1, 1))], axis=2).reshape(-1, 3)

        # In Weltkoordinaten umrechnen
        rays_world = rays_camera @ R
        rays_world = rays_world / np.linalg.norm(rays_world, axis=1)[:, np.newaxis]

        # Endpunkte für Visualisierung: Kamera + Strahl
        line_len = 1000  # z. B. 1000 mm
        for ray in rays_world:
            p_end = cam_position.flatten() + ray * line_len
            ax.plot([cam_position[0, 0], p_end[0]],
                    [cam_position[1, 0], p_end[1]],
                    [cam_position[2, 0], p_end[2]], 'gray', linestyle='--', linewidth=0.5)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D-Szene mit Kamera und Rückprojektionsstrahlen')
    ax.legend()
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    plt.show()


# ▒▒▒ Beispielverwendung ▒▒▒
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




overlay = draw_axes_overlay(image, K, dist_coeffs, rvec, tvec)
cv2.imshow("Koordinatensystem Overlay", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

show_3d_scene(obj_points, rvec, tvec)
