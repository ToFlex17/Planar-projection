import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_reprojection_3d(obj_points, img_points, rvec, tvec, K, dist_coeffs, image_size=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 1. Originale 3D-Objektpunkte
    ax.scatter(obj_points[:, 0], obj_points[:, 1], obj_points[:, 2], c='b', label='3D Objektpunkte')

    # 2. Kamera-Koordinatensystem
    axis_length = 500  # je nach Einheit
    axis_3d = np.float32([
        [0, 0, 0],
        [axis_length, 0, 0],  # X
        [0, axis_length, 0],  # Y
        [0, 0, axis_length],  # Z
    ])

    # 3. Transformation: Welt → Kamera
    R, _ = cv2.Rodrigues(rvec)
    camera_pos = -R.T @ tvec.reshape(3)  # Kamerazentrum in Weltkoordinaten

    # Achsen transformieren in Weltkoordinaten
    cam_axes = (R.T @ axis_3d.T).T + camera_pos

    ax.plot([camera_pos[0], cam_axes[1][0]], [camera_pos[1], cam_axes[1][1]], [camera_pos[2], cam_axes[1][2]], 'r-', label='X-Achse')
    ax.plot([camera_pos[0], cam_axes[2][0]], [camera_pos[1], cam_axes[2][1]], [camera_pos[2], cam_axes[2][2]], 'g-', label='Y-Achse')
    ax.plot([camera_pos[0], cam_axes[3][0]], [camera_pos[1], cam_axes[3][1]], [camera_pos[2], cam_axes[3][2]], 'b-', label='Z-Achse')

    # 4. Reprojektion berechnen (in Bildkoordinaten)
    img_points_proj, _ = cv2.projectPoints(obj_points, rvec, tvec, K, dist_coeffs)
    img_points_proj = img_points_proj.reshape(-1, 2)

    # 5. Bildpunkte auf virtuelle Bildebene in 3D projizieren
    if image_size is None:
        image_size = (K[1, 2] * 2, K[0, 2] * 2)  # Breite, Höhe heuristisch

    width, height = image_size
    f = K[0, 0]  # Brennweite in Pixel

    # Rückprojektion in 3D: Bildpunkte auf Bildebene Z=f (Kamerakoordinaten)
    img_points_h = cv2.undistortPoints(img_points_proj[:, np.newaxis], K, dist_coeffs)
    img_points_cam = np.concatenate([img_points_h.squeeze(), np.ones((len(img_points_h), 1))], axis=1) * f

    # In Weltkoordinaten transformieren
    img_points_world = (R.T @ img_points_cam.T).T + camera_pos

    # 6. Punkte auf Bildebene plotten
    ax.scatter(img_points_world[:, 0], img_points_world[:, 1], img_points_world[:, 2], c='g', marker='x', label='Reprojizierte Punkte')
    for p_proj in img_points_world:
        ax.plot(
            [p_proj[0], camera_pos[0]],
            [p_proj[1], camera_pos[1]],
            [p_proj[2], camera_pos[2]],
            color='cyan', linestyle='--', linewidth=0.7
        )

    # 7. Linien vom Objektpunkt zum Kamerazentrum (statt zur Bildebene)
    for p3d in obj_points:
        ax.plot(
            [p3d[0], camera_pos[0]],
            [p3d[1], camera_pos[1]],
            [p3d[2], camera_pos[2]],
            'orange', linestyle='--', linewidth=0.7
        )

    reference_vec = img_points_world[0] - camera_pos
    reference_vec /= np.linalg.norm(reference_vec)  # normalisieren

    for i, p_proj in enumerate(img_points_world[1:], start=1):
        vec = p_proj - camera_pos
        vec /= np.linalg.norm(vec)  # normalisieren

        angle_rad = np.arccos(np.clip(np.dot(reference_vec, vec), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        print(f"Winkel zwischen Strahl 0 und Strahl {i}: {angle_deg:.6f}°")

    # Z-Achse der Kamera (optische Achse) im Kamerakoordinatensystem
    optical_axis_cam = np.array([0, 0, 1])

    # Drehmatrix aus Rotationsvektor
    R, _ = cv2.Rodrigues(rvec)

    # Optische Achse im Weltkoordinatensystem
    optical_axis_world = -R.T @ optical_axis_cam  # Kamera nach Welt

    # Achsen beschriften
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('3D Reprojektion')

    plt.tight_layout()
    plt.show()



def show_camera_image_view(img_points, obj_points, rvec, tvec, K, dist_coeffs):
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2

    # 1. Reprojezierte Punkte berechnen
    img_points_proj, _ = cv2.projectPoints(obj_points, rvec, tvec, K, dist_coeffs)
    img_points_proj = img_points_proj.reshape(-1, 2)
    img_points = img_points.reshape(-1, 2)

    # 2. Plot vorbereiten
    plt.figure(figsize=(6, 5))
    plt.gca().invert_yaxis()  # Bildursprung oben links

    # 3. Punkte plotten
    plt.scatter(img_points[:, 0], img_points[:, 1], c='r', label='Gemessene Punkte')
    plt.scatter(img_points_proj[:, 0], img_points_proj[:, 1], c='g', marker='+', label='Reprojizierte Punkte')

    # 4. Fehlerlinien zeichnen
    for p_meas, p_proj in zip(img_points, img_points_proj):
        plt.plot([p_meas[0], p_proj[0]], [p_meas[1], p_proj[1]], 'y-', linewidth=0.8)

    # 5. Achsen & Legende
    plt.title("Kameraansicht (2D)")
    plt.xlabel("Bild X [px]")
    plt.ylabel("Bild Y [px]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


obj_points = np.array([
    [0   , 0    , 0    ],
    [0   , 30   , 0    ],
    [7   , 87   , 0    ],
    [30  , 20   , 0    ],
    [72  , 0    , 0    ],
    [35  , 61.5 , 0    ],
    [71  , 73   , 0    ],
    [-23 , 0    , 12.5],
    [-25 , 107  , 12.5],
    [-23 , 47.5 , 95  ]
], dtype=np.float32)

# obj_points = (obj_points + np.array([-30, -55.5, 0], dtype=np.float32)) *10
obj_points = (obj_points) *10

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

# Extrinsische Parameter der Kamera
rvec = rvecs[0]
tvec = tvecs[0]


R, _ = cv2.Rodrigues(rvec)
camera_position = -R.T @ tvec



visualize_reprojection_3d(obj_points, img_points, rvec, tvec, K, dist_coeffs )