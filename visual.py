import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# def visualize_reprojection_3d(obj_points, img_points, rvec, tvec, K, dist_coeffs, image_size=None):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # 1. Originale 3D-Objektpunkte
#     ax.scatter(obj_points[:, 0], obj_points[:, 1], obj_points[:, 2], c='b', label='3D Objektpunkte')

#     # 2. Kamera-Koordinatensystem
#     axis_length = 500  # je nach Einheit
#     axis_3d = np.float32([
#         [0, 0, 0],
#         [axis_length, 0, 0],  # X
#         [0, axis_length, 0],  # Y
#         [0, 0, axis_length],  # Z
#     ])

#     # 3. Transformation: Welt → Kamera
#     R, _ = cv2.Rodrigues(rvec)
#     camera_pos = -R.T @ tvec.reshape(3)  # Kamerazentrum in Weltkoordinaten

#     # Achsen transformieren in Weltkoordinaten
#     cam_axes = (R.T @ axis_3d.T).T + camera_pos

#     ax.plot([camera_pos[0], cam_axes[1][0]], [camera_pos[1], cam_axes[1][1]], [camera_pos[2], cam_axes[1][2]], 'r-', label='X-Achse')
#     ax.plot([camera_pos[0], cam_axes[2][0]], [camera_pos[1], cam_axes[2][1]], [camera_pos[2], cam_axes[2][2]], 'g-', label='Y-Achse')
#     ax.plot([camera_pos[0], cam_axes[3][0]], [camera_pos[1], cam_axes[3][1]], [camera_pos[2], cam_axes[3][2]], 'b-', label='Z-Achse')

#     # 4. Reprojektion berechnen (in Bildkoordinaten)
#     img_points_proj, _ = cv2.projectPoints(obj_points, rvec, tvec, K, dist_coeffs)
#     img_points_proj = img_points_proj.reshape(-1, 2)

#     # 5. Bildpunkte auf virtuelle Bildebene in 3D projizieren
#     if image_size is None:
#         image_size = (K[1, 2] * 2, K[0, 2] * 2)  # Breite, Höhe heuristisch

#     width, height = image_size
#     f = K[0, 0]  # Brennweite in Pixel

#     # Rückprojektion in 3D: Bildpunkte auf Bildebene Z=f (Kamerakoordinaten)
#     img_points_h = cv2.undistortPoints(img_points_proj[:, np.newaxis], K, dist_coeffs)
#     img_points_cam = np.concatenate([img_points_h.squeeze(), np.ones((len(img_points_h), 1))], axis=1) * f

#     # In Weltkoordinaten transformieren
#     img_points_world = (R.T @ img_points_cam.T).T + camera_pos

#     # 6. Punkte auf Bildebene plotten
#     ax.scatter(img_points_world[:, 0], img_points_world[:, 1], img_points_world[:, 2], c='g', marker='x', label='Reprojizierte Punkte')
#     for p_proj in img_points_world:
#         ax.plot(
#             [p_proj[0], camera_pos[0]],
#             [p_proj[1], camera_pos[1]],
#             [p_proj[2], camera_pos[2]],
#             color='cyan', linestyle='--', linewidth=0.7
#         )

#     # 7. Linien vom Objektpunkt zum Kamerazentrum (statt zur Bildebene)
#     for p3d in obj_points:
#         ax.plot(
#             [p3d[0], camera_pos[0]],
#             [p3d[1], camera_pos[1]],
#             [p3d[2], camera_pos[2]],
#             'orange', linestyle='--', linewidth=0.7
#         )

#     reference_vec = img_points_world[0] - camera_pos
#     reference_vec /= np.linalg.norm(reference_vec)  # normalisieren

#     for i, p_proj in enumerate(img_points_world[1:], start=1):
#         vec = p_proj - camera_pos
#         vec /= np.linalg.norm(vec)  # normalisieren

#         angle_rad = np.arccos(np.clip(np.dot(reference_vec, vec), -1.0, 1.0))
#         angle_deg = np.degrees(angle_rad)

#         print(f"Winkel zwischen Strahl 0 und Strahl {i}: {angle_deg:.6f}°")

#     # Z-Achse der Kamera (optische Achse) im Kamerakoordinatensystem
#     optical_axis_cam = np.array([0, 0, 1])

#     # Drehmatrix aus Rotationsvektor
#     R, _ = cv2.Rodrigues(rvec)

#     # Optische Achse im Weltkoordinatensystem
#     optical_axis_world = -R.T @ optical_axis_cam  # Kamera nach Welt

#     # Achsen beschriften
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.legend()
#     ax.set_title('3D Reprojektion')

#     plt.tight_layout()
#     plt.show()

def visualize_reprojection_3d(df_image_points, object_points, img_points, K, dist_coeffs, rvec, tvec, plane_pts):
    # --- Ebene definieren ---
    p1, p2, p3 = map(np.asarray, plane_pts)
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    # --- Kamera extrahieren ---
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    cam_origin = (-R.T @ t).flatten()

    # --- Neue Punkte rückprojizieren ---
    reproj_points = []
    for _, row in df_image_points.iterrows():
        u, v = row['u'], row['v']
        pts = np.array([[[u, v]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(pts, K, dist_coeffs, P=K)
        x, y = undistorted[0, 0]
        ray_cam = np.array([x, y, 1.0]).reshape(3, 1)
        ray_world = R.T @ ray_cam

        denom = np.dot(normal, ray_world.flatten())
        if np.abs(denom) < 1e-6:
            point_on_plane = [np.nan, np.nan, np.nan]
        else:
            d = np.dot(normal, (p1 - cam_origin))
            s = d / denom
            point_on_plane = cam_origin + s * ray_world.flatten()

        reproj_points.append(point_on_plane)
    reproj_points = np.array(reproj_points)

    # --- Plot erstellen ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Kamera-Koordinatensystem darstellen
    axis_length = 500
    axis_3d = np.float32([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length],
    ])
    cam_axes = (R.T @ axis_3d.T).T + cam_origin

    ax.plot([cam_origin[0], cam_axes[1][0]], [cam_origin[1], cam_axes[1][1]], [cam_origin[2], cam_axes[1][2]], 'r-', label='X-Achse')
    ax.plot([cam_origin[0], cam_axes[2][0]], [cam_origin[1], cam_axes[2][1]], [cam_origin[2], cam_axes[2][2]], 'g-', label='Y-Achse')
    ax.plot([cam_origin[0], cam_axes[3][0]], [cam_origin[1], cam_axes[3][1]], [cam_origin[2], cam_axes[3][2]], 'b-', label='Z-Achse')

    # Kameraursprung
    ax.scatter(*cam_origin, c='red', s=100, label='Kamera')

    # Ebene visualisieren
    plane_array = np.array(plane_pts)
    grid_x, grid_y = np.meshgrid(
        np.linspace(plane_array[:, 0].min(), plane_array[:, 0].max(), 10),
        np.linspace(plane_array[:, 1].min(), plane_array[:, 1].max(), 10)
    )
    grid_z = (-(normal[0] * (grid_x - p1[0]) + normal[1] * (grid_y - p1[1])) / normal[2]) + p1[2]
    ax.plot_surface(grid_x, grid_y, grid_z, color='green', alpha=0.3, edgecolor='none', label='Projektionsebene')

    # Kalibrierungspunkte
    object_points_np = np.array(object_points).reshape(-1, 3)
    ax.scatter(object_points_np[:, 0], object_points_np[:, 1], object_points_np[:, 2],
               c='blue', s=60, label='Kalibrierungspunkte')

    # Linien von Kalibrierungspunkten zur Kamera
    for p3d in object_points_np:
        ax.plot(
            [p3d[0], cam_origin[0]],
            [p3d[1], cam_origin[1]],
            [p3d[2], cam_origin[2]],
            'orange', linestyle='--', linewidth=0.7
        )



    # Zurückprojizierte Punkte
    ax.scatter(reproj_points[:, 0], reproj_points[:, 1], reproj_points[:, 2],
               c='green', s=60, label='Reprojizierte Punkte')

    # Strahlen von Kamera zu reprojizierten Punkten
    for pt in reproj_points:
        ax.plot([cam_origin[0], pt[0]],
                [cam_origin[1], pt[1]],
                [cam_origin[2], pt[2]],
                color='gray', linestyle='--', linewidth=0.7)

    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.set_title('3D-Visualisierung: Kamera, Kalibrierungspunkte & reprojizierte Punkte')
    ax.legend()
    ax.view_init(elev=20, azim=130)
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

def compute_plane_from_points(P1, P2, P3):
    v1 = P2 - P1
    v2 = P3 - P1
    n = np.cross(v1, v2)
    n = n / np.linalg.norm(n)
    a, b, c = n
    d = -np.dot(n, P1)
    return np.array([a, b, c, d]), n

# def backproject_point_to_plane(p_norm, R, t, plane):
#     R_inv = np.linalg.inv(R)
#     ray_dir = R_inv @ p_norm.reshape(3, 1)
#     cam_origin = -R_inv @ t

#     a, b, c, d = plane
#     n = np.array([a, b, c]).reshape(1, 3)

#     denom = ray_dir.T @ n.T
#     if abs(denom[0, 0]) < 1e-6:
#         return np.array([np.nan, np.nan, np.nan])

#     lam = -(cam_origin.T @ n.T + d) / denom
#     X_w = cam_origin + lam * ray_dir
#     return X_w.flatten()

# def convert_dataframe_to_world_plane(df, camera_matrix, dist_coeffs, rvec, tvec, plane_points, x_col='x', y_col='y'):
#     rvec = np.array(rvec, dtype=np.float64).reshape(3, 1)
#     tvec = np.array(tvec, dtype=np.float64).reshape(3, 1)
#     R, _ = cv2.Rodrigues(rvec)

#     P1, P2, P3 = [np.array(p, dtype=np.float64) for p in plane_points]
#     plane, _ = compute_plane_from_points(P1, P2, P3)

#     points = df[[x_col, y_col]].values.astype(np.float32).reshape(-1, 1, 2)
#     undistorted = cv2.undistortPoints(points, camera_matrix, dist_coeffs, P=camera_matrix)

#     world_points = []
#     for p in undistorted:
#         x, y = p[0]
#         p_norm = np.array([x, y, 1.0], dtype=np.float64)
#         X_w = backproject_point_to_plane(p_norm, R, tvec, plane)
#         world_points.append(X_w)

#     world_points = np.array(world_points)
#     df_out = df.copy()
#     df_out[['X_w', 'Y_w', 'Z_w']] = world_points
#     return df_out

# def plot_3d_projection(df, rvec, tvec, plane_points):
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')

#     # Kameraposition
#     rvec = np.array(rvec, dtype=np.float64).reshape(3, 1)
#     tvec = np.array(tvec, dtype=np.float64).reshape(3, 1)
#     R, _ = cv2.Rodrigues(rvec)
#     cam_pos = -R.T @ tvec

#     # Ebene zeichnen
#     P1, P2, P3 = [np.array(p) for p in plane_points]
#     plane, normal = compute_plane_from_points(P1, P2, P3)

#     # Gitter für die Ebene
#     xx, yy = np.meshgrid(np.linspace(-20, 0, 10), np.linspace(-15, 0, 10))
#     a, b, c, d = plane
#     zz = (-a * xx - b * yy - d) / c
#     ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

#     # Weltpunkte
#     ax.scatter(df['X_w'], df['Y_w'], df['Z_w'], c='green', label='Weltpunkte')

#     # Kamera
#     ax.scatter(*cam_pos.flatten(), c='red', marker='o', s=100, label='Kamera')

#     # Verbindende Linien
#     for i, row in df.iterrows():
#         ax.plot([cam_pos[0, 0], row['X_w']],
#                 [cam_pos[1, 0], row['Y_w']],
#                 [cam_pos[2, 0], row['Z_w']], 'k--', alpha=0.5)

#     ax.set_xlabel('X [m]')
#     ax.set_ylabel('Y [m]')
#     ax.set_zlabel('Z [m]')
#     ax.set_title('3D-Rückprojektion auf Welt-Ebene')
#     ax.legend()
#     ax.view_init(elev=20, azim=-60)
#     plt.tight_layout()
#     plt.show()

def backproject_dataframe_to_plane(df, K, dist_coeffs, rvec, tvec, plane_pts):
    # 1. Ebene definieren
    p1, p2, p3 = map(np.asarray, plane_pts)
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    # 2. Rotation/Translation vorbereiten
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    cam_origin = -R.T @ t

    results = []

    for _, row in df.iterrows():
        u, v = row['u'], row['v']
        pts = np.array([[[u, v]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(pts, K, dist_coeffs, P=K)
        x, y = undistorted[0, 0]
        ray_cam = np.array([x, y, 1.0]).reshape(3, 1)
        ray_world = R.T @ ray_cam

        denom = np.dot(normal, ray_world.flatten())
        if np.abs(denom) < 1e-6:
            point_on_plane = [np.nan, np.nan, np.nan]
        else:
            d = np.dot(normal, (p1 - cam_origin.flatten()))
            s = d / denom
            point_on_plane = cam_origin.flatten() + s * ray_world.flatten()

        results.append({
            'u': u,
            'v': v,
            'X': point_on_plane[0],
            'Y': point_on_plane[1],
            'Z': point_on_plane[2]
        })

    return pd.DataFrame(results)


obj_points = np.array([
    [0   , 0    , 0    ],
    [0   , 30   , 0    ],
    [7   , 87   , 0    ],
    [30  , 20   , 0    ],
    [72  , 0    , 0    ],
    [35  , 61.5 , 0    ],
    [71  , 73   , 0    ],
    [-23 , 0    , -12.5],
    [-25 , 107  , -12.5],
    [-23 , 47.5 , -95  ]
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

print(img_points)


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



df = pd.DataFrame({
    'v': [1682, 1683, 1682, 1682, 1682, 100, 100, 100, 100, 100, 1.588E3],
    'u': [512, 158, 1682, 2182, 1182, 512, 158, 1682, 2182, 1182, 1.942E3]
})

plane_pts = [
    np.array([0, 0, 0]),
    np.array([0, 10000, 0]),
    np.array([10000, 0, 0])
]


df_3d = backproject_dataframe_to_plane(df, K, dist_coeffs, rvec, tvec, plane_pts)
print(df_3d)
camera_position = -R.T @ tvec
print("Kameraposition im Weltkoordinatensystem:\n", camera_position)

visualize_reprojection_3d(df, obj_points, img_points, K, dist_coeffs, rvec, tvec, plane_pts)


projected, _ = cv2.projectPoints(
    obj_points,    # (10×3)
    rvec,          # (3×1)
    tvec,          # (3×1)
    K,
    dist_coeffs
)

proj_img_points, _ = cv2.projectPoints(obj_points, rvec, tvec, K, dist_coeffs)
for i, (orig, proj) in enumerate(zip(img_points, proj_img_points.squeeze())):
    error = np.linalg.norm(orig - proj)
    print(f"Punkt {i}: Fehler {error:.2f} Pixel")



# visualize_reprojection_3d(obj_points, img_points, rvec, tvec, K, dist_coeffs )
# show_camera_image_view(img_points, obj_points, rvec, tvec, K, dist_coeffs)