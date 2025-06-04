import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Beispiel-Datenframe mit x, y, z Punkten
data = {
    'x': [0 ,0  ,7  ,30 ,72 ,35   ,71 ,-23   ,-25   ,-23  ],
    'y': [0 ,30 ,87 ,20 ,0  ,61.5 ,73 ,0     ,107   ,47.5 ],
    'z': [0 ,0  ,0  ,0  ,0  ,0    ,0  ,-12.5 ,-12.5 ,-95  ]
}


kamera_pos = np.array([30, -198, 55.5]) 


df = pd.DataFrame(data)

# 3D-Plot erzeugen
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Punkte aus dem DataFrame plotten
ax.scatter(df['x'], df['z'], df['y'], c='red', marker='o')
ax.scatter(kamera_pos[0], kamera_pos[1], kamera_pos[2], c='blue', marker='^', s=100, label='Hervorgehobener Punkt')

# Alle Punkte aus dem DataFrame und Geraden zum festen Punkt plotten
for i, row in df.iterrows():
    point = np.array([row['x'], row['z'], row['y']])
    line = np.array([kamera_pos, point])
    
    # Gerade plotten
    ax.plot(line[:,0], line[:,1], line[:,2], color='gray', linestyle='--')


# Punkte nummerieren
for i, row in df.iterrows():
    ax.text(row['x'], row['z'], row['y'], str(i), color='black', fontsize=10)

# Achsen beschriften
ax.set_xlabel('X-Achse')
ax.set_ylabel('Z-Achse')
ax.set_zlabel('Y-Achse')

# Plot anzeigen
plt.title('3D-Punktwolke aus DataFrame')
plt.show()
