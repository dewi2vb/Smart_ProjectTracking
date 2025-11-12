import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load log
df = pd.read_csv('movement_log.csv')

# Ambil koordinat
x = df['x_center'].values
y = df['y_center'].values

# Ukuran heatmap sesuai resolusi video (ubah sesuai video)
frame = cv2.VideoCapture('videos/test_video.mp4')
width = int(frame.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(frame.get(cv2.CAP_PROP_FRAME_HEIGHT))

heatmap, xedges, yedges = np.histogram2d(x, y, bins=(50, 50), range=[[0, width], [0, height]])
heatmap = np.rot90(heatmap)
heatmap = np.flipud(heatmap)

plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.title('Heatmap Pergerakan Individu')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.colorbar(label='Kepadatan')
plt.show()
