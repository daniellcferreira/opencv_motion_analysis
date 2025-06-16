import numpy as np
import cv2
import os

# Caminho do video (ajustavel conforme a estrutura do projeto)
VIDEO_PATH = os.path.join('data', 'Rua.mp4')

# Abre o video com o OpenCV
cap = cv2.VideoCapture(VIDEO_PATH)

# Verifica se o video foi carregado corretamente
if not cap.isOpened():
  raise IOError(f"Não foi possivel abrir o video: {VIDEO_PATH}")

# Lê o primero frame apenas para garantir que a leitura está ok
hasFrame, frame = cap.read()
if not hasFrame:
  raise RuntimeError("Não foi possivel ler o primeiro frame do video.")

# Seleciona 72 frames aleatórios do video para calcular o frame medio
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames_ids = total_frames * np.random.uniform(size=72)

# Lista para armazenar os frames capturados
frames = []

for fid in frames_ids:
  cap.set(cv2.CAP_PROP_POS_FRAMES, fid)  # vai para o frame especifico
  hasFrame, frame = cap.read()
  if hasFrame:
    frames.append(frame)

# Calcula a frame mediano com base nos frames capturados
median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)

# Exibe o frame mediano na tela
cv2.imshow('Frame Mediano', median_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Salva o frame mediano como imagem
cv2.imwrite('outputs/median_frame.jpg', median_frame)