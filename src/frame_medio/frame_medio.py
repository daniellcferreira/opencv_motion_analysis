import numpy as np
from time import sleep
import cv2

# Define o atraso entre os frames na exibição
delay = 60

# Caminho do video de entrada
VIDEO = 'data/Rua.mp4'
# Caminho do video de saída com os frames filtrados
VIDEO_OUT = 'outputs/filtragem.avi'

# Inicializa a captura do video
cap = cv2.VideoCapture(VIDEO)
hasFrame, frame = cap.read()

# Inicializa o writer para gravar o video em escala de cinza
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(VIDEO_OUT, fourcc, 72, (frame.shape[1], frame.shape[0]), False)

# Gera 72 indices de frames aleatórios ao longo do video
framesIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=72)

# Coleta os frames aleatórios
frames = []
for fid in framesIds:
  cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
  hasFrame, frame = cap.read()
  frames.append(frame)

# Calcula o frame mediano com base nos frames amostrados
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
# Salva a imagem do frame mediano
cv2.imwrite('outputs/median_frame-pisa.jpg', medianFrame)

# Reinicia o video desde o primeiro frame para aplicar a filtragem
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Converte o frame mediano para tons de cinza
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', grayMedianFrame)
cv2.waitKey(0)

# Aplica a subtração de fundo em tempo real
while True:
  tempo = float(1 / delay)
  sleep(tempo)
  hasFrame, frame = cap.read()

  if not hasFrame:
    print('Acabaram os frames do video.')
    break

  # Converte frame atual para cinza
  frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Subtrai o fundo (frame mediano)
  dframe = cv2.absdiff(frameGray, grayMedianFrame)

  # Aplica limiarização binaria com método de Otsu
  th, dframe = cv2.threshold(dframe, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

  # Exibwe e salva frame processado
  cv2.imshow('frame', dframe)
  writer.write(dframe)

  # Interrompe o loop ao pressionar 'c'
  if cv2.waitKey(1) & 0xFF == ord('c'):
    break

# Libera recursos
writer.release()
cap.release()