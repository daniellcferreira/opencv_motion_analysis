import numpy as np
import cv2
import sys

# Caminho do vídeo de entrada
VIDEO = 'data/Ponte.mp4'

# Algoritmos disponíveis de subtração de fundo
algorithm_types = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']

# Função para retornar diferentes tipos de kernel para operações morfológicas
def Kernel(KERNEL_TYPE):
  if KERNEL_TYPE == 'dilation':
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
  if KERNEL_TYPE in ['opening', 'closing']:
    return np.ones((3, 3), np.uint8)

# Aplica o filtro morfológico desejado à imagem
def Filter(img, filter):
  if filter == 'closing':
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
  if filter == 'opening':
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, Kernel('opening'), iterations=2)
  if filter == 'dilation':
    return cv2.dilate(img, Kernel('dilation'), iterations=2)
  if filter == 'combine':
    # Combinação de filtros: fechamento → abertura → dilatação
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, Kernel('opening'), iterations=2)
    dilation = cv2.dilate(opening, Kernel('dilation'), iterations=2)
    return dilation

# Retorna o subtrator de fundo baseado no algoritmo escolhido
def Subtractor(algorithm_type):
  if algorithm_type == 'GMG':
    return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120, decisionThreshold=0.8)
  if algorithm_type == 'MOG':
    return cv2.bgsegm.createBackgroundSubtractorMOG(history=100, nmixtures=5, backgroundRatio=0.7, noiseSigma=0)
  if algorithm_type == 'MOG2':
    return cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=True, varThreshold=100)
  if algorithm_type == 'KNN':
    return cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)
  if algorithm_type == 'CNT':
    return cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, useHistory=True,
                                                    maxPixelStability=15*60, isParallel=True)
  print('Detector inválido')
  sys.exit(1)

# Parâmetros da detecção
w_min = 40           # Largura mínima do retângulo
h_min = 40           # Altura mínima do retângulo
offset = 2           # Tolerância para cruzamento de linha
linha_ROI = 620      # Posição da linha de contagem
carros = 0           # Contador de veículos

# Calcula o centro de um retângulo (bounding box)
def centroide(x, y, w, h):
  x1 = w // 2
  y1 = h // 2
  cx = x + x1
  cy = y + y1
  return cx, cy

# Lista que armazena os centros dos veículos detectados
detec = []

# Atualiza a contagem com base nos objetos que cruzam a linha
def set_info(detec):
  global carros
  for (x, y) in detec:
    if (linha_ROI + offset) > y > (linha_ROI - offset):
      carros += 1
      cv2.line(frame, (25, linha_ROI), (1200, linha_ROI), (0, 127, 255), 3)
      detec.remove((x, y))  # Remove centroide já contado
      print("Carros detectados até o momento:", carros)

# Exibe informações na tela
def show_info(frame, mask):
  frame = cv2.resize(frame, None, fx=0.75, fy=0.75)
  mask = cv2.resize(mask, None, fx=0.75, fy=0.75)

  text = f'Carros: {carros}'
  cv2.putText(frame, text, (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
  cv2.imshow("Video Original", frame)
  cv2.imshow("Detectar", mask)

# Inicialização da leitura do vídeo e do subtrator de fundo
cap = cv2.VideoCapture(VIDEO)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
algorithm_type = algorithm_types[1]  # MOG2
background_subtractor = Subtractor(algorithm_type)

# Loop principal de processamento frame a frame
while True:
  ok, frame = cap.read()
  if not ok:
    break

  # Aplica subtração de fundo e filtros
  mask = background_subtractor.apply(frame)
  mask = Filter(mask, 'combine')

  # Encontra contornos (possíveis veículos)
  contorno, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # Desenha a linha de contagem na tela
  cv2.line(frame, (25, linha_ROI), (1200, linha_ROI), (255, 127, 0), 3)

  for c in contorno:
    (x, y, w, h) = cv2.boundingRect(c)
    validar_contorno = (w >= w_min) and (h >= h_min)
    if not validar_contorno:
      continue

    # Desenha bounding box e registra centro
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    centro = centroide(x, y, w, h)
    detec.append(centro)
    cv2.circle(frame, centro, 4, (0, 0, 255), -1)

  # Atualiza contagem e mostra resultados
  set_info(detec)
  show_info(frame, mask)

  # Pressione ESC para sair
  if cv2.waitKey(1) == 27:
    break

# Libera recursos
cv2.destroyAllWindows()
cap.release()
