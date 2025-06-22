import numpy as np
import cv2
import sys

# Caminho do vídeo de entrada
VIDEO = 'data/Ponte.mp4'

# Lista de algoritmos disponíveis para subtração de fundo
algorithm_types = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']
algorithm = algorithm_types[0]  # Padrão: GMG

# Função que cria diferentes tipos de kernel morfológicos
def kernel(KERNEL_TYPE):
  if KERNEL_TYPE == 'dilation':
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
  if KERNEL_TYPE in ['opening', 'closing']:
    return np.ones((3, 3), np.uint8)

# Função que aplica filtros morfológicos
def filter(img, filtro):
  if filtro == 'closing':
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel('closing'), iterations=2)
  if filtro == 'opening':
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel('opening'), iterations=2)
  if filtro == 'dilation':
    return cv2.dilate(img, kernel('dilation'), iterations=2)
  if filtro == 'combine':
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel('closing'), iterations=2)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel('opening'), iterations=2)
    dilation = cv2.dilate(opening, kernel('dilation'), iterations=2)
    return dilation

# Função que retorna o subtrator de fundo
def subtractor(algorithm):
  if algorithm == 'GMG':
    return cv2.bgsegm.createBackgroundSubtractorGMG()
  if algorithm == 'MOG2':
    return cv2.createBackgroundSubtractorMOG2()
  if algorithm == 'MOG':
    return cv2.bgsegm.createBackgroundSubtractorMOG()
  if algorithm == 'KNN':
    return cv2.createBackgroundSubtractorKNN()
  if algorithm == 'CNT':
    return cv2.bgsegm.createBackgroundSubtractorCNT()
  print(f'Algoritmo {algorithm} não implementado.')
  sys.exit(1)

# Parâmetros da contagem
w_min = 30       # Largura mínima do retângulo do carro
h_min = 30       # Altura mínima do retângulo do carro
offset = 10      # Margem de erro
line = 300       # Linha de contagem
carros = 0       # Contador de carros
detec = []       # Lista de centroids detectados

# Função que calcula o centroide de um retângulo
def centroide(x, y, w, h):
  cx = x + w // 2
  cy = y + h // 2
  return cx, cy

# Atualiza a contagem se o centroide cruzar a linha
def set_info(detec, frame):
  global carros
  for (x, y) in list(detec):
    if (line - offset) < y < (line + offset):
      carros += 1
      cv2.line(frame, (25, line), (1200, line), (0, 127, 255), 3)
      detec.remove((x, y))
      print(f'Carros detectados até o momento: {carros}')

# Exibe informações na tela
def show_info(frame, mask):
  text = f'Carros: {carros}'
  cv2.putText(frame, text, (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
  cv2.imshow('Video Original', frame)
  cv2.imshow('Mascara Filtrada', mask)

# Inicialização da captura e subtrator
cap = cv2.VideoCapture(VIDEO)
background_subtractor = subtractor(algorithm)

# Função principal
def main():
  while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
      print('Não foi possível ler o vídeo.')
      break

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    mask = background_subtractor.apply(frame)
    mask_filter = filter(mask, 'combine')

    # Desenhar linha de contagem
    cv2.line(frame, (25, line), (1200, line), (255, 0, 0), 2)

    # Detectar contornos
    contornos, _ = cv2.findContours(mask_filter, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contornos:
      x, y, w, h = cv2.boundingRect(cnt)
      if w >= w_min and h >= h_min:
        centro = centroide(x, y, w, h)
        detec.append(centro)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, centro, 4, (0, 0, 255), -1)

    set_info(detec, frame)
    show_info(frame, mask_filter)

    # Tecla "c" encerra
    if cv2.waitKey(1) & 0xFF == ord("c"):
      break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
