import numpy as np
import cv2
import sys

# Caminho do video a ser processado
VIDEO = 'data/Ponte.mp4'

# Lista de algoritmos de subtração de fundo disponíveis
algorithms = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']
algorithm_type = algorithms[0] # 'GMG' é o algoritmo padrão

# Função para obter kernel com base no tipo de operação morfológica
def kernel(KERNEL_TYPE):
  if KERNEL_TYPE == 'dilation':
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
  if KERNEL_TYPE == 'opening' or KERNEL_TYPE == 'closing':
    return np.ones((3, 3), np.uint8)
  
# Função para aplicar o filtro morfológico a imagem binária
def filter(img, filter):
  if filter == 'closing':
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel('closing'), iterations=2)
  if filter == 'opening':
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel('opening'), iterations=2)
  if filter == 'dilation':
    return cv2.dilate(img, kernel('dilation'), iterations=2)
  if filter == 'combine':
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel('closing'), iterations=2)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel('opening'), iterations=2)
    dilation = cv2.dilate(opening, kernel('dilation'), iterations=2)
    return dilation
  
# Fiunção que cria o subtrator de fundo de acorodo com algoritmo selecionado
def subtractor(algorithm_type):
  if algorithm_type == 'GMG':
    return cv2.bgsegm.createBackgroundSubtractorGMG()
  if algorithm_type == 'MOG2':
    return cv2.createBackgroundSubtractorMOG2()
  if algorithm_type == 'MOG':
    return cv2.bgsegm.createBackgroundSubtractorMOG()
  if algorithm_type == 'KNN':
    return cv2.createBackgroundSubtractorKNN()
  if algorithm_type == 'CNT':
    return cv2.bgsegm.createBackgroundSubtractorCNT()
  print('Algoritmo de subtração de fundo inválido. Usando GMG como padrão.')
  sys.exit(1)

# Parametros minimos para detecção de veiculos
w_min = 30        # Largura mínima do retângulo
h_min = 30        # Altura mínima do retângulo
offset = 10       # Margem de erro no cruzamento de retângulos
linha_ROI = 500   # Posição da linha de contagem
carros = 0        # Contador de veículos

# Calcula o centroide do retângulo de um objeto detectado
def centroide(x, y, w, h):
  cx = x + w // 2
  cy = y + h // 2
  return cx, cy

# Inicialização da captura de vídeo e subtrator de fundo
cap = cv2.VideoCapture(VIDEO)
background_subtractor = subtractor(algorithm_type)

# Loop principal de leitura e processamento dos frames do vídeo
def main():
  while cap.isOpened():
    ok, frame = cap.read()

    if not ok:
      print('Não foi possível ler o vídeo ou o vídeo terminou.')
      break

    # Redimensiona o frame para melhorar a performance
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Aplica subtração de fundo e filtros morfológicos
    mask = background_subtractor.apply(frame)
    mask_filter = filter(mask, 'combine')
    cars_after_mask = cv2.bitwise_and(frame, frame, mask=mask_filter)

    # Exibe os resultados
    cv2.imshow('Frame', frame)
    cv2.imshow('Mascara', mask)
    cv2.imshow('Mascara filtrada', mask_filter)
    cv2.imshow('Carros detectados', cars_after_mask)

    # Encerra se pressionar 'c'
    if cv2.waitKey(1) & 0xFF == ord('c'):
      break

  # Libera os recursos
  cap.release()
  cv2.destroyAllWindows()

# Executa a função principal
if __name__ == '__main__':
  main()