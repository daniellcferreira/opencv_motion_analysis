import numpy as np
import cv2
import sys

# Caminho do video de entrada
VIDEO = 'data/Ponte.mp4'

# Lista de algoritmos disponíveis para subtração de fundo
algorithms = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']
algorithm_type = algorithms[1]  # MOG2 por padrão

# Função que retorna o kernel conforme o tipo de algoritmo escolhido
def kernel(KERNEL_TYPE):
  if KERNEL_TYPE == 'dilation':
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
  if KERNEL_TYPE == 'opening':
    return np.ones((3, 3), np.uint8)
  if KERNEL_TYPE == 'closing':
    return np.ones((3, 3), np.uint8)
  
# Função que aplica o filtro morfológico escolhido a imagem binária
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
  
# Função que cria o objeto de subtração de fundo conforme o algoritmo escolhido
def subtractor(algorithm_type):
  if algorithm_type == 'GMG':
    return cv2.bgsegm.createBackgroundSubtractorGMG()
  elif algorithm_type == 'MOG2':
    return cv2.createBackgroundSubtractorMOG2()
  elif algorithm_type == 'MOG':
    return cv2.bgsegm.createBackgroundSubtractorMOG()
  elif algorithm_type == 'KNN':
    return cv2.createBackgroundSubtractorKNN()
  elif algorithm_type == 'CNT':
    return cv2.bgsegm.createBackgroundSubtractorCNT()
  else:
    print(f"Algoritmo {algorithm_type} não suportado.")
    sys.exit(1)

# Inicializa leitura do vídeo
cap = cv2.VideoCapture(VIDEO)
background_subtractor = subtractor(algorithm_type)

# Função principal que processa o vídeo
def main():
  while cap.isOpened():
    ok, frame = cap.read()

    if not ok:
      print("Não foi possível ler o vídeo ou o vídeo terminou.")
      break

    # Reduz o tamanho do frame para acelerar o processamento
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Aplica a subtração de fundo e filtros morfológicos
    mask = background_subtractor.apply(frame)
    mask_filter = filter(mask, 'combine')
    result = cv2.bitwise_and(frame, frame, mask=mask_filter)

    # Exibe os resultados
    cv2.imshow('Frame Original', frame)
    cv2.imshow('Mascara', mask)
    cv2.imshow('Mascara Filtrada', mask_filter)
    cv2.imshow('Resultado', result)

    # Tecla 'c' para encerra o loop
    if cv2.waitKey(1) & 0xFF == ord('c'):
      break 

  # Libera os recursos
  cap.release()
  cv2.destroyAllWindows()

# Executa a função principal se o script for executado diretamente
if __name__ == "__main__":
  main()