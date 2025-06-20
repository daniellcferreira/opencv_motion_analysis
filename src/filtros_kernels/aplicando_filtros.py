import numpy as np
import cv2
import sys

# Caminho do video de entrada
VIDEO = 'data/Ponte.mp4'

# Lista de algoritmos disponíveis
algorithms = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']
algorithm_type = algorithms[0]  # GMG é o algoritmo padrão

# Função que retorn o kernel apropriado conforme o tipo de filtro
def kernel(KERNEL_TYPE):
  if KERNEL_TYPE == 'dilation':
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Elipse para dilatação
  if KERNEL_TYPE == 'opening':
    return np.ones((3, 3), np.uint8)  # Quadrado simples para abertura
  if KERNEL_TYPE == 'closing':
    return np.ones((3, 3), np.uint8)  # Quadrado simples para fechamento
  
# Função que aplica o filtro morfológico selecionado sobre a imagem binária
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
  
# Função que retorna o objeto de subtração de fundo conforme o tipo de algoritmo escolhido
def subtractor(algorithm):
  if algorithm == 'GMG':
    return cv2.bgsegm.createBackgroundSubtractorGMG()
  elif algorithm == 'MOG2':
    return cv2.createBackgroundSubtractorMOG2()
  elif algorithm == 'MOG':
    return cv2.bgsegm.createBackgroundSubtractorMOG()
  elif algorithm == 'KNN':
    return cv2.createBackgroundSubtractorKNN()
  elif algorithm == 'CNT':
    return cv2.bgsegm.createBackgroundSubtractorCNT()
  else:
    print(f"Algoritmo {algorithm} não reconhecido.")
    sys.exit(1)

# Inicializa a captura de vídeo e o algoritmo de subtração de fundo
cap = cv2.VideoCapture(VIDEO)
background_subtractor = subtractor(algorithm_type)

# Função principal que processa cada frame do vídeo
def main():
  while cap.isOpened():
    ok, frame = cap.read()

    if not ok:
      print("Não foi possível ler o frame do vídeo.")
      break

    # Redimensiona o frame para melhorar a performance
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Aplica a subtração de fundo
    mask = background_subtractor.apply(frame)

    # Aplica a sequencia de filtros morfológicos na máscara
    mask_filter = filter(mask, 'combine')

    # Aplica a máscara filtrada no frame original
    result = cv2.bitwise_and(frame, frame, mask=mask_filter)

    # Exibe os resultados
    cv2.imshow('Frame', frame)
    cv2.imshow('Mascara', mask)
    cv2.imshow('Mascara Filtrada', mask_filter)
    cv2.imshow('Resultado', result)

    # Pressiona 'c' para encerrar a execução
    if cv2.waitKey(1) & 0xFF == ord('c'):
      break

  # Libera os recursos
  cap.release()
  cv2.destroyAllWindows()

# Executa a função principal se o script for executado diretamente
if __name__ == "__main__":
  main()