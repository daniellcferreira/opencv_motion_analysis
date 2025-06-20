import numpy as np
import cv2
import sys

# Caminho do video de entrada
VIDEO = 'data/Ponte.mp4'

# Lista com os nomes dos algoritmos disponíveis
algorithms = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']
algorithm_type = algorithms[1]  # MOG2 é o algoritmo padrão

# Função que retorna diferentes tipos de kernel morfológico
def kernel(KERNEL_TYPE):
  if KERNEL_TYPE == 'dilation':
    # Elipse para dilatação (ajuda a preencher lacunas no objeto)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
  if KERNEL_TYPE == 'opening':
    # Quadrado simples (remoção de ruídos pequenos)
    kernel = np.ones((3, 3), np.uint8)
  if KERNEL_TYPE == 'closing':
    # Quadrado simples (preenchimento de buracos no objeto)
    kernel = np.ones((3, 3), np.uint8)
  return kernel

# Aplica o filtro morfológico selecionado sobre a imagem binária
def filter(img, filter):
  if filter == 'closing':
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel('closing'), iterations=2)
  if filter == 'opening':
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel('opening'), iterations=2)
  if filter == 'dilation':
    return cv2.dilate(img, kernel('dilation'), iterations=2)
  if filter == 'combine':
    # Sequencia de fechamento, abertura e dilatação para refinar a mascara
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel('closing'), iterations=2)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel('opening'), iterations=2)
    dilation = cv2.dilate(opening, kernel('dilation'), iterations=2)
    return dilation
  
# Retorna o objeto de subtração de fundo conforme o tipo de algoritmo escolhido
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

# Função principal que processa so frame do vídeo
def main():
  while cap.isOpened():
    ok, frame = cap.read()

    if not ok:
      print("Não foi possível ler o frame do vídeo, ou frames acabaram.")
      break

    # Redimensiona o frame para facilitar o processamento
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Aplica a subtração de fundo para detectar movimento
    mask = background_subtractor.apply(frame)

    # Exibe o frame original e a máscara binária resultante
    cv2.imshow('Frame', frame)
    cv2.imshow('Mascara', mask)

    # Pressiona 'c' para encerrar a execução
    if cv2.waitKey(1) & 0xFF == ord('c'):
      break

  # libera os recursos
  cap.release()
  cv2.destroyAllWindows()

# Ponto de entrada do script
if __name__ == "__main__":
  main()

