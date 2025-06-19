import numpy as np
import cv2
import sys

# Caminho do video de entrada
VIDEO = 'data/Ponte.mp4'

# Lista de algoritmos de subtração de fundo disponíveis
algorithms = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']
algorithm_type = algorithms[1]  # MOG2

# Função que retorna o kernel estrutural usando em operações morfológicas
def kernel(KERNEL_TYPE):
  if KERNEL_TYPE == 'dilation':
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
  elif KERNEL_TYPE == 'opening':
    return np.ones((3, 3), np.uint8)
  elif KERNEL_TYPE == 'closing':
    return np.ones((3, 3), np.uint8)
  else:
    print('Tipo de kernel desconhecido. Usando kernel de dilatação por padrão.')
    return None

# Exibe os kernels
print("Dilatação:")
print(kernel('dilation'))

print("Abertura:")
print(kernel('opening'))

print("Fechamento:")
print(kernel('closing'))

# Função que retorn o subtrator de fundo confome o algoritmo escolhido
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
    print('Algoritmo desconhecido. Usando MOG2 por padrão.')
    sys.exit(1)

# Captura do vídeo
cap = cv2.VideoCapture(VIDEO)

# Verifica se o vídeo foi aberto corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    sys.exit(1)

# Inicializa o subtrator de fundo
background_subtractor = subtractor(algorithm_type)

# Função principal para processar o vídeo
def main():
  while cap.isOpened():
    ok, frame = cap.read()

    if not ok:
      print("Fim do vídeo ou erro na leitura do frame.")
      break

    # Redimensiona o frame
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Aplica o subtrator de fundo
    mask = background_subtractor.apply(frame)

    # Exibe o resultado
    cv2.imshow('Frame', frame)
    cv2.imshow('Máscara', mask)

    # Encerra com a tecla 'c'
    if cv2.waitKey(1) & 0xFF == ord('c'):
      break

  # Libera os recursos
  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()