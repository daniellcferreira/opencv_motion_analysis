import cv2
import sys

# Caminho absoluto do vídeo de entrada
VIDEO = 'data/Ponte.mp4'
# Lista de algoritmos disponíveis para subtração de fundo
algorithms_types = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']
# Seleciona o algoritmo a ser utilizado
algorithms_type = algorithms_types[3]

# Função que retorna o subtrator de fundo conforme o algoritmo escolhido
def subtractor(algorithms_type):
  if algorithms_type == 'GMG':
    return cv2.bgsegm.createBackgroundSubtractorGMG()
  if algorithms_type == 'MOG':
    return cv2.bgsegm.createBackgroundSubtractorMOG()
  if algorithms_type == 'MOG2':
    return cv2.createBackgroundSubtractorMOG2()
  if algorithms_type == 'KNN':
    return cv2.createBackgroundSubtractorKNN()
  if algorithms_type == 'CNT':
    return cv2.bgsegm.createBackgroundSubtractorCNT()

  raise ValueError(f"Algoritmo {algorithms_type} não implementado.")
  sys.exit(1)

# Inicializa captura e subtrator
cap = cv2.VideoCapture(VIDEO)
background_subtractor = subtractor(algorithms_type)

# Função principal que processa o vídeo
def main():
  while cap.isOpened():
    ok, frame = cap.read()

    if not ok:
      print("Não foi possível ler o vídeo ou o vídeo terminou.")
      break

    # Reduz dimensão do frame para acelerar o processamento
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Aplica o subtrator de fundo
    mask = background_subtractor.apply(frame)

    # Exibe o frame original e a máscara gerada
    cv2.imshow('Frame', frame)
    cv2.imshow('Máscara', mask)

    # Interrompe ao pressionar 'c'
    if cv2.waitKey(1) & 0xFF == ord('c'):
      break

main()