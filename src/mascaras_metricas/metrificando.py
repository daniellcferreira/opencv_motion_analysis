import cv2
import sys

# Caminho absoluto do vídeo de entrada
VIDEO = 'data/Ponte.mp4'

# Lista de algoritmos disponíveis para subtração de fundo
algorithms_types = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']
algorithms_type = algorithms_types[3]  # Seleciona o algoritmo a ser utilizado

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
e1 = cv2.getTickCount()  # Inicia contagem de tempo
cap = cv2.VideoCapture(VIDEO)
background_subtractor = subtractor(algorithms_type)

# Função principal que processa o vídeo
def main():
  frame_count = 0  # Contador de frames processados

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

    frame_count += 1  # Incrementa contador de frames

    # Interrompe ao pressionar 'c' ou ao atingir 300 frames
    if cv2.waitKey(1) & 0xFF == ord('c') or frame_count >= 300:
      break

    # Tempo total de processamento
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    print(f"Frames processados: {frame_count}, Tempo total: {time:.2f} segundos")

main()

    