import cv2
import sys

# Caminho absoluto do vídeo de entrada
VIDEO = 'data/Ponte.mp4'

# Lista de algoritmos de subtração de fundo
algorithm_types = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']

# Função que retorna o subtrator de fundo conforme o algoritmo escolhido
def subtractor(algorithm_type):
  if algorithm_type == 'GMG':
    return cv2.bgsegm.createBackgroundSubtractorGMG()
  if algorithm_type == 'MOG':
    return cv2.bgsegm.createBackgroundSubtractorMOG()
  if algorithm_type == 'MOG2':
    return cv2.createBackgroundSubtractorMOG2()
  if algorithm_type == 'KNN':
    return cv2.createBackgroundSubtractorKNN()
  if algorithm_type == 'CNT':
    return cv2.bgsegm.createBackgroundSubtractorCNT()

  print('Detector inválido')
  sys.exit(1)

# Inicializa captura de vídeo
cap = cv2.VideoCapture(VIDEO)

# Cria instâncias dos subtratores para todos os algoritmos
background_subtractor = [subtractor(a) for a in algorithm_types]

# Função principal
def main():
  frame_number = 0

  while cap.isOpened():
    ok, frame = cap.read()

    if not ok:
      print('Erro na captura')
      break

    frame_number += 1

    # Reduz a resolução para agilizar o processamento
    frame = cv2.resize(frame, (0, 0), fx=0.35, fy=0.35)

    # Aplica os algoritmos de subtração de fundo
    gmg = background_subtractor[0].apply(frame)
    mog2 = background_subtractor[1].apply(frame)
    mog = background_subtractor[2].apply(frame)
    knn = background_subtractor[3].apply(frame)
    cnt = background_subtractor[4].apply(frame)

    # Exibe o frame original e as máscaras de cada algoritmo
    cv2.imshow('Original', frame)
    cv2.imshow('GMG', gmg)
    cv2.imshow('MOG2', mog2)
    cv2.imshow('MOG', mog)
    cv2.imshow('KNN', knn)
    cv2.imshow('CNT', cnt)

    # Aguarda tecla; encerra ao pressionar ESC (27)
    if cv2.waitKey(0) & 0xFF == 27:
      break

main()
