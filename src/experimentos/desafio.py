import numpy as np
import cv2
import sys
from random import randint
import csv
import os

# Cria pasta de saida, se não existir
os.makedirs('outputs', exist_ok=True)

# Cria arquivo CSV para armazenar a contagem de pixels de cada algoritmo
csv_path = os.path.join('outputs', 'Results.csv')
fp = open(csv_path, mode='w', newline='')
writer = csv.DictWriter(fp, fieldnames=['Frame', 'Pixel Count'])
writer.writeheader()

# Configurações visuais para as legendas sobre os frames
TEXT_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
BORDER_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SIZE = 1.2
TITLE_TEXT_POSITION = (100, 40)

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
  
  print(f'Algoritmo {algorithm_type} não implementado.')
  sys.exit(1)

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(VIDEO)

# Cria instâncias dos subtratores para todos os algoritmos
background_subtractor = [subtractor(a) for a in algorithm_types]

# Função principal que processa frame a frame
def main():
  frame_number = 0

  while cap.isOpened():
    ok, frame = cap.read()

    if not ok:
      print('Não foi possível ler o vídeo ou o vídeo terminou.')
      break

    frame_number += 1
    frame = cv2.resize(frame, (0, 0), fx=0.35, fy=0.35)

    # Aplica os algoritmos
    gmg = background_subtractor[0].apply(frame)
    mog2 = background_subtractor[1].apply(frame)
    mog = background_subtractor[2].apply(frame)
    knn = background_subtractor[3].apply(frame)
    cnt = background_subtractor[4].apply(frame)

    # Conta os pixels diferentes de zero (movimento detectado)
    gmg_count = np.count_nonzero(gmg)
    mog2_count = np.count_nonzero(mog2)
    mog_count = np.count_nonzero(mog)
    knn_count = np.count_nonzero(knn)
    cnt_count = np.count_nonzero(cnt)

    # Salva os dados no CSV
    writer.writerow({'Frame': 'GMG', 'Pixel Count': gmg_count})
    writer.writerow({'Frame': 'MOG2', 'Pixel Count': mog2_count})
    writer.writerow({'Frame': 'MOG', 'Pixel Count': mog_count})
    writer.writerow({'Frame': 'KNN', 'Pixel Count': knn_count})
    writer.writerow({'Frame': 'CNT', 'Pixel Count': cnt_count})

    # Insere texto em cada frame
    cv2.putText(gmg, 'GMG', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(mog2, 'MOG2', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(mog, 'MOG', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(knn, 'KNN', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(cnt, 'CNT', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)

    # Exibe os frames com as máscaras
    cv2.imshow('Original', frame)
    cv2.imshow('GMG', gmg)
    cv2.imshow('MOG2', mog2)
    cv2.imshow('MOG', mog)
    cv2.imshow('KNN', knn)
    cv2.imshow('CNT', cnt)

    # Aguarda tecla para avançar; ESC para sair
    k = cv2.waitKey(0) & 0xff
    if k == 27:  # ESC
      break

  # Libera recursos
  cap.release()
  cv2.destroyAllWindows()
  fp.close()


main()
