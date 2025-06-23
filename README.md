# Projeto de Processamento e Análise de Vídeo com OpenCV

[![Python Language](https://img.shields.io/badge/Python-Language%20Programming-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/) 
[![OpenCV Library](https://img.shields.io/badge/OpenCV-Computer%20Vision-brightgreen?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/) 
[![NumPy Package](https://img.shields.io/badge/NumPy-Numerical%20Computing-orange?style=flat-square&logo=NumPy&logoColor=white)](https://numpy.org/) 

---

## Descrição

Este projeto é uma coleção de scripts desenvolvidos em Python que utilizam a biblioteca OpenCV para realizar o processamento de vídeos focado na remoção de fundo, segmentação e análise de objetos em movimento.

A abordagem principal consiste na aplicação de diversos algoritmos de subtração de fundo para detecção de movimento em vídeos urbanos, com ênfase na contagem e análise de veículos. Além disso, são usados filtros morfológicos para melhorar a qualidade das máscaras binárias geradas, reduzindo ruídos e preenchendo lacunas em regiões de interesse.

O projeto também implementa técnicas para cálculo do frame mediano a partir de amostragem aleatória dos frames do vídeo, visando obter uma imagem representativa do fundo estático da cena, fundamental para aplicações de remoção de fundo mais robustas.

Diversos módulos possibilitam a comparação visual e quantitativa dos diferentes algoritmos de subtração de fundo, permitindo ao usuário avaliar qual deles se adapta melhor ao cenário analisado.

---

## Funcionalidades

- Captura e leitura de vídeos a partir de arquivos locais.
- Implementação de múltiplos algoritmos de subtração de fundo disponíveis no OpenCV:
  - GMG (Gaussian Mixture-based Background/Foreground Segmentation Algorithm)
  - MOG (Mixture of Gaussians)
  - MOG2 (Melhorias no MOG)
  - KNN (K-Nearest Neighbors)
  - CNT (Pixel-wise background/foreground segmentation method)
- Aplicação de filtros morfológicos variados para pós-processamento das máscaras de movimento:
  - Dilatação
  - Abertura
  - Fechamento
  - Combinação sequencial destes filtros para resultados mais refinados
- Cálculo e geração do frame mediano a partir de uma amostra aleatória de frames para obter o fundo estático.
- Visualização simultânea do frame original, da máscara binária e dos resultados filtrados em janelas OpenCV.
- Interface simples para interrupção da execução pelo usuário via tecla.
- Controle de resolução e redimensionamento dos frames para otimização de desempenho.
- Registro e exibição do tempo total de processamento e contagem de frames processados.
- Exportação do frame mediano como imagem estática para análise posterior.
- Modularidade no código permitindo fácil substituição ou adição de algoritmos e filtros.

---

## Tecnologias Abordadas

- **Python Language**: Linguagem de programação versátil e de alto nível, amplamente utilizada em ciência de dados, desenvolvimento web, automação e processamento de imagens.
- **OpenCV Library**: Biblioteca open source para visão computacional e processamento de imagens, fornecendo uma ampla gama de funcionalidades para análise e manipulação de vídeos e imagens.
- **NumPy Package**: Biblioteca fundamental para computação científica em Python, especializada em manipulação eficiente de arrays multidimensionais e operações matemáticas.
- **Sistemas de janelas OpenCV**: Para exibição em tempo real dos vídeos, máscaras e resultados do processamento.
- **Manipulação de arquivos e diretórios em Python**: Para carregamento e salvamento de vídeos e imagens, garantindo portabilidade e organização do projeto.
- **Controle de fluxo e interação com o usuário via teclado**: Para permitir a parada da execução a qualquer momento pelo usuário.
- **Conceitos de processamento de imagem digital**: Aplicação de técnicas morfológicas para melhorar a qualidade da segmentação, reduzindo ruídos e preenchendo falhas.
- **Técnicas avançadas de segmentação de vídeo**: Uso de algoritmos baseados em modelos estatísticos para separar fundo de objetos móveis em cenas urbanas.

---

Este projeto é ideal para quem deseja estudar e aplicar técnicas de visão computacional para análise de vídeos, especialmente em cenários de monitoramento e contagem automática de objetos em movimento, como veículos em vias públicas.
