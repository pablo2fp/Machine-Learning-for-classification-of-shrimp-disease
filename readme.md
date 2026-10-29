# 🦐 Shrimp Disease Detection: MobileNetV3, Feature Extraction & Statistical Benchmarking

Pipeline completo de Visão Computacional e Aprendizagem Automática (*Machine Learning*) voltado para a classificação patológica binária em camaricultura (detecção de animais saudáveis vs. infectados com Mancha Branca / WSSV e Brânquias Enegrecidas / Black Gill).

O projeto destaca-se pelo rigor metodológico contra *data leakage*, comparação empírica de múltiplos classificadores via *Nested Cross-Validation*, comprovação estatística paramétrica/não paramétrica dos resultados e explicabilidade visual por meio de mapas de calor Grad-CAM.

---

## 📌 Principais Destaques de Engenharia e Metodologia

- **Divisão com Prevenção de Fuga por Hashing Percetual (`imagehash`):** 
  Agrupamento de imagens quase-duplicadas via *pHash* antes da separação, aplicando `GroupShuffleSplit` para garantir que variantes do mesmo espécime ou capturas em sequência nunca partilhem conjuntos de treino, validação e teste.
- **Extração de Embeddings com Deep Learning:**
  Utilização da arquitetura **MobileNetV3Large** pré-treinada na ImageNet como extrator denso de características (*Global Average Pooling*).
- **Redução de Dimensionalidade (PCA vs. Dense Espacial):**
  Avaliação da variância acumulada (curvas de 90%, 95% e 99%) para verificar o impacto de compressão de vetores de características latentes.
- **Benchmark com Classificadores Clássicos:**
  Comparação sistemática entre **SGDClassifier (Log-Loss)**, **Support Vector Machines (SVC RBF)**, **Random Forest** e **XGBoost**, ambos nas abordagens com e sem PCA.
- **Validação Estatística Rigorosa:**
  Comprovação de significância com testes de **Shapiro-Wilk** (normalidade), **Levene** (homocedasticidade), **ANOVA one-way** com post-hoc de **Tukey**, e teste não paramétrico de **Friedman** com post-hoc de **Nemenyi** para eficiência de tempo computacional.
- **CNN Cirúrgica Unificada & Fine-Tuning:**
  Integração do gargalo latente (*PCA-like projection*) com cabeça densa de classificação ajustada em duas etapas (camadas congeladas e posterior afinação das últimas 10 camadas da convolução).
- **Explicabilidade Visual (XAI) com Grad-CAM:**
  Mapeamento de ativação ponderada por gradiente na última camada convolucional (`conv_1`) para auditar as regiões biológicas anatómicas que sustentam o diagnóstico.
---

## 🛠️ Tecnologias e Bibliotecas
Linguagem: Python

Deep Learning & Visão Computacional: TensorFlow, Keras (MobileNetV3Large), OpenCV, PIL

Machine Learning & Pré-processamento: Scikit-Learn (Pipeline, StandardScaler, PCA, GroupShuffleSplit, StratifiedKFold), XGBoost

Validação Estatística: SciPy (stats.shapiro, stats.levene, stats.f_oneway, stats.friedmanchisquare), Statsmodels (pairwise_tukeyhsd), Scikit-Posthocs (posthoc_nemenyi_friedman)

Deduplicação de Imagens: ImageHash (phash)

Visualização de Dados: Matplotlib, Seaborn

---
## 🔬 Benchmark de Modelos e Avaliação Estatística:
O processo de treino foi conduzido com validação cruzada aninhada (Nested Cross-Validation com $k=10$ folds externos e $k=3$ internos)

1. Resultados: 

Modelo | Acurácia Média (%) | Precisão Macro (%) | Recall Macro (%) | F1-Score Macro (%) | Tempo de Treino (s)
 :--- | :---: | :---: | :---: | :---: | :---: |
SGD Classifier sem PCA | 85 | 85 | 82 | 83 | 0.11
SGD Classifier + PCA (400 comp.) | 86 | 85 | 85 | 85 | 1.23
SVC (RBF) sem PCA | 89 | 89 | 88 | 88 | 3.65
SVC + PCA (200 comp.) | 90 | 90 | 89 | 89 | 2.90
Random Forest sem PCA | 83 | 82 | 81 | 81 | 6.02
Random Forest + PCA (300 comp.) | 82 | 82 | 78 | 79 | 8.60
XGBoost sem PCA | 85 | 84 | 83 | 83 | 13.03
XGBoost + PCA (300 comp.) | 84 | 84 | 81 | 82 | 9.17


2. Inferência Estatística:
* Acurácia, Precisão, Recall e F1-Score: Avaliados via ANOVA one-way após verificação de normalidade residual (Shapiro-Wilk) e homogeneidade de variâncias (Levene). O teste post-hoc de Tukey identificou que a compressão vetorial permitiu redução substancial de dimensionalidade mantendo paridade estatística de desempenho ($p > 0.05$).

* Custo Computacional: A análise via teste de Friedman confirmou diferença estatisticamente significativa nos tempos de treino ($p < 0.001$), onde o teste de Nemenyi comprovou que modelos baseados em gradiente linear e redução PCA oferecem latência de processamento significativamente inferior.

---
## 🩺 Interpretabilidade com Grad-CAM
A validação de modelos em patologias animais exige a certeza de que a rede não está a focar o fundo da imagem ou artefactos de iluminação:

Utilizou-se a técnica Grad-CAM (Gradient-weighted Class Activation Mapping) extraindo os mapas da última camada de convolução (conv_1).

Os mapas de ativação comprovaram foco nas zonas de necrose branquial e nas manchas cuticulares características das doenças estudadas (WSSV / Mancha Branca e Black Gill), garantindo validade biológica ao modelo.

<img width="912" height="455" alt="image" src="https://github.com/user-attachments/assets/6aeade70-12e0-4fe1-87a2-a945be004363" />

<img width="912" height="455" alt="image" src="https://github.com/user-attachments/assets/78fc0f7e-c022-49d5-bbab-d137480484bd" />

<img width="912" height="455" alt="image" src="https://github.com/user-attachments/assets/82e44616-0a1c-4c61-b3c9-5a51a2207c89" />



---
💻 Como Reproduzir o Projeto

1. Clone o repositório:
```text
git clone [https://github.com/pablo2fp/mobilenet-shrimp-disease-detection.git](https://github.com/pablo2fp/mobilenet-shrimp-disease-detection.git)
cd mobilenet-shrimp-disease-detection
```

2. Instale as dependências necessárias:
```text
pip install -r requirements.txt
```

3. Execute o script principal:
```text
python mobilenet_binario_análise_de_imagem_de_camarão.py
```
---



## 📊 Arquitetura do Pipeline

```text
Imagens Brutas (224x224x3)
     │
     ├──> Cálculo de pHash (Agrupamento anti-leakage)
     │         │
     │         └──> GroupShuffleSplit (Treino / Validação / Teste)
     │
     ├──> MobileNetV3Large (Extração de Embeddings de 960 dimensões)
     │         │
     │         ├──> Benchmark Tradicional (Nested CV & Testes de Hipótese)
     │         │     ├── SGD | SVC | Random Forest | XGBoost
     │         │     └── Avaliação com e sem PCA (200 a 400 componentes)
     │         │
     │         └──> CNN Cirúrgica End-to-End
     │               ├── Projeção Linear (Redução de dimensionalidade)
     │               ├── Batch Normalization + Dropout
     │               ├── Fine-tuning das camadas convolucionais finais
     │               └── Classificador Sigmóide Binário
     │
     └──> Interpretabilidade com Grad-CAM (Inspeção visual anatómica)

