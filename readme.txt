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

---
🔬 Benchmark de Modelos e Avaliação EstatísticaO processo de treino foi conduzido com validação cruzada aninhada (Nested Cross-Validation com $k=10$ folds externos e $k=3$ internos)

---
🩺 Interpretabilidade com Grad-CAM
A validação de modelos em patologias animais exige a certeza de que a rede não está a focar o fundo da imagem ou artefactos de iluminação:

Utilizou-se a técnica Grad-CAM (Gradient-weighted Class Activation Mapping) extraindo os mapas da última camada de convolução (conv_1).

Os mapas de ativação comprovaram foco nas zonas de necrose branquial e nas manchas cuticulares características das doenças estudadas (WSSV / Mancha Branca e Black Gill), garantindo validade biológica ao modelo.

🛠️ Tecnologias e Bibliotecas
Linguagem: Python

Deep Learning & Visão Computacional: TensorFlow, Keras (MobileNetV3Large), OpenCV, PIL

Machine Learning & Pré-processamento: Scikit-Learn (Pipeline, StandardScaler, PCA, GroupShuffleSplit, StratifiedKFold), XGBoost

Validação Estatística: SciPy (stats.shapiro, stats.levene, stats.f_oneway, stats.friedmanchisquare), Statsmodels (pairwise_tukeyhsd), Scikit-Posthocs (posthoc_nemenyi_friedman)

Deduplicação de Imagens: ImageHash (phash)

Visualização de Dados: Matplotlib, Seaborn

---
💻 Como Reproduzir o Projeto
1. Clone o repositório:
Bash
git clone [https://github.com/pablo2fp/mobilenet-shrimp-disease-detection.git](https://github.com/pablo2fp/mobilenet-shrimp-disease-detection.git)
cd mobilenet-shrimp-disease-detection

2. Instale as dependências necessárias:
Bash
pip install -r requirements.txt

3. Execute o script principal:
Bash
python mobilenet_binario_análise_de_imagem_de_camarão.py
