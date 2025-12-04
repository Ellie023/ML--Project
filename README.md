# ML--Project
![KakaoTalk_20241203_155334725](https://github.com/user-attachments/assets/dc24a7fd-aa34-4891-8c9c-03293ca71f14)


# Advanced Hybrid Movie Recommendation System

## 1\. Project Overview

This project implements a comprehensive movie recommendation engine that integrates **Content-Based Filtering**, **Item-Based Collaborative Filtering**, and **Visual Analysis** of movie posters. By combining metadata analysis with deep learning-based image feature extraction, the system aims to solve the cold-start problem and provide more diverse and accurate recommendations.

The core innovation lies in the **Hybrid Approach**, which ensembles traditional textual/numerical features with visual clusters derived from movie posters using **ResNet50**, alongside user behavior patterns.

## 2\. Technical Architecture

The system operates on a multi-stage pipeline:

### A. Data Preprocessing & Feature Engineering

  * **Structured Data**: Handling missing values, encoding genres, and normalizing numerical features (runtime, ratings).
  * **Textual Data**: Processing movie keywords and actor information using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
  * **Dimensionality Reduction**: Applying **SVD (Singular Value Decomposition)** to reduce the sparsity of the TF-IDF matrices.

### B. Visual Feature Extraction (Deep Learning)

  * **Model**: **ResNet50** (pre-trained on ImageNet).
  * **Process**: Extracting high-dimensional feature vectors from movie posters.
  * **Clustering**: Applying **K-Means Clustering** to group movies based on visual aesthetic similarities (e.g., color palette, composition).

### C. Recommendation Models

1.  **Content-Based Filtering (CBF)**: Calculates similarity between movies based on genres, keywords, and poster clusters using Cosine Similarity and **FAISS** for efficient retrieval.
2.  **Item-Based Collaborative Filtering (CF)**: Utilizes user-item interaction matrices to find similar items based on rating patterns.
3.  **Hybrid Model**: Computes a weighted average of scores from CF, CBF, and Poster Clustering to generate the final ranking.

## 3\. Project Structure & File Descriptions

The codebase is organized into specific modules for preprocessing, modeling, and evaluation.

| Module | Notebook File | Description |
| :--- | :--- | :--- |
| **Preprocessing** | `기본전처리.ipynb` | Basic data cleaning, merging movie metadata with poster URLs. |
| | `actor전처리.ipynb` | Processing and encoding actor names and casting information. |
| **Visual Analysis** | `poster_restnet50.ipynb` | implementation of **ResNet50** to extract feature vectors from poster images. |
| | `poster_clustering.ipynb` | Performing **K-Means clustering** on extracted image features to categorize visual styles. |
| | `poster.ipynb` | Data handling for poster datasets and mapping images to movie IDs. |
| **Modeling** | `contentbased.ipynb` | Implementation of Content-Based Filtering using TF-IDF and SVD. |
| | `item_based_final.ipynb` | Implementation of Item-Based Collaborative Filtering. |
| | `아이템_추천.ipynb` | Sandbox for testing item-based recommendation logic. |
| **Hybrid & Eval** | `하이브리드 추천.ipynb` | **Core Logic**: Ensembling CF, Content, and Cluster scores. |
| | `하이브리드성능.ipynb` | Performance evaluation of the Hybrid model vs. individual baselines. |
| **Analytics** | `user.ipynb` | User segmentation and behavior analysis. |
| | `기학 리뷰 분석.ipynb` | Exploratory Data Analysis (EDA) on critic reviews. |

## 4\. Installation & Requirements

To replicate the environment, ensure the following dependencies are installed:

```bash
# Data Manipulation
pip install pandas numpy

# Machine Learning & Similarity Search
pip install scikit-learn faiss-cpu

# Deep Learning (for ResNet50)
pip install tensorflow keras  # or torch/torchvision depending on the specific implementation

# Image Processing
pip install pillow requests

# Visualization
pip install matplotlib seaborn
```

## 5\. Methodology Details

### Visual Clustering with ResNet50

We utilized a transfer learning approach to analyze movie posters. By removing the final classification layer of ResNet50, we obtained a feature embedding for each poster.

  * **Input**: Raw image files of movie posters.
  * **Output**: 2048-dimensional feature vectors.
  * **Clustering**: These vectors were clustered into $K$ groups to identify movies with similar visual vibes (e.g., "Dark/Horror", "Bright/Comedy").

### The Hybrid Algorithm

The final recommendation score $S_{hybrid}$ is calculated as:

$$S_{hybrid} = \alpha \cdot S_{CF} + \beta \cdot S_{Content} + \gamma \cdot S_{Visual}$$

Where:

  * $S_{CF}$: Similarity score from Collaborative Filtering.
  * $S_{Content}$: Similarity score from Metadata (Genre, Keywords).
  * $S_{Visual}$: Similarity score based on Poster Clusters.
  * $\alpha, \beta, \gamma$: Hyperparameters determined through validation.

## 6\. Evaluation Metrics

The system's performance is evaluated using standard Information Retrieval metrics:

  * **RMSE (Root Mean Square Error)**: To measure rating prediction accuracy.
  * **MAE (Mean Absolute Error)**: To measure the average magnitude of errors.
  * **Precision @ K**: To evaluate the relevance of the top-K recommended items.
  * **Recall**: To measure the coverage of relevant items retrieved.
  * **F1-Score**: The harmonic mean of Precision and Recall.

## 7\. Usage

1.  **Preprocess Data**: Run `기본전처리.ipynb` and `actor전처리.ipynb` to prepare the dataset.
2.  **Extract Visual Features**: Run `poster_restnet50.ipynb` to generate embeddings and `poster_clustering.ipynb` to assign clusters.
3.  **Train Models**: Execute `contentbased.ipynb` and `item_based_final.ipynb` to generate similarity matrices.
4.  **Generate Recommendations**: Use `하이브리드 추천.ipynb` to input a user ID or movie title and receive the final recommendation list.



-----

# 고급 하이브리드 영화 추천 시스템 (Advanced Hybrid Movie Recommendation System)

## 1\. 프로젝트 개요

본 프로젝트는 **콘텐츠 기반 필터링(Content-Based Filtering)**, **아이템 기반 협업 필터링(Item-Based Collaborative Filtering)**, 그리고 **영화 포스터의 시각적 분석**을 통합한 포괄적인 영화 추천 엔진을 구현합니다. 메타데이터 분석과 딥러닝 기반의 이미지 특성 추출을 결합하여 콜드 스타트(Cold-Start) 문제를 해결하고, 더욱 다양하고 정확한 추천을 제공하는 것을 목표로 합니다.

**핵심 혁신**은 기존의 텍스트/수치 특성 데이터와 ResNet50을 통해 추출한 영화 포스터의 시각적 클러스터(Visual Clusters), 그리고 사용자 행동 패턴을 앙상블(Ensemble)하는 **하이브리드 접근 방식**에 있습니다.

## 2\. 기술 아키텍처

이 시스템은 다음과 같은 다단계 파이프라인으로 작동합니다.

**A. 데이터 전처리 및 특성 엔지니어링 (Data Preprocessing & Feature Engineering)**

  * **정형 데이터:** 결측치 처리, 장르 인코딩, 수치형 특성(상영 시간, 평점) 정규화.
  * **텍스트 데이터:** TF-IDF(Term Frequency-Inverse Document Frequency)를 사용하여 영화 키워드 및 배우 정보 처리.
  * **차원 축소:** SVD(특이값 분해)를 적용하여 TF-IDF 행렬의 희소성(Sparsity) 감소.

**B. 시각적 특성 추출 (Deep Learning)**

  * **모델:** ResNet50 (ImageNet으로 사전 학습됨).
  * **과정:** 영화 포스터에서 고차원 특성 벡터 추출.
  * **클러스터링:** K-Means 클러스터링을 적용하여 시각적 미적 유사성(예: 색감, 구도 등)을 기반으로 영화 그룹화.

**C. 추천 모델 (Recommendation Models)**

  * **콘텐츠 기반 필터링 (CBF):** 코사인 유사도와 FAISS를 사용하여 장르, 키워드, 포스터 클러스터를 기반으로 영화 간 유사도 계산 및 효율적인 검색 수행.
  * **아이템 기반 협업 필터링 (CF):** 사용자-아이템 상호작용 행렬을 활용하여 평점 패턴에 따른 유사 아이템 탐색.
  * **하이브리드 모델:** CF, CBF, 포스터 클러스터링 점수의 가중 평균을 계산하여 최종 순위 생성.

## 3\. 프로젝트 구조 및 파일 설명

코드베이스는 전처리, 모델링, 평가를 위한 특정 모듈로 구성되어 있습니다.

| 모듈 | 노트북 파일 | 설명 |
| :--- | :--- | :--- |
| **Preprocessing** | `기본전처리.ipynb` | 기본 데이터 정제, 영화 메타데이터와 포스터 URL 병합. |
| | `actor전처리.ipynb` | 배우 이름 및 캐스팅 정보 처리 및 인코딩. |
| **Visual Analysis** | `poster_restnet50.ipynb` | 포스터 이미지에서 특성 벡터를 추출하기 위한 ResNet50 구현. |
| | `poster_clustering.ipynb` | 추출된 이미지 특성에 K-Means 클러스터링을 수행하여 시각적 스타일 분류. |
| | `poster.ipynb` | 포스터 데이터셋 처리 및 이미지를 영화 ID에 매핑. |
| **Modeling** | `contentbased.ipynb` | TF-IDF와 SVD를 사용한 콘텐츠 기반 필터링 구현. |
| | `item_based_final.ipynb` | 아이템 기반 협업 필터링 구현. |
| | `아이템_추천.ipynb` | 아이템 기반 추천 로직 테스트를 위한 샌드박스. |
| **Hybrid & Eval** | `하이브리드 추천.ipynb` | **핵심 로직:** CF, 콘텐츠, 클러스터 점수 앙상블. |
| | `하이브리드성능.ipynb` | 하이브리드 모델 대 개별 베이스라인 모델 성능 비교 평가. |
| **Analytics** | `user.ipynb` | 사용자 세분화(Segmentation) 및 행동 분석. |
| | `기학 리뷰 분석.ipynb` | 평론가 리뷰에 대한 탐색적 데이터 분석(EDA). |

## 4\. 설치 및 요구사항

환경을 복제하려면 다음 종속성들이 설치되어 있는지 확인하십시오.

```bash
# 데이터 조작 (Data Manipulation)
pip install pandas numpy

# 머신러닝 및 유사도 검색 (Machine Learning & Similarity Search)
pip install scikit-learn faiss-cpu

# 딥러닝 (ResNet50용)
pip install tensorflow keras  # 또는 구현 방식에 따라 torch/torchvision

# 이미지 처리 (Image Processing)
pip install pillow requests

# 시각화 (Visualization)
pip install matplotlib seaborn
```

## 5\. 상세 방법론

### ResNet50을 이용한 시각적 클러스터링

영화 포스터 분석을 위해 전이 학습(Transfer Learning) 접근 방식을 사용했습니다. ResNet50의 마지막 분류 레이어를 제거하여 각 포스터에 대한 특성 임베딩(Feature Embedding)을 획득했습니다.

  * **입력:** 영화 포스터 원본 이미지 파일.
  * **출력:** 2048차원 특성 벡터.
  * **클러스터링:** 이 벡터들을 $K$개의 그룹으로 클러스터링하여 유사한 시각적 분위기(예: "어두운/호러", "밝은/코미디")를 가진 영화들을 식별합니다.

### 하이브리드 알고리즘

최종 추천 점수 $S_{hybrid}$는 다음과 같이 계산됩니다:

$$S_{hybrid} = \alpha \cdot S_{CF} + \beta \cdot S_{Content} + \gamma \cdot S_{Visual}$$

여기서:

  * $S_{CF}$: 협업 필터링(Collaborative Filtering) 유사도 점수
  * $S_{Content}$: 메타데이터(장르, 키워드) 유사도 점수
  * $S_{Visual}$: 포스터 클러스터 기반 유사도 점수
  * $\alpha, \beta, \gamma$: 검증(Validation)을 통해 결정된 하이퍼파라미터 가중치

## 6\. 평가 지표

시스템의 성능은 표준 정보 검색(Information Retrieval) 지표를 사용하여 평가됩니다.

  * **RMSE (Root Mean Square Error):** 평점 예측 정확도 측정.
  * **MAE (Mean Absolute Error):** 오차의 평균 크기 측정.
  * **Precision @ K:** 상위 K개 추천 아이템의 관련성 평가.
  * **Recall:** 관련 아이템을 얼마나 잘 찾아냈는지(커버리지) 측정.
  * **F1-Score:** 정밀도(Precision)와 재현율(Recall)의 조화 평균.

## 7\. 사용법

1.  **데이터 전처리:** `기본전처리.ipynb`와 `actor전처리.ipynb`를 실행하여 데이터셋을 준비합니다.
2.  **시각적 특성 추출:** `poster_restnet50.ipynb`를 실행하여 임베딩을 생성하고, `poster_clustering.ipynb`로 클러스터를 할당합니다.
3.  **모델 훈련:** `contentbased.ipynb`와 `item_based_final.ipynb`를 실행하여 유사도 행렬을 생성합니다.
4.  **추천 생성:** `하이브리드 추천.ipynb`를 사용하여 사용자 ID나 영화 제목을 입력하고 최종 추천 목록을 확인합니다.
