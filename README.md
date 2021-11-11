# 음원 특성 분석을 통한 음악 추천 AI


## 프로젝트에 사용한 기법 및 분석 과정 간단 소개


### 콘텐츠 기반 필터링(Content-Based Filtering)
곡의 멜로디, 분위기, 감정, 템포 등 여러 특징들이 유사한 곡을 찾아낼 수 있도록 콘텐츠기반필터링 분석방법에 기반하였다.


### librosa
Audio 분석에 사용되는 패키지이다. 곡의 분석을 위해 여러 특징들을 추출하는데 'librosa' 패키지를 사용하였다.


### AWS(Amazon Web Service) - Mysql
모델링에 사용하기 위한 약 1500곡의 데이터들을 AWS에 DB를 구축하여 저장하였다.


### DTW(Dynamic Time Warping)
두 개의 Sequence의 유사도를 측정하는 알고리즘이다. 시간이 지남에 따라 변화하는 곡의 멜로디를 비교하기 위하여 사용하였다.

### t-SNE(t-Stochastic Neighbor Embbeding)
차원 축소의 기법 중 하나이다. 다른 기법보다 안정적인 임베딩 학습결과를 보여준다. 차원의 저주를 방지하기 위해서 사용하였다.

### K-means Clustering
Machine Learning의 군집화 알고리즘 중 하나이다. 방대한 곡의 데이터들을 일일이 비교하기에는 연산량과 시간이 너무 많이 들기 때문에, 곡의 여러 특징들을 분석하여 유사한 곡들을 군집화시켜 연산량과 검색시간을 줄이기 위하여 사용하였다.

### XGBoost
여러 개의 Decision Tree를 조합하여 사용하는 Ensemble 알고리즘이다. Classification, Regression 문제를 모두 지원하며, 성능과 자원 효율이 좋은 Machine Learning 학습 라이브러리이다.

### DNN(Deep Neural Network)
은닉층을 2개 이상 지닌 학습 방법이다. 많은 데이터와 반복학습, 사전학습과 오차역전파 기법을 통해 현재 널리 사용되고 있는 신경망 학습방법이다.