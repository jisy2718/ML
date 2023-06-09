# XGBoost와 사이킷런을 활용한 그레이디언트 부스팅

<a href="https://tensorflow.blog/handson-gb"><img src="cover.jpg" alt="Hands-On Gradient Boosting with XGBoost and scikit-learn" height="400px" align="right" border="1" style="margin-left:10px"></a>

이 깃허브 저장소는 <[XGBoost와 사이킷런을 활용한 그레이디언트 부스팅](https://tensorflow.blog/handson-gb)>(한빛미디어, 2022)의 코드를 담고 있습니다.

XGBoost는 업계에서 입증된 그레이디언트 부스팅을 위한 오픈소스 라이브러리로 수십억 개의 데이터 빠르고 효율적으로 처리할 수 있습니다.

이 책은 다음과 같은 내용을 다룹니다.
* 밑바닥부터 그레이디언트 부스팅 모델을 만들어 봅니다.
* 성능과 속도를 모두 만족시키는 XGBoost 회귀 모델과 분류 모델을 구축합니다.
* XGBoost 하이퍼파라미터를 미세 튜닝하면서 분산과 편향을 분석합니다.
* 누락된 값을 자동으로 처리하고 불균형한 데이터를 조정합니다.
* dart, 선형 모델, XGBoost 랜덤 포레스트 같은 다른 기본 학습기를 적용합니다.
* 한국어판 부록에서는 LightGBM, CatBoost, 사이킷런의 히스토그램 기반 그레이디언트 부스팅을 소개합니다.

온라인/오프라인 서점에서 판매중입니다! [Yes24](http://www.yes24.com/Product/Goods/108708980), [교보문고](http://www.kyobobook.co.kr/product/detailViewKor.laf?ejkGb=KOR&mallGb=KOR&barcode=9791162245392&orderClick=LOA&Kc=#N), [한빛미디어](https://www.hanbit.co.kr/store/books/look.php?p_code=B5725043400)




# 공부내용

## Ch01. 머신러닝개요
+ [결측치처리](./Ch01_머신러닝개요/Ch01.머신러닝개요.ipynb#1.2.3.-결측치-처리하기)

+ [선형회귀, XGBRegressor, 예측값 시각화](./Ch01_머신러닝개요/Ch01.머신러닝개요.ipynb#1.3.4.-선형회귀모형-만들기)

+ [CV, 로지스틱, XGBClassifier](./Ch01_머신러닝개요/Ch01.머신러닝개요.ipynb#1.4.6.-모델링)






## Ch02. 결정트리

+ [결정트리 분류 코드](./Ch02_결정트리/Ch02.결정트리.ipynb#2.2.1.-첫번째-결정트리-모델)

+ [결정트리 작동원리 & 트리플랏](./Ch02_결정트리/Ch02.결정트리.ipynb#2.2.2.-결정트리의-작동원리)

+ [분산과 편향 의미](./Ch02_결정트리/Ch02.결정트리.ipynb#2.3.-분산과편향)

+ [결정트리 회귀 코드 & 트리객체 attribute와 method](./Ch02_결정트리/Ch02.결정트리.ipynb#2.4.1.결정트리회귀모델)

+ [결정트리 하이퍼파라미터](./Ch02_결정트리/Ch02.결정트리.ipynb#2.4.2.하이퍼파라미터)
+ [결정트리 특성중요도](./Ch02_결정트리/Ch02.결정트리.ipynb#2.5.5.-특성-중요도)

+ [GridSearchCV](./Ch02_결정트리/Ch02.결정트리.ipynb#GridSearchCV)

+ [RandomizedSearchCV](./Ch02_결정트리/Ch02.결정트리.ipynb#2.5.3.-하이퍼-파라미터-선택)






## Ch03. 랜덤포레스트

+ [RF 분류&회귀 코드](./Ch03_배깅과RF/Ch03.배깅과랜덤포레스트.ipynb#3.2.1.-랜덤-포레스트-분류-모델)

+ [RF 매개변수](./Ch03_배깅과RF/Ch03.배깅과랜덤포레스트.ipynb#3.3.-랜덤-포레스트-매개변수)
  + oob_score : 사용하지 않은 샘플로 모델 평가하는 옵션
  +  warm_start : 적절한 n_estimators를 찾을 수 있음
  + 그 외
+ [하이퍼파라미터튜닝 - RandomizedSearchCV](./Ch03_배깅과RF/Ch03.배깅과랜덤포레스트.ipynb#3.4.4.-하이퍼파라미터튜닝)

+ [ExtraTreesClassifier](./Ch03_배깅과RF/Ch03.배깅과랜덤포레스트.ipynb#부록.-ExtraTreesClassifier-&-ExtraTreesRegressor)







## Ch04. Gradient Boosting to XGBoost

+ [Gradient Boosting 구현](./Ch04_GradientBoosting/Ch04.%20Gradient%20Boosting%20to%20XGBoost.ipynb#4.2.3.-그레디언트-부스팅-모델-만들기)
+ [Gradient Boosting 파라미터 튜닝](./Ch04_GradientBoosting/Ch04.%20Gradient%20Boosting%20to%20XGBoost.ipynb#4.3.-GradientBoosting-매개변수-튜닝)
  + [learning_rate & n_estimators : Plot 그리기](./Ch04_GradientBoosting/Ch04.%20Gradient%20Boosting%20to%20XGBoost.ipynb#4.3.1.-learning_rate)
  + [조기종료](./Ch04_GradientBoosting/Ch04.%20Gradient%20Boosting%20to%20XGBoost.ipynb#4.3.3.-validation_fraction-&-n_iter_no_change-&-tol)
  + [validation_fraction & n_iter_no_change & tol](./Ch04_GradientBoosting/Ch04.%20Gradient%20Boosting%20to%20XGBoost.ipynb#4.3.3.-validation_fraction-&-n_iter_no_change-&-tol)
  + [subsample & oob_improvement_](./Ch04_GradientBoosting/Ch04.%20Gradient%20Boosting%20to%20XGBoost.ipynb#4.3.5.-subsample-&-obb_improvement_)
    + 과적합을 막을 수 있음
    + 예를들어 `n_iter_no_change = 10`, `validation_fraction= 0.2`, `subsample=0.5` 를 쓴다면, `validation_fraction`에 의해 전체 데이터 중 80%를 훈련 데이터로 쓰고, subsample에 의해 80% 중 랜덤하게 절반만 이용해서 트리를 만드는 과정을 반복함
  + [기본학습기 매개변수](./Ch04_GradientBoosting/Ch04.%20Gradient%20Boosting%20to%20XGBoost.ipynb#4.3.4.-기본-학습기)
+ [Gradient Boosting Attributes](./Ch04_GradientBoosting/Ch04.%20Gradient%20Boosting%20to%20XGBoost.ipynb#4.3.2.-속성들) : `init_`, `estimators_`, `train_score_`, `oob_improvement_`
+ [시간 측정 방법](./Ch04_GradientBoosting/Ch04.%20Gradient%20Boosting%20to%20XGBoost.ipynb#4.4.4.-시간측정) : `%timeit, %%timeit`, `time.time()`
+ Plot 배경 옵션
  + `sns.set()`
