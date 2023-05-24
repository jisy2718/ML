















## 2. Model



### 2.1. Tree

#### 2.1.1. 예측모델코드

```python
from sklearn.tree import DecisionTreeClassifier
# 분류 모델을 만듭니다.
clf = DecisionTreeClassifier(random_state=2)
clf.fit(X_train, y_train)

# 테스트 데이터에 대한 예측을 만듭니다.
y_pred = clf.predict(X_test)
```



#### 2.1.2. plot_tree

좌측화살표가 True, 우측 화살표가 False임

```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(13,8))
plot_tree(clf, max_depth=2, feature_names=list(X.columns), class_names=['0','1'], filled=True, rounded=True, fontsize=14)
plt.show()
```





## 3. metrics



### 3.1. accuracy

> 전체 예측 값 중, 정확하게 맞은 값의 개수를 측정

```python
from sklearn.metrics import accuracy_score
# 분류 모델을 만듭니다.
model = DecisionTreeClassifier(random_state=2)
model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측을 만듭니다.
y_pred = model.predict(X_test)

# 정확도를 계산합니다.
accuracy_score(y_pred, y_test)
```



### 3.2. mean_squared_error & rmse

```python
from sklearn.metrics import mean_squared_error
rmse_test = mean_squared_error(y_test, y_pred)**0.5
```







## 4. model_selection

### 4.1. GridSearchCV

파이썬라이브러리를 활용한 머신러닝의 5장을 참고



### 4.1.1. 매개변수

+ `refit`
  + `True`인 경우 교차 검증을 통해 찾은 최적의 매개변수를 사용해 (훈련 폴드와 검증 폴드를 합친) 전체 훈련 세트에서 최종 모델을 훈련하고, 이를 `best_estimator_`에 저장



#### 4.1.2. method &attribute

+ `fit()`

+ `predict()` : best_estimator_ 로 predict함

+ `best_estimator_`

+ `best_score_`

  

















