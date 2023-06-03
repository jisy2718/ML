

[toc]



# 0. 모델일반

## 0.1. 분산과 편향

**분산**

- 분산이란 다른 훈련 데이터가 주어졌을 때, 모델이 얼마나 변화하는지를 나타냄

- 분산이 큰 모델은 과적합되기 쉬움

- 분산이 낮은 모델은 training data가 달라져도 모델이 크게 달라지지 않음

    

**편향**

- 편향이란 모델이 학습 데이터의 본질적인 패턴과 구조를 적절하게 파악하지 못하고, 일관되게 잘못된 가정을 하는 경향을 의미

- 편향이 큰 모델은 과소적합되기 쉬움

- 편향이 낮은 것은 오차가 크지 않다는 것

    

**편향-분산 trade off**

- 모델의 복잡성이 증가하면 편향은 감소하지만 분산은 증가하며,반대로 모델의 복잡성이 감소하면 편향은 증가하고 분산은 감





# 1. 결정트리(Desicion Tree)

+ 앙상블을 구성하는 개별 모델을 XGBoost에서는 **기본학습기(base learner)** 라고 함
    + XGBoost의 기본 학습기로 가장 널리 사용되는 것이 결정트리

+ **결정트리**는 과적합되기 쉬우므로, 분산과 편향을 다루며 이 문제를 살펴 볼 것

+ 과적합을 막는 방법은

    1. 하이퍼 파라미터 튜닝

    2. 여러 트리의 예측을 모으는 것 (RF와 XGBoost)



## 1.1. 결정트리 알고리즘

결정트리는 **가지(branch)** 분할을 통해 데이터를 두 개의 **노드(node)** 로 나눕니다. 가지 분할은 예측을 만드는 **리프 노드(leaf node)** 까지 계속됩니다.



### 1.1.1. 용어 설명

<img src="model.assets/image-20230602154625759.png" alt="image-20230602154625759" style="zoom:80%;" />

위의 내용 해석

**루트노드**

- 첫 노드인 루트노드는 현재 결혼 유무에 따라서 분할을 하였음
- 왼쪽이 결혼을 안한 True로 0값, 오른쪽이 결혼을 한 True로 1임

**지니불순도**

- $gini = 1 - \sum_{i=1}^c(p_i)^2$ , $p_i$는 전체 샘플에서 해당 클래스 샘플의 비율, c는 총 클래스 개수
- 불순도 값이 가장 낮은 분할을 찾는 것이 트리의 목표
- 루트노드의 gini=0.367
- gini값이 0이면 하나의 클래스로만 이루어진 노드가 됨
- gini값이 0.5이면 클래스 간 샘플 개수가 동일함

**samples**

- sample 수

**value**

- 현재 노드에서 y값이 0인 개수와 y값이 1인 개수를 각각 나타냄

**class**

- 현재 노드의 다수 클래스가 무엇인지 나타냄

**리프노드**

- 트리의 끝에 있는 노드
- 리프노드에서 최종 예측이 결정되고, 다수인 클래스가 예측 클래스가 됨

**분할방식**

- 왼쪽이 True, 오른쪽이 False 노드로 분류됨



## 1.2. 결정트리 코드

### 1.2.1. Classifier

```python
import pandas as pd
import numpy as np

# 1. 데이터 준비
df_census = pd.read_csv('census_cleaned.csv')

## 1.1. 데이터를 X와 y로 나누기
X = df_census.iloc[:,:-1]
y = df_census.iloc[:,-1]

## 1.2. 데이터를 훈련 세트와 테스트 세트로 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 2. 모델적합
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

## 2.1. 분류 모델 생성
clf = DecisionTreeClassifier(random_state=2)
clf.fit(X_train, y_train)

## 2.2. 테스트 데이터에 대한 예측 생성
y_pred = clf.predict(X_test)

## 2.3. 정확도 계산
accuracy_score(y_pred, y_test)

df_census.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>workclass_ ?</th>
      <th>workclass_ Federal-gov</th>
      <th>workclass_ Local-gov</th>
      <th>workclass_ Never-worked</th>
      <th>...</th>
      <th>native-country_ Puerto-Rico</th>
      <th>native-country_ Scotland</th>
      <th>native-country_ South</th>
      <th>native-country_ Taiwan</th>
      <th>native-country_ Thailand</th>
      <th>native-country_ Trinadad&amp;Tobago</th>
      <th>native-country_ United-States</th>
      <th>native-country_ Vietnam</th>
      <th>native-country_ Yugoslavia</th>
      <th>income_ &gt;50K</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>77516</td>
      <td>13</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>83311</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>215646</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>234721</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>338409</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 93 columns</p>





### 1.2.2. 트리플랏

```python
# 트리플랏 코드
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(13,8))
plot_tree(clf, max_depth=2, feature_names=list(X.columns), class_names=['0','1'], filled=True, rounded=True, fontsize=14)
plt.show()
```

<img src="model.assets/image-20230602154534565.png" alt="image-20230602154534565" style="zoom:80%;" />





### 1.2.3. Regressor

```python
# 모델 적합 & validation score 확인
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

# DecisionTreeRegressor 객체를 생성
reg = DecisionTreeRegressor(random_state=42)

# 평균 제곱 오차로 교차 검증 점수를 계산 : sklearn은 최대화시키는 방향으로 훈련함
scores = cross_val_score(reg, X_bikes, y_bikes, scoring='neg_mean_squared_error', cv=5)

# rmse 계산
rmse = np.sqrt(-scores)

# 평균을 출력
print('RMSE 평균: %0.2f' % (rmse.mean()))
```





## 1.3. 결정트리 하이퍼파라미터 & 속성 & 메서드 

### 1.3.1. 하이퍼파라미터

- sklearn Regressor 트리 문서 : https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

    

#### 1.3.1.1. tips

- 일반적으로 **max**이름을 가진 매개변수를 감소시키고, **min**이름을 가진 매개변수를 증가시키면 분산을 줄이고 과적합을 방지함

- `max_features`와 나머지 `min` 접두사가 붙은 매개변수는 모두 **0~1** 사이의 실수를 사용할 수 있음

    

#### 1.3.1.2. 주로 사용되는 하이퍼파라미터

- max_depth, max_features, min_samples_leaf, max_leaf_nodes, min_impurity_decrease, min_samples_split

    

#### 1.3.1.3. `max_depth`

> 트리의 깊이(분할 개수)를 설정함

- `DecisionTreeRegressor()`의 `max_depth=None` 인 경우, 리프노드가 순수노드가 될 때까지 트리가 성장됨.
    - 회귀모델의 타깃은 임의의 실수이기 때문에 리프노드가 순수노드가 될 때까지 성장한다면, 대부분 리프노드에 샘플이 1개씩만 들어가게 되어 과적합이 발생하게 됨



#### 1.3.1.4. `max_features`

> 분할마다 모든 특성을 고려하지 않고, 매번 지정된 개수의 특성 중에서 선택

- 모델의 분산을 줄이는 데 효과적인 매개변수 (분할을 제한함)
- 옵션 설명
    - None , 'auto' : 전체특성 사용
    - 'sqrt` : 전체 특성 개수의 제곱근을 사용 (DecisionTreeClassifier의 경우 'auto' == 'sqrt')
    - 'log2' : 전체 특성 개수의 로그를 사용



#### 1.3.1.5. `min_sample_leaf`

> 리프 노드가 가질 수 있는 최소 샘플의 개수를 제한함

- 과적합 방지



#### 1.3.1.6. `max_leaf_nodes`

> 리프 노드가 가질 수 있는 최대 샘플의 개수를 제한함



#### 1.3.1.7. `min_impurity_decrease`

> 분할하기 위한 최소 불순도 감소를 지정

- default = 0
- 해당 값 만큼의 불순도가 감소되어야 분할이 진행됨



#### 1.3.1.8. `min_samples_split`

> 분할하기 위해 필요한 최소 샘플 개수를 제한함

- default = 2
- 5로 설정하면, 5개 보다 적은 노드는 더 이상 분할되지 않음



#### 1.3.1.9. `splitter`

> 노드를 분할하기 위한 **특성 선택 방법**

- 옵션
    - default = 'best' : **정보이득**이 가장 큰 특성을 선택
    - 'random' : 랜덤하게 노드를 분할
- 'random'을 선택하면 과적합을 막고, 다양한 트리 생성 가능



#### 1.3.1.10. `criterion`

> 분할 품질을 측정하는 메저를 정함
>
> + criterion이 가장 좋은 분할을 선택해서 분할하게 됨

- 회귀모델
    - 'squared_error(default)', 'friedman_mse', 'absolute_error', 'poisson'
- 분류모델
    - 'gini(default)', 'entropy' : 보통 둘 다 비슷한 결과 나옴



#### 1.3.1.11. `min_weight_fraction_leaf`

> 리프 노드가 되기 위한 전체 가중치의 최소 비율 (`sample_weight` 설정을 따로 하지 않으면 그냥 sample의 개수 비율이 됨)

- 샘플이 500개이고, `sample_weight` 없이 해당 매개변수를 0.01로 지정하면, 리프 노드가 되기 위한 최소 샘플 개수는 5개
    - `.fit()` 에서 `sample_weight` 매개변수를 지정하지 않으면, 샘플은 모두 동일한 가중치를 가짐

- **분산을 줄이고 과적합을 막을 수 있음**
- default = 0



#### 1.3.1.12. `ccp_alpha`

>  트리를 만든 이후에 가지치기하는 기능

- default = 0으로 가지치기 실행하지 않음
- 0보다 크면, 최대 ccp_alpha의 비용복잡도를 가진 부분 트리를 선택
- 공식문서 참고 : https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning





#### 1.3.1.100. RandomizedSearchCV 예시

+ `n_iter` : Random Search를 실행할 횟수

```python
# 1. 임포트
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

# 2. RandomizedSearchCV 함수 생성
def randomized_search_clf(params, runs=20, clf=DecisionTreeClassifier(random_state=42)):

    # RandomizedSearchCV 객체를 만듭니다.
    rand_clf = RandomizedSearchCV(clf, params, n_iter=runs, 
                                  cv=5, n_jobs=-1, random_state=42)
    
    # X_train와 y_train에서 랜덤 서치를 수행
    rand_clf.fit(X_train, y_train)

    # 최상의 모델을 추출
    best_model = rand_clf.best_estimator_
    
    # 최상의 점수를 추출
    best_score = rand_clf.best_score_

    # 최상의 점수를 출력
    print("훈련 점수: {:.3f}".format(best_score))

    # 테스트 세트에 대한 예측을 생성
    y_pred = best_model.predict(X_test)

    # 정확도를 계산
    accuracy = accuracy_score(y_test, y_pred)
    print('테스트 점수: {:.3f}'.format(accuracy))
            
    # 최상의 모델을 반환
    return best_model

# 3. RandomizedSearchCV 진행
## 3.1. params 1
params={
        'max_depth':[None, 2,4,6,8],  # 트리의 깊이(분할 개수)를 설정함
        'max_features':['sqrt', 0.95, 0.90, 0.85, 0.80, 0.75, 0.70], # 분할마다 모든 특성을 고려하지 않고, 매번 지정된 개수의 특성 중에서 선택
        'min_samples_leaf':[1, 0.01, 0.02, 0.03, 0.04], # 리프 노드가 가질 수 있는 최소 샘플의 개수를 제한함
        'max_leaf_nodes':[10, 15, 20, 25, 30, 35, 40, 45, 50, None], # 리프 노드가 가질 수 있는 최대 샘플의 개수를 제한함
        'min_impurity_decrease':[0.0, 0.0005, 0.005, 0.05, 0.10, 0.15, 0.2], # 분할하기 위한 최소 불순도 감소를 지정
        'min_samples_split':[2, 3, 4, 5, 6, 8, 10], # 분할하기 위해 필요한 최소 샘플 개수를 제한함
        'splitter':['random', 'best'],  # 노드를 분할하기 위한 특성 선택 방법
        'criterion':['entropy', 'gini'],
        'min_weight_fraction_leaf':[0.0, 0.0025, 0.005, 0.0075, 0.01, 0.05], # 리프 노드가 되기 위한 전체 가중치의 최소 비율
#       'ccp_alpha' : [0, ...]
}
randomized_search_clf(params=params)

## 3.2. params 2 : loguniform 이용
from sklearn.utils.fixes import loguniform
from scipy.stats import randint
params = {'max_depth': randint(1,100),
          'max_features': loguniform(1e-5,1),
          'min_samples_leaf': loguniform(1e-5,1),
          'max_leaf_nodes': randint(1,100),
          'min_impurity_decrease': loguniform(1e-5,1),
          'min_samples_split': loguniform(1e-5,1),
          'min_weight_fraction_leaf': loguniform(1e-5,1)}
dtc = DecisionTreeClassifier(random_state=0)
rs = RandomizedSearchCV(dtc, params, n_iter=100, n_jobs=-1, random_state=42)
rs.fit(X_train, y_train)
```







### 1.3.2. 속성

#### 1.3.2.1. `.tree_`

+  훈련된 트리 객체가 저장되어 있음

#### 1.3.2.2. 트리 객체의 속성

- `.node_count` : 트리 전체 노드 개수
- `.n_nodes_smples` : 각 노드에 포함된 샘플 개수
- `.children_left`, `.children_right` : 자식 노드의 인덱스를 담고 있음, 2개 다 -1이면 리프노드

```python
# 1. 리프노드 개수 counting
leaf_node_count = 0
tree = reg.tree_
for i in range(tree.node_count):
    # 리프노드 개수 counting
    if (tree.children_left[i] == -1) and (tree.children_right[i] == -1):
        leaf_node_count += 1
        if tree.n_node_samples[i] > 1:
            print('노드 인덱스:', i, ', 샘플 개수:', tree.n_node_samples[i])
print('전체 리프 노드 개수:', leaf_node_count)
```

```
노드 인덱스: 124 , 샘플 개수: 2
전체 리프 노드 개수: 547
```

```python
# 2. 트리 전체 속성
tree = reg.tree_
print('트리 객체 : ', tree)
print('트리 노드 개수 : ',tree.node_count)
print('각 노드에 포함된 트리 개수 :', tree.n_node_samples)
print('각 좌측노드의 인덱스 :', tree.children_left) 
print('각 우측노드의 인덱스 :', tree.children_right) 
```

```
트리 객체 :  <sklearn.tree._tree.Tree object at 0x0000013EFC25C6C0>
트리 노드 개수 :  1093
각 노드에 포함된 트리 개수 : [548 327 183 ...   1   1   1]
각 좌측노드의 인덱스 : [ 1  2  3 ... -1 -1 -1]
각 우측노드의 인덱스 : [652 365 150 ...  -1  -1  -1]
```



#### 1.3.2.3. `feature_importances_`

> 노드에 사용된 특성별로 감소된 불순도량을 더한 후, 전체 값이 1이 되도록 정규화한 것

```python
# 1. 특성 중요도 가져오기
best_model.feature_importances_
```

```
array([0.00170337, 0.        , 0.12246496, 0.00197446, 0.        ,
       0.        , 0.02172684, 0.02172684, 0.06972671, 0.03740602,
       0.09972507, 0.42061239, 0.20293334])
```

```python
# 2. 특성 중요도 해석
## 2.1. 열과 feature_importances_를 딕셔너리로 묶기
feature_dict = dict(zip(X.columns, best_model.feature_importances_))

## 2.2.  튜플에 있는 값을 기준으로 딕셔너리를 정렬
sorted(feature_dict.items(), key=lambda x : x[1], reverse=True)
```

```
[('ca', 0.42061238935298506),
 ('thal', 0.2029333435096573),
 ('cp', 0.12246496463353763),
 ('slope', 0.09972506749405596),
 ('exang', 0.06972670737588196),
 ('oldpeak', 0.037406020256486906),
 ('thalach', 0.021726842018324793),
 ('restecg', 0.021726842018324786),
 ('trestbps', 0.0019744579851423446),
 ('age', 0.0017033653556032876),
 ('sex', 0.0),
 ('chol', 0.0),
 ('fbs', 0.0)]
```





### 1.3.3. 메서드

#### 1.3.3.1. `permutation_importance()`

> sklearn에서 추천하는 특성 중요도 측정 방법은 `permutation_importance()` 함수

1. 전달 받은 특성 값을 그대로 사용해 모델을 훈련하고 타깃을 사용해 모델의 점수를 계산
2. 특성 하나를 랜덤하게 섞은 후 모델을 훈련하여 계산
3. 이 방식으로 각 특성을 `n_repeats` 횟수만큼 순서대로 테스트하여 모델 성능에 큰 영향을 미치는 특성을 찾음

```python
# 1. 특성 중요도 측정
from sklearn.inspection import permutation_importance
result = permutation_importance(best_model, X, y, n_jobs=-1, random_state=0)

# 2. 특성 중요도 정렬
feature_dict = dict(zip(X.columns, result.importances_mean))
sorted(feature_dict.items(), key=lambda x : x[1], reverse=True)
```

```
[('ca', 0.12079207920792083),
 ('thal', 0.06534653465346538),
 ('cp', 0.045544554455445564),
 ('slope', 0.04026402640264033),
 ('thalach', 0.013861386138613896),
 ('exang', 0.0105610561056106),
 ('age', 0.0),
 ('sex', 0.0),
 ('trestbps', 0.0),
 ('chol', 0.0),
 ('fbs', 0.0),
 ('restecg', 0.0),
 ('oldpeak', 0.0)]
```







