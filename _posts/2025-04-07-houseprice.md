---
layout: post
title: "주택 가격 예측"
date: 2025-04-07
categories: [머신러닝, 1주차]
---

# 1990년 미국 캘리포니아 주에서 수집한 주택 가격 데이터를 활용하여 주택 가격을 예측


주택 가격을 예측하기 위해 훈련시킬 모델의 특성은 다음과 같다.
1. 지도 학습 : 구역별 주택 중위 가격을 타깃, 즉 최대한 정확하게 예측해야 하는 목표로 지정한다.
2. 회귀 : 주택 중위가격, 즉 이산형 값이 아닌 연속형 값을 예측한다. 보다 세분화하면 다중 회귀이자 단변량 회귀모델이다.
   * 다중 회귀 : 구역별로 여러 특성을 주택 가격 예측에 사용
   * 단변량 회귀 : 구역별로 한 종류의 값만 예측
4. 배치 학습 : 빠르게 변하는 데이터에 적응할 필요가 없고, 데이터셋 전체를 대상으로 훈련을 진행한다.   

```python
import sys
assert sys.version_info>=(3,7)

import sklearn
assert sklearn.__version__>="1.0.1"

import numpy as np

np.random.seed(42)
```
여기까지는 기본적으로 세팅해야 할 부분이다.

```python
from pathlib import Path  #pathlib은 파일과 디렉터리 경로를 다루는데 사용하며, Path 클래스는 경로 객체를 생성하고 경로 조직을 단순화할 때 사용한다.
import pandas as pd
import tarfile  #.tar, .tar.gz 등 압축된 tar 파일을 읽고 쓰는 데 사용한다. 주로 대규모 데이터셋을 다운로드하거나 저장할 때 이용한다.
import urllib.request  #HTTP 요청을 보내고 데이터를 다운로드하기 위한 모듈이다.
```
여기서 urlib이란 파이썬의 표준 라이브러리로 웹에서 데이터를 가져오는 데 사용한다.  

```python
def load_housing_data():  #주택 데이터셋을 불러온다. 데이터가 이미 로컬에 존재하면 바로 사용하고, 없으면 다운로드 후 처리한다.
  tarball_path=Path("datasets/housing.tgz")  #tarball_path : 다운로드된 압축 파일의 경로
  if not tarball_path.is_file():  #경로에 해당 파일이 없으면 실행한다.
    Path("datasets").mkdir(parents=True, exist_ok=True)
    url = "https://github.com/ageron/data/raw/main/housing.tgz"
    urllib.request.urlretrieve(url, tarball_path)  #지정된 URL에서 데이터를 다운로드하고, tarball_path 경로에 저장한다. 
    with tarfile.open(tarball_path) as housing_tarball:  #.tar 또는 tgz 파일로 읽는다.
        housing_tarball.extractall(path="datasets")  #.extractall() : 모든 압축된 파일을 데이터셋 디렉터리에 압축해제 한다. 
  return pd.read_csv(Path("datasets/housing/housing.csv"))  #.read_csv() : 압축 해제된 csv파일을 읽어 pandas의 dataFrame으로 반환한다.
```
.mkdir() : 데이터셋 티렉터리가 없으면 새로 생성한다.  
parents=True : 상위 디렉터리도 함께 생성한다.  
exist_ok=True : 디렉터리가 이미 존재해도 에러 발생 없이 넘어간다.  

```python
housing = load_housing_data()
```
이전에 정의된 함수로 데이터를 로드하여 housing 변수에 저장한다. 이 데이터는 pandas의 dataFrame으로 반환한다.

```python
housing.head()  #처음 5개의 데이터 불러오기

housing.info()
```
.info : dataFrame에 포함된 정보(열 이름, 데이터 유형, 비어있는 값의 개수 등) 출력.  
주로 데이터의 전반적인 구조를 이해하고 결측치가 있는지 확인하는데 유용하다.

```python
housing["ocean_proximity"].value_counts()
```
ocean_proximity 열을 선택하여 카테고리 데이터를 가져와 (각 카테고리의) 고유값의 개수를 세고 내림차순으로 반환한다. value_counts() 함수를 통해 데이터의 분포를 확인할 수 있다.  

```python
housing.describe()
```
.describe() : 수치형 열에 대한 요약 통계량(평균, 표준편차, 최소값, 최대값 등)을 반환한다. 데이터의 범위와 스케일을 파악하는 데 유용하다.  

```python
import matplotlib.pyplot as plt  #데이터를 시각화하기 위한 파이썬의 인기 있는 라이브러리로 주로 plt라는 약칭을 사용한다.

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
```
데이터를 시각화한다.

```python
IMAGES_PATH = Path() / "images" / "end_to_end_project"  #현재 작업 디렉토리/"경로 객체", OS에 상관 없이 플랫폼 독립적으로 경로를 관리할 수 있다.
IMAGES_PATH.mkdir(parents=True, exist_ok=True)  #디렉토리를 생성한다. 
```

```python
housing.hist(bins=50, figsize=(12, 8))  #데이터 프레임의 각 열에 대해 히스토그램을 생성한다.
plt.show()  #히스토그램 표시
```
히스토그램은 데이터의 분포를 이해하는데 유용하다. 여기서 bins는 구역을 의미하며 bins=50은 각 히스토그램의 데이터를 50개의 구간으로 나눴음을 의미한다. figsize=(12,8)는 그래프의 크기를 설정하는 것으로 가로 12인치, 세로 8인치를 의미한다.  
housing 데이터의 분포를 시각적으로 분석할 수 있다. median_income이나 housing_median_age와 같은 열에서 값의 집중도나 이상값을 발견할 수 있다. 

```python
from sklearn.model_selection import train_test_split

train_set, test_set=train_test_split(housing,test_size=0.2,random_state=42)
```
train_test_split(나눌 데이터 프레임, test_size=테스트 세트로 할당할 데이터의 양, random_state=난수 생성 시드 고정) : 데이터를 훈련 세트와 테스트 세트를 무작위로 나눈다.  
여기서는 20%의 데이터를 테스트 세트로 할당하고 나머지 80%를 훈련 세트에 포함한다. 난수 생성 시드를 42로 고정하여 실행할 때마다 동일한 분할을 보장한다.  

# Use np.inf instead of inf
housing['median_income'].hist(bins=1000, range=(0, 1))
housing["income_cat"]=pd.cut(housing["median_income"],bins=[0.,1.5,3.0,4.5,6.,np.inf],labels=[1,2,3,4,5])
housing

housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.show()

strat_train_set, strat_test_set=train_test_split(housing, test_size=0.2,stratify=housing["income_cat"],random_state=42)

def income_cat_proportions(data):
  return data["income_cat"].value_counts() / len(data)

train_set, test_set=train_test_split(housing, test_size=0.2, random_state=42)

compare_props=pd.DataFrame({"전체(%)":income_cat_proportions(housing),"계층 샘플링(%)":income_cat_proportions(strat_test_set),"무작위 샘플링(%)":income_cat_proportions(test_set)})
compare_props.sort_index="소득 구간"
compare_props["계층 샘플링 오류율(%)"]=(compare_props["계층 샘플링(%)"]/compare_props["전체(%)"]-1)
compare_props["무작위 샘플링 오류율(%)"]=(compare_props["무작위 샘플링(%)"]/compare_props["전체(%)"] - 1)
(compare_props * 100).round(2)

for set_ in(strat_train_set,strat_test_set):
  set_.drop("income_cat",axis=1,inplace=True)

housing=strat_train_set.copy()
housing = housing.reset_index(drop=True)

housing.plot(kind="scatter",x="longitude",y="latitude",grid=True)
plt.show()

housing.plot(kind="scatter",x="longitude",y="latitude",grid=True,alpha=0.2)
plt.show()

housing.plot(kind="scatter",x="longitude",y="latitude",grid=True,s=housing["population"]/100,label="population",c="median_house_value",cmap="jet",colorbar=True,legend=True,sharex=False,figsize=(10,7))
plt.show()

filename="california.png"
if not (IMAGES_PATH/filename).is_file():
  homl3_root="https://github.com/ageron/handson-ml3/raw/main/"
  url=homl3_root+"images/end_to_end_project/"+filename
  print("Downloading",filename)
  urllib.request.urlretrieve(url,IMAGES_PATH/filename)

housing_renamed=housing.rename(columns={
    "latitude":"Latitude","longitude":"Longitude","population":"Population","median_house_value":"Median house value(USD)"})

housing_renamed.plot(kind="scatter",x="Longitude",y="Latitude",s=housing_renamed["Population"]/100,label="Population",c="Median house value(USD)",cmap="jet",colorbar=True,legend=True,sharex=False,figsize=(10,7))

california_img=plt.imread(IMAGES_PATH/filename)
axis=-124.55,-113.95,32.45,42.05 #x축 y축 눈금
plt.axis(axis)
plt.imshow(california_img,extent=axis)
plt.show

corr_matrix=housing.corr(numeric_only=True)
corr_matrix

corr_matrix["median_house_value"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes=["median_house_value","median_income","total_rooms","housing_median_age"]
scatter_matrix(housing[attributes],figsize=(12,8))
plt.show()

housing.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.1,grid=True)
plt.show()

null_rows_idx=housing.isnull().any(axis=1)
null_rows_idx

housing.loc[null_rows_idx].head()

housing.loc[null_rows_idx].shape

median=housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median,inplace=True)

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="median")

housing_num=housing.select_dtypes(include=[np.number])

imputer.fit(housing_num)
imputer.statistics_

X=imputer.transform(housing_num)
X

X=imputer.fit_transform(housing_num)
X

housing_tr=pd.DataFrame(X,columns=housing_num.columns,index=housing_num.index)
housing_tr.loc[null_rows_idx].head()

#입력 데이터셋 지정
housing=strat_train_set.drop("median_house_value",axis=1)
#타깃 데이터셋 지정
housing_labels=strat_train_set["median_house_value"].copy()

housing_cat=housing[["ocean_proximity"]]

from sklearn.preprocessing import OneHotEncoder
cat_encoder=OneHotEncoder()
housing_cat_1hot=cat_encoder.fit_transform(housing_cat)

housing_cat_1hot
housing_cat_1hot.toarray()

cat_encoder=OneHotEncoder(sparse_output=False)
housing_cat_1hot=cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

cat_encoder.categories_
cat_encoder.feature_names_in_
cat_encoder.get_feature_names_out()

housing_cat_onehot=pd.DataFrame(housing_cat_1hot, columns=cat_encoder.get_feature_names_out(),index=housing_cat.index)
housing_cat_onehot

from sklearn.preprocessing import MinMaxScaler
min_max_scaler=MinMaxScaler(feature_range=(0,1))
housing_num_min_max_scaled=min_max_scaler.fit_transform(housing_num)

from sklearn.preprocessing import StandardScaler
std_scaler=StandardScaler()
housing_num_std_scaled=std_scaler.fit_transform(housing_num)

from sklearn.preprocessing import FunctionTransformer
log_transformer=FunctionTransformer(np.log,feature_names_out="one-to-one")

ratio_transformer=FunctionTransformer(lambda X: X[:, [0]]/X[:,[1]])

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

class ClusterSimilarity(BaseEstimator, TransformerMixin):
  def __init__(self,n_clusters=10,gamma=1.0,random_state=None):
    self.n_clusters=n_clusters
    self.gamma=gamma
    self.random_state=random_state

  # Indent fit method within the class definition
  def fit(self,X,y=None,sample_weight=None):
    self.kmeans_=KMeans(self.n_clusters,random_state=self.random_state,n_init=10).fit(X,sample_weight=sample_weight)
    return self

  # Indent transform method within the class definition
  def transform(self,X):
    return rbf_kernel(X,self.kmeans_.cluster_centers_,gamma=self.gamma)

  def get_feature_names_out(self,names=None):
    return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

housing['median_income']

cluster_simil=ClusterSimilarity(n_clusters=10,gamma=1.,random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]],sample_weight=housing['median_income'])

similarities[:5].round(2)

housing_renamed=housing.rename(columns={"latitude":"Latitude","longitude":"Longitude","population":"Population","median_house_value":"Median house value(USD)"})
housing_renamed["Max cluster similarity"]=similarities.max(axis=1)
housing_renamed.plot(kind="scatter",x="Longitude",y="Latitude",grid=True,s=housing_renamed["Population"]/100,label="Population", c="Max cluster similarity", cmap="jet",colorbar=True,legend=True, sharex=False, figsize=(10,7))
plt.plot(cluster_simil.kmeans_.cluster_centers_[:,1],cluster_simil.kmeans_.cluster_centers_[:,0],linestyle="",color="black",marker="X",markersize=20,label="Cluster centers")
plt.legend(loc="upper right")
plt.show()

from sklearn.pipeline import Pipeline
num_pipeline=Pipeline([("impute",SimpleImputer(strategy="median")),("standardize",StandardScaler()),])

from sklearn.pipeline import make_pipeline

num_pipeline=make_pipeline(SimpleImputer(strategy="median"),StandardScaler())

housing_num_prepared=num_pipeline.fit_transform(housing_num)
num_pipeline.get_feature_names_out()

df_housing_num_prepared = pd.DataFrame(housing_num_prepared,
                                       columns=num_pipeline.get_feature_names_out(),
                                       index=housing_num.index)
df_housing_num_prepared.head()

from sklearn.compose import ColumnTransformer
num_attribs=["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income"]
cat_attribs=["ocean_proximity"]

cat_pipeline=make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessing=ColumnTransformer([("num",num_pipeline,num_attribs),("cat",cat_pipeline,cat_attribs),])

from sklearn.compose import make_column_selector
preprocessing=ColumnTransformer([("num",num_pipeline,make_column_selector(dtype_include=np.number)),("cat",cat_pipeline,make_column_selector(dtype_include=object))])

housing_prepared=preprocessing.fit_transform(housing)

preprocessing.get_feature_names_out()

from sklearn.compose import make_column_transformer
preprocessing=make_column_transformer(
    (num_pipeline,make_column_selector(dtype_include=np.number)),(cat_pipeline,make_column_selector(dtype_include=object)),
    )
housing_prepared=preprocessing.fit_transform(housing)
preprocessing.get_feature_names_out()

housing_prepared_fr = pd.DataFrame(housing_prepared,
                                   columns=preprocessing.get_feature_names_out(),
                                   index=housing.index)

housing_prepared_fr.head()

def column_ratio(X):
  return X[:,[0]]/X[:,[1]]

def ratio_name(function_transformer,feature_names_in):
  return ["ratio"]

ratio_pipeline=make_pipeline(
    SimpleImputer(strategy="median"), FunctionTransformer(column_ratio,feature_names_out=ratio_name),
    StandardScaler()
)

log_pipeline=make_pipeline(
    SimpleImputer(strategy="median"),FunctionTransformer(np.log,feature_names_out="one-to-one"),
    StandardScaler()
)

cluster_simil=ClusterSimilarity(n_clusters=10,gamma=1.,random_state=42)

default_num_pipeline=make_pipeline(SimpleImputer(strategy="median"),StandardScaler())

preprocessing=ColumnTransformer([
    ("bedrooms",ratio_pipeline,["total_bedrooms","total_rooms"]),
    ("rooms_per_house",ratio_pipeline,["total_rooms","households"]),
    ("people_per_house",ratio_pipeline,["population","households"]),
    ("log",log_pipeline,["total_bedrooms","total_rooms","population","households","median_income"]),
    ("geo",cluster_simil,["latitude","longitude"]),
    ("cat",cat_pipeline,make_column_selector(dtype_include=object)),],remainder=default_num_pipeline)

housing_prepard=preprocessing.fit_transform(housing)
housing_prepared.shape
preprocessing.get_feature_names_out()
housing_prepared

