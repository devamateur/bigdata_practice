# bigdata_practice
- 빅데이터 분석 기사 실기 준비를 위한 저장소

- 참고 사이트: https://www.kaggle.com/datasets/agileteam/bigdatacertificationkr/discussion?sort=undefined

<br>

### 작업형 2유형 풀이 흐름
1) 데이터 불러오기 pd.read_csv()
2) train/test split 
3) 결측값, 이상치 등 확인: 결측값이 많은 컬럼은 되도록 삭제하지 않고 다른 값으로 대체
   - df.drop(컬럼명, axis=1): 결측값이 조금 있거나, 회원 ID같이 분석과 관계없는 컬럼 삭제
   - df.isnull().sum(): 컬럼별 결측값 합계
   - df.fillna(x): 결측값을 x로 대체
4) 피쳐 선택
   - 어떤 변수로 분석할지 선택
   - 범주형 변수: 원핫인코딩으로 변환 (pd.get_dummies(범주형 변수))
   - 범주가 너무 다양한 변수는 삭제?

5) 스케일링, 정규화
   - 스케일링: StandardScaler
   - 정규화: min-max 정규화, z-score 정규화
      - min-max 정규화: X-min(X) / max(X)-min(X)
      - z-score 정규화: X-X의 평균 / X의 표준편차

6) 분석 및 평가지표
   - 회귀 분석: LogisticRegression(), RandomForestRegressor()
     - 평가지표: MSE, MAE, RMSE
   - 분류: RandomForestClassifier()
     - 평가지표: roc-auc, f1-score
   - 클러스터링: KMeans()
     - 평가지표: silhouette_score

7) 결과 제출
   - 분석 결과 csv 파일로 만들어 제출하는 것 잊지 말기
     - df.to_csv("파일명.csv", index=False)

<br>

#### 예시코드
   ```
   import pandas as pd

   train = pd.read_csv("data/customer_train.csv")
   test = pd.read_csv("data/customer_test.csv")

   # train/test split
   from sklearn.model_selection import train_test_split
   X_train, X_val, y_train, y_val = train_test_split(train, train['성별'], test_size=0.2)


   X_train =  X_train.drop(['성별', '회원ID'], axis=1)
   X_test = test.drop('회원ID', axis=1)

   X_val = X_val.drop('회원ID', axis=1)
   print(X_train.shape, y_train.shape)
   print(X_val.shape)

   #print(y_train.head())

   # 결측값 확인
   #print(X_train.isnull().sum())
   #print(X_test.isnull().sum())

   # 환불금애 컬럼 결측값 평균으로 대체
   # print(X_train['환불금액'].value_counts())

   X_train['환불금액'] = X_train['환불금액'].fillna(X_train['환불금액'].mean())
   X_val['환불금액'] = X_val['환불금액'].fillna(X_val['환불금액'].mean())
   X_test['환불금액'] = X_test['환불금액'].fillna(X_test['환불금액'].mean())

   #print(X_train['주구매상품'].value_counts())
   #print(X_train['주구매지점'].value_counts())

   # 피쳐 선택
   # 주구매상품, 주구매지점 컬럼 삭제
   X_train = X_train.drop(['주구매상품', '주구매지점'], axis = 1)
   X_val = X_val.drop(['주구매상품', '주구매지점'], axis = 1)
   X_test = X_test.drop(['주구매상품', '주구매지점'], axis = 1)

   ''' 스케일링
   cols = X_train.columns
   print(cols)
   from sklearn.preprocessing import StandardScaler

   scaler = StandardScaler()
   X_train[cols] = scaler.fit_transform(X_train[cols])
   X_test = scaler.fit_transform(X_test[cols])'''


   from sklearn.ensemble import RandomForestClassifier
   rf = RandomForestClassifier(max_depth = 8)
   rf.fit(X_train, y_train)
   pred = rf.predict(X_test)
   print(len(pred))
   print(rf.score(X_train, y_train))

   result = pd.DataFrame({'pred':pred})

   result.to_csv("result.csv", index=False)
   ```