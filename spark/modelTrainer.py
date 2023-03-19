import numpy as np
import pandas as pd
import time
import xgboost as xgb
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from pyspark.ml.regression import GBTRegressor
import matplotlib.pyplot as plt
from utils.timePresenter import cal_run_time

# Create SparkSession
spark = SparkSession.builder \
      .master('local[1]') \
      .appName('modelTrainer') \
      .getOrCreate()

start_time = time.time()
# Use this line at Linux
# x_train = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load("hdfs://172.31.58.70:9000/data/diabetic_data.csv")
# x_train = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load("file:///~/cs230/spark/data/diabetic_data.csv")
# Use this line at Windows
x_train = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('./data/diabetic_data.csv')

x_train = x_train.drop('encounter_id', 'patient_nbr')
x_train = x_train.toPandas()

x_train.loc[x_train.age== '[0-10)','age'] = 0
x_train.loc[x_train.age== '[10-20)','age'] = 10
x_train.loc[x_train.age== '[20-30)','age'] = 20
x_train.loc[x_train.age== '[30-40)','age'] = 30
x_train.loc[x_train.age== '[40-50)','age'] = 40
x_train.loc[x_train.age== '[50-60)','age'] = 50
x_train.loc[x_train.age== '[60-70)','age'] = 60
x_train.loc[x_train.age== '[70-80)','age'] = 70
x_train.loc[x_train.age== '[80-90)','age'] = 80
x_train.loc[x_train.age== '[90-100)','age'] = 90
x_train.age = x_train.age.astype(np.int32)

x_train.loc[x_train.weight== '[0-25)','weight'] = 0
x_train.loc[x_train.weight== '[25-50)','weight'] = 25
x_train.loc[x_train.weight== '[50-75)','weight'] = 50
x_train.loc[x_train.weight== '[75-100)','weight'] = 75
x_train.loc[x_train.weight== '[100-125)','weight'] = 100
x_train.loc[x_train.weight== '[125-150)','weight'] = 125
x_train.loc[x_train.weight== '[150-175)','weight'] = 150
x_train.loc[x_train.weight== '[175-200)','weight'] = 175
x_train.loc[x_train.weight== '>200','weight'] = -100
x_train.loc[x_train.weight== '?','weight'] = None
x_train.weight = x_train.weight.astype(np.float32)

x_train.loc[x_train.max_glu_serum== 'None','max_glu_serum'] = 0
x_train.loc[x_train.max_glu_serum== 'Norm','max_glu_serum'] = 100
x_train.loc[x_train.max_glu_serum== '>200','max_glu_serum'] = 200
x_train.loc[x_train.max_glu_serum== '>300','max_glu_serum'] = 300
x_train.max_glu_serum = x_train.max_glu_serum.astype(np.int32)

x_train.loc[x_train.A1Cresult== 'None','A1Cresult'] = 0
x_train.loc[x_train.A1Cresult== 'Norm','A1Cresult'] = 5
x_train.loc[x_train.A1Cresult== '>7','A1Cresult'] = 7
x_train.loc[x_train.A1Cresult== '>8','A1Cresult'] = 8
x_train.A1Cresult = x_train.A1Cresult.astype(np.int32)

x_train.loc[x_train.change== 'No','change'] = 0
x_train.loc[x_train.change== 'Ch','change'] = 1
x_train.change = x_train.change.astype(np.int8)

x_train.loc[x_train.diabetesMed== 'No','diabetesMed'] = 0
x_train.loc[x_train.diabetesMed== 'Yes','diabetesMed'] = 1
x_train.diabetesMed = x_train.diabetesMed.astype(np.int8)

medications = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]
for med in medications:
    x_train.loc[x_train[med] == 'No', med] = -20
    x_train.loc[x_train[med] == 'Down', med] = -10
    x_train.loc[x_train[med] == 'Steady', med] = 0
    x_train.loc[x_train[med] == 'Up', med] = 10
    x_train[med] = x_train[med].astype(np.int32)

categoricals = ['race', 'gender', 'payer_code', 'medical_specialty','diag_1', 'diag_2', 'diag_3']
for c in categoricals:
    x_train[c] = pd.Categorical(x_train[c]).codes

x_train.loc[x_train.readmitted != 'NO','readmitted'] = 0
x_train.loc[x_train.readmitted == 'NO','readmitted'] = 1

x_train.readmitted = x_train.readmitted.astype(np.int8)
y_train = x_train.readmitted
x_train = x_train.drop('readmitted', axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=42)
# Train model
# Use "gpu_hist" for training the model.
reg = xgb.XGBRegressor(tree_method="approx")
# Fit the model using predictor X and response y.
reg.fit(X_train, Y_train)

cal_run_time(start_time, time.time())

# Save model into JSON format.
reg.save_model("model.json")

# importance = reg.get_booster().get_score()
# tuples = [(k, importance[k]) for k in importance]
# tuples = sorted(tuples, key=lambda x: x[1],reverse=True)
# print(tuples)
# print(len(tuples))
#
# # Drawing result
# xgb.plot_importance(reg)
# # dtest = xgb.DMatrix(X_test)
# y_pred = reg.predict(X_test)
#
# y_pred_tag = [1 if y >= 0.5 else 0 for y in y_pred]
# ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(Y_test, y_pred_tag)).plot()
#
# fpr, tpr, thresholds = roc_curve(Y_test, y_pred)
# plt.plot(fpr, tpr, label=f'AUC:{auc(fpr, tpr)}')
# plt.xlabel('fpr')
# plt.ylabel('tpr')
# plt.legend()
# plt.title('ROC curve')
# plt.show()

