import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from utils.timePresenter import cal_run_time


# Create SparkSession
spark = SparkSession.builder \
      .master("local[1]") \
      .appName("modelTrainer") \
      .getOrCreate()

start_time = time.time()
# x_train = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load("hdfs://172.31.58.70:9000/data/new_diabetic_data.csv")  # Use this line at Linux
# x_train = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load("file:///~/cs230/spark/data/new_diabetic_data.csv")
x_train = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('./data/new_diabetic_data.csv')  # Use this line at Windows

# todo: Data Processing
columns = [
    'num_lab_procedures', 'diag_1_code', 'diag_2_code', 'diag_3_code', 'num_medications', 'age',
    'discharge_disposition_id', 'medical_specialty_code', 'time_in_hospital', 'num_procedures',
    'readmitted']
x_train = x_train.select(columns)

x_train = x_train.withColumn('age', when(x_train.age == '[0-10)', 0).otherwise(x_train.age))
x_train = x_train.withColumn('age', when(x_train.age == '[10-20)', 10).otherwise(x_train.age))
x_train = x_train.withColumn('age', when(x_train.age == '[20-30)', 20).otherwise(x_train.age))
x_train = x_train.withColumn('age', when(x_train.age == '[30-40)', 30).otherwise(x_train.age))
x_train = x_train.withColumn('age', when(x_train.age == '[40-50)', 40).otherwise(x_train.age))
x_train = x_train.withColumn('age', when(x_train.age == '[50-60)', 50).otherwise(x_train.age))
x_train = x_train.withColumn('age', when(x_train.age == '[60-70)', 60).otherwise(x_train.age))
x_train = x_train.withColumn('age', when(x_train.age == '[70-80)', 70).otherwise(x_train.age))
x_train = x_train.withColumn('age', when(x_train.age == '[80-90)', 80).otherwise(x_train.age))
x_train = x_train.withColumn('age', when(x_train.age == '[90-100)', 90).otherwise(x_train.age))
x_train = x_train.withColumn("age", x_train["age"].cast('int'))

x_train = x_train.withColumn('readmitted', when(x_train.readmitted != 'NO', 0).otherwise(1))
x_train = x_train.withColumn("readmitted", x_train["readmitted"].cast('int'))

pick_columns = ['num_lab_procedures', 'diag_1_code', 'diag_2_code', 'diag_3_code', 'num_medications', 'age',
                'discharge_disposition_id', 'medical_specialty_code', 'time_in_hospital', 'num_procedures']
assembler = VectorAssembler(inputCols=pick_columns, outputCol="features")
x_train = assembler.transform(x_train)

# todo: Split data to train and test
x_train = x_train.select(['features', 'readmitted'])
print('features and readmitted (label): ')
x_train.show(5)
(train, test) = x_train.randomSplit([0.7, 0.3])

# todo: Prediction
reg = DecisionTreeRegressor(featuresCol='features', labelCol='readmitted')
reg_model = reg.fit(train)

reg_prediction = reg_model.transform(test)
cal_run_time(start_time, time.time())

print('Prediction result: ')
reg_prediction.select('features', 'readmitted', 'prediction').show(5)

reg_evaluator = RegressionEvaluator(labelCol='readmitted', predictionCol='prediction', metricName='rmse')
print("r2 = %g" % reg_evaluator.evaluate(reg_prediction, {reg_evaluator.metricName: "r2"}))
print("mse = %g" % reg_evaluator.evaluate(reg_prediction, {reg_evaluator.metricName: "mse"}))
print("rmse = %g" % reg_evaluator.evaluate(reg_prediction, {reg_evaluator.metricName: "rmse"}))
print("mae = %g" % reg_evaluator.evaluate(reg_prediction, {reg_evaluator.metricName: "mae"}))

# reg_model.write().overwrite().save("hdfs://172.31.58.70:9000/data/model")  # Run this when hdfs is running!




