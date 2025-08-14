# Databricks notebook source
# MAGIC %md
# MAGIC # NOTE: The following code takes 4+ hours to run to completion. 
# MAGIC

# COMMAND ----------

#Load Data
df_2018 = spark.read.format("parquet").load("/FileStore/tables/Combined_Flights_2018.parquet")
df_2019 = spark.read.format("parquet").load("/FileStore/tables/Combined_Flights_2019.parquet")
df_2020 = spark.read.format("parquet").load("/FileStore/tables/Combined_Flights_2020.parquet")
df_2021 = spark.read.format("parquet").load("/FileStore/tables/Combined_Flights_2021.parquet")
df_2022 = spark.read.format("parquet").load("/FileStore/tables/Combined_Flights_2022.parquet")

# COMMAND ----------

#For now, will just model on 2018
print(f'num rows: {df_2018.count()}')
print(f'num cols: {len(df_2018.columns)}')

# COMMAND ----------

# MAGIC %md
# MAGIC # Random Forest

# COMMAND ----------

#Random Forest for Arrival
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

#note: for smaller subset of data, scikit learn would likely work. But since training/testing models for a larger dataset, will use pyspark

#Trying to predict ArrDelayMinutes, using Departure Features, Scheduled & Actual Flight Time, Flight Schedule Information, Flight Route/Distance,
#Day & Time Factors, and Airport/Airline Information
y = "ArrDelayMinutes"

relevant_features = [
    "DepDelayMinutes", "DepartureDelayGroups", "CRSElapsedTime", "ActualElapsedTime", "AirTime", "TaxiOut", "TaxiIn",
    "CRSDepTime", "CRSArrTime", "DepTimeBlk", "ArrTimeBlk",
    "Distance", "DayOfWeek", "Month", "Quarter", "DayofMonth",
    "OriginAirportID", "DestAirportID", "Marketing_Airline_Network"
]

#Split into numerical and categorical
num_features = [
    "DepDelayMinutes", "DepartureDelayGroups", "CRSElapsedTime", "ActualElapsedTime",
    "AirTime", "TaxiOut", "TaxiIn", "CRSDepTime", "CRSArrTime", "Distance", 
    "DayOfWeek", "Month", "Quarter", "DayofMonth", "OriginAirportID", "DestAirportID"
]
cat_features = ["Marketing_Airline_Network", "DepTimeBlk", "ArrTimeBlk"]

#Convert categorical features to numerical (encoded)
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_Index", handleInvalid="keep") for col in cat_features]

#Assemble features in a single vector
assembler = VectorAssembler(inputCols=num_features + [f"{col}_Index" for col in cat_features], outputCol="features")

#split data
train_df, test_df = df_2018.randomSplit([0.8, 0.2], seed=6242)
#drop rows with null values
train_df = train_df.na.drop()
test_df = test_df.na.drop()

#Define RF model
rf = RandomForestRegressor(featuresCol="features", labelCol=y, numTrees=100, maxDepth=10, seed=42)

#Build pipeline (Encoding → Vectorization → RF Model)
pipeline = Pipeline(stages=indexers + [assembler, rf])

#Train model
rf_model = pipeline.fit(train_df)

#Make predictions
predictions = rf_model.transform(test_df)

#Evaluate w/ RMSE
evaluator = RegressionEvaluator(labelCol=y, predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

print(f"Root Mean Squared Error (RMSE): {rmse}")

# COMMAND ----------

# #Save the model (Doesn't work in community edition?)
# rf_model.save("rf_model")
# from pyspark.ml.regression import RandomForestRegressionModel
# loaded_model = RandomForestRegressionModel.load("rf_model")

# COMMAND ----------

#Variable importance
#NOTE: IN OUR FINAL MODEL WE USE THE FULL MODEL. 
import pandas as pd

#Get feature importances
rf_stage = rf_model.stages[-1]  # Last stage is the RF model
importances = rf_stage.featureImportances
all_feature_names = num_features + [f"{col}_Index" for col in cat_features]

#Create a DataFrame of feature importance
importance_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': importances.toArray()
})

#Sort by importance and get top 10
top_10_features = importance_df.sort_values('importance', ascending=False).head(10)['feature'].values

print("Top 10 most important features:")
print(top_10_features)

#Identify which original features these correspond to
top_10_original_features = []
encoded_to_original = {f"{col}_Index": col for col in cat_features}
for feature in top_10_features:
    if feature in encoded_to_original:
        top_10_original_features.append(encoded_to_original[feature])
    else:
        top_10_original_features.append(feature)
print("\nImportant feature names:")
print(top_10_original_features)

# #Separate back into numerical and categorical
# new_num_features = [f for f in top_10_original_features if f in num_features]
# new_cat_features = [f for f in top_10_original_features if f in cat_features]

# print("\Important numerical features:", new_num_features)
# print("Important categorical features:", new_cat_features)

# #Retrain RF model with only these features
# new_indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_Index", handleInvalid="keep") 
#                 for col in new_cat_features]

# new_assembler = VectorAssembler(
#     inputCols=new_num_features + [f"{col}_Index" for col in new_cat_features],
#     outputCol="features"
# )

# #New RF model
# new_rf = RandomForestRegressor(featuresCol="features", labelCol=y, 
#                               numTrees=100, maxDepth=10, seed=6242)

# new_pipeline = Pipeline(stages=new_indexers + [new_assembler, new_rf])
# new_rf_model = new_pipeline.fit(train_df)
# new_predictions = new_rf_model.transform(test_df)
# #Evaluate
# new_rmse = evaluator.evaluate(new_predictions)

# print(f"\nOriginal RMSE: {rmse}")
# print(f"New RMSE with top 10 features: {new_rmse}")

# COMMAND ----------

#Random Forest for Departure
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

#note: for smaller subset of data, scikit learn would likely work. But since training/testing models for a larger dataset, will use pyspark

#Trying to predict DepDelayMinutes
y = "DepDelayMinutes"

relevant_features = [
    "CRSElapsedTime", "ActualElapsedTime", "AirTime", "TaxiOut", "TaxiIn",
    "CRSDepTime", "CRSArrTime", "DepTimeBlk", "ArrTimeBlk",
    "Distance", "DayOfWeek", "Month", "Quarter", "DayofMonth",
    "OriginAirportID", "DestAirportID", "Marketing_Airline_Network"
]

#Split into numerical and categorical
num_features = [
    "CRSElapsedTime", "ActualElapsedTime",
    "AirTime", "TaxiOut", "TaxiIn", "CRSDepTime", "CRSArrTime", "Distance", 
    "DayOfWeek", "Month", "Quarter", "DayofMonth", "OriginAirportID", "DestAirportID"
]
cat_features = ["Marketing_Airline_Network", "DepTimeBlk", "ArrTimeBlk"]

#Convert categorical features to numerical (encoded)
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_Index", handleInvalid="keep") for col in cat_features]

#Assemble features in a single vector
assembler = VectorAssembler(inputCols=num_features + [f"{col}_Index" for col in cat_features], outputCol="features")

#split data
train_df, test_df = df_2018.randomSplit([0.8, 0.2], seed=6242)
#drop rows with null values
train_df = train_df.na.drop()
test_df = test_df.na.drop()

#Define RF model
rf_dep = RandomForestRegressor(featuresCol="features", labelCol=y, numTrees=100, maxDepth=10, seed=42)

#Build pipeline (Encoding → Vectorization → RF Model)
pipeline = Pipeline(stages=indexers + [assembler, rf])

#Train model
rf_dep_model = pipeline.fit(train_df)

#Make predictions
predictions = rf_dep_model.transform(test_df)

#Evaluate w/ RMSE
evaluator = RegressionEvaluator(labelCol=y, predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

print(f"Root Mean Squared Error (RMSE): {rmse}")

# COMMAND ----------

#Variable importance
import pandas as pd

#Get feature importances
rf_stage = rf_dep_model.stages[-1]  # Last stage is the RF model
importances = rf_stage.featureImportances
all_feature_names = num_features + [f"{col}_Index" for col in cat_features]

#Create a DataFrame of feature importance
importance_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': importances.toArray()
})

#Sort by importance and get top 10
top_10_features = importance_df.sort_values('importance', ascending=False).head(10)['feature'].values

print("Top 10 most important features:")
print(top_10_features)

#note: the following can be used to retrain RF model with top 10 important features
# #Identify which original features these correspond to
# top_10_original_features = []
# encoded_to_original = {f"{col}_Index": col for col in cat_features}
# for feature in top_10_features:
#     if feature in encoded_to_original:
#         top_10_original_features.append(encoded_to_original[feature])
#     else:
#         top_10_original_features.append(feature)
# print("\nImportant feature names:")
# print(top_10_original_features)

# #Separate back into numerical and categorical
# new_num_features = [f for f in top_10_original_features if f in num_features]
# new_cat_features = [f for f in top_10_original_features if f in cat_features]

# print("\Important numerical features:", new_num_features)
# print("Important categorical features:", new_cat_features)

# #Retrain RF model with only these features
# new_indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_Index", handleInvalid="keep") 
#                 for col in new_cat_features]

# new_assembler = VectorAssembler(
#     inputCols=new_num_features + [f"{col}_Index" for col in new_cat_features],
#     outputCol="features"
# )

# #New RF model
# new_dep_rf = RandomForestRegressor(featuresCol="features", labelCol=y, 
#                               numTrees=100, maxDepth=10, seed=6242)

# new_pipeline = Pipeline(stages=new_indexers + [new_assembler, new_rf])
# new_dep_rf_model = new_pipeline.fit(train_df)
# new__dep_predictions = new_dep_rf_model.transform(test_df)
# #Evaluate
# new_rmse = evaluator.evaluate(new_predictions)

# print(f"\nOriginal RMSE: {rmse}")
# print(f"New RMSE with top 10 features: {new_rmse}")

# COMMAND ----------

# MAGIC %md
# MAGIC Link Flights (Joel's Code)

# COMMAND ----------

df_2018.createOrReplaceTempView("df_2018")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     CRSDepTime 
# MAGIC   , FlightDate
# MAGIC   , LEFT(FlightDate::STRING, 11)
# MAGIC   , LPAD(CRSDepTime::STRING, 4, '0') AS dept_string 
# MAGIC   , CONCAT(LEFT(FlightDate::STRING, 11), LEFT(dept_string, 2), ':', RIGHT(dept_string, 2), ':00.000')::TIMESTAMP AS flight_time 
# MAGIC   , Origin 
# MAGIC   , Airline 
# MAGIC   , Dest 
# MAGIC   , Diverted
# MAGIC   , ArrDelayMinutes
# MAGIC   , DepDelayMinutes -- only positive delay 
# MAGIC   , DepDelay -- positive and negative (positive delay is bad)
# MAGIC   , Distance 
# MAGIC   , Month AS month_date 
# MAGIC   , DayOfWeek 
# MAGIC   , Tail_Number 
# MAGIC   , Cancelled
# MAGIC FROM
# MAGIC     df_2018 

# COMMAND ----------

_sqldf.createOrReplaceTempView("_sqldf")

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT 
# MAGIC     df1.flight_time  
# MAGIC   , df1.Origin 
# MAGIC   , df1.Airline 
# MAGIC   , df1.Dest 
# MAGIC   , df1.Diverted
# MAGIC   , df1.DepDelayMinutes -- only positive delay 
# MAGIC   , df1.DepDelay -- positive and negative (positive delay is bad)
# MAGIC   , df1.Distance 
# MAGIC   , df1.month_date 
# MAGIC   , df1.DayOfWeek 
# MAGIC   , df1.Tail_Number 
# MAGIC
# MAGIC
# MAGIC -- flights departing before flight we are considering 
# MAGIC   , df2.flight_time AS flight_time_pre
# MAGIC   , df2.Origin AS Origin_pre
# MAGIC   , df2.Airline AS  Airline_pre
# MAGIC   , df2.Dest AS  Dest_pre
# MAGIC   , df2.Diverted AS Diverted_pre
# MAGIC   , df2.ArrDelayMinutes AS ArrDelay_pre
# MAGIC   , df2.DepDelayMinutes AS  DepDelayMinutes_pre -- only positive delay 
# MAGIC   , df2.DepDelay AS DepDelay_pre -- positive and negative (positive delay is bad)
# MAGIC   , df2.Distance AS  Distance_pre
# MAGIC   , df2.month_date AS month_date_pre
# MAGIC   , df2.DayOfWeek AS  DayOfWeek_pre
# MAGIC   , df2.Tail_Number AS  Tail_Number_pre
# MAGIC
# MAGIC
# MAGIC FROM 
# MAGIC     _sqldf AS df1 
# MAGIC LEFT JOIN 
# MAGIC     _sqldf AS df2 
# MAGIC     ON df1.Tail_Number = df2.Tail_Number
# MAGIC     AND df1.Airline = df2.Airline 
# MAGIC     AND DATEDIFF(MINUTE, df2.flight_time, df1.flight_time) < (60*24*2)
# MAGIC     AND df1.flight_time > df2.flight_time
# MAGIC WHERE 
# MAGIC     df1.Cancelled = FALSE 
# MAGIC QUALIFY 
# MAGIC     ROW_NUMBER() OVER (PARTITION BY df1.flight_time, df1.Tail_Number, df1.Airline ORDER BY df2.flight_time DESC) = 1

# COMMAND ----------

linked_df = _sqldf
# print(f'num rows: {linked_df.count()}')

# COMMAND ----------

# MAGIC %md
# MAGIC # Propagation Model

# COMMAND ----------

#Delay Propagation Model 1(Markov-Based)
from pyspark.sql import functions as F
from pyspark.sql.window import Window

#Define state mapping function
def get_delay_state(delay):
    if delay <= 0: #early
        return 0
    elif delay <= 15: #on-time
        return 1
    elif delay <= 60: #small delay
        return 2
    elif delay <= 180: #significant delay
        return 3
    else: #severe delay
        return 4

#Convert to UDF (to work with spark cols)
get_delay_state_udf = F.udf(get_delay_state)

#Remove the first flights of each chain (need a previous state to calculate state transnitions) and remove missing data points
cleaned_df = linked_df.filter(F.col("DepDelay").isNotNull() & F.col("ArrDelay_pre").isNotNull())

#Assign states to past and current flights
df_markov = cleaned_df.select(
    F.col("Tail_Number"),
    F.col("Airline"),
    get_delay_state_udf(F.col("DepDelay")).alias("current_state"),
    get_delay_state_udf(F.col("ArrDelay_pre")).alias("previous_state")
).dropna()

#Count num transitions for each pair 
transition_counts = df_markov.groupBy("previous_state", "current_state").count()

#Get transition probabilities and comp matrix
transition_matrix = transition_counts.withColumn(
    "probability", 
    F.col("count") / F.sum("count").over(Window.partitionBy("previous_state"))
).drop("count")

transition_matrix.show()

# COMMAND ----------

#store and pshow transition matrix (I store as separate df b/c pyspark's lazy eval)
from pyspark.sql import Row

rows = [
    Row(previous_state='0', current_state='0', probability=0.783344254870463),
    Row(previous_state='0', current_state='1', probability=0.136727926346852),
    Row(previous_state='0', current_state='2', probability=0.053644231517716245),
    Row(previous_state='0', current_state='3', probability=0.02135573309933409),
    Row(previous_state='0', current_state='4', probability=0.004927854165634664),
    Row(previous_state='1', current_state='0', probability=0.5361136068155633),
    Row(previous_state='1', current_state='1', probability=0.3039257137421121),
    Row(previous_state='1', current_state='2', probability=0.12538712780238248),
    Row(previous_state='1', current_state='3', probability=0.028284376420349446),
    Row(previous_state='1', current_state='4', probability=0.006289175219592631), 
    Row(previous_state='2', current_state='0', probability=0.31949769291918323),
    Row(previous_state='2', current_state='1', probability=0.18524553617262185),
    Row(previous_state='2', current_state='2', probability=0.42301126113573495),
    Row(previous_state='2', current_state='3', probability=0.062443716321460105),
    Row(previous_state='2', current_state='4', probability=0.009801793450999833),
    Row(previous_state='3', current_state='0', probability=0.2975370954468231),
    Row(previous_state='3', current_state='1', probability=0.07019613522133646),
    Row(previous_state='3', current_state='2', probability=0.1644314621663529),
    Row(previous_state='3', current_state='3', probability=0.4404477237647508),
    Row(previous_state='3', current_state='4', probability=0.027387583400736788),
    Row(previous_state='4', current_state='0', probability=0.1432647100726599),
    Row(previous_state='4', current_state='1', probability=0.08690696680438809),
    Row(previous_state='4', current_state='2', probability=0.0937811654081778),
    Row(previous_state='4', current_state='3', probability=0.26382319418720614),    
    Row(previous_state='4', current_state='4', probability=0.41222396352756804),
]

#Convert to DataFrame
transition_matrix_df = spark.createDataFrame(rows)

#Show matrix  
transition_matrix_df.show(25, False)


# COMMAND ----------

#Get avg delay for each state
def get_delay_state(delay):
    if delay <= 0: #early
        return 0
    elif delay <= 15: #on-time
        return 1
    elif delay <= 60: #small delay
        return 2
    elif delay <= 180: #significant delay
        return 3
    else: #severe delay
        return 4


#Convert to UDF (to work with spark cols)
get_delay_state_udf = F.udf(get_delay_state)

df = df_2018.na.drop()
df_with_states = df.withColumn("DelayState", get_delay_state_udf(col("DepDelayMinutes")))
#Group by state and calculate average delay
avg_delay_per_state = df_with_states.groupBy("DelayState").avg("DepDelayMinutes").orderBy("DelayState")

#Show result
avg_delay_per_state.show()

# COMMAND ----------

#Convert this transition matrix to a function. The Markov Model will take a previous arrival delay value (prediction) and return a departure delay prediction
#as a weighted linear combination using the avg delay of each state and the probabilities

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

def get_delay_state(delay):
    if delay <= 0: #early
        return 0
    elif delay <= 15: #on-time
        return 1
    elif delay <= 60: #small delay
        return 2
    elif delay <= 180: #significant delay
        return 3
    else: #severe delay
        return 4
    
state_delays = {
    0: 0.0, 1: 6.412491631003583, 2: 32.00369774602746,
    3: 100.45319493282997, 4: 305.50422827753323}

transition_matrix = {
    0: {0: 0.783344254870463, 1: 0.136727926346852, 2: 0.053644231517716245, 3: 0.02135573309933409, 4: 0.004927854165634664},
    1: {0: 0.5361136068155633, 1: 0.3039257137421121, 2: 0.12538712780238248, 3: 0.028284376420349446, 4: 0.006289175219592631},
    2: {0: 0.31949769291918323, 1: 0.18524553617262185, 2: 0.42301126113573495, 3: 0.062443716321460105, 4: 0.009801793450999833},
    3: {0: 0.2975370954468231, 1: 0.07019613522133646, 2: 0.1644314621663529, 3: 0.4404477237647508, 4: 0.027387583400736788},
    4: {0: 0.1432647100726599, 1: 0.08690696680438809, 2: 0.0937811654081778, 3: 0.2638231941872061, 4: 0.412223963527568}
}

def get_markov_pred(prev_delay_val):
    if prev_delay_val is None: #return nulls for nulls
        return None
    prev_state = get_delay_state(prev_delay_val)
    transitions = transition_matrix[prev_state]
    expected_delay = sum(state_delays[state] * prob for state, prob in transitions.items())
    return expected_delay

print(get_markov_pred(54))
#udf to use on spark cols
get_markov_pred_udf = F.udf(get_markov_pred, DoubleType())


# COMMAND ----------

# MAGIC %md
# MAGIC # Making Predictions

# COMMAND ----------

#Example Hybrid Model Predictions
#steps:
#If have previous flight:
#  Predict previous flights' ArrDelay w/ first RF model
#  Convert to delay state and use Markov model to get Markov_Pred_DepDelay
#  Use RF model to get RF_Pred_DepDelay (based on current flight’s features)
#  Hybrid prediction = weighted combo of Markov_Pred_DepDelay and RF_Pred_DepDelay
#If no previous flight:
#  Use RF model to get RF_Pred_DepDelay

#Example day of flights that I will predict delay on
from pyspark.sql.functions import col
df_first_day = df_2018.filter(
    (col("Year") == 2018) & (col("Month") == 1) & (col("DayofMonth") == 1)
)
df_first_day = df_first_day.na.drop()
print(f'num rows: {df_first_day.count()}')
display(df_first_day.head(10))


# COMMAND ----------

#Get rf predictions
rf_arr_predictions = rf_model.transform(df_first_day)
rf_dep_predictions = rf_dep_model.transform(df_first_day)


# COMMAND ----------

display(rf_arr_predictions) 

# COMMAND ----------

display(rf_dep_predictions)

# COMMAND ----------

#Combine Predictions
from pyspark.sql.functions import monotonically_increasing_id, col

#create row_id to join (many overlapping same features)
arr_pred = rf_arr_predictions.withColumn("row_id", monotonically_increasing_id())
dep_pred = rf_dep_predictions.withColumn("row_id", monotonically_increasing_id())
dep_pred_sel = dep_pred.select("row_id", col("prediction").alias("RFPredDep"))

#Join on row_id
joined_df = arr_pred.join(dep_pred_sel, on="row_id", how="inner")

#Rename DepDelayMinutes → Actual and prediction → PredArr
combined_df = joined_df.withColumnRenamed("DepDelayMinutes", "ActualDep") \
                    .withColumnRenamed("prediction", "RFPredArr")

combined_df.select("Origin", "Dest", "ActualDep", "RFPredArr", "RFPredDep").show(10)


# COMMAND ----------

#Now link flights 
from pyspark.sql.window import Window
from pyspark.sql.functions import lag

#ordered by time within each aircraft (Tail_Number)
window_spec = Window.partitionBy("Tail_Number", "FlightDate").orderBy("CRSDepTime")

#Add previous flight's predicted arrival delay
linked_df = combined_df.withColumn("Prev_RFPredArr", lag("RFPredArr").over(window_spec))

linked_df.select("Origin", "Dest", "ActualDep", "Prev_RFPredArr", "RFPredDep").show(10)

# COMMAND ----------

#Extract and store useful columns
from pyspark.sql.functions import col, when

#Get RF predictions
rf_pred_df = linked_df.select("Airline", "Origin", "Dest", "ActualDep", "Prev_RFPredArr", "RFPredDep")
rf_pred_df.show(10)

#Get Markov Predictions
markov_pred_df = rf_pred_df.withColumn("MarkovPredDep", get_markov_pred_udf(F.col("Prev_RFPredArr")))
markov_pred_df.show(10)

#Create Hybrid Predictions
hybrid_pred_df = markov_pred_df.withColumn("HybridPredDep",when(
        col("MarkovPredDep").isNotNull(),
        0.5 * col("RFPredDep") + 0.5 * col("MarkovPredDep")
    ).otherwise(col("RFPredDep"))
)
hybrid_pred_df.show(10)

# COMMAND ----------

final_df = hybrid_pred_df.select("Airline", "Origin", "Dest", "ActualDep", "RFPredDep", "MarkovPredDep", "HybridPredDep")
final_df = final_df.withColumnRenamed("ActualDep", "Actual Departure Delay") \
                    .withColumnRenamed("RFPredDep", "RF Prediction") \
                     .withColumnRenamed("MarkovPredDep", "Markov Prediction")\
                      .withColumnRenamed("HybridPredDep", "Hybrid Model Prediction")
final_df.show(10)

# COMMAND ----------

display(final_df) #Results can be downloaded.

# COMMAND ----------

#Predictions on whole data
df_cleaned = df_2018.na.drop()
print(f'num rows: {df_cleaned.count()}')
display(df_cleaned.head(10))

# COMMAND ----------

#get rf predictions
rf_arr_predictions3 = rf_model.transform(df_cleaned)
rf_dep_predictions3 = rf_dep_model.transform(df_cleaned)

# COMMAND ----------

display(rf_arr_predictions3)

# COMMAND ----------

display(rf_dep_predictions3)

# COMMAND ----------

#Combine Predictions
from pyspark.sql.functions import monotonically_increasing_id, col

#create row_id to join (many overlapping same features)
arr_pred = rf_arr_predictions3.withColumn("row_id", monotonically_increasing_id())
dep_pred = rf_dep_predictions3.withColumn("row_id", monotonically_increasing_id())
dep_pred_sel = dep_pred.select("row_id", col("prediction").alias("RFPredDep"))

#Join on row_id
joined_df = arr_pred.join(dep_pred_sel, on="row_id", how="inner")

#Rename DepDelayMinutes → Actual and prediction → PredArr
combined_df = joined_df.withColumnRenamed("DepDelayMinutes", "ActualDep") \
                    .withColumnRenamed("prediction", "RFPredArr")

combined_df.select("Origin", "Dest", "ActualDep", "RFPredArr", "RFPredDep").show(10)

# COMMAND ----------

#Now link flights 
from pyspark.sql.window import Window
from pyspark.sql.functions import lag

#ordered by time within each aircraft (Tail_Number)
window_spec = Window.partitionBy("Tail_Number", "FlightDate").orderBy("CRSDepTime")

#Add previous flight's predicted arrival delay
linked_df = combined_df.withColumn("Prev_RFPredArr", lag("RFPredArr").over(window_spec))

linked_df.select("Origin", "Dest", "ActualDep", "Prev_RFPredArr", "RFPredDep").show(10)

# COMMAND ----------

#Extract and store useful columns
from pyspark.sql.functions import col, when

#Get RF predictions
rf_pred_df = linked_df.select("FlightDate", "CRSDepTime", "CRSArrTime", "Airline", "Origin", "Dest", "ActualDep", "Prev_RFPredArr", "RFPredDep")
rf_pred_df.show(10)

#Get Markov Predictions
markov_pred_df = rf_pred_df.withColumn("MarkovPredDep", get_markov_pred_udf(F.col("Prev_RFPredArr")))
markov_pred_df.show(10)

#Create Hybrid Predictions
hybrid_pred_df = markov_pred_df.withColumn("HybridPredDep",when(
        col("MarkovPredDep").isNotNull(),
        0.5 * col("RFPredDep") + 0.5 * col("MarkovPredDep")
    ).otherwise(col("RFPredDep"))
)
hybrid_pred_df.show(10)

# COMMAND ----------

final_df = hybrid_pred_df.select("FlightDate", "CRSDepTime", "CRSArrTime", "Airline", "Origin", "Dest", "ActualDep", "RFPredDep", "MarkovPredDep", "HybridPredDep")
final_df = final_df.withColumnRenamed("ActualDep", "Actual Departure Delay") \ 
                    .withColumnRenamed("RFPredDep", "RF Prediction") \
                     .withColumnRenamed("MarkovPredDep", "Markov Prediction")\
                      .withColumnRenamed("HybridPredDep", "Hybrid Model Prediction")

# COMMAND ----------

display(final_df) #NOTE: Results should be downloaded as a csv.

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluating/Predicting On New Data

# COMMAND ----------

#NEW DATA (2019, which was not used to train)
#first 2 months of 2019
from pyspark.sql.functions import col
df_2019_2_months = df_2019.filter(
    (col("Year") == 2019) & (col("Month").isin([1,2]))
)
df_2019_2_months = df_2019_2_months.na.drop()
print(f'num rows: {df_2019_2_months.count()}')
display(df_2019_2_months.head(10))

# COMMAND ----------

rf_arr_predictions_2019 = rf_model.transform(df_2019_2_months)
rf_dep_predictions_2019 = rf_dep_model.transform(df_2019_2_months)

# COMMAND ----------

display(rf_arr_predictions_2019)

# COMMAND ----------

display(rf_dep_predictions_2019)

# COMMAND ----------

#Combine Predictions
from pyspark.sql.functions import monotonically_increasing_id, col

#create row_id to join (many overlapping same features)
arr_pred = rf_arr_predictions_2019.withColumn("row_id", monotonically_increasing_id())
dep_pred = rf_dep_predictions_2019.withColumn("row_id", monotonically_increasing_id())
dep_pred_sel = dep_pred.select("row_id", col("prediction").alias("RFPredDep"))

#Join on row_id
joined_df = arr_pred.join(dep_pred_sel, on="row_id", how="inner")

#Rename DepDelayMinutes → Actual and prediction → PredArr
combined_df = joined_df.withColumnRenamed("DepDelayMinutes", "ActualDep") \
                    .withColumnRenamed("prediction", "RFPredArr")

combined_df.select("Origin", "Dest", "ActualDep", "RFPredArr", "RFPredDep").show(10)

# COMMAND ----------

#Now link flights 
from pyspark.sql.window import Window
from pyspark.sql.functions import lag

#ordered by time within each aircraft (Tail_Number)
window_spec = Window.partitionBy("Tail_Number", "FlightDate").orderBy("CRSDepTime")

#Add previous flight's predicted arrival delay
linked_df = combined_df.withColumn("Prev_RFPredArr", lag("RFPredArr").over(window_spec))

linked_df.select("Origin", "Dest", "ActualDep", "Prev_RFPredArr", "RFPredDep").show(10)

#Extract and store useful columns
from pyspark.sql.functions import col, when

#Get RF predictions
rf_pred_df = linked_df.select("FlightDate", "Airline", "Origin", "Dest", "ActualDep", "Prev_RFPredArr", "RFPredDep")
rf_pred_df.show(10)

#Get Markov Predictions
markov_pred_df = rf_pred_df.withColumn("MarkovPredDep", get_markov_pred_udf(F.col("Prev_RFPredArr")))
markov_pred_df.show(10)

#Create Hybrid Predictions
hybrid_pred_df = markov_pred_df.withColumn("HybridPredDep",when(
        col("MarkovPredDep").isNotNull(),
        0.5 * col("RFPredDep") + 0.5 * col("MarkovPredDep")
    ).otherwise(col("RFPredDep"))
)
hybrid_pred_df.show(10)

# COMMAND ----------

final_df = hybrid_pred_df.select("FlightDate","Airline", "Origin", "Dest", "ActualDep", "RFPredDep", "MarkovPredDep", "HybridPredDep")
final_df = final_df.withColumnRenamed("ActualDep", "Actual Departure Delay") \
                    .withColumnRenamed("RFPredDep", "RF Prediction") \
                     .withColumnRenamed("MarkovPredDep", "Markov Prediction")\
                      .withColumnRenamed("HybridPredDep", "Hybrid Model Prediction")

# COMMAND ----------

display(final_df) #Note: Output hidden, but can be found as 2019_Hybrid_Predictions.csv

# COMMAND ----------

#Evaluate Models
from pyspark.sql.functions import col, when, pow, abs, avg, sqrt
#RF
rf_errors = final_df.withColumn("RF_Error", col("RF Prediction") - col("Actual Departure Delay")).select("RF_Error")
rf_errors = rf_errors.withColumn("RF_Squared", pow(col("RF_Error"), 2))\
                     .withColumn("RF_Abs", abs(col("RF_Error")))

rf_metrics = rf_errors.agg(
    sqrt(avg("RF_Squared")).alias("RF_RMSE"),
    avg("RF_Abs").alias("RF_MAE"),
)
rf_metrics.show()

#Markov
#remove rows with null
markov_cleaned = final_df.filter(col("Markov Prediction").isNotNull())
markov_errors = markov_cleaned.withColumn("Markov_Error", col("Markov Prediction") - col("Actual Departure Delay")).select("Markov_Error")
markov_errors = markov_errors.withColumn("Markov_Squared", pow(col("Markov_Error"), 2))\
                     .withColumn("Markov_Abs", abs(col("Markov_Error")))

markov_metrics = markov_errors.agg(
    sqrt(avg("Markov_Squared")).alias("Markov_RMSE"),
    avg("Markov_Abs").alias("Markov_MAE"),
)
markov_metrics.show()

#Hybrid
hybrid_errors = final_df.withColumn("Hybrid_Error", col("Hybrid Model Prediction") - col("Actual Departure Delay")).select("Hybrid_Error")
hybrid_errors = hybrid_errors.withColumn("Hybrid_Squared", pow(col("Hybrid_Error"), 2))\
                     .withColumn("Hybrid_Abs", abs(col("Hybrid_Error")))

hybrid_metrics = hybrid_errors.agg(
    sqrt(avg("Hybrid_Squared")).alias("Hybrid_RMSE"),
    avg("Hybrid_Abs").alias("Hybrid_MAE"),
)
hybrid_metrics.show()
