# flights

## Description 

These files import, transform, and model data to predict flight delays in the US from 2018 to 2022. We use Databricks to import our data and run anomaly detection, hybrid model predictions, and some summary statistics. We then use Snowflake to explore other
big-data tools and find it had a nicer SQL experience. 

We run some boosted classification models, hybrid random forest/markov chain models, and anomaly detection using SQL and python. Finally, we use Tableau to provide interactive visuals to display airports and identify the best flights to take for target dates. 

The Tableau notebook can be found [here](https://public.tableau.com/app/profile/rebecca.barth/viz/FlightDVAProjectv2/USFlightDelays#1). 

## Installation 

We downloaded parquet files for our imported data, and then used Databricks, Snowflake, and Tableau to manipulate and model our data using SQL and Python. 

First, download the data from [Kaggle](https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022/data). 

We recommend downloading the Parquet files, as they are better compressed and contain column metadata already. 
For any Databricks work, use the load_flight_data file to load data. Then, you can run the other Databricks files using the loaded data. 

We also explored using Snowflake, which is a key Databricks competitor. Snowflake has native parquet import abilities. 
Once imported into Snowflake from their import tool to a newly created flights.public database/schema, see the classification_snowflake_sql.sql, snowflake_notebook_environment.yml, and snowflake_notebook_python files. Note that snowflake_notebook_python is not a direct python file, but a notebook file and requires classification_snowflake_sql.sql. 

Finally, you will need to either download Tableau Desktop, or use our link to the Tableau workbook in Tableau Public. A free version of Tableau Desktop for students and instructors can be downloaded [here](https://www.tableau.com/community/academic).


## Execution 

For Snowflake, after importing your data via the data import tool in Snowflake, run classification_snowflake_sql.sql. This will create the relevant tables and views. Some key transformations are finding the datetimes, reducing the columns used for faster query execution, and doing a self-join to get the preceding flight in the time series data. This provides a clear 1-step delay propagation table. 

Then, run the Python scripts with the corresponding yml file. This loads the data, takes a sample and removes some outliers, and splits the data into train and test data. While pandas might not normally be good for data this size, we also use Modin pandas, which is more parallelizable on Snowflake's engine. We then run boosted models on training data to classify delays 
greater than 20 minutes. We did a gridsearch to find the best parameters, and then plot the predictions and also some evaluation tables.
Finally, we use SHAP plots to identify the key features and how they impact our prediction output.

The random forest, markov, and hybrid models are contained within a Databricks source file: RF_and_Propagation_Models.py. Once the Parquet files are downloaded, the source code can be imported into databricks. This file trains two random forest models using pyspark’s pyspark.ml.regression.RandomForestRegressor and displays important features. It then links sequential flight data and uses this to create a markov model, calculating transition probabilities between delay states. These models are then combined to make predictions on our data. For further visualization, the result predictions made on the full 2018 data can be downloaded after running the source file, namely the display(final_df). Evaluation statistics are then provided on a sample of unseen data at the end.

Load the results of the hybrid model predictions into Snowflake in flights.public.flight_prediction_data, then run the two queries in tableau_snowflake_sql.sql. The results of each query are loaded into Tableau Desktop. Within Tableau Desktop, create 3 visualizations: (1) a map with the amount of delay on Size and Color, (2) a bar chart showing delays by Airline, and (3) a table showing flight information with average historical delay time and predicted delay time. Report 2 should be used in the tooltip of report 1. Add report 1 and 3 in a dashboard along with necessary parameters for interactivity. The final dashboard can be uploaded to Tableau Public.
