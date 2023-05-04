#======================================================================================
# Arxiv_classifiying.py
#======================================================================================
'''
This file is used for classifying all the articles using the following steps:

- Read the configurations that will be passed when executing 'spark-submit', such as the file name.
- Read the pre-processing delta table into a dataframe.
- Create a pipeline that contains all the transformations needed to classify the articles into their main category.
- Run 'pipeline.fit()', which will produce a pipeline model that we will use for testing the data.
- After predicting the test data, we will save the prediction results into a delta table.
- To save the results in a delta table, we will check if the delta table exists. If it does, we will merge the data, otherwise we will insert a new file
'''
#======================================================================================

import sys
import pyspark
import time
from pyspark.sql import SparkSession 
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import col, split, when, count
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC, NaiveBayes
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.feature import StringIndexer, RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import concat_ws
from delta import DeltaTable, configure_spark_with_delta_pip


def Create_ML_pipline(ML_model = "NB"):
  # Convert the main_category column to numeric using StringIndexer
  labelIndexer = StringIndexer(inputCol="main_category", outputCol="label")

  # Define the regular expression tokenizer
  regexTokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\W")

  # Define the stop words remover
  stopWordsRemover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")

  # Define the TF-IDF Vectorizer
  countVectorizer = CountVectorizer(inputCol="filtered_tokens", outputCol="vectorize_features")
  idf = IDF(inputCol="vectorize_features", outputCol="features")

  if ML_model == 'LR': # Create logistic regression classifier     
    # it take time to execute
    #ML_Model = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100)
    ML_Model = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
  elif ML_model == 'RF': # Create a Random Forest classifier
    ML_Model = RandomForestClassifier(numTrees=100, maxDepth=5, labelCol="label", featuresCol="features")
  elif ML_model == 'NB': # Create a Naive Bayes classifier
    ML_Model = NaiveBayes(smoothing=1.0, modelType="multinomial", labelCol="label", featuresCol="features")

  # Define the Pipeline
  pipeline = Pipeline(stages=[labelIndexer, regexTokenizer, stopWordsRemover, countVectorizer, idf, ML_Model])

  return pipeline

   
if __name__ == '__main__':
  
    # ===================================================================
    # Read Configuration & application arguments
    # ===================================================================    
    testpart = sys.argv[1]  # determine the testing file that we will predict it  
    # input files = the delta tables that we save preprocessing, Training and Testing files 
    #======================================================================================
    delta_preprocessing_file = "hdfs:///Dat500_Group09/spark_result/preprocessing"
    delta_training_file = "hdfs:///Dat500_Group09/spark_result/training"
    delta_testing_file = "hdfs:///Dat500_Group09/spark_result/testing/"+ testpart
    
    # output files : the prediction result will be saved into Delta table
    #======================================================================================
    delta_table_path = "hdfs:///Dat500_Group09/spark_result/final_result/arxiv_meta"
    
    spark = (SparkSession
      .builder      
      .master('yarn')
      .appName("Arxiv_Classification_NB")
      .config('spark.driver.memory', "8g")      
      .config('spark.executor.instances', 12)
      .config("spark.executor.memory", "1g")       
      #.config("spark.sql.shuffle.partitions", 24)
      .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
      .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
      .config("spark.databricks.delta.properties.defaults.autoOptimize.optimizeWrite", "true")
      .config("spark.databricks.delta.properties.defaults.autoOptimize.autoCompact", "true")      
      .getOrCreate())  
      
    
  # create the pipeline 
    pipeline = Create_ML_pipline()

    # Reading from delta table Training & Testing data
    trainingData = spark.read.format("delta").load(delta_training_file)

    trainingData.cache()

    trainingData.count()

    
    testData = spark.read.format("delta").load(delta_testing_file)
         

    # Fit the model
    ML_model = pipeline.fit(trainingData)

    # Make predictions on the testing data
    predictions = ML_model.transform(testData)
    
    # Display dataframe with the original main_category and the predicted one
    df_Prediction = predictions.select("id", "main_category", "label", "prediction")
    #df_Prediction.show(3)

    # Evaluate the model using the Accuracy
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy_score = evaluator.evaluate(df_Prediction)
    print("Accuracy_score = %g" % accuracy_score)

    # save the prediction result into delta table
    #======================================================
    print("="*100)
    print("Save the prediction result into delta table")
    print("="*100)

    # check if the Delta table exists  
    #DeltaTable
    if DeltaTable.isDeltaTable(spark, delta_table_path):
        print("update delta table")
        deltaTable = DeltaTable.forPath(spark, delta_table_path)
        #"target.id = updates.id and target.main_category = updates.main_category") \
        deltaTable.alias("target") \
            .merge(
            source = df_Prediction.alias("updates"),
            condition = "target.id = updates.id") \
            .whenMatchedUpdate( set = 
            {
              "label": "updates.label",
              "prediction": "updates.prediction"     
            }) \
            .whenNotMatchedInsert(values =
            {
              "id": "updates.id",
              "main_category": "updates.main_category",
              "label": "updates.label",
              "prediction": "updates.prediction"        
            }) \
            .execute()
    else: # file not exists
        print("Create delta table first time")
        df_Prediction.write.format("delta").save(delta_table_path)        

   
    spark.stop()





