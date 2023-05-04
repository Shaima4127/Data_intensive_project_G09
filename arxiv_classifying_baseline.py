import sys
import pyspark
from pyspark.sql import SparkSession 
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import col, split, when, count
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.feature import StringIndexer, RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import concat_ws
from delta import DeltaTable, configure_spark_with_delta_pip



def Create_ML_pipline(ML_model = "LR"):
  # Convert the main_category column to numeric using StringIndexer
  labelIndexer = StringIndexer(inputCol="main_category", outputCol="label")

  # Define the regular expression tokenizer
  regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")

  # Define the stop words remover
  stopWordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

  # Define the TF-IDF Vectorizer
  countVectorizer = CountVectorizer(inputCol="filtered_words", outputCol="raw_features")
  # hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
  idf = IDF(inputCol="raw_features", outputCol="features")

  if ML_model == 'LR': # Create logistic regression classifier     
    ML_Model = LogisticRegression(featuresCol="features", labelCol="label")
  elif ML_model == 'RF': # Create a Random Forest classifier
    ML_Model = RandomForestClassifier(numTrees=100, maxDepth=5, labelCol="label", featuresCol="features")
  elif ML_model == 'GBT': # Create a Gradient-Boosted Trees classifier
    ML_Model = GBTClassifier(maxDepth=5, maxIter=100, labelCol="label", featuresCol="features")

  # Define the Pipeline
  pipeline = Pipeline(stages=[labelIndexer, regexTokenizer, stopWordsRemover, countVectorizer, idf, ML_Model])

  return pipeline

   
if __name__ == '__main__':
    # Read application arguments
    filepath = sys.argv[1]    
    partition = int(sys.argv[2])  # number of partition to repartition the dataframe     
    test_ratio = float(sys.argv[3])  # splitting ratio for the testing dataset
    bprint = int(sys.argv[4])  # pointer to print some result
  
    
    if not filepath:
        print("Error: Please enter the path for the data file",filepath, bprint )
        sys.exit(1)
    
    #filepath = 'hdfs:///Dat500_Group09/output_meta/part*'


    builder = pyspark.sql.SparkSession.builder.appName("Arxiv_Classification") \
      .master('yarn') \
      .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
      .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    
    # create data schema
    dbschema = StructType([
    StructField("id", StringType(), True),
    StructField("authors", StringType(), True),
    StructField("title", StringType(), True),
    StructField("abstract", StringType(), True),
    StructField("journal_ref", StringType(), True),
    StructField("category", StringType(), True),
    StructField("update_date", StringType(), True),
    ])

    # import csv file for the data
    try:
      arxiv_df =spark.read.options(delimiter="::", header=False, schema=dbschema).csv(filepath)    
    except:
        print(f"Error: Could not read the data for this file {filepath}")
        sys.exit(1)


    arxiv_df = arxiv_df.repartition(partition)
    
    # change the column names to the same name for the Arxiv metadata 
    arxiv_df = arxiv_df.selectExpr("_c0 as id", "_c1 as authors", "_c2 as title", "_c3 as abstract", 
                                "_c4 as journal_ref", "_c5 as category", "_c6 as update_date")
    
    # add new column for arxiv data to recategorize the articles
    arxiv_df = arxiv_df.withColumn("main_category",
                   when(split(col("category"), "\\.")[0] == "cs", "computer science")
                   .when(split(col("category"), "\\.")[0] == "math", "mathematics")
                   .when(split(col("category"), "\\.")[0] == "econ", "economics")
                   .when(split(col("category"), "\\.")[0] == "eess", "electrical engineering and systems science")
                   .when(split(col("category"), "\\.")[0] == "q-bio", "quantitative biology")
                   .when(split(col("category"), "\\.")[0] == "q-fin", "quantitative finance")
                   .when(split(col("category"), "\\.")[0] == "stat", "statistics")                   
                   .otherwise("physics"))
           
    if bprint == 1:      
      print("="*100)
      print("Spark driver configuration")
      print("="*100)
      print("The number of partition for the Data:",arxiv_df.rdd.getNumPartitions())
      print("the No of shuffle.partitions",  spark.conf.get("spark.sql.shuffle.partitions"))
      print('spark.driver.maxResultSize', spark.conf.get("spark.driver.maxResultSize"))
      print("spark.executor.memory",  spark.conf.get("spark.executor.memory"))
      print("spark.driver.memory",  spark.conf.get("spark.driver.memory"))
      print("="*100)
      print("Scheme for Arxiv Dataframe:")
      print("="*100)
      arxiv_df.printSchema()

    # determine the columns that we used for machine learning model
    clean_arxiv_df = arxiv_df.select("id", "title", "abstract", "main_category")

    # Concatenate the title and abstract into a single column
    clean_arxiv_df = clean_arxiv_df.withColumn('text', concat_ws(' ', clean_arxiv_df['title'], clean_arxiv_df['abstract']))

    pipeline = Create_ML_pipline()

    # Split the data into training and testing sets
    (trainingData, testData) = clean_arxiv_df.randomSplit([1-test_ratio, test_ratio], seed=24)

    if bprint == 1:
      print("="*100)
      print("Training Data size: ", trainingData.count())
      print("Testing Data size: ", testData.count())
      print("="*100)


    # Fit the model
    ML_model = pipeline.fit(trainingData)

    # Make predictions on the testing data
    predictions = ML_model.transform(testData)
    
    # Print the dataframe with the original main_category and the predicted one
    df_Prediction = predictions.select("id", "main_category", "label", "prediction").show(10)

    # Evaluate the model using the F1 score
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    f1_score = evaluator.evaluate(predictions)
    print("F1 score = %g" % f1_score)

    # save the prediction result into delta table
    #======================================================
    print("="*100)
    print("Save the prediction result into delta table")
    print("="*100)

    # set the path to the Delta table
    delta_table_path = "hdfs:///Dat500_Group09/result"
    
    # set the path to the Delta table
    #delta_table_path = "hdfs:///Dat500_Group09/result/arxiv_result"
    
    # check if the Delta table exists  
    # DeltaTable.isDeltaTable(spark, "spark-warehouse/table1") # True 
    if DeltaTable.isDeltaTable(spark, delta_table_path):
        print("update delta table")
        deltaTable = DeltaTable.forPath(spark, delta_table_path)
        deltaTable.alias("target") \
            .merge(
            df_Prediction.alias("updates"),
            "target.id = updates.id") \
            .whenMatchedUpdate( set = 
            {
            "label": "updates.label",
            "prediction": "updates.prediction"      
            }) \
            .whenNotMatchedInsertAll() \
            .execute()
    else: # file not exists
        print("Create delta table first time")
        df_Prediction.select("id", "title", "main_category", "label", "prediction").write.format("delta").save(delta_table_path)


    spark.stop()





