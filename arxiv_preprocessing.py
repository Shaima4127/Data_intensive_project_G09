#======================================================================================
# Arxiv_preprocessing.py
#======================================================================================
'''
This file used to prepare  arxiv data for anlayizing and classification by the following steps:
- Read configurations that will be passed by user when running spark-submit like file name
- Read the CSV file that was generated in hadoop mapreduce and convert it to spark dataframe (arxiv_df)
- Read delta table for categories that we got it from scraping the arxiv website for the detail categories (category_df)
- Join between arxiv dataframe with category_df where we take the first category from arxiv_df match the sub_category in category_df 
- Select the columns that we need for analyizing & classification into arxiv_df
- Add new column article_date to arxiv_df which will be extracted from article_id by follwoing the format in Arxiv website
- Clean update_date column ans save the result into clean_arxiv_df
- Add new columns for authers_num to clean_arxiv_df
- Extracting the number of authors:
- The number of authors for each article is extracted from the "authors" column in the Arxiv dataset, 
  where the names of all authors are listed and separated by commas. Some articles may use "and" to separate the authors as well.
- Determine the required columns that we will used for analyizing & classification and saved it into delta table "delta_preprocessing_file"
- Split the cleaned dataframe into training & testing part and save these splitted dataframe into delta tables
- Determine three parts for testing data and saved it into three different delta table file that will be used in classifying_arxiv.py

'''
import time
import sys
import pyspark
from pyspark.sql import SparkSession 
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import concat_ws
from delta import DeltaTable, configure_spark_with_delta_pip

   
if __name__ == '__main__':
    
    start_time = time.time()
    # ===================================================================
    # Path for all files in preprocessing
    #==================================================================== 
    # input file  
    csv_file = sys.argv[1]   
    #csv_file = "hdfs:///Dat500_Group09/output_meta/part*" 
    #partition = int(sys.argv[2])  # number of partition to repartition the dataframe
    # delta table contains all main_categories and sub categories   
    delta_category_file = "hdfs:///Dat500_Group09/spark_result/category"

    # output files
    delta_preprocessing_file = "hdfs:///Dat500_Group09/spark_result/preprocessing"
    delta_training_file = "hdfs:///Dat500_Group09/spark_result/training"
    delta_testing_file = "hdfs:///Dat500_Group09/spark_result/testing"

    #==================================================================== 
    # Set the configuration properties for Delta tables
    #====================================================================     
    spark = (SparkSession
      .builder      
      .master('yarn')
      .appName("arxiv_preprocessing")
      #.config('spark.executor.instances', '8')
      .config('spark.executor.instances', '12')
      .config("spark.executor.memory", "1g")  
      #.config("spark.executor.memory", "2g")  
      #.config("spark.executor.cores", 2)  
      #.config('spark.driver.memory', "4g") 
      #.config('spark.driver.memory', '2g')  # if we did not determine the drive it will take from the config file around 7gb
      .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
      .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") 
      .config("spark.databricks.delta.properties.defaults.autoOptimize.optimizeWrite", "true")
      .config("spark.databricks.delta.properties.defaults.autoOptimize.autoCompact", "true")      
      .getOrCreate())  
    
      
    # Create data schema
    #==================================================================== 
    dbschema = StructType([
    StructField("id", StringType(), True),
    StructField("authors", StringType(), True),
    StructField("title", StringType(), True),
    StructField("abstract", StringType(), True),
    StructField("journal_ref", StringType(), True),
    StructField("category", StringType(), True),
    StructField("update_date", StringType(), True),
    ])

    # 1- Import csv file for the data
    #==================================================================== 
    try:
      arxiv_df =spark.read.options(delimiter="::", header=False, schema=dbschema).csv(csv_file)    
    except:
        print(f"Error: Could not read the data for this file {csv_file}")
        sys.exit(1)

    # Change the column names to the same name for the Arxiv metadata 
    #====================================================================
    arxiv_df = arxiv_df.selectExpr("_c0 as id", "_c1 as authors", "_c2 as title", "_c3 as abstract", 
                                "_c4 as journal_ref", "_c5 as category", "_c6 as update_date")
    
    print("Add new column for arxiv data to recategorize the articles")
    '''
      # Add new column "main_category" for labeling the arxiv data to recategorize the articles
        - The main_category column will be used in analysing and classifying the Arxiv data using machine learning models.
        - Arxiv data contains a category column which combine one or more sub category and starting with the main category 
        - for the scientic articles, regarding to the arxiv website they shows main_category for each sub category-
        - we add a new column which contains the main category for each article.
        - we scrap all the main categories & sub categories from arxiv url and save them into delta table
        - read the category delta table and join them with the arxiv data where the subcategory in the first table matche the the first category from the arxiv table
        - select all the requied columns from the joined data 
    '''
    #====================================================================
    #delta_category_file   
    category_df = spark.read.format("delta").load(delta_category_file)
    #category_df.count()
    
    # Take the first category from the category column 
    arxiv_df = arxiv_df.withColumn("category1", split(col("category"), " ").getItem(0))
    
    # join the two data frames where "category1" in arxiv_df equal to "category" in category_df
    #arxiv_joined_df = arxiv_df.join(category_df, arxiv_df.category1 == category_df.sub_category, "inner")
    # Optimization - use broadcast (1min---9sec)
    #arxiv_joined_df = arxiv_df.join(broadcast(category_df), arxiv_df.category1 == category_df.sub_category, "inner")
    arxiv_joined_df = arxiv_df.join(category_df, arxiv_df.category1 == category_df.sub_category, "inner")

    # select the required columns from the joined data frame
    arxiv_df = arxiv_joined_df.select("id", "authors", "title", "abstract", "category", "update_date", "main_category", col("description").alias("main_topic"))
    # old method
    '''arxiv_df = arxiv_df.withColumn("main_category",
                   when(split(col("category"), "\\.")[0] == "cs", "computer science")
                   .when(split(col("category"), "\\.")[0] == "math", "mathematics")
                   .when(split(col("category"), "\\.")[0] == "econ", "economics")
                   .when(split(col("category"), "\\.")[0] == "eess", "electrical engineering and systems science")
                   .when(split(col("category"), "\\.")[0] == "q-bio", "quantitative biology")
                   .when(split(col("category"), "\\.")[0] == "q-fin", "quantitative finance")
                   .when(split(col("category"), "\\.")[0] == "stat", "statistics")                   
                   .otherwise("physics"))
    '''
    #arxiv_df.show(3)       

    print("Extract date from article IDs")      
    # Extract date from article IDs  
    #====================================================================    
    arxiv_df = arxiv_df.withColumn("date_string",    
            when( arxiv_df["id"].contains("/") & (substring(split(arxiv_df["id"], "/")[1], 1, 2) >= "90"), 
                    concat(lit("19"), substring(split(arxiv_df["id"], "/")[1], 1, 2), substring(split(arxiv_df["id"], "/")[1], 3, 2))
                )
            .when( arxiv_df["id"].contains("/") & (substring(split(arxiv_df["id"], "/")[1], 1, 2) < "90"),
                    concat(lit("20"), substring(split(arxiv_df["id"], "/")[1], 1, 2), substring(split(arxiv_df["id"], "/")[1], 3, 2))                
                    # extract date for the second format of article IDs (YYMM.123)         
                ).otherwise( concat(lit("20"), substring("id", 1, 2), substring("id", 3, 2)))
    ).withColumn("article_date", to_date("date_string", "yyyyMM"))

    # Cleaning, the date contains extra character \t at the end we remove this char
    arxiv_df = arxiv_df.withColumn('update_date', regexp_replace(arxiv_df['update_date'], "\t", ""))

    clean_arxiv_df = arxiv_df.drop("date_string")

    clean_arxiv_df.show(3)

    print("Add authors no column to the dataframe") 
    # Add authors no column to the dataframe
    #====================================================================
    clean_arxiv_df = clean_arxiv_df.withColumn('splitted_authors', regexp_replace('authors', ' and ', ','))\
                    .withColumn('authers_num', size(split('splitted_authors', ',')))  
    
    clean_arxiv_df = clean_arxiv_df.drop("splitted_authors")  
        

    # Concatenate the title and abstract into a single column
    clean_arxiv_df = clean_arxiv_df.withColumn('text', concat_ws(' ', clean_arxiv_df['title'], clean_arxiv_df['abstract']))

    clean_arxiv_df.show(1)

    # determine the columns that we used for machine learning model
    clean_arxiv_df = clean_arxiv_df.select("id", "text","title", "abstract", "authers_num", "article_date", "main_category", "main_topic")
    
    #print("Total record in Data: ",clean_arxiv_df.count())

    # save the cleaned data into delta table
    #=====================================================================
    #delta_preprocessing_file 

    # Repartition the DataFrame
    #num_partitions = 30
    #delta_df = clean_arxiv_df.repartition(num_partitions)       
    print("saving into delta table") 

    # DeltaTable
    if not DeltaTable.isDeltaTable(spark, delta_preprocessing_file):
      clean_arxiv_df.write.format("delta").save(delta_preprocessing_file)
      
    
    # splitting the data into training and testing part
    #=====================================================================
    training_df, _ = clean_arxiv_df.randomSplit([0.8, 0.2], seed=24)
    #print("Training Data size: ", training_df.count())

    # save training data
    #=====================================================================
    if not DeltaTable.isDeltaTable(spark, delta_training_file):
      print("save training data")      
      training_df.write.format("delta").save(delta_training_file)

    # save testing data
    #=====================================================================
    # split the cleaned dataframe 3 times with different ratio (0.5, 0.6, 0.7) to get 3 different parts for test data(test1: 0.5, test2: 0.6, test3: 0.7)

    # Saving first test file with rate = 0.5    
    delta_test1 = delta_testing_file + '/test1'
    if not DeltaTable.isDeltaTable(spark, delta_test1):
      print("save testing file 1")  
      _, test_df = clean_arxiv_df.randomSplit([0.5, 0.5], seed=24)        
      #print("Testing Data size: ", test_df.count())
      print("="*100)          
      test_df.write.format("delta").save(delta_test1)

    # second test file
    delta_test2 = delta_testing_file + '/test2'
    if not DeltaTable.isDeltaTable(spark, delta_test2):
      print("save testing file 2")  
      _, test_df = clean_arxiv_df.randomSplit([0.4, 0.6], seed=24)        
      #print("Testing Data 2 size: ", test_df.count())
      print("="*100)        
      test_df.write.format("delta").save(delta_test2)    
      
    
    # Third test file
    delta_test3 = delta_testing_file + '/test3'
    if not DeltaTable.isDeltaTable(spark, delta_test3):
      print("save testing file 3")  
      _, test_df = clean_arxiv_df.randomSplit([0.3, 0.7], seed=24)        
      #print("Testing Data 3 size: ", test_df.count())
      print("="*100)          
      test_df.write.format("delta").save(delta_test3)  
    
    # stop the timer
    end_time = time.time()
    dif_time = end_time - start_time 
    print(f"Total time taken: {dif_time:.2f} seconds")


    
    spark.stop()





