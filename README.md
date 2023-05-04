# Data_intensive_project_G09
Project in data intensive course

#============================
# Packages
#============================
# hadoop
hadoop-3.3.3.tar.gz
# spark
spark-3.3.2-bin-hadoop3.tgz
# delta table
packages io.delta:delta-core_2.12:2.0.0 

#======================================================
# Project Files tree
#======================================================
- Dat500_Group09
    - baseline:
        # contains 5 files, we worked in jupyter note book files first then preapre the arixv_classifying_baseline.py 
        - analyize_arxiv_data_baseline
        - classifying_arxiv_data_baseline1
        - classifying_arxiv_data_baseline2
        - preprocessing_arxiv_baseline
        - arxiv_classifying_basline.py   

    - map_reduce:
        # converting a JSON data file into a CSV format 
        - arxiv_convert_to_csv.py

    # we have three files in jupyter notebook
    - scrap_categories   
    - arxiv_data_analyizing
    - arxiv_data_classifying

    # Two main files used in spark-submit
    - arxiv_preprocessing.py
    - arxiv_classifying.py

#======================================================
# How to Run Files in our project
#======================================================
Firstly, execute the map_reduce file to convert the json file to csv format and store the output in the (/output_meta/*parts) directory.

# Copy this command and run it in the terminal.
# Note: You should delete the output "hdfs:///Dat500_Group09/output_meta" before running the command to delelte the csv file write this command "hadoop fs -rm -r /Dat500_Group09/output*"
# ================================================
- python3 ./map_reduce/arxiv_convert_to_csv.py --hadoop-streaming-jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.3.jar -r hadoop hdfs:///Dat500_Group09/input/arxiv-metadata.json --output-dir hdfs:///Dat500_Group09/output_meta --no-output
# ================================================

Next, execute the scrap_categories.ipynb file to extract data from the arxiv URL and create a delta table that contains all the main and sub categories. 

After that, run the arxiv_preprocessing.py file, which will perform all necessary preprocessing, transformations, cleaning the data, and generate a clean dataframe. The resulting dataframe will be stored in the delta table located at "hdfs:///Dat500_Group09/spark_result/preprocessing", and will split the preprocessed dataframe into training & testing parts then will saved all these parts into delta tables.
# Note: running this command first time will generate delta tables but if you run it next time it will repeat the preprocessing without generating or updating the delta table that is aleardy generated, (the running time will reduce second time)
# copy this command and run it in the terminal 
# ================================================
/usr/local/spark/bin/spark-submit  --packages io.delta:delta-core_2.12:2.0.0 Dat500_Group09/arxiv_preprocessing.py hdfs:///Dat500_Group09/output_meta/part*
# ================================================

Then, execute the arxiv_data_analyizing.ipynb file to analyze the data and create visualizations

Finally, run the arxiv_classifying.py file to apply a machine learning model and perform the classification task then save the prediction result into a delta table  "hdfs:///Dat500_Group09/spark_result/final_result/arxiv_meta.
# copy this command and run it in the terminal
# Note:  before you run command you should delete the previous final_result or it will be merge the new result with the existing delta table
# to delete the previous result you can write this command
#  hadoop fs -rm -r /Dat500_Group09/spark_result/final_result/arxiv_meta
# ================================================
# first run will generate test1 delta table (50% part of the data)
/usr/local/spark/bin/spark-submit  --packages io.delta:delta-core_2.12:2.0.0 Dat500_Group09/arxiv_classifying.py test1
# second run will generate test2 delta table (60% part of the data)
/usr/local/spark/bin/spark-submit  --packages io.delta:delta-core_2.12:2.0.0 Dat500_Group09/arxiv_classifying.py test2
# third run generate test3 delta table (60% part of the data)
/usr/local/spark/bin/spark-submit  --packages io.delta:delta-core_2.12:2.0.0 Dat500_Group09/arxiv_classifying.py test3
# ================================================


#======================================================
# Our project contains the following files located in HDFS in a Dat500_group09 directory.
# Some files are the data file like json other are the csv file that we generated in mapreduce
# finally the delta files that we generated in our project.
#======================================================

- hdfs:///Dat500_group09
    - /input
        sample_metadata_m.json
        arxiv-metadata.json
        category.csv

    # Results from map-reduce in mrjob    
    - /output_meta/*parts   # 27 files
    - /output_sample/*parts 

    #=============================================== 
    # delta tables that we generated in our project
    #===============================================
    - /spark_result   
        # extracting main_category & sub category from arxiv url and save it as a delta table in this path
        - /category
        # save the predictions results that we got from classifying the data by machine learning model, 
        - /final_result
            - /arxiv_meta     
            - /arxiv_sample

        # save the data after cleaning & transformation into this path
        - /preprocessing
        
        # training part (80%)
        - /training

        # testing parts (we tried more than rate for testing)   
        - /testing
            - /test1(first testing part 50% of the data)
            - /test2(60% of the data)
            - /test3(70% of the data)

        



