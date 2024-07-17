Movie Recommendation with MLlib - Collaborative Filtering

(implementaiton 2)
Step 1: Convert MoveLens' data (UserID, MovieID, rating, Timestamp) into the format of (UserID, MovieID, rating)

# Load the data file and convert it to the required format

input_file_path = '/mnt/data/New data.txt'
output_file_path = '/mnt/data/formatted_data.txt'

# Read the input file
with open(input_file_path, 'r') as file:
    data = file.readlines()

# Process the data
formatted_data = []
for line in data:
    parts = line.split()
    if len(parts) == 4:
        formatted_data.append(f"{parts[0]},{parts[1]},{parts[2]}\n")

# Write the formatted data to a new file
with open(output_file_path, 'w') as file:
    file.writelines(formatted_data)

output_file_path
















Step 2 Implement this version of MLlib - Collaborative Filtering Examples

Using the GCP Console to Upload the File:
Navigate to Cloud Storage, create a new bucket.
Click the "Upload Files" button and select C:\Users\KANG\Downloads\formatted_data.txt to upload.

 
Create a new Python script named recommendation.py:
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

# Initialize Spark session
spark = SparkSession.builder.appName("MovieRecommendation").getOrCreate()
sc = spark.sparkContext

# Load and parse the data
data = sc.textFile("gs://movie_bucket_kang_1/formatted_data.txt")
ratings = data.map(lambda l: l.split(','))\
              .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# Train the recommendation model
rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)

# Predict user ratings for movies
testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))

# Calculate Mean Squared Error
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
print("Mean Squared Error = " + str(MSE))

# Save the model
model.save(sc, "gs://movie_bucket_kang_1/model/myCollaborativeFilter")
sameModel = MatrixFactorizationModel.load(sc, "gs://movie_bucket_kang_1/model/myCollaborativeFilter")

spark.stop()
 
Submit the PySpark Job
In Cloud Shell, submit the PySpark job to your Dataproc cluster using the following command:

gcloud dataproc jobs submit pyspark recommendation.py --cluster=cluster-a9c6 --region=us-centr
 
 
 

You can see the results by open predictions.

(implementaiton 3)


Using the GCP Console to Upload the File:
Navigate to Cloud Storage, create a new bucket.
Click the "Upload Files" button and select C:\Users\KANG\Downloads\formatted_data.txt to upload.
 

Create a new Python script named recommendation.py:
Import Libraries
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS, Rating
•	SparkSession: Entry point to programming Spark with the DataFrame and SQL API.
•	ALS: Alternating Least Squares, a collaborative filtering algorithm for recommender systems.
•	Rating: Class used to store user, product, and rating information.
Initialize Spark Session
spark = SparkSession.builder.appName("MovieRecommendation").getOrCreate()
sc = spark.sparkContext
•	Initializes a Spark session named "MovieRecommendation".
•	sc is the SparkContext object, which is the entry point for using Spark functionality.
Load and Parse the Data
data = sc.textFile("gs://movie_bucket_kang_1/formatted_data.txt")
ratings = data.map(lambda l: l.split(',')) \
              .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
•	Loads the data from Google Cloud Storage (GCS) as a text file.
•	Splits each line of the file by commas and maps it to a Rating object containing user, product, and rating.
Train the Recommendation Model Using ALS
rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)
•	rank: Number of latent factors in the model.
•	numIterations: Number of iterations to run the ALS algorithm.
•	Trains the ALS model using the provided ratings data.
Generate Predictions
users_products = ratings.map(lambda r: (r.user, r.product))
predictions = model.predictAll(users_products).map(lambda r: (r.user, r.product, r.rating))
•	Creates a list of (user, product) pairs from the ratings data.
•	Uses the trained model to predict ratings for all user-product pairs.
•	Maps the predictions to include user, product, and predicted rating.
Convert Predictions to DataFrame
predictions_df = predictions.toDF(["user", "product", "rating"])
•	Converts the predictions RDD to a DataFrame with columns "user", "product", and "rating".
Save Predictions as a Single CSV File to GCS
predictions_df.coalesce(1).write.mode("overwrite").option("header", "true").csv("gs://movie_bucket_kang_1/predictions/predictions.csv")
•	coalesce(1): Ensures the DataFrame is written to a single CSV file instead of multiple part files.
•	mode("overwrite"): If the file already exists, it will be overwritten.
•	option("header", "true"): Writes the DataFrame with a header row.
•	csv("gs://movie_bucket_kang_1/predictions/predictions.csv"): Specifies the GCS path where the CSV file will be saved.

 

