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
Create a new Python script named recommendation.py

 

