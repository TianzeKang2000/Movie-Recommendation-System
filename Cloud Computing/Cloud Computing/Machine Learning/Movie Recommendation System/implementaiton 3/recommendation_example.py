from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

# Initialize Spark session
spark = SparkSession.builder.appName("MovieRecommendation").getOrCreate()
sc = spark.sparkContext

# Load and parse the data
data = sc.textFile("gs://movie_bucket_kang_1/formatted_data.txt")
ratings = data.map(lambda l: l.split(',')) \
              .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# Train the recommendation model using ALS
rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)

# Generate predictions
users_products = ratings.map(lambda r: (r.user, r.product))
predictions = model.predictAll(users_products).map(lambda r: (r.user, r.product, r.rating))

# Collect and print predictions
results = predictions.collect()
for result in results:
    print(f"User: {result[0]}, Product: {result[1]}, Rating: {result[2]}")

# Save predictions to a text file in GCS
predictions.saveAsTextFile("gs://movie_bucket_kang_1/predictions")

spark.stop()
