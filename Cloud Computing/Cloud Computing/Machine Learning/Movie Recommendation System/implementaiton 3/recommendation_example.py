from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS, Rating

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

# Convert predictions to DataFrame
predictions_df = predictions.toDF(["user", "product", "rating"])

# Show top N predictions
top_n_predictions = predictions_df.orderBy(predictions_df['rating'].desc()).take(20)

# Print the top N predictions
print("Top 20 Predictions:")
for prediction in top_n_predictions:
    print(f"User: {prediction['user']}, Product: {prediction['product']}, Rating: {prediction['rating']}")

# Optionally, save predictions as a single CSV file to GCS
predictions_df.coalesce(1).write.mode("overwrite").option("header", "true").csv("gs://movie_bucket_kang_1/predictions/predictions.csv")

spark.stop()
