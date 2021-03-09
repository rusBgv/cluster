from sklearn import datasets
from sklearn.cluster import KMeans

iris_df = datasets.load_iris()

model = KMeans(n_clusters=3)

model.fit(iris_df.data)

all_predictions = model.predict(iris_df.data)

print(all_predictions)
