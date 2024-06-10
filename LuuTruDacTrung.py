from sklearn.cluster import KMeans
from Class import Cluster
from Class import Feature
import jsonpickle as json


def kmeans(data):
    # Khởi tạo mô hình K-means với 11 cụm
    kmeans = KMeans(n_clusters=11, random_state=0, n_init=10)
    # Huấn luyện mô hình trên dữ liệu
    kmeans.fit(data)
    # Dự đoán nhãn cụm cho các điểm dữ liệu
    labels = kmeans.predict(data)
    # Lấy tâm của các cụm
    centers = kmeans.cluster_centers_
    return labels, centers


def ClusterUseKmeans(features):
    dataFeatures = []
    for feature in features:
        dataFeatures.append(feature.feature)
    labels, centers = kmeans(dataFeatures)
    Clusters = []
    for i in range(len(centers)):
        listFeature = []
        for j in range(len(labels)):
            if labels[j] == i:
                listFeature.append(features[j])
        cluster = Cluster(center=centers[i], features=listFeature)
        Clusters.append(cluster)
    return Clusters


def save(data):
    data_json = json.dumps(data)
    with open("metadata/data1.json", "w") as file:
        file.write(data_json)
