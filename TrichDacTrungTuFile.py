import TrichRutDacTrung as ft
import os
from Class import Feature
import LuuTruDacTrung as luu

# Đường dẫn đến thư mục chứa các file
folder_path = 'data'

# Lấy tất cả các file trong thư mục
files = os.listdir(folder_path)

listFeatures = []

for file in files:
    features = ft.features(os.path.join(folder_path, file))
    for feature in features:
        feature = Feature(link=os.path.join(folder_path, file), feature=feature)
        listFeatures.append(feature)


Clusters = luu.ClusterUseKmeans(listFeatures)
# Luu vào file json
luu.save(Clusters)