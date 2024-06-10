import math
import jsonpickle as json
import TrichRutDacTrung
from Class import Cluster
from collections import Counter
import pandas as pd


def SimilarityCalculation(clusters, features):  # features là list các đặc trưng
    labelCount = []

    # Tìm tâm gần mỗi feature
    for feature in features:
        dmin = 9999999999999999.9
        tmp = 0

        # Duyệt qua từng tâm cụm
        for c in range(len(clusters)):

            # Tính khoảng cách euclid
            d = euclideanDistance(clusters[c].center, feature)

            # Nếu nhỏ hơn dmin thì gán lại giấ trị
            if (d < dmin):
                dmin = d
                tmp = c
        count = []
        lb = []
        for i in range(1):
            count.append(9999999999999999.9)
            lb.append("")
        for f in clusters[tmp].features:
            d = euclideanDistance(feature, f.feature)

            # Thêm điều kiện nếu d > ngưỡng thì loại f
            if (d > 5000):
                continue

            for i in range(1):
                if (count[i] > d):
                    count[i] = d
                    lb[i] = f.link
                    break

        for i in range(1):
            if (lb[i] != ""):
                labelCount.append((lb[i], count[i]))
                # lb nhãn : link đến file âm thanh
                # count: khoảng cách euclid từ 1 feature đến tâm gần nhất

        # Đếm số lần xuất hiện của các link trong labelCount
        # link_counts = Counter(labelCount)
        # Lấy ra 3 link có số lần xuất hiện nhiều nhất
        # top_3_links = [link for link, _ in link_counts.most_common(3)]
     # Tạo DataFrame từ link_counts
    df = pd.DataFrame(labelCount, columns=['f_link', 'count'])

    # Tính trung bình count theo từng f.link
    mean_counts = df.groupby('f_link')['count'].mean()

    # Lấy ra 3 f.link có trung bình count thấp nhất
    top_3_links = mean_counts.nsmallest(3).index.tolist()
    return top_3_links


def euclideanDistance(feature1, feature2):
    d = 0.0
    for i in range(len(feature1)):
        d += (feature1[i] - feature2[i])**2
    d = math.sqrt(d)
    return d


# # Mở file JSON và đọc nội dung của file
# with open('metadata/data.json', 'r') as file:
#     json_data = file.read()

# # Chuyển đổi nội dung file JSON thành đối tượng Python
# clusters = json.loads(json_data)

# features = TrichRutDacTrung.features("data test/Tieng-cu-gay-210s.wav")
# print(SimilarityCalculation(clusters=clusters, features=features))

# data\Tieng-bo-cau-10s.wav            304.331244
# data\Tieng-bo-cau-170s.wav           302.509192
# data\Tieng-bo-cau-20s.wav            384.933522
# data\Tieng-bo-cau-220s.wav           426.454309
# data\Tieng-bo-cau-280s.wav          1218.690611
# data\Tieng-bo-cau-30s.wav            301.958591
# data\Tieng-bo-cau-330s.wav           516.208640
# data\Tieng-chim-quoc-230s.wav        346.014742
# data\Tieng-chim-sao-den-190s.wav     374.708247
# data\Tieng-chim-sao-den-90s.wav      560.940608
# data\Tieng-cu-gay-120s.wav           293.493376
# data\Tieng-cu-gay-50s.wav            543.350789
# data\Tieng-cu-gay-70s.wav            231.478371
# data\Tieng-cu-gay-80s.wav             92.937475
# data\Tieng-cu-gay-90s.wav            197.579892
# data\Tieng-cu-keu-150s.wav           239.001545
