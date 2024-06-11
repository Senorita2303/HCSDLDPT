import jsonpickle
import json
with open('metadata/data.json', 'r') as file:
    json_data = file.read()

# Chuyển đổi nội dung file JSON thành đối tượng Python
Features = []
Clusters = []
clusters = jsonpickle.loads(json_data)
with open('metadata/data1.json', 'w') as file:
    # Duyệt qua các đối tượng trong clusters
    for x in clusters:
        # Ghi thông tin Center
        # Duyệt qua các đối tượng features trong mỗi cluster
        for y in x.features:
            Features.append({
                "link": y.link,
                "feature": y.feature
            })
        Clusters.append({
            "center": x.__dict__["center"].tolist(),
            "features": Features
        })
    json.dump(Clusters, file, indent=4)
    # print(data_json)
