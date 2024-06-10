import jsonpickle as json
with open('metadata/data.json', 'r') as file:
    json_data = file.read()

# Chuyển đổi nội dung file JSON thành đối tượng Python

clusters = json.loads(json_data)
with open('output.txt', 'w') as file:
    # Duyệt qua các đối tượng trong clusters
    for x in clusters:
        # Ghi thông tin Center
        file.write(f'Center: {x.__dict__["center"]}\n')

        # Duyệt qua các đối tượng features trong mỗi cluster
        for y in x.features:
            # Ghi thông tin Link
            file.write(f'Link: {y.link}\n')
            # Ghi thông tin Feature
            file.write(f'Feature: {y.feature}\n')
# for x in clusters:
#     print('Center', x.__dict__['center'])
#     for y in x.features:
#         print('Link', y.link)
#         print('Feature', y.feature)
