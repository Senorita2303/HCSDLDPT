class Cluster:
    def __init__(self) -> None:
        pass

    def __init__(self, center, features):
        self.center = center
        self.features = features


class Feature:
    def __init__(self) -> None:
        pass

    def __init__(self, link, feature):
        self.link = link
        self.feature = feature
