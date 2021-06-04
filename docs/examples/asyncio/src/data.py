from sklearn import datasets


class IrisData(object):
    def __init__(self):
        iris = datasets.load_iris()
        self.X = iris.data  # we only take the first two features.
        self.y = iris.target
