class OutlierDetector:
    """Base class for univariate outlier detection"""
    def __init__(self, log_transform=False, *args, **kwargs):
        self.log_transform=log_transform

    def fit(self, X):
        return self

    def transform(self, X):
        return X.copy()

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class ZScore(OutlierDetector):
    pass

class ModifiedZScore(OutlierDetector):
    pass

class IQR(OutlierDetector):
    pass

class RHO(OutlierDetector):
    pass

class Residuals(OutlierDetector):
    pass

