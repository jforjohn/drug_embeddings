import sklearn_crfsuite

class CRFClassifier(object):
    def __init__(self, *args, **kwargs):
        self.classifier = sklearn_crfsuite.CRF(
            *args, **kwargs,
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self
    
    def predict(self, X):
        return self.classifier.predict(X)