import joblib
import sklearn


class Rf_Model:

    def predict(lst):
        rf_model = joblib.load("src/models/random_forest_engagement.joblib")
        prd = rf_model.predict(lst)
        return prd



model = Rf_Model()


def rf_get_model():
    return model