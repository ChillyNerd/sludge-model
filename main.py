from model.boosting import Boosting, BoostingType


lightgbm_model = Boosting(BoostingType.lightgbm, verbose=0, n_estimators=100)
xgboost_model = Boosting(BoostingType.xgboost)
catboost_model = Boosting(BoostingType.catboost, silent=True)
