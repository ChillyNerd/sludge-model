import os

from data.data_prepare import prepare_dataset, Dataset, Target
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from model import *


def select_hyperparameters(model: BaseModel, parameters: dict):
    vector = {}
    tries = 1
    for parameter in parameters.keys():
        vector[parameter] = None
        parameter_values = parameters[parameter] if parameters[parameter] is not None else []
        parameter_count = len(parameter_values) if len(parameter_values) > 0 else 1
        tries = tries * parameter_count


def train_predict(dataset: Dataset, target: Target):
    data_dir = os.path.join(os.getcwd(), 'data/raw')
    datasets = prepare_dataset(os.path.join(data_dir, 'correct_data.csv'), dataset, target)
    lr = Pipeline([
        ('scaler_', StandardScaler()),
        ('model_', LogReg(max_iter=500))
    ])

    knn = Pipeline([
        ('scaler_', StandardScaler()),
        ('model_', KNeighbour(n_neighbors=5))
    ])

    svc = Pipeline([
        ('scaler_', StandardScaler()),
        ('model_', SupportVector(probability=True))
    ])

    rf = Pipeline([
        ('model_', RandomForest())
    ])

    lgbm = Pipeline([
        ('model_', LightGBM(verbose=0, n_estimators=100))
    ])

    xgbm = Pipeline([
        # ('encoder_', LabelEncoderWrapper()),
        ('model_', XGBoost())
    ])

    cbm = Pipeline([
        ('model_', CatBoost(silent=True))
    ])
    models = {
        'KNN': knn,
        'Logistic Regression': lr,
        'Support Vector': svc,
        'Random Forest': rf,
        'LightGBM': lgbm,
        'CatBoost': cbm
    }

    use_roc = False
    for label, dataset in datasets.items():
        X_train, X_test, y_train, y_test = dataset
        print(f'--------------- {label} --------------------')
        for title, model in models.items():
            print(f'~~~~~~ MODEL {title} ~~~~~~~~~~')
            model.fit(X_train, y_train)
            if use_roc:
                y_proba = model.predict_proba(X_test)
                labels = np.argmax(y_proba, axis=1)
                classes = model.classes_
                y_pred = [classes[i] for i in labels]
                print('ROC AUC score', roc_auc_score(y_test, y_proba, multi_class="ovr", average="micro"))
            else:
                y_pred = model.predict(X_test)
            cr = classification_report(y_test, y_pred, output_dict=True)
            print('accuracy', cr['accuracy'])
            print('macro f score', cr['macro avg']['f1-score'])
            print('weighted f score', cr['weighted avg']['f1-score'])


for dataset_type in [Dataset.RAW, Dataset.RAW_VELOCITY, Dataset.VELOCITY]:
    for target_type in [Target.SLUDGE, Target.DIRECTION]:
        print(f'~Dataset is {dataset_type}, Target is {target_type}~')
        train_predict(dataset_type, target_type)
