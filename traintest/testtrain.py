from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def tratest(df):

    y= df["Potability"]
    X=df.drop("Potability",axis=1)


    #modelling

    classifiers = [('KNN', KNeighborsClassifier()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier()),
                   ('LightGBM', LGBMClassifier()),
                   ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=["roc_auc","accuracy"])
        print(f"AUC: {round(cv_results['test_roc_auc'].mean(),4)} ({name}) ")


    knn_params = {"n_neighbors": range(2, 50)}

    cart_params = {'max_depth': range(1, 20),
                   "min_samples_split": range(2, 30)}

    rf_params = {"max_depth": [5, 8, 15, None],
                 "max_features": [5, 7, "auto"],
                 "min_samples_split": [8, 15, 20],
                 "n_estimators": [200, 500, 1000]}

    xgboost_params = {"learning_rate": [0.1, 0.01],
                      "max_depth": [5, 8, 12, 20],
                      "n_estimators": [100, 200],
                      "colsample_bytree": [0.5, 0.8, 1]}

    lightgbm_params = {"learning_rate": [0.01, 0.1],
                       "n_estimators": [300, 500, 1500],
                       "colsample_bytree": [0.5, 0.7, 1]}

    catboost_params = {"iterations": [200, 500,1000],
                       "learning_rate": [0.01, 0.1],
                       "depth": [3, 6,9]}

    adaboost_params = {"n_estimators":[50,100,200,400],
                       "learning_rate":[1.0,0.5,0.1]}


    classifiers = [('KNN', KNeighborsClassifier(), knn_params),
                   ("CART", DecisionTreeClassifier(), cart_params),
                   ("RF", RandomForestClassifier(), rf_params),
                   ('XGBoost', XGBClassifier(), xgboost_params),
                   ('LightGBM', LGBMClassifier(), lightgbm_params),
                   ("Adaboost",AdaBoostClassifier(),adaboost_params),
                   ("catboost",CatBoostClassifier(),catboost_params)]

    best_models = {}

    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=10, scoring=["roc_auc"])
        print(f"AUC (Before): {round(cv_results['test_roc_auc'].mean(),4)}")


        gs_best = GridSearchCV(classifier, params, cv=10, n_jobs=-1, verbose=True).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=10, scoring=["roc_auc"])
        print(f"AUC (After): {round(cv_results['test_roc_auc'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

        best_models[name] = final_model


    voting_clf = VotingClassifier(
        estimators=[("KNN",best_models["KNN"]),
                    ("CART",best_models["CART"]),
                    ("RF",best_models["RF"]),
                    ("XGBoost",best_models["XGBoost"]),
                    ("LightGBM",best_models["LightGBM"]),
                    ("Adaboost",best_models["Adaboost"]),
                    ("catboost",best_models["catboost"])],
        voting='soft')

    voting_clf.fit(X,y)

    cvNewResult = cross_validate(voting_clf,X,y,cv=10,scoring=["accuracy", "f1", "roc_auc"])

    return cvNewResult