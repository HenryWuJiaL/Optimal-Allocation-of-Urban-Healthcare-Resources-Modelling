x = data_features
y = data['readmitted']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 101)

model_eval(x_train, x_test, y_train, y_test, XGBClassifier)
model_eval(x_train, x_test, y_train, y_test, RandomForestClassifier)
model_eval(x_train, x_test, y_train, y_test, GradientBoostingClassifier)
model_eval(x_train, x_test, y_train, y_test, LogisticRegression)
model_eval(x_train, x_test, y_train, y_test, DecisionTreeClassifier)

