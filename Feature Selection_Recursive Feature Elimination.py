start_time = time.time()
mod = XGBClassifier()      

rfe_smote = RFE(estimator=mod, n_features_to_select=10, step=1)
rfe_smote = rfe_smote.fit(x_bal, y_bal)
print('10 best features selected by rfe:',x_bal.columns[rfe_smote.support_])

x_train_rfe_smote = rfe_smote.transform(x_bal)
x_test_rfe_smote = rfe_smote.transform(x_test)

rfe_nearmiss = RFE(estimator=mod, n_features_to_select=10, step=1)
rfe_nearmiss = rfe_nearmiss.fit(x_miss, y_miss)
print('10 best features selected by rfe:',x_miss.columns[rfe_smote.support_])

x_train_rfe_nearmiss = rfe_nearmiss.transform(x_miss)
x_test_rfe_nearmiss = rfe_nearmiss.transform(x_test)

rfe_over = RFE(estimator=mod, n_features_to_select=10, step=1)
rfe_over = rfef.fit(x_over, y_over)
print('10 best features selected by rfe:',x_over.columns[rfe_smote.support_])

x_train_rfe_over = rfe_smote.transform(x_over)
x_test_rfe_over = rfe_smote.transform(x_test)

elapsed_time = (time.time() - start_time)

print("Time taken for execution: ", elapsed_time)
