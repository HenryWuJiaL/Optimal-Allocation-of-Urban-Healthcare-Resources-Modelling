### SMOTE
sfm_smote = SelectFromModel(XGBClassifier(), threshold=0.01) 
sfm_smote = sfm_smote.fit(x_bal,y_bal)

x_train_sf_smote = sfm_smote.transform(x_bal) 
x_test_sf_smote = sfm_smote.transform(x_test)

shape_smote = np.shape(x_train_sf_smote)

print("Shape of the dataset using SMOTE",shape_smote)

### Over Sampling

sfm_over = SelectFromModel(XGBClassifier(), threshold=0.01) 
sfm_over = sfm_over.fit(x_over,y_over)

x_train_sf_over = sfm_over.transform(x_over) 
x_test_sf_over = sfm_over.transform(x_test)

shape_over = np.shape(x_train_sf_over)

print("Shape of the dataset using Over Sampling",shape_over)

### NearMiss

sfm_nearmiss = SelectFromModel(XGBClassifier(), threshold=0.01) 
sfm_nearmiss = sfm_nearmiss.fit(x_miss,y_miss)

x_train_sf_nearmiss = sfm_nearmiss.transform(x_miss) 
x_test_sf_nearmiss = sfm_nearmiss.transform(x_test)

shape_nearmiss = np.shape(x_train_sf_nearmiss)


print("Shape of the dataset using NearMiss",shape_nearmiss)
