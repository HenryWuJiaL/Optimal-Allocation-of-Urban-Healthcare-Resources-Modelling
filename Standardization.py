scaler = StandardScaler()
scaler.fit(x)
scaled_features = scaler.transform(x)

data_features = pd.DataFrame(scaled_features, columns = x.columns)

data_features.head()

