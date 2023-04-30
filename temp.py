import pandas as pd
from sklearn.linear_model import LinearRegression

# create a sample DataFrame with missing values
df = pd.DataFrame({'A': [1, 2, None, 4, 5],
                   'B': [2, None, 4, 5, 6],
                   'C': [3, 4, 5, None, 7]})

# create a list of columns with missing values
cols_with_missing = df.columns[df.isnull().any()]

# loop through each column with missing values and impute them using regression
for col in cols_with_missing:
    # split the data into a training set and a test set
    train = df[df[col].notnull()]
    test = df[df[col].isnull()]

    # fit a linear regression model to the training data
    model = LinearRegression()
    model.fit(train.drop(col, axis=1), train[col])

    # use the model to predict the missing values in the test data
    imputed_values = model.predict(test.drop(col, axis=1))

    # replace the missing values with the predicted values
    df.loc[df[col].isnull(), col] = imputed_values

# display the result
print(df)
