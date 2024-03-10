import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data=pd.read_csv(r"C:\Users\ramiu\Desktop\MLOPSPROJECTS\ML_TEST\instagram_reach.csv")

# Split the dataset into input features and target variables
x = np.array(data[['username', 'caption', 'hashtags', 'followers', 'time_since_posted']])
y_likes = np.array(data["likes"])
y_time = np.array(data["time_since_posted"])

# Split the data into training and test sets
x_train, x_test, y_train_likes, y_test_likes = train_test_split(x, y_likes, test_size=0.2, random_state=42)
x_train, x_test, y_train_time, y_test_time = train_test_split(x, y_time, test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression

# Train a linear regression model to predict the number of likes
model_likes = LinearRegression()
model_likes.fit(x_train, y_train_likes)

# Train a linear regression model to predict the Time Since posted
model_time = LinearRegression()
model_time.fit(x_train, y_train_time)


# Predict the number of likes and Time Since posted for a new post
new_post = np.array([['new_username', 'new_caption', 'new_hashtags', 'new_followers', 'new_time_since_posted']])
likes_pred = model_likes.predict(new_post)
time_pred = model_time.predict(new_post)

print(f"Predicted number of likes: {likes_pred[0]}")
print(f"Predicted Time Since posted: {time_pred[0]}")

