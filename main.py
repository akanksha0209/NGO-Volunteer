# # import pandas as pd
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.model_selection import train_test_split
# #
# # # Load the data from the CSV file
# # df = pd.read_csv('NGO1.csv')
# #
# # # Split the data into training and testing sets
# # X = df.iloc[:, 2:]
# # y = df.iloc[:, 1]
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# #
# # # Train a random forest classifier on the training data
# # clf = RandomForestClassifier()
# # clf.fit(X_train, y_train)
# #
# # # Evaluate the model on the test data
# # accuracy = clf.score(X_test, y_test)
# # print('Accuracy:', accuracy)
# #
# # # Define a function to get the top N recommendations for a given volunteer
# # def get_recommendations(volunteer_id, N=5):
# #     # Get the skills for the volunteer
# #     volunteer_skills = df.loc[df['volunteer_id'] == volunteer_id, df.columns[2:]]
# #
# #     # Use the random forest model to predict the likelihood of the volunteer being interested in each NGO
# #     scores = clf.predict_proba(volunteer_skills)[:, 1]
# #
# #     # Sort the scores and get the top N NGOs
# #     top_ngos = scores.argsort()[-N:][::-1]
# #
# #     # Get the ngo_ids for the top NGOs
# #     recommended_ngos = df.loc[df.index.isin(top_ngos), 'ngo_id'].unique().tolist()
# #
# #     return recommended_ngos
#
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
#
# # Load the data from the CSV file
# df = pd.read_csv('NGO1.csv')
#
# # Split the data into training and testing sets
# X = df.iloc[:, 2:]
# y = df.iloc[:, 1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# # Train a logistic regression model on the training data
# clf = LogisticRegression()
# clf.fit(X_train, y_train)
#
# # Evaluate the model on the test data
# accuracy = clf.score(X_test, y_test)
# print('Accuracy:', accuracy)
#
# # Define a function to get the top N recommendations for a given volunteer
# def get_recommendations(volunteer_id, N=5):
#     # Get the skills for the volunteer
#     volunteer_skills = df.loc[df['volunteer_id'] == volunteer_id, df.columns[2:8]]
#
#     # Get the skill requirements for each NGO
#     ngo_skill_requirements = df.iloc[:, 8:]
#
#     # Compute the dot product of the volunteer skills and the NGO skill requirements
#     scores = volunteer_skills.dot(ngo_skill_requirements.T)
#
#     # Use the logistic regression model to predict the likelihood of the volunteer being interested in each NGO
#     scores *= clf.predict_proba(volunteer_skills)[:, 1]
#
#     # Sort the scores and get the top N NGOs
#     top_ngos = scores.argsort()[-N:][::-1]
#
#     # Get the ngo_ids for the top NGOs
#     recommended_ngos = df.loc[df.index.isin(top_ngos), 'ngo_id'].unique().tolist()
#
#     return recommended_ngos
#
# get_recommendations(1,5)

# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
# # Load the CSV file into a Pandas DataFrame
# data = pd.read_csv('NGO1.csv')
#
# # Split the data into training and test sets
# X = data.drop(['ngo_id', 'volunteer_id'], axis=1)
# y = data['ngo_id']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Train a random forest classifier on the training data
# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)
#
# # Evaluate the accuracy of the model on the test data
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy}')
#
# # Use the model to predict the NGO for a new volunteer
# new_volunteer = pd.DataFrame([[1, 1, 0, 1, 0, 0, 1, 1]], columns=X.columns)
# predicted_ngo = clf.predict(new_volunteer)
# print(f'Predicted NGO: {predicted_ngo}')

# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
#
# # Load the data
# data = pd.read_csv('NGO1.csv')
#
# # Split the data into features and target
# X = data.iloc[:, 2:9]  # Features - volunteer skills
# y = data.iloc[:, 1]    # Target - NGO ID
#
# # Train a random forest classifier
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X, y)
#
# # Define a function to make predictions for a new volunteer
# def predict_ngo(volunteer_skills):
#     return rf.predict([volunteer_skills])[0]
#
# # Example usage: predict which NGO is the best match for a new volunteer
# new_volunteer_skills = [1, 0, 1, 0, 1, 1, 0]  # Example volunteer skills
# predicted_ngo_id = predict_ngo(new_volunteer_skills)
# print("The best match for the new volunteer is NGO ID:", predicted_ngo_id)


# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
# # Load the NGO1.csv dataset
# data = pd.read_csv('NGO1.csv')
#
# # Split the data into training and testing sets
# X = data.iloc[:, :-1] # Get all columns except the target column
# y = data.iloc[:, -1]  # Get the target column
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Train a random forest classifier
# clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
# clf.fit(X_train, y_train)
#
# # Evaluate the model on the testing set
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Test set accuracy: {accuracy}')
#
# # Use the trained model to predict the NGO for a new volunteer
# new_volunteer = [[3, 1, 1, 1, 0, 1, 0, 0]] # Example volunteer with skills [1, 1, 0, 1, 0, 0, 1]
# predicted_ngo = clf.predict(new_volunteer)
# print(f'Predicted NGO for the new volunteer: {predicted_ngo}')

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the NGO1.csv dataset
data = pd.read_csv('NGO1.csv')

# Split the data into training and testing sets
X = data.iloc[:, :-1] # Get all columns except the target column
y = data.iloc[:, -1]  # Get the target column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test set accuracy: {accuracy}')

# Use the trained model to predict the NGO for a new volunteer
new_volunteer = [4, 6, 0, 1, 1, 1, 0, 1, 4] # Example volunteer with skills [1, 1, 0, 1, 0, 0, 1]
predicted_ngo = clf.predict([new_volunteer])
print(f'Predicted NGO for the new volunteer: {predicted_ngo}')


joblib.dump(clf, 'trained_model.joblib')
