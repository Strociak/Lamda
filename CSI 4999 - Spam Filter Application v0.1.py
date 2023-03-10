import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

spam = pd.read_csv('spam_ham_dataset.csv')

z = spam['text']
y = spam["label"]
z_train, z_test, y_train, y_test = train_test_split(z,y,test_size = 0.2)

cv = CountVectorizer()
features = cv.fit_transform(z_train)

model = svm.SVC()
model.fit(features,y_train)

features_test = cv.transform(z_test)
print("Accuracy: {}".format(model.score(features_test,y_test)))

print("Please eneter the text of your email to determine whether it is spam or ham:")
inputstring = str(input("Enter text here:"))

inputarray = (cv.transform([inputstring]))

result = model.predict(inputarray)
if (result == 'spam'):
    print("The email is most likley spam.")
elif (result == 'ham'):
    print("The email is most likley ham.")
