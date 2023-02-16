import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

dataset = pd.read_csv('OurDataset.csv')

z = dataset['email']
y = dataset['class']
z_train, z_test, y_train, y_test = train_test_split(z,y,test_size = 0.2)

vect = CountVectorizer(stop_words='english')
vect.fit(z_train)

z_train_df = vect.transform(z_train)
z_test_df = vect.transform(z_test)

model = MultinomialNB()
model.fit(z_train_df,y_train)

print("Please eneter the text of your email to determine whether it is spam or ham:")
inputstring = str(input("Enter text here:"))

inputarray = (vect.transform([inputstring]))

result = model.predict(inputarray)
if (result):
    print("The email is most likley spam.")
else:
    print("The email is most likley ham.")