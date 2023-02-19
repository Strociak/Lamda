import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import tkinter as tk
from tkinter import *

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

root = tk.Tk()
root.title("Spam of Ham?")

def stringPredict(*args):
    try:
        value=str(msg.get())
        msgarray = (vect.transform([value]))
        resultbool = model.predict(msgarray)
        if (resultbool):
            result.set("The email is most likley spam.")
        else:
            result.set("The email is most likley ham.")
    except ValueError:
        pass

canvas = tk.Canvas(root, height=500, width=600, bg="#808080")
canvas.pack()

title1 = tk.Label(canvas, text="Spam or Ham?", fg="black", bg="#808080", font=("Arial", 25))
title1.place(relx=0.32, rely=0.05)
title2 = tk.Label(canvas, text="Team Lambda", fg="black", bg="#808080", font=("Arial", 15))
title2.place(relx=0.4, rely=0.15)

frame = tk.Frame(root, bg="#949494")
frame.place(relwidth=0.8, relheight=0.5, relx=0.1, rely=0.25)

msg = StringVar()
submit = tk.Button(frame, text="Submit", padx=10, pady=5, fg="black", bg="#808080", command=stringPredict)
submit.place(relx=0.43, rely=0.5)

emailEntry = tk.Entry(frame, width=7, justify=CENTER, textvariable=msg)
emailEntry.place(relwidth=0.8, relheight=0.1, relx=0.1, rely=0.33)

result = StringVar()
resultLabel = tk.Label(frame, textvariable=result, fg="black", bg="#949494", font=("Arial", 15))
resultLabel.place(relx=0.24, rely=0.8)

instructions = tk.Label(frame, text="Please enter the subject or the text of an email to determine \nwhether it is spam or ham!", fg="black", bg="#949494", font=("Arial", 12))
instructions.place(relx=0.08, rely=0.1)

root.mainloop()