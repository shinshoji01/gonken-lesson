# Homework

Answer the following three questions.

1. What is the difference between OpenCV and Pillow? Please answer from the viewpoint of usage in Japanese or English. **(difficulty : ☆)**
2. There is face.jpg in current folder. Create a python program that looks for human faces and automatically mosaics them. Then mosaic the man's face in this face.jpg. **(difficulty : ☆☆☆)**
3. Create a machine learning program to classify iris varieties from the length and width of sepals and petals. Fill in the ?????? below to complete the program. One sentence corresponds to one line. iris.csv is in the current folder.  **(difficulty : ☆☆)**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Reading iris data
iris_data = pd.read_csv("iris.csv", encoding="utf-8")

# Separate iris data into label and input data
y = iris_data.loc[:,"Name"]
x = iris_data.loc[:,["SepalLength","SepalWidth","PetalLength","PetalWidth"]]

# Separate for learning (80%) and test (20%) (shuffle = True)
# (1) x_train, X_test, y_train, y_test = ??????

# Learning
clf = SVC()
# (2) clf.fit(??????)

# Evaluation and displaying the accuracy rate
# (3) y_pred = ??????
# (4) print(??????)
```
