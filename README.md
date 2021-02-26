# Diabetes Prediction

This Project check whether or not a person is having Diabetes on the basis of following inputs. The Model is trained on the features such as **No of Time Pregnant**, **BMI**, **Skin Fold Thickness**, **Age**, etc and according to that data the model will predict is the person affected with Diabetes or not.

## To run this Project please follow these steps

1. Open _command prompt_ or _powershell window_.
2. Type this command<br>`git clone https://github.com/YashasviBhatt/Diabetes-Prediction`<br>and press enter.
3. Go inside the _Cloned Repository_ folder and open _command-prompt_ or _powershell window_.

4. Type<br>`pip install -r requirements.txt`<br> and press enter in either _command_prompt_ or _powershell window_ as _administrator_.
5. After Installing all the required _libraries_ run the python file using<br>`python diabetes_prediction.py`.

## Working

1. Firstly, _data_ is imported using `pandas library`.
2. Secondly, data is **preprocessed** and **cleaned** using `Data Preprocessing Techniques`.
2. Thirdly, we divide the _features_ and _label_ into seperate _dataframes_.
3. Now, after creating the separate dataframes for features and label, we split them into _training_ and _testing_ sets.
4. The _training set_ is used to train the _model_ using several types of classifiers like **Decision Tree Classifier**, **Logistic Regression**, **K-Nearest Neighbor Classifier**, **Random Forest Classifier**.
5. The **Accuracy Score** for each and every model is then analyzed and a plot is plotted to see which model best fits the data. And by the analysis it was found out that **Random Forest Classifier** has the highest accuracy score and thus should be used for further processing.<br><br><br>

**I have used Kaggle Dataset on Pima Indian Diabetes and you can download the dataset from [here](https://www.kaggle.com/uciml/pima-indians-diabetes-database/download). You can visit their website and check more info on the dataset from [here](https://www.kaggle.com/uciml/pima-indians-diabetes-database).**