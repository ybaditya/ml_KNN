import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Machine Learning Apps")
st.title("Machine Learning Apps")
st.subheader("K-Nearest Neighbors Classification")
st.write(":blue[Created by : YB Aditya]")
st.markdown("---")

st.caption(
    """The k-nearest neighbors (KNN) algorithm is a data classification method for estimating the likelihood that a data point will become a member of one group or another based on what group the data points nearest to it belong to"""
)

image = Image.open("knn.jpg")
st.image(image, caption="K-Nearest Neighbors")

uploaded_file = st.file_uploader("Choose a XLSX file", type="xlsx")
st.caption(
    "_please make sure there is no Null-data, null data will make machine learning error_"
)
if uploaded_file:
    st.markdown("---")
    data = pd.read_excel(uploaded_file, engine="openpyxl")
    st.write("**Data Preview :**")
    st.write(data.head())
    st.write("**Data Information :**")
    st.caption(
        """**count** : The number of not-empty values. 
                  ,**mean** : The average (mean) value.
                  ,**std** : The standard deviation.
                  ,**min** : the minimum value.
                  ,**25%** : The 25% percentile*.
                  ,**50%** : The 50% percentile*.
                  ,**75%** : The 75% percentile*.
                  ,**max** : the maximum value."""
    )
    describe = data.describe()
    st.write(describe)

    st.write("**Corelation Between Data :**")
    st.caption(
        "_Note: **1** is Perfect corelation, **0,9-06** Good Corelation, **<0,5** Bad Corelation, use this data to select Independent Variables_"
    )
    correlation = data.corr(numeric_only=True, method="pearson")
    st.write(correlation)

    st.write("**Select The Target Variable (_Y_):**")
    target_var = st.selectbox("Target Variable", list(data.columns))

    st.write("**Select The Independent Variables (_X1, X2, X3, ..._):**")
    independent_vars = st.multiselect("Independent Variables", list(data.columns))


def knn_classification():
    # Plot Graph Correlation
    st.write("**Graph Corelation Between Data :**")
    fig = sns.pairplot(data, x_vars=independent_vars, y_vars=independent_vars, hue=target_var)
    st.pyplot(fig)
    fig_1 = sns.pairplot(data, x_vars=independent_vars, y_vars=target_var, hue=target_var)
    st.pyplot(fig_1)
    # Train a Multiple Regression model and display the results
    X = data[independent_vars]
    y = data[target_var]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test, random_state=random
    )
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    model = knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    st.write("**Performance Evaluation**")
    model_score = model.score(x_test, y_test)
    st.write("Score :", model_score)
    st.caption(
        f"Score also mean model accuracy, your model accurancy is {model_score*100}%"
    )

    con_matrix = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix :")
    st.dataframe(pd.DataFrame(con_matrix))
    st.caption("Confusion Matrix :  matrix shows the number of correct and incorrect predictions made by the classifier.")

    st.write("Classification Report :")
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())
    st.caption("Classification Report :  report shows precision, recall, f1-score, support and accuracy metrics for each class in the dataset.")

    st.write("**Do you want to download this model ?**")
    st.caption("**WARNING THIS WILL RELOAD THE PAGE!**")
    st.download_button(
        "Download Model",
        data=pickle.dumps(model),
        file_name="model.pkl",
    )

    predict_var = knn.predict([variable])
    st.sidebar.write("Prediction Value :", predict_var[0])


st.sidebar.write("_**Input Machine Learning Parameter**_")
test = st.sidebar.slider("Input test size", min_value=0.1, max_value=0.4)
st.sidebar.caption("define test size from the dataset. 0.2 mean 20% from total dataset")
random = st.sidebar.number_input("Input random state", min_value=0, max_value=50)
st.sidebar.caption(
    "random state is a model hyperparameter used to control the randomness involved in machine learning models"
)
neighbors = st.sidebar.number_input("Input k_neighbors", min_value=0, max_value=20)
st.sidebar.caption(
    "k_neighbors : the number of nearest neighbours to include in the majority of the voting process."
)
ml_button = st.sidebar.button("Run")


if uploaded_file and independent_vars is not None:
    variable = []
    st.sidebar.markdown("---")
    st.sidebar.write("_**Input Data for Prediction**_")
    for i in range(len(independent_vars)):
        var = st.sidebar.number_input(f"Input {independent_vars[i]}")
        variable.append(var)

if ml_button:
    knn_classification()


