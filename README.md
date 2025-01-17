This project uses a Gradient Boosting Regressor model to predict house prices in California based on features such as median income, house age, and more. The app is built using Streamlit to provide an interactive web interface where users can input values and receive predictions.


The model predicts the price of a house in California given the following input features:

- Median income of the area
  
- Age of the house
  
- Average number of rooms in the house
  
- Average number of occupants per house
  
- Latitude and longitude of the house
  
- Median house value in the area

- The dependencies include:

streamlit: For building the web application.

scikit-learn: For implementing the Gradient Boosting Regressor model.

numpy: For numerical operations.


Model: Gradient Boosting Regressor

Model type: Supervised regression model

Key hyperparameters:

n_estimators: 100
random_state: 42

California Housing dataset: The dataset is publicly available from the UCI Machine Learning Repository.
