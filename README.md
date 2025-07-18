Update:
- Added a prediction counter
- Added a prediction chart
- Added a prediciton summary panel


This project uses a Gradient Boosting Regressor model to predict house prices in California based on features such as median income, house age, and more. The app is built using Streamlit to provide an interactive web interface where users can input values and receive predictions.


Try the app here: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://california-housing-price-prediction-2yngxsjve5y7doemqpkwm8.streamlit.app/)


Approach

 Models Tried:
 
1. **MLP Regressor**:
   - Initially, I tried using an MLP Regressor to predict house prices.
   - However, the MLP Regressor produced worse results compared to Gradient Boosting Regressor. Despite being a powerful model for learning complex relationships, MLP struggled with the dataset, resulting in a lower R-squared score and higher Mean Squared Error.
   
   **MLP Regressor Results:**
   - R-squared: 0.46 (lower than Gradient Boosting)
   - Mean Squared Error: 0.71 (worse than Gradient Boosting)

2. **Gradient Boosting Regressor**:
   - After experimenting with MLP Regressor, I switched to Gradient Boosting Regressor.
   - This model delivered better performance with a higher R-squared score and lower Mean Squared Error, making it more suitable for this particular regression task.
   
   **Gradient Boosting Regressor Results:**
   - R-squared: 0.78
   - Mean Squared Error: 0.29

Conclusion:

- The Gradient Boosting Regressor outperformed the MLP Regressor due to its ability to model complex interactions between features more effectively for this dataset. The MLP Regressor struggled with overfitting and failed to capture the relationships between the features as well as Gradient Boosting.


The model predicts the price of a house in California given the following input features:

- Median income of the area
  
- Age of the house
  
- Average number of rooms in the house
  
- Average number of occupants per house
  
- Latitude and longitude of the house
  
- Median house value in the area
  

  The dependencies include:

streamlit: For building the web application.

scikit-learn: For implementing the Gradient Boosting Regressor model.

numpy: For numerical operations.




Model: Gradient Boosting Regressor

Model type: Supervised regression model

Key hyperparameters:

n_estimators: 100
random_state: 42

California Housing dataset: The dataset is publicly available from the UCI Machine Learning Repository.

This project is licensed under the MIT License.
