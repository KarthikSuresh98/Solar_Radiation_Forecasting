# Solar_Radiation_Forecasting

This project is aimed at forecasting and predicting the solar radiation data.

  1 . Forecasting:
    In this task , recurrent neural networks were used and trained on 10 years historical UV radiation data. The goal of this 
    network is to accuractely predict the uv radiation data for the next day given today's data. 

    - uv_radiation_forecasting.py and uv_radiation_forecasting2.py correspond to the forecasting task.

2. Predicting solar radiation data from climate / weather parameters:
   In this task , the data is initially pre processed wherein any parameter that has very low or very high correlation is removed.
   The parameter with very high correlation is removed since in almost all cases these parameters tend to derivatives of the 
   target output varibale i.e solar radiation.
   
   Random Forest Classifier is then used to predict the solar radiation from this processed data. The results were quite
   good wherein the model is learning the dependance of solar radiation with these climate parameters such as temperature , 
   humidity etc. The model thus is generalizing well to unseen data and thus giving good testing set accuracy
   
   a) german_solar_data_predictions.py : In this program , the classifier is trained on a dataset from Germany. This dataset
      is not uploaded since the data was acquired confidentially through proper permissions. Here we train the model on data
      corresponding to a geographical location in Germany. This model is now tested on data that corresponds to different 
      location in germany. The model seems to performing well and is giving good accuracy.
      
   b) radiation_data_preprocess.py & radiation_data_forecasting.py : Here the dataset corresponds to 3 months of solar
      radiatin data along with several weather parameters such as temperature , wind speed , humidity etc. Initially the
      data is pre processed and appropriate parameters are selected. Then a random forest classifier is trained and tested
      on this data. The classifier gives satisfactory results as we were able to see the dependance of radiation data on
      temperature mainly.
