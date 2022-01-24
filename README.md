# Application : Système de recommandation de footballeurs


## Solution 
Je suis fier de vous présenter le site hébergeant le système de recommandation : [http://footballerprediction.herokuapp.com/players](http://footballerprediction.herokuapp.com/players)


## Introduction :

Football is no longer just a sport, but also a major business. Player transfers are important for the clubs both in a sporting and a business sense. In the market, many millions of dollars are often paid for football players. Up to now in professional football, market values have primarily been estimated by sports magazines, or fans on internet platforms. 

Both experts and fans estimates have repeatedly been shown to be highly accurate, but the estimation process is usually extremely complex. Market values are therefore often only updated once or twice a year. Our goal through this project is to implement a conjecturing model in order to predict the market value of a player according to his personal football skills and attributes. 


*Comment trouver un joueur qui correspondrait à toutes nos attentes ?*

Nowadays we think that some players might be over evaluated that's why **we implemented an application in order to predict any footballer's price only by entering his name**. The application will also predict your own market value according to your abilities.

You can find the website that we scrapped for this project on the following link : [https://www.fifaindex.com/fr/players/](https://www.fifaindex.com/fr/players/)
<br>


## This project is implemented in Python and gathers tasks of :
* Web scrapping : getting and storing data from a wesite
* Data visualisation : interesting plots about players'market value and features on the Jupyter Notebook
* Machine learning algorithm modelisation on the Jupyter Notebook
* Transformation of the model into an API Flask
  

## You will find in this repositery : 
* A [LaTeX report of the study](https://github.com/thomastrg/Price_prediction_footballers/blob/main/rapport/Player_Value_Prediction.pdf)
* The [Scrapping function](https://github.com/thomastrg/Price_prediction_footballers/blob/main/Scrapping/Scrapping_fifaindex_function.py)
* The [Data exploration processing and modeling Notebook](https://github.com/thomastrg/Price_prediction_footballers/tree/main/Modeling)
* The [Flask application deployed with Heroku](http://footballerprediction.herokuapp.com/players)
<br> 

## Conclusion : 
Overview of the website :  <br> <br>
<img src="https://github.com/thomastrg/Price_prediction_footballers/blob/main/rapport/Hnet-image.gif" width="600" height="370"/>
