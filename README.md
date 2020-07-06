# ✈Flight Price Prediction
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)[![forthebadge](https://forthebadge.com/images/badges/its-not-a-lie-if-you-believe-it.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/built-by-developers.svg)](https://forthebadge.com)

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Technical Aspect](#technical-aspect)
  * [Bug / Feature Request](#bug---feature-request)
  * [Technologies Used](#technologies-used)
  * [License](#license)
  * [Contact](#contact-)

## Demo
![](demos.gif)


## Overview

A Flask based Web Application that Predicts the Flight Price using RandomForestRegressor.Its GUI is based on Swagger API. This is hosted on the Heroku platform.

## Motivation

People who are travelling in the flights face the problem of price fluctuation to avoid this I thought of to predict the price of the flight ticket so that passenger will have a idea of how much price of flight will be.

## Technical Aspect

1. Firstly I collected dataset from kaggle.I am providing the link of the dataset [here](https://drive.google.com/drive/folders/1RFnObzTjFsrGVOzpw6q_WYQQJoehglA2?usp=sharing).

2. Now I used did exploratory data analysis by cleaning the dataset.

3. Now I did the feature engineering and identified which features affect our prices .

4. Then I build a model using random forest regression and tuned hyper parameters to improve the performance.

5. I used the Swagger API to create a GUI for web application and used flask to deal with the requests and predict the outputs and run the application in the development server .
6. Now I add the requirements and ProcFile and deployed using Heroku CLI.





## Bug / Feature Request
If you find a bug (gave undesired results), kindly open an issue [here](https://github.com/ksdkamesh99/Flight-Price-Prediction/issues/new/choose) by including your search query and the expected result.

If you'd like to request a new function, feel free to do so by opening an issue [here](https://github.com/ksdkamesh99/Flight-Price-Prediction/issues/new/). Please include sample queries and their corresponding results.

## Technologies Used


[<img target="_blank" src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width=200>](https://scikit-learn.org/stable/) [<img target="_blank" src="https://static1.smartbear.co/swagger/media/assets/images/swagger_logo.svg" width=170>](https://swagger.io/blog/api-development/automatically-generating-swagger-specifications-wi/)[<img target="_blank" src="https://miro.medium.com/max/3600/1*fIjRtO5P8zc3pjs0E5hYkw.png" width=200>](https://www.heroku.com/)[<img target="_blank" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png" width=170>](https://flask.palletsprojects.com/en/1.1.x/) 



## 📧Contact:-
For any kind of suggesstions/ help in code Please mail me at ksdkamesh99@gmail.com.
