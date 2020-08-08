# Web Scraping Spotify & Classification

### Project 3 at Metis - ["This Repo"](https://github.com/er-arcadio/Spotify_Classifier_Project)

**Objective: "Can a song be predicted to hit the top 100 on Spotify Charts?"**

The goal of the project is to create a predictor that will classify a song as a top hit or not, and extract the features that determine such classification. Data will be scraped from spotify through their API, explored and visualized via Jupyter Notebook, modeled via a flask app, and summarized in presentation slides.

## Spotify API and Webscraping 

- Learn the Spotify API and sign up for credentials
- Write python library that will make the webscraping process easier
- Scrape the "Spotify Charts," the Top 200 most streamed songs as of late
- Scrape a user's playlist of about 1000 hits from the past decade


**Files Refferenced**

[Spotify API Python File](https://github.com/er-arcadio/Spotify_Classifier_Project/blob/master/Spotify.py) *Note: Won't run without credentials*

[Spotify Webscraping Notebook](https://github.com/er-arcadio/Spotify_Classifier_Project/blob/master/Spotify_Notebooks/Spotify1_Scraping.ipynb)


## EDA, Model Selection, and Feature Engineering

- Combine the data into 1, clean, by removing duplicates and fixing nulls etc.
- Come up with a baseline model 
- EDA -- Analyze the pair plots and class imbalance
- Find better models, feature engineer, tune hyperparameters, and finalize model.

**Files Refferenced**

[Model Selection Python File](https://github.com/er-arcadio/Spotify_Classifier_Project/blob/master/model_selection.py)

[Cleaning, EDA, and Baseline Notebook](https://github.com/er-arcadio/Spotify_Classifier_Project/blob/master/Spotify_Notebooks/Spotify2_EDA_MVP.ipynb)

[Feature Engineering, and Final Model Notebook](https://github.com/er-arcadio/Spotify_Classifier_Project/blob/master/Spotify_Notebooks/Spotify3_Model_Tuning.ipynb)


## Flask App Predictor & Presentation

- Convert the Model into an App/ Predictor
- Create a Presentation with highlighted points

**Files Refferenced**

[Flask App Files](https://github.com/er-arcadio/Spotify_Classifier_Project/tree/master/Flask_App)

[Presentation (pdf)](https://github.com/er-arcadio/Spotify_Classifier_Project/blob/master/Project_3_Presentation.pdf)

