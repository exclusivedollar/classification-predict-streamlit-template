"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Model Building
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/vectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	
	title_temp ="""
	<div style="background-color:#ED7117;padding:10px;border-radius:10px;margin:10px;">
	<h1 style="color:white;text-align:center;">CLASSIFICATION STREAMLIT APP</h1>
	<h2 style="color:white;text-align:center;">TEAM 2</h2>
	<h3 style="color:white;text-align:right;">2020-11-18</h3>
	</div>
	"""
	st.markdown(title_temp, unsafe_allow_html = True)

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Information", "Exploratory Data Analysis", "Prediction", "About Us"]
	selection = st.sidebar.selectbox("Choose Option", options)
	
	# To bring the background colour
	st.markdown(
		"""
		<style>
		.sidebar .sidebar-content {
		background-image: linear-gradient(#FC6A03, #C5C6D0);
		font color: white;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)

	# Building out the "Information" page
	if selection == "Information":
		if st.info("Problem Statement"):
			st.write("Build a Natural Language Processing models to classify whether or not a person believes in climate change or based on their novel tweet data")
			
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")
		st.markdown("Some information here")
		st.markdown("Many companies are built around lessening one’s environmental impact or carbon footprint. They offer products and services that are environmentally friendly and sustainable, in line with their values and ideals. They would like to determine how people perceive climate change and whether or not they believe it is a real threat. This would add to their market research efforts in gauging how their product/service may be received.")
		st.subheader("==========================================================")
		st.subheader("Benefits of using this classification tool")
		st.subheader("==========================================================")
		st.markdown("1. Quicker near real-time results compared to a survey without having to pay for expensive reports")
		st.markdown("2. Easily accessible on your internet browser")
		st.markdown("3. You don't need to understand complicated statistical techniques, just need to understand insights")
		st.markdown("If you are a startup or even an established business looking to launch a new product, are you aware of your potential customers sentiments regarding climate change? As a not for profit organisation looking for donors for environmental projects, do you know what your donors thoughts are regarding climate change? Knowing this information can help you better prepare to take your organisations strategy forward. Not knowing this information can make you seem irrelevant to your target market and cause you to miss out on an opporunity of a lifetime. The tweet classifier will help you be more prepared and relevant to your audience.")
		st.subheader("==========================================================")
		st.subheader("Instructions for using this tool")
		st.subheader("==========================================================")
		st.markdown("Let us help you turn insights from your potential customers to action.")
		st.markdown("Get started by:")
		st.markdown("1. Navigating to the sidebar at the top left of this page")
		st.markdown("2. Choose an option by clicking the 'Choose Option' dropdown")
		st.markdown("3. Select the option you wish to view")
		st.markdown("4. Get insights that will help you be better prepared")


		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
			
	if selection == "Exploratory Data Analysis":
		st.markdown("Exploratory Data Analysis refers to the critical process of performing initial investigations on data so as to discover patterns,to spot anomalies,to test hypothesis and to check assumptions with the help of summary statistics and diagramatic representations")
		
	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		st.subheader("==========================================================")
		st.markdown("Predictive modeling is a process that uses data and statistics to predict outcomes with classification data models. These models can also be used to predict our twitter data. We get predictions from models such as Logistic Regression, Decision Tree Classifier, Random Forest Classifier and many more.")
		st.subheader("==========================================================")
		st.markdown("LogisticRegression- Is used to obtain odds ratios in the presence of more than one exploratory variable. It explains the relationship between one dependent binary variable and one or more independent variables")
		st.subheader("==========================================================")
		st.markdown("Random Forest-  is an ensemble of decision trees. This is to say that many trees, constructed in a certain “random” way form a Random Forest. Each tree is created from a different sample of rows and at each node, a different sample of features is selected for splitting. Each of the trees makes its own individual prediction. These predictions are then averaged to produce a single result.")
		st.subheader("==========================================================")
		st.markdown("Decision Tree- builds regression or classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes.")
		st.subheader("==========================================================")
		st.markdown("Random Forest is the model that performs best, you can check the other models to compare the results")
		st.subheader("==========================================================")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		#if st.button("Classify"):
			# Transforming user input with vectorizer
			#vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			#predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			#prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			
	if st.button("Random forest"):
		# Transforming user input with vectorizer
		vect_text = tweet_cv.transform([tweet_text]).toarray()
		# Load your .pkl file with the model of your choice + make predictions
		# Try loading in multiple models to give the user a choice
		predictor = joblib.load(open(os.path.join("resources/Random_Forest.pkl"),"rb"))
		prediction = predictor.predict(vect_text)

		# When model has successfully run, will print prediction
		# You can use a dictionary or similar structure to make this output
		# more human interpretable.
			
		if prediction[0] == 1:
  			st.success('Your message has been classified as showing positive belief in climate change')
		elif prediction[0] == 0:
  			st.success('Your message has been classified as showing being neutral towards climate change')
		elif prediction[0] == 2:
  			st.success('Your message has been classified as news')
		else:
  			st.success('Your message has been classified as showing negative belief in climate change')

		st.success("Text Categorized as: {}".format(prediction))
			
	if st.button("Decision Tree"):
		# Transforming user input with vectorizer
		vect_text = tweet_cv.transform([tweet_text]).toarray()
		# Load your .pkl file with the model of your choice + make predictions
		# Try loading in multiple models to give the user a choice
		predictor = joblib.load(open(os.path.join("resources/Decision Tree.pkl"),"rb"))
		prediction = predictor.predict(vect_text)

		# When model has successfully run, will print prediction
		# You can use a dictionary or similar structure to make this output
		# more human interpretable.

			
		if prediction[0] == 1:
  			st.success('Your message has been classified as showing positive belief in climate change')
		elif prediction[0] == 0:
  			st.success('Your message has been classified as showing being neutral towards climate change')
		elif prediction[0] == 2:
  			st.success('Your message has been classified as news')
		else:
  			st.success('Your message has been classified as showing negative belief in climate change')


		st.success("Text Categorized as: {}".format(prediction))
			
	if st.button("Logistic Regression"):
		# Transforming user input with vectorizer
		vect_text = tweet_cv.transform([tweet_text]).toarray()
		# Load your .pkl file with the model of your choice + make predictions
		# Try loading in multiple models to give the user a choice
		predictor = joblib.load(open(os.path.join("resources/Logistic Regression.pkl"),"rb"))
		prediction = predictor.predict(vect_text)

		# When model has successfully run, will print prediction
		# You can use a dictionary or similar structure to make this output
		# more human interpretable.

			
		if prediction[0] == 1:
  			st.success('Your message has been classified as showing positive belief in climate change')
		elif prediction[0] == 0:
  			st.success('Your message has been classified as showing being neutral towards climate change')
		elif prediction[0] == 2:
  			st.success('Your message has been classified as news')
		else:
  			st.success('Your message has been classified as showing negative belief in climate change')



			st.success("Text Categorized as: {}".format(prediction))
		
	if selection == "About Us":
		st.info("Explore Data Scientists")
		st.markdown("Mfumo Baloyi === Email:baloyimfumoe@gmailcom")
		st.markdown("Chuene Mokgokong === Email:mokgokonggrewies01@gmail.com")
		st.markdown("Nthabiseng Moela === Email:nthabisengmoela1@gmail.com")
		st.markdown("Sammy Maakwana === Email:maakwana@gmail.com")
		st.markdown("Bukelwa Mqhamane === Email:mabum7@gmail.com")

		st.write("We are a team of data scientists from the explore academy. We are all have blue belts ranking so we come with a experience in our field. Under the supervision of Siphesihle Yapi.")
		
		



# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
