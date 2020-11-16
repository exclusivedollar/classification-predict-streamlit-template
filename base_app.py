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

# Data dependencies
import pandas as pd

# Model Building
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
#

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

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home", "Information", "Exploratory Data Analysis(EDA)", "Prediction", "About Us"]
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

# Building out the "Home" page
	if selection == "Home":
		st.markdown("<h1 style='text-align: center; color: blue;'>Home</h1>", unsafe_allow_html=True)
		st.image('resources/imgs/1320_effects-.jpeg',use_column_width=True)


# Building out the "Information" page
	if selection == "Information":
		st.markdown("<h1 style='text-align: center; color: blue;'>Information</h1>", unsafe_allow_html=True)
		st.markdown("")
		#st.image('resources/imgs/info.png',width=500)
		st.markdown("")
		st.info("An information on the project and application:")
		st.markdown("")
		st.markdown("")
		if st.checkbox("Project Information"):
		   st.subheader("Project Information")
		   st.subheader("==========================================================")
		   st.info("General Information")
		   st.markdown("In a [research article](https://www.barrons.com/articles/two-thirds-of-north-americans-prefer-eco-friendly-brands-study-finds-51578661728) conducted, 19,000 customers from 28 countries where given a poll to find out how individual shopping decisions are changing. Nearly 70% of consumers in the U.S. and Canada find that it is important for a company or brand to be sustainable or eco-friendly. More than a third (40%) of the respondents globally said that they are purpose-driven consumers, who select brands based on how well they align with their personal beliefs. Many companies are built around lessening their environmental impact or carbon footprint. They offer products and services that are environmentally friendly and sustainable, in line with their values and ideals. They would like to determine how people perceive climate change and whether or not they believe it is a real threat. This would add to their market research efforts in gauging how their product/service may be received.  The goal of this challenge is to build a Classification Machine Learning model that will determine whether a person believes in Climate Change using tweet data. This model will provide insights of public opinion of Climate Change & consumer sentiment to companies looking to market their new or improved products or services to consumers, in response to CER. As the demand for sustainable, eco-friendly products and services by consumers increases, a sentiment classification model that identifies these potential customers is key and could be used any business or organisation committed to carbon neutrality & wanting to inform marketing strategies. This includes, but is not limited to companies in the retail, automotive, government, agriculture & food, pharmaceutical spheres. The model could also be used by sectors in government wanting to identify the various belief sentiments in order to better direct environmental awareness and education campaigns in alignment with their legislative directives and climate change response plans.")
		   st.subheader("==========================================================")
		   st.info("Problem Statement")
		   st.subheader("==========================================================")
		   st.markdown("Build a machine learning model that is able to classify whether or not an individual believes in man-made climate change based on historical tweet data to increase insights about customers and inform future marketing strategies.You can find the project overview [here](https://www.kaggle.com/c/climate-change-edsa2020-21).")
		   st.subheader("==========================================================")
		#st.subheader("We are excited to have you on our Weather Prection App!")
		#st.image('resources/imgs/1320_effects-.jpeg',use_column_width=True) 
		if st.checkbox("App Information"):
		   st.subheader("App Information") 
		   st.info("Navigation Bar")  
		   st.subheader("==========================================================")
		   st.markdown("* Home - Landing page, name of the App")
		   st.subheader("==========================================================")
		   st.markdown("* Information - A detailed project and application to provide clear understanding of the project and usage of the application(App)")
		   st.subheader("==========================================================")
		   st.markdown("* EXploratory Data Analysis(EDA) - Exploratory data analysis shows how we analysing the tweets data sets to summarize their main characteristics, using visuals. EDA is basically for displaying what the data can tell us beyond the formal modeling.")
		   st.subheader("==========================================================")
		   st.markdown("* Predictions - This is where the magic happens. This is the part where a tweet can be typed in and be classified based on the models we created. User aslo has option on which model they want to use for classification. This page also shows the accuracy score of the model used. This is a reflection of efficiency of the used model.")
		   st.subheader("==========================================================")
		   st.markdown("* About Us -  Displays team information: names, github accounts and email adresses.")
		   st.subheader("==========================================================")
		   st.info("App Usage") 
		   st.subheader("==========================================================")
		   st.markdown("The App require user's interaction for navigation from one page to the other using the selection/navigation bar, top left. For predictions, the user is required to input text (ideally a tweet relating to climate change), and will classify it according to whether or not they believe in climate change through using different model options amde availble on the page.")
		   st.markdown("")
		   st.subheader("==========================================================")
		   
####################EDA

	if selection == "Exploratory Data Analysis(EDA)":
		st.markdown("<h1 style='text-align: center; color: blue;'>Exploratory Data Analysis(EDA)</h1>", unsafe_allow_html=True)  
		st.markdown("")
		st.markdown("Exploratory Data Analysis(EDA) is a critical process of performing initial investigations on data so as to discover patterns, to spot anomalies, to test hypothesis and to check assumptions with the help of summary statistics and graphical representations. The section is an exploration of the data through an analysis of the different Climate Change sentiments that people have on Twitter.")
		st.markdown("")
		

		st.markdown("")
		st.markdown("")
	
		# Building checkbox to give user options
		if st.checkbox('Understanding the distribution of sentiments'):
		   st.subheader("==========================================================")
		   st.subheader("Distribution of sentiments")
		   st.markdown("As seen in the bar graph, sentiment class 1 has the highest number of tweets in the train data accounting for 8530 tweets(53.92%).The lowest sentiment class is class -1 which accounts for 1296 tweets (8.19%).The distribution of sentiments classes are imbalanced because the classes do not have the same ammount of tweets in their class as seen in dataframe which compares the value counts and percentage of each sentiment class. The class imbalance of the training data has an impact on the classification made on the unseen data (testing data) in the modeling phase.A class imbalance could result in the model classifying most of the tweets into sentiment class 1 since the model gets better a classifying class 1 tweets as the model has more evidence of class 1 tweets.This will be taken into consideration in the preprocessing and modeling section of the notebook")
		   st.image('resources/imgs/sentiments(graph).PNG',use_column_width=True)
		   st.image('resources/imgs/sentiments.PNG',use_column_width=True)
		   st.subheader("==========================================================")
		   
		if st.checkbox("The main topics on climate change"):
		   st.markdown("An understanding of the main topics dicussed in the climate change discussion on twitter is essential as it illustrates the sentiments attatched to climate change. This is done through extracting the most frequently used words and hashtags.")
		   st.subheader("==========================================================")
		   st.subheader("Top 30 used words from tweets:")
		   st.image('resources/imgs/Stlit.png',use_column_width=True)
		   st.markdown("")
		   st.markdown("")
		   st.image('resources/imgs/usedwords.png',use_column_width=True)
		   st.markdown("")
		   st.markdown("")
		   st.image('resources/imgs/usedwords_test.png',use_column_width=True)
		   st.subheader("==========================================================")

		   st.subheader("Wordcloud- Vocabulary from tweets:")
		   st.image('resources/imgs/wordcloud.PNG',use_column_width=True)
		   st.subheader("==========================================================")
		
		   st.subheader("The top 10 influencial Twitter accounts in the climate change debate")
		   st.markdown("The accounts that recieved the most mentions are Twitter accounts that have engaged with the climate change topic.Twitter users mention these accounts when reposting(retweeting) the twitter accounts sentiment on climate change or responding to the twitter accounts comment on climate change.Within the data these Twitter accounts have played a vital role in fueling the climate change debate on Twitter.")
		   st.image('resources/imgs/accounts.PNG',use_column_width=True)
		   
		   #Sentiments
		   st.subheader("==========================================================")
		   st.subheader("Top 10 hashtags used in Sentiment class")
		   st.markdown("* Class -1 tweets")
		   st.markdown("In class -1 the hashtag that was used the most is #MAGA and the second highest being #climate.These keywords were the most used when people were discussing their sentiments concerning climate change.Other interesting hashtags that form part of the top ten hashtags used in class one are #fakenews and #ClimateScam which insinuate that some of the people who were tweeting about climate change believe that is is simply fake news or a scam. The third highest hashtag used is #Trump when discussing climate change. The class focuses more on discussing climate change as being linked to politics hence the hashtag that has been used the most is #MAGA as well as the example of one of the tweets provided in the cell above.")
		   st.markdown("RT @Cernovich: Same 'experts' who said Hillary would win claim 'climate change' is real. LOL! Go away, morons, you know nothing and you losÃ¢â‚¬Â¦")
		   st.image('resources/imgs/sentiments(-1).PNG',use_column_width=True)
		   
		   st.markdown("* Class 0 tweets")
		   st.markdown("'RT @CivilJustUs: How do they expect us to care about global warming with all this penguin on penguin crime?? https://t.co/HypysWHvVV'")
		   st.markdown("The keyword that is used the most when discussing climate change is #climate followed by #climatechange.#Trump is a prominent hashtag in class 0 as well.Donald Trump's views on climate change is discussed in the class.An interesting hashtag used by people is #BeforeTheFlood which is a movie that depicts the impacts of climate change on the Earth,as well as #amreading people use this hashtage to tell mention what they a book or article they are currently reading. The sentiments within class 0 are open conversations surrounding climate change including people asking questions about climate change as well as sarcasm")
		   st.image('resources/imgs/sentiments(0).png',use_column_width=True)

		   st.markdown("* Class 1 tweets")
		   st.markdown("'RT @AstroKatie: Governments of several world powers are failing us on climate change. We need to act without them if we want any hope for tÃ¢â‚¬Â¦'")
		   st.markdown("The opinions on climate change in class 1 shift towards climate change does exist as the conversations in this class discuss a movie called Before the flood.The movie highlights the impact of climate change on the Earth.As well as using the hashtag #ActOnClimate, the tweets associated with the hastag on Twitter mainly discuss ways to combat climate change (http://www.tweepy.net/hashtag/ActOnClimate).")
		   st.image('resources/imgs/sentiments(1).PNG',use_column_width=True)
		   
		   st.markdown("* Class 2 tweets")
		   st.markdown("'RT @tveitdal: We only have a 5 percent chance of avoiding ‘dangerous’ global warming, a study finds https://t.co/xUBTqNxhkK https://t.co/of…'")
		   st.markdown("The opinions in class one mainly focus on the climate this is evident in the high hashtag count of the word #climate, the second highest is #enviroment .The class is mainly focused on informing people about climate change and its effect on the enviroment.")
		   st.image('resources/imgs/sentiments(2).PNG',use_column_width=True)
		   st.subheader("==========================================================")
		   
		if st.checkbox("The key findings from the Exploratory Data Analysis(EDA)"):
		   st.markdown("")
		   st.markdown("* There are polarised views on climate change on twitter")
		   st.markdown("* Within the data there exists a class imbalance,this will be considered in the preprocessing and model training section")
		   st.markdown("* An analysis of the hashtags has shown that the tweets in class 1 believe in climate change,class 2 believe and inform people about climate change,class 0 are more neutral and tend to downplay the existence of climate change and class -1 do not believe that climate change exists.")
		   st.subheader("==========================================================")
# Building out the predication page
	if selection == "Prediction":
		st.markdown("<h1 style='text-align: center; color: blue;'>Prediction with ML Models</h1>", unsafe_allow_html=True)
		st.subheader("==========================================================")
		st.markdown("Predictive modeling is a process that uses data and statistics to predict outcomes with classification data models. These models can also be used to predict our twitter data. We get predictions from models such as Logistic Regression, Decision Tree Classifier, Random Forest Classifier and many more.")
		st.subheader("==========================================================")
		st.markdown("LogisticRegression- Is used to obtain odds ratios in the presence of more than one exploratory variable. It explains the relationship between one dependent binary variable and one or more independent variables")
		st.subheader("=========================================================")
		st.subheader("==========================================================")
		st.markdown("Decision Tree- builds regression or classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes.")
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

		#if st.button("Random forest"):
		# Transforming user input with vectorizer
			#vect_text = tweet_cv.transform([tweet_text]).toarray()
		# Load your .pkl file with the model of your choice + make predictions
		# Try loading in multiple models to give the user a choice
			#predictor = joblib.load(open(os.path.join("resources/Random_Forest.pkl"),"rb"))
			#prediction = predictor.predict(vect_text)

		# When model has successfully run, will print prediction
		# You can use a dictionary or similar structure to make this output
		# more human interpretable.
			
			#if prediction[0] == 1:
  				#st.success('Your message has been classified as showing positive belief in climate change')
			#elif prediction[0] == 0:
  				#st.success('Your message has been classified as showing being neutral towards climate change')
			#elif prediction[0] == 2:
  				#st.success('Your message has been classified as news')
			#else:
  				#st.success('Your message has been classified as showing negative belief in climate change')

			#st.success("Text Categorized as: {}".format(prediction))
			
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


# Building out the "About Us" page
	if selection == "About Us":
		st.markdown("<h1 style='text-align: center; color: blue;'>About Us</h1>", unsafe_allow_html=True)
		st.markdown("")
		st.markdown("")
		st.info("1. Nthabiseng Moela")
		st.image('resources/imgs/Nthabi.PNG')
		st.markdown("* Github account:NthabisengMoela")        
		st.markdown("* email:nthabisengmoela1@gmail.com")
		st.markdown("")
		st.info("2.Bukelwa Mqhamane")
		st.image('resources/imgs/Bukelwa.PNG',width=300)
		st.markdown("* Github account:")        
		st.markdown("* email:bmqhamane@gmail.com")
		st.markdown("")
		st.info("3. Mfumo Baloyi")
		st.image('resources/imgs/Mfumo.PNG')
		st.markdown("* Github account : mfumoB")        
		st.markdown("* Email:www.baloyimfumoe@gmail.com")
		st.markdown("")
		st.info("4. Sammy Maakwana")
		st.image('resources/imgs/Sammy.PNG')
		st.markdown("* Github account:exclusivedollar")        
		st.markdown("* email:maakwana@gmail.com")
		st.markdown("")
		st.info("5. Chuene Mokgokong")
		st.image('resources/imgs/Chuene.PNG')
		st.markdown("* Github account:Grewies")        
		st.markdown("* email:mokgokonggrewies01@gmail.com")
#######################################################
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
