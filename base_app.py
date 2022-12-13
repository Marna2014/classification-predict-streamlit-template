"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
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

# Vectorizer
news_vectorizer = open("vectoriser.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Amped Solutions Tweet Classifer")
	st.subheader("Climate change tweet classification")
	st.image("Amped_Solutions.png")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "About Us", "Contact Us", "Customer Appreciation"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Customer Appreciation":
		st.info("Customer Appreciation about Amped Solutions services")
		# You can read a markdown file from supporting resources folder
		st.write("I have been working with Amped Solutions for ten years and I continue to recommend their easy-to-use platform that allows you to discover what people are saying about your brand on twitter all around the word.")


	# Building out the "Information" page
	if selection == "Contact Us":
		st.info("Contact Information for Amped Solutions")
		# You can read a markdown file from supporting resources folder
		st.write("hello@ampedsolutions.com, (011)  666 2345, www.ampetsolutions.co.za  ")


	# Building out the "About Us" page
	if selection == "About Us":
		st.info("General Information about Amped Solutions")
		# You can read a markdown file from supporting resources folder
		st.write("Amped Solutions have created a new platform centred around sentiment analysis. Together we make up a team of experts in data science and machine learning, passionate about making high quality twitter (social media) sentiment analysis so that companies can deliver more effective marketing and achieve favourable results. We have architected a solution with high accuracy and precision. We’re so excited to assist companies in expanding their potential, one tweet at a time…")

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/SVC_model.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))
			st.image("Amped_Solutions.png")


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
