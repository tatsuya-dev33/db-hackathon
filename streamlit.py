import streamlit as st 
import numpy as np 
import json
import requests
import pandas as pd
from dotenv import load_dotenv
import os
from databricks.sdk.runtime import dbutils

# Load environment variables from .env file
load_dotenv(".env")

# Retrieve secret scope name, key, and serving endpoint URL from environment variables
SECRET_SCOPE_NAME = os.getenv("SECRET_SCOPE_NAME")
SECRET_SCOPE_KEY = os.getenv("SECRET_SCOPE_KEY")
SERVING_ENDPOINT_URL = os.getenv("SERVING_ENDPOINT_URL")

# Set the title of the Streamlit app
st.title('Copen Q&A bot')

def generate_answer(question):
    """
    Generate an answer for a given question by making a request to the serving endpoint.

    Args:
        question (str): The question to ask the model.

    Returns:
        dict: The JSON response from the serving endpoint containing the answer.
    """
    # Retrieve the token from Databricks secrets
    token = dbutils.secrets.get(SECRET_SCOPE_NAME, SECRET_SCOPE_KEY)
    url = SERVING_ENDPOINT_URL

    # Set up headers for the request
    headers = {
        "Content-Type": "application/json",
        "Authentication": f"Bearer {token}"
    }

    # Prepare the data payload
    data = {
        "query": question
    }

    # Convert question to DataFrame and then to dictionary format required by the API
    prompt = pd.DataFrame({"query": [question]})
    ds_dict = {"dataframe_split": prompt.to_dict(orient="split")}

    # Make a POST request to the serving endpoint
    response = requests.post(url, headers=headers, data=json.dumps(ds_dict))
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")

    # Parse the JSON response
    response_json = response.json()
    return response_json

# Initialize the session state if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Respond to user input
if prompt := st.chat_input("What do you want to know about Copen?"):
    # Display user message in chat container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display bot response
    with st.spinner('Generating response...'):
        bot_response = generate_answer(prompt)
        answer = bot_response["predictions"][0]["result"]
        url = bot_response["predictions"][0]["source_documents"][1]["metadata"]["url"]

        response = f"""
        {answer} \n
        URL: {url}
        """

    # Display assistant's response in chat container
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add assistant's response to session state
    st.session_state.messages.append({"role": "assistant", "content": response})
