# Sentiment-analysis
Analysed twitter tweets. The dataset has been derived from kaggle.<br>
This model has an accuracy of 78.99% and also measures its confidence%.<br>
For the frontend, streamlit has been used.<br>
The project has been deployed on AWS cloud server in EC2 instance.<br>
Steps to deploy it on AWS Cloud virtual server.<br>
AMI: Ubuntu(Free tier)<br>
Instance type: t2.small<br>
```
sudo apt update
```
```
sudo apt-get update
```
```
sudo apt upgrade -y
```
```
sudo apt install git curl unzip tar make sudo vim wget -y
```
```
git clone "Your-repository"
```
```
cd "repository name"
```
In my case, I had to install Streamlit by creating a Python Virtual Environment<br>
Below are the steps<br>
```
sudo apt install python3-venv
```
```
python3 -m venv ~/streamlit-env
```
```
source ~/streamlit-env/bin/activate
```
```
pip install streamlit
```
```
streamlit run your_app.py
````
Copy the Public address on new tab and after that mention the port number, in this case it's 8501.

