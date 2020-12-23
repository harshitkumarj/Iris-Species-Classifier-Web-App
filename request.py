import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'sepal_length':4, 'sepal_width:3, 'petal_length':4, 'petal_width':1})

print(r.json())