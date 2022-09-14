import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'absences':6,'G1':5,'G2':6})

print(r.json())