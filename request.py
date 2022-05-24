import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'radius_mean':17.99,'texture_mean':10.38,'area_mean':1001, 'concavity_mean':0.3001,'concave points_mean':0.1471,'area_se':153.4,'radius_worst':25.38,'texture_worst':17.33,'perimeter_worst':184.6,'area_worst':2019,'concavity_worst':0.7119,'concave points_worst':0.2654,'symmetry_worst':0.4601})

print(r.json())