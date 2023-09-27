import json
import requests
import numpy as np
from tensorflow.keras.preprocessing import image

filename = '/home/lenovo/Projects/Freelance/skin_analysis_app/aiprojectphase1backend_working/static/1_1590387428.jpeg'
#Read the image
#img = cv2.imread(filename)
img = image.load_img(filename, target_size=(224,224))

#print(img.shape)

#Preprocess the image
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#image = cv2.resize(image,(224,224), interpolation = cv2.INTER_NEAREST)
img = image.img_to_array(img)
image = img/255
image = np.expand_dims(image,axis = 0)
img = np.concatenate((image, image, image))
print(img.shape)

#Query the model
#1. Create json string
data = json.dumps({"signature_name": "serving_default", "instances":img.tolist()})
#2. Headers for post request
headers = {"content-type": "application/json"}
#3. Request
#json_response = requests.post('http://localhost:9000/v1/models/first_model/versions/1:predict', data = data, headers = headers)
resp_wpr = requests.post('http://localhost:9000/v1/models/wpr_model:predict', data =data, headers = headers)
#resp_acn = requests.post('http://localhost:9000/v1/models/acne_model:predict', data= data, headers = headers)

#Get the predictions
#predictions = json.load(json_response.text)

print(resp_wpr)
print(json.loads(resp_wpr.text))