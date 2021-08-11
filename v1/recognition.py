import os
import segmentation
from sklearn.externals import joblib
'''
car plate pattern
'''
current_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(current_dir, 'models/svc/svc.pkl')
model = joblib.load(model_dir)

classification_result = []

for each_char in segmentation.characters:
    each_char = each_char.reshape(1, -1)
    result = model.predict(each_char)
    classification_result.append(result)

print classification_result

plate_string = ''

for each_predict in classification_result:
    plate_string += each_predict[0]

print(plate_string)
