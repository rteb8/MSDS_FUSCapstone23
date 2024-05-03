#do joblib load
import joblib

filename = 'fus_model.joblib'
loaded_model = joblib.load(filename)

# Write information about the loaded model to a file
with open('model_info.txt', 'w') as file:
    print(loaded_model, file=file)

with open('model_info.txt', 'a') as file:
    # Append model attributes to the file
    model_attributes = dir(loaded_model)
    print(model_attributes, file=file)
