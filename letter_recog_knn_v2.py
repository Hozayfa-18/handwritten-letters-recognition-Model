import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from joblib import load, dump

alphabet=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 
          'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

train_data = pd.read_csv("./emnist_letters/emnist-letters-train.csv", header = None)
test_data = pd.read_csv("./emnist_letters/emnist-letters-test.csv", header = None)
mapping = pd.read_csv("./emnist_letters/emnist-letters-mapping.txt", sep = ' ', header = None)

train_data=train_data.rename(columns={0:'label'})
test_data=test_data.rename(columns={0:'label'})

num_train,num_validation = int(len(train_data)*0.8),int(len(train_data)*0.2)

x_train,y_train=train_data.iloc[:num_train,1:].values,train_data.iloc[:num_train,0].values
x_validation,y_validation=train_data.iloc[num_train:,1:].values,train_data.iloc[num_train:,0].values

index=1

print("Label: " + str(y_train[index]) + ' as ' + alphabet[(y_train[index])-1])

#image fixing
image= x_train[index]
image = image.reshape([28, 28])
plt.imshow(image)
plt.show()
image = np.fliplr(image)
image = np.rot90(image)
print(image)
plt.imshow(image)
plt.show()

model=RandomForestClassifier()
model.fit(x_train,y_train)

# prediction_validation = model.predict(x_validation)
# print("Validation Accuracy: " + str(accuracy_score(y_validation,prediction_validation)))


# index=75
# print("Predicted " + str(y_validation[y_validation==prediction_validation][index]) + " as " + 
#     alphabet[(y_validation[y_validation==prediction_validation][index])-1])

# #image fixing
# image= x_validation[y_validation==prediction_validation][index]
# image = image.reshape([28, 28])
# image = np.fliplr(image)
# image = np.rot90(image)
# plt.imshow(image)
# plt.show()

# print("Validation Confusion Matrix: \n" + str(confusion_matrix(y_validation,prediction_validation)))

# matrix = confusion_matrix(y_validation,prediction_validation)
# fig, ax = plt.subplots(figsize=(7.5, 7.5))
# ax.matshow(matrix, cmap=plt.cm.Blues, alpha=0.3)
# for i in range(matrix.shape[0]):
#     for j in range(matrix.shape[1]):
#         ax.text(x=j, y=i,s=matrix[i, j], va='center', ha='center', size='larger')
# plt.xlabel('Predictions', fontsize=18)
# plt.ylabel('Actuals', fontsize=18)
# plt.title('Confusion Matrix', fontsize=18)
# plt.show()

y_test = test_data['label']
x_test=test_data.drop("label", axis=1)

prediction_test = model.predict(x_test)
print("TEST Accuracy: " + str(accuracy_score(y_test,prediction_test)))

index = 10000
print("Predicted " + alphabet[(prediction_test[index])-1])

#image fixing
image= x_test.iloc[index].values
image = image.reshape([28, 28])
image = np.fliplr(image)
image = np.rot90(image)
plt.imshow(image)
plt.show()

print("Test Confusion Matrix: \n" + str(confusion_matrix(y_test,prediction_test)))

matrix = confusion_matrix(y_test,prediction_test)
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(x=j, y=i,s=matrix[i, j], va='center', ha='center', size='larger')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

# dump(model,'./KNN_LETTER_MODEL_v2.joblib')


# model = load('./KNN_LETTER_MODEL.joblib')


# import PIL

# file = r"C:\Users\hozay\OneDrive\Desktop\HW_recognition\letters\letter5.png"
# original_img = PIL.Image.open(file)
# img_resized = original_img.resize((28, 28), PIL.Image.Resampling.LANCZOS)
# img_resized = np.array(img_resized)
# img_resized = img_resized[:,:,0]
# img_resized = np.invert(np.array([img_resized]))
# img_resized = img_resized[0,:,:]
# print(img_resized)
# white_threshold = 10
# mask = img_resized > white_threshold
# img_resized[mask] = 255
# img_resized[~mask] = 0
# print(img_resized)
# plt.imshow(img_resized, cmap=plt.cm.binary)
# plt.show()

# img_pre = np.array(img_resized)
# print(img_pre.shape)
# img_pre = np.fliplr(img_pre)
# img_pre = np.rot90(img_pre)
# plt.imshow(img_pre, cmap=plt.cm.binary)
# plt.show()
# img_pre = img_pre.reshape(784)
# prediction = model.predict([img_pre])
# print("the Digit is:",alphabet[prediction[0]])

# plt.imshow(original_img, cmap=plt.cm.binary)
# plt.show()