import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten

X_train = np.loadtxt('input.csv',delimiter=',')
Y_train = np.loadtxt('labels.csv',delimiter=',')

X_test = np.loadtxt('input_test.csv',delimiter=',')
Y_test = np.loadtxt('labels_test.csv',delimiter=',')

X_train = X_train.reshape(len(X_train),100,100,3)
Y_train = Y_train.reshape(len(Y_train),1)

X_test = X_test.reshape(len(X_test),100,100,3)
Y_test = Y_test.reshape(len(Y_test),1)

X_train = X_train/255
X_test = X_test/255

print("Shape of X-train",X_train.shape)
print("Shape of Y-train",Y_train.shape)
print("Shape of X-test",X_test.shape)
print("Shape of Y-test",Y_test.shape)

model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
          
model.add(Flatten())
model.add(Dense(64,activation = 'relu'))
model.add(Dense(1,activation='sigmoid'))

model.fit(X_train, Y_train, epochs=25, validation_split = 0.2)
model.fit(X_train, Y_train, epochs=5, batch_size=64)

model.evaluate(X_test,Y_test)

idx2 = random.randint(0, len(Y_test) - 1)  # Adjust the range to prevent index out of bounds
plt.imshow(X_test[idx2, :])
plt.show()

y_pred = model.predict(X_test[idx2, :].reshape(1, 100, 100, 3))
print(y_pred)
if(y_pred > 0.8):
    print("Our model says its a cat")
else:
    print("Our model says its a dog")