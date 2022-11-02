import tensorflow as tf
from tensorflow import keras
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Custom Loss1 (for example) 
@tf.function() 
def customLoss1(yTrue,yPred):
  return tf.reduce_mean(yTrue-yPred) 

# Custom Loss2 (for example) 
@tf.function() 
def customLoss2(yTrue, yPred):
  return tf.reduce_mean(tf.square(tf.subtract(yTrue,yPred))) 
  
def create_model():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),  
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
  model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', customLoss1, customLoss2])
  return model
  
# Create a basic model instance
model=create_model()

# Fit and evaluate model 
model.fit(x_train, y_train, epochs=5)
loss, acc,loss1, loss2 = model.evaluate(x_test, y_test,verbose=1)
print("Original model, accuracy: {:5.2f}%".format(100*acc))

# save the model in *.h5 format
model.save('./MyModel.h5',save_format='h5')

# load the saved model (Don't compile while loading as custom objects are not serialized)
loaded_model = tf.keras.models.load_model('./MyModel.h5', custom_objects={customLoss1:customLoss1, customLoss2:customLoss2},compile=False)

# compile the loaded model with the custome objects
loaded_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
              metrics=['accuracy', customLoss1, customLoss2])