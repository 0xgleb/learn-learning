import tensorflow as tf
print(tf.__version__)

# assign mnist datasets to a variable
mnist = tf.keras.datasets.mnist

# load mnist training and testing data
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# normalize the inputs
training_images = training_images / 255.0
test_images = test_images / 255.0

# define the model
# 10 classes, hence 10 neurons in the output layer
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(), # convert 2 dimensional images into one dimensional arrays
    tf.keras.layers.Dense(1024, activation=tf.nn.relu), # layer with 1024 neurons and relu activation function
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) # 10 neurons and softmax activation
])

# assign hyperparameters
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy'
)

# define a callback class
class EpochEndCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss') < 0.4):
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True

# instantiate the callback class
callback = EpochEndCallback()

# train the model
model.fit(
    training_images,
    training_labels,
    epochs=5,
    callbacks = [callback]
)

# evaluate the model's accuracy using test data
print("\nTest results:\n")
model.evaluate(test_images, test_labels)

# create predictions for the test dataset
classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])
