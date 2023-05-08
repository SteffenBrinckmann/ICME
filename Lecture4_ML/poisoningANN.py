import matplotlib.pyplot as plt
import tensorflow as tf

#All parameters in a nice format
fileName = "poisLinSVC.csv"
columnNames = ['pO2', 'pCrO3', 'component']
batchSize = 32
numEpochs = 200
numNodesLayer1 = 5
numNodesLayer2 = 5
numOutput = 5
numInput = 2
learningRate = 0.01

#GET DATA INTO SUITABLE FORMAT
# specify column names etc of CSV file
featureNames = columnNames[:-1]
labelName = columnNames[-1]
print("Features: {}".format(featureNames))
print("Label: {}".format(labelName))

trainDataset = tf.data.experimental.make_csv_dataset(
    fileName, 
    batchSize, 
    column_names=columnNames, 
    label_name=labelName, 
    num_epochs=1)
features, labels = next(iter(trainDataset))
#data is stored in list of columns
#print(features)

#plot for checking
#values are shuffled during reading, result in different each time
plt.scatter(features['pO2'], features['pCrO3'], c=labels, cmap='viridis')
plt.xlabel("log-pressure $O_2$")
plt.ylabel("log-pressure $CrO_3$")
plt.show()

#repackage data into matrix
def repackVectorsTensor(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels
trainDataset = trainDataset.map(repackVectorsTensor)
features, labels = next(iter(trainDataset))
#print("After repacking",features)

## BUILD MODEL AND DO SIMPLE TESTS
#build model in one go
model = tf.keras.Sequential([
  tf.keras.layers.Dense(numNodesLayer1, activation=tf.nn.relu, input_shape=(numInput,)),
  tf.keras.layers.Dense(numNodesLayer2, activation=tf.nn.relu),
  tf.keras.layers.Dense(numOutput)
])

#get simple predictions: first 5
predictions = model(features)
print("\nTest predictions:",predictions[:5])
#convert into softmax: first 5
print("Test predictions:",tf.nn.softmax(predictions[:5]))

#convert all into labels and compare to real ones
print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))


## TRAIN MODEL
#function that describes the error: should be minimized
lossObject = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def loss(model, x, y, training):
  y_ = model(x, training=training)
  return lossObject(y_true=y, y_pred=y_)
l = loss(model, features, labels, training=False)
print("\nLoss test: {}".format(l))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    lossValue = loss(model, inputs, targets, training=True)
  return lossValue, tape.gradient(lossValue, model.trainable_variables)
#statistical gradient descent
optimizer = tf.keras.optimizers.SGD(learning_rate=learningRate)
#1 single optimization step
lossValue, grads = grad(model, features, labels)
print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          lossValue.numpy()))
optimizer.apply_gradients(zip(grads, model.trainable_variables))
print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                          loss(model, features, labels, training=True).numpy()))


## Note: Rerunning this cell uses the same model variables
# Keep results for plotting
train_loss_results = []
train_accuracy_results = []
for epoch in range(numEpochs):
  epochLossAvg = tf.keras.metrics.Mean()
  epochAccuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # Training loop - using batches of 32
  for x, y in trainDataset:
    #x.shape: 32,4
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # Track progress
    epochLossAvg.update_state(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    epochAccuracy.update_state(y, model(x, training=True))
  # End epoch
  train_loss_results.append(epochLossAvg.result())
  train_accuracy_results.append(epochAccuracy.result())
  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.0%}".format(epoch,
                                                                epochLossAvg.result(),
                                                                epochAccuracy.result()))

#Visualizierung
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')
axes[0].set_ylabel("Loss", fontsize=14)
# axes[0].set_ylim(bottom=0)
axes[0].plot(train_loss_results)
axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_ylim(top=1)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.subplots_adjust(hspace=0)
plt.show()

