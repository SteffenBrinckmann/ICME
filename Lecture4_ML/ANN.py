# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import tensorflow as tf

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
featureNames = columnNames[:-1]
labelName = columnNames[-1]
print("Features: {}".format(featureNames))
print("Label: {}".format(labelName))

trainDataset = tf.data.experimental.make_csv_dataset(
    fileName, batchSize, column_names=columnNames, 
    label_name=labelName, num_epochs=1)
features, labels = next(iter(trainDataset))

plt.scatter(features['pO2'], features['pCrO3'], c=labels)
plt.colorbar()
plt.xlabel("log-pressure $O_2$")
plt.ylabel("log-pressure $CrO_3$")
plt.show()
