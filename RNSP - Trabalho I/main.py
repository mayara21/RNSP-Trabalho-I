import struct, time
from PyWANN import WiSARD
from array import array

startTime = time.time()

trainImagesFilename = "train-images.idx3-ubyte"
trainLabelsFilename = "train-labels.idx1-ubyte"
testImagesFilename = "t10k-images.idx3-ubyte"
testLabelsFilename = "t10k-labels.idx1-ubyte"

# a threshold to binarize the pixels data
threshold = 40


print("Started reading and preprocessing the data.")

with open(trainImagesFilename, mode = 'rb') as file:
	magicNumberTrainImages = struct.unpack('>i', file.read(4))[0]
	numberImagesTrainImages = struct.unpack('>i', file.read(4))[0]
	numberRowsTrainImages = struct.unpack('>i', file.read(4))[0]
	numberColumnsTrainImages = struct.unpack('>i', file.read(4))[0]

	trainImagesArray = array('B', file.read())
	trainImages = []
	for i in range(numberImagesTrainImages):
		trainImages.append([])
		trainImages[i][:] = trainImagesArray[i * numberRowsTrainImages * numberColumnsTrainImages: (i + 1) * numberRowsTrainImages * numberColumnsTrainImages]

		# using the threshold to binarize the data
		for j in range(numberRowsTrainImages * numberColumnsTrainImages):
			if trainImages[i][j] <= threshold:
				trainImages[i][j] = 0
			else:
				trainImages[i][j] = 1


with open(trainLabelsFilename, mode = 'rb') as file:
	magicNumberTrainLabels = struct.unpack('>i', file.read(4))[0]
	numberItemsTrainLabels = struct.unpack('>i', file.read(4))[0]

	trainLabelsArray = array('B', file.read())
	trainLabels = []
	for i in range(numberItemsTrainLabels):
		trainLabels.append([])
		trainLabels[i] = trainLabelsArray[i]


with open(testImagesFilename, mode = 'rb') as file:
	magicNumberTestImages = struct.unpack('>i', file.read(4))[0]
	numberImagesTestImages = struct.unpack('>i', file.read(4))[0]
	numberRowsTestImages = struct.unpack('>i', file.read(4))[0]
	numberColumnsTestImages = struct.unpack('>i', file.read(4))[0]

	testImagesArray = array('B', file.read())
	testImages = []
	for i in range(numberImagesTestImages):
		testImages.append([])
		testImages[i][:] = testImagesArray[i * numberRowsTestImages * numberColumnsTestImages: (i + 1) * numberRowsTestImages * numberColumnsTestImages]
		for j in range(numberRowsTestImages * numberColumnsTestImages):
			if testImages[i][j] <= threshold:
				testImages[i][j] = 0
			else:
				testImages[i][j] = 1

with open(testLabelsFilename, mode = 'rb') as file:
	magicNumberTestLabels = struct.unpack('>i', file.read(4))[0]
	numberItemsTestLabels = struct.unpack('>i', file.read(4))[0]

	testLabelsArray = array('B', file.read())
	testLabels = []
	for i in range(numberItemsTestLabels):
		testLabels.append([])
		testLabels[i] = testLabelsArray[i]

print("Finished reading and preprocessing data.")
# retinaLength = numberRowsTrainImages*numberColumnsTrainImages (in the updated PyWANN, passing this value as an argument to the WiSARD function isn't needed anymore)


probFile = open("predict_proba.txt", 'w') # file for the probabilities of the wrong classifications
difFile = open("difference_proba.txt", 'w') # file for the diference between the probability of the guessed class and the expected one

numberBits = 32
# bleaching = False
correct = 0
numberClasses = 10

wisard = WiSARD.WiSARD(numberBits)
# wisard = WiSARD.WiSARD(numberBits, bleaching)

print("Started training.")
wisard.fit(trainImages, trainLabels)
print("Finished training.")

print("Started classifying.")
result = wisard.predict(testImages)
print("Finished classifying.")

# gets the probabilitles for each class
prob = wisard.predict_proba(testImages)

# to count the frequency of wrong classifications in each expected class
wrongLabelsFreq = [0] * numberClasses

for i in range(numberItemsTestLabels):
	if result[i] == testLabels[i]:
		correct += 1
	else:
		probFile.write(str(testLabels[i]) + '-' + str(result[i]) + '\n')
		difFile.write(str(testLabels[i]) + '-' + str(result[i]) + '\n')

		for j in range(numberClasses):
			probFile.write(str(prob[i][j]) + ' ')
		probFile.write('\n')

		difFile.write(str(prob[i][result[i]]) + '-' + str(prob[i][testLabels[i]]) + ' = ' + str(prob[i][result[i]] - prob[i][testLabels[i]]) + '\n')

		wrongLabelsFreq[testLabels[i]] += 1

probFile.close()
difFile.close()

print(wrongLabelsFreq)

print(correct*100.0/numberItemsTestLabels) # accuracy

print("Time to execute: %s seconds" % (time.time() - startTime))
