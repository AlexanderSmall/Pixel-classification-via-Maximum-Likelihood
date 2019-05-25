import cv2
import numpy as np
#import matplotlib.pyplot as plt
import csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from random import *
import math

def importImage(filepath):
    return cv2.imread(filepath, 1)

def MorphologicalOperations2(im):

    alteredImage = np.zeros(shape=(211, 356, 4))
    kernel = np.ones((2, 2), np.uint8)

    kernelA = np.array([[1, -2, 1],
                        [-2, 1, -2],
                        [1, -2, 1]], np.uint8)

    kernel2 = np.array([[1, 0, 1],
                        [0, 1, 0],
                        [1, 0, 1]], np.uint8)

    kernal2Compliment = np.array([[1, 1, 1],
                                  [1, 0, 1],
                                  [1, 1, 1]], np.uint8)

    #print(im[:, :, 0])

    # Buildings
    imOP = im[:, :, 0].astype(np.uint8)
    imOP = cv2.erode(imOP, kernel, iterations=3)
    imOP = cv2.dilate(imOP, kernel, iterations=5)
    alteredImage[:, :, 3] = imOP

    # Vegetation
    imOP2 = im[:, :, 1].astype(np.uint8)
    imOP2 = cv2.erode(imOP2, kernel, iterations=1)
    imOP2 = cv2.dilate(imOP2, kernel, iterations=2)
    alteredImage[:, :, 2] = imOP2

    # Cars
    imOP3 = im[:, :, 2].astype(np.uint8)
    imOP3 = cv2.erode(imOP3, kernel, iterations=1)
    imOP3 = cv2.dilate(imOP3, kernel, iterations=2)
    alteredImage[:, :, 1] = imOP3

    # Ground

    imOP4 = im[:, :, 3].astype(np.uint8)
    imOP4 = cv2.erode(imOP4, kernel, iterations=1)
    imOP4 = cv2.dilate(imOP4, kernel, iterations=2)
    alteredImage[:, :, 0] = imOP4


    cv2.imshow('buildings', imOP)
    cv2.imwrite('buildings.png', imOP)
    cv2.imshow('vegetation', imOP2)
    cv2.imwrite('vegetation.png', imOP2)
    cv2.imshow('cars', imOP3)
    cv2.imwrite('cars.png', imOP3)
    cv2.imshow('ground', imOP4)
    cv2.imwrite('ground.png', imOP4)
    #cv2.imshow('Dilation', img_dilation)
    #cv2.waitKey(0)
    return alteredImage

def combineImage(alteredImage):
    #G/V/C/B - print order
    finalImage = np.zeros(shape=(211, 356, 3))
    finalImage.fill(0)
    # GreyGBR color
    colourkey = {4: [0, 0, 255], 3: [0, 255, 0], 2: [255, 0, 0], 1: [128, 128, 128]}  # Colour key
    for x in range(0, 356):
        for y in range(0, 211):
            blackCount = 0
            for classNum in range(0, 4):
                if (alteredImage[y, x, classNum] != 0):
                    finalImage[y, x, :] = colourkey.get(classNum + 1)
                else:
                    blackCount += 1

            if blackCount == 4:
                finalImage[y, x, :] = colorkey.get(int(secondProbValues[y, x]) + 1)

    finalImage = finalImage.astype(np.uint8)
    return finalImage
    #cv2.imshow('Final Image', finalImage)
    #cv2.waitKey(0)

def calculateCorrectPercentage(im, im2):
    colorMatch = 0
    correct = 0
    total = 0
    for x in range(0, 356):
        for y in range(0, 211):
            #print("im: " + str(im[y, x]) + "im2: " + str(im2[y, x]))
            #if (im[y, x, :] == im2[y, x, :]):
            colorMatch = 0
            for i in range(0, 3):
                if (im[y, x, i] == im2[y, x, i]):
                    colorMatch += 1

            if (colorMatch == 3):
                correct += 1
            total += 1

    error = correct/total
    print("After morphological operations:")
    print("% correct: " + str(error))


def createDifferenceImages(predicted):
    allDifferenceImage = np.zeros(shape=(211, 356, 0))
    differenceImage = np.zeros(shape=(211, 356))
    for i in range(0, 4):
        for x in range(0, 356):
            for y in range(0, 211):
                if predicted[y, x] == i + 1:
                    differenceImage[y, x] = 255
                else:
                    differenceImage[y, x] = 0

        #print(differenceImage)
        #print("____")
        allDifferenceImage = np.dstack((allDifferenceImage, differenceImage))
        printImage = differenceImage.astype(np.uint8)
        #cv2.imshow("difference image", printImage)
        #cv2.waitKey(0)

    return allDifferenceImage

def calculateConfutionImage(prediction, groundTruth) :
    # CLASSIFICATIONS:
    # buildings (RED), vegetation (GREEN), car (BLUE), ground (GREY)
    # MISS-CLASSIFICATIONS:
    # building-vegetation (YELLOW), building-car (MAGENTA), building-ground (pink)
    # vegetation-building (light yellow), vegetation-car (CYAN), vegetation-ground (light green)
    # car-building (light magenta), car-vegetation (light cyan), car-ground (lavender)
    # ground-building (light pink), ground-vegetation (very light green), ground-car (dark lavender)
    colorkeyConfusion = {'11': [0, 0, 255], '12': [0, 255, 255], '13': [0, 255, 0], '14': [255, 0, 0],
                         '21': [0, 255, 255], '22': [0, 255, 0], '23': [255, 255, 0], '24': [128, 255, 128],
                         '31': [128, 255, 255], '32': [255, 255, 128], '33': [255, 0, 0], '34': [255, 128, 128],
                         '41': [180, 255, 255], '42': [70, 255, 70], '43': [255, 70, 70], '44': [128, 128, 128]
                        }  # Colour key

    confusionImage = np.zeros(shape=(211, 356, 3))
    for x in range(0, prediction.shape[1]):
        for y in range(0, prediction.shape[0]):
            key = str(int(prediction[y, x])) + str(int(groundTruth[y, x]))
            colorVector = colorkeyConfusion.get(key)   # the predicted pixel colour
            for dimention in range(0, 3):
                confusionImage[y, x, dimention] = colorVector[dimention]
    # print image
    confusionImage = confusionImage.astype(np.uint8)
    cv2.imshow("Confusion image", confusionImage)
    cv2.imwrite('confusionImage.png', confusionImage)
    #cv2.waitKey(0)

#def createDifferenceImages(predicted):
#    allDifferenceImage = np.zeros(shape=(211, 356))
#    differenceImage = np.zeros(shape=(211, 356))
#    for i in range(0, 4):
#        for x in range(0, 356):
#            for y in range(0, 211):
#                if predicted[y, x] == i + 1:
#                    differenceImage[y, x] = 255
#                else:
#                    differenceImage[y, x] = 0
#
#        print(differenceImage)
#        print("____")
#        allDifferenceImage = np.dstack((allDifferenceImage, differenceImage))
#        printImage = differenceImage[i].astype(np.uint8)
#        cv2.imshow("difference image", printImage)
#        cv2.waitKey(0)
#    return allDifferenceImage



def printGroundTruth(ground_truth) :
    groundTruthImage = np.zeros(shape=(211, 356, 3))
    colourkey = {1: [0, 0, 255], 2: [0, 255, 0], 3: [255, 0, 0], 4: [128, 128, 128]}  # Colour key
    for x in range(0, 356):
        for y in range(0, 211):
            for i in range(0, 3):
                groundTruthImage[y, x, i] = colourkey.get(ground_truth[y, x])[i]
    groundTruthImage = groundTruthImage.astype(np.uint8)

def importImage(filepath):
    return cv2.imread(filepath, 0)

def printPredictedClassValues(cv):
    f2 = open("cv.csv", "w")
    f2.write("")
    f2.close()

    f = open("cv.csv", "a")
    for y in range(211):
        for x in range(356):
            string = str(cv[y, x]) + ","
            f.write(string)
        f.write("\n")


def MorphologicalOperations(im):
    kernel = np.ones((5, 5), np.unit8)

    img_erosion = cv2.erode(im, kernel, iterations=1)
    img_dilation = cv2.dilate(im, kernel, iterations=1)
    cv2.imshow('Input', )
    cv2.imshow('Erosion', img_erosion)
    cv2.imshow('Dilation', img_dilation)

# load in RGB NIR and LIDAR components and place into one nparray
allLayers = np.zeros(shape=(211, 356, 0))
for i in range(1, 7):
    name = str(i) + '.bmp'
    layer = cv2.imread(name, 0)
    allLayers = np.dstack((allLayers, layer))

# converted ground truths to a csv file
# read in ground truths
ground_truth = np.loadtxt(open("ground_truth.csv", "rb"), delimiter=',', skiprows = 0)

allSamples = [30, 30, 30, 30]   # ability to change the number of samples per class
numSamples = 30    # number of samples per class - make sure numSamples is the same size as the max element of allSamples
numClasses = 4      # number of classes
numDimensions = 6   # number of dimensions per vector

XYbuffer = {} # used to keep track of which pixels have been randomly selected

classWeights = [1, 1, 1, 1]   # weights alter class condition pdf

## SELECT RANDOM DISTROBUTION OF SAMPLE PIXELS FROM THE PROVIDED ground_truth.csv
means = []  # holds the vector of means of each class
covs = []   # holds the covariance matrix of each class
testSamples = np.zeros(shape=(numClasses, numSamples, numDimensions), dtype = object) # holds tuples of positions for each class
# classType is the number corrosponding to the class
for classType in range(1, numClasses + 1):
    testCount = 0
    while (testCount < allSamples[classType - 1]) :   # loop until enough samples have been collected

        x = randint(0, len(ground_truth[0]) - 1)    # generate random position tuple
        y = randint(0, len(ground_truth) - 1)

        if (x, y) not in XYbuffer :
            if ground_truth[y, x] == classType :    #OpenCV uses (x, y) Numpy uses(y , x)
                for dimention in range(0, numDimensions):
                    testSamples[classType - 1, testCount, dimention] = allLayers[y, x, dimention]  # add tuple position to the test samples

                XYbuffer = {(x, y), 0}  # add tuple to buffer to ensure no reselection of pixels
                testCount += 1          # increase number of samples taken

    ## Calculate the mean and covariance of each class samples
    mean = 0
    for numTests in range(0, allSamples[classType - 1]):
        mean += testSamples[classType - 1, numTests, :]
    means.append(mean / allSamples[classType - 1])

    ## calculate cov matrix
    covs.append(np.cov(testSamples[classType - 1, :, :].astype(float), rowvar=False))

# print covariance matrix for each class
#print("COVARIANCE MATRICES")
#for i in range(0, 4):
#    print("Class [" + str(i + 1) + "]")
#    print(np.int_(covs[i]))
#    print("\n")


## CLASSIFICATION
# Calculate the class conditional pdf for every pixel in the image
# Select highest probability and place corrosponding class colour in the x, y positions of the classified output image
finalImg = np.zeros(shape=(211, 356, 3))    # final image - holds RGB representation of  classified pixels

secondBest = 0
secondProbValues = np.zeros(shape=(211, 356))
maxProbValues = np.zeros(shape=(211, 356))    # holds classified pixels in integer form
colorkey = {1: [0, 0, 255], 2: [0, 255, 0], 3: [255, 0, 0], 4: [128, 128, 128]} # Colour key
classPredictions = np.zeros(shape=(211, 356, 4))
for x1 in range(0, 356):
    for y1 in range(0, 211):
        classMax = 0;
        currentMax = 0;
        for classCount in range(0, numClasses):
            x = allLayers[y1, x1, :]  # 6x1 feature vector

            # calculate the normal
            n = 1.0 / ((2 * math.pi ** (3)) * (math.sqrt(np.linalg.det(covs[classCount]))))

            # a convertion from lists to np matrix - (np.dot does not work without this conversion - not sure why)
            cov = np.zeros(shape=(6,6))
            for i in range(0, 6):
                for j in range(0, 6):
                    cov[j, i] = covs[classCount][j, i]

            # calculate the exponetional
            pixelVariance = (np.transpose(x - means[classCount]))   # pixel variance - pixel (feature vector of the image) minus the vector of means
            # calculate the inverse of the covariance matrix to divide by variance
            a = np.dot(np.linalg.inv(cov), pixelVariance)   # in order to square the variance transpose matrix and preform the dot product
            # dot product with non-transposed feature vector to square the vectors values
            exponent = -0.5 * np.dot(a, (x - means[classCount]))    # calculate final exponent (multiply by -0.5 in accordance with normal distrobution)

            pdf = n * math.exp(exponent)    # the normal multipled by euler's number to the power of
            if (pdf > currentMax * classWeights[classCount]):
                secondBest = classMax   # second highest probability
                currentMax = pdf * classWeights[classCount]
                classMax = classCount + 1 # classMax holds the class integer with the highest pdf.


        colour = colorkey.get(classMax)  # get the corrosponding colour
        maxProbValues[y1, x1] = classMax    # class of max value - used for confusion matrix and visualising difference image
        secondProbValues[y1, x1] = secondBest
        for i in range(0, 3):  # add the RGB colour to the pixels
            finalImg[y1, x1, i] = colour[i]

# construct confusion matrix
a = ground_truth.flatten()
b = maxProbValues.flatten()
cm = confusion_matrix(a, b)

# print confusion matrix
print("CONFUSION MATRIX:")
print("\n")
print("B | V | C | G")
print(cm)

# calculate the overall accuracy
total = np.sum(cm)
correct = cm[0, 0] + cm[1, 1] + cm[2, 2] + cm[3, 3]
error = correct/total
print("Total error: " + str(error))

# calculate user error for each class
print("\n")
print("[1] - buildings (RED), [2] - vegetation (GREEN), [3] - car (BLUE), [4] - ground (GREY)")
count = 0
for i in range(0, 4):
    total = 0
    for j in range(0, 4):
        total += cm[j, i]
    correct = cm[i, i]
    error = correct/total
    print("User error for class [" + str(i + 1) + "] error: " + str(error))

print("\n")

# calculate producer error
count = 0
for i in range(0, 4):
    total = 0
    for j in range(0, 4):
        total += cm[i, j]
    correct = cm[i, i]
    error = correct / total
    print("Producer error for class [" + str(i + 1) + "] error: " + str(error))


# calculate Cohen's Kappa score
print("\n")
print("Cohen's Kappa score: " + str(cohen_kappa_score(a, b))) # (a , b) flat matrix of predicted and ground truth respectively

# print the final image
finalImg = finalImg.astype(np.uint8)
cv2.imshow("Predictions", finalImg)
cv2.imwrite('prediction.png', finalImg)

# in real world example you would not usually aquire so much ground truth values
# however, we can be use it to visualise the difference between the predicted values from n (20) sample

#printGroundTruth(ground_truth)
#createDifferenceImages(maxProbValues, ground_truth)
calculateConfutionImage(maxProbValues, ground_truth)
printPredictedClassValues(maxProbValues)

images = createDifferenceImages(maxProbValues)
alteredImage = MorphologicalOperations2(images)
finalImage = combineImage(alteredImage)

cv2.imshow('Predicted + Morphological operations', finalImage)
cv2.imwrite('predictionModified.png', finalImage)

groundTruth = cv2.imread('ground_truth.bmp', 1)
printGroundTruth(ground_truth)

calculateCorrectPercentage(finalImage, groundTruth)
cv2.waitKey(0)


