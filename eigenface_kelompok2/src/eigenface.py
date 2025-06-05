import numpy as np
import os
import cv2

def imagetoVector(imagePath):
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (16,16), interpolation= cv2.INTER_AREA)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = grayImage.flatten() 
    return result

def vectortoMatrix(datasetPath):
    dir = os.listdir(datasetPath)
    matrixImage = np.array([
        imagetoVector(os.path.join(datasetPath, image)) for image in dir
    ])
    return matrixImage

def mean(imageVectors):
    meanVector = [[0 for _ in range(imageVectors.shape[1])]]
    for i in range(len(imageVectors)):
        temp = 1/len(imageVectors) * imageVectors[i]
        meanVector = np.add(temp, meanVector)
    return meanVector

def different(meanVector, imageVectors):
    diffMatrix = [[0 for _ in range(imageVectors.shape[1])] for _ in range(len(imageVectors))]
    for i in range(len(imageVectors)):
        diffMatrix[i] = np.subtract(imageVectors[i], meanVector[0])
    return np.array(diffMatrix)
    

def covariance(diffMatrix):
    covarianceMatrix = diffMatrix @ np.transpose(diffMatrix)
    covarianceMatrix = np.divide(covarianceMatrix, len(covarianceMatrix))
    return covarianceMatrix

def QR(matrix):
    (rowCount, colCount) = np.shape(matrix)
    Q_matrix = np.identity(rowCount)
    R_matrix = np.copy(matrix)

    for k in range(0, rowCount - 1):
        x = np.copy(R_matrix[k:, k])
        x[0] += np.copysign(np.linalg.norm(x), x[0])

        v = x / np.linalg.norm(x)

        H = np.identity(rowCount)
        H[k:, k:] -= 2.0 * np.outer(v, v)

        Q_matrix = Q_matrix @ H
        R_matrix = H @ R_matrix
    return (Q_matrix, np.triu(R_matrix))

def eigQR(matrix):
    (rowCount, colCount) = np.shape(matrix)
    eigenVecs = np.identity(rowCount)
    for _ in range(100):
        s = matrix.item(rowCount-1, colCount-1) * np.identity(rowCount)

        Q_matrix, R_matrix = QR(np.subtract(matrix, s))

        matrix = np.add(R_matrix @ Q_matrix, s)
        eigenVecs = eigenVecs @ Q_matrix
    return np.diag(matrix), eigenVecs

def eig(covMatrix):
    _, eigenVec = eigQR(covMatrix)
    return np.array(eigenVec).T
    

def projection(orgMatrix, reducedeigenVec):
    eigenfaceProjection = np.dot(orgMatrix.T , reducedeigenVec)
    return eigenfaceProjection.T

def weightDataset(datasetProjection, diffMatrix):
    return np.array([np.dot(datasetProjection, i) for i in diffMatrix])

def recogniseUnknownFace(pathDataset, pathTestImage, meanDatasetVector, projectionVec, weightDataset, threshold = 500000):
    testFaceVector = cv2.imread(pathTestImage)
    testFaceVector = cv2.resize(testFaceVector, (16,16), interpolation= cv2.INTER_AREA)
    grayImage = cv2.cvtColor(testFaceVector, cv2.COLOR_BGR2GRAY)
    testFaceVector = grayImage.flatten()

    testFacediffVector = np.subtract(testFaceVector, meanDatasetVector)

    weightTestFace = np.dot(testFacediffVector, projectionVec.T)
    
    euclideanMatrix = np.absolute(weightDataset - weightTestFace)
    euclideanDistance = np.linalg.norm(euclideanMatrix, axis=1)

    minDistance = min(euclideanDistance)
    maxDistance = max(euclideanDistance)

    if (minDistance < threshold):
        percentage = ((maxDistance - minDistance) / maxDistance) * 100
        minimumIndex = np.where(euclideanDistance == minDistance)[0]
        imageFiles = [os.path.join(pathDataset, p) for p in os.listdir(pathDataset)]
        return (imageFiles[int(minimumIndex)], percentage, minDistance)
    else:
        dummyImage = os.path.join(os.path.dirname(__file__), 'dummy', 'makasih.jpg')
        return (dummyImage, 0, minDistance)
        
