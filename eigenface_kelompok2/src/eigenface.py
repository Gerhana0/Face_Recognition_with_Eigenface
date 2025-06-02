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
    matrixImage = [[0 for i in range(256)] for j in range(len(dir))]
    matrixImage = np.array(matrixImage)
    i = 0
    for image in dir:
        matrixImage[i] = imagetoVector(datasetPath + "/" + image)
        i += 1
    return matrixImage

def mean(imageVectors):
    meanVector = [[0 for i in range(imageVectors.shape[1])] for j in range(1)]
    for i in range(len(imageVectors)):
        temp = 1/len(imageVectors) * imageVectors[i]
        meanVector = np.add(temp, meanVector)
    return meanVector

def selisih(meanVector, imageVectors):
    diffMatrix = [[0 for i in range(imageVectors.shape[1])] for j in range(len(imageVectors))]
    for i in range(len(imageVectors)):
        diffMatrix[i] = np.subtract(imageVectors[i], meanVector[0])
    return np.array(diffMatrix)
    

def covariance(diffMatrix):
    covarianceMatrix = diffMatrix @ np.transpose(diffMatrix)
    covarianceMatrix = np.divide(covarianceMatrix, len(covarianceMatrix))
    return covarianceMatrix

def QR(matrix):
    # QR decomposition using Householder reflection
    # Source: https://rpubs.com/aaronsc32/qr-decomposition-householder

    (rowCount, colCount) = np.shape(matrix)

    # Initialize Q as matrix orthogonal and R as matrix upper triangular
    Q_matrix = np.identity(rowCount)
    R_matrix = np.copy(matrix)

    for j in range(0, rowCount - 1):
        x = np.copy(R_matrix[j:, j])
        x[0] += np.copysign(np.linalg.norm(x), x[0])

        v = x / np.linalg.norm(x)

        H = np.identity(rowCount)
        H[j:, j:] -= 2.0 * np.outer(v, v)

        Q_matrix = Q_matrix @ H
        R_matrix = H @ R_matrix
    return (Q_matrix, np.triu(R_matrix))

def eigQR(matrix):
    # Source: https://www.andreinc.net/2021/01/25/computing-eigenvalues-and-eigenvectors-using-qr-decomposition

    (rowCount, colCount) = np.shape(matrix)
    eigenVecs = np.identity(rowCount)
    for k in range(100):
        s = matrix.item(rowCount-1, colCount-1) * np.identity(rowCount)

        Q_matrix, R_matrix = QR(np.subtract(matrix, s))

        matrix = np.add(R_matrix @ Q_matrix, s)
        eigenVecs = eigenVecs @ Q_matrix
    return np.diag(matrix), eigenVecs

def eig(covMatrix):
    eigenVal , eigenVec = eigQR(covMatrix)
    # Grouping eigen pairs
    reducedeigenVec = np.array(eigenVec).transpose()
    # Forming eigenspace
    return reducedeigenVec

def projection(orgMatrix, reducedeigenVec):
    # Calc Eig Faces
    eigenfaceProjection = np.dot(orgMatrix.transpose(), reducedeigenVec)
    eigenfaceProjection = eigenfaceProjection.transpose()
    return eigenfaceProjection

def weightDataset(datasetProjection, diffMatrix):
    weightDataset = np.array([np.dot(datasetProjection, i) for i in diffMatrix])
    return weightDataset

def recogniseUnknownFace(pathDataset, pathTestImage, meanDatasetVector, projectionVec, weightDataset, threshold = 500000):
    # Get test face vector
    testFaceVector = cv2.imread(r"" + pathTestImage)
    testFaceVector = cv2.resize(testFaceVector, (16,16), interpolation= cv2.INTER_AREA)
    grayImage = cv2.cvtColor(testFaceVector, cv2.COLOR_BGR2GRAY)
    testFaceVector = grayImage.flatten()

    # Get test face normalised vector face
    testFacediffVector = np.subtract(testFaceVector, meanDatasetVector)

    # Calc test face weight
    weightTestFace = np.dot(testFacediffVector, projectionVec.transpose())
    
    # Calculate euclidean distance (in matrix form)
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