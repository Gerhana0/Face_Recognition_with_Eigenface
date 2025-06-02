import eigenface

def run(datasetPath, testImagePath, threshold = 500000):

    imageMatrix = eigenface.vectortoMatrix(datasetPath)
    
    meanVector = eigenface.mean(imageMatrix)

    diffMatrix = eigenface.selisih(meanVector, imageMatrix)

    covMatrix = eigenface.covariance(diffMatrix)

    eigenVectors = eigenface.eig(covMatrix)

    projectedMatrix = eigenface.projection(imageMatrix, eigenVectors)

    datasetWights = eigenface.weightDataset(projectedMatrix, diffMatrix)

    matchedPath, matchPercentage, minDistance = eigenface.recogniseUnknownFace(
        datasetPath, testImagePath, meanVector, projectedMatrix, datasetWights, threshold
    )

    return (matchedPath, matchPercentage, minDistance)