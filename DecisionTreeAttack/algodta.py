from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

digits = load_digits()
xtrain, xtest, ytrain, ytest = train_test_split(digits.data, digits.target, train_size = .8)

dtc = DecisionTreeClassifier().fit(xtrain, ytrain)

tree = dtc.tree_


# params
alpha = 0.1



def getPredictionNodes(predictionPath):
    nodesNumber = predictionPath.shape[1]
    listPredictionNodes = [i for i in range(nodesNumber) if predictionPath[0,i] != 0]
    return listPredictionNodes

def getAdversialNodes(predictionPath):
    # renvoie le chemin adverse
    listPredictionNodes = getPredictionNodes(predictionPath)
    predictedClass = np.argmax(tree.value[listPredictionNodes[-1]])
    currentClass = predictedClass
    currentAdversialNodes = np.copy(listPredictionNodes)
    currentDepth = len(listPredictionNodes) - 1

    while predictedClass == currentClass and currentDepth>=0:
        currentNode = listPredictionNodes[currentDepth]
        otherNode = getOtherNode(currentNode, currentDepth, listPredictionNodes)
        pathsUnderNodes = exploreUnderNode(otherNode)
        for path in pathsUnderNodes:
            if np.argmax(tree.value[path[-1]]) != currentClass:
                return listPredictionNodes[:currentDepth] + path
        currentDepth -= 1

    raise Exception("adversial class not found")
    
def getOtherNode(currentNode, currentDepth, listPredictionNodes):
    # renvoie le frere de currentNode
    parent = listPredictionNodes[currentDepth-1]
    if tree.children_left[parent] == currentNode:
        return tree.children_right[parent]
    else:
        return tree.children_left[parent]

def exploreUnderNode(node):
    # renvoie tous les chemins partant de node
    if node == -1:
        return []

    if tree.children_left[node] == -1:
        return [[node]]

    leftPaths = exploreUnderNode(tree.children_left[node])
    rightPaths = exploreUnderNode(tree.children_right[node])

    allPaths = []
    for path in leftPaths + rightPaths:
       allPaths.append([node] + path)

    return allPaths

def getChanges(adversialNodes, predictionPath):
    # renvoie la liste des changements à faire sous la forme
    # [[numero de feature, valeur a mettre] ...]
    listPredictionNodes = getPredictionNodes(predictionPath)
    changes = []

    for i in range(len(adversialNodes)):
        if i >= len(listPredictionNodes) or adversialNodes[i] != listPredictionNodes[i]:
            parent = adversialNodes[i-1]

            if tree.children_left[parent] == adversialNodes[i]:
                newValue = tree.threshold[parent] - alpha
            else:
                newValue = tree.threshold[parent] + alpha

            change = [tree.feature[parent], newValue]
            changes.append(change)

    return changes

def getSuccessInfo(aSample, X, model):
    y = model.predict(X)
    yadversial = model.predict(aSample)
    nSuccesses = len(y) - sum(y==yadversial)
    s = f"Nombre de réussites : {nSuccesses} sur {len(y)}\n {nSuccesses/len(y)*100}% de réussite"
    return s

    
# génération du sample adverse
def generate(X, model):
    xtest = X.copy()
    aSample = []
    for i in range(len(xtest)):
        print(f"Element numéro {i+1} sur {len(xtest)}", end = "\r")
        x = xtest[i]
        predictionPath = dtc.decision_path([x]) 
       
        # récupérer le chemin adverse
        adversialNodes = getAdversialNodes(predictionPath)

        # récupérer les changements
        changes = getChanges(adversialNodes, predictionPath)
        # modifier l'échantillon
        for change in changes:
            x[change[0]] = change[1]
        
        aSample.append(x)

    print(getSuccessInfo(aSample, X, model))
    
    return aSample

aSample = generate(xtest, dtc)
