import csv
import sys
import numpy as np
#from numpy.linalg import norm #To help calculate one of the stop conditions
import pandas as pd
import math
import random
from timeit import default_timer as timer
import time 
import os
#import json 

# Hyperopt imports, for hyperparameter optimization
from hyperopt import STATUS_OK 
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin

from functools import partial



## TREE LOGIC
class Node(object):
    def __init__(self, data, decisionColumn=None, depth=0, decision=-1):
        self.data = data # A dataframe with all of the nodes currently
        self.decisionColumn = decisionColumn
        self.children = {} # Dictionary to keep track of routes
        self.depth = depth # What depth this node is at
        self.decision = decision #0 or 1 for leaf nodes, -1 for non-leaf
        if (decision == -1):
            self.set_decision_by_average()

    def add_child(self, obj, index):
        self.children[index] = obj
        self.children[index].depth = self.depth + 1
        self.decision = -1
    
    # Set the decision by computing the average score
    def set_decision_by_average(self):
        # Loop through dataframe
        tempFrame = self.data[["decision"]]
        total = 0
        success = 0
        for index,row in tempFrame.iterrows():
            if(row.get(key = "decision") == 1):
                success = success + 1
            total = total + 1
        avg = success/total
        if (avg >= 0.5):
            self.decision = 1
        else:
            self.decision = 0
            
    def add_children_by_col(self, columnName):
        self.decisionColumn = columnName
        nodeObjects = [] # For reference and append it
        numNodes = 0
#        print("Decision column: " + str(columnName))
#        print("Data in column name")
#        print(self.data[[columnName]])
        uniques = self.data[[columnName]].value_counts().iteritems()
        for val, cnt in uniques:
#            print("Value: " + str(val))
#            print("Count: " + str(cnt))
            # Create a new node object 
            newData = self.data[self.data[columnName] == val[0]]
#            print("New data: ")
#            print(newData)
            newData = newData.drop(columns = [columnName])
#            print("New data after dropping: ")
#            print(newData)
            nodeObjects.append(Node(newData))
            self.add_child(nodeObjects[numNodes], val[0])
#            self.data = self.data.drop(nodeObjects[numNodes].index) # AFAIK this should end up dropping literally everything, but that doesn't make a difference
            numNodes = numNodes + 1
        self.data = None
        return nodeObjects
    
    def nodeName(self):
        if (self.decisionColumn is None):
            return "Result: " + str(self.decision)
        else:
            return str(self.decisionColumn)
        
    def printTree(self, pathTaken = None, toFile=sys.stdout):
        # Print all nodes below yourself
        if (pathTaken is None):
            print('\t' * self.depth + self.nodeName(), file=toFile)
        else:
            print('\t' * self.depth + "-" + str(pathTaken) + "-" + self.nodeName(), file=toFile)
        for child in self.children:
            self.children[child].printTree(child, toFile=toFile) # Passess in the path taken
            
    
            
## END TREE LOGIC

# Compute Gini index of a given set X
def predict_DT(sample, top):
    # Loop through attributes
    currentNode = top
    while currentNode.decision == -1:
        # Move via decision attribute
        try: # Use try block to handle case that it's not in the tree
            currentNode = currentNode.children[sample[currentNode.decisionColumn]]
        except:
#            print("Column " + currentNode.decisionColumn + " has no value corresponding to sample value of " + str(sample[currentNode.decisionColumn]))
#            print("Children: ")
#            print(currentNode.children)
            return 0
    return currentNode.decision
     
def predict_Bagging(sample, bag):
    # Loop through the bag and get results
    results = 0
    for tree in bag:
        results = results + predict_DT(sample, tree)
        
    average = results / len(bag)
    if (average >= 0.5):
        return 1
    else:
        return 0
    
def gini(X):
    eSum = 0
    uniques = X.value_counts().iteritems()
    for val, cnt in uniques:
        eSum += (cnt / len(X.index))**2
    return 1 - eSum
        
# Compute gini gain
def gini_gain(S, A, aColName):
    # Get our decision set
    decisionSet = S[["decision"]]
    giniS = gini(decisionSet)
    # A is the potential column to split
    # For every value it can take on, compute the gini index
    eSum = 0
    uniques = A.value_counts().iteritems()
    for val, cnt in uniques:
        decisionSubset = S[S[aColName] == val]
        eSum += ((cnt/len(S.index)) * gini(decisionSubset[["decision"]]))
    return giniS - eSum

def trainDecisionTree(training_set, test_set, depthLimit, exampleLimit):
#    depthLimit = 8
#    exampleLimit = 50
    
    # Grow using training set
    # Start with a node with everything in it
    top = Node(training_set)
    # Check first requirement before going through the iterations
    if (len(top.data.index) >= exampleLimit):
        # Now we loop through
        dfsQueue = [] # The next level to grow so to speak
        # We need a set of new nodes; compute GINI gain to determine which one
        maxGiniGain = -999999
        giniColumn = "invalid"
        for columnName in top.data:
            if columnName == "decision":
                continue
            curGain = gini_gain(top.data, top.data[columnName], columnName)
            if (curGain > maxGiniGain):
                maxGiniGain = curGain
                giniColumn = columnName
        if (giniColumn == "invalid"):
            return top # Done
        # Now that we have the column to split by, we should do so
        nodeObjects = top.add_children_by_col(giniColumn)
        # Loop through and add to dfs queue
        for obj in nodeObjects:
            dfsQueue.append(obj)
        # Now we execute dfs
        while len(dfsQueue) > 0:
            # Pop off node
            currentNode = dfsQueue.pop(0)
            if (currentNode.depth >= depthLimit or len(currentNode.data.index) < exampleLimit):
                continue
            # Compute next column to split
            maxGiniGain = -999999
            giniColumn = "invalid"
            for columnName in currentNode.data:
                if columnName == "decision":
                    continue
                curGain = gini_gain(currentNode.data, currentNode.data[columnName], columnName)
                if (curGain > maxGiniGain):
                    maxGiniGain = curGain
                    giniColumn = columnName
            # Split by column
            if giniColumn == "invalid":
                continue
            nodeObjects = currentNode.add_children_by_col(giniColumn)
            # Loop through and add to dfs queue
            for obj in nodeObjects:
                dfsQueue.append(obj)
    return top # This is the trained DT

def decisionTree(training_set, test_set, depthLimit, exampleLimit): #I'd do this recursively, but runtime would go through the roof
    top = trainDecisionTree(training_set, test_set, depthLimit, exampleLimit)
    
    # Test time
    result_test = 0
    total = 0
    # Test
    total = 0
    for index, row in test_set.iterrows():
        expected = row["decision"]
        current_tab = row.drop(columns="decision")
        result = predict_DT(current_tab, top)
        if result == expected:
            result_test = result_test + 1
        total = total + 1
    result_test = result_test / total
    
    return result_test, top



def trainRandomForest(training_set, test_set, depthLimit, exampleLimit):
#    depthLimit = 8
#    exampleLimit = 50
    
    # Grow using training set
    # Start with a node with everything in it
    top = Node(training_set)
    # Get an array of column names
    columnNames = []
    for columnName in top.data:
        if columnName == "decision":
            continue
        columnNames.append(columnName)
    p_val = int(math.sqrt(len(columnNames)))
    # Check first requirement before going through the iterations
    if (len(top.data.index) >= exampleLimit):
        # Now we loop through
        dfsQueue = [] # The next level to grow so to speak
        # We need a set of new nodes; compute GINI gain to determine which one
        maxGiniGain = -999999
        giniColumn = "invalid"
        # Pick sqrt p samples
        validSamples = False
        while not validSamples:
            pickedNames = random.sample(columnNames, p_val)
            toRemove = []
            for name in pickedNames:
                if name not in top.data.columns.tolist() or name == "decision":
                    toRemove.append(name)
            for name in toRemove:
                pickedNames.remove(name)

            if len(pickedNames) > 0:
                validSamples = True
        
        for columnName in pickedNames:
            if columnName == "decision":
                continue
            curGain = gini_gain(top.data, top.data[columnName], columnName)
            if (curGain > maxGiniGain):
                maxGiniGain = curGain
                giniColumn = columnName
                
        if giniColumn == "invalid":
            return top # Done
        # Now that we have the column to split by, we should do so
        nodeObjects = top.add_children_by_col(giniColumn)
        # Loop through and add to dfs queue
        for obj in nodeObjects:
            dfsQueue.append(obj)
        # Now we execute dfs
        while len(dfsQueue) > 0:
            # Pop off node
            currentNode = dfsQueue.pop(0)
            if (currentNode.depth >= depthLimit or len(currentNode.data.index) < exampleLimit):
                continue
                
            # Pick sqrt p samples
            validSamples = False
            while not validSamples:
                pickedNames = random.sample(columnNames, p_val)
                toRemove = []
                for name in pickedNames:
                    if name not in currentNode.data.columns.tolist() or name == "decision":
                        toRemove.append(name)
                for name in toRemove:
                    pickedNames.remove(name)
                
                if len(pickedNames) > 0:
                    validSamples = True
                    
            # Compute next column to split
            maxGiniGain = -999999
            giniColumn = "invalid"
            for columnName in pickedNames:
                if columnName == "decision":
                    continue
                curGain = gini_gain(currentNode.data, currentNode.data[columnName], columnName)
                if (curGain > maxGiniGain):
                    maxGiniGain = curGain
                    giniColumn = columnName
            if giniColumn == "invalid":
                continue
            # Split by column
            nodeObjects = currentNode.add_children_by_col(giniColumn)
            # Loop through and add to dfs queue
            for obj in nodeObjects:
                dfsQueue.append(obj)
    return top # This is the trained DT

def bagging(trainingSet, testSet, depthLimit, exampleLimit, bagNum):
    bag = [] #Array of learned trees
#    bagNum = 30
    for i in range(0, bagNum):
#        print("Training iteration " + str(i))
        thisTrainSet = trainingSet.sample(frac = 1, replace = True)
        tree = trainDecisionTree(thisTrainSet, testSet, depthLimit, exampleLimit)
        bag.append(tree)
        
    # Now test
    result_test = 0
    # Test results
    total = 0
    for index, row in testSet.iterrows():
        expected = row["decision"]
        current_tab = row.drop(columns="decision")
        result = predict_Bagging(current_tab, bag)
        if result == expected:
            result_test = result_test + 1
        total = total + 1
    result_test = result_test / total
    
    # Return results and the bag
    return result_test, bag

def randomForests(trainingSet, testSet, depthLimit, exampleLimit, bagNum):
    bag = [] #Array of learned trees
#    bagNum = 30
    for i in range(0, bagNum):
#        print("Training iteration " + str(i))
        thisTrainSet = trainingSet.sample(frac = 1, replace = True)
        tree = trainRandomForest(thisTrainSet, testSet, depthLimit, exampleLimit)
        bag.append(tree)
        
    # Now test
    result_test = 0
    # Test results
    total = 0
    for index, row in testSet.iterrows():
        expected = row["decision"]
        current_tab = row.drop(columns="decision")
        result = predict_Bagging(current_tab, bag)
        if result == expected:
            result_test = result_test + 1
        total = total + 1
    result_test = result_test / total
    
    # Return accuracy and bag
    return result_test, bag

# Cross validation
# Conduct 10-fold cross validation on a decision tree, given params
def conductCrossValidationDT(totalSet, depth, exampleLimit):
    #sections = []
    sections = np.array_split(totalSet, 10)
    decisionTreeAcc = []
    # We return the best tree we get out of the cross validation
    bestTree = None
    bestTreeScore = -99999
    for test_section in sections:
        training_section = totalSet.drop(test_section.index)
        # Perform the cross validation
        result, tree = decisionTree(training_section, test_section, depth, exampleLimit)
        if (result > bestTreeScore):
            bestTree = tree
            bestTreeScore = result
        decisionTreeAcc.append(result)
    # Average out in the end
#    decisionTreeAvg = sum(decisionTreeAcc)/len(decisionTreeAcc)
    # Return the BEST SCORE and best tree
    return bestTreeScore, bestTree

# Conduct 10-fold cross validation on bagging
def conductCrossValidationBT(totalSet, depth, exampleLimit, bagNum):
    #sections = []
    sections = np.array_split(totalSet, 10)
    baggingAcc = []
    # We return the best tree we get out of the cross validation
    bestBag = None
    bestTreeScore = -99999
    for test_section in sections:
        training_section = totalSet.drop(test_section.index)
        # Perform the cross validation
        result, bag = bagging(training_section, test_section, depth, exampleLimit, bagNum)
        if (result > bestTreeScore):
            bestBag = bag
            bestTreeScore = result
        baggingAcc.append(result)
    # Average out in the end
#    baggingAvg = sum(baggingAcc)/len(baggingAcc)
    # Return the best score and best tree
    return bestTreeScore, bestBag

# Conduct 10-fold cross validation on Random Forest
def conductCrossValidationRF(totalSet, depth, exampleLimit, bagNum):
    #sections = []
    sections = np.array_split(totalSet, 10)
    baggingAcc = []
    # We return the best tree we get out of the cross validation
    bestBag = None
    bestTreeScore = -99999
    for test_section in sections:
        training_section = totalSet.drop(test_section.index)
        # Perform the cross validation
        result, bag = randomForests(training_section, test_section, depth, exampleLimit, bagNum)
        if (result > bestTreeScore):
            bestBag = bag
            bestTreeScore = result
        baggingAcc.append(result)
    # Average out in the end
#    baggingAvg = sum(baggingAcc)/len(baggingAcc)
    # Return the bestTreeScore and best tree
    return bestTreeScore, bestBag

# Functions related to hyperparameter tuning
def objective_DT(params, dataset):
    # Keep track of evals
    global ITERATION_DT
    ITERATION_DT += 1
    
    # Extract the parameters (there should be two)
    depthLimit = int(params['depthLimit'])
    exampleLimit = int(params['exampleLimit'])
    
    start = timer()
    
    # 10 fold cross validation, get best result
    bestScore, bestTree = conductCrossValidationDT(dataset, depthLimit, exampleLimit)
    
    run_time = timer() - start
    # Get loss
    loss = 1 - bestScore
    
    # Return dictionary with information
    return {'loss': loss, 'params': params, 'iteration': ITERATION_DT, 'bestTree': bestTree, 'runtime': run_time, 'status': STATUS_OK}

# Decision tree search space
def searchspace_DT():
    space = {
        'depthLimit': hp.choice('depthLimit', range(3, 1000)),
        'exampleLimit': hp.choice('exampleLimit', range(1,2000))
    }
    return space

def objective_BT(params, dataset):
    # Keep track of evals
    global ITERATION_BT
    ITERATION_BT += 1
    
    # Extract the parameters (there should be three)
    depthLimit = int(params['depthLimit'])
    exampleLimit = int(params['exampleLimit'])
    bagNum = int(params['bagNum'])

    start = timer()
    
    # 10 fold cross validation, get best result
    bestScore, bestBag = conductCrossValidationBT(dataset, depthLimit, exampleLimit, bagNum)
    
    run_time = timer() - start
    # Get loss
    loss = 1 - bestScore
    
    # Return dictionary with information
    return {'loss': loss, 'params': params, 'iteration': ITERATION_BT, 'bestBag': bestBag, 'runtime': run_time, 'status': STATUS_OK}

# Decision tree search space
def searchspace_BT():
    space = {
        'depthLimit': hp.choice('depthLimit', range(3, 1000)),
        'exampleLimit': hp.choice('exampleLimit', range(1,2000)),
        'bagNum': hp.choice('bagNum', range(10,100))
    }
    return space

def objective_RF(params, dataset):
    # Keep track of evals
    global ITERATION_RF
    ITERATION_RF += 1
    
    # Extract the parameters (there should be three)
    depthLimit = int(params['depthLimit'])
    exampleLimit = int(params['exampleLimit'])
    bagNum = int(params['bagNum'])

    start = timer()
    
    # 10 fold cross validation, get best result
    bestScore, bestBag = conductCrossValidationRF(dataset, depthLimit, exampleLimit, bagNum)
    
    run_time = timer() - start
    # Get loss
    loss = 1 - bestScore
    
    # Return dictionary with information
    return {'loss': loss, 'params': params, 'iteration': ITERATION_RF, 'bestBag': bestBag, 'runtime': run_time, 'status': STATUS_OK}

# Decision tree search space
def searchspace_RF():
    space = {
        'depthLimit': hp.choice('depthLimit', range(3, 1000)),
        'exampleLimit': hp.choice('exampleLimit', range(1,2000)),
        'bagNum': hp.choice('bagNum', range(10,100))
    }
    return space

def optimize_3(dataset, bt_rf_enabled=False):
    # How many iterations hyperparameter tuning should run
    dt_max_tuning = 200
    bt_rf_max_tuning = 20
    # History objects
    dt_trials = Trials()
    if bt_rf_enabled:
        bt_trials = Trials()
        rf_trials = Trials()
    
    # Iteration objects
    global ITERATION_DT
    global ITERATION_BT
    global ITERATION_RF
    ITERATION_DT = 0
    ITERATION_BT = 0
    ITERATION_RF = 0
    
    # Pass in the dataset to obtain a partial function
    dt_obj = partial(objective_DT, dataset=dataset)
    # BT_RF
    if bt_rf_enabled:
        bt_obj = partial(objective_BT, dataset=dataset)  
        rf_obj = partial(objective_RF, dataset=dataset)
    
    # Perform hyperparameter optimization on all 3
    best_dt = fmin(fn = dt_obj, space = searchspace_DT(), algo = tpe.suggest, max_evals = dt_max_tuning, trials = dt_trials, rstate = np.random.RandomState(69))
    if bt_rf_enabled:
        best_bt = fmin(fn = bt_obj, space = searchspace_BT(), algo = tpe.suggest, max_evals = bt_rf_max_tuning, trials = bt_trials, rstate = np.random.RandomState(69))
        best_rf = fmin(fn = rf_obj, space = searchspace_RF(), algo = tpe.suggest, max_evals = bt_rf_max_tuning, trials = rf_trials, rstate = np.random.RandomState(69))
    
    # Now we can get the best results for each
    best_results_dt = sorted(dt_trials.results, key = lambda x: x['loss'])[0]
    if bt_rf_enabled:
        best_results_bt = sorted(bt_trials.results, key = lambda x: x['loss'])[0]
        best_results_rf = sorted(rf_trials.results, key = lambda x: x['loss'])[0]
    
    # Return all 3 if enabled
    if bt_rf_enabled:
        return {'dt': best_results_dt, 'bt': best_results_bt, 'rf': best_results_rf}
    else:
        return {'dt': best_results_dt}

def performance_DT(dt, test_set):
    top = dt
    # Test time
    result_test = 0
    total = 0
    # Test
    total = 0
    for index, row in test_set.iterrows():
        expected = row["decision"]
        current_tab = row.drop(columns="decision")
        result = predict_DT(current_tab, top)
        if result == expected:
            result_test = result_test + 1
        total = total + 1
    result_test = result_test / total
    
    return result_test

def performance_BT(bt, testSet):
    bag = bt
    # Now test
    result_test = 0
    # Test results
    total = 0
    for index, row in testSet.iterrows():
        expected = row["decision"]
        current_tab = row.drop(columns="decision")
        result = predict_Bagging(current_tab, bag)
        if result == expected:
            result_test = result_test + 1
        total = total + 1
    result_test = result_test / total
    
    # Return results and the bag
    return result_test

def performance_RF(rf, testSet): # Identical to bagging
    bag = rf
    # Now test
    result_test = 0
    # Test results
    total = 0
    for index, row in testSet.iterrows():
        expected = row["decision"]
        current_tab = row.drop(columns="decision")
        result = predict_Bagging(current_tab, bag)
        if result == expected:
            result_test = result_test + 1
        total = total + 1
    result_test = result_test / total
    
    # Return results and the bag
    return result_test

def main():
    # SET TO TRUE TO ENABLE BT AND RF (Extremely slow, but might lead to better results!)
    enable_boosted_trees = False
    # Load in the two datasets
    training_set_channelspec = pd.read_csv("trainingSet-ChannelSpec.csv")
    test_set_channelspec = pd.read_csv("testSet-ChannelSpec.csv")
    
    training_set_nochannelspec = pd.read_csv("trainingSet-NoChannelSpec.csv")
    test_set_nochannelspec = pd.read_csv("testSet-NoChannelSpec.csv")
#    print(training_set_loc)
#    print(test_set_loc)
    
    # Uncomment below for speedhack; you're probably going to get poor results, but it's fast so you can test the program
#    training_set_channelspec = training_set_channelspec.sample(30, random_state = 47)
#    test_set_channelspec = test_set_channelspec.sample(10, random_state = 47)
#    training_set_nochannelspec = training_set_nochannelspec.sample(30, random_state = 47)
#    test_set_nochannelspec = test_set_nochannelspec.sample(10, random_state = 47)

    # Hyperparameter tuning here
    # Get results on channelspec
    channelspec_results = optimize_3(training_set_channelspec, bt_rf_enabled = enable_boosted_trees)
#    channelspec_results = {'dt': {'loss': 0.1428571428571429, 'params': {'depthLimit': 16, 'exampleLimit': 43}, 'iteration': 66, 'bestTree': conductCrossValidationDT(training_set_channelspec, 16, 43)[1], 'runtime': 268.1861009000004, 'status': 'ok'}}
    # Get results on non_channelspec
    nochannelspec_results = optimize_3(training_set_nochannelspec, bt_rf_enabled = enable_boosted_trees)
#    nochannelspec_results = {'dt': {'loss': 0.09523809523809523, 'params': {'depthLimit': 449, 'exampleLimit': 50}, 'iteration': 33, 'bestTree': conductCrossValidationDT(training_set_nochannelspec, 449, 50)[1], 'runtime': 268.1861009000004, 'status': 'ok'}}
    # Finally, compute performance on test set
    dt_test_performance_channelspec = performance_DT(channelspec_results['dt']['bestTree'], test_set_channelspec)
    if enable_boosted_trees:
        bt_test_performance_channelspec = performance_BT(channelspec_results['bt']['bestBag'], test_set_channelspec)
        rf_test_performance_channelspec = performance_RF(channelspec_results['rf']['bestBag'], test_set_channelspec)
    
    dt_test_performance_nochannelspec = performance_DT(nochannelspec_results['dt']['bestTree'], test_set_nochannelspec)
    if enable_boosted_trees:
        bt_test_performance_nochannelspec = performance_BT(nochannelspec_results['bt']['bestBag'], test_set_nochannelspec)
        rf_test_performance_nochannelspec = performance_RF(nochannelspec_results['rf']['bestBag'], test_set_nochannelspec)
    # Prepare file to write results to
    base_path = os.getcwd()
    results_dir = str(time.time()) + "-results"
    results_path = os.path.join(base_path, results_dir)
    try: 
        os.makedirs(results_path, exist_ok = True) 
    except OSError as error: 
        print("Directory '%s' can not be created" % results_dir) 

    outfile = open(results_path + "/results.txt","w") 
    
    print("Writing results to file...")
    # Write some preliminary results
    outfile.write("CHANNEL_SPEC:\n")
    outfile.write("-----------------\n\n")
    outfile.write("Decision Tree Final Performance: ")
    outfile.write(str(dt_test_performance_channelspec))
    if enable_boosted_trees:
        outfile.write("\n")
        outfile.write("Bagging Final Performance: ")
        outfile.write(str(bt_test_performance_channelspec))
        outfile.write("\n")
        outfile.write("Random Forest Final Performance: ")
        outfile.write(str(rf_test_performance_channelspec))
    outfile.write("\n")
    outfile.write("Decision Tree Best Hyperparameters:\n")
    outfile.write(str(channelspec_results['dt']['params']))
    if enable_boosted_trees:
        outfile.write("\n")
        outfile.write("Bagging Best Hyperparameters:\n")
        outfile.write(str(channelspec_results['bt']['params']))
        outfile.write("\n")
        outfile.write("Random Forest Best Hyperparameters:\n")
        outfile.write(str(channelspec_results['rf']['params']))
    outfile.write("\n\n\n")
    
    outfile.write("NO_CHANNEL_SPEC:\n")
    outfile.write("-----------------\n\n")
    outfile.write("Decision Tree Final Performance: ")
    outfile.write(str(dt_test_performance_nochannelspec))
    if enable_boosted_trees:
        outfile.write("\n")
        outfile.write("Bagging Final Performance: ")
        outfile.write(str(bt_test_performance_nochannelspec))
        outfile.write("\n")
        outfile.write("Random Forest Final Performance: ")
        outfile.write(str(rf_test_performance_nochannelspec))
        
    outfile.write("\n")
    outfile.write("Decision Tree Best Hyperparameters:\n")
    outfile.write(str(nochannelspec_results['dt']['params']))
    if enable_boosted_trees:
        outfile.write("\n")
        outfile.write("Bagging Best Hyperparameters:\n")
        outfile.write(str(nochannelspec_results['bt']['params']))
        outfile.write("\n")
        outfile.write("Random Forest Best Hyperparameters:\n")
        outfile.write(str(nochannelspec_results['rf']['params']))
    outfile.write("\n")
    
    # When done
    outfile.close()
    
    # Create a directory to store the final trees
    channelspec_path = os.path.join(results_path, "channelspec")
    os.makedirs(channelspec_path)

    # Create a file storing the dict object
    control_file_c = open(channelspec_path + "/resultobj.txt","w") 
    # Write out the results object
    control_file_c.write(str(channelspec_results))
    control_file_c.close()
    
    # Print out the decision tree to a file
    dt_file = open(channelspec_path + "/decisiontree.txt", "w", encoding="utf-8")
    channelspec_results['dt']['bestTree'].printTree(toFile = dt_file)
    dt_file.close()
    if enable_boosted_trees:
        # Create a path for the random forest and binary trees
        bt_path = os.path.join(channelspec_path, "binarytree")
        os.makedirs(bt_path)
        # Write out each BT in succession
        bt_bag = channelspec_results['bt']['bestBag']
        for i in range(0, len(bt_bag)):
            bt_file = open(bt_path + "/" + str(i) + ".txt", "w", encoding="utf-8")
            bt_bag[i].printTree(toFile = bt_file)
            bt_file.close()

        rf_path = os.path.join(channelspec_path, "randomforest")
        os.makedirs(rf_path)
        # Write out each RF in succession
        rf_bag = channelspec_results['rf']['bestBag']
        for i in range(0, len(rf_bag)):
            rf_file = open(rf_path + "/" + str(i) + ".txt", "w", encoding="utf-8")
            rf_bag[i].printTree(toFile = rf_file)
            rf_file.close()
    
    # Now do the same for nochannelspec
    nochannelspec_path = os.path.join(results_path, "nochannelspec")
    os.makedirs(nochannelspec_path)
    
    # Create a file storing the dict object
    control_file_c = open(nochannelspec_path + "/resultobj.txt","w") 
    # Write out the results object
    control_file_c.write(str(nochannelspec_results))
    control_file_c.close()
    
    # Print out the decision tree to a file
    dt_file = open(nochannelspec_path + "/decisiontree.txt", "w", encoding="utf-8")
    nochannelspec_results['dt']['bestTree'].printTree(toFile = dt_file)
    dt_file.close()
    if enable_boosted_trees:
        # Create a path for the random forest and binary trees
        bt_path = os.path.join(nochannelspec_path, "binarytree")
        os.makedirs(bt_path)
        # Write out each BT in succession
        bt_bag = nochannelspec_results['bt']['bestBag']
        for i in range(0, len(bt_bag)):
            bt_file = open(bt_path + "/" + str(i) + ".txt", "w", encoding="utf-8")
            bt_bag[i].printTree(toFile = bt_file)
            bt_file.close()

        rf_path = os.path.join(nochannelspec_path, "randomforest")
        os.makedirs(rf_path)
        # Write out each RF in succession
        rf_bag = nochannelspec_results['rf']['bestBag']
        for i in range(0, len(rf_bag)):
            rf_file = open(rf_path + "/" + str(i) + ".txt", "w", encoding="utf-8")
            rf_bag[i].printTree(toFile = rf_file)
            rf_file.close()

    print("Results written to directory " + str(results_dir))
if __name__ == "__main__":
    main()