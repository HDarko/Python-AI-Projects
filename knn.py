
# coding: utf-8

# In[44]:

'''Herschel Darko'''
#A K nearest neighbour for classification and labeling of files


import csv 
import sys
import math
import random

def main(file,k,trainpercent,seed):
    random.seed(seed)
    with open(file,'r') as csvfile:
        lines=csv.reader(csvfile)
        data=list(lines)
        titles=data.pop(0)
        random.shuffle(data)
        trainpercent=float(trainpercent)
        k=int(k)
        seed=int(seed)
        trainset,testset=datasplit(data,trainpercent)
        labels=labelcollect(trainset)
        matrix=cmatrix(labels,testset)
        for example in testset:
            closestNeighs=collectNeighbours(example,trainset,k)
            predlabel=predictclass(closestNeighs)
            column=labels.index(predlabel)
            actual=example[0]
            row=labels.index(actual)
            matrix[row][column]+=1
        mainfile=file.split("/")[-1][:-4]
        filename="results_"+mainfile+"_"+str(k)+"_"+str(seed)
        output(matrix,labels,filename)
        
def output(matrix,labels,file):
    #print("",end="")
    outData=[]
    outData.append(labels)
    count=0
    for row in matrix:
        row.append(labels[count])
        outData.append(row)
        count+=1
    myFile = open(file, 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(outData)

def cmatrix(labels,testset):
    confusionmatrix=[[0 for i in range(len(labels))] for i in range(len(labels))]
    return confusionmatrix
            
def labelcollect(trainset):
    #so we need using the number of labels, we create an array
    #
    labels=set()
    for example in trainset:
        labels.add(example[0])
    labels=list(labels)
    return labels
                          
def datasplit(data,trainpercent):
    #This function takes the the training percent of the datatset and transfers them to training set variable.
    #The rest is left as the test set
    trainingset=[]
    #testset=[]
    total=len(data)
    numberfortrain=total*trainpercent
    for x in range(0,int(numberfortrain)):
        trainingset.append(data.pop(0))
    testset=data
    return (trainingset,testset)

def collectNeighbours(example,trainingset,k):
    #finds the k closest neighbours to our example using euclidean distance and reurns them in a list
    exdistances=[]
    for i in range(len(trainingset)):
        exdistance= euclidean_distance(example,trainingset[i])
        exdistances.append((trainingset[i],exdistance))
    shdis=sorted(exdistances,key=lambda x:x[1])
    bestneighs=[]
    for i in range(k):
        bestneighs.append(shdis[i][0])
    return bestneighs

def predictclass(neighbours):
    #Looks in the list of examples and returns the most common classification or label
    predictions={}
    for i in range(len(neighbours)):
        current=neighbours[i][0]
        if current in predictions:
            predictions[current]+=1
        else:
            predictions[current]=1
    bestpred=max(predictions.items(),key=lambda x:x[1])
    return (bestpred[0])   
            
def euclidean_distance(X, Y):
    xfeatures=X[1:]
    yfeatures=Y[1:]
    ouya=[]
    for a, b in zip(xfeatures,yfeatures):
        if isinstance(a, int) and isinstance(b, int):
            result=(a-b)**2
            ouya.append(result)
        else:
            if (a==b):
                ouya.append(0)
            else:
                ouya.append(1)
    ouya=math.sqrt(sum(ouya))
    return ouya

main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]) 


# In[ ]:




# In[ ]:



