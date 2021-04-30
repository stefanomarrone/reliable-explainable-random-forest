#!/usr/bin/python3
from itertools import combinations_with_replacement 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing,tree
from sklearn.tree import _tree, DecisionTreeClassifier
from collections import Counter
import pandas as pd
import numpy as np
from pomegranate import *
import itertools
import random
import json
from time import time
from io import StringIO

# Python 3 program to print all
# possible strings of length k
	
# The method that prints all
# possible strings of length k.
# It is mainly a wrapper over
# recursive function printAllKLengthRec()
def printAllKLength(set, k):
  n = len(set)
  retval = printAllKLengthRec(set, list(), n, k)
  return retval

# The main recursive method
# to print all possible
# strings of length k
def printAllKLengthRec(set, prefix, n, k):
  retval = prefix
  if (k == 0):
    retval = prefix.append() + '\n'
  else:
    for i in range(n):
      newPrefix = prefix + ',' + set[i]
      printAllKLengthRec(set, newPrefix, n, k - 1)    
  return retval

class Reputation:
  def __init__(self,nname,aalpha,classnum):
    self.name = nname
    self.value = 1
    self.minval = 1.0/(classnum - 1)
    self.maxval = 1
    self.cnum = classnum
    self.alpha = aalpha
  
  def update(self,aalpha):
    temp = self.value * (1+aalpha)
    temp = max(self.minval,temp)
    temp = min(self.maxval,temp)
    retval = (temp != self.value)
    self.value = temp
    return retval

  def random(self):
    self.value = random.random()

  def increase(self):
    return self.update(self.alpha)
  
  def decrease(self):
    return self.update(-self.alpha)
  
  def getPositiveValue(self):
    return self.value
  
  def getNegativeValue(self):
    return float(1-self.value)/float(self.cnum)


def dtDump(tree):
  children_left = tree.children_left
  children_right = tree.children_right
  retval = list()
  stack = [(0,0,None)]
  while len(stack) > 0:
    node_id, depth, parent_id = stack.pop()
    is_split_node = children_left[node_id] != children_right[node_id]
    if is_split_node:
      stack.append((children_left[node_id], depth + 1, node_id))
      stack.append((children_right[node_id], depth + 1, node_id))
    temp = (node_id, depth, parent_id, not is_split_node)
    retval.append(temp)
  return retval

def computename(values,classes):
  index = np.argmax(values)
  return str(classes[index])

def computeimpurity(values):
  maxx = max(values[0])
  summ = sum(values[0])
  return 1 - float(maxx)/float(summ)
  
  
def recurse(tree, feature_names, classes, node, depth):
  left = tree.tree_.children_left
  right = tree.tree_.children_right
  threshold = tree.tree_.threshold
  features  = [feature_names[i] for i in tree.tree_.feature]
  value = tree.tree_.value
  indent = "  " * depth
  if (threshold[node] != -2):
    print(indent + "if (" + features[node] + " <= " + str(threshold[node]) + "):")
    if left[node] != -1:
      recurse (tree, feature_names, classes, left[node], depth + 1)
      print(indent + "else:")
      if right[node] != -1:
        recurse (tree, feature_names, classes, right[node], depth + 1)
  else:
    classname = computename(value[node],classes)
    impurity = computeimpurity(value[node])
    print(indent + "retval = " + classname + "# impurity = " + str(impurity))

def get_code(tree, feature_names, classes):
  print("def tree({}):".format(", ".join(feature_names)))
  recurse(tree, feature_names, classes, 0, 1)
  print("{}return retval".format("  "))

def getValues(dt,node):
  return dt.tree_.value[node][0]

def extractResult(string):
  f = StringIO(string)
  data = json.load(f)
  return data['parameters'][0]

def getMaxClass(res):
  ks = list(res.keys())
  retval = ks[0]
  confidence = res[ks[0]]
  for k in ks:
    if (res[k] > res[retval]):
      retval = k
      confidence = res[k]
  return retval, confidence

def getNames(psize):
  retval = ['DT_' + str(i) for i in range(0,psize)]
  return retval

def getDTsNumber(rf):
  return len(rf.estimators_)

def rorfBuild(rf, reputations):
  network = BayesianNetwork("RORF model")
  register = dict()
  popsize = len(rf.estimators_)
  dtnames = getNames(popsize)
  # BN Decision Layer
  for counter in range(0,popsize):
    name = dtnames[counter]
    dt = rf.estimators_[counter]
    register[name] = dict() # guard
    # leaf node
    finals = dtDump(dt.tree_)
    finals = list(filter(lambda r: r[3] == True,finals))
    finalnames = list(map(lambda r: 'L' + str(r[0]),finals))
    finalNumber = len(finalnames)
    distrodict = dict()
    for f in finalnames:
      distrodict[f] = 1./finalNumber
    leafCPT = DiscreteDistribution(distrodict)
    node_leaf = Node(leafCPT, name="L" + str(counter))
    network.add_node(node_leaf)
    register[name]['leaf'] = node_leaf # guard
    # class node
    distrolist = list()
    for i in range(0,len(finals)):
      node_id, depth, parent_id, flag = finals[i]
      values = getValues(dt,node_id)
      valuesum = sum(values)
      for j in range(0,len(rf.classes_)):
        accuracy = values[j]/valuesum
        temp = [finalnames[i], str(rf.classes_[j]), accuracy]
        distrolist.append(temp)
    classCPT = ConditionalProbabilityTable (distrolist,[leafCPT])
    register[name]['classcpt'] = classCPT # guard
    node_class = State(classCPT, name="C" + str(counter))
    network.add_states(node_class)
    register[name]['class'] = node_class # guard
    network.add_edge(node_leaf,node_class)
    # reputation node
    distrolist = list()
    for ci in rf.classes_:
      for cj in rf.classes_:
        if (ci == cj):
          accuracy = reputations[name].getPositiveValue()
        else:
          accuracy = reputations[name].getNegativeValue()
        temp = [str(ci), str(cj), accuracy]
        distrolist.append(temp)
    repCPT = ConditionalProbabilityTable (distrolist,[classCPT])
    register[name]['rep_distro'] = repCPT # guard
    node_repo = State(repCPT, name="R" + str(counter))
    network.add_states(node_repo)
    register[name]['reputation'] = node_repo # guard
    network.add_edge(node_class,node_repo)
  classes = list(map(lambda x: str(x),list(rf.classes_)))
  comb = [p for p in itertools.product(classes, repeat=popsize)]
  distrolisttemp = list(map(lambda x: list(x),comb))
  distrolist = list()
  for d in distrolisttemp:
    count = Counter(d)
    for c in classes:
      row = list(d)
      index = float(count.get(c,0))
      ratio = index/popsize
      temp = [c, ratio]
      row.extend(temp)
      distrolist.append(row)
  parentdistros = [register[dn]['rep_distro'] for dn in dtnames]
  votingCPT = ConditionalProbabilityTable (distrolist,parentdistros)
  node_voting = Node(votingCPT, name="VOTING")
  network.add_node(node_voting)
  for dtname in dtnames:
    network.add_edge(register[dtname]['reputation'],node_voting)
  network.bake()
  return network


def updateReputation(reputations,sds,d):
  popsize = len(reputations)
  names = getNames(popsize)
  changed = False
  for i in range(0,popsize):
    name = names[i]
    if (d == sds[i]):
      localChange = reputations[name].increase()
    else:
      localChange = reputations[name].decrease()
    changed = changed or localChange
  return reputations, changed


def getDecision(net,belfs,psize):
  singleDecisions = list()
  indict = "\n".join( "'{}': '{}',".format(state.name, belief) for state, belief in zip(net.states, belfs))
  indict = indict.replace("\n","")
  indict = '{' + indict[:-1] + '}'
  results = eval(indict)
  for i in range(0,psize):
    singledec = extractResult(results["C" + str(i)])
    singledec = getMaxClass(singledec)
    singleDecisions.append(singledec)
  decision_result = extractResult(results['VOTING'])
  decision, confidence = getMaxClass(decision_result)
  return decision, confidence, singleDecisions


def evalRorf(rf,val_X,reps,updt=False):
  psize = len(rf.estimators_)
  innerReps = dict(reps)
  pred_val_rorf = list()
  firstTrip = True
  debugFlag = False
  changeflag = False
  for j in range(0,len(val_X)):
    if (updt == True) and (firstTrip == False):
      innerReps, changeflag = updateReputation(innerReps,singledecs,dec)
    network = rorfBuild(rf,innerReps)
    if ((changeflag and debugFlag)== True):
      bucket = open('dump_' + str(j) + '.bn','w')
      bucket.write(str(network))
      bucket.close()
    observations = dict()
    test_single_dt = val_X.values[j].reshape(1,-1)
    for i in range(0,psize):
      thisDT = rf.estimators_[i]
      node_indicator = thisDT.decision_path(test_single_dt)
      decision_node = node_indicator[-1]
      temp = str(node_indicator).split('\n')[-1].split('\t')[0].split(',')[1]
      temp = temp.strip().rstrip()[:-1]
      decision_node = 'L' + temp
      observations['L' + str(i)] = decision_node
    belfs = list(map(str, network.predict_proba(observations)))
    dec, confidence, singledecs = getDecision(network,belfs,psize)    
    pred_val_rorf.append((int(dec),confidence))
    firstTrip = False
    #if (j%(len(val_X)/10) == 0):
      #print(str(j/len(val_X)))
  return pred_val_rorf

def core(testsetratio,rfdepth,learners,alphafactor,separator):
  retval = str(testsetratio) + separator + str(rfdepth) + separator + str(learners) + separator + str(alphafactor) + separator
  try:
    # Encoding
    enc = preprocessing.LabelEncoder()
    # Feature settings
    dataset_name = 'stroke_comma.csv'
    features = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','smoking_status']
    output = 'stroke'
    target = [output]
    df = pd.read_csv(dataset_name)
    superfeatures = list(features)
    superfeatures.append(output)
    # Encoding, balancing & splitting
    for f in features:
      df[f]= enc.fit_transform(df[f])
    truelen = len(df[df.stroke.eq(1)])
    falselen = len(df[df.stroke.eq(0)])
    halflen = min(truelen,falselen)
    truedf = df[df.stroke.eq(1)].sample(n=halflen)
    falsedf = df[df.stroke.eq(0)].sample(n=halflen)
    balancedDF = pd.concat([truedf, falsedf])
    X = balancedDF[features]
    y = balancedDF[target]
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = testsetratio)
    list_y = list(val_y.loc[:][output])
    # Training
    t = time()
    rf = RandomForestClassifier(n_jobs=1, max_depth=rfdepth, n_estimators=learners)
    rf.fit(train_X, train_y.values.ravel())
    retval += str(time() - t) + separator
    # Prediction
    t = time()
    preds_val = list(rf.predict(val_X))
    retval += str(time() - t) + separator
    accuracy_randomforest = accuracy_score(list_y, preds_val)
    retval += str(accuracy_randomforest) + separator
    #RORF
    reps = dict()
    dtSize = getDTsNumber(rf)
    names = getNames(dtSize)
    for n in names:
      reps[n] = Reputation(n,alphafactor,dtSize)
    t = time()
    temprslt = evalRorf(rf,val_X,reps,True)
    full_pred_val_rorf = list(map(lambda x: x[0],temprslt))
    confidence = list(map(lambda x: x[1],temprslt))
    retval += str(time() - t) + separator
    full_accuracy_rorf = accuracy_score(list_y, full_pred_val_rorf)
    retval += str(full_accuracy_rorf) + separator
    retval += str(float(sum(confidence))/len(confidence)) + separator
    retval += '\n'
  except ZeroDivisionError:
    pass
  retval = retval.replace('.',',')
  return retval


if __name__ == '__main__':
  separator = ';'
  bucket = open(sys.argv[1],'w')
  header = 'testsetratio' + separator + 'rfdepth' + separator + 'learners' + separator + 'alphafactor' + separator
  header += 'RF training (sec)' + separator + 'RF evaluation (sec)' + separator + 'RF accuracy' + separator #+ 'RORF building (sec)' + separator
  header += 'RORF evaluation (sec)' + separator + 'RORF accuracy' + separator #+ 'RORF full evaluation (sec)' + separator + 'RORF full accuracy' + separator
  #header += 'RF fake accuracy' + separator + 'RORF fake accuracy' + separator + 'RORF fake full accuracy' + separator + '\n'
  header += 'confidence' + separator
  header += '\n'
  bucket.write(header)
  testsetratio_range = [0.9]
  rfdepth_range = list(range(2,5))
  learners_range = list(range(2,15))
  alphafactor_range = [0.2]
  counter = 0
  for tsr in testsetratio_range:
    for rfd in rfdepth_range:
      for ls in learners_range:
        for af in alphafactor_range:
          print('lap ' + str(counter) + '/' + str(len(testsetratio_range) * len(rfdepth_range) * len(learners_range) * len(alphafactor_range)))
          row = core(tsr,rfd,ls,af,separator)
          bucket.write(row)
          counter += 1
  bucket.close()