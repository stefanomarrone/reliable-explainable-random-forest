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


def confidenceCompare(oracle,valrfs,rorfs):
  agree_ok = list()
  disagree_ok = list()
  agree_ko = list()
  disagree_ko = list()
  valrorfs = list(map(lambda x: x[0],rorfs))
  confs = list(map(lambda x: x[1],rorfs)) 
  for i in range(0,len(oracle)):
    if valrorfs[i] == valrfs[i]:
      if oracle[i] == valrorfs[i]:
        agree_ok.append(confs[i])
      else:
        agree_ko.append(confs[i])
    else:
      if oracle[i] == valrorfs[i]:
        disagree_ok.append(confs[i])
      else:
        disagree_ko.append(confs[i])
  conf_avg = float(sum(confs))/len(confs)
  aok_avg = float(sum(agree_ok))/len(agree_ok) if len(agree_ok) > 0 else None
  ako_avg = float(sum(agree_ko))/len(agree_ko) if len(agree_ko) > 0 else None
  daok_avg = float(sum(disagree_ok))/len(disagree_ok) if len(disagree_ok) > 0 else None
  dako_avg = float(sum(disagree_ko))/len(disagree_ko) if len(disagree_ko) > 0 else None
  return conf_avg, aok_avg, ako_avg, daok_avg, dako_avg 
  
def core_confidence(testsetratio,rfdepth,learners,alphafactor):
  enc = preprocessing.LabelEncoder()
  dataset_name = 'stroke_comma.csv'
  features = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','smoking_status']
  output = 'stroke'
  target = [output]
  df = pd.read_csv(dataset_name)
  superfeatures = list(features)
  superfeatures.append(output)
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
  print(len(list_y))
  t = time()
  rf = RandomForestClassifier(n_jobs=1, max_depth=rfdepth, n_estimators=learners)
  rf.fit(train_X, train_y.values.ravel())
  t = time()
  preds_val = list(rf.predict(val_X))
  accuracy_randomforest = accuracy_score(list_y, preds_val)
  reps = dict()
  dtSize = getDTsNumber(rf)
  names = getNames(dtSize)
  for n in names:
    reps[n] = Reputation(n,alphafactor,dtSize)
  temprslt = evalRorf(rf,val_X,reps,True)
  conf_avg, aok_avg, ako_avg, daok_avg, dako_avg = confidenceCompare(list_y,preds_val,temprslt)
  return conf_avg, aok_avg, ako_avg, daok_avg, dako_avg


if __name__ == '__main__':
  testsetratio_range = [0.9]
  rfdepth_range = list(range(3,4))
  learners_range = list(range(11,12))
  alphafactor_range = [0.2]
  for tsr in testsetratio_range:
    for rfd in rfdepth_range:
      for ls in learners_range:
        for af in alphafactor_range:
          conf_avg, aok_avg, ako_avg, daok_avg, dako_avg  = core_confidence(tsr,rfd,ls,af)
          print(conf_avg)
          print(aok_avg)
          print(ako_avg)
          print(daok_avg)
          print(dako_avg)
