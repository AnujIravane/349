import math
from node import Node
import sys
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def ID3(data_set, attribute_metadata, numerical_splits_count, depth):
    '''
    See Textbook for algorithm.
    Make sure to handle unknown values, some suggested approaches were
    given in lecture.
    ========================================================================================================
    Input:  A data_set, attribute_metadata, maximum number of splits to consider for numerical attributes,
	maximum depth to search to (depth = 0 indicates that this node should output a label)
    ========================================================================================================
    Output: The node representing the decision tree learned over the given data set
    ========================================================================================================
    '''
    # Your code here
    print depth
    Dtree = Node()
    if len(data_set) == 0:
        return Dtree
    c = check_homogenous([[element[0]] for element in data_set])
    if isinstance(c,int):
         Dtree.label = c
         return Dtree
    elif len(data_set[0]) == 1 or depth <= 0 or [0]*(len(numerical_splits_count)-1) == numerical_splits_count[1:]:
         Dtree.label = mode(data_set)
         return Dtree
    else:
         data_set = missingValues(data_set)
         best_attribute,threshold = pick_best_attribute(data_set,attribute_metadata,numerical_splits_count)
         if not(best_attribute):
             Dtree.label = mode(data_set)
             return Dtree
         
         Dtree.decision_attribute = best_attribute
         Dtree.modeVal = mode([[element[Dtree.decision_attribute]] for element in data_set])
         Dtree.name = attribute_metadata[best_attribute]['name']
         if threshold:
             Dtree.is_nominal = False
             Dtree.splitting_value = threshold
             less,greater = split_on_numerical(data_set,best_attribute,threshold)
             new_nsc = numerical_splits_count
             new_nsc[best_attribute] -= 1
             Dtree.children = [ID3(less,attribute_metadata,new_nsc,depth-1),ID3(greater,attribute_metadata,new_nsc,depth-1)]
         else:
             Dtree.is_nominal = True
             n_dict = split_on_nominal(data_set,best_attribute)
             new_attribute_metadata = attribute_metadata
             new_attribute_metadata.pop(best_attribute)
             #try:
             Dtree.children = [ID3(removeAttribute(value,best_attribute),new_attribute_metadata,numerical_splits_count,depth-1) for key,value in n_dict.iteritems()]
             #except AttributeError:
              #   print n_dict
               #  print best_attribute
                # print threshold
                 #raise Exception("wut")
    return Dtree       
    pass


def removeAttribute(data_set,i):
    new_list = data_set
    for j in range(0,len(data_set)):
         new_list[j].pop(i)
    return new_list

         
def missingValues(data_set):
    new_data_set = data_set
    for j in range(0,len(data_set)):
         element1 = data_set[j]
         for i in range(1,len(element1)):
             if element1[i] == None:
                 data_set_iter = [el for el in data_set]
                 data_set_iter.pop(j)
                 new_data_set[j][i] = mode([[element[i]] for element in data_set_iter])
    return new_data_set     
             
                 
         
def check_homogenous(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Checks if the output value (index 0) is the same for all examples in the the data_set, if so return that output value, otherwise return None.
    ========================================================================================================
    Output: Return either the homogenous attribute or None
    ========================================================================================================
     '''
    # Your code here
    if ([data_set[0] for i in data_set] == data_set):
        return data_set[0][0]
    else:
        return None
    pass
# ======== Test Cases =============================
# data_set = [[0],[1],[1],[1],[1],[1]]
# check_homogenous(data_set) ==  None
# data_set = [[0],[1],[None],[0]]
# check_homogenous(data_set) ==  None
# data_set = [[1],[1],[1],[1],[1],[1]]
# check_homogenous(data_set) ==  1

def pick_best_attribute(data_set, attribute_metadata, numerical_splits_count):
    '''
    ========================================================================================================
    Input:  A data_set, attribute_metadata, splits counts for numeric
    ========================================================================================================
    Job:    Find the attribute that maximizes the gain ratio. If attribute is numeric return best split value.
            If nominal, then split value is False.
            If gain ratio of all the attributes is 0, then return False, False
            Only consider numeric splits for which numerical_splits_count is greater than zero
    ========================================================================================================
    Output: best attribute, split value if numeric
    ========================================================================================================
    '''
    # Your code here
    best_gr = 0
    best_attr = None
    attr_thresh = False
    for i in range(1,len(attribute_metadata)):
        gr = 0
        is_numeric = False
        if attribute_metadata[i]['is_nominal'] == True:
            gr = gain_ratio_nominal(data_set,i)
        else:
            if numerical_splits_count[i] > 0:
                is_numeric = True
                gr,threshold = gain_ratio_numeric(data_set,i,5)
        if gr > best_gr:
            best_attr = i
            best_gr = gr
            if is_numeric:
                attr_thresh = threshold
            else:
                attr_thresh = False
    return (best_attr,attr_thresh)
                
            
    pass

# # ======== Test Cases =============================
# numerical_splits_count = [20,20]
# attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "opprundifferential",'is_nominal': False}]
# data_set = [[1, 0.27], [0, 0.42], [0, 0.86], [0, 0.68], [0, 0.04], [1, 0.01], [1, 0.33], [1, 0.42], [0, 0.51], [1, 0.4]]
# pick_best_attribute(data_set, attribute_metadata, numerical_splits_count) == (1, 0.51)
# attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "weather",'is_nominal': True}]
# data_set = [[0, 0], [1, 0], [0, 2], [0, 2], [0, 3], [1, 1], [0, 4], [0, 2], [1, 2], [1, 5]]
# pick_best_attribute(data_set, attribute_metadata, numerical_splits_count) == (1, False)

# Uses gain_ratio_nominal or gain_ratio_numeric to calculate gain ratio.

         
def mode(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Takes a data_set and finds mode of index 0.
    ========================================================================================================
    Output: mode of index 0.
    ========================================================================================================
    '''
    # Your code here
    counts = {}
    for element in data_set:
        if element[0] in counts.keys():
            counts[element[0]] += 1
        else:
            counts[element[0]] = 1
    return max(counts, key = counts.get)
    pass
# ======== Test case =============================
# data_set = [[0],[1],[1],[1],[1],[1]]
# mode(data_set) == 1
# data_set = [[0],[1],[0],[0]]
# mode(data_set) == 0

def entropy(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Calculates the entropy of the attribute at the 0th index, the value we want to predict.
    ========================================================================================================
    Output: Returns entropy. See Textbook for formula
    ========================================================================================================
    '''
    counts = {}
    num = len(data_set)
    for element in data_set:
        if element[0] in counts.keys():
            counts[element[0]] += 1
        else:
            counts[element[0]] = 1
    entropy_val = 0
    for element in counts:
        p = counts[element]*1.0/num*1.0
        entropy_val -= p*math.log(p,2)
    return entropy_val


# ======== Test case =============================
# data_set = [[0],[1],[1],[1],[0],[1],[1],[1]]
# entropy(data_set) == 0.811
# data_set = [[0],[0],[1],[1],[0],[1],[1],[0]]
# entropy(data_set) == 1.0
# data_set = [[0],[0],[0],[0],[0],[0],[0],[0]]
# entropy(data_set) == 0


def gain_ratio_nominal(data_set, attribute):
    '''
    ========================================================================================================
    Input:  Subset of data_set, index for a nominal attribute
    ========================================================================================================
    Job:    Finds the gain ratio of a nominal attribute in relation to the variable we are training on.
    ========================================================================================================
    Output: Returns gain_ratio. See https://en.wikipedia.org/wiki/Information_gain_ratio
    ========================================================================================================
    '''
    # Your code here
    entropy_before = entropy([[element[0]] for element in data_set])
    entropy_dict = {}
    entropy_after = 0
    iv = 0
    entropy_dict = split_on_nominal(data_set,attribute)
    for element in entropy_dict:
        n_ratio = len(entropy_dict[element])*1.0/len(data_set)*1.0
        entropy_after += n_ratio*(entropy(entropy_dict[element]))*1.0
        iv -= n_ratio*math.log(n_ratio,2)
    if iv == 0:
        return 0.5
    return (entropy_before - entropy_after)/iv                            
    pass
# ======== Test case =============================
# data_set, attr = [[1, 2], [1, 0], [1, 0], [0, 2], [0, 2], [0, 0], [1, 3], [0, 4], [0, 3], [1, 1]], 1
# gain_ratio_nominal(data_set,attr) == 0.11470666361703151
# data_set, attr = [[1, 2], [1, 2], [0, 4], [0, 0], [0, 1], [0, 3], [0, 0], [0, 0], [0, 4], [0, 2]], 1
# gain_ratio_nominal(data_set,attr) == 0.2056423328155741
# data_set, attr = [[0, 3], [0, 3], [0, 3], [0, 4], [0, 4], [0, 4], [0, 0], [0, 2], [1, 4], [0, 4]], 1
# gain_ratio_nominal(data_set,attr) == 0.06409559743967516

def gain_ratio_numeric(data_set, attribute, steps=1):
    '''
    ========================================================================================================
    Input:  Subset of data set, the index for a numeric attribute, and a step size for normalizing the data.
    ========================================================================================================
    Job:    Calculate the gain_ratio_numeric and find the best single threshold value
            The threshold will be used to split examples into two sets
                 those with attribute value GREATER THAN OR EQUAL TO threshold
                 those with attribute value LESS THAN threshold
            Use the equation here: https://en.wikipedia.org/wiki/Information_gain_ratio
            And restrict your search for possible thresholds to examples with array index mod(step) == 0
    ========================================================================================================
    Output: This function returns the gain ratio and threshold value
    ========================================================================================================
    '''
    # Your code here
    best_gain_ratio = 0
    best_threshold = None
    entropy_before = entropy([[element[0]] for element in data_set])
    for index in range(0,len(data_set),steps):
        element = data_set[index]
        entropy_after = 0
        ig = 0
        iv = 0
        less, greater = split_on_numerical(data_set,attribute,element[attribute])
        l_ratio = len(less)*1.0/len(data_set)*1.0
        g_ratio = len(greater)*1.0/len(data_set)*1.0
        if (l_ratio > 0 and g_ratio > 0):
            entropy_after = l_ratio*1.0*entropy(less) + g_ratio*1.0*entropy(greater)
            iv = - l_ratio*math.log(l_ratio,2) - g_ratio*math.log(g_ratio,2)
            ig = (entropy_before - entropy_after)/iv*1.0
        if (best_gain_ratio <= ig):
            best_gain_ratio = ig
            best_threshold = element[attribute]

    return (best_gain_ratio,best_threshold)
    pass
# ======== Test case =============================
# data_set,attr,step = [[0,0.05], [1,0.17], [1,0.64], [0,0.38], [0,0.19], [1,0.68], [1,0.69], [1,0.17], [1,0.4], [0,0.53]], 1, 2
# gain_ratio_numeric(data_set,attr,step) == (0.31918053332474033, 0.64)
# data_set,attr,step = [[1, 0.35], [1, 0.24], [0, 0.67], [0, 0.36], [1, 0.94], [1, 0.4], [1, 0.15], [0, 0.1], [1, 0.61], [1, 0.17]], 1, 4
# gain_ratio_numeric(data_set,attr,step) == (0.11689800358692547, 0.94)
# data_set,attr,step = [[1, 0.1], [0, 0.29], [1, 0.03], [0, 0.47], [1, 0.25], [1, 0.12], [1, 0.67], [1, 0.73], [1, 0.85], [1, 0.25]], 1, 1
# gain_ratio_numeric(data_set,attr,step) == (0.23645279766002802, 0.29)

def split_on_nominal(data_set, attribute):
    '''
    ========================================================================================================
    Input:  subset of data set, the index for a nominal attribute.
    ========================================================================================================
    Job:    Creates a dictionary of all values of the attribute.
    ========================================================================================================
    Output: Dictionary of all values pointing to a list of all the data with that attribute
    ========================================================================================================
    '''
    
    # Your code here
    entropy_dict = {}
    for element in data_set:
        if element[attribute] in entropy_dict:
            entropy_dict[element[attribute]].append(element)
        else:
            entropy_dict[element[attribute]] = [element]
    return entropy_dict
    pass
# ======== Test case =============================
# data_set, attr = [[0, 4], [1, 3], [1, 2], [0, 0], [0, 0], [0, 4], [1, 4], [0, 2], [1, 2], [0, 1]], 1
# split_on_nominal(data_set, attr) == {0: [[0, 0], [0, 0]], 1: [[0, 1]], 2: [[1, 2], [0, 2], [1, 2]], 3: [[1, 3]], 4: [[0, 4], [0, 4], [1, 4]]}
# data_set, attr = [[1, 2], [1, 0], [0, 0], [1, 3], [0, 2], [0, 3], [0, 4], [0, 4], [1, 2], [0, 1]], 1
# split on_nominal(data_set, attr) == {0: [[1, 0], [0, 0]], 1: [[0, 1]], 2: [[1, 2], [0, 2], [1, 2]], 3: [[1, 3], [0, 3]], 4: [[0, 4], [0, 4]]}

def split_on_numerical(data_set, attribute, splitting_value):
    '''
    ========================================================================================================
    Input:  Subset of data set, the index for a numeric attribute, threshold (splitting) value
    ========================================================================================================
    Job:    Splits data_set into a tuple of two lists, the first list contains the examples where the given
	attribute has value less than the splitting value, the second list contains the other examples
    ========================================================================================================
    Output: Tuple of two lists as described above
    ========================================================================================================
    '''
    # Your code here
    less = []
    greater = []
    for element in data_set:
        if (element[attribute] >= splitting_value):
            greater.append(element)
        else:
            less.append(element)
    return (less,greater)
    pass
# ======== Test case =============================
# d_set,a,sval = [[1, 0.25], [1, 0.89], [0, 0.93], [0, 0.48], [1, 0.19], [1, 0.49], [0, 0.6], [0, 0.6], [1, 0.34], [1, 0.19]],1,0.48
# split_on_numerical(d_set,a,sval) == ([[1, 0.25], [1, 0.19], [1, 0.34], [1, 0.19]],[[1, 0.89], [0, 0.93], [0, 0.48], [1, 0.49], [0, 0.6], [0, 0.6]])
# d_set,a,sval = [[0, 0.91], [0, 0.84], [1, 0.82], [1, 0.07], [0, 0.82],[0, 0.59], [0, 0.87], [0, 0.17], [1, 0.05], [1, 0.76]],1,0.17
# split_on_numerical(d_set,a,sval) == ([[1, 0.07], [1, 0.05]],[[0, 0.91],[0, 0.84], [1, 0.82], [0, 0.82], [0, 0.59], [0, 0.87], [0, 0.17], [1, 0.76]])
