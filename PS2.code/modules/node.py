# DOCUMENTATION
# =====================================
# Class node attributes:
# ----------------------------
# children - a list of 2 nodes if numeric, and a dictionary (key=attribute value, value=node) if nominal.  
#            For numeric, the 0 index holds examples < the splitting_value, the 
#            index holds examples >= the splitting value
#
# label - is None if there is a decision attribute, and is the output label (0 or 1 for
#   the homework data set) if there are no other attributes
#       to split on or the data is homogenous
#
# decision_attribute - the index of the decision attribute being split on
#
# is_nominal - is the decision attribute nominal
#
# value - Ignore (not used, output class if any goes in label)
#
# splitting_value - if numeric, where to split
#
# name - name of the attribute being split on
#import matplotlib

class Node:
    def __init__(self):
        # initialize all attributes
        self.label = None
        self.decision_attribute = None
        self.is_nominal = None
        self.value = None
        self.splitting_value = None
        self.children = {}
        self.name = None
        self.modeVal = 0  #should be one of children or decimal
        #self.visits = 1

    def __classify_helper__(self,node,instance):
        if not(node.label == None):
            return node.label
        if not(node.decision_attribute == None):
            if (instance[node.decision_attribute] == None):
                instance[node.decision_attribute] = node.modeVal
            if node.is_nominal:
                i = node.children[instance[node.decision_attribute]]
                return node.__classify_helper__(i,instance)
            else:
                if instance[node.decision_attribute] < node.splitting_value:
                    return node.__classify_helper__(node.children[0],instance)
                else:
                    return node.__classify_helper__(node.children[1],instance)
        else:
            print "bug" + str(node.decision_attribute)
            
    def classify(self, instance):
        '''
        given a single observation, will return the output of the tree
        '''
        return self.__classify_helper__(self,instance)
    # Your code here
    
    pass

    
        
    def print_tree(self, indent = 0):
        '''
        returns a string of the entire tree in human readable form
        IMPLEMENTING THIS FUNCTION IS OPTIONAL
        '''
        # Your code here
        pass


    def print_dnf_tree(self):
        '''
        returns the disjunct normalized form of the tree.
        '''
        return dnf_recur(self,[])
        pass
    def dnf_recur(self,dnf):
        if not(self.label == None):
            return self.label and dnf
        if not(self.is_nominal):
            new_dnf1 = self.name + "<" + str(self.splitting_value)
            new_dnf2 = self.name + ">=" + str(self.splitting_value)
            return [self.dnf_recur(new_dnf1),self.dnf_recur(new_dnf2)]
        else:
            return [self.dnf_recur(dnf.append(self.dnf_rep(child))) for child in self.children]
            
    
    def dnf_rep(self,child):
            return self.name + "=" + str(child)
            #return "(" + "^".join([(self.name + "=" + i.key) for i in self.children]) + ")"
