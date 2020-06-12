import numpy as np
from scoring import H, split_score, SplitStats, robustness

class Split:
    def __init__(self, attribute, cut_off, left_node, right_node):
        self.attribute = attribute
        self.cut_off = cut_off
        self.left_node = left_node
        self.right_node = right_node
        
    def predict(self, sample):
        if sample[self.attribute] < self.cut_off:
            return self.left_node.predict(sample)
        else:
            return self.right_node.predict(sample)      
        
    def __str__(self):
        return 'Split(' + str(self.attribute) + ', ' + str(self.cut_off) + \
            ', [' + str(self.left_node) + ', ' +  str(self.right_node) + '])'
        
class Leaf:
    def __init__(self, label):
        self.label = label

    def predict(self, sample):
        if self.label > 0.5:
            return 1
        else: 
            return 0             
        
    def __str__(self):
        return 'Leaf(' + str(self.label) + ')'


class Trees:

    def __init__(self, trees):
        self.trees = trees

    def predict(self, data):    
        predicted_classes = []

        for row in range(0, len(data)):
            
            predictions = [tree.predict(data.iloc[row]) for tree in self.trees]
            num_positive = np.sum(predictions)
            if num_positive > len(predictions) / 2:
                predicted_class = 1
            else:
                predicted_class = 0
            
            predicted_classes.append(predicted_class)        

        return predicted_classes            


class ExtremelyRandomizedTrees:

    def __init__(self, num_trees, d, n_min):
        self.num_trees = num_trees
        self.d = d
        self.n_min = n_min


    def split(self, samples, attribute_candidates, label_attribute, constant_attributes):

        attribute_candidates_for_splitting = []    
        for candidate in attribute_candidates:
            if candidate not in constant_attributes:
                attribute_candidates_for_splitting.append(candidate)

        to_take = self.d
        if len(attribute_candidates_for_splitting) < self.d:
            to_take = len(attribute_candidates_for_splitting)
        
        attributes = np.random.choice(attribute_candidates_for_splitting, to_take, replace=False)
        
        max_score = -1        
        the_attribute = None
        the_cutoff = None
        the_left_samples = None
        the_right_samples = None
        
        all_scores = []

        best_stats = None    
        all_stats = []

        for attribute in attributes:
            attribute_values = np.array(samples[attribute])

            min_attribute_value = np.min(attribute_values)
            max_attribute_value = np.max(attribute_values)

            cut_offs = np.random.uniform(min_attribute_value, max_attribute_value, size=1)    

            for cut_off in cut_offs:
                left_samples = samples[samples[attribute] < cut_off].copy(deep=True)
                right_samples = samples[samples[attribute] >= cut_off].copy(deep=True)

                num_plus_left = np.sum(left_samples[label_attribute] == 1)
                num_minus_left = np.sum(left_samples[label_attribute] == 0)
                num_plus_right = np.sum(right_samples[label_attribute] == 1)
                num_minus_right = np.sum(right_samples[label_attribute] == 0)

                score = split_score(num_plus_left, num_minus_left, num_plus_right, num_minus_right)
                all_scores.append((attribute, cut_off, score, num_plus_left, num_minus_left, num_plus_right, num_minus_right))
                all_stats.append(SplitStats(num_plus_left, num_minus_left, num_plus_right, num_minus_right))

                if score > max_score:
                    max_score = score
                    the_attribute = attribute
                    the_cutoff = cut_off
                    the_left_samples = left_samples
                    the_right_samples = right_samples     
                    best_stats = SplitStats(num_plus_left, num_minus_left, num_plus_right, num_minus_right)       
        
        min_robustness = None
        for other_stats in all_stats:
            if best_stats != other_stats:
               curr_robustness = robustness(best_stats, other_stats)
               if min_robustness is None or curr_robustness < min_robustness:
                 min_robustness = curr_robustness
             

        print('attribute:', the_attribute, ', cut_off:',the_cutoff, 
             ', robustness:', min_robustness, ', num samples:', len(samples))          

        return the_attribute, the_cutoff, the_left_samples, the_right_samples, max_score, all_scores       



    def stop_split(self, samples, attribute_candidates, label_attribute, known_constant_attributes):
        
        constant_attributes = set()
        for known_constant_attribute in known_constant_attributes:
            constant_attributes.add(known_constant_attribute)

        for attribute in attribute_candidates:            
            if attribute not in known_constant_attributes and len(samples[attribute].unique()) == 1:
                constant_attributes.add(attribute)
                
        if len(constant_attributes) == len(attribute_candidates):
            return True, constant_attributes                

        if len(samples) < self.n_min:
            return True, constant_attributes
        
        if len(samples[label_attribute].unique()) == 1:
            return True, constant_attributes
        
        return False, constant_attributes


    def split_node(self, samples, attribute_candidates, label_attribute, known_constant_attributes):
        
        should_stop, updated_constant_attributes = \
            self.stop_split(samples, attribute_candidates, label_attribute, known_constant_attributes)

        if should_stop:            
            num_positive = float(np.sum(samples[label_attribute]))
            label = num_positive / len(samples[label_attribute])
            
            return Leaf(label)
        
        attribute, cut_off, left_samples, right_samples, score, all_scores = \
            self.split(samples, attribute_candidates, label_attribute, updated_constant_attributes)
        
        #print(all_scores)

        if score == 0.0:
            
            num_positive = float(np.sum(samples[label_attribute]))
            label = num_positive / len(samples[label_attribute])
            
            return Leaf(label)    

        left_child = self.split_node(left_samples, attribute_candidates, label_attribute, updated_constant_attributes)        
        right_child = self.split_node(right_samples, attribute_candidates, label_attribute, updated_constant_attributes)
        
        return Split(attribute, cut_off, left_child, right_child)    


    def fit(self, data, attribute_candidates, label_attribute):
        trees = []
        for num_trees_fitted in range(0, self.num_trees):
            trees.append(self.split_node(data, attribute_candidates, label_attribute, set()))       
            print(num_trees_fitted + 1, " trees fitted")

        return Trees(trees)                