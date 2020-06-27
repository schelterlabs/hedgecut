import numpy as np
from scoring import H, split_score, SplitStats, robustness, split_score_from_stats

class Split:
    def __init__(self, attribute, cut_off, left_node, right_node, stats=None, 
                 branching_candidates=None, branching_splits=None):
        self.attribute = attribute
        self.cut_off = cut_off
        self.left_node = left_node
        self.right_node = right_node
        self.stats = stats
        self.branching_candidates = branching_candidates
        self.branching_splits = branching_splits

    def predict(self, sample):
        if sample[self.attribute] < self.cut_off:
            return self.left_node.predict(sample)
        else:
            return self.right_node.predict(sample)      
        
    def forget(self, sample, label):

        if self.branching_candidates is not None:
            #print("Invoking forgetting procedure on NON-ROBUST split")

            #print("best before", split_score_from_stats(self.stats))

            # TODO update stats
            if sample[self.attribute] <= self.cut_off:
                if label == 1:
                    self.stats.num_plus_left -= 1
                else:    
                    self.stats.num_minus_left -= 1
            else:        
                if label == 1:
                    self.stats.num_plus_right -= 1
                else:    
                    self.stats.num_minus_right -= 1                
            # TODO restructure if needed

            best_score_after = split_score_from_stats(self.stats)

            #print("best after", best_score_after)

            candidate_stats_to_remove = []
            for candidate_stats in self.branching_candidates:    
                #print("candidate before", split_score_from_stats(candidate_stats))

                # TODO update stats
                if sample[self.attribute] <= self.cut_off:
                    if label == 1:
                        candidate_stats.num_plus_left -= 1
                    else:    
                        candidate_stats.num_minus_left -= 1
                else:        
                    if label == 1:
                        candidate_stats.num_plus_right -= 1
                    else:    
                        candidate_stats.num_minus_right -= 1                
                # TODO restructure if needed

                if candidate_stats.num_plus_left >= 0 and candidate_stats.num_minus_left >= 0 and \
                    candidate_stats.num_plus_right >= 0 and candidate_stats.num_minus_right >= 0:

                    candidate_score_after = split_score_from_stats(candidate_stats)
                    #print("candidate after", candidate_score_after)

                    if candidate_score_after > best_score_after:
                        print("!!!!!!!!!!!!!!!!!!!!!!REORGANISATION REQUIRED!!!!!!!!!!!!!!!!!!!")     
                else:
                    print("Invalidating split candidate", candidate_stats)           
                    candidate_stats_to_remove.append(candidate_stats)                   

            for stats_to_remove in candidate_stats_to_remove:
                self.branching_candidates.remove(stats_to_remove)        


        if sample[self.attribute] < self.cut_off:
            return self.left_node.forget(sample, label)
        else:
            return self.right_node.forget(sample, label)   

    def __str__(self):
        return 'Split(' + str(self.attribute) + ', ' + str(self.cut_off) + \
            ', [' + str(self.left_node) + ', ' +  str(self.right_node) + '])'
        
class Leaf:
    def __init__(self, num_positive, num_samples):
        self.num_positive = num_positive
        self.num_samples = num_samples

    def forget(self, sample, label):
        #print("Invoking forgetting procedure on leaf")

        if label == 1:
            self.num_positive -= 1

        self.num_samples -= 1

    def predict(self, sample):

        label = float(self.num_positive) / self.num_samples

        if label > 0.5:
            return 1
        else: 
            return 0             
        
    def __str__(self):
        return 'Leaf(' + str(self.label) + ')'


class Trees:

    def __init__(self, trees):
        self.trees = trees
        self.num_training_samples = None


    def forget(self, sample, label):
        for tree in self.trees:
            tree.forget(sample, label)

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
        self.percentiles = {}


    def split(self, samples, attribute_candidates, label_attribute, constant_attributes, node_id, num_tries=1):

        attribute_candidates_for_splitting = []    
        for candidate in attribute_candidates:
            if candidate not in constant_attributes:
                attribute_candidates_for_splitting.append(candidate)

        to_take = self.d
        if len(attribute_candidates_for_splitting) < self.d:
            to_take = len(attribute_candidates_for_splitting)
        
        attributes = np.random.choice(attribute_candidates_for_splitting, to_take, replace=False)
        #attributes = attribute_candidates_for_splitting
        
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
 
            cut_offs = np.random.choice(self.percentiles[attribute], 1, replace=False)

            for cut_off in cut_offs:

                left_samples = samples[samples[attribute] < cut_off].copy(deep=True)
                right_samples = samples[samples[attribute] >= cut_off].copy(deep=True)

                #print(cut_off, len(left_samples), len(right_samples))

                num_plus_left = np.sum(left_samples[label_attribute] == 1)
                num_minus_left = np.sum(left_samples[label_attribute] == 0)
                num_plus_right = np.sum(right_samples[label_attribute] == 1)
                num_minus_right = np.sum(right_samples[label_attribute] == 0)

                if num_plus_left == 0 and num_minus_left == 0:
                    continue    
                if num_plus_right == 0 and num_minus_right == 0:
                    continue

                score = split_score(num_plus_left, num_minus_left, num_plus_right, num_minus_right)
                all_scores.append((attribute, cut_off, score, num_plus_left, num_minus_left, num_plus_right, num_minus_right))
                all_stats.append(SplitStats(attribute, cut_off, num_plus_left, num_minus_left, num_plus_right, num_minus_right))

                if score > max_score:
                    max_score = score
                    the_attribute = attribute
                    the_cutoff = cut_off
                    the_left_samples = left_samples
                    the_right_samples = right_samples     
                    best_stats = SplitStats(attribute, cut_off, num_plus_left, num_minus_left, num_plus_right, num_minus_right)       
        
        # Try once more
        if max_score == -1:
            if num_tries == 25:
                #print(f"Giving up split selection for {len(samples)} samples")
                return None, None, None, None, None, None, None, None
            else:    
                return self.split(samples, attribute_candidates, label_attribute, constant_attributes, node_id, num_tries + 1)

        branching_candidates = []        

        #print("Split inspection")
        min_robustness = None
        for other_stats in all_stats:
            if best_stats != other_stats:
                curr_robustness = robustness(best_stats, other_stats, samples, label_attribute)

                if curr_robustness <= self.target_robustness:

                    if num_tries < 25:
                        return self.split(samples, attribute_candidates, label_attribute, constant_attributes, node_id, num_tries + 1)
                    else:    
                        fraction = (float(len(samples)) / self.num_training_samples) * 100
                        print(f"Non-robust split ({curr_robustness}) detected at {node_id} on {fraction:.2f}% of the data: {best_stats} VS {other_stats}") 
                        branching_candidates.append(other_stats)

                if min_robustness is None or curr_robustness < min_robustness:
                    min_robustness = curr_robustness
             
        # if min_robustness is not None and min_robustness < 3:         
        #     print('attribute:', the_attribute, ', cut_off:',the_cutoff, 
        #           ', robustness:', min_robustness, ', num samples:', len(samples))          

        return the_attribute, the_cutoff, the_left_samples, the_right_samples, max_score, all_scores, best_stats, branching_candidates       



    def stop_split(self, samples, attribute_candidates, label_attribute, known_constant_attributes):

        constant_attributes = set()
        for known_constant_attribute in known_constant_attributes:
            constant_attributes.add(known_constant_attribute)

        for attribute in attribute_candidates:            
            if attribute not in known_constant_attributes and len(samples[attribute].unique()) == 1:
                constant_attributes.add(attribute)
                
        if len(constant_attributes) == len(attribute_candidates):
            return True, constant_attributes                

        if len(samples) <= self.n_min:
            return True, constant_attributes
        
        if len(samples[label_attribute].unique()) == 1:
            return True, constant_attributes
        
        return False, constant_attributes


    def split_node(self, samples, attribute_candidates, label_attribute, known_constant_attributes, node_id):
        
        should_stop, updated_constant_attributes = \
            self.stop_split(samples, attribute_candidates, label_attribute, known_constant_attributes)

        if should_stop:                      
            return Leaf(num_positive=np.sum(samples[label_attribute]), 
                        num_samples=len(samples[label_attribute]))

        attribute, cut_off, left_samples, right_samples, score, all_scores, best_stats, branching_candidates = \
            self.split(samples, attribute_candidates, label_attribute, updated_constant_attributes, node_id)
        
        if score == 0.0 or attribute is None:            
            return Leaf(num_positive=np.sum(samples[label_attribute]), 
                        num_samples=len(samples[label_attribute]))  

        left_child = self.split_node(left_samples, attribute_candidates, label_attribute, 
                                     updated_constant_attributes, node_id + "0")        
        right_child = self.split_node(right_samples, attribute_candidates, label_attribute, 
                                      updated_constant_attributes, node_id + "1")
        

        if len(branching_candidates) > 0:


            branching_splits = []
            for branching_candidate in branching_candidates:

                print("Building alternate subtree at ", node_id)

                attribute_branch = branching_candidate.attribute
                cut_off_branch = branching_candidate.cut_off

                # This duplicates work already done...
                left_samples_branch = samples[samples[attribute_branch] < cut_off_branch].copy(deep=True)
                right_samples_branch = samples[samples[attribute_branch] >= cut_off_branch].copy(deep=True)

                left_child_branch = self.split_node(left_samples_branch, attribute_candidates, 
                                                    label_attribute, updated_constant_attributes, 
                                                    node_id + "0")        
                right_child_branch = self.split_node(right_samples_branch, attribute_candidates, 
                                                     label_attribute, updated_constant_attributes, 
                                                     node_id + "1")

                # TODO could there be cases where we need to branch twice?
                branching_split = Split(attribute_branch, cut_off_branch, left_child_branch, 
                    right_child_branch)

                branching_splits.append(branching_split)
                # TODO link into tree        

            return Split(attribute, cut_off, left_child, right_child, 
                         stats=best_stats, branching_candidates=branching_candidates, 
                         branching_splits=branching_splits)    



        else:    
            return Split(attribute, cut_off, left_child, right_child)    

    def fit(self, data, attribute_candidates, label_attribute):

        self.num_training_samples = len(data)
        self.target_robustness = np.max([int(self.num_training_samples / 1000), 1])

        for attribute in attribute_candidates:
            self.percentiles[attribute] = np.percentile(data[attribute].values, range(5, 100, 5))

            
        trees = []
        for num_trees_fitted in range(0, self.num_trees):
            trees.append(self.split_node(data, attribute_candidates, label_attribute, set(), "0"))       
            print(num_trees_fitted + 1, " trees fitted with target robustness " + str(self.target_robustness))

        return Trees(trees)                
