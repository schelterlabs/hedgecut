import numpy as np
from copy import deepcopy

class SplitStats:
    def __init__(self, attribute, cut_off, num_plus_left, num_minus_left, num_plus_right, num_minus_right):
        self.attribute = attribute
        self.cut_off = cut_off
        self.num_plus_left = num_plus_left
        self.num_minus_left = num_minus_left
        self.num_plus_right = num_plus_right
        self.num_minus_right = num_minus_right

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ 

    def __str__(self):
        return f"{self.attribute}: [{self.num_plus_left},{self.num_minus_left},{self.num_plus_right},{self.num_minus_right}]"   

def H(a, b, a_plus_b):    
    assert a >= 0   
    assert b >= 0   

    if a == 0 and b == 0:
        return 0.0
    
    p_a = float(a) / float(a_plus_b)
    p_b = float(b) / float(a_plus_b)
    
    log2_p_a = 0.0
    if a != 0:
        log2_p_a = np.log2(p_a)
        
    log2_p_b = 0.0
    if b != 0:
        log2_p_b = np.log2(p_b)        
        
    return -(p_a * log2_p_a + p_b * log2_p_b)        

def split_score_diff(stats_a, stats_b):
    assert stats_a is not None
    assert stats_b is not None


    return split_score(stats_a.num_plus_left, stats_a.num_minus_left, stats_a.num_plus_right, 
        stats_a.num_minus_right) - split_score(stats_b.num_plus_left, stats_b.num_minus_left, 
        stats_b.num_plus_right, stats_b.num_minus_right)

def split_score_gini(num_plus_left, num_minus_left, num_plus_right, num_minus_right):
    assert num_plus_left >= 0
    assert num_minus_left >= 0
    assert num_plus_right >= 0
    assert num_minus_left >= 0

    if num_plus_left == 0 and num_minus_left == 0:
        return 0.

    if num_plus_right == 0 and num_minus_right == 0:
        return 0.

    p1 = float(num_plus_left) / (num_plus_left + num_minus_left)
    p2 = float(num_plus_right) / (num_minus_right + num_minus_right)

    return 1. - ((p1 * (1-p1) + (p2 * (1 - p2))))  


def split_score(num_plus_left, num_minus_left, num_plus_right, num_minus_right):
    assert num_plus_left >= 0
    assert num_minus_left >= 0
    assert num_plus_right >= 0
    assert num_minus_left >= 0

    num_left = num_plus_left + num_minus_left
    num_right = num_plus_right + num_minus_right
    num_plus = num_plus_left + num_plus_right
    num_minus = num_minus_left + num_minus_right

    num_samples = num_left + num_right

    # Prior "classification entropy" H_C(S)
    hcs = H(num_plus, num_minus, num_samples)

    # Entropy of S with respect to test T H_T(S)
    hts = H(num_left, num_right, num_samples)

    # Posterior "classification entropy" H_{C|T}(S) of S given the outcome of the test T
    p_sys = float(num_left) / float(num_samples)
    p_sns = float(num_right) / float(num_samples)

    hcsy = H(num_plus_left, num_minus_left, num_left)
    hcsn = H(num_plus_right, num_minus_right, num_right)

    hcts = p_sys * hcsy + p_sns * hcsn;

    # Information gain of applying test T
    icts = hcs - hcts;



    if hcts + hts == 0:
        return 0.0

    score = 2.0 * icts / (hcs + hts);
    
    return score    


def weaken_split(current_champion_stats, current_runnerup_stats, current_budget):

    best_candidate = None
    next_stats_version = (None, None, None)
    score_diff_to_beat = split_score_diff(current_champion_stats, current_runnerup_stats)

    for is_positive in [True, False]:
        for passes_first in [True, False]:
            for passes_second in [True, False]:

                champion_stats = deepcopy(current_champion_stats)
                runnerup_stats = deepcopy(current_runnerup_stats)
                budget = deepcopy(current_budget)

                is_negative = not is_positive    

                if is_positive and passes_first and passes_second:
                    if champion_stats.num_plus_left == 0 or runnerup_stats.num_plus_left == 0 \
                        or budget.positive_first_second == 0:
                        continue
                    else:
                        champion_stats.num_plus_left -= 1                            
                        runnerup_stats.num_plus_left -= 1
                        budget.positive_first_second -= 1


                if is_positive and not passes_first and passes_second:
                    if champion_stats.num_plus_right == 0 or runnerup_stats.num_plus_left == 0 \
                        or budget.positive_notfirst_second == 0:
                        continue                                    
                    else:
                        champion_stats.num_plus_right -= 1                            
                        runnerup_stats.num_plus_left -= 1
                        budget.positive_notfirst_second -= 1


                if is_positive and passes_first and not passes_second:
                    if champion_stats.num_plus_left == 0 or runnerup_stats.num_plus_right == 0 \
                        or budget.positive_first_notsecond == 0:
                        continue
                    else:
                        champion_stats.num_plus_left -= 1                            
                        runnerup_stats.num_plus_right -= 1
                        budget.positive_first_notsecond -= 1

                if is_positive and not passes_first and not passes_second:
                    if champion_stats.num_plus_right == 0 or runnerup_stats.num_plus_right == 0 \
                        or budget.positive_notfirst_notsecond == 0:
                        continue
                    else:
                        champion_stats.num_plus_right -= 1                            
                        runnerup_stats.num_plus_right -= 1
                        budget.positive_notfirst_notsecond -= 1


                if is_negative and passes_first and passes_second:
                    if champion_stats.num_plus_left == 0 or runnerup_stats.num_plus_left == 0 \
                        or budget.negative_first_second == 0:
                        continue
                    else:
                        champion_stats.num_plus_left -= 1                            
                        runnerup_stats.num_plus_left -= 1
                        budget.negative_first_second -= 1


                if is_negative and not passes_first and passes_second:
                    if champion_stats.num_plus_right == 0 or runnerup_stats.num_plus_left == 0 \
                        or budget.negative_notfirst_second == 0:
                        continue                                    
                    else:
                        champion_stats.num_plus_right -= 1                            
                        runnerup_stats.num_plus_left -= 1
                        budget.negative_notfirst_second -= 1


                if is_negative and passes_first and not passes_second:
                    if champion_stats.num_plus_left == 0 or runnerup_stats.num_plus_right == 0 \
                        or budget.negative_first_notsecond == 0:
                        continue
                    else:
                        champion_stats.num_plus_left -= 1                            
                        runnerup_stats.num_plus_right -= 1
                        budget.negative_first_notsecond -= 1

                if is_negative and not passes_first and not passes_second:
                    if champion_stats.num_plus_right == 0 or runnerup_stats.num_plus_right == 0 \
                        or budget.negative_notfirst_notsecond == 0:
                        continue
                    else:
                        champion_stats.num_plus_right -= 1                            
                        runnerup_stats.num_plus_right -= 1
                        budget.negative_notfirst_notsecond -= 1

                # if is_positive:

                #     if passes_first:
                #         if champion_stats.num_plus_left == 0:
                #             continue
                #         else:     
                #             champion_stats.num_plus_left -= 1 
                #     else:   
                #         if champion_stats.num_plus_right == 0:
                #             continue
                #         else:
                #             champion_stats.num_plus_right -= 1 

                #     if passes_second:
                #         if runnerup_stats.num_plus_left == 0:
                #             continue
                #         else: 
                #             runnerup_stats.num_plus_left -= 1 
                #     else:
                #         if runnerup_stats.num_plus_right == 0:
                #             continue
                #         else:   
                #             runnerup_stats.num_plus_right -= 1 
                    
                # else:       

                #     if passes_first:
                #         if champion_stats.num_minus_left == 0:
                #             continue
                #         else:
                #             champion_stats.num_minus_left -= 1 
                #     else:   
                #         if champion_stats.num_minus_right == 0:
                #             continue
                #         else:
                #             champion_stats.num_minus_right -= 1 

                #     if passes_second:
                #         if runnerup_stats.num_minus_left == 0:
                #             continue
                #         else:
                #             runnerup_stats.num_minus_left -= 1 
                #     else:   
                #         if runnerup_stats.num_minus_right == 0:
                #             continue
                #         else:
                #             runnerup_stats.num_minus_right -= 1                 

                score_diff = split_score_diff(champion_stats, runnerup_stats)

                if score_diff < score_diff_to_beat:
                    score_diff_to_beat = score_diff
                    best_candidate = (is_positive, passes_first, passes_second)     
                    next_stats_version = (deepcopy(champion_stats), deepcopy(runnerup_stats), deepcopy(budget))   

    #print(best_candidate, score_diff_to_beat)               
    stop_now = False
    if next_stats_version[0] is None:
        stop_now = True
    #assert next_stats_version[0] is not None
    #assert next_stats_version[1] is not None

    return next_stats_version[0], next_stats_version[1], score_diff_to_beat, stop_now, best_candidate, next_stats_version[2]

class RemovalBudget():
    def __init__(self, positive_first_second, positive_notfirst_second, positive_first_notsecond, 
                 positive_notfirst_notsecond, negative_first_second, negative_notfirst_second, 
                 negative_first_notsecond, negative_notfirst_notsecond):

        self.positive_first_second = positive_first_second
        self.positive_notfirst_second = positive_notfirst_second
        self.positive_first_notsecond = positive_first_notsecond
        self.positive_notfirst_notsecond = positive_notfirst_notsecond
        self.negative_first_second = negative_first_second
        self.negative_notfirst_second = negative_notfirst_second
        self.negative_first_notsecond = negative_first_second
        self.negative_notfirst_notsecond = negative_notfirst_notsecond


def robustness(champion_stats, runnerup_stats, samples, label_attribute):
    assert champion_stats is not None
    assert runnerup_stats is not None

    positive = samples[(samples[label_attribute] == 1)]
    negative = samples[(samples[label_attribute] == 0)]
    att1 = champion_stats.attribute
    c1 = champion_stats.cut_off
    att2 = runnerup_stats.attribute
    c2 = runnerup_stats.cut_off

    positive_first_second = len(positive[(positive[att1] <= c1) & (positive[att2] <= c2)])
    positive_notfirst_second = len(positive[(positive[att1] > c1) & (positive[att2] <= c2)])
    positive_first_notsecond = len(positive[(positive[att1] <= c1) & (positive[att2] > c2)])
    positive_notfirst_notsecond = len(positive[(positive[att1] > c1) & (positive[att2] > c2)])
    negative_first_second = len(negative[(negative[att1] <= c1) & (negative[att2] <= c2)])
    negative_notfirst_second = len(negative[(negative[att1] > c1) & (negative[att2] <= c2)])
    negative_first_notsecond = len(negative[(negative[att1] <= c1) & (negative[att2] > c2)])
    negative_notfirst_notsecond = len(negative[(negative[att1] > c1) & (negative[att2] > c2)])

    budget = RemovalBudget(positive_first_second, positive_notfirst_second, positive_first_notsecond, 
                 positive_notfirst_notsecond, negative_first_second, negative_notfirst_second, 
                 negative_first_notsecond, negative_notfirst_notsecond)

    points_to_remove = []

    rounds = 0
    score_diff = None

    while True:
        # add budget here
        champion_stats, runnerup_stats, score_diff, stop_now, point_to_remove, budget = \
            weaken_split(champion_stats, runnerup_stats, budget)

        points_to_remove.append(point_to_remove)
        rounds += 1

        if score_diff <= 0 or stop_now:
            break

    return len(points_to_remove)