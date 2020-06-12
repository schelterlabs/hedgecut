from copy import deepcopy
import numpy as np

class SplitStats:
	def __init__(self, num_plus_left, num_minus_left, num_plus_right, num_minus_right):
		self.num_plus_left = num_plus_left
		self.num_minus_left = num_minus_left
		self.num_plus_right = num_plus_right
		self.num_minus_right = num_minus_right

def H(a, b, a_plus_b):    
   
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


def split_score(stats):

    num_left = stats.num_plus_left + stats.num_minus_left
    num_right = stats.num_plus_right + stats.num_minus_right
    num_plus = stats.num_plus_left + stats.num_plus_right
    num_minus = stats.num_minus_left + stats.num_minus_right

    num_samples = num_left + num_right

    # Prior "classification entropy" H_C(S)
    hcs = H(num_plus, num_minus, num_samples)

    # Entropy of S with respect to test T H_T(S)
    hts = H(num_left, num_right, num_samples)

    # Posterior "classification entropy" H_{C|T}(S) of S given the outcome of the test T
    p_sys = float(num_left) / float(num_samples)
    p_sns = float(num_right) / float(num_samples)

    hcsy = H(stats.num_plus_left, stats.num_minus_left, num_left)
    hcsn = H(stats.num_plus_right, stats.num_minus_right, num_right)

    hcts = p_sys * hcsy + p_sns * hcsn;

    # Information gain of applying test T
    icts = hcs - hcts;

    score = 2.0 * icts / (hcs + hts);
    
    return score    




def weaken_split(current_champion_stats, current_runnerup_stats):

	best_candidate = None
	next_stats_version = (None, None)
	score_diff_to_beat = split_score(current_champion_stats) - split_score(current_runnerup_stats)

	for is_positive in [True, False]:
		for passes_first in [True, False]:
			for passes_second in [True, False]:

				champion_stats = deepcopy(current_champion_stats)
				runnerup_stats = deepcopy(current_runnerup_stats)

				if is_positive:

					if passes_first:
						champion_stats.num_plus_left -= 1 
					else:	
						champion_stats.num_plus_right -= 1 

					if passes_second:
						runnerup_stats.num_plus_left -= 1 
					else:	
						runnerup_stats.num_plus_right -= 1 
					
				else:		

					if passes_first:
						champion_stats.num_minus_left -= 1 
					else:	
						champion_stats.num_minus_right -= 1 

					if passes_second:
						runnerup_stats.num_minus_left -= 1 
					else:	
						runnerup_stats.num_minus_right -= 1 				

				score_diff = split_score(champion_stats) - split_score(runnerup_stats)

				if score_diff < score_diff_to_beat:
					score_diff_to_beat = score_diff
					best_candidate = (is_positive, passes_first, passes_second)		
					next_stats_version = (deepcopy(champion_stats), deepcopy(runnerup_stats))	

	print(best_candidate, score_diff_to_beat)				



	return next_stats_version[0], next_stats_version[1], score_diff_to_beat, best_candidate


def robustness(champion_stats, runnerup_stats):
	points_to_remove = []

	while True:
		champion_stats, runnerup_stats, score_diff, point_to_remove = weaken_split(champion_stats, runnerup_stats)
		points_to_remove.append(point_to_remove)
		if score_diff < 0:
			break

	print('Split decision broken after removal of', len(points_to_remove), 
		  'points with characteristics', points_to_remove)


robustness(SplitStats(20, 113, 4, 42), SplitStats(5, 35, 19, 120))
robustness(SplitStats(19, 148, 1, 5), SplitStats(3, 22, 17, 131))
robustness(SplitStats(5, 44, 1, 15), SplitStats(1, 11, 5, 48))

robustness(SplitStats(842, 5012, 546, 114), SplitStats(317, 1864, 1071, 3262))

