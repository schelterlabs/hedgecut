
#[derive(Debug, Clone)]
pub struct SplitStats {
    pub num_plus_left: u32,
    pub num_minus_left: u32,
    pub num_plus_right: u32,
    pub num_minus_right: u32,
    pub impurity_left: f64,
    pub impurity_right: f64,
    pub score: i64,
}

impl SplitStats {

    pub fn new(
        num_plus_left: u32,
        num_minus_left: u32,
        num_plus_right: u32,
        num_minus_right: u32,
    ) -> SplitStats {
        SplitStats {
            num_plus_left,
            num_minus_left,
            num_plus_right,
            num_minus_right,
            impurity_left: 0.0,
            impurity_right: 0.0,
            score: 0
        }
    }

    pub fn update_score(&mut self, impurity_before: f64) {
        let (score, impurity_left, impurity_right) = gini(
            impurity_before,
            self.num_plus_left,
            self.num_minus_left,
            self.num_plus_right,
            self.num_minus_right
        );

        self.score = score;
        self.impurity_left = impurity_left;
        self.impurity_right = impurity_right;
    }

    pub fn update_score_and_impurity_before(&mut self) {
        let (score, impurity_left, impurity_right) = gini_with_impurity_before(
            self.num_plus_left,
            self.num_minus_left,
            self.num_plus_right,
            self.num_minus_right
        );

        self.score = score;
        self.impurity_left = impurity_left;
        self.impurity_right = impurity_right;
    }
}

pub fn is_robust(
    current_champion_stats: &SplitStats,
    current_runnerup_stats: &SplitStats,
    threshold: usize
) -> (bool, usize) {

    let mut weakened_stats = vec![(current_champion_stats.clone(), current_runnerup_stats.clone())];
    let mut num_removals = 0;

    let mut is_robust = true;

    //println!("Starting robustness check");

    loop {

        // We could not find a way to decrease the split score difference with
        // the given number of removal operations
        if weakened_stats.is_empty() || num_removals > threshold {
            break;
        }

        for (stats_a, stats_b) in &weakened_stats {
            // Runner up beats champion, split not robust
            if (stats_a.score as f64 - stats_b.score as f64) < 0.0 {
                is_robust = false;
                break;
            }
        }

        if num_removals == threshold {
            break;
        }

        let mut next_weakened_stats = Vec::new();

        for (stats_a, stats_b) in &weakened_stats {
            next_weakened_stats.extend(weaken_split(stats_a, stats_b));
        }

        let min_score_diff = next_weakened_stats.iter()
            .map(|(stats_a, stats_b)| stats_a.score - stats_b.score)
            .min();

        weakened_stats = next_weakened_stats.into_iter()
            .filter(|(stats_a, stats_b)| {
                (stats_a.score - stats_b.score) == min_score_diff.unwrap()
            })
            .collect();

        //eprintln!("{},{},{}", weakened_stats.len(), threshold, num_removals);

        num_removals += 1;
    };

    (is_robust, num_removals)
}

fn weaken_split(
    current_champion_split_stats: &SplitStats,
    current_runnerup_split_stats: &SplitStats,
) -> Vec<(SplitStats, SplitStats)> {
    let truefalse = [true, false];

    let mut weakest_pairs: Vec<(SplitStats, SplitStats)> = Vec::new();

    let mut score_diff_to_beat =
        current_champion_split_stats.score as f64 - current_runnerup_split_stats.score as f64;

    for is_plus in truefalse.iter() {
        for passes_first in truefalse.iter() {
            for passes_second in truefalse.iter() {
                let mut champion_stats = current_champion_split_stats.clone();
                let mut runnerup_stats = current_runnerup_split_stats.clone();

                if *is_plus && *passes_first && *passes_second {
                    if champion_stats.num_plus_left != 0 && runnerup_stats.num_plus_left != 0 {
                        champion_stats.num_plus_left -= 1;
                        runnerup_stats.num_plus_left -= 1;
                    } else {
                        continue;
                    }
                } else if *is_plus && !*passes_first && *passes_second {
                    if champion_stats.num_plus_right != 0 && runnerup_stats.num_plus_left != 0 {
                        champion_stats.num_plus_right -= 1;
                        runnerup_stats.num_plus_left -= 1;
                    } else {
                        continue;
                    }
                } else if *is_plus && *passes_first && !*passes_second {
                    if champion_stats.num_plus_left != 0 && runnerup_stats.num_plus_right != 0 {
                        champion_stats.num_plus_left -= 1;
                        runnerup_stats.num_plus_right -= 1;
                    } else {
                        continue;
                    }
                } else if *is_plus && !*passes_first && !*passes_second {
                    if champion_stats.num_plus_right != 0 && runnerup_stats.num_plus_right != 0 {
                        champion_stats.num_plus_right -= 1;
                        runnerup_stats.num_plus_right -= 1;
                    } else {
                        continue;
                    }
                } else if !*is_plus && *passes_first && *passes_second {
                    if champion_stats.num_minus_left != 0 && runnerup_stats.num_minus_left != 0 {
                        champion_stats.num_minus_left -= 1;
                        runnerup_stats.num_minus_left -= 1;
                    } else {
                        continue;
                    }
                } else if !*is_plus && !*passes_first && *passes_second {
                    if champion_stats.num_minus_right != 0 && runnerup_stats.num_minus_left != 0 {
                        champion_stats.num_minus_right -= 1;
                        runnerup_stats.num_minus_left -= 1;
                    } else {
                        continue;
                    }
                } else if !*is_plus && *passes_first && !*passes_second {
                    if champion_stats.num_minus_left != 0 && runnerup_stats.num_minus_right != 0 {
                        champion_stats.num_minus_left -= 1;
                        runnerup_stats.num_minus_right -= 1;
                    } else {
                        continue;
                    }
                } else if !*is_plus && !*passes_first && !*passes_second {
                    if champion_stats.num_minus_right != 0 && runnerup_stats.num_minus_right != 0 {
                        champion_stats.num_minus_right -= 1;
                        runnerup_stats.num_minus_right -= 1;
                    } else {
                        continue;
                    }
                }

                champion_stats.update_score_and_impurity_before();
                runnerup_stats.update_score_and_impurity_before();

                let new_score_diff = champion_stats.score as f64 - runnerup_stats.score as f64;

                if new_score_diff == score_diff_to_beat {
                    weakest_pairs.push((champion_stats, runnerup_stats))
                } else if new_score_diff < score_diff_to_beat {
                    score_diff_to_beat = new_score_diff;
                    weakest_pairs.clear();
                    weakest_pairs.push((champion_stats, runnerup_stats));
                }
            }
        }
    }

    weakest_pairs
}

pub fn gini_with_impurity_before(
    num_plus_left: u32,
    num_minus_left: u32,
    num_plus_right: u32,
    num_minus_right: u32,
) -> (i64, f64, f64) {


    let num_plus = num_plus_left + num_plus_right;
    let num_minus = num_minus_left + num_minus_right;
    let num_samples = num_plus + num_minus;

    let p_plus = num_plus as f64 / num_samples as f64;

    let impurity_before = 2.0 * p_plus * (1.0 - p_plus);

    gini(
        impurity_before,
        num_plus_left,
        num_minus_left,
        num_plus_right,
        num_minus_right,
    )
}

#[inline(always)]
pub fn gini_impurity(num_plus: u32, num_samples: u32) -> f64 {
    let p_plus = num_plus as f64 / num_samples as f64;
    2.0 * p_plus * (1.0 - p_plus)
}

fn gini(
    impurity_before: f64,
    num_plus_left: u32,
    num_minus_left: u32,
    num_plus_right: u32,
    num_minus_right: u32
) -> (i64, f64, f64) {

    let num_samples_left = num_plus_left + num_minus_left;
    let num_samples_right = num_plus_right + num_minus_right;

    if num_samples_left == 0 || num_samples_right == 0 {
        return (-1, 0.0, 0.0);
    }

    let gini_left = gini_impurity(num_plus_left, num_samples_left);
    let gini_right = gini_impurity(num_plus_right, num_samples_right);

    let num_samples = num_samples_left + num_samples_right;

    let score = impurity_before -
        (num_samples_left as f64 / num_samples as f64) * gini_left -
        (num_samples_right as f64 / num_samples as f64) * gini_right;

    if score.is_nan() {
        println!("[{},{},{},{}]", num_plus_left, num_minus_left, num_plus_right, num_minus_right);
        println!("{}, {}, {}, {}", score, gini_left, gini_right, impurity_before);
        panic!("Invalid score encountered!");
    }

    ((score * 1_000_000_000_000_f64) as i64, gini_left, gini_right)
}
