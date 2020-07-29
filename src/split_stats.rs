
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

    let mut scratch_space = Vec::with_capacity(8);

    let mut candidates = Vec::new();

    let mut frontier = vec![(current_champion_stats.clone(), current_runnerup_stats.clone())];
    let mut current_minimal_score_diff =
        current_champion_stats.score - current_runnerup_stats.score;

    let mut num_removals = 0;
    let mut is_robust = true;

    //println!("Starting robustness check");

    loop {

        // We have been able to break the split
        if current_minimal_score_diff < 0 {
            //println!("broken");
            is_robust = false;
            break;
        }

        // We have not been able to break the split with the given number of removal operations
        if num_removals == threshold {
            //println!("not broken");
            break;
        }

        //println!("Frontier {}, {}", frontier.len(), current_minimal_score_diff);

        for (stats_a, stats_b) in &frontier {
            //println!("\tComparison {} vs {}", (stats_a.score - stats_b.score), current_minimal_score_diff);
            if stats_a.score - stats_b.score <= current_minimal_score_diff {
                scratch_space.clear();
                let score_diff_found = weaken_split(stats_a, stats_b, &mut scratch_space);
                // TODO we should only do this if these are good candidates
                candidates.extend(scratch_space.drain(..));
                //println!("Candidates {}", candidates.len());
                if score_diff_found <= current_minimal_score_diff {
                    current_minimal_score_diff = score_diff_found;
                }
            }
        }
        //println!("Candidates2 {}", candidates.len());

        // No improvements possible
        if candidates.is_empty() {
            //println!("No improvements possible");
            break;
        }

        frontier.clear();

        std::mem::swap(&mut frontier, &mut candidates);
        candidates.clear();

        num_removals += 1;
    };

    (is_robust, num_removals)
}

fn weaken_split(
    initial_champion: &SplitStats,
    initial_runnerup: &SplitStats,
    weakest_pairs: &mut Vec<(SplitStats, SplitStats)>
) -> i64 {
    let truefalse = [true, false];

    // TODO handle case where both scores are zero
    let mut score_diff_to_beat =
        initial_champion.score as f64 - initial_runnerup.score as f64;
    //
    // println!("Weaken ({},{},{},{}) vs ({},{},{},{}) with {}",
    //          initial_champion.num_plus_left,
    //          initial_champion.num_minus_left,
    //          initial_champion.num_plus_right,
    //          initial_champion.num_minus_right,
    //          initial_runnerup.num_plus_left,
    //          initial_runnerup.num_minus_left,
    //          initial_runnerup.num_plus_right,
    //          initial_runnerup.num_minus_right,
    //          (score_diff_to_beat * 1_000_000_000_000_f64) as i64
    // );

    // Stop if we produce a split which is constant on both sides
    if (initial_runnerup.num_minus_left == 0 && initial_runnerup.num_minus_right == 0) ||
       (initial_runnerup.num_plus_left == 0 && initial_runnerup.num_minus_right == 0) ||
       (initial_runnerup.num_minus_left == 0 && initial_runnerup.num_plus_right == 0) ||
       (initial_runnerup.num_plus_left == 0 && initial_runnerup.num_plus_right == 0) ||
        initial_runnerup.score < 0 {
        return (score_diff_to_beat * 1_000_000_000_000_f64) as i64
    }


    for is_plus in truefalse.iter() {
        for passes_first in truefalse.iter() {
            for passes_second in truefalse.iter() {
                let mut champion = initial_champion.clone();
                let mut runnerup = initial_runnerup.clone();

                if *is_plus && *passes_first && *passes_second {
                    if champion.num_plus_left != 0 && runnerup.num_plus_left != 0 {
                        champion.num_plus_left -= 1;
                        runnerup.num_plus_left -= 1;
                    } else {
                        continue;
                    }
                } else if *is_plus && !*passes_first && *passes_second {
                    if champion.num_plus_right != 0 && runnerup.num_plus_left != 0 {
                        champion.num_plus_right -= 1;
                        runnerup.num_plus_left -= 1;
                    } else {
                        continue;
                    }
                } else if *is_plus && *passes_first && !*passes_second {
                    if champion.num_plus_left != 0 && runnerup.num_plus_right != 0 {
                        champion.num_plus_left -= 1;
                        runnerup.num_plus_right -= 1;
                    } else {
                        continue;
                    }
                } else if *is_plus && !*passes_first && !*passes_second {
                    if champion.num_plus_right != 0 && runnerup.num_plus_right != 0 {
                        champion.num_plus_right -= 1;
                        runnerup.num_plus_right -= 1;
                    } else {
                        continue;
                    }
                } else if !*is_plus && *passes_first && *passes_second {
                    if champion.num_minus_left != 0 && runnerup.num_minus_left != 0 {
                        champion.num_minus_left -= 1;
                        runnerup.num_minus_left -= 1;
                    } else {
                        continue;
                    }
                } else if !*is_plus && !*passes_first && *passes_second {
                    if champion.num_minus_right != 0 && runnerup.num_minus_left != 0 {
                        champion.num_minus_right -= 1;
                        runnerup.num_minus_left -= 1;
                    } else {
                        continue;
                    }
                } else if !*is_plus && *passes_first && !*passes_second {
                    if champion.num_minus_left != 0 && runnerup.num_minus_right != 0 {
                        champion.num_minus_left -= 1;
                        runnerup.num_minus_right -= 1;
                    } else {
                        continue;
                    }
                } else if !*is_plus && !*passes_first && !*passes_second {
                    if champion.num_minus_right != 0 && runnerup.num_minus_right != 0 {
                        champion.num_minus_right -= 1;
                        runnerup.num_minus_right -= 1;
                    } else {
                        continue;
                    }
                }

                champion.update_score_and_impurity_before();
                runnerup.update_score_and_impurity_before();

                let new_score_diff = champion.score as f64 - runnerup.score as f64;

                //println!("{} vs {}", score_diff_to_beat, new_score_diff);

                if new_score_diff == score_diff_to_beat {
                    weakest_pairs.push((champion, runnerup))
                } else if new_score_diff < score_diff_to_beat {
                    score_diff_to_beat = new_score_diff;
                    weakest_pairs.clear();
                    weakest_pairs.push((champion, runnerup));
                }
            }
        }
    }

    //println!("Returning {} weakest pairs", weakest_pairs.len());

    return (score_diff_to_beat * 1_000_000_000_000_f64) as i64
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


#[cfg(test)]
mod tests {

    use crate::split_stats::{SplitStats, is_robust};

    fn evaluate3(s: &mut SplitStats, t: &mut SplitStats) {

        s.update_score_and_impurity_before();
        t.update_score_and_impurity_before();

        let mut diffs1 = Vec::new();
        let mut diffs2 = Vec::new();
        let mut diffs3 = Vec::new();

        let mut diff_paths = Vec::new();

        let enumerated = weaken_split_dbg(&s, &t);

        for (_step, s_hat, t_hat) in enumerated {
            let score_diff = s_hat.score - t_hat.score;
            let enumerated_again = weaken_split_dbg(&s_hat, &t_hat);

            diffs1.push(score_diff);

            for (_step, s_hat, t_hat) in enumerated_again {
                let score_diff_again = s_hat.score - t_hat.score;

                diffs2.push(score_diff_again);

                let enumerated_again_again = weaken_split_dbg(&s_hat, &t_hat);

                for (_step, s_hat, t_hat) in enumerated_again_again {
                    let score_diff_again_again = s_hat.score - t_hat.score;

                    diffs3.push(score_diff_again_again);

                    diff_paths.push((score_diff, score_diff_again, score_diff_again_again));
                }
            }
        }

        let mins_path = (
            *diffs1.iter().min().unwrap(),
            *diffs2.iter().min().unwrap(),
            *diffs3.iter().min().unwrap()
        );

        println!("{:?}", mins_path);
    }

    #[derive(Eq,PartialEq,Debug)]
    struct Step {
        label: bool,
        passes_first: bool,
        passes_second: bool,
    }

    fn weaken_split_dbg(
        current_champion_split_stats: &SplitStats,
        current_runnerup_split_stats: &SplitStats,
    ) -> Vec<(Step, SplitStats, SplitStats)> {
        let truefalse = [true, false];

        let mut enumerated = Vec::new();

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

                    let step = Step {
                        label: *is_plus,
                        passes_first: *passes_first,
                        passes_second: *passes_second
                    };

                    enumerated.push((step, champion_stats, runnerup_stats));
                }
            }
        }

        enumerated
    }

    #[test]
    fn examples() {

        let mut s = SplitStats::new(8, 6, 12, 2);
        let mut t = SplitStats::new(10, 3, 10, 3);

        s.update_score_and_impurity_before();
        t.update_score_and_impurity_before();

        let (split_robust, num_steps) = is_robust(&s, &t, 3);

        assert!(!split_robust);
        assert_eq!(num_steps, 3);
        evaluate3(&mut s, &mut t);


        let mut s = SplitStats::new(7, 5, 13, 3);
        let mut t = SplitStats::new(10, 3, 10, 3);

        s.update_score_and_impurity_before();
        t.update_score_and_impurity_before();

        let (split_robust, num_steps) = is_robust(&s, &t, 3);

        assert!(!split_robust);
        assert_eq!(num_steps, 2);
        evaluate3(&mut s, &mut t);


        let mut s = SplitStats::new(70, 50, 130, 30);
        let mut t = SplitStats::new(10, 3, 10, 3);

        s.update_score_and_impurity_before();
        t.update_score_and_impurity_before();

        let (split_robust, num_steps) = is_robust(&s, &t, 3);

        assert!(!split_robust);
        assert_eq!(num_steps, 3);
        evaluate3(&mut s, &mut t);


        let mut s = SplitStats::new(2, 2, 1, 0);
        let mut t = SplitStats::new(1, 1, 2, 1);

        s.update_score_and_impurity_before();
        t.update_score_and_impurity_before();

        let (split_robust, num_steps) = is_robust(&s, &t, 3);

        assert!(!split_robust);
        assert_eq!(num_steps, 1);
        evaluate3(&mut s, &mut t);
    }
}