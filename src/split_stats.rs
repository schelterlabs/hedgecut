
#[derive(Debug, Clone, Copy)]
pub struct SplitStats {
    pub num_plus_left: u32,
    pub num_minus_left: u32,
    pub num_plus_right: u32,
    pub num_minus_right: u32,
    pub impurity_left: f64,
    pub impurity_right: f64,
    pub score: Option<i64>,
}

impl SplitStats {

    pub fn fmt(&self) -> String {
        format!("({},{},{},{})", self.num_plus_left, self.num_minus_left, self.num_plus_right, self.num_minus_right)
    }

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
            score: None
        }
    }

    pub fn has_positive_score(&self) -> bool {
        match self.score {
            Some(the_score) => the_score > 0,
            _ => false,
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

fn h2(s: &SplitStats, t: &SplitStats, r: u32) -> bool {
    let threshold = r;

    // We reject these to avoid false positives
    if s.num_plus_left <= threshold || s.num_plus_right <= threshold ||
        t.num_plus_left <= threshold || t.num_plus_right <= threshold ||
        s.num_minus_left <= threshold || s.num_minus_right <= threshold ||
        t.num_minus_left <= threshold || t.num_minus_right <= threshold {

        false
    } else {
        true
    }
}

pub fn is_robust(
    current_champion_stats: &SplitStats,
    current_runnerup_stats: &SplitStats,
    threshold: usize
) -> (bool, usize) {

    assert!(current_champion_stats.has_positive_score());
    assert!(current_runnerup_stats.has_positive_score());

    if !h2(current_champion_stats, current_runnerup_stats, threshold as u32) {
        return (false, threshold)
    }

    let (robust_via_heuristic, _) = heuristic(current_champion_stats, current_runnerup_stats, threshold as u32);

    if !robust_via_heuristic {
        return (false, threshold)
    }

    let (is_robust, num_removals,_) = is_robust2(current_champion_stats, current_runnerup_stats, threshold, false);


    (is_robust, num_removals)
}

pub fn is_robust2(
    current_champion_stats: &SplitStats,
    current_runnerup_stats: &SplitStats,
    threshold: usize,
    _dbg: bool,
) -> (bool, usize, i64) {

    assert!(current_champion_stats.has_positive_score());
    assert!(current_runnerup_stats.has_positive_score());

    let mut scratch_space = Vec::with_capacity(8);

    let mut candidates = Vec::new();

    let mut frontier = vec![(current_champion_stats.clone(), current_runnerup_stats.clone())];
    let mut current_minimal_score_diff =
        current_champion_stats.score.unwrap() - current_runnerup_stats.score.unwrap();

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
            if stats_a.score.unwrap() - stats_b.score.unwrap() <= current_minimal_score_diff {
                scratch_space.clear();
                let score_diff_found = weaken_split(stats_a, stats_b, &mut scratch_space);
                candidates.extend(scratch_space.drain(..));
                //println!("Candidates {}", candidates.len());
                if score_diff_found <= current_minimal_score_diff {
                    current_minimal_score_diff = score_diff_found;
                }
            }
        }

        // if dbg {
        //     println!("GREEDY [{}]", num_removals + 1);
        //     for (s,t) in &candidates {
        //         println!("{} {} {}", s.fmt(), t.fmt(), (s.score - t.score) as f64 / 1_000_000_000_000_f64);
        //     }
        //     println!("----");
        // }
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

    (is_robust, num_removals, current_minimal_score_diff)
}

fn weaken_split(
    initial_champion: &SplitStats,
    initial_runnerup: &SplitStats,
    weakest_pairs: &mut Vec<(SplitStats, SplitStats)>
) -> i64 {
    let truefalse = [true, false];

    let mut score_diff_to_beat =
        initial_champion.score.unwrap() as f64 - initial_runnerup.score.unwrap() as f64;
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
        initial_runnerup.score.is_none() {

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

                let new_score_diff = champion.score.unwrap() as f64 - runnerup.score.unwrap() as f64;

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
) -> (Option<i64>, f64, f64) {


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
) -> (Option<i64>, f64, f64) {

    let num_samples_left = num_plus_left + num_minus_left;
    let num_samples_right = num_plus_right + num_minus_right;

    // We don't want such splits
    if num_samples_left == 0 || num_samples_right == 0 {
        return (None, 0.0, 0.0);
        //return (0, 0.0, 0.0);
    }

    let gini_left = gini_impurity(num_plus_left, num_samples_left);
    let gini_right = gini_impurity(num_plus_right, num_samples_right);

    let num_samples = num_samples_left + num_samples_right;

    let score = impurity_before -
        (num_samples_left as f64 / num_samples as f64) * gini_left -
        (num_samples_right as f64 / num_samples as f64) * gini_right;

    // if score.is_nan() {
    //     println!("[{},{},{},{}]", num_plus_left, num_minus_left, num_plus_right, num_minus_right);
    //     println!("{}, {}, {}, {}", score, gini_left, gini_right, impurity_before);
    //     panic!("Invalid score encountered!");
    // }

    (Some((score * 1_000_000_000_000_f64) as i64), gini_left, gini_right)
}

pub fn fmt_score(s: i64) -> f64 {
    s as f64 / 1_000_000_000_000_f64
}



pub fn heuristic(s: &SplitStats, t: &SplitStats, r: u32) -> (bool, Option<(SplitStats, SplitStats)>) {

    if s.num_plus_left >= r && t.num_plus_right >= r {
        let mut s_plus_a = s.clone();
        let mut t_plus_a = t.clone();

        s_plus_a.num_plus_left -= r;
        t_plus_a.num_plus_right -= r;

        s_plus_a.update_score_and_impurity_before();
        t_plus_a.update_score_and_impurity_before();

        if t_plus_a.score > s_plus_a.score {
            return (false, Some((s_plus_a, t_plus_a)));
        }
    }
    // ---

    if s.num_plus_right >= r && t.num_plus_left >= r {
        let mut s_plus_b = s.clone();
        let mut t_plus_b = t.clone();

        s_plus_b.num_plus_right -= r;
        t_plus_b.num_plus_left -= r;

        s_plus_b.update_score_and_impurity_before();
        t_plus_b.update_score_and_impurity_before();

        if t_plus_b.score > s_plus_b.score {
            return (false, Some((s_plus_b, t_plus_b)));
        }
    }
    // ---

    if s.num_plus_left >= r && t.num_plus_left >= r {
        let mut s_plus_c = s.clone();
        let mut t_plus_c = t.clone();

        s_plus_c.num_plus_left -= r;
        t_plus_c.num_plus_left -= r;

        s_plus_c.update_score_and_impurity_before();
        t_plus_c.update_score_and_impurity_before();

        if t_plus_c.score > s_plus_c.score {
            return (false, Some((s_plus_c, t_plus_c)));
        }
    }
    // ---

    if s.num_plus_right >= r && t.num_plus_right >= r {
        let mut s_plus_d = s.clone();
        let mut t_plus_d = t.clone();

        s_plus_d.num_plus_right -= r;
        t_plus_d.num_plus_right -= r;

        s_plus_d.update_score_and_impurity_before();
        t_plus_d.update_score_and_impurity_before();

        if t_plus_d.score > s_plus_d.score {
            return (false, Some((s_plus_d, t_plus_d)));
        }
    }
    // ---

    if s.num_minus_left >= r && t.num_minus_right >= r {
        let mut s_minus_a = s.clone();
        let mut t_minus_a = t.clone();

        s_minus_a.num_minus_left -= r;
        t_minus_a.num_minus_right -= r;

        s_minus_a.update_score_and_impurity_before();
        t_minus_a.update_score_and_impurity_before();

        if t_minus_a.score > s_minus_a.score {
            return (false, Some((s_minus_a, t_minus_a)));
        }
    }
    // ---

    if s.num_minus_right >= r && t.num_minus_left >= r {
        let mut s_minus_b = s.clone();
        let mut t_minus_b = t.clone();

        s_minus_b.num_minus_right -= r;
        t_minus_b.num_minus_left -= r;

        s_minus_b.update_score_and_impurity_before();
        t_minus_b.update_score_and_impurity_before();

        if t_minus_b.score > s_minus_b.score {
            return (false, Some((s_minus_b, t_minus_b)));
        }
    }
    // ---

    if s.num_minus_left >= r && t.num_minus_left >= r {
        let mut s_minus_c = s.clone();
        let mut t_minus_c = t.clone();

        s_minus_c.num_minus_left -= r;
        t_minus_c.num_minus_left -= r;

        s_minus_c.update_score_and_impurity_before();
        t_minus_c.update_score_and_impurity_before();

        if t_minus_c.score > s_minus_c.score {
            return (false, Some((s_minus_c, t_minus_c)));
        }
    }

    // ---

    if s.num_minus_right >= r && t.num_minus_right >= r {
        let mut s_minus_d = s.clone();
        let mut t_minus_d = t.clone();

        s_minus_d.num_minus_right -= r;
        t_minus_d.num_minus_right -= r;

        s_minus_d.update_score_and_impurity_before();
        t_minus_d.update_score_and_impurity_before();

        if t_minus_d.score > s_minus_d.score {
            return (false, Some((s_minus_d, t_minus_d)));
        }
    }

    return (true, None);
}
