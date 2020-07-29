extern crate hedgecut;

use hedgecut::split_stats::{SplitStats};

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

                //println!("{},{},{}", score_diff, score_diff_again, score_diff_again_again);
                diff_paths.push((score_diff, score_diff_again, score_diff_again_again));
            }
        }
    }

    //println!("---------------------------------------------------------------------------");

    let mins_path = (
        *diffs1.iter().min().unwrap(),
        *diffs2.iter().min().unwrap(),
        *diffs3.iter().min().unwrap()
    );

    let found = diff_paths.contains(&mins_path);

    println!("{}: {:?}", found, mins_path);
}

fn main() {

    evaluate3(
        &mut SplitStats::new(8, 6, 12, 2),
        &mut SplitStats::new(10, 3, 10, 3)
    );

    evaluate3(
        &mut SplitStats::new(12, 2, 8, 6),
        &mut SplitStats::new(10, 3, 10, 3)
    );

    evaluate3(
        &mut SplitStats::new(7, 5, 13, 3),
        &mut SplitStats::new(10, 3, 10, 3)
    );

    evaluate3(
        &mut SplitStats::new(70, 50, 130, 30),
        &mut SplitStats::new(100, 30, 100, 30)
    );

    evaluate3(
        &mut SplitStats::new(2, 2, 1, 0),
        &mut SplitStats::new(1, 1, 2, 1)
    );
}