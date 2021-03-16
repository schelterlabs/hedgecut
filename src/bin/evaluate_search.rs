extern crate hedgecut;

use hedgecut::split_stats::SplitStats;
use hedgecut::split_stats::is_robust;

use rand::Rng;
use std::time::Instant;

fn main() {

    let mut rng = rand::thread_rng();

    let mut timer = Instant::now();

    let configs = &[
        (2, 1_000_000, 100_000),
        (3, 1_000_000, 100_000),
        (4, 1_000_000, 100_000),
        (5, 1_000_000, 10_000),
        (6, 100_000, 1_000),
        (7, 100_000, 1_000),
        (8, 100_000, 1_000)
    ];

    for (robustness, num_tests, batch_size) in configs {

        let mut num_robust = 0;
        let mut num_non_robust = 0;
        let mut num_failures = 0;

        for test in 1..*num_tests {
            let num_samples = rng.gen_range(4 * robustness + 4, 1000);
            let num_plus_samples = rng.gen_range(2 * (robustness + 1), num_samples - 2 * (robustness + 1) + 1);
            let num_minus_samples = num_samples - num_plus_samples;

            let num_plus_left_s = rng.gen_range(robustness + 1, num_plus_samples - robustness);
            let num_minus_left_s = rng.gen_range(robustness + 1, num_minus_samples - robustness);
            let num_plus_right_s = num_plus_samples - num_plus_left_s;
            let num_minus_right_s = num_minus_samples - num_minus_left_s;

            let num_plus_left_t = rng.gen_range(robustness + 1, num_plus_samples - robustness);
            let num_minus_left_t = rng.gen_range(robustness + 1, num_minus_samples - robustness);
            let num_plus_right_t = num_plus_samples - num_plus_left_t;
            let num_minus_right_t = num_minus_samples - num_minus_left_t;

            let mut s = SplitStats::new(num_plus_left_s, num_minus_left_s, num_plus_right_s, num_minus_right_s);
            let mut t = SplitStats::new(num_plus_left_t, num_minus_left_t, num_plus_right_t, num_minus_right_t);

            s.update_score_and_impurity_before();
            t.update_score_and_impurity_before();

            //println!("S: {} {}, T: {} {}", s.fmt(), s.score as f64 / 1_000_000_000_000_f64, t.fmt(), t.score as f64 / 1_000_000_000_000_f64);

            if s.has_positive_score() && t.has_positive_score() {

                if s.score.unwrap() < t.score.unwrap() {
                    std::mem::swap(&mut s, &mut t);
                }

                let (split_robust, num_steps) = is_robust(&s, &t, *robustness as usize);

                let exhaustive_result = if *robustness == 2 {
                    exhaustive2(&mut s, &mut t)
                } else if *robustness == 3 {
                    exhaustive3(&mut s, &mut t)
                } else if *robustness == 4 {
                    exhaustive4(&mut s, &mut t)
                } else if *robustness == 5 {
                    exhaustive5(&mut s, &mut t)
                } else if *robustness == 6 {
                    exhaustive6(&mut s, &mut t)
                } else if *robustness == 7 {
                    exhaustive7(&mut s, &mut t)
                } else if *robustness == 8 {
                    exhaustive8(&mut s, &mut t)
                } else {
                    panic!("Cannot handle unknown robustness")
                };

                let mut is_robust_via_exhaustive = exhaustive_result.is_none();

                if let Some(breakpoint) = exhaustive_result {
                    is_robust_via_exhaustive = breakpoint > *robustness as usize;
                }

                if is_robust_via_exhaustive != split_robust {
                    num_failures += 1;

                    println!(
                        "S: {} {}, T: {} {}",
                        s.fmt(),
                        s.score.unwrap() as f64 / 1_000_000_000_000_f64,
                        t.fmt(),
                        t.score.unwrap() as f64 / 1_000_000_000_000_f64
                    );

                    println!("{:?}", exhaustive_result);

                    println!("DISAGREEMENT, exhaustive {}, greedy {}", is_robust_via_exhaustive, split_robust);

                    if !is_robust_via_exhaustive {
                        println!("EXHAUSTIVE says broken after {}", exhaustive_result.unwrap());
                    }
                    if !split_robust {
                        println!("GREEDY says broken after {}", num_steps);
                    }
                } else {
                    if split_robust {
                        num_robust += 1;
                    } else {
                        num_non_robust += 1;
                    }

                    if test % *batch_size == 0 || test == num_tests - 1 {

                        println!(
                            "Robustness({}): {} / {} / {}   [{}/{} ; {}s per {}]",
                            robustness,
                            num_robust,
                            num_non_robust,
                            num_failures,
                            test,
                            num_tests,
                            timer.elapsed().as_secs(),
                            batch_size
                        );

                        timer = Instant::now();
                    }
                }
            }
        }
    }
}

fn exhaustive7(s: &mut SplitStats, t: &mut SplitStats) -> Option<usize> {

    s.update_score_and_impurity_before();
    t.update_score_and_impurity_before();


    let enumerated1 = enumerate_changes(&s, &t);

    for (_step, s_hat, t_hat) in enumerated1 {
        let score_diff1 = s_hat.score.unwrap() - t_hat.score.unwrap();

        if score_diff1 < 0 {
            return Some(1);
        }

        let enumerated2 = enumerate_changes(&s_hat, &t_hat);


        for (_step, s_hat, t_hat) in enumerated2 {
            let score_diff2 = s_hat.score.unwrap() - t_hat.score.unwrap();

            if score_diff2 < 0 {
                return Some(2);
            }

            let enumerated3 = enumerate_changes(&s_hat, &t_hat);

            for (_step, s_hat, t_hat) in enumerated3 {
                let score_diff3 = s_hat.score.unwrap() - t_hat.score.unwrap();

                if score_diff3 < 0 {
                    return Some(3);
                }

                let enumerated4 = enumerate_changes(&s_hat, &t_hat);

                for (_step, s_hat, t_hat) in enumerated4 {
                    let score_diff4 = s_hat.score.unwrap() - t_hat.score.unwrap();

                    if score_diff4 < 0 {
                        return Some(4);
                    }

                    let enumerated5 = enumerate_changes(&s_hat, &t_hat);

                    for (_step, s_hat, t_hat) in enumerated5 {
                        let score_diff5 = s_hat.score.unwrap() - t_hat.score.unwrap();

                        if score_diff5 < 0 {
                            return Some(5);
                        }

                        let enumerated6 = enumerate_changes(&s_hat, &t_hat);

                        for (_step, s_hat, t_hat) in enumerated6 {
                            let score_diff6 = s_hat.score.unwrap() - t_hat.score.unwrap();

                            if score_diff6 < 0 {
                                return Some(6);
                            }

                            let enumerated7 = enumerate_changes(&s_hat, &t_hat);

                            for (_step, s_hat, t_hat) in enumerated7 {
                                let score_diff7 = s_hat.score.unwrap() - t_hat.score.unwrap();

                                if score_diff7 < 0 {
                                    return Some(7);
                                }
                            }

                        }
                    }
                }
            }
        }
    }

    None
}

fn exhaustive8(s: &mut SplitStats, t: &mut SplitStats) -> Option<usize> {

    s.update_score_and_impurity_before();
    t.update_score_and_impurity_before();


    let enumerated1 = enumerate_changes(&s, &t);

    for (_step, s_hat, t_hat) in enumerated1 {
        let score_diff1 = s_hat.score.unwrap() - t_hat.score.unwrap();

        if score_diff1 < 0 {
            return Some(1);
        }

        let enumerated2 = enumerate_changes(&s_hat, &t_hat);


        for (_step, s_hat, t_hat) in enumerated2 {
            let score_diff2 = s_hat.score.unwrap() - t_hat.score.unwrap();

            if score_diff2 < 0 {
                return Some(2);
            }

            let enumerated3 = enumerate_changes(&s_hat, &t_hat);

            for (_step, s_hat, t_hat) in enumerated3 {
                let score_diff3 = s_hat.score.unwrap() - t_hat.score.unwrap();

                if score_diff3 < 0 {
                    return Some(3);
                }

                let enumerated4 = enumerate_changes(&s_hat, &t_hat);

                for (_step, s_hat, t_hat) in enumerated4 {
                    let score_diff4 = s_hat.score.unwrap() - t_hat.score.unwrap();

                    if score_diff4 < 0 {
                        return Some(4);
                    }

                    let enumerated5 = enumerate_changes(&s_hat, &t_hat);

                    for (_step, s_hat, t_hat) in enumerated5 {
                        let score_diff5 = s_hat.score.unwrap() - t_hat.score.unwrap();

                        if score_diff5 < 0 {
                            return Some(5);
                        }

                        let enumerated6 = enumerate_changes(&s_hat, &t_hat);

                        for (_step, s_hat, t_hat) in enumerated6 {
                            let score_diff6 = s_hat.score.unwrap() - t_hat.score.unwrap();

                            if score_diff6 < 0 {
                                return Some(6);
                            }

                            let enumerated7 = enumerate_changes(&s_hat, &t_hat);

                            for (_step, s_hat, t_hat) in enumerated7 {
                                let score_diff7 = s_hat.score.unwrap() - t_hat.score.unwrap();

                                if score_diff7 < 0 {
                                    return Some(7);
                                }

                                let enumerated8 = enumerate_changes(&s_hat, &t_hat);

                                for (_step, s_hat, t_hat) in enumerated8 {
                                    let score_diff8 = s_hat.score.unwrap() - t_hat.score.unwrap();

                                    if score_diff8 < 0 {
                                        return Some(8);
                                    }
                                }
                            }


                        }
                    }
                }
            }
        }
    }

    None
}


fn exhaustive6(s: &mut SplitStats, t: &mut SplitStats) -> Option<usize> {

    s.update_score_and_impurity_before();
    t.update_score_and_impurity_before();


    let enumerated1 = enumerate_changes(&s, &t);

    for (_step, s_hat, t_hat) in enumerated1 {
        let score_diff1 = s_hat.score.unwrap() - t_hat.score.unwrap();

        if score_diff1 < 0 {
            return Some(1);
        }

        let enumerated2 = enumerate_changes(&s_hat, &t_hat);


        for (_step, s_hat, t_hat) in enumerated2 {
            let score_diff2 = s_hat.score.unwrap() - t_hat.score.unwrap();

            if score_diff2 < 0 {
                return Some(2);
            }

            let enumerated3 = enumerate_changes(&s_hat, &t_hat);

            for (_step, s_hat, t_hat) in enumerated3 {
                let score_diff3 = s_hat.score.unwrap() - t_hat.score.unwrap();

                if score_diff3 < 0 {
                    return Some(3);
                }

                let enumerated4 = enumerate_changes(&s_hat, &t_hat);

                for (_step, s_hat, t_hat) in enumerated4 {
                    let score_diff4 = s_hat.score.unwrap() - t_hat.score.unwrap();

                    if score_diff4 < 0 {
                        return Some(4);
                    }

                    let enumerated5 = enumerate_changes(&s_hat, &t_hat);

                    for (_step, s_hat, t_hat) in enumerated5 {
                        let score_diff5 = s_hat.score.unwrap() - t_hat.score.unwrap();

                        if score_diff5 < 0 {
                            return Some(5);
                        }

                        let enumerated6 = enumerate_changes(&s_hat, &t_hat);

                        for (_step, s_hat, t_hat) in enumerated6 {
                            let score_diff6 = s_hat.score.unwrap() - t_hat.score.unwrap();

                            if score_diff6 < 0 {
                                return Some(6);
                            }
                        }
                    }

                }

            }
        }
    }

    None
}


fn exhaustive5(s: &mut SplitStats, t: &mut SplitStats) -> Option<usize> {

    s.update_score_and_impurity_before();
    t.update_score_and_impurity_before();

    //let mut diffs1 = Vec::new();
    //let mut diffs2 = Vec::new();
    //let mut diffs3 = Vec::new();
    //let mut diffs4 = Vec::new();
    //let mut diffs5 = Vec::new();

    let enumerated1 = enumerate_changes(&s, &t);

    for (_step, s_hat, t_hat) in enumerated1 {
        let score_diff1 = s_hat.score.unwrap() - t_hat.score.unwrap();

        if score_diff1 < 0 {
            return Some(1);
        }

        let enumerated2 = enumerate_changes(&s_hat, &t_hat);

        //diffs1.push((s_hat.clone(), t_hat.clone(), score_diff1));

        for (_step, s_hat, t_hat) in enumerated2 {
            let score_diff2 = s_hat.score.unwrap() - t_hat.score.unwrap();

            if score_diff2 < 0 {
                return Some(2);
            }
            //diffs2.push((s_hat.clone(), t_hat.clone(), score_diff2));

            let enumerated3 = enumerate_changes(&s_hat, &t_hat);

            for (_step, s_hat, t_hat) in enumerated3 {
                let score_diff3 = s_hat.score.unwrap() - t_hat.score.unwrap();

                if score_diff3 < 0 {
                    return Some(3);
                }
                //diffs3.push((s_hat.clone(), t_hat.clone(), score_diff3));

                let enumerated4 = enumerate_changes(&s_hat, &t_hat);

                for (_step, s_hat, t_hat) in enumerated4 {
                    let score_diff4 = s_hat.score.unwrap() - t_hat.score.unwrap();

                    //diffs4.push((s_hat.clone(), t_hat.clone(), score_diff4));
                    if score_diff4 < 0 {
                        return Some(4);
                    }

                    let enumerated5 = enumerate_changes(&s_hat, &t_hat);

                    for (_step, s_hat, t_hat) in enumerated5 {
                        let score_diff5 = s_hat.score.unwrap() - t_hat.score.unwrap();

                        if score_diff5 < 0 {
                            return Some(5);
                        }

                        //diffs5.push((s_hat.clone(), t_hat.clone(), score_diff5));
                    }

                }

            }
        }
    }

    None
}

fn exhaustive4(s: &mut SplitStats, t: &mut SplitStats) -> Option<usize> {

    s.update_score_and_impurity_before();
    t.update_score_and_impurity_before();

    let enumerated1 = enumerate_changes(&s, &t);

    for (_step, s_hat, t_hat) in enumerated1 {
        let score_diff1 = s_hat.score.unwrap() - t_hat.score.unwrap();

        if score_diff1 < 0 {
            return Some(1);
        }

        let enumerated2 = enumerate_changes(&s_hat, &t_hat);

        for (_step, s_hat, t_hat) in enumerated2 {
            let score_diff2 = s_hat.score.unwrap() - t_hat.score.unwrap();

            if score_diff2 < 0 {
                return Some(2);
            }

            let enumerated3 = enumerate_changes(&s_hat, &t_hat);

            for (_step, s_hat, t_hat) in enumerated3 {
                let score_diff3 = s_hat.score.unwrap() - t_hat.score.unwrap();

                if score_diff3 < 0 {
                    return Some(3);
                }

                let enumerated4 = enumerate_changes(&s_hat, &t_hat);

                for (_step, s_hat, t_hat) in enumerated4 {
                    let score_diff4 = s_hat.score.unwrap() - t_hat.score.unwrap();

                    if score_diff4 < 0 {
                        return Some(4);
                    }
                }

            }
        }
    }

    None
}

fn exhaustive2(s: &mut SplitStats, t: &mut SplitStats) -> Option<usize> {

    s.update_score_and_impurity_before();
    t.update_score_and_impurity_before();


    let enumerated1 = enumerate_changes(&s, &t);

    for (_step, s_hat, t_hat) in enumerated1 {
        let score_diff1 = s_hat.score.unwrap() - t_hat.score.unwrap();

        if score_diff1 < 0 {
            return Some(1);
        }

        let enumerated2 = enumerate_changes(&s_hat, &t_hat);

        for (_step, s_hat, t_hat) in enumerated2 {
            let score_diff2 = s_hat.score.unwrap() - t_hat.score.unwrap();

            if score_diff2 < 0 {
                return Some(2);
            }
        }
    }

    None
}

fn exhaustive3(s: &mut SplitStats, t: &mut SplitStats) -> Option<usize> {

    s.update_score_and_impurity_before();
    t.update_score_and_impurity_before();


    let enumerated1 = enumerate_changes(&s, &t);

    for (_step, s_hat, t_hat) in enumerated1 {
        let score_diff1 = s_hat.score.unwrap() - t_hat.score.unwrap();

        if score_diff1 < 0 {
            return Some(1);
        }

        let enumerated2 = enumerate_changes(&s_hat, &t_hat);

        for (_step, s_hat, t_hat) in enumerated2 {
            let score_diff2 = s_hat.score.unwrap() - t_hat.score.unwrap();

            if score_diff2 < 0 {
                return Some(2);
            }

            let enumerated3 = enumerate_changes(&s_hat, &t_hat);

            for (_step, s_hat, t_hat) in enumerated3 {
                let score_diff3 = s_hat.score.unwrap() - t_hat.score.unwrap();

                if score_diff3 < 0 {
                    return Some(3);
                }
            }
        }
    }

    None
}


#[derive(Eq,PartialEq,Debug)]
struct Step {
    label: bool,
    passes_first: bool,
    passes_second: bool,
}

fn enumerate_changes(
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

                enumerated.push((step, champion_stats.clone(), runnerup_stats.clone()));
            }
        }
    }

    enumerated
}