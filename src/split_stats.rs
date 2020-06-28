
#[derive(Eq,PartialEq,Debug)]
pub struct SplitStats {
    pub num_plus_left: u32,
    pub num_minus_left: u32,
    pub num_plus_right: u32,
    pub num_minus_right: u32,
    pub score: Option<u64>,
}

impl SplitStats {

    pub fn new() -> SplitStats {
        SplitStats {
            num_plus_left: 0,
            num_minus_left: 0,
            num_plus_right: 0,
            num_minus_right: 0,
            score: None
        }
    }

    pub fn update(&mut self, plus: bool, is_left: bool) {
        if is_left {
            if plus {
                self.num_plus_left += 1;
            } else {
                self.num_minus_left += 1;
            }
        } else {
            if plus {
                self.num_plus_right += 1;
            } else {
                self.num_minus_right += 1;
            }
        }
    }

    pub fn update_score(&mut self) {
        self.score = normalized_information_gain(
            self.num_plus_left,
            self.num_minus_left,
            self.num_plus_right,
            self.num_minus_right
        );
    }
}

fn normalized_information_gain(
    num_plus_left: u32,
    num_minus_left: u32,
    num_plus_right: u32,
    num_minus_right: u32)
    -> Option<u64> {

    let num_left = num_plus_left + num_minus_left;
    let num_right = num_plus_right + num_minus_right;

    if num_left == 0 || num_right == 0 {
        return None
    }

    let num_plus = num_plus_left + num_plus_right;
    let num_minus = num_minus_left + num_minus_right;

    let num_samples = num_left + num_right;

    // TODO this is independent of the split, could be reused
    // Prior "classification entropy" H_C(S)
    let hcs = H(num_plus, num_minus, num_samples);

    // Entropy of S with respect to test T H_T(S)
    let hts = H(num_left, num_right, num_samples);

    // Posterior "classification entropy" H_{C|T}(S) of S given the outcome of the test T
    // TODO this is computed twice
    let p_sys = num_left as f64 / num_samples as f64;
    let p_sns = num_right as f64 / num_samples as f64;

    let hcsy = H(num_plus_left, num_minus_left, num_left);
    let hcsn = H(num_plus_right, num_minus_right, num_right);

    let hcts = p_sys * hcsy + p_sns * hcsn;

    // Information gain of applying test T
    let icts = hcs - hcts;

    let score = 2.0 * icts / (hcs + hts);

    if score.is_nan() {
        println!("[{},{},{},{}]", num_plus_left, num_minus_left, num_plus_right, num_minus_right);
        println!("H_C(S) {}, H_T(s) {}, H_C|T(S) {}, Score {}", hcs, hts, hcts, score);
        panic!("Invalid score encountered!");
    }

    //println!("H_C(S) {}, H_T(s) {}, H_C|T(S) {}, Score {}", hcs, hts, hcts, score);

    Some((score * 1_000_000_000_000_f64) as u64)
}

#[allow(non_snake_case)]
fn H(a: u32, b: u32, a_plus_b: u32) -> f64 {

    //TODO we could precompute the logarithms for small numbers
    let p_a_times_log_of_pa = if a != 0 {
        let p_a = a as f64 / a_plus_b as f64;
        p_a * p_a.log2()
    } else {
        0.0
    };

    let p_b_times_log_of_pb = if b != 0 {
        let p_b = b as f64 / a_plus_b as f64;
        p_b * p_b.log2()
    } else {
        0.0
    };

    -(p_a_times_log_of_pa + p_b_times_log_of_pb)
}