pub fn normalized_information_gain(
    num_plus_left: u32,
    num_minus_left: u32,
    num_plus_right: u32,
    num_minus_right: u32)
    -> u64 {

    let num_left = num_plus_left + num_minus_left;
    let num_right = num_plus_right + num_minus_right;
    let num_plus = num_plus_left + num_plus_right;
    let num_minus = num_minus_left + num_minus_right;

    let num_samples = num_left + num_right;

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

    println!("H_C(S) {}, H_T(s) {}, H_C|T(S) {}, Score {}", hcs, hts, hcts, score);

    (score * 1_000_000_000_000_f64) as u64
}

#[allow(non_snake_case)]
fn H(a: u32, b: u32, a_plus_b: u32) -> f64 {
    let p_a = a as f64 / a_plus_b as f64;
    let p_b = b as f64 / a_plus_b as f64;
    -(p_a * p_a.log2() + p_b * p_b.log2())
}