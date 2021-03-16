use std::arch::x86_64::*;

use crate::dataset::Sample;
use crate::split_stats::SplitStats;
use crate::tree::Split;

pub fn scan_with_branches<S: Sample>(
    samples: &[S],
    split: &Split,
) -> SplitStats {

    let mut num_left: u32 = 0;
    let mut num_plus_left: u32 = 0;
    let mut num_plus_right: u32 = 0;

    for sample in samples {
        let is_left = sample.is_left_of(split);
        let is_plus = sample.true_label();

        if is_left {
            num_left += 1;

            if is_plus {
                num_plus_left += 1;
            }
        } else {
            if is_plus {
                num_plus_right += 1;
            }
        }
    }

    let num_minus_left = num_left - num_plus_left;
    let num_minus_right: u32 =  ((samples.len() - num_left as usize) as u32) - num_plus_right;

    SplitStats::new(num_plus_left, num_minus_left, num_plus_right, num_minus_right)
}

fn simd_sum_mlpack(labels: &[bool]) -> [usize; 2] {
    let batch_size = 4;
    let mut i = 0;
    let leftovers = labels.len() % batch_size;

    let mut counts = [0,0];
    let mut counts2 = [0,0];
    let mut counts3 = [0,0];
    let mut counts4 = [0,0];

    /*
           // SIMD loop: add counts for four elements simultaneously (if the compiler
           // manages to vectorize the loop).
           for (size_t i = 3; i < labels.n_elem; i += 4)
           {
             counts[labels[i - 3]]++;
             counts2[labels[i - 2]]++;
             counts3[labels[i - 1]]++;
             counts4[labels[i]]++;
           }
    */

    while i < labels.len() - leftovers {
        counts[labels[i] as usize] += 1;
        counts2[labels[i + 1] as usize] += 1;
        counts3[labels[i + 2] as usize] += 1;
        counts4[labels[i + 3] as usize] += 1;

        i += batch_size;
    }

    /*
           // Handle leftovers.
           if (labels.n_elem % 4 == 1)
           {
             counts[labels[labels.n_elem - 1]]++;
           }
           else if (labels.n_elem % 4 == 2)
           {
             counts[labels[labels.n_elem - 2]]++;
             counts2[labels[labels.n_elem - 1]]++;
           }
           else if (labels.n_elem % 4 == 3)
           {
             counts[labels[labels.n_elem - 3]]++;
             counts2[labels[labels.n_elem - 2]]++;
             counts3[labels[labels.n_elem - 1]]++;
           }
    */
    if leftovers == 1 {
        counts[labels[labels.len() - 1] as usize] += 1;
    }
    else if leftovers == 2  {
        counts[labels[labels.len() - 2] as usize] += 1;
        counts2[labels[labels.len() - 1] as usize] += 1;
    }
    else if leftovers == 3 {
        counts[labels[labels.len() - 3] as usize] += 1;
        counts2[labels[labels.len() - 2] as usize] += 1;
        counts3[labels[labels.len() - 1] as usize] += 1;
    }

    /*
       counts += counts2 + counts3 + counts4;
    */
    let all_counts = [
        counts[0] + counts2[0] + counts3[0] + counts4[0],
        counts[1] + counts2[1] + counts3[1] + counts4[1],
    ];

    all_counts
}

pub fn scan_mlpack<S: Sample>(
    samples: &[S],
    split: &Split,
) -> SplitStats {

    let mut labels_left = Vec::new();
    let mut labels_right = Vec::new();

    // Non-SIMD threshold comparisons
    for sample in samples {
        if sample.is_left_of(split) {
            labels_left.push(sample.true_label());
        } else {
            labels_right.push(sample.true_label());
        }
    }

    // mlpack only uses SIMD to sum up the final counts
    let counts_left = simd_sum_mlpack(&labels_left);
    let counts_right = simd_sum_mlpack(&labels_right);

    /*
       for (size_t i = 0; i < numClasses; ++i)
       {
         const double f = ((double) counts[i] / (double) labels.n_elem);
         impurity += f * (1.0 - f);
       }
     }

     return -impurity;
*/
    // We just compute the stats not the gain in this method
    SplitStats::new(
        counts_left[1] as u32,
        counts_left[0] as u32,
        counts_right[1] as u32,
        counts_right[0] as u32
    )
}

pub fn scan<S: Sample>(
    samples: &[S],
    split: &Split,
) -> SplitStats {

    let mut num_left: u32 = 0;
    let mut num_plus_left: u32 = 0;
    let mut num_plus_right: u32 = 0;

    for sample in samples {
        let is_left = sample.is_left_of(split);
        let is_plus = sample.true_label();

        num_left += is_left as u32;
        num_plus_left += (is_left & is_plus) as u32;
        num_plus_right += (!is_left & is_plus) as u32;
    }

    let num_minus_left = num_left - num_plus_left;
    let num_minus_right: u32 =  ((samples.len() - num_left as usize) as u32) - num_plus_right;

    SplitStats::new(num_plus_left, num_minus_left, num_plus_right, num_minus_right)
}

pub fn scan_simd_numerical<S: Sample>(
    samples: &[S],
    split: &Split,
) -> SplitStats {

    let (attribute_index, cut_off) = match split {
        Split::Numerical { attribute_index, cut_off } => (*attribute_index, *cut_off as i8),
        Split::Categorical { attribute_index: _, subset: _ } => {
            panic!("Don't call this method with a categorical split!")
        }
    };

    let mut offset = 0;
    let batch_size = 16;

    let mut num_left: isize = 0;
    let mut num_plus_left: isize = 0;
    let mut num_plus_right: isize = 0;

    unsafe {

        let cut_off_batch = _mm_set1_epi8(cut_off);
        let additional_samples = samples.len() % batch_size;

        while offset < samples.len() - additional_samples {

            let attribute_values_batch = _mm_set_epi8(
                samples.get_unchecked(offset + 0).attribute_value(attribute_index) as i8,
                samples.get_unchecked(offset + 1).attribute_value(attribute_index) as i8,
                samples.get_unchecked(offset + 2).attribute_value(attribute_index) as i8,
                samples.get_unchecked(offset + 3).attribute_value(attribute_index) as i8,
                samples.get_unchecked(offset + 4).attribute_value(attribute_index) as i8,
                samples.get_unchecked(offset + 5).attribute_value(attribute_index) as i8,
                samples.get_unchecked(offset + 6).attribute_value(attribute_index) as i8,
                samples.get_unchecked(offset + 7).attribute_value(attribute_index) as i8,
                samples.get_unchecked(offset + 8).attribute_value(attribute_index) as i8,
                samples.get_unchecked(offset + 9).attribute_value(attribute_index) as i8,
                samples.get_unchecked(offset + 10).attribute_value(attribute_index) as i8,
                samples.get_unchecked(offset + 11).attribute_value(attribute_index) as i8,
                samples.get_unchecked(offset + 12).attribute_value(attribute_index) as i8,
                samples.get_unchecked(offset + 13).attribute_value(attribute_index) as i8,
                samples.get_unchecked(offset + 14).attribute_value(attribute_index) as i8,
                samples.get_unchecked(offset + 15).attribute_value(attribute_index) as i8,
            );

            let is_plus_batch = _mm_set_epi8(
                samples.get_unchecked(offset + 0).true_label() as i8,
                samples.get_unchecked(offset + 1).true_label() as i8,
                samples.get_unchecked(offset + 2).true_label() as i8,
                samples.get_unchecked(offset + 3).true_label() as i8,
                samples.get_unchecked(offset + 4).true_label() as i8,
                samples.get_unchecked(offset + 5).true_label() as i8,
                samples.get_unchecked(offset + 6).true_label() as i8,
                samples.get_unchecked(offset + 7).true_label() as i8,
                samples.get_unchecked(offset + 8).true_label() as i8,
                samples.get_unchecked(offset + 9).true_label() as i8,
                samples.get_unchecked(offset + 10).true_label() as i8,
                samples.get_unchecked(offset + 11).true_label() as i8,
                samples.get_unchecked(offset + 12).true_label() as i8,
                samples.get_unchecked(offset + 13).true_label() as i8,
                samples.get_unchecked(offset + 14).true_label() as i8,
                samples.get_unchecked(offset + 15).true_label() as i8,
            );

            let shifted_is_plus_batch = _mm_slli_epi16(is_plus_batch, 7);

            let is_left_batch = _mm_cmplt_epi8(attribute_values_batch, cut_off_batch);

            let plus_left_batch = _mm_and_si128(is_left_batch, shifted_is_plus_batch);
            let plus_right_batch = _mm_andnot_si128(is_left_batch, shifted_is_plus_batch);

            let left_result = _mm_movemask_epi8(is_left_batch);
            let plus_left_result = _mm_movemask_epi8(plus_left_batch);
            let plus_right_result = _mm_movemask_epi8(plus_right_batch);

            num_left += left_result.count_ones() as isize;
            num_plus_left += plus_left_result.count_ones() as isize;
            num_plus_right += plus_right_result.count_ones() as isize;

            offset += batch_size;
        }

        if offset < samples.len() {

            // Process last samples without SIMD
            while offset < samples.len() {

                let sample = samples.get_unchecked(offset);
                let is_left = sample.is_left_of(split);
                let is_plus = sample.true_label();

                num_left += is_left as isize;
                num_plus_left += (is_left & is_plus) as isize;
                num_plus_right += (!is_left & is_plus) as isize;

                offset += 1;
            }
        }

        let num_minus_left: u32 = (num_left - num_plus_left) as u32;
        let num_minus_right: u32 =
            ((samples.len() - num_left as usize) as u32) - num_plus_right as u32;

        SplitStats::new(
            num_plus_left as u32,
            num_minus_left,
            num_plus_right as u32,
            num_minus_right,
        )
    }
}

pub fn scan_simd_categorical<S: Sample>(
    samples: &[S],
    split: &Split,
) -> SplitStats {

    let (attribute_index, subset) = match split {
        Split::Numerical { attribute_index: _, cut_off: _ } => {
            panic!("Don't call this method with a numerical split!");
        }
        Split::Categorical { attribute_index, subset } => {
            (*attribute_index, *subset)
        }
    };

    let mut offset = 0;
    let batch_size = 4;

    let mut num_left: isize = 0;
    let mut num_plus_left: isize = 0;
    let mut num_plus_right: isize = 0;

    unsafe {

        let subset_batch = _mm_set1_epi32(subset as i32);
        let no_match_batch = _mm_set1_epi32(0);
        let indexes_batch = _mm_set1_epi32(1);

        let additional_samples = samples.len() % batch_size;

        while offset < samples.len() - additional_samples {

            let attribute_values_batch = _mm_set_epi32(
                samples.get_unchecked(offset + 0).attribute_value(attribute_index) as i32,
                samples.get_unchecked(offset + 1).attribute_value(attribute_index) as i32,
                samples.get_unchecked(offset + 2).attribute_value(attribute_index) as i32,
                samples.get_unchecked(offset + 3).attribute_value(attribute_index) as i32,
            );

            let is_plus_batch = _mm_set_epi32(
                samples.get_unchecked(offset + 0).true_label() as i32,
                samples.get_unchecked(offset + 1).true_label() as i32,
                samples.get_unchecked(offset + 2).true_label() as i32,
                samples.get_unchecked(offset + 3).true_label() as i32,
            );

            let positions = _mm_sllv_epi32(indexes_batch, attribute_values_batch);

            let is_in = _mm_and_si128(positions, subset_batch);
            let is_left_batch = _mm_cmpeq_epi32(is_in, no_match_batch);
            let shifted_is_plus_batch = _mm_slli_epi32(is_plus_batch, 15);

            let plus_left_batch = _mm_andnot_si128(is_left_batch, shifted_is_plus_batch);
            let plus_right_batch = _mm_and_si128(is_left_batch, shifted_is_plus_batch);

            let left_result = _mm_movemask_epi8(is_left_batch);
            let plus_left_result = _mm_movemask_epi8(plus_left_batch);
            let plus_right_result = _mm_movemask_epi8(plus_right_batch);

            num_left += (4 - left_result.count_ones()  / 4) as isize;
            num_plus_left += plus_left_result.count_ones() as isize;
            num_plus_right += plus_right_result.count_ones() as isize;

            offset += batch_size;
        }

        if offset < samples.len() {

            // Process last samples without SIMD
            while offset < samples.len() {

                let sample = samples.get_unchecked(offset);
                let is_left = sample.is_left_of(split);
                let is_plus = sample.true_label();

                num_left += is_left as isize;
                num_plus_left += (is_left & is_plus) as isize;
                num_plus_right += (!is_left & is_plus) as isize;

                offset += 1;
            }
        }

        let num_minus_left: u32 = (num_left - num_plus_left) as u32;
        let num_minus_right: u32 =
            ((samples.len() - num_left as usize) as u32) - num_plus_right as u32;

        SplitStats::new(
            num_plus_left as u32,
            num_minus_left,
            num_plus_right as u32,
            num_minus_right,
        )
    }
}

#[cfg(test)]
mod tests {

    use crate::dataset::{GiveMeSomeCreditDataset, ShoppingDataset};
    use crate::tree::Split;
    use crate::scan::{scan, scan_simd_numerical, scan_simd_categorical, scan_mlpack};

    #[test]
    fn mlpack_impl() {
        let samples =
            GiveMeSomeCreditDataset::samples_from_csv("datasets/givemesomecredit-train.csv");

        let split = Split::Numerical { attribute_index: 3, cut_off: 5 };

        let stats = scan(&samples, &split);
        let stats_mlpack = scan_mlpack(&samples, &split);

        eprintln!("{:?}", stats);
        eprintln!("{:?}", stats_mlpack);

        assert_eq!(stats.num_plus_left, stats_mlpack.num_plus_left);
        assert_eq!(stats.num_plus_right, stats_mlpack.num_plus_right);
        assert_eq!(stats.num_minus_left, stats_mlpack.num_minus_left);
        assert_eq!(stats.num_minus_right, stats_mlpack.num_minus_right);
    }

    #[test]
    fn same_result_numerical() {
        let samples =
            GiveMeSomeCreditDataset::samples_from_csv("datasets/givemesomecredit-train.csv");

        let split = Split::Numerical { attribute_index: 3, cut_off: 5 };

        let stats = scan(&samples, &split);
        let stats_simd = scan_simd_numerical(&samples, &split);

        assert_eq!(stats.num_plus_left, stats_simd.num_plus_left);
        assert_eq!(stats.num_plus_right, stats_simd.num_plus_right);
        assert_eq!(stats.num_minus_left, stats_simd.num_minus_left);
        assert_eq!(stats.num_minus_right, stats_simd.num_minus_right);
    }

    #[test]
    fn same_result_categorical() {
        let samples =
            ShoppingDataset::samples_from_csv("datasets/shopping-train.csv");

        let possible_values = vec![0, 7, 12];

        let mut subset: u64 = 0;
        for bit_to_set in possible_values.iter() {
            subset |= 1_u64 << *bit_to_set as u64
        }

        let split = Split::Categorical { attribute_index: 14, subset };

        let stats = scan(&samples, &split);
        let stats_simd = scan_simd_categorical(&samples, &split);

        println!("SCAN {:?}", stats);
        println!("SIMD {:?}", stats_simd);

        assert_eq!(stats.num_plus_left, stats_simd.num_plus_left);
        assert_eq!(stats.num_plus_right, stats_simd.num_plus_right);
        assert_eq!(stats.num_minus_left, stats_simd.num_minus_left);
        assert_eq!(stats.num_minus_right, stats_simd.num_minus_right);
    }
}