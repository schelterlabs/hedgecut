use std::arch::x86_64::*;

use crate::dataset::Sample;
use crate::split_stats::SplitStats;

// TODO write some unit tests using this
#[allow(dead_code)]
pub fn scan<S: Sample>(
    samples: &[S],
    attribute_index: u8,
    cut_off: i8
) -> SplitStats {

    let mut num_left: u32 = 0;
    let mut num_plus_left: u32 = 0;
    let mut num_plus_right: u32 = 0;

    for sample in samples {
        let is_left = sample.is_smaller_than(attribute_index, cut_off as u8);
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

    SplitStats::new(
        num_plus_left,
        num_minus_left,
        num_plus_right,
        num_minus_right,
    )
}

pub fn scan_simd<S: Sample>(
    samples: &[S],
    attribute_index: u8,
    cut_off: i8
) -> SplitStats {
    let mut offset = 0;
    let batch_size = 16;

    let mut num_left: isize = 0;
    let mut num_plus_left: isize = 0;
    let mut num_plus_right: isize = 0;

    unsafe {

        let cut_off_batch = _mm_set_epi8(
            cut_off,
            cut_off,
            cut_off,
            cut_off,
            cut_off,
            cut_off,
            cut_off,
            cut_off,
            cut_off,
            cut_off,
            cut_off,
            cut_off,
            cut_off,
            cut_off,
            cut_off,
            cut_off,
        );

        let mut left_accumulator =
            _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let mut plus_left_accumulator =
            _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let mut plus_right_accumulator =
            _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

        let mut result_buffer: Vec<i8> = Vec::with_capacity(16);
        result_buffer.set_len(16);
        let results_buffer_addr =
            std::mem::transmute::<&mut i8, &mut __m128i>(&mut result_buffer[0]);

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
                (samples.get_unchecked(offset + 0).true_label() as i8) * -1,
                (samples.get_unchecked(offset + 1).true_label() as i8) * -1,
                (samples.get_unchecked(offset + 2).true_label() as i8) * -1,
                (samples.get_unchecked(offset + 3).true_label() as i8) * -1,
                (samples.get_unchecked(offset + 4).true_label() as i8) * -1,
                (samples.get_unchecked(offset + 5).true_label() as i8) * -1,
                (samples.get_unchecked(offset + 6).true_label() as i8) * -1,
                (samples.get_unchecked(offset + 7).true_label() as i8) * -1,
                (samples.get_unchecked(offset + 8).true_label() as i8) * -1,
                (samples.get_unchecked(offset + 9).true_label() as i8) * -1,
                (samples.get_unchecked(offset + 10).true_label() as i8) * -1,
                (samples.get_unchecked(offset + 11).true_label() as i8) * -1,
                (samples.get_unchecked(offset + 12).true_label() as i8) * -1,
                (samples.get_unchecked(offset + 13).true_label() as i8) * -1,
                (samples.get_unchecked(offset + 14).true_label() as i8) * -1,
                (samples.get_unchecked(offset + 15).true_label() as i8) * -1,
            );

            // Subtraction needed, https://devblogs.microsoft.com/oldnewthing/20141201-00/?p=43503
            let is_left_batch = _mm_cmplt_epi8(attribute_values_batch, cut_off_batch);
            left_accumulator = _mm_sub_epi8(left_accumulator, is_left_batch);

            let plus_left_batch = _mm_and_si128(is_left_batch, is_plus_batch);
            plus_left_accumulator = _mm_sub_epi8(plus_left_accumulator, plus_left_batch);

            let plus_right_batch = _mm_andnot_si128(is_left_batch, is_plus_batch);
            plus_right_accumulator = _mm_sub_epi8(plus_right_accumulator, plus_right_batch);

            offset += batch_size;

            if offset % 127 == 0 { // i8.MAX_VALUE

                _mm_store_si128(results_buffer_addr, left_accumulator);
                for count in result_buffer.iter() {
                    num_left += *count as isize;
                }

                _mm_store_si128(results_buffer_addr, plus_left_accumulator);
                for count in result_buffer.iter() {
                    num_plus_left += *count as isize;
                }

                _mm_store_si128(results_buffer_addr, plus_right_accumulator);
                for count in result_buffer.iter() {
                    num_plus_right += *count as isize;
                }

                left_accumulator =
                    _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
                plus_left_accumulator =
                    _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
                plus_right_accumulator =
                    _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            }
        }

        // Collect remaining results in the accumulators
        if offset % 127 != 0 { // i8.MAX_VALUE

            _mm_store_si128(results_buffer_addr, left_accumulator);
            for count in result_buffer.iter() {
                num_left += *count as isize;
            }

            _mm_store_si128(results_buffer_addr, plus_left_accumulator);
            for count in result_buffer.iter() {
                num_plus_left += *count as isize;
            }

            _mm_store_si128(results_buffer_addr, plus_right_accumulator);
            for count in result_buffer.iter() {
                num_plus_right += *count as isize;
            }
        }

        if offset < samples.len() {

            // Process last samples without SIMD
            while offset < samples.len() {

                let sample = samples.get_unchecked(offset);
                let is_left = sample.is_smaller_than(attribute_index, cut_off as u8);
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