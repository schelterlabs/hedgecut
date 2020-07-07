fn main() {}
// extern crate hedgecut;
//
// use std::arch::x86_64::*;
//
// use hedgecut::dataset::{DefaultsDataset, Sample, DefaultsSample};
//
// fn main() {
//
//     let all_samples = DefaultsDataset::samples_from_csv("datasets/defaults-train.csv");
//
//     let samples = &all_samples[..23577];
//
//     let attribute_index: u8 = 1;
//     let cut_off = 2;
//
//     split_counts(&samples, attribute_index, cut_off);
//     split_counts_simd(&samples, attribute_index, cut_off);
// }
//
// fn split_counts(
//     samples: &[DefaultsSample],
//     attribute_index: u8,
//     cut_off: i8
// ) {
//     let mut num_plus: isize = 0;
//     let mut num_left: isize = 0;
//     let mut num_plus_left: isize = 0;
//     let mut num_plus_right: isize = 0;
//
//     for sample in samples {
//         let is_left = sample.is_left_of(attribute_index, cut_off as u8);
//         let is_plus = sample.true_label();
//
//         if is_left {
//             num_left += 1;
//
//             if is_plus {
//                 num_plus_left += 1;
//             }
//         }
//
//         if is_plus {
//             num_plus += 1;
//             if !is_left {
//                 num_plus_right += 1;
//             }
//         }
//     }
//
//     println!(
//         "NO-SIMD: plus {}, left {}, plus left {}, plus right {}",
//         num_plus,
//         num_left,
//         num_plus_left,
//         num_plus_right
//     );
// }
//
// fn split_counts_simd(
//     samples: &[DefaultsSample],
//     attribute_index: u8,
//     cut_off: i8
// ) {
//     let mut offset = 0;
//     let batch_size = 16;
//
//     let mut num_plus: isize = 0;
//     let mut num_left: isize = 0;
//     let mut num_plus_left: isize = 0;
//     let mut num_plus_right: isize = 0;
//
//     unsafe {
//
//         let cut_off_batch = _mm_set_epi8(
//             cut_off,
//             cut_off,
//             cut_off,
//             cut_off,
//             cut_off,
//             cut_off,
//             cut_off,
//             cut_off,
//             cut_off,
//             cut_off,
//             cut_off,
//             cut_off,
//             cut_off,
//             cut_off,
//             cut_off,
//             cut_off,
//         );
//
//         let mut left_accumulator =
//             _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
//         let mut plus_accumulator =
//             _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
//         let mut plus_left_accumulator =
//             _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
//         let mut plus_right_accumulator =
//             _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
//
//         let mut result_buffer: Vec<i8> = Vec::with_capacity(16);
//         result_buffer.set_len(16);
//         let results_buffer_addr =
//             std::mem::transmute::<&mut i8, &mut __m128i>(&mut result_buffer[0]);
//
//         let additional_samples = samples.len() % batch_size;
//
//         while offset < samples.len() - additional_samples {
//
//             //println!("SIMD counting for offset {}", offset);
//
//             let attribute_values_batch = _mm_set_epi8(
//                 samples.get_unchecked(offset + 0).attribute_value(attribute_index) as i8,
//                 samples.get_unchecked(offset + 1).attribute_value(attribute_index) as i8,
//                 samples.get_unchecked(offset + 2).attribute_value(attribute_index) as i8,
//                 samples.get_unchecked(offset + 3).attribute_value(attribute_index) as i8,
//                 samples.get_unchecked(offset + 4).attribute_value(attribute_index) as i8,
//                 samples.get_unchecked(offset + 5).attribute_value(attribute_index) as i8,
//                 samples.get_unchecked(offset + 6).attribute_value(attribute_index) as i8,
//                 samples.get_unchecked(offset + 7).attribute_value(attribute_index) as i8,
//                 samples.get_unchecked(offset + 8).attribute_value(attribute_index) as i8,
//                 samples.get_unchecked(offset + 9).attribute_value(attribute_index) as i8,
//                 samples.get_unchecked(offset + 10).attribute_value(attribute_index) as i8,
//                 samples.get_unchecked(offset + 11).attribute_value(attribute_index) as i8,
//                 samples.get_unchecked(offset + 12).attribute_value(attribute_index) as i8,
//                 samples.get_unchecked(offset + 13).attribute_value(attribute_index) as i8,
//                 samples.get_unchecked(offset + 14).attribute_value(attribute_index) as i8,
//                 samples.get_unchecked(offset + 15).attribute_value(attribute_index) as i8,
//             );
//
//             let is_plus_batch = _mm_set_epi8(
//                 (samples.get_unchecked(offset + 0).true_label() as i8) * -1,
//                 (samples.get_unchecked(offset + 1).true_label() as i8) * -1,
//                 (samples.get_unchecked(offset + 2).true_label() as i8) * -1,
//                 (samples.get_unchecked(offset + 3).true_label() as i8) * -1,
//                 (samples.get_unchecked(offset + 4).true_label() as i8) * -1,
//                 (samples.get_unchecked(offset + 5).true_label() as i8) * -1,
//                 (samples.get_unchecked(offset + 6).true_label() as i8) * -1,
//                 (samples.get_unchecked(offset + 7).true_label() as i8) * -1,
//                 (samples.get_unchecked(offset + 8).true_label() as i8) * -1,
//                 (samples.get_unchecked(offset + 9).true_label() as i8) * -1,
//                 (samples.get_unchecked(offset + 10).true_label() as i8) * -1,
//                 (samples.get_unchecked(offset + 11).true_label() as i8) * -1,
//                 (samples.get_unchecked(offset + 12).true_label() as i8) * -1,
//                 (samples.get_unchecked(offset + 13).true_label() as i8) * -1,
//                 (samples.get_unchecked(offset + 14).true_label() as i8) * -1,
//                 (samples.get_unchecked(offset + 15).true_label() as i8) * -1,
//             );
//
//             plus_accumulator = _mm_sub_epi8(plus_accumulator, is_plus_batch);
//
//             // Subtraction needed, https://devblogs.microsoft.com/oldnewthing/20141201-00/?p=43503
//             let is_left_batch = _mm_cmplt_epi8(attribute_values_batch, cut_off_batch);
//             left_accumulator = _mm_sub_epi8(left_accumulator, is_left_batch);
//
//             let plus_left_batch = _mm_and_si128(is_left_batch, is_plus_batch);
//             plus_left_accumulator = _mm_sub_epi8(plus_left_accumulator, plus_left_batch);
//
//             let plus_right_batch = _mm_andnot_si128(is_left_batch, is_plus_batch);
//             plus_right_accumulator = _mm_sub_epi8(plus_right_accumulator, plus_right_batch);
//
//             offset += batch_size;
//
//             if offset % 127 == 0 { // i8.MAX_VALUE
//                 _mm_store_si128(results_buffer_addr, plus_accumulator);
//                 for count in result_buffer.iter() {
//                     num_plus += *count as isize;
//                 }
//
//                 _mm_store_si128(results_buffer_addr, left_accumulator);
//                 for count in result_buffer.iter() {
//                     num_left += *count as isize;
//                 }
//
//                 _mm_store_si128(results_buffer_addr, plus_left_accumulator);
//                 for count in result_buffer.iter() {
//                     num_plus_left += *count as isize;
//                 }
//
//                 _mm_store_si128(results_buffer_addr, plus_right_accumulator);
//                 for count in result_buffer.iter() {
//                     num_plus_right += *count as isize;
//                 }
//
//                 left_accumulator =
//                     _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
//                 plus_accumulator =
//                     _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
//                 plus_left_accumulator =
//                     _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
//                 plus_right_accumulator =
//                     _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
//             }
//         }
//
//         println!(
//             "[Offset {}] plus {}, left {}, plus left {}, plus right {}",
//             offset,
//             num_plus,
//             num_left,
//             num_plus_left,
//             num_plus_right
//         );
//
//
//         // Collect remaining results in the accumulators
//         if offset % 127 != 0 { // i8.MAX_VALUE
//
//             println!("Emptying accumulators");
//
//             _mm_store_si128(results_buffer_addr, plus_accumulator);
//             for count in result_buffer.iter() {
//                 num_plus += *count as isize;
//             }
//
//             _mm_store_si128(results_buffer_addr, left_accumulator);
//             for count in result_buffer.iter() {
//                 num_left += *count as isize;
//             }
//
//             _mm_store_si128(results_buffer_addr, plus_left_accumulator);
//             for count in result_buffer.iter() {
//                 num_plus_left += *count as isize;
//             }
//
//             _mm_store_si128(results_buffer_addr, plus_right_accumulator);
//             for count in result_buffer.iter() {
//                 num_plus_right += *count as isize;
//             }
//         }
//
//         if offset < samples.len() {
//
//             println!("Processing last samples without SIMD");
//
//             // Process last samples without SIMD
//             while offset < samples.len() {
//
//
//                 let sample = samples.get_unchecked(offset);
//                 let is_left = sample.is_left_of(attribute_index, cut_off as u8);
//                 let is_plus = sample.true_label();
//
//                 if is_left {
//                     num_left += 1;
//
//                     if is_plus {
//                         num_plus_left += 1;
//                     }
//                 }
//
//                 if is_plus {
//                     num_plus += 1;
//                     if !is_left {
//                         num_plus_right += 1;
//                     }
//                 }
//
//                 offset += 1;
//             }
//         }
//
//         println!(
//             "plus {}, left {}, plus left {}, plus right {}",
//             num_plus,
//             num_left,
//             num_plus_left,
//             num_plus_right
//         );
//     }
// }