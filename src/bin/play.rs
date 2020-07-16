extern crate hedgecut;

use std::arch::x86_64::*;

fn main() {

    let possible_values = vec![0, 7, 29];

    let mut subset: i32 = 0;
    for bit_to_set in possible_values.iter() {
        subset |= 1_i32 << *bit_to_set as i32
    }

    unsafe {

        let subset_batch = _mm_set1_epi32(subset);
        let no_match_batch = _mm_set1_epi32(0);


         let mut result_buffer: Vec<i32> = Vec::with_capacity(4);
         result_buffer.set_len(4);
         let results_buffer_addr =
             std::mem::transmute::<&mut i32, &mut __m128i>(&mut result_buffer[0]);

        let values_batch = _mm_set_epi32(1, 4, 29, 30);
        let indexes_batch = _mm_set1_epi32(1);

        let positions = _mm_sllv_epi32(indexes_batch, values_batch);


        let is_in = _mm_and_si128(positions, subset_batch);
        let is_in_match = _mm_cmpeq_epi32(is_in, no_match_batch);

        _mm_store_si128(results_buffer_addr, is_in_match);
        for count in result_buffer.iter() {
            println!("{:#034b}", count);
        }

    }
    //println!("{:#066b}", subset);

//     unsafe {
//
//         let mut result_buffer: Vec<i8> = Vec::with_capacity(16);
//         result_buffer.set_len(16);
//         let results_buffer_addr =
//             std::mem::transmute::<&mut i8, &mut __m128i>(&mut result_buffer[0]);
//
//         let values = _mm_set_epi8(0, 0, 0, 0,  0,  0,  0,  0, 5, 5, 5, 5,  5,  5,  5,  5);
//         let labels = _mm_set_epi8(0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);
// //        let labels = _mm_set_epi8(0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1);
//
//         let shifted_labels = _mm_slli_epi16(labels, 7);
//
//         _mm_store_si128(results_buffer_addr, shifted_labels);
//         for count in result_buffer.iter() {
//             println!("{:#010b}", count);
//         }
//
//         let cut_off = _mm_set1_epi8(5);
//
//         let is_left = _mm_cmplt_epi8(values, cut_off);
//         let is_plus_left = _mm_and_si128(is_left, shifted_labels);
//         let is_plus_right = _mm_andnot_si128(is_left, shifted_labels);
//
//         let is_left_result = _mm_movemask_epi8(is_left);
//         let is_plus_left_result = _mm_movemask_epi8(is_plus_left);
//         let is_plus_right_result = _mm_movemask_epi8(is_plus_right);
//
//         //println!("{:b}", 1_i8);
//         //println!("{:b}", -1_i8);
//
//         println!(
//             "num_left: {}, num_plus_left: {}, num_plus_right: {}",
//             is_left_result.count_ones(),
//             is_plus_left_result.count_ones(),
//             is_plus_right_result.count_ones(),
//         );
//         //result.count_ones();
// //        let result = _mm_popcnt_epi8(is_plus_left);
//
// //        println!("{:b}", is_plus_right_result);
//     }
}
