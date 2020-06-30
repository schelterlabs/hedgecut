fn main() {



    let mut record_ids: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    //let values: Vec<u8> = vec![0, 0, 7, 6, 7, 8, 9, 1, 3, 8];
    let values: Vec<u8> = vec![3, 3, 7, 6, 7, 8, 9, 3, 3, 8];

    let cut_off = 5;

    let mut cursor = 0;
    let mut cursor_end = record_ids.len();

    let mut constant_value_on_the_left = true;
    let mut first_value_on_the_left: Option<u8> = None;
    let mut constant_value_on_the_right = true;
    let mut first_value_on_the_right: Option<u8> = None;

    loop {
        let record_id = record_ids.get(cursor).unwrap();
        let attribute_value = *values.get(*record_id as usize).unwrap();
        if attribute_value < cut_off {

            if constant_value_on_the_left {
                if first_value_on_the_left.is_none() {
                    first_value_on_the_left = Some(attribute_value);
                } else if attribute_value != first_value_on_the_left.unwrap() {
                    constant_value_on_the_left = false;
                }
            }

            cursor += 1;
        } else {

            if constant_value_on_the_right {
                if first_value_on_the_right.is_none() {
                    first_value_on_the_right = Some(attribute_value);
                } else if attribute_value != first_value_on_the_right.unwrap() {
                        constant_value_on_the_right = false;
                }
            }

            cursor_end -= 1;
            //println!("Swapping {} and {} with record {}({}), cursor_end is now {}", cursor, cursor_end, record_id, value, cursor_end);
            record_ids.swap(cursor, cursor_end);
        }

        if cursor == cursor_end - 1 {
            break;
        }
    }

    let (left_record_ids, right_record_ids) = record_ids.split_at_mut(cursor);

    for index in 0..left_record_ids.len() {
        let record_id = left_record_ids.get(index).unwrap();
        println!("{}({})", record_id, values.get(*record_id as usize).unwrap());
    }

    println!("---------------");

    for index in 0..right_record_ids.len() {
        let record_id = right_record_ids.get(index).unwrap();
        println!("{}({})", record_id, values.get(*record_id as usize).unwrap());
    }

    println!("Constant on the left? {}, Constant on the right? {}", constant_value_on_the_left, constant_value_on_the_right);

}