fn main() {

    let mut a = vec![0, 1, 2, 3];
    let mut b = vec![4, 5, 6, 7];

    std::mem::swap(&mut a, &mut b);

    println!("a: {:?}", a);
    println!("a: {:?}", b);
}