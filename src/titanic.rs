use std::error::Error;
use std::str::FromStr;

#[derive(Debug)]
pub struct Passenger {
    pub survived: bool,
    pub class: u8,
    pub male: bool,
    pub age: u8,
    pub siblings: u8,
    pub parents: u8,
    pub fare: u32
}


pub fn titanic_data() -> Result<Vec<Passenger>, Box<dyn Error>> {
    // Build the CSV reader and iterate over each record.
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b',')
        .from_path("titanic.csv")?;

    let mut passengers = Vec::with_capacity(887);

    for result in reader.records() {
        let record = result?;

        let survived: bool = u8::from_str(record.get(0).unwrap()).unwrap() == 0;
        let class: u8 = u8::from_str(record.get(1).unwrap()).unwrap();
        let male: bool = record.get(3).unwrap() == "male";
        let age: u8 = f32::from_str(record.get(4).unwrap()).unwrap() as u8;
        let siblings: u8 = u8::from_str(record.get(5).unwrap()).unwrap();
        let parents: u8 = u8::from_str(record.get(6).unwrap()).unwrap();
        let fare: u32 = (f32::from_str(record.get(7).unwrap()).unwrap() * 10000_f32) as u32;

        let passenger = Passenger { survived, class, male, age, siblings, parents, fare };

        passengers.push(passenger);
    }

    Ok(passengers)
}