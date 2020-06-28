use std::str::FromStr;

pub trait Dataset {
    fn num_records(&self) -> u32;
    fn num_attributes(&self) -> u8;
    fn attribute(&self, index: u8) -> &Vec<u8>;
    fn labels(&self) -> &Vec<bool>;
}

pub trait Sample {
    fn is_smaller_than(&self, attribute_index: u8, cut_off: u8) -> bool;
    fn true_label(&self) -> bool;
}

pub struct TitanicDataset {
    pub age: Vec<u8>,
    pub fare: Vec<u8>,
    pub siblings: Vec<u8>,
    pub children: Vec<u8>,
    pub gender: Vec<u8>,
    pub pclass: Vec<u8>,
    pub labels: Vec<bool>,
}

pub struct TitanicSample {
    pub age: u8,
    pub fare: u8,
    pub siblings: u8,
    pub children: u8,
    pub gender: u8,
    pub pclass: u8,
    pub label: bool,
}

impl Sample for TitanicSample {
    fn is_smaller_than(&self, attribute_index: u8, cut_off: u8) -> bool {
        let attribute = match attribute_index {
            0 => &self.age,
            1 => &self.fare,
            2 => &self.siblings,
            3 => &self.children,
            4 => &self.gender,
            5 => &self.pclass,
            _ => panic!("Requested non exsting attribute!")
        };

        *attribute < cut_off
    }

    fn true_label(&self) -> bool {
        self.label
    }
}


impl TitanicDataset {

    pub fn from_csv() -> TitanicDataset {
        let num_records = 886;

        let mut age: Vec<u8> = Vec::with_capacity(num_records);
        let mut fare: Vec<u8> = Vec::with_capacity(num_records);
        let mut siblings: Vec<u8> = Vec::with_capacity(num_records);
        let mut children: Vec<u8> = Vec::with_capacity(num_records);
        let mut gender: Vec<u8> = Vec::with_capacity(num_records);
        let mut pclass: Vec<u8> = Vec::with_capacity(num_records);

        let mut labels: Vec<bool> = Vec::with_capacity(num_records);

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .delimiter(b'\t')
            .from_path("titanic-attributes.csv")
            .unwrap();

        for result in reader.records() {
            let record = result.unwrap();

            let record_id: u32 = u32::from_str(record.get(0).unwrap()).unwrap();
            let attribute_name = record.get(1).unwrap();
            let attribute_value = u8::from_str(record.get(2).unwrap()).unwrap();

            match attribute_name {
                "age" => age.insert(record_id as usize, attribute_value),
                "fare" => fare.insert(record_id as usize, attribute_value),
                "siblings" => siblings.insert(record_id as usize, attribute_value),
                "children" => children.insert(record_id as usize, attribute_value),
                "gender" => gender.insert(record_id as usize, attribute_value),
                "pclass" => pclass.insert(record_id as usize, attribute_value),
                "label" => labels.insert(record_id as usize, attribute_value == 1),

                _ => println!("UNKNOWN ATTRIBUTE ENCOUNTERED")
            }
        }

        TitanicDataset {
            age,
            fare,
            siblings,
            children,
            gender,
            pclass,
            labels
        }
    }
}

impl Dataset for TitanicDataset {

    fn num_records(&self) -> u32 { 886 }

    fn num_attributes(&self) -> u8 { 6 }

    fn attribute(&self, index: u8) -> &Vec<u8> {
        match index {
            0 => &self.age,
            1 => &self.fare,
            2 => &self.siblings,
            3 => &self.children,
            4 => &self.gender,
            5 => &self.pclass,
            _ => panic!("Requested non exsting attribute!")
        }
    }

    fn labels(&self) -> &Vec<bool> {
        return &self.labels;
    }
}