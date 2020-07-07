use std::str::FromStr;
use crate::tree::Split;

pub trait Dataset {
    fn num_records(&self) -> u32;
    fn num_plus(&self) -> u32;
    fn num_attributes(&self) -> u8;
    fn attribute_range(&self, index: u8) -> (u8, u8);
    fn attribute_type(&self, index: u8) -> AttributeType;
}

pub enum AttributeType {
    Numerical,
    Categorical
}

pub trait Sample: Clone {

    fn is_left_of(&self, split: &Split) -> bool {

        let attribute_index = split.attribute_index();
        let attribute_value = self.attribute_value(attribute_index);

        match split {
            Split::Numerical { attribute_index: _, cut_off } => {
                attribute_value < *cut_off
            },
            Split::Categorical { attribute_index: _, subset } => {
                *subset & (1 << attribute_value) as u64 != 0
            }

        }
    }

    fn attribute_value(&self, attribute_index: u8) -> u8;
    fn true_label(&self) -> bool;
}

pub struct TitanicDataset {
    pub num_records: u32,
    pub num_plus: u32,
}

#[derive(Eq,PartialEq,Debug,Clone)]
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

    fn attribute_value(&self, attribute_index: u8) -> u8 {
        match attribute_index {
            0 => self.age,
            1 => self.fare,
            2 => self.siblings,
            3 => self.children,
            4 => self.gender,
            5 => self.pclass,
            _ => panic!("Requested range for non-existing attribute {}!", attribute_index)
        }
    }

    fn true_label(&self) -> bool {
        self.label
    }
}


impl TitanicDataset {

    pub fn from_samples(samples: &Vec<TitanicSample>) -> TitanicDataset {
        let num_plus = samples.iter().filter(|sample| sample.true_label()).count();

        TitanicDataset {
            num_records: samples.len() as u32,
            num_plus: num_plus as u32
        }
    }

    pub fn samples_from_csv(file: &str) -> Vec<TitanicSample> {

        let mut samples: Vec<TitanicSample> = Vec::new();


        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .delimiter(b'\t')
            .from_path(file)
            .unwrap();

        for result in reader.records() {
            let record = result.unwrap();

            let age = u8::from_str(record.get(1).unwrap()).unwrap();
            let fare = u8::from_str(record.get(2).unwrap()).unwrap();
            let siblings = u8::from_str(record.get(3).unwrap()).unwrap();
            let children = u8::from_str(record.get(4).unwrap()).unwrap();
            let gender = u8::from_str(record.get(5).unwrap()).unwrap();
            let pclass = u8::from_str(record.get(6).unwrap()).unwrap();
            let label = u8::from_str(record.get(4).unwrap()).unwrap() == 1;

            let sample = TitanicSample { age, fare, siblings, children, gender, pclass, label };

            samples.push(sample);
        }

        samples
    }

}

impl Dataset for TitanicDataset {

    fn num_records(&self) -> u32 { self.num_records }

    fn num_plus(&self) -> u32 { self.num_plus }

    fn num_attributes(&self) -> u8 { 6 }

    fn attribute_range(&self, index: u8) -> (u8, u8) {
        match index {
            0 => (0, 19),
            1 => (0, 19),
            2 => (0, 8),
            3 => (0, 6),
            4 => (0, 1),
            5 => (0, 2),
            _ => panic!("Requested range for non-existing attribute {}!", index)
        }
    }

    fn attribute_type(&self, index: u8) -> AttributeType {
        match index {
            0 => AttributeType::Numerical,
            1 => AttributeType::Numerical,
            2 => AttributeType::Numerical,
            3 => AttributeType::Numerical,
            4 => AttributeType::Categorical,
            5 => AttributeType::Categorical,
            _ => panic!("Requested range for non-existing attribute {}!", index)
        }
    }
}

pub struct DefaultsDataset {
    pub num_records: u32,
    pub num_plus: u32,
}

impl DefaultsDataset {

    pub fn from_samples(samples: &Vec<DefaultsSample>) -> DefaultsDataset {
        let num_plus = samples.iter().filter(|sample| sample.true_label()).count();

        DefaultsDataset {
            num_records: samples.len() as u32,
            num_plus: num_plus as u32
        }
    }

    pub fn samples_from_csv(file: &str) -> Vec<DefaultsSample> {

        let mut samples: Vec<DefaultsSample> = Vec::new();

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .delimiter(b'\t')
            .from_path(file)
            .unwrap();

        for result in reader.records() {
            let record = result.unwrap();

            let limit = u8::from_str(record.get(1).unwrap()).unwrap();
            let sex = u8::from_str(record.get(2).unwrap()).unwrap();
            let education = u8::from_str(record.get(3).unwrap()).unwrap();
            let marriage = u8::from_str(record.get(4).unwrap()).unwrap();
            let age = u8::from_str(record.get(5).unwrap()).unwrap();
            let pay0 = u8::from_str(record.get(6).unwrap()).unwrap();
            let pay2 = u8::from_str(record.get(7).unwrap()).unwrap();
            let pay3 = u8::from_str(record.get(8).unwrap()).unwrap();
            let pay4 = u8::from_str(record.get(9).unwrap()).unwrap();
            let pay5 = u8::from_str(record.get(10).unwrap()).unwrap();
            let pay6 = u8::from_str(record.get(11).unwrap()).unwrap();
            let bill_amt1 = u8::from_str(record.get(12).unwrap()).unwrap();
            let bill_amt2 = u8::from_str(record.get(13).unwrap()).unwrap();
            let bill_amt3 = u8::from_str(record.get(14).unwrap()).unwrap();
            let bill_amt4 = u8::from_str(record.get(15).unwrap()).unwrap();
            let bill_amt5 = u8::from_str(record.get(16).unwrap()).unwrap();
            let bill_amt6 = u8::from_str(record.get(17).unwrap()).unwrap();
            let pay_amt1 = u8::from_str(record.get(18).unwrap()).unwrap();
            let pay_amt2 = u8::from_str(record.get(19).unwrap()).unwrap();
            let pay_amt3 = u8::from_str(record.get(20).unwrap()).unwrap();
            let pay_amt4 = u8::from_str(record.get(21).unwrap()).unwrap();
            let pay_amt5 = u8::from_str(record.get(22).unwrap()).unwrap();
            let pay_amt6 = u8::from_str(record.get(23).unwrap()).unwrap();
            let label = u8::from_str(record.get(24).unwrap()).unwrap() == 1;

            let sample = DefaultsSample {
                limit,
                sex,
                education,
                marriage,
                age,
                pay0,
                pay2,
                pay3,
                pay4,
                pay5,
                pay6,
                bill_amt1,
                bill_amt2,
                bill_amt3,
                bill_amt4,
                bill_amt5,
                bill_amt6,
                pay_amt1,
                pay_amt2,
                pay_amt3,
                pay_amt4,
                pay_amt5,
                pay_amt6,
                label
            };

            samples.push(sample);
        }

        samples
    }
}

impl Dataset for DefaultsDataset {

    fn num_records(&self) -> u32 {
        self.num_records
    }

    fn num_plus(&self) -> u32 { self.num_plus }

    fn num_attributes(&self) -> u8 {
        23
    }

    fn attribute_range(&self, index: u8) -> (u8, u8) {
        match index {
            0 => (0, 19),
            1 => (1, 2),
            2 => (1, 4),
            3 => (1, 3),
            4 => (0, 19),
            5 => (0, 11),
            6 => (0, 11),
            7 => (0, 11),
            8 => (0, 11),
            9 => (0, 11),
            10 => (0, 11),
            11 => (0, 19),
            12 => (0, 19),
            13 => (0, 19),
            14 => (0, 19),
            15 => (0, 19),
            16 => (0, 19),
            17 => (0, 19),
            18 => (0, 19),
            19 => (0, 19),
            20 => (0, 19),
            21 => (0, 19),
            22 => (0, 19),
            _ => panic!("Requested non-existing attribute!")
        }
    }

    fn attribute_type(&self, index: u8) -> AttributeType {
        match index {
            0 => AttributeType::Numerical,
            1 => AttributeType::Numerical,
            2 => AttributeType::Numerical,
            3 => AttributeType::Numerical,
            4 => AttributeType::Numerical,
            5 => AttributeType::Numerical,
            6 => AttributeType::Numerical,
            7 => AttributeType::Numerical,
            8 => AttributeType::Numerical,
            9 => AttributeType::Numerical,
            10 => AttributeType::Numerical,
            11 => AttributeType::Numerical,
            12 => AttributeType::Numerical,
            13 => AttributeType::Numerical,
            14 => AttributeType::Numerical,
            15 => AttributeType::Numerical,
            16 => AttributeType::Numerical,
            17 => AttributeType::Numerical,
            18 => AttributeType::Numerical,
            19 => AttributeType::Numerical,
            20 => AttributeType::Numerical,
            21 => AttributeType::Numerical,
            22 => AttributeType::Numerical,
            _ => panic!("Requested non-existing attribute!")
        }
    }
}

#[derive(Eq,PartialEq,Debug,Clone)]
pub struct DefaultsSample {
    pub limit: u8,
    pub sex: u8,
    pub education: u8,
    pub marriage: u8,
    pub age: u8,
    pub pay0: u8,
    pub pay2: u8,
    pub pay3: u8,
    pub pay4: u8,
    pub pay5: u8,
    pub pay6: u8,
    pub bill_amt1: u8,
    pub bill_amt2: u8,
    pub bill_amt3: u8,
    pub bill_amt4: u8,
    pub bill_amt5: u8,
    pub bill_amt6: u8,
    pub pay_amt1: u8,
    pub pay_amt2: u8,
    pub pay_amt3: u8,
    pub pay_amt4: u8,
    pub pay_amt5: u8,
    pub pay_amt6: u8,
    pub label: bool,
}

impl Sample for DefaultsSample {

    fn attribute_value(&self, attribute_index: u8) -> u8 {
        match attribute_index {
            0 => self.limit,
            1 => self.sex,
            2 => self.education,
            3 => self.marriage,
            4 => self.age,
            5 => self.pay0,
            6 => self.pay2,
            7 => self.pay3,
            8 => self.pay4,
            9 => self.pay5,
            10 => self.pay6,
            11 => self.bill_amt1,
            12 => self.bill_amt2,
            13 => self.bill_amt3,
            14 => self.bill_amt4,
            15 => self.bill_amt5,
            16 => self.bill_amt6,
            17 => self.pay_amt1,
            18 => self.pay_amt2,
            19 => self.pay_amt3,
            20 => self.pay_amt4,
            21 => self.pay_amt5,
            22 => self.pay_amt6,
            _ => panic!("Requested non-existing attribute!")
        }
    }


    fn true_label(&self) -> bool {
        self.label
    }
}