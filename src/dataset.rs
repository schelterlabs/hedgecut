use std::str::FromStr;

pub trait Dataset {
    fn num_records(&self) -> u32;
    fn num_attributes(&self) -> u8;
    fn attribute(&self, index: u8) -> &Vec<u8>;
    fn attribute_range(&self, index: u8) -> (u8, u8);
    fn labels(&self) -> &Vec<bool>;
}

pub trait Sample {
    fn is_smaller_than(&self, attribute_index: u8, cut_off: u8) -> bool;
    fn true_label(&self) -> bool;
}

pub struct TitanicDataset {
    pub num_records: u32,
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

    pub fn dataset_from_csv() -> TitanicDataset {

        let mut age: Vec<u8> = Vec::new();
        let mut fare: Vec<u8> = Vec::new();
        let mut siblings: Vec<u8> = Vec::new();
        let mut children: Vec<u8> = Vec::new();
        let mut gender: Vec<u8> = Vec::new();
        let mut pclass: Vec<u8> = Vec::new();

        let mut labels: Vec<bool> = Vec::new();

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .delimiter(b'\t')
            .from_path("titanic-train.csv")
            .unwrap();

        for result in reader.records() {
            let record = result.unwrap();

            let record_age = u8::from_str(record.get(1).unwrap()).unwrap();
            let record_fare = u8::from_str(record.get(2).unwrap()).unwrap();
            let record_sibling = u8::from_str(record.get(3).unwrap()).unwrap();
            let record_children = u8::from_str(record.get(4).unwrap()).unwrap();
            let record_gender = u8::from_str(record.get(5).unwrap()).unwrap();
            let record_pclass = u8::from_str(record.get(6).unwrap()).unwrap();
            let record_label = u8::from_str(record.get(4).unwrap()).unwrap() == 1;

            age.push(record_age);
            fare.push(record_fare);
            siblings.push(record_sibling);
            children.push(record_children);
            gender.push(record_gender);
            pclass.push(record_pclass);
            labels.push(record_label);

        }

        TitanicDataset {
            num_records: age.len() as u32,
            age,
            fare,
            siblings,
            children,
            gender,
            pclass,
            labels
        }
    }

    pub fn samples_from_csv() -> Vec<TitanicSample> {

        let mut samples: Vec<TitanicSample> = Vec::new();


        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .delimiter(b'\t')
            .from_path("titanic-test.csv")
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

    fn num_attributes(&self) -> u8 { 6 }

    fn attribute(&self, index: u8) -> &Vec<u8> {
        match index {
            0 => &self.age,
            1 => &self.fare,
            2 => &self.siblings,
            3 => &self.children,
            4 => &self.gender,
            5 => &self.pclass,
            _ => panic!("Requested non-existing attribute!")
        }
    }

    fn attribute_range(&self, index: u8) -> (u8, u8) {
        match index {
            0 => (0, 19),
            1 => (0, 19),
            2 => (0, 8),
            3 => (0, 6),
            4 => (0, 1),
            5 => (1, 3),
            _ => panic!("Requested range for non-existing attribute!")
        }
    }


    fn labels(&self) -> &Vec<bool> {
        return &self.labels;
    }
}