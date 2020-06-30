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
            .from_path("datasets/titanic-train.csv")
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
            .from_path("datasets/titanic-test.csv")
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

pub struct DefaultsDataset {
    pub num_records: u32,
    pub limit: Vec<u8>,
    pub sex: Vec<u8>,
    pub education: Vec<u8>,
    pub marriage: Vec<u8>,
    pub age: Vec<u8>,
    pub pay0: Vec<u8>,
    pub pay2: Vec<u8>,
    pub pay3: Vec<u8>,
    pub pay4: Vec<u8>,
    pub pay5: Vec<u8>,
    pub pay6: Vec<u8>,
    pub bill_amt1: Vec<u8>,
    pub bill_amt2: Vec<u8>,
    pub bill_amt3: Vec<u8>,
    pub bill_amt4: Vec<u8>,
    pub bill_amt5: Vec<u8>,
    pub bill_amt6: Vec<u8>,
    pub pay_amt1: Vec<u8>,
    pub pay_amt2: Vec<u8>,
    pub pay_amt3: Vec<u8>,
    pub pay_amt4: Vec<u8>,
    pub pay_amt5: Vec<u8>,
    pub pay_amt6: Vec<u8>,
    pub labels: Vec<bool>,
}

impl DefaultsDataset {

    pub fn dataset_from_csv() -> DefaultsDataset {

        let mut limit: Vec<u8> = Vec::new();
        let mut sex: Vec<u8> = Vec::new();
        let mut education: Vec<u8> = Vec::new();
        let mut marriage: Vec<u8> = Vec::new();
        let mut age: Vec<u8> = Vec::new();
        let mut pay0: Vec<u8> = Vec::new();
        let mut pay2: Vec<u8> = Vec::new();
        let mut pay3: Vec<u8> = Vec::new();
        let mut pay4: Vec<u8> = Vec::new();
        let mut pay5: Vec<u8> = Vec::new();
        let mut pay6: Vec<u8> = Vec::new();
        let mut bill_amt1: Vec<u8> = Vec::new();
        let mut bill_amt2: Vec<u8> = Vec::new();
        let mut bill_amt3: Vec<u8> = Vec::new();
        let mut bill_amt4: Vec<u8> = Vec::new();
        let mut bill_amt5: Vec<u8> = Vec::new();
        let mut bill_amt6: Vec<u8> = Vec::new();
        let mut pay_amt1: Vec<u8> = Vec::new();
        let mut pay_amt2: Vec<u8> = Vec::new();
        let mut pay_amt3: Vec<u8> = Vec::new();
        let mut pay_amt4: Vec<u8> = Vec::new();
        let mut pay_amt5: Vec<u8> = Vec::new();
        let mut pay_amt6: Vec<u8> = Vec::new();

        let mut labels: Vec<bool> = Vec::new();

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .delimiter(b'\t')
            .from_path("datasets/defaults-train.csv")
            .unwrap();

        for result in reader.records() {
            let record = result.unwrap();

            let record_limit = u8::from_str(record.get(1).unwrap()).unwrap();
            let record_sex = u8::from_str(record.get(2).unwrap()).unwrap();
            let record_education = u8::from_str(record.get(3).unwrap()).unwrap();
            let record_marriage = u8::from_str(record.get(4).unwrap()).unwrap();
            let record_age = u8::from_str(record.get(5).unwrap()).unwrap();
            let record_pay0 = u8::from_str(record.get(6).unwrap()).unwrap();
            let record_pay2 = u8::from_str(record.get(7).unwrap()).unwrap();
            let record_pay3 = u8::from_str(record.get(8).unwrap()).unwrap();
            let record_pay4 = u8::from_str(record.get(9).unwrap()).unwrap();
            let record_pay5 = u8::from_str(record.get(10).unwrap()).unwrap();
            let record_pay6 = u8::from_str(record.get(11).unwrap()).unwrap();
            let record_bill_amt1 = u8::from_str(record.get(12).unwrap()).unwrap();
            let record_bill_amt2 = u8::from_str(record.get(13).unwrap()).unwrap();
            let record_bill_amt3 = u8::from_str(record.get(14).unwrap()).unwrap();
            let record_bill_amt4 = u8::from_str(record.get(15).unwrap()).unwrap();
            let record_bill_amt5 = u8::from_str(record.get(16).unwrap()).unwrap();
            let record_bill_amt6 = u8::from_str(record.get(17).unwrap()).unwrap();
            let record_pay_amt1 = u8::from_str(record.get(18).unwrap()).unwrap();
            let record_pay_amt2 = u8::from_str(record.get(19).unwrap()).unwrap();
            let record_pay_amt3 = u8::from_str(record.get(20).unwrap()).unwrap();
            let record_pay_amt4 = u8::from_str(record.get(21).unwrap()).unwrap();
            let record_pay_amt5 = u8::from_str(record.get(22).unwrap()).unwrap();
            let record_pay_amt6 = u8::from_str(record.get(23).unwrap()).unwrap();
            let label = u8::from_str(record.get(24).unwrap()).unwrap() == 1;

            limit.push(record_limit);
            sex.push(record_sex);
            education.push(record_education);
            marriage.push(record_marriage);
            age.push(record_age);
            pay0.push(record_pay0);
            pay2.push(record_pay2);
            pay3.push(record_pay3);
            pay4.push(record_pay4);
            pay5.push(record_pay5);
            pay6.push(record_pay6);
            bill_amt1.push(record_bill_amt1);
            bill_amt2.push(record_bill_amt2);
            bill_amt3.push(record_bill_amt3);
            bill_amt4.push(record_bill_amt4);
            bill_amt5.push(record_bill_amt5);
            bill_amt6.push(record_bill_amt6);
            pay_amt1.push(record_pay_amt1);
            pay_amt2.push(record_pay_amt2);
            pay_amt3.push(record_pay_amt3);
            pay_amt4.push(record_pay_amt4);
            pay_amt5.push(record_pay_amt5);
            pay_amt6.push(record_pay_amt6);
            labels.push(label);
        }

        DefaultsDataset {
            num_records: labels.len() as u32,
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
            labels
        }
    }

    pub fn samples_from_csv() -> Vec<DefaultsSample> {

        let mut samples: Vec<DefaultsSample> = Vec::new();

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .delimiter(b'\t')
            .from_path("datasets/defaults-train.csv")
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

    fn num_attributes(&self) -> u8 {
        23
    }

    fn attribute(&self, index: u8) -> &Vec<u8> {
        match index {
            0 => &self.limit,
            1 => &self.sex,
            2 => &self.education,
            3 => &self.marriage,
            4 => &self.age,
            5 => &self.pay0,
            6 => &self.pay2,
            7 => &self.pay3,
            8 => &self.pay4,
            9 => &self.pay5,
            10 => &self.pay6,
            11 => &self.bill_amt1,
            12 => &self.bill_amt2,
            13 => &self.bill_amt3,
            14 => &self.bill_amt4,
            15 => &self.bill_amt5,
            16 => &self.bill_amt6,
            17 => &self.pay_amt1,
            18 => &self.pay_amt2,
            19 => &self.pay_amt3,
            20 => &self.pay_amt4,
            21 => &self.pay_amt5,
            22 => &self.pay_amt6,
            _ => panic!("Requested non-existing attribute!")
        }
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

    fn labels(&self) -> &Vec<bool> {
       &self.labels
    }
}

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
    fn is_smaller_than(&self, attribute_index: u8, cut_off: u8) -> bool {
        let attribute = match attribute_index {
            0 => &self.limit,
            1 => &self.sex,
            2 => &self.education,
            3 => &self.marriage,
            4 => &self.age,
            5 => &self.pay0,
            6 => &self.pay2,
            7 => &self.pay3,
            8 => &self.pay4,
            9 => &self.pay5,
            10 => &self.pay6,
            11 => &self.bill_amt1,
            12 => &self.bill_amt2,
            13 => &self.bill_amt3,
            14 => &self.bill_amt4,
            15 => &self.bill_amt5,
            16 => &self.bill_amt6,
            17 => &self.pay_amt1,
            18 => &self.pay_amt2,
            19 => &self.pay_amt3,
            20 => &self.pay_amt4,
            21 => &self.pay_amt5,
            22 => &self.pay_amt6,
            _ => panic!("Requested non-existing attribute!")
        };

        *attribute < cut_off
    }

    fn true_label(&self) -> bool {
        self.label
    }
}