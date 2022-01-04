fn gcd(a: u32, b: u32) -> u32 {
     if a == 0 || b == 0 {
        0
     } else if a == b {
         a
     } else if a > b {
         gcd(a - b, b)
     } else {
         gcd(a, b - a)
     }
}

pub fn make_coprime(numbers: &mut [u32]) {
    for n in numbers.iter_mut() {
        if *n <= 1 {
            *n = 2;
        }
    }
    for i in 1..numbers.len() {
        let mut done = false;
        while !done {
            done = true;
            for j in 0..i {
                while gcd(numbers[i], numbers[j]) != 1 {
                    numbers[i] += 1;
                    done = false;
                }
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct SchmittTrigger {
    triggered: bool
}

impl SchmittTrigger {
    #[inline]
    pub fn tick(&mut self, trigger: f64) -> bool {
        if trigger > 0.5 {
            if !self.triggered {
                self.triggered = true;
                true
            } else {
                false
            }
        } else {
            self.triggered = false;
            false
        }
    }
}
