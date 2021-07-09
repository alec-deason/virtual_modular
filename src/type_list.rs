use packed_simd_2::f32x8;
#[derive(Copy, Clone, Debug, Default)]
pub struct Value<A: Clone>(pub A);
#[derive(Copy, Clone, Default)]
pub struct NoValue;

impl<A, B> std::ops::Add for Value<(A, A)>
where
    A: std::ops::Add<Output = B> + Copy,
    B: Clone,
{
    type Output = Value<(B, B)>;
    fn add(self, other: Self) -> Self::Output {
        Value(((self.0).0 + (other.0).0, (self.0).1 + (other.0).0))
    }
}
impl<A, B> std::ops::Mul for Value<(A, A)>
where
    A: std::ops::Mul<Output = B> + Copy,
    B: Clone,
{
    type Output = Value<(B, B)>;
    fn mul(self, other: Self) -> Self::Output {
        Value(((self.0).0 * (other.0).0, (self.0).1 * (other.0).0))
    }
}
impl<A> std::ops::Mul<A> for Value<(A, A)>
where
    A: std::ops::Mul<Output = A> + Copy,
{
    type Output = Value<(A, A)>;
    fn mul(self, other: A) -> Self::Output {
        Value(((self.0).0 * other, (self.0).1 * other))
    }
}
impl std::ops::Mul<f32> for Value<(f32x8, f32x8)> {
    type Output = Value<(f32x8, f32x8)>;
    fn mul(self, other: f32) -> Self::Output {
        Value(((self.0).0 * other, (self.0).1 * other))
    }
}
pub trait ValueT: Clone {
    type Car: Clone;
    type Cdr: Clone;
    type Inner: Clone;
    fn car(&self) -> &Self::Car;
    fn cdr(&self) -> &Self::Cdr;
    fn inner(&self) -> &Self::Inner;
    fn map_car(self, f: impl Fn(Self::Car) -> Self::Car) -> Self;
    fn from_inner(inner: Self::Inner) -> Self;
}
impl<A: Clone> ValueT for Value<(A,)> {
    type Car = A;
    type Cdr = NoValue;
    type Inner = (A,);
    fn car(&self) -> &Self::Car {
        &(self.0).0
    }
    fn map_car(self, f: impl Fn(Self::Car) -> Self::Car) -> Self {
        Value((f((self.0).0),))
    }
    fn inner(&self) -> &Self::Inner {
        &self.0
    }
    fn cdr(&self) -> &Self::Cdr {
        &NoValue
    }
    fn from_inner(inner: Self::Inner) -> Self {
        Value(inner)
    }
}
impl<A: Clone, B: Clone> ValueT for Value<(A, B)> {
    type Car = A;
    type Cdr = B;
    type Inner = (A, B);
    fn car(&self) -> &Self::Car {
        &(self.0).0
    }
    fn cdr(&self) -> &Self::Cdr {
        &(self.0).1
    }
    fn inner(&self) -> &Self::Inner {
        &self.0
    }
    fn map_car(self, f: impl Fn(Self::Car) -> Self::Car) -> Self {
        Value((f((self.0).0), (self.0).1))
    }
    fn from_inner(inner: Self::Inner) -> Self {
        Value(inner)
    }
}
impl ValueT for NoValue {
    type Car = NoValue;
    type Cdr = NoValue;
    type Inner = NoValue;
    fn car(&self) -> &NoValue {
        &NoValue
    }
    fn map_car(self, _f: impl Fn(Self::Car) -> Self::Car) -> Self {
        NoValue
    }
    fn inner(&self) -> &Self::Inner {
        &NoValue
    }
    fn cdr(&self) -> &NoValue {
        &NoValue
    }
    fn from_inner(inner: Self::Inner) -> Self {
        NoValue
    }
}

pub trait Combine<A, B> {
    type Output: ValueT;
    fn combine(a: A, b: B) -> Self::Output;
    fn split(v: Self::Output) -> (A, B);
}
impl<A: Clone> Combine<Value<(A,)>, NoValue> for (Value<(A,)>, NoValue) {
    type Output = Value<(A,)>;
    fn combine(a: Value<(A,)>, _b: NoValue) -> Self::Output {
        a
    }
    fn split(v: Value<(A,)>) -> (Value<(A,)>, NoValue) {
        (v, NoValue)
    }
}
impl<A: Copy> Combine<NoValue, Value<(A,)>> for (NoValue, Value<(A,)>) {
    type Output = Value<(A,)>;
    fn combine(_a: NoValue, b: Value<(A,)>) -> Self::Output {
        b
    }
    fn split(v: Value<(A,)>) -> (NoValue, Value<(A,)>) {
        (NoValue, v)
    }
}
impl<A: Copy, B:Copy> Combine<NoValue, Value<(A,B)>> for (NoValue, Value<(A,B)>) {
    type Output = Value<(A,B)>;
    fn combine(_a: NoValue, b: Value<(A,B)>) -> Self::Output {
        b
    }
    fn split(v: Value<(A,B)>) -> (NoValue, Value<(A,B)>) {
        (NoValue, v)
    }
}
impl<A: Clone, B: Clone> Combine<Value<(A,)>, Value<(B,)>> for (Value<(A,)>, Value<(B,)>) {
    type Output = Value<(A, B)>;
    fn combine(a: Value<(A,)>, b: Value<(B,)>) -> Self::Output {
        Value(((a.0).0, (b.0).0))
    }
    fn split(v: Value<(A, B)>) -> (Value<(A,)>, Value<(B,)>) {
        (Value(((v.0).0,)), Value(((v.0).1,)))
    }
}
impl<A: Clone, B: Clone, C: Clone> Combine<Value<(A, B)>, Value<(C,)>>
    for (Value<(A, B)>, Value<(C,)>)
{
    type Output = Value<((A, B), C)>;
    fn combine(a: Value<(A, B)>, b: Value<(C,)>) -> Self::Output {
        Value((a.0, (b.0).0))
    }
    fn split(v: Value<((A, B), C)>) -> (Value<(A, B)>, Value<(C,)>) {
        (Value((v.0).0), Value(((v.0).1,)))
    }
}
impl<A: Clone, B: Clone, C: Clone> Combine<Value<(A,)>, Value<(B, C)>>
    for (Value<(A,)>, Value<(B, C)>)
{
    type Output = Value<(A, (B, C))>;
    fn combine(a: Value<(A,)>, b: Value<(B, C)>) -> Self::Output {
        Value(((a.0).0, b.0))
    }
    fn split(v: Value<(A, (B, C))>) -> (Value<(A,)>, Value<(B, C)>) {
        (Value(((v.0).0,)), Value((v.0).1))
    }
}
impl<A: Clone, B: Clone> Combine<Value<(A, A)>, Value<(B, B)>> for (Value<(A, A)>, Value<(B, B)>) {
    type Output = Value<((A, A), (B, B))>;
    fn combine(a: Value<(A, A)>, b: Value<(B, B)>) -> Self::Output {
        Value((a.0, b.0))
    }
    fn split(v: Value<((A, A), (B, B))>) -> (Value<(A, A)>, Value<(B, B)>) {
        (Value((v.0).0), Value((v.0).1))
    }
}
impl Combine<NoValue, NoValue> for (NoValue, NoValue) {
    type Output = NoValue;
    fn combine(a: NoValue, b: NoValue) -> Self::Output {
        NoValue
    }
    fn split(v: NoValue) -> (NoValue, NoValue) {
        (NoValue, NoValue)
    }
}

pub trait DynamicValue:ValueT {
    fn get(&self, i: usize) -> f32x8;
    fn set(&mut self, i: usize, v: f32x8);
    fn add(&mut self, i: usize, v: f32x8);
    fn len(&self) -> usize;
}

impl DynamicValue for Value<(f32x8,)> {
    fn get(&self, i: usize) -> f32x8 {
        assert_eq!(i, 0);
        (self.0).0
    }

    fn set(&mut self, i: usize, v: f32x8) {
        assert_eq!(i, 0);
        (self.0).0 = v;
    }
    fn add(&mut self, i: usize, v: f32x8) {
        assert_eq!(i, 0);
        (self.0).0 += v;
    }
    fn len(&self) -> usize { 1 }
}
impl DynamicValue for Value<(f32,)> {
    fn get(&self, i: usize) -> f32x8 {
        assert_eq!(i, 0);
        f32x8::splat((self.0).0)
    }

    fn set(&mut self, i: usize, v: f32x8) {
        assert_eq!(i, 0);
        (self.0).0 = v.max_element();
    }
    fn add(&mut self, i: usize, v: f32x8) {
        assert_eq!(i, 0);
        (self.0).0 += v.max_element();
    }
    fn len(&self) -> usize { 1 }
}

impl DynamicValue for Value<(f32x8, f32x8)> {
    fn get(&self, i: usize) -> f32x8 {
        match i {
            0 => (self.0).0,
            1 => (self.0).1,
            _ => panic!()
        }
    }

    fn set(&mut self, i: usize, v: f32x8) {
        match i {
            0 => (self.0).0 = v,
            1 => (self.0).1 = v,
            _ => panic!()
        }
    }

    fn add(&mut self, i: usize, v: f32x8) {
        match i {
            0 => (self.0).0 += v,
            1 => (self.0).1 += v,
            _ => panic!()
        }
    }
    fn len(&self) -> usize { 2 }
}

impl DynamicValue for Value<((f32x8, f32x8), f32x8)> {
    fn get(&self, i: usize) -> f32x8 {
        let Value(((a,b),c)) = self.clone();
        match i {
            0 => a,
            1 => b,
            2 => c,
            _ => panic!()
        }
    }

    fn set(&mut self, i: usize, v: f32x8) {
        let Value(((a,b),c)) = self.clone();
        match i {
            0 => { *self = Value(((v,b),c)); }
            1 => { *self = Value(((a,v),c)); }
            2 => { *self = Value(((a,b),v)); }
            _ => panic!()
        }
    }

    fn add(&mut self, i: usize, v: f32x8) {
        let Value(((a,b),c)) = self.clone();
        match i {
            0 => { *self = Value(((a+v,b),c)); }
            1 => { *self = Value(((a,b+v),c)); }
            2 => { *self = Value(((a,b),c+v)); }
            _ => panic!()
        }
    }

    fn len(&self) -> usize { 3 }
}

impl DynamicValue for Value<((f32, f32), f32)> {
    fn get(&self, i: usize) -> f32x8 {
        let Value(((a,b),c)) = self.clone();
        f32x8::splat(match i {
            0 => a,
            1 => b,
            2 => c,
            _ => panic!()
        })
    }

    fn set(&mut self, i: usize, v: f32x8) {
        let Value(((a,b),c)) = self.clone();
        let v = v.max_element();
        match i {
            0 => { *self = Value(((v,b),c)); }
            1 => { *self = Value(((a,v),c)); }
            2 => { *self = Value(((a,b),v)); }
            _ => panic!()
        }
    }

    fn add(&mut self, i: usize, v: f32x8) {
        let Value(((a,b),c)) = self.clone();
        let v = v.max_element();
        match i {
            0 => { *self = Value(((a+v,b),c)); }
            1 => { *self = Value(((a,b+v),c)); }
            2 => { *self = Value(((a,b),c+v)); }
            _ => panic!()
        }
    }

    fn len(&self) -> usize { 3 }
}

impl DynamicValue for Value<(f32x8, (f32x8, f32x8))> {
    fn get(&self, i: usize) -> f32x8 {
        let Value((a,(b,c))) = self.clone();
        match i {
            0 => a,
            1 => b,
            2 => c,
            _ => panic!()
        }
    }

    fn set(&mut self, i: usize, v: f32x8) {
        let Value((a,(b,c))) = self.clone();
        match i {
            0 => { *self = Value((v,(b,c))); }
            1 => { *self = Value((a,(v,c))); }
            2 => { *self = Value((a,(b,v))); }
            _ => panic!()
        }
    }

    fn add(&mut self, i: usize, v: f32x8) {
        let Value((a,(b,c))) = self.clone();
        match i {
            0 => { *self = Value((a+v,(b,c))); }
            1 => { *self = Value((a,(b+v,c))); }
            2 => { *self = Value((a,(b,c+v))); }
            _ => panic!()
        }
    }

    fn len(&self) -> usize { 3 }
}

impl DynamicValue for Value<((f32, f32), (f32, f32))> {
    fn get(&self, i: usize) -> f32x8 {
        let Value(((a,b),(c,d))) = self.clone();
        f32x8::splat(match i {
            0 => a,
            1 => b,
            2 => c,
            3 => d,
            _ => panic!()
        })
    }

    fn set(&mut self, i: usize, v: f32x8) {
        let Value(((a,b),(c,d))) = self.clone();
        let v = v.max_element();
        match i {
            0 => { *self = Value(((v,b),(c,d))); }
            1 => { *self = Value(((a,v),(c,d))); }
            2 => { *self = Value(((a,b),(v,d))); }
            3 => { *self = Value(((a,b),(c,v))); }
            _ => panic!()
        }
    }

    fn add(&mut self, i: usize, v: f32x8) {
        let Value(((a,b),(c,d))) = self.clone();
        let v = v.max_element();
        match i {
            0 => { *self = Value(((a+v,b),(c,d))); }
            1 => { *self = Value(((a,b+v),(c,d))); }
            2 => { *self = Value(((a,b),(c+v,d))); }
            3 => { *self = Value(((a,b),(c,d+v))); }
            _ => panic!()
        }
    }

    fn len(&self) -> usize { 4 }
}

impl DynamicValue for NoValue {
    fn get(&self, i: usize) -> f32x8 {
        panic!()
    }

    fn set(&mut self, i: usize, v: f32x8) {
        panic!()
    }
    fn add(&mut self, i: usize, v: f32x8) {
        panic!()
    }

    fn len(&self) -> usize { 0 }
}
