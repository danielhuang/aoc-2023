#![allow(incomplete_features)]

use derive_more::{Deref, DerefMut};
use itertools::Itertools;
use std::iter::{self};
use std::ops::{Deref, Neg};
use std::{
    array::from_fn,
    fmt::Debug,
    ops::{Add, AddAssign, Div, Mul, Sub, SubAssign},
};

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Deref, DerefMut)]
pub struct Point<const N: usize>(pub [i64; N]);

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Deref, DerefMut)]
pub struct Vector<const N: usize>(pub [i64; N]);

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Deref, DerefMut)]
pub struct Cell<const N: usize>(pub [i64; N]);

pub type Point2 = Point<2>;
pub type Vector2 = Vector<2>;
pub type Cell2 = Cell<2>;

pub type Point3 = Point<3>;
pub type Vector3 = Vector<3>;
pub type Cell3 = Cell<3>;

pub const fn count_adj_diag(dim: usize) -> usize {
    (3usize).pow(dim as _) - 1
}

pub const fn count_corners(dim: usize) -> usize {
    (2usize).pow(dim as _)
}

#[must_use]
pub fn p2(x: i64, y: i64) -> Point<2> {
    Point::new([x, y])
}

#[must_use]
pub fn p3(x: i64, y: i64, z: i64) -> Point<3> {
    Point::new([x, y, z])
}

#[must_use]
pub fn p4(w: i64, x: i64, y: i64, z: i64) -> Point<4> {
    Point::new([w, x, y, z])
}

#[must_use]
pub fn v2(x: i64, y: i64) -> Vector<2> {
    Vector::new([x, y])
}

#[must_use]
pub fn v3(x: i64, y: i64, z: i64) -> Vector<3> {
    Vector::new([x, y, z])
}

#[must_use]
pub fn v4(w: i64, x: i64, y: i64, z: i64) -> Vector<4> {
    Vector::new([w, x, y, z])
}

#[must_use]
pub fn c2(x: i64, y: i64) -> Cell<2> {
    Cell::new([x, y])
}

#[must_use]
pub fn c3(x: i64, y: i64, z: i64) -> Cell<3> {
    Cell::new([x, y, z])
}

#[must_use]
pub fn c4(w: i64, x: i64, y: i64, z: i64) -> Cell<4> {
    Cell::new([w, x, y, z])
}

impl<const N: usize> Point<N> {
    pub fn origin() -> Self {
        Self(from_fn(|_| 0))
    }

    pub fn new(x: [i64; N]) -> Self {
        Self(x)
    }

    pub fn vector(&self) -> Vector<N> {
        *self - Self::origin()
    }

    pub fn adj(&self) -> [Self; N * 2] {
        Vector::<N>::adj().map(|x| x + *self)
    }

    pub fn adj_diag(&self) -> [Self; count_adj_diag(N)] {
        Vector::<N>::adj_diag().map(|x| x + *self)
    }
}

impl<const N: usize> Default for Point<N> {
    fn default() -> Self {
        Self(from_fn(|_| 0))
    }
}

impl<const N: usize> Default for Vector<N> {
    fn default() -> Self {
        Self(from_fn(|_| 0))
    }
}

impl<const N: usize> Default for Cell<N> {
    fn default() -> Self {
        Self(from_fn(|_| 0))
    }
}

impl<const N: usize> Add for Vector<N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl<const N: usize> Add<Point<N>> for Vector<N> {
    type Output = Point<N>;

    fn add(self, rhs: Point<N>) -> Self::Output {
        Point(from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl<const N: usize> Mul<i64> for Vector<N> {
    type Output = Self;

    fn mul(self, rhs: i64) -> Self::Output {
        Self(self.0.map(|x| x * rhs))
    }
}

impl<const N: usize> Div<i64> for Vector<N> {
    type Output = Self;

    fn div(self, rhs: i64) -> Self::Output {
        Self(self.0.map(|x| x / rhs))
    }
}

impl<const N: usize> Add<Vector<N>> for Point<N> {
    type Output = Self;

    fn add(self, rhs: Vector<N>) -> Self::Output {
        Self(from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl<const N: usize> Sub<Vector<N>> for Point<N> {
    type Output = Self;

    fn sub(self, rhs: Vector<N>) -> Self::Output {
        Self(from_fn(|i| self.0[i] - rhs.0[i]))
    }
}

impl<const N: usize> Sub for Point<N> {
    type Output = Vector<N>;

    fn sub(self, rhs: Self) -> Self::Output {
        Vector(from_fn(|i| self.0[i] - rhs.0[i]))
    }
}

impl<const N: usize> Sub for Vector<N> {
    type Output = Vector<N>;

    fn sub(self, rhs: Self) -> Self::Output {
        Vector(from_fn(|i| self.0[i] - rhs.0[i]))
    }
}

impl<const N: usize> AddAssign<Vector<N>> for Point<N> {
    fn add_assign(&mut self, rhs: Vector<N>) {
        *self = *self + rhs;
    }
}

impl<const N: usize> SubAssign<Vector<N>> for Point<N> {
    fn sub_assign(&mut self, rhs: Vector<N>) {
        *self = *self - rhs;
    }
}

impl<const N: usize> AddAssign for Vector<N> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const N: usize> Neg for Vector<N> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * -1
    }
}

pub trait Cartesian<const N: usize>:
    Sized + Default + Clone + Copy + Deref<Target = [i64; N]> + Eq
{
    fn inner(&self) -> [i64; N];

    fn new(x: [i64; N]) -> Self;

    fn manhat(&self) -> i64 {
        self.inner().into_iter().map(|x| x.abs()).sum()
    }

    fn manhat_diag(&self) -> i64 {
        self.inner().into_iter().map(|x| x.abs()).max().unwrap()
    }

    fn point(&self) -> Point<N> {
        Point(self.inner())
    }

    fn vector(&self) -> Vector<N> {
        Vector(self.inner())
    }

    fn cell(&self) -> Cell<N> {
        Cell(self.inner())
    }
}

pub trait Cartesian2: Cartesian<2> {
    fn up(&self, n: i64) -> Self {
        let mut x = self.inner();
        x[1] += n;
        Self::new(x)
    }

    fn down(&self, n: i64) -> Self {
        let mut x = self.inner();
        x[1] -= n;
        Self::new(x)
    }

    fn left(&self, n: i64) -> Self {
        let mut x = self.inner();
        x[0] -= n;
        Self::new(x)
    }

    fn right(&self, n: i64) -> Self {
        let mut x = self.inner();
        x[0] += n;
        Self::new(x)
    }

    fn diamond(&self, radius: i64) -> Vec<Self> {
        assert!(radius >= 0);
        if radius == 0 {
            return vec![*self];
        }
        let mut output = vec![];
        let mut pos = self.inner();
        pos[1] -= radius;
        for (vel_x, vel_y) in [(1, 1), (-1, 1), (-1, -1), (1, -1)] {
            for _ in 0..radius {
                output.push(Self::new(pos));
                pos[0] += vel_x;
                pos[1] += vel_y;
            }
        }
        output
    }

    fn filled_diamond(&self, radius: i64) -> Vec<Self> {
        (0..=radius).flat_map(|r| self.diamond(r)).collect()
    }

    fn square_border(&self, radius: i64) -> Vec<Self> {
        assert!(radius >= 0);
        if radius == 0 {
            return vec![*self];
        }
        let mut output = vec![];
        let mut pos = self.down(radius).left(radius);
        for _ in 0..radius * 2 {
            output.push(pos);
            pos = pos.up(1);
        }
        for _ in 0..radius * 2 {
            output.push(pos);
            pos = pos.right(1);
        }
        for _ in 0..radius * 2 {
            output.push(pos);
            pos = pos.down(1);
        }
        for _ in 0..radius * 2 {
            output.push(pos);
            pos = pos.left(1);
        }
        output
    }

    fn filled_square(&self, radius: i64) -> Vec<Self> {
        (0..=radius).flat_map(|r| self.square_border(r)).collect()
    }

    fn rotate3(&self) -> Self {
        Self::new([self[1], -self[0]])
    }

    fn rotate(&self) -> Self {
        Self::new([-self[1], self[0]])
    }

    fn turn_left(&self) -> Self {
        self.rotate()
    }

    fn turn_right(&self) -> Self {
        self.rotate3()
    }

    fn flip_real_part(&self) -> Self {
        Self::new([-self[0], self[1]])
    }

    fn flip_imag_part(&self) -> Self {
        Self::new([self[0], -self[1]])
    }

    fn flip_over_fwslash(&self) -> Self {
        Self::new([self[1], self[0]])
    }

    fn flip_over_bkslash(&self) -> Self {
        Self::new([-self[1], -self[0]])
    }
}

impl<T: Cartesian<2>> Cartesian2 for T {}

impl<const N: usize> Cartesian<N> for Point<N> {
    fn inner(&self) -> [i64; N] {
        self.0
    }

    fn new(x: [i64; N]) -> Self {
        Self(x)
    }
}

impl<const N: usize> Cartesian<N> for Vector<N> {
    fn inner(&self) -> [i64; N] {
        self.0
    }

    fn new(x: [i64; N]) -> Self {
        Self(x)
    }
}

impl<const N: usize> Debug for Point<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Point[{}]", self.0.iter().join(", "))
    }
}

impl<const N: usize> Debug for Vector<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vector[{}]", self.0.iter().join(", "))
    }
}

impl<const N: usize> Cell<N> {
    pub fn origin() -> Self {
        Self(from_fn(|_| 0))
    }

    pub fn new(x: [i64; N]) -> Self {
        Self(x)
    }

    pub fn vector(&self) -> Vector<N> {
        *self - Self::origin()
    }

    pub fn point(&self) -> Point<N> {
        Point::new(self.0)
    }

    pub fn adj(&self) -> [Self; N * 2] {
        Vector::<N>::adj().map(|x| x + *self)
    }

    pub fn adj_diag(&self) -> [Self; count_adj_diag(N)] {
        Vector::<N>::adj_diag().map(|x| x + *self)
    }

    pub fn corners(&self) -> [Point<N>; count_corners(N)] {
        from_fn(|i| {
            let mut point = self.point();
            for n in 0..N {
                if i & (1 << n) != 0 {
                    point.0[n] += 1;
                }
            }
            point
        })
    }

    pub fn corners_minmax(&self) -> [Point<N>; 2] {
        [self.point(), self.point() + Vector::new(from_fn(|_| 1))]
    }
}

impl<const N: usize> Vector<N> {
    pub fn zero() -> Self {
        Self(from_fn(|_| 0))
    }

    pub fn new(x: [i64; N]) -> Self {
        Self(x)
    }

    pub fn point(&self) -> Self {
        Self::zero() + *self
    }

    pub fn adj_diag() -> [Self; count_adj_diag(N)] {
        let mut result = [Self::default(); count_adj_diag(N)];

        let mut i = 0;
        let mut v = [-1; N];
        for _ in 0..(result.len() - 1) {
            if v.iter().all(|&x| x == 0) {
                v[0] = 1;
            }
            result[i] = Self::new(v);
            let mut j = 0;
            loop {
                v[j] += 1;
                if v[j] > 1 {
                    v[j] = -1;
                    j += 1;
                } else {
                    break;
                }
            }
            i += 1;
        }
        result[i] = Self::new(v);

        result
    }

    pub fn adj() -> [Self; N * 2] {
        from_fn(|i| {
            let axis = i / 2;
            let mut result = Self::default().inner();

            result[axis] = if i % 2 == 0 { 1 } else { -1 };

            Self::new(result)
        })
    }

    /// normalize to the parallel basis vector
    fn normalize(self) -> Self {
        assert!(self.is_aligned());
        self / self.manhat()
    }

    /// reduces all axis values to 1, 0, or -1
    fn normalize_diag(self) -> Self {
        Self(self.map(|x| x.signum()))
    }

    /// vector is a multiple of a basis vector (single axis)
    fn is_aligned(self) -> bool {
        self.iter().any(|&x| x == 0)
    }
}

impl Vector<3> {
    pub fn cross(&self, other: Self) -> Self {
        let Vector([a, b, c]) = self;
        let Vector([x, y, z]) = other;
        Vector([b * z - c * y, c * x - a * z, a * y - b * x])
    }
}

impl<const N: usize> Add<Cell<N>> for Vector<N> {
    type Output = Cell<N>;

    fn add(self, rhs: Cell<N>) -> Self::Output {
        Cell(from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl<const N: usize> Add<Vector<N>> for Cell<N> {
    type Output = Self;

    fn add(self, rhs: Vector<N>) -> Self::Output {
        Self(from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl<const N: usize> Sub<Vector<N>> for Cell<N> {
    type Output = Self;

    fn sub(self, rhs: Vector<N>) -> Self::Output {
        Self(from_fn(|i| self.0[i] - rhs.0[i]))
    }
}

impl<const N: usize> Sub for Cell<N> {
    type Output = Vector<N>;

    fn sub(self, rhs: Self) -> Self::Output {
        Vector(from_fn(|i| self.0[i] - rhs.0[i]))
    }
}

impl<const N: usize> AddAssign<Vector<N>> for Cell<N> {
    fn add_assign(&mut self, rhs: Vector<N>) {
        *self = *self + rhs;
    }
}

impl<const N: usize> SubAssign<Vector<N>> for Cell<N> {
    fn sub_assign(&mut self, rhs: Vector<N>) {
        *self = *self - rhs;
    }
}

impl<const N: usize> Cartesian<N> for Cell<N> {
    fn inner(&self) -> [i64; N] {
        self.0
    }

    fn new(x: [i64; N]) -> Self {
        Self(x)
    }
}

impl<const N: usize> Debug for Cell<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cell[{}]", self.0.iter().join(", "))
    }
}

pub trait Boundable<const N: usize> {
    fn points(&self) -> impl Iterator<Item = Point<N>>;
}

impl<const N: usize> Boundable<N> for Point<N> {
    fn points(&self) -> impl Iterator<Item = Point<N>> {
        [*self].into_iter()
    }
}

impl<const N: usize> Boundable<N> for Cell<N> {
    fn points(&self) -> impl Iterator<Item = Point<N>> {
        self.corners_minmax().into_iter()
    }
}

#[derive(Clone, Copy, Eq, PartialEq, PartialOrd, Ord, Hash, Debug)]
pub struct Matrix<const N: usize> {
    pub cols: [Vector<N>; N],
}

impl<const N: usize> Matrix<N> {
    pub fn new(cols: [Vector<N>; N]) -> Self {
        Self { cols }
    }
}

impl<const N: usize> Default for Matrix<N> {
    fn default() -> Self {
        Self {
            cols: from_fn(|i| Vector(from_fn(|j| if i == j { 1 } else { 0 }))),
        }
    }
}

impl<const N: usize> Add for Matrix<N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            cols: from_fn(|i| self.cols[i] + rhs.cols[i]),
        }
    }
}

impl<const N: usize> Sub for Matrix<N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            cols: from_fn(|i| self.cols[i] - rhs.cols[i]),
        }
    }
}

impl<const N: usize> Mul<i64> for Matrix<N> {
    type Output = Self;

    fn mul(self, rhs: i64) -> Self::Output {
        Self {
            cols: from_fn(|i| self.cols[i] * rhs),
        }
    }
}

impl<const N: usize> Mul<Vector<N>> for Matrix<N> {
    type Output = Vector<N>;

    fn mul(self, rhs: Vector<N>) -> Self::Output {
        (0..N)
            .map(|i| self.cols[i] * rhs[i])
            .reduce(|a, b| a + b)
            .unwrap_or_default()
    }
}

impl<const N: usize> Mul for Matrix<N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            cols: rhs.cols.map(|col| self * col),
        }
    }
}

pub fn all_3d_rotations() -> Vec<Matrix<3>> {
    let base = [
        v3(1, 0, 0),
        v3(0, 1, 0),
        v3(0, 0, 1),
        v3(-1, 0, 0),
        v3(0, -1, 0),
        v3(0, 0, -1),
    ];
    let mut result = vec![];
    for &v1 in &base {
        for &v2 in &base {
            if v1 != v2 && v1 * -1 != v2 {
                result.push(Matrix {
                    cols: [v1, v2, v1.cross(v2)],
                })
            }
        }
    }
    assert_eq!(result.len(), 24);
    result
}

pub trait Absolute<const N: usize>:
    Cartesian<N>
    + Sub<Output = Vector<N>>
    + Add<Vector<N>, Output = Self>
    + AddAssign<Vector<N>>
    + Sub<Vector<N>, Output = Self>
    + SubAssign<Vector<N>>
{
    fn is_aligned_with(self, other: Self) -> bool {
        (other - self).is_aligned()
    }

    fn go_straight(self, vel: Vector<N>) -> impl Iterator<Item = Self> {
        let mut pos = self;
        iter::from_fn(move || {
            pos += vel;
            Some(pos)
        })
    }

    fn goto_straight(self, dest: Self) -> Vec<Self> {
        let unit = (dest - self).normalize();
        std::iter::once(self)
            .chain(self.go_straight(unit).take((dest - self).manhat() as _))
            .collect()
    }

    fn goto(self, dest: Self) -> Vec<Self> {
        let unit = (dest - self).normalize_diag();
        let result: Vec<_> = std::iter::once(self)
            .chain(
                self.go_straight(unit)
                    .take((dest - self).manhat_diag() as _),
            )
            .collect();
        if !result.contains(&dest) {
            println!("WARNING: line is not straight and does not reach destination")
        }
        result
    }
}

impl<const N: usize> Absolute<N> for Point<N> {}
impl<const N: usize> Absolute<N> for Cell<N> {}

#[track_caller]
pub fn charvel(c: char) -> Vector<2> {
    match c {
        '^' => v2(0, 1),
        'v' | 'V' => v2(0, -1),
        '<' => v2(-1, 0),
        '>' => v2(1, 0),
        _ => panic!("charvel: {c} is not a valid direction"),
    }
}
