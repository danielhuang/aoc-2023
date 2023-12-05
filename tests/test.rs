use aoc_2023::*;

#[test]
fn intervals() {
    let mut intervals = Intervals::default();
    intervals.add(0, 10);
    intervals.add(10, 20);
    assert!(intervals.iter().cv() == (0..20).cv());
}

#[test]
fn intervals2() {
    let mut intervals = Intervals::default();
    intervals.add(0, 20);
    intervals.remove(5, 10);
    assert!(intervals.iter().cv() == (0..5).chain(10..20).cv());
}

#[test]
fn intervals3() {
    let mut intervals = Intervals::default();
    intervals.add(0, 20);
    intervals.remove(0, 20);
    assert!(intervals.iter().cv().is_empty());
}

#[test]
fn intervals4() {
    let mut intervals = Intervals::default();
    intervals.add(0, 20);
    intervals.remove(-1, 21);
    assert!(intervals.iter().cv().is_empty());
}

#[test]
fn intervals5() {
    let mut intervals = Intervals::default();
    intervals.add(0, 20);
    intervals.remove(5, 20);
    assert!(intervals.iter().cv() == (0..5).cv());
}

#[test]
fn intervals6() {
    let mut intervals = Intervals::default();
    intervals.add(0, 5);
    intervals.add(6, 20);
    assert!(intervals.iter().cv() == (0..5).chain(6..20).cv());
}

#[test]
fn intervals7() {
    let mut a = Intervals::default();
    a.add(0, 5);
    let mut b = Intervals::default();
    b.add(5, 10);
    let intervals = Intervals::union(a, b);
    assert!(intervals.iter().cv() == (0..10).cv());
}

#[test]
fn intervals_iter() {
    let mut a = Intervals::default();
    a.add(1, 4);
    a.add(5, 10);
    a.add(15, 20);
    assert!(a.iter().cv() == (1..4).chain(5..10).chain(15..20).cv());
    assert!(a.iter().rev().cv() == (1..4).chain(5..10).chain(15..20).rev().cv());
}
