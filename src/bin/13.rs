use aoc_2023::*;

fn main() {
    let input = load_input();

    let grids = input.paras();

    let mut hs1 = 0;
    let mut vs1 = 0;

    let mut hs2 = 0;
    let mut vs2 = 0;

    for grid in grids {
        let grid = parse_2d(&grid);

        if let Some(h) = find_reflection(grid.clone(), 0) {
            hs1 += h;
        } else if let Some(v) = find_reflection(transpose(grid.clone()), 0) {
            vs1 += v;
        }

        if let Some(h) = find_reflection(grid.clone(), 1) {
            hs2 += h;
        } else if let Some(v) = find_reflection(transpose(grid.clone()), 1) {
            vs2 += v;
        }
    }

    cp(hs1 * 100 + vs1);
    cp(hs2 * 100 + vs2);
}

fn diff(a: &[char], b: &[char]) -> usize {
    assert!(a.len() == b.len());
    let mut count = 0;
    for (a, b) in a.ii().zip(b) {
        if a != b {
            count += 1;
        }
    }
    count
}

fn find_reflection(grid: Vec<Vec<char>>, errors_needed: usize) -> Option<usize> {
    for test in 1..grid.len() {
        let mut i = test - 1;
        let mut j = test;
        let mut errors = 0;
        loop {
            if grid[i] != grid[j] {
                errors += diff(&grid[i], &grid[j]);
            }
            if errors > errors_needed {
                break;
            }
            if i == 0 || j == grid.len() - 1 {
                break;
            }
            i -= 1;
            j += 1;
        }
        if errors == errors_needed {
            return Some(test);
        }
    }
    None
}
