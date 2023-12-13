use aoc_2023::*;

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum Reflection {
    H(usize),
    V(usize),
}

fn main() {
    let input = load_input();

    let grids = input.paras();

    let mut hs1 = 0;
    let mut vs1 = 0;

    let mut hs2 = 0;
    let mut vs2 = 0;

    for grid in grids {
        let grid = parse_2d(&grid);

        let before = find_all_reflections(grid.clone(), 0, 0, false);

        match before.one() {
            Reflection::H(x) => hs1 += x,
            Reflection::V(x) => vs1 += x,
        }

        'a: for r in 0..grid.len() {
            for c in 0..grid[0].len() {
                let mut grid = grid.clone();

                let orig = grid[r][c];
                if orig == '#' {
                    grid[r][c] = '.';
                } else if orig == '.' {
                    grid[r][c] = '#';
                } else {
                    panic!();
                }

                let after = find_all_reflections(grid.clone(), r, c, true);
                if let Some(new) = after.cii().find(|x| !before.contains(x)) {
                    match new {
                        Reflection::H(h) => {
                            hs2 += h;
                        }
                        Reflection::V(v) => {
                            vs2 += v;
                        }
                    }
                    break 'a;
                }
            }
        }
    }

    cp(hs1 * 100 + vs1);
    cp(hs2 * 100 + vs2);
}

fn find_all_reflections(x: Vec<Vec<char>>, r: usize, c: usize, need: bool) -> Vec<Reflection> {
    let mut result = vec![];
    for x in find_reflecting_rows(x.clone(), r, need) {
        result.push(Reflection::H(x))
    }
    for x in find_reflecting_rows(transpose(x.clone()), c, need) {
        result.push(Reflection::V(x))
    }
    result
}

fn find_reflecting_rows(grid: Vec<Vec<char>>, r: usize, need: bool) -> Vec<usize> {
    let mut result = vec![];
    for test in 1..(grid.len()) {
        let mut i = test - 1;
        let mut j = test;
        let mut good = true;
        let mut counts = 0;
        let mut used = false;
        loop {
            assert!(i != j);
            if grid[i] == grid[j] {
                counts += 1;
            } else {
                good = false;
            }

            if i == r || j == r {
                used = true;
            }

            if i == 0 || i == grid.len() - 1 || j == 0 || j == grid.len() - 1 {
                break;
            }

            i -= 1;
            j += 1;
        }
        if good && counts > 0 && (used || !need) {
            result.push(test);
        }
    }
    result
}
