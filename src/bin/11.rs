use aoc_2023::*;

fn main() {
    let input = load_input();

    let grid = parse_grid(&input, |x| x, '.');
    let galaxies = grid.findv('#');

    let b = bounds(grid.find(|x| x == '#'));
    let mut cols = (0..b.length(0)).cset();
    let mut rows = (0..b.length(1)).map(|x| -x).cset();

    for galaxy in galaxies.iter() {
        let x = galaxy[0];
        let y = galaxy[1];
        cols.remove(&x);
        rows.remove(&y);
    }

    for expansion in [2, 1000000] {
        let mut sum = 0;
        for (&a, &b) in galaxies.iter().tuple_combinations() {
            let (x1, x2) = (a[0], b[0]);
            let (y1, y2) = (a[1], b[1]);
            let extra_x = (x1.min(x2)..=x1.max(x2))
                .filter(|x| cols.contains(x))
                .count();
            let extra_y = (y1.min(y2)..=y1.max(y2))
                .filter(|x| rows.contains(x))
                .count();
            sum += (b - a).manhat();
            sum += extra_x as i64 * (expansion - 1);
            sum += extra_y as i64 * (expansion - 1);
        }

        cp(sum);
    }
}
