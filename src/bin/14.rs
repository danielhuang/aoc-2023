use aoc_2023::*;

fn weight(grid: &DefaultHashMap<Cell<2>, char>) -> i64 {
    let b = bounds(grid.keys().copied());
    let height = b.length(1);

    let mut sum = 0;
    for rock in grid.findv('O') {
        let y = rock[1];
        let h = height + y;
        sum += h;
    }

    sum
}

fn main() {
    let input = load_input();

    let grid_start = parse_grid(&input, |x| x, ' ');

    let mut grid = grid_start.clone();
    tick(&mut grid, 0);
    cp(weight(&grid));

    let grid = use_cycles(
        grid_start,
        |mut grid| {
            for i in 0..4 {
                tick(&mut grid, i);
            }
            grid
        },
        unparse_grid,
        1000000000,
    );

    cp(weight(&grid));
}

fn tick(grid: &mut DefaultHashMap<Cell<2>, char>, i: usize) {
    loop {
        let mut changed = false;
        for mut rock in grid.findv('O') {
            let mut next = [rock.up(1), rock.left(1), rock.down(1), rock.right(1)][i % 4];
            while grid[next] == '.' {
                grid[rock] = '.';
                grid[next] = 'O';
                rock = next;
                next = [rock.up(1), rock.left(1), rock.down(1), rock.right(1)][i % 4];
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}
