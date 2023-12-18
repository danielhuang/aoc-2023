use aoc_2023::*;

fn main() {
    let input = load_input();

    let grid = parse_grid(&input, |x| x.int(), 99999);

    let b = bounds(grid.keys().cloned());
    let width = b.length(0);
    let height = b.length(1);

    for (min, max) in [(1, 3), (4, 10)] {
        let result = dijkstra(
            [(c2(0, 0), v2(0, 1)), (c2(0, 0), v2(1, 0))],
            |&(pos, vel)| {
                let mut results = vec![];
                for vel in [vel.turn_left(), vel.turn_right()] {
                    let mut cost = 0;
                    let mut new_pos = pos;
                    for i in 1..=max {
                        new_pos += vel;
                        cost += grid[new_pos];
                        if i >= min {
                            results.push(((new_pos, vel), cost));
                        }
                    }
                }
                results
            },
            |x| x.0 == c2(width - 1, -(height - 1)),
        );

        cp(result.unwrap().1);
    }
}
