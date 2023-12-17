use aoc_2023::*;

fn main() {
    let input = load_input();

    let grid = parse_grid(&input, |x| x.int(), 99999);

    let b = bounds(grid.keys().cloned());
    let width = b.length(0);
    let height = b.length(1);

    for range in [1..=3, 4..=10] {
        let result = dijkstra(
            [(c2(0, 0), v2(0, 1)), (c2(0, 0), v2(1, 0))],
            |prev| {
                let (pos, vel) = *prev;
                let mut results = vec![];
                for vel in [vel.turn_left(), vel.turn_right()] {
                    for amount in range.clone() {
                        let new_pos = vel * amount + pos;
                        let cost = pos.goto(new_pos).ii().skip(1).map(|x| grid[x]).sumi();
                        results.push(((new_pos, vel), cost));
                    }
                }
                results
            },
            |x| x.0 == c2(width - 1, -(height - 1)),
        );

        cp(result.unwrap().1);
    }
}
