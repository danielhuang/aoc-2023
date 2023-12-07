use aoc_2023::*;

fn main() {
    let input = load_input();

    let [time, distance] = input.lines().ca();

    for part2 in [false, true] {
        let (time, distance) = if part2 {
            (
                time.replace(' ', "").ints(),
                distance.replace(' ', "").ints(),
            )
        } else {
            (time.ints(), distance.ints())
        };

        let mut total = 1;

        for (total_time, prev_distance) in time.ii().zip(distance) {
            let mut count = 0;
            for speed in 0..total_time {
                let travel_time = total_time - speed;
                let total_distance = travel_time * speed;
                if total_distance > prev_distance {
                    count += 1;
                }
            }
            total *= count;
        }

        cp(total);
    }
}
