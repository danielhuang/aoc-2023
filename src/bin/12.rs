use aoc_2023::*;

fn main() {
    let input = load_input();

    for part2 in [false, true] {
        let mut sum = 0;
        let mut cache = FxHashMap::default();
        for line in input.lines() {
            let [mut unknown, known] = line.words().ca();
            let mut known = known.ints().cv();

            if part2 {
                unknown = [&*unknown].repeat(5).join("?");
                known = known.repeat(5);
            }
            let unknown = unknown.chars().map(|x| x as u8).cv();

            cache.clear();
            let cnt = count(&unknown, &known, 0, 0, 0, &mut cache);

            sum += cnt;
        }

        cp(sum);
    }
}

fn count(
    template: &[u8],
    counts: &[i64],
    template_i: usize,
    counts_i: usize,
    block_size: i64,
    cache: &mut FxHashMap<(usize, usize), usize>,
) -> usize {
    if counts_i == counts.len() {
        return template[template_i..]
            .iter()
            .all(|&x| x == b'.' || x == b'?') as usize;
    }
    if template_i == template.len() {
        return (counts[counts_i..] == [block_size]) as usize;
    }
    if block_size == 0 {
        if let Some(&result) = cache.get(&(template_i, counts_i)) {
            return result;
        }
    }
    let mut sum = 0;
    if (template[template_i] == b'#' || template[template_i] == b'?')
        && block_size < counts[counts_i]
    {
        sum += count(
            template,
            counts,
            template_i + 1,
            counts_i,
            block_size + 1,
            cache,
        );
    }
    if (template[template_i] == b'.' || template[template_i] == b'?')
        && block_size == counts[counts_i]
    {
        sum += count(template, counts, template_i + 1, counts_i + 1, 0, cache);
    }
    if (template[template_i] == b'.' || template[template_i] == b'?') && block_size == 0 {
        sum += count(template, counts, template_i + 1, counts_i, 0, cache);
    }
    if block_size == 0 {
        cache.insert((template_i, counts_i), sum);
    }
    sum
}
