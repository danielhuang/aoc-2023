use aoc_2023::*;

fn letter(s: char) -> usize {
    ['x', 'm', 'a', 's'].ii().position(|x| x == s).unwrap()
}

fn accepts(rules: &HashMap<String, Vec<(String, String)>>, nums: [i64; 4]) -> bool {
    let mut cur = "in".tos();
    while rules.contains_key(&cur) {
        let cond = rules[&cur].clone();
        for (src, dest) in cond {
            if src.contains('<') {
                let min = src.ints()[0];
                let var = src.chars().next().unwrap();
                let num = nums[letter(var)];
                if num < min {
                    cur = dest.tos();
                    break;
                }
            } else if src.contains('>') {
                let max = src.ints()[0];
                let var = src.chars().next().unwrap();
                let num = nums[letter(var)];
                if num > max {
                    cur = dest.tos();
                    break;
                }
            } else {
                cur = dest.tos();
            }
        }
    }
    match &*cur {
        "A" => true,
        "R" => false,
        _ => unreachable!(),
    }
}

fn count(
    rules: &HashMap<String, Vec<(String, String)>>,
    label: String,
    mut intervals: [Intervals; 4],
) -> usize {
    if label == "R" {
        return 0;
    }
    if label == "A" {
        return intervals.ii().map(|x| x.covered_size() as usize).product();
    }
    let mut sum = 0;
    for (src, dest) in rules[&label].clone() {
        if src.contains('<') {
            let hi = src.ints()[0];
            let var = src.chars().next().unwrap();
            let (accept, next) = intervals[letter(var)].clone().split_at(hi);
            intervals[letter(var)] = accept;
            sum += count(rules, dest.tos(), intervals.clone());
            intervals[letter(var)] = next;
        } else if src.contains('>') {
            let lo = src.ints()[0];
            let var = src.chars().next().unwrap();
            let (next, accept) = intervals[letter(var)].clone().split_at(lo + 1);
            intervals[letter(var)] = accept;
            sum += count(rules, dest.tos(), intervals.clone());
            intervals[letter(var)] = next;
        } else {
            sum += count(rules, dest.tos(), intervals.clone());
        }
    }
    sum
}

fn main() {
    let input = load_input();

    let (head, body) = input.split_once("\n\n").unwrap();

    let mut rules = HashMap::new();

    for rule in head.lines() {
        let name = rule.alphanumeric_words()[0].clone();
        let rule = rule.split('{').nth(1).unwrap();
        let rule = rule.strip_suffix('}').unwrap();
        let mut labels = vec![];
        for cond in rule.split(',') {
            if let Some((left, right)) = cond.split_once(':') {
                labels.push((left.tos(), right.tos()));
            } else {
                labels.push(("".tos(), cond.tos()));
            }
        }
        rules.insert(name, labels);
    }

    let mut sum = 0;
    for cand in body.lines() {
        let xmas = cand.ints().ca();
        if accepts(&rules, xmas) {
            sum += xmas.ii().sumi();
        }
    }
    cp(sum);

    let mut i = Intervals::default();
    i.add(1, 4001);
    cp(count(&rules, "in".tos(), vec![i; 4].ca()));
}
