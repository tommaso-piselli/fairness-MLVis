import re


def calculate_percentages(data):
    total = sum(data.values())
    return {k: (v, round(v / total * 100, 1)) for k, v in data.items()}


def read_file_content(filename):
    with open(filename, 'r') as file:
        content = file.read()
    return content


def parse_file_content(content):
    sections = re.split(r'(\w+):\n', content)[1:]
    return dict(zip(sections[::2], sections[1::2]))


def get_percentage(filename, feature, *values):
    content = read_file_content(filename)
    sections = parse_file_content(content)

    if feature not in sections:
        return f"Error: Feature '{feature}' not found in the file."

    items = re.findall(
        r'  (\w+(?:[\s-]\w+)*(?:\+)?): (\d+) - ([\d.]+)%', sections[feature])
    percentages = {k: (int(v), float(p)) for k, v, p in items}

    results = {}
    total_count = 0
    total_percentage = 0

    for value in values:
        if value not in percentages:
            return f"Error: Value '{value}' not found in feature '{feature}'."
        count, percentage = percentages[value]
        results[value] = {"count": count, "percentage": percentage}
        total_count += count
        total_percentage += percentage

    if len(values) > 1:
        results["Combined"] = {"count": total_count,
                               "percentage": round(total_percentage, 1)}

    return results


def process_file(filename):
    with open(filename, 'r') as file:
        content = file.read()
    sections = re.split(r'(\w+):\n', content)[1:]
    sections = dict(zip(sections[::2], sections[1::2]))
    for section, data in sections.items():
        if section in ['Region', 'Ethnicity', 'Age', 'Gender', 'Status']:
            # Updated regex to handle '65+' and similar cases
            items = re.findall(r'  (\w+(?:[\s-]\w+)*(?:\+)?): (\d+)', data)
            data_dict = {k: int(v) for k, v in items}
            percentages = calculate_percentages(data_dict)
            new_data = f"{section}:\n"
            for k, (v, p) in percentages.items():
                new_data += f"  {k}: {v} - {p}%\n"
            sections[section] = new_data
    updated_content = "Graph: " + \
        content.split("Graph: ", 1)[1].split("Region:", 1)[
            0] + "\n".join(sections.values())
    with open(filename, 'w') as file:
        file.write(updated_content)


def process_multiple_files(file_pattern, feature, *values):
    all_results = []
    for index in range(0, 24):
        filename = file_pattern.format(index)
        results = get_percentage(filename, feature, *values)
        if isinstance(results, str):  # Error occurred
            print(f"Error in file {filename}: {results}")
            continue
        all_results.append(results)

    # Compute min and max for each value and combined
    min_max = {}
    for value in list(values) + (["Combined"] if len(values) > 1 else []):
        percentages = [result[value]["percentage"]
                       for result in all_results if value in result]
        if percentages:
            min_max[value] = {"min": min(percentages), "max": max(percentages)}

    return min_max


# Usage
# Adjust this pattern to match your file naming
file_pattern = r'data\preprocessing\summary\AVN\summary\graph_spa_500_{}_summary.txt'
min_max_results = process_multiple_files(file_pattern, 'Status', 'normal')

# Print results
for value, stats in min_max_results.items():
    print(f"{value}:")
    print(f"  Min: {stats['min']}%")
    print(f"  Max: {stats['max']}%")
