def filter_tags(row, filtered_tags):
    return any(tag in eval(row["tags"]) for tag in filtered_tags)