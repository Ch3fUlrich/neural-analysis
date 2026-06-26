import sys

filepath = "src/neural_analysis/metrics/pairwise_core.py"
with open(filepath, "r") as f:
    content = f.read()

# We want to add a comment about feature_similarity supersession.
# Currently the docstring of compare_datasets says:
#     compare_distribution_groups : Legacy API for group comparisons
# We can update it to also mention feature_similarity.

search_str = "    compare_distribution_groups : Legacy API for group comparisons"
replace_str = "    compare_distribution_groups : Legacy API for group comparisons\n    feature_similarity : Legacy API from todo/Manimeasure.py superseded by mode='all-pairs'"

if search_str in content:
    new_content = content.replace(search_str, replace_str)
    with open(filepath, "w") as f:
        f.write(new_content)
    print("Successfully updated pairwise_core.py docstring")
else:
    print("Could not find search string in pairwise_core.py")
