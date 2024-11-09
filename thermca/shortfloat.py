#!/Users/schroeder/mambaforge/bin/python

"""Strip leading and trailing zeros of floats
in given python file after Black formatter

Examples:
    0.0 -> 0.,
    0.1 -> .1,
    1.0 -> 1.
    1.e01 -> 1.e1
"""
# Add to PyCharm Tools
# PyCharm -> Preferences… -> Tools -> External Tools -> Click + -> select "shortfloat.py"
# Insert "$FilePath$" to Arguments
# Insert "$FileDir$" to Working directory
# Make file executable:  $sudo chmod +x shortfloat.py

import shutil
import sys
import re

# Find float values containing a dot
float_pattern = r'[-+]?[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?'


def strip_leading_and_trailing_zeros(match):
    # print(f"{match=}")
    s = match.group()  # Returns the entire string that matched
    if float(s) == 0.0 or float(s) == -0.0:
        return '0.'
    # TODO: add underscores!
    # Remove leading zeros before dot
    s = re.sub(r'^([+-]?)(0*)(0\.)', r'\1.', s)
    # Remove trailing zeros after dot and optionally some counting digits
    s = re.sub(r'\.([0-9]*?)([1-9]*)(0+)([-eE]|$)', r'.\1\2\4', s)
    # Remove leading zero of exponent
    s = re.sub(r'e([+-]?)0+', r'e\1', s)
    # print(f"{s=}")
    return s


# Original and stripped combinations
test_strip_combs = [
    ('0.0', '0.'),
    ('0.', '0.'),
    ('.0', '0.'),
    ('0.1', '.1'),
    ('.1', '.1'),
    ('1.0', '1.'),
    ('1.', '1.'),
    ('10.0', '10.'),
    ('10.', '10.'),
    ('10.00', '10.'),
    ('10.01', '10.01'),
    ('10.010', '10.01'),
    ('10.0101', '10.0101'),
    ('10.01010', '10.0101'),
    ('1.0e01', '1.e1'),
    ('1.00e01', '1.e1'),
    ('1.0e001', '1.e1'),
    ('1.0e0001', '1.e1'),
    ('1.0e-01', '1.e-1'),
    ('1.00e-01', '1.e-1'),
    ('1.0e-001', '1.e-1'),
    ('1.0e-0001', '1.e-1'),
    ('-0.0', '0.'),
    ('-0.', '0.'),
    ('-.0', '0.'),
    ('-0.1', '-.1'),
    ('-.1', '-.1'),
    ('-1.0', '-1.'),
    ('-1.', '-1.'),
    ('-10.0', '-10.'),
    ('-10.', '-10.'),
    ('-10.00', '-10.'),
    ('-10.01', '-10.01'),
    ('-10.010', '-10.01'),
    ('-10.0101', '-10.0101'),
    ('-10.01010', '-10.0101'),
    ('-1.0e01', '-1.e1'),
    ('-1.00e01', '-1.e1'),
    ('-1.0e001', '-1.e1'),
    ('-1.0e0001', '-1.e1'),
    ('-1.0e-01', '-1.e-1'),
    ('-1.00e-01', '-1.e-1'),
    ('-1.0e-001', '-1.e-1'),
    ('-1.0e-0001', '-1.e-1'),
]

if __name__ == '__main__':
    # Run tests if no file as argument
    if len(sys.argv) == 1:
        print(
            "Use python file containing floats as argument.\n"
            "Unnecessary zeros of floats of will be deleted."
        )
        print("Running tests ... ", end="")
        for float_str, expected_str in test_strip_combs:
            stripped_str = re.sub(
                float_str, strip_leading_and_trailing_zeros, float_str
            )
            assert (
                stripped_str == expected_str
            ), f"Shortened '{float_str} ' should be '{expected_str}' not '{stripped_str}'!"
        print("OK")
    else:
        old = sys.argv[1]
        new = sys.argv[1] + '_new'
        stripped_zeros = 0
        with open(old) as old_file, open(new, 'w') as new_file:
            for line in old_file:
                new_line = re.sub(float_pattern, strip_leading_and_trailing_zeros, line)
                new_file.write(new_line)
                stripped_zeros += len(line) - len(new_line)
        shutil.move(new, old)
        print(f"Stripped {stripped_zeros} zeros from floats.")
