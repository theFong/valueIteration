from project3cs360s2019 import main as run
import pytest
from shutil import copyfile

@pytest.mark.parametrize("input_file_num,output_file_num", [
    (0,0),
    (1,1),
    (2,2),
    (3,3),
    (4,4),
    (5,5)
])
def test_files(input_file_num, output_file_num):
    input_file_name = "dev_cases/input-{}.txt".format(input_file_num)
    output_file_name = "dev_cases/output-{}.txt".format(output_file_num)

    copyfile(input_file_name, "input.txt")

    run()

    res, expected = "", ""
    with open(output_file_name, 'r') as f:
        res = f.read().replace("\r", "")
            
    with open("output.txt", 'r') as f:
        expected = f.read()
    
    assert res == expected
