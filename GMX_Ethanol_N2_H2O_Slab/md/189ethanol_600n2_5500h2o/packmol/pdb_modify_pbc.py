import os

def insert_cryst1_line(input_filename):
    with open(input_filename, 'r') as f:
        lines = f.readlines()
    
    new_line = "CRYST1    40.00    40.00   159.33  90.00  90.00  90.00 P 1           1\n"
    
    lines.insert(5, new_line)
    
    with open(input_filename, 'w') as f:
        f.writelines(lines)

if __name__ == '__main__':
    path = './'
    for filename in os.listdir(path):
        if filename.endswith('.pdb'):
            input_file = os.path.join(path, filename)
            insert_cryst1_line(input_file)
            print(f"Inserted CRYST1 line in {input_file}")
