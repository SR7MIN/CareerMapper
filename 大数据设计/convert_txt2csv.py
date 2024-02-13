import csv


def convert(path, path2):
    with open(path2, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                temp = line.replace('\r', '').replace('\n', '').replace('   ', '').split(' ')
                if len(temp) == 4:
                    csv_writer.writerow(temp[1:])


convert("try.txt", "111.csv")
