import csv

def get_all_enterprise():
    all_e = {}
    with open('current_jobs.txt', 'r', encoding='ANSI') as file:
        lines = file.readlines()
        for line in lines:
            temp = line.replace('\n', '').replace('\r', '').split('|')
            if not temp[1] in all_e:
                all_e[temp[1]] = 0
    with open('all_enterprise.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for e in all_e:
            writer.writerow([e])

# get_all_enterprise()

