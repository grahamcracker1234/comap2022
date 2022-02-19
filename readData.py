import csv

# get the data working in a list
def openFile(filePath):
    file = open(filePath, "r")
    file = csv.reader(file)
    rows = []
    for row in file:
        rows.append(row)

    rows = rows[1:]
    for i in range(len(rows)):
        rows[i] = [rows[i][0], float(rows[i][1])]
    
    return rows
