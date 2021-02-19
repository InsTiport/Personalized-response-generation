import csv
import os

indexed_data_file = "utterance_index.csv"

# navigate to root directory in the project
os.chdir("../")

with open(os.path.join("data", indexed_data_file), 'r') as file_in:
    with open(os.path.join("data", "naive_small.txt"), 'w') as file_out:
        reader = csv.reader(file_in)
        in_question = True
        data_point = ''
        next(reader) # skip the header
        for row in reader:

            # utterance is of the form "[1, 2, 3, 4]"
            utterance = row[-1][1: len(row[-1]) - 1].split(',')
            for i in range(len(utterance)):
                utterance[i] = utterance[i].strip()

            if not in_question and row[3] == '[Q]':
                file_out.write(data_point + "\n")
                data_point = ''
                for index in utterance:
                    data_point += index + ' '
            elif row[3] == '[Q]':
                for index in utterance:
                    data_point += index + ' '
            else:
                in_question = False
                data_point += '| '
                for index in utterance:
                    data_point += index + ' '
