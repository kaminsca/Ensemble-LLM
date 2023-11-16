import csv
import json

delim = "%@%"

casehold_filenames = [
    "test.csv",
    "val.csv",
    "train.csv"
]

arc_filenames = [
    "ARC-Challenge-Test.csv",
    "ARC-Challenge-Train.csv",
    "ARC-Easy-Test.csv",
    "ARC-Easy-Train.csv",
    "ARC-Challenge-Dev.csv",
    "ARC-Easy-Dev.csv"
]

codeqa_filenames = [   
    "dev",
    "test",
    "train"
    ]

aquarat_filenames = [
    "train.json",
    "dev.json",
    "test.json"
]

coqa_filenames = [
    "coqa-dev.json",
    "coqa-train.json"
]

def process(raw, processed, filenames):
    for file in filenames:

        with open(f'{raw}{file}', 'r') as f:
            #with open(f'{processed}{file}', 'w') as f2:

                # reader = csv.reader(f)
                # next(reader)
                # f2.write("question, label\n")
                # for row in reader:
                #     #casehold
                #     #f2.write(f'{row[1]}{delim}LAW\n')

                #     # arc
                #     x = row[9].split("(A)")[0] if "(A)" in row[9] else row[9].split("(1)")[0]
                #     f2.write(f'{x}{delim}SCIENCE\n')

            # # aquarat
            # with open(f'{processed}{file.replace(".json", ".csv")}', 'w') as f2:
            #     data = f.read()
            #     part = data.split("\n")
            #     for line in part:
            #         if line:
            #             line = json.loads(line)
            #             x = line["question"].replace("\n", "\t")
            #             f2.write(f'{x}{delim}MATH\n')
            
            # coqa
            with open(f'{processed}{file.replace(".json", ".csv")}', 'w') as f2:
                data = json.loads(f.read())
                data = data["data"]
                for d in data:
                    for q in d["questions"]:
                        f2.write(f'{q["input_text"]}{delim}CONV\n')



        # # process code QA
        # with open (f'{raw}{file}.code') as c:
        #     with open (f'{raw}{file}.question') as q:
        #         with open (f'{processed}{file}.csv', 'a') as f2:
        #             for code, question in zip(c, q):
        #                 f2.write(f'{question.strip()}\t{code.strip()}{delim}CODE\n')
    





#process("data/casehold_raw/", "data/casehold_processed/", casehold_filenames)
#process("data/arc_raw/", "data/arc_processed/", arc_filenames)
#process("data/codeQA_raw/", "data/codeQA_processed/", codeqa_filenames)
#process("data/aquarat_raw/", "data/aquarat_processed/", aquarat_filenames)
process("data/coqa_raw/", "data/coqa_processed/", coqa_filenames)
