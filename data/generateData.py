delim = "%@%"
num_train = 3000 # stratify test set to get fair training data
# we don't need to stratify dev and test

dev_files = [
    "data/aquarat_processed/dev.csv",
    "data/arc_processed/ARC-Dev.csv",
    "data/casehold_processed/val.csv",
    "data/codeQA_processed/dev.csv",
    "data/coqa_processed/coqa-dev.csv"
]

train_files = [
    "data/aquarat_processed/train.csv",
    "data/arc_processed/ARC-Train.csv",
    "data/casehold_processed/train.csv",
    "data/codeQA_processed/train.csv",
    "data/coqa_processed/coqa-train.csv"
]

test_files = [
    "data/aquarat_processed/test.csv",
    "data/arc_processed/ARC-Test.csv",
    "data/casehold_processed/test.csv",
    "data/codeQA_processed/test.csv",
    "data/coqa_processed/coqa-test.csv"
]


# get the data and process it into a list of [question, label]

# train
train_data = [open(f, "r").read().strip() for f in train_files]
train_data = [d.split("\n")[:num_train] for d in train_data]
train_data = [[d.split(delim) for d in dataset] for dataset in train_data]
train_data = [item for sublist in train_data for item in sublist]

# val
val_data = [open(f, "r").read().strip() for f in dev_files]
val_data = [d.split("\n") for d in val_data]
val_data = [[d.split(delim) for d in dataset] for dataset in val_data]
val_data = [item for sublist in val_data for item in sublist]

# test
test_data = [open(f, "r").read().strip() for f in test_files]
test_data = [d.split("\n") for d in test_data]
test_data = [[d.split(delim) for d in dataset] for dataset in test_data]
test_data = [item for sublist in test_data for item in sublist]

# shuffle
import random
random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)

# write to file
dir = "data/final_data/"
for data, name in [(train_data, "train"), (val_data, "val"), (test_data, "test")]:
    with open(f'{dir}{name}.csv', 'w') as f:
        f.write("question, label\n")
        for row in data:
            f.write(f'{row[0]}{delim}{row[1]}\n')
