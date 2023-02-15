import re
import json
from sklearn.model_selection import train_test_split

with open('recipes.json') as f:
    data = json.load(f)

def build_text_files(data_json, dest_path):
    f = open(dest_path, 'w')
    data = ''
    for texts in data_json:
        summary = str(texts['Instructions']).strip()
        summary = re.sub(r"\s", " ", summary)
        data += summary + "  "
    f.write(data)

train, test = train_test_split(data,test_size=0.2)

build_text_files(train,'train_dataset.txt')
build_text_files(test,'test_dataset.txt')

print("Train dataset length: "+ str(len(train)))
print("Test dataset length: "+ str(len(test)))

#Train dataset length: 10361
#Test dataset length: 1829
