import os
import subprocess
from zipfile import ZipFile
from pathlib import Path
import tqdm
import pandas as pd
import numpy as np
import json, bs4, shutil
from bs4 import BeautifulSoup as bs
import tqdm
from pathlib import Path
from typing import List
from io import StringIO
from bs4 import NavigableString
import re


## raw doc file -> odt file 
src_directory = 'raw'
dest_directory = 'odt'

for root, dirs, files in os.walk(src_directory):
        for file in files:
            # 원본 파일 경로
            src_path = os.path.join(root, file)
            # 생성 파일 경로
            relative_path = os.path.relpath(src_path, src_directory)
            new_full_path = os.path.join(dest_directory, relative_path)
            new_path = os.path.dirname(new_full_path)
            
            # 경로가 없다면 새로 생성
            os.makedirs(new_path, exist_ok=True)
            
            # 이미 처리된 파일인지 확인
            base, _ = os.path.splitext(relative_path)
            new_relative_path = f"{base}.odt"
            dest_file_path = os.path.join(dest_directory, new_relative_path)
            if os.path.exists(dest_file_path):
                print(f"Skipping already processed file: {dest_file_path}")
                continue
            
            try:
                if src_path.endswith(".doc") or src_path.endswith(".docx") or src_path.endswith(".hwp") or src_path.endswith(".DOC") or src_path.endswith(".DOCX"):
                    subprocess.run(['libreoffice', '--headless', '--convert-to', 'odt', src_path, '--outdir', new_path])
                print("File conversion completed.")
            except Exception as e:
                print("An error occurred while converting the files: ", str(e))
                
                
                
# odt file -> unzip
src_directory = 'odt'
dest_directory = 'unzip'

def main(args):
    for root, dirs, files in os.walk(args.src_directory):
        if files:
            for f in tqdm.tqdm(files,desc='root:%s'%root):
                if f[-3:]=='odt':
                    file_in = os.path.join(root, f)
                    zip_file = ZipFile(file_in)
                    
                    src_path = file_in
                    relative_path = os.path.relpath(src_path, src_directory)
                    new_full_path = os.path.join(dest_directory, relative_path)
                    new_path = os.path.dirname(new_full_path)
                    output_point = Path(new_path)
                    # 경로가 없다면 새로 생성
                    os.makedirs(new_path, exist_ok=True)
                    # 이미 처리된 파일인지 확인
                    base, _ = os.path.splitext(relative_path)
                    new_relative_path = f"{base}"
                    dest_file_path = os.path.join(dest_directory, new_relative_path)
                    if os.path.exists(dest_file_path):
                        print(f"Skipping already processed file: {dest_file_path}")
                        continue
                    
                    #이부분의 output path를 각 폴더별로 분배하게 만들기
                    zip_file.extractall(path=output_point/f[:-4])

entry_point = Path(src_directory)
main(args=type('obj', (object,), {'src_directory': src_directory}))


# unzip folder / context.xml -> .json (dict 형태) 

def load(file_path):
    with open(file_path, 'r') as file_in:
        raw = file_in.readlines()
        file_in.close()
    raw = ''.join(raw)
    doc = bs(raw, 'xml')
    
    # Replace <text:tab/> with a space
    for tab in doc.find_all('text:tab'):
        tab.replace_with(NavigableString(' '))
    
    return doc

def process_text(doc):
    merged_tags = []
    for tag in doc.find_all(['text:p', 'text:h']):
        merged_tags.append(tag)
    
    results = {i:x.getText() for i, x in enumerate(merged_tags)}
    return results


unzip_path = 'unzip'
entry_point = Path(unzip_path)

dest_directory = 'parsed'

err_cases = []
for r, dirs, files in tqdm.tqdm(os.walk(entry_point, topdown=True)):
    dirs[:] = [d for d in dirs if "Object" not in d]
    if files:
        for f in files:
            if f == 'content.xml':
                try:
                    file_path = Path(r) / f
                    
                    src_path = file_path
                    relative_path = os.path.relpath(src_path, unzip_path)
                    new_full_path = os.path.join(dest_directory, relative_path)
                    new_path = os.path.dirname(new_full_path)
                    output_point = Path(os.path.dirname(new_path))
                    
                    output_point.mkdir(exist_ok=True, parents=True)
                    # 이미 처리된 파일인지 확인
                    base, _ = os.path.splitext(os.path.dirname(relative_path))
                    new_relative_path = f"{base}.json"
                    dest_file_path = os.path.join(dest_directory, new_relative_path)
                    if os.path.exists(dest_file_path):
                            print(f"Skipping already processed file: {dest_file_path}")
                            continue
                    
                    dir_out = output_point / file_path.parent.name
                    file_out = output_point / (dir_out.name + '.json')

                    doc = load(file_path)
                    text = process_text(doc)
                    #figs = process_figures(doc, file_path, dir_out, copy=True)
                    #tabs = process_tables(doc, dir_out)

                    result = {'text':text}
                    with open(str(file_out), 'w') as out_:
                        json.dump(result, out_, ensure_ascii=False)
                except Exception as e:
                    print(e)
                    err_cases.append(str(file_path)+'\n')
with open(output_point/'error_cases.log', 'w') as errout_:
    errout_.writelines(err_cases)
    

# .json 최종 list형태로 변환하는 코드
def divide_sections(values, sections):
    divided_sections = []
    end = len(values)
    for start, idx in sections:
        divided_sections.append((idx, values[start:end]))
        end = start
    divided_sections.append((None, values[:end]))  # 첫번째 섹션을 추가합니다.
    return list(reversed(divided_sections))

def any_to_list(data):
    if isinstance(data, tuple) or isinstance(data, np.ndarray):
        return [any_to_list(item) for item in data]
    elif isinstance(data, list):
        return [any_to_list(item) for item in data]
    else:
        return data

parsed_path = 'parsed'
entry_point = Path(parsed_path)
dest_directory = 'final'

for r, d, files in tqdm.tqdm(os.walk(entry_point)):
        if files:
            for f in files:
                json_file_path = os.path.join(r,f)
                
                 # 생성 파일 경로
                relative_path = os.path.relpath(json_file_path, parsed_path)
                new_full_path = os.path.join(dest_directory, relative_path)
                new_path = os.path.dirname(new_full_path)
                
                # 경로가 없다면 새로 생성
                os.makedirs(new_path, exist_ok=True)
                
                # 이미 처리된 파일인지 확인
                base, _ = os.path.splitext(relative_path)
                new_relative_path = f"{base}.json"
                dest_file_path = os.path.join(dest_directory, new_relative_path)
                if os.path.exists(dest_file_path):
                    print(f"Skipping already processed file: {dest_file_path}")
                    continue
                
                try:
                    context = pd.read_json(json_file_path)  # Load the JSON file
                    try:
                        values = context.sort_index()["text"].values  # Get the 'text' column values
                    except KeyError:
                        print(f"'text' key does not exist in JSON file: {json_file_path}")
                        continue
                except ValueError:
                    print(f"Could not parse JSON file: {json_file_path}")
                    continue
                except FileNotFoundError:
                    print(f"JSON file does not exist: {json_file_path}")
                    continue
                except Exception as e:
                    print(f"Unexpected error occurred when processing JSON file {json_file_path}. Error: {str(e)}")
                
                sections = [(0,1)]
                divided_sections_res = divide_sections(values, sections)
                divided_sections_res = any_to_list(divided_sections_res)
                
                json_file_path = json_file_path.replace('.json', '_sectioned.json')  # 섹션별 JSON 파일 경로를 만듭니다.
                with open(json_file_path, 'w', encoding='utf-8') as json_file:
                    json.dump(divided_sections_res, json_file, ensure_ascii=False)
                
                # 원본 파일을 새로운 경로로 복사합니다.
                shutil.copy2(json_file_path, new_path)