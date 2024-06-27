import os 
import csv
import re
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

token = 'hugging_face - token' # 개인 허깅페이스 키를 넣으세요 / Llama3 허가를 받은 계정이어야함

# 모델 불러오기
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model =  AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto"
)

# 모델 GPU에 넣기
if torch.cuda.is_available():
    model = model.to("cuda")

# 챗형식 모델로 설정
def generate_response(system_message, user_message):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.1
    )

    response = outputs[0][input_ids.shape[-1]:]

    return tokenizer.decode(response, skip_special_tokens=True)



# LLM 사용코드

# 일본어 추출 모델
def make_japen_prompt(text_data):
    text_data = re.sub(r'·|…', '', text_data)
    while len(text_data) > 7000:
        one_third_length = len(text_data) // 3
        text_data = text_data[:one_third_length*2]
        
    main_질문_prompt = 'コンテキスト：' + text_data + ',  質問 : 次の空欄を最大限埋めてくれ,  試験表題: , 試験番号: , 試験責任者: , 使用した動物種 :, 投与経路 : , 試験開始日:, 試験終了日:, 実験開始日:, 実験終了日: , 動物搬入日 : , 被験物質: , 病理責任者: , 遵守 GLP :'
    main_프롬프트_message =  '真面目な日本人エージェントとして機能します。提供されたコンテキストに基づいて、直接的な回答のみを提供してください。質問に対する答えは、存在する情報を最大限に活用してください。質問文は日本語のままで回答し、英語には翻訳しないでください。提供された情報が不足している場合は「なし」と回答してください。'
    llama3_inference_result = generate_response(system_message=main_프롬프트_message,
                                            user_message=main_질문_prompt)
    
    # print(llama3_inference_result)
    
    lines = llama3_inference_result.split('\n')
    parsed_list = []
    for line in lines:
        if '：' in line:  # 콜론이 있는 줄만 처리
            parts = line.split('：')
            if len(parts)>2:
                value = ''.join(parts[1:])  # 콜론 기준으로 한 번만 분리
            else:
                value = parts[1] # 콜론 기준으로 한 번만 분리
            parsed_list.append( value.strip())
        elif ':' in line:
            parts = line.split(':')
            if len(parts)>2:
                value = ''.join(parts[1:])  # 콜론 기준으로 한 번만 분리
            else:
                value = parts[1] # 콜론 기준으로 한 번만 분리
            parsed_list.append( value.strip())
            
    parsed_list = [item for item in parsed_list if item.strip()]
    # 시험 제목 
    study_title = parsed_list[0]
    # 시험 번호
    study_number = parsed_list[1]
    # 시험 책임자
    study_direct = parsed_list[2]
    # 시험 동물
    study_animal = parsed_list[3]
    # 시험 투여경로
    study_inject = parsed_list[4]
    # 시험 개시일
    study_start = parsed_list[5]
    # 시험 종료일
    study_end = parsed_list[6]
    # 실험 개시일
    exper_start = parsed_list[7]
    # 실험 종료일
    exper_end = parsed_list[8]
    # 동물입수일
    animal_day = parsed_list[9]
    # 시험물질
    study_material = parsed_list[10]
    # 병리 책임자
    study_battle = parsed_list[11]
    # GLP
    study_GLP = parsed_list[12]

    result = ['일본어', study_number, study_GLP, study_title, study_start, exper_start,
              exper_end, study_end, animal_day, study_direct,study_battle,  study_animal, 
              study_inject, study_material, '요약']
    return result


# LLM 사용코드 - 영어 추출모델 (프롬프트만 다름 - 코드 최적화가 가능할것같은데?)
def make_english_prompt(text_data):
    text_data = re.sub(r'·|…', '', text_data)
    while len(text_data) > 7000:
        one_third_length = len(text_data) // 3
        text_data = text_data[:one_third_length*2]
        
    main_question_prompt = 'Context: ' + text_data + ', Question: Please fill in the blanks as much as possible, Study_title: , Study_No.: , Study_Director: , Animal_Species_used: , Administration_route: , Initiation of study date: , Completion of study date: , Initiation of experiments date: , Completion of experiments date: , Animal receipt date: , Test materials: , Pathology director: , GLP Regulations Performed: '
    main_prompt_message = 'Function as a serious US agent. Provide direct and succinct answers based solely on the provided context. Utilize all available information to comprehensively answer the questions. If the provided information is insufficient to form a response, simply state "none". Just answer me'
    llama3_inference_result = generate_response(system_message=main_prompt_message,
                                            user_message=main_question_prompt)
    
    
    lines = llama3_inference_result.split('\n')
    
    parsed_list = []
    for line in lines:
        if '：' in line:  # 콜론이 있는 줄만 처리
            parts = line.split('：')
            if len(parts)>2:
                value = ''.join(parts[1:])  # 콜론 기준으로 한 번만 분리
            else:
                value = parts[1] # 콜론 기준으로 한 번만 분리
            parsed_list.append( value.strip())
        elif ':' in line:
            parts = line.split(':')
            if len(parts)>2:
                value = ''.join(parts[1:])  # 콜론 기준으로 한 번만 분리
            else:
                value = parts[1] # 콜론 기준으로 한 번만 분리
            parsed_list.append( value.strip())
    
    parsed_list = [item for item in parsed_list if item.strip()]

    # 시험 제목 
    study_title = parsed_list[0]
    # 시험 번호
    study_number = parsed_list[1]
    # 시험 책임자
    study_direct = parsed_list[2]
    # 시험 동물
    study_animal = parsed_list[3]
    # 시험 투여경로
    study_inject = parsed_list[4]
    # 시험 개시일
    study_start = parsed_list[5]
    # 시험 종료일
    study_end = parsed_list[6]
    # 실험 개시일
    exper_start = parsed_list[7]
    # 실험 종료일
    exper_end = parsed_list[8]
    # 동물입수일
    animal_day = parsed_list[9]
    # 시험물질
    study_material = parsed_list[10]
    # 병리 책임자
    study_battle = parsed_list[11]
    # GLP
    study_GLP = parsed_list[12]

    result = ['영어', study_number, study_GLP, study_title, study_start, exper_start,
              exper_end, study_end, animal_day, study_direct, study_battle,  study_animal, 
              study_inject, study_material, '요약']
    return result


# LLM 사용코드  -한국어 프롬프트
def make_korea_prompt(text_data):
    text_data = re.sub(r'·|…', '', text_data)
    while len(text_data) > 7000:
        one_third_length = len(text_data) // 3
        text_data = text_data[:one_third_length*2]
        
    main_질문_prompt = '컨텍스트: ' + text_data + ', 질문: 다음 빈칸만 채워주세요, 시험제목: , 시험번호: , 시험책임자: , 사용한 동물 종류: , 투여경로: , 시험 개시일: , 시험 종료일: , 실험 개시일: , 실험 종료일: , 동물 입수일: , 시험물질: , 병리 책임자: , 준수한 GLP규정: '
    main_프롬프트_message = '제공된 컨텍스트에 기반하여 답변만을 제공해 주세요. 질문에 대한 답변은 컨텍스트 정보를 최대한 활용해 주세요. 질문은 한국어로 답하고 영어로 번역하지 마세요. 제공된 정보가 부족할 경우 "none" 으로 답해 주세요., 답변만을 해주세요.'
    llama3_inference_result = generate_response(system_message=main_프롬프트_message,
                                            user_message=main_질문_prompt)
    
    lines = llama3_inference_result.split('\n')
    parsed_list = []
    for line in lines:
        # : 콜론 확인 - 일본어라서 이상할수도있음
        if '：' in line:  # 콜론이 있는 줄만 처리
            parts = line.split('：')
            if len(parts)>2:
                value = ''.join(parts[1:])  # 콜론 기준으로 한 번만 분리
            else:
                value = parts[1] # 콜론 기준으로 한 번만 분리
            parsed_list.append( value.strip())
        elif ':' in line:
            parts = line.split(':')
            if len(parts)>2:
                value = ''.join(parts[1:])  # 콜론 기준으로 한 번만 분리
            else:
                value = parts[1] # 콜론 기준으로 한 번만 분리
            parsed_list.append( value.strip())
    # 시험 제목 
    study_title = parsed_list[0]
    # 시험 번호
    study_number = parsed_list[1]
    # 시험 책임자
    study_direct = parsed_list[2]
    # 시험 동물
    study_animal = parsed_list[3]
    # 시험 투여경로
    study_inject = parsed_list[4]
    # 시험 개시일
    study_start = parsed_list[5]
    # 시험 종료일
    study_end = parsed_list[6]
    # 실험 개시일
    exper_start = parsed_list[7]
    # 실험 종료일
    exper_end = parsed_list[8]
    # 동물입수일
    animal_day = parsed_list[9]
    # 시험물질
    study_material = parsed_list[10]
    # 병리 책임자
    study_battle = parsed_list[11]
    # GLP
    study_GLP = parsed_list[12]
    
    result = ['한국어', study_number, study_GLP, study_title, study_start, exper_start,
              exper_end, study_end, animal_day, study_direct,study_battle,  study_animal, 
              study_inject, study_material, '요약']
    return result

def check_keywords(text):
    jap_keywords = ['試験番号', '試験責任者']
    kor_keywords = [ '시험번호', '시험책임자']
    eng_keywords = ['study director']

    # 일본어 키워드 검사
    if all(keyword in text for keyword in jap_keywords):
        result_jap = make_japen_prompt(text)
        return result_jap
    
    # 한국어 키워드 검사
    if all(keyword in text for keyword in kor_keywords):
        result_kor = make_korea_prompt(text)
        return result_kor
    
    
    # 영어 키워드 검사
    lower_text = text.lower()
    if all(keyword in lower_text for keyword in eng_keywords):
        result_eng = make_english_prompt(text)
        return result_eng
    
    # 어떤 언어의 키워드도 충족하지 않는 경우
    print('해당하는 언어의 키워드가 없습니다.')
    
    result_error = ['error', '시험번호', 'GLP', '시험제목', 
        '시험개시일', '실험개시일', '실험종료일','시험종료일', '동물입수일','시험책임자',
        '병리책임자','동물종','투여경로','시험물질','요약']
    return result_error

# main function code
def extract_model(json_data):
        raw_j_data = json_data[1][1]
        text_data  = ''.join(raw_j_data)

        result = check_keywords(text_data)
        return result
    
## log file def 
def get_last_processed_file(log_file_path):
    try:
        with open(log_file_path, 'r') as log_file:
            last_processed = log_file.read().strip()
        return last_processed
    except FileNotFoundError:
        return None

def set_last_processed_file(log_file_path, file_name):
    with open(log_file_path, 'w') as log_file:
        log_file.write(file_name)
        
# main code

raw_file_list_old = []
json_file_list = []
json_path_list = []

for r, d, files in os.walk('raw'):
    for f in files:       
        file_path = os.path.join(r,f)
        raw_file_list_old.append(file_path)    
raw_file_list = sorted(raw_file_list_old, key=lambda x: (x.split('/')[1], x.split('/')[-1]))

for r, d, files in os.walk('final'):
    for f in files:       
        file_path = os.path.join(r,f)
        json_file_list.append(f) 
        json_path_list.append(file_path) 


csv_path = 'japen_main_result.csv'

col_name = ['file_link', 'file_name',  '예외', '시험그룹', '언어 ','시험번호', 'GLP' ,'시험제목', 
            '시험개시일', '실험개시일', '실험종료일','시험종료일', '동물입수일','시험책임자',
            '병리책임자','동물종','투여경로','시험물질','요약']


log_file_path = 'log/process_log.txt'
last_processed = get_last_processed_file(log_file_path)
# 파일 처리 시작 지점 찾기
start_index = raw_file_list.index(last_processed) + 1 if last_processed in raw_file_list else 0

with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    if start_index == 0:
        writer.writerow(col_name) 
    
    for r_f in raw_file_list[start_index:]:
        file_link = r_f
        file_name = r_f.split('/')[-1]
        
        defalut_row = [file_link, file_name, '', '']
        
        file_name0 = re.sub(r'(.doc|.DOC|.docx|.DOCX|.pdf|.hwp)', '_sectioned.json', file_name)
        file_name1 = re.sub('.jsonx', '.json', file_name0)
        
        print(file_name + ' : ongoing')
        
        if file_name1 in json_file_list:
            index = json_file_list.index(file_name1)
            file_path = json_path_list[index]
            print('file_path : ' + file_path)
            if file_name1.endswith('.json'):  # JSON 파일인지 확인합니다.
                with open(file_path, 'r') as json_file:
                    json_data = json.load(json_file)
                    try:
                        extract_list = extract_model(json_data)
                        defalut_row = defalut_row + extract_list
                    except Exception as e:
                        print('[Error]')
                        writer.writerow(defalut_row)
                        continue 
                    
            
            writer.writerow(defalut_row)
            set_last_processed_file(log_file_path, file_link)
        else :
            defalut_row = [file_link, file_name, '암호', '']
            writer.writerow(defalut_row)
            set_last_processed_file(log_file_path, file_link)