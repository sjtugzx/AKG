import os
import openai
import requests

import datetime
import time
api_key = "Your-OpenAI-API-Key"

def get_chatgpt_response(messages):

    proxies = {
        'http': 'http://10.10.1.3:10000',
        'https': 'http://10.10.1.3:10000'
    }

    resp = requests.post(url='https://openai.acemap.cn/v1/chat/completions',
                         headers={
                             "Content-Type": "application/json",
                             "Authorization": f"Bearer {api_key}"
                         },
                         json={
                             "model": "gpt-3.5-turbo",
                             "messages": messages,
                         },
                         proxies=proxies,
    )
    # 解析响应
    if resp.status_code == 200:
        data = resp.json()
        text = data["choices"][0]["message"]
        return text
    else:
        print(resp.json())
        return "Sorry, something went wrong."

def read_resource_file(file_path):
    print("read source file!")
    resource_list = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            new_dict={}
            new_dict["role"]="assistant"
            new_dict["content"]="Summarize the following table content: "+line.strip()
            resource_list.append(new_dict)
    return resource_list

def write_output_file(file_path, predicted_list):
    with open(file_path, 'w') as f:
        f.writelines("%s\n" % item for item in predicted_list)


def attempt_request(message):
    try:
        response = get_chatgpt_response([message])['content'].replace("\n", " ")
        return response, True
    except Exception as e:
        print(f"An error occurred: {e}. Retrying in 5 seconds...")
        time.sleep(5)  # 等待5秒后重试
        return None, False


if __name__ == "__main__":
    file_list = ["../data/test-gpt3.5/book/test_table.txt",
                 "../data/test-gpt3.5/human/test_table.txt",
                 "../data/test-gpt3.5/song/test_table.txt",]

    for file in file_list:
        resource_list = read_resource_file(file)
        response_list = []
        idx = 0

        output_file = file.replace("test_table.txt", "test-gpt.out")

        with open(output_file, "a") as f:
            for message in resource_list:
                success = False
                while not success:
                    response, success = attempt_request(message)
                    if success:
                        response_list.append(response)
                        idx += 1
                        f.write(response + "\n")
                        print(f"finish writing the {idx}th record into the file.")
                        time.sleep(1)  # Sleep for 1 second to comply with rate limits

        print(f"finish writing the {idx}th record into the file.")

