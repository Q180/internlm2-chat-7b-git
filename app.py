import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

# download internlm2 to the base_path directory using git tool
base_path = './course4'
print("工作目录：")
print(os.getcwd())

os.system(f'git clone https://code.openxlab.org.cn/Q180/course4.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

print(os.getcwd())
tokenizer = AutoTokenizer.from_pretrained('/home/xlab-app-center/course4', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="course4",
                description="""
InternLM is mainly developed by Shanghai AI Laboratory.  
                 """,
                 ).queue(1).launch()
