import json
import os
import uuid
import pandas as pd
import streamlit as st
import argparse
import traceback
from typing import Dict
import requests
from utils.utils import load_data_split
from utils.normalizer import post_process_sql
from nsql.database import NeuralDB
from nsql.nsql_exec import Executor
from nsql.nsql_exec_python import Executor
from generation.generator import Generator
import time

import sys
sys.path.append('.')

st.set_page_config(
    page_title="Binder Demo",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Check out our [website](https://lm-code-binder.github.io/) for more details!"
    }
)

ROOT_DIR = os.path.join(os.path.dirname(__file__), "./")
# todo: Add more binder questions, need careful cherry-picks
EXAMPLE_TABLES = {
    "Estonia men's national volleyball team": (558, "what is the number of players from france?"),
    # 'how old is kert toobal'
    "Highest mountain peaks of California": (5, "which is the lowest mountain?"),
    # 'which mountain is in the most north place?'
    "2010–11 UAB Blazers men's basketball team": (1, "how many players come from alabama?"),
    # 'how many players are born after 1996?'
    "Nissan SR20DET": (438, "which car has power more than 170 kw?"),
    # ''
}


@st.cache
def load_data():
    return load_data_split("missing_squall", "validation")


@st.cache
def get_key():
    # # print the public IP of the demo machine
    # ip = requests.get('https://checkip.amazonaws.com').text.strip()
    # print(ip)

    # URL = "http://54.242.37.195:8080/api/predict"
    # # The springboard machine we built to protect the key, 20217 is the birthday of Tianbao's girlfriend
    # # we will only let the demo machine have the access to the keys

    # one_key = requests.post(url=URL, json={"data": "Hi, binder server. Give me a key!"}).json()['data'][0]
    return "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"


def read_markdown(path):
    with open(path, "r") as f:
        output = f.read()
    st.markdown(output, unsafe_allow_html=True)


def generate_binder_program(_args, _generator, _data_item):
    n_shots = _args.n_shots
    few_shot_prompt = _generator.build_few_shot_prompt_from_file(
        file_path=_args.prompt_file,
        n_shots=n_shots
    )
    generate_prompt = _generator.build_generate_prompt(
        data_item=_data_item,
        generate_type=(_args.generate_type,)
    )
    prompt = few_shot_prompt + "\n\n" + generate_prompt

    # Ensure the input length fit Codex max input tokens by shrinking the n_shots
    max_prompt_tokens = _args.max_api_total_tokens - _args.max_generation_tokens
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(ROOT_DIR, "utils", "gpt2"))
    while len(tokenizer.tokenize(prompt)) >= max_prompt_tokens:
        n_shots -= 1
        assert n_shots >= 0
        few_shot_prompt = _generator.build_few_shot_prompt_from_file(
            file_path=_args.prompt_file,
            n_shots=n_shots
        )
        prompt = few_shot_prompt + "\n\n" + generate_prompt

    response_dict = _generator.generate_one_pass(
        prompts=[("0", prompt)],  # the "0" is the place taker, take effect only when there are multi threads
        verbose=_args.verbose
    )
    print(response_dict)
    return response_dict["0"][0][0]


def remove_row_id(table):
    new_table = {"header": [], "rows": []}
    header: list = table['header']
    rows = table['rows']

    if not 'row_id' in header:
        return table

    new_table['header'] = header[1:]
    new_table['rows'] = [row[1:] for row in rows]

    return new_table


# Set up
import nltk

nltk.download('punkt')
parser = argparse.ArgumentParser()

parser.add_argument('--prompt_file', type=str, default='templates/prompts/wikitq_binder.txt')
# Binder program generation options
parser.add_argument('--prompt_style', type=str, default='create_table_select_3_full_table',
                    choices=['create_table_select_3_full_table',
                             'create_table_select_full_table',
                             'create_table_select_3',
                             'create_table',
                             'create_table_select_3_full_table_w_all_passage_image',
                             'create_table_select_3_full_table_w_gold_passage_image',
                             'no_table'])
parser.add_argument('--generate_type', type=str, default='nsql',
                    choices=['nsql', 'sql', 'answer', 'npython', 'python'])
parser.add_argument('--n_shots', type=int, default=14)
parser.add_argument('--seed', type=int, default=42)

# Codex options
# todo: Allow adjusting Codex parameters
parser.add_argument('--engine', type=str, default="code-davinci-002")
parser.add_argument('--max_generation_tokens', type=int, default=512)
parser.add_argument('--max_api_total_tokens', type=int, default=8001)
parser.add_argument('--temperature', type=float, default=0.)
parser.add_argument('--sampling_n', type=int, default=1)
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument('--stop_tokens', type=str, default='\n\n',
                    help='Split stop tokens by ||')
parser.add_argument('--qa_retrieve_pool_file', type=str, default='templates/qa_retrieve_pool/qa_retrieve_pool.json')

# debug options
parser.add_argument('-v', '--verbose', action='store_false')
args = parser.parse_args()
keys = [get_key()]

# The title
st.markdown("# Binder Playground")

# Demo description
read_markdown('resources/demo_description.md')

# Upload tables/Switch tables

st.markdown('### Try Binder!')
col1, _ = st.columns(2)
with col1:
    selected_table_title = st.selectbox(
        "Select an example table (We use WikiTQ examples for this demo. But task inputs can include free-form texts and images as well)",
        (
            "Estonia men's national volleyball team",
            "Highest mountain peaks of California",
            "2010–11 UAB Blazers men's basketball team",
            "Nissan SR20DET",
        )
    )

# Here we just use ourselves'
data_items = load_data()
data_item = data_items[EXAMPLE_TABLES[selected_table_title][0]]
table = data_item['table']
header, rows, title = table['header'], table['rows'], table['page_title']
db = NeuralDB(
    [{"title": title, "table": table}])  # todo: try to cache this db instead of re-creating it again and again.
df = db.get_table_df()
st.markdown("Title: {}".format(title))
st.dataframe(df)

# Let user input the question
with col1:
    selected_language = st.selectbox(
        "Select a target Binder program",
        ("Binder-SQL", "Binder-Python"),
    )
if selected_language == 'Binder-SQL':
    args.prompt_file = 'templates/prompts/wikitq_binder.txt'
    args.generate_type = 'nsql'
elif selected_language == 'Binder-Python':
    args.prompt_file = 'templates/prompts/wikitq_binder.txt'
    args.generate_type = 'npython'
else:
    raise ValueError(f'{selected_language} language is not supported.')

question = st.text_input(
    "Ask a question about the table:",
    value=EXAMPLE_TABLES[selected_table_title][1],
)

button = st.button("Run Binder!")
if not button:
    st.stop()

# Print the question we just input.
st.subheader("Question")
st.markdown("{}".format(question))

# Generate Binder Program
generator = Generator(args, keys=keys)
with st.spinner("Generating Binder program to solve the question...will be finished in 10s, please refresh the page if not"):
    binder_program = generate_binder_program(args, generator,
                                             {"question": question, "table": db.get_table_df(), "title": title})

# Do execution
st.subheader("Binder program")
if selected_language == 'Binder-SQL':
    # Post process
    binder_program = post_process_sql(binder_program, df, selected_table_title, True)
    st.markdown('```sql\n' + binder_program + '\n```')
    executor = Executor(args, keys=keys)
elif selected_language == 'Binder-Python':
    st.code(binder_program, language='python')
    executor = Executor(args, keys=keys)
    db = db.get_table_df()
else:
    raise ValueError(f'{selected_language} language is not supported.')
try:
    stamp = '{}'.format(uuid.uuid4())
    os.makedirs('tmp_for_vis/', exist_ok=True)
    with st.spinner("Executing... will be finished in 30s, please refresh the page if not"):
        exec_answer = executor.nsql_exec(stamp, binder_program, db)
    if selected_language == 'Binder-SQL':
        with open("tmp_for_vis/{}_tmp_for_vis_steps.txt".format(stamp), "r") as f:
            steps = json.load(f)
        for i, step in enumerate(steps):
            col1, _, _ = st.columns([7, 1, 2])
            with col1:
                st.markdown(f'**Step #{i + 1}**')
            col1, col1_25, col1_5, col2, col3 = st.columns([4, 1, 2, 1, 2])
            with col1:
                st.markdown('```sql\n' + step + '\n```')
            with col1_25:
                st.markdown("executes\non")
            with col1_5:
                if i == len(steps) - 1:
                    st.markdown("Full table")
                else:
                    with open("tmp_for_vis/{}_result_step_{}_input.txt".format(stamp, i), "r") as f:
                        sub_tables_input = json.load(f)
                    for sub_table in sub_tables_input:
                        sub_table_to_print = remove_row_id(sub_table)
                        st.table(pd.DataFrame(sub_table_to_print['rows'], columns=sub_table_to_print['header']))
            with col2:
                st.markdown('$\\rightarrow$')
                if i == len(steps) - 1:
                    # The final step
                    st.markdown("{} Interpreter".format(selected_language.replace("Binder-", "")))
                else:
                    st.markdown("GPT3 Codex")
            with st.spinner('...'):
                time.sleep(1)
            with open("tmp_for_vis/{}_result_step_{}.txt".format(stamp, i), "r") as f:
                result_in_this_step = json.load(f)
            with col3:
                if isinstance(result_in_this_step, Dict):

                    rows = remove_row_id(result_in_this_step)["rows"]
                    header = remove_row_id(result_in_this_step)["header"]
                    if isinstance(header, list):
                        for idx in range(len(header)):
                            if header[idx].startswith('col_'):
                                header[idx] = step
                    st.table(pd.DataFrame(rows, columns=header))
                    # hard code here, use use_container_width after the huggingface update their streamlit version
                else:
                    st.markdown(result_in_this_step)
            with st.spinner('...'):
                time.sleep(1)
    elif selected_language == 'Binder-Python':
        pass
    if isinstance(exec_answer, list) and len(exec_answer) == 1:
        exec_answer = exec_answer[0]
    # st.subheader(f'Execution answer')
    st.text('')
    st.markdown(f"**Execution answer:** {exec_answer}")
    # todo: Remove tmp files
except Exception as e:
    traceback.print_exc()
