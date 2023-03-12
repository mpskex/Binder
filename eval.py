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
from sqlalchemy import exc

import sys
sys.path.append('.')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

ROOT_DIR = os.getcwd()
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
key = [
    "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", 
    "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"
    ]


import nltk

nltk.download('punkt')
# debug options

class Args:
    prompt_file = 'templates/prompts/wikitq_binder.txt'
    prompt_style = 'create_table_select_3_full_table'
    generate_type = 'nsql'
    n_shots = 14
    seed = 12
    engine = "code-davinci-002"
    max_generation_tokens = 512
    max_api_total_tokens = 8001
    temperature = 0.
    sampling_n = 1
    top_p = 1.0
    stop_tokens = '\n\n'
    qa_retrieve_pool_file = 'templates/qa_retrieve_pool/qa_retrieve_pool.json'
    verbose = False

args = Args()
keys = key

selected_table_title = list(EXAMPLE_TABLES.keys())[0]
selected_language = ("Binder-SQL", "Binder-Python")[0]

dataset_str = ['wikitq', 'tab_fact'][0]
dataset = load_data_split(dataset_str, 'test')
# For TabFact test split, we load the small test set (about 2k examples) to test,
# since it is expensive to test on full set
if dataset_str == "tab_fact":
    with open(os.path.join(ROOT_DIR, "utils", "tab_fact", "small_test_id.json"), "r") as f:
        small_test_ids_for_iter = json.load(f)
    dataset = [data_item for data_item in dataset if data_item['table']['id'] in small_test_ids_for_iter]

import copy
from tqdm import tqdm
from binder import translate2SQL, executeSQL

g_dict = dict()
result_dict = dict()
print(len(dataset))
for g_eid in tqdm(range(len(dataset)//10)):
    success = False
    count = 3
    while not success or count > 0:
        try:
            g_data_item = dataset[g_eid]
            question = g_data_item['question']
            result_dict[g_eid] = {}
            result_dict[g_eid]['question'] = g_data_item['question']
            result_dict[g_eid]['gold_answer'] = g_data_item['answer_text']
            g_dict[g_eid] = {
                'generations': [],
                'ori_data_item': copy.deepcopy(g_data_item)
            }
            db = NeuralDB(
                tables=[{'title': g_data_item['table']['page_title'], 'table': g_data_item['table']}]
            )

            # Here we just use ourselves'
            # data_items = load_data_split("missing_squall", "validation")
            # data_item = data_items[EXAMPLE_TABLES[selected_table_title][0]]
            # table = data_item['table']
            # header, rows, title = table['header'], table['rows'], table['page_title']
            # print(title)
            # print(table)
            
            # question = EXAMPLE_TABLES[selected_table_title][1]
            # question = "What are the names of the player from bigbank tartu who are taller than 175?"

            args, sql_str = translate2SQL(args, question, db, keys, selected_table_title, ROOT_DIR)
            result_dict[g_eid]['nsql'] = sql_str
            pred_answer = executeSQL(args, sql_str, db, keys)
            result_dict[g_eid]['pred_answer'] = pred_answer
            result_dict[g_eid]['excutable'] = 'success'
            success = True
        except Exception as e:
            if type(e) is exc.OperationalError:
                result_dict[g_eid]['excutable'] = 'invalid sql'
                success = True
            else:
                print(f'Error! {str(e)}')
                result_dict[g_eid]['excutable'] = 'timeout'
                time.sleep(10)
                count -= 1
    
import json
json.dump(result_dict, open(f'{dataset_str}_simple_binder.json', 'w'))