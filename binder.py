import os
import uuid
from utils.utils import load_data_split
from utils.normalizer import post_process_sql
from nsql.database import NeuralDB
from nsql.nsql_exec import Executor
from generation.generator import Generator

import nltk
from transformers import AutoTokenizer

nltk.download('punkt')

def read_markdown(path):
    with open(path, "r") as f:
        output = f.read()


def generate_binder_program(_args, _generator: Generator, _data_item, root_dir='./'):
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
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(root_dir, "utils", "gpt2"))
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

def translate2SQL(args, question, db: NeuralDB, keys, selected_table_title, root_dir='./'):
    df = db.get_table_df()

    # Let user input the question
    args.prompt_file = os.path.join(root_dir, 'templates/prompts/wikitq_binder.txt')
    args.generate_type = 'nsql'

    # Generate Binder Program
    generator = Generator(args, keys=keys)
    binder_program = generate_binder_program(args, generator,
                                            {"question": question, 
                                             "table": db.get_table_df(), 
                                             "title": db.raw_tables[0]['title']},
                                            root_dir=root_dir)
    # Post process
    binder_program = post_process_sql(binder_program, df, selected_table_title, False)
    return args, binder_program

def executeSQL(args, binder_program, db, keys):
    executor = Executor(args, keys=keys)
    stamp = '{}'.format(uuid.uuid4())
    os.makedirs('tmp_for_vis/', exist_ok=True)
    exec_answer = executor.nsql_exec(binder_program, db, verbose=False, stamp=stamp)
    
    if isinstance(exec_answer, list) and len(exec_answer) == 1:
        exec_answer = exec_answer[0]
    return exec_answer