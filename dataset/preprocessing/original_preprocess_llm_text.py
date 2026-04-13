import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from utils import check_path, set_device, amazon_dataset2fullname


class TextProcessor:
    """Text processor using Qwen3 embedding model for generating item embeddings"""
    def __init__(self, model_name='Qwen/Qwen3-Embedding-0.6B', device='cuda'):
        self.device = device
        # Load tokenizer and base model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.base_model = AutoModel.from_pretrained(model_name).to(device)
        
        # Freeze base model parameters to save memory
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Store dimensions
        self.base_embedding_dim = self.base_model.config.hidden_size
        
    def last_token_pool(self, last_hidden_states, attention_mask):
        """Pool the last token representation"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def process_texts(self, text_list, batch_size=4):
        """
        Process a list of texts and return their embeddings
        Args:
            text_list: List of text strings
            batch_size: Batch size for processing
        Returns:
            embeddings: numpy array of shape [num_texts, embedding_dim]
        """
        embeddings = []
        
        for start in tqdm(range(0, len(text_list), batch_size), desc='Processing texts'):
            batch_texts = text_list[start:start + batch_size]
            
            # Tokenize text inputs
            tokenized_texts = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=8192, 
                return_tensors='pt'
            ).to(self.device)
            
            # Get embeddings from the model
            with torch.no_grad():
                outputs = self.base_model(**tokenized_texts)
                batch_embeddings = self.last_token_pool(outputs.last_hidden_state, tokenized_texts['attention_mask'])
                
                # Normalize embeddings
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                embeddings.append(batch_embeddings.cpu())
        
        # Concatenate all embeddings
        embeddings = torch.cat(embeddings, dim=0).numpy()
        return embeddings


def load_ratings(file):
    users, items, inters = set(), set(), set()
    with open(file, 'r') as fp:
        for line in tqdm(fp, desc='Load ratings'):
            try:
                item, user, rating, time = line.strip().split(',')
                users.add(user)
                items.add(item)
                inters.add((user, item, float(rating), int(time)))
            except ValueError:
                print(line)
    return users, items, inters


def load_meta_items(file):
    items = set()
    with gzip.open(file, 'r') as fp:
        for line in tqdm(fp, desc='Load metas'):
            data = json.loads(line)
            items.add(data['asin'])
    return items


def get_user2count(inters):
    user2count = collections.defaultdict(int)
    for unit in inters:
        user2count[unit[0]] += 1
    return user2count


def get_item2count(inters):
    item2count = collections.defaultdict(int)
    for unit in inters:
        item2count[unit[1]] += 1
    return item2count


def generate_candidates(unit2count, threshold):
    cans = set()
    for unit, count in unit2count.items():
        if count >= threshold:
            cans.add(unit)
    return cans, len(unit2count) - len(cans)


def filter_inters(inters, can_items=None,
                  user_k_core_threshold=0, item_k_core_threshold=0):
    new_inters = []

    # filter by meta items
    if can_items:
        print('\nFiltering by meta items: ')
        for unit in inters:
            if unit[1] in can_items:
                new_inters.append(unit)
        inters, new_inters = new_inters, []
        print('    The number of inters: ', len(inters))

    # filter by k-core
    if user_k_core_threshold or item_k_core_threshold:
        print('\nFiltering by k-core:')
        idx = 0
        user2count = get_user2count(inters)
        item2count = get_item2count(inters)

        while True:
            new_user2count = collections.defaultdict(int)
            new_item2count = collections.defaultdict(int)
            users, n_filtered_users = generate_candidates(
                user2count, user_k_core_threshold)
            items, n_filtered_items = generate_candidates(
                item2count, item_k_core_threshold)
            if n_filtered_users == 0 and n_filtered_items == 0:
                break
            for unit in inters:
                if unit[0] in users and unit[1] in items:
                    new_inters.append(unit)
                    new_user2count[unit[0]] += 1
                    new_item2count[unit[1]] += 1
            idx += 1
            inters, new_inters = new_inters, []
            user2count, item2count = new_user2count, new_item2count
            print('    Epoch %d The number of inters: %d, users: %d, items: %d'
                    % (idx, len(inters), len(user2count), len(item2count)))
    return inters


def make_inters_in_order(inters):
    user2inters, new_inters = collections.defaultdict(list), list()
    for inter in inters:
        user, item, rating, timestamp = inter
        user2inters[user].append((user, item, rating, timestamp))
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        for inter in user_inters:
            new_inters.append(inter)
    return new_inters


def preprocess_rating(args):
    dataset_full_name = amazon_dataset2fullname[args.dataset]

    print('Process rating data: ')
    print(' Dataset: ', dataset_full_name)

    # load ratings
    rating_file_path = os.path.join(args.input_path, 'Ratings', dataset_full_name + '.csv')
    rating_users, rating_items, rating_inters = load_ratings(rating_file_path)

    # load item IDs with meta data
    meta_file_path = os.path.join(args.input_path, 'Metadata', f'meta_{dataset_full_name}.json.gz')
    meta_items = load_meta_items(meta_file_path)

    # 1. Filter items w/o meta data;
    # 2. K-core filtering;
    print('The number of raw inters: ', len(rating_inters))
    rating_inters = filter_inters(rating_inters, can_items=meta_items,
                                  user_k_core_threshold=args.user_k,
                                  item_k_core_threshold=args.item_k)

    # sort interactions chronologically for each user
    rating_inters = make_inters_in_order(rating_inters)
    print('\n')

    # return: list of (user_ID, item_ID, rating, timestamp)
    return rating_inters


def get_user_item_from_ratings(ratings):
    users, items = set(), set()
    for line in ratings:
        user, item, rating, time = line
        users.add(user)
        items.add(item)
    return users, items


def clean_text(raw_text):
    if isinstance(raw_text, list):
        cleaned_text = ' '.join(raw_text)
    elif isinstance(raw_text, dict):
        cleaned_text = str(raw_text)
    else:
        cleaned_text = raw_text
    cleaned_text = html.unescape(cleaned_text)
    cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'
    if len(cleaned_text) >= 2000:
        cleaned_text = ''
    return cleaned_text


def generate_text(args, items, features):
    item_text_list = []
    already_items = set()

    dataset_full_name = amazon_dataset2fullname[args.dataset]
    meta_file_path = os.path.join(args.input_path, 'Metadata', f'meta_{dataset_full_name}.json.gz')
    with gzip.open(meta_file_path, 'r') as fp:
        for line in tqdm(fp, desc='Generate text'):
            data = json.loads(line)
            item = data['asin']
            if item in items and item not in already_items:
                already_items.add(item)
                text = ''
                for meta_key in features:
                    if meta_key in data:
                        meta_value = clean_text(data[meta_key])
                        text += meta_value + ' '
                item_text_list.append([item, text])
    return item_text_list


def load_text(file):
    item_text_list = []
    with open(file, 'r') as fp:
        fp.readline()
        for line in fp:
            try:
                item, text = line.strip().split('\t', 1)
            except ValueError:
                item = line.strip()
                text = '.'
            item_text_list.append([item, text])
    return item_text_list


def write_text_file(item_text_list, file):
    print('Writing text file: ')
    with open(file, 'w') as fp:
        fp.write('item_id:token\ttext:token_seq\n')
        for item, text in item_text_list:
            fp.write(item + '\t' + text + '\n')


def preprocess_text(args, rating_inters):
    print('Process text data: ')
    print(' Dataset: ', args.dataset)
    rating_users, rating_items = get_user_item_from_ratings(rating_inters)

    # load item text and clean
    item_text_list = generate_text(args, rating_items, ['title', 'category', 'brand'])
    print('\n')

    # return: list of (item_ID, cleaned_item_text)
    return item_text_list


def convert_inters2dict(inters):
    user2items = collections.defaultdict(list)
    user2index, item2index = dict(), dict()
    for inter in inters:
        user, item, rating, timestamp = inter
        if user not in user2index:
            user2index[user] = len(user2index)
        if item not in item2index:
            item2index[item] = len(item2index)
        user2items[user2index[user]].append(item2index[item])
    return user2items, user2index, item2index


def generate_training_data(args, rating_inters):
    print('Split dataset: ')
    print(' Dataset: ', args.dataset)

    # generate train valid test
    user2items, user2index, item2index = convert_inters2dict(rating_inters)
    train_inters, valid_inters, test_inters = dict(), dict(), dict()
    for u_index in range(len(user2index)):
        inters = user2items[u_index]
        # leave one out
        train_inters[u_index] = [str(i_index) for i_index in inters[:-2]]
        valid_inters[u_index] = [str(inters[-2])]
        test_inters[u_index] = [str(inters[-1])]
        assert len(user2items[u_index]) == len(train_inters[u_index]) + \
               len(valid_inters[u_index]) + len(test_inters[u_index])
    return train_inters, valid_inters, test_inters, user2index, item2index


def load_unit2index(file):
    unit2index = dict()
    with open(file, 'r') as fp:
        for line in fp:
            unit, index = line.strip().split('\t')
            unit2index[unit] = int(index)
    return unit2index


def write_remap_index(unit2index, file):
    with open(file, 'w') as fp:
        for unit in unit2index:
            fp.write(unit + '\t' + str(unit2index[unit]) + '\n')


def generate_item_embedding(args, item_text_list, item2index, text_processor):
    """
    Generate item embeddings using TextProcessor (Qwen3-based)
    """
    print(f'Generate Text Embedding using {args.plm_name}: ')
    print(' Dataset: ', args.dataset)

    # Extract items and texts
    items, texts = zip(*item_text_list)
    order_texts = [[0]] * len(items)
    for item, text in zip(items, texts):
        order_texts[item2index[item]] = text
    for text in order_texts:
        assert text != [0]

    embeddings = text_processor.process_texts(order_texts, batch_size=args.batch_size)
    print('Embeddings shape: ', embeddings.shape)

    # Save embeddings to file
    output_file = os.path.join(args.output_path, args.dataset, f'{args.dataset}.feat1LLM')
    embeddings.tofile(output_file)
    print(f'Embeddings saved to: {output_file}')


def convert_to_atomic_files(args, train_data, valid_data, test_data):
    print('Convert dataset: ')
    print(' Dataset: ', args.dataset)
    uid_list = list(train_data.keys())
    uid_list.sort(key=lambda t: int(t))

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.train.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid]
            seq_len = len(item_seq)
            for target_idx in range(1, seq_len):
                target_item = item_seq[-target_idx]
                seq = item_seq[:-target_idx][-50:]
                file.write(f'{uid}\t{" ".join(seq)}\t{target_item}\n')

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.valid.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid][-50:]
            target_item = valid_data[uid][0]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.test.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = (train_data[uid] + valid_data[uid])[-50:]
            target_item = test_data[uid][0]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Scientific', help='Pantry / Scientific / Instruments / Arts / Office')
    parser.add_argument('--user_k', type=int, default=5, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='item k-core filtering')
    parser.add_argument('--input_path', type=str, default='../raw/')
    parser.add_argument('--output_path', type=str, default='../bert/')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--plm_name', type=str, default='Qwen/Qwen3-Embedding-0.6B', help='Pre-trained language model name')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for text processing')
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio, do not drop by default')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # load interactions from raw rating file
    rating_inters = preprocess_rating(args)

    # load item text from raw meta data file
    item_text_list = preprocess_text(args, rating_inters)

    # split train/valid/test
    train_inters, valid_inters, test_inters, user2index, item2index = \
        generate_training_data(args, rating_inters)

    # device initialization
    device = set_device(args.gpu_id)
    args.device = device

    # create output dir
    check_path(os.path.join(args.output_path, args.dataset))

    # Initialize TextProcessor with specified LLM
    text_processor = TextProcessor(model_name=args.plm_name, device=device)
    
    # generate text embeddings and save to file
    generate_item_embedding(args, item_text_list, item2index, text_processor)

    # save interaction sequences into atomic files
    convert_to_atomic_files(args, train_inters, valid_inters, test_inters)

    # save useful data
    write_text_file(item_text_list, os.path.join(args.output_path, args.dataset, f'{args.dataset}.text'))
    write_remap_index(user2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.user2index'))
    write_remap_index(item2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item2index'))