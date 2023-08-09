import torch
import os
import pickle
from collections import defaultdict
from transformers import AutoTokenizer
from utils.misc import invert_dict

def collate(batch):
    batch = list(zip(*batch))
    topic_entity, question, answer, entity_range,relation_embedd = batch
    topic_entity = torch.stack(topic_entity)
    question = {k:torch.cat([q[k] for q in question], dim=0) for k in question[0]}
    answer = torch.stack(answer)
    entity_range = torch.stack(entity_range)
    # relation_embedd = {k:torch.cat([q[k] for q in relation_embedd[0]], dim=0) for k in relation_embedd[0][0]}
    # relation_embedd = {k: torch.cat([q[k] for q in relation_embedd], dim=0) for k in relation_embedd[0]}

    # relation_embedd = {k: torch.cat([q[k] for q in relation_embedd], dim=0) for k in relation_embedd[0]}
    relation_embedd = relation_embedd[0]
    return topic_entity, question, answer, entity_range,relation_embedd


class Dataset(torch.utils.data.Dataset):
    def __init__(self, questions, ent2id,relation_embedd):
        self.questions = questions
        self.ent2id = ent2id
        self.relation_embedd=relation_embedd

    def __getitem__(self, index):
        topic_entity, question, answer, entity_range = self.questions[index]
        topic_entity = self.toOneHot(topic_entity)
        answer = self.toOneHot(answer)
        entity_range = self.toOneHot(entity_range)
        relation_embedd = self.relation_embedd
        return topic_entity, question, answer, entity_range,relation_embedd

    def __len__(self):
        return len(self.questions)

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        vec_len = len(self.ent2id)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, input_dir, fn, bert_name, ent2id, rel2id, batch_size, relation_embedd,training=False):
        print('Reading questions from {}'.format(fn))
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.id2ent = invert_dict(ent2id)
        self.id2rel = invert_dict(rel2id)



        sub_map = defaultdict(list)
        so_map = defaultdict(list)
        for line in open(os.path.join(input_dir, 'fbwq_full/train.txt')):
            l = line.strip().split('\t')
            s = l[0]
            p = l[1].strip()
            o = l[2]
            sub_map[s].append((p, o))
            so_map[(s, o)].append(p)


        data = []
        for line in open(fn):
            line = line.strip()
            if line == '':
                continue
            line = line.split('\t')
            # if no answer
            if len(line) != 2:
                continue
            question = line[0].split('[')
            question_1 = question[0]
            question_2 = question[1].split(']')
            head = question_2[0]
            question_2 = question_2[1]
            # question = question_1 + 'NE' + question_2
            question = question_1.strip()
            ans = line[1].split('|')


            # if (head, ans[0]) not in so_map:
            #     continue

            entity_range = set()
            for p, o in sub_map[head]:
                entity_range.add(o)
                for p2, o2 in sub_map[o]:
                    entity_range.add(o2)
            entity_range = [ent2id[o] for o in entity_range]

            head = [ent2id[head]]
            question = self.tokenizer(question.strip(), max_length=64, padding='max_length', return_tensors="pt")
            ans = [ent2id[a.strip()] for a in ans]
            data.append([head, question, ans, entity_range])

        print('data number: {}'.format(len(data)))
        # relation_embedd={k: torch.cat([q[k] for q in relation_embedd], dim=0) for k in relation_embedd[0]}
        dataset = Dataset(data, ent2id,relation_embedd)

        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate, 
            )


def load_data(input_dir, bert_name, batch_size):
    cache_fn = os.path.join(input_dir, 'processed.pt')
    if os.path.exists(cache_fn):
        print('Read from cache file: {} (NOTE: delete it if you modified data loading process)'.format(cache_fn))
        with open(cache_fn, 'rb') as fp:
            ent2id, rel2id, triples, train_data, test_data = pickle.load(fp)
        print('Train number: {}, test number: {}'.format(len(train_data.dataset), len(test_data.dataset)))
    else:
        print('Read data...')
        ent2id = {}
        for line in open(os.path.join(input_dir, 'fbwq_full/entity_ids.del')):
            l = line.strip().split('\t')
            ent2id[l[1]] = len(ent2id)
        # print(len(ent2id))
        # print(max(ent2id.values()))
        rel2id = {}
        for line in open(os.path.join(input_dir, 'fbwq_full/relations.dict')):
            l = line.strip().split('\t')
            rel2id[l[0].strip()] = int(l[1])

        triples = []
        for line in open(os.path.join(input_dir, 'fbwq_full/train.txt')):
            l = line.strip().split('\t')
            s = ent2id[l[0]]
            p = rel2id[l[1].strip()]
            o = ent2id[l[2]]
            triples.append((s, p, o))
            p_rev = rel2id[l[1].strip()+'_reverse']
            triples.append((o, p_rev, s))
        triples = torch.LongTensor(triples)
        tokenizer = AutoTokenizer.from_pretrained(bert_name)
        relation_embedd=[]
        for line in open(os.path.join(input_dir, 'fbwq_full/relations.dict')):
            l = line.strip().split('\t')
            sourse_level=l[0].strip()
            relation_level = sourse_level.split('.')[-1]
            relation = relation_level
            relation_e=tokenizer(relation.strip(), max_length=15, padding='max_length', return_tensors="pt")
            relation_embedd.append(relation_e)
            # relation.append(sourse_level)
            # relation.append(relation_level)
            # relation.append(type_level)
        relation_embedd=tuple(relation_embedd)
        # i=0
        # for k in relation_embedd[0]:
        #     # for q in relation_embedd:
        #     l=torch.cat([q[k] for q in relation_embedd], dim=0)
        #     relation_embedd={k:l}
        relation_embedd = {k: torch.cat([q[k] for q in relation_embedd], dim=0) for k in relation_embedd[0]}
        train_data = DataLoader(input_dir, os.path.join(input_dir, 'QA_data/WebQuestionsSP/qa_train_webqsp.txt'), bert_name, ent2id, rel2id, batch_size,relation_embedd, training=True)
        test_data = DataLoader(input_dir, os.path.join(input_dir, 'QA_data/WebQuestionsSP/qa_test_webqsp_fixed.txt'), bert_name, ent2id, rel2id, batch_size,relation_embedd)
    
        with open(cache_fn, 'wb') as fp:
            pickle.dump((ent2id, rel2id, triples, train_data, test_data), fp)

    return ent2id, rel2id, triples, train_data, test_data
