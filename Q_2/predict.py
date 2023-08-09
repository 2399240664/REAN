import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from collections import defaultdict
from utils.misc import batch_device
from data import load_data
from model import REAN
import logging
from IPython import embed


def validate(args, model, data, device,verbose = False):
    model.eval()
    count = 0
    correct = 0
    hop_count = defaultdict(list)
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            # batch = list(batch)
            # batch.append(relation_embedd)
            # batch = tuple(batch)
            outputs = model(*batch_device(batch, device)) # [bsz, Esize]
            e_score = outputs['e_score'].cpu()
            scores, idx = torch.max(e_score, dim = 1) # [bsz], [bsz]
            match_score = torch.gather(batch[2], 1, idx.unsqueeze(-1)).squeeze().tolist()
            count += len(match_score)
            correct += sum(match_score)
            # for i in range(len(match_score)):
            #     h = outputs['hop_attn'][i].argmax().item()
            #     hop_count[h].append(match_score[i])

            if verbose:
                with open('data/true.txt','a+') as f:
                    answers = batch[2]
                    for i in range(len(match_score)):
                        if match_score[i] == 1:
                            print('================================================================',file=f)
                            question_ids = batch[1]['input_ids'][i].tolist()
                            question_tokens = data.tokenizer.convert_ids_to_tokens(question_ids)
                            print(' '.join(question_tokens),file=f)
                            topic_id = batch[0][i].argmax(0).item()
                            print('> topic entity: {}'.format(data.id2ent[topic_id]),file=f)
                            for t in range(2):
                                print('>>>>>>> step {}'.format(t),file=f)
                                tmp = ' '.join(['{}: {:.3f}'.format(x, y) for x,y in
                                    zip(question_tokens, outputs['word_attns'][t][i].tolist())])
                                print('> Attention: ' + tmp,file=f)
                                print('> Relation:',file=f)
                                rel_idx = outputs['rel_probs'][t][i].gt(0.9).nonzero().squeeze(1).tolist()
                                for x in rel_idx:
                                    print('  {}: {:.3f}'.format(data.id2rel[x], outputs['rel_probs'][t][i][x].item()),file=f)

                                print('> Entity: {}'.format('; '.join([data.id2ent[_] for _ in outputs['ent_probs'][t][i].gt(0.8).nonzero().squeeze(1).tolist()])),file=f)
                            print('----',file=f)
                            print('> max is {}'.format(data.id2ent[idx[i].item()]),file=f)
                            print('> golden: {}'.format('; '.join([data.id2ent[_] for _ in answers[i].gt(0.9).nonzero().squeeze(1).tolist()])),file=f)
                            print('> prediction: {}'.format('; '.join([data.id2ent[_] for _ in e_score[i].gt(0.9).nonzero().squeeze(1).tolist()])),file=f)
                            print(' '.join(question_tokens),file=f)
                            # print(outputs['hop_attn'][i].tolist(),file=f)

                        # embed()
    acc = correct / count
    print(acc)
    # print('pred hop accuracy: 1-hop {} (total {}), 2-hop {} (total {})'.format(
    #     sum(hop_count[0])/(len(hop_count[0])+0.1),
    #     len(hop_count[0]),
    #     sum(hop_count[1])/(len(hop_count[1])+0.1),
    #     len(hop_count[1]),
    #     ))
    return acc


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', default = './input')
    parser.add_argument('--ckpt', required = True)
    parser.add_argument('--mode', default='vis', choices=['val', 'vis', 'test'])
    parser.add_argument('--bert_name', default='bert-base-cased', choices=['roberta-base', 'bert-base-uncased'])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ent2id, rel2id, triples, train_loader, val_loader = load_data(args.input_dir, 'bert-base-cased',32)
    # for k in relation_embedd:
    #     if isinstance(relation_embedd[k], torch.Tensor):
    #         relation_embedd[k] = relation_embedd[k].to(device)
    model = REAN(args, ent2id, rel2id, triples)
    # missing, unexpected = model.load_state_dict(torch.load(args.ckpt), strict=False)
    model.load_state_dict(torch.load(args.ckpt))
    # if missing:
    #     print("Missing keys: {}".format("; ".join(missing)))
    # if unexpected:
    #     print("Unexpected keys: {}".format("; ".join(unexpected)))
    model = model.to(device)
    # model.triples = model.triples.to(device)
    model.Msubj = model.Msubj.to(device)
    model.Mobj = model.Mobj.to(device)
    model.Mrel = model.Mrel.to(device)
    # acc = validate(args, model, val_loader, device)
    # logging.info(acc)
    if args.mode == 'vis':
        validate(args, model, val_loader, device, True)
    elif args.mode == 'val':
        validate(args, model, val_loader, device, False)

if __name__ == '__main__':
    main()
