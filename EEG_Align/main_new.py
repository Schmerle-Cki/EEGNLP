import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import json
from torch.utils.data import DataLoader
import time
from transformers import BertModel, BertTokenizer

from config import EEG_LEN, TEXT_LEN, d_model, d_inner, d_k, d_v, class_num, dropout
from optim_new import ScheduledOptim, early_stopping
from trainer import train
from evaluator import eval, inference
from model_new import Transformer
from utils import open_file
from new_plot import plot_learning_curve
from dataset_new import prepare_sr_eeg_data, EEGDataset, clean_dic, shuffle_split_data
torch.set_num_threads(2)

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--model', type=str, help="Please choose a model from the following list: ['transformer', 'biLSTM', 'MLP', 'resnet']")
    parser.add_argument('--modality', type = str, default = None, help="Please choose a modality from the following list: ['eeg', 'text', fusion]")
    parser.add_argument('--dataset', type=str, help="Please choose a dataset from the following list: ['KEmoCon', 'ZuCo']")
    # NOTE: Nicki adds the option for data storage
    parser.add_argument('--dataRoot', type=str, default = "/home/nicki", help="The paremt directory for 'datasets'")
    parser.add_argument('--task', default ='SA', type=str, help="If dataset == Zuco, please choose a task from the following list: ['SA', 'RD']")
    parser.add_argument('--level', type=str, default = 'sentence', help="If ZuCo, please choose the level of EEG feature you want to work with from this list: ['word', 'concatword', 'sentence']")
    parser.add_argument('--batch_size', type=int, default = 64)
    parser.add_argument('--text_feature_len', type = int, default = 768)
    parser.add_argument('--eeg_feature_len', type = int, default = 832)
    parser.add_argument('--lr', type = float, default = 1e-5)
    parser.add_argument('--eps', type = float, default = 1e-4)
    parser.add_argument('--weight_decay', type = float, default = 1e-2)
    parser.add_argument('--warm_steps', type = int, default = 2000)
    parser.add_argument('--epochs', type = int, default = 200)
    parser.add_argument('--device', type = str, default = 'cpu')
    parser.add_argument('--inference', type = int, default = 0)
    parser.add_argument('--checkpoint', type = str, default = None)
    parser.add_argument('--dev', type = int, default = 0)
    parser.add_argument('--loss', type = str, default = 'CE', help = "Please choose one of the following loss functions [CE, CCA, WD, CCAWD]")
    parser.add_argument('--num_layers', type = int, default = 1, help = 'Please choose how many layers the encoder should have')
    parser.add_argument('--num_heads', type = int, default = 1, help = 'Please choose how many heads the encoder should have')
    parser.add_argument('--dropout', type= float, default = 0.3, help = 'Please indicate the dropout proportion')
    parser.add_argument('--text_llm', type=str, default = 'bert', help = 'Please choose which LLM to encode text')
    parser.add_argument('--ce_weight', type = float, default = 1, help = 'Please choose the ce loss weight')
    parser.add_argument('--cca_weight', type = float, default = 1, help = 'Please choose the cca loss weight')
    parser.add_argument('--wd_weight', type = float, default = 1, help = 'Please choose the wd loss weight')
    return parser.parse_args()


if __name__ == '__main__':
    
    
    args = get_args()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    device = torch.device(args.device)
    
    # print(device)
    
    if args.dataset == 'KEmoCon':
        ###### COMING SOON #####
        pass
    
    elif args.dataset == 'ZuCo':
        
        if args.task == 'RD':
            ###### COMING SOON #####
            pass
        
        elif args.task == 'SA':
            
            assert (args.level == 'sentence' or args.level == 'word' or args.level == 'concatword'), 'Please choose a correct eeg feature type'
            
            if args.level == 'word':
                ###### COMING SOON #####
                pass
            elif args.level == 'concatword':
                ###### COMING SOON #####
                pass      
            
            else:
                # Load csv
                sentiment_labels = pd.read_csv('./preprocessed/ZuCo/sentiment_labels_clean.csv')
                
                sr_eeg_data_path = os.path.join(args.dataRoot, 'datasets/ZuCo/task1-SR/Matlab_files')
                
                sentence_list = sentiment_labels.sentence.tolist()
                labels_list = sentiment_labels.sentiment_label.tolist()
                sentence_ids_list = sentiment_labels.sentence_id.tolist()
                
                eeg_dict = prepare_sr_eeg_data(sr_eeg_data_path, sentence_list, labels_list, sentence_ids_list, args)
                
                eeg_train_split, eeg_val_split, eeg_test_split = shuffle_split_data(eeg_dict)
                
                train_set, train_id_mapping = clean_dic(eeg_train_split)
                val_set, val_id_mapping = clean_dic(eeg_val_split)
                test_set, test_id_mapping = clean_dic(eeg_test_split)
                
                
                train_dataset = EEGDataset(train_set, args)
                val_dataset = EEGDataset(val_set, args)
                test_dataset = EEGDataset(test_set,args)
                                
                train_loader = DataLoader(
                    dataset=train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True, 
                    drop_last = True
                )
                val_loader = DataLoader(
                    dataset=val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                )
                test_loader = DataLoader(
                    dataset=test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last = True
                )
                
                if args.model == 'transformer':
                    model = Transformer(device = device, d_feature_text = TEXT_LEN, d_feature_eeg = EEG_LEN,\
                                            d_model = d_model, d_inner = d_inner, n_layers = args.num_layers, \
                                            n_head=args.num_heads, d_k = d_k, d_v = d_v, dropout= dropout, \
                                            class_num = class_num, args = args)
                elif args.model == 'bert':
                    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                    model = BertModel.from_pretrained("bert-base-uncased")
                    
                model = model.to(device)
                
                torch.manual_seed(2)
                
                optimizer = ScheduledOptim(
                    Adam(filter(lambda x: x.requires_grad, model.parameters()), 
                         betas = (0.9, 0.98), eps = args.eps, lr = args.lr, weight_decay = args.weight_decay),
                    d_model = d_model, n_warmup_steps = args.warm_steps
                )
                
                all_train_loss, all_train_acc, all_val_loss, all_val_acc = [], [], [], []
                all_pred_val, all_label_val = [], []
                eva_indices = []
                all_epochs  = []
                if args.inference == 1:
                    chkpt_path = os.path.join('baselines', args.checkpoint)
                    print(chkpt_path)
                    checkpoint = torch.load(chkpt_path, map_location = 'cuda')
                    model.load_state_dict(checkpoint['model'])
                    model = model.to(device)
                    inference(test_loader, device, model, test_dataset.__len__(), args)
                
                else:
                    since = time.time()
                    for epoch in range(args.epochs):
                        
                        print('[ Epoch', epoch, ']')
                        start = time.time()
                        
                        train_loss, train_acc, train_cm, train_preds, train_labels = train(train_loader, device, model, optimizer, train_dataset.__len__(), args)
                        val_loss, val_acc, val_cm, val_preds, val_labels = eval(val_loader, device, model, val_dataset.__len__(), args)
                        
                        model_state_dict = model.state_dict()
                        
                        checkpoint = {
                            'model' : model_state_dict,
                            'config_file' : 'config',
                            'epoch' : epoch
                        }
                    
                        all_pred_val.extend(val_preds)
                        all_label_val.extend(val_labels)
                        all_train_loss.append(train_loss)
                        all_train_acc.append(train_acc)
                        all_val_loss.append(val_loss)
                        all_val_acc.append(val_acc)
                        all_epochs.append(epoch)
                        
                        
                        if val_loss <= max(all_val_loss):
                                torch.save(checkpoint, f'./baselines/{args.model}_{args.modality}_{args.level}_{args.num_layers}_{args.num_heads}_{args.batch_size}_{args.loss}_{args.ce_weight}_{args.cca_weight}_{args.wd_weight}.chkpt')
                                print('    - [Info] The checkpoint file has been updated.')
                            
                        early_stop = early_stopping(all_val_loss, patience = 10, delta = 0.01)
                        
                        if early_stop:
                            print('Validation loss has stopped decreasing. Early stopping...')
                            break   
                    time_elapsed = time.time() - since
                    print(f'EEG_Text_Align Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

                    plot_learning_curve(all_train_acc, all_train_loss, all_val_acc, all_val_loss, all_epochs, args, checkpoint_name='ckpt')
                    checkpoint = torch.load(f'./baselines/{args.model}_{args.modality}_{args.level}_{args.num_layers}_{args.num_heads}_{args.batch_size}_{args.loss}_{args.ce_weight}_{args.cca_weight}_{args.wd_weight}.chkpt', map_location = 'cuda')
                    model.load_state_dict(checkpoint['model'])
                    model = model.to(device)
                    inference(test_loader, device, model, test_dataset.__len__(), args)    
                                        
                    
                    
                    
            
                
