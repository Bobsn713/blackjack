import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

#Helper functions for data processing
def card_to_num(card):
    raw_rank = card[:-1]
    
    ranks = {
        '2' : 0,
        '3' : 1,
        '4' : 2, 
        '5' : 3,
        '6' : 4, 
        '7' : 5, 
        '8' : 6, 
        '9' : 7, 
        '10': 8, 
        'J' : 9, 
        'Q' : 10, 
        'K' : 11, 
        'A': 12
    }

    return ranks[raw_rank]

def hand_to_list(hand):
    '''Takes hand like KH-AC and outputs list of card numbers'''
    hand_list_1 = hand.split("-")
    hand_list_2 = [card_to_num(card) for card in hand_list_1]
    return hand_list_2

result_mapping = {
    'hit' : 0,
    'stand' : 1,
    'double down' : 2
}

# Data processing
split_or_not_raw_df = pd.read_csv('CSVs/split_or_not.csv')
hit_stand_dd_df = pd.read_csv('CSVs/hit_stand_dd.csv')

# Cleaned split_or_not
split_or_not_raw_df['dealer_upcard'] = split_or_not_raw_df['dealer_upcard'].apply(card_to_num)
split_or_not_raw_df['player_hand'] = split_or_not_raw_df['player_hand'].apply(hand_to_list)
split_or_not_raw_df['player_hand'] = split_or_not_raw_df['player_hand'].apply(lambda hand: hand[0])
split_or_not_df = split_or_not_raw_df.rename(columns = {'player_hand':'player_upcard'})

#Cleaned hit_stand_dd
hit_stand_dd_df['dealer_upcard'] = hit_stand_dd_df['dealer_upcard'].apply(card_to_num)
hit_stand_dd_df['player_hand'] = hit_stand_dd_df['player_hand'].apply(hand_to_list)
hit_stand_dd_df['result'] = hit_stand_dd_df['result'].map(result_mapping)

class Split_or_Not_Dataset(Dataset):
    def __init__(self, dataframe):
        data = dataframe

        self.X = data.drop(columns=['result']).values.astype('float32')
        self.y = data['result'].values.astype('int64')

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
    
