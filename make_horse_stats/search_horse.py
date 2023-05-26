import pandas as pd
from datetime import datetime

df = pd.read_pickle("/Users/kari/Desktop/toto_horse2/make_horse_stats/horses.pkl")

def search_horse(name):
    horse_stats = df.query("name == @name")
    #horse_stats['h_w_s'] = horse_stats['h_w_s'].fillna(0.0)
    #res_days =  days_between_races(list(horse_stats['day']))
    #horse_stats['rest_days'] = res_days
    print(horse_stats)
    return  horse_stats.iloc[-1:]



def days_between_races(days):
    
    days_arr = [0]
    try:
        for i in range(len(days)):
            d1 = datetime.strptime(days[i], "%Y-%m-%d")
            d2 = datetime.strptime(days[i +1], "%Y-%m-%d")
            days_arr.append(abs((d2 - d1).days))
        
    except:
        print("")
    
    print(days_arr)

    return days_arr


search_horse("Roughenough Vice")
