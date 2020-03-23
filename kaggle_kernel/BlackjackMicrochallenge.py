import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# 观察不同dealer_card_val、player_total下平均胜率
def insight_data():
    move_df = pd.read_csv("./data/all_games.csv")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    p_table = 100*move_df.query('not cur_move').pivot_table(index='dealer_card_val', columns='player_total',
                                                            aggfunc='mean', values='result')
    sns.heatmap(p_table, ax=ax1, annot=True, fmt='2.0f', vmin=10, vmax=90)
    ax1.set_title('If player stays')
    ax1.set_ylabel('Dealer Card')
    ax1.set_xlabel('Player Total')

    p_table = 100*move_df.query('cur_move').pivot_table(index='dealer_card_val', columns='player_total',
                                                            aggfunc='mean', values='result')
    sns.heatmap(p_table, ax=ax2, annot=True, fmt='2.0f', vmin=10, vmax=90)
    ax1.set_title('If player hits')
    ax1.set_ylabel('Dealer Card')
    ax1.set_xlabel('Player Total')
    plt.show()


# 建立预测模型
def build_model():
    move_df = pd.read_csv("./data/all_games.csv")
    x_vars = ['player_total', 'dealer_card_val', 'player_aces']
    hit_df = move_df.query('cur_move').groupby(x_vars).agg('mean').reset_index()
    hit_forest = RandomForestRegressor()
    hit_forest.fit(hit_df[x_vars], hit_df['result'])

    stand_df = move_df.query('not cur_move').groupby(x_vars).agg('mean').reset_index()
    stand_forest = RandomForestRegressor()
    stand_forest.fit(stand_df[x_vars], stand_df['result'])

    player_total = 12
    player_aces = 0
    dealer_card_val = 7
    #actual_val = 0.0
    hit_score = hit_forest.predict(np.reshape([player_total, dealer_card_val, player_aces], (1, -1)))[0]
    stand_score = stand_forest.predict(np.reshape([player_total, dealer_card_val, player_aces], (1, -1)))[0]
    print("hit_score vs stand_score: %s -- %s" % (hit_score, stand_score))


if __name__ == '__main__':
    # insight_data()
    build_model()
