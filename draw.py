import os

import numpy as np
from matplotlib import pyplot as plt


def visual_result():
    visual_dir = './VisualResults'
    visual_dir = os.path.join(visual_dir, '{}_KES_assist093.npy')
    RR_result = np.load(visual_dir.format('RR'))
    RRM_result = np.load('./VisualResults/{}_KES_assist09m3.npy'.format('RR'))
    DQN_result = np.load(visual_dir.format('DQN'))
    random_result = np.load(visual_dir.format('Random'))
    plt.plot(RR_result, label='ReRank')
    plt.plot(RRM_result, label='ReRank-MLP')
    plt.plot(DQN_result, label='DQN')
    plt.plot(random_result, label='Random')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Result')
    plt.legend()
    plt.show()


def f(str_data1, str_data2):
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 14  # 设置x轴刻度数字大小
    plt.rcParams['ytick.labelsize'] = 14  # 设置y轴刻度数字大小
    plt.subplots(1, 2, figsize=(10, 5))
    for j, str_data in enumerate((str_data1, str_data2)):
        models_reward = str_data.split()
        models_reward = np.array(models_reward, dtype=float).reshape((4, -1))
        print(models_reward)
        x = np.arange(1, models_reward.shape[1] + 1) * 5
        models = ['SRC', 'GRU4Rec', 'Rule-Based', 'DQN']
        plt.subplot(1, 2, j + 1)
        for i, model in enumerate(models):
            plt.plot(x, models_reward[i], label=model, marker='x')
        plt.xlabel('Lengths', fontsize=20)
        plt.ylabel(r'$E_T$', fontsize=20)
        plt.title(f'p={j + 2}', fontsize=20, weight='bold')
        if j == 0:
            plt.legend(loc='lower center', fontsize=20, bbox_to_anchor=(1.05, 1.05), ncol=4)
        plt.gcf().subplots_adjust(bottom=0.13, top=0.8)
    plt.subplots_adjust(wspace=0.25)
    # plt.savefig('length.pdf', format='pdf')
    plt.show()
    plt.close()


# f('0.0256 0.1175 0.1875 0.2345 0.2627 0.3145 0.3577\n'
#   '-0.0052 0.0747 0.1126 0.1505 0.2179 0.2344 0.2702\n'
#   '0.0091 0.0847 0.1418 0.1950 0.2145 0.2562 0.2834\n'
#   '-0.0323 0.0137 0.0630 0.0723 0.0891 0.1313 0.1636')
f('0.0256 0.1175 0.1875 0.2345 0.2627 0.3145 0.3577\n'
  '-0.0052 0.0747 0.1126 0.1505 0.2179 0.2344 0.2702\n'
  '0.0091 0.0847 0.1418 0.1950 0.2145 0.2562 0.2834\n'
  '-0.0323 0.0137 0.0630 0.0723 0.0891 0.1313 0.1636',
  '0.2949 0.4511 0.5072 0.5567 0.5659 0.5718 0.5941\n'
  '-0.0357 0.0034 0.0547 0.0755 0.1007 0.1120 0.1688\n'
  '0.2345 0.3229 0.3947 0.4233 0.4279 0.4255 0.4339\n'
  '0.1546 0.2064 0.2419 0.2949 0.3076 0.3054 0.3101')
