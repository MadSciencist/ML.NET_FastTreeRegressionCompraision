import numpy as np
import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt

def main():
    python_res = pd.read_csv('../../results/python_results.csv')
    dotnet_res = pd.read_csv('../../results/dotnet_results.csv')

    results = python_res.join(dotnet_res)
    results = results.sort_values(['test'])

    results = results.reset_index(drop=True)
    results_python = results['pred'].values
    results_dotnet = results['dotnet_pred'].values

    print(f'ML.NET variance={np.var(results_dotnet)}  SK LEARN variance={np.var(results_python)}')

    plt.hist(results_python, 50, normed=1, facecolor='blue', alpha=0.5)  
    plt.hist(results_dotnet, 50, normed=1, facecolor='red', alpha=0.5)
    plt.show()

    t2, p2 = stats.ttest_ind(results_dotnet, results_python)
    print(f't= {t2}  p= {p2}')

    if p2 > 0.05:
	    print('fail to reject H0')
    else:
	    print('reject H0)')

    plt.title('Prediction comparision')
    plt.ylabel('Predicted value', fontsize=10)
    plt.xlabel('Test sample number', fontsize=10)

    plt.scatter(x=results.index.values, y=results_python,  color='blue', s=2, alpha=0.25, label='SK-Learn')
    plt.scatter(x=results.index.values, y=results_dotnet,  color='green', s=2, alpha=0.325, label='ML.NET')
    plt.scatter(x=results.index.values, y=results['test'].values,  color='red', s=2, label='Test data')

    axes = plt.gca()
    axes.set_ylim(bottom=0)

    lgnd =  plt.legend(loc='upper left', fontsize=10)
    lgnd.legendHandles[0]._sizes = [30]
    lgnd.legendHandles[1]._sizes = [30]
    lgnd.legendHandles[2]._sizes = [30]

    plt.show()


if __name__=='__main__':
    main()
