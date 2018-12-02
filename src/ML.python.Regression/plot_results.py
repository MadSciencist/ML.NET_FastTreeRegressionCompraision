import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

def main():
    python_res = pd.read_csv('../../results/python_results.csv')
    dotnet_res = pd.read_csv('../../results/dotnet_results.csv')

    results = python_res.join(dotnet_res)
    results = results.sort_values(['test'])

    results = results.reset_index(drop=True)

    plt.title('Prediction comparision')
    plt.ylabel('Predicted value', fontsize=10)
    plt.xlabel('Test sample number', fontsize=10)

    plt.scatter(x=results.index.values, y=results['pred'].values,  color='blue', s=2, alpha=0.25, label='SK-Learn')
    plt.scatter(x=results.index.values, y=results['dotnet_pred'].values,  color='green', s=2, alpha=0.325, label='ML.NET')
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
