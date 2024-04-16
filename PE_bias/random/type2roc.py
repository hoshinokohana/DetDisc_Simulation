import numpy as np

def type2roc(correct, conf, Nratings):
    """
    Calculate area under type 2 ROC curve in Python.
    
    Parameters:
    - correct: np.array of shape (n_trials,), 0 for error, 1 for correct trial
    - conf: np.array of shape (n_trials,) of confidence ratings taking values 1:Nratings
    - Nratings: int, number of confidence levels available
    
    Returns:
    - auroc2: float, area under the type 2 ROC curve
    """
    H2 = np.zeros(Nratings + 1)
    FA2 = np.zeros(Nratings + 1)
    
    for c in range(1, Nratings + 1):
        i = Nratings + 1 - c
        H2[i] = (np.sum((conf == c) & (correct == 1)) + 0.5)
        FA2[i] = (np.sum((conf == c) & (correct == 0)) + 0.5)
    
    # Calculating the cumulative sums for hits and false alarms
    CH2 = np.cumsum(H2) / np.sum(correct == 1)
    CFA2 = np.cumsum(FA2) / np.sum(correct == 0)
    
    # Calculating the area under the ROC curve by trapezoidal rule
    auroc2 = np.trapz(CH2, CFA2)
    
    return auroc2



# Converting 'Class.Acc' to binary
correct_binary = (data_df['Class.Acc'] > 75).astype(int)

# Using 'Conf(corre)' as 'conf'
conf = data_df['Conf(corre)']

# Inferring 'Nratings' from the unique values in 'Conf(corre)'
Nratings = conf.nunique()

# Since the Python version expects numpy arrays, we convert the pandas series to numpy arrays
correct_np = correct_binary.to_numpy()
conf_np = conf.to_numpy()

# Calculate the area under the type 2 ROC curve
auroc2 = type2roc(correct_np, conf_np, Nratings)
auroc2

0.78125

# Using 'Conf(incorre)' as 'conf' for incorrect confidence ratings
conf_incorrect = data_df['Conf(incorre)']

# Inferring 'Nratings' from the unique values in 'Conf(incorre)'
Nratings_incorrect = conf_incorrect.nunique()

# Convert 'Conf(incorre)' pandas series to numpy array
conf_incorrect_np = conf_incorrect.to_numpy()

# Calculate the area under the type 2 ROC curve using the new 'conf'
auroc2_incorrect = type2roc(correct_np, conf_incorrect_np, Nratings_incorrect)
auroc2_incorrect

0.66125

