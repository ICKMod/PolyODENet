import matplotlib.pyplot as plt
import seaborn as sns
from torchdiffeq import odeint_adjoint as odeint
import torch
import numpy as np
import pandas as pd
import string


def plot_write_ode(ode, real_concentrations, real_timestamps, name, device):
    tensor_concentrations = torch.tensor(np.array(real_concentrations), dtype=torch.float64, requires_grad=False, device=device)
    tensor_timestamps = torch.tensor(np.array(real_timestamps), dtype=torch.float64, requires_grad=False, device=device)

    with open(f"Rate_equations_of_{name.replace('/','_')}.txt", "w") as f:
        f.write(str(ode))

    for i in range(len(real_timestamps)):
        tensor_conc0 = tensor_concentrations[i][0]
        pred_conc = odeint(ode, tensor_conc0, tensor_timestamps[i])
        timestamps = tensor_timestamps[i].cpu().numpy()
        concentrations = tensor_concentrations[i].cpu().numpy()
        pred_conc = pred_conc.detach().cpu().numpy()
        plt.figure()
        sns.set_palette("Dark2", concentrations.shape[1])
        for conc in concentrations[::2].T:
            plt.scatter(timestamps[::2], conc, s=15.0)

        j = 0
        for conc in pred_conc[::2].T:
            plt.plot(timestamps[::2], conc, label=string.ascii_uppercase[j])
            j += 1

        plt.xlabel("Time")
        plt.ylabel("Concentration")
        plt.ylim(bottom=-0.1)
        plt.legend()
        title = f"Plot_of_{name.replace('/','_')}_set_{i}"
        plt.title(title)
        plt.savefig(f"{title}.jpg", dpi=300)

        pred_df = pd.DataFrame(data=np.hstack([timestamps[:, np.newaxis], pred_conc]),
                               columns=["## Time"] + [f'Set {j + 1} ' for j in range(pred_conc.shape[1])])
        pred_df.to_csv(f"Predicted_Concentrations_of_{name.replace('/','_')}_set_{i}.csv", sep='\t', index=False, float_format='%8.6f')
