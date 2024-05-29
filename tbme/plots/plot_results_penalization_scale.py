import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# read dataframe
nb_seeds = 30
results_dir = "/storage/store2/work/aheurteb/mvicad/tbme/data/"
save_name = f"DataFrame_with_{nb_seeds}_by_penalization_scale"
save_path = results_dir + save_name
df = pd.read_csv(save_path)

# plot
fig = plt.figure(figsize=(6, 4))
sns.lineplot(data=df, x="penalization scale", y="Amari LBFGSB", linewidth=2.5, label="LBFGSB", estimator=np.median)
sns.lineplot(data=df, x="penalization scale", y="Amari MVICAD", linewidth=2.5, label="MVICAD", estimator=np.median)
sns.lineplot(data=df, x="penalization scale", y="Amari MVICAD ext", linewidth=2.5, label="MVICAD extended", estimator=np.median)
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Amari distance")
plt.grid()
plt.title("Amari distance with respect to penalization scale")
figures_dir = "/storage/store2/work/aheurteb/mvicad/tbme/figures/"
plt.savefig(figures_dir + "amari_distance_wrt_penalization_scale.pdf")
plt.show()
