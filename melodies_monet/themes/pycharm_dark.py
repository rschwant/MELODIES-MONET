import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('dark_background')
sns.set_context('notebook', font_scale=2)

textcolor = (175 / 255, 177 / 255, 179 / 255)
facecolor = (60 / 255, 63 / 255, 65 / 255)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['text.color'] = textcolor
plt.rcParams['axes.edgecolor'] = textcolor
plt.rcParams['axes.labelcolor'] = textcolor
plt.rcParams['xtick.color'] = textcolor
plt.rcParams['ytick.color'] = textcolor
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 4
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 2
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 4
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 2
