import numpy as np
import matplotlib.pyplot as plt
import data

from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

X = data.import_feature("no_categorical_new_imp")
y = data.import_feature("labels")

lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 5)

scores = list()
scores_std = list()

n_folds = 3

for alpha in alphas:
    lasso.alpha = alpha
    this_scores = cross_val_score(lasso, X, y, cv=n_folds, n_jobs=1)
    # print ("done for alpha: " + str(alpha))
    scores.append(np.mean(this_scores))
    # print (str(np.mean(this_scores)))
    scores_std.append(np.std(this_scores))

scores, scores_std = np.array(scores), np.array(scores_std)

plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)

# plot error lines showing +/- std. errors of the scores
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.title("Results for LASSO regression with multiple alfas")
plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])

print ("show")
plt.show()
