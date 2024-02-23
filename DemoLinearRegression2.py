import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


X = np.array([55,60,65,70,75,80]).reshape(-1,1)
y = np.array([316,292,268,246,227,207]).reshape(-1,1)

scalerX = StandardScaler()
scalerX.fit(X)
scalerY = StandardScaler()
scalerY.fit(y)
xScaled = scalerX.transform(X)
yScaled = scalerY.transform(y)

# Building the model
beta0 = 0
beta1 = 0

a = 0.0001 # learning rate, alpha
n_iterations = 10000

# Perform gradient descent
for i in range(n_iterations):
    yPred = beta0 + beta1*xScaled
    D_beta0 = -2 * sum(yScaled-yPred)
    D_beta1 = -2 * sum(xScaled*(yScaled-yPred))
    
    beta0 = beta0 - a * D_beta0
    beta1 = beta1 - a * D_beta1
    
    print('iteration', i, ', beta0 =', beta0, ', beta1 =', beta1)
    
# Plot the results
yPred = beta0 + beta1*xScaled
plt.scatter(X, y)
plt.plot(X, scalerY.inverse_transform(yPred), color = 'r') # regression line
plt.show()
    