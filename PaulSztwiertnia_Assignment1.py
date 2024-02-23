from sklearn import datasets 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer

# Fetch the breast cancer wisconsin dataset. 
X, y = datasets.load_breast_cancer(return_X_y=True) 
data = load_breast_cancer()
# Check how many instances we have in the dataset, and how many features describe these instances
print("There are", X.shape[0], "instances described by", X.shape[1], "features.")   

# Create a training and test set such that the test set has 40% of the instances from the 
# complete breast cancer wisconsin dataset and that the training set has the remaining 60% of  
# the instances from the complete breast cancer wisconsin dataset, using the holdout method. 
# Ensure that the training and test sets # contain approximately the same 
# percentage of instances of each target class as the complete set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state = 42)  

# Create a decision tree classifier. Then Train the classifier using the training dataset created earlier.
# Measure the quality of a split, using the entropy criteria.
# Ensure that nodes with less than 6 training instances are not further split
clf = tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_split=6)
clf.fit(X_train,y_train) 

# Apply the decision tree to classify the data 'testData'.
predC = clf.predict(X_test) 

# Compute the accuracy of the classifier on 'testData'
accuracy = accuracy_score(y_test, predC)  
print('The accuracy of the classifier is', accuracy)

# Visualize the tree created.
plt.figure(figsize=(15, 10))
tree.plot_tree(clf, fontsize=12)
plt.show()

# Visualize the training and test error as a function of the maximum depth of the decision tree
# Initialize 2 empty lists and save the training and testing accuracies 
# as we iterate through the different decision tree depth options.
trainAccuracy = [] 
testAccuracy = [] 
# Use the range function to create different depths options, ranging from 1 to 15, for the decision trees
depthOptions = range(1,16) 
for depth in depthOptions: 
    # Use a decision tree classifier that still measures the quality of a split using the entropy criteria.
    # Also, ensure that nodes with less than 6 training instances are not further split
  cltree = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=6, max_depth=depth) 
    # Decision tree training
  cltree.fit(X_train, y_train) 
    # Training error
  y_predTrain = cltree.predict(X_train) 
    # Testing error
  y_predTest = cltree.predict(X_test) 
    # Training accuracy
  trainAccuracy.append(accuracy_score(y_train, y_predTrain)) 
    # Testing accuracy
  testAccuracy.append(accuracy_score(y_test, y_predTest)) 

# Plot of training and test accuracies vs the tree depths (use different markers of different colors)
plt.plot(depthOptions, trainAccuracy, marker='o', label='Training Accuracy', color='blue') 
plt.plot(depthOptions, testAccuracy, marker='s', label='Test accuracy', color='red') 
plt.legend(['Training Accuracy', 'Test Accuracy']) # add a legend for the training accuracy and test accuracy 
plt.xlabel('Tree Depth')  # name the horizontal axis 'Tree Depth' 
plt.ylabel('Classifier Accuracy') # name the horizontal axis 'Classifier Accuracy' 
plt.show()

# Use sklearn's GridSearchCV function to perform an exhaustive search to find the best tree depth.
# First define the parameter to be optimized: the max depth of the tree
parameters = {'max_depth':depthOptions} 
# Grow a decision tree classifier by measuring the quality of a split using the entropy criteria. 
# Continue to ensure that nodes with less than 6 training instances are not further split.
clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=6)
clf.fit(X_train,y_train)  
grid_search = GridSearchCV(clf,parameters,cv=10,scoring='accuracy')
grid_search.fit(X_train, y_train)
tree_model = grid_search.best_estimator_
print("The maximum depth of the tree should be", clf.tree_.max_depth) 
print("Using GridSearchCV, The maximum depth of the tree should be", grid_search.best_params_['max_depth'])



# The best model is tree_model. Visualize that decision tree 
plt.figure(figsize=(15, 10))
tree.plot_tree(clf, fontsize=12)
plt.show()
