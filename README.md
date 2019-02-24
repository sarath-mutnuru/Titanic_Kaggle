# Titanic_Kaggle

This is related to the titanic survival prediction problem originally being hosted in Kaggle.

This is a preliminary dealing of the problem and there is scope to improve which I plan to do soon.
The feature engineering done includes converting categorical data into numerical and  categorizing few numeric features.
The reason to categorize few features namely Age and Fare is by observing their effect to the label by using KDE plots(skipped here).
Some features like Cabin, Ticket are dropped as they don't seem to contribute more to the survival but we can employ some techniques to use these too which might increase the accuracy of the model.

Different Classifiers like Logistic Regression, Desicion Trees, Multi-Layer Perceptrons and ensemble techniques like Random Forests, Adaboost and Gradient Boosted Trees are employed to do the prediction.
Cross validation techniques like Grid Search is also used
The accuracy obtained is close to 0.78.

**Requirements**:

numpy,sklearn,pandas
