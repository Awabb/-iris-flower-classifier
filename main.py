
from sklearn.datasets import load_iris
import pandas as pd

# Load the flower data
iris = load_iris()

# Turn the data into a table
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the correct flower type
data["flower_type"] = iris.target

# Show the first 5 flowers
print(data.head())
