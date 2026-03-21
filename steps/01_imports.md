# 📦 Step 1: Import Libraries

We first import the required libraries.

## Code

```python
import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

## Explaination
- pandas (pd) → work with table data  
- load_iris → get example dataset  
- seaborn (sns) → make nice graphs  
- matplotlib.pyplot (plt) → draw charts  
- KMeans → group similar data  
- silhouette_score → check grouping quality  