import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

df = pd.read_csv("StudentPerformanceFactors.csv")
df.drop(["Gender", "School_Type", "Sleep_Hours"], axis=1, inplace=True)

oe = OrdinalEncoder(categories=[["Low","Medium","High"], ["High School", "College", "Postgraduate"], ["Near", "Moderate", "Far"]])
df[["Parental_Involvement", "Access_to_Resources", "Motivation_Level", "Family_Income", "Teacher_Quality", "Parental_Education_Level", "Distance_from_Home"]] = oe.fit_transform(df[["Parental_Involvement", "Access_to_Resources", "Motivation_Level", "Family_Income", "Teacher_Quality", "Parental_Education_Level", "Distance_from_Home"]])

ohe = OneHotEncoder(sparse_output=False, drop="first")
encoded = ohe.fit_transform(["Extracurricular_Activities", "Internet_Access", "Peer_Influence", "Learning_Disabilities"])
encoded = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(""))
