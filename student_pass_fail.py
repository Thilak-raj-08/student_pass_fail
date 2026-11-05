import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("student_pass_fail.csv")

print("First five students:")
print(df.head())

print("Dataset shape:", df.shape)

df.info()

print("Missing values:")
print(df.isna().sum())

print("Duplicate rows:", df.duplicated().sum())

print("Statistical summary:")
print(df.describe())

print("Pass and Fail percentage:")
print((df["Pass"].value_counts(normalize=True) * 100).round(2))

df["Result"] = df["Pass"].map({0: "Fail", 1: "Pass"})

print("Median study hours by result:")
print(df.groupby("Result")["Study_Hours"].median())

plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Result",
    y="Study_Hours",
    order=["Fail", "Pass"]
)
plt.title("Study Hours by Pass/Fail")
plt.xlabel("Exam Result")
plt.ylabel("Study Hours")
plt.tight_layout()
plt.savefig("study_hours_vs_result.png", dpi=300, bbox_inches="tight")
plt.show()

print("Median attendance by result:")
print(df.groupby("Result")["Attendance"].median())

plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Result",
    y="Attendance",
    order=["Fail", "Pass"]
)
plt.title("Attendance by Pass/Fail")
plt.xlabel("Exam Result")
plt.ylabel("Attendance (%)")
plt.tight_layout()
plt.savefig("attendance_vs_result.png", dpi=300, bbox_inches="tight")
plt.show()


feature_columns = [
    "Study_Hours",
    "Attendance",
    "Assignment_Score",
    "Previous_Score"
]

X = df[feature_columns]
y = df["Pass"]

print("Features:", feature_columns)
print("Target:", y.name)
print("Classes:", sorted(y.unique().tolist()))
print("0 = Fail, 1 = Pass")

print("X shape:", X.shape)
print("y shape:", y.shape)


df["Rule_Prediction"] = (
    (df["Study_Hours"] >= 4) &
    (df["Attendance"] >= 75)
).astype(int)

print("Rule-based predictions:")
print(df[
    ["Study_Hours", "Attendance", "Pass", "Rule_Prediction"]
].head(10))

correct_predictions = (df["Pass"] == df["Rule_Prediction"]).sum()
rule_accuracy = correct_predictions / len(df) * 100

print("Correct predictions:", correct_predictions)
print("Total students:", len(df))
print("Rule accuracy (%):", round(rule_accuracy, 2))

print("Always predicting Pass accuracy (%):", round(df["Pass"].mean() * 100, 2))