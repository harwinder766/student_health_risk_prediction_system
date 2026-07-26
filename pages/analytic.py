import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
# Helper functions 
def apply_range_filter(df, column, selected_range):
    col_min = float(df[column].min())
    col_max = float(df[column].max())

    selected_min, selected_max = selected_range

    if selected_min == col_min and selected_max == col_max:
        return df

    # User actually changed slider
    return df[df[column].between(selected_min, selected_max)]

def lifestyle_health_chart(df, feature, title):
    # Count each health condition within each category
    plot_df = (
        df.groupby([feature, "health_condition"],observed=True)
        .size()
        .reset_index(name="count")
    )
    # Convert counts to percentages within each feature category
    plot_df["percentage"] = (
        plot_df["count"]
        / plot_df.groupby(feature)["count"].transform("sum")
        * 100
    )
    fig = px.bar(plot_df,x=feature,y="percentage",color="health_condition",barmode="group",text_auto= '.1f',title=title,
        labels={
            feature: feature.replace("_", " ").title(),
            "percentage": "Students (%)",
            "health_condition": "Health Condition"
        },
        hover_data={
            "count": True,
            "percentage": ":.1f"
        }
    )

    fig.update_layout(yaxis_title="Students (%)",xaxis_title=None,legend_title="Health Condition")

    return fig

def numerical_health_chart(df, feature, title, y_label):
    fig = px.box(
        df,
        x="health_condition",
        y=feature,
        color="health_condition",
        title=title,
        labels={
            "health_condition": "Health Condition",
            feature: y_label
        },
        category_orders={
            "health_condition": ["fit", "at-risk", "unhealthy"]
        }
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title="Health Condition",
        yaxis_title=y_label
    )

    return fig

def relationship_plot(df, x, y, numerical_cols, categorical_cols):

    if len(df) > 20000:
        plot_data = df.sample(20000,random_state=42)
        
    if x in numerical_cols and y in numerical_cols:

        fig = px.scatter(plot_data,x=x,y=y,color="health_condition",opacity=0.5,
            title=f"{x.replace('_', ' ').title()} vs "
                  f"{y.replace('_', ' ').title()}"
        )
        fig.update_layout(
            xaxis_title=x.replace("_", " ").title(), yaxis_title = y.replace("_"," ").title(),
            legend_title_text="Health Condition"if "health_condition" not in [x, y] else None
            )
        return fig
    elif x in categorical_cols and y in numerical_cols:
        fig = px.box(df,x=x, y=y,color="health_condition",
            title=f"{y.replace('_', ' ').title()} by "
                  f"{x.replace('_', ' ').title()}"
        )
        fig.update_layout(
                    xaxis_title=x.replace("_", " ").title(), yaxis_title = y.replace("_"," ").title(),
                    legend_title_text="Health Condition"if "health_condition" not in [x, y] else None
                    )
        return fig
    elif x in numerical_cols and y in categorical_cols:

        fig = px.box(df, x=y, y=x, color="health_condition",
            title=f"{x.replace('_', ' ').title()} by "
                  f"{y.replace('_', ' ').title()}"
        )
        fig.update_layout(
                    xaxis_title=y.replace("_", " ").title(), yaxis_title = x.replace("_"," ").title(),
                    legend_title_text="Health Condition"if "health_condition" not in [x, y] else None
                    )
        return fig
    else:
        plot_df = (
            df.groupby([x, y], observed=True)
              .size()
              .reset_index(name="count")
        )
        plot_df["percentage"] = (
            plot_df["count"]
            / plot_df.groupby(x)["count"].transform("sum")
            * 100
        )
        fig = px.bar(plot_df, x=x, y="percentage", color=y, barmode="group", text_auto= '.1f',
            title=f"{x.replace('_', ' ').title()} vs "
                  f"{y.replace('_', ' ').title()}",
            labels={"percentage": "Students (%)"}
        )
        fig.update_layout(
                    xaxis_title=x.replace("_", " ").title(), yaxis_title = y.replace("_"," ").title(),
                    legend_title_text="Health Condition"if "health_condition" not in [x, y] else None
                    )
        return fig

def lifestyle_insight(df, feature, target="health_condition"):
    # Calculate percentage of each health condition
    rates = (
        pd.crosstab(
            df[feature],
            df[target],
            normalize="index"
        ) * 100
    )

    # Make sure unhealthy exists
    if "unhealthy" not in rates.columns:
        return "Not enough data to calculate unhealthy rates."

    unhealthy_rates = rates["unhealthy"]

    highest_group = unhealthy_rates.idxmax()
    highest_rate = unhealthy_rates.max()

    lowest_group = unhealthy_rates.idxmin()
    lowest_rate = unhealthy_rates.min()

    difference = highest_rate - lowest_rate

    return (
        f"Students in the **{highest_group}** group have the highest "
        f"unhealthy rate at **{highest_rate:.1f}%**, compared with "
        f"**{lowest_rate:.1f}%** for the **{lowest_group}** group "
        f"(**{difference:.1f}% difference**)."
    )

def reset_filters():
    # Categorical filters
    st.session_state.health_filter = []
    st.session_state.gender_filter = []
    st.session_state.diet_filter = []
    st.session_state.stress_level_filter = []
    st.session_state.sleep_quality_filter = []
    st.session_state.physical_activity_level_filter = []
    st.session_state.smoking_alcohol_filter = []

    # Numerical filters
    st.session_state.sleep_duration = (float(df["sleep_duration"].min()),float(df["sleep_duration"].max()))
    st.session_state.heart_rate = (float(df["heart_rate"].min()),float(df["heart_rate"].max()))
    st.session_state.bmi = (float(df["bmi"].min()),float(df["bmi"].max()))
    st.session_state.exercise_duration = (float(df["exercise_duration"].min()),float(df["exercise_duration"].max()))
    st.session_state.water_intake = (float(df["water_intake"].min()),float(df["water_intake"].max()))
    st.session_state.calorie_expenditure = (int(df["calorie_expenditure"].min()),int(df["calorie_expenditure"].max()))
    st.session_state.step_count = (int(df["step_count"].min()),int(df["step_count"].max()))

df = pd.read_parquet(r'C:\Users\harwi\Downloads\student_health_risk_prediction_system\data\analytics_data.parquet')

st.title("📊 Student Health Analytics")

st.caption(
    "Explore student health patterns, lifestyle factors, and relationships "
    "using the filters in the sidebar."
)

st.divider()

st.sidebar.header("Filters")

st.sidebar.button("🔄 Reset Filters", on_click=reset_filters)

health_filter = st.sidebar.multiselect(
    "Health Condition",
    options=df["health_condition"].unique(),
    key = 'health_filter'
)
gender_filter = st.sidebar.multiselect(
    "Gender",
    options=df["gender"].unique(),
    key = 'gender_filter'
)
diet_filter = st.sidebar.multiselect(
    "Diet Type",
    options=df["diet_type"].unique(),
    key = 'diet_filter'
)
stress_level_filter = st.sidebar.multiselect(
    "Stress Level",
    options=df["stress_level"].unique(),
    key = 'stress_level_filter'
)
sleep_quality_filter = st.sidebar.multiselect(
    "Sleep Quality",
    options=df["sleep_quality"].unique(),
    key = 'sleep_quality_filter'
)
physical_activity_level_filter = st.sidebar.multiselect(
    "Physical Activity Level",
    options=df["physical_activity_level"].unique(),
    key = 'physical_activity_level_filter'
)
smoking_alcohol_filter = st.sidebar.multiselect(
    "Smoking Alcohol",
    options=df["smoking_alcohol"].unique(),
    key = 'smoking_alcohol_filter'
)
sleep_duration = st.sidebar.slider(
    "Sleep Duration",
    float(df["sleep_duration"].min()),
    float(df["sleep_duration"].max()),
    (
        float(df["sleep_duration"].min()),
        float(df["sleep_duration"].max())
    ),
    key  = 'sleep_duration'
)
heart_rate = st.sidebar.slider(
    "Heart Rate",
    float(df["heart_rate"].min()),
    float(df["heart_rate"].max()),
    (
        float(df["heart_rate"].min()),
        float(df["heart_rate"].max())
    ),
    key = 'heart_rate'
)
bmi = st.sidebar.slider(
    "BMI",
    float(df["bmi"].min()),
    float(df["bmi"].max()),
    (
        float(df["bmi"].min()),
        float(df["bmi"].max())
    ),
    key = 'bmi'
)
calorie_expenditure = st.sidebar.slider(
    "Calorie Expenditure",
    int(df["calorie_expenditure"].min()),
    int(df["calorie_expenditure"].max()),
    (
        int(df["calorie_expenditure"].min()),
        int(df["calorie_expenditure"].max())
    ),
    key = 'calorie_expenditure'
)
step_count = st.sidebar.slider(
    "Step Count",
    int(df["step_count"].min()),
    int(df["step_count"].max()),
    (
        int(df["step_count"].min()),
        int(df["step_count"].max())
    ),
    key = 'step_count'
)
exercise_duration = st.sidebar.slider(
    "Exercise Duration (in minutes)",
    float(df["exercise_duration"].min()),
    float(df["exercise_duration"].max()),
    (
        float(df["exercise_duration"].min()),
        float(df["exercise_duration"].max())
    ),
    key = 'exercise_duration'
)
water_intake = st.sidebar.slider(
    "Water Intake",
    float(df["water_intake"].min()),
    float(df["water_intake"].max()),
    (
        float(df["water_intake"].min()),
        float(df["water_intake"].max())
    ),
    key = 'water_intake'
)

filtered_df = df.copy()

# Filtering categorical features
categorical_filters = {
    "health_condition": health_filter,
    "gender": gender_filter,
    "diet_type": diet_filter,
    "stress_level": stress_level_filter,
    "sleep_quality": sleep_quality_filter,
    "physical_activity_level": physical_activity_level_filter,
    "smoking_alcohol": smoking_alcohol_filter,
}

for col, selected_values in categorical_filters.items():
    # Empty selection = no filtering
    if selected_values:
        filtered_df = filtered_df[
            filtered_df[col].isin(selected_values)
        ]

# Filtering numerical features
filtered_df = apply_range_filter(filtered_df, "sleep_duration", sleep_duration)
filtered_df = apply_range_filter(filtered_df, "heart_rate", heart_rate)
filtered_df = apply_range_filter(filtered_df, "bmi", bmi)
filtered_df = apply_range_filter(filtered_df, "calorie_expenditure", calorie_expenditure)
filtered_df = apply_range_filter(filtered_df, "step_count", step_count)
filtered_df = apply_range_filter(filtered_df, "exercise_duration", exercise_duration)
filtered_df = apply_range_filter(filtered_df, "water_intake", water_intake)

if filtered_df.empty:
    st.warning("No students match the selected filters.")
    st.stop()

if len(filtered_df) == len(df):
    st.caption(f"Showing all {len(df):,} students")
else:
    st.caption(
        f"Showing {len(filtered_df):,} of {len(df):,} students"
    )

# Overview
st.subheader("📌 Overview")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Students", f"{len(filtered_df):,}")
col2.metric("Avg Sleep", f"{filtered_df['sleep_duration'].mean():.1f} hrs")
col3.metric("Avg BMI", f"{filtered_df['bmi'].mean():.1f}")
col4.metric("Avg Daily Steps", f"{filtered_df['step_count'].mean():,.0f}")
col5.metric("Avg Heart Rate", f"{filtered_df['heart_rate'].mean():.0f} bpm")

st.header("🏥 Health Condition Overview")

condition_counts = (
    filtered_df["health_condition"]
    .value_counts()
    .rename_axis("Health Condition")
    .reset_index(name="Students")
)

condition_counts["Percentage"] = (
    condition_counts["Students"]
    / condition_counts["Students"].sum()
    * 100
)

fig = px.bar(
    condition_counts,
    x="Health Condition",
    y="Students",
    color="Health Condition",
    text="Percentage",
    title="Distribution of Student Health Conditions"
)
fig.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)
fig.update_layout(
    showlegend=False,
    xaxis_title="Health Condition",
    yaxis_title="Number of Students"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("🌱 Lifestyle & Health")

stress_tab, sleep_tab, activity_tab, diet_tab, gender_tab,smoking_alcohol_tab= st.tabs([
    "Stress",
    "Sleep Quality",
    "Physical Activity",
    "Diet",
    "Gender",
    "Smoking Alcohal"
])
with stress_tab:
    st.plotly_chart(
        lifestyle_health_chart(
            filtered_df,
            "stress_level",
            "Stress Level vs Health Condition"
        ),
        use_container_width=True
    )
    st.info(
    "💡 **Key Insight:** " +
    lifestyle_insight(
        filtered_df,
        "stress_level"
    )
)
with sleep_tab:
    st.plotly_chart(
        lifestyle_health_chart(
            filtered_df,
            "sleep_quality",
            "Sleep Quality vs Health Condition"
        ),
        use_container_width=True
    )

    st.info(
    "💡 **Key Insight:** " +
    lifestyle_insight(
        filtered_df,
        "sleep_quality"
    )
)
with activity_tab:
    st.plotly_chart(
        lifestyle_health_chart(
            filtered_df,
            "physical_activity_level",
            "Physical Activity vs Health Condition"
        ),
        use_container_width=True
    )
    st.info(
    "💡 **Key Insight:** " +
    lifestyle_insight(
        filtered_df,
        "physical_activity_level"
    )
)
with diet_tab:
    st.plotly_chart(
        lifestyle_health_chart(
            filtered_df,
            "diet_type",
            "Diet vs Health Condition"
        ),
        use_container_width=True
    )
    st.info(
    "💡 **Key Insight:** " +
    lifestyle_insight(
        filtered_df,
        "diet_type"
    )
)
with gender_tab:
    st.plotly_chart(
        lifestyle_health_chart(
            filtered_df,
            "gender",
            "Gender vs Health Condition"
        ),
        use_container_width=True
    )
    st.info(
    "💡 **Key Insight:** " +
    lifestyle_insight(
        filtered_df,
        "gender"
    )
)
with smoking_alcohol_tab:
    st.plotly_chart(
        lifestyle_health_chart(
            filtered_df,
            "smoking_alcohol",
            "Smoking Alcohol vs Health Condition"
        ),
        use_container_width=True
    )
    st.info(
    "💡 **Key Insight:** " +
    lifestyle_insight(
        filtered_df,
        "smoking_alcohol"
    )
)

st.subheader("📊 Numerical Health Analysis")

numerical_feature = st.selectbox(
    "Select health metric",
    ["Sleep Duration","BMI","Daily Step Count","Heart Rate","Exercise Duration","Water Intake","Calorie Expenditure"]
)
feature_config = {
    "Sleep Duration": {
        "column": "sleep_duration",
        "label": "Sleep Duration (hours)"
    },
    "BMI": {
        "column": "bmi",
        "label": "BMI"
    },
    "Daily Step Count": {
        "column": "step_count",
        "label": "Daily Steps"
    },
    "Heart Rate": {
            "column": "heart_rate",
            "label": "Heart Rate"
    },
    "Calorie Expenditure": {
            "column": "calorie_expenditure",
            "label": "Calorie Expenditure"
    },
    "Exercise Duration": {
                "column": "exercise_duration",
                "label": "Exercise Duration (minutes)"
    },
    "Water Intake": {
                "column": "water_intake",
                "label": "Water Intake (litres)"
     },
}
config = feature_config[numerical_feature]
fig = numerical_health_chart(
    filtered_df,
    config["column"],
    f"{numerical_feature} by Health Condition",
    config["label"]
)
st.plotly_chart(
    fig,
    use_container_width=True
)


st.subheader("🔎 Relationship Explorer")

numerical_cols = ["sleep_duration", "bmi", "step_count", "heart_rate", "calorie_expenditure", "exercise_duration", "water_intake"]
categorical_cols = ["health_condition", "gender", "diet_type", "stress_level", "sleep_quality", "physical_activity_level", "smoking_alcohol"]
available_cols = categorical_cols + numerical_cols

col1, col2 = st.columns(2)
with col1:
    x_variable = st.selectbox("X Variable", available_cols, index=available_cols.index("stress_level"))

with col2:
    y_variable = st.selectbox("Y Variable", available_cols, index=available_cols.index("health_condition"))

if x_variable == y_variable:
    st.info("Please select two different variables.")
else:
    fig = relationship_plot(filtered_df, x_variable, y_variable, numerical_cols, categorical_cols)
    st.plotly_chart(fig,use_container_width=True)