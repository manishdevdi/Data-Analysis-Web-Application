import streamlit as st  # For creating web apps
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs ,plt.tight_layout()
import seaborn as sns  # For statistical data visualization
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For data preprocessing
from sklearn.decomposition import PCA  # For Principal Component Analysis
from sklearn.cluster import KMeans  # For KMeans clustering
from io import BytesIO  # For handling bytes-like objects
import base64  # For base64 encoding/decoding

# Custom CSS for Streamlit elements
custom_css = """
<style>
"container": {"padding": "5!important","background-color":'black'}
/* Sidebar styling */
.sidebar .sidebar-content {
    background-color: #0B3C5D; /* Navy blue */
    color: #FFFFFF; /* White */
    font-size: 18px;
}

/* Main content styling */
.css-1l02zno, .css-14zkor0 {
    color: #FFFFFF; /* White */
    font-size: 18px;
}

/* Header and button styling */
.st-cq {
    background-color: #17BEBB; /* Cyan */
    color: #FFFFFF; /* White */
    font-size: 24px;
    font-weight: bold;
    border-radius: 10px;
    padding: 8px 12px;
    margin-bottom: 20px;
}

/* Background pattern */
body {
    background-color: #000000; /* Black */
}

/* Footer styling */
.stFooter {
    font-size: 14px;
    color: #FFFFFF; /* White */
    text-align: center;
    margin-top: 20px;
}
</style>
"""

# Define the main function
def main():
    # Set the title and configuration of the Streamlit page
    st.set_page_config(page_title="Manish Devdi with Data", layout="wide")

    # Display custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)

    # Sidebar with activities selection
    activities = ["Home", "Exploratory Data Analysis", "Data Visualization", "Data Preprocessing", "Machine Learning"]
    choice = st.sidebar.radio("Select Activity", activities)

    # Main content based on the selected activity
    if choice == "Home":
        st.header("Welcome to Data Analyzing Web Application with Manish Devdi")
        st.write("""
        This application offers a variety of tools to help you explore, visualize, preprocess, and analyze your data. 
        You can perform Exploratory Data Analysis, visualize your data, preprocess it for machine learning, and even apply 
        clustering algorithms. Please select an activity from the sidebar to get started.
        """)
    elif choice == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis ")

        # Upload dataset
        uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "txt", "xlsx"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())

                # Data preprocessing
                st.subheader("Data Preprocessing")
                st.write("Handling missing values:")
                st.write(df.isnull().sum())

                # Handling missing values
                if st.checkbox("Handle Missing Values"):
                    df = handle_missing_values(df)
                    st.write("Missing values handled. New data:")
                    st.dataframe(df.head())

                # Show dataset shape
                if st.checkbox("Show Dataset Shape"):
                    st.write(df.shape)

                # Show columns
                if st.checkbox("Show Columns"):
                    all_columns = df.columns.tolist()
                    st.write(all_columns)

                # Show summary statistics
                if st.checkbox("Show Summary Statistics"):
                    st.write(df.describe())

                # Show selected columns
                if st.checkbox("Show Selected Columns"):
                    selected_columns = st.multiselect("Select Columns", all_columns)
                    if selected_columns:
                        selected_df = df[selected_columns]
                        st.dataframe(selected_df)

                # Show value counts
                if st.checkbox("Show Value Counts"):
                    column = st.selectbox("Select column", df.columns)
                    st.write(df[column].value_counts())

                # Plot correlation matrix
                if st.checkbox("Plot Correlation Matrix"):
                    try:
                        numeric_df = df.select_dtypes(include=np.number)  # Select only numeric columns
                        fig, ax = plt.subplots()
                        ax = sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"An error occurred while plotting the correlation matrix: {e}")

            except Exception as e:
                st.error(f"An error occurred: {e}")

    elif choice == "Data Visualization":
        st.header("Data Visualization ")
        # Upload dataset
        uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "txt", "xlsx"])

        if uploaded_file is not None:
            try:
                # Read the uploaded file
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())

                # Show value counts as bar plot
                if st.checkbox("Show Value Counts (Bar Plot)"):
                    column = st.selectbox("Select column for value counts", df.columns)
                    st.bar_chart(df[column].value_counts())

                # Customizable plot
                st.subheader("Customizable Plot")
                plot_type = st.selectbox("Select plot type", ["area", "bar", "line", "scatter"])
                selected_columns = st.multiselect("Select columns to plot", df.columns)

                if st.button("Generate Plot"):
                    if plot_type and selected_columns:
                        if len(selected_columns) < 2:
                            st.error("Please select at least two columns for plotting.")
                        else:
                            x_col = selected_columns[0]
                            y_col = selected_columns[1]

                            fig, ax = plt.subplots()
                            if plot_type == "area":
                                ax.fill_between(df[x_col], df[y_col])
                            elif plot_type == "bar":
                                ax.bar(df[x_col], df[y_col])
                            elif plot_type == "line":
                                ax.plot(df[x_col], df[y_col])
                            elif plot_type == "scatter":
                                ax.scatter(df[x_col], df[y_col])

                            st.pyplot(fig)
            except Exception as e:
                st.error(f"An error occurred: {e}")

    elif choice == "Data Preprocessing":
        st.header("Data Preprocessing ")

        # Upload dataset
        uploaded_file = st.file_uploader("Upload a dataset for preprocessing", type=["csv", "txt"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())

                # Data preprocessing
                st.subheader("Data Preprocessing")
                st.write("Handling missing values:")
                st.write(df.isnull().sum())

                # Handling missing values
                if st.checkbox("Handle Missing Values"):
                    df = handle_missing_values(df)
                    st.write("Missing values handled. New data:")
                    st.dataframe(df.head())

                # Encoding categorical data if present
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                if categorical_cols:
                    st.write("Encoding categorical data:")
                    df_encoded = encode_categorical_data(df, categorical_cols)
                    st.write("Encoded data:")
                    st.dataframe(df_encoded.head())

                # Scaling data if numerical columns exist
                numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                if numerical_cols:
                    st.write("Scaling numerical data:")
                    df_scaled = scale_numerical_data(df[numerical_cols])
                    st.write("Scaled data:")
                    st.dataframe(df_scaled.head())

                # Download link for preprocessed data
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
                href = f'<a href="data:file/csv;base64,{b64}" download="preprocessed_data.csv">Download Preprocessed Data</a>'
                st.markdown(href, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")

    elif choice == "Machine Learning":
        st.header("Machine Learning ")

        # Upload dataset
        uploaded_file = st.file_uploader("Upload a dataset for clustering", type=["csv", "txt"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())

                # Preprocessing for clustering (label encoding and PCA)
                le = LabelEncoder()
                df_encoded = df.apply(le.fit_transform)

                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(df_encoded)
                df_pca = pd.DataFrame(data=principal_components, columns=["Principal Component 1", "Principal Component 2"])

                # KMeans clustering
                k = st.slider("Select number of clusters", min_value=2, max_value=10)
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(df_encoded)
                df["Cluster"] = kmeans.labels_

                # Visualize clusters
                st.subheader("Cluster Visualization")
                fig, ax = plt.subplots()
                scatter = ax.scatter(df_pca["Principal Component 1"], df_pca["Principal Component 2"], c=df["Cluster"], cmap="viridis")
                legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
                ax.add_artist(legend)
                st.pyplot(fig)

                # Show cluster details
                st.subheader("Cluster Details")
                # Select numeric columns for aggregation
                numeric_df = df.select_dtypes(include=np.number)
                st.write(numeric_df.groupby("Cluster").mean())

                # Download link for clustering results
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
                href = f'<a href="data:file/csv;base64,{b64}" download="clustering_results.csv">Download Clustering Results</a>'
                st.markdown(href, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Footer
    st.markdown('<p class="stFooter">Developed by Manish Devdi</p>', unsafe_allow_html=True)

def handle_missing_values(df):
    """Handle missing values in the DataFrame."""
    # Replace missing numerical values with mean
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for col in numerical_cols:
        df[col].fillna(df[col].mean(), inplace=True)
    
    # Replace missing categorical values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def encode_categorical_data(df, categorical_cols):
    """Encode categorical data using LabelEncoder."""
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    return df

def scale_numerical_data(df):
    """Scale numerical data using StandardScaler."""
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    return df_scaled

# Run the main function
if __name__ == '__main__':
    main()


# streamlit run app.py
