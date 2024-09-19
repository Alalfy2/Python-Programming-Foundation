import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class DataPrepKit:
    @staticmethod
    def read_data(file_path, file_type):
        if file_type == 'csv':
            return pd.read_csv(file_path)
        elif file_type == 'excel':
            return pd.read_excel(file_path)
        elif file_type == 'json':
            return pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file type. Supported types are 'csv', 'excel', and 'json'.")

    @staticmethod
    def summarize_data(df):
        summary = {}
        summary['shape'] = df.shape
        summary['columns'] = df.columns.tolist()
        summary['data_types'] = df.dtypes.to_dict()
        summary['missing_values'] = df.isnull().sum().to_dict()
        summary['numerical_summary'] = df.describe().to_dict()
        summary['categorical_summary'] = df.describe(include=['object', 'category']).to_dict()
        
        for col in df.select_dtypes(include=['object', 'category']):
            summary[f'{col}_most_frequent'] = df[col].mode().iloc[0] if not df[col].mode().empty else None
        
        return summary

    @staticmethod
    def handle_missing_values(df, strategy='remove'):
        df_clean = df.copy()
        
        if strategy == 'remove':
            return df_clean.dropna()
        elif strategy in ['mean', 'median', 'most_frequent']:
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            categorical_columns = df_clean.select_dtypes(exclude=[np.number]).columns
            
            if numeric_columns.any():
                imputer_numeric = SimpleImputer(strategy=strategy)
                df_clean[numeric_columns] = imputer_numeric.fit_transform(df_clean[numeric_columns])
            
            if categorical_columns.any():
                imputer_categorical = SimpleImputer(strategy='most_frequent')
                df_clean[categorical_columns] = imputer_categorical.fit_transform(df_clean[categorical_columns])
            
            return df_clean
        else:
            raise ValueError("Unsupported strategy. Supported strategies are 'remove', 'mean', 'median', and 'most_frequent'.")

    @staticmethod
    def encode_categorical(df, columns, method='label'):
        df_encoded = df.copy()
        
        if method == 'label':
            le = LabelEncoder()
            for col in columns:
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        elif method == 'onehot':
            ohe = OneHotEncoder(sparse=False)
            encoded_cols = ohe.fit_transform(df_encoded[columns])
            encoded_df = pd.DataFrame(encoded_cols, columns=ohe.get_feature_names_out(columns))
            df_encoded = pd.concat([df_encoded.drop(columns, axis=1), encoded_df], axis=1)
        else:
            raise ValueError("Unsupported method. Supported methods are 'label' and 'onehot'.")
        
        return df_encoded
    
    @staticmethod
    def plot_data(df):
        # Plot distributions of numerical columns
        df_numeric = df.select_dtypes(include=[np.number])
        for column in df_numeric.columns:
            plt.figure(figsize=(6, 6))  # Create a new figure for each plot
            sns.histplot(df[column].dropna(), kde=True)
            plt.title(f'Distribution of {column}')
            plt.show()

        # Plot count plots for categorical columns
        df_categorical = df.select_dtypes(include=['object', 'category'])
        for column in df_categorical.columns:
            plt.figure(figsize=(10, 6))  # Create a new figure for each plot
            sns.countplot(x=column, data=df)
            plt.title(f'Count Plot of {column}')
            plt.xticks(rotation=45)
            plt.show()

def process_real_data(file_path):
    # قراءة البيانات من ملف CSV
    print("قراءة البيانات...")
    df = DataPrepKit.read_data(file_path, 'csv')
    
    # عرض معلومات أساسية عن البيانات
    print("\nمعلومات أساسية عن البيانات:")
    print(df.info())
    
    # تلخيص البيانات
    print("\nملخص البيانات:")
    summary = DataPrepKit.summarize_data(df)
    for key, value in summary.items():
        print(f"{key}:")
        print(value)
        print()
    
    # معالجة القيم المفقودة
    print("\nمعالجة القيم المفقودة...")
    df_clean = DataPrepKit.handle_missing_values(df, strategy='mean')
    print("عدد القيم المفقودة بعد المعالجة:")
    print(df_clean.isnull().sum())
    
    # ترميز المتغيرات الفئوية
    categorical_columns = df_clean.select_dtypes(include=['object']).columns
    print(f"\nترميز المتغيرات الفئوية: {', '.join(categorical_columns)}")
    df_encoded = DataPrepKit.encode_categorical(df_clean, categorical_columns, method='onehot')
    
    print("\nأبعاد البيانات بعد الترميز:")
    print(df_encoded.shape)
    
    print("\nالأعمدة بعد الترميز:")
    print(df_encoded.columns.tolist())
    
    # رسم الأشكال البيانية لتوضيح البيانات
    print("\nعرض الأشكال البيانية:")
    DataPrepKit.plot_data(df_encoded)
    
    # حفظ البيانات المعالجة
    output_file = 'processed_data.csv'
    df_encoded.to_csv(output_file, index=False)
    print(f"\nتم حفظ البيانات المعالجة في {output_file}")

if __name__ == "__main__":
    # استبدل 'path/to/your/train.csv' بالمسار الفعلي لملف البيانات الخاص بك
    process_real_data('/Users/mohamedalalfy/Downloads/titanic/train.csv')