### Business Problem

# There is a dataset from E-Commerce Wholesale Company named Online_Retail2 ,
#We want to create a recommender system using Apriori algoritm also called Market Basket Analysis .
#Our main goal is to offer items to our customers based on  item association.

## Importing Libraries

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

### Reading Data


df_ = pd.read_excel('online_retail_II.xlsx',sheet_name="Year 2010-2011")

df = df_.copy()


### Data Understanding
df.head()

df.shape

df.info()

df.describe()

### Data Prepration

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    return dataframe

df = retail_data_prep(df)


### Capping Outliers for Price and Quantity

def capping_outlier(df, variable, upper_limit=0.75, lower_limit=0.25):
    q1 = df[variable].quantile(lower_limit)
    q3 = df[variable].quantile(upper_limit)

    IQR = q3 - q1

    upper_limit = q3 + 1.5 * IQR
    lower_limit = q1 - 1.5 * IQR

    df.loc[df[variable] > upper_limit, variable] = upper_limit
    df.loc[df[variable] < lower_limit, variable] = lower_limit

capping_outlier(df,'Quantity',upper_limit=0.99,lower_limit=0.01)
capping_outlier(df,'Price',upper_limit=0.99,lower_limit=0.01)

df.describe()

# We will use only the observation from Germany

df = df[df.Country == 'Germany']


df.head()

### Creating Data Frame for ARL

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


df_invoice_description = create_invoice_product_df(df)

df_invoice_stock_code = create_invoice_product_df(df,id=True)

df_invoice_stock_code.head()

df_invoice_description.head()

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

### Checking Name of the Items

check_id(df,21987)

check_id(df,23235)

check_id(df,22747)


### Association Rule Learning


# Now We are rady to implement our model to new dataframe


def get_rules(apriori_df, min_support=0.01):
    frequent_itemsets = apriori(apriori_df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=min_support)
    return rules


rules = get_rules(df_invoice_stock_code,min_support=0.01)

rules.head()

rules.sort_values("support", ascending=False).head()

sorted_rules = rules.sort_values("lift", ascending=False)

sorted_rules.head()


### Recommendation System


def arl_recommender(rules_df, product_id, rec_count=1):

    sorted_rules = rules_df.sort_values("lift", ascending=False)

    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})

    return recommendation_list[:rec_count]

### Example of Recommendation


arl_recommender(rules, 22492, 5)


recommended = arl_recommender(rules,21988,10)


for i in recommended:
    check_id(df,i)






