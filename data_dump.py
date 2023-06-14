import pymongo
import pandas as pd
import json
from store_sales.constant import *

# Provide the mongodb localhost url to connect python to mongodb.
client = pymongo.MongoClient(DATABASE_CLIENT_URL_KEY)




if __name__== "__main__":
    df = pd.read_csv(DATA_PATH)
    print(f"Rows and columns: {df.shape}")

    #Convert dataframe to json so that we can dump these record in mongo db
    df.reset_index(drop=True,inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    #insert converted json record to mongo db
    client[DATABASE_NAME_KEY][DATABASE_COLLECTION_NAME_KEY].insert_many(json_record)