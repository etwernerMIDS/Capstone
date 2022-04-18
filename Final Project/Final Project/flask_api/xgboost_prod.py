import pandas as pd
import sys

def get_x_and_y(df, fill_method):
    column_list = ['Price_USD','Price_increase','Category_Art', 'Category_Collectible','Category_Games', 'Category_Metaverse', 'Category_Other',
                                'Category_Utility','Collection_cleaned','nft_sales_count', 'First_item_Unique_id_collection','GoogleTrends', 'BTC_price', 'ETH_price',
                               'MANA_price', 'WAX_price','GoogleTrends_7d_rolling_avg', 'GoogleTrends_14d_rolling_avg','GoogleTrends_30d_rolling_avg',
                               'GoogleTrends_60d_rolling_avg','GoogleTrends_90d_rolling_avg', 'BTC_price_7d_rolling_avg','BTC_price_14d_rolling_avg','BTC_price_30d_rolling_avg',
                               'BTC_price_60d_rolling_avg', 'BTC_price_90d_rolling_avg','ETH_price_7d_rolling_avg', 'ETH_price_14d_rolling_avg','ETH_price_30d_rolling_avg',
                               'ETH_price_60d_rolling_avg','ETH_price_90d_rolling_avg', 'MANA_price_7d_rolling_avg','MANA_price_14d_rolling_avg', 'MANA_price_30d_rolling_avg',
                               'MANA_price_60d_rolling_avg', 'MANA_price_90d_rolling_avg','WAX_price_7d_rolling_avg', 'WAX_price_14d_rolling_avg','WAX_price_30d_rolling_avg',
                               'WAX_price_60d_rolling_avg','WAX_price_90d_rolling_avg', 'Ggl_trends_collection','Ggl_trends_collection_7d_rolling_avg',
                               'Ggl_trends_collection_14d_rolling_avg','Ggl_trends_collection_30d_rolling_avg','Ggl_trends_collection_60d_rolling_avg',
                               'Ggl_trends_collection_90d_rolling_avg','Price_USD_median_7d', 'Price_USD_max_7d', 'Price_USD_median_14d','Price_USD_max_14d',
                               'Price_USD_median_30d', 'Price_USD_max_30d','Price_USD_median_60d', 'Price_USD_max_60d','Price_USD_median_90d', 'Price_USD_max_90d',
                               'Price_USD_median_180d','Price_USD_max_180d', 'Price_USD_median_365d', 'Price_USD_max_365d','Price_USD_median_730d','Price_USD_max_730d',
                               'Price_Crypto_median_7d','Price_Crypto_max_7d', 'Price_Crypto_median_14d','Price_Crypto_max_14d', 'Price_Crypto_median_30d',
                               'Price_Crypto_max_30d', 'Price_Crypto_median_60d','Price_Crypto_max_60d', 'Price_Crypto_median_90d','Price_Crypto_max_90d',
                               'Price_Crypto_median_180d','Price_Crypto_max_180d', 'Price_Crypto_median_365d','Price_Crypto_max_365d', 'Price_Crypto_median_730d',
                               'Price_Crypto_max_730d', 'Price_USD_collection_cum_median','Price_USD_collection_cum_max', 'Price_Crypto_collection_cum_median',
                               'Price_Crypto_collection_cum_max','p_resale', 'pca_1','pca_2', 'pca_3', 'pca_4', 'pca_5', 'trans_count_seller','deg_centrality_seller',
                               'pgrank_seller', 'trans_count_buyer', 'deg_centrality_buyer', 'pgrank_buyer']

    drop_list = ['Price_USD','Price_increase', 'Collection_cleaned', 'Category_Utility']
    fillna_median_list = ['p_resale', 'pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5']

    fillna_zero_list = ['Ggl_trends_collection', 'Ggl_trends_collection_7d_rolling_avg', 'Ggl_trends_collection_14d_rolling_avg', 'Ggl_trends_collection_30d_rolling_avg',
                    'Ggl_trends_collection_60d_rolling_avg', 'Ggl_trends_collection_90d_rolling_avg']

    temp_df = df[column_list]
    if fill_method == 0:
        temp_df2 = temp_df.fillna(0)
    else:
        temp_df2 = temp_df.dropna(subset=['Price_USD'])
        for item in fillna_median_list:
            temp_df2[item] = temp_df2.groupby(['Collection_cleaned'], sort=False)[item].apply(lambda x: x.fillna(x.median()))
        for col in fillna_zero_list:
            temp_df2[col] = temp_df2[col].fillna(0)
        temp_df2 = temp_df2.bfill().ffill()
    print(temp_df2.columns, file=sys.stderr)
    temp_df3 = temp_df2
    print(temp_df3.columns, file=sys.stderr)
    temp_X = temp_df3.drop(columns=drop_list)
    temp_y = temp_df3['Price_increase']
    temp_indices = temp_X.index
    return temp_X, temp_y, temp_indices
