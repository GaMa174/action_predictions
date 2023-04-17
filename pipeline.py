import dill
import pandas as pd

from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


def filter_data(df):
    colum_to_drop = ['device_model',
                     'device_os',
                     'utm_keyword',
                     ]
    return df.drop(colum_to_drop, axis=1)


def clean_feat(df):
    df = df.copy()

    df['utm_adcontent'] = df.utm_adcontent.fillna('Other_adcont')
    utm_adcontent_list = list(df.utm_adcontent.value_counts(dropna=False, normalize=True)[
                                  df.utm_adcontent.value_counts(dropna=False, normalize=True).apply(
                                      lambda x: f'{x:0.3f}') >= '0.01'].index)

    def utm_adcontent_freq(data):
        if data in utm_adcontent_list:
            return data
        return 'Other_adcont'

    df['utm_adcontent'] = df.utm_adcontent.apply(utm_adcontent_freq)

    df.utm_campaign = df.utm_campaign.fillna('Other_campaign')
    utm_campaign_list = list(df.utm_campaign.value_counts(dropna=False, normalize=True)[
                                 df.utm_campaign.value_counts(dropna=False, normalize=True).apply(
                                     lambda x: f'{x:0.3f}') >= '0.01'].index)

    def utm_campaign_freq(data):
        if data in utm_campaign_list:
            return data
        return 'Other_campaign'

    df['utm_campaign'] = df.utm_campaign.apply(utm_campaign_freq)

    category_list = list(df.device_category.unique())
    brand_dict = {i: df[df['device_category'] == i]['device_brand'].mode(dropna=False)[0] for i in
                  category_list}
    brand_dict['desktop'] = 'desktop'
    df['device_brand'] = df['device_brand'].fillna(df.device_category.map(brand_dict))

    brand_dict = {i: df[df['device_category'] == i]['device_brand'].mode(dropna=False)[0] for i in
                  category_list}
    brand_dict['desktop'] = 'desktop'
    df['device_brand'] = df['device_brand'].fillna(df.device_category.map(brand_dict))

    return df


def create_feat(df):
    import pandas as pd
    df = df.copy()

    df.visit_date = pd.to_datetime(df.visit_date, utc=True)
    df['weekday'] = df.visit_date.map(lambda x: x.weekday())
    df['day'] = df.visit_date.map(lambda x: x.day)
    df['month'] = df.visit_date.map(lambda x: x.month)
    df['visit_hour'] = df.visit_time.apply(lambda x: x.replace(':', '.')[:2])

    def device_comby(category, brand):
        if brand == '':
            return str(category)
        return str(category) + '-' + str(brand)

    df['device+brand'] = df.apply(lambda x: device_comby(x['device_category'], x['device_brand']), axis=1)

    device_brand_list = list(df['device+brand'].value_counts(dropna=False, normalize=True)[
                                 df['device+brand'].value_counts(dropna=False, normalize=True).apply(
                                     lambda x: f'{x:0.3f}') >= '0.001'].index)

    def dev_brand_freq(data):
        if data in device_brand_list:
            return data
        return 'Other_device'

    df['device+brand'] = df['device+brand'].apply(dev_brand_freq)

    device_browser_list = list(df.device_browser.value_counts(dropna=False, normalize=True)[
                                   df.device_browser.value_counts(dropna=False, normalize=True).apply(
                                       lambda x: f'{x:0.3f}') >= '0.001'].index)

    def browser_freq(data):
        if data in device_browser_list:
            return data
        return 'Other_brows'

    df['device_browser'] = df.device_browser.apply(browser_freq)

    utm_source_list = list(df.utm_source.value_counts(dropna=False, normalize=True)[
                               df.utm_source.value_counts(dropna=False, normalize=True).apply(
                                   lambda x: f'{x:0.3f}') >= '0.01'].index)

    def utm_source_freq(data):
        if data in utm_source_list:
            return data
        return 'Other_us'

    df['utm_source'] = df.utm_source.apply(utm_source_freq)

    geo_country_list = list(df.geo_country.value_counts(dropna=False, normalize=True)[
                                df.geo_country.value_counts(dropna=False, normalize=True).apply(
                                    lambda x: f'{x:0.3f}') >= '0.001'].index)

    def geo_country_freq(data):
        if data in geo_country_list:
            return data
        return 'Other_country'

    df['geo_country'] = df.geo_country.apply(geo_country_freq)

    geo_city_list = list(df.geo_city.value_counts(dropna=False, normalize=True)[
                             df.geo_city.value_counts(dropna=False, normalize=True).apply(
                                 lambda x: f'{x:0.3f}') >= '0.01'].index)

    def geo_city_freq(data):
        if data in geo_city_list:
            return data
        return 'Other_city'

    df['geo_city'] = df.geo_city.apply(geo_city_freq)

    organic_list = ['organic', 'referral', '(none)']
    df['is_organic'] = df.utm_medium.apply(lambda x: 1 if x in organic_list else 0)

    df.device_screen_resolution = df.device_screen_resolution.apply(
        lambda x: '0x0' if x == '(not set)' else x)

    df['device_screen_width'] = df.device_screen_resolution.apply(lambda x: x.split('x')[0])
    df['device_screen_height'] = df.device_screen_resolution.apply(lambda x: x.split('x')[-1])

    df['device_screen_width'] = df.device_screen_width.astype(int)
    df['device_screen_height'] = df.device_screen_height.astype(int)

    columns_drop = ['device_category',
                    'device_brand',
                    'device_screen_resolution',
                    'utm_medium',
                    'visit_date',
                    'visit_time',
                    'geo_country',
                    'geo_city',
                    'session_id',
                    'client_id'
                    ]

    return df.drop(columns_drop, axis=1)


def main():
    t_actions = ['sub_car_claim_click', 'sub_car_claim_submit_click',
                 'sub_open_dialog_click', 'sub_custom_question_submit_click',
                 'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
                 'sub_car_request_submit_click']

    df1 = pd.read_csv('data/ga_sessions.csv', low_memory=False)
    df2 = pd.read_csv('data/ga_hits.csv')

    target_sessions = list(df2[df2.event_action.isin(t_actions)].session_id.unique())
    df1['target'] = df1.session_id.isin(target_sessions)
    df1['target'] = df1.target.astype(int)
    df = df1.copy()

    X = df.drop('target', axis=1)
    y = df['target']

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=object)

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('outlier_remover', FunctionTransformer(clean_feat)),
        ('feature_creator', FunctionTransformer(create_feat)),
        ('column_transformer', column_transformer)
    ], verbose=True)

    model = RandomForestClassifier(n_jobs=-1, random_state=42, verbose=2)

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ], verbose=True)

    pipe.fit(X, y)
    roc_auc = roc_auc_score(y, pipe.predict_proba(X)[:, 1])
    print(f'model: {type(model).__name__}, ROC AUC: {roc_auc:.4f}')

    with open('model_pipeline.pkl', 'wb') as file:
        dill.dump({
            'model': pipe,
            'metadata': {
                'name': 'Action predictions model',
                'author': 'Aleksandr',
                'version': 1,
                'date': datetime.now(),
                'type': type(pipe.named_steps["classifier"]).__name__,
                'ROC AUC': roc_auc
            }
        }, file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
