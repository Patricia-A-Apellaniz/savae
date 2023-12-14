# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 14/11/2023

# Packages to import
import pandas as pd
from pycox import datasets


def join_sa_data(df, time, label):
    survival_df = time.join(label)
    df = df.join(survival_df)
    df = df.rename(columns={df.columns[-1]: 'event', df.columns[-2]: 'time'})
    return df


# Function that takes a df and a column name and returns a df with the column name as a categorical variable
def columns_to_categorical(df, column_names):
    for column_name in column_names:
        df = df.rename(columns={column_name: column_name + '.0'})
        column_names = df.columns[df.columns.str.startswith(column_name)]
        for sample in range(df.shape[0]):
            cat = None
            for column in column_names:
                if df.loc[sample, column] == 1:
                    cat = column.split('.')[1]
                    break
            if cat is None:
                raise RuntimeError('No cat found for sample ')
            df.loc[sample, column_name] = cat

        # Remove column_names from df
        df = df.drop(labels=column_names, axis=1)
        df[column_name] = df[column_name].astype(int)
    return df


####################################################################################################
#                               SURVIVAL ANALYSIS DATASETS                                         #
####################################################################################################


def preprocess_whas(args):
    # Load data
    data_filename = args['input_dir'] + 'whas/whas.csv'
    df = pd.read_csv(data_filename, sep=',')

    return df


def preprocess_support(args):
    # Load data
    data_filename = args['input_dir'] + 'support/support.csv'
    df = pd.read_csv(data_filename, sep=',')

    # Variables
    df = df.dropna()
    df.loc[:, 'sex'] = df.loc[:, 'sex'].replace(['male', 'female'], [0, 1])
    df.loc[:, 'race'] = df.loc[:, 'race'].replace(['white', 'black', 'hispanic', 'asian', 'other'], [0, 1, 2, 3, 4])
    df.loc[:, 'ca'] = df.loc[:, 'ca'].replace(['metastatic', 'no', 'yes'], [0, 1, 2])
    df.loc[:, 'meanbp'] = df.loc[:, 'meanbp'].astype(int)
    df.loc[:, 'hrt'] = df.loc[:, 'hrt'].astype(int)
    df.loc[:, 'resp'] = df.loc[:, 'resp'].astype(int)
    df.loc[:, 'sod'] = df.loc[:, 'sod'].astype(int)

    return df


def preprocess_gbsg():
    # Load data
    df = datasets.gbsg.read_df()

    # Variables
    df.loc[:, 'x0'] = df.loc[:, 'x0'].astype(int)
    df.loc[:, 'x1'] = df.loc[:, 'x1'].astype(int)
    df.loc[:, 'x2'] = df.loc[:, 'x2'].astype(int)
    df.loc[:, 'x3'] = df.loc[:, 'x3'].astype(int)
    df.loc[:, 'x4'] = df.loc[:, 'x4'].astype(int)
    df.loc[:, 'x5'] = df.loc[:, 'x5'].astype(int)
    df.loc[:, 'x6'] = df.loc[:, 'x6'].astype(int)
    label = df[['event']]
    time = df[['duration']]
    df = df.drop(labels=['event', 'duration'], axis=1)

    # Join data
    df = join_sa_data(df, time, label)

    return df


def preprocess_flchain():
    # Load data
    df = datasets.flchain.read_df(
        processed=False)  # Processed is set to False because there is a bug in Pycox library.
    # The following lines are copied from pycox library

    # Process dataset
    df = (df
          .drop(['chapter', 'rownames'], axis=1)
          .loc[lambda x: x['creatinine'].isna() == False]
          .reset_index(drop=True)
          .assign(sex=lambda x: (x['sex'] == 'M')))

    categorical = ['sample.yr', 'flc.grp']
    for col in categorical:
        df[col] = df[col].astype('category')
    for col in df.columns.drop(categorical):
        df[col] = df[col].astype('float32')

    # Variables
    df.loc[:, 'age'] = df.loc[:, 'age'].astype(int)
    df.loc[:, 'sex'] = df.loc[:, 'sex'].astype(int)
    df.loc[:, 'mgus'] = df.loc[:, 'mgus'].astype(int)
    label = df[['death']].astype(int)
    time = df[['futime']].astype(int)
    df = df.drop(labels=['death', 'futime'], axis=1)

    # Join data
    df = join_sa_data(df, time, label)

    return df


def preprocess_nwtco():
    # Load data
    df = datasets.nwtco.read_df(processed=False)  # Processed is set to False because there is a bug.
    # The following lines are copied from pycox library

    # Process dataset
    df = (df
          .assign(instit_2=df['instit'] - 1,
                  histol_2=df['histol'] - 1,
                  study_4=df['study'] - 3,
                  stage=df['stage'].astype('category'))
          .drop(['rownames', 'seqno', 'instit', 'histol', 'study'], axis=1))
    for col in df.columns.drop('stage'):
        df[col] = df[col].astype('float32')

    # Variables
    df.loc[:, 'age'] = df.loc[:, 'age'].astype(int)
    df.loc[:, 'in.subcohort'] = df.loc[:, 'in.subcohort'].astype(int)
    df.loc[:, 'instit_2'] = df.loc[:, 'instit_2'].astype(int)
    df.loc[:, 'histol_2'] = df.loc[:, 'histol_2'].astype(int)
    df.loc[:, 'study_4'] = df.loc[:, 'study_4'].astype(int)
    label = df[['rel']].astype(int)
    time = df[['edrel']].astype(int)
    df = df.drop(labels=['rel', 'edrel'], axis=1)

    # Join data
    df = join_sa_data(df, time, label)

    return df


def preprocess_metabric(args):
    # Load data
    data_filename = args['input_dir'] + 'metabric/cleaned_features.csv'
    df = pd.read_csv(data_filename, sep=',')
    surv_filename = args['input_dir'] + 'metabric/label.csv'
    surv_df = pd.read_csv(surv_filename, sep=',')

    # Extract info
    surv_df = surv_df.rename(columns={'label': 'event', 'event_time': 'time'})
    label = surv_df[['event']]
    time = surv_df[['time']]
    df = df.drop(labels=['NPI'], axis=1)  # This column is redundant
    df = df.drop(df[df['size'] == 5.5].index).reset_index(
        drop=True)  # Drop sample that has no integer in "size" feature
    df['size'] = df['size'].astype(int)

    # Convert one-hot-encoded variables to single categorical one to reduce dimensionality
    col_to_cat = ['grade', 'histological', 'HER2_IHC_status', 'HER2_SNP6_state', 'Treatment', 'group', 'cellularity',
                  'Pam50_Subtype', 'int_clust_memb', 'site', 'Genefu']
    df = columns_to_categorical(df, col_to_cat)

    # Remove ER_IHC_status.1, ER_Expr.1, PR_Expz.1
    df = df.drop(labels=['ER_IHC_status', 'ER_Expr', 'PR_Expz', 'Her2_Expr', 'inf_men_status'], axis=1)

    # Join data
    df = join_sa_data(df, time, label)

    return df


def preprocess_pbc(args):
    # Load data
    data_filename = args['input_dir'] + 'pbc/pbc.csv'
    df = pd.read_csv(data_filename, sep=',', index_col=0)

    # Extract info
    df['treatment'] = df['treatment'].apply(lambda x: 0 if x == 1 else 1)
    df["edema"] = df["edema"].apply(
        lambda x: 0 if x < 0.25 else (2 if x > 0.75 else 1))  # Make this a categorical variable (3 values)
    df['stage'] = df['stage'].replace([1, 2, 3, 4], [0, 1, 2, 3])
    label = df[['status']]
    time = df[['days']]
    df = df.drop(labels=["days", "status"], axis=1)

    # Join data
    df = join_sa_data(df, time, label)
    df.loc[:, 'time'] = df.loc[:, 'time'] / 30

    return df


def preprocess_std(args):
    # Load data
    data_filename = args['input_dir'] + 'std/std.csv'
    df = pd.read_csv(data_filename, sep=',', index_col=0)

    # Extract info
    label = df[['rinfct']]
    time = df[['time']]
    df = df.drop(labels=["obs", "time", "rinfct"], axis=1)
    df["race"] = df["race"].apply(lambda x: 0 if x == "B" else 1)
    df["marital"] = df["marital"].apply(lambda x: 0 if x == "S" else (1 if x == "M" else 2))
    df['iinfct'] = df['iinfct'].replace([1, 2, 3], [0, 1, 2])
    df['condom'] = df['condom'].replace([1, 2, 3], [0, 1, 2])

    # Join data
    df = join_sa_data(df, time, label)
    df.loc[:, 'time'] = df.loc[:, 'time'] / 30

    return df


def preprocess_pneumon(args):
    # Load data
    data_filename = args['input_dir'] + 'pneumon/pneumon.csv'
    df = pd.read_csv(data_filename, sep=',', index_col=0)

    # Extract info
    label = df[['hospital']]
    time = df[['chldage']]  # Convert months to days to get rid of decimals
    df = df.drop(labels=["chldage", "hospital"], axis=1)
    df['region'] = df['region'].replace([1, 2, 3, 4], [0, 1, 2, 3])
    df['race'] = df['race'].replace([1, 2, 3], [0, 1, 2])

    # Join data
    df = join_sa_data(df, time, label)

    return df
