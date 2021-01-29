import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')
combined = pd.concat([train_df, test_df])


def q(name):
    def run(f):
        print(name.center(100, '-'))
        f()

    return run


@q('Questions 1-6')
def _():
    print('Train df info:')
    print(train_df.info())
    print()
    print('Test df info:')
    print(test_df.info())
    print()
    print('Train df head:')
    print(train_df.head())
    print()

    def columns_with_blank_values(df):
        def has_blank_values(column):
            try:
                return df[column].str.contains('^\\s+$').any()
            except AttributeError:
                # non-string columns
                return False

        return [column for column in df.columns if has_blank_values(column)]

    print('Columns with whitespace (blank) values:')
    print('Train:', columns_with_blank_values(train_df))
    print('Test: ', columns_with_blank_values(test_df))


@q('Question 7')
def _():
    print(train_df[['Age', 'SibSp', 'Parch', 'Fare']].describe())


@q('Question 8')
def _():
    print(train_df[['Survived', 'Pclass', 'Sex', 'Embarked']].astype('category').describe())


@q('Question 9')
def _():
    print('Survived among 1st class:')
    print(train_df[train_df['Pclass'] == 1]['Survived'].value_counts(normalize=True))


@q('Question 10')
def _():
    print(train_df.groupby('Sex')['Survived'].value_counts())


def plot_age(filter_array, *, xlabel='Age', **kwargs):
    train_df[filter_array]['Age'].hist(bins=20, **kwargs).set_xlabel(xlabel)


@q('Question 11')
def _():
    print('Rendering the plots')

    _, ax = plt.subplots(1, 2, sharey=True, figsize=(12.8, 4.8))

    plot_age(train_df['Survived'] == 0, ax=ax[0])
    ax[0].set_title('Survived = 0')
    plot_age(train_df['Survived'] == 1, ax=ax[1])
    ax[1].set_title('Survived = 1')
    plt.show()

    plot_age(train_df['Survived'] == 0, label='Survived = 0', alpha=0.5)
    plot_age(train_df['Survived'] == 1, label='Survived = 1', alpha=0.5)
    plt.title('Merged')
    plt.legend()
    plt.show()


@q('Question 12')
def _():
    print('Rendering the plots')

    _, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(12.8, 9.6))

    for row_index, row in enumerate(axes):
        for col_index, ax in enumerate(row):
            pclass = row_index + 1
            survived = col_index

            plot_age(np.logical_and(
                train_df['Pclass'] == pclass,
                train_df['Survived'] == survived,
            ), ax=ax, xlabel='')
            ax.set_title('Pclass = %d | Survived = %d' % (pclass, survived))

    for ax in row:
        ax.set_xlabel('Age')

    plt.show()


@q('Question 13')
def _():
    print('Rendering the plots')

    _, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(12.8, 9.6))
    ports = ('S', 'C', 'Q')

    for row_index, row in enumerate(axes):
        for col_index, ax in enumerate(row):
            embarked = ports[row_index]
            survived = col_index

            train_df[np.logical_and(
                train_df['Embarked'] == embarked,
                train_df['Survived'] == survived
            )].groupby('Sex')['Fare'].mean().plot.bar(ax=ax)
            ax.set_title('Embarked = %s | Survived = %d' % (embarked, survived))

    plt.show()


@q('Question 14')
def _():
    print(
        'Ticket duplicates rate:',
        train_df.duplicated(subset=['Ticket']).value_counts(normalize=True)[True]
    )


@q('Question 15')
def _():
    print(
        'Null Cabin values: %d out of %d' %
        (combined['Cabin'].isnull().sum(), len(combined))
    )


@q('Question 16')
def _():
    train_df['Gender'] = train_df['Sex'].map({'female': 1, 'male': 0})
    print('With Gender:')
    print(train_df[['PassengerId', 'Sex', 'Gender']].head())


@q('Question 17')
def _():
    print('Age before completing null values:')
    print(train_df['Age'].describe())
    print()

    mean = train_df['Age'].mean()
    std = train_df['Age'].std()
    age_low, age_high = (max(mean - std, 0), mean + std)

    def random_age():
        return np.random.randint(age_low, age_high)

    train_df['Age'] = train_df['Age'].map(lambda age: random_age() if pd.isnull(age) else age)

    print('Age after completing null values:')
    print(train_df['Age'].describe())


@q('Question 18')
def _():
    print('Embarked values before completing null values:')
    print(train_df['Embarked'].value_counts(dropna=False))
    print()
    train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
    print('Embarked values after completing null values:')
    print(train_df['Embarked'].value_counts(dropna=False))


@q('Question 19')
def _():
    mode = test_df['Fare'].mode()[0]
    print('Mode Fare:', mode)
    print('Null Fare values in test data before completing:', test_df['Fare'].isnull().sum())
    test_df['Fare'] = test_df['Fare'].fillna(mode)
    print('Null Fare values in test data after completing:', test_df['Fare'].isnull().sum())


@q('Question 20')
def _():
    train_df['Fare'] = pd.qcut(train_df['Fare'], 4)
    print(train_df['Fare'].describe())
    print(train_df['Fare'].cat.categories)
