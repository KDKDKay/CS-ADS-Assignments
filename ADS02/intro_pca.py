"""This file contains a set of functions to implement using PCA.

All of them take at least a dataframe df as argument. To test your functions
locally, we recommend using the wine dataset that you can load from sklearn by
importing sklearn.datasets.load_wine"""


def get_cumulated_variance(df, scale):
    """Apply PCA on a DataFrame and return a new DataFrame containing
    the cumulated explained variance from with only the first component,
    up to using all components together. Values should be expressed as
    a percentage of the total variance explained.

    The DataFrame will have one row and each column should correspond to a
    principal component.

    Example:
             PC1        PC2        PC3        PC4    PC5
    0  36.198848  55.406338  66.529969  73.598999  100.0

    If scale is True, you should standardise the data first
    Tip: use the StandardScaler from sklearn

    :param df: pandas DataFrame
    :param scale: boolean, whether to scale or not
    :return: a new DataFrame with cumulated variance in percent
    """

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import numpy as np

    if scale:
        df = StandardScaler().fit_transform(df)

    pca = PCA()
    pca.fit(df)
    cum_expl_ratio = np.cumsum(pca.explained_variance_ratio_)*100
    length = len(cum_expl_ratio)
    cum_expl_ratio = np.reshape(cum_expl_ratio, (1, length))
    final = pd.DataFrame(cum_expl_ratio,
                         columns=['PC'+str(index+1) for index in range(df.shape[1])])
    return final


def get_coordinates_of_first_two(df, scale):
    """Apply PCA on a given DataFrame df and return a new DataFrame
    containing the coordinates of the first two principal components
    expressed in the original basis (with the original columns).

    Example:
    if the original DataFrame was:

          A    B
    0   1.3  1.2
    1  27.0  2.1
    2   3.3  6.8
    3   5.1  3.2

    we want the components PC1 and PC2 expressed as a linear combination
    of A and B, presented in a table as:

              A      B
    PC1    0.99  -0.06
    PC2    0.06   0.99

    If scale is True, you should standardise the data first
    Tip: use the StandardScaler from sklearn

    :param df: pandas DataFrame
    :param scale: boolean, whether to scale or not
    :return: a new DataFrame with coordinates of PC1 and PC2
    """

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    features = list(df[:0])
    if scale:
        df = StandardScaler().fit_transform(df)

    pca = PCA()
    pca.fit(df)
    components = pca.components_[0:2]
    final = pd.DataFrame(components,
                         columns=features,
                         index=['PC'+str(index+1) for index in range(2)])
    return final


def get_most_important_two(df, scale):
    """Apply PCA on a given DataFrame df and use it to determine the
    'most important' features in your dataset. To do so we will focus
    on the principal component that exhibits the highest explained
    variance (that's PC1).

    PC1 can be expressed as a vector with weight on each of the original
    columns. Here we want to return the names of the two features that
    have the highest weights in PC1 (in absolute value).

    Example:
        if the original DataFrame was:

          A    B     C
     0  1.3  1.2   0.1
     1  2.0  2.1   1.2
     2  3.3  6.8  23.4
     3  5.1  3.2   4.5

    and PC1 can be written as [0.05, 0.22, 0.97] in [A, B, C].

    Then you should return C, B as the two most important features.

    If scale is True, you should standardise the data first
    Tip: use the StandardScaler from sklearn

    :param df: pandas DataFrame
    :param scale: boolean, whether to scale or not
    :return: names of the two most important features as a tuple
    """

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    features = df[:0]
    if scale:
        df = pd.DataFrame(StandardScaler().fit_transform(df), columns=[features])

    pca = PCA(n_components=1)
    pca.fit_transform(df)
    components = abs(pca.components_[0])
    comp = list(zip(components, features))
    _col1, val1 = max1 = max(comp)
    comp.remove(max1)
    _col2, val2 = max(comp)

    return val1, val2


def distance_in_n_dimensions(df, point_a, point_b, n, scale):
    """Write a function that applies PCA on a given DataFrame df in order to find
    a new subspace of dimension n.

    Transform the two points point_a and point_b to be represented into that
    n dimensions space, compute the Euclidean distance between the points in
    that space and return it.

    Example:
        if the original DataFrame was:

          A    B     C
     0  1.3  1.2   0.1
     1  2.0  2.1   1.2
     2  3.3  6.8  23.4
     3  5.1  3.2   4.5

    and n = 2, you can learn a new subspace with two columns [PC1, PC2].

    Then given two points:

    point_a = [1, 2, 3]
    point_b = [2, 3, 4]
    expressed in [A, B, C]

    Transform them to be expressed in [PC1, PC2], here we would have:
    point_a -> [-4.57, -1.74]
    point_b -> [-3.33, -0.65]

    and return the Euclidean distance between the points
    in that space.

    If scale is True, you should standardise the data first
    Tip: use the StandardScaler from sklearn

    :param df: pandas DataFrame
    :param point_a: a numpy vector expressed in the same basis as df
    :param point_b: a numpy vector expressed in the same basis as df
    :param n: number of dimensions of the new space
    :param scale: whether to scale data or not
    :return: distance between points in the subspace
    """

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import numpy as np

    features = list(df[:0])
    points = pd.DataFrame([list(point_a), list(point_b)], columns=features)

    if scale:
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=features)
        points = pd.DataFrame(scaler.transform(points), columns=points.columns)

    pca = PCA(n_components=n)
    pca.fit(df)
    new_points = pca.transform(points)
    distance = np.linalg.norm(new_points[0]-new_points[1])
    return distance


def find_outliers_pca(df, n, scale):
    """Apply PCA on a given DataFrame df and transofmr all the data to be expressed
    on the first principal component (you can discard other components)

    With all those points in a one-dimension space, find outliers by looking for points
    that lie at more than n standard deviations from the mean.

    You should return a new dataframe containing all the rows of the original dataset
    that have been found to be outliers when projected.

    Example:
        if the original DataFrame was:

          A    B     C
     0  1.3  1.2   0.1
     1  2.0  2.1   1.2
     2  3.3  6.8  23.4
     3  5.1  3.2   4.5

    Once projected on PC1 it will be:
          PC1
    0   -7.56
    1   -6.26
    2   16.46
    3   -2.65

    Compute the mean of this one dimensional dataset and find all rows that lie at more
    than n standard deviations from it.

    Here, if n==1, only the row 2 is an outlier.

    So you should return:
         A    B     C
    2  3.3  6.8  23.4


    If scale is True, you should standardise the data first
    Tip: use the StandardScaler from sklearn

    :param df: pandas DataFrame
    :param n: number of standard deviations from the mean to be considered outlier
    :param scale: whether to scale data or not
    :return: pandas DataFrame containing outliers only
    """

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    if scale:
        newdf = StandardScaler().fit_transform(df)
    else:
        newdf = df.copy()

    pca = PCA()
    pca.fit(newdf)
    wmatrix = pca.components_[0]

    aprod = np.dot(newdf, wmatrix.T)
    aprod_std = aprod.std(ddof=0)
    outliers = df[abs(aprod/aprod_std) > n]
    return outliers
