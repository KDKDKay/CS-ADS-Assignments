# Part 1: Essentials


def business_ids_count():
    """
    Write a SQL query that finds the number of business ids in the businesses table
    :return: a string representing the SQL query
    :rtype: str
    """
    query = "SELECT COUNT(business_id) FROM businesses"
    return query


def unique_business_names_count():
    """
    Write a SQL query that finds out how many unique business names are registered
    with San Francisco Food health investigation organization
    and name the column as unique restaurant name count.
    :return: a string representing the SQL query
    :rtype: str
    """
    query = """SELECT COUNT (DISTINCT name) AS 'unique restaurant name count'
    FROM businesses"""
    return query


def first_and_last_investigation():
    """
    Write a SQL query that finds out what is the earliest and latest date
    a health investigation is recorded in this database.
    :return: a string representing the SQL query
    :rtype: str
    """
    query = "SELECT MIN(date) AS 'earliest', MAX(date) AS 'latest' FROM inspections"
    return query


def business_local_owner_count():
    """
    How many businesses are there in San Francisco where their owners live
    in the same area (postal code/ zip code) as the business is located?
    :return: a string representing the SQL query
    :rtype: str
    """
    query = "SELECT COUNT(name) AS 'live in same area as business' FROM businesses WHERE postal_code = owner_zip"
    return query


def business_local_owner_reg_count():
    """
    Out of those businesses, how many of them has a registered business certificate?
    :return: a string representing the SQL query
    :rtype: str
    """
    query = "SELECT COUNT(name) FROM businesses WHERE postal_code = owner_zip AND business_certificate IS NOT NULL"
    return query
