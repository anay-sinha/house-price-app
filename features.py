# features.py
def add_features(d):
    d = d.copy()
    d["TotalSF"]   = d["TotalBsmtSF"].fillna(0) + d["1stFlrSF"] + d["2ndFlrSF"]
    d["HouseAge"]  = d["YrSold"] - d["YearBuilt"]
    d["RemodAge"]  = d["YrSold"] - d["YearRemodAdd"]
    d["TotalBath"] = (d["FullBath"] + 0.5 * d["HalfBath"]
                      + d["BsmtFullBath"].fillna(0) + 0.5 * d["BsmtHalfBath"].fillna(0))
    d["HasGarage"] = (d["GarageArea"].fillna(0) > 0).astype(int)
    d["QualxSF"]   = d["OverallQual"] * d["TotalSF"]
    return d
