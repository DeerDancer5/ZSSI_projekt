from sklearn.feature_selection import SelectKBest, mutual_info_classif


def build_feature_selector(k="all"):
    return SelectKBest(score_func=mutual_info_classif, k=k)