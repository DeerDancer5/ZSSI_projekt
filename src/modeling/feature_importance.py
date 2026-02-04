import numpy as np
import matplotlib.pyplot as plt

def visualize_feature_importance(model, X_train):
    preprocessor = model.named_steps["preprocess"]
    selector = model.named_steps["select"]

    num_features = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.transformers_[1][1]
    cat_features = preprocessor.transformers_[1][2]

    ohe_feature_names = cat_encoder.get_feature_names_out(cat_features)

    all_features = np.concatenate([num_features, ohe_feature_names])

    mask = selector.get_support()
    selected_features = all_features[mask]

    scores = selector.scores_[mask]

    indices = np.argsort(scores)[::-1]
    selected_features = selected_features[indices]
    scores = scores[indices]

    plt.figure(figsize=(10, 6))
    plt.barh(selected_features, scores)
    plt.gca().invert_yaxis()
    plt.xlabel("Mutual Information Score")
    plt.title("Feature Importance (SelectKBest + MI) after Preprocessing")
    plt.tight_layout()
    plt.show()
