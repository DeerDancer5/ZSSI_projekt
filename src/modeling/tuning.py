from sklearn.model_selection import GridSearchCV, StratifiedKFold


def get_param_grid():
    return {
        "select__k": [5, 8, 10, "all"],
        "mlp__hidden_layer_sizes": [(16,), (32,), (32,16), (64,32)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [0.0001, 0.001, 0.01],
        "mlp__learning_rate_init": [0.001, 0.01],
        "mlp__batch_size": [16, 32, 64],
    }

def build_grid_search(pipeline):
    param_grid = get_param_grid()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=2
    )
    return grid