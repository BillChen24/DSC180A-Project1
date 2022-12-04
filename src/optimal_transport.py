import numpy as np
import ot

def sample_color_with_class(X_train, y_train, X_gray, n, class_label):
    X_target = X_train[y_train == class_label]
    
    X_target_r_pool = np.array([x[:1024] for x in X_target]).flatten()
    X_target_g_pool = np.array([x[1024:2048] for x in X_target]).flatten()
    X_target_b_pool = np.array([x[2048:] for x in X_target]).flatten()
    
    X_source_r_pool = np.array([x[:1024] for x in X_gray]).flatten()
    X_source_g_pool = np.array([x[1024:2048] for x in X_gray]).flatten()
    X_source_b_pool = np.array([x[2048:] for x in X_gray]).flatten()
    
    sample1 = np.random.randint(len(X_target_r_pool), size = n)
    sample2 = np.random.randint(len(X_source_r_pool), size = n)
    
    X_target_r = X_target_r_pool[sample1]
    X_target_g = X_target_g_pool[sample1]
    X_target_b = X_target_b_pool[sample1]
    
    X_source_r = X_source_r_pool[sample1]
    X_source_g = X_source_g_pool[sample1]
    X_source_b = X_source_b_pool[sample1]
    
    target_sample = np.array([[r, g, b] for r,g,b in zip(X_target_r, X_target_g, X_target_b)])
    source_sample = np.array([[r, g, b] for r,g,b in zip(X_source_r, X_source_g, X_source_b)])
    
    return target_sample, source_sample

def sample_color_within_class(X_train, y_train, X_gray, n, class_label):
    X_target = X_train[y_train == class_label]
    X_gray = X_gray[y_train == class_label]
    
    X_target_r_pool = np.array([x[:1024] for x in X_target]).flatten()
    X_target_g_pool = np.array([x[1024:2048] for x in X_target]).flatten()
    X_target_b_pool = np.array([x[2048:] for x in X_target]).flatten()
    
    X_source_r_pool = np.array([x[:1024] for x in X_gray]).flatten()
    X_source_g_pool = np.array([x[1024:2048] for x in X_gray]).flatten()
    X_source_b_pool = np.array([x[2048:] for x in X_gray]).flatten()
    
    sample1 = np.random.randint(len(X_target_r_pool), size = n)
    sample2 = np.random.randint(len(X_source_r_pool), size = n)
    
    X_target_r = X_target_r_pool[sample1]
    X_target_g = X_target_g_pool[sample1]
    X_target_b = X_target_b_pool[sample1]
    
    X_source_r = X_source_r_pool[sample1]
    X_source_g = X_source_g_pool[sample1]
    X_source_b = X_source_b_pool[sample1]
    
    target_sample = np.array([[r, g, b] for r,g,b in zip(X_target_r, X_target_g, X_target_b)])
    source_sample = np.array([[r, g, b] for r,g,b in zip(X_source_r, X_source_g, X_source_b)])
    
    return target_sample, source_sample

def color_ot_build(X_source, X_target, ot_type = ot.da.EMDTransport()):
    ot_type.fit(Xs=X_source, Xt=X_target)
    return ot_type

def color_ot_transform(X_test_gray, ot_type):
    r = X_test_gray[:1024]
    g = X_test_gray[1024:2048]
    b = X_test_gray[2048:]
    to_transform = np.array([[r0,g0,b0] for r0,b0,g0 in zip(r,g,b)])
    transformed = ot_type.transform(to_transform)
    transformed = transformed.transpose().flatten().astype(int)
    transformed = np.array([x if x > 0 else 0 for x in transformed])
    return transformed

def color_ot_transform_with_allclass(X_train, y_train, X_train_gray, X_test_gray, n):
    out = {}
    for label in set(y_train):
        print(label)
        target_sample, source_sample = sample_color_with_class(X_train, y_train, X_train_gray, n, label)
        ot_transformer = color_ot_build(source_sample, target_sample)
        transformed_ls = []
        for x in X_test_gray:
            transformed = color_ot_transform(x, ot_transformer)
            transformed_ls.append(transformed)
        out[label] = np.array(transformed_ls)
    return out

def color_ot_transform_within_allclass(X_train, y_train, X_train_gray, X_test_gray, n):
    out = {}
    for label in set(y_train):
        print(label)
        target_sample, source_sample = sample_color_within_class(X_train, y_train, X_train_gray, n, label)
        ot_transformer = color_ot_build(source_sample, target_sample)
        transformed_ls = []
        for x in X_test_gray:
            transformed = color_ot_transform(x, ot_transformer)
            transformed_ls.append(transformed)
        out[label] = np.array(transformed_ls)
    return out




