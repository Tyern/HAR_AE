import numpy as np
from copy import deepcopy

def limit_filter_data_by_class(data, label, limit_number=-1, seed=None):
    if seed: 
        np.random.seed(seed=seed)

    # limit number is the maximum number of instance to filter each class
    print("limit_filter_data_by_class", "limit_number=", limit_number)
    assert len(data) == len(label)

    if limit_number == -1:
        return (data, label, np.arange(len(label)))
    else:
        choice_limited_list = []

        label_set = set(label)
        for i in label_set:
            one_class_idx = np.where(label == i)[0]
            choice_idx_list = np.random.choice(
                one_class_idx, 
                min(limit_number, len(one_class_idx)), 
                replace=False)
            
            choice_limited_list.extend(choice_idx_list)

        limited_train_data = data[choice_limited_list]
        limited_train_label = label[choice_limited_list]

        print("limit_filter_data_by_class: ", np.unique(limited_train_label, return_counts=True))
    
        return (limited_train_data, limited_train_label, np.array(choice_limited_list))

def merge_data_set(label, label_merge_dict):
    # label_merge_dict = {
    #     1:1,
    #     2:2,
    #     3:3,
    #     4:4,
    #     5: [5, 6, 7, 8],
    # }
    new_label = np.array(deepcopy(label))
    label = np.array(label)
    assert sorted(label_merge_dict.keys()) == list(range(1, len(label_merge_dict) + 1))
    
    val_list = []
    
    for key, val in label_merge_dict.items():
        if isinstance(val, int):
            if val in val_list:
                raise ValueError("Repeat key value")
            else:
                val_list.append(val)
            new_label[label == val] = key

        elif isinstance(val, list):
            for val_i in val:
                if val_i in val_list:
                    raise ValueError("Repeat key value")
                else:
                    val_list.append(val_i)

                new_label[label == val_i] = key
                
    return new_label